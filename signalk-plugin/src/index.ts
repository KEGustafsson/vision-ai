// Marine Vision-AI SignalK plugin entrypoint.
//
// Consumes detection events from the YOLOv8 vision container, enriches them
// with own-ship navigation, fuses with AIS, computes CPA/TCPA, raises
// notifications, and publishes the vision.* tree. Also serves a captain webapp
// and proxies the annotated video stream (see router.ts).

import { ContainerClient, ControlBody, HealthInfo } from './containerClient';
import { CpaEstimator } from './cpa';
import { EventStream } from './eventStream';
import { PluginConfig, schema, uiSchema, withDefaults } from './config';
import { collectAisContacts, fuse } from './aisFusion';
import { enrichTarget } from './enrich';
import { NotificationManager } from './notifications';
import { Publisher } from './publisher';
import { readOwnShip } from './nav';
import { registerRoutes, SharedState } from './router';
import { Plugin, ServerApp } from './skapp';
import { DetectionEvent, EnrichedTarget } from './types';

export = function (app: ServerApp): Plugin {
  const pluginId = 'signalk-vision-ai';

  let cfg: PluginConfig = withDefaults(undefined);
  let client: ContainerClient | null = null;
  let stream: EventStream | null = null;
  let publisher: Publisher | null = null;
  let notifier: NotificationManager | null = null;
  let cpa: CpaEstimator | null = null;
  let processTimer: NodeJS.Timeout | null = null;
  let syncTimer: NodeJS.Timeout | null = null;

  const targets = new Map<string, EnrichedTarget>();
  // Last cycle's AIS association (target key -> mmsi). Persisted across process
  // cycles so fuse() can keep a stable target on the same contact instead of
  // flapping its identity as the noisy monocular range wanders.
  let aisAssignment = new Map<string, string>();
  const lastEventByCamera = new Map<string, DetectionEvent>();
  const frameCount = new Map<string, number>();
  let activeCamera = 'forward';
  let modeHint: string | null = null;
  let lastStatsAt = 0;
  // Live master on/off. Seeded from the persisted config in start(), but the
  // captain webapp can flip it at runtime (see router POST /detection). Held
  // here (not in cfg) so a live toggle isn't reverted by the next sync.
  let detectionEnabled = true;
  let maxTargets = cfg.maxTargets;

  const shared: SharedState = {
    get targets() {
      return [...targets.values()];
    },
    get ownShip() {
      return readOwnShip(app, cfg.ownNavMaxAgeS);
    },
    get system() {
      return {
        activeCamera,
        cameras: [...lastEventByCamera.keys()],
        detectionEnabled,
        maxTargets,
      };
    },
    setDetection(on: boolean) {
      detectionEnabled = on;
      // Push immediately so the webapp toggle takes effect without waiting for
      // the next periodic sync.
      void syncContainer();
    },
    setMaxTargets(value: number) {
      // Only the container's per-frame detection cap; the plugin's target list
      // keeps every tracked target (recently-seen but not currently-active ones
      // included) so the captain view can show them all.
      maxTargets = value;
      void syncContainer();
    },
    client: () => client,
  };

  // Per-camera timestamp (ms) of the last event accepted, to drop out-of-order /
  // replayed frames whose timestamp is not newer than the last one we took.
  const lastEventTsByCamera = new Map<string, number>();

  // --- Ingest: cheap, runs per WebSocket frame ---
  // Only update the target map here; the heavy fusion/CPA/notify/publish work
  // is debounced to a fixed cadence in processCycle() so it doesn't scale with
  // (and flap at) the camera frame rate. Tracked detections are keyed by
  // camera.track_id; untracked boxes are dropped (no stable key for CPA/fusion
  // persistence) EXCEPT untracked person-in-water, which is a man-overboard
  // candidate too important to discard (see below).
  function handleEvent(ev: DetectionEvent): void {
    // eventStream already rejected invalid/over-age timestamps; here we enforce
    // monotonic per-camera ordering so a buffered burst of older-but-not-stale
    // frames can't rewind a track's position into CPA/fusion history. Use the
    // event time (not arrival time) so freshness reflects when the frame was
    // actually captured.
    const evTs = Date.parse(ev.timestamp);
    if (!Number.isFinite(evTs)) return; // belt-and-braces; eventStream filters these
    const prevTs = lastEventTsByCamera.get(ev.camera);
    if (prevTs !== undefined && evTs <= prevTs) return; // out-of-order / replayed
    lastEventTsByCamera.set(ev.camera, evTs);

    lastEventByCamera.set(ev.camera, ev);
    frameCount.set(ev.camera, (frameCount.get(ev.camera) ?? 0) + 1);
    // Evaluate own-ship freshness against the frame's capture time (evTs), not
    // delivery time, so georeferencing uses the nav that was current when the
    // frame was shot and a lagged-but-accepted frame doesn't null otherwise-valid
    // nav just because delivery was slow.
    const own = readOwnShip(app, cfg.ownNavMaxAgeS, evTs);

    // `targets` is optional in the wire contract (default empty list); guard so a
    // valid event that omits it can't throw in this per-frame hot path.
    for (const raw of ev.targets ?? []) {
      // NB: do NOT re-filter by raw.confidence here. The container is the single
      // source of truth for what to surface — its stabilizer keeps a track shown
      // via EMA/hysteresis/coasting even when the per-frame confidence dips below
      // the threshold (and context control may lower that threshold at night).
      // Re-gating on raw.confidence would drop exactly those stabilized tracks,
      // so they'd appear on the video overlay but never in the list. The
      // operator's minConfidence is enforced on the container (syncContainer).
      if (cfg.detectClasses.length > 0 && !cfg.detectClasses.includes(raw.label)) continue;
      // NB: minimum-range filtering (minTargetRangeM) is done in the container
      // (pushed via syncContainer), so events already exclude too-close objects
      // from both the target list AND the annotated overlay.
      if (raw.track_id === null) {
        // Untracked detection: normally dropped (no stable id to persist across
        // frames). EXCEPTION: an untracked person-in-water is a MOB candidate, so
        // route it through with a per-camera stable key ("mob-anon") instead of
        // enrich's churning anon-x-y key — that lets the MOB persistence counter
        // accumulate across frames and fire even without a track id. Non-MOB
        // untracked boxes stay excluded from the AIS/CPA path.
        if (!raw.is_person_in_water) continue;
        const t = enrichTarget(raw, ev.camera, own, cfg, evTs);
        t.key = `${ev.camera}.mob-anon`;
        targets.set(t.key, t);
        continue;
      }
      const t = enrichTarget(raw, ev.camera, own, cfg, evTs);
      targets.set(t.key, t);
    }
    pruneLabelSelection();
  }

  function pruneLabelSelection(): void {
    if (cfg.detectClasses.length === 0) return;
    for (const [key, t] of targets) {
      if (!cfg.detectClasses.includes(t.label)) targets.delete(key);
    }
  }

  // --- Process: fixed cadence, does the expensive correlated work ---
  function processCycle(): void {
    const now = Date.now();
    const own = readOwnShip(app, cfg.ownNavMaxAgeS, now);

    // Age out stale tracks first so downstream sees only live targets.
    const timeoutMs = cfg.trackTimeoutS * 1000;
    for (const [key, t] of [...targets]) {
      if (now - t.lastSeen > timeoutMs) targets.delete(key);
    }
    pruneLabelSelection();
    const all = [...targets.values()];

    let darkKeys = new Set<string>();
    let aisCount = 0;
    if (cfg.enableAisFusion) {
      const contacts = collectAisContacts(
        app.getPath('vessels'), own, cfg.ownAisMinRangeM, cfg.aisMaxAgeS * 1000, now);
      const res = fuse(all, contacts, cfg, aisAssignment);
      aisAssignment = res.assignment;
      darkKeys = new Set(res.darkTargetKeys);
      aisCount = res.aisCorrelatedCount;
    } else {
      // Drop hysteresis state while fusion is off so a later re-enable starts
      // clean instead of reusing target→MMSI mappings nothing has maintained.
      aisAssignment = new Map<string, string>();
    }

    if (cfg.enableCollision && cpa) cpa.update(all, own, cfg, now);
    if (notifier) notifier.evaluate(all, darkKeys, now);
    if (publisher) {
      publisher.publishTargets(all);
      if (cfg.enableAisFusion) publisher.publishFusionSummary(darkKeys.size, aisCount);
    }

    publishSystemStats(now);
  }

  function publishSystemStats(now: number): void {
    if (!publisher) return;
    const perCameraCounts: Record<string, number> = {};
    for (const cam of lastEventByCamera.keys()) {
      perCameraCounts[cam] = [...targets.values()].filter((t) => t.camera === cam).length;
    }
    // Inference FPS over the actual elapsed window (not a hardcoded interval).
    let totalFrames = 0;
    for (const [, c] of frameCount) totalFrames += c;
    const elapsedS = lastStatsAt ? Math.max((now - lastStatsAt) / 1000, 1e-3) : 1;
    const fps = totalFrames / elapsedS;
    frameCount.clear();
    lastStatsAt = now;

    const sample = lastEventByCamera.get(activeCamera) ?? [...lastEventByCamera.values()][0];
    publisher.publishSystem({
      activeCamera,
      mode: modeHint ?? undefined,
      backend: sample?.inference.backend,
      inferenceFps: fps,
      horizonY: sample?.horizon_y ?? null,
      perCameraCounts,
    });
    if (app.setPluginStatus) {
      app.setPluginStatus(
        `${targets.size} targets, ${fps.toFixed(1)} fps, active=${activeCamera}`
      );
    }
  }

  // Push runtime settings to the container. Always syncs the operator's
  // object-type selection (so it survives a container restart even with context
  // control off); when context control is on it also picks the active camera
  // and a day/night confidence. Re-run on a timer so a restarted container
  // re-learns the settings.
  async function syncContainer(): Promise<void> {
    if (!client) return;
    // Always carry the master on/off and the object-type selection so a
    // restarted container re-learns both even when context control is off.
    const body: ControlBody = {
      labels: cfg.detectClasses,
      enabled: detectionEnabled,
      max_targets: maxTargets,
      // The container does the minimum-range filtering EARLY (events + overlay);
      // this is the single source of truth, pushed so it survives a restart.
      min_target_range_m: cfg.minTargetRangeM,
      // Always push the confidence threshold so the container (and its stabilizer)
      // gates at the operator's value even when context control is off. The
      // context-control block below overrides it with the night-lowered value.
      confidence: cfg.minConfidence,
    };
    let nextCamera: string | null = null;
    let nextModeHint: string | null = null;
    if (cfg.enableContextControl) {
      const own = readOwnShip(app, cfg.ownNavMaxAgeS);
      const underway = (own.sog ?? 0) >= cfg.underwaySogMs;
      const hour = new Date().getHours();
      const night = hour < 6 || hour >= 21;
      // Underway: watch ahead. Low speed (docking/manoeuvring): watch astern.
      nextCamera = underway ? 'forward' : 'aft';
      nextModeHint = underway ? 'underway' : 'docking';
      body.active_camera = nextCamera;
      body.mode_hint = nextModeHint;
      body.confidence = night ? Math.max(0.25, cfg.minConfidence - 0.1) : cfg.minConfidence;
    }
    try {
      await client.control(body);
      // Only reflect a camera switch locally once the container accepted it.
      if (nextCamera) {
        activeCamera = nextCamera;
        modeHint = nextModeHint;
      }
    } catch (e) {
      app.debug(`vision-ai: container sync failed: ${e}`);
    }
  }

  let lastMismatchSig: string | null = null;
  // Transition tracking for the container health notifications so we emit on
  // change only, not every 5s poll cycle.
  let containerDownActive = false;
  let lastDegradedSig: string | null = null;

  async function checkHealth(): Promise<void> {
    if (!client || !notifier) return;
    let info: HealthInfo;
    try {
      info = await client.health();
    } catch (e) {
      // Container unreachable supersedes any status-derived alert: clear a
      // previously-latched degraded/label warning so it can't linger
      // contradicting the containerDown alarm through a sustained outage.
      if (lastDegradedSig !== null) {
        lastDegradedSig = null;
        notifier.clearContainerDegraded();
      }
      if (lastMismatchSig !== null) {
        lastMismatchSig = null;
        notifier.clearLabelMismatch();
      }
      // Container unreachable: the vision sensor is offline. Raise once on the
      // down transition; the poll keeps running and clears it on recovery.
      if (!containerDownActive) {
        containerDownActive = true;
        const msg = `Vision container unreachable at ${cfg.containerUrl} (${e}). No detections or video.`;
        notifier.setContainerDown(msg);
        app.error(`vision-ai: ${msg}`);
      }
      return;
    }
    if (containerDownActive) {
      containerDownActive = false;
      notifier.clearContainerDown();
      app.debug('vision-ai: container reachable again');
    }

    // Degraded: container is up but reporting a camera stall or pipeline
    // restart (app/api/rest.py sets status="degraded"). Surface the detail.
    if (info.status === 'degraded') {
      const parts: string[] = [];
      const errs = info.camera_errors ?? {};
      for (const [cam, err] of Object.entries(errs)) parts.push(`${cam}: ${err}`);
      if (info.pipeline_restarts) {
        parts.push(
          `pipeline restarted ${info.pipeline_restarts}x` +
            (info.pipeline_last_error ? ` (${info.pipeline_last_error})` : '')
        );
      }
      const detail = parts.length ? parts.join('; ') : 'unspecified';
      const sig = detail;
      if (sig !== lastDegradedSig) {
        lastDegradedSig = sig;
        const msg = `Vision container degraded — ${detail}.`;
        notifier.setContainerDegraded(msg);
        app.error(`vision-ai: ${msg}`);
      }
    } else if (lastDegradedSig !== null) {
      lastDegradedSig = null;
      notifier.clearContainerDegraded();
    }

    const modelLabels = info.model_labels;
    if (!modelLabels || modelLabels.length === 0) {
      if (lastMismatchSig !== null) {
        lastMismatchSig = null;
        notifier.clearLabelMismatch();
      }
      return; // old container, skip
    }

    const selected = cfg.detectClasses;
    if (selected.length === 0) {
      if (lastMismatchSig !== null) {
        lastMismatchSig = null;
        notifier.clearLabelMismatch();
      }
      return; // "all" — always valid
    }

    const invalid = selected.filter((l) => !modelLabels.includes(l));
    if (invalid.length > 0) {
      const sig = `${info.model ?? 'unknown'}|${invalid.sort().join(',')}`;
      if (sig === lastMismatchSig) return;
      lastMismatchSig = sig;
      const msg =
        `Detection model "${info.model ?? 'unknown'}" cannot produce: ${invalid.join(', ')}. ` +
        `It supports: ${modelLabels.join(', ')}. ` +
        'Uncheck invalid labels in the plugin settings or switch the container model.';
      notifier.setLabelMismatch(msg);
      app.error(`vision-ai: ${msg}`);
    } else if (lastMismatchSig !== null) {
      lastMismatchSig = null;
      notifier.clearLabelMismatch();
    }
  }

  return {
    id: pluginId,
    name: 'Marine Vision-AI',
    description:
      'YOLOv8 visual radar: georeferenced targets, AIS fusion / dark-target ' +
      'alerts, man-overboard and collision (CPA/TCPA) detection.',
    schema,
    uiSchema,

    start(settings: Partial<PluginConfig>) {
      cfg = withDefaults(settings);
      detectionEnabled = cfg.enableDetection;
      maxTargets = cfg.maxTargets;
      client = new ContainerClient(cfg.containerUrl);
      publisher = new Publisher(app, pluginId, cfg);
      notifier = new NotificationManager(app, pluginId, cfg);
      cpa = new CpaEstimator();

      stream = new EventStream(
        client.wsUrl(),
        handleEvent,
        {
          debug: (m, ...a) => app.debug(m, ...a),
          error: (m) => app.error(m),
        },
        (version) => {
          if (!notifier) return;
          if (version === null) {
            notifier.clearSchemaMismatch();
          } else {
            notifier.setSchemaMismatch(
              `Vision container speaks event schema ${version}, but this plugin ` +
              `understands 1.x. Update the plugin or the container so they match — ` +
              `events are being ignored until then.`
            );
          }
        },
        () => cfg.eventMaxAgeS * 1000,
        (stale) => {
          if (!notifier) return;
          if (stale) {
            notifier.setStaleEvents(
              `Vision detection events are arriving older than ${cfg.eventMaxAgeS}s and ` +
              `are being ignored — detection is effectively offline. Check the vision ` +
              `container, the network link, and clock sync between the container and SignalK.`
            );
          } else {
            notifier.clearStaleEvents();
          }
        }
      );
      stream.start();

      lastStatsAt = Date.now();
      processTimer = setInterval(processCycle, cfg.processIntervalMs);
      // Always sync (the object-type selection must reach the container even
      // when context control is off); push once now, then keep it in sync.
      // The container health check (reachability, degraded status, model
      // labels) runs once on start and then every sync cycle.
      void syncContainer();
      void checkHealth();
      syncTimer = setInterval(() => {
        void syncContainer();
        void checkHealth();
      }, 5000);
      app.debug(`vision-ai: started, container=${cfg.containerUrl}`);
    },

    stop() {
      if (processTimer) clearInterval(processTimer);
      if (syncTimer) clearInterval(syncTimer);
      processTimer = syncTimer = null;
      if (stream) stream.stop();
      if (notifier) notifier.clearAll();
      if (publisher) publisher.reset();
      if (cpa) cpa.reset();
      targets.clear();
      aisAssignment = new Map<string, string>();
      lastEventByCamera.clear();
      lastEventTsByCamera.clear();
      frameCount.clear();
      activeCamera = 'forward';
      modeHint = null;
      maxTargets = cfg.maxTargets;
      lastStatsAt = 0;
      lastMismatchSig = null;
      containerDownActive = false;
      lastDegradedSig = null;
      stream = null;
    },

    registerWithRouter(router: any) {
      registerRoutes(router, shared, () => cfg);
    },

    statusMessage() {
      return `${targets.size} targets tracked`;
    },
  };
};
