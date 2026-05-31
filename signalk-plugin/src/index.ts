// Marine Vision-AI SignalK plugin entrypoint.
//
// Consumes detection events from the YOLOv8 vision container, enriches them
// with own-ship navigation, fuses with AIS, computes CPA/TCPA, raises
// notifications, and publishes the vision.* tree. Also serves a captain webapp
// and proxies the annotated video stream (see router.ts).

import { ContainerClient, ControlBody } from './containerClient';
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
  const lastEventByCamera = new Map<string, DetectionEvent>();
  const frameCount = new Map<string, number>();
  let activeCamera = 'forward';
  let modeHint: string | null = null;
  let lastStatsAt = 0;
  // Live master on/off. Seeded from the persisted config in start(), but the
  // captain webapp can flip it at runtime (see router POST /detection). Held
  // here (not in cfg) so a live toggle isn't reverted by the next sync.
  let detectionEnabled = true;

  const shared: SharedState = {
    get targets() {
      return [...targets.values()];
    },
    get ownShip() {
      return readOwnShip(app);
    },
    get system() {
      return {
        activeCamera,
        cameras: [...lastEventByCamera.keys()],
        detectionEnabled,
      };
    },
    setDetection(on: boolean) {
      detectionEnabled = on;
      // Push immediately so the webapp toggle takes effect without waiting for
      // the next periodic sync.
      void syncContainer();
    },
    client: () => client,
  };

  // --- Ingest: cheap, runs per WebSocket frame ---
  // Only update the target map here; the heavy fusion/CPA/notify/publish work
  // is debounced to a fixed cadence in processCycle() so it doesn't scale with
  // (and flap at) the camera frame rate. Only tracked detections are kept —
  // untracked boxes (track_id null) have no stable key, so they cannot be
  // associated across frames for CPA/MOB persistence and would churn the map.
  function handleEvent(ev: DetectionEvent): void {
    const now = Date.now();
    lastEventByCamera.set(ev.camera, ev);
    frameCount.set(ev.camera, (frameCount.get(ev.camera) ?? 0) + 1);
    const own = readOwnShip(app);

    for (const raw of ev.targets) {
      if (raw.confidence < cfg.minConfidence) continue;
      if (raw.track_id === null) continue;
      const t = enrichTarget(raw, ev.camera, own, cfg, now);
      targets.set(t.key, t);
    }
  }

  // --- Process: fixed cadence, does the expensive correlated work ---
  function processCycle(): void {
    const now = Date.now();
    const own = readOwnShip(app);

    // Age out stale tracks first so downstream sees only live targets.
    const timeoutMs = cfg.trackTimeoutS * 1000;
    for (const [key, t] of [...targets]) {
      if (now - t.lastSeen > timeoutMs) targets.delete(key);
    }
    const all = [...targets.values()];

    let darkKeys = new Set<string>();
    let aisCount = 0;
    if (cfg.enableAisFusion) {
      const contacts = collectAisContacts(app.getPath('vessels'), own);
      const res = fuse(all, contacts, cfg);
      darkKeys = new Set(res.darkTargetKeys);
      aisCount = res.aisCorrelatedCount;
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
    const body: ControlBody = { labels: cfg.detectClasses, enabled: detectionEnabled };
    let nextCamera: string | null = null;
    let nextModeHint: string | null = null;
    if (cfg.enableContextControl) {
      const own = readOwnShip(app);
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
      client = new ContainerClient(cfg.containerUrl);
      publisher = new Publisher(app, pluginId, cfg);
      notifier = new NotificationManager(app, pluginId, cfg);
      cpa = new CpaEstimator();

      stream = new EventStream(client.wsUrl(), handleEvent, {
        debug: (m, ...a) => app.debug(m, ...a),
        error: (m) => app.error(m),
      });
      stream.start();

      lastStatsAt = Date.now();
      processTimer = setInterval(processCycle, cfg.processIntervalMs);
      // Always sync (the object-type selection must reach the container even
      // when context control is off); push once now, then keep it in sync.
      void syncContainer();
      syncTimer = setInterval(() => void syncContainer(), 5000);
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
      lastEventByCamera.clear();
      frameCount.clear();
      activeCamera = 'forward';
      modeHint = null;
      lastStatsAt = 0;
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
