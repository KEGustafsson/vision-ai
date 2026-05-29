// Marine Vision-AI SignalK plugin entrypoint.
//
// Consumes detection events from the YOLOv8 vision container, enriches them
// with own-ship navigation, fuses with AIS, computes CPA/TCPA, raises
// notifications, and publishes the vision.* tree. Also serves a captain webapp
// and proxies the annotated video stream (see router.ts).

import { ContainerClient } from './containerClient';
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
  let systemTimer: NodeJS.Timeout | null = null;
  let contextTimer: NodeJS.Timeout | null = null;

  const targets = new Map<string, EnrichedTarget>();
  const lastEventByCamera = new Map<string, DetectionEvent>();
  const frameCount = new Map<string, number>();
  let activeCamera = 'forward';

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
      };
    },
    client: () => client,
  };

  function handleEvent(ev: DetectionEvent): void {
    const now = Date.now();
    lastEventByCamera.set(ev.camera, ev);
    frameCount.set(ev.camera, (frameCount.get(ev.camera) ?? 0) + 1);
    const own = readOwnShip(app);

    for (const raw of ev.targets) {
      if (raw.confidence < cfg.minConfidence) continue;
      const t = enrichTarget(raw, ev.camera, own, cfg, now);
      targets.set(t.key, t);
    }

    // Age out stale tracks.
    const timeoutMs = cfg.trackTimeoutS * 1000;
    for (const [key, t] of [...targets]) {
      if (now - t.lastSeen > timeoutMs) targets.delete(key);
    }

    const all = [...targets.values()];

    // AIS fusion.
    let darkKeys = new Set<string>();
    let aisCount = 0;
    if (cfg.enableAisFusion) {
      const contacts = collectAisContacts(app.getPath('vessels'), own);
      const res = fuse(all, contacts, cfg);
      darkKeys = new Set(res.darkTargetKeys);
      aisCount = res.aisCorrelatedCount;
    }

    // Collision risk.
    if (cfg.enableCollision && cpa) cpa.update(all, own, cfg, now);

    // Notifications.
    if (notifier) notifier.evaluate(all, darkKeys, now);

    // Publish.
    if (publisher) {
      publisher.publishTargets(all);
      if (cfg.enableAisFusion) publisher.publishFusionSummary(darkKeys.size, aisCount);
    }
  }

  function publishSystemStats(): void {
    if (!publisher) return;
    const perCameraCounts: Record<string, number> = {};
    for (const cam of lastEventByCamera.keys()) {
      perCameraCounts[cam] = [...targets.values()].filter((t) => t.camera === cam).length;
    }
    // Inference FPS over the ~2s window since last publish.
    let totalFrames = 0;
    for (const [, c] of frameCount) totalFrames += c;
    const fps = totalFrames / 2;
    frameCount.clear();

    const sample = lastEventByCamera.get(activeCamera) ?? [...lastEventByCamera.values()][0];
    publisher.publishSystem({
      activeCamera,
      mode: undefined,
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

  async function contextControl(): Promise<void> {
    if (!cfg.enableContextControl || !client) return;
    const own = readOwnShip(app);
    const underway = (own.sog ?? 0) >= cfg.underwaySogMs;
    const hour = new Date().getHours();
    const night = hour < 6 || hour >= 21;
    // Underway: watch ahead. Low speed (docking/manoeuvring): watch astern.
    activeCamera = underway ? 'forward' : 'aft';
    const confidence = night ? Math.max(0.25, cfg.minConfidence - 0.1) : cfg.minConfidence;
    try {
      await client.control({
        active_camera: activeCamera,
        confidence,
        mode_hint: underway ? 'underway' : 'docking',
      });
    } catch (e) {
      app.debug(`vision-ai: context control failed: ${e}`);
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
      client = new ContainerClient(cfg.containerUrl);
      publisher = new Publisher(app, pluginId, cfg);
      notifier = new NotificationManager(app, pluginId, cfg);
      cpa = new CpaEstimator();

      stream = new EventStream(client.wsUrl(), handleEvent, {
        debug: (m, ...a) => app.debug(m, ...a),
        error: (m) => app.error(m),
      });
      stream.start();

      systemTimer = setInterval(publishSystemStats, 2000);
      if (cfg.enableContextControl) {
        contextTimer = setInterval(() => void contextControl(), 5000);
      }
      app.debug(`vision-ai: started, container=${cfg.containerUrl}`);
    },

    stop() {
      if (systemTimer) clearInterval(systemTimer);
      if (contextTimer) clearInterval(contextTimer);
      systemTimer = contextTimer = null;
      if (stream) stream.stop();
      if (notifier) notifier.clearAll();
      if (publisher) publisher.reset();
      if (cpa) cpa.reset();
      targets.clear();
      lastEventByCamera.clear();
      frameCount.clear();
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
