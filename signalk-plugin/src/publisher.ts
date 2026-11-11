// Build and emit SignalK deltas: the vision.* telemetry tree (fusion + system
// stats on own context) and one `targets.camera:<id>` sensor target per
// georeferenced camera track (see sensorTargets.ts). A target is never
// published without a real position, and a departed track is withdrawn with a
// null position, which consumers read as "track lost".

import { PluginConfig } from './config';
import {
  displayName,
  overlayNumber,
  sensorTargetContext,
  sensorTargetId,
  sourceLabel,
} from './sensorTargets';
import { ServerApp } from './skapp';
import { EnrichedTarget } from './types';

export class Publisher {
  private metaSent = new Set<string>();
  // Published target contexts -> the camera and the lastSeen of the
  // observation last published, so each detection is published once, stamped
  // with its capture time.
  private published = new Map<string, { camera: string; lastSeen: number }>();

  constructor(
    private app: ServerApp,
    private pluginId: string,
    private cfg: PluginConfig
  ) {}

  private emit(values: Array<{ path: string; value: unknown }>, meta?: Array<{ path: string; value: unknown }>): void {
    if (values.length === 0 && (!meta || meta.length === 0)) return;
    this.app.handleMessage(this.pluginId, {
      context: 'vessels.self',
      updates: [
        {
          source: { label: 'vision-ai' },
          timestamp: new Date().toISOString(),
          values,
          ...(meta && meta.length ? { meta } : {}),
        },
      ],
    });
  }

  publishTargets(targets: EnrichedTarget[], nowMs = Date.now()): void {
    // Single output: each georeferenced track as a sensor target. No
    // vision.targets.* data tree — the captain webapp reads the plugin REST API,
    // and collision/MOB run off the in-memory target set.
    if (this.cfg.enableVisualRadar) this.publishSensorTargets(targets, nowMs);
  }

  publishFusionSummary(darkCount: number, aisCount: number): void {
    this.emit([
      { path: 'vision.fusion.darkTargetCount', value: darkCount },
      { path: 'vision.fusion.aisCorrelatedCount', value: aisCount },
      { path: 'vision.fusion.lastUpdate', value: new Date().toISOString() },
    ]);
  }

  publishSystem(stats: {
    activeCamera?: string;
    mode?: string;
    backend?: string;
    inferenceFps?: number;
    horizonY?: number | null;
    attitudeCompensated?: boolean;
    perCameraCounts?: Record<string, number>;
  }): void {
    const values: Array<{ path: string; value: unknown }> = [];
    const meta: Array<{ path: string; value: unknown }> = [];
    if (stats.activeCamera !== undefined) values.push({ path: 'vision.system.activeCamera', value: stats.activeCamera });
    if (stats.mode !== undefined) values.push({ path: 'vision.system.mode', value: stats.mode });
    if (stats.backend !== undefined) values.push({ path: 'vision.system.backend', value: stats.backend });
    if (stats.horizonY !== undefined) values.push({ path: 'vision.system.horizonY', value: stats.horizonY });
    if (stats.attitudeCompensated !== undefined) {
      values.push({ path: 'vision.system.attitudeCompensated', value: stats.attitudeCompensated });
    }
    if (stats.inferenceFps !== undefined) {
      const path = 'vision.system.inferenceFps';
      values.push({ path, value: stats.inferenceFps });
      if (!this.metaSent.has(path)) {
        this.metaSent.add(path);
        // No zones: zone metadata makes SignalK auto-raise
        // notifications.vision.system.inferenceFps. Keep this a plain telemetry
        // value with units only.
        meta.push({
          path,
          value: {
            units: 'Hz',
            description: 'Vision inference frame rate',
          },
        });
      }
    }
    for (const [cam, count] of Object.entries(stats.perCameraCounts ?? {})) {
      // Camera names are free-form on the wire; a dot/space would corrupt the
      // SignalK path structure, so sanitize the segment like notification keys.
      const seg = cam.replace(/[^a-zA-Z0-9]/g, '_');
      values.push({ path: `vision.${seg}.targetCount`, value: count });
    }
    this.emit(values, meta);
  }

  /**
   * Publish each recently detected, georeferenced track as a sensor target.
   *
   * Observations, not repetitions: a target is published only when the
   * container saw it again since the last publish, stamped with the capture
   * time. Consumers age a target by its updates (Freeboard greys a ring after
   * 20 s, Target Fusion and Collision Alerts ignore one older than a minute),
   * so republishing a held position every cycle would make a boat the cameras
   * lost look live. A coasted frame is the container's prediction across a
   * detector dropout, not a sighting, so it is not published either.
   *
   * A track is held for blipHoldS after its last detection, so a detection gap
   * of a few seconds (waves, occlusion) does not drop the ring and its fusion
   * link. After that its position is published as null once: the convention's
   * "track lost", which removes it from the chart at once and withdraws its
   * fusion link. The server prunes the context itself later.
   */
  private publishSensorTargets(targets: EnrichedTarget[], now: number): void {
    // The floor of two process cycles keeps the gate sane if blipHoldS is
    // misconfigured low.
    const activeMs = Math.max(this.cfg.blipHoldS * 1000, this.cfg.processIntervalMs * 2);
    // Two targets can share an id only on the track_id fallback (an older
    // container without stable_id, whose numbers are recycled). Never let two
    // write one context in a cycle: the most recently seen wins. Dedup BEFORE
    // the maxTargets cap so the cap counts distinct targets.
    const byContext = new Map<string, EnrichedTarget>();
    for (const t of targets) {
      const id = sensorTargetId(t);
      if (!t.position || id === null || now - t.lastSeen > activeMs) continue;
      const context = sensorTargetContext(id);
      const prev = byContext.get(context);
      if (!prev || t.lastSeen > prev.lastSeen) byContext.set(context, t);
    }
    // Cap the chart to maxTargets, keeping the closest (most
    // collision-relevant) so the count tracks the detection limit.
    const eligible = [...byContext.entries()]
      .sort(([, a], [, b]) => (a.geometry.range_m ?? Infinity) - (b.geometry.range_m ?? Infinity))
      .slice(0, this.cfg.maxTargets);
    const current = new Set<string>();
    for (const [context, t] of eligible) {
      current.add(context);
      const publishedAt = this.published.get(context)?.lastSeen;
      if (t.coasting || (publishedAt !== undefined && t.lastSeen <= publishedAt)) continue;
      this.published.set(context, { camera: t.camera, lastSeen: t.lastSeen });
      this.emitTarget(context, t.camera, new Date(t.lastSeen).toISOString(), Publisher.observation(t));
    }
    for (const context of [...this.published.keys()]) {
      if (!current.has(context)) this.withdraw(context);
    }
  }

  // Course and speed are written value-or-null on every observation, so a
  // target whose estimate is lost never keeps a stale course line. The name
  // rides the root delta (path ''), as the convention asks, so consumers read
  // a plain string.
  private static observation(t: EnrichedTarget): Array<{ path: string; value: unknown }> {
    const n = overlayNumber(t);
    return [
      ...(n !== null ? [{ path: '', value: { name: displayName(t.camera, n) } }] : []),
      { path: 'navigation.position', value: t.position },
      { path: 'navigation.speedOverGround', value: t.sog },
      { path: 'navigation.courseOverGroundTrue', value: t.cog },
    ];
  }

  private withdraw(context: string): void {
    const camera = this.published.get(context)?.camera ?? '';
    this.published.delete(context);
    this.emitTarget(context, camera, new Date().toISOString(), [
      { path: 'navigation.position', value: null },
    ]);
  }

  private emitTarget(
    context: string,
    camera: string,
    ts: string,
    values: Array<{ path: string; value: unknown }>
  ): void {
    this.app.handleMessage(this.pluginId, {
      context,
      updates: [{ source: { label: sourceLabel(camera) }, timestamp: ts, values }],
    });
  }

  /** Whether a target context is on the chart: published and not withdrawn. */
  isPublished(context: string): boolean {
    return this.published.has(context);
  }

  /** Withdraw every published target, then clear bookkeeping. Called on plugin
   * stop and when detection is switched off, so no vision contact lingers on a
   * chart nobody is watching. */
  reset(): void {
    for (const context of [...this.published.keys()]) this.withdraw(context);
    this.metaSent.clear();
    this.published.clear();
  }
}
