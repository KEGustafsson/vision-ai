// Build and emit SignalK deltas: the vision.* telemetry tree (fusion + system
// stats on own context) and synthetic AIS vessel blips — one vessels.* contact
// per georeferenced target. A blip is never published without a real position,
// and departed blips are fully retracted.

import { PluginConfig } from './config';
import { ServerApp } from './skapp';
import { EnrichedTarget } from './types';

export class Publisher {
  private metaSent = new Set<string>();
  private publishedBlips = new Set<string>();

  // Every leaf a synthetic vessel can carry; used to fully retract a departed
  // blip so none ever lingers without a real location. position + name identify
  // the contact; the kinematics are published value-or-null each cycle so they
  // never go stale on a live blip.
  private static readonly BLIP_LEAVES = [
    'navigation.position',
    'name',
    'navigation.speedOverGround',
    'navigation.courseOverGroundTrue',
    'navigation.cpa',
    'navigation.tcpa',
  ];

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

  publishTargets(targets: EnrichedTarget[]): void {
    // Single output: render each georeferenced target as a synthetic AIS vessel
    // (chart blip). No vision.targets.* data tree — the captain webapp reads the
    // plugin REST API, and collision/MOB run off the in-memory target set.
    if (this.cfg.enableVisualRadar) this.publishBlips(targets);
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
    perCameraCounts?: Record<string, number>;
  }): void {
    const values: Array<{ path: string; value: unknown }> = [];
    const meta: Array<{ path: string; value: unknown }> = [];
    if (stats.activeCamera !== undefined) values.push({ path: 'vision.system.activeCamera', value: stats.activeCamera });
    if (stats.mode !== undefined) values.push({ path: 'vision.system.mode', value: stats.mode });
    if (stats.backend !== undefined) values.push({ path: 'vision.system.backend', value: stats.backend });
    if (stats.horizonY !== undefined) values.push({ path: 'vision.system.horizonY', value: stats.horizonY });
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
      values.push({ path: `vision.${cam}.targetCount`, value: count });
    }
    this.emit(values, meta);
  }

  /** Render each georeferenced target as a synthetic AIS vessel so it draws as a
   * chart blip. position + name identify it; SOG/COG let the chartplotter draw a
   * vector and compute CPA natively; cpa/tcpa expose our own estimate too. A blip
   * is NEVER created without a real position. */
  private publishBlips(targets: EnrichedTarget[]): void {
    const now = Date.now();
    const ts = new Date(now).toISOString();
    const current = new Set<string>();
    // Draw ONLY actively-detected vessels, so the chart matches the live video.
    // A track retained longer for CPA/notification continuity (trackTimeoutS) is
    // not drawn once it stops being seen; it's pruned within ~2 process cycles.
    // The container coasts through brief flicker, so this won't blink contacts.
    const activeMs = this.cfg.processIntervalMs * 2;
    const eligible = targets
      .filter((t) => t.position && t.track_id !== null && now - t.lastSeen <= activeMs)
      // Cap the chart to maxTargets vessels, keeping the closest (most
      // collision-relevant) so the count tracks the detection limit.
      .sort((a, b) => (a.geometry.range_m ?? Infinity) - (b.geometry.range_m ?? Infinity))
      .slice(0, this.cfg.maxTargets);
    for (const t of eligible) {
      const uuid = `urn:mrn:signalk:uuid:vision-${t.camera}-${t.track_id}`;
      current.add(uuid);
      // position + name always identify the contact. The four kinematics are
      // written value-or-null every cycle: emitting an explicit null when an
      // estimate is lost clears any previously published value, so a live chart
      // contact never carries a stale SOG/COG vector or CPA.
      const values: Array<{ path: string; value: unknown }> = [
        { path: 'navigation.position', value: t.position },
        { path: 'name', value: `VIS-${t.label}-${t.track_id}` },
        { path: 'navigation.speedOverGround', value: t.sog },
        { path: 'navigation.courseOverGroundTrue', value: t.cog },
        { path: 'navigation.cpa', value: t.cpa },
        { path: 'navigation.tcpa', value: t.tcpa },
      ];
      this.emitVessel(uuid, ts, values);
    }
    // Age out departed blips: null every leaf so none lingers without a location.
    for (const uuid of this.publishedBlips) {
      if (current.has(uuid)) continue;
      this.emitVessel(uuid, ts, Publisher.BLIP_LEAVES.map((path) => ({ path, value: null })));
    }
    this.publishedBlips = current;
  }

  private emitVessel(uuid: string, ts: string, values: Array<{ path: string; value: unknown }>): void {
    this.app.handleMessage(this.pluginId, {
      context: `vessels.${uuid}`,
      updates: [{ source: { label: 'vision-ai' }, timestamp: ts, values }],
    });
  }

  /** Retract all synthetic blips, then clear bookkeeping. Called on plugin stop
   * so no stale synthetic vessel lingers. */
  reset(): void {
    const ts = new Date().toISOString();
    for (const uuid of this.publishedBlips) {
      this.emitVessel(uuid, ts, Publisher.BLIP_LEAVES.map((path) => ({ path, value: null })));
    }
    this.metaSent.clear();
    this.publishedBlips.clear();
  }
}
