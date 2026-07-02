// Build and emit SignalK deltas: the vision.* telemetry tree (fusion + system
// stats on own context) and synthetic AIS vessel blips — one vessels.* contact
// per georeferenced target. A blip is never published without a real position,
// and departed blips are fully retracted.

import { createHash } from 'crypto';
import { PluginConfig } from './config';
import { ServerApp } from './skapp';
import { EnrichedTarget } from './types';

// The SignalK spec constrains vessels.* keys to `urn:mrn:imo:mmsi:<9 digits>`
// or `urn:mrn:signalk:uuid:<UUID v4>` (version-4/variant bits enforced by the
// schema regex), so a readable token like "vision-forward-3" is rejected by
// strict consumers. Hash the camera/track key into an RFC-4122-shaped UUID
// instead: deterministic, so the same track always maps to the same context
// and a retraction always finds the blip it published.
export function blipUrn(camera: string, trackId: number | string): string {
  // sha256 (not sha1/md5) purely to stay off SAST weak-hash lists; this is
  // non-cryptographic ID derivation and only the first 16 bytes are used.
  const h = createHash('sha256').update(`signalk-vision-ai:${camera}:${trackId}`).digest();
  h[6] = (h[6] & 0x0f) | 0x40; // version nibble = 4 (required by the spec regex)
  h[8] = (h[8] & 0x3f) | 0x80; // RFC 4122 variant
  const x = h.subarray(0, 16).toString('hex');
  return (
    'urn:mrn:signalk:uuid:' +
    `${x.slice(0, 8)}-${x.slice(8, 12)}-${x.slice(12, 16)}-${x.slice(16, 20)}-${x.slice(20, 32)}`
  );
}

export class Publisher {
  private metaSent = new Set<string>();
  private publishedBlips = new Set<string>();

  // Every DATA leaf a synthetic vessel carries, built value-or-null in one
  // place so a live blip, a lost-estimate blip and a full retraction always
  // cover the same set and no kinematics linger stale. `name` is identity, not
  // data: it is set on every live cycle and NEVER nulled, exactly like a real
  // AIS contact that stops transmitting — it keeps its name at its last
  // position until the chartplotter ages it out (Freeboard ignores
  // position:null and removes targets only via aisMaxAge, default 9 min).
  // Nulling the name instead leaves anonymous uuid shells in vessel lists (a
  // SignalK context can never be deleted), which reads as a nameless vessel
  // receiving data whenever its track revives. Blip flicker that once made
  // name-kept ghosts litter the chart is gone since blipHoldS. `name` rides
  // the empty path (root merge) per the SignalK delta convention for
  // vessel-root attributes, so vessel.name stays the plain string consumers
  // expect in the full model. CPA/TCPA go in the spec's
  // navigation.closestApproach container ({ distance, timeTo }) so
  // chartplotters/MFDs pick them up natively.
  private static blipValues(t: EnrichedTarget | null): Array<{ path: string; value: unknown }> {
    const hasCpa = t !== null && t.cpa !== null && t.tcpa !== null;
    return [
      { path: 'navigation.position', value: t ? t.position : null },
      // Track IDs count independently per camera, so the camera (not the class
      // label) is what makes the display name unique on the chart.
      ...(t ? [{ path: '', value: { name: `VIS-${t.camera}-${t.track_id}` } }] : []),
      { path: 'navigation.speedOverGround', value: t ? t.sog : null },
      { path: 'navigation.courseOverGroundTrue', value: t ? t.cog : null },
      {
        path: 'navigation.closestApproach',
        value: hasCpa ? { distance: t.cpa, timeTo: t.tcpa } : null,
      },
    ];
  }

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
      // Camera names are free-form on the wire; a dot/space would corrupt the
      // SignalK path structure, so sanitize the segment like notification keys.
      const seg = cam.replace(/[^a-zA-Z0-9]/g, '_');
      values.push({ path: `vision.${seg}.targetCount`, value: count });
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
    // Draw recently-detected vessels, holding each blip blipHoldS after its
    // last detection. Detection gaps of several seconds are routine (waves,
    // occlusion — measured 8-9 s on live water), and retracting on them blinks
    // the contact and strips its name label right when a chartplotter user is
    // looking at it. Real AIS reports every 10-30 s, so a held blip with a
    // slightly stale position is in character for a chart. The floor of two
    // process cycles keeps the gate sane if blipHoldS is misconfigured low.
    const activeMs = Math.max(this.cfg.blipHoldS * 1000, this.cfg.processIntervalMs * 2);
    const eligible = targets
      .filter((t) => t.position && t.track_id !== null && now - t.lastSeen <= activeMs)
      // Cap the chart to maxTargets vessels, keeping the closest (most
      // collision-relevant) so the count tracks the detection limit.
      .sort((a, b) => (a.geometry.range_m ?? Infinity) - (b.geometry.range_m ?? Infinity))
      .slice(0, this.cfg.maxTargets);
    for (const t of eligible) {
      const uuid = blipUrn(t.camera, t.track_id as number);
      current.add(uuid);
      // position + name always identify the contact. The kinematics are written
      // value-or-null every cycle: emitting an explicit null when an estimate is
      // lost clears any previously published value, so a live chart contact
      // never carries a stale SOG/COG vector or CPA.
      this.emitVessel(uuid, ts, Publisher.blipValues(t));
    }
    // Age out departed blips: null every leaf so none lingers without a location.
    for (const uuid of this.publishedBlips) {
      if (current.has(uuid)) continue;
      this.emitVessel(uuid, ts, Publisher.blipValues(null));
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
      this.emitVessel(uuid, ts, Publisher.blipValues(null));
    }
    this.metaSent.clear();
    this.publishedBlips.clear();
  }
}
