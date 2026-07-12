// Build and emit SignalK deltas: the vision.* telemetry tree (fusion + system
// stats on own context) and synthetic AIS vessel blips — one vessels.* contact
// per georeferenced target. A blip is never published without a real position,
// and departed blips are fully retracted.

import { PluginConfig } from './config';
import { ServerApp } from './skapp';
import { EnrichedTarget } from './types';

// The blip identity token: readable, deterministic, unique (ids count per
// camera). Doubles as the vessel display name, so context and name can
// never diverge. Camera names are free-form config strings; sanitize the
// segment so a dot/space can't corrupt the context key.
export function blipName(camera: string, trackId: number | string): string {
  return `VIS-${String(camera).replace(/[^a-zA-Z0-9]/g, '_')}-${trackId}`;
}

// The id a blip is keyed and named by: the container's per-session stable_id
// (never recycled) when present, else the 2-digit track_id (older containers).
// Keying on the recycled display number would let a freed number's blip
// identity land on a DIFFERENT physical vessel later in the session.
export function blipId(t: Pick<EnrichedTarget, 'stable_id' | 'track_id'>): number | null {
  return t.stable_id ?? t.track_id;
}

// DELIBERATE spec deviation (owner's call): the SignalK spec wants vessels.*
// keys to be `urn:mrn:imo:mmsi:<9 digits>` or `urn:mrn:signalk:uuid:<UUID v4>`,
// but an opaque hashed UUID makes every retracted blip an anonymous shell in
// vessel lists — a context can never be deleted, so after full retraction the
// uuid is all a client has left to show. A readable token in the uuid slot
// keeps every shell identifiable as ours, is deterministic (the same
// camera/track always maps back to the same context, so a retraction always
// finds the blip it published and a revived track gets its name back), and is
// accepted end-to-end by signalk-server + Freeboard (ran in production before
// and verified live). Strict spec-validating consumers may reject these
// contexts; readability won.
export function blipUrn(camera: string, trackId: number | string): string {
  return `urn:mrn:signalk:uuid:${blipName(camera, trackId)}`;
}

export class Publisher {
  private metaSent = new Set<string>();
  private publishedBlips = new Set<string>();
  // Blips retracted last cycle whose (now all-null) context gets deleted this
  // cycle — see publishBlips for why deletion is deferred by one cycle.
  private pendingDelete = new Set<string>();

  // Every leaf a synthetic vessel carries, built value-or-null in one place so
  // a live blip, a lost-estimate blip and a full retraction always cover the
  // same set and nothing lingers stale — including `name`: a fully-retracted
  // blip carries no data and no name, just its readable context (blipUrn),
  // which is what keeps the shell identifiable (a SignalK context can never be
  // deleted; Freeboard ignores position:null and removes targets only via
  // aisMaxAge). When the track revives, every live cycle republishes the name
  // (derived from the same camera/track token as the context), so the name
  // becomes visible again by itself. `name` rides the empty path (root merge)
  // per the SignalK delta convention for vessel-root attributes, so
  // vessel.name stays the plain string consumers expect in the full model.
  // CPA/TCPA go in the spec's navigation.closestApproach container
  // ({ distance, timeTo }) so chartplotters/MFDs pick them up natively.
  private static blipValues(t: EnrichedTarget | null): Array<{ path: string; value: unknown }> {
    const hasCpa = t !== null && t.cpa !== null && t.tcpa !== null;
    return [
      { path: 'navigation.position', value: t ? t.position : null },
      { path: '', value: { name: t ? blipName(t.camera, blipId(t) as number) : null } },
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
      .filter((t) => t.position && blipId(t) !== null && now - t.lastSeen <= activeMs)
      // Cap the chart to maxTargets vessels, keeping the closest (most
      // collision-relevant) so the count tracks the detection limit.
      .sort((a, b) => (a.geometry.range_m ?? Infinity) - (b.geometry.range_m ?? Infinity))
      .slice(0, this.cfg.maxTargets);
    for (const t of eligible) {
      const uuid = blipUrn(t.camera, blipId(t) as number);
      current.add(uuid);
      // position + name always identify the contact. The kinematics are written
      // value-or-null every cycle: emitting an explicit null when an estimate is
      // lost clears any previously published value, so a live chart contact
      // never carries a stale SOG/COG vector or CPA.
      this.emitVessel(uuid, ts, Publisher.blipValues(t));
    }
    // Delete contexts retracted LAST cycle (unless the track revived since):
    // one cycle in between guarantees the all-null retraction delta has been
    // processed and forwarded to live subscribers before the vessel entry
    // disappears from the model, regardless of handleMessage's internal timing.
    for (const uuid of this.pendingDelete) {
      if (!current.has(uuid)) this.deleteVessel(uuid);
    }
    this.pendingDelete.clear();
    // Age out departed blips: null every leaf so none lingers without a
    // location, then schedule the emptied shell for deletion next cycle.
    for (const uuid of this.publishedBlips) {
      if (current.has(uuid)) continue;
      this.emitVessel(uuid, ts, Publisher.blipValues(null));
      this.pendingDelete.add(uuid);
    }
    this.publishedBlips = current;
  }

  private emitVessel(uuid: string, ts: string, values: Array<{ path: string; value: unknown }>): void {
    this.app.handleMessage(this.pluginId, {
      context: `vessels.${uuid}`,
      updates: [{ source: { label: 'vision-ai' }, timestamp: ts, values }],
    });
  }

  /** Remove the synthetic vessel's context entirely, the same way the server's
   * own pruneContextsMinutes sweep does. Undocumented internals, so fail soft:
   * if either hook is missing the shell just lingers (all-null, readable
   * context) until the server age-prunes it — the pre-deletion behavior. */
  private deleteVessel(uuid: string): void {
    const key = `vessels.${uuid}`;
    try {
      this.app.signalk?.deleteContext?.(key);
      this.app.deltaCache?.deleteContext?.(key);
    } catch (e) {
      this.app.debug(`deleteContext failed for ${key}: ${e}`);
    }
  }

  /** Retract all synthetic blips, then clear bookkeeping. Called on plugin stop
   * so no stale synthetic vessel lingers. Contexts are deleted immediately —
   * there is no next cycle to defer to; if the null delta lands after the
   * delete and resurrects an all-null shell, the server's prune sweep is the
   * backstop. */
  reset(): void {
    const ts = new Date().toISOString();
    for (const uuid of this.publishedBlips) {
      this.emitVessel(uuid, ts, Publisher.blipValues(null));
      this.deleteVessel(uuid);
    }
    for (const uuid of this.pendingDelete) this.deleteVessel(uuid);
    this.metaSent.clear();
    this.publishedBlips.clear();
    this.pendingDelete.clear();
  }
}
