import { describe, it, expect, beforeEach } from 'vitest';
import { Publisher, blipUrn } from '../src/publisher';
import { withDefaults } from '../src/config';
import { Delta, ServerApp } from '../src/skapp';
import { EnrichedTarget } from '../src/types';

class FakeApp implements ServerApp {
  deltas: Delta[] = [];
  handleMessage(_id: string, delta: Delta): void { this.deltas.push(delta); }
  getSelfPath(): any { return null; }
  getPath(): any { return null; }
  debug(): void {}
  error(): void {}

  values(): Array<{ path: string; value: unknown }> {
    return this.deltas.flatMap((d) => d.updates.flatMap((u) => u.values || []));
  }
  // Values for a single context (e.g. a synthetic vessel's vessels.<uuid>).
  valuesFor(context: string): Array<{ path: string; value: unknown }> {
    return this.deltas
      .filter((d) => d.context === context)
      .flatMap((d) => d.updates.flatMap((u) => u.values || []));
  }
  blipContexts(): string[] {
    return [...new Set(this.deltas
      .map((d) => d.context)
      .filter((c) => c !== 'vessels.self' && c.startsWith('vessels.urn:mrn:signalk:uuid:')))];
  }
}

function tgt(id: number, partial: Partial<EnrichedTarget> = {}): EnrichedTarget {
  return {
    track_id: id, label: 'vessel', coco_class: 8, confidence: 0.9,
    bbox: { x: 0, y: 0, w: 1, h: 1 }, is_person_in_water: false,
    geometry: { relative_bearing_deg: 0, range_m: 500, range_method: 'horizon', range_confidence: 0.7 },
    pixel_velocity: { vx: 0, vy: 0 }, first_seen: null, age_frames: 0,
    key: `forward.${id}`, camera: 'forward', bearingTrue: 0.1,
    position: { latitude: 60, longitude: 25 },
    aisCorrelated: false, aisMmsi: null, aisCog: null, aisSog: null,
    cpa: null, tcpa: null, sog: null, cog: null, threatLevel: 'none', lastSeen: Date.now(),
    ...partial,
  };
}

const ctx = (camera: string, id: number) => `vessels.${blipUrn(camera, id)}`;
const VIS1 = ctx('forward', 1);
// The name rides the empty path (root merge) per the SignalK convention for
// vessel-root attributes, so match on the object value.
const nameOf = (vals: Array<{ path: string; value: unknown }>): unknown =>
  (vals.find((v) => v.path === '')?.value as { name?: unknown } | undefined)?.name;

describe('blipUrn', () => {
  it('produces a readable, deterministic context token per camera/track', () => {
    // Deliberate spec deviation: a readable token in the uuid slot keeps a
    // fully-retracted shell identifiable as ours (contexts cannot be deleted).
    expect(blipUrn('forward', 1)).toBe('urn:mrn:signalk:uuid:VIS-forward-1');
    expect(blipUrn('forward', 1)).toBe(blipUrn('forward', 1)); // deterministic
    expect(blipUrn('forward', 1)).not.toBe(blipUrn('forward', 2));
    expect(blipUrn('forward', 1)).not.toBe(blipUrn('aft', 1));
    // Free-form camera names are sanitized so they can't corrupt the context.
    expect(blipUrn('bow cam.2', 7)).toBe('urn:mrn:signalk:uuid:VIS-bow_cam_2-7');
  });
});

describe('Publisher', () => {
  let app: FakeApp;
  // Single output toggle is on for most cases; default config has it off.
  const cfg = withDefaults({ enableVisualRadar: true });

  beforeEach(() => {
    app = new FakeApp();
  });

  it('publishes nothing when the output toggle is off (default)', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', withDefaults(undefined));
    pub.publishTargets([tgt(1)]);
    expect(app.deltas).toHaveLength(0);
  });

  it('projects a positioned target as a synthetic AIS vessel', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1)]);

    const vals = app.valuesFor(VIS1);
    expect(vals.find((v) => v.path === 'navigation.position')!.value).toEqual({ latitude: 60, longitude: 25 });
    expect(nameOf(vals)).toBe('VIS-forward-1');
    // No vision.targets.* data tree on the own context anymore.
    expect(app.valuesFor('vessels.self')).toHaveLength(0);
  });

  it('enriches the blip with SOG / COG / closestApproach when available', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1, { sog: 3.2, cog: 1.1, cpa: 50, tcpa: 120 })]);

    const vals = app.valuesFor(VIS1);
    expect(vals.find((v) => v.path === 'navigation.speedOverGround')!.value).toBe(3.2);
    expect(vals.find((v) => v.path === 'navigation.courseOverGroundTrue')!.value).toBe(1.1);
    // CPA/TCPA go in the spec's navigation.closestApproach container.
    expect(vals.find((v) => v.path === 'navigation.closestApproach')!.value)
      .toEqual({ distance: 50, timeTo: 120 });
  });

  it('writes kinematics as null when unavailable so stale values never linger', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1)]); // sog/cog/cpa all null
    const vals = app.valuesFor(VIS1);
    // position + name are always real; the kinematics are emitted as explicit
    // null so a chartplotter clears any previously published vector.
    expect(vals.find((v) => v.path === 'navigation.position')!.value).toEqual({ latitude: 60, longitude: 25 });
    expect(nameOf(vals)).toBe('VIS-forward-1');
    expect(vals.find((v) => v.path === 'navigation.speedOverGround')!.value).toBeNull();
    expect(vals.find((v) => v.path === 'navigation.courseOverGroundTrue')!.value).toBeNull();
    expect(vals.find((v) => v.path === 'navigation.closestApproach')!.value).toBeNull();
  });

  it('clears a stale kinematic when a live blip loses its estimate', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1, { sog: 3.2, cog: 1.1, cpa: 50, tcpa: 120 })]);
    app.deltas = [];
    // Same track still actively detected, but the velocity estimate is gone.
    pub.publishTargets([tgt(1)]);
    const vals = app.valuesFor(VIS1);
    expect(vals.find((v) => v.path === 'navigation.position')!.value).toEqual({ latitude: 60, longitude: 25 });
    expect(vals.find((v) => v.path === 'navigation.speedOverGround')!.value).toBeNull();
    expect(vals.find((v) => v.path === 'navigation.courseOverGroundTrue')!.value).toBeNull();
    expect(vals.find((v) => v.path === 'navigation.closestApproach')!.value).toBeNull();
  });

  it('publishes half a CPA pair as unresolved (never a partial closestApproach)', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1, { cpa: 50, tcpa: null })]);
    expect(app.valuesFor(VIS1).find((v) => v.path === 'navigation.closestApproach')!.value).toBeNull();
    app.deltas = [];
    pub.publishTargets([tgt(1, { cpa: null, tcpa: 120 })]);
    expect(app.valuesFor(VIS1).find((v) => v.path === 'navigation.closestApproach')!.value).toBeNull();
  });

  it('publishes fusion summary counts', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishFusionSummary(2, 3);
    const v = app.values();
    expect(v.find((x) => x.path === 'vision.fusion.darkTargetCount')!.value).toBe(2);
    expect(v.find((x) => x.path === 'vision.fusion.aisCorrelatedCount')!.value).toBe(3);
  });

  it('skips targets without a georeferenced position', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1, { position: null })]);
    expect(app.blipContexts()).toHaveLength(0);
  });

  it('ages out a departed blip by nulling every data leaf', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1, { sog: 3, cog: 1, cpa: 40, tcpa: 90 })]);
    app.deltas = [];
    pub.publishTargets([]); // track 1 gone

    const vals = app.valuesFor(VIS1);
    // No synthetic vessel lingers carrying any data without a location.
    for (const path of ['navigation.position', 'navigation.speedOverGround',
      'navigation.courseOverGroundTrue', 'navigation.closestApproach']) {
      expect(vals.find((v) => v.path === path)!.value).toBeNull();
    }
    // name is nulled too — a retracted shell carries no data and no name; its
    // readable context token is what keeps it identifiable, and a revived
    // track republishes the (context-derived) name every live cycle.
    expect(nameOf(vals)).toBeNull();
  });

  it('draws recently-detected vessels, holding through gaps up to blipHoldS', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg); // activeMs = blipHoldS (15 s)
    const now = Date.now();
    pub.publishTargets([
      tgt(1, { lastSeen: now }),               // active → drawn
      tgt(2, { lastSeen: now - 10_000 }),      // detection gap < hold → still drawn
      tgt(3, { lastSeen: now - 20_000 }),      // beyond hold → not drawn
    ]);
    expect(app.blipContexts().sort()).toEqual([ctx('forward', 1), ctx('forward', 2)].sort());
  });

  it('prunes a vessel once its detection stops', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    const now = Date.now();
    pub.publishTargets([tgt(1, { lastSeen: now })]);
    app.deltas = [];
    // Same track still retained in the map, but not detected for > blipHoldS.
    pub.publishTargets([tgt(1, { lastSeen: now - 20_000 })]);
    const vals = app.valuesFor(VIS1);
    expect(vals.find((v) => v.path === 'navigation.position')!.value).toBeNull();
    expect(nameOf(vals)).toBeNull(); // fully nulled; context stays readable
  });

  it('brings the name back when data resumes after a full retraction', () => {
    // INVARIANT: data is never published over a nulled blip without its name.
    // Every live cycle carries name + data in the same update, so a revived
    // track re-names its vessel in the very delta that resumes its data.
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    const now = Date.now();
    pub.publishTargets([tgt(1, { lastSeen: now })]);
    pub.publishTargets([tgt(1, { lastSeen: now - 20_000 })]); // gone → all null
    app.deltas = [];
    pub.publishTargets([tgt(1, { lastSeen: Date.now() })]); // track revives
    const vals = app.valuesFor(VIS1);
    expect(nameOf(vals)).toBe('VIS-forward-1');
    expect(vals.find((v) => v.path === 'navigation.position')!.value).toEqual({ latitude: 60, longitude: 25 });
    // name and position arrive in the SAME update — no window where a client
    // can see data on a nameless vessel.
    const revival = app.deltas.find((d) => d.context === VIS1)!;
    const paths = revival.updates[0].values!.map((v) => v.path);
    expect(paths).toContain('');
    expect(paths).toContain('navigation.position');
  });

  it('caps blips to maxTargets, keeping the closest', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', { ...cfg, maxTargets: 2 });
    const near = (id: number, range: number) =>
      tgt(id, { geometry: { relative_bearing_deg: 0, range_m: range, range_method: 'horizon', range_confidence: 0.7 } });
    pub.publishTargets([near(1, 300), near(2, 100), near(3, 200)]);

    // Only the two closest (ids 2 @100m and 3 @200m) get a vessel context.
    expect(app.blipContexts().sort()).toEqual([ctx('forward', 2), ctx('forward', 3)].sort());
    expect(app.valuesFor(ctx('forward', 1))).toHaveLength(0);
  });

  it('retracts all blips on reset', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1)]);
    app.deltas = [];
    pub.reset();
    expect(app.valuesFor(VIS1).find((v) => v.path === 'navigation.position')!.value).toBeNull();
    expect(nameOf(app.valuesFor(VIS1))).toBeNull();
  });

  it('never writes a synthetic vessel that lacks a real position', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    // Mix of positioned and position-less targets across two cycles, including a
    // target that loses its position while still being tracked.
    pub.publishTargets([tgt(1), tgt(2, { position: null })]);
    pub.publishTargets([tgt(1, { position: null }), tgt(3)]);

    // Every non-null navigation.position ever written to a vision blip context
    // must carry a real lat/lon — a blip is never created location-less.
    for (const d of app.deltas) {
      if (d.context === 'vessels.self' || !d.context.startsWith('vessels.urn:mrn:signalk:uuid:')) continue;
      for (const u of d.updates) {
        for (const v of u.values || []) {
          if (v.path !== 'navigation.position' || v.value === null) continue;
          expect(v.value).toHaveProperty('latitude');
          expect(v.value).toHaveProperty('longitude');
        }
      }
    }
    // Target 2 (never positioned) gets no context at all.
    expect(app.valuesFor(ctx('forward', 2))).toHaveLength(0);
  });
});
