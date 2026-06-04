import { describe, it, expect, beforeEach } from 'vitest';
import { Publisher } from '../src/publisher';
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
      .filter((c) => c.startsWith('vessels.urn:mrn:signalk:uuid:vision-')))];
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

const VIS1 = 'vessels.urn:mrn:signalk:uuid:vision-forward-1';

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
    expect(vals.find((v) => v.path === 'name')!.value).toBe('VIS-vessel-1');
    // No vision.targets.* data tree on the own context anymore.
    expect(app.valuesFor('vessels.self')).toHaveLength(0);
  });

  it('enriches the blip with SOG / COG / CPA when available', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1, { sog: 3.2, cog: 1.1, cpa: 50, tcpa: 120 })]);

    const vals = app.valuesFor(VIS1);
    expect(vals.find((v) => v.path === 'navigation.speedOverGround')!.value).toBe(3.2);
    expect(vals.find((v) => v.path === 'navigation.courseOverGroundTrue')!.value).toBe(1.1);
    expect(vals.find((v) => v.path === 'navigation.cpa')!.value).toBe(50);
    expect(vals.find((v) => v.path === 'navigation.tcpa')!.value).toBe(120);
  });

  it('omits kinematics when unavailable — an active vessel never carries null data', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1)]); // sog/cog/cpa all null
    const vals = app.valuesFor(VIS1);
    const paths = vals.map((v) => v.path);
    expect(paths).toContain('navigation.position');
    expect(paths).toContain('name');
    expect(paths).not.toContain('navigation.speedOverGround');
    expect(paths).not.toContain('navigation.courseOverGroundTrue');
    expect(paths).not.toContain('navigation.cpa');
    expect(paths).not.toContain('navigation.tcpa');
    // No value written to a live synthetic vessel is ever null.
    expect(vals.every((v) => v.value !== null)).toBe(true);
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

  it('ages out a departed blip by nulling every leaf', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1, { sog: 3, cog: 1, cpa: 40, tcpa: 90 })]);
    app.deltas = [];
    pub.publishTargets([]); // track 1 gone

    const vals = app.valuesFor(VIS1);
    // No synthetic vessel lingers carrying any data without a location.
    for (const path of ['navigation.position', 'name', 'navigation.speedOverGround',
      'navigation.courseOverGroundTrue', 'navigation.cpa', 'navigation.tcpa']) {
      expect(vals.find((v) => v.path === path)!.value).toBeNull();
    }
  });

  it('draws only actively-detected vessels (prunes stale tracks)', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg); // activeMs = processIntervalMs*2
    const now = Date.now();
    pub.publishTargets([
      tgt(1, { lastSeen: now }),               // active → drawn
      tgt(2, { lastSeen: now - 10_000 }),      // stale (still retained for CPA) → not drawn
    ]);
    expect(app.blipContexts()).toEqual(['vessels.urn:mrn:signalk:uuid:vision-forward-1']);
  });

  it('prunes a vessel once its detection stops', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    const now = Date.now();
    pub.publishTargets([tgt(1, { lastSeen: now })]);
    app.deltas = [];
    // Same track still retained in the map, but no longer actively detected.
    pub.publishTargets([tgt(1, { lastSeen: now - 10_000 })]);
    const vals = app.valuesFor(VIS1);
    expect(vals.find((v) => v.path === 'navigation.position')!.value).toBeNull();
    expect(vals.find((v) => v.path === 'name')!.value).toBeNull();
  });

  it('caps blips to maxTargets, keeping the closest', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', { ...cfg, maxTargets: 2 });
    const near = (id: number, range: number) =>
      tgt(id, { geometry: { relative_bearing_deg: 0, range_m: range, range_method: 'horizon', range_confidence: 0.7 } });
    pub.publishTargets([near(1, 300), near(2, 100), near(3, 200)]);

    // Only the two closest (ids 2 @100m and 3 @200m) get a vessel context.
    expect(app.blipContexts().sort()).toEqual([
      'vessels.urn:mrn:signalk:uuid:vision-forward-2',
      'vessels.urn:mrn:signalk:uuid:vision-forward-3',
    ]);
    expect(app.valuesFor('vessels.urn:mrn:signalk:uuid:vision-forward-1')).toHaveLength(0);
  });

  it('retracts all blips on reset', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1)]);
    app.deltas = [];
    pub.reset();
    expect(app.valuesFor(VIS1).find((v) => v.path === 'navigation.position')!.value).toBeNull();
    expect(app.valuesFor(VIS1).find((v) => v.path === 'name')!.value).toBeNull();
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
      if (!d.context.startsWith('vessels.urn:mrn:signalk:uuid:vision-')) continue;
      for (const u of d.updates) {
        for (const v of u.values || []) {
          if (v.path !== 'navigation.position' || v.value === null) continue;
          expect(v.value).toHaveProperty('latitude');
          expect(v.value).toHaveProperty('longitude');
        }
      }
    }
    // Target 2 (never positioned) gets no context at all.
    expect(app.valuesFor('vessels.urn:mrn:signalk:uuid:vision-forward-2')).toHaveLength(0);
  });
});
