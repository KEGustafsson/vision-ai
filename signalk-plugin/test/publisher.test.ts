import { describe, it, expect, beforeEach } from 'vitest';
import { Publisher } from '../src/publisher';
import { sensorTargetId } from '../src/sensorTargets';
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
  valuesFor(context: string): Array<{ path: string; value: unknown }> {
    return this.deltas
      .filter((d) => d.context === context)
      .flatMap((d) => d.updates.flatMap((u) => u.values || []));
  }
  updatesFor(context: string): Delta['updates'] {
    return this.deltas.filter((d) => d.context === context).flatMap((d) => d.updates);
  }
  targetContexts(): string[] {
    return [...new Set(this.deltas
      .map((d) => d.context as string)
      .filter((c) => c.startsWith('targets.')))];
  }
}

const NOW = Date.parse('2026-11-11T12:00:00.000Z');

function tgt(id: number, partial: Partial<EnrichedTarget> = {}): EnrichedTarget {
  return {
    track_id: id, stable_id: id + 100, label: 'vessel', coco_class: 8, confidence: 0.9,
    bbox: { x: 0, y: 0, w: 1, h: 1 }, is_person_in_water: false,
    geometry: { relative_bearing_deg: 0, range_m: 500, range_method: 'horizon', range_confidence: 0.7 },
    pixel_velocity: { vx: 0, vy: 0 }, first_seen: null, age_frames: 0,
    key: `forward.${id + 100}`, camera: 'forward', bearingTrue: 0.1,
    position: { latitude: 60, longitude: 25 },
    aisCorrelated: false, aisMmsi: null, aisCog: null, aisSog: null, aisPosition: null,
    cpa: null, tcpa: null, sog: null, cog: null, threatLevel: 'none', lastSeen: NOW,
    ...partial,
  };
}

const ctx = (camera: string, stableId: number) => `targets.camera:visionai-${camera}-${stableId}`;
const T1 = ctx('forward', 101);
const nameOf = (vals: Array<{ path: string; value: unknown }>): unknown =>
  (vals.find((v) => v.path === '')?.value as { name?: unknown } | undefined)?.name;
const positionsOf = (vals: Array<{ path: string; value: unknown }>) =>
  vals.filter((v) => v.path === 'navigation.position').map((v) => v.value);

describe('sensorTargetId', () => {
  it('is a camera target unique to this plugin, keyed on the never-recycled stable_id', () => {
    expect(sensorTargetId({ camera: 'forward', stable_id: 137, track_id: 14 }))
      .toBe('camera:visionai-forward-137');
    // Older containers without stable_id fall back to the track number.
    expect(sensorTargetId({ camera: 'aft', stable_id: undefined, track_id: 14 }))
      .toBe('camera:visionai-aft-14');
    expect(sensorTargetId({ camera: 'aft', stable_id: null, track_id: null })).toBeNull();
  });

  it('keeps free-form camera names from splitting the context', () => {
    expect(sensorTargetId({ camera: 'bow cam.2', stable_id: 7, track_id: 7 }))
      .toBe('camera:visionai-bow_20_cam_2e_2-7');
  });

  it('never gives two cameras one context, whatever their names', () => {
    const names = ['bow cam', 'bow.cam', 'bow_cam', 'bow-cam', 'bow_20_cam', 'bowcam'];
    const ids = names.map((camera) => sensorTargetId({ camera, stable_id: 1, track_id: 1 }));
    expect(new Set(ids).size).toBe(names.length);
    for (const id of ids) expect(id).not.toContain('.');
  });
});

describe('Publisher — sensor targets', () => {
  let app: FakeApp;
  const cfg = withDefaults({ enableVisualRadar: true });

  beforeEach(() => {
    app = new FakeApp();
  });

  it('publishes nothing when the output toggle is off (default)', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', withDefaults(undefined));
    pub.publishTargets([tgt(1)], NOW);
    expect(app.deltas).toHaveLength(0);
  });

  it('publishes a positioned track as a camera target, not as a vessel', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1, { sog: 3.2, cog: 1.1 })], NOW);
    expect(app.targetContexts()).toEqual([T1]);
    expect(app.deltas.some((d) => String(d.context).startsWith('vessels.'))).toBe(false);
    const vals = app.valuesFor(T1);
    expect(positionsOf(vals)).toEqual([{ latitude: 60, longitude: 25 }]);
    expect(vals.find((v) => v.path === 'navigation.speedOverGround')!.value).toBe(3.2);
    expect(vals.find((v) => v.path === 'navigation.courseOverGroundTrue')!.value).toBe(1.1);
    // Not part of the convention, and consumers ignore it on targets.
    expect(vals.some((v) => v.path === 'navigation.closestApproach')).toBe(false);
  });

  it('names the target with the overlay number, as a root delta', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(14, { stable_id: 137 })], NOW);
    expect(nameOf(app.valuesFor(ctx('forward', 137)))).toBe('VIS-forward-14');
  });

  it('stamps each observation with its capture time and a per-camera source', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    const seen = NOW - 700;
    pub.publishTargets([tgt(1, { lastSeen: seen }), tgt(2, { camera: 'aft', lastSeen: seen })], NOW);
    const [update] = app.updatesFor(T1);
    expect(update.timestamp).toBe(new Date(seen).toISOString());
    expect(update.source).toEqual({ label: 'vision-ai-forward' });
    expect(app.updatesFor(ctx('aft', 102))[0].source).toEqual({ label: 'vision-ai-aft' });
  });

  it('publishes each sighting once, not again every cycle while it is held', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    const t = tgt(1, { lastSeen: NOW });
    pub.publishTargets([t], NOW);
    pub.publishTargets([t], NOW + 1000);
    pub.publishTargets([t], NOW + 2000);
    expect(positionsOf(app.valuesFor(T1))).toHaveLength(1);
    // A new sighting is published.
    pub.publishTargets([{ ...t, lastSeen: NOW + 2500 }], NOW + 3000);
    expect(positionsOf(app.valuesFor(T1))).toHaveLength(2);
  });

  it('does not publish a coasted frame as a sighting', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1, { coasting: true })], NOW);
    expect(app.deltas).toHaveLength(0);
    pub.publishTargets([tgt(1, { coasting: false, lastSeen: NOW + 500 })], NOW + 1000);
    expect(positionsOf(app.valuesFor(T1))).toHaveLength(1);
  });

  it('ages the hold from the last real detection, not from coasted frames', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg); // blipHoldS 15 s
    pub.publishTargets([tgt(1, { observedAt: NOW })], NOW);
    // The container keeps coasting the track: lastSeen advances, observedAt stays.
    const coasted = tgt(1, { coasting: true, observedAt: NOW, lastSeen: NOW + 14_000 });
    pub.publishTargets([coasted], NOW + 14_000);
    expect(pub.isPublished(T1)).toBe(true);
    pub.publishTargets([{ ...coasted, lastSeen: NOW + 16_000 }], NOW + 16_000);
    expect(pub.isPublished(T1)).toBe(false);
    expect(app.valuesFor(T1).at(-1)).toEqual({ path: 'navigation.position', value: null });
  });

  it('gives a coasted track not yet on the chart no maxTargets slot', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', { ...cfg, maxTargets: 1 });
    const near = tgt(1, { coasting: true, observedAt: NOW - 500,
      geometry: { relative_bearing_deg: 0, range_m: 100, range_method: 'horizon', range_confidence: 0.7 } });
    const far = tgt(2, { geometry: { relative_bearing_deg: 0, range_m: 900, range_method: 'horizon', range_confidence: 0.7 } });
    pub.publishTargets([near, far], NOW);
    expect(app.targetContexts()).toEqual([ctx('forward', 102)]);
  });

  it('never publishes a track that has only been coasted', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1, { coasting: true, observedAt: undefined })], NOW);
    expect(app.deltas).toHaveLength(0);
  });

  it('writes course and speed as null when lost, so no stale course line lingers', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1, { sog: 3, cog: 1 })], NOW);
    app.deltas = [];
    pub.publishTargets([tgt(1, { lastSeen: NOW + 1000 })], NOW + 1000);
    const vals = app.valuesFor(T1);
    expect(vals.find((v) => v.path === 'navigation.speedOverGround')!.value).toBeNull();
    expect(vals.find((v) => v.path === 'navigation.courseOverGroundTrue')!.value).toBeNull();
  });

  it('holds a target through a detection gap up to blipHoldS', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg); // blipHoldS 15 s
    pub.publishTargets([tgt(1)], NOW);
    app.deltas = [];
    pub.publishTargets([tgt(1)], NOW + 10_000);
    expect(app.deltas).toHaveLength(0);
    expect(pub.isPublished(T1)).toBe(true);
  });

  it('withdraws a departed track with a null position, once', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1, { sog: 3, cog: 1 })], NOW);
    app.deltas = [];
    pub.publishTargets([tgt(1)], NOW + 20_000); // beyond the hold
    // Only the position: the convention's "track lost".
    expect(app.valuesFor(T1)).toEqual([{ path: 'navigation.position', value: null }]);
    expect(app.updatesFor(T1)[0].source).toEqual({ label: 'vision-ai-forward' });
    expect(pub.isPublished(T1)).toBe(false);
    pub.publishTargets([], NOW + 21_000);
    expect(app.valuesFor(T1)).toHaveLength(1);
  });

  it('skips targets without a georeferenced position', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1, { position: null })], NOW);
    expect(app.targetContexts()).toHaveLength(0);
  });

  it('withdraws a published target that loses its position', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1)], NOW);
    app.deltas = [];
    pub.publishTargets([tgt(1, { position: null, lastSeen: NOW + 1000 })], NOW + 1000);
    expect(app.valuesFor(T1)).toEqual([{ path: 'navigation.position', value: null }]);
  });

  it('never lets two targets sharing an id write one context in a cycle', () => {
    // Only on the track_id fallback (no stable_id), whose numbers recycle.
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    const old = tgt(14, { stable_id: undefined, lastSeen: NOW - 5000, position: { latitude: 59, longitude: 24 } });
    const fresh = tgt(14, { stable_id: undefined, lastSeen: NOW });
    pub.publishTargets([old, fresh], NOW);
    expect(positionsOf(app.valuesFor(ctx('forward', 14)))).toEqual([{ latitude: 60, longitude: 25 }]);
  });

  it('caps the chart to maxTargets, keeping the closest', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', { ...cfg, maxTargets: 2 });
    const near = (id: number, range: number) =>
      tgt(id, { geometry: { relative_bearing_deg: 0, range_m: range, range_method: 'horizon', range_confidence: 0.7 } });
    pub.publishTargets([near(1, 300), near(2, 100), near(3, 200)], NOW);
    expect(app.targetContexts().sort()).toEqual([ctx('forward', 102), ctx('forward', 103)].sort());
  });

  it('withdraws every target on reset', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1), tgt(2)], NOW);
    app.deltas = [];
    pub.reset();
    expect(app.valuesFor(T1)).toEqual([{ path: 'navigation.position', value: null }]);
    expect(app.valuesFor(ctx('forward', 102))).toEqual([{ path: 'navigation.position', value: null }]);
    expect(pub.isPublished(T1)).toBe(false);
  });

  it('never writes a target position that is not a real lat/lon', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishTargets([tgt(1), tgt(2, { position: null })], NOW);
    pub.publishTargets([tgt(1, { position: null, lastSeen: NOW + 1 }), tgt(3)], NOW + 1000);
    for (const v of app.values()) {
      if (v.path !== 'navigation.position' || v.value === null) continue;
      expect(v.value).toHaveProperty('latitude');
      expect(v.value).toHaveProperty('longitude');
    }
    expect(app.valuesFor(ctx('forward', 102))).toHaveLength(0);
  });

  it('publishes fusion summary counts on the own vessel', () => {
    const pub = new Publisher(app, 'signalk-vision-ai', cfg);
    pub.publishFusionSummary(2, 3);
    const v = app.values();
    expect(v.find((x) => x.path === 'vision.fusion.darkTargetCount')!.value).toBe(2);
    expect(v.find((x) => x.path === 'vision.fusion.aisCorrelatedCount')!.value).toBe(3);
  });
});
