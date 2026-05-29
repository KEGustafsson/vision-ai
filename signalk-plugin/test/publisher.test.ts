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
  metas(): Array<{ path: string; value: unknown }> {
    return this.deltas.flatMap((d) => d.updates.flatMap((u) => u.meta || []));
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
    aisCorrelated: false, aisMmsi: null, cpa: null, tcpa: null, threatLevel: 'none', lastSeen: 0,
    ...partial,
  };
}

describe('Publisher', () => {
  let app: FakeApp;
  let pub: Publisher;
  const cfg = withDefaults(undefined);

  beforeEach(() => {
    app = new FakeApp();
    pub = new Publisher(app, 'signalk-vision-ai', cfg);
  });

  it('emits vision.targets.* values with metadata once', () => {
    pub.publishTargets([tgt(1)]);
    const paths = app.values().map((v) => v.path);
    expect(paths).toContain('vision.targets.forward.1.bearingTrue');
    expect(paths).toContain('vision.targets.forward.1.distance');
    expect(paths).toContain('vision.targets.forward.1.position');
    // Distance metadata sent with units.
    const distMeta = app.metas().find((m) => m.path === 'vision.targets.forward.1.distance');
    expect(distMeta).toBeTruthy();
    expect((distMeta!.value as any).units).toBe('m');

    // A second publish for the same path must not resend metadata.
    const before = app.metas().length;
    pub.publishTargets([tgt(1)]);
    expect(app.metas().length).toBe(before);
  });

  it('ages out a disappeared track by publishing null', () => {
    pub.publishTargets([tgt(1)]);
    app.deltas = [];
    pub.publishTargets([]); // track 1 gone
    const nulls = app.values().filter((v) => v.path.startsWith('vision.targets.forward.1.') && v.value === null);
    expect(nulls.length).toBeGreaterThan(0);
  });

  it('publishes fusion summary counts', () => {
    pub.publishFusionSummary(2, 3);
    const v = app.values();
    expect(v.find((x) => x.path === 'vision.fusion.darkTargetCount')!.value).toBe(2);
    expect(v.find((x) => x.path === 'vision.fusion.aisCorrelatedCount')!.value).toBe(3);
  });

  it('does not publish targets when visual radar is disabled', () => {
    const pub2 = new Publisher(app, 'signalk-vision-ai', { ...cfg, enableVisualRadar: false });
    pub2.publishTargets([tgt(1)]);
    expect(app.values()).toHaveLength(0);
  });
});
