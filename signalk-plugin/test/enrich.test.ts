import { describe, it, expect } from 'vitest';
import { enrichTarget } from '../src/enrich';
import { withDefaults } from '../src/config';
import { bearingTo, rad2deg } from '../src/geo';
import { RawTarget } from '../src/types';

function raw(relBrg: number, range: number | null, rangeConf = 0.6): RawTarget {
  return {
    track_id: 1,
    label: 'vessel',
    coco_class: 8,
    confidence: 0.9,
    bbox: { x: 0, y: 0, w: 10, h: 10 },
    is_person_in_water: false,
    geometry: {
      relative_bearing_deg: relBrg,
      range_m: range,
      range_method: range ? 'horizon' : null,
      range_confidence: rangeConf,
    },
    pixel_velocity: { vx: 0, vy: 0 },
    first_seen: null,
    age_frames: 0,
  };
}

describe('enrichTarget', () => {
  const cfg = withDefaults(undefined);
  const own = { position: { latitude: 60, longitude: 25 }, headingTrue: 0, sog: 0, cog: 0 };

  it('computes true bearing = heading + relative', () => {
    const t = enrichTarget(raw(30, 500), 'forward', { ...own, headingTrue: Math.PI / 2 }, cfg, 0);
    expect(rad2deg(t.bearingTrue!)).toBeCloseTo(120, 5);
  });

  it('georeferences a target with sufficient range confidence', () => {
    const t = enrichTarget(raw(0, 500), 'forward', own, cfg, 0);
    expect(t.position).not.toBeNull();
    // Straight ahead (heading 0 = north) at 500 m -> ~500 m due north.
    expect(bearingTo(own.position, t.position!)).toBeCloseTo(0, 1);
  });

  it('does not georeference when range confidence is below the gate', () => {
    const t = enrichTarget(raw(0, 500, 0.1), 'forward', own, cfg, 0);
    expect(t.position).toBeNull();
  });

  it('key combines camera and track id', () => {
    const t = enrichTarget(raw(0, 500), 'aft', own, cfg, 0);
    expect(t.key).toBe('aft.1');
  });
});
