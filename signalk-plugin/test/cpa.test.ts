import { describe, it, expect } from 'vitest';
import { CpaEstimator, classify } from '../src/cpa';
import { withDefaults } from '../src/config';
import { destinationPoint, deg2rad } from '../src/geo';
import { EnrichedTarget, LatLon, OwnShip } from '../src/types';

const cfg = withDefaults(undefined);

function target(pos: LatLon): EnrichedTarget {
  return {
    track_id: 1, label: 'vessel', coco_class: 8, confidence: 0.9,
    bbox: { x: 0, y: 0, w: 1, h: 1 }, is_person_in_water: false,
    geometry: { relative_bearing_deg: 0, range_m: 1000, range_method: 'horizon', range_confidence: 0.8 },
    pixel_velocity: { vx: 0, vy: 0 }, first_seen: null, age_frames: 0,
    key: 'forward.1', camera: 'forward', bearingTrue: 0, position: pos,
    aisCorrelated: false, aisMmsi: null, cpa: null, tcpa: null, threatLevel: 'none', lastSeen: 0,
  };
}

describe('CpaEstimator', () => {
  it('detects a head-on collision course (CPA ~0, positive TCPA)', () => {
    const est = new CpaEstimator();
    const own0: OwnShip = { position: { latitude: 60, longitude: 25 }, headingTrue: 0, sog: 5, cog: 0 };

    // t=0: target 1000 m due north.
    const tgtAbs0 = destinationPoint(own0.position!, 0, 1000);
    est.update([target(tgtAbs0)], own0, cfg, 0);

    // t=1s: own advanced 5 m north; target advanced 5 m south (closing).
    const ownPos1 = destinationPoint(own0.position!, 0, 5);
    const own1: OwnShip = { ...own0, position: ownPos1 };
    const tgtAbs1 = destinationPoint(tgtAbs0, deg2rad(180), 5);
    const t = target(tgtAbs1);
    est.update([t], own1, cfg, 1000);

    expect(t.tcpa).not.toBeNull();
    expect(t.tcpa!).toBeGreaterThan(0);
    expect(t.cpa!).toBeLessThan(50);
    expect(t.threatLevel).toBe('high');
  });

  it('classify thresholds', () => {
    expect(classify(50, 120, cfg)).toBe('high');
    expect(classify(50, 400, cfg)).toBe('medium');
    expect(classify(250, 400, cfg)).toBe('low');
    expect(classify(50, -5, cfg)).toBe('none');
    expect(classify(5000, 100, cfg)).toBe('none');
  });
});
