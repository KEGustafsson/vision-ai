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
    aisCorrelated: false, aisMmsi: null, aisCog: null, aisSog: null, cpa: null, tcpa: null, threatLevel: 'none', lastSeen: 0,
  };
}

describe('CpaEstimator', () => {
  it('detects a head-on collision course (CPA ~0, positive TCPA)', () => {
    const est = new CpaEstimator();
    const own0: OwnShip = { position: { latitude: 60, longitude: 25 }, headingTrue: 0, sog: 5, cog: 0, stale: false };

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

  it('uses AIS velocity for a correlated target on the first sample', () => {
    const est = new CpaEstimator();
    const own: OwnShip = { position: { latitude: 60, longitude: 25 }, headingTrue: 0, sog: 0, cog: 0, stale: false };
    // Target 1000 m due north, moving due south (180°T) at 5 m/s via AIS.
    const tgtAbs = destinationPoint(own.position!, 0, 1000);
    const t = { ...target(tgtAbs), aisCorrelated: true, aisCog: deg2rad(180), aisSog: 5 };
    // Single update: finite-difference would yield nothing, but AIS velocity
    // gives an immediate closing solution.
    est.update([t], own, cfg, 0);
    expect(t.tcpa).not.toBeNull();
    expect(t.tcpa!).toBeGreaterThan(0);
    expect(t.cpa!).toBeLessThan(50);
  });

  it('leaves CPA unresolved when own velocity is unknown (no SOG/COG)', () => {
    const est = new CpaEstimator();
    // Own position is known but SOG/COG are null (e.g. aged out as stale by
    // readOwnShip). Must NOT assume a stationary own ship and fabricate a CPA.
    const own0: OwnShip = { position: { latitude: 60, longitude: 25 }, headingTrue: 0, sog: null, cog: null, stale: true };
    const tgtAbs0 = destinationPoint(own0.position!, 0, 1000);
    est.update([target(tgtAbs0)], own0, cfg, 0);

    const own1: OwnShip = { ...own0, position: destinationPoint(own0.position!, 0, 5) };
    const tgtAbs1 = destinationPoint(tgtAbs0, deg2rad(180), 5);
    const t = target(tgtAbs1);
    est.update([t], own1, cfg, 1000);

    expect(t.cpa).toBeNull();
    expect(t.tcpa).toBeNull();
    expect(t.threatLevel).toBe('none');
  });

  it('classify thresholds', () => {
    expect(classify(50, 120, cfg)).toBe('high');
    expect(classify(50, 400, cfg)).toBe('medium');
    expect(classify(250, 400, cfg)).toBe('low');
    expect(classify(50, -5, cfg)).toBe('none');
    expect(classify(5000, 100, cfg)).toBe('none');
  });
});
