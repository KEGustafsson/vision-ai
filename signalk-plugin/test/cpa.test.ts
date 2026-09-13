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
    aisCorrelated: false, aisMmsi: null, aisCog: null, aisSog: null, aisPosition: null,
    cpa: null, tcpa: null, sog: null, cog: null, threatLevel: 'none', lastSeen: 0,
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

  it('solves a correlated target from the AIS position, not the monocular one', () => {
    // Monocular range carries tens of percent of error. Pairing that position
    // with the precise AIS velocity put a known vessel's CPA well out and made
    // the threat level flap; once correlated, the contact's own fix is better.
    const est = new CpaEstimator();
    const own: OwnShip = { position: { latitude: 60, longitude: 25 }, headingTrue: 0, sog: 5, cog: 0, stale: false };
    const seen = destinationPoint(own.position!, 0, 1000);   // camera says 1000 m
    const actual = destinationPoint(own.position!, 0, 300);  // AIS says 300 m
    const t = {
      ...target(seen),
      aisCorrelated: true, aisPosition: actual, aisCog: 0, aisSog: 0,
    };

    est.update([t], own, cfg, 0);

    // Closing at own 5 m/s on a stationary target: 300 m away is 60 s out,
    // 1000 m would have been 200 s.
    expect(t.tcpa!).toBeCloseTo(60, 0);
    expect(t.cpa!).toBeLessThan(5);
    // The visual estimate is left alone — it is what the camera saw, and what
    // the synthetic blip publishes.
    expect(t.position).toBe(seen);
  });

  it('does not difference an AIS fix against a monocular one when correlation drops', () => {
    // The two positions can sit tens of metres apart. Differencing across the
    // change would read that gap as target motion in a single cycle.
    const est = new CpaEstimator();
    const own: OwnShip = { position: { latitude: 60, longitude: 25 }, headingTrue: 0, sog: 5, cog: 0, stale: false };
    const seen = destinationPoint(own.position!, 0, 1000);
    const actual = destinationPoint(own.position!, 0, 300);

    const correlated = {
      ...target(seen), aisCorrelated: true, aisPosition: actual, aisCog: 0, aisSog: 0,
    };
    est.update([correlated], own, cfg, 0);

    // Next cycle the AIS match is gone; only the visual estimate remains.
    const lost = { ...target(seen) };
    est.update([lost], own, cfg, 1000);

    expect(lost.tcpa).toBeNull();
    expect(lost.threatLevel).toBe('none');
    expect(lost.sog).toBeNull();   // no velocity invented from the source change
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
