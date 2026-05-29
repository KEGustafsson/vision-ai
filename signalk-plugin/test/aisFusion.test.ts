import { describe, it, expect } from 'vitest';
import { collectAisContacts, fuse } from '../src/aisFusion';
import { withDefaults } from '../src/config';
import { destinationPoint, deg2rad } from '../src/geo';
import { EnrichedTarget } from '../src/types';

const own = { position: { latitude: 60, longitude: 25 }, headingTrue: 0, sog: 0, cog: 0 };

function visualTarget(bearingRad: number, range: number, key = 'forward.1'): EnrichedTarget {
  return {
    track_id: 1, label: 'vessel', coco_class: 8, confidence: 0.9,
    bbox: { x: 0, y: 0, w: 10, h: 10 }, is_person_in_water: false,
    geometry: { relative_bearing_deg: 0, range_m: range, range_method: 'horizon', range_confidence: 0.7 },
    pixel_velocity: { vx: 0, vy: 0 }, first_seen: null, age_frames: 0,
    key, camera: 'forward', bearingTrue: bearingRad, position: destinationPoint(own.position, bearingRad, range),
    aisCorrelated: false, aisMmsi: null, cpa: null, tcpa: null, threatLevel: 'none', lastSeen: 0,
  };
}

describe('aisFusion', () => {
  const cfg = withDefaults(undefined);

  it('collects AIS contacts with bearing and range', () => {
    const aisPos = destinationPoint(own.position, deg2rad(90), 600);
    const vessels = {
      self: {},
      'urn:mrn:imo:mmsi:123456789': { navigation: { position: { value: aisPos } }, name: { value: 'TESTER' } },
    };
    const contacts = collectAisContacts(vessels, own);
    expect(contacts).toHaveLength(1);
    expect(contacts[0].mmsi).toBe('123456789');
    expect(contacts[0].range).toBeCloseTo(600, -1);
  });

  it('correlates a visual target with a co-located AIS contact', () => {
    const brg = deg2rad(90);
    const aisPos = destinationPoint(own.position, brg, 600);
    const vessels = { 'urn:mrn:imo:mmsi:111': { navigation: { position: { value: aisPos } } } };
    const contacts = collectAisContacts(vessels, own);
    const res = fuse([visualTarget(brg, 610)], contacts, cfg);
    expect(res.aisCorrelatedCount).toBe(1);
    expect(res.targets[0].aisCorrelated).toBe(true);
    expect(res.targets[0].aisMmsi).toBe('111');
    expect(res.darkTargetKeys).toHaveLength(0);
  });

  it('flags an uncorrelated in-range vessel as a dark target', () => {
    const res = fuse([visualTarget(deg2rad(270), 400)], [], cfg);
    expect(res.darkTargetKeys).toContain('forward.1');
    expect(res.targets[0].aisCorrelated).toBe(false);
  });

  it('does not flag distant uncorrelated vessels', () => {
    const res = fuse([visualTarget(deg2rad(270), 5000)], [], cfg);
    expect(res.darkTargetKeys).toHaveLength(0);
  });
});
