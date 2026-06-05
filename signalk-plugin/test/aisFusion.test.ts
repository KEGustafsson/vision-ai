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
    aisCorrelated: false, aisMmsi: null, aisCog: null, aisSog: null, cpa: null, tcpa: null, threatLevel: 'none', lastSeen: 0,
  };
}

describe('aisFusion', () => {
  const cfg = withDefaults(undefined);

  // A real AIS vessel: real MMSI context + a reported AIS class.
  const aisVessel = (pos: any, extra: any = {}) => ({
    navigation: { position: { value: pos }, ...extra },
    sensors: { ais: { class: { value: 'A' } } },
  });

  it('collects AIS contacts with bearing and range', () => {
    const aisPos = destinationPoint(own.position, deg2rad(90), 600);
    const vessels = {
      self: {},
      'urn:mrn:imo:mmsi:123456789': { ...aisVessel(aisPos), name: { value: 'TESTER' } },
    };
    const contacts = collectAisContacts(vessels, own);
    expect(contacts).toHaveLength(1);
    expect(contacts[0].mmsi).toBe('123456789');
    expect(contacts[0].aisClass).toBe('A');
    expect(contacts[0].range).toBeCloseTo(600, -1);
  });

  it('ignores contacts without an MMSI or an AIS class', () => {
    const pos = destinationPoint(own.position, deg2rad(90), 600);
    const vessels = {
      // Our own synthetic blip: UUID context, position + name, no MMSI/class.
      'urn:mrn:signalk:uuid:vision-forward-1': { navigation: { position: { value: pos } }, name: { value: 'VIS-vessel-1' } },
      // Real MMSI context but no AIS class — still rejected.
      'urn:mrn:imo:mmsi:999999999': { navigation: { position: { value: pos } } },
      // 'mmsi:' prefix but a malformed (non-9-digit) identity — rejected.
      'urn:mrn:imo:mmsi:111': aisVessel(pos),
    };
    expect(collectAisContacts(vessels, own)).toHaveLength(0);
  });

  it('filters AIS contacts too close to own ship', () => {
    const nearPos = destinationPoint(own.position, deg2rad(90), 5);
    const farPos = destinationPoint(own.position, deg2rad(90), 100);
    const vessels = {
      'urn:mrn:imo:mmsi:111111111': aisVessel(nearPos),
      'urn:mrn:imo:mmsi:222222222': aisVessel(farPos),
    };
    const contacts = collectAisContacts(vessels, own, 25);
    expect(contacts.map((c) => c.mmsi)).toEqual(['222222222']);
  });

  it('correlates a visual target with a co-located AIS contact', () => {
    const brg = deg2rad(90);
    const aisPos = destinationPoint(own.position, brg, 600);
    const vessels = { 'urn:mrn:imo:mmsi:111111111': aisVessel(aisPos) };
    const contacts = collectAisContacts(vessels, own);
    const res = fuse([visualTarget(brg, 610)], contacts, cfg);
    expect(res.aisCorrelatedCount).toBe(1);
    expect(res.targets[0].aisCorrelated).toBe(true);
    expect(res.targets[0].aisMmsi).toBe('111111111');
    expect(res.darkTargetKeys).toHaveLength(0);
  });

  it('captures AIS COG/SOG from a correlated contact', () => {
    const brg = deg2rad(90);
    const aisPos = destinationPoint(own.position, brg, 600);
    const vessels = {
      'urn:mrn:imo:mmsi:111111111': aisVessel(aisPos, {
        courseOverGroundTrue: { value: deg2rad(270) },
        speedOverGround: { value: 4 },
      }),
    };
    const contacts = collectAisContacts(vessels, own);
    expect(contacts[0].mmsi).toBe('111111111');
    expect(contacts[0].cog).toBeCloseTo(deg2rad(270), 5);
    expect(contacts[0].sog).toBe(4);
    const res = fuse([visualTarget(brg, 610)], contacts, cfg);
    expect(res.targets[0].aisCorrelated).toBe(true);
    expect(res.targets[0].aisSog).toBe(4);
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
