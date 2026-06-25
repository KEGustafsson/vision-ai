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

  it('assigns one-to-one: two targets cannot share one AIS contact', () => {
    const brg = deg2rad(90);
    const aisPos = destinationPoint(own.position, brg, 600);
    // A single real AIS vessel, but two visual targets near it (one slightly off).
    const vessels = { 'urn:mrn:imo:mmsi:111111111': aisVessel(aisPos) };
    const contacts = collectAisContacts(vessels, own);
    const near = visualTarget(brg, 610, 'forward.1');
    const off = visualTarget(deg2rad(91), 650, 'forward.2');
    const res = fuse([near, off], contacts, cfg);
    // Exactly one target may carry the MMSI — the single contact cannot be
    // claimed twice. The loser sits beside the same real vessel, so it is
    // ambiguous (near-miss), not a confident dark target.
    const correlated = res.targets.filter((t) => t.aisCorrelated);
    expect(correlated).toHaveLength(1);
    expect(correlated[0].key).toBe('forward.1'); // the closer match wins
    expect(res.aisCorrelatedCount).toBe(1);
    expect(res.targets.find((t) => t.key === 'forward.2')!.aisCorrelated).toBe(false);
    expect(res.darkTargetKeys).not.toContain('forward.2');
  });

  it('keeps a target on its previous AIS identity when two contacts both gate (hysteresis)', () => {
    const brg = deg2rad(90);
    // Two AIS contacts straddling the target so either could match.
    const aPos = destinationPoint(own.position, deg2rad(89.5), 600);
    const bPos = destinationPoint(own.position, deg2rad(90.5), 600);
    const vessels = {
      'urn:mrn:imo:mmsi:111111111': aisVessel(aPos),
      'urn:mrn:imo:mmsi:222222222': aisVessel(bPos),
    };
    const contacts = collectAisContacts(vessels, own);
    const tgt = visualTarget(brg, 600, 'forward.1');
    // Pin a prior assignment to the contact that is NOT the marginally-best one.
    const prev = new Map([['forward.1', '222222222']]);
    const res = fuse([tgt], contacts, cfg, prev);
    expect(res.targets[0].aisMmsi).toBe('222222222');
    expect(res.assignment.get('forward.1')).toBe('222222222');
  });

  it('does not flag a dark target when a near-miss AIS contact sits just outside the gate', () => {
    const brg = deg2rad(270);
    // Vessel detected at 400 m; a real AIS vessel at 520 m on the same bearing —
    // beyond the correlation range gate (~96 m at conf 0.7) but inside the
    // 1.5× near-miss band (~144 m): ambiguous, so we hold off the dark call.
    const aisPos = destinationPoint(own.position, brg, 520);
    const vessels = { 'urn:mrn:imo:mmsi:111111111': aisVessel(aisPos) };
    const contacts = collectAisContacts(vessels, own);
    const res = fuse([visualTarget(brg, 400)], contacts, cfg);
    expect(res.targets[0].aisCorrelated).toBe(false);
    expect(res.darkTargetKeys).toHaveLength(0);
  });

  it('correlates nothing (no NaN) when the bearing tolerance is misconfigured to 0', () => {
    const brg = deg2rad(90);
    const aisPos = destinationPoint(own.position, brg, 600);
    const vessels = { 'urn:mrn:imo:mmsi:111111111': aisVessel(aisPos) };
    const contacts = collectAisContacts(vessels, own);
    const zeroCfg = withDefaults({ correlationBearingDeg: 0 });
    const res = fuse([visualTarget(brg, 610)], contacts, zeroCfg);
    expect(res.aisCorrelatedCount).toBe(0);
    expect(res.targets[0].aisCorrelated).toBe(false);
  });

  it('does not correlate a vessel detection with an aid-to-navigation (ATON) contact', () => {
    const brg = deg2rad(90);
    const atonPos = destinationPoint(own.position, brg, 600);
    const vessels = {
      'urn:mrn:imo:mmsi:111111111': {
        navigation: { position: { value: atonPos } },
        sensors: { ais: { class: { value: 'ATON' } } },
      },
    };
    const contacts = collectAisContacts(vessels, own);
    expect(contacts).toHaveLength(1); // collected (a buoy could match it)
    const res = fuse([visualTarget(brg, 610)], contacts, cfg); // label 'vessel'
    expect(res.targets[0].aisCorrelated).toBe(false);
    expect(res.darkTargetKeys).toContain('forward.1');
  });
});
