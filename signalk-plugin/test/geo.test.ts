import { describe, it, expect } from 'vitest';
import {
  bearingTo, destinationPoint, haversine, normalizeRad, angularDiff, deg2rad, rad2deg,
  solarElevation, isNight,
} from '../src/geo';

describe('geo', () => {
  it('haversine is ~0 for same point and positive otherwise', () => {
    const a = { latitude: 60, longitude: 25 };
    expect(haversine(a, a)).toBeCloseTo(0, 3);
    expect(haversine(a, { latitude: 60.01, longitude: 25 })).toBeGreaterThan(1000);
  });

  it('destinationPoint then bearingTo round-trips', () => {
    const start = { latitude: 60, longitude: 25 };
    const brg = deg2rad(45);
    const dest = destinationPoint(start, brg, 1000);
    expect(haversine(start, dest)).toBeCloseTo(1000, 0);
    expect(bearingTo(start, dest)).toBeCloseTo(brg, 2);
  });

  it('normalizeRad wraps into [0,2pi)', () => {
    expect(normalizeRad(-Math.PI / 2)).toBeCloseTo((3 * Math.PI) / 2, 6);
  });

  it('angularDiff is symmetric and handles wrap', () => {
    expect(angularDiff(deg2rad(350), deg2rad(10))).toBeCloseTo(deg2rad(20), 6);
  });
});

describe('solar elevation', () => {
  const helsinki = { latitude: 60.17, longitude: 24.94 };

  it('matches the textbook midwinter maximum at 60°N', () => {
    // Highest the sun gets at the solstice is 90 − latitude − 23.44 = 6.39°.
    const noon = new Date('2026-12-21T10:00:00Z'); // ~solar noon at 24.94°E
    expect(rad2deg(solarElevation(helsinki, noon))).toBeCloseTo(6.39, 0);
  });

  it('puts the sun high at midsummer noon and below the horizon at midnight', () => {
    const noon = rad2deg(solarElevation(helsinki, new Date('2026-06-21T09:00:00Z')));
    const midnight = rad2deg(solarElevation(helsinki, new Date('2026-06-21T21:00:00Z')));
    expect(noon).toBeGreaterThan(45);
    expect(midnight).toBeLessThan(0);
  });

  it('is overhead at the equator at equinox noon and opposite at midnight', () => {
    const equator = { latitude: 0, longitude: 0 };
    expect(rad2deg(solarElevation(equator, new Date('2026-03-20T12:00:00Z')))).toBeGreaterThan(85);
    expect(rad2deg(solarElevation(equator, new Date('2026-03-20T00:00:00Z')))).toBeLessThan(-85);
  });
});

describe('isNight', () => {
  const helsinki = { latitude: 60.17, longitude: 24.94 };

  it('is not night at local midnight in midsummer', () => {
    // The case the wall-clock rule gets wrong: 00:00 local at the solstice is
    // civil twilight, not darkness, so the detection threshold must not drop.
    expect(isNight(helsinki, new Date('2026-06-21T21:00:00Z'))).toBe(false);
  });

  it('is night in the small hours of a Baltic winter', () => {
    expect(isNight(helsinki, new Date('2026-01-15T01:00:00Z'))).toBe(true);
  });

  it('is night well before the 21:00 the clock rule waits for, in midwinter', () => {
    // Dark by ~17:00 local in December; the hour rule would still call it day.
    expect(isNight(helsinki, new Date('2026-12-21T15:00:00Z'))).toBe(true);
  });

  it('is day at local noon in every season', () => {
    expect(isNight(helsinki, new Date('2026-06-21T09:00:00Z'))).toBe(false);
    expect(isNight(helsinki, new Date('2026-12-21T10:00:00Z'))).toBe(false);
  });
});
