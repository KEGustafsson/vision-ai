import { describe, it, expect } from 'vitest';
import {
  bearingTo, destinationPoint, haversine, normalizeRad, angularDiff, deg2rad,
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
