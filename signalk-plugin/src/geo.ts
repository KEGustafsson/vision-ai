// Pure geodesy / unit helpers. No SignalK dependency so they are trivially
// unit-testable. Distances in metres, angles in radians unless suffixed.

import { LatLon } from './types';

export const EARTH_RADIUS_M = 6371000;

export const deg2rad = (d: number): number => (d * Math.PI) / 180;
export const rad2deg = (r: number): number => (r * 180) / Math.PI;
export const kn2ms = (kn: number): number => kn * 0.514444;
export const ms2kn = (ms: number): number => ms / 0.514444;

/** Normalise an angle (radians) to [0, 2π). */
export function normalizeRad(r: number): number {
  const twoPi = 2 * Math.PI;
  return ((r % twoPi) + twoPi) % twoPi;
}

/** Great-circle distance (metres) between two points. */
export function haversine(a: LatLon, b: LatLon): number {
  const dLat = deg2rad(b.latitude - a.latitude);
  const dLon = deg2rad(b.longitude - a.longitude);
  const la1 = deg2rad(a.latitude);
  const la2 = deg2rad(b.latitude);
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(la1) * Math.cos(la2) * Math.sin(dLon / 2) ** 2;
  return 2 * EARTH_RADIUS_M * Math.asin(Math.min(1, Math.sqrt(h)));
}

/** Initial true bearing (radians, 0..2π) from point a to point b. */
export function bearingTo(a: LatLon, b: LatLon): number {
  const la1 = deg2rad(a.latitude);
  const la2 = deg2rad(b.latitude);
  const dLon = deg2rad(b.longitude - a.longitude);
  const y = Math.sin(dLon) * Math.cos(la2);
  const x =
    Math.cos(la1) * Math.sin(la2) -
    Math.sin(la1) * Math.cos(la2) * Math.cos(dLon);
  return normalizeRad(Math.atan2(y, x));
}

/** Destination point given start, true bearing (rad) and distance (m). */
export function destinationPoint(
  start: LatLon,
  bearingRad: number,
  distanceM: number
): LatLon {
  const ang = distanceM / EARTH_RADIUS_M;
  const la1 = deg2rad(start.latitude);
  const lo1 = deg2rad(start.longitude);
  const la2 = Math.asin(
    Math.sin(la1) * Math.cos(ang) +
      Math.cos(la1) * Math.sin(ang) * Math.cos(bearingRad)
  );
  const lo2 =
    lo1 +
    Math.atan2(
      Math.sin(bearingRad) * Math.sin(ang) * Math.cos(la1),
      Math.cos(ang) - Math.sin(la1) * Math.sin(la2)
    );
  return { latitude: rad2deg(la2), longitude: rad2deg(lo2) };
}

/** Smallest absolute difference between two bearings (radians). */
export function angularDiff(a: number, b: number): number {
  let d = Math.abs(normalizeRad(a) - normalizeRad(b));
  if (d > Math.PI) d = 2 * Math.PI - d;
  return d;
}

/**
 * Convert a local geographic offset (metres) to a lat/lon near a reference
 * point using an equirectangular approximation (fine for short ranges).
 */
export function offsetToLatLon(
  ref: LatLon,
  eastM: number,
  northM: number
): LatLon {
  const dLat = northM / EARTH_RADIUS_M;
  const dLon = eastM / (EARTH_RADIUS_M * Math.cos(deg2rad(ref.latitude)));
  return {
    latitude: ref.latitude + rad2deg(dLat),
    longitude: ref.longitude + rad2deg(dLon),
  };
}
