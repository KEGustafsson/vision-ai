// Pure geodesy / unit helpers. No SignalK dependency so they are trivially
// unit-testable. Distances in metres, angles in radians unless suffixed.

import { LatLon } from './types';

export const EARTH_RADIUS_M = 6371000;

export const deg2rad = (d: number): number => (d * Math.PI) / 180;
export const rad2deg = (r: number): number => (r * 180) / Math.PI;

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
 * Solar elevation (radians, positive above the horizon) at a position and
 * instant, by the standard low-precision solar-position formulae. Accurate to
 * a few hundredths of a degree, which is far finer than any daylight decision
 * needs.
 */
export function solarElevation(pos: LatLon, at: Date): number {
  const n = at.getTime() / 86400000 + 2440587.5 - 2451545.0; // days since J2000.0
  const meanLon = deg2rad((280.46 + 0.9856474 * n) % 360);
  const meanAnom = deg2rad((357.528 + 0.9856003 * n) % 360);
  // Ecliptic longitude: mean longitude plus the equation of centre.
  const eclipticLon =
    meanLon + deg2rad(1.915) * Math.sin(meanAnom) + deg2rad(0.02) * Math.sin(2 * meanAnom);
  const obliquity = deg2rad(23.439 - 0.0000004 * n);
  const declination = Math.asin(Math.sin(obliquity) * Math.sin(eclipticLon));
  const rightAscension = Math.atan2(
    Math.cos(obliquity) * Math.sin(eclipticLon), Math.cos(eclipticLon));
  // Greenwich mean sidereal time -> local hour angle.
  const gmstHours = (18.697374558 + 24.06570982441908 * n) % 24;
  const hourAngle = deg2rad(gmstHours * 15) + deg2rad(pos.longitude) - rightAscension;
  const lat = deg2rad(pos.latitude);
  return Math.asin(
    Math.sin(lat) * Math.sin(declination) +
      Math.cos(lat) * Math.cos(declination) * Math.cos(hourAngle)
  );
}

// Civil twilight: the sun 6° below the horizon, the point where the unaided eye
// stops making out a dark hull against the water.
export const CIVIL_TWILIGHT_RAD = deg2rad(-6);

/** Whether it is dark enough at *pos* to warrant night detection settings. */
export function isNight(pos: LatLon, at: Date): boolean {
  return solarElevation(pos, at) < CIVIL_TWILIGHT_RAD;
}
