// Read own-ship navigation state from SignalK. headingTrue/cog are radians and
// sog is m/s in the SignalK model, so no conversion is needed.
//
// Heading comes from `navigation.headingTrue` and nothing else. Not from
// `navigation.headingMagnetic` + `navigation.magneticVariation`: a fluxgate
// compass carries whatever deviation its installation gives it, and nothing in
// SignalK says how large that is. Measured on Arabella at the dock, the compass
// plus variation read a steady 33° off the GNSS-compass true heading — every
// target would have been placed 33° wrong and correlated against the wrong AIS
// contacts, while looking healthy. And not from COG: leeway and current make
// course over ground the wrong answer for where the cameras point. With no true
// heading the result is null, which downstream treats as unknown.
//
// Reads are freshness-aware: SignalK retains the last value of a path long after
// the sensor producing it goes quiet, so an unguarded read can hand back a frozen
// position/heading/SOG/COG as if it were live. A stale own-ship fix would
// georeference targets to where we *were*, and a stale (or missing) SOG/COG fed
// into CPA as zero would make a moving vessel look stationary and suppress a real
// collision warning. So when a max age is configured, any value whose timestamp
// is older than it (or unparseable) is dropped to null and the result is flagged
// `stale`, which downstream treats as "unknown", never as "stationary".

import { ServerApp } from './skapp';
import { Attitude, OwnShip, LatLon } from './types';

function num(v: any): number | null {
  return typeof v === 'number' && isFinite(v) ? v : null;
}

function pos(v: any): LatLon | null {
  if (v && typeof v.latitude === 'number' && typeof v.longitude === 'number') {
    return { latitude: v.latitude, longitude: v.longitude };
  }
  return null;
}

interface Read {
  raw: any;
  stale: boolean;
}

// Resolve a self path to its current value, enforcing max age when the node
// carries a timestamp (full-model shape). Delta-only nodes (a bare value with no
// timestamp) can't be aged, so they pass through unflagged — matching how AIS
// contacts without a timestamp are handled in aisFusion.collectAisContacts.
function readPath(app: ServerApp, path: string, maxAgeMs: number, now: number): Read {
  const node: any = app.getSelfPath(path);
  if (node && typeof node === 'object' && 'value' in node) {
    if (maxAgeMs > 0 && typeof node.timestamp === 'string') {
      const ageMs = now - Date.parse(node.timestamp);
      if (!Number.isFinite(ageMs) || ageMs > maxAgeMs) {
        return { raw: null, stale: true };
      }
    }
    return { raw: node.value, stale: false };
  }
  // Bare value (delta-only shape) — no timestamp to check.
  return { raw: node, stale: false };
}

export function readOwnShip(app: ServerApp, maxAgeS = 0, now: number = Date.now()): OwnShip {
  const maxAgeMs = maxAgeS > 0 ? maxAgeS * 1000 : 0;
  const p = readPath(app, 'navigation.position', maxAgeMs, now);
  const h = readPath(app, 'navigation.headingTrue', maxAgeMs, now);
  const s = readPath(app, 'navigation.speedOverGround', maxAgeMs, now);
  const c = readPath(app, 'navigation.courseOverGroundTrue', maxAgeMs, now);
  return {
    position: pos(p.raw),
    headingTrue: num(h.raw),
    sog: num(s.raw),
    cog: num(c.raw),
    stale: p.stale || h.stale || s.stale || c.stale,
  };
}

/**
 * Boat attitude from `navigation.attitude` (SignalK: an object of radians —
 * roll +ve = list to starboard, pitch +ve = bow up, yaw). NMEA 2000 PGN 127257
 * maps here via n2k-signalk; a field the sensor doesn't send arrives as null.
 *
 * Returns null — never a guessed level boat — when pitch or roll is missing,
 * non-finite, outside +/-90° (a source publishing degrees as radians), or older
 * than `maxAgeS` (SignalK keeps the last value after the IMU goes quiet).
 */
export function readAttitude(app: ServerApp, maxAgeS = 0, now: number = Date.now()): Attitude | null {
  const r = readPath(app, 'navigation.attitude', maxAgeS > 0 ? maxAgeS * 1000 : 0, now);
  const v = r.raw;
  if (r.stale || !v || typeof v !== 'object') return null;
  const pitch = num(v.pitch);
  const roll = num(v.roll);
  if (pitch === null || roll === null) return null;
  if (Math.abs(pitch) > Math.PI / 2 || Math.abs(roll) > Math.PI / 2) return null;
  return { pitch, roll };
}
