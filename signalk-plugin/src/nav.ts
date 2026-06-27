// Read own-ship navigation state from SignalK. headingTrue/cog are radians and
// sog is m/s in the SignalK model, so no conversion is needed.
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
import { OwnShip, LatLon } from './types';

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
