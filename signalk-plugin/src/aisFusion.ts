// Correlate visual targets with AIS-reported vessels. A visual "vessel" that
// matches no AIS contact within tolerance, yet is within alert range, is a
// "dark target" — a craft that is seen but not transmitting AIS.

import { PluginConfig } from './config';
import { angularDiff, bearingTo, deg2rad, haversine } from './geo';
import { EnrichedTarget, LatLon, OwnShip } from './types';

export interface AisContact {
  mmsi: string; // real AIS identity — never null (a contact without one is dropped)
  aisClass: string; // AIS class ("A"/"B"/ATON type) reported by the transmitter
  name?: string;
  position: LatLon;
  bearing: number; // rad, from own ship
  range: number; // m
  cog: number | null; // rad
  sog: number | null; // m/s
}

const num = (v: any): number | null =>
  typeof v === 'number' && isFinite(v) ? v : null;

const VESSEL_LABELS = new Set([
  'vessel', 'boat', 'ship', 'ferry', 'sail boat', 'speed boat', 'kayak',
  // marine-surveillance model canonical labels (no spaces/hyphens)
  'sailboat', 'speedboat', 'warship',
]);

/**
 * Extract REAL AIS contacts (excluding self) to correlate against. A contact
 * qualifies only if it carries both an MMSI and an AIS class — i.e. it is an
 * actual transmitting vessel, not a position-only blip. This deliberately
 * excludes our own synthetic camera vessels (published as UUID contexts with no
 * MMSI/class), which would otherwise self-correlate with the visual targets that
 * spawned them.
 */
export function collectAisContacts(
  vessels: any,
  own: OwnShip,
  minRangeM = 0,
  maxAgeMs = 0,
  nowMs: number = Date.now()
): AisContact[] {
  const out: AisContact[] = [];
  if (!vessels || !own.position) return out;
  for (const [id, v] of Object.entries<any>(vessels)) {
    if (id === 'self') continue;
    // Require a real AIS identity: exactly 9 digits (the MMSI format). A bare
    // `mmsi:` prefix check would let a malformed token through; UUID-only /
    // synthetic contexts have no MMSI at all.
    const mmsi = id.match(/mmsi:(\d{9})(?::|$)/)?.[1] ?? null;
    if (!mmsi) continue;
    // Require an AIS class — confirms this came from an AIS transmitter, not a
    // bare position injected by some other plugin (ours included).
    const aisClass = v?.sensors?.ais?.class?.value ?? v?.sensors?.ais?.class;
    if (typeof aisClass !== 'string' || aisClass.trim().length === 0) continue;
    const posNode = v?.navigation?.position;
    const p = posNode?.value ?? posNode;
    // Validate BOTH lat and lon: a numeric lat with a missing/NaN lon would
    // produce NaN range/bearing that silently corrupts correlation.
    if (!p || typeof p.latitude !== 'number' || typeof p.longitude !== 'number') continue;
    // Drop stale contacts: SignalK retains an AIS vessel's last-known position
    // long after it stops transmitting. A stale fix would let a no-longer-
    // transmitting vessel keep correlating (suppressing its dark-target alarm)
    // and feed a wrong AIS SOG/COG into CPA. Only enforced when a timestamp is
    // present (full-model shape); delta-only shapes without one are kept.
    if (maxAgeMs > 0 && typeof posNode?.timestamp === 'string') {
      const ageMs = nowMs - Date.parse(posNode.timestamp);
      if (Number.isFinite(ageMs) && ageMs > maxAgeMs) continue;
    }
    const pos: LatLon = { latitude: p.latitude, longitude: p.longitude };
    const range = haversine(own.position, pos);
    if (minRangeM > 0 && range < minRangeM) continue;
    out.push({
      mmsi,
      aisClass,
      name: v?.name?.value ?? v?.name,
      position: pos,
      bearing: bearingTo(own.position, pos),
      range,
      cog: num(v?.navigation?.courseOverGroundTrue?.value ?? v?.navigation?.courseOverGroundTrue),
      sog: num(v?.navigation?.speedOverGround?.value ?? v?.navigation?.speedOverGround),
    });
  }
  return out;
}

export interface FusionResult {
  targets: EnrichedTarget[];
  darkTargetKeys: string[];
  aisCorrelatedCount: number;
}

export function fuse(
  targets: EnrichedTarget[],
  ais: AisContact[],
  cfg: PluginConfig
): FusionResult {
  const darkTargetKeys: string[] = [];
  let aisCorrelatedCount = 0;
  const bearingTol = deg2rad(cfg.correlationBearingDeg);

  for (const t of targets) {
    // Reset correlation state every pass — targets persist across cycles, so a
    // target that loses its AIS match must not retain stale identity/kinematics.
    t.aisCorrelated = false;
    t.aisMmsi = null;
    t.aisCog = null;
    t.aisSog = null;

    if (t.bearingTrue === null) continue;
    const range = t.geometry.range_m;

    let best: AisContact | null = null;
    let bestScore = Infinity;
    for (const a of ais) {
      const dB = angularDiff(t.bearingTrue, a.bearing);
      if (dB > bearingTol) continue;
      // If we have a visual range, also require range agreement.
      if (range !== null) {
        const rangeTol = Math.max(cfg.correlationRangeFrac * range, 50);
        if (Math.abs(range - a.range) > rangeTol) continue;
      }
      const score = dB + (range !== null ? Math.abs(range - a.range) / 1000 : 0);
      if (score < bestScore) {
        bestScore = score;
        best = a;
      }
    }

    if (best) {
      t.aisCorrelated = true;
      t.aisMmsi = best.mmsi;
      t.aisCog = best.cog;
      t.aisSog = best.sog;
      aisCorrelatedCount += 1;
    } else if (
      VESSEL_LABELS.has(t.label) &&
      range !== null &&
      range <= cfg.darkTargetRangeM
    ) {
      darkTargetKeys.push(t.key);
    }
  }

  return { targets, darkTargetKeys, aisCorrelatedCount };
}
