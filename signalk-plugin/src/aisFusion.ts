// Correlate visual targets with AIS-reported vessels. A visual "vessel" that
// matches no AIS contact within tolerance, yet is within alert range, is a
// "dark target" — a craft that is seen but not transmitting AIS.

import { PluginConfig } from './config';
import { angularDiff, bearingTo, deg2rad, haversine } from './geo';
import { EnrichedTarget, LatLon, OwnShip } from './types';

export interface AisContact {
  mmsi: string;
  name?: string;
  position: LatLon;
  bearing: number; // rad, from own ship
  range: number; // m
}

const VESSEL_LABELS = new Set(['vessel', 'boat', 'ship', 'ferry', 'sail boat', 'speed boat']);

/** Extract AIS contacts (excluding self) that have a position. */
export function collectAisContacts(vessels: any, own: OwnShip): AisContact[] {
  const out: AisContact[] = [];
  if (!vessels || !own.position) return out;
  for (const [id, v] of Object.entries<any>(vessels)) {
    if (id === 'self') continue;
    const p = v?.navigation?.position?.value ?? v?.navigation?.position;
    if (!p || typeof p.latitude !== 'number') continue;
    const pos: LatLon = { latitude: p.latitude, longitude: p.longitude };
    const mmsi = id.includes('mmsi:') ? id.split('mmsi:').pop()! : id;
    out.push({
      mmsi,
      name: v?.name?.value ?? v?.name,
      position: pos,
      bearing: bearingTo(own.position, pos),
      range: haversine(own.position, pos),
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
