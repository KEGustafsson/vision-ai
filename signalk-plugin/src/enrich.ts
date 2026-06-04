// Turn a container detection (relative bearing + range) into a navigation-relative
// target: true bearing and a georeferenced lat/lon, using own-ship state.

import { deg2rad, destinationPoint, normalizeRad } from './geo';
import { PluginConfig } from './config';
import { EnrichedTarget, OwnShip, RawTarget } from './types';

export function targetKey(camera: string, raw: RawTarget): string {
  const id = raw.track_id ?? `anon-${Math.round(raw.bbox.x)}-${Math.round(raw.bbox.y)}`;
  return `${camera}.${id}`;
}

export function enrichTarget(
  raw: RawTarget,
  camera: string,
  own: OwnShip,
  cfg: PluginConfig,
  now: number
): EnrichedTarget {
  let bearingTrue: number | null = null;
  let position = null as EnrichedTarget['position'];

  if (own.headingTrue !== null) {
    bearingTrue = normalizeRad(own.headingTrue + deg2rad(raw.geometry.relative_bearing_deg));
  }

  const rangeOk =
    raw.geometry.range_m !== null &&
    raw.geometry.range_confidence >= cfg.minRangeConfidence;

  if (own.position && bearingTrue !== null && rangeOk) {
    position = destinationPoint(own.position, bearingTrue, raw.geometry.range_m as number);
  }

  return {
    ...raw,
    key: targetKey(camera, raw),
    camera,
    bearingTrue,
    position,
    aisCorrelated: false,
    aisMmsi: null,
    aisCog: null,
    aisSog: null,
    cpa: null,
    tcpa: null,
    sog: null,
    cog: null,
    threatLevel: 'none',
    lastSeen: now,
  };
}
