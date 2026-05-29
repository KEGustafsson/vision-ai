// Closest Point of Approach (CPA) and Time to CPA (TCPA) for visual targets.
//
// Target ground velocity is estimated from successive georeferenced positions
// of the same track (or taken from AIS when correlated). Own velocity comes
// from SOG/COG. We solve the standard relative-motion CPA in a local ENU frame.

import { PluginConfig } from './config';
import { EARTH_RADIUS_M, deg2rad } from './geo';
import { EnrichedTarget, LatLon, OwnShip, ThreatLevel } from './types';

interface Sample {
  t: number; // s
  e: number; // east metres in a per-sample local frame around own
  n: number; // north metres
  refLat: number;
  refLon: number;
}

function toLocal(ref: LatLon, p: LatLon): { e: number; n: number } {
  const e =
    deg2rad(p.longitude - ref.longitude) *
    EARTH_RADIUS_M *
    Math.cos(deg2rad(ref.latitude));
  const n = deg2rad(p.latitude - ref.latitude) * EARTH_RADIUS_M;
  return { e, n };
}

export class CpaEstimator {
  private history = new Map<string, Sample>();

  /** Compute and attach cpa/tcpa/threatLevel for each target. */
  update(targets: EnrichedTarget[], own: OwnShip, cfg: PluginConfig, nowMs: number): void {
    const t = nowMs / 1000;
    const active = new Set<string>();

    // Own velocity (east, north) m/s.
    const voE = own.sog !== null && own.cog !== null ? own.sog * Math.sin(own.cog) : 0;
    const voN = own.sog !== null && own.cog !== null ? own.sog * Math.cos(own.cog) : 0;

    for (const tgt of targets) {
      if (!tgt.position || !own.position) continue;
      active.add(tgt.key);
      const cur = toLocal(own.position, tgt.position);
      const sample: Sample = {
        t,
        e: cur.e,
        n: cur.n,
        refLat: own.position.latitude,
        refLon: own.position.longitude,
      };
      const prev = this.history.get(tgt.key);
      this.history.set(tgt.key, sample);

      let vtE: number;
      let vtN: number;
      if (tgt.aisCorrelated && tgt.aisSog !== null && tgt.aisCog !== null) {
        // Prefer accurate AIS kinematics over noisy monocular finite-difference;
        // this also yields a solution on the very first sample (no baseline needed).
        vtE = tgt.aisSog * Math.sin(tgt.aisCog);
        vtN = tgt.aisSog * Math.cos(tgt.aisCog);
      } else {
        if (!prev) continue; // need a previous fix for finite-difference velocity
        const dt = t - prev.t;
        if (dt < 0.2) continue; // need a meaningful baseline
        // Re-express the previous target position in the current local frame so
        // the difference is a ground displacement.
        const prevAbs: LatLon = {
          latitude: prev.refLat + (prev.n / EARTH_RADIUS_M) * (180 / Math.PI),
          longitude:
            prev.refLon +
            (prev.e / (EARTH_RADIUS_M * Math.cos(deg2rad(prev.refLat)))) *
              (180 / Math.PI),
        };
        const prevInCur = toLocal(own.position, prevAbs);
        vtE = (cur.e - prevInCur.e) / dt;
        vtN = (cur.n - prevInCur.n) / dt;
      }

      // Relative motion: target relative to own.
      const rE = cur.e;
      const rN = cur.n;
      const vE = vtE - voE;
      const vN = vtN - voN;
      const vv = vE * vE + vN * vN;

      let cpa: number;
      let tcpa: number;
      if (vv < 1e-6) {
        tcpa = 0;
        cpa = Math.hypot(rE, rN);
      } else {
        tcpa = -(rE * vE + rN * vN) / vv;
        const cE = rE + vE * Math.max(tcpa, 0);
        const cN = rN + vN * Math.max(tcpa, 0);
        cpa = Math.hypot(cE, cN);
      }

      tgt.cpa = cpa;
      tgt.tcpa = tcpa;
      tgt.threatLevel = classify(cpa, tcpa, cfg);
    }

    for (const key of [...this.history.keys()]) {
      if (!active.has(key)) this.history.delete(key);
    }
  }

  reset(): void {
    this.history.clear();
  }
}

export function classify(cpa: number, tcpa: number, cfg: PluginConfig): ThreatLevel {
  if (tcpa <= 0) return 'none'; // diverging or passed
  if (cpa <= cfg.collisionCpaM && tcpa <= cfg.collisionAlarmTcpaS) return 'high';
  if (cpa <= cfg.collisionCpaM && tcpa <= cfg.collisionTcpaS) return 'medium';
  if (cpa <= cfg.collisionCpaM * 3 && tcpa <= cfg.collisionTcpaS) return 'low';
  return 'none';
}
