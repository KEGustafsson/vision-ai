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
  typeof v === 'number' && Number.isFinite(v) ? v : null;

// Read a numeric AIS field with freshness: a SignalK node may be { value,
// timestamp } or a bare value. When a max age is set and the node carries a
// timestamp, a value older than it is dropped to null so stale kinematics can't
// be fed into CPA (a vessel that stopped transmitting keeps its last COG/SOG in
// SignalK indefinitely). Bare/timestamp-less values pass through num().
const freshNum = (node: any, maxAgeMs: number, nowMs: number): number | null => {
  if (node && typeof node === 'object' && 'value' in node) {
    if (maxAgeMs > 0 && typeof node.timestamp === 'string') {
      const ageMs = nowMs - Date.parse(node.timestamp);
      // Reject both too-old AND future-dated values (negative age, e.g. a
      // transmitter/clock ahead of us): a future timestamp would otherwise read
      // as "fresh" indefinitely. Fail closed on anything outside the window.
      if (!Number.isFinite(ageMs) || ageMs < 0 || ageMs > maxAgeMs) return null;
    }
    return num(node.value);
  }
  return num(node);
};

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
    // Validate BOTH lat and lon with Number.isFinite (NOT typeof number, which
    // admits NaN/±Infinity): a non-finite coordinate would produce NaN
    // range/bearing/score that silently corrupts the association and could
    // suppress a real dark-target alarm.
    if (!p || !Number.isFinite(p.latitude) || !Number.isFinite(p.longitude)) continue;
    // Stale-data handling, FAIL-CLOSED: SignalK retains an AIS vessel's last-known
    // position long after it stops transmitting. A stale fix would let a
    // no-longer-transmitting vessel keep correlating (suppressing its dark-target
    // alarm) and feed a wrong AIS SOG/COG into CPA. So when the age check is on
    // (aisMaxAgeS > 0) we require a parseable, recent position timestamp — a
    // contact we can't prove is fresh is dropped entirely rather than trusted.
    // Set aisMaxAgeS = 0 to disable (then timestamp-less contacts are kept).
    if (maxAgeMs > 0) {
      const ts = typeof posNode?.timestamp === 'string' ? Date.parse(posNode.timestamp) : NaN;
      const ageMs = nowMs - ts;
      // Drop anything without a parseable timestamp, older than the window, or
      // future-dated (negative age) — a future fix would otherwise stay "fresh".
      if (!Number.isFinite(ageMs) || ageMs < 0 || ageMs > maxAgeMs) continue;
    }
    const pos: LatLon = { latitude: p.latitude, longitude: p.longitude };
    const range = haversine(own.position, pos);
    if (!Number.isFinite(range)) continue;
    if (minRangeM > 0 && range < minRangeM) continue;
    const bearing = bearingTo(own.position, pos);
    if (!Number.isFinite(bearing)) continue;
    out.push({
      mmsi,
      aisClass,
      name: v?.name?.value ?? v?.name,
      position: pos,
      bearing,
      range,
      cog: freshNum(v?.navigation?.courseOverGroundTrue, maxAgeMs, nowMs),
      sog: freshNum(v?.navigation?.speedOverGround, maxAgeMs, nowMs),
    });
  }
  return out;
}

export interface FusionResult {
  targets: EnrichedTarget[];
  darkTargetKeys: string[];
  aisCorrelatedCount: number;
  // key -> mmsi for the matches made this pass. Feed back as `prev` next cycle:
  // a stable target then keeps its identity instead of flapping between two
  // bearing/range-coincident contacts as the noisy monocular range wanders.
  assignment: Map<string, string>;
}

// AIS classes that denote an actual moving vessel. A "vessel"-labelled visual
// detection must not bind to an aid-to-navigation or base station that merely
// happens to share its bearing/range — those can only explain non-vessel
// detections (e.g. a buoy correlating to an ATON).
const VESSEL_AIS_CLASSES = new Set(['A', 'B']);

// Correlation tunables. Kept here (not as operator config) because they are
// qualitative gate-shaping factors rather than values a captain would tune; the
// user-facing knobs stay correlationBearingDeg / correlationRangeFrac.
const BEARING_ONLY_TOL_FACTOR = 0.6; // tighten the bearing gate when there is no visual range to back it up
const HYSTERESIS_BONUS = 0.7;        // score multiplier for sticking with last cycle's match (lower = stickier)
const DARK_NEAR_MISS_FACTOR = 1.5;   // a contact within this × the gate makes a "dark" call ambiguous, so we hold off
const KINEMATIC_WEIGHT = 0.5;        // weight of course disagreement in the match score
const KINEMATIC_MIN_SOG = 0.5;       // m/s below which a course is not meaningful enough to compare

const clamp = (v: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, v));

// A vessel detection may only correlate with a transmitting vessel (class A/B);
// any label may correlate with any class (so a buoy can match an ATON).
function classAllowed(label: string, aisClass: string): boolean {
  if (VESSEL_LABELS.has(label)) return VESSEL_AIS_CLASSES.has(aisClass.trim().toUpperCase());
  return true;
}

// Gate half-widths for a target. The range gate tightens with the container's
// reported range_confidence (a trusted monocular range gates hard; a shaky one
// gates loose) and the bearing gate tightens when there is no range to support
// it. `widen` inflates both for the dark-target near-miss test.
function gatesFor(t: EnrichedTarget, cfg: PluginConfig, widen = 1) {
  const range = t.geometry.range_m;
  const hasRange = range !== null;
  const rangeConf = t.geometry.range_confidence ?? 0;
  const confFactor = clamp(1.3 - rangeConf, 0.2, 1.4);
  const bearingTol =
    deg2rad(cfg.correlationBearingDeg) * (hasRange ? 1 : BEARING_ONLY_TOL_FACTOR) * widen;
  const rangeTol = hasRange
    ? Math.max(cfg.correlationRangeFrac * (range as number) * confFactor, 50) * widen
    : 0;
  return { range, hasRange, bearingTol, rangeTol };
}

// A normalized (dimensionless) residual for a target/contact pair, or null when
// the pair does not gate. Bearing and range each contribute their fraction of
// the gate, so the two are weighed on the same scale; course disagreement is a
// soft add-on used only when both sides have a meaningful motion estimate.
function pairScore(
  t: EnrichedTarget,
  a: AisContact,
  cfg: PluginConfig
): number | null {
  if (t.bearingTrue === null) return null;
  if (!classAllowed(t.label, a.aisClass)) return null;
  const { range, hasRange, bearingTol, rangeTol } = gatesFor(t, cfg);
  // A zero/negative bearing gate (correlationBearingDeg misconfigured to 0)
  // would make the dB/bearingTol score NaN and corrupt the assignment sort.
  // Treat it as "nothing correlates" rather than dividing by it.
  if (bearingTol <= 0) return null;
  const dB = angularDiff(t.bearingTrue, a.bearing);
  if (dB > bearingTol) return null;
  let score = dB / bearingTol;
  if (hasRange) {
    const dR = Math.abs((range as number) - a.range);
    if (dR > rangeTol) return null;
    score += dR / rangeTol;
  }
  // Course agreement disambiguates crossing traffic. Engages once a visual
  // motion estimate is present (CPA fills t.sog/t.cog on a later cycle); a
  // no-op until then, so it never hurts a first-sighting match.
  if (
    t.cog != null && t.sog != null && a.cog != null && a.sog != null &&
    t.sog >= KINEMATIC_MIN_SOG && a.sog >= KINEMATIC_MIN_SOG
  ) {
    score += KINEMATIC_WEIGHT * (angularDiff(t.cog, a.cog) / Math.PI);
  }
  // Defensive: a non-finite score (from any unexpected NaN/±Infinity input) must
  // not enter the assignment sort, where it would scramble the ordering and could
  // let a target borrow the wrong identity (suppressing a dark-target alarm).
  if (!Number.isFinite(score)) return null;
  return score;
}

// True when some plausible AIS contact sits just outside the correlation gate —
// the detection is ambiguous (likely that vessel, lost to noise) rather than a
// confident non-transmitter, so we suppress the dark-target call.
function hasNearMissContact(t: EnrichedTarget, ais: AisContact[], cfg: PluginConfig): boolean {
  if (t.bearingTrue === null) return false;
  const { range, hasRange, bearingTol, rangeTol } = gatesFor(t, cfg, DARK_NEAR_MISS_FACTOR);
  for (const a of ais) {
    if (!classAllowed(t.label, a.aisClass)) continue;
    if (angularDiff(t.bearingTrue, a.bearing) > bearingTol) continue;
    if (hasRange && Math.abs((range as number) - a.range) > rangeTol) continue;
    return true;
  }
  return false;
}

export function fuse(
  targets: EnrichedTarget[],
  ais: AisContact[],
  cfg: PluginConfig,
  prev: Map<string, string> = new Map()
): FusionResult {
  const darkTargetKeys: string[] = [];
  const assignment = new Map<string, string>();

  // Build every gating target/contact pair, then assign one-to-one in ascending
  // score order. This makes the association mutually exclusive: a single AIS
  // contact can no longer be claimed by two visual targets (which would let one
  // real dark target borrow another's identity and suppress its alarm). Greedy
  // global-minimum order is a standard, cheap near-optimal data association.
  interface Pair { tKey: string; aIdx: number; score: number; }
  const pairs: Pair[] = [];
  for (const t of targets) {
    for (let i = 0; i < ais.length; i++) {
      const s = pairScore(t, ais[i], cfg);
      if (s === null) continue;
      // Stickiness: discount keeping last cycle's identity while it still gates.
      pairs.push({ tKey: t.key, aIdx: i, score: prev.get(t.key) === ais[i].mmsi ? s * HYSTERESIS_BONUS : s });
    }
  }
  pairs.sort((x, y) => x.score - y.score);

  const assignedContact = new Map<string, AisContact>();
  const usedContacts = new Set<number>();
  for (const p of pairs) {
    if (assignedContact.has(p.tKey) || usedContacts.has(p.aIdx)) continue;
    assignedContact.set(p.tKey, ais[p.aIdx]);
    usedContacts.add(p.aIdx);
  }

  let aisCorrelatedCount = 0;
  for (const t of targets) {
    // Reset correlation state every pass — targets persist across cycles, so a
    // target that loses its AIS match must not retain stale identity/kinematics.
    t.aisCorrelated = false;
    t.aisMmsi = null;
    t.aisCog = null;
    t.aisSog = null;

    const a = assignedContact.get(t.key);
    if (a) {
      t.aisCorrelated = true;
      t.aisMmsi = a.mmsi;
      t.aisCog = a.cog;
      t.aisSog = a.sog;
      assignment.set(t.key, a.mmsi);
      aisCorrelatedCount += 1;
      continue;
    }

    const range = t.geometry.range_m;
    if (
      VESSEL_LABELS.has(t.label) &&
      range !== null &&
      range <= cfg.darkTargetRangeM &&
      !hasNearMissContact(t, ais, cfg)
    ) {
      darkTargetKeys.push(t.key);
    }
  }

  return { targets, darkTargetKeys, aisCorrelatedCount, assignment };
}
