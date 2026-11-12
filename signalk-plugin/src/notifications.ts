// Lifecycle management for vision-driven SignalK notifications: MOB, dark
// targets and collision risk. Tracks active notifications so they can be
// cleared (value: null) when the condition resolves, with simple hysteresis to
// avoid flapping.

import { PluginConfig } from './config';
import { aisContext, sensorTargetId } from './sensorTargets';
import { ServerApp } from './skapp';
import { EnrichedTarget } from './types';

type State = 'normal' | 'alert' | 'warn' | 'alarm' | 'emergency';

// Notifications managed outside evaluate() (set/cleared by their own lifecycle),
// so the per-cycle clear loop must not touch them.
const EXTERNAL_PATHS = new Set([
  'notifications.vision.labelMismatch',
  'notifications.vision.schemaMismatch',
  'notifications.vision.staleEvents',
  'notifications.vision.containerDown',
  'notifications.vision.containerDegraded',
]);

/** The `$source` this plugin's notifications carry (their source label). */
export const NOTIFICATION_SOURCE = 'vision-ai';

const COLLISION_PREFIX = 'notifications.navigation.closestApproach.';
const DARK_PREFIX = 'notifications.vision.darkTarget.';

const isHoldable = (path: string): boolean =>
  path.startsWith(COLLISION_PREFIX) || path.startsWith(DARK_PREFIX);

/**
 * What the notifier needs from the rest of the plugin to alarm each boat once:
 * which targets the chart has, how fusion linked them, and whether another
 * plugin already alarms a boat (see alarmCoordination.ts).
 */
export interface AlarmCoordination {
  /** The sensor-target context a target is published under, or null. */
  publishedContext(t: EnrichedTarget): string | null;
  /** The `sameAs` link a fusion plugin set on a published context, or null. */
  sameAs(context: string): string | null;
  /** Contexts another producer currently raises a closest-approach alarm on. */
  othersAlarmed: Set<string>;
  /** Whether a linked `vessels.*` context is an AIS transmitter. A fusion
   * link may name a vessel some other sensor put in the model, which says
   * nothing about whether the boat sends AIS. */
  isAisVessel(context: string): boolean;
}

/** How fusion links the targets on the chart. */
export type TrackLinks = Pick<AlarmCoordination, 'publishedContext' | 'sameAs'>;

const NO_COORDINATION: AlarmCoordination = {
  publishedContext: () => null,
  sameAs: () => null,
  othersAlarmed: new Set(),
  isAisVessel: () => false,
};

/** One physical boat, as far as this plugin can tell, and how to name it. */
export interface Boat {
  // The context consumers locate the boat by: an AIS vessel a fusion link
  // names, else this plugin's AIS match, else the target a fusion link names,
  // else the track's own published target; null when nothing names it.
  targetRef: string | null;
  // Groups this cycle's alarms: targetRef, or the track when nothing names it.
  key: string;
  // The notification key this plugin raises the boat's alarms under.
  token: string;
  // Every context that names this boat, to match other producers' alarms.
  refs: string[];
  // Some track of the boat is an AIS vessel: by a fusion link to an AIS
  // transmitter, or by this plugin's own AIS match. Such a boat is never
  // called dark or "no AIS".
  hasAis: boolean;
  // The tracks of this boat.
  targets: EnrichedTarget[];
}

interface TrackRefs {
  t: EnrichedTarget;
  own: string | null;
  link: string | null;
  ais: string | null;
}

/**
 * The boats behind this cycle's tracks. Tracks that share any context — the
 * target one is published as, the target or vessel a fusion link names, the
 * AIS vessel this plugin matched — are one boat, followed transitively: a
 * stern track linked to the bow track's target, which matches an AIS vessel,
 * is that vessel.
 */
export function groupBoats(targets: EnrichedTarget[], coord: AlarmCoordination): Boat[] {
  const overridden = overriddenAisMatches(targets, coord);
  const tracks: TrackRefs[] = linkedTracks(targets, coord).map(({ t, own, link }) => ({
    t,
    own,
    link,
    ais: t.aisMmsi && !overridden.has(t.key) ? aisContext(t.aisMmsi) : null,
  }));
  return unionGroups(tracks, (track) => [track.own, track.link, track.ais])
    .map((members) => boatOf(members, coord.isAisVessel));
}

/**
 * The tracks whose own AIS match a fusion link overrides. A fusion link to an
 * AIS vessel identifies the boat; this plugin's own match is a position
 * match, and a different vessel there must neither name the boat nor carry
 * its collision assessment, nor let that vessel's alarm silence this boat's.
 * The link speaks for every track joined to it by published targets and
 * links, not only the track it is set on: a stern track linked to the bow
 * track's target is the bow track's vessel, whatever the stern track matched.
 */
export function overriddenAisMatches(targets: EnrichedTarget[], coord: TrackLinks): Set<string> {
  const overridden = new Set<string>();
  for (const group of unionGroups(linkedTracks(targets, coord), (track) => [track.own, track.link])) {
    const vessels = new Set(group.map((m) => m.link).filter((l) => l?.startsWith('vessels.') ?? false));
    if (vessels.size === 0) continue;
    for (const { t } of group) {
      if (t.aisMmsi && !vessels.has(aisContext(t.aisMmsi))) overridden.add(t.key);
    }
  }
  return overridden;
}

function linkedTracks(
  targets: EnrichedTarget[],
  coord: TrackLinks
): Array<{ t: EnrichedTarget; own: string | null; link: string | null }> {
  return targets.map((t) => {
    const own = coord.publishedContext(t);
    return { t, own, link: own ? coord.sameAs(own) : null };
  });
}

/** Items that share any ref, grouped transitively (union-find). */
function unionGroups<T>(items: T[], refsOf: (item: T) => Array<string | null>): T[][] {
  const parent = items.map((_, i) => i);
  const find = (i: number): number => {
    while (parent[i] !== i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  };
  const byRef = new Map<string, number>();
  items.forEach((item, i) => {
    for (const ref of refsOf(item)) {
      if (ref === null) continue;
      const j = byRef.get(ref);
      if (j === undefined) byRef.set(ref, i);
      else parent[find(i)] = find(j);
    }
  });
  const groups = new Map<number, T[]>();
  items.forEach((item, i) => {
    const root = find(i);
    groups.set(root, [...(groups.get(root) ?? []), item]);
  });
  return [...groups.values()];
}

const nonNull = (r: string | null): r is string => r !== null;
const firstOf = (refs: Array<string | null>): string | null => refs.filter(nonNull).sort()[0] ?? null;

/** Name a group of tracks: the most authoritative context first, by id so the
 * choice does not depend on the order tracks arrive in. */
function boatOf(members: TrackRefs[], isAisVessel: (context: string) => boolean): Boat {
  const links = members.map((m) => m.link);
  const targetRef =
    firstOf(links.filter((l) => l?.startsWith('vessels.') ?? false)) ??
    firstOf(members.map((m) => m.ais)) ??
    firstOf(links) ??
    firstOf(members.map((m) => m.own));
  const rep = members.map((m) => m.t).sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0))[0];
  const key = targetRef ?? `vision:${rep.key}`;
  return {
    targetRef,
    key,
    token: boatToken(key, rep),
    refs: [...new Set(members.flatMap((m) => [m.own, m.link, m.ais]).filter(nonNull))],
    hasAis: members.some((m) => m.ais !== null || (m.link !== null && isAisVessel(m.link))),
    targets: members.map((m) => m.t),
  };
}

/**
 * A notification key per boat, never one another producer uses: two sources
 * writing one path would let this plugin's clear (value null) wipe the other's
 * still-standing alarm off every display that keys alarms by path.
 *
 * Every key starts `visionai`, then keeps its kind apart so no two boats can
 * share one: this plugin's own target ids keep their `visionai-` form, and
 * every other kind follows `visionai_` with a name no own id can take
 * (`mmsi_` for an AIS vessel, `ctx_` for any other context, `key_` for a
 * track with no id).
 */
export function boatToken(key: string, rep: EnrichedTarget): string {
  const mmsi = /^vessels\.urn:mrn:imo:mmsi:(\d+)$/.exec(key)?.[1];
  if (mmsi) return `${OWN_KEY_PREFIX}_mmsi_${mmsi}`;
  // This plugin's own target ids are already one segment of plain
  // characters (see sensorTargets.ts) and are used as they are.
  const ownId = /^targets\.camera:(visionai-[a-zA-Z0-9_-]+)$/.exec(key)?.[1];
  if (ownId) return ownId;
  if (key.startsWith('vision:')) {
    const id = sensorTargetId(rep);
    return id ? id.slice(id.indexOf(':') + 1) : `${OWN_KEY_PREFIX}_key_${encodeToken(rep.key)}`;
  }
  return `${OWN_KEY_PREFIX}_ctx_${encodeToken(key.replace(/^(vessels|targets)\./, ''))}`;
}

/** What every notification key this plugin raises begins with, and no other
 * producer's does. */
export const OWN_KEY_PREFIX = 'visionai';

/**
 * Any context as one path segment, one-to-one: every character outside
 * [a-zA-Z0-9-] is written as its hex code between underscores, so contexts
 * that differ only in punctuation (`radar:a:b`, `radar:a_b`) keep apart.
 */
function encodeToken(text: string): string {
  return Array.from(text, (ch) =>
    /[a-zA-Z0-9-]/.test(ch) ? ch : `_${(ch.codePointAt(0) as number).toString(16)}_`
  ).join('');
}

const THREAT_RANK: Record<string, number> = { high: 2, medium: 1 };

/** The more urgent of two threatening targets: higher threat, then sooner. */
function moreUrgent(a: EnrichedTarget, b: EnrichedTarget): EnrichedTarget {
  const ra = THREAT_RANK[a.threatLevel] ?? 0;
  const rb = THREAT_RANK[b.threatLevel] ?? 0;
  if (ra !== rb) return ra > rb ? a : b;
  return (a.tcpa ?? Infinity) <= (b.tcpa ?? Infinity) ? a : b;
}

// The container reports a camera stall's age as a live counter ("forward: no
// frames for 26s"), so the degraded detail changes on every health poll while the
// fault itself does not. This is the part that identifies the fault — which
// cameras, which causes, how many restarts — for deciding when to log.
export function degradedFaultKey(detail: string): string {
  return detail.replace(/\bno frames for \d+s\b/g, 'no frames');
}

export class NotificationManager {
  private active = new Set<string>();
  private mobCounters = new Map<string, number>();
  // lastSeen at the time each MOB candidate was last counted, so the persistence
  // counter only advances on a genuinely new sighting (see evaluateMob).
  private mobLastCounted = new Map<string, number>();
  private mobHoldUntil = new Map<string, number>();
  // Anti-flap hold for collision/dark notifications: keep them up briefly after
  // they would clear so a target hovering around a threshold doesn't toggle the
  // audible alarm every cycle.
  private holdUntil = new Map<string, number>();
  // Target keys behind each collision/dark path raised last cycle, so a path
  // whose boat is now alarmed under another path is cleared at once instead of
  // being held alongside it.
  private members = new Map<string, Set<string>>();
  private lastMobMessage: string | null = null;

  constructor(
    private app: ServerApp,
    private pluginId: string,
    private cfg: PluginConfig
  ) {}

  /** Raise or update a notification on the own vessel; `extra` adds fields
   * such as a collision alarm's `data` beside state, method and message. */
  private send(
    path: string,
    state: State,
    message: string,
    methods: string[],
    extra: Record<string, unknown> = {}
  ): void {
    this.app.handleMessage(this.pluginId, {
      context: 'vessels.self',
      updates: [
        {
          source: { label: NOTIFICATION_SOURCE },
          timestamp: new Date().toISOString(),
          values: [{ path, value: { state, method: methods, message, ...extra } }],
        },
      ],
    });
    this.active.add(path);
  }

  private clear(path: string): void {
    if (!this.active.has(path)) return;
    this.app.handleMessage(this.pluginId, {
      context: 'vessels.self',
      updates: [
        {
          source: { label: NOTIFICATION_SOURCE },
          timestamp: new Date().toISOString(),
          values: [{ path, value: null }],
        },
      ],
    });
    this.active.delete(path);
  }

  /**
   * Evaluate the current target set and emit/clear notifications.
   *
   * Collision and dark-target alerts are raised per boat, not per camera
   * track: tracks fusion links together, or that match the same AIS vessel,
   * raise one alert, the most urgent. A boat another plugin already alarms
   * raises none here (see alarmCoordination.ts), and a boat under a collision
   * alarm raises no dark-target alert as well, the collision alarm saying it
   * has no AIS instead.
   */
  evaluate(
    targets: EnrichedTarget[],
    darkTargetKeys: Set<string>,
    nowMs: number,
    coord: AlarmCoordination = NO_COORDINATION
  ): void {
    const wantActive = new Set<string>();
    // Paths raised under this cycle's grouping -> their target keys.
    const raised = new Map<string, Set<string>>();
    // Paths to clear now, skipping their anti-flap hold: another alarm took
    // over the boat, and holding them would alarm it twice.
    const release = new Set<string>();

    if (this.cfg.enableMob) this.evaluateMob(targets, nowMs, wantActive);
    const boats = groupBoats(targets, coord);
    // Boats another producer already alarms, whether or not this plugin
    // raises collision alarms itself: none gets a dark-target alert either.
    const underCollisionAlarm = new Set<string>();
    for (const boat of boats) {
      if (boat.refs.some((r) => coord.othersAlarmed.has(r))) underCollisionAlarm.add(boat.key);
    }
    // Collision/dark notifications need their computation on (they read values
    // it produces) and the notification itself enabled.
    if (this.cfg.enableCollision && this.cfg.notifyCollision) {
      this.evaluateCollision(boats, darkTargetKeys, wantActive, raised, release, underCollisionAlarm);
    }
    if (this.cfg.enableAisFusion && this.cfg.notifyDarkTarget) {
      this.evaluateDark(boats, darkTargetKeys, underCollisionAlarm, wantActive, raised, release);
    }

    // A path held from last cycle whose boat is now alarmed under another path
    // (its grouping changed: a fusion link or AIS match arrived) goes at once.
    const nowRaisedFor = new Map<string, string>();
    for (const [path, keys] of raised) for (const k of keys) nowRaisedFor.set(k, path);
    for (const [path, keys] of this.members) {
      if (raised.has(path)) continue;
      for (const k of keys) {
        const other = nowRaisedFor.get(k);
        if (other !== undefined && other !== path) release.add(path);
      }
    }
    for (const [path, keys] of raised) this.members.set(path, keys);
    for (const path of [...this.members.keys()]) {
      if (!raised.has(path) && (release.has(path) || !this.active.has(path))) this.members.delete(path);
    }

    this.applyHold(wantActive, nowMs, release);

    // Clear any previously-active notification no longer wanted. Skip paths owned
    // by a separate lifecycle (label/schema mismatch) so we don't clear them here.
    for (const path of [...this.active]) {
      if (EXTERNAL_PATHS.has(path)) continue;
      if (!wantActive.has(path)) this.clear(path);
    }
  }

  /** Release every path of a kind raised for any of the boat's tracks under
   * an earlier grouping, whose name (token) may differ from the current one. */
  private releaseHeld(prefix: string, boat: Boat, release: Set<string>): void {
    const keys = new Set(boat.targets.map((t) => t.key));
    for (const [held, members] of this.members) {
      if (held.startsWith(prefix) && [...members].some((k) => keys.has(k))) release.add(held);
    }
  }

  /** Refresh the hold for currently-wanted holdable paths, then re-add any
   * held path whose hold hasn't expired so it isn't cleared yet (anti-flap).
   * A released path drops its hold. */
  private applyHold(want: Set<string>, nowMs: number, release: Set<string>): void {
    const holdMs = this.cfg.notifyHoldS * 1000;
    for (const path of want) {
      if (isHoldable(path)) this.holdUntil.set(path, nowMs + holdMs);
    }
    for (const path of release) {
      if (!want.has(path)) this.holdUntil.delete(path);
    }
    for (const [path, until] of [...this.holdUntil]) {
      if (until <= nowMs) {
        this.holdUntil.delete(path);
      } else if (!want.has(path) && this.active.has(path)) {
        want.add(path);
      }
    }
  }

  private evaluateMob(targets: EnrichedTarget[], nowMs: number, want: Set<string>): void {
    const path = 'notifications.mob';
    let fire = false;
    const qualifying = new Set<string>();

    for (const t of targets) {
      if (!t.is_person_in_water || t.confidence < this.cfg.mobMinConfidence) continue;
      qualifying.add(t.key);
      // Targets outlive their detections (retained trackTimeoutS after the last
      // frame), so only count when lastSeen has actually advanced — otherwise a
      // SINGLE sighting lingering in the map would count itself once per process
      // cycle and satisfy mobPersistFrames on its own, defeating the
      // false-positive guard the persistence threshold exists to provide.
      const countedAt = this.mobLastCounted.get(t.key);
      if (countedAt !== undefined && t.lastSeen <= countedAt) continue;
      this.mobLastCounted.set(t.key, t.lastSeen);
      const c = (this.mobCounters.get(t.key) ?? 0) + 1;
      this.mobCounters.set(t.key, c);
      if (c >= this.cfg.mobPersistFrames) {
        fire = true;
        // Hold the alarm for 60s after last sighting (drift / occlusion).
        this.mobHoldUntil.set(t.key, nowMs + 60000);
        // Capture the last-known position/bearing so the message stays useful
        // even if the detection drops out during the hold window.
        if (t.position) {
          this.lastMobMessage =
            `MAN OVERBOARD — person in water at ` +
            `${t.position.latitude.toFixed(5)}, ${t.position.longitude.toFixed(5)}` +
            ` (${t.camera} camera)`;
        } else {
          this.lastMobMessage = `MAN OVERBOARD — person in water, bearing ${
            t.bearingTrue !== null ? ((t.bearingTrue * 180) / Math.PI).toFixed(0) + '°T' : 'unknown'
          } (${t.camera} camera)`;
        }
      }
    }

    // Decay counters for tracks that no longer QUALIFY as a MOB candidate —
    // not merely tracks that left the map. A retained track whose latest frame
    // stopped qualifying (reclassified out of the water, or confidence dropped)
    // must restart its persistence count: otherwise an old partial count could
    // sit dormant on a non-qualifying track and later fire the alarm off a
    // single fresh qualifying frame.
    for (const key of [...this.mobCounters.keys()]) {
      if (!qualifying.has(key)) {
        this.mobCounters.delete(key);
        this.mobLastCounted.delete(key);
      }
    }

    // Honour hold windows even if the detection briefly drops out.
    if (!fire) {
      for (const [, until] of this.mobHoldUntil) {
        if (until > nowMs) {
          fire = true;
          break;
        }
      }
    }
    for (const [key, until] of [...this.mobHoldUntil]) {
      if (until <= nowMs) this.mobHoldUntil.delete(key);
    }

    if (fire) {
      this.send(path, 'emergency',
        this.lastMobMessage ?? 'Man overboard detected by vision system',
        ['visual', 'sound']);
      want.add(path);
    } else {
      this.lastMobMessage = null;
    }
  }

  /** One dark-target alert per boat that has no AIS, unless the boat is
   * under a collision alarm (this plugin's or another producer's). */
  private evaluateDark(
    boats: Boat[],
    darkKeys: Set<string>,
    underCollisionAlarm: Set<string>,
    want: Set<string>,
    raised: Map<string, Set<string>>,
    release: Set<string>
  ): void {
    for (const boat of boats) {
      const path = `${DARK_PREFIX}${boat.token}`;
      // A boat some track of which is an AIS vessel is not dark, whatever one
      // linked track's own AIS correlation says, and a boat under a collision
      // alarm needs no second alert. Either way a standing dark-target alert
      // goes at once, before the check below that skips boats with no dark
      // track: a boat whose only dark track just matched AIS has none.
      if (boat.hasAis || underCollisionAlarm.has(boat.key)) {
        release.add(path);
        this.releaseHeld(DARK_PREFIX, boat, release);
        continue;
      }
      const dark = boat.targets.filter((t) => darkKeys.has(t.key));
      if (dark.length === 0) continue;
      // The nearest sighting speaks for the boat.
      const t = dark.reduce((a, b) =>
        (a.geometry.range_m ?? Infinity) <= (b.geometry.range_m ?? Infinity) ? a : b);
      const rng = t.geometry.range_m !== null ? `${t.geometry.range_m.toFixed(0)} m` : 'unknown range';
      const brg = t.bearingTrue !== null ? `${((t.bearingTrue * 180) / Math.PI).toFixed(0)}°T` : '';
      this.send(path, 'alert', `Non-AIS ${t.label} detected at ${brg} ${rng}`, ['visual']);
      want.add(path);
      // Every track of the boat, so the alert is released if a later grouping
      // raises it under another path, whichever track qualifies then.
      raised.set(path, new Set(boat.targets.map((d) => d.key)));
    }
  }

  /**
   * One closest-approach alarm per boat, on the standard
   * `notifications.navigation.closestApproach.*` path in the shared collision
   * alarm shape, so chartplotters draw it (Freeboard: a ring on the boat and
   * both course lines on to the closest approach). A boat another producer
   * already alarms raises none here, and an alarm this plugin still holds for
   * it is released at once, even when the boat no longer threatens. Until
   * such an alarm exists, this plugin keeps alarming a boat it finds risky,
   * Collision Alerts running or not: its limits may differ from this
   * plugin's, and a boat must not go without an alarm meanwhile.
   */
  private evaluateCollision(
    boats: Boat[],
    darkKeys: Set<string>,
    want: Set<string>,
    raised: Map<string, Set<string>>,
    release: Set<string>,
    underCollisionAlarm: Set<string>
  ): void {
    for (const boat of boats) {
      const path = `${COLLISION_PREFIX}${boat.token}`;
      if (underCollisionAlarm.has(boat.key)) {
        // Another producer's alarm stands for this boat: ours goes at once.
        release.add(path);
        this.releaseHeld(COLLISION_PREFIX, boat, release);
        continue;
      }
      const risky = boat.targets.filter((t) => t.threatLevel === 'medium' || t.threatLevel === 'high');
      if (risky.length === 0) continue;
      const t = risky.reduce(moreUrgent);
      const state: State = t.threatLevel === 'high' ? 'alarm' : 'warn';
      const tcpa = t.tcpa !== null ? `${(t.tcpa / 60).toFixed(1)} min` : '?';
      const cpa = t.cpa !== null ? `${t.cpa.toFixed(0)} m` : '?';
      // The vessel the boat is named by, a fusion link before this plugin's
      // own match, and "no AIS" only when AIS was checked and no track is AIS.
      const mmsi = /^vessels\.urn:mrn:imo:mmsi:(\d+)$/.exec(boat.targetRef ?? '')?.[1];
      const noAis = !boat.hasAis && risky.some((r) => darkKeys.has(r.key));
      const who = mmsi
        ? `${boat.hasAis ? 'AIS' : 'vessel'} ${mmsi}${noAis ? ' (no AIS)' : ''}`
        : `${t.label}${noAis ? ' (no AIS)' : ''}`;
      this.send(path, state, `Collision risk: ${who} CPA ${cpa} in ${tcpa}`, ['visual', 'sound'],
        collisionData(boat, t));
      want.add(path);
      raised.set(path, new Set(boat.targets.map((r) => r.key)));
      underCollisionAlarm.add(boat.key);
    }
  }

  setLabelMismatch(message: string): void {
    this.send('notifications.vision.labelMismatch', 'warn', message, ['visual']);
  }

  clearLabelMismatch(): void {
    this.clear('notifications.vision.labelMismatch');
  }

  setSchemaMismatch(message: string): void {
    this.send('notifications.vision.schemaMismatch', 'warn', message, ['visual']);
  }

  clearSchemaMismatch(): void {
    this.clear('notifications.vision.schemaMismatch');
  }

  setStaleEvents(message: string): void {
    this.send('notifications.vision.staleEvents', 'warn', message, ['visual']);
  }

  clearStaleEvents(): void {
    this.clear('notifications.vision.staleEvents');
  }

  // Container unreachable: the whole vision sensor is offline (no events, no
  // stream). Raised at `alarm` so it's distinct from the data-quality warnings
  // above, but visual-only to avoid an audible alarm flapping on a routine
  // container restart — the plugin recovers on its own when it comes back.
  setContainerDown(message: string): void {
    this.send('notifications.vision.containerDown', 'alarm', message, ['visual']);
  }

  clearContainerDown(): void {
    this.clear('notifications.vision.containerDown');
  }

  // Container reachable but reporting status="degraded" (camera/RTSP stall or a
  // pipeline auto-restart). The sensor is partially working, so this is a warn.
  setContainerDegraded(message: string): void {
    this.send('notifications.vision.containerDegraded', 'warn', message, ['visual']);
  }

  clearContainerDegraded(): void {
    this.clear('notifications.vision.containerDegraded');
  }

  /** Forget per-track MOB persistence counts, as when the tracks behind them
   * are gone (a new event-stream connection). A MOB alarm already raised
   * keeps its hold; only the counts towards a new one restart. */
  forgetTracks(): void {
    this.mobCounters.clear();
    this.mobLastCounted.clear();
  }

  clearAll(): void {
    for (const path of [...this.active]) this.clear(path);
    this.mobCounters.clear();
    this.mobLastCounted.clear();
    this.mobHoldUntil.clear();
    this.holdUntil.clear();
    this.members.clear();
    this.lastMobMessage = null;
  }
}

/**
 * The notification's `data`, in the shape Collision Alerts and MaYaRa share,
 * plus `other` for consumers older than `data.targetRef`. Without a published
 * target or an AIS match there is no context to name, and Freeboard then draws
 * the course lines from `cpaPositions` alone.
 */
function collisionData(boat: Boat, t: EnrichedTarget): Record<string, unknown> {
  const data: Record<string, unknown> = {
    source: 'camera',
    cpa: t.cpa,
    tcpa: t.tcpa,
    targetCog: t.sog !== null && t.sog > 0 ? t.cog : null,
    targetSog: t.sog,
  };
  if (boat.targetRef) data.targetRef = boat.targetRef;
  if (t.approach) {
    data.range = t.approach.range;
    data.cpaPositions = { self: t.approach.self, target: t.approach.target };
  }
  return boat.targetRef ? { data, other: boat.targetRef } : { data };
}
