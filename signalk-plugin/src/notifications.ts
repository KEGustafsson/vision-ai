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
  /** Collision Alerts runs and evaluates AIS vessels and sensor targets. */
  collisionAlertsRunning: boolean;
  /** Contexts another producer currently raises a closest-approach alarm on. */
  othersAlarmed: Set<string>;
}

const NO_COORDINATION: AlarmCoordination = {
  publishedContext: () => null,
  sameAs: () => null,
  collisionAlertsRunning: false,
  othersAlarmed: new Set(),
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
  // Collision Alerts sees this boat: it is published, or it is an AIS vessel.
  seenByCollisionAlerts: boolean;
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
  const tracks: TrackRefs[] = targets.map((t) => {
    const own = coord.publishedContext(t);
    return {
      t,
      own,
      link: own ? coord.sameAs(own) : null,
      ais: t.aisMmsi ? aisContext(t.aisMmsi) : null,
    };
  });
  const parent = tracks.map((_, i) => i);
  const find = (i: number): number => {
    while (parent[i] !== i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  };
  const byRef = new Map<string, number>();
  tracks.forEach((track, i) => {
    for (const ref of [track.own, track.link, track.ais]) {
      if (ref === null) continue;
      const j = byRef.get(ref);
      if (j === undefined) byRef.set(ref, i);
      else parent[find(i)] = find(j);
    }
  });
  const groups = new Map<number, TrackRefs[]>();
  tracks.forEach((track, i) => {
    const root = find(i);
    groups.set(root, [...(groups.get(root) ?? []), track]);
  });
  return [...groups.values()].map(boatOf);
}

const nonNull = (r: string | null): r is string => r !== null;
const firstOf = (refs: Array<string | null>): string | null => refs.filter(nonNull).sort()[0] ?? null;

/** Name a group of tracks: the most authoritative context first, by id so the
 * choice does not depend on the order tracks arrive in. */
function boatOf(members: TrackRefs[]): Boat {
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
    seenByCollisionAlerts: members.some(
      (m) => m.own !== null || m.ais !== null || (m.link?.startsWith('vessels.') ?? false)
    ),
    targets: members.map((m) => m.t),
  };
}

/**
 * A notification key per boat, never one another producer uses: two sources
 * writing one path would let this plugin's clear (value null) wipe the other's
 * still-standing alarm off every display that keys alarms by path.
 */
export function boatToken(key: string, rep: EnrichedTarget): string {
  const mmsi = /^vessels\.urn:mrn:imo:mmsi:(\d+)$/.exec(key)?.[1];
  if (mmsi) return `visionai-mmsi-${mmsi}`;
  const ownId = /^targets\.camera:(visionai-.+)$/.exec(key)?.[1];
  if (ownId) return sanitizeToken(ownId);
  if (key.startsWith('vision:')) {
    const id = sensorTargetId(rep);
    return sanitizeToken(id ? id.slice(id.indexOf(':') + 1) : `visionai-${rep.key}`);
  }
  return sanitizeToken(`visionai-${key.replace(/^(vessels|targets)\./, '')}`);
}

/** Keep a notification key to one path segment of plain characters. */
function sanitizeToken(token: string): string {
  return token.replace(/[^a-zA-Z0-9_-]/g, '_');
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
      this.evaluateCollision(boats, darkTargetKeys, coord, wantActive, raised, release, underCollisionAlarm);
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
      const dark = boat.targets.filter((t) => darkKeys.has(t.key));
      if (dark.length === 0) continue;
      const path = `${DARK_PREFIX}${boat.token}`;
      if (underCollisionAlarm.has(boat.key)) {
        release.add(path);
        continue;
      }
      // The nearest sighting speaks for the boat.
      const t = dark.reduce((a, b) =>
        (a.geometry.range_m ?? Infinity) <= (b.geometry.range_m ?? Infinity) ? a : b);
      const rng = t.geometry.range_m !== null ? `${t.geometry.range_m.toFixed(0)} m` : 'unknown range';
      const brg = t.bearingTrue !== null ? `${((t.bearingTrue * 180) / Math.PI).toFixed(0)}°T` : '';
      this.send(path, 'alert', `Non-AIS ${t.label} detected at ${brg} ${rng}`, ['visual']);
      want.add(path);
      raised.set(path, new Set(dark.map((d) => d.key)));
    }
  }

  /**
   * One closest-approach alarm per boat, on the standard
   * `notifications.navigation.closestApproach.*` path in the shared collision
   * alarm shape, so chartplotters draw it (Freeboard: a ring on the boat and
   * both course lines on to the closest approach). A boat another plugin
   * covers raises none here, and an alarm this plugin still holds for it is
   * released at once, even when the boat no longer threatens.
   */
  private evaluateCollision(
    boats: Boat[],
    darkKeys: Set<string>,
    coord: AlarmCoordination,
    want: Set<string>,
    raised: Map<string, Set<string>>,
    release: Set<string>,
    underCollisionAlarm: Set<string>
  ): void {
    for (const boat of boats) {
      const path = `${COLLISION_PREFIX}${boat.token}`;
      const covered =
        underCollisionAlarm.has(boat.key) ||
        (coord.collisionAlertsRunning && boat.seenByCollisionAlerts);
      if (covered) {
        release.add(path);
        // A path raised under an earlier grouping of these tracks goes too.
        const keys = new Set(boat.targets.map((t) => t.key));
        for (const [held, members] of this.members) {
          if (held.startsWith(COLLISION_PREFIX) && [...members].some((k) => keys.has(k))) {
            release.add(held);
          }
        }
        continue;
      }
      const risky = boat.targets.filter((t) => t.threatLevel === 'medium' || t.threatLevel === 'high');
      if (risky.length === 0) continue;
      const t = risky.reduce(moreUrgent);
      const state: State = t.threatLevel === 'high' ? 'alarm' : 'warn';
      const tcpa = t.tcpa !== null ? `${(t.tcpa / 60).toFixed(1)} min` : '?';
      const cpa = t.cpa !== null ? `${t.cpa.toFixed(0)} m` : '?';
      const noAis = risky.some((r) => darkKeys.has(r.key));
      const who = t.aisMmsi ? `AIS ${t.aisMmsi}` : `${t.label}${noAis ? ' (no AIS)' : ''}`;
      this.send(path, state, `Collision risk: ${who} CPA ${cpa} in ${tcpa}`, ['visual', 'sound'],
        collisionData(boat, t));
      want.add(path);
      raised.set(path, new Set(risky.map((r) => r.key)));
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
