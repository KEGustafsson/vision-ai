// No double alarms: a boat must alarm once, whichever plugins see it.
//
// A closest-approach alarm another producer raised
// (`notifications.navigation.closestApproach.*`: Collision Alerts, MaYaRa,
// SK Derived Data) that names the same boat makes this plugin stand down for
// it, at once, and take the boat back the moment that alarm clears. That is
// fail-safe: only an alarm that is actually there makes this plugin stand
// down, so a boat this plugin finds risky is never left without one, even
// where another producer's limits differ. A missing alarm is worse than a
// duplicate, so anything unreadable leaves this plugin's own alarm up.

import { ServerApp } from './skapp';

const ACTIVE_STATES = new Set(['alert', 'warn', 'alarm', 'emergency']);

const CPA_PATH_PREFIX = 'notifications.navigation.closestApproach.';

/**
 * Whether a delta raises, updates or clears another producer's
 * closest-approach alarm: any value on a key this plugin does not use, from a
 * source other than this plugin's. A clear matters as much as a raise: the
 * boat may now need this plugin's own alarm.
 */
export function touchesForeignAlarm(
  delta: { updates?: Array<{ $source?: unknown; values?: Array<{ path: string; value: unknown }> }> },
  ownLabel: string,
  ownKeyPrefix = 'visionai'
): boolean {
  for (const u of delta.updates ?? []) {
    if (isOwnSource(u.$source, ownLabel)) continue;
    for (const v of u.values ?? []) {
      if (!v.path.startsWith(CPA_PATH_PREFIX)) continue;
      if (!v.path.slice(CPA_PATH_PREFIX.length).startsWith(ownKeyPrefix)) return true;
    }
  }
  return false;
}

/**
 * Calls back as soon as another producer raises or clears a closest-approach
 * alarm, so this plugin stands down for a boat the moment another alarm
 * covers it, and takes it back the moment that alarm goes, instead of on its
 * next process cycle: a double alarm, or none, for up to a cycle.
 */
export class ForeignAlarmWatch {
  private unsubscribes: Array<() => void> = [];

  start(app: Pick<ServerApp, 'subscriptionmanager'>, ownLabel: string, onRaised: () => void): void {
    this.stop();
    const sm = app.subscriptionmanager;
    if (!sm || typeof sm.subscribe !== 'function') return;
    sm.subscribe(
      {
        context: 'vessels.self',
        subscribe: [{ path: 'notifications.navigation.closestApproach.*', policy: 'instant' }],
      },
      this.unsubscribes,
      () => undefined,
      (delta) => {
        if (touchesForeignAlarm(delta, ownLabel)) onRaised();
      }
    );
  }

  stop(): void {
    for (const unsubscribe of this.unsubscribes.splice(0)) {
      try {
        unsubscribe();
      } catch {
        // already gone with the server's subscription
      }
    }
  }
}

/**
 * The contexts other producers currently raise a closest-approach alarm on.
 * A notification names its boat in `data.targetRef` (the shared collision
 * alarm shape), in the older `other` field, or only by its path key, which
 * Collision Alerts and MaYaRa set to the target id (`urn:mrn:imo:mmsi:…` for
 * AIS, `radar:…` / `camera:…` for sensor targets).
 *
 * This plugin's own alarms are left out twice over: by path, since every one
 * it raises is keyed `visionai…` and no other producer uses that prefix, and
 * by source, since servers store a plugin's source label either bare
 * (`vision-ai`) or with a suffix (`vision-ai.XX` on older ones).
 * @param ownLabel the source label this plugin's notifications carry
 * @param ownKeyPrefix the path key prefix this plugin's alarms use
 */
export function othersAlarmedRefs(
  app: ServerApp,
  ownLabel: string,
  ownKeyPrefix = 'visionai'
): Set<string> {
  const refs = new Set<string>();
  let tree: unknown;
  try {
    tree = app.getSelfPath('notifications.navigation.closestApproach');
  } catch {
    return refs;
  }
  if (tree === null || typeof tree !== 'object') return refs;
  for (const [key, node] of Object.entries(tree as Record<string, unknown>)) {
    if (node === null || typeof node !== 'object' || key === 'value' || key === 'meta') continue;
    if (key.startsWith(ownKeyPrefix)) continue;
    for (const entry of sourcedValues(node as Record<string, unknown>)) {
      if (isOwnSource(entry.source, ownLabel)) continue;
      const v = entry.value as {
        state?: unknown;
        other?: unknown;
        data?: { targetRef?: unknown };
      } | null;
      if (!v || typeof v.state !== 'string' || !ACTIVE_STATES.has(v.state)) continue;
      const named = [v.data?.targetRef, v.other].filter(
        (r): r is string => typeof r === 'string' && r.length > 0
      );
      for (const r of named.length > 0 ? named : [refFromKey(key)]) refs.add(r);
    }
  }
  return refs;
}

/** Whether a `$source` is this plugin's label, bare or with a suffix. */
function isOwnSource(source: unknown, ownLabel: string): boolean {
  return typeof source === 'string' && (source === ownLabel || source.startsWith(`${ownLabel}.`));
}

/** A leaf's value per source: its latest value with its `$source`, or every
 * source's own value under `values` when several write it. */
function sourcedValues(node: Record<string, unknown>): Array<{ source: unknown; value: unknown }> {
  const out: Array<{ source: unknown; value: unknown }> = [];
  const values = node.values;
  if (values && typeof values === 'object') {
    for (const [source, v] of Object.entries(values as Record<string, unknown>)) {
      out.push({ source, value: (v as { value?: unknown } | null)?.value });
    }
  }
  if (out.length === 0 && 'value' in node) out.push({ source: node.$source, value: node.value });
  return out;
}

/** The context a notification names only by its path key. */
function refFromKey(key: string): string {
  return key.startsWith('urn:') ? `vessels.${key}` : `targets.${key}`;
}
