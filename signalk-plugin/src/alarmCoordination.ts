// No double alarms: a boat must alarm once, whichever plugins see it.
//
// Two signals tell this plugin another one already alarms a boat:
//  1. Collision Alerts announces the `collisionAlerts` PropertyValue
//     `{ evaluatesTargets: true }` while it runs. It then evaluates every AIS
//     vessel and every `targets.*` sensor target, following `sameAs` links, so
//     a sensor plugin leaves those alarms to it (MaYaRa does the same).
//  2. A closest-approach alarm another producer already raised
//     (`notifications.navigation.closestApproach.*`, e.g. SK Derived Data)
//     names the same boat.
// Both are fail-safe: only an announcement or an alarm that is actually there
// makes this plugin stand down. A missing alarm is worse than a duplicate, so
// anything unreadable leaves this plugin's own alarm up.

import { ServerApp } from './skapp';

export const COLLISION_ALERTS_PROPERTY = 'collisionAlerts';

// Any plugin can emit under any name, so check the shape instead of casting.
export function evaluatesTargets(value: unknown): boolean {
  return (
    typeof value === 'object' &&
    value !== null &&
    (value as { evaluatesTargets?: unknown }).evaluatesTargets === true
  );
}

/** Whether Collision Alerts is running and evaluating sensor targets. */
export class CollisionAlertsWatch {
  private latest: unknown = undefined;
  private unsubscribe: (() => void) | undefined;

  /** The history replays on subscribe, so plugin start order does not matter. */
  start(app: Pick<ServerApp, 'onPropertyValues'>): void {
    this.stop();
    if (typeof app.onPropertyValues !== 'function') return;
    this.unsubscribe = app.onPropertyValues(COLLISION_ALERTS_PROPERTY, (history) => {
      this.latest = history[history.length - 1]?.value;
    });
  }

  /** Forgets the announcement so a restart never inherits a stale one. */
  stop(): void {
    this.unsubscribe?.();
    this.unsubscribe = undefined;
    this.latest = undefined;
  }

  running(): boolean {
    return evaluatesTargets(this.latest);
  }
}

const ACTIVE_STATES = new Set(['alert', 'warn', 'alarm', 'emergency']);

/**
 * The contexts other producers currently raise a closest-approach alarm on.
 * A notification names its boat in `data.targetRef` (the shared collision
 * alarm shape), in the older `other` field, or only by its path key, which
 * Collision Alerts and MaYaRa set to the target id (`urn:mrn:imo:mmsi:…` for
 * AIS, `radar:…` / `camera:…` for sensor targets).
 * @param ownSource the `$source` this plugin's notifications carry
 */
export function othersAlarmedRefs(app: ServerApp, ownSource: string): Set<string> {
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
    for (const entry of sourcedValues(node as Record<string, unknown>)) {
      if (entry.source === ownSource) continue;
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

// A leaf carries its latest value with its `$source`, and every source's own
// value under `values` when several write it.
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

function refFromKey(key: string): string {
  return key.startsWith('urn:') ? `vessels.${key}` : `targets.${key}`;
}
