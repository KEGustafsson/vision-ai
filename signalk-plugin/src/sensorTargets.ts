// Identity of a visual target in SignalK's sensor-targets convention
// (signalk-server docs: develop/plugins/sensor_targets.md). Each camera track is
// a `targets.camera:<id>` context; a fusion plugin (signalk-target-fusion) may
// set `sameAs` on it to name the AIS vessel or other target it is the same boat
// as. Chartplotters (Freeboard-SK) and collision alarms (Collision Alerts)
// follow those links, so a boat several sensors see is drawn and alarmed once.

import { randomBytes } from 'crypto';
import { ServerApp } from './skapp';
import { EnrichedTarget } from './types';

export const SENSOR_TYPE = 'camera';

/**
 * A camera name as an id segment. Names are free-form config strings and a
 * `.` would split the context, so every character outside [a-zA-Z0-9] is
 * written as its hex code between underscores (`bow cam` → `bow_20_cam`,
 * `bow.cam` → `bow_2e_cam`). The encoding is
 * one-to-one, so two cameras whose names differ only in punctuation never
 * share a target context; the usual names (`forward`, `aft`) are unchanged.
 */
export function cameraSegment(camera: string): string {
  return Array.from(String(camera), (ch) =>
    /[a-zA-Z0-9]/.test(ch) ? ch : `_${(ch.codePointAt(0) as number).toString(16)}_`
  ).join('');
}

/** The number the container burns into the video overlay for this target. */
export function overlayNumber(t: Pick<EnrichedTarget, 'stable_id' | 'track_id'>): number | null {
  return t.track_id ?? t.stable_id ?? null;
}

/** The chart label: the overlay number, so a ring and its video box match. */
export function displayName(camera: string, n: number): string {
  return `VIS-${cameraSegment(camera)}-${n}`;
}

/**
 * The target id after `targets.`. Keyed on the never-recycled `stable_id`:
 * the recycled 2-digit `track_id` would hand one context to a different boat
 * later in the session. `stable_id` restarts with the container, so the id
 * also carries the event-stream connection it was seen on (`session`): no id
 * is ever reused, and a consumer that reads the model on a timer (Target
 * Fusion) cannot carry an old boat's link over to a new one, even if it never
 * saw the old track withdrawn. The `visionai-` prefix keeps the id unique to
 * this plugin among other camera sensors. `track_id` is the fallback for
 * containers that predate `stable_id`.
 */
export function sensorTargetId(
  t: Pick<EnrichedTarget, 'camera' | 'stable_id' | 'track_id' | 'session'>
): string | null {
  const n = t.stable_id ?? t.track_id;
  if (n === null || n === undefined) return null;
  const session = t.session ? `${t.session}-` : '';
  return `${SENSOR_TYPE}:visionai-${cameraSegment(t.camera)}-${session}${n}`;
}

/**
 * Tokens for event-stream connections, never the same twice. Each is a time
 * in base 36 that only moves forward within this plugin run, whatever the
 * clock does (boats' computers are often RTC-less and stepped by GPS or
 * chrony after boot), plus a random suffix so two runs never share one either.
 * Letters and digits only, so a token sits in an id as is.
 */
export class SessionTokens {
  private lastMs = -Infinity;

  constructor(private readonly suffix: () => string = () => randomBytes(4).toString('hex')) {}

  next(nowMs: number): string {
    this.lastMs = Math.max(Math.floor(nowMs), this.lastMs + 1);
    return `${this.lastMs.toString(36)}${this.suffix()}`;
  }
}

export function sensorTargetContext(id: string): string {
  return `targets.${id}`;
}

/**
 * One source label per camera. Target Fusion never folds two targets from the
 * same sensor into one boat, and identifies the sensor by `$source`; a single
 * label would leave a boat seen at the edge of both the bow and the stern
 * camera as two rings.
 */
export function sourceLabel(camera: string): string {
  return `vision-ai-${cameraSegment(camera)}`;
}

export function aisContext(mmsi: string): string {
  return `vessels.urn:mrn:imo:mmsi:${mmsi}`;
}

/**
 * The `sameAs` link a fusion plugin set on one of our targets, or null. Read
 * from the full model; a bare value and a `{ value }` leaf are both accepted.
 */
export function readSameAs(app: ServerApp, context: string): string | null {
  let node: unknown;
  try {
    node = app.getPath(`${context}.sameAs`);
  } catch {
    return null;
  }
  const value =
    node !== null && typeof node === 'object' && 'value' in (node as object)
      ? (node as { value: unknown }).value
      : node;
  return typeof value === 'string' && value.length > 0 ? value : null;
}
