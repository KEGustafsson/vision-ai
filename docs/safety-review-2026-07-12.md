# Safety review findings — 2026-07-12

A wide review of the vision service, the SignalK plugin, the shared
event contract, and the CI/test setup. No code was changed as part of
this review; each finding lists the affected code and a recommended fix.
Findings are ordered by safety impact.

## 1. Own-ship nav accepts future-dated timestamps as fresh

**Where:** `signalk-plugin/src/nav.ts:36-49` (`readPath`)

`readPath()` drops values whose timestamp is unparseable or older than
`maxAgeMs`, but it does not reject `ageMs < 0`. A future-dated
position/heading/SOG/COG (clock skew on a sensor or gateway, a bad NTP
step on the server) is treated as fresh and stays trusted until wall
time catches up — potentially for a long time if the skew is large.
This contradicts the fail-closed stale-data intent documented at the
top of the same file: a frozen future-dated fix would georeference
targets against a stale own-ship position without setting `stale`.

**Fix:** treat any timestamp outside `[-smallSkewAllowance, maxAgeMs]`
as stale (a small negative tolerance, e.g. a few seconds, absorbs benign
clock jitter). Add `nav.test.ts` covering future-dated, too-old,
unparseable, and delta-only (no timestamp) shapes.

## 2. Own-ship position allows non-finite coordinates

**Where:** `signalk-plugin/src/nav.ts:20-25` (`pos`)

`pos()` only checks `typeof v.latitude === 'number'`, so `NaN`,
`Infinity`, and `-Infinity` pass through into georeferencing and CPA.
The sibling helper `num()` (line 16) correctly requires `isFinite()`.
NaN coordinates propagate silently through trig — downstream range or
bearing math yields NaN rather than throwing, so a poisoned fix can
suppress alerts without any visible error.

**Fix:** require `Number.isFinite` on both latitude and longitude in
`pos()`; add tests for NaN/±Infinity inputs.

## 3. `undistort_before_detect` mixes coordinate spaces for the configured horizon

**Where:** `vision-service/app/pipeline.py:198, 251-258, 264-268`

When `undistort_before_detect` is enabled the frame is undistorted
*before* detection, so detections are in corrected image coordinates.
But `_resolve_horizon()` returns the configured `camera.horizon_y`
unchanged — a row calibrated against the *distorted* image. The
display-only path (lines 251-258) demonstrates that `horizon_y` needs
remapping (`u.horizon_y(...)`) when crossing coordinate spaces, yet the
detect-time path never applies it. Result: range estimation and
person-in-water classification compare corrected-space bbox rows
against a distorted-space horizon row. Near the horizon (exactly where
range sensitivity is greatest) even a few rows of lens-correction shift
produce large range errors.

**Fix:** define the horizon's coordinate space explicitly. When
`undistort_before_detect` is set, remap the configured `horizon_y`
through the undistorter once (or require the calibration to be done on
corrected frames and document that). Add a test that runs the pipeline
both ways and asserts the horizon row used for range matches the space
the detections are in.

## 4. Auto-horizon is a weak heuristic but is treated as calibrated geometry

**Where:** `vision-service/app/geometry/horizon.py:16-28`
(`detect_horizon_y`), consumed at `vision-service/app/pipeline.py:264-275`

`detect_horizon_y()` picks the single strongest row-mean intensity
gradient and rejects only near-edge rows. In marina/shoreline scenes, a
cloud bank, a deck edge, or a wake crossing the frame, the strongest
gradient is often not the horizon — but the result still drives
`CalibrationStatus.auto` and horizon-based range estimates, producing
plausible-looking but wrong ranges with no quality signal.

**Fix:** add quality gating — e.g. require the winning gradient to
exceed the runner-up by a margin, require temporal stability across
frames before trusting it, or clamp it to a plausible band around the
configured mounting geometry. When gating fails, publish no horizon
(range falls back to bearing-only behavior) rather than a wrong one.

## 5. Event-schema docs narrow `camera` incorrectly

**Where:** `docs/event-schema.md:22` (`"camera": "forward", // "forward" | "aft"`)

The generated contract and both codebases treat `camera` as a free-form
configured camera name; only the docs claim it is an enum of
`"forward" | "aft"`. An integrator coding to the docs will break on any
install with differently named or additional cameras. (Lines 61-74 of
the same doc also use forward/aft as if exhaustive.)

**Fix:** document `camera` as the configured camera name, with
forward/aft as the conventional defaults.

## 6. Schema defaults vs. TypeScript required fields

**Where:** `signalk-plugin/schema/detection-event.schema.json`
(generated from `vision-service/app/schemas.py`) vs.
`signalk-plugin/src/types.ts`, validated in
`signalk-plugin/src/eventStream.ts:71-73`

The generated schema carries `default` on a number of fields that are
*not* in the corresponding `required` arrays:

- root: `schema_version`, `horizon_y`, `calibration_status`
- `$defs.Geometry`: `range_m`, `range_method`, `range_confidence`
- `$defs.Inference`: `latency_ms`
- `$defs.PixelVelocity`: `vx`, `vy`
- `$defs.Target`: `track_id`, `stable_id`, `is_person_in_water`,
  `first_seen`, `age_frames`, `coasting`

But the TypeScript mirror declares several of these non-optional —
`DetectionEvent.schema_version` and `.calibration_status`,
`Geometry.range_confidence`, `PixelVelocity.vx`/`vy`,
`RawTarget.is_person_in_water` and `.age_frames`
(`signalk-plugin/src/types.ts:14-70`) — and the AJV instance in
`eventStream.ts:71` (`new Ajv({ allErrors: false, strict: false })`)
does not set `useDefaults`, so validation passes an event that omits
them and downstream reads `undefined` where the types promise a value
(e.g. `is_person_in_water` in MOB evaluation). Resolve one way or the
other: make defaulted fields required in the Pydantic source and
regenerate the schema, or enable AJV `useDefaults: true` so validated
events are normalized.

## 7. Retained tracks flow into dark-target and collision alerting

**Where:** `signalk-plugin/src/notifications.ts:191-211`
(`evaluateDark`, `evaluateCollision`); retention in
`signalk-plugin/src/index.ts:177-186`

Targets are retained in the map for `trackTimeoutS` after their last
detection (`index.ts:179-184`) so chart blips don't flicker. The MOB
path explicitly guards against alerting on retained-but-stale tracks —
it only counts a candidate when `lastSeen` has advanced
(`notifications.ts:126-133`) and decays counters for tracks that stop
qualifying (`notifications.ts:154-165`). But `evaluateDark`
(`notifications.ts:191-199`) and `evaluateCollision`
(`notifications.ts:201-211`) iterate all targets with no `lastSeen`
freshness check, so a vessel that disappeared keeps raising
dark-target/collision notifications until the retention timeout prunes
it — extended further by the anti-flap hold (`applyHold`,
`notifications.ts:103-115`). The comment at `index.ts:177`
("MOB/notification logic already guards against retained-but-stale")
overstates the guard: only MOB has one. Add a freshness gate (or a
staleness cutoff shorter than `trackTimeoutS`) to the dark/collision
paths and test it explicitly.

## 8. Test and CI gaps

- **CI lint gap:** `.github/workflows/ci.yml` — the
  `vision-service (Python)` job (lines 13-40) runs `pytest` and the
  schema diff-check but not `ruff check` or `black --check`, even
  though `vision-service/pyproject.toml` configures both `[tool.ruff]`
  and `[tool.black]`. (The TypeScript job does run its linter,
  `ci.yml:58-59`.)
- **Notification coverage:** `signalk-plugin/test/notifications.test.ts`
  covers MOB well; the `evaluateDark` and `evaluateCollision` paths
  (finding 7) have no direct tests.
- **CPA coverage:** `signalk-plugin/test/cpa.test.ts` covers
  head-on/threshold basics; add crossing, diverging, already-passed,
  zero-relative-velocity, and non-finite-input cases.
- **Missing test file:** there is no `signalk-plugin/test/nav.test.ts`
  covering `src/nav.ts` freshness handling (findings 1-2).

## Verification notes

- `signalk-plugin`: `npm test` passes (8 files, 70 tests) in the review
  environment.
- `vision-service`: pytest could not be run in the review environment
  (Python 3.14 there vs. CI's 3.11 constrained install); Python findings
  were verified by reading the code, not by execution.
- Docker compose configs could not be validated (no Docker in the
  review environment).
