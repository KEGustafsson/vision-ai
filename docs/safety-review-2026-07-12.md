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

**Where:** generated JSON schema vs. `signalk-plugin/src/types.ts`

The TypeScript mirror of the event contract assumes several
schema-defaulted fields are always present, but AJV validation as
configured does not insert defaults — so a producer that legitimately
omits a defaulted field yields `undefined` where the types promise a
value. Resolve one way or the other: make those fields required in the
Pydantic source and regenerate, or normalize defaults on the plugin
side immediately after validation (e.g. AJV `useDefaults`).

## 7. Retained (coasting) tracks flow into alerting

**Where:** SignalK plugin fusion/notification path

Visual-radar tracks retained for chart continuity appear to continue
through AIS fusion, CPA, dark-target, and collision notifications while
stale. Holding a display blip is fine; alerting on it is not clearly
intended. Alert computation should distinguish fresh detections from
held tracks, and that distinction should be tested explicitly.

## 8. Test and CI gaps

- CI does not run `ruff` or `black --check` for the Python service even
  though the repo is configured for both; Python CI covers install,
  pytest, and schema diff-check only.
- Notification tests cover MOB well; dark-target and collision alert
  paths have no direct coverage.
- CPA tests cover head-on/threshold basics; add crossing, diverging,
  already-passed, zero-relative-velocity, and non-finite-input cases.
- No tests exist for `nav.ts` freshness handling (see findings 1-2).

## Verification notes

- `signalk-plugin`: `npm test` passes (8 files, 70 tests) in the review
  environment.
- `vision-service`: pytest could not be run in the review environment
  (Python 3.14 there vs. CI's 3.11 constrained install); Python findings
  were verified by reading the code, not by execution.
- Docker compose configs could not be validated (no Docker in the
  review environment).
