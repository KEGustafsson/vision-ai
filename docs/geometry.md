# Monocular geometry & calibration

The container converts each detection's pixel box into a relative bearing and a
range. Both depend only on per-camera config (`config/*.yaml` → `cameras[]`).

## Where geometry sits in the pipeline

A single camera is treated as a **bearing/range sensor**. The detector gives a
tracked pixel box; geometry turns that box into two numbers — a **relative
bearing** (from the box's horizontal position) and a **range** (from the box's
vertical position relative to the horizon). Those two numbers are everything the
plugin needs to place the object on the chart once it adds the boat's own heading
and position.

![Full process from pixels to georeferenced targets: the container decodes a camera frame, detects and tracks with a YOLO detector and tracker to get a bounding box and track id, derives a relative bearing from the pixel column and a range from the horizon depression or known size with a confidence, applies operator filters, and emits a DetectionEvent. The plugin enriches it to a true bearing and lat/lon, fuses with AIS, computes CPA/TCPA, raises MOB / collision / dark-target notifications, and publishes synthetic AIS vessels and vision.* paths.](images/detection-process.svg)

The two measurements below — bearing and range — are the geometry container's
entire job. Everything downstream is navigation math done by the plugin.

## Relative bearing

A detection's **horizontal** pixel position maps linearly to an angle off the
camera's optical axis. Looking straight down from above the boat, the lens spreads
its horizontal field of view (`HFOV`) symmetrically about the optical axis; a
detection centred at pixel column `px` therefore sits at a fraction
`2·px/W − 1` of the half-FOV, left (port) or right (starboard) of centre.

![Top-down view of relative bearing: the camera at the bottom looks up its optical axis with a symmetric HFOV cone; the image sensor of width W spans the cone; a detected object at pixel column px lies along a ray at angle theta off the optical axis, negative to port (left) and positive to starboard (right). relative_bearing = (HFOV/2) times (2·px/W minus 1), plus the camera mount offset.](images/geometry-bearing.svg)

For a rectilinear lens of horizontal field of view `HFOV` over image width `W`,
a detection centred at pixel column `px` has bearing relative to the optical
axis:

```text
relative_bearing_deg = (HFOV / 2) * (2 * px / W - 1)
```

Positive = starboard (right of centre), negative = port. The camera's mounting
offset (`bearing_offset_deg`: forward = 0, aft = 180) is added so the value is
relative to the bow. The plugin then adds own `navigation.headingTrue` to get
true bearing. Only the true heading is used: never magnetic heading plus
variation (a compass's installation deviation is unknown to SignalK — on Arabella
it measured 33°) and never COG. With no true heading there is no true bearing.

Implemented in `app/geometry/bearing.py`; verified in `tests/test_geometry.py`.

## Range by horizon depression (preferred)

This is how the system measures **distance**. For an object floating on the
water, its waterline contact (the bottom of the bounding box) sits **below the
horizon** by a small depression angle θ. The farther away the object is, the
closer its waterline creeps to the horizon, so θ shrinks with distance — and from
θ and the known camera height the range follows by simple trigonometry.

In the image this is purely a **row** measurement: the horizon is a known pixel
row (`horizon_y`, calibrated or auto-detected) and the object's waterline is the
bottom row of its box (`object_y`). The gap between them in pixels, scaled by the
vertical degrees-per-pixel (`IFOV`), is the depression angle θ. With camera height
`h` above the waterline, the object is `h / tan(θ)` metres away.

![Side view of range by horizon depression: a camera mounted at height h above the waterline looks out to sea; the horizon is a horizontal eye-level reference line (image row horizon_y); a detected vessel on the water surface lies along a ray that drops below the horizon by the depression angle theta, where the vessel's waterline is image row object_y. The pixel gap between horizon_y and object_y times IFOV gives theta, and range equals h divided by tan(theta). VFOV = 2·atan(tan(HFOV/2)·H/W), IFOV = VFOV/H. Confidence falls as theta approaches zero near the horizon.](images/geometry-range-horizon.svg)

With camera height `h` above the waterline:

```text
VFOV  = 2 * atan(tan(HFOV/2) * H / W)        # vertical FOV (square pixels)
IFOV  = VFOV / H                              # degrees per pixel (vertical)
θ     = (object_y - horizon_y) * IFOV         # depression angle
range = h / tan(θ)
```

Confidence decreases as the object nears the horizon (θ → 0 is numerically
noisy). Implemented in `app/geometry/range.py`.

## Range by known size (fallback)

When the horizon is unavailable (uncalibrated, or hidden by haze/land), distance
falls back to a pinhole projection: a target of known real-world width that
appears `pixel_width` pixels wide must be at the range where the camera's focal
length projects it to that size. It needs no horizon, only the object's assumed
width (`geometry.known_widths_m`), so it works anywhere — but a wrong width
assumption scales the range directly, which is why it reports a low fixed
confidence.

![Range by known size, a pinhole projection seen from above: rays from the camera through the top and bottom of the object converge at the lens; the object of known real width (metres) projects onto the image plane at distance focal_px as a span of pixel_width. focal_px = (W/2)/tan(HFOV/2), and range = focal_px times real_width_m divided by pixel_width, reported at a low fixed confidence.](images/geometry-range-knownsize.svg)

If the horizon is unavailable but the object's real-world width is known
(`geometry.known_widths_m`):

```text
focal_px = (W / 2) / tan(HFOV / 2)
range    = focal_px * real_width_m / pixel_width
```

Coarse (no precise intrinsics), so it reports a low fixed confidence.

## Calibration procedure (on installation)

1. **HFOV** — from the camera/lens datasheet, or measure: place two markers a
   known distance apart at a known range and solve.
2. **Height** — measure the lens height above the waterline at design trim.
3. **Horizon row** — with the boat level and a clear horizon, read the pixel row
   of the horizon **at the frame's centre column** and set `horizon_y`. Set
   `geometry.auto_horizon: true` to let `app/geometry/horizon.py` estimate it
   from the sky/sea intensity edge instead.
   If the boat has an attitude sensor, note its pitch/roll at the same moment
   (`GET /health` → `attitude`, once the plugin forwards it) and set the
   camera's `horizon_ref_pitch_deg` / `horizon_ref_roll_deg` — see below.
4. **Bearing offset** — forward camera 0°, aft 180°; adjust for any yaw in the
   mount.

## Accuracy caveats

Monocular range is **coarse**, and on a pitching boat the horizon row moves, so
range jitters frame-to-frame. Mitigations: gate georeferencing on
`range_confidence` (`minRangeConfidence`), the plugin smooths motion through
track continuity, and CPA/TCPA require a velocity baseline before they trigger.
Slow attitude changes — trim, squat, heel — are corrected by IMU horizon
compensation, below.

## IMU horizon compensation

A calibrated `horizon_y` is right only at the attitude it was measured at. The
cameras are fixed to the hull, so when the hull changes attitude the horizon
moves in the image:

| Situation | Typical attitude change | Horizon error, 1280×960 / 90° camera |
|-----------|-------------------------|---------------------------------------|
| Moored vs underway (squat, planing trim) | 2–5° pitch | 22–56 px, whole frame |
| Sailing heel | 10–25° roll | slope of 0.18–0.47 rows/column — up to ±110–300 px at the frame edges |
| Loading, tanks, crew position | 0.5–2° | 6–22 px |

A vessel 1 km away sits only ~2 px below the horizon from a 2.5 m mount, so any
of these makes horizon-depression range meaningless — or puts the waterline
*above* the stale row and the target gets no range at all.

**Where the attitude comes from.** SignalK carries it as
`navigation.attitude`, an object of radians — `roll` (+ve = list to starboard),
`pitch` (+ve = bow up), `yaw`. NMEA 2000 attitude/heading sensors (PGN 127257)
are mapped there by signalk-server's `n2k-signalk`; a field the sensor doesn't
send arrives as `null`. NMEA 0183 has no standard attitude sentence (XDR
pitch/roll is not mapped by the stock parser), so a 0183-only IMU needs a
plugin that publishes `navigation.attitude`.

**Who does what.** The boundary holds: the plugin owns SignalK state, the
container owns pixels and geometry.

1. The plugin (`enableAttitudeCompensation`) reads `navigation.attitude`,
   rejects it when stale (`attitudeMaxAgeS`), incomplete or beyond ±90°, and
   low-pass filters it (`attitudeSmoothingS`) — `src/nav.ts`, `src/attitude.ts`.
2. It pushes `{pitch_rad, roll_rad}` to `POST /attitude` (~5 Hz).
3. Each frame, the container turns the attitude *change* from the camera's
   calibration reference into a horizon **line** for that camera
   (`app/geometry/attitude.py`): its row at the centre column and its slope.
4. Range and the person-in-water rule are evaluated against that line **at the
   target's own column**; the depression is measured perpendicular to a tilted
   horizon. The event reports the line (`horizon_y`, `horizon_slope`) and
   `attitude_compensated`; the overlay draws it tilted.

**The math.** World "up" is rotated into the boat frame by the attitude, then
into a camera looking `bearing_offset_deg` off the bow and tilted down by its
mount angle; the horizon is where a pixel ray is perpendicular to "up" — a
straight line for a pinhole camera. The mount angle is solved from `horizon_y`
at the reference attitude, so no extra calibration is needed beyond recording
that attitude. Camera azimuth matters: bow-up pitch lowers the forward
camera's horizon and raises the aft camera's; a starboard list tilts the forward
camera's horizon (higher on the right) and raises a starboard camera's.
Verified against an independent ray projection in `tests/test_attitude.py`.

**Reference attitude.** `horizon_ref_pitch_deg` / `horizon_ref_roll_deg`
(per camera, default 0) are the IMU's reading when `horizon_y` was measured.
Only the change from it is applied, which also cancels any misalignment
between the IMU and the hull. An auto-detected horizon (`auto_horizon`) already
follows pitch, so for it only the heel slope is added.

**Smoothing — why the default ignores waves.** The attitude reaches a frame
late: SignalK, the HTTP push and the camera's own RTSP/decode latency add up to
a few hundred milliseconds. Wave-induced roll and pitch have periods of a few
seconds, and a correction that far out of phase can be *larger* than the motion
it cancels. The default 5 s low-pass keeps the slow component — exactly the
moored/underway/heel changes above — and leaves wave motion as uncompensated as
it was before (within a few percent). `attitudeSmoothingS: 0` sends the raw
attitude for an install with a fast, high-rate sensor and low camera latency;
check it on the overlay in a seaway before relying on it.

**Fail-safe behaviour.**

- **No IMU, or the feature off: nothing changes.** Range, the person-in-water
  rule, the event's `horizon_y`, the overlay and the published SignalK paths
  are exactly what they were before this feature existed (`horizon_slope` is
  `0`, `attitude_compensated` is `false`). Locked in by
  `tests/test_attitude.py`.
- Off by default. A sensor mounted backwards, or reporting the opposite sign
  convention, *doubles* the error instead of removing it, so enable it only
  after the check below.
- Missing, stale (plugin `attitudeMaxAgeS`, container
  `geometry.attitude_max_age_s` — 2 s each), out-of-range or geometrically
  degenerate attitude → the container uses the plain calibrated row, exactly as
  without the feature; `attitude_compensated` reads `false`.
- **Man-overboard is never suppressed by compensation:** a person counts as in
  the water if their waterline is below *either* the compensated line or the
  plain calibrated row, so compensation can add MOB candidates (where heel
  lowered the horizon) but never removes one the uncompensated rule would raise.

**On-board check before enabling.** With the video overlay open: heel the boat
(crew to one side, or underway on a tack) and confirm the grey horizon line
tilts *with* the real horizon; compare moored vs underway trim and confirm the
line stays on the horizon instead of drifting off it. If it tilts the wrong
way, the sensor's sign convention or mounting is wrong — fix that in the
sensor/SignalK, not here. `vision.system.attitudeCompensated` shows whether
compensation is live.

Not covered: lens distortion of the *change* (the calibrated row absorbs it
at the reference attitude) and camera roll about the optical axis from the mount
itself (level the mount, or use `undistort_rotation_deg` for the display).
