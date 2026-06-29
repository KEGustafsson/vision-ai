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

![Full process from pixels to georeferenced targets: the container decodes a camera frame, detects and tracks with YOLOv8 + ByteTrack to get a bounding box and track id, derives a relative bearing from the pixel column and a range from the horizon depression or known size with a confidence, applies operator filters, and emits a DetectionEvent. The plugin enriches it to a true bearing and lat/lon, fuses with AIS, computes CPA/TCPA, raises MOB / collision / dark-target notifications, and publishes synthetic AIS vessels and vision.* paths.](images/detection-process.svg)

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
relative to the bow. The plugin then adds own `headingTrue` to get true bearing.

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
   of the horizon and set `horizon_y`. Set `geometry.auto_horizon: true` to let
   `app/geometry/horizon.py` estimate it from the sky/sea intensity edge instead.
4. **Bearing offset** — forward camera 0°, aft 180°; adjust for any yaw in the
   mount.

## Accuracy caveats

Monocular range is **coarse**, and on a pitching boat the horizon row moves, so
range jitters frame-to-frame. Mitigations: gate georeferencing on
`range_confidence` (`minRangeConfidence`), the plugin smooths motion through
track continuity, and CPA/TCPA require a velocity baseline before they trigger.
For better range, feed the camera pitch from an IMU and offset `horizon_y`
per frame — a natural future extension.
