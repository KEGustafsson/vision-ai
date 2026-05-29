# Monocular geometry & calibration

The container converts each detection's pixel box into a relative bearing and a
range. Both depend only on per-camera config (`config/*.yaml` → `cameras[]`).

## Relative bearing

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

For an object floating on the water, its waterline contact (bottom of the bbox)
sits below the horizon by a depression angle θ. With camera height `h` above the
waterline:

```text
VFOV  = 2 * atan(tan(HFOV/2) * H / W)        # vertical FOV (square pixels)
IFOV  = VFOV / H                              # degrees per pixel (vertical)
θ     = (object_y - horizon_y) * IFOV         # depression angle
range = h / tan(θ)
```

Confidence decreases as the object nears the horizon (θ → 0 is numerically
noisy). Implemented in `app/geometry/range.py`.

## Range by known size (fallback)

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
