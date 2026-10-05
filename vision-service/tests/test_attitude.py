"""IMU horizon compensation: the closed-form horizon line against an independent
ray projection, the sign conventions, the fail-safe fallbacks, and the API."""

import math
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.attitude import AttitudeStore
from app.config import CameraConfig, GeometryConfig, load_settings
from app.detector.base import RawTrack
from app.geometry import estimate_range, horizon_line, person_in_water, resolve_horizon
from app.geometry.range import focal_px
from app.main import create_app

W, H, HFOV = 1280, 960, 90.0


def _project_horizon(pitch, roll, bearing_deg, tilt, xs):
    """Independent reference: rotate world-horizontal rays into the camera with
    full rotation matrices and project them. Returns {x: y} of the horizon at
    (approximately) the requested columns, by sampling azimuths densely."""
    f = focal_px(HFOV, W)
    cx, cy = W / 2.0, H / 2.0
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    r_nb = ry @ rx  # yaw irrelevant to the horizon
    b = math.radians(bearing_deg)
    z_c = np.array([math.cos(tilt) * math.cos(b), math.cos(tilt) * math.sin(b), math.sin(tilt)])
    x_c = np.array([-math.sin(b), math.cos(b), 0.0])
    y_c = np.cross(z_c, x_c)
    pts = []
    for a in np.linspace(b - 1.2, b + 1.2, 4001):
        d_b = r_nb.T @ np.array([math.cos(a), math.sin(a), 0.0])  # level ray, NED -> boat
        z = d_b @ z_c
        if z <= 1e-6:
            continue
        pts.append((cx + f * (d_b @ x_c) / z, cy + f * (d_b @ y_c) / z))
    pts = np.array(pts)
    out = {}
    for x in xs:
        i = int(np.argmin(np.abs(pts[:, 0] - x)))
        out[x] = (pts[i, 0], pts[i, 1])
    return out


@pytest.mark.parametrize("bearing", [0.0, 180.0, 90.0, -90.0, 35.0])
@pytest.mark.parametrize(
    "pitch_deg,roll_deg", [(4.0, 0.0), (0.0, 18.0), (-3.0, -12.0), (6.0, 25.0)]
)
def test_horizon_line_matches_ray_projection(bearing, pitch_deg, roll_deg):
    tilt = math.radians(4.0)  # camera looks 4 deg down
    ref_p, ref_r = math.radians(1.0), math.radians(-2.0)  # calibrated off-level
    # Calibrated row = where the reference attitude puts the horizon at centre.
    ref_pts = _project_horizon(ref_p, ref_r, bearing, tilt, [W / 2.0])
    _, base_y = ref_pts[W / 2.0]
    p, r = math.radians(pitch_deg), math.radians(roll_deg)
    line = horizon_line(base_y, W, H, HFOV, bearing, p, r, ref_p, ref_r)
    assert line is not None
    for x, (px, py) in _project_horizon(p, r, bearing, tilt, [200.0, W / 2.0, 1100.0]).items():
        assert line.y_at(px) == pytest.approx(py, abs=0.6)


def test_reference_attitude_reproduces_calibrated_row():
    line = horizon_line(400.0, W, H, HFOV, 0.0, 0.05, -0.1, 0.05, -0.1)
    assert line.y_center == pytest.approx(400.0)


def test_sign_conventions():
    f = focal_px(HFOV, W)
    up = math.radians(3.0)
    # Bow up: forward camera tilts up -> horizon moves DOWN the image...
    fwd = horizon_line(H / 2, W, H, HFOV, 0.0, up, 0.0)
    assert fwd.y_center == pytest.approx(H / 2 + f * math.tan(up))
    assert fwd.slope == pytest.approx(0.0, abs=1e-12)
    # ...and the aft camera tilts down -> horizon moves UP.
    aft = horizon_line(H / 2, W, H, HFOV, 180.0, up, 0.0)
    assert aft.y_center == pytest.approx(H / 2 - f * math.tan(up))
    # List to starboard: forward camera's right side drops, so the horizon rises
    # on the right (negative slope); a starboard-facing camera tilts down.
    heel = math.radians(15.0)
    fwd = horizon_line(H / 2, W, H, HFOV, 0.0, 0.0, heel)
    assert fwd.slope == pytest.approx(-math.tan(heel))
    stbd = horizon_line(H / 2, W, H, HFOV, 90.0, 0.0, heel)
    assert stbd.y_center < H / 2


def test_degenerate_geometry_returns_none():
    assert horizon_line(H / 2, W, H, HFOV, 0.0, math.radians(85), 0.0) is None
    assert horizon_line(H / 2, 0, H, HFOV, 0.0, 0.0, 0.0) is None


def _cam(**kw):
    return CameraConfig(name="forward", hfov_deg=HFOV, height_m=2.5, **kw)


def test_resolve_horizon_without_attitude_is_the_plain_row():
    line, comp = resolve_horizon(_cam(horizon_y=300.0), 300.0, W, H, None)
    assert (line.y_center, line.slope, comp) == (300.0, 0.0, False)
    assert resolve_horizon(_cam(), None, W, H, (0.1, 0.1)) == (None, False)


def test_resolve_horizon_calibrated_uses_reference():
    cam = _cam(horizon_y=300.0, horizon_ref_pitch_deg=2.0, horizon_ref_roll_deg=0.0)
    # At the reference attitude nothing moves.
    line, comp = resolve_horizon(cam, 300.0, W, H, (math.radians(2.0), 0.0))
    assert comp and line.y_center == pytest.approx(300.0)
    # Squat 3 deg further bow up -> horizon lower.
    line, comp = resolve_horizon(cam, 300.0, W, H, (math.radians(5.0), 0.0))
    assert comp and line.y_center > 330.0


def test_resolve_horizon_auto_adds_only_heel_slope():
    line, comp = resolve_horizon(_cam(), 300.0, W, H, (math.radians(5.0), math.radians(10.0)))
    assert comp
    assert line.y_center == pytest.approx(300.0)
    assert line.slope < 0


def test_resolve_horizon_degenerate_falls_back_to_level():
    # Aft camera already looking down; 89 deg bow-up points it at the water.
    cam = CameraConfig(name="aft", hfov_deg=HFOV, horizon_y=300.0, bearing_offset_deg=180.0)
    line, comp = resolve_horizon(cam, 300.0, W, H, (math.radians(89), 0.0))
    assert (line.y_center, line.slope, comp) == (300.0, 0.0, False)


def _track(x, y, w, h, label="vessel"):
    return RawTrack(track_id=1, cls=8, label=label, confidence=0.9, x=x, y=y, w=w, h=h)


def test_estimate_range_uses_horizon_at_target_column():
    geo = GeometryConfig(known_widths_m={})
    cam = _cam(horizon_y=300.0)
    # Heeled: horizon at the right edge sits 100 px higher than at centre.
    slope = -100.0 / (W / 2)
    right = _track(1200, 230, 40, 20)  # waterline 250, horizon there ~206
    level_rng, _, _ = estimate_range(right, cam, geo, W, H, 300.0)
    tilt_rng, method, _ = estimate_range(right, cam, geo, W, H, 300.0, slope)
    assert level_rng is None  # above the level row: no range at all
    assert method == "horizon" and tilt_rng is not None


def test_estimate_range_measures_depression_perpendicular_to_tilted_horizon():
    geo = GeometryConfig(known_widths_m={})
    cam = _cam()
    slope = math.tan(math.radians(20.0))
    t = _track(W / 2 - 20, 380, 40, 20)  # 100 px below the centre row
    tilted, _, _ = estimate_range(t, cam, geo, W, H, 300.0, slope)
    level, _, _ = estimate_range(t, cam, geo, W, H, 300.0)
    # Perpendicular depression is 100*cos(20) px -> slightly farther.
    assert tilted > level
    nudged = _track(W / 2 - 20, 300 + 100 * math.cos(math.radians(20.0)) - 20, 40, 20)
    assert tilted == pytest.approx(estimate_range(nudged, cam, geo, W, H, 300.0)[0])


def test_person_in_water_fails_toward_alerting():
    level = 300.0
    line, _ = resolve_horizon(_cam(), level, W, H, (0.0, math.radians(15.0)))
    # Right side: tilted horizon is higher than the level row. A waterline
    # between them is in-water by the compensated line.
    right = _track(1150, 250, 30, 30, label="person")  # waterline 280
    assert line.y_at(right.cx) < 280 < level
    assert person_in_water(right, line, level)
    # Left side: tilted horizon is LOWER than the level row; a waterline between
    # them stays in-water because the uncompensated row still says so.
    left = _track(100, 280, 30, 30, label="person")  # waterline 310
    assert level < 310 < line.y_at(left.cx)
    assert person_in_water(left, line, level)
    # Clearly above both: not in the water. Not a person: never.
    assert not person_in_water(_track(600, 100, 30, 30, label="person"), line, level)
    assert not person_in_water(_track(600, 500, 30, 30), line, level)


def test_attitude_store_ages_out():
    s = AttitudeStore()
    assert s.get(2.0) is None
    s.set(0.1, -0.2, now=100.0)
    assert s.get(2.0, now=101.0) == (0.1, -0.2)
    assert s.get(2.0, now=102.5) is None
    assert s.get(0.0, now=100.0) is None  # disabled
    assert s.status(now=103.0) == pytest.approx((0.1, -0.2, 3.0))


def test_attitude_endpoint_compensates_events():
    settings = load_settings("mock")
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/health").json()["attitude"] is None
        # Out of range (degrees sent as radians) is refused.
        assert client.post("/attitude", json={"pitch_rad": 5.0, "roll_rad": 0.0}).status_code == 422
        deadline = time.time() + 5
        ev = None
        while time.time() < deadline:
            assert (
                client.post(
                    "/attitude",
                    json={"pitch_rad": math.radians(3.0), "roll_rad": math.radians(10.0)},
                ).status_code
                == 200
            )
            events = [e for e in client.get("/events/recent").json() if e["attitude_compensated"]]
            if events:
                ev = events[-1]
                break
            time.sleep(0.1)
        assert ev is not None, "no compensated event produced"
        # List to starboard: the forward camera's horizon rises on its right
        # (negative slope); the aft camera looks backwards, so it tilts the other way.
        cam = settings.camera(ev["camera"])
        assert ev["horizon_slope"] * math.cos(math.radians(cam.bearing_offset_deg)) < 0
        assert ev["horizon_y"] != pytest.approx(cam.horizon_y)
        att = client.get("/health").json()["attitude"]
        assert att["active"] is True
        assert att["pitch_deg"] == pytest.approx(3.0)
        assert att["roll_deg"] == pytest.approx(10.0)


# --- No IMU: behaviour must be exactly what it was before compensation ------


@pytest.mark.parametrize("label", ["vessel", "person"])
@pytest.mark.parametrize("y", [250.0, 299.0, 320.0, 600.0])
def test_without_attitude_range_and_mob_are_unchanged(label, y):
    from app.detector.classmap import is_person_in_water
    from app.geometry.range import range_by_horizon, range_by_size

    cam = _cam(horizon_y=300.0)
    geo = GeometryConfig()
    t = _track(900, y, 40, 20, label=label)
    line, comp = resolve_horizon(cam, 300.0, W, H, None)
    assert not comp
    got = estimate_range(t, cam, geo, W, H, line.y_center, line.slope)
    # The pre-compensation algorithm, inlined.
    wl = t.y + t.h
    res = range_by_horizon(wl, 300.0, cam.height_m, cam.hfov_deg, W, H)
    if res is not None:
        want = (res[0], "horizon", res[1])
    else:
        ks = range_by_size(t.w, geo.known_widths_m[label], cam.hfov_deg, W)
        want = (ks[0], "known_size", ks[1])
    assert got == want
    assert person_in_water(t, line, 300.0) == is_person_in_water(label, wl, 300.0)


def test_mock_events_without_attitude_are_uncompensated():
    settings = load_settings("mock")
    app = create_app(settings)
    with TestClient(app) as client:
        deadline = time.time() + 5
        events = []
        while time.time() < deadline and not events:
            events = client.get("/events/recent").json()
            time.sleep(0.2)
        assert events
        for ev in events:
            assert ev["attitude_compensated"] is False
            assert ev["horizon_slope"] == 0.0
            assert ev["horizon_y"] == settings.camera(ev["camera"]).horizon_y
        assert client.get("/health").json()["attitude"] is None
