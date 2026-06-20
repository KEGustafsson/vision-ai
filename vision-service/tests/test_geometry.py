import math

from app.geometry.bearing import relative_bearing_deg
from app.geometry.range import range_by_horizon, range_by_size, vfov_from_hfov


def test_bearing_center_is_zero():
    assert relative_bearing_deg(960, 1920, 90.0) == 0.0


def test_bearing_edges():
    # Right edge -> +half FOV (starboard); left edge -> -half FOV (port).
    assert relative_bearing_deg(1920, 1920, 90.0) == 45.0
    assert relative_bearing_deg(0, 1920, 90.0) == -45.0


def test_vfov_smaller_than_hfov_for_landscape():
    vfov = vfov_from_hfov(90.0, 1920, 1080)
    assert 0 < vfov < 90.0


def test_range_by_horizon_decreases_with_depression():
    # Lower in frame (larger object_y) => closer.
    near = range_by_horizon(700, 540, 2.5, 90.0, 1920, 1080)
    far = range_by_horizon(560, 540, 2.5, 90.0, 1920, 1080)
    assert near is not None and far is not None
    assert near[0] < far[0]


def test_range_by_horizon_none_above_horizon():
    assert range_by_horizon(500, 540, 2.5, 90.0, 1920, 1080) is None


def test_range_by_horizon_none_at_horizon_noise_band():
    # An object 1px below the horizon is within the noise band -> declined.
    assert range_by_horizon(541, 540, 2.5, 90.0, 1920, 1080) is None


def test_range_by_horizon_capped_for_unrealistic_height():
    # A near-horizon object with a very high mount yields a range beyond the cap.
    assert range_by_horizon(546, 540, 40.0, 90.0, 1920, 1080) is None


def test_range_by_size_scales_inversely_with_pixels():
    big = range_by_size(200, 4.0, 90.0, 1920)
    small = range_by_size(50, 4.0, 90.0, 1920)
    assert big is not None and small is not None
    assert big[0] < small[0]


def test_range_by_horizon_matches_formula():
    h, hy, oy = 2.5, 540, 700
    res = range_by_horizon(oy, hy, h, 90.0, 1920, 1080)
    vfov = vfov_from_hfov(90.0, 1920, 1080)
    ifov = vfov / 1080
    theta = math.radians((oy - hy) * ifov)
    assert abs(res[0] - h / math.tan(theta)) < 1e-6


def test_range_by_horizon_confidence_is_full_image_height_fraction():
    # Confidence = depression_frac * 1.5, where depression_frac is measured
    # against the FULL image height (horizon-position independent), so an object
    # 0.2 of the image height below the horizon scores exactly 0.3.
    res = range_by_horizon(756, 540, 2.5, 90.0, 1920, 1080)  # (756-540)/1080 = 0.2
    assert res is not None
    assert abs(res[1] - 0.3) < 1e-9


def test_range_by_horizon_confidence_independent_of_horizon_row():
    # The SAME object geometry (object 216 px below the horizon, same image) must
    # score the same confidence regardless of where the horizon sits in frame.
    a = range_by_horizon(756, 540, 2.5, 90.0, 1920, 1080)
    b = range_by_horizon(516, 300, 2.5, 90.0, 1920, 1080)
    assert a is not None and b is not None
    assert abs(a[1] - b[1]) < 1e-9
