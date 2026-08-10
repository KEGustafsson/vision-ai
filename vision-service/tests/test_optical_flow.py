"""Global image motion from NVIDIA OFA flow vectors.

Everything here is hardware-independent: the DeepStream probe hands the module
raw S10.5 vectors, so the fixed-point conversion, the robust (median) estimate,
the input hardening and the per-camera diagnostic state machine are all testable
without a Jetson.
"""

import math

import numpy as np
import pytest

from app.motion import (
    OF_FIXED_POINT_DIVISOR,
    CameraFlowState,
    OpticalFlowState,
    estimate_global_motion,
    raw_to_px,
)

# nvof reports S10.5 fixed point, so a raw component of 32 is one pixel.
PX = OF_FIXED_POINT_DIVISOR


# ── Fixed-point conversion ────────────────────────────────────────────────────


@pytest.mark.parametrize("raw, px", [(32, 1.0), (16, 0.5), (-32, -1.0), (0, 0.0), (8, 0.25)])
def test_s10_5_fixed_point_conversion(raw, px):
    assert raw_to_px(raw) == px


def test_estimate_applies_fixed_point_scaling():
    # A uniform field of raw (32, -16) is exactly (1.0, -0.5) px of motion.
    stats = estimate_global_motion([(32, -16)] * 10)
    assert stats.global_dx == 1.0
    assert stats.global_dy == -0.5
    assert stats.vector_count == 10


# ── Robust (median) global motion ─────────────────────────────────────────────


def test_median_ignores_a_strong_outlier():
    # Three consistent vectors of (1, 2) px plus one wild (100, -100) px vector:
    # the median reports the dominant background motion, a mean would not.
    vectors = [(1 * PX, 2 * PX)] * 3 + [(100 * PX, -100 * PX)]
    stats = estimate_global_motion(vectors)
    assert stats.global_dx == pytest.approx(1.0, abs=1e-6)
    assert stats.global_dy == pytest.approx(2.0, abs=1e-6)
    # |(100, -100)| = 141 px > the 128 px magnitude gate, so it is dropped
    # before the median is taken — belt and braces on top of the median itself.
    assert stats.vector_count == 3
    assert stats.valid

    # With the magnitude gate off the outlier is kept and the median still holds.
    kept = estimate_global_motion(vectors, max_magnitude_px=0)
    assert kept.vector_count == 4
    assert kept.global_dx == pytest.approx(1.0, abs=1e-6)
    assert kept.global_dy == pytest.approx(2.0, abs=1e-6)


def test_moving_targets_do_not_capture_the_estimate():
    # 60% background pans right by 3 px; 40% of the frame (a passing vessel,
    # wake and spray) moves the other way. The background still wins.
    vectors = [(3 * PX, 0)] * 60 + [(-9 * PX, 5 * PX)] * 40
    stats = estimate_global_motion(vectors)
    assert stats.global_dx == pytest.approx(3.0)
    assert stats.global_dy == pytest.approx(0.0)


def test_empty_input_yields_no_estimate():
    stats = estimate_global_motion([])
    assert stats.vector_count == 0
    assert not stats.valid
    assert stats.global_dx == 0.0
    assert stats.global_dy == 0.0
    assert stats.confidence == 0.0


def test_none_input_yields_no_estimate():
    assert estimate_global_motion(None).vector_count == 0


# ── Input hardening ───────────────────────────────────────────────────────────


def test_non_finite_and_malformed_vectors_are_ignored():
    vectors = [
        (1 * PX, 1 * PX),
        (float("nan"), 1 * PX),
        (1 * PX, float("inf")),
        (float("-inf"), float("nan")),
        (7,),                 # malformed: too short
        42,                   # malformed: not indexable
        ("a", "b"),           # malformed: non-numeric
        (1 * PX, 1 * PX),
        (1 * PX, 1 * PX),
    ]
    stats = estimate_global_motion(vectors)
    assert stats.vector_count == 3
    assert stats.global_dx == pytest.approx(1.0)
    assert stats.global_dy == pytest.approx(1.0)
    # confidence = share of the frame's vectors that survived filtering.
    assert stats.confidence == pytest.approx(3 / 9)


def test_all_invalid_input_yields_no_estimate_without_raising():
    stats = estimate_global_motion([(float("nan"), float("nan")), None, "xy"])
    assert stats.vector_count == 0
    assert not stats.valid


def test_absurd_magnitudes_are_rejected():
    vectors = [(2 * PX, 0)] * 3 + [(5000 * PX, 5000 * PX)]
    stats = estimate_global_motion(vectors, max_magnitude_px=128.0)
    assert stats.vector_count == 3
    assert stats.global_dx == pytest.approx(2.0)


def test_magnitude_rejection_can_be_disabled():
    vectors = [(2 * PX, 0)] * 3 + [(5000 * PX, 5000 * PX)]
    assert estimate_global_motion(vectors, max_magnitude_px=0).vector_count == 4


def test_static_scene_reports_near_zero():
    # NVIDIA notes the quantisation floor makes a static scene report +-0.5 px
    # (raw +-16) rather than exactly 0; the estimate must stay in that band.
    vectors = [(16, -16), (0, 16), (-16, 0), (16, 16), (0, 0)]
    stats = estimate_global_motion(vectors)
    assert abs(stats.global_dx) <= 0.5
    assert abs(stats.global_dy) <= 0.5


# ── numpy path (what pyds actually hands the probe) ───────────────────────────


def test_flat_float32_array_matches_the_pair_path():
    # pyds.get_optical_flow_vectors() returns a FLAT float32 array of raw
    # components (x, y, x, y, ...) — the shape the probe passes in.
    pairs = [(1 * PX, 2 * PX)] * 3 + [(100 * PX, -100 * PX)]
    flat = np.array([c for pair in pairs for c in pair], dtype=np.float32)
    from_array = estimate_global_motion(flat)
    from_pairs = estimate_global_motion(pairs)
    assert from_array.global_dx == pytest.approx(from_pairs.global_dx)
    assert from_array.global_dy == pytest.approx(from_pairs.global_dy)
    assert from_array.vector_count == from_pairs.vector_count == 3
    assert from_array.confidence == pytest.approx(from_pairs.confidence)


def test_rows_cols_2_array_is_accepted():
    # The DeepStream sample reshapes to (rows, cols, 2); either shape must work.
    grid = np.full((4, 5, 2), 32.0, dtype=np.float32)
    grid[0, 0] = (32 * 500, -32 * 500)  # one absurd vector
    stats = estimate_global_motion(grid)
    assert stats.vector_count == 19
    assert stats.global_dx == pytest.approx(1.0)
    assert stats.global_dy == pytest.approx(1.0)


def test_array_with_non_finite_values_is_filtered():
    flat = np.array([32, 32, np.nan, 32, 32, np.inf, 32, 32], dtype=np.float32)
    stats = estimate_global_motion(flat)
    assert stats.vector_count == 2
    assert stats.global_dx == pytest.approx(1.0)
    assert stats.confidence == pytest.approx(0.5)


def test_odd_length_array_drops_the_dangling_component():
    stats = estimate_global_motion(np.array([32, 64, 32], dtype=np.float32))
    assert stats.vector_count == 1
    assert stats.global_dy == pytest.approx(2.0)


def test_empty_array_yields_no_estimate():
    assert estimate_global_motion(np.array([], dtype=np.float32)).vector_count == 0


# ── Per-camera diagnostic state ───────────────────────────────────────────────


def test_state_disabled_when_feature_is_off():
    flow = CameraFlowState("forward", enabled=False)
    assert flow.state(now=100.0) is OpticalFlowState.disabled
    snap = flow.snapshot(now=100.0)
    assert snap["enabled"] is False
    assert snap["active"] is False
    assert snap["global_dx"] is None
    assert snap["vectors"] == 0


def test_state_no_data_until_the_first_flow_arrives():
    # Optical flow needs a previous frame, so the first frame after start has
    # none — that is normal, not an error.
    flow = CameraFlowState("forward", enabled=True)
    assert flow.state(now=100.0) is OpticalFlowState.no_data
    assert flow.snapshot(now=100.0)["error"] is None


def test_state_active_then_stale():
    flow = CameraFlowState("forward", enabled=True)
    flow.update(estimate_global_motion([(32, -64)], updated_at=100.0))
    assert flow.state(now=100.2, stale_after_s=2.0) is OpticalFlowState.active
    assert flow.state(now=105.0, stale_after_s=2.0) is OpticalFlowState.stale

    snap = flow.snapshot(now=100.5, stale_after_s=2.0)
    assert snap["active"] is True
    assert snap["state"] == "active"
    assert snap["global_dx"] == pytest.approx(1.0)
    assert snap["global_dy"] == pytest.approx(-2.0)
    assert snap["vectors"] == 1
    assert snap["age_ms"] == 500


def test_error_state_wins_over_a_stored_estimate():
    flow = CameraFlowState("forward", enabled=True)
    flow.update(estimate_global_motion([(32, 32)], updated_at=100.0))
    assert flow.fail("nvof failed: caps negotiation") is True
    assert flow.state(now=100.1) is OpticalFlowState.error
    assert flow.snapshot(now=100.1)["error"] == "nvof failed: caps negotiation"
    # Repeating the same failure is not "new" — the caller logs once, not per frame.
    assert flow.fail("nvof failed: caps negotiation") is False
    # A fresh estimate clears the error.
    flow.update(estimate_global_motion([(32, 32)], updated_at=101.0))
    assert flow.state(now=101.1) is OpticalFlowState.active


def test_reset_clears_flow_history_on_pipeline_rebuild():
    flow = CameraFlowState("forward", enabled=True)
    flow.update(estimate_global_motion([(32, 32)], updated_at=100.0))
    flow.fail("boom")
    flow.reset()
    # A rebuilt pipeline starts a new nvof epoch: no stale motion, no stale error.
    assert flow.stats is None
    assert flow.error is None
    assert flow.state(now=100.1) is OpticalFlowState.no_data


def test_cameras_keep_independent_state():
    fwd = CameraFlowState("forward", enabled=True)
    aft = CameraFlowState("aft", enabled=True)
    fwd.update(estimate_global_motion([(320, 0)], updated_at=100.0))
    aft.update(estimate_global_motion([(0, 32)], updated_at=100.0))
    # Camera 0's motion must not appear as camera 1's.
    assert fwd.snapshot(now=100.0)["global_dx"] == pytest.approx(10.0)
    assert aft.snapshot(now=100.0)["global_dx"] == pytest.approx(0.0)
    assert aft.snapshot(now=100.0)["global_dy"] == pytest.approx(1.0)
    # A stalled camera does not drag the other's freshness.
    fwd.reset()
    assert fwd.state(now=100.0) is OpticalFlowState.no_data
    assert aft.state(now=100.0) is OpticalFlowState.active


def test_snapshot_is_json_friendly():
    flow = CameraFlowState("forward", enabled=True)
    flow.update(estimate_global_motion([(33, -17)] * 4, updated_at=100.0))
    snap = flow.snapshot(now=100.05)
    assert set(snap) == {
        "enabled", "state", "active", "global_dx", "global_dy",
        "vectors", "confidence", "age_ms", "error",
    }
    assert all(not isinstance(v, (np.generic,)) for v in snap.values())
    assert math.isfinite(snap["global_dx"]) and math.isfinite(snap["global_dy"])
