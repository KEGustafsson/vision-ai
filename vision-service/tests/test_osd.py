"""Pure-logic tests for the DeepStream GPU overlay (app.api.osd).

The nvdsosd/pyds drawing needs a Jetson + DeepStream, so here we only cover the
host-side helpers: colour mapping, coasted dimming, timestamp formatting, and the
dashed-edge segment generation. The full render is validated on hardware via the
standalone pipeline test."""

from app.api import osd
from app.schemas import BBox, Geometry, PixelVelocity, RangeMethod, Target


def _target(range_m=None, piw=False):
    return Target(
        track_id=1, label="vessel", coco_class=8, confidence=0.9,
        bbox=BBox(x=0, y=0, w=10, h=10), is_person_in_water=piw,
        geometry=Geometry(relative_bearing_deg=0.0, range_m=range_m,
                          range_method=RangeMethod.horizon if range_m else None,
                          range_confidence=0.5),
        pixel_velocity=PixelVelocity(vx=0.0, vy=0.0), age_frames=1, coasting=False)


def test_severity_colour_matches_overlay_rules():
    assert osd._severity_colour(_target(piw=True)) == osd._RED       # person-in-water
    assert osd._severity_colour(_target(range_m=50)) == osd._AMBER   # < 100 m
    assert osd._severity_colour(_target(range_m=500)) == osd._GREEN  # far
    assert osd._severity_colour(_target(range_m=None)) == osd._GREEN # unknown range


def test_dim_is_60_percent_brightness():
    assert osd._dim((1.0, 0.5, 0.0, 1.0)) == (0.6, 0.3, 0.0, 1.0)


def test_format_timestamp():
    assert osd._format_timestamp("2026-05-31T12:34:56.789Z") == "2026-05-31 12:34:56 UTC"
    assert osd._format_timestamp("2026-05-31T12:34:56+00:00") == "2026-05-31 12:34:56 UTC"


def test_dashed_edges_cover_perimeter_in_segments():
    segs = list(osd._dashed_edges(0, 0, 100, 50))
    assert segs, "expected dash segments"
    # every segment stays on the rectangle perimeter (x in {0,100} or y in {0,50})
    for x1, y1, x2, y2 in segs:
        on_v = x1 in (0, 100) and x2 in (0, 100)
        on_h = y1 in (0, 50) and y2 in (0, 50)
        assert on_v or on_h
