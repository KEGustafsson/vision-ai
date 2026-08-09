"""DeepStream-side optical-flow (OFA) plumbing, without DeepStream.

The GStreamer/pyds glue is faked here so the parts that decide *whether* the
pipeline gets an nvof element, *what* it does when nvof is missing or broken,
and *how* flow metadata is turned into per-camera state are all covered on a
machine with no Jetson, no DeepStream and no cameras.
"""

import logging
from types import SimpleNamespace

import pytest

from app.config import load_settings
from app.motion import OpticalFlowState
from app.pipeline_deepstream import DeepStreamPipeline

LOG = logging.getLogger("test-ds-of")


# ── Fakes ─────────────────────────────────────────────────────────────────────


class _FakeElement:
    def __init__(self, factory: str, name: str):
        self.factory = factory
        self.name = name
        self.props: dict = {}

    def set_property(self, key, value):
        self.props[key] = value

    def get_name(self):
        return self.name


class _FakePipeline:
    def __init__(self):
        self.removed: list = []

    def remove(self, element):
        self.removed.append(element)


def _fake_gst(have_nvof: bool = True):
    return SimpleNamespace(
        ElementFactory=SimpleNamespace(
            find=lambda name: object() if (name != "nvof" or have_nvof) else None),
        Caps=SimpleNamespace(from_string=lambda s: s),
    )


def _maker(created: list):
    def make(factory, name, **props):
        el = _FakeElement(factory, name)
        for k, v in props.items():
            el.set_property(k.replace("_", "-"), v)
        created.append(el)
        return el
    return make


class _Node:
    """One GList node of frame_meta.frame_user_meta_list."""

    def __init__(self, data, nxt=None):
        self.data = data
        self.next = nxt


class _UserMeta:
    def __init__(self, meta_type, payload):
        self.base_meta = SimpleNamespace(meta_type=meta_type)
        self.user_meta_data = payload


def _fake_pyds(vectors, frame_num=7, raise_on_vectors=False):
    """Minimal stand-in for the pyds surface _read_optical_flow touches."""
    def get_vectors(of_meta):
        if raise_on_vectors:
            raise RuntimeError("bad meta")
        return of_meta.vectors

    return SimpleNamespace(
        NvDsMetaType=SimpleNamespace(NVDS_OPTICAL_FLOW_META="OF"),
        NvDsUserMeta=SimpleNamespace(cast=lambda d: d),
        NvDsOpticalFlowMeta=SimpleNamespace(cast=lambda d: d),
        get_optical_flow_vectors=get_vectors,
    ), SimpleNamespace(vectors=vectors, frame_num=frame_num)


def _pipeline(**detector_overrides) -> DeepStreamPipeline:
    settings = load_settings("deepstream")
    for key, value in detector_overrides.items():
        setattr(settings.detector, key, value)
    return DeepStreamPipeline(settings, LOG)


# ── Disabled by default: nothing is created, nothing changes ──────────────────


def test_disabled_by_default_creates_no_elements():
    p = _pipeline()
    assert p.settings.detector.optical_flow is False
    created: list = []
    assert p._build_optical_flow(_fake_gst(), _FakePipeline(), _maker(created), []) == ()
    assert created == []
    assert all(s["state"] == "disabled" for s in p.optical_flow_status().values())


def test_disabled_pipeline_ignores_flow_metadata():
    p = _pipeline()
    pyds, of_meta = _fake_pyds([(32, 32)])
    frame = SimpleNamespace(frame_user_meta_list=_Node(_UserMeta("OF", of_meta)))
    p._read_optical_flow(pyds, frame, "forward")
    assert p.optical_flow_status()["forward"]["state"] == "disabled"


# ── Enabled: element creation ─────────────────────────────────────────────────


def test_enabled_creates_nv12_nvmm_branch_with_preset():
    p = _pipeline(optical_flow=True, optical_flow_preset_level=1)
    created: list = []
    chain = p._build_optical_flow(
        _fake_gst(), _FakePipeline(), _maker(created), p.settings.cameras)

    assert [el.factory for el in chain] == ["nvvideoconvert", "capsfilter", "nvof"]
    assert [el.name for el in chain] == ["ofconv", "ofcaps", "of"]
    # Frames must stay in NVMM and reach nvof as NV12 — the only format the
    # DeepStream optical-flow plugin accepts.
    assert chain[1].props["caps"] == "video/x-raw(memory:NVMM),format=NV12"
    assert chain[2].props["preset-level"] == 1
    assert all(s["state"] == "no_data" for s in p.optical_flow_status().values())


# ── Enabled but unavailable: fail-safe fallback ───────────────────────────────


def test_missing_nvof_falls_back_without_ofa():
    p = _pipeline(optical_flow=True)
    created: list = []
    # No nvof plugin (wrong DeepStream build / no OFA block on this SoC).
    assert p._build_optical_flow(
        _fake_gst(have_nvof=False), _FakePipeline(), _maker(created), []) == ()
    assert created == []          # nothing half-built is left in the pipeline
    status = p.optical_flow_status()
    for cam in status.values():
        # Enabled in config, but reported as an actionable error rather than
        # silently missing — and detection keeps running.
        assert cam["enabled"] is True
        assert cam["state"] == OpticalFlowState.error.value
        assert "nvof" in cam["error"]


def test_required_ofa_fails_startup_with_an_ofa_specific_error():
    p = _pipeline(optical_flow=True, optical_flow_required=True)
    with pytest.raises(RuntimeError, match="optical flow"):
        p._build_optical_flow(_fake_gst(have_nvof=False), _FakePipeline(), _maker([]), [])


def test_fallback_is_sticky_across_rebuilds():
    p = _pipeline(optical_flow=True)
    p._build_optical_flow(_fake_gst(have_nvof=False), _FakePipeline(), _maker([]), [])
    # A later rebuild must not re-add nvof (and re-break the pipeline) even
    # though the plugin now "exists": the failure was recorded.
    created: list = []
    assert p._build_optical_flow(_fake_gst(), _FakePipeline(), _maker(created), []) == ()
    assert created == []


def test_bus_error_from_nvof_drops_ofa_but_keeps_the_pipeline():
    p = _pipeline(optical_flow=True)
    err = SimpleNamespace(message="caps negotiation failed")
    p._note_optical_flow_error(err, "debug info", _FakeElement("nvof", "of"))
    assert p._of_failed is not None
    assert p.optical_flow_status()["forward"]["state"] == "error"
    # Next rebuild comes up without OFA.
    assert p._build_optical_flow(_fake_gst(), _FakePipeline(), _maker([]), []) == ()


def test_bus_error_from_another_element_does_not_touch_ofa():
    p = _pipeline(optical_flow=True)
    err = SimpleNamespace(message="rtsp connect failed")
    p._note_optical_flow_error(err, "gstrtspsrc.c", _FakeElement("rtspsrc", "src0"))
    assert p._of_failed is None
    assert p.optical_flow_status()["forward"]["state"] == "no_data"


def test_required_ofa_keeps_retrying_instead_of_dropping():
    p = _pipeline(optical_flow=True, optical_flow_required=True)
    err = SimpleNamespace(message="caps negotiation failed")
    p._note_optical_flow_error(err, "debug", _FakeElement("nvof", "of"))
    # Error surfaced, but OFA is NOT removed from the next build.
    assert p._of_failed is None
    assert p.optical_flow_status()["aft"]["state"] == "error"


# ── Metadata parsing ──────────────────────────────────────────────────────────


def test_flow_metadata_updates_only_its_own_camera():
    p = _pipeline(optical_flow=True)
    pyds, of_meta = _fake_pyds([(32 * 2, -32 * 3)] * 5)
    frame = SimpleNamespace(frame_user_meta_list=_Node(_UserMeta("OF", of_meta)))
    p._read_optical_flow(pyds, frame, "forward")

    status = p.optical_flow_status()
    assert status["forward"]["state"] == "active"
    assert status["forward"]["global_dx"] == pytest.approx(2.0)
    assert status["forward"]["global_dy"] == pytest.approx(-3.0)
    assert status["forward"]["vectors"] == 5
    # The other camera keeps its own (empty) history — no shared state.
    assert status["aft"]["state"] == "no_data"
    assert status["aft"]["global_dx"] is None


def test_unrelated_user_meta_is_skipped():
    p = _pipeline(optical_flow=True)
    pyds, of_meta = _fake_pyds([(64, 64)])
    chain = _Node(_UserMeta("SOMETHING_ELSE", object()),
                  _Node(_UserMeta("OF", of_meta)))
    p._read_optical_flow(pyds, SimpleNamespace(frame_user_meta_list=chain), "forward")
    assert p.optical_flow_status()["forward"]["global_dx"] == pytest.approx(2.0)


def test_absent_metadata_is_normal_not_an_error():
    # Optical flow needs a previous frame: the first frame after startup, an
    # RTSP reconnect or a detection off/on toggle has no flow meta at all.
    p = _pipeline(optical_flow=True)
    pyds, _ = _fake_pyds([])
    p._read_optical_flow(pyds, SimpleNamespace(frame_user_meta_list=None), "forward")
    cam = p.optical_flow_status()["forward"]
    assert cam["state"] == "no_data"
    assert cam["error"] is None


def test_broken_metadata_records_an_error_without_raising():
    p = _pipeline(optical_flow=True)
    pyds, of_meta = _fake_pyds([], raise_on_vectors=True)
    frame = SimpleNamespace(frame_user_meta_list=_Node(_UserMeta("OF", of_meta)))
    p._read_optical_flow(pyds, frame, "forward")  # must not raise: probe safety
    assert p.optical_flow_status()["forward"]["state"] == "error"


def test_missing_pyds_binding_is_reported_as_an_error():
    p = _pipeline(optical_flow=True)
    pyds = SimpleNamespace()  # bindings without the optical-flow meta type
    frame = SimpleNamespace(frame_user_meta_list=None)
    p._read_optical_flow(pyds, frame, "forward")
    p._read_optical_flow(pyds, frame, "forward")  # repeat: still one message
    cam = p.optical_flow_status()["forward"]
    assert cam["state"] == "error"
    assert "NVDS_OPTICAL_FLOW_META" in cam["error"]


def test_unknown_camera_name_is_ignored():
    p = _pipeline(optical_flow=True)
    pyds, of_meta = _fake_pyds([(32, 32)])
    frame = SimpleNamespace(frame_user_meta_list=_Node(_UserMeta("OF", of_meta)))
    p._read_optical_flow(pyds, frame, "starboard")  # not a configured camera
    assert "starboard" not in p.optical_flow_status()


# ── Rebuild resets temporal state ─────────────────────────────────────────────


def test_rebuild_resets_flow_state_per_camera():
    p = _pipeline(optical_flow=True)
    pyds, of_meta = _fake_pyds([(32, 32)])
    frame = SimpleNamespace(frame_user_meta_list=_Node(_UserMeta("OF", of_meta)))
    p._read_optical_flow(pyds, frame, "forward")
    assert p.optical_flow_status()["forward"]["state"] == "active"

    # What _bring_up does on every (re)build: a new pipeline is a new nvof
    # epoch, so no motion estimate may survive it.
    for flow in p._flow.values():
        flow.reset()
    assert p.optical_flow_status()["forward"]["state"] == "no_data"
    assert p.optical_flow_status()["forward"]["global_dx"] is None
