"""DeepStream graph construction, without DeepStream.

``DeepStreamPipeline._build_pipeline`` only ever ran on a Jetson, so a wrong
link, a probe on the wrong pad or a tuning property that silently stopped
being applied would surface on the water, not in CI. This fakes just enough
of ``Gst`` (elements, pads, links, probes) to build the graph on any machine
and assert its shape: which element the probe hangs off, where the probe
queue sits, what the decoders and the tracker are configured with.
"""

import logging
from types import SimpleNamespace

import pytest

from app.config import load_settings
from app.pipeline_deepstream import DeepStreamPipeline, _tracker_dims

LOG = logging.getLogger("test-ds-graph")

_OK = 0  # Gst.PadLinkReturn.OK
_BUFFER = 16  # Gst.PadProbeType.BUFFER


class _Pad:
    def __init__(self, element, name):
        self.element = element
        self.name = name
        self.peer = None
        self.probes: list = []

    def link(self, other):
        self.peer, other.peer = other, self
        return _OK

    def is_linked(self):
        return self.peer is not None

    def add_probe(self, mask, callback, user_data):
        self.probes.append((mask, callback))


class _Element:
    """Records properties, pads, links and signal connections."""

    def __init__(self, factory, name, graph):
        self.factory = factory
        self.name = name
        self.props: dict = {}
        self.pads: dict = {}
        self.signals: list = []
        self._graph = graph

    def set_property(self, key, value):
        if (key, self.factory) in self._graph.unsupported:
            raise TypeError(f"object of type {self.factory} does not have property {key!r}")
        self.props[key] = value

    def get_name(self):
        return self.name

    def link(self, other):
        self._graph.links.append((self.name, other.name))
        return True

    def _pad(self, name):
        return self.pads.setdefault(name, _Pad(self, name))

    get_static_pad = _pad
    get_request_pad = _pad

    def connect(self, signal, callback, *args):
        self.signals.append((signal, callback, args))


class _Graph:
    """Everything the fake Gst created while _build_pipeline ran."""

    def __init__(self, unsupported=()):
        self.elements: list = []
        self.links: list = []
        self.added: list = []
        # (property, factory) pairs the fake refuses, to model a plugin build
        # without that knob.
        self.unsupported = set(unsupported)

    def make(self, factory, name):
        el = _Element(factory, name, self)
        self.elements.append(el)
        return el

    def by_name(self, name):
        return next(el for el in self.elements if el.name == name)

    def by_factory(self, factory):
        return [el for el in self.elements if el.factory == factory]

    def gst(self):
        graph = self

        class _Pipeline:
            def add(self, el):
                graph.added.append(el.name)

            def remove(self, el):
                graph.added.remove(el.name)

        return SimpleNamespace(
            Pipeline=SimpleNamespace(new=lambda name: _Pipeline()),
            ElementFactory=SimpleNamespace(make=self.make, find=lambda name: object()),
            Caps=SimpleNamespace(from_string=lambda s: s),
            PadLinkReturn=SimpleNamespace(OK=_OK),
            PadProbeType=SimpleNamespace(BUFFER=_BUFFER),
        )


def _pipeline(undistort=False, optical_flow=False) -> DeepStreamPipeline:
    settings = load_settings("deepstream")
    for i, cam in enumerate(settings.cameras):
        cam.url = f"rtsp://user:pass@192.0.2.{i + 1}/stream"
        cam.undistort = undistort
    settings.detector.optical_flow = optical_flow
    settings.detector.optical_flow_required = False
    return DeepStreamPipeline(settings, LOG)


def _build(p: DeepStreamPipeline, unsupported=()):
    graph = _Graph(unsupported)
    p._build_pipeline(graph.gst())
    if p._dewarp_tmp is not None:
        p._dewarp_tmp.cleanup()
        p._dewarp_tmp = None
    return graph


# ── Probe placement ───────────────────────────────────────────────────────────


def test_probe_runs_on_its_own_queue_thread_not_the_gpu_chain():
    graph = _build(_pipeline())
    probeq = graph.by_name("probeq")
    assert probeq.factory == "queue"

    # The probe hangs off the queue's src pad (the queue's own streaming
    # thread) — nowhere else in the graph carries a probe.
    probed = [
        (el.name, pad.name) for el in graph.elements for pad in el.pads.values() if pad.probes
    ]
    assert probed == [("probeq", "src")]
    ((mask, callback),) = probeq.pads["src"].probes
    assert mask == _BUFFER
    assert callback.__func__ is DeepStreamPipeline._probe_callback

    # ... and it sits between the RGBA capsfilter and the demux.
    assert ("dispcaps", "probeq") in graph.links
    assert ("probeq", "demux") in graph.links
    assert ("dispcaps", "demux") not in graph.links


def test_probe_queue_is_shallow_and_not_leaky():
    # Back-pressure, not batch loss: a slow probe must push back to the leaky
    # per-camera source queues (drop a stale decoded frame) rather than drop a
    # batch that nvinfer/nvtracker already processed. Two buffers keep the GPU
    # chain busy; more only adds latency.
    probeq = _build(_pipeline()).by_name("probeq")
    assert probeq.props["max-size-buffers"] == 2
    assert probeq.props["max-size-bytes"] == 0
    assert probeq.props["max-size-time"] == 0
    assert "leaky" not in probeq.props


def test_batched_chain_order_is_mux_infer_track_convert_queue_demux():
    graph = _build(_pipeline())
    main = [
        (a, b)
        for a, b in graph.links
        if a in ("mux", "pgie", "tracker", "dispconv", "dispcaps", "probeq")
    ]
    assert main == [
        ("mux", "pgie"),
        ("pgie", "tracker"),
        ("tracker", "dispconv"),
        ("dispconv", "dispcaps"),
        ("dispcaps", "probeq"),
        ("probeq", "demux"),
    ]


def test_optical_flow_branch_still_precedes_the_probe_queue():
    graph = _build(_pipeline(optical_flow=True))
    assert ("tracker", "ofconv") in graph.links
    assert ("of", "dispconv") in graph.links
    assert ("dispcaps", "probeq") in graph.links


# ── Per-camera front end ──────────────────────────────────────────────────────


def test_every_decoder_runs_at_max_performance():
    graph = _build(_pipeline())
    decoders = graph.by_factory("nvv4l2decoder")
    assert len(decoders) == 2
    assert all(d.props.get("enable-max-performance") is True for d in decoders)


def test_decoder_without_the_clock_knob_still_builds():
    # A plugin build that lacks the property must degrade to its default, not
    # fail the pipeline build (which would take detection down at startup).
    graph = _build(_pipeline(), unsupported={("enable-max-performance", "nvv4l2decoder")})
    decoders = graph.by_factory("nvv4l2decoder")
    assert len(decoders) == 2
    assert all("enable-max-performance" not in d.props for d in decoders)
    assert ("dispcaps", "probeq") in graph.links  # the rest of the graph is intact


def test_source_queues_are_leaky_so_load_is_shed_at_the_decoder():
    graph = _build(_pipeline())
    for i in range(2):
        q = graph.by_name(f"q{i}")
        assert q.props["leaky"] == 2
        assert q.props["max-size-buffers"] == 4
        assert (f"q{i}", "mux") not in graph.links  # request-pad link, not element link
        assert q.pads["src"].peer is graph.by_name("mux").pads[f"sink_{i}"]


# ── Tracker working resolution ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "imgsz,mux_w,mux_h,expected",
    [
        (768, 1280, 960, (768, 576)),  # the deployed 4:3 domes
        (640, 1280, 960, (640, 480)),
        (768, 1920, 1080, (768, 448)),  # 16:9: 432 rounds to the nearest multiple of 32
        (640, 1920, 1080, (640, 352)),  # 360 → 352
        (800, 1280, 960, (800, 608)),  # 600 rounds up to the nearest multiple of 32
        (700, 1280, 960, (672, 512)),  # width itself snaps down to a multiple of 32
        (16, 1280, 960, (32, 32)),  # never below the 32-px NvDCF minimum
        (768, 0, 0, (768, 768)),  # degenerate mux size: fall back to square
    ],
)
def test_tracker_dims_keep_mux_aspect_on_a_32_grid(imgsz, mux_w, mux_h, expected):
    w, h = _tracker_dims(imgsz, mux_w, mux_h)
    assert (w, h) == expected
    assert w % 32 == 0 and h % 32 == 0


def test_tracker_is_configured_with_aspect_correct_dims():
    p = _pipeline()
    d = p.settings.detector
    tracker = _build(p).by_name("tracker")
    w, h = _tracker_dims(d.imgsz, d.mux_width, d.mux_height)
    assert tracker.props["tracker-width"] == w
    assert tracker.props["tracker-height"] == h
    # A 4:3 source must not be handed to NvDCF squashed 2:1 (the old fixed 384).
    assert abs(w / h - d.mux_width / d.mux_height) < 0.1
    assert tracker.props["display-tracking-id"] == 0


# ── Display tail ──────────────────────────────────────────────────────────────


def test_each_camera_gets_its_own_osd_and_hw_jpeg_tail():
    graph = _build(_pipeline())
    for i in range(2):
        for a, b in [
            (f"encq{i}", f"osd{i}"),
            (f"osd{i}", f"jconv{i}"),
            (f"jconv{i}", f"jcaps{i}"),
            (f"jcaps{i}", f"nvjpegenc{i}"),
            (f"nvjpegenc{i}", f"appsink{i}"),
        ]:
            assert (a, b) in graph.links
        assert graph.by_name(f"jcaps{i}").props["caps"] == "video/x-raw(memory:NVMM),format=I420"
        sink = graph.by_name(f"appsink{i}")
        assert sink.props["sync"] is False and sink.props["drop"] is True
        assert sink.props["max-buffers"] == 1
        assert [s[0] for s in sink.signals] == ["new-sample"]
