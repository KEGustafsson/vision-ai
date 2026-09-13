import threading

import cv2
import numpy as np

from app.camera import rtsp_cpu
from app.camera.rtsp_cpu import _FFMPEG_LOW_LATENCY_OPTIONS, RtspCpuSource
from app.camera.rtsp_gstreamer import build_pipeline


def test_ffmpeg_rtsp_options_prefer_live_frames():
    assert "rtsp_transport;tcp" in _FFMPEG_LOW_LATENCY_OPTIONS
    assert "fflags;nobuffer" in _FFMPEG_LOW_LATENCY_OPTIONS
    assert "flags;low_delay" in _FFMPEG_LOW_LATENCY_OPTIONS
    assert "reorder_queue_size;0" in _FFMPEG_LOW_LATENCY_OPTIONS


def test_gstreamer_pipeline_drops_late_frames():
    pipeline = build_pipeline("rtsp://camera.example/live")

    assert "latency=50" in pipeline
    assert "drop-on-latency=true" in pipeline
    assert "appsink sync=false drop=true max-buffers=1" in pipeline


def test_gstreamer_pipeline_runs_nvdec_at_max_performance_when_supported():
    # NVDEC must not DVFS down between frames: steady, minimal decode latency.
    pipeline = build_pipeline("rtsp://camera.example/live", max_performance=True)
    assert "nvv4l2decoder enable-max-performance=1 !" in pipeline


def test_gstreamer_pipeline_omits_the_clock_knob_when_the_plugin_lacks_it():
    # gst_parse_launch rejects an unknown property outright, so the capture
    # would never open: the property must be left out, not risked.
    pipeline = build_pipeline("rtsp://camera.example/live", max_performance=False)
    assert "enable-max-performance" not in pipeline
    assert "! nvv4l2decoder ! nvvidconv !" in pipeline


def test_gstreamer_pipeline_probes_the_plugin_once_by_default(monkeypatch):
    from app.camera import rtsp_gstreamer as mod

    calls = []

    def probe():
        calls.append(1)
        return True

    monkeypatch.setattr(mod, "_probe_nvdec_max_performance", probe)
    monkeypatch.setattr(mod, "_NVDEC_MAX_PERF", None)
    assert "enable-max-performance=1" in build_pipeline("rtsp://camera.example/live")
    assert "enable-max-performance=1" in build_pipeline("rtsp://camera.example/live")
    assert len(calls) == 1  # cached: the plugin set does not change at runtime

    # No PyGObject / no plugin (this CI host): the knob is omitted, never guessed.
    monkeypatch.setattr(mod, "_NVDEC_MAX_PERF", None)
    monkeypatch.setattr(mod, "_probe_nvdec_max_performance", lambda: False)
    assert "enable-max-performance" not in build_pipeline("rtsp://camera.example/live")


def _bare_source() -> RtspCpuSource:
    """RtspCpuSource with just the read()-path state (no capture, no reader
    thread), to exercise the consumer-side delivery logic in isolation."""
    src = RtspCpuSource.__new__(RtspCpuSource)
    src._closed = False
    src._lock = threading.Lock()
    src._frame_ready = threading.Condition(src._lock)
    src._latest_img = np.zeros((2, 2, 3), dtype=np.uint8)
    src._latest_seq = 1
    src._last_delivered_seq = 0
    return src


def test_read_reports_a_stall_instead_of_reserving_the_old_frame(monkeypatch):
    # Keep the wait window short so the stall path is exercised quickly.
    monkeypatch.setattr(rtsp_cpu, "_READ_WAIT_S", 0.05)
    src = _bare_source()

    first = src.read()
    assert first is not None and first.seq == 1

    # No new frame arrives: read() must return None (stall) so the pipeline's
    # stall detection / health reporting can trigger — NOT re-serve frame 1 as
    # if it were fresh (a frozen feed would then keep publishing stale scene
    # data under current event timestamps forever).
    assert src.read() is None

    # A fresh frame resumes delivery.
    with src._frame_ready:
        src._latest_img = np.ones((2, 2, 3), dtype=np.uint8)
        src._latest_seq = 2
        src._frame_ready.notify_all()
    nxt = src.read()
    assert nxt is not None and nxt.seq == 2


def test_capture_params_are_accepted_by_the_installed_opencv(tmp_path):
    """The FFmpeg backend rejects the WHOLE open-parameter list if any entry
    goes unused, and then the capture never opens — so an unsupported property
    here (CAP_PROP_BUFFERSIZE was one) silently kills every CPU-path camera and
    every reconnect. Prove the list this module builds actually opens a file."""
    import cv2

    clip = tmp_path / "probe.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened(), "cannot encode a probe clip on this host"
    for i in range(5):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()

    cap = cv2.VideoCapture(str(clip), cv2.CAP_FFMPEG, rtsp_cpu.capture_params())
    try:
        assert cap.isOpened(), "OpenCV rejected the capture parameters"
        assert cap.read()[0]
    finally:
        cap.release()


def test_capture_params_carry_the_open_and_read_timeouts():
    # Without these the FFmpeg defaults (30 s each) apply, so one half-dead
    # camera blocks its reader thread for 30 s per attempt.
    params = rtsp_cpu.capture_params()
    for name in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
        prop = getattr(cv2, name, None)
        if prop is not None:
            assert prop in params
            assert params[params.index(prop) + 1] == 5000
    # CAP_PROP_BUFFERSIZE is not consumed by the FFmpeg backend: including it
    # makes the open fail outright.
    assert cv2.CAP_PROP_BUFFERSIZE not in params
