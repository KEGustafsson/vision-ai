import threading

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


def test_gstreamer_pipeline_runs_nvdec_at_max_performance():
    # NVDEC must not DVFS down between frames: steady, minimal decode latency.
    pipeline = build_pipeline("rtsp://camera.example/live")
    assert "nvv4l2decoder enable-max-performance=1 !" in pipeline


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
