from app.camera.rtsp_cpu import _FFMPEG_LOW_LATENCY_OPTIONS
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
