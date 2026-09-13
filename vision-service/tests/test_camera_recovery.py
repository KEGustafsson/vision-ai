"""A camera that stops delivering must come back on its own.

A GStreamer capture never recovers once rtspsrc has errored or hit EOS (camera
reboot, TCP reset, a cable pulled and replaced): every later read() returns None
forever, so without re-creating the source the camera stays dead until someone
restarts the container. Re-creating it restarts the source's own frame counter,
which the long-lived stabilizer and tracker use to age their state — so the
worker has to keep the sequence it publishes monotonic across the re-create.
"""

import logging
import time

import numpy as np

from app import pipeline as pipeline_mod
from app.camera.base import Frame
from app.config import load_settings
from app.detector.mock import MockDetector
from app.pipeline import CameraWorker
from app.util import EventBuffer, LatestFrame

LOG = logging.getLogger("test-camera-recovery")


class _StallingSource:
    """Delivers `good` frames, then nothing — like a wedged RTSP feed."""

    def __init__(self, good: int = 2):
        self._good = good
        self._reads = 0
        self.seq = 0
        self.closed = False

    def read(self):
        self._reads += 1
        if self._reads > self._good:
            return None
        self.seq += 1
        return Frame(image=np.zeros((32, 32, 3), dtype=np.uint8), seq=self.seq)

    def close(self):
        self.closed = True


def _worker(monkeypatch, sources):
    settings = load_settings("mock")
    settings.server.target_fps = 50.0        # keep the test quick
    monkeypatch.setattr(pipeline_mod, "create_source",
                        lambda cam, s: sources.append(_StallingSource()) or sources[-1])
    monkeypatch.setattr(pipeline_mod, "STALL_TIMEOUT_S", 0.02)
    monkeypatch.setattr(pipeline_mod, "STALL_REOPEN_S", 0.05)
    return CameraWorker(settings.cameras[0], settings, EventBuffer(maxlen=50),
                        LatestFrame(), LOG, MockDetector())


def _run_until(worker, predicate, timeout=10.0):
    worker.start()
    try:
        deadline = time.time() + timeout
        while time.time() < deadline and not predicate():
            time.sleep(0.02)
        return predicate()
    finally:
        worker.stop()
        worker.join(timeout=2.0)


def test_worker_reopens_a_source_that_stopped_delivering(monkeypatch):
    sources: list = []
    worker = _worker(monkeypatch, sources)
    assert _run_until(worker, lambda: len(sources) >= 3), \
        f"source was not re-created (created {len(sources)})"
    # Every superseded source is released, not leaked.
    assert all(s.closed for s in sources[:-1])


def test_frame_sequence_stays_monotonic_across_a_reopen(monkeypatch):
    sources: list = []
    worker = _worker(monkeypatch, sources)
    events = worker._events
    assert _run_until(
        worker, lambda: len(sources) >= 3 and len(events.recent(50)) >= 5)
    seqs = [e["frame_seq"] for e in events.recent(50)]
    # Each source restarts its own counter at 1 (two frames each), so a naive
    # pass-through would publish 1, 2, 1, 2, 1, 2 — and the stabilizer, which
    # ages its state on the difference between sequence numbers, would then
    # coast every pre-reopen track for as many frames as the camera had been up.
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)
    assert max(seqs) >= 5


def test_a_stalled_worker_is_reported_in_health(monkeypatch):
    """The worker's own flag only fires when read() RETURNS without a frame.
    camera_errors() also has to catch a worker that is stuck inside a blocking
    read or a wedged GPU call, which would otherwise keep /health green while
    the camera produces nothing at all."""
    settings = load_settings("mock")
    p = pipeline_mod.Pipeline(settings, LOG)
    monkeypatch.setattr(pipeline_mod, "STALL_TIMEOUT_S", 0.01)

    class _WedgedWorker:
        error = None
        last_frame_at = time.monotonic() - 60.0

    p.workers = {"forward": _WedgedWorker()}
    assert "forward" in p.camera_errors()
    assert "worker stalled" in p.camera_errors()["forward"]

    # Detection switched off: silence is expected, not a fault.
    p.enabled = False
    assert p.camera_errors() == {}


def test_worker_join_actually_stops_the_thread(monkeypatch):
    """threading.Thread keeps a private _stop() that join()/is_alive() call once
    the thread has finished. A stop Event stored under that name made both raise
    TypeError — so Pipeline.stop() aborted on the first worker and the rest kept
    their capture devices across a restart."""
    sources: list = []
    worker = _worker(monkeypatch, sources)
    worker.start()
    worker.stop()
    worker.join(timeout=2.0)          # used to raise TypeError
    assert worker.is_alive() is False  # ...and so did this
