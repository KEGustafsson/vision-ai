"""JPEG encoder backends.

The hardware (nvjpegenc) path needs python3-gi and Jetson plugins, absent on
dev/CI, so here we cover the CPU encoder and the factory's safe HW->CPU
fallback."""

import cv2
import numpy as np

import app.api.jpeg as jpeg_mod
from app.api.jpeg import CpuJpegEncoder, make_jpeg_encoder


def _frame():
    img = np.zeros((64, 96, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (40, 40), (0, 200, 0), -1)
    return img


def test_cpu_encoder_round_trips_to_valid_jpeg():
    enc = CpuJpegEncoder(quality=80)
    data = enc.encode(_frame())
    assert data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"  # JPEG SOI/EOI
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (64, 96, 3)


def test_factory_returns_cpu_when_hw_disabled():
    enc = make_jpeg_encoder(quality=80, hw=False)
    assert isinstance(enc, CpuJpegEncoder)
    assert enc.backend == "cpu"


def test_factory_falls_back_to_cpu_when_hw_unavailable(monkeypatch):
    # Force the HW encoder to fail at construction so the fallback is exercised
    # deterministically — otherwise this passes only on hosts without gi/nvjpegenc
    # and would return a real HwJpegEncoder on a Jetson.
    def _boom(*_a, **_k):
        raise RuntimeError("nvjpegenc unavailable (test)")
    monkeypatch.setattr(jpeg_mod, "HwJpegEncoder", _boom)
    enc = make_jpeg_encoder(quality=80, hw=True)
    assert isinstance(enc, CpuJpegEncoder)
    assert enc.encode(_frame())
