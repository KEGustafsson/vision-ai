"""Lockstep checks on the nvinfer (pgie) configs.

These files are parsed by DeepStream, not by the app, so nothing in the code
path can catch a drifted key. Every model config must agree with the deployed
camera set and carry the pre-processing settings the detector was trained
with — a mismatch here shows up as degraded detection on the water, not as an
error.
"""

import configparser
from pathlib import Path

import pytest

from app.config import load_settings
from app.detector.classmap import MODEL_PGIE_CONFIG

_DS_DIR = Path(__file__).resolve().parents[1] / "deepstream"


def _load(name: str) -> configparser.ConfigParser:
    cp = configparser.ConfigParser(inline_comment_prefixes=None, strict=True)
    cp.optionxform = str  # keep key case
    cp.read(_DS_DIR / name)
    return cp


@pytest.mark.parametrize("model,cfg_name", sorted(MODEL_PGIE_CONFIG.items()))
def test_pgie_config_matches_the_deployment(model, cfg_name):
    cp = _load(cfg_name)
    prop = cp["property"]
    cameras = load_settings("deepstream").cameras

    # nvstreammux batches one frame per camera; nvinfer's engine is built for
    # exactly this batch, so the two must move together.
    assert int(prop["batch-size"]) == len(cameras)
    # Sole primary detector — nvtracker correlates on this id.
    assert int(prop["gie-unique-id"]) == 1
    assert int(prop["network-type"]) == 0
    # FP16 is the Jetson sweet spot; INT8 needs a calibration set.
    assert int(prop["network-mode"]) == 2
    # The parser is the image's own build, by absolute path outside the
    # bind-mount (a relative path would resolve to the host's, possibly
    # other-board, copy).
    assert prop["custom-lib-path"].startswith("/opt/vision-service/lib/")
    assert prop["parse-bbox-func-name"] == "NvDsInferParseYolo"


@pytest.mark.parametrize("cfg_name", sorted(set(MODEL_PGIE_CONFIG.values())))
def test_pgie_prescale_matches_ultralytics_letterbox(cfg_name):
    """The network input is a letterboxed, BILINEAR downscale of the mux frame,
    exactly as Ultralytics prepares frames for training and for the torch/
    TensorRT backends. nvinfer's default filter is nearest-neighbour, which on
    the 1280→768 downscale drops 40% of the source rows outright — small,
    distant targets alias away. Pinned so a config rewrite can't lose it."""
    prop = _load(cfg_name)["property"]
    assert int(prop["scaling-filter"]) == 1
    assert int(prop["maintain-aspect-ratio"]) == 1
    assert int(prop["symmetric-padding"]) == 1
    # RGB, 1/255 — Ultralytics normalisation.
    assert int(prop["model-color-format"]) == 0
    assert float(prop["net-scale-factor"]) == pytest.approx(1 / 255, rel=1e-6)


def test_every_registered_model_has_its_config_file():
    for cfg_name in MODEL_PGIE_CONFIG.values():
        assert (_DS_DIR / cfg_name).is_file(), cfg_name
