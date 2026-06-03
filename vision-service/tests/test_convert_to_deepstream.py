"""Exercises scripts/convert_to_deepstream.py end-to-end without model weights.

We build a tiny stand-in for a stock YOLOv8 detection ONNX — a single Identity op
whose output is shaped [1, 4+nc, N] (channels-first, the stock layout) — run the
converter, and assert the rewritten graph produces the parser's [1, N, 6]
[x1,y1,x2,y2,score,class] tensor that matches a NumPy reference decode.

Skipped automatically when onnx/onnxruntime are not installed (they are build/
provisioning deps, not runtime deps of the vision service).
"""

import sys
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import convert_to_deepstream as conv  # noqa: E402


def _make_stock_onnx(path: Path, nc: int = 6, n: int = 20) -> None:
    """A stock-layout YOLOv8 stand-in: output [1, 4+nc, n] = passthrough of input."""
    c = 4 + nc
    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, c, n])
    out = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, c, n])
    node = helper.make_node("Identity", ["images"], ["output0"])
    graph = helper.make_graph([node], "stock", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    # Pin an IR version old enough for any onnxruntime build (newer onnx stamps a
    # higher default than some runtimes accept). Mirrors what real exports carry.
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_convert_rewrites_head_and_matches_reference(tmp_path):
    stock = tmp_path / "stock.onnx"
    converted = tmp_path / "stock_ds.onnx"
    _make_stock_onnx(stock, nc=6, n=20)

    n, nc = conv.convert(stock, converted, force=False)
    assert (n, nc) == (20, 6)

    m = onnx.load(str(converted))
    assert len(m.graph.output) == 1
    out = m.graph.output[0]
    assert out.name == "output"
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    assert dims[1:] == [20, 6]  # [batch, N, 6]

    # validate() raises SystemExit on any mismatch; passing == graph math is right.
    conv.validate(stock, converted)


def test_convert_refuses_double_conversion(tmp_path):
    """A 6-channel output looks already-converted; convert() must refuse."""
    already = tmp_path / "already.onnx"
    _make_stock_onnx(already, nc=2, n=20)  # 4+2 = 6 channels
    with pytest.raises(SystemExit, match="already"):
        conv.convert(already, tmp_path / "x.onnx", force=False)


def test_force_overrides_double_conversion_guard(tmp_path):
    already = tmp_path / "already.onnx"
    converted = tmp_path / "already_ds.onnx"
    _make_stock_onnx(already, nc=2, n=20)
    n, nc = conv.convert(already, converted, force=True)
    assert (n, nc) == (20, 2)
    conv.validate(already, converted)
