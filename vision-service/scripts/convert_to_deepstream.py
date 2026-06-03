"""Convert a stock Ultralytics YOLOv8 ONNX into a DeepStream-Yolo-compatible ONNX.

WHY THIS EXISTS
===============
The custom parser this project builds (``NvDsInferParseYolo`` from
marcoslucianops/DeepStream-Yolo) does NOT read the stock Ultralytics export.
A stock export emits one tensor shaped ``[batch, 4+nc, 8400]`` — channels-first,
boxes as ``(cx, cy, w, h)`` in net-input pixels, and one score per class. The
parser instead expects the layout produced by DeepStream-Yolo's
``utils/export_yoloV8.py``: a single tensor shaped ``[batch, 8400, 6]`` where each
row is ``[x1, y1, x2, y2, max_score, class_index]`` (corner boxes, argmax folded
into the graph). See nvdsparsebbox_Yolo.cpp::decodeTensorYolo — it indexes
``output[b*6 + 0..5]`` exactly.

Feeding a stock ONNX to that parser yields ZERO detections (the bytes are
misread), which is exactly what the published SkipperDon/signalk-forward-watch
``forward-watch.onnx`` does. This tool appends the same post-processing
DeepStream-Yolo bakes in, but as pure ONNX graph surgery — so it needs neither
the ``.pt`` weights nor ultralytics, only the published ONNX.

WHAT IT DOES (mirrors export_yoloV8.py's DeepStreamOutput, in pixel space)
    transpose [b,4+nc,N] -> [b,N,4+nc]
    boxes(cx,cy,w,h) -> (x1,y1,x2,y2) = (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
    score = max(class_scores), label = argmax(class_scores)
    out   = concat([x1,y1,x2,y2,score,label]) -> [b,N,6], named "output"

USAGE
    python3 scripts/convert_to_deepstream.py IN.onnx [-o OUT.onnx]
    python3 scripts/convert_to_deepstream.py forward-watch.onnx --inplace
The default OUT is ``<in>_ds.onnx``. ``--validate`` (default on) cross-checks the
rewritten graph against a NumPy decode of the original on random input.

NOTE: after replacing the ONNX you MUST delete any cached TRT engine
(``<onnx>_b<N>_gpu0_fp16.engine``) so nvinfer rebuilds it from the new graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper


def _find_output_layout(model: onnx.ModelProto):
    """Return (out_name, n_dim_idx, c_dim_idx, n, nc) for the YOLOv8 head output.

    Stock export is [batch, 4+nc, N] (channels-first). Some exports transpose to
    [batch, N, 4+nc]. We pick the larger of the two non-batch dims as the anchor
    count N and the smaller as the channel count C=4+nc.
    """
    if len(model.graph.output) != 1:
        raise SystemExit(
            f"expected exactly 1 output, found {len(model.graph.output)}: "
            f"{[o.name for o in model.graph.output]}. This does not look like a "
            "stock YOLOv8 detection export."
        )
    out = model.graph.output[0]
    dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
    if len(dims) != 3:
        raise SystemExit(f"output {out.name!r} has rank {len(dims)} (dims={dims}); "
                         "expected rank 3 [batch, C, N] or [batch, N, C].")
    # dims[0] is batch (often 0/symbolic). The channel dim C=4+nc is always static;
    # the anchor dim N may be dynamic (0) when exported with dynamic axes — that's
    # fine, the appended graph works symbolically. Identify C as the static, small
    # non-batch dim; the other non-batch dim is N.
    d1, d2 = dims[1], dims[2]
    static = [(i, d) for i, d in ((1, d1), (2, d2)) if d > 0]
    if not static:
        raise SystemExit(f"output {out.name!r} has BOTH non-batch dims dynamic "
                         f"(dims={dims}); re-export with a fixed image size.")
    if len(static) == 2:
        # both static: channels is the smaller, anchors the larger
        if d2 >= d1:
            c_idx, n_idx, c, n = 1, 2, d1, d2
        else:
            c_idx, n_idx, c, n = 2, 1, d2, d1
    else:
        # one dynamic: the STATIC one is the channel dim (4+nc is never dynamic)
        c_idx, c = static[0]
        n_idx = 2 if c_idx == 1 else 1
        n = dims[n_idx]  # 0 → dynamic anchors
        if c > 1000:
            raise SystemExit(f"static dim {c} on axis {c_idx} is too large to be the "
                             f"YOLO channel dim (4+nc); dims={dims}.")
    nc = c - 4
    if nc < 1:
        raise SystemExit(f"channel dim C={c} implies {nc} classes; not a YOLOv8 head.")
    if c == 6:
        raise SystemExit(
            "output already has 6 channels — this looks ALREADY converted "
            "(or is a 2-class stock model). Refusing to double-convert. "
            "Use --force to override if you are certain it is stock 2-class."
        )
    return out.name, n_idx, c_idx, n, nc


def _pin_input_spatial(model: onnx.ModelProto, imgsz: int) -> bool:
    """Fix dynamic input H/W to ``imgsz`` (keep batch dynamic).

    ultralytics ``export(dynamic=True)`` leaves batch AND spatial dims symbolic.
    nvinfer/TRT can only auto-profile the batch dim (via batch-size), so dynamic
    spatial dims make the engine build fail with ``setDimensions ... x >= 0``.
    Pinning H/W makes the anchor count static too — only batch stays dynamic.
    """
    inp = model.graph.input[0]
    dims = inp.type.tensor_type.shape.dim
    changed = False
    if len(dims) == 4:
        for ax in (2, 3):  # N, C, H, W → pin H, W
            d = dims[ax]
            if d.dim_value <= 0:  # dynamic (dim_param set or 0)
                d.ClearField("dim_param")
                d.dim_value = imgsz
                changed = True
    return changed


def _resolve_anchor_count(model: onnx.ModelProto, imgsz: int, n_idx: int) -> int:
    """Run the (spatially-pinned) model once on CPU to get the static anchor count."""
    import numpy as np
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(model.SerializeToString(), so,
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else (1 if i == 0 else imgsz)
             for i, d in enumerate(inp.shape)]
    out = sess.run(None, {inp.name: np.zeros(shape, np.float32)})[0]
    return int(out.shape[n_idx])


def convert(in_path: Path, out_path: Path, force: bool = False,
            imgsz: int = 640) -> tuple[int, int]:
    model = onnx.load(str(in_path))
    if _pin_input_spatial(model, imgsz):
        print(f"  pinned dynamic input H/W to {imgsz} (kept batch dynamic)")
    opset = max((imp.version for imp in model.opset_import if imp.domain in ("", "ai.onnx")),
                default=0)
    try:
        src, n_idx, c_idx, n, nc = _find_output_layout(model)
    except SystemExit:
        if not force:
            raise
        # --force: assume stock channels-first [batch, C, N].
        out = model.graph.output[0]
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        src, n_idx, c_idx, n, nc = out.name, 2, 1, dims[2], dims[1] - 4

    # nvinfer needs the anchor count static (only batch may be dynamic). If the
    # export left it symbolic, resolve it by running the spatially-pinned model.
    if not (isinstance(n, int) and n > 0):
        n = _resolve_anchor_count(model, imgsz, n_idx)
        print(f"  resolved dynamic anchor count → {n}")

    g = model.graph
    P = "ds_"  # prefix for all tensors/nodes we add, to avoid name clashes
    nodes = []

    def const(name, arr):
        t = numpy_helper.from_array(arr, P + name)
        g.initializer.append(t)
        return P + name

    import numpy as np
    half = const("half", np.array([0.5], dtype=np.float32))

    # 1. make tensor [batch, N, C]
    if c_idx == 1:  # channels-first -> transpose
        nodes.append(helper.make_node("Transpose", [src], [P + "nc"], P + "tr", perm=[0, 2, 1]))
        nc_tensor = P + "nc"
    else:
        nc_tensor = src

    def slc(name, start, end):
        s = const(name + "_s", np.array([start], dtype=np.int64))
        e = const(name + "_e", np.array([end], dtype=np.int64))
        a = const(name + "_a", np.array([2], dtype=np.int64))
        nodes.append(helper.make_node("Slice", [nc_tensor, s, e, a], [P + name], P + name + "_n"))
        return P + name

    cx, cy, w, h = slc("cx", 0, 1), slc("cy", 1, 2), slc("w", 2, 3), slc("h", 3, 4)
    classes = slc("cls", 4, 4 + nc)

    nodes += [
        helper.make_node("Mul", [w, half], [P + "hw"], P + "hw_n"),
        helper.make_node("Mul", [h, half], [P + "hh"], P + "hh_n"),
        helper.make_node("Sub", [cx, P + "hw"], [P + "x1"], P + "x1_n"),
        helper.make_node("Sub", [cy, P + "hh"], [P + "y1"], P + "y1_n"),
        helper.make_node("Add", [cx, P + "hw"], [P + "x2"], P + "x2_n"),
        helper.make_node("Add", [cy, P + "hh"], [P + "y2"], P + "y2_n"),
        helper.make_node("Concat", [P + "x1", P + "y1", P + "x2", P + "y2"],
                         [P + "xyxy"], P + "xyxy_n", axis=2),
    ]

    # score = max over classes ; label = argmax over classes
    if opset >= 18:  # ReduceMax moved axes to an input in opset 18
        axes = const("rax", np.array([2], dtype=np.int64))
        nodes.append(helper.make_node("ReduceMax", [classes, axes], [P + "score"],
                                      P + "score_n", keepdims=1))
    else:
        nodes.append(helper.make_node("ReduceMax", [classes], [P + "score"],
                                      P + "score_n", axes=[2], keepdims=1))
    nodes += [
        helper.make_node("ArgMax", [classes], [P + "lbl_i"], P + "lbl_n", axis=2, keepdims=1),
        helper.make_node("Cast", [P + "lbl_i"], [P + "lbl_f"], P + "cast_n", to=TensorProto.FLOAT),
        helper.make_node("Concat", [P + "xyxy", P + "score", P + "lbl_f"],
                         [P + "output"], P + "out_n", axis=2),
    ]

    g.node.extend(nodes)

    # Replace the graph output with our [batch, N, 6] tensor named "output".
    # Keep batch and (if it was dynamic) the anchor dim symbolic, not a fixed 0.
    del g.output[:]
    n_dim = n if (isinstance(n, int) and n > 0) else "anchors"
    g.output.append(helper.make_tensor_value_info(
        P + "output", TensorProto.FLOAT, ["batch", n_dim, 6]))
    # Rename to the conventional "output" (matches DeepStream-Yolo) without clashing.
    for node in g.node:
        node.output[:] = ["output" if o == P + "output" else o for o in node.output]
    g.output[0].name = "output"

    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    return n, nc


def validate(in_path: Path, out_path: Path, imgsz: int = 640) -> None:
    import numpy as np
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    src = ort.InferenceSession(str(in_path), so, providers=["CPUExecutionProvider"])
    dst = ort.InferenceSession(str(out_path), so, providers=["CPUExecutionProvider"])

    inp = src.get_inputs()[0]
    # Dynamic export can leave batch AND spatial dims symbolic. Use batch=1 and a
    # real 640 for any dynamic spatial dim (feeding 1 breaks the model's reshapes).
    shape = [d if isinstance(d, int) and d > 0 else (1 if i == 0 else imgsz)
             for i, d in enumerate(inp.shape)]
    x = (np.random.rand(*shape).astype(np.float32))

    raw = src.run(None, {inp.name: x})[0]          # [1, 4+nc, N] (or [1,N,4+nc])
    got = dst.run(None, {dst.get_inputs()[0].name: x})[0]  # [1, N, 6]

    # NumPy reference decode of the stock output (same math the graph now does).
    r = raw[0]
    if r.shape[0] < r.shape[1]:   # [C, N] -> [N, C]
        r = r.T
    cx, cy, w, h = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    cls = r[:, 4:]
    ref = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
                    cls.max(1), cls.argmax(1).astype(np.float32)], axis=1)

    g = got[0]
    assert g.shape == ref.shape, f"shape {g.shape} != {ref.shape}"
    # boxes + score: float-close; label: exact integer match
    if not np.allclose(g[:, :5], ref[:, :5], atol=1e-3):
        diff = np.abs(g[:, :5] - ref[:, :5]).max()
        raise SystemExit(f"VALIDATION FAILED: box/score max diff {diff:.4g}")
    if not np.array_equal(g[:, 5], ref[:, 5]):
        n_bad = int((g[:, 5] != ref[:, 5]).sum())
        raise SystemExit(f"VALIDATION FAILED: {n_bad} label mismatches")
    finite = np.isfinite(g).all()
    print(f"  validation OK: output {g.shape}, labels in "
          f"[{int(g[:,5].min())},{int(g[:,5].max())}], all-finite={finite}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="stock YOLOv8 ONNX")
    ap.add_argument("-o", "--out", type=Path, help="output path (default <in>_ds.onnx)")
    ap.add_argument("--inplace", action="store_true", help="overwrite the input file")
    ap.add_argument("--force", action="store_true",
                    help="assume stock [batch,C,N] even if layout detection bails")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="square input size to pin dynamic H/W to (default 640)")
    ap.add_argument("--no-validate", dest="validate", action="store_false",
                    help="skip the onnxruntime cross-check")
    args = ap.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"no such file: {args.input}")
    out = args.input if args.inplace else (args.out or
          args.input.with_name(args.input.stem + "_ds.onnx"))

    # When overwriting in place, convert via a temp then move, so a failure can't
    # corrupt the original.
    tmp = out.with_suffix(out.suffix + ".tmp") if args.inplace else out
    print(f"converting {args.input} -> {out}")
    n, nc = convert(args.input, tmp, force=args.force, imgsz=args.imgsz)
    print(f"  rewrote head: {n} anchors, {nc} classes -> output [batch, {n}, 6]")
    if args.validate:
        validate(args.input, tmp, imgsz=args.imgsz)
    if args.inplace:
        tmp.replace(out)
    print(f"saved {out}")
    print("  reminder: delete the cached *_gpu0_fp16.engine so nvinfer rebuilds it.")


if __name__ == "__main__":
    main()
