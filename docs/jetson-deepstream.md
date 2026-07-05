# DeepStream GPU pipeline (Jetson)

The `deepstream` backend — a fully GPU-resident alternative to the Ultralytics/
TensorRT `jetson` backend covered in
[Jetson setup & deployment](jetson-setup.md). It's a **different pipeline**,
not a tuning variant: both default to the same model generation (YOLO11n), but
a different tracker (NvDCF vs BoT-SORT), different image
(`Dockerfile.deepstream` vs `Dockerfile`), different compose file. It shares
only the basics with the
Ultralytics path — see [Jetson setup & deployment](jetson-setup.md) for:

- [§1 Prerequisites](jetson-setup.md#1-prerequisites-on-the-jetson) (JetPack,
  NVIDIA Container Runtime, Super mode) — plus **DeepStream 7.x** on the host
  for this backend (needs JetPack 6.1/6.2; see `Dockerfile.deepstream`'s
  `DS_BASE`).
- [§3 Camera URLs & calibration](jetson-setup.md#3-camera-urls--calibration).
- [§5 Autostart](jetson-setup.md#5-autostart).

Everything below is specific to this backend.

`VISION_MODE=deepstream` runs a fully GPU-resident pipeline instead of the
Python/TensorRT one: frames stay in NVMM end to end.

![DeepStream GStreamer pipeline, fully GPU-resident in NVMM: rtspsrc → nvv4l2decoder → optional nvdewarper → nvstreammux (batches both cameras) → nvinfer (TensorRT) → nvtracker (NvDCF); then per camera after nvstreamdemux: nvvideoconvert (RGBA) → a pad probe that reads metadata only → nvstreamdemux → nvdsosd (GPU overlay) → nvjpegenc (I420 to JPEG) → appsink → LatestFrame. Detection metadata also feeds the bearing/range geometry into the DetectionEvent without copying any pixels.](images/deepstream-pipeline.svg)

Overlays are drawn on the GPU (`nvdsosd`) and the MJPEG is encoded on
the NVJPG block (`nvjpegenc`), so the CPU only handles the finished JPEG bytes —
no per-frame pixel copy (the lone exception is throttled auto-horizon detection,
~1/s, skipped when `horizon_y` is calibrated). It exposes the same API and
`DetectionEvent` contract as the default backend. Built from
`Dockerfile.deepstream`; needs JetPack 6 + **DeepStream 7.x** on the host.

## Run

The image builds **from a clean clone in a single command** — no prior
`vision-ai:jetson` build, no manual ONNX export, no parser-compile step:

```bash
export VISION_CAMERA_FORWARD_URL="rtsp://user:pass@192.168.1.10:554/stream"
export VISION_CAMERA_AFT_URL="rtsp://user:pass@192.168.1.11:554/stream"
docker compose -f docker-compose.deepstream.yml up -d --build
```

For a reproducible / air-gapped build, pin the export base to a digest by
exporting `VISION_JETSON_BASE` before the command (it feeds the `EXPORT_BASE`
build arg — resolve the aarch64 digest on the board):

```bash
export VISION_JETSON_BASE="ultralytics/ultralytics@sha256:<digest>"
docker compose -f docker-compose.deepstream.yml up -d --build
```

What the single build does for you:

- **ONNX export** — the build's stage-1 `export` runs on the public Ultralytics
  Jetson base (the same base the jetson path uses, so it's already cached if you
  built that image first; otherwise it's pulled fresh) and fetches the YOLO11n
  weights itself. To build OFFLINE, drop the weights at
  `vision-service/models/yolo11n.pt` first (`python3 scripts/download_models.py
  --model yolo11n.pt`) and the build uses them as-is.
- **Custom parser** — `deepstream/libnvdsinfer_custom_impl_Yolo.so` is committed
  and baked in, so no compile step is required. Rebuild it only when the
  DeepStream version changes: `vision-service/scripts/build_yolo_parser.sh` (needs
  host nvcc + the DS 7.1 SDK at the same versions; the DS samples image has none).
- **TRT engine** — nvinfer auto-builds it next to the ONNX in the bind-mounted
  `deepstream/` on first start, so it persists across container recreates.

## Tuning

Tuning lives in `config/deepstream.yaml` (bind-mounted, takes effect on restart):

- **`detector.mux_width` / `mux_height`** — nvstreammux output = display +
  geometry resolution (native camera res, default `1280x960`). nvinfer rescales
  this to `imgsz` internally on the GPU, so detection still runs at `imgsz` while
  bboxes and `horizon_y` stay in native pixels. Set these to your stream size.
- **GPU lens correction** — when a camera sets `undistort: true`, an `nvdewarper`
  stage straightens barrel (`undistort_k1`) + rotation (`undistort_rotation_deg`)
  in NVMM before inference. A starter config is generated from those knobs;
  override with `camera.dewarper_config` once tuned on the hardware.
- **`detector.mux_batch_timeout_ms`** — nvstreammux batch timeout (default 40 ms).
  Kept short so both cameras pipeline at full input rate; a per-camera PTS guard
  in the probe drops any muxer frame repeats so output never exceeds the camera's
  real frame rate.

The deepstream compose bind-mounts `app/`, `config/`, `deepstream/`, and `models/`,
so code, config, model ONNX/labels, and the built TRT engine all live on the host
and survive recreates — a plain **restart** picks up app/config edits and an
engine rebuild only happens when the ONNX actually changes. An image rebuild is
only needed to bake changes for a clean redeploy. ⚠ The Orin is NVMM-tight: stop
the GPU overlay co-tenants before recreating the container, or buffer-pool
allocation can OOM (`failed to activate bufferpool`); restart them after.

## Runtime behaviour

- **Disable detection (master off):** unlike the CPU/Jetson backend (whose workers
  release the camera capture device), DeepStream transitions the whole GStreamer
  graph to **PAUSED** — decoders and `nvinfer` stop pulling data, so disabling
  actually drops the GPU/thermal load. Re-enabling returns it to PLAYING. A
  pipeline that recovers from a fault while disabled comes back PAUSED.
- **Auto-recovery:** a fatal GStreamer error or EOS (e.g. a transient RTSP/decoder
  glitch) no longer takes detection down until a manual container restart. A
  supervisor rebuilds the pipeline with exponential backoff (2 s → 30 s) and keeps
  retrying. `GET /health` reports `pipeline_restarts` and `pipeline_last_error` so
  a flapping feed is visible; `status` goes `degraded` while restarts have occurred.
- **Non-root:** the runtime image runs as a non-root user (UID 10001, in the
  `video` group for GPU access). nvinfer writes the TRT engine into the
  bind-mounted `deepstream/` and `models/`, so those host dirs must be writable by
  UID 10001 (as the jetson image already requires for `models/`). Run
  `vision-service/scripts/fix_host_permissions.sh` once per host to grant this via
  ACL (`setfacl`) without changing ownership or opening the dirs to everyone. A
  container stuck restart-looping with `WARN could not write .../yolo11n_ds.onnx`
  or `ERROR: Cannot access ONNX file` in its logs means this hasn't been run (or
  the dirs were recreated since). If your nvidia container runtime denies GPU
  access to non-root, set `user: root` on the deepstream compose service as a
  documented exception instead.

## Detection model selection

Exactly **one** detection model runs at a time — the two are never active
together. Pick it with `detector.model` in `config/deepstream.yaml` (or the
`VISION_DETECTOR_MODEL` env var) and restart:

| `detector.model` | nvinfer config | classes |
|---|---|---|
| `coco` (default) | `deepstream/pgie_yolo11n.txt` | 80 COCO (person, vessel, buoy, …) |
| `forward-watch` | `deepstream/pgie_forward_watch.txt` | 6 marine (ship, boat, debris, buoy, kayak, log) |
| `marine-surveillance` | `deepstream/pgie_marine_surveillance.txt` | 7 marine (boat, buoy, kayak, sailboat, speedboat, vessel, warship) |

All models use a deepstream-yolo-compatible YOLO ONNX and the SAME custom parser
(YOLO11 shares YOLOv8's output layout) — only the ONNX weights, input size, label
file, and `num-detected-classes` differ, so switching is purely a config change.
(`coco` is YOLO11n at 768×768; the marine models are YOLOv8n/s at 640×640.)

`coco` keeps person/man-overboard detection; `forward-watch` drops it but adds
debris/kayak/log. The `forward-watch.onnx` is **not** vendored — fetch it before
building the image so `COPY deepstream` bakes it in:

```bash
python3 scripts/download_forward_watch.py     # downloads AND converts → deepstream/forward-watch.onnx
```

> **The published forward-watch ONNX is a stock Ultralytics export and will not
> work with our parser as-is** — its output is `[1, 4+nc, 8400]`, but
> `NvDsInferParseYolo` expects `[1, 8400, 6]` (`x1,y1,x2,y2,score,class`), so fed
> raw it produces **zero detections**. The download script above automatically
> rewrites it via `scripts/convert_to_deepstream.py` (needs
> `pip install onnx onnxruntime`). To convert an ONNX you already have:
> `python3 scripts/convert_to_deepstream.py forward-watch.onnx --inplace`. After
> replacing the ONNX, delete any cached `*_gpu0_fp16.engine` so nvinfer rebuilds
> it. COCO needs no conversion — its build-time `export_yolo11.py` already emits
> the parser layout.

### marine-surveillance (train on-box)

Roboflow only exports the *dataset*, so this model is trained on the Jetson and
exported to a parser-ready ONNX by `training/train_marine_surveillance.py` (it does
a stock export then runs `vision-service/scripts/convert_to_deepstream.py`). The
dataset and runs land under `training/_train_marine_surveillance/`. Run it inside
an Ultralytics Jetson container — mount the repo root (the script reaches into
`vision-service/`) — with the GPU co-tenants stopped so it doesn't OOM:

```bash
docker stop vision-service-deepstream gstreamer_in_overlay gstreamer_out_overlay
docker run --rm -it --runtime nvidia --network host \
  -v $PWD:/work -w /work \
  ultralytics/ultralytics:latest-jetson-jetpack6 \
  bash -lc "pip install -q roboflow onnx onnxslim onnxruntime && \
    python3 training/train_marine_surveillance.py \
      --api-key \$ROBOFLOW_KEY --workspace WS --project PROJ --version N --batch 8"
```

It writes `vision-service/deepstream/marine-surveillance.onnx` and regenerates the label file.
Because `deepstream/` is bind-mounted (see the compose volumes), the new ONNX is
already visible to the container — just set `detector.model: marine-surveillance`
in `config/deepstream.yaml`, delete any stale
`marine-surveillance.onnx_b*_gpu0_fp16.engine`, and **restart** the container (no
image rebuild needed; nvinfer rebuilds the engine on first start). Rebuild the
image only when you want it baked in for a clean redeploy. **No person class →
man-overboard detection is off while this model is active**; keep `coco` if MOB
matters.

## Troubleshooting

- **`nvbufsurface: Unable to allocate HW buffer` (error 12)** →
  NVMM is fragmented/saturated (`tegrastats` shows a tiny `lfb`). Stop other NVMM
  tenants (e.g. GStreamer overlay containers) to defragment, start vision so it
  allocates its pools, then restart the others — the running pipeline survives
  them returning. Same condition can stall the first-run TRT engine build.
- **Empty target list / video** in the Captain view after switching
  backend → the SignalK plugin drops events failing schema validation; ensure its
  installed `schema/detection-event.schema.json` includes `deepstream` in the
  `Backend` enum and restart the plugin.
- **Container restart-looping on `WARN could not write .../yolo11n_ds.onnx`** →
  see the Non-root note above; run `vision-service/scripts/fix_host_permissions.sh`.

See also [Jetson setup & deployment](jetson-setup.md#troubleshooting) for issues
shared with the Ultralytics backend (OpenCV/GStreamer, `nvv4l2decoder`, low FPS).
