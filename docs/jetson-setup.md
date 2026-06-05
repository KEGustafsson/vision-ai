# Jetson Orin Nano setup & deployment

Target: **NVIDIA Jetson Orin Nano Super**, JetPack 6.x (CUDA 12.6, TensorRT 10.3).

## 1. Prerequisites on the Jetson

- JetPack 6.x flashed (L4T r36.x).
- Docker + the **NVIDIA Container Runtime** (`nvidia-docker2`); confirm with
  `docker info | grep -i runtime` showing `nvidia`.
- (Optional) set the Orin Nano to its higher-power "Super" mode:
  `sudo nvpmodel -m 0 && sudo jetson_clocks`.

## 2. Build the TensorRT engine

Ultralytics loads a `.engine` exactly like a `.pt`. Build it **on the Jetson**
(engines are device/TensorRT-specific and not portable):

```bash
cd vision-service
python3 scripts/download_models.py --model yolov8n.pt
python3 scripts/export_engine.py --weights models/yolov8n.pt --imgsz 640
# → models/yolov8n.engine  (FP16)
```

Expected throughput: YOLOv8n ≈ 15–20 FPS, YOLOv8s ≈ 8–12 FPS at 640 (FP16).
INT8 adds ~25–35% but needs a calibration dataset — see the Ultralytics TensorRT
guide. With two cameras, the plugin's context control prioritises one camera so
you don't pay full rate on both simultaneously.

## 3. Camera URLs & calibration

Set the RTSP URLs and per-camera geometry. The GStreamer pipeline uses HW decode
(`nvv4l2decoder`) — see `app/camera/rtsp_gstreamer.py`. For H.265 cameras change
the depay/parse in `build_pipeline`. Calibrate `hfov_deg`, `height_m`,
`horizon_y`, `bearing_offset_deg` per `docs/geometry.md`.

## 4. Run

```bash
export VISION_CAMERA_FORWARD_URL="rtsp://user:pass@192.168.1.10:554/stream"
export VISION_CAMERA_AFT_URL="rtsp://user:pass@192.168.1.11:554/stream"
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up -d
```

The service binds to `127.0.0.1:7000` on the boat network; the SignalK plugin
(running on the same box or reachable host) proxies the video and control behind
SignalK's authentication. Point the plugin's `containerUrl` at it.

## 5. Autostart

`restart: unless-stopped` (compose) brings the container back after reboot. Run
the SignalK server as a service (its standard systemd unit / Docker restart
policy) and enable the plugin in the SignalK admin UI.

## 6. DeepStream GPU pipeline (alternative backend)

`VISION_MODE=deepstream` runs a fully GPU-resident pipeline instead of the
Python/TensorRT one: frames stay in NVMM end to end — `nvv4l2decoder →
[nvdewarper] → nvstreammux → nvinfer (TRT) → nvtracker (NvDCF) → nvdsosd →
nvjpegenc`. Overlays are drawn on the GPU (`nvdsosd`) and the MJPEG is encoded on
the NVJPG block (`nvjpegenc`), so the CPU only handles the finished JPEG bytes —
no per-frame pixel copy (the lone exception is throttled auto-horizon detection,
~1/s, skipped when `horizon_y` is calibrated). It exposes the same API and
`DetectionEvent` contract as the default backend. Built from
`Dockerfile.deepstream`; needs JetPack 6 + **DeepStream 7.x** on the host.

```bash
# 1. Build the deepstream-yolo nvinfer parser ON THE HOST (the DS samples image
#    has no nvcc). Needs host nvcc + the DS 7.1 SDK at the same versions.
vision-service/scripts/build_yolo_parser.sh
#    → vision-service/deepstream/libnvdsinfer_custom_impl_Yolo.so

# 2. Export a raw (no end-to-end NMS) YOLOv8n ONNX — the build's export stage
#    does this automatically; see config/deepstream.yaml for the manual command.

# 3. Run (nvinfer auto-builds the TRT engine next to the ONNX in the bind-mounted
#    deepstream/ on first start, so it persists across container recreates).
export VISION_CAMERA_FORWARD_URL="rtsp://user:pass@192.168.1.10:554/stream"
export VISION_CAMERA_AFT_URL="rtsp://user:pass@192.168.1.11:554/stream"
docker compose -f docker-compose.yml -f docker-compose.deepstream.yml up -d
```

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

## Detection model selection

Exactly **one** detection model runs at a time — the two are never active
together. Pick it with `detector.model` in `config/deepstream.yaml` (or the
`VISION_DETECTOR_MODEL` env var) and restart:

| `detector.model` | nvinfer config | classes |
|---|---|---|
| `coco` (default) | `deepstream/pgie_yolov8n.txt` | 80 COCO (person, vessel, buoy, …) |
| `forward-watch` | `deepstream/pgie_forward_watch.txt` | 6 marine (ship, boat, debris, buoy, kayak, log) |
| `marine-surveillance` | `deepstream/pgie_marine_surveillance.txt` | 7 marine (boat, buoy, kayak, sailboat, speedboat, vessel, warship) |

The two models share the same YOLOv8n architecture, 640×640 input, and the same
deepstream-yolo custom parser — only the ONNX, label file, and
`num-detected-classes` differ, so switching is purely a config change.

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
> it. COCO needs no conversion — its build-time `export_yoloV8.py` already emits
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

- **OpenCV has no GStreamer** → use the Ultralytics Jetson base image; do not
  `pip install opencv-python` over it (that shadows the CUDA/GStreamer build).
- **`nvv4l2decoder` not found** → run with the NVIDIA runtime; verify L4T
  multimedia packages are present.
- **Low FPS** → confirm Super mode (`nvpmodel`/`jetson_clocks`), use the
  `.engine` (not `.pt`), keep `imgsz: 640`, and let context control limit the
  inactive camera.
- **DeepStream: `nvbufsurface: Unable to allocate HW buffer` (error 12)** →
  NVMM is fragmented/saturated (`tegrastats` shows a tiny `lfb`). Stop other NVMM
  tenants (e.g. GStreamer overlay containers) to defragment, start vision so it
  allocates its pools, then restart the others — the running pipeline survives
  them returning. Same condition can stall the first-run TRT engine build.
- **DeepStream: empty target list / video** in the Captain view after switching
  backend → the SignalK plugin drops events failing schema validation; ensure its
  installed `schema/detection-event.schema.json` includes `deepstream` in the
  `Backend` enum and restart the plugin.
