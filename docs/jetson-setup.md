# Jetson Orin Nano setup & deployment

Target: **NVIDIA Jetson Orin Nano Super**, JetPack 6.x (CUDA 12.6, TensorRT 10.3).

This covers the **`jetson` backend** (Ultralytics YOLO11 on TensorRT, Python
decode/inference/track loop). There's also a **`deepstream` backend** — a
different pipeline entirely (different tracker, image, compose file) —
covered in [DeepStream GPU pipeline](jetson-deepstream.md). It shares this
doc's prerequisites, camera calibration, and autostart steps (§1, §3, §5
below); only §2 and §4 are `jetson`-backend-specific.

The `deepstream` backend additionally runs on **JetPack 5** boards (Jetson
Xavier NX / AGX Xavier) from a second compose file; see
[Hardware targets](jetson-deepstream.md#hardware-targets). Everything in this
document assumes the Orin Nano / JetPack 6 target unless stated otherwise.

## 1. Prerequisites on the Jetson

- JetPack 6.x flashed (L4T r36.x). *(For the `deepstream` backend on a Xavier,
  JetPack 5.1.x / L4T r35.x — see
  [Xavier NX notes](jetson-deepstream.md#xavier-nx-notes).)*
- Docker + the **NVIDIA Container Runtime** (`nvidia-docker2`); confirm with
  `docker info | grep -i runtime` showing `nvidia`.
- (Optional) set the Orin Nano to its higher-power "Super" mode:
  `sudo nvpmodel -m 0 && sudo jetson_clocks`. On a Xavier NX the equivalent is
  `sudo nvpmodel -m 8 && sudo jetson_clocks` (20 W, 6 cores).

## 2. Build the TensorRT engine

Ultralytics loads a `.engine` exactly like a `.pt`. Build it **on the Jetson**
(engines are device/TensorRT-specific and not portable):

```bash
cd vision-service
python3 scripts/download_models.py --model yolo11n.pt
python3 scripts/export_engine.py --weights models/yolo11n.pt --imgsz 640
# → models/yolo11n.engine  (FP16)
```

Expected throughput: YOLO11n ≈ 15–20 FPS, YOLO11s ≈ 8–12 FPS at 640 (FP16) —
similar ballpark to YOLOv8n/s at the same input size; re-measure on your board
before relying on these. INT8 adds ~25–35% but needs a calibration dataset —
see the Ultralytics TensorRT guide. With two cameras, the plugin's context
control prioritises one camera so you don't pay full rate on both
simultaneously.

## 3. Camera URLs & calibration

Set the RTSP URLs and per-camera geometry. The GStreamer pipeline uses HW decode
(`nvv4l2decoder`) — see `app/camera/rtsp_gstreamer.py`. For H.265 cameras change
the depay/parse in `build_pipeline`. Calibrate `hfov_deg`, `height_m`,
`horizon_y`, `bearing_offset_deg` per `docs/geometry.md`.

## 4. Run

```bash
export VISION_CAMERA_FORWARD_URL="rtsp://user:pass@192.168.1.10:554/stream"
export VISION_CAMERA_AFT_URL="rtsp://user:pass@192.168.1.11:554/stream"
docker compose -f docker-compose.jetson.yml up -d
```

The service binds to `127.0.0.1:7000` on the boat network; the SignalK plugin
(running on the same box or reachable host) proxies the video and control behind
SignalK's authentication. Point the plugin's `containerUrl` at it.

## 5. Autostart

`restart: unless-stopped` (compose) brings the container back after reboot. Run
the SignalK server as a service (its standard systemd unit / Docker restart
policy) and enable the plugin in the SignalK admin UI.

## DeepStream backend (alternative)

For the fully GPU-resident pipeline — same model generation (YOLO11n) but a
different tracker (NvDCF vs BoT-SORT), different image and compose file — see
[DeepStream GPU pipeline](jetson-deepstream.md). It covers that backend's
build/run, tuning, runtime behaviour, detection model selection, and
troubleshooting.

## Troubleshooting

- **OpenCV has no GStreamer** → use the Ultralytics Jetson base image; do not
  `pip install opencv-python` over it (that shadows the CUDA/GStreamer build).
- **`nvv4l2decoder` not found** → run with the NVIDIA runtime; verify L4T
  multimedia packages are present.
- **Low FPS** → confirm Super mode (`nvpmodel`/`jetson_clocks`), use the
  `.engine` (not `.pt`), keep `imgsz: 640`, and let context control limit the
  inactive camera.

See also [DeepStream GPU pipeline](jetson-deepstream.md#troubleshooting) for
issues specific to that backend.
