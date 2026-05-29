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

The service binds to `127.0.0.1:8000` on the boat network; the SignalK plugin
(running on the same box or reachable host) proxies the video and control behind
SignalK's authentication. Point the plugin's `containerUrl` at it.

## 5. Autostart

`restart: unless-stopped` (compose) brings the container back after reboot. Run
the SignalK server as a service (its standard systemd unit / Docker restart
policy) and enable the plugin in the SignalK admin UI.

## Troubleshooting

- **OpenCV has no GStreamer** → use the Ultralytics Jetson base image; do not
  `pip install opencv-python` over it (that shadows the CUDA/GStreamer build).
- **`nvv4l2decoder` not found** → run with the NVIDIA runtime; verify L4T
  multimedia packages are present.
- **Low FPS** → confirm Super mode (`nvpmodel`/`jetson_clocks`), use the
  `.engine` (not `.pt`), keep `imgsz: 640`, and let context control limit the
  inactive camera.
