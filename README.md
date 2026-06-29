# Marine Vision-AI 🛥️📡

A **"visual radar"** for boats: two cameras (forward + aft) feeding a YOLOv8
detector on an **NVIDIA Jetson Orin Nano Super**, turned into georeferenced,
AIS-fused, navigation-aware situational awareness inside **[SignalK](https://signalk.org)**.

## What it does

The system doesn't just draw boxes on a video feed. It treats each camera as a
bearing/range sensor and, using the boat's own position and heading from
SignalK, produces:

- 🎯 **Visual radar targets** — every detection becomes a true-bearing + range
  track, georeferenced to a lat/lon and published as a synthetic AIS vessel
  (`vessels.*` chart blip) when enabled.
- 🕶️ **Dark-target alerts** — visual targets are correlated against AIS; a
  vessel that is *seen but not transmitting AIS* is flagged as a collision
  hazard (`notifications.vision.darkTarget.*`).
- 🆘 **Man-overboard detection** — a person detected in the water raises an
  `notifications.mob` **emergency** with the estimated drop position.
- ⚠️ **Collision risk** — per-track CPA/TCPA with graded
  `notifications.vision.collision.*` alarms.
- 📺 **Captain webapp** — an annotated live MJPEG stream with a colour-coded
  target list and own-ship readout, embedded in the SignalK UI.
- 🧭 **Context-aware control** — the plugin steers the container (active camera,
  confidence, day/night) based on the boat's speed and time of day.

## Demo

Running on a Jetson Orin Nano Super — 2× cameras (bow + aft), 10 fps each, the
DeepStream backend.

**Captain webapp** — annotated live stream embedded in the SignalK UI:

![Captain webapp: annotated live camera stream with detection boxes, a colour-coded target list, and own-ship readout.](./docs/images/Marine_Vision-AI.png)

**Synthetic AIS vessels on the chart** — each detection georeferenced as a blip:

![Chart view: camera-detected targets published as synthetic AIS vessels and drawn as blips on the map.](./docs/images/synthetic_vessels.png)

**Jetson load** (`jtop`):

![NVIDIA Jetson jtop view showing GPU, CPU, and memory utilisation while the pipeline runs.](./docs/images/nVidia_Nano_jtop.png)

## How it works

### Architecture

![System architecture: two cameras feed the vision-service container over RTSP; it emits DetectionEvent JSON over WebSocket (plus MJPEG video and a REST control channel) to the signalk-vision-ai plugin, which publishes vision.* deltas and notifications to the SignalK server for the MFD/chart and Captain view.](docs/images/architecture-overview.svg)

Two processes, one contract. The container owns the GPU/pixels/geometry and
emits a single JSON event schema (`docs/event-schema.md`). The plugin owns all
navigation-relative math because only it has live SignalK state. The boundary
schema is generated from the container's Pydantic models, so the two sides can't
drift.

| Component | Path | Stack |
|-----------|------|-------|
| Vision service | [`vision-service/`](vision-service/) | Python, FastAPI, Ultralytics YOLOv8, OpenCV, TensorRT or DeepStream (Jetson) |
| SignalK plugin | [`signalk-plugin/`](signalk-plugin/) | TypeScript, ws, ajv |

### From pixels to targets

Every detection runs the same pipeline: the camera frame is decoded and a
YOLOv8 + ByteTrack pass yields a tracked bounding box; the container turns that
box's **pixel column** into a relative **bearing** and its **waterline row**
into a **range** (monocular geometry — see [Geometry & calibration](docs/geometry.md));
the plugin then fuses in the boat's own heading and position to georeference the
target, correlates it against AIS, and raises any MOB / collision / dark-target
alerts.

![The full detection process: the container decodes a camera frame, detects and tracks objects with YOLOv8 + ByteTrack, computes a relative bearing from the pixel column and a range from the horizon depression or known size, then filters and emits a DetectionEvent; the plugin enriches it to a true bearing and lat/lon, fuses it with AIS, estimates CPA/TCPA, raises notifications, and publishes synthetic AIS vessels and vision.* paths to SignalK.](docs/images/detection-process.svg)

## Quick start (no GPU, no cameras)

Everything runs in a **mock mode** with synthetic frames and a deterministic
detector, so you can develop and demo on a laptop.

```bash
# 1. Vision container (synthetic forward + aft scenes, MOB on aft)
cd vision-service
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # or just the light deps for mock
VISION_MODE=mock python -m uvicorn app.main:app --port 7000

# open http://localhost:7000/stream/forward.mjpg  → annotated video
# open http://localhost:7000/stream/aft.mjpg       → includes a person-in-water

# 2. SignalK plugin
cd ../signalk-plugin
npm install && npm run build && npm test
```

Then install the plugin into a SignalK server and point it at the container —
see [`docs/dev-quickstart.md`](docs/dev-quickstart.md). Or skip the manual setup
and bring the whole stack up with Docker — see the `mock` mode under
[Deployment modes](#deployment-modes) below.

## Deployment modes

Three run modes, selected by `VISION_MODE`; all emit the same `DetectionEvent`.
Each has a **self-contained compose file** (a single `-f`, no base needed).
`deepstream` builds and runs from a clean clone in one command — no prior image
build, no manual model step. `jetson` needs one device-specific step first: build
the TensorRT engine on the board (engines aren't portable), then run.

| Mode | Where | Command |
|------|-------|---------|
| **`mock`** | any laptop, no GPU/cameras | `docker compose up` |
| **`jetson`** | Jetson, TensorRT | build the engine once (below), then `docker compose -f docker-compose.jetson.yml up -d` |
| **`deepstream`** | Jetson, full-GPU pipeline | `docker compose -f docker-compose.deepstream.yml up -d --build` |

Set the camera URLs first for the GPU modes (`.env` or exported):

```bash
export VISION_CAMERA_FORWARD_URL="rtsp://user:pass@192.168.1.10:554/stream"
export VISION_CAMERA_AFT_URL="rtsp://user:pass@192.168.1.11:554/stream"
```

### `mock` — laptop / dev (no GPU, no cameras)

`docker compose up` runs the vision service alone on synthetic frames. For the
full stack (vision service **+** a SignalK server with demo data + the plugin),
layer the dev override — this is the only mode that needs two `-f` files:

```bash
docker compose -f docker-compose.yml -f docker-compose.mock.yml up
# SignalK:  http://localhost:3000     Captain view: http://localhost:3000/signalk-vision-ai/
```

### `jetson` — Ultralytics YOLOv8 on TensorRT

Decode in a GStreamer pipeline, inference + ByteTrack in Python. Optional
`server.hw_jpeg` offloads the MJPEG encode to the Jetson NVJPG block
(`nvjpegenc`) instead of CPU `cv2.imencode`. TensorRT engines are
device-specific, so build the engine **on the Jetson** once before running:

```bash
cd vision-service
python3 scripts/download_models.py --model yolov8n.pt
python3 scripts/export_engine.py --weights models/yolov8n.pt --imgsz 640  # → models/yolov8n.engine
cd ..
docker compose -f docker-compose.jetson.yml up -d
```

### `deepstream` — fully GPU-resident NVIDIA DeepStream

The most efficient backend — every stage runs on the GPU (see
[DeepStream architecture](#deepstream-architecture) for the design). The image
builds from a clean clone in one command (it exports the ONNX and bakes the
committed parser itself; nvinfer builds the TensorRT engine on first start):

```bash
docker compose -f docker-compose.deepstream.yml up -d --build
```

For a reproducible / air-gapped build, pin the export base to a digest by
exporting `VISION_JETSON_BASE` before the command (it feeds the `EXPORT_BASE`
build arg — resolve the aarch64 digest on the board):

```bash
export VISION_JETSON_BASE="ultralytics/ultralytics@sha256:<digest>"
docker compose -f docker-compose.deepstream.yml up -d --build
```

See [Jetson setup & deployment](docs/jetson-setup.md) for prerequisites,
calibration, model selection, and tuning.

## DeepStream architecture

On a Jetson the `deepstream` backend replaces the Python decode/inference/track
loop with a **single GStreamer graph that lives entirely in GPU memory (NVMM)**.
Both cameras are batched once and run through inference and tracking in one pass;
the CPU never touches a pixel — it only ever receives the finished JPEG bytes.

![DeepStream GStreamer pipeline, fully GPU-resident in NVMM: rtspsrc → nvv4l2decoder → optional nvdewarper → nvstreammux (batches both cameras) → nvinfer (TensorRT) → nvtracker (NvDCF); then per camera after nvstreamdemux: nvvideoconvert (RGBA) → a pad probe that reads metadata only → nvstreamdemux → nvdsosd (GPU overlay) → nvjpegenc (I420 to JPEG) → appsink → LatestFrame. Detection metadata also feeds the bearing/range geometry into the DetectionEvent without copying any pixels.](docs/images/deepstream-pipeline.svg)

**One batched front end, two zero-copy outputs.** `nvstreammux` batches both
camera streams so `nvinfer` (TensorRT) and `nvtracker` (NvDCF) run on the batch
in a single pass at the camera's native resolution (inference rescales to `imgsz`
internally on the GPU). A pad probe then reads the detection metadata **without
copying pixels** and the graph forks:

- **Geometry / event path** — the probe turns each track's box into a
  bearing/range and emits the `DetectionEvent` (host side, metadata only). This
  is the same contract every other backend produces, so the plugin is unchanged.
- **Display path** — the probe attaches GPU overlay metadata; a per-camera
  `nvdsosd` burns the boxes/labels/HUD onto the NVMM surface and `nvjpegenc`
  encodes the JPEG on the NVJPG block, delivered via `appsink` → `LatestFrame`
  for MJPEG.

**GPU lens correction** (`nvdewarper`, optional) applies barrel + rotation before
inference. The lone host-side pixel access is auto-horizon detection, throttled to
~1/s per camera and skipped entirely when a camera has an explicit `horizon_y`
calibration. A per-camera PTS guard in the probe drops `nvstreammux` frame repeats
so output never exceeds the camera's input rate.

See [Architecture & data flow](docs/architecture.md#inference-backends) for how
this fits the wider system and [Jetson setup & deployment](docs/jetson-setup.md)
for the build, model selection, and tuning.

## Documentation

- [Architecture & data flow](docs/architecture.md)
- [Detection event contract](docs/event-schema.md)
- [SignalK `vision.*` paths](docs/signalk-paths.md)
- [Geometry & calibration](docs/geometry.md)
- [Jetson setup & deployment](docs/jetson-setup.md)
- [Dev quickstart (end-to-end)](docs/dev-quickstart.md)
- [Onboard verification runbook](docs/onboard-verification.md)

## Tests

```bash
cd vision-service && pytest          # geometry, schema, synthetic pipeline
cd signalk-plugin && npm test        # geo, enrich, AIS fusion, CPA, publisher
```

## Safety note

Vision-derived alerts are an *aid*, not a replacement for proper lookout, radar
or AIS. Range is monocular and coarse, and auto-MOB is a best-effort guess —
all alerting features are configurable and the synthetic-AIS-blip projection
("Publish targets as synthetic AIS vessels" / `enableVisualRadar`) is **off by
default** to avoid being confused with real AIS contacts.
