# Marine Vision-AI 🛥️📡

A **"visual radar"** for boats: two cameras (forward + aft) feeding a YOLOv8
detector on an **NVIDIA Jetson Orin Nano Super**, turned into georeferenced,
AIS-fused, navigation-aware situational awareness inside **[SignalK](https://signalk.org)**.

Example: 2x Cams (bow, aft), 10fps each, DeepStream

WebUI
![image](./docs/Marine_Vision-AI.png)

Camera detected synthetic vessels on map
![image](./docs/synthetic_vessels.png)

nVidia jtop process views:
![image](./docs/nVidia_Nano_jtop.png)

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

## Architecture

```text
  ┌─────────────┐  RTSP   ┌─────────────────────────────────┐
  │ fwd camera  ├────────►│  vision-service (Python)        │
  ├─────────────┤  RTSP   │  YOLOv8 + ByteTrack             │
  │ aft camera  ├────────►│  + monocular geometry           │
  └─────────────┘         │  (bearing / range)              │
                          └───────────┬──────────────┬──────┘
            WebSocket events          │  MJPEG       │ REST control
            (DetectionEvent JSON)     ▼              ▼
                          ┌─────────────────────────────────┐
                          │  signalk-vision-ai plugin (TS)  │
                          │  enrich · AIS fusion · CPA/TCPA │
                          │  notifications · publisher      │
                          └───────────┬──────────────┬──────┘
                  vision.* deltas +   │              │  webapp + MJPEG proxy
                  notifications.*     ▼              ▼
                          ┌──────────────────────────────────┐
                          │  SignalK server  →  MFD / chart  │
                          │                  →  Captain view │
                          └──────────────────────────────────┘
```

Two processes, one contract. The container owns the GPU/pixels/geometry and
emits a single JSON event schema (`docs/event-schema.md`). The plugin owns all
navigation-relative math because only it has live SignalK state. The boundary
schema is generated from the container's Pydantic models, so the two sides can't
drift.

| Component | Path | Stack |
|-----------|------|-------|
| Vision service | [`vision-service/`](vision-service/) | Python, FastAPI, Ultralytics YOLOv8, OpenCV, TensorRT or DeepStream (Jetson) |
| SignalK plugin | [`signalk-plugin/`](signalk-plugin/) | TypeScript, ws, ajv |

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

**End-to-end zero-copy in NVMM** — decode → inference → tracking → GPU overlay →
hardware JPEG (nvv4l2decoder → nvstreammux → nvinfer → nvtracker → nvdsosd →
nvjpegenc), with optional GPU lens correction (nvdewarper). The annotated MJPEG
is drawn on the GPU by `nvdsosd` and encoded on the NVJPG block by `nvjpegenc`,
so the CPU never touches a pixel — only the finished JPEG bytes. The image builds
from scratch in one command (it exports the ONNX and bakes the committed parser
itself; nvinfer builds the TensorRT engine on first start):

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
