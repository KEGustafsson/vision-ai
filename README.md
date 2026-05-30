# Marine Vision-AI 🛥️📡

A **"visual radar"** for boats: two cameras (forward + aft) feeding a YOLOv8
detector on an **NVIDIA Jetson Orin Nano Super**, turned into georeferenced,
AIS-fused, navigation-aware situational awareness inside **[SignalK](https://signalk.org)**.

The system doesn't just draw boxes on a video feed. It treats each camera as a
bearing/range sensor and, using the boat's own position and heading from
SignalK, produces:

- 🎯 **Visual radar targets** — every detection becomes a true-bearing + range
  track, georeferenced to a lat/lon and published under `vision.targets.*`.
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
  ┌─────────────┐  RTSP   ┌──────────────────────────┐
  │ fwd camera  ├────────►│  vision-service (Python)  │
  ├─────────────┤  RTSP   │  YOLOv8 + ByteTrack        │
  │ aft camera  ├────────►│  + monocular geometry      │
  └─────────────┘         │  (bearing / range)         │
                          └───────────┬──────────────┬─┘
            WebSocket events          │  MJPEG       │ REST control
            (DetectionEvent JSON)     ▼              ▼
                          ┌────────────────────────────────┐
                          │  signalk-vision-ai plugin (TS)  │
                          │  enrich · AIS fusion · CPA/TCPA  │
                          │  notifications · publisher       │
                          └───────────┬──────────────┬─────┘
                  vision.* deltas +   │              │  webapp + MJPEG proxy
                  notifications.*     ▼              ▼
                          ┌────────────────────────────────┐
                          │  SignalK server  →  MFD / chart │
                          │                  →  Captain view │
                          └────────────────────────────────┘
```

Two processes, one contract. The container owns the GPU/pixels/geometry and
emits a single JSON event schema (`docs/event-schema.md`). The plugin owns all
navigation-relative math because only it has live SignalK state. The boundary
schema is generated from the container's Pydantic models, so the two sides can't
drift.

| Component | Path | Stack |
|-----------|------|-------|
| Vision service | [`vision-service/`](vision-service/) | Python, FastAPI, Ultralytics YOLOv8, OpenCV, TensorRT (Jetson) |
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
see [`docs/dev-quickstart.md`](docs/dev-quickstart.md). Or bring the whole stack
up with Docker:

```bash
docker compose -f docker-compose.yml -f docker-compose.mock.yml up
# SignalK:  http://localhost:3000     Captain view: http://localhost:3000/signalk-vision-ai/
```

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
all alerting features are configurable and the synthetic-AIS-blip projection is
**off by default** to avoid being confused with real AIS contacts.
