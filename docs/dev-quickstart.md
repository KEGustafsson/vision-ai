# Dev quickstart — full stack on a laptop

No GPU or cameras required. You'll run the vision container in mock mode, a
SignalK server, and the plugin, then watch detections flow end-to-end.

## Option A — Docker (simplest)

```bash
cd signalk-plugin && npm install && npm run build && cd ..
docker compose -f docker-compose.yml -f docker-compose.mock.yml up
```

- Vision container: http://localhost:8000/stream/forward.mjpg
- SignalK server: http://localhost:3000 (enable the **Marine Vision-AI** plugin
  under *Server → Plugin Config*, set `containerUrl` to `http://vision-service:8000`)
- Captain view: http://localhost:3000/signalk-vision-ai/

## Option B — local processes

### 1. Vision container

```bash
cd vision-service
python3 -m venv .venv && . .venv/bin/activate
pip install fastapi "uvicorn[standard]" pydantic numpy PyYAML opencv-python-headless websockets
VISION_MODE=mock python -m uvicorn app.main:app --port 8000
```

Sanity checks:

```bash
curl localhost:8000/health
curl "localhost:8000/events/recent?n=1" | jq '.[0].targets[].label'
# open localhost:8000/stream/aft.mjpg  → note the red "MOB!" box (person in water)
```

### 2. SignalK server + plugin

```bash
npm install -g signalk-server        # or use the Docker image
cd signalk-plugin && npm install && npm run build
# Make the plugin discoverable:
mkdir -p ~/.signalk/node_modules
ln -s "$PWD" ~/.signalk/node_modules/signalk-vision-ai
signalk-server
```

In the SignalK admin UI (http://localhost:3000):
1. *Server → Plugin Config → Marine Vision-AI* → enable, set
   `containerUrl = http://localhost:8000`, save.
2. *Data Browser* → you should see `vision.targets.*`, `vision.system.inferenceFps`,
   `vision.fusion.darkTargetCount` populate.
3. The aft camera's person-in-water raises `notifications.mob` (emergency) —
   visible in the notifications panel.
4. Open the **Vision-AI** webapp from the SignalK menu for the annotated stream
   + target list.

### Seeing AIS fusion

SignalK's demo data / any AIS source populates `vessels.*`. A visual target that
lines up (bearing + range) with an AIS vessel shows `aisCorrelated: true`;
in-range vessels with no AIS match raise dark-target alerts.

## Running the tests

```bash
cd vision-service && pytest
cd signalk-plugin && npm test
```
