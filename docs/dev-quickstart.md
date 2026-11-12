# Dev quickstart: mock container + your Signal K server

No GPU or cameras required. You run the vision container in mock mode
(synthetic forward and aft cameras), install the plugin into the Signal K
server you already use for development, and watch detections flow end to end.
Nothing here starts a Signal K server of its own.

Use a test server: the aft camera's made-up person in the water raises a real
`notifications.mob` alarm.

## 1. Start the mock vision container

**Docker** (simplest):

```bash
docker compose -f docker-compose.mock.yml up -d --build
```

Or pull the published mock image instead of building it:
`docker compose -f deploy/docker-compose.mock.yml up -d`.

**Or as a local process:**

```bash
cd vision-service
python3 -m venv .venv && . .venv/bin/activate
pip install -c constraints.txt -r requirements.txt
VISION_MODE=mock python -m uvicorn app.main:app --port 7000
```

Sanity checks:

```bash
curl localhost:7000/health
curl "localhost:7000/events/recent?n=1" | jq '.[0].targets[].label'
# open localhost:7000/stream/aft.mjpg  → note the red "MOB!" box (person in water)
```

## 2. Install the plugin into your Signal K server

Build it and link it into the server's `node_modules`, then restart the
server:

```bash
cd signalk-plugin && npm ci && npm run build
mkdir -p ~/.signalk/node_modules
ln -s "$PWD" ~/.signalk/node_modules/signalk-vision-ai
```

(Or install **Marine Vision-AI** from the App Store to try the published
version.)

If your Signal K server runs in Docker, the link must point at a path inside
its container, and `localhost:7000` there is the Signal K container itself:
give it host networking, or set the plugin's **Vision container base URL** to
an address that reaches the vision container.

## 3. Watch it work

In the Signal K admin:

1. *Server → Plugin Config → Marine Vision-AI*: enable it. The default
   **Vision container base URL**, `http://localhost:7000`, is right for a
   container on the same computer. Save.
2. *Data → Data Browser*: `vision.system.inferenceFps` and
   `vision.fusion.darkTargetCount` populate. Turn on **Publish targets to
   the chart** to also see `targets.camera:visionai-*` camera targets.
3. The aft camera's person in the water raises `notifications.mob`
   (emergency).
4. Open the **Vision-AI** webapp from the Signal K menu for the annotated
   video and the target list.

Stop the container (`docker compose -f docker-compose.mock.yml stop`) to see
what a user without one sees: a banner at once, and after 30 s the
container setup guide; it closes by itself when you start the container again.

### Own ship and AIS

Targets get a true bearing and a position only from a current own-ship
position and true heading. The plugin treats data older than **Own-ship nav
max age** (5 s) as missing, so the server's bundled sample recordings
(`--sample-nmea0183-data`, `--sample-n2k-data`), which replay timestamps from
the past, show targets without bearings. Feed live navigation data, or a
simulator that stamps the current time.

Any live AIS source populates `vessels.*`. A visual target that lines up in
bearing and range with an AIS vessel shows `aisCorrelated: true`; in-range
vessels with no AIS match raise dark-target alerts.

## Running the tests

```bash
cd vision-service && pytest
cd signalk-plugin && npm test
```
