# Onboard verification runbook (Jetson + SignalK)

> **Audience: the operator agent (me) running on/near the boat's NVIDIA Jetson
> Orin Nano, plus the human skipper.** When live access is enabled, read this
> file top-to-bottom and execute the phases in order. Each step lists the exact
> command, the **PASS** criterion, and what to do on failure. Steps marked
> **[NEEDS YOU]** require the skipper (physical measurement or visual
> confirmation); everything else I can run and judge myself.

---

## 0. Before starting

### 0.1 Information to collect from the skipper
- `NANO_HOST` / port of the **vision container** (default `:8000`) and whether it
  is bound to localhost or a LAN IP.
- `SK_HOST` / port of the **SignalK server** (default `:3000`) and an admin
  login / token if the API requires auth.
- Camera **RTSP URLs** already configured (in `VISION_CAMERA_FORWARD_URL` /
  `VISION_CAMERA_AFT_URL` or `config/jetson.yaml`).
- Whether a **TensorRT `.engine`** has been built, and whether a **maritime
  model** is in use (stock COCO only detects `person`/`boat`; `buoy` and robust
  person-in-water need a marine-trained model — see `docs/jetson-setup.md`).
- Calibration values if known: camera **height above waterline (m)**, **HFOV
  (deg)**, mounting **yaw offset**, and **horizon row (px)** — used in Phase 7.

### 0.2 Safety preconditions ⚠️
- The alerting features publish **real** `notifications.*` to the boat's
  SignalK, which may sound alarms on the MFD/plotter. **Tell the crew before
  testing**, and prefer the destructive notification tests (Phase 5) on a
  **second, isolated test instance** (different ports, `MODE=cpu` with a
  recorded clip) so the live helm display isn't spammed.
- **Never stage a real man-overboard.** Use a fender/mannequin, a person safely
  on a dock at close range with the crew informed, or a recorded video clip.
- I will not change thresholds or restart the live service without confirming
  with the skipper first.

### 0.3 Set convenience variables
```bash
export NANO=http://127.0.0.1:8000          # vision container
export SK=http://127.0.0.1:3000            # SignalK server
export OUT=/tmp/vision-verify && mkdir -p $OUT
```

---

## 1. Environment discovery

```bash
docker ps                                   # find vision-service + signalk containers
cat /etc/nv_tegra_release                    # L4T / JetPack version
sudo nvpmodel -q                             # power mode (want MAXN / "Super")
sudo jetson_clocks --show | head             # clocks pinned?
tegrastats --interval 1000 | head -n 3       # baseline GPU/CPU/EMC/temp/power
```
**PASS:** containers running; power mode is the high-performance one (else
`sudo nvpmodel -m 0 && sudo jetson_clocks`). Record JetPack version for the report.

---

## 2. Vision container

### 2.1 Health & backend
```bash
curl -s $NANO/health | jq
```
**PASS:** `status:"ok"`, `mode:"jetson"`, `backend:"tensorrt"` (NOT `mock`),
both cameras listed, `camera_errors:{}`, an `active_camera`.
**FAIL → backend mock/torch:** engine not found/loaded → check
`models/yolov8n.engine` and `VISION_MODEL_ENGINE`; rebuild with
`scripts/export_engine.py` on the Jetson.
**FAIL → camera_errors non-empty:** go to 2.2 debug.

### 2.2 Cameras & ingestion
```bash
curl -s $NANO/cameras
for c in forward aft; do curl -s $NANO/snapshot/$c -o $OUT/snap_$c.jpg; done
ls -l $OUT/*.jpg
```
I will then **send the snapshots to the skipper** (SendUserFile) for a visual
sanity check (right scene, right camera, focused, horizon roughly level).
**PASS:** a non-trivial JPEG (>5 KB) for each camera and the skipper confirms the
view. **[NEEDS YOU]** confirm forward/aft aren't swapped.
**FAIL:** RTSP/GStreamer issue — check the URL, try `MODE=cpu` (software decode)
to isolate HW-decode vs connectivity; see `docs/jetson-setup.md` troubleshooting.

### 2.3 Inference performance
```bash
# watch a few seconds of events and read backend/latency/seq
curl -s "$NANO/events/recent?n=5" | jq '.[] | {cam:.camera, seq:.frame_seq, backend:.inference.backend, ms:.inference.latency_ms, n:(.targets|length)}'
# in parallel, sample the GPU/thermals
timeout 20 tegrastats --interval 1000 | tee $OUT/tegrastats.log
```
**PASS:** `frame_seq` advancing for both cameras; `latency_ms` consistent with a
usable FPS (YOLOv8n target ≈ 15–20 FPS / camera; with two cameras the context
loop prioritises one); no thermal throttling in `tegrastats` (watch the temps
and the `... GR3D_FREQ` load). Record measured FPS and peak temp.

### 2.4 Detection sanity
```bash
curl -s "$NANO/events/recent?n=1" | jq '.[0].targets[] | {label, confidence, bbox}'
```
**PASS:** labels are sensible for what the cameras see. **Note the model limit:**
with stock COCO only `person`/`vessel` are reliable; absence of `buoy` is
expected unless a maritime model is loaded — flag this, don't fail on it.

### 2.5 Geometry sign & calibration state
```bash
curl -s "$NANO/events/recent?n=1" | jq '.[0] | {horizon_y, calibration_status, targets:[.targets[]|{label, brg:.geometry.relative_bearing_deg, rng:.geometry.range_m, method:.geometry.range_method, rconf:.geometry.range_confidence}]}'
```
**PASS:** an object visually on the **right** of the forward frame reports a
**positive** `relative_bearing_deg` (starboard); `calibration_status:"ok"` if
`horizon_y` is set. **[NEEDS YOU]** confirm bearing sign against the live view.
If `uncalibrated`/range mostly null → Phase 7.

---

## 3. SignalK plugin

### 3.1 Plugin connected
- Check the plugin status in the SignalK admin UI (or server log) — it should
  read `N targets, X fps, active=...` and not be stuck reconnecting.
```bash
curl -s "$SK/signalk/v1/api/vessels/self/vision/system" | jq
```
**PASS:** `inferenceFps`, `backend`, `activeCamera`, `horizonY` present →
the plugin's WS consumer is receiving container events.
**FAIL:** check the plugin's `containerUrl` points at `$NANO`; check reachability
from the SignalK host to the container.

### 3.2 vision.* tree & metadata
```bash
curl -s "$SK/signalk/v1/api/vessels/self/vision/targets" | jq 'keys'
curl -s "$SK/signalk/v1/api/vessels/self/vision/fusion" | jq
# metadata + zones on the fps path:
curl -s "$SK/signalk/v1/api/vessels/self/vision/system/inferenceFps/meta" | jq
```
**PASS:** `vision.targets.<camera>.<id>.*` populate while targets are in view;
`vision.fusion.darkTargetCount`/`aisCorrelatedCount` present; `inferenceFps`
carries `units:"Hz"` + zones.

### 3.3 Live delta stream
```bash
# stream self deltas for ~15s and grep the vision/notification paths
( printf '{"context":"vessels.self","subscribe":[{"path":"vision.*"},{"path":"notifications.*"}]}\n'; sleep 15 ) \
  | websocat -n1 "$SK/signalk/v1/stream?subscribe=none" 2>/dev/null | tee $OUT/sk-deltas.log | head -40
```
(If `websocat` is unavailable I'll use a short Node/Python WS client instead.)
**PASS:** live deltas for `vision.targets.*` arrive at roughly the processing
cadence; values change as targets move.

### 3.4 Plugin REST proxy (same-origin, behind SignalK auth)
```bash
curl -s "$SK/plugins/signalk-vision-ai/targets" | jq '{own:.ownShip, system:.system, n:(.targets|length)}'
curl -s "$SK/plugins/signalk-vision-ai/config" | jq          # MUST show no RTSP creds
curl -s "$SK/plugins/signalk-vision-ai/snapshot/forward" -o $OUT/sk_proxy_forward.jpg && ls -l $OUT/sk_proxy_forward.jpg
```
**PASS:** `/targets` returns own-ship + enriched targets; `/config` shows camera
URLs redacted (`***redacted***`); the proxied snapshot is a valid JPEG.

---

## 4. End-to-end enrichment & fusion (the high-value ground-truth checks)

### 4.1 Own-ship nav present (prerequisite)
```bash
for p in navigation/position navigation/headingTrue navigation/speedOverGround navigation/courseOverGroundTrue; do
  echo -n "$p = "; curl -s "$SK/signalk/v1/api/vessels/self/$p" | jq -c '.value'
done
```
**PASS:** position + heading at minimum. Without heading, `bearingTrue`/`position`
on targets will be null and 4.2–4.4 can't run — fix the nav source first.

### 4.2 True-bearing check **[NEEDS YOU]**
Pick a target visible in the camera and by eye/compass. Compare its
`vision.targets.<cam>.<id>.bearingTrue` (radians → ×57.2958 for degrees) to the
hand-compass bearing. **PASS:** within a few degrees once calibrated.

### 4.3 AIS cross-check (best quantitative validation)
When a **real AIS vessel** is within camera view, compare the vision estimate to
the AIS ground truth:
```bash
# list AIS contacts with bearing/range as SignalK sees them, and the vision targets
curl -s "$SK/plugins/signalk-vision-ai/targets" | jq '.targets[] | select(.aisCorrelated==true) | {id:.key, mmsi:.aisMmsi, visBrgDeg:(.bearingTrue*57.2958), visRngM:.geometry.range_m, cpa, tcpa}'
```
Then for each `aisMmsi`, read the AIS contact's actual position from SignalK and
compute true bearing/range from own position (I'll do this math) and diff.
**PASS:** a vessel in view shows `aisCorrelated:true` with the right MMSI, and
vision bearing matches AIS within a few degrees; range within the coarse
monocular tolerance. This validates geometry **and** fusion together.

### 4.4 Dark-target check **[NEEDS YOU]**
With a small craft in view that is **not** transmitting AIS (and within
`darkTargetRangeM`): **PASS:** `aisCorrelated:false` and a
`notifications.vision.darkTarget.*` alert appears; it **clears** when the craft
leaves range/view.

### 4.5 CPA/TCPA sanity
For a moving correlated target, check `cpa`/`tcpa` signs and magnitudes are
physical (closing target → positive `tcpa`, decreasing range; diverging →
`threatLevel:"none"`). Cross-check against the MFD's own CPA if available.

---

## 5. Notifications ⚠️ (real alarms — see 0.2)

Prefer a **second isolated instance** for these:
```bash
# example: a throwaway CPU instance replaying a clip, on a spare port
VISION_MODE=cpu VISION_MOCK_SOURCE=/path/clip.mp4 VISION_PORT=8001 \
  python -m uvicorn app.main:app --port 8001
```
- **MOB:** trigger safely (fender/mannequin/clip). **PASS:** `notifications.mob`
  = `emergency` with a lat/lon in the message; it persists through brief dropouts
  (60 s hold) and clears afterwards.
- **Collision:** with a closing target crossing the TCPA/CPA thresholds, expect
  `notifications.vision.collision.*` `warn`→`alarm`; **PASS:** it fires and later
  clears (`value:null`) when the risk passes.
```bash
curl -s "$SK/signalk/v1/api/vessels/self/notifications" | jq
```

---

## 6. Captain webapp
Open `${SK}/signalk-vision-ai/` in a browser (or I fetch the proxied MJPEG/
snapshot and send a frame to the skipper). **PASS:** annotated stream renders,
the target list matches `/targets`, own-ship line shows live nav, threat colours
and the camera switch work.

---

## 7. Calibration loop **[NEEDS YOU for measurements]**
When 2.5 / 4.2 show bearing or range error:
1. Collect per-camera **HFOV** (datasheet), **height_m** (measure), **yaw
   offset**, and **horizon_y** (read the pixel row of the horizon from a Phase
   2.2 snapshot with the boat level).
2. I set them in `config/jetson.yaml` (`cameras[]`), restart the container, and
   re-run **4.3 (AIS cross-check)** to measure the new error.
3. Iterate to minimise bearing/range error; then tune detection/alert thresholds
   from observed false-positives/negatives: `minConfidence`,
   `darkTargetRangeM`, `correlationBearingDeg`/`correlationRangeFrac`,
   `collisionTcpaS`/`collisionCpaM`, `mobMinConfidence`/`mobPersistFrames`
   (plugin config). Background and formulas: `docs/geometry.md`.

---

## 8. Soak & resilience
```bash
timeout 600 tegrastats --interval 2000 | tee $OUT/soak-tegrastats.log   # 10 min
```
- Watch for thermal throttling, memory growth, and FPS decay over time.
- **Recovery test [NEEDS YOU]:** briefly disconnect one camera → `/health`
  should show that camera in `camera_errors` and the other keep running; on
  reconnect it should recover without a restart.

---

## 9. Troubleshooting quick map
| Symptom | Check | Likely fix |
|---------|-------|-----------|
| `backend:"mock"` on Jetson | engine path / `VISION_MODEL_ENGINE` | build `.engine` with `scripts/export_engine.py` |
| `camera_errors` set | RTSP URL, GStreamer | verify URL; test `MODE=cpu`; check L4T multimedia/`nvv4l2decoder` |
| `vision.system.*` absent in SignalK | plugin `containerUrl`, reachability | fix URL; confirm WS connects |
| `bearingTrue`/`position` null | own-ship heading/position | fix nav source; see 4.1 |
| range mostly null / wild | `horizon_y`, `calibration_status` | calibrate (Phase 7) |
| low FPS / hot | `nvpmodel`, both cameras full-rate | set MAXN+`jetson_clocks`; rely on context-camera prioritisation; prefer v8n |
| RTSP creds visible in `/config` | should be redacted | regression — check `rest.get_config` |

---

## 10. Output of this runbook
At the end I produce a **verification report**: PASS/FAIL per phase, measured
numbers (FPS, latency, temps, bearing/range error vs AIS), the saved annotated
snapshots, and a list of recommended config/calibration changes — with anything
still requiring a sea trial called out separately. I will **not** declare the
system field-ready on my own; that needs the skipper's sea-trial sign-off.
