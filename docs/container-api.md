# Vision container API

The HTTP/WebSocket interface exposed by the vision service (`vision-service/`).
It is a small control + introspection surface around three transports: a
**WebSocket** event stream, an **MJPEG** video stream, and a **REST** control
channel (the same split described in [architecture.md](architecture.md)).

- **Base URL** — `http://<container>:7000` (default port `7000`). On a boat the
  container usually binds to `127.0.0.1:7000` and the SignalK plugin reverse-
  proxies the streams and control behind SignalK's authentication
  (`/plugins/signalk-vision-ai/…`, see [the plugin README](../signalk-plugin/README.md)).
  Endpoints here are **unauthenticated** — don't expose the port to an untrusted
  network.
- **Units** — the container speaks its native units (**degrees**, **metres**,
  **pixels**). The plugin converts to SI at the boundary. See
  [event-schema.md](event-schema.md) and [geometry.md](geometry.md).
- **Source of truth** — request/response models live in
  `vision-service/app/schemas.py`; routes in `vision-service/app/api/`.

## REST endpoints

| Method & path | Purpose | Success | Errors |
|---|---|---|---|
| `GET /health` | Liveness + backend/camera state | `200` `HealthResponse` | — |
| `GET /config` | Effective settings (RTSP creds redacted) | `200` settings object | — |
| `GET /cameras` | Configured camera names | `200` `["forward","aft"]` | — |
| `GET /events/recent?n=` | Last `n` detection events (`n` 1–1000, default 20) | `200` `DetectionEvent[]` | — |
| `POST /control` | Change runtime behaviour, no restart | `200` `{ "applied": { … } }` | `404` unknown camera |
| `GET /ptz` | Names of PTZ-capable cameras | `200` `{ "cameras": [...] }` | — |
| `POST /ptz/{camera}` | ONVIF pan/tilt/zoom | `200` `{ "ok": true, "action": … }` | `404` no PTZ · `400` bad action · `502` camera unreachable |
| `GET /snapshot/{camera}` | Latest annotated frame | `200` `image/jpeg` | `404` no frame yet |
| `GET /stream/{camera}.mjpg` | Annotated MJPEG stream | `200` `multipart/x-mixed-replace` | `404` unknown camera · `503` too many clients |

### `GET /health`

Polled by the Docker `HEALTHCHECK` and by the plugin. `status` is `"degraded"`
when any camera has an error or (DeepStream) the pipeline is actively
restarting, else `"ok"`.

```jsonc
{
  "status": "ok",                  // "ok" | "degraded"
  "mode": "jetson",                // run mode from config/env
  "backend": "tensorrt",           // mock | torch-cpu | torch-cuda | tensorrt | deepstream
  "cameras": ["forward", "aft"],
  "uptime_s": 1843.2,
  "active_camera": "forward",      // camera prioritised by context control; may be null
  "camera_errors": {},             // { "<camera>": "<last error>" } when a feed is failing
  "detection_enabled": true,       // master on/off (see POST /control `enabled`)
  "max_targets": 20,               // active per-frame detection cap
  "labels": ["person", "vessel", "buoy"],  // active label filter; null => all
  "model": "coco",                 // active detection model
  "model_labels": ["buoy", "person", "vessel"],  // labels this model can produce
  "pipeline_restarts": 0,          // DeepStream auto-restart count (0 on other backends)
  "pipeline_last_error": null
}
```

### `POST /control`

Every field is optional; only those present are applied, and the response echoes
what actually took effect. The plugin owns these settings and pushes them so the
annotated overlay and the event stream always agree.

```jsonc
// request body (ControlRequest) — any subset:
{
  "active_camera": "aft",          // switch prioritised camera (404 if unknown)
  "confidence": 0.45,              // detection confidence floor, 0..1
  "max_targets": 20,              // per-frame cap, 1..300
  "min_target_range_m": 8,        // drop closer detections (person exempt); 0 disables
  "mode_hint": "docking",         // "underway" | "docking" | "anchored"
  "labels": ["person", "vessel"], // canonical labels to surface; [] => all
  "enabled": true                  // master on/off: false releases cameras, stops inference
}

// response:
{ "applied": { "active_camera": "aft", "confidence": 0.45 } }
```

### `POST /ptz/{camera}`

For ONVIF PTZ cameras only (those listed by `GET /ptz`). `move` uses normalised
velocities in `-1..1` (`+pan` = right, `+tilt` = up, `+zoom` = in).

```jsonc
{ "action": "move", "pan": 0.5, "tilt": 0.0, "zoom": 0.0 }  // action: move | stop | home
```

### `GET /snapshot/{camera}` · `GET /stream/{camera}.mjpg`

`snapshot` returns the single latest annotated JPEG. `stream` is a
`multipart/x-mixed-replace; boundary=frame` MJPEG feed that emits only newly
produced frames (output rate tracks real production, not the poll rate), so a
slow client can never stall inference. Stream clients are capped
(`server.max_stream_clients`); over the cap returns `503`.

## WebSocket — `GET /ws/events`

The primary output. On connect the client receives the recent buffered events
(late-join catch-up, last 20), then a live push of every new `DetectionEvent` as
frames are processed.

- **Query** — optional `?camera=<name>` restricts the stream to one camera.
- **Payload** — one `DetectionEvent` JSON per message. Its shape is the wire
  contract documented in [event-schema.md](event-schema.md).
- **Backpressure / limits** — subscribers are capped
  (`server.max_ws_clients`); over the cap the server closes with code `1013`
  ("try again later").

```bash
# stream forward-camera events with websocat
websocat "ws://localhost:7000/ws/events?camera=forward"
```

## Quick reference

```bash
curl -s localhost:7000/health | jq
curl -s localhost:7000/cameras | jq
curl -s "localhost:7000/events/recent?n=1" | jq '.[0].targets[].label'
curl -s -X POST localhost:7000/control -H 'content-type: application/json' \
  -d '{"active_camera":"aft","confidence":0.45}' | jq
curl -s localhost:7000/snapshot/forward -o forward.jpg
# open http://localhost:7000/stream/forward.mjpg in a browser
```

The plugin re-exposes the streams and control behind SignalK auth; the operator
runbook in [onboard-verification.md](onboard-verification.md) exercises these
endpoints end to end.
