# SignalK paths published by the plugin

`vision.*` telemetry (fusion / system) and notifications are published on
`vessels.self` with source label `vision-ai`, in SI units. Visual targets
themselves are published as separate synthetic vessels (see below), not as a
`vision.targets.*` tree.

## Visual targets — synthetic AIS vessels (`enableVisualRadar`, default OFF)

With **Publish targets as synthetic AIS vessels** on, each georeferenced target
is published as its own vessel context `vessels.urn:mrn:signalk:uuid:<uuid>`,
so it renders as a blip on any chartplotter that draws `vessels.*`. The UUID is
a spec-valid v4-format identifier derived deterministically from
`<camera>` + `<trackId>` (same track ⇒ same context). Off by default to avoid
confusion with real AIS contacts.

| Leaf | Units | Description |
|------|-------|-------------|
| `navigation.position` | — | `{ latitude, longitude }` georeferenced fix (always set on a live blip) |
| `name` | — | `VIS-<label>-<trackId>`, published via the root-merge delta (`path: ""`) so `vessel.name` is a plain string |
| `navigation.speedOverGround` | m/s | Target ground speed (needs `enableCollision`) |
| `navigation.courseOverGroundTrue` | rad | Target ground course `[0,2π)` (needs `enableCollision`) |
| `navigation.closestApproach` | — | `{ distance (m), timeTo (s) }` — the standard SignalK CPA/TCPA container (vision-AI estimate); `null` when unresolved |

A blip is **never** published without a real `navigation.position`. When a track
disappears (ages out after `trackTimeoutS`), every leaf — position, name and all
kinematics — is published as `null`, so no synthetic vessel ever lingers without
a location. SOG/COG let a chartplotter draw the vector and compute CPA natively;
`navigation.closestApproach` exposes our own estimate on the standard path.

## Fusion summary — `vision.fusion.*`

| Path | Description |
|------|-------------|
| `vision.fusion.darkTargetCount` | Visual vessels with no AIS correlation, in range |
| `vision.fusion.aisCorrelatedCount` | Visual targets matched to AIS |
| `vision.fusion.lastUpdate` | ISO-8601 timestamp |

## System / statistics — `vision.system.*`

| Path | Units | Notes |
|------|-------|-------|
| `vision.system.activeCamera` | — | Camera prioritised by context control |
| `vision.system.backend` | — | `tensorrt` / `torch-cpu` / `mock` … |
| `vision.system.inferenceFps` | Hz | Zones: alarm < 3, warn 3–6, normal ≥ 6 |
| `vision.system.horizonY` | px | Current horizon calibration |
| `vision.<camera>.targetCount` | — | Tracks currently held per camera |

## Notifications — standard paths (alarm on any MFD)

| Path | State | Trigger |
|------|-------|---------|
| `notifications.mob` | `emergency` | Person-in-water, persisted, with lat/lon in the message |
| `notifications.vision.darkTarget.<key>` | `alert` | In-range vessel with no AIS correlation |
| `notifications.vision.collision.<key>` | `warn` / `alarm` | TCPA/CPA thresholds (alarm = high threat) |
| `notifications.vision.labelMismatch` | `warn` | Selected labels not producible by the active model |
| `notifications.vision.schemaMismatch` | `warn` | Vision container emits an incompatible event `schema_version` (major mismatch); events are ignored until it matches |
| `notifications.vision.staleEvents` | `warn` | Detection events are arriving older than `eventMaxAgeS` and are being ignored (detection effectively offline — check the container, network link, and container/SignalK clock sync) |
| `notifications.vision.containerDown` | `alarm` | The vision container is unreachable on its `/health` endpoint — the whole sensor is offline (no detections or video). Cleared automatically when it responds again |
| `notifications.vision.containerDegraded` | `warn` | The container is up but reports `status: "degraded"` on `/health` — a camera/RTSP stall or a pipeline auto-restart. The message carries the per-camera error and restart count |

Notifications are cleared (`value: null`) automatically when the condition
resolves or the track ages out; MOB has a 60 s hold to ride out brief dropouts.
Collision and dark-target notifications also have a short anti-flap hold
(`notifyHoldS`) so a target near a threshold doesn't toggle the alarm each cycle.

