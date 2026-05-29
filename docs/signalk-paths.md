# SignalK paths published by the plugin

All values are published on `vessels.self` with source label `vision-ai`, in SI
units. Metadata (units / zones) is sent once per concrete path on first publish.

## Visual-radar targets — `vision.targets.<camera>.<trackId>.*`

| Leaf | Units | Description |
|------|-------|-------------|
| `bearingTrue` | rad | True bearing to the target |
| `distance` | m | Estimated range |
| `position` | — | `{ latitude, longitude }` georeferenced fix |
| `label` | — | `vessel` / `buoy` / `person` / … |
| `confidence` | ratio | Detection confidence 0..1 |
| `rangeMethod` | — | `horizon` \| `known_size` \| null |
| `aisCorrelated` | — | boolean: matched to an AIS contact |
| `aisMmsi` | — | matched MMSI, or null |
| `cpa` | m | Closest point of approach |
| `tcpa` | s | Time to CPA |
| `threatLevel` | — | `none` / `low` / `medium` / `high` |

When a track disappears (ages out after `trackTimeoutS`), every leaf is
published as `null` so consumers can drop the blip.

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

Notifications are cleared (`value: null`) automatically when the condition
resolves or the track ages out; MOB has a 60 s hold to ride out brief dropouts.

## Optional: synthetic AIS blips (default OFF)

With `enableAisBlips` on, high-confidence georeferenced targets are also
published as `vessels.urn:mrn:signalk:uuid:vision-<camera>-<id>` with a
`navigation.position` and a `VIS-<label>-<id>` name, so they render as blips on
chartplotters that only draw `vessels.*`. Off by default to avoid confusion with
real AIS contacts.
