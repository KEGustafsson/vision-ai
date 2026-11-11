# SignalK paths published by the plugin

`vision.*` telemetry (fusion / system) and notifications are published on
`vessels.self` with source label `vision-ai`, in SI units. Visual targets
themselves are published as Signal K **sensor targets** (see below), not as a
`vision.targets.*` tree.

## Visual targets — sensor targets (`enableVisualRadar`, default OFF)

With **Publish targets to the chart** on, each georeferenced camera track is
published as its own context in the server's *sensor targets* convention
(signalk-server docs, *Sensor Targets*):

```
targets.camera:visionai-<camera>-<stable_id>
```

Radar, camera and AIS plugins share this convention, so the targets are ordinary
Signal K data that chartplotters and collision alarms read without knowing this
plugin. Freeboard-SK (3.5 and later) draws each one as a camera target: an
orange ring with a course line, shown with its Vessels layer, never mistaken
for an AIS vessel.

| Leaf | Units | Description |
|------|-------|-------------|
| `name` | — | `VIS-<camera>-<track_id>`: the number the container burns into the video overlay, so a chart ring and its on-screen box carry the same label. Published at the root (`path: ""`) so it is a plain string |
| `navigation.position` | — | `{ latitude, longitude }` georeferenced fix; `null` once when the track is lost |
| `navigation.speedOverGround` | m/s | Target ground speed (needs `enableCollision`), `null` while unknown |
| `navigation.courseOverGroundTrue` | rad | Target ground course `[0,2π)` (needs `enableCollision`), `null` while unknown |

- **Identity.** The id is keyed on the container's never-recycled
  `stable_id` (`track_id` for older containers without it): consumers treat a
  track whose position went null as finished, so a recycled number must never
  name a second boat. The `visionai-` prefix keeps it unique among camera
  sensors. Camera names are written one-to-one into the id (any character
  outside `[a-zA-Z0-9]` as its hex code between underscores, so `bow cam` is
  `bow_20_cam`), so two cameras never share a context.
- **Container restarts.** `stable_id` is unique within one container session,
  and a restart always drops the event-stream connection. So whenever the
  connection (re)opens, before its first event, every target of the previous
  connection is withdrawn (null position) and its CPA and AIS history dropped:
  a new boat that receives a reused serial starts as a new target, never as an
  old one. A reconnect within one session only restarts its tracks.
- **One source per camera** (`vision-ai-<camera>`). Target Fusion never merges
  two targets from one sensor and identifies the sensor by `$source`; with one
  label for both cameras a boat seen by both would stay two rings.
- **Observations, not repetitions.** A target is published only when the
  container saw it again, stamped with the frame's capture time. Nothing is
  republished while a track is held through a detection gap (`blipHoldS`), so
  consumers age a target from its real last sighting (Freeboard greys a ring
  after 20 s; Target Fusion and Collision Alerts ignore one older than a
  minute). A frame the container coasted across a detector dropout is a
  prediction, not a sighting, and is not published; the hold runs from the
  last real detection however long the container coasts.
- **Track lost.** After `blipHoldS` without a detection, or when the target loses
  its georeference, `navigation.position` is published as `null` once: Freeboard
  removes it at once and Target Fusion withdraws its link. The server prunes the
  context later. A target is never published without a real position.
- **No AIS shortcut.** Root `mmsi` is not set: this plugin's own AIS correlation
  is a position match, not an identification, and Target Fusion links a target
  that carries an MMSI to that vessel unconditionally.

### With Target Fusion

Install [signalk-target-fusion](https://github.com/dirkwa/signalk-target-fusion)
and it sets `sameAs` on each camera target that is a boat AIS already reports,
or the same boat another sensor (the other camera, a radar) tracks. Freeboard
then draws such a boat once, as its AIS vessel, and Collision Alerts alarms it
once. This plugin reads those links too, to alarm each boat once (below). It
never writes `sameAs` itself: one fusion plugin at a time.

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
| `vision.system.inferenceFps` | Hz | Plain telemetry — no zones (zone metadata would make SignalK auto-raise a notification on it) |
| `vision.system.horizonY` | px | Current horizon row at the frame centre (IMU-compensated when active) |
| `vision.system.attitudeCompensated` | — | `true` while the container applies IMU attitude to the horizon. Published only when `enableAttitudeCompensation` is on |
| `vision.<camera>.targetCount` | — | Tracks currently held per camera |

## Notifications — standard paths (alarm on any MFD)

| Path | State | Trigger |
|------|-------|---------|
| `notifications.mob` | `emergency` | Person-in-water, persisted, with lat/lon in the message |
| `notifications.vision.darkTarget.<boat>` | `alert` | In-range vessel with no AIS correlation, once per boat; none while the boat is under a collision alarm |
| `notifications.navigation.closestApproach.<boat>` | `warn` / `alarm` | TCPA/CPA thresholds (alarm = high threat), once per boat — see *Collision alarms* below |
| `notifications.vision.labelMismatch` | `warn` | Selected labels not producible by the active model |
| `notifications.vision.schemaMismatch` | `warn` | Vision container emits an incompatible event `schema_version` (major mismatch); events are ignored until it matches |
| `notifications.vision.staleEvents` | `warn` | Detection events are arriving older than `eventMaxAgeS` and are being ignored (detection effectively offline — check the container, network link, and container/SignalK clock sync) |
| `notifications.vision.containerDown` | `alarm` | The vision container is unreachable on its `/health` endpoint — the whole sensor is offline (no detections or video). Cleared automatically when it responds again |
| `notifications.vision.containerDegraded` | `warn` | The container is up but reports `status: "degraded"` on `/health` — a camera/RTSP stall or a pipeline auto-restart. The message carries the per-camera error and restart count |

Notifications are cleared (`value: null`) automatically when the condition
resolves or the track ages out; MOB has a 60 s hold to ride out brief dropouts.
Collision and dark-target notifications also have a short anti-flap hold
(`notifyHoldS`) so a target near a threshold doesn't toggle the alarm each cycle.
The hold is skipped when another alarm takes the boat over, so it never keeps
two alarms up for one boat.

`<boat>` is `visionai-<camera>-<stable_id>` for a boat known only to the
cameras, or `visionai-mmsi-<mmsi>` once it is matched to an AIS vessel. The
`visionai-` prefix keeps the path apart from those Collision Alerts and SK
Derived Data write (they key by the bare target id): two producers on one path
would let this plugin's clear wipe the other's standing alarm off displays that
key alarms by path.

## Collision alarms — one per boat

The collision alarm uses the standard `navigation.closestApproach` notification
in the shape Collision Alerts and MaYaRa share, so chartplotters draw it
(Freeboard: a flashing ring on the boat, both course lines on to the closest
approach, and the closest-approach line between them):

| `data` field | Units | Description |
|--------------|-------|-------------|
| `targetRef` | — | The boat's context: its AIS vessel (fusion link or own AIS match), else its camera target. Omitted when the target is not published and not matched to AIS |
| `source` | — | `camera` |
| `cpa` / `tcpa` / `range` | m / s / m | Closest approach, time to it, current distance |
| `targetCog` / `targetSog` | rad / m/s | Target ground course (`null` when stopped) and speed |
| `cpaPositions` | — | `{ self, target }`: where both boats will be at the closest approach |

`other` repeats `targetRef` for consumers older than `data.targetRef`.

A boat alarms **once**, whichever plugins and sensors see it:

1. **Tracks of one boat raise one alarm.** Tracks that share any context — the
   target one is published as, the target or vessel a Target Fusion link
   (`sameAs`) names, the AIS vessel this plugin matched — are one boat,
   followed transitively; it alarms with its most urgent track. The same
   grouping gives one dark-target alert per boat.
2. **Collision Alerts takes the boats it can see.** While the Collision Alerts
   plugin runs it announces the `collisionAlerts` PropertyValue
   `{ evaluatesTargets: true }`. It then evaluates every AIS vessel and every
   sensor target, so this plugin leaves to it every boat that is published as a
   target or matched to AIS, as the MaYaRa radar plugin does. A boat known only
   to this plugin (not published, no AIS match) keeps its alarm here. Collision
   Alerts applies its own limits (its *Alert sensitivity*), not
   `collisionCpaM` / `collisionTcpaS`.
3. **An alarm already raised elsewhere wins.** A `closestApproach` alarm that
   another producer raised (SK Derived Data, MaYaRa, Collision Alerts) and that
   names the same boat — by its AIS vessel, its camera target or the target it
   is linked to — makes this plugin clear its own at once.
4. **A collision alarm replaces the dark-target alert** for the same boat; its
   message says `(no AIS)` instead. That holds for another producer's alarm
   too, even with this plugin's own collision alarms off.

Each hand-over clears this plugin's alarm at once, skipping its anti-flap
hold, including an alarm still held for a boat that has stopped threatening.

All of this fails safe: only an announcement or an alarm that is actually there
makes this plugin stand down, and anything it cannot read leaves its own alarm
up: a duplicate alarm is a lesser failure than a missing one.

