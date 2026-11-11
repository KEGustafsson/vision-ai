# signalk-vision-ai

SignalK server plugin for the [Marine Vision-AI](../README.md) system. Consumes
detection events from the YOLO-based vision container, enriches them with own-ship
navigation, fuses with AIS, computes CPA/TCPA, raises notifications, publishes
the `vision.*` tree, and serves a captain webapp.

## Install (development)

```bash
npm install
npm run build           # tsc → dist/
npm test                # vitest
# discover from a local SignalK server:
ln -s "$PWD" ~/.signalk/node_modules/signalk-vision-ai
```

Then enable **Marine Vision-AI** in the SignalK admin UI and set `containerUrl`
to the vision container (default `http://localhost:7000`).

## Configuration

All options have sensible defaults (`src/config.ts`). The toggles separate
**computation** (run the analysis + publish its data paths) from **notification**
(raise/clear the SignalK alarm), so you can e.g. keep course and speed on the chart targets without an
audible alarm. Highlights:

| Option | Default | Meaning |
|--------|---------|---------|
| `containerUrl` | `http://localhost:7000` | Vision container base URL |
| `enableDetection` | on | Master detection on/off (also toggled live from the webapp); off releases the cameras and stops inference |
| `enableVisualRadar` | **off** | Publish each georeferenced target as a sensor target, `targets.camera:visionai-<camera>-<id>` (`VIS-` name), which Freeboard-SK draws as a camera target and Target Fusion links to its AIS vessel. Data only — never alarms. Off by default |
| `enableAisFusion` / `enableCollision` | on | Compute: AIS correlation + dark targets / CPA-TCPA + threat level |
| `enableMob` / `notifyCollision` / `notifyDarkTarget` | on | Notify: `notifications.mob` / `…navigation.closestApproach.visionai-*` / `…vision.darkTarget.*`. One collision alarm per boat: none for boats Collision Alerts or another plugin already alarms ([details](../docs/signalk-paths.md#collision-alarms--one-per-boat)) |
| `enableContextControl` | on | Steer active camera/confidence by SOG + time of day |
| `detectClasses` | `person, vessel, buoy` | Object types to surface; empty => all |
| `minConfidence` | 0.4 | Minimum detection confidence |
| `minRangeConfidence` | 0.3 | Gate for georeferencing a target |
| `minTargetRangeM` | 8 | Ignore detections closer than this (m). Pushed to the container, which filters at the source so too-close objects (own-hull / very-near clutter) leave **both** the target list and the annotated overlay. `person` is exempt (MOB); unknown-range kept; 0 disables. |
| `darkTargetRangeM` | 800 | Alert range for non-AIS vessels |
| `collisionTcpaS` / `collisionAlarmTcpaS` / `collisionCpaM` | 600 / 180 / 100 | CPA/TCPA warn / alarm / CPA thresholds |
| `mobMinConfidence` / `mobPersistFrames` | 0.5 / 3 | MOB sensitivity |
| `enableAttitudeCompensation` | **off** | Forward `navigation.attitude` (pitch/roll) to the container to move/tilt each camera's horizon with trim and heel — see [geometry.md](../docs/geometry.md#imu-horizon-compensation). Enable after the on-board sign check |
| `attitudeSmoothingS` / `attitudeMaxAgeS` / `attitudeIntervalMs` | 5 / 2 / 200 | Low-pass time constant (0 = raw, follows waves) / ignore older attitude / push cadence |

Freshness guards (`ownNavMaxAgeS`, `aisMaxAgeS`, `eventMaxAgeS`) expire stale
own-ship, AIS, and detection data so it is never fused as current; see
`src/config.ts` for these and the remaining correlation/anti-flap thresholds.
See [`docs/signalk-paths.md`](../docs/signalk-paths.md) for everything published.

## REST API (mounted at `/plugins/signalk-vision-ai/`)

| Route | Purpose |
|-------|---------|
| `GET /targets` | Current enriched targets + own-ship + system state (webapp) |
| `GET /config` | Active config summary (camera URLs redacted) |
| `GET` / `POST /detection` | Read / set the master detection toggle live |
| `POST /target-limit` | Set max targets per frame |
| `POST /control` | Proxy arbitrary control to the container |
| `GET /ptz` · `POST /ptz/:camera` | PTZ capabilities / move a PTZ camera |
| `GET /stream/:camera` | Reverse-proxied annotated MJPEG |
| `GET /snapshot/:camera` | Reverse-proxied single JPEG |

## Module map

`index.ts` (lifecycle) · `eventStream.ts` (WS ingest + schema validation) ·
`containerClient.ts` (REST/control to the container) · `nav.ts` (own-ship) ·
`enrich.ts` (true bearing + georef) · `aisFusion.ts` (correlation + dark
targets) · `cpa.ts` (CPA/TCPA) · `notifications.ts` (MOB/dark/collision) ·
`publisher.ts` (`vision.*` deltas + sensor targets) · `sensorTargets.ts`
(target identity) · `alarmCoordination.ts` (one alarm per boat) · `geo.ts` (geodesy) ·
`router.ts` (webapp API + proxy) · `skapp.ts` (webapp registration) ·
`config.ts` (config schema + defaults) · `types.ts` (contract mirror).
