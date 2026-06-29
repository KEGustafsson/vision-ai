# signalk-vision-ai

SignalK server plugin for the [Marine Vision-AI](../README.md) system. Consumes
detection events from the YOLOv8 vision container, enriches them with own-ship
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
(raise/clear the SignalK alarm), so you can e.g. keep CPA on the blips without an
audible alarm. Highlights:

| Option | Default | Meaning |
|--------|---------|---------|
| `containerUrl` | `http://localhost:7000` | Vision container base URL |
| `enableDetection` | on | Master detection on/off (also toggled live from the webapp); off releases the cameras and stops inference |
| `enableVisualRadar` | **off** | Publish each georeferenced target as a synthetic `vessels.*` AIS blip (`VIS-` name). Data only — never alarms. Off by default to avoid confusion with real AIS |
| `enableAisFusion` / `enableCollision` | on | Compute: AIS correlation + dark targets / CPA-TCPA + threat level |
| `enableMob` / `notifyCollision` / `notifyDarkTarget` | on | Notify: `notifications.mob` / `…vision.collision.*` / `…vision.darkTarget.*` |
| `enableContextControl` | on | Steer active camera/confidence by SOG + time of day |
| `detectClasses` | `person, vessel, buoy` | Object types to surface; empty => all |
| `minConfidence` | 0.4 | Minimum detection confidence |
| `minRangeConfidence` | 0.3 | Gate for georeferencing a target |
| `minTargetRangeM` | 8 | Ignore detections closer than this (m). Pushed to the container, which filters at the source so too-close objects (own-hull / very-near clutter) leave **both** the target list and the annotated overlay. `person` is exempt (MOB); unknown-range kept; 0 disables. |
| `darkTargetRangeM` | 800 | Alert range for non-AIS vessels |
| `collisionTcpaS` / `collisionAlarmTcpaS` / `collisionCpaM` | 600 / 180 / 100 | CPA/TCPA warn / alarm / CPA thresholds |
| `mobMinConfidence` / `mobPersistFrames` | 0.5 / 3 | MOB sensitivity |

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
`publisher.ts` (`vision.*` deltas + synthetic vessels) · `geo.ts` (geodesy) ·
`router.ts` (webapp API + proxy) · `skapp.ts` (webapp registration) ·
`config.ts` (config schema + defaults) · `types.ts` (contract mirror).
