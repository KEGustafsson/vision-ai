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

All options have sensible defaults (`src/config.ts`). Highlights:

| Option | Default | Meaning |
|--------|---------|---------|
| `containerUrl` | `http://localhost:7000` | Vision container base URL |
| `enableVisualRadar` / `enableAisFusion` / `enableMob` / `enableCollision` | on | Feature toggles |
| `enableAisBlips` | **off** | Project targets as synthetic `vessels.*` blips |
| `enableContextControl` | on | Steer active camera/confidence by SOG + time of day |
| `minRangeConfidence` | 0.3 | Gate for georeferencing a target |
| `darkTargetRangeM` | 800 | Alert range for non-AIS vessels |
| `collisionTcpaS` / `collisionAlarmTcpaS` / `collisionCpaM` | 600 / 180 / 100 | CPA/TCPA thresholds |
| `mobMinConfidence` / `mobPersistFrames` | 0.5 / 3 | MOB sensitivity |

See [`docs/signalk-paths.md`](../docs/signalk-paths.md) for everything published.

## REST API (mounted at `/plugins/signalk-vision-ai/`)

| Route | Purpose |
|-------|---------|
| `GET /targets` | Current enriched targets + own-ship + system state (webapp) |
| `GET /config` | Active config summary |
| `POST /control` | Proxy control to the container |
| `GET /stream/:camera` | Reverse-proxied annotated MJPEG |
| `GET /snapshot/:camera` | Reverse-proxied single JPEG |

## Module map

`index.ts` (lifecycle) · `eventStream.ts` (WS + schema validation) ·
`nav.ts` (own-ship) · `enrich.ts` (true bearing + georef) ·
`aisFusion.ts` (correlation + dark targets) · `cpa.ts` (CPA/TCPA) ·
`notifications.ts` (MOB/dark/collision) · `publisher.ts` (`vision.*` deltas) ·
`geo.ts` (geodesy) · `router.ts` (webapp API + proxy).
