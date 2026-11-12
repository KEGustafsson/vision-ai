# Changelog

All notable changes to `signalk-vision-ai` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/).

## [1.0.0]

First release on the Signal K App Store.

### Added

- Ingest of `DetectionEvent`s from the Marine Vision-AI container over
  WebSocket, validated against the generated JSON schema.
- Own-ship enrichment: true bearing and lat/lon for every georeferenced camera
  target, with freshness guards that drop stale own-ship, AIS and detection
  data instead of fusing it as current.
- AIS fusion and dark-target detection (`vision.fusion.*`,
  `notifications.vision.darkTarget.*`).
- CPA/TCPA and threat level per target, with one collision alarm per boat
  (`notifications.navigation.closestApproach.visionai*`), coordinated with
  other collision plugins.
- Man-overboard alarm (`notifications.mob`) when a person in the water persists.
- Optional chart targets (`targets.camera:visionai-*`), off by default.
- Optional IMU horizon compensation from `navigation.attitude`, off by default.
- Context control of the active camera and confidence by speed and time of day.
- Captain webapp: annotated live video, camera switch, PTZ pad, target list,
  live detection on/off.
- Signal K App Store metadata: display name, app icon and screenshots.
- Signal K plugin CI (`.github/workflows/signalk-ci.yml`).

[1.0.0]: https://github.com/KEGustafsson/vision-ai/releases/tag/v1.0.0
