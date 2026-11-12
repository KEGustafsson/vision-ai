# Changelog

All notable changes to `signalk-vision-ai` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/).

## [0.1.0]

First pre-release, the first version published to npm and the Signal K App Store.

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
- Signal K App Store metadata: display name, app icon, screenshots, and
  recommended companions (Freeboard-SK, Target Fusion, Collision Alerts).
- Signal K plugin CI (`.github/workflows/signalk-ci.yml`).
- Vision container setup guide in the webapp: while the container does not
  answer, a banner and step-by-step setup for Jetson Orin or Xavier
  (DeepStream), Jetson Orin (TensorRT), CPU or mock, using the container
  release that matches the plugin's version. It opens at once if the
  container never answered, otherwise after 30 s, and closes when it answers.
- `GET /setup`, and a `container` status in `GET /targets`, for the guide.
- The admin shows a missing container as the plugin's error instead of a
  green "0 targets" status.
- Prebuilt vision-service images on GHCR and compose files that pull them,
  published with each GitHub Release (`docs/release.md`).

[0.1.0]: https://github.com/KEGustafsson/vision-ai/releases/tag/v0.1.0
