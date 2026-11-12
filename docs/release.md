# Prebuilt images and releases

The vision service is published as ready-made container images, so a boat no
longer has to build one from source (an hour or more on a Jetson). Each
release also carries compose files that pull those images. This page covers
installing from a release, updating, calibrating, and cutting a release.

The Signal K plugin's webapp shows the same steps, filled in for the user's
hardware and camera addresses, whenever it cannot reach the container.

## Images

All images are on GitHub Container Registry as
`ghcr.io/kegustafsson/vision-ai:<tag>`:

| Variant | Tag | Platforms | Hardware | Compose file |
|---------|-----|-----------|----------|--------------|
| DeepStream, JetPack 6 | `X.Y.Z-deepstream` | arm64 | Jetson Orin (L4T R36), DeepStream 7.1 | `docker-compose.deepstream.yml` |
| DeepStream, JetPack 5 | `X.Y.Z-deepstream-xavier` | arm64 | Jetson Xavier (L4T R35), DeepStream 6.3 | `docker-compose.deepstream-xavier.yml` |
| TensorRT | `X.Y.Z-jetson` | arm64 | Jetson Orin, Ultralytics on TensorRT | `docker-compose.jetson.yml` |
| CPU | `X.Y.Z-cpu` | amd64, arm64 | any computer, no GPU (about 5 fps) | `docker-compose.cpu.yml` |
| Mock | `X.Y.Z-mock` | amd64, arm64 | none: synthetic cameras, for bench tests | `docker-compose.mock.yml` |

Tags:

- `X.Y.Z-<variant>`: a release. Use these; the release's compose files do.
- `<variant>` (e.g. `deepstream`): the latest stable release (1.0.0 or later).
- `edge-<variant>`: the latest build of `main`. The compose files in `deploy/`
  of a git checkout use these.

The CPU, TensorRT and DeepStream images carry the YOLO11n weights. Each
DeepStream image also carries the ONNX and the nvinfer parser for its
DeepStream version. The TensorRT engine is never in an image, because an
engine only runs on the board and TensorRT version that built it.

Run the image whose version matches the Signal K plugin. Both sides use the
`DetectionEvent` contract (`docs/event-schema.md`), and a matching release is
the only combination that has been tested together. The plugin's setup guide
always points at its own version.

## Install from a release

The steps are the same for every variant: download its compose file, save
the camera addresses, and start it. Run them on the computer that has the
GPU and can reach the cameras. Replace `0.1.0` with the plugin's version
(shown in the Signal K App Store):

```bash
mkdir -p ~/vision-ai && cd ~/vision-ai
curl -fsSLO https://github.com/KEGustafsson/vision-ai/releases/download/v0.1.0/docker-compose.deepstream.yml
```

```bash
(umask 077
touch .env && chmod 600 .env && cat > .env <<'EOF'
VISION_CAMERA_FORWARD_URL='rtsp://user:password@192.168.1.10:554/stream'
VISION_CAMERA_AFT_URL='rtsp://user:password@192.168.1.11:554/stream'
EOF
)
```

The addresses usually hold camera passwords, so `.env` is made readable by
you only before they are written, and nothing is written if that fails. Keep the single quotes: Compose reads an
unquoted value with a `$` in it as a variable, which would change a password
such as `pa$word`. A password containing `'` needs double quotes instead,
with `$` written as `$$`, `"` as `\"` and `\` as `\\`. The plugin's setup
guide writes the file this way for you.

```bash
docker compose -f docker-compose.deepstream.yml up -d
```

```bash
curl -s http://127.0.0.1:7000/health
```

Notes per variant:

- **Jetson (all three):** needs the NVIDIA Container Runtime
  (`docker info | grep -i runtime` lists `nvidia`). Set full power mode
  first, as in [Jetson setup](jetson-setup.md#1-prerequisites-on-the-jetson).
- **DeepStream:** the first start builds the TensorRT engine from the ONNX,
  which takes several minutes and a lot of RAM (on a Xavier, stop other heavy
  containers until `/health` answers). The engine is kept in a Docker volume,
  so later starts take seconds.
- **TensorRT (`jetson`):** build the engine once before the first start, into
  the `models` volume:

  ```bash
  docker compose -f docker-compose.jetson.yml run --rm vision-service \
    python3 scripts/export_engine.py --weights models/yolo11n.pt --imgsz 640
  ```

- **CPU and the Jetson variants** refuse to start without both camera
  addresses. In `cpu` mode a camera with no address falls back to the
  synthetic test picture, whose made-up person in the water would raise a real
  man-overboard alarm.
- **Mock** needs no `.env`. It makes up its targets, including that person in
  the water on the aft camera, so use it on a test server only.

Every stack listens on `127.0.0.1:7000` only, because the service has no
authentication. The Signal K plugin reaches it there and passes the video
through behind Signal K's own login. If Signal K runs in a container, give
that container host networking, or set the plugin's **Vision container base
URL** to an address it can reach.

### Signal K on another computer

Running both on the same computer is simplest. If Signal K runs elsewhere,
don't publish port 7000 on the boat's network: anyone on it could then watch
the cameras and change detection settings. Reach it through an SSH tunnel
from the Signal K computer instead:

```bash
ssh -N -L 7000:127.0.0.1:7000 user@vision-computer
```

and leave **Vision container base URL** at `http://localhost:7000`. Keep the
tunnel up across reboots with `autossh` or a systemd service, and log in with
an SSH key rather than a password. An authenticated reverse proxy (HTTPS
with a login) on the vision computer also works; then set the base URL to
that proxy's address.

## Update

Download the compose file of the new release and start it again. Docker
pulls the new image and replaces the container:

```bash
cd ~/vision-ai
curl -fsSLO https://github.com/KEGustafsson/vision-ai/releases/download/v0.2.0/docker-compose.deepstream.yml
docker compose -f docker-compose.deepstream.yml up -d
```

On DeepStream the container notices that the image changed and, on its first
start, copies the new image's ONNX, pgie configs and labels into its volume
and deletes the old engine, then builds a new one (several minutes again). It
keeps any files you added, such as another model's ONNX, but an edit made inside
the volume to one of the image's own files (say a pgie setting) is replaced:
keep such tuning in a source build, or re-apply it. The same happens when
you pull a new `edge` image, so it never runs the previous image's model or
settings.

On the TensorRT variant, build the engine again with the `export_engine.py`
step: its `models` volume is named per release (`vision-ai-models-0.2.0`, …),
so a new release starts without one rather than with one built for an older
TensorRT. On `edge` that volume is shared between builds; after pulling a new
`edge` image, run `docker compose -f deploy/docker-compose.jetson.yml down -v`
before building the engine again. List and remove old volumes with
`docker volume ls` and `docker volume rm`.

Update the Signal K plugin to the same version.

## Calibrate

The images use the calibration shipped in `vision-service/config/`. To set
the horizon, camera height, field of view and bearing offsets for the boat
(see [Geometry & calibration](geometry.md)), copy the files out, edit them,
and mount them back:

```bash
docker compose -f docker-compose.deepstream.yml cp vision-service:/opt/vision-service/config ./config
# edit config/deepstream.yaml (or cpu.yaml / jetson.yaml)
```

Then uncomment the `./config:/opt/vision-service/config:ro` line under
`volumes:` in the compose file and run `docker compose ... up -d` again.

## Build from source instead

The compose files at the repository root still build the images on the
board from a git checkout. Use them to change the code, pin a different base
image, or run offline. See [Jetson setup](jetson-setup.md) and
[DeepStream GPU pipeline](jetson-deepstream.md).

## How images are built

`.github/workflows/docker-images.yml` builds every variant on a native runner
for its architecture (GitHub's arm64 runners for the Jetson images, so nothing
is built under emulation), bakes in the YOLO11n weights from a pinned
Ultralytics release, and joins the per-architecture images into one tag. It
runs:

- on every pull request that touches `vision-service/` or `deploy/`: the mock
  and CPU images for amd64, built but not pushed;
- on every push to `main`: all variants, pushed as `edge-<variant>`;
- from `release.yml` on a version tag: all variants, pushed as
  `X.Y.Z-<variant>`.

The Jetson and DeepStream images are built on runners with no GPU. Building
them needs no GPU, but it means CI only proves they build: the pipeline itself
is verified on the boat (see [Onboard verification](onboard-verification.md)).

## Cut a release

1. Set the version in `signalk-plugin/package.json` (`npm version X.Y.Z
   --no-git-tag-version` in `signalk-plugin/`), and add a `## [X.Y.Z]`
   section to `signalk-plugin/CHANGELOG.md`.
2. Merge to `main`.
3. Tag and push:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

`release.yml` then checks that the tag matches `package.json` and that the
changelog has the section, runs the full CI suite on the tagged commit,
builds and pushes every image, and creates a GitHub Release with:

- the changelog section as release notes;
- the `deploy/` compose files pinned to `X.Y.Z`;
- the plugin's npm tarball (`signalk-vision-ai-X.Y.Z.tgz`).

Versions below 1.0.0 are marked pre-release, and they don't move the bare
`<variant>` tags.

Publishing the plugin to npm, which is what puts it in the Signal K App
Store, is a separate step: run `npm publish` in `signalk-plugin/`, or add
an npm token and a publish step to `release.yml`.

**The first time images are pushed**, GitHub creates the `vision-ai` package
as private. Make it public once (GitHub → your profile → Packages →
vision-ai → Package settings → Change visibility), or no boat can pull it
without logging in.
