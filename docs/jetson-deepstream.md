# DeepStream GPU pipeline (Jetson)

The `deepstream` backend — a fully GPU-resident alternative to the Ultralytics/
TensorRT `jetson` backend covered in
[Jetson setup & deployment](jetson-setup.md). It's a **different pipeline**,
not a tuning variant: both default to the same model generation (YOLO11n), but
a different tracker (NvDCF vs BoT-SORT), different image
(`Dockerfile.deepstream` vs `Dockerfile`), different compose file. It shares
only the basics with the
Ultralytics path — see [Jetson setup & deployment](jetson-setup.md) for:

- [§1 Prerequisites](jetson-setup.md#1-prerequisites-on-the-jetson) (JetPack,
  NVIDIA Container Runtime, power mode) — the DeepStream version comes from the
  image, not the host, so nothing beyond the container runtime has to be
  installed for this backend (see
  [Hardware targets](#hardware-targets) below).
- [§3 Camera URLs & calibration](jetson-setup.md#3-camera-urls--calibration).
- [§5 Autostart](jetson-setup.md#5-autostart).

Everything below is specific to this backend.

`VISION_MODE=deepstream` runs a fully GPU-resident pipeline instead of the
Python/TensorRT one: frames stay in NVMM end to end.

![DeepStream GStreamer pipeline, fully GPU-resident in NVMM: rtspsrc → nvv4l2decoder → optional nvdewarper → nvstreammux (batches both cameras) → nvinfer (TensorRT) → nvtracker (NvDCF); then per camera after nvstreamdemux: nvvideoconvert (RGBA) → a pad probe that reads metadata only → nvstreamdemux → nvdsosd (GPU overlay) → nvjpegenc (I420 to JPEG) → appsink → LatestFrame. Detection metadata also feeds the bearing/range geometry into the DetectionEvent without copying any pixels.](images/deepstream-pipeline.svg)

Overlays are drawn on the GPU (`nvdsosd`) and the MJPEG is encoded on
the NVJPG block (`nvjpegenc`), so the CPU only handles the finished JPEG bytes —
no per-frame pixel copy (the lone exception is throttled auto-horizon detection,
~1/s, skipped when `horizon_y` is calibrated). It exposes the same API and
`DetectionEvent` contract as the default backend.

## Hardware targets

The backend runs on **both** JetPack generations. The pipeline, the application
code, `config/deepstream.yaml` and the `DetectionEvent` contract are identical
on the two — only the platform layer (base images, DeepStream/CUDA/TensorRT
versions, and therefore the container's Python) differs, and all of that
difference is confined to the Dockerfile and compose file:

| | Orin Nano Super | Xavier NX |
|---|---|---|
| L4T / JetPack | r36.x / JetPack 6.1–6.2 | r35.x / JetPack 5.1.x |
| Compose file | `docker-compose.deepstream.yml` | `docker-compose.deepstream.xavier.yml` |
| Dockerfile | `vision-service/Dockerfile.deepstream` | `vision-service/Dockerfile.deepstream.xavier` |
| DeepStream | 7.1 (`deepstream:7.1-samples-multiarch`) | 6.3 (`deepstream-l4t:6.3-samples`) |
| CUDA / TensorRT | 12.6 / 10.3 | 11.4 / 8.5.2 |
| Export base | `ultralytics/…:latest-jetson-jetpack6` | `ultralytics/…:latest-jetson-jetpack5` |
| Container OS / Python | Ubuntu 22.04 / 3.10 | Ubuntu 20.04 / 3.8 |
| pyds | 1.2.0 (cp310) | 1.1.8 |
| nvinfer parser `.so` | committed, built on the host | compiled during the image build |
| L4T multimedia libs | from the host (CSV mount) | from the host, with an in-image fallback |
| Python deps | `requirements-deepstream.txt` + `constraints.txt` | `requirements-deepstream-xavier.txt` + `constraints-xavier.txt` |

DeepStream 6.3 is the last release for JetPack 5 (6.4 and later require
JetPack 6), and it fixes the container's interpreter at Python 3.8 because the
official `pyds` binding is compiled against it.

> **Version-pairing note.** NVIDIA pairs DeepStream 6.3 with JetPack 5.1.2 /
> L4T **35.4.1**. The board this was built and verified on runs L4T **35.6.4**,
> a later JetPack 5.1.x maintenance release, so this is a *supported-adjacent*
> rather than a documented-by-NVIDIA combination. It is the only option that
> exists — there is no newer DeepStream for JetPack 5, and downgrading L4T to
> 35.4.1 to satisfy the pairing on paper would mean reflashing the boat's
> board. It is verified working end to end (see
> [Verified on this hardware](#verified-on-this-hardware)), and every L4T
> ABI-sensitive piece is pulled from the host's **own** release, not 35.4.1:
> `L4T_RELEASE` must track the board (`head -1 /etc/nv_tegra_release`), because
> the glue libraries below talk to the running kernel driver and deliberately
> mismatching them would be worse than the version-pairing gap it papers over.
> If you move to a different L4T, re-run the verification rather than assuming
> it carries. That costs nothing in the
application: every module already carries `from __future__ import annotations`
and the Pydantic models use `Optional`/`Dict`/`List`, so the same sources run
unchanged — but the dependency versions have to be the last ones that still
publish 3.8 wheels, which is what `constraints-xavier.txt` pins (and why it
exists as a second file rather than a tweak to `constraints.txt`). CI still runs
the test suite on 3.11.

Everything below applies to both boards unless a section says otherwise; see
[Xavier NX notes](#xavier-nx-notes) for the JetPack 5 specifics.

**Tested hardware.** The JetPack 5 path is verified on a **Jetson Xavier NX**
(L4T R35.6.4) — see [Verified on this hardware](#verified-on-this-hardware).
Nothing in it is NX-specific, so other JetPack 5 Jetsons (AGX Xavier, or an Orin
still on JetPack 5) are *expected* to work, but they are **untested**: at
minimum the `L4T_SOC` build arg changes (`t194` → `t234` for Orin) and the
`nvpmodel` mode IDs differ per board and JetPack release, so the power-mode
command below does not carry over unchanged. Treat those boards as needing their
own bring-up rather than as supported configurations.

## Run

The image builds **from a clean clone in a single command** — no prior
`vision-ai:jetson` build, no manual ONNX export, no parser-compile step:

```bash
export VISION_CAMERA_FORWARD_URL="rtsp://user:pass@192.168.1.10:554/stream"
export VISION_CAMERA_AFT_URL="rtsp://user:pass@192.168.1.11:554/stream"

# Orin Nano Super (JetPack 6)
docker compose -f docker-compose.deepstream.yml up -d --build
# Xavier NX (JetPack 5)
docker compose -f docker-compose.deepstream.xavier.yml up -d --build
```

Use the file that matches the board, and only one at a time per checkout: both
bind-mount the same `vision-service/deepstream/` and would fight over the
TensorRT engine in it.

For a reproducible / air-gapped build, pin the export base to a digest by
exporting `VISION_JETSON_BASE` before the command (it feeds the `EXPORT_BASE`
build arg — resolve the aarch64 digest on the board):

```bash
export VISION_JETSON_BASE="ultralytics/ultralytics@sha256:<digest>"
docker compose -f docker-compose.deepstream.yml up -d --build
```

What the single build does for you:

- **ONNX export** — the build's stage-1 `export` runs on the public Ultralytics
  Jetson base (the same base the jetson path uses, so it's already cached if you
  built that image first; otherwise it's pulled fresh) and fetches the YOLO11n
  weights itself. To build OFFLINE, drop the weights at
  `vision-service/models/yolo11n.pt` first (`python3 scripts/download_models.py
  --model yolo11n.pt`) and the build uses them as-is.
- **Custom parser** — on JetPack 6 the committed
  `deepstream/libnvdsinfer_custom_impl_Yolo.so` is baked in, so no compile step
  is required; rebuild it only when the DeepStream version changes with
  `vision-service/scripts/build_yolo_parser.sh` (needs host nvcc + the DS 7.1 SDK
  at the same versions; the DS samples image has none). On JetPack 5 that
  committed `.so` is the wrong ABI (DeepStream 7.1 / CUDA 12.6 / TensorRT 10), so
  `Dockerfile.deepstream.xavier` **compiles its own** from the same pinned
  DeepStream-Yolo commit as the ONNX — nothing to run on the host, nothing extra
  to commit. Either way the parser is installed at
  `/opt/vision-service/lib/libnvdsinfer_custom_impl_Yolo.so`, which is what every
  `deepstream/pgie_*.txt` names in `custom-lib-path`: an absolute path outside the
  `deepstream/` bind-mount, so the host's copy can never shadow the image's with a
  build for the other board (an ABI mismatch that would show up as zero detections
  rather than an error).
- **TRT engine** — nvinfer auto-builds it next to the ONNX in the bind-mounted
  `deepstream/` on first start, so it persists across container recreates.

## Tuning

Tuning lives in `config/deepstream.yaml` (bind-mounted, takes effect on restart):

- **`detector.mux_width` / `mux_height`** — nvstreammux output = display +
  geometry resolution (native camera res, default `1280x960`). nvinfer rescales
  this to `imgsz` internally on the GPU, so detection still runs at `imgsz` while
  bboxes and `horizon_y` stay in native pixels. Set these to your stream size.
- **GPU lens correction** — when a camera sets `undistort: true`, an `nvdewarper`
  stage straightens barrel (`undistort_k1`) + rotation (`undistort_rotation_deg`)
  in NVMM before inference. A starter config is generated from those knobs;
  override with `camera.dewarper_config` once tuned on the hardware.
- **`detector.mux_batch_timeout_ms`** — nvstreammux batch timeout (default 40 ms).
  Kept short so both cameras pipeline at full input rate; a per-camera PTS guard
  in the probe drops any muxer frame repeats so output never exceeds the camera's
  real frame rate.
- **Tracker working resolution** — derived, not configured: NvDCF runs at
  `imgsz` wide and a height that keeps the `mux_width:mux_height` aspect ratio,
  both rounded to NvDCF's 32-px grid (1280×960 at `imgsz: 768` → 768×576). The
  tracker therefore sees a uniform downscale of the camera frame rather than a
  squashed one. Logged once per pipeline build
  (`nvtracker NvDCF working resolution …`).
- **nvinfer pre-scale filter** — every `deepstream/pgie_*.txt` sets
  `scaling-filter=1` (bilinear on the VIC). nvinfer's default is
  nearest-neighbour, which on the 1280→768 letterbox simply discards 40% of the
  source rows and aliases exactly the small, distant targets the larger `imgsz`
  exists to keep; bilinear is what Ultralytics uses for its own letterbox, so
  the network gets the input it was trained on. Keep it when adding a model
  config (`tests/test_pgie_configs.py` pins it).

Two properties of the graph itself are worth knowing when reading
`tegrastats`/`jtop` or the `latency_ms` in events:

- **The host-side probe has its own thread.** A pad probe runs on whichever
  thread pushes the buffer, and upstream of it the muxer, `nvinfer`,
  `nvtracker` and the RGBA convert form one GPU chain. The probe (stabilizer,
  geometry, event serialisation, overlay meta — milliseconds of Python per
  batch) therefore hangs off a two-buffer `queue` placed after the RGBA
  capsfilter, so inference of batch *N+1* overlaps the host processing of batch
  *N* instead of waiting for it. The queue is not leaky: a slow probe
  back-pressures the leaky per-camera source queues, which shed load at the
  decoder (a stale frame skipped) rather than dropping a batch that was already
  inferred and tracked.
- **NVDEC runs at max performance.** Each `nvv4l2decoder` is created with
  `enable-max-performance` so the decoder does not clock down between frames;
  it costs a little power and buys steadier per-frame decode latency. The
  property is set best-effort — a plugin build without it just uses its default.

The deepstream compose bind-mounts `app/`, `config/`, `deepstream/`, and `models/`,
so code, config, model ONNX/labels, and the built TRT engine all live on the host
and survive recreates — a plain **restart** picks up app/config edits and an
engine rebuild only happens when the ONNX actually changes. An image rebuild is
only needed to bake changes for a clean redeploy. ⚠ The Orin is NVMM-tight: stop
the GPU overlay co-tenants before recreating the container, or buffer-pool
allocation can OOM (`failed to activate bufferpool`); restart them after.

## Runtime behaviour

- **Disable detection (master off):** unlike the CPU/Jetson backend (whose workers
  release the camera capture device), DeepStream transitions the whole GStreamer
  graph to **PAUSED** — decoders and `nvinfer` stop pulling data, so disabling
  actually drops the GPU/thermal load. Re-enabling returns it to PLAYING. A
  pipeline that recovers from a fault while disabled comes back PAUSED.
- **Auto-recovery:** a fatal GStreamer error or EOS (e.g. a transient RTSP/decoder
  glitch) no longer takes detection down until a manual container restart. A
  supervisor rebuilds the pipeline with exponential backoff (2 s → 30 s) and keeps
  retrying. `GET /health` reports `pipeline_restarts` and `pipeline_last_error` so
  a flapping feed is visible; `status` goes `degraded` while restarts have occurred.
- **Non-root:** the runtime image runs as a non-root user (UID 10001, in the
  `video` group for GPU access). nvinfer writes the TRT engine into the
  bind-mounted `deepstream/` and `models/`, so those host dirs must be writable by
  UID 10001 (as the jetson image already requires for `models/`). Run
  `vision-service/scripts/fix_host_permissions.sh` once per host to grant this via
  ACL (`setfacl`) without changing ownership or opening the dirs to everyone. A
  container stuck restart-looping with `WARN could not write .../yolo11n_ds.onnx`
  or `ERROR: Cannot access ONNX file` in its logs means this hasn't been run (or
  the dirs were recreated since). If your nvidia container runtime denies GPU
  access to non-root, set `user: root` on the deepstream compose service as a
  documented exception instead.

## Xavier NX notes

JetPack 5 specifics. Everything not listed here is identical to the Orin path.

### Prerequisites

- JetPack 5.1.x flashed (L4T r35.x) — check with `cat /etc/nv_tegra_release`.
- Docker + the **NVIDIA Container Runtime**; confirm with
  `docker info | grep -i runtime` showing `nvidia`.
- Nothing else. DeepStream 6.3, CUDA and TensorRT all come from the image; the
  host needs no `deepstream-6.3` package and no `nvcc` (unlike the Orin path,
  whose parser is compiled host-side).
- Put the board in its highest power mode before running — the Xavier NX
  defaults well below its ceiling:

  ```bash
  sudo nvpmodel -m 8 && sudo jetson_clocks     # 20 W, 6 CPU cores (NX dev kit)
  nvpmodel -q                                  # confirm
  ```

### Build & run

```bash
export VISION_CAMERA_FORWARD_URL="rtsp://user:pass@192.168.1.10:554/stream"
export VISION_CAMERA_AFT_URL="rtsp://user:pass@192.168.1.11:554/stream"
docker compose -f docker-compose.deepstream.xavier.yml up -d --build
```

The pipeline is hardware-accelerated end to end exactly as on the Orin —
`nvv4l2decoder` (NVDEC) → `nvdewarper` → `nvstreammux` → `nvinfer` (TensorRT) →
`nvtracker` (NvDCF, GPU) → `nvdsosd` (GPU) → `nvjpegenc` (NVJPG) — with pixels
staying in NVMM the whole way; only the finished JPEG bytes reach the CPU.

Expect the **first** build to be slow and bandwidth-hungry: it pulls the
Ultralytics JetPack 5 base and the ~3 GB DeepStream 6.3 base, exports the ONNX,
and compiles the nvinfer parser. Subsequent builds reuse those layers.

The image build deliberately uses **no Dockerfile heredocs** (the container
entrypoint is a committed script, `scripts/deepstream_entrypoint.sh`). Heredocs
need BuildKit's dockerfile 1.4+ frontend, which the Docker shipped with
JetPack 5 does not have — with them, the build fails at parse time with
`unknown instruction: ECHO`. The JetPack 6 image was moved onto the same script
so the two stay identical.

### L4T multimedia libraries

Four libraries that the `nv*` GStreamer plugins need ship in neither the
DeepStream image nor DeepStream itself — they belong to the **host's** L4T
GStreamer userspace and are normally injected by the NVIDIA container runtime
from its CSV mount list (`/etc/nvidia-container-runtime/host-files-for-container.d/l4t.csv`):

| library | package | needed by |
|---|---|---|
| `libnvdsbufferpool.so.1.0.0` | `nvidia-l4t-multimedia` | `nvstreammux`, `nvvideoconvert`, `nvdewarper`, `nvstreamdemux` |
| `libtegrav4l2.so` | `nvidia-l4t-multimedia` | `nvv4l2decoder`, `nvjpegenc` (indirectly — see below) |
| `libgstnvdsseimeta.so.1.0.0` | `nvidia-l4t-gstreamer` | `nvv4l2decoder`, `nvstreammux` |
| `libgstnvexifmeta.so` | `nvidia-l4t-gstreamer` | `nvjpegenc` |

On a stock **Ubuntu** JetPack host they are simply present and nothing special
happens. On a **Yocto / meta-tegra** rootfs — which is what this boat's Xavier NX
runs — the L4T GStreamer userspace is not installed at all, so those CSV entries
resolve to nothing and the mounts silently do not happen.

They fail in two different, equally unhelpful ways.

**Missing at link time** — the element cannot be created at all:

```text
RuntimeError: Cannot create GStreamer element 'nvstreammux' (name='mux').
```

`nvinfer`, `nvtracker`, `nvdsosd` and `nvof` load fine, while `nvstreammux`,
`nvstreamdemux`, `nvvideoconvert`, `nvdewarper`, `nvv4l2decoder` and `nvjpegenc`
do not. Confirm with:

```bash
docker run --rm --runtime nvidia --entrypoint ldd \
    nvcr.io/nvidia/deepstream-l4t:6.3-samples \
    /opt/nvidia/deepstream/deepstream/lib/gst-plugins/libgstnvvideoconvert.so | grep "not found"
```

**Missing at *runtime* dlopen — `libtegrav4l2.so`.** Nothing links to it
directly, so every element still loads and the pipeline still builds; it fails
later, at state change. NVIDIA's `libv4l2` dlopens the NVDEC/NVJPG shim
`libv4l/plugins/nv/libv4l2_nvvideocodec.so`, and *that* is what needs
`libtegrav4l2.so`. Without it the shim never loads, so nothing claims
`/dev/nvhost-nvdec` and the decoder blames the device node:

```text
libv4l2: error getting capabilities: Inappropriate ioctl for device
ERROR ... nvv4l2decoder0: Error getting capabilities for device '/dev/nvhost-nvdec':
  It isn't a v4l2 driver. Check if it is a v4l1 driver.
RuntimeError: DeepStream pipeline failed to enter PLAYING state.
```

The device node is present and fine — the library is not. Confirm with:

```bash
docker run --rm --runtime nvidia --entrypoint ldd vision-ai:deepstream-xavier \
    /usr/lib/aarch64-linux-gnu/libv4l/plugins/nv/libv4l2_nvvideocodec.so | grep "not found"
```

`Dockerfile.deepstream.xavier` therefore carries all four files itself: its
`l4tglue` stage pulls `nvidia-l4t-multimedia` and `nvidia-l4t-gstreamer` from the
public L4T apt repo for the board's own release and extracts exactly those four
`.so`s (never installs the packages — that would drag in `nvidia-l4t-core`/`-cuda`
and fight the CUDA the DeepStream base ships). Because they come from the same
L4T release the board was flashed from, they are ABI-matched to the kernel
driver; and on a host that *does* provide them the runtime's bind mounts land on
these exact paths and win, so the fallback never overrides a working host.

To re-audit this on a different board — every CSV entry resolvable in neither the
rootfs nor the runtime's `alt-roots`:

```bash
CSV=/etc/nvidia-container-runtime/host-files-for-container.d/l4t.csv
ALT=/usr/share/nvidia-container-passthrough      # from config.toml's alt-roots
awk -F', ' '$1=="lib"||$1=="sym"{print $2}' "$CSV" | while read -r p; do
  [ -e "$p" ] || [ -e "$ALT$p" ] || echo "$p"
done | sort
```

On this board that lists 19 paths; the other 15 are OpenMAX, EGL/Vulkan/KMS
display and Orin-only (`tegra23x`) firmware, none of which this headless
pipeline touches.

Set the release/SoC in the compose file if the board is not an R35.6 Xavier:

```bash
head -1 /etc/nv_tegra_release        # "# R35 (release), REVISION: 6.4" -> r35.6
export VISION_L4T_RELEASE=r35.6      # compose build arg L4T_RELEASE
export VISION_L4T_SOC=t194           # t194 = Xavier, t234 = Orin
```

Verify after building — this is the single most useful post-build check:

```bash
for e in nvstreammux nvstreamdemux nvvideoconvert nvdewarper \
         nvv4l2decoder nvjpegenc nvinfer nvtracker nvdsosd; do
  docker run --rm --runtime nvidia --entrypoint gst-inspect-1.0 \
      vision-ai:deepstream-xavier "$e" >/dev/null 2>&1 \
      && echo "OK   $e" || echo "MISS $e"
done
```

All nine must print `OK` before the pipeline can start.

### Verified on this hardware

Checked on the boat's Xavier NX (L4T R35.6.4, Yocto rootfs, Docker 20.10.25)
against a freshly built `vision-ai:deepstream-xavier`:

| check | result |
|---|---|
| ONNX export on the JetPack 5 base | YOLO11n exported at 768, 10.7 MB |
| nvinfer parser compile (CUDA 11.4 / TRT 8.5) | `NvDsInferParseYolo` symbol verified |
| `pyds` 1.1.8 + PyGObject import on Python 3.8 | OK |
| Python 3.8 dependency stack | all wheels, no source builds |
| App imports (`config`, `schemas`, `pipeline_deepstream`, `main`) | OK |
| All 13 pipeline elements loadable | OK |
| NVMM convert → `nvjpegenc` (NVJPG block) | ran, `NvMMLiteBlockCreate BlockType = 1` |
| `nvstreammux` → `nvtracker` (NvDCF) → `nvstreamdemux` → `nvdsosd` → `nvjpegenc` | ran to clean EOS |

Then confirmed again on a full deployment against the boat's two live Hikvision
domes (`docker compose -f docker-compose.deepstream.xavier.yml up -d`):

| check | result |
|---|---|
| Live RTSP ingest, both cameras | both NVDEC instances opened (`Opening in BLOCKING MODE` ×2), `camera_errors: {}` |
| nvinfer TensorRT engine build | built from the ONNX in ~19 min, 7.3 MB, persisted to the bind-mounted `deepstream/` |
| `nvdewarper` | configured and running per camera (`GPU dewarp via nvdewarper`), pipeline reached PLAYING |
| End-to-end detections | `/events/recent` returns targets with label, confidence, bearing and range |
| `DetectionEvent` output | `schema_version: 1.0`, consumed by the SignalK plugin |
| Hardware JPEG via `nvdsosd` → `nvjpegenc` | `/snapshot/<cam>` returns a ~120 KB JPEG |
| Steady state | `/health` `status: "ok"`, `pipeline_restarts: 0`, ~5.9 Hz per camera |

**Still unverified:** sustained multi-hour running, behaviour under way (all of
the above was measured at a berth), and track continuity in chop — see
[Tracker parameter differences](#tracker-parameter-differences), which is the
one place this board is known to behave differently from an Orin. Run
[onboard verification](onboard-verification.md) before relying on it.

### RAM

The NX dev kit has 8 GB shared between CPU and GPU and, by default, **no swap**.
The first start is the tight moment: nvinfer builds the TensorRT engine from the
ONNX, which is far more memory-hungry than steady-state inference and takes
appreciably longer here than on an Orin (the image's healthcheck start-period is
raised to 600 s to cover it). Give it room:

```bash
free -h                                  # check what is actually free
docker stop <other GPU/NVMM tenants>     # e.g. overlay/streaming containers
docker compose -f docker-compose.deepstream.xavier.yml up -d --build
# …wait for `GET /health` to answer, then restart the co-tenants
```

The engine is written into the bind-mounted `deepstream/`, so this cost is paid
once — later starts deserialise it in seconds. If the build OOMs anyway, add
swap (`sudo systemctl enable --now nvzramconfig`, or a swapfile) before
retrying.

TensorRT engines are **not portable** across boards, TensorRT versions or
JetPack generations, and the engine lives in the bind-mounted `deepstream/`
under a platform-neutral filename. Moving a checkout between an Orin and a
Xavier therefore leaves a stale `deepstream/*_gpu0_fp16.engine` this board
cannot deserialise. That is handled, not fatal — nvinfer logs the failure and
rebuilds from the ONNX:

```text
deserialize backend context from engine from file :...engine failed, try rebuild
Info from NvDsInferContextImpl::buildModel(): Trying to create engine from model files
```

The cost is the rebuild (~19 min here), not an outage. Delete the stale engine
first if you would rather not wait for the fallback.

### Inference size

`detector.imgsz: 768` is shared with the Orin and is what
`Dockerfile.deepstream.xavier` bakes into the ONNX (build arg `YOLO_IMGSZ`).
The Xavier NX is the slower board, so if `GET /health` shows the event rate
falling short of `server.target_fps`, or `tegrastats` shows the GPU pinned, drop
to 640 — in **both** places, they must match:

```bash
# 1. config/deepstream.yaml:  detector.imgsz: 640
# 2. rebuild the ONNX at the same size and clear the stale engine.
#    The trailing * is doing real work here: nvinfer names the engine after the
#    ONNX (yolo11n_ds.onnx_b2_gpu0_fp16.engine), so this one glob removes BOTH
#    the old ONNX and the engine built from it. Drop the * and you keep a 768
#    engine next to a 640 ONNX.
rm -f vision-service/deepstream/yolo11n_ds.onnx*
docker compose -f docker-compose.deepstream.xavier.yml build \
    --build-arg YOLO_IMGSZ=640
docker compose -f docker-compose.deepstream.xavier.yml up -d
```

(`imgsz` also drives the NvDCF `tracker_width`; see the coupling note in
`config/deepstream.yaml`.)

### Tracker parameter differences

`deepstream/nvdcf_config.yml` is shared by both boards. The NvDCF library parses
it with OpenCV `FileStorage` and **silently ignores keys it does not know**, so
two DeepStream 7.x parameters in it have no effect under DeepStream 6.3. Both are
logged once at startup and are expected:

```text
!! [WARNING][NvTrackerParamsTrajectoryManager] Unknown param found: useUncoveredAreaDetection
!! [WARNING][VisualTrackerConfigParams] Unknown param found: searchRegionPaddingScale
```

- `useUncoveredAreaDetection: 0` — **no difference.** We set it off and 6.3 has
  no such feature, so both boards behave identically.
- `searchRegionPaddingScale: 2` — **a real difference.** It exists nowhere in
  6.3's stock presets, so NvDCF falls back to its own internal search-region
  size instead of the value tuned for wave-induced apparent motion. There is no
  6.3 equivalent to set in its place. Treat track continuity in chop as
  **unvalidated on the Xavier**: re-check it when running
  [onboard verification](onboard-verification.md) on that board, and watch for
  more ID churn on vessels than the Orin shows.

You will also see two plugin-scanner warnings on every start
(`libnvdsgst_udp.so: librivermax.so.0` and
`libnvdsgst_inferserver.so: libtritonserver.so`). Both are DeepStream plugins
this pipeline does not use, absent by design from the `-samples` base. Ignore
them.

### Optical flow

The Xavier SoC carries an optical-flow engine and DeepStream 6.3 ships the
`nvof` element, so `detector.optical_flow` is expected to work here too — but it
has **not** been validated on this board (the hardware validation below was done
on the Orin), and it is off by default. If you enable it, the fail-safe path
applies unchanged: a missing or failing `nvof` is logged, surfaced in `/health`,
and the pipeline runs without it.

## Optical flow (OFA)

The Orin SoC carries a dedicated **Optical Flow Accelerator** — a hardware block
separate from the GPU and from NVDEC. DeepStream drives it with the `nvof`
element, which emits a map of block-level motion vectors per frame. The vision
service can turn that map into **one robust global image-motion vector per
camera** and report it in `GET /health`.

**This is a measurement only.** Nothing in detection, tracking, geometry,
bearing/range, collision logic, alerting or the `DetectionEvent` contract reads
it. OFA is *not* a tracker and is *not* a replacement for NvDCF —
`deepstream/nvdcf_config.yml` is untouched by this feature.

### Enable

`config/deepstream.yaml`:

```yaml
detector:
  optical_flow: true            # default false
  # optical_flow_required: false      # true => /health degrades without flow
  # optical_flow_preset_level: 0      # 0 = fast, 1 = medium (2 is dGPU-only)
  # optical_flow_stale_ms: 2000       # age at which flow is reported stale
```

Restart the container. With `optical_flow: false` (the default) **no OFA element
is created at all** and the pipeline is exactly what it was before the feature
existed.

### Where nvof sits

```text
nvstreammux → nvinfer → nvtracker → nvvideoconvert(NV12/NVMM) → nvof → nvvideoconvert(RGBA) → …
```

One `nvof` on the batched stream, spliced in **after the tracker**, three
deliberate choices:

- **After nvstreammux, not per camera.** `nvof` publishes its result as user meta
  on `NvDsFrameMeta.frame_user_meta_list`, and frame meta only exists inside the
  `NvDsBatchMeta` that nvstreammux creates — an `nvof` placed on a per-camera
  branch before the muxer has nowhere to attach its output. NVIDIA's multi-source
  optical-flow sample and their Orin Nano reference pipeline both put `nvof`
  after the muxer, and the plugin emits **one flow map per source** (one
  `NvDsOpticalFlowMeta` per `NvDsFrameMeta`), so each camera still gets its own
  temporal flow stream — the per-source isolation lives in the plugin's
  per-batch-slot state rather than in separate elements. ⚠ This is the one part
  of the design that still wants an on-hardware check: see *Multi-camera* below.
- **After nvtracker, not before nvinfer.** nvinfer therefore sees byte-for-byte
  the same buffer it sees today — enabling OFA cannot change a single detection.
- **NV12 in NVMM.** `nvof` accepts NV12 only, so an `nvvideoconvert` +
  capsfilter (`video/x-raw(memory:NVMM),format=NV12`) precedes it. That is a GPU
  colour conversion inside NVMM — no host copy — and is a near no-op when
  dewarping is off (the branch is already NV12). With `undistort: true` the
  dewarper's RGBA is converted here, so flow is measured on the **corrected**
  image, in the same coordinate space detection and tracking used.

### Diagnostics

`GET /health` gains one entry per camera:

```jsonc
"optical_flow": {
  "forward": {
    "enabled": true,
    "state": "active",      // disabled | no_data | active | stale | error
    "active": true,
    "global_dx": 1.42,      // median flow, px per frame interval
    "global_dy": -0.31,
    "vectors": 51840,       // vectors that survived filtering
    "confidence": 0.98,     // share of the frame's vectors kept (coverage)
    "age_ms": 42,
    "error": null
  },
  "aft": { "…": "…" }
}
```

| `state` | meaning |
|---------|---------|
| `disabled` | feature off in config — no `nvof` element exists |
| `no_data` | enabled, no flow metadata received yet (normal right after start) |
| `active` | flow metadata received within `optical_flow_stale_ms` |
| `stale` | enabled, last metadata older than that |
| `error` | `nvof` unavailable, or metadata parsing failed (see `error`) |

Stale or missing flow does **not** mark the service unhealthy — it is an
optional diagnostic, not part of the detection path. Set
`optical_flow_required: true` to opt into `status: "degraded"` when flow is not
active.

### Failure behaviour

Fail-safe by default. If `nvof` cannot be created (plugin missing, no OFA block)
the pipeline is built **without** it, the reason is logged once and surfaced as
`state: "error"` in `/health`, and detection/tracking continue untouched. If a
fatal bus error comes from the OFA elements (e.g. caps negotiation), OFA is
dropped and the supervisor's next rebuild comes up without it. With
`optical_flow_required: true` the opposite happens on purpose: a missing `nvof`
fails startup with an error that names OFA, and a runtime OFA fault keeps the
pipeline retrying with OFA rather than silently running without it.

The first frame after startup, an RTSP reconnect, a DeepStream rebuild or a
detection off→on toggle legitimately carries **no** flow metadata (optical flow
needs a previous frame). That is reported as `no_data`, not as an error, and
every rebuild resets each camera's flow history so no estimate survives from the
previous pipeline epoch.

### Vector representation

`nvof` reports one vector per 4×4 pixel block (the only grid size DeepStream
supports) as a pair of **S10.5 fixed-point** int16s: the pixel value is
`raw / 32.0` (raw `32` → `1.0` px). `pyds.get_optical_flow_vectors()` widens the
raw int16s to float32 but does **not** scale them — the `/32` is applied in
`app/motion/optical_flow.py`, which is where the unit tests pin it.

The reported `global_dx`/`global_dy` are the **median** of the frame's valid
vectors, not the mean: swell, spray, wakes, glitter and independently moving
vessels put a large minority of grossly different vectors in every marine frame,
and a mean follows them. Filtering before the median is deliberately simple:
drop malformed and non-finite components, drop magnitudes above 128 px/frame,
take the median. (RANSAC/affine ego-motion models, per-object flow and
object-mask exclusion are explicitly out of scope here.)

NVIDIA notes that the quantisation floor makes a genuinely static scene report
±0.5 px rather than exactly 0 — treat sub-pixel magnitudes as noise.

### Validation on hardware

✅ **Performed 2026-08-10** on the boat's Orin with both cameras (`forward`,
`aft`) live, `optical_flow: true` in `config/deepstream.yaml`. All five items
below are confirmed for this board's actual running configuration; the one gap
is `undistort: false`, which this boat doesn't run in production so it was
never exercised live (see item 4).

**1. OFA engine is clocked (`jtop`)**

```bash
sudo pip3 install -U jetson-stats && sudo jtop      # page 1 (or the ENGINES page)
```

| step | expected | observed |
|------|----------|----------|
| `optical_flow: false`, cameras live | `OFA` off / idle | confirmed — idle in `jtop` |
| `optical_flow: true`, cameras live | `OFA` active, non-zero clock | confirmed — clocks up in `jtop` |
| back to `optical_flow: false` | `OFA` returns to idle/off | confirmed — drops back to idle |

`jtop`'s presentation of OFA varies by JetPack and jetson-stats version (it may
show as a clocked engine row rather than a utilisation percentage). Note also
that NVIDIA's own documentation is inconsistent about whether **Orin Nano**
exposes an OFA: the DeepStream FAQ lists AGX Orin and Orin NX, while the Orin
Nano forum thread reports the OFA working as a VPI backend and appearing in
`jtop`, and NVIDIA staff confirmed an `nvof` pipeline running on Orin Nano. This
board's `jtop` toggling the OFA engine on/off in lockstep with `optical_flow`
settles that for this module.

**2. Flow responds to real motion**

Confirmed via `POST /ptz/forward` (pan/tilt) while polling `GET /health`:

- Camera static → `global_dx`/`global_dy` sat at `0.0`, occasional ±0.5 px
  single-frame jitter (water surface noise), `confidence: 1.0`,
  `vectors: 76800` (the full 1280×960 frame at the 4×4 block grid) throughout.
- Pan (`pan: 0.4`, then `pan: -0.4`) → `global_dx` grew monotonically over the
  ~4 s of motion (e.g. `0 → -0.5 → -1.5 → -3.1` px, and the mirror-image
  `+1.0 → +2.0 → +0.9` px on the reverse pan), settling back to `0.0` within
  ~1 poll (≈0.5 s) of `POST /ptz/forward {"action":"stop"}`.
- Tilt (`tilt: -0.5`, physically down) → the same pattern in `global_dy`
  (`0 → -0.5 → -0.75`, settling back to `0.0` on stop). Tilting *up* first
  produced no motion — this dome was already at its mechanical tilt limit in
  that direction, not an OFA issue.
- The other camera (`aft`) stayed pinned at `0.0` throughout every `forward`
  move — see item 5.

**3. Sign convention — measured, not assumed**

```text
image content moves right  →  global_dx  =  +   (camera panned left, pan: -0.4)
image content moves left   →  global_dx  =  -   (camera panned right, pan: +0.4)
image content moves up     →  global_dy  =  -   (camera tilted down, tilt: -0.5)
image content moves down   →  global_dy  =  +   (inferred from the above; not directly measured — tilt-up hit the dome's mechanical limit)
```

**4. Dewarper on and off**

`undistort: true` (both cameras' shipped setting on this boat) confirmed:
the display stream is visibly flat where the raw feed shows heavy barrel
distortion — the nvdewarper correction is doing real geometric work — and
`/health.optical_flow` stayed `state: "active"` on both cameras throughout
(`vectors: 76800`, `confidence: 1.0`), with no caps-negotiation error in the
logs (the one bus error seen this session, `failed to activate bufferpool`,
is the pre-existing NVMM/VIC tightness noted in item 5, not a dewarper/OFA
caps mismatch). `undistort: false` was not separately tested — this boat runs
`true` on both cameras in production, so there was nothing to compare it
against live.

**5. Multi-camera**

Confirmed: panning/tilting `forward` alone left `aft.global_dx/dy` at `0.0` the
entire time (and vice versa isn't re-tested, but the topology is symmetric) —
the single batched `nvof` keeps each source's vectors isolated, so no per-camera
`nvof` split is needed. A pipeline rebuild (`POST /control {"enabled": false}`
then `{"enabled": true}`) showed `no_data` on both cameras for several seconds,
then both returned to `state: "active"` on their own. That rebuild happened to
hit a real, pre-existing failure mode on this board — `pipeline_last_error:
"failed to activate bufferpool"` (`pipeline_restarts: 1`), the known NVMM/VIC
allocation tightness on this Orin, not something OFA introduced — and the
pipeline's exponential-backoff supervisor recovered from it automatically, with
`/health` correctly reporting `status: "degraded"` while the restart was
recent (see the existing NVMM-tightness note earlier in this doc); this just
confirms OFA doesn't change how the supervisor handles it.

### Cost

Only the compact vector map reaches the CPU (~77k vectors at 1280×960 with the
4×4 grid — metadata, not pixels): one median per axis per camera per frame,
which is a fraction of a millisecond of NumPy work on an array that is already in
host memory. No frame surface is mapped, no OpenCV/CUDA optical flow is used, and
the pixels stay in NVMM through the whole OFA path. The added GPU work is one
NVMM colour conversion per batch; the flow itself runs on the OFA block, not the
GPU.

## Detection model selection

Exactly **one** detection model runs at a time — the two are never active
together. Pick it with `detector.model` in `config/deepstream.yaml` (or the
`VISION_DETECTOR_MODEL` env var) and restart:

| `detector.model` | nvinfer config | classes |
|---|---|---|
| `coco` (default) | `deepstream/pgie_yolo11n.txt` | 80 COCO (person, vessel, buoy, …) |
| `forward-watch` | `deepstream/pgie_forward_watch.txt` | 6 marine (ship, boat, debris, buoy, kayak, log) |
| `marine-surveillance` | `deepstream/pgie_marine_surveillance.txt` | 7 marine (boat, buoy, kayak, sailboat, speedboat, vessel, warship) |

All models use a deepstream-yolo-compatible YOLO ONNX and the SAME custom parser
(YOLO11 shares YOLOv8's output layout) — only the ONNX weights, input size, label
file, and `num-detected-classes` differ, so switching is purely a config change.
(`coco` is YOLO11n at 768×768; the marine models are YOLOv8n/s at 640×640.)

`coco` keeps person/man-overboard detection; `forward-watch` drops it but adds
debris/kayak/log. The `forward-watch.onnx` is **not** vendored — fetch it before
building the image so `COPY deepstream` bakes it in:

```bash
python3 scripts/download_forward_watch.py     # downloads AND converts → deepstream/forward-watch.onnx
```

> **The published forward-watch ONNX is a stock Ultralytics export and will not
> work with our parser as-is** — its output is `[1, 4+nc, 8400]`, but
> `NvDsInferParseYolo` expects `[1, 8400, 6]` (`x1,y1,x2,y2,score,class`), so fed
> raw it produces **zero detections**. The download script above automatically
> rewrites it via `scripts/convert_to_deepstream.py` (needs
> `pip install onnx onnxruntime`). To convert an ONNX you already have:
> `python3 scripts/convert_to_deepstream.py forward-watch.onnx --inplace`. After
> replacing the ONNX, delete any cached `*_gpu0_fp16.engine` so nvinfer rebuilds
> it. COCO needs no conversion — its build-time `export_yolo11.py` already emits
> the parser layout.

### marine-surveillance (train on-box)

Roboflow only exports the *dataset*, so this model is trained on the Jetson and
exported to a parser-ready ONNX by `training/train_marine_surveillance.py` (it does
a stock export then runs `vision-service/scripts/convert_to_deepstream.py`). The
dataset and runs land under `training/_train_marine_surveillance/`. Run it inside
an Ultralytics Jetson container — mount the repo root (the script reaches into
`vision-service/`) — with the GPU co-tenants stopped so it doesn't OOM:

```bash
docker stop vision-service-deepstream gstreamer_in_overlay gstreamer_out_overlay
docker run --rm -it --runtime nvidia --network host \
  -v $PWD:/work -w /work \
  ultralytics/ultralytics:latest-jetson-jetpack6 \
  bash -lc "pip install -q roboflow onnx onnxslim onnxruntime && \
    python3 training/train_marine_surveillance.py \
      --api-key \$ROBOFLOW_KEY --workspace WS --project PROJ --version N --batch 8"
```

It writes `vision-service/deepstream/marine-surveillance.onnx` and regenerates the label file.
Because `deepstream/` is bind-mounted (see the compose volumes), the new ONNX is
already visible to the container — just set `detector.model: marine-surveillance`
in `config/deepstream.yaml`, delete any stale
`marine-surveillance.onnx_b*_gpu0_fp16.engine`, and **restart** the container (no
image rebuild needed; nvinfer rebuilds the engine on first start). Rebuild the
image only when you want it baked in for a clean redeploy. **No person class →
man-overboard detection is off while this model is active**; keep `coco` if MOB
matters.

## Troubleshooting

- **`nvbufsurface: Unable to allocate HW buffer` (error 12)** →
  NVMM is fragmented/saturated (`tegrastats` shows a tiny `lfb`). Stop other NVMM
  tenants (e.g. GStreamer overlay containers) to defragment, start vision so it
  allocates its pools, then restart the others — the running pipeline survives
  them returning. Same condition can stall the first-run TRT engine build.
- **Empty target list / video** in the Captain view after switching
  backend → the SignalK plugin drops events failing schema validation; ensure its
  installed `schema/detection-event.schema.json` includes `deepstream` in the
  `Backend` enum and restart the plugin.
- **Container restart-looping on `WARN could not write .../yolo11n_ds.onnx`** →
  see the Non-root note above; run `vision-service/scripts/fix_host_permissions.sh`.
- **`optical flow: … continuing WITHOUT OFA`** in the logs, or
  `optical_flow.<cam>.state == "error"` in `/health` → the `nvof` element could
  not be created or failed at runtime; the message names the reason. Detection is
  unaffected. Check that the DeepStream optical-flow plugin is present
  (`gst-inspect-1.0 nvof`) and see [Optical flow (OFA)](#optical-flow-ofa).
- **`Cannot create GStreamer element 'nvstreammux'`** (JetPack 5) → the host's
  L4T multimedia libraries are missing and the image's fallback did not land; see
  [L4T multimedia libraries](#l4t-multimedia-libraries) and run the nine-element
  check there.
- **`unknown instruction: ECHO` while building** (JetPack 5) → the Dockerfile
  being built uses a heredoc and the board's Docker has no BuildKit dockerfile
  1.4+ frontend. `Dockerfile.deepstream.xavier` avoids heredocs on purpose; if
  you added one, move the script into `scripts/` and `COPY` it instead.
- **Detections stop entirely after moving a checkout between an Orin and a
  Xavier** → a stale TensorRT engine or the wrong-ABI parser. Delete
  `vision-service/deepstream/*_gpu0_fp16.engine` and rebuild the image; the
  parser itself is version-correct by construction (each image installs its own
  at `/opt/vision-service/lib/`, outside the bind-mount).
- **`Could not open library … libnvdsinfer_custom_impl_Yolo.so`** → the image was
  built without its parser, or `custom-lib-path` in `deepstream/pgie_*.txt` was
  changed back to a relative path (which resolves into the bind-mounted host
  `deepstream/`, i.e. the other board's build). It must stay absolute.

See also [Jetson setup & deployment](jetson-setup.md#troubleshooting) for issues
shared with the Ultralytics backend (OpenCV/GStreamer, `nvv4l2decoder`, low FPS).
