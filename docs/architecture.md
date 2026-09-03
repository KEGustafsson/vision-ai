# Architecture

## Design principles

1. **Two processes, one contract.** The vision container does everything that
   needs the GPU and the camera intrinsics (decode → detect → track → monocular
   geometry) and emits a single JSON event type. The SignalK plugin does
   everything that needs live navigation state (true bearing, georeferencing,
   AIS fusion, CPA/TCPA, notifications, publishing). The interface is the
   `DetectionEvent` schema and nothing else.

2. **Units convert once, at the boundary.** The container speaks its native
   units (degrees, metres, pixels). The plugin converts to SI (radians, metres,
   m·s⁻¹) before anything reaches SignalK.

3. **Three transports, each best-fit.** WebSocket for detection events, MJPEG
   over HTTP for annotated video, REST for control.

4. **Inference is never blocked by consumers.** The pipeline writes to ring
   buffers (latest annotated frame + recent events). Slow MJPEG/WS clients read
   from the buffers and can never stall the camera loop.

5. **Mock mode is first-class.** Mode is chosen by config/env and swaps only the
   frame source and inference backend. The whole stack runs on a laptop.

6. **Operator filters are applied in the container, before the boundary.** Object-
   type selection (`detectClasses`) and the minimum-range gate (`minTargetRangeM`)
   are owned by the plugin's config but pushed to the container via `POST /control`
   and enforced when the `DetectionEvent` is built — so the annotated overlay and
   the event stream always agree (you never see an object on the video that is
   absent from the target list). `person` is exempt from the range gate so a
   close man-overboard is never filtered.

## Data flow

![Container data flow: the camera frame from FrameSource.read() goes to Detector.detect_and_track() producing RawTrack[], then geometry adds bearing and range, operator filters drop unwanted classes and out-of-range targets, and the result becomes a DetectionEvent. That event fans out to an EventBuffer (served over WebSocket) and to annotate() which writes the LatestFrame (served as MJPEG). Both are ring buffers, so slow clients never stall the camera loop.](images/dataflow.svg)

### Inference backends

The frame-source + detector half of that flow has interchangeable backends,
chosen by `VISION_MODE`/`detector.backend`; all of them emit the identical
`DetectionEvent`, so the plugin is backend-agnostic:

- **`mock`** — synthetic frames + deterministic detector (laptop dev/demo).
- **`torch-cpu` / `torch-cuda`** — Ultralytics YOLO11 in PyTorch.
- **`jetson` (TensorRT)** — YOLO11 `.engine` via Ultralytics; GStreamer HW decode,
  inference + tracking (BoT-SORT with camera-motion compensation by default —
  see [Tracking stability](tracking-stability.md)) in Python.
- **`deepstream`** — a fully GPU-resident NVIDIA DeepStream pipeline
  (`pipeline_deepstream.py`). Default detection model is **YOLO11n** (COCO,
  768×768) — the same model generation as the other backends above (they run
  it at 640×640 instead); two YOLOv8-based marine alternatives are also
  selectable
  (see [Detection model selection](jetson-deepstream.md#detection-model-selection)).
  Frames stay in NVMM end to end:

  ![DeepStream GStreamer pipeline, fully GPU-resident in NVMM: rtspsrc → nvv4l2decoder → optional nvdewarper → nvstreammux (batches both cameras) → nvinfer (TensorRT) → nvtracker (NvDCF) → nvvideoconvert (RGBA) → a queue whose source-pad probe reads metadata only on its own thread → nvstreamdemux; then per camera: nvdsosd (GPU overlay) → nvjpegenc (I420 to JPEG) → appsink → LatestFrame. Detection metadata also feeds the bearing/range geometry into the DetectionEvent without copying any pixels.](images/deepstream-pipeline.svg)

  Both cameras are batched by `nvstreammux`; inference + GPU tracking run on the
  batch in one pass. `nvstreammux` outputs the native camera resolution (display
  + geometry coords) and `nvinfer` rescales to `imgsz` internally on the GPU.
  Lens correction (barrel + rotation) is applied on the GPU by `nvdewarper`
  before inference. A per-camera PTS guard in the probe drops `nvstreammux` frame
  repeats so output never exceeds the camera's input rate.

  The display path is also zero-copy: the pad probe reads detection metadata only
  (no pixel copy) and attaches GPU overlay metadata (`NvDsDisplayMeta`); a
  per-camera `nvdsosd` burns the boxes/labels/HUD onto the NVMM surface and
  `nvjpegenc` encodes the JPEG on the NVJPG block — so the CPU only ever sees the
  finished JPEG bytes (delivered via `appsink` → `LatestFrame`). The one host
  pixel access is auto-horizon detection, throttled to ~1/s per camera and skipped
  entirely when a camera has an explicit `horizon_y` calibration.

On the plugin side, each `DetectionEvent` is enriched in stages:

![Plugin-side enrichment of each DetectionEvent: enrichTarget() combines the relative bearing with own heading for a true bearing and own position + bearing + range for the target lat/lon; collectAisContacts() enumerates vessels.* with positions; fuse() correlates visual versus AIS into aisCorrelated or darkTarget; CpaEstimator.update() turns per-track ground velocity into CPA/TCPA and a threatLevel; then the NotificationManager raises MOB / dark-target / collision alerts (set and clear with hysteresis) and the Publisher emits synthetic AIS vessels (vessels.*) plus vision.fusion.* and vision.system.* paths.](images/plugin-enrichment.svg)

## Why "visual radar"?

A fixed camera with a known horizontal FOV gives a **relative bearing** from a
detection's pixel column. A camera at a known **height** above the waterline,
with a known/detected horizon row, gives a **range** from the depression angle
of the object's waterline. Combine those with the boat's position and heading
and each detection becomes a georeferenced contact — exactly what radar/AIS
produce — which is why the targets are published in a form that can sit next to
real radar/AIS on a chartplotter.

## Failure behaviour

- Container unreachable → plugin WS reconnects with exponential backoff; the
  MJPEG proxy returns 502; SignalK keeps running.
- No navigation data → targets still flow but `bearingTrue`/`position` are null
  and georeferenced features (fusion, CPA, MOB position) degrade gracefully.
- Uncalibrated horizon → range falls back to known-size estimation or is null;
  `calibration_status` reflects the state.
