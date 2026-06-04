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

```text
camera → FrameSource.read() → Detector.detect_and_track() → RawTrack[]
       → geometry (bearing + range) → operator filters (classes, min range)
       → DetectionEvent
       → EventBuffer (→ WebSocket)   and   annotate() → LatestFrame (→ MJPEG)
```

### Inference backends

The frame-source + detector half of that flow has interchangeable backends,
chosen by `VISION_MODE`/`detector.backend`; all of them emit the identical
`DetectionEvent`, so the plugin is backend-agnostic:

- **`mock`** — synthetic frames + deterministic detector (laptop dev/demo).
- **`torch-cpu` / `torch-cuda`** — Ultralytics YOLOv8 in PyTorch.
- **`jetson` (TensorRT)** — YOLOv8 `.engine` via Ultralytics; GStreamer HW decode,
  inference + ByteTrack in Python.
- **`deepstream`** — a fully GPU-resident NVIDIA DeepStream pipeline
  (`pipeline_deepstream.py`). Frames stay in NVMM end to end:

  ```text
  rtspsrc → nvv4l2decoder → [nvdewarper] → nvstreammux → nvinfer (TRT)
          → nvtracker (NvDCF) → pad probe → annotate → LatestFrame
  ```

  Both cameras are batched by `nvstreammux`; inference + GPU tracking run on the
  batch in one pass. `nvstreammux` outputs the native camera resolution (display
  + geometry coords) and `nvinfer` rescales to `imgsz` internally on the GPU.
  Lens correction (barrel + rotation) is applied on the GPU by `nvdewarper`
  before inference. Only the displayed frame is copied to the CPU, after
  inference, for MJPEG annotation. A per-camera PTS guard in the probe drops
  `nvstreammux` frame repeats so output never exceeds the camera's input rate.

On the plugin side, each `DetectionEvent` is:

```text
enrichTarget()       relative bearing + own heading → true bearing
                     own position + bearing + range → target lat/lon
collectAisContacts() enumerate vessels.* with positions
fuse()               correlate visual vs AIS → aisCorrelated | darkTarget
CpaEstimator.update() per-track ground velocity → CPA / TCPA → threatLevel
NotificationManager  MOB / dark-target / collision (set & clear, hysteresis)
Publisher            vision.targets.* + vision.fusion.* + vision.system.*
```

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
