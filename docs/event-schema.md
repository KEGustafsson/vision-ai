# Detection event contract

The vision container emits one `DetectionEvent` per processed frame, per camera,
over the WebSocket at `ws://<container>/ws/events`.

**The authoritative definition is `vision-service/app/schemas.py`** (Pydantic).
The JSON Schema the plugin validates against
(`signalk-plugin/schema/detection-event.schema.json`) is generated from it:

```bash
cd vision-service && python scripts/export_schema.py
```

A CI step / the plugin's `EventStream` validates every inbound event against
that schema, so the two sides cannot silently drift.

## Shape

```jsonc
{
  "schema_version": "1.0",
  "camera": "forward",                 // "forward" | "aft"
  "timestamp": "2026-05-29T12:00:00.123Z",
  "frame_seq": 48213,
  "frame_size": { "w": 1280, "h": 720 },  // processed-frame px (bbox/horizon_y space)
  "horizon_y": 324,                    // px row of the horizon; null if uncalibrated
  "inference": { "backend": "mock", "latency_ms": 4.2 },  // see backend values below
  "calibration_status": "ok",          // ok | uncalibrated | auto
  "targets": [
    {
      "track_id": 17,                  // stable across frames; null if untracked
      "label": "vessel",               // canonical marine label
      "coco_class": 8,
      "confidence": 0.88,
      "bbox": { "x": 980, "y": 360, "w": 120, "h": 46 },  // px, top-left origin
      "is_person_in_water": false,     // container-side MOB candidate rule
      "geometry": {
        "relative_bearing_deg": 3.2,   // + starboard / - port, incl. mount offset
        "range_m": 412.0,              // null if not estimable
        "range_method": "horizon",     // horizon | known_size | null
        "range_confidence": 0.6        // 0..1; plugin gates georeferencing on this
      },
      "pixel_velocity": { "vx": -2.1, "vy": 0.3 },  // px/frame
      "first_seen": null,
      "age_frames": 35
    }
  ]
}
```

## Field notes

- **`inference.backend`** is one of `mock`, `torch-cpu`, `torch-cuda`, `tensorrt`,
  or `deepstream` (the GPU NVMM pipeline). All emit this same schema.
- **`frame_size`** is the resolution everything else (bbox, `horizon_y`) is
  expressed in — the processed frame, not necessarily the raw sensor. For the
  `deepstream` backend this is `nvstreammux`'s output (`detector.mux_width` ×
  `mux_height`), and inference runs at `imgsz` independently of it.
- **`relative_bearing_deg`** already includes the camera's mounting offset
  (forward = 0°, aft = 180°), so the plugin only adds own heading.
- **`range_method`** lets the plugin treat `horizon` ranges (more reliable) and
  `known_size` ranges (coarse) differently; `range_confidence` gates whether a
  target is georeferenced at all (`minRangeConfidence`).
- **`is_person_in_water`** is decided in the container (person whose waterline is
  below the horizon) so MOB latency is one frame, not a round-trip.
- Labels are canonical marine terms from `app/detector/classmap.py`
  (`person`, `vessel`, `buoy`, …), independent of the underlying model's class
  ids — swap in a maritime-trained model without touching the plugin.
