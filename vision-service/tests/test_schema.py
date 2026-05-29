from app.schemas import (
    BBox, Backend, DetectionEvent, FrameSize, Geometry, Inference, Target,
)


def _event() -> DetectionEvent:
    return DetectionEvent(
        camera="forward",
        timestamp="2026-05-29T12:00:00.000Z",
        frame_seq=1,
        frame_size=FrameSize(w=1280, h=720),
        horizon_y=324,
        inference=Inference(backend=Backend.mock, latency_ms=5.0),
        targets=[Target(
            label="vessel", coco_class=8, confidence=0.9,
            bbox=BBox(x=10, y=350, w=120, h=40),
            geometry=Geometry(relative_bearing_deg=3.2, range_m=400.0),
        )],
    )


def test_event_roundtrips_json():
    ev = _event()
    data = ev.model_dump(mode="json")
    again = DetectionEvent.model_validate(data)
    assert again.targets[0].label == "vessel"
    assert again.camera == "forward"


def test_event_accepts_arbitrary_camera_name():
    ev = _event()
    data = ev.model_dump(mode="json")
    data["camera"] = "port-quarter"
    again = DetectionEvent.model_validate(data)
    assert again.camera == "port-quarter"


def test_schema_generates():
    schema = DetectionEvent.model_json_schema()
    assert "targets" in schema["properties"]
    assert schema["properties"]["camera"]
