from app.detector.classmap import (
    MODEL_COCO, MODEL_FORWARD_WATCH, MODEL_PGIE_CONFIG,
    is_person_in_water, label_for, label_for_model,
)


def test_registry_covers_both_models():
    assert MODEL_PGIE_CONFIG[MODEL_COCO] == "pgie_yolov8n.txt"
    assert MODEL_PGIE_CONFIG[MODEL_FORWARD_WATCH] == "pgie_forward_watch.txt"


def test_coco_model_mapping_keeps_raw_ids():
    assert label_for_model(MODEL_COCO, 0) == ("person", 0)
    assert label_for_model(MODEL_COCO, 8) == ("vessel", 8)
    assert label_for_model(MODEL_COCO, 80) == ("buoy", 80)


def test_forward_watch_mapping_remaps_ids():
    # ship + boat both collapse to the canonical "vessel" label, but keep
    # distinct synthetic class ids so the wire coco_class stays unambiguous.
    assert label_for_model(MODEL_FORWARD_WATCH, 0) == ("vessel", 81)
    assert label_for_model(MODEL_FORWARD_WATCH, 1) == ("vessel", 82)
    assert label_for_model(MODEL_FORWARD_WATCH, 2) == ("debris", 83)
    assert label_for_model(MODEL_FORWARD_WATCH, 3) == ("buoy", 84)
    assert label_for_model(MODEL_FORWARD_WATCH, 4) == ("kayak", 85)
    assert label_for_model(MODEL_FORWARD_WATCH, 5) == ("log", 86)


def test_forward_watch_ids_never_collide_with_coco():
    coco_ids = {label_for_model(MODEL_COCO, c)[1] for c in (0, 8, 80)}
    fw_ids = {label_for_model(MODEL_FORWARD_WATCH, c)[1] for c in range(6)}
    assert coco_ids.isdisjoint(fw_ids)


def test_label_for_resolves_wire_ids_from_both_models():
    assert label_for(8) == "vessel"      # COCO id on the wire
    assert label_for(83) == "debris"     # forward-watch synthetic id
    assert label_for(85) == "kayak"


def test_person_in_water_rule_unchanged():
    assert is_person_in_water("person", waterline_y=400, horizon_y=350) is True
    assert is_person_in_water("person", waterline_y=300, horizon_y=350) is False
    assert is_person_in_water("vessel", waterline_y=400, horizon_y=350) is False
    assert is_person_in_water("person", waterline_y=400, horizon_y=None) is False
