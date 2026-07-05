from app.detector.classmap import (
    MODEL_COCO,
    MODEL_FORWARD_WATCH,
    MODEL_LABELS,
    MODEL_MARINE_SURVEILLANCE,
    MODEL_PGIE_CONFIG,
    is_person_in_water,
    label_for,
    label_for_model,
)


def test_registry_covers_all_models():
    assert MODEL_PGIE_CONFIG[MODEL_COCO] == "pgie_yolo11n.txt"
    assert MODEL_PGIE_CONFIG[MODEL_FORWARD_WATCH] == "pgie_forward_watch.txt"
    assert MODEL_PGIE_CONFIG[MODEL_MARINE_SURVEILLANCE] == "pgie_marine_surveillance.txt"


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


def test_marine_surveillance_mapping_keeps_classes_distinct():
    exp = {
        0: ("boat", 87), 1: ("buoy", 88), 2: ("kayak", 89), 3: ("sailboat", 90),
        4: ("speedboat", 91), 5: ("vessel", 92), 6: ("warship", 93),
    }
    for raw, want in exp.items():
        assert label_for_model(MODEL_MARINE_SURVEILLANCE, raw) == want


def test_ids_never_collide_across_all_models():
    coco = {label_for_model(MODEL_COCO, c)[1] for c in (0, 8, 80)}
    fw = {label_for_model(MODEL_FORWARD_WATCH, c)[1] for c in range(6)}
    ms = {label_for_model(MODEL_MARINE_SURVEILLANCE, c)[1] for c in range(7)}
    assert coco.isdisjoint(fw) and coco.isdisjoint(ms) and fw.isdisjoint(ms)


def test_label_for_resolves_wire_ids_from_all_models():
    assert label_for(8) == "vessel"      # COCO id on the wire
    assert label_for(83) == "debris"     # forward-watch synthetic id
    assert label_for(85) == "kayak"
    assert label_for(90) == "sailboat"   # marine-surveillance synthetic id
    assert label_for(93) == "warship"


def test_model_labels_match_producible_labels():
    for model, expected in MODEL_LABELS.items():
        assert model in MODEL_PGIE_CONFIG, f"model {model} not in PGIE config registry"
        assert sorted(expected) == expected, f"MODEL_LABELS[{model}] must be sorted"
    assert "person" in MODEL_LABELS[MODEL_COCO]
    assert "person" not in MODEL_LABELS[MODEL_FORWARD_WATCH]
    assert "debris" in MODEL_LABELS[MODEL_FORWARD_WATCH]
    assert "debris" not in MODEL_LABELS[MODEL_COCO]
    assert MODEL_LABELS[MODEL_MARINE_SURVEILLANCE] == [
        "boat", "buoy", "kayak", "sailboat", "speedboat", "vessel", "warship"]
    assert "person" not in MODEL_LABELS[MODEL_MARINE_SURVEILLANCE]  # no man-overboard


def test_person_in_water_rule_unchanged():
    assert is_person_in_water("person", waterline_y=400, horizon_y=350) is True
    assert is_person_in_water("person", waterline_y=300, horizon_y=350) is False
    assert is_person_in_water("vessel", waterline_y=400, horizon_y=350) is False
    assert is_person_in_water("person", waterline_y=400, horizon_y=None) is False
