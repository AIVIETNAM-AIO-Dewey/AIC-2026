from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic2026.object_description import load_organizer_detections
from aic2026.object_description.organizer import parse_organizer_detections

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_parse_organizer_string_arrays() -> None:
    detections = load_organizer_detections(FIXTURES / "objects" / "000001.json")

    assert len(detections) == 5
    assert detections[0].score == 0.92
    assert detections[0].bbox_yxyx_norm == (0.1, 0.1, 0.8, 0.45)


def test_mismatched_arrays_fail() -> None:
    payload = json.loads((FIXTURES / "objects" / "000001.json").read_text())
    payload["detection_scores"].pop()

    with pytest.raises(ValueError, match="equal lengths"):
        parse_organizer_detections(payload)


def test_small_boundary_error_is_clipped() -> None:
    payload = json.loads((FIXTURES / "objects" / "000001.json").read_text())
    payload["detection_boxes"][0] = ["-0.001", "0", "1.001", "1"]

    detection = parse_organizer_detections(payload)[0]

    assert detection.bbox_yxyx_norm == (0.0, 0.0, 1.0, 1.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("detection_class_names", "", "non-empty string"),
        ("detection_class_entities", None, "non-empty string"),
        ("detection_class_labels", "1.5", "non-negative integer"),
    ],
)
def test_invalid_detector_metadata_fails_before_model_load(
    field: str, value: object, message: str
) -> None:
    payload = json.loads((FIXTURES / "objects" / "000001.json").read_text())
    payload[field][0] = value

    with pytest.raises(ValueError, match=message):
        parse_organizer_detections(payload)
