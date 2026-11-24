"""Parser for organizer OpenImages Faster R-CNN JSON artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .geometry import Detection

REQUIRED_ARRAYS = (
    "detection_scores",
    "detection_class_names",
    "detection_class_entities",
    "detection_boxes",
    "detection_class_labels",
)


def _bounded_coordinate(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("coordinate must be numeric, not boolean")
    coordinate = float(value)
    if not math.isfinite(coordinate):
        raise ValueError("coordinate must be finite")
    if coordinate < -0.01 or coordinate > 1.01:
        raise ValueError("coordinate lies outside the accepted clipping tolerance")
    return min(1.0, max(0.0, coordinate))


def _required_text(value: Any, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}[{index}] must be a non-empty string")
    return value.strip()


def _class_label(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("class label must be an integer, not boolean")
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
        raise ValueError("class label must be a non-negative integer")
    return int(numeric)


def parse_organizer_detections(payload: dict[str, Any]) -> list[Detection]:
    arrays: dict[str, list[Any]] = {}
    for key in REQUIRED_ARRAYS:
        value = payload.get(key)
        if not isinstance(value, list):
            raise ValueError(f"{key} must be an array")
        arrays[key] = value
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("organizer detection arrays must have equal lengths")

    detections: list[Detection] = []
    for index in range(len(arrays["detection_scores"])):
        box_raw = arrays["detection_boxes"][index]
        if not isinstance(box_raw, list | tuple) or len(box_raw) != 4:
            raise ValueError(f"detection_boxes[{index}] must contain four coordinates")
        try:
            if isinstance(arrays["detection_scores"][index], bool):
                raise ValueError("score must be numeric, not boolean")
            score = float(arrays["detection_scores"][index])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("score outside [0, 1]")
            box = tuple(_bounded_coordinate(value) for value in box_raw)
            label = _class_label(arrays["detection_class_labels"][index])
            class_name = _required_text(
                arrays["detection_class_names"][index], "detection_class_names", index
            )
            class_entity = _required_text(
                arrays["detection_class_entities"][index], "detection_class_entities", index
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid detection at source index {index}: {error}") from error
        detections.append(
            Detection(
                source_index=index,
                score=score,
                class_name=class_name,
                class_entity=class_entity,
                class_label=label,
                bbox_yxyx_norm=box,  # type: ignore[arg-type]
            )
        )
    return detections


def load_organizer_detections(path: Path) -> list[Detection]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Organizer object file must contain one JSON object: {path}")
    return parse_organizer_detections(payload)


def index_object_files(objects_dir: Path) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for path in objects_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        try:
            keyframe_n = int(path.stem)
        except ValueError as error:
            raise ValueError(f"Objects JSON filename stem must be numeric: {path.name}") from error
        if keyframe_n in indexed:
            raise ValueError(f"Multiple object files resolve to keyframe n={keyframe_n}")
        indexed[keyframe_n] = path
    return indexed
