"""Deterministic geometry filtering for organizer Faster R-CNN boxes."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Detection:
    source_index: int
    score: float
    class_name: str
    class_entity: str
    class_label: int
    bbox_yxyx_norm: tuple[float, float, float, float]

    @property
    def area_ratio(self) -> float:
        ymin, xmin, ymax, xmax = self.bbox_yxyx_norm
        return (ymax - ymin) * (xmax - xmin)


@dataclass(frozen=True, slots=True)
class FilterConfig:
    minimum_score: float = 0.10
    minimum_area_ratio: float = 0.001
    maximum_area_ratio: float = 0.90
    same_class_iou: float = 0.70
    cross_label_duplicate_iou: float = 0.95
    maximum_regions: int = 20
    fallback_scene: bool = True

    def __post_init__(self) -> None:
        for name in (
            "minimum_score",
            "minimum_area_ratio",
            "maximum_area_ratio",
            "same_class_iou",
            "cross_label_duplicate_iou",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.minimum_area_ratio >= self.maximum_area_ratio:
            raise ValueError("minimum_area_ratio must be below maximum_area_ratio")
        if self.maximum_regions < 1:
            raise ValueError("maximum_regions must be positive")


def valid_normalized_box(box: tuple[float, float, float, float]) -> bool:
    if not all(math.isfinite(value) for value in box):
        return False
    ymin, xmin, ymax, xmax = box
    return 0 <= ymin < ymax <= 1 and 0 <= xmin < xmax <= 1


def intersection_over_union(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    ly1, lx1, ly2, lx2 = left
    ry1, rx1, ry2, rx2 = right
    intersection_h = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection_w = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    intersection = intersection_h * intersection_w
    if intersection == 0:
        return 0.0
    left_area = (ly2 - ly1) * (lx2 - lx1)
    right_area = (ry2 - ry1) * (rx2 - rx1)
    return intersection / (left_area + right_area - intersection)


def filter_detections(
    detections: list[Detection], config: FilterConfig | None = None
) -> list[Detection]:
    config = config or FilterConfig()
    candidates = [
        detection
        for detection in detections
        if detection.score >= config.minimum_score
        and valid_normalized_box(detection.bbox_yxyx_norm)
        and config.minimum_area_ratio <= detection.area_ratio <= config.maximum_area_ratio
    ]
    candidates.sort(key=lambda detection: (-detection.score, detection.source_index))

    kept: list[Detection] = []
    for candidate in candidates:
        duplicate = False
        for existing in kept:
            iou = intersection_over_union(candidate.bbox_yxyx_norm, existing.bbox_yxyx_norm)
            same_class = (
                candidate.class_entity == existing.class_entity
                or candidate.class_label == existing.class_label
            )
            if same_class and iou >= config.same_class_iou:
                duplicate = True
                break
            if not same_class and iou >= config.cross_label_duplicate_iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
            if len(kept) == config.maximum_regions:
                break
    return kept


def normalized_to_pixels(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    if not valid_normalized_box(box):
        raise ValueError("normalized box is invalid")
    ymin, xmin, ymax, xmax = box
    x1 = max(0, min(width - 1, math.floor(xmin * width)))
    y1 = max(0, min(height - 1, math.floor(ymin * height)))
    x2 = max(x1 + 1, min(width, math.ceil(xmax * width)))
    y2 = max(y1 + 1, min(height, math.ceil(ymax * height)))
    return x1, y1, x2, y2
