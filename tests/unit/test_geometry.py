from __future__ import annotations

import math

from aic2026.object_description.geometry import (
    Detection,
    FilterConfig,
    filter_detections,
    intersection_over_union,
    normalized_to_pixels,
)


def _detection(index: int, score: float, entity: str, box: tuple[float, ...]) -> Detection:
    return Detection(index, score, entity, entity, index, box)  # type: ignore[arg-type]


def test_class_and_cross_label_deduplication_is_stable() -> None:
    detections = [
        _detection(0, 0.92, "person", (0.1, 0.1, 0.8, 0.45)),
        _detection(1, 0.80, "person", (0.11, 0.11, 0.79, 0.44)),
        _detection(2, 0.79, "man", (0.1, 0.1, 0.8, 0.45)),
        _detection(3, 0.15, "microphone", (0.3, 0.46, 0.7, 0.55)),
        _detection(4, 0.01, "tree", (0.0, 0.0, 1.0, 1.0)),
    ]

    kept = filter_detections(detections, FilterConfig())

    assert [item.source_index for item in kept] == [0, 3]


def test_iou_and_half_open_pixel_conversion() -> None:
    assert intersection_over_union((0, 0, 1, 1), (0, 0, 1, 1)) == 1
    assert intersection_over_union((0, 0, 0.2, 0.2), (0.8, 0.8, 1, 1)) == 0
    assert normalized_to_pixels((0.1, 0.2, 0.8, 0.9), 100, 50) == (20, 5, 90, 40)


def test_region_cap_applies_after_deduplication() -> None:
    detections = [
        _detection(i, 0.99 - i / 1000, f"class-{i}", (0.1, i / 100, 0.2, i / 100 + 0.01))
        for i in range(30)
    ]
    config = FilterConfig(minimum_area_ratio=0.0001, maximum_regions=20)

    assert len(filter_detections(detections, config)) == 20


def test_invalid_and_non_finite_boxes_are_rejected_from_candidates() -> None:
    detections = [
        _detection(0, 0.9, "reversed", (0.8, 0.1, 0.2, 0.5)),
        _detection(1, 0.9, "empty", (0.1, 0.1, 0.1, 0.5)),
        _detection(2, 0.9, "nan", (0.1, math.nan, 0.5, 0.5)),
    ]

    assert filter_detections(detections) == []
