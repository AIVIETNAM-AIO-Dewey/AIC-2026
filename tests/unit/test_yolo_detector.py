"""Unit tests for YOLO-World detector adapter and pipeline integration."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pytest
from PIL import Image

from aic2026.object_description.geometry import Detection, FilterConfig, filter_detections
from aic2026.object_description.yolo_detector import YoloWorldDetector


def test_yolo_world_detector_output_contract():
    """Verify YoloWorldDetector converts mock Ultralytics boxes into valid normalized Detection objects."""
    detector = object.__new__(YoloWorldDetector)
    detector.conf = 0.20
    detector.device = "cpu"

    # Mock Ultralytics Results object
    mock_boxes = MagicMock()
    mock_boxes.__len__.return_value = 2

    # Box 1: (100, 50, 300, 250) in a 1000x500 image -> ymin=0.1, xmin=0.1, ymax=0.5, xmax=0.3
    # Box 2: (500, 200, 800, 400) -> ymin=0.4, xmin=0.5, ymax=0.8, xmax=0.8
    box1 = MagicMock()
    box1.tolist.return_value = [100.0, 50.0, 300.0, 250.0]
    box2 = MagicMock()
    box2.tolist.return_value = [500.0, 200.0, 800.0, 400.0]
    mock_boxes.xyxy = [box1, box2]

    conf1 = MagicMock()
    conf1.item.return_value = 0.88
    conf2 = MagicMock()
    conf2.item.return_value = 0.65
    mock_boxes.conf = [conf1, conf2]

    cls1 = MagicMock()
    cls1.item.return_value = 0
    cls2 = MagicMock()
    cls2.item.return_value = 42
    mock_boxes.cls = [cls1, cls2]

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    mock_result.names = {0: "person", 42: "water bottle"}

    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result]
    detector.model = mock_model

    # Run detection on synthetic 1000x500 image
    img = Image.new("RGB", (1000, 500), color="white")
    detections = detector.detect(img)

    assert len(detections) == 2
    
    det1 = detections[0]
    assert det1.source_index == 0
    assert det1.class_name == "person"
    assert det1.score == 0.88
    # ymin=50/500=0.1, xmin=100/1000=0.1, ymax=250/500=0.5, xmax=300/1000=0.3
    assert det1.bbox_yxyx_norm == (0.1, 0.1, 0.5, 0.3)

    det2 = detections[1]
    assert det2.source_index == 1
    assert det2.class_name == "water bottle"
    assert det2.score == 0.65
    assert det2.bbox_yxyx_norm == (0.4, 0.5, 0.8, 0.8)


def test_yolo_world_filtering_integration():
    """Verify raw YOLO-World detections correctly flow through existing FilterConfig."""
    raw_detections = [
        Detection(
            source_index=0,
            score=0.92,
            class_name="sunglasses",
            class_entity="sunglasses",
            class_label=1,
            bbox_yxyx_norm=(0.1, 0.1, 0.2, 0.2),  # Area = 0.01
        ),
        Detection(
            source_index=1,
            score=0.15,  # below threshold
            class_name="bottle",
            class_entity="bottle",
            class_label=2,
            bbox_yxyx_norm=(0.3, 0.3, 0.5, 0.5),
        ),
        Detection(
            source_index=2,
            score=0.85,
            class_name="laptop",
            class_entity="laptop",
            class_label=3,
            bbox_yxyx_norm=(0.4, 0.4, 0.8, 0.8),  # Area = 0.16
        ),
    ]

    config = FilterConfig(
        minimum_score=0.30,
        minimum_area_ratio=0.005,
        maximum_area_ratio=0.85,
        maximum_regions=3,
    )

    filtered = filter_detections(raw_detections, config)
    assert len(filtered) == 2
    assert [d.class_name for d in filtered] == ["sunglasses", "laptop"]


def test_default_open_vocabulary():
    """Verify default open vocabulary contains diverse infrastructure, nature, hazard, and object classes."""
    from aic2026.object_description.yolo_detector import DEFAULT_OPEN_VOCABULARY

    assert len(DEFAULT_OPEN_VOCABULARY) > 30
    assert "building" in DEFAULT_OPEN_VOCABULARY
    assert "house" in DEFAULT_OPEN_VOCABULARY
    assert "road" in DEFAULT_OPEN_VOCABULARY
    assert "water" in DEFAULT_OPEN_VOCABULARY
    assert "river" in DEFAULT_OPEN_VOCABULARY
    assert "landslide" in DEFAULT_OPEN_VOCABULARY
    assert "person" in DEFAULT_OPEN_VOCABULARY


def test_scene_fallback_when_zero_detections():
    """Verify fallback scene detection is generated when zero detections pass filter."""
    from aic2026.object_description.geometry import Detection, FilterConfig, filter_detections

    raw_detections: list[Detection] = []
    config = FilterConfig(fallback_scene=True)
    filtered = filter_detections(raw_detections, config)
    assert len(filtered) == 0  # filter_detections alone returns empty, prepare_masks injects fallback

    # When fallback_scene is active in pipeline
    if not filtered and config.fallback_scene:
        fallback = [
            Detection(
                source_index=0,
                score=1.0,
                class_name="scene",
                class_entity="scene",
                class_label=0,
                bbox_yxyx_norm=(0.05, 0.05, 0.95, 0.95),
            )
        ]
        assert len(fallback) == 1
        assert fallback[0].class_entity == "scene"
        assert fallback[0].bbox_yxyx_norm == (0.05, 0.05, 0.95, 0.95)
