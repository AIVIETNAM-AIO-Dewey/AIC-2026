"""Versioned cross-team data contracts."""

from .models import (
    CaptionResult,
    DetectorMetadata,
    FrameRef,
    MaskRLE,
    ObjectFrameRecord,
    ObjectRegion,
    RunManifest,
    SegmentationResult,
)
from .query import AnswerSource, QueryEvent, QuerySpec, TaskType, TemporalOperator

__all__ = [
    "AnswerSource",
    "CaptionResult",
    "DetectorMetadata",
    "FrameRef",
    "MaskRLE",
    "ObjectFrameRecord",
    "ObjectRegion",
    "QueryEvent",
    "QuerySpec",
    "RunManifest",
    "SegmentationResult",
    "TaskType",
    "TemporalOperator",
]
