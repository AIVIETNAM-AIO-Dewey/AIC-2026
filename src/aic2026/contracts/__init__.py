"""Versioned cross-team data contracts."""

from .asr import AsrKeyframeRef, AsrSegmentRecord, AsrVideoManifest
from .models import (
    CaptionResult,
    DetectorMetadata,
    FrameRef,
    MaskRLE,
    ObjectFrameRecord,
    ObjectRegion,
    RunManifest,
    SegmentationResult,
    ShotRecord,
)
from .query import AnswerSource, QueryEvent, QuerySpec, TaskType, TemporalOperator

__all__ = [
    "AnswerSource",
    "AsrKeyframeRef",
    "AsrSegmentRecord",
    "AsrVideoManifest",
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
    "ShotRecord",
    "TaskType",
    "TemporalOperator",
]
