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
from .scene_embedding import SCENE_EMBEDDING_SCHEMA, SceneEmbeddingRecord

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
    "SCENE_EMBEDDING_SCHEMA",
    "SceneEmbeddingRecord",
    "SegmentationResult",
    "ShotRecord",
    "TaskType",
    "TemporalOperator",
]
