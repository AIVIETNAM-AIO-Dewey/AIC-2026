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
from .scene_embedding import SCENE_EMBEDDING_SCHEMA, SceneEmbeddingRecord

__all__ = [
    "SCENE_EMBEDDING_SCHEMA",
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
    "SceneEmbeddingRecord",
    "SegmentationResult",
    "TaskType",
    "TemporalOperator",
]
