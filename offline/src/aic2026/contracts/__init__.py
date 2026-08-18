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
)
from .ocr import LegacyOcrFrameRecord, OcrError, OcrFrameRecord, OcrText
from .query import AnswerSource, QueryEvent, QuerySpec, TaskType, TemporalOperator
from .scene_embedding import SCENE_EMBEDDING_SCHEMA, SceneEmbeddingRecord

__all__ = [
    "SCENE_EMBEDDING_SCHEMA",
    "AnswerSource",
    "AsrKeyframeRef",
    "AsrSegmentRecord",
    "AsrVideoManifest",
    "CaptionResult",
    "DetectorMetadata",
    "FrameRef",
    "LegacyOcrFrameRecord",
    "MaskRLE",
    "ObjectFrameRecord",
    "ObjectRegion",
    "OcrError",
    "OcrFrameRecord",
    "OcrText",
    "QueryEvent",
    "QuerySpec",
    "RunManifest",
    "SceneEmbeddingRecord",
    "SegmentationResult",
    "TaskType",
    "TemporalOperator",
]
