"""Pydantic Contracts & Data Schemas for Multimodal Video Retrieval Engine."""

from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, Field


TaskType = Literal["KIS", "TRAKE", "VQA"]


class TrakeEvent(BaseModel):
    """Sub-event description in a chronological action sequence."""
    order: int = 1
    description: str
    scene_en: str = ""
    objects_en: list[str] = Field(default_factory=list)
    speech_vi: str = ""
    ocr_keywords: list[str] = Field(default_factory=list)


TRAKEEvent = TrakeEvent


class ParsedQuery(BaseModel):
    """Normalized structured query output from LLM Query Decomposer."""
    task_type: TaskType = "KIS"
    language: str = "vi"
    original_query: str
    
    # 4-Channel Sub-Queries
    global_scene_en: str = ""
    objects_en: list[str] = Field(default_factory=list)
    ocr_keywords: list[str] = Field(default_factory=list)
    speech_vi: str = ""
    
    # TRAKE Temporal Sequence Fields
    is_temporal_trake: bool = False
    trake_events: list[TrakeEvent] = Field(default_factory=list)
    
    # VQA Question Field
    vqa_question: str = ""
    
    # Dynamic Modality Fusion Weights (Sum to 1.0)
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "vis": 0.45,
            "dam": 0.40,
            "asr": 0.15,
            "ocr": 0.00,
        }
    )


class MatchedObject(BaseModel):
    """Explainability metadata for detected DAM objects."""
    region_id: str | int = 1
    class_entity: str
    description_en: str
    score: float
    bbox: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])


class SpeechEvidence(BaseModel):
    """Spoken dialogue evidence for keyframe."""
    start_s: float
    end_s: float
    transcript_raw: str
    score: float = 0.0


class SearchResult(BaseModel):
    """Single candidate keyframe result card."""
    rank: int
    video_id: str
    keyframe_n: int
    frame_idx: int
    pts_time_s: float
    submission_string: str  # Format: "<VIDEO_ID>, <FRAME_IDX>"
    
    # Scores
    final_score: float
    stage1_score: float
    stage2_rerank_score: float
    visual_similarity: float
    
    # Explainability & Visual Artifacts
    image_relpath: str
    image_available: bool = True
    best_matching_objects: list[MatchedObject] = Field(default_factory=list)
    dam_full_captions: list[str] = Field(default_factory=list)
    
    # Speech & Text Evidence
    has_speech: bool = False
    speech_evidence: Optional[SpeechEvidence] = None
    ocr_text: str = ""
    
    # Adjacent Keyframes for shot navigation
    adjacent_keyframes: list[int] = Field(default_factory=list)
    
    # VQA Answer (if VQA task)
    vqa_answer: Optional[str] = None


class SearchResponse(BaseModel):
    """Full search response payload returned to Frontend UI."""
    task_type: TaskType
    original_query: str
    parsed_query: ParsedQuery
    execution_time_ms: float
    total_candidates_evaluated: int
    results: list[SearchResult]
