from __future__ import annotations

import pytest
from aic2026.contracts import CaptionResult, FrameRef, QuerySpec, SegmentationResult
from aic2026.contracts.models import MaskRLE
from pydantic import ValidationError


def test_frame_uid_must_use_original_frame_idx() -> None:
    with pytest.raises(ValidationError, match="frame_uid"):
        FrameRef(
            video_id="L21_V011",
            frame_uid="L21_V011:1",
            keyframe_n=1,
            frame_idx=0,
            pts_time_s=0,
            fps=25,
            frame_relpath="keyframes/L21_V011/000001.jpg",
            width=1920,
            height=1080,
        )


def test_frame_contract_does_not_coerce_numeric_strings() -> None:
    with pytest.raises(ValidationError):
        FrameRef(
            video_id="L21_V011",
            frame_uid="L21_V011:0",
            keyframe_n="1",  # type: ignore[arg-type]
            frame_idx=0,
            pts_time_s=0.0,
            fps=25.0,
            frame_relpath="keyframes/L21_V011/000001.jpg",
            width=1920,
            height=1080,
        )


def test_query_contract_enforces_task_specific_shape() -> None:
    query = QuerySpec(
        task_type="qa",
        raw_query_vi="Có bao nhiêu người?",
        scene_en="people on a stage",
        question_vi="Có bao nhiêu người?",
        question_en="How many people are there?",
        answer_sources=["visual"],
    )
    assert query.schema_version == "aic26.query.v1"

    with pytest.raises(ValidationError, match="question_vi"):
        QuerySpec(task_type="qa", raw_query_vi="Ai?", scene_en="a person")


def test_trake_requires_ordered_events() -> None:
    with pytest.raises(ValidationError, match="ordered event"):
        QuerySpec(task_type="trake", raw_query_vi="Một chuỗi", scene_en="an event")


def test_query_signals_reject_empty_strings_and_duplicate_answer_sources() -> None:
    with pytest.raises(ValidationError, match="empty strings"):
        QuerySpec(
            task_type="kis",
            raw_query_vi="Tìm người",
            scene_en="a person",
            objects_en=[" "],
        )

    with pytest.raises(ValidationError, match="duplicates"):
        QuerySpec(
            task_type="qa",
            raw_query_vi="Ai?",
            scene_en="a person",
            question_vi="Ai?",
            question_en="Who?",
            answer_sources=["visual", "visual"],
        )


def test_failed_caption_requires_structured_error() -> None:
    with pytest.raises(ValidationError, match="structured error"):
        CaptionResult(status="error")


def test_mask_source_and_quality_score_must_agree() -> None:
    rle = MaskRLE(size=(2, 2), counts="04")
    with pytest.raises(ValidationError, match="require the raw"):
        SegmentationResult(mask_source="sam", sam_iou_score=None, mask_rle=rle)
    with pytest.raises(ValidationError, match="cannot claim"):
        SegmentationResult(mask_source="bbox_fallback", sam_iou_score=0.5, mask_rle=rle)
