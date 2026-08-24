"""Unit tests for the Unified Multi-Modal Pipeline schemas, sampling, and normalization."""

from __future__ import annotations

import pytest
from aic2026.contracts import ShotRecord
from aic2026.frame_extraction import (
    adaptive_samples_from_shots,
    dedupe_samples,
    sample_indices_for_shot,
)
from aic2026.ocr import normalize_vietnamese_text
from aic2026.unified import (
    DamRegionCaption,
    UnifiedFrameRecord,
    UnifiedOcrResult,
    UnifiedOcrSpan,
)


def test_unified_frame_record_valid() -> None:
    rec = UnifiedFrameRecord(
        video_id="L21_V003",
        frame_uid="L21_V003:50",
        keyframe_n=2,
        frame_idx=50,
        pts_time_s=2.0,
        fps=25.0,
        shot_id="L21_V003:shot_0000",
        image_relpath="frames/L21_V003/002.jpg",
        siglip_embedding=[0.1] * 768,
        ocr=UnifiedOcrResult(full_text="test text"),
        dam_descriptions=[
            DamRegionCaption(
                region_id="reg_001",
                class_label="Person",
                bbox_xyxy_px=(10, 10, 100, 100),
                sam_iou=0.95,
                caption_en="A person standing outdoors.",
                word_count=4,
            )
        ],
    )
    assert rec.frame_uid == "L21_V003:50"
    assert rec.keyframe_n == 2
    assert len(rec.siglip_embedding) == 768


def test_unified_frame_record_invalid_uid() -> None:
    with pytest.raises(ValueError, match="frame_uid must equal"):
        UnifiedFrameRecord(
            video_id="L21_V003",
            frame_uid="L21_V003:999",  # Mismatch with frame_idx=50
            keyframe_n=2,
            frame_idx=50,
            pts_time_s=2.0,
            fps=25.0,
            image_relpath="frames/L21_V003/002.jpg",
        )


def test_dam_caption_word_count_cap() -> None:
    # 51 words should fail
    long_caption = "word " * 51
    with pytest.raises(ValueError):
        DamRegionCaption(
            region_id="reg_001",
            class_label="Car",
            bbox_xyxy_px=(0, 0, 50, 50),
            caption_en=long_caption.strip(),
            word_count=51,
        )


def test_adaptive_shot_sampling() -> None:
    fps = 25.0
    # 1. Ultra short shot: 1.0s (25 frames: 0..24) -> exactly 1 frame at midpoint (12)
    indices_short = sample_indices_for_shot(shot_start_idx=0, shot_end_idx=24, fps=fps)
    assert len(indices_short) == 1
    assert indices_short[0] == 12

    # 2. Medium shot: 3.0s (75 frames: 0..74) -> 2 frames
    indices_med = sample_indices_for_shot(shot_start_idx=0, shot_end_idx=74, fps=fps)
    assert len(indices_med) == 2

    # 3. Long shot: 10.0s (250 frames: 0..249) -> multiple cadence frames capped <= 10
    indices_long = sample_indices_for_shot(shot_start_idx=0, shot_end_idx=249, fps=fps)
    assert 4 <= len(indices_long) <= 10


def test_text_normalization() -> None:
    raw = "  TRUNG TÂM THƯƠNG MẠI !!! SÀI GÒN - TP.HCM  "
    normalized = normalize_vietnamese_text(raw)
    assert normalized == "trung tâm thương mại sài gòn - tp.hcm"
