from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from aic2026.asr.normalizer import normalize_transcript
from aic2026.asr.pipeline import _deduplicate_segments
from aic2026.asr.validation import validate_jsonl, validate_manifest
from aic2026.contracts.asr import AsrKeyframeRef, AsrSegmentRecord, AsrVideoManifest


def test_asr_keyframe_ref_validation() -> None:
    # Valid keyframe ref
    kf = AsrKeyframeRef(
        keyframe_n=1,
        frame_idx=90,
        pts_time_s=3.0,
        frame_uid="L21_V001:90",
    )
    assert kf.frame_uid == "L21_V001:90"

    # Invalid frame_uid suffix mismatch
    with pytest.raises(ValidationError, match="does not match frame_idx"):
        AsrKeyframeRef(
            keyframe_n=1,
            frame_idx=90,
            pts_time_s=3.0,
            frame_uid="L21_V001:100",
        )

    # Invalid frame_uid missing colon
    with pytest.raises(ValidationError, match="video_id:frame_idx"):
        AsrKeyframeRef(
            keyframe_n=1,
            frame_idx=90,
            pts_time_s=3.0,
            frame_uid="L21_V001_90",
        )


def test_asr_segment_record_validation() -> None:
    kf = AsrKeyframeRef(
        keyframe_n=1,
        frame_idx=90,
        pts_time_s=3.0,
        frame_uid="L21_V001:90",
    )

    # Valid record
    rec = AsrSegmentRecord(
        segment_id="L21_V001:seg_0",
        video_id="L21_V001",
        start_ms=0,
        end_ms=5000,
        transcript_raw="Bản tin thời sự hôm nay.",
        transcript_normalized="bản tin thời sự hôm nay",
        keyframes=[kf],
    )
    assert rec.schema_version == "aic26.asr_segments.v1"

    # Invalid: end_ms <= start_ms
    with pytest.raises(ValidationError, match="greater than start_ms"):
        AsrSegmentRecord(
            segment_id="L21_V001:seg_0",
            video_id="L21_V001",
            start_ms=5000,
            end_ms=3000,
            transcript_raw="Test",
            transcript_normalized="test",
        )

    # Invalid: segment_id does not start with video_id:
    with pytest.raises(ValidationError, match="segment_id must be prefixed"):
        AsrSegmentRecord(
            segment_id="seg_0",
            video_id="L21_V001",
            start_ms=0,
            end_ms=5000,
            transcript_raw="Test",
            transcript_normalized="test",
        )

    # Invalid: keyframe timestamp outside segment range
    kf_outside = AsrKeyframeRef(
        keyframe_n=10,
        frame_idx=900,
        pts_time_s=30.0,  # 30000ms > 5000ms
        frame_uid="L21_V001:900",
    )
    with pytest.raises(ValidationError, match="outside segment"):
        AsrSegmentRecord(
            segment_id="L21_V001:seg_0",
            video_id="L21_V001",
            start_ms=0,
            end_ms=5000,
            transcript_raw="Test",
            transcript_normalized="test",
            keyframes=[kf_outside],
        )

    # Invalid: duplicate keyframe frame_uid in one segment
    with pytest.raises(ValidationError, match="duplicate keyframe"):
        AsrSegmentRecord(
            segment_id="L21_V001:seg_0",
            video_id="L21_V001",
            start_ms=0,
            end_ms=5000,
            transcript_raw="Test",
            transcript_normalized="test",
            keyframes=[kf, kf],
        )


def test_asr_video_manifest_validation() -> None:
    from datetime import datetime, timezone

    # Terminal state requires ended_at
    with pytest.raises(ValidationError, match="terminal manifests require ended_at"):
        AsrVideoManifest(
            video_id="L21_V001",
            status="completed",
            segment_count=5,
            keyframe_count=10,
            audio_duration_s=60.0,
            model_id="vinai/PhoWhisper-large",
            engine="faster_whisper",
            started_at=datetime.now(timezone.utc),
            ended_at=None,
        )

    # Failed status requires error_message
    with pytest.raises(ValidationError, match="failed manifests require error_message"):
        AsrVideoManifest(
            video_id="L21_V001",
            status="failed",
            segment_count=0,
            keyframe_count=0,
            audio_duration_s=0.0,
            model_id="vinai/PhoWhisper-large",
            engine="faster_whisper",
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            error_message=None,
        )


def test_transcript_normalization() -> None:
    # Lowercase & strip punctuation while preserving Vietnamese diacritics
    raw = "Chào bạn! Đây là bản tin 60s trên Facebook và YouTube."
    norm = normalize_transcript(raw)
    assert "chào bạn" in norm
    assert "bản tin 60s" in norm
    assert "facebook" in norm
    assert "youtube" in norm

    # Loanword expansion test: phonetic -> English alias
    phonetic_raw = "Tôi mua chiếc ai phôn mới để lướt phây búc."
    norm_phonetic = normalize_transcript(phonetic_raw)
    assert "ai phôn" in norm_phonetic
    assert "phây búc" in norm_phonetic
    assert "iphone" in norm_phonetic
    assert "facebook" in norm_phonetic


def test_deduplicate_overlapping_segments() -> None:
    raw_segments = [
        {
            "start_ms": 1000,
            "end_ms": 8000,
            "text": "Bản tin thời sự hôm nay trên kênh VTV1",
            "window_start_s": 0.0,
            "window_end_s": 15.0,
        },
        {
            "start_ms": 1050,
            "end_ms": 8050,
            "text": "Bản tin thời sự hôm nay trên kênh VTV1",
            "window_start_s": 7.5,
            "window_end_s": 22.5,
        },
        {
            "start_ms": 12000,
            "end_ms": 16000,
            "text": "Xin cảm ơn quý vị đã quan tâm theo dõi",
            "window_start_s": 7.5,
            "window_end_s": 22.5,
        },
    ]

    deduped = _deduplicate_segments(
        raw_segments,
        time_overlap_threshold=0.80,
        text_similarity_threshold=0.85,
        merge_gap_ms=500,
    )

    # Should collapse the two near-identical segments into 1, leaving 2 segments total
    assert len(deduped) == 2
    assert deduped[0]["start_ms"] in (1000, 1050)
    assert deduped[1]["start_ms"] == 12000


def test_artifact_validation_utility() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        jsonl_file = tmp_path / "L21_V001.jsonl"

        rec = AsrSegmentRecord(
            segment_id="L21_V001:seg_0",
            video_id="L21_V001",
            start_ms=0,
            end_ms=5000,
            transcript_raw="Bản tin hôm nay.",
            transcript_normalized="bản tin hôm nay",
        )

        jsonl_file.write_text(rec.model_dump_json() + "\n", encoding="utf-8")

        res = validate_jsonl(jsonl_file)
        assert res.is_valid
        assert res.valid_records == 1
        assert res.total_lines == 1
