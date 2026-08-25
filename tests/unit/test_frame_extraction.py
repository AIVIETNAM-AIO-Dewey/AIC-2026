"""Unit tests for TransNetV2 shot sampling, deduplication, and frame extraction."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aic2026.contracts import ShotRecord
from aic2026.frame_extraction.sampling import (
    FrameSampleCandidate,
    adaptive_samples_from_shots,
    dedupe_samples,
    fallback_samples,
    sample_indices_for_shot,
)


def test_fallback_sampling_builds_smoke_timestamps() -> None:
    samples = fallback_samples(fps=25.0, timestamps_s=[0, 1, 2, 3, 4])

    assert [sample.frame_idx for sample in samples] == [0, 25, 50, 75, 100]
    assert [sample.sample_n for sample in samples] == [1, 2, 3, 4, 5]
    assert all(sample.sampling_source == "fallback" for sample in samples)


def test_adaptive_sampling_duration_bands_and_cap_rules() -> None:
    # Short shot (<2s) -> 1 frame at midpoint
    assert sample_indices_for_shot(shot_start_idx=0, shot_end_idx=48, fps=25.0) == [24]
    # Medium shot (2s <= T < 4s) -> 2 frames at 25% and 75%
    assert sample_indices_for_shot(shot_start_idx=0, shot_end_idx=49, fps=25.0) == [12, 38]
    assert sample_indices_for_shot(shot_start_idx=0, shot_end_idx=74, fps=25.0) == [19, 56]
    # Long shot (4s <= T < 7s) -> 1.5s cadence
    assert sample_indices_for_shot(shot_start_idx=0, shot_end_idx=99, fps=25.0) == [19, 56, 94]

    long_indices = sample_indices_for_shot(shot_start_idx=0, shot_end_idx=249, fps=25.0)
    assert long_indices == [19, 56, 94, 131, 169, 206, 244]

    # Extreme shot (>= 7s) -> capped at max_frames_per_shot (10)
    capped = sample_indices_for_shot(shot_start_idx=0, shot_end_idx=999, fps=25.0)
    assert len(capped) == 10
    assert capped[0] == 19
    assert capped[-1] <= 999


def test_adaptive_samples_from_shot_records() -> None:
    shot = ShotRecord(
        video_id="L21_V001",
        shot_id="L21_V001:s00001",
        shot_start_idx=0,
        shot_end_idx=48,
        start_time_s=0.0,
        end_time_s=1.92,
        fps=25.0,
        source_video="/kaggle/input/video.mp4",
    )

    samples = adaptive_samples_from_shots([shot])

    assert len(samples) == 1
    assert samples[0].shot_id == "L21_V001:s00001"
    assert samples[0].sampling_source == "transnetv2"


def test_dedupe_samples_tolerance() -> None:
    s1 = FrameSampleCandidate(
        sample_n=1,
        pts_time_s=10.0,
        fps=30.0,
        frame_idx=300,
        sampling_source="transnetv2",
    )
    s2 = FrameSampleCandidate(
        sample_n=2,
        pts_time_s=10.3,
        fps=30.0,
        frame_idx=309,
        sampling_source="transnetv2",
    )
    s3 = FrameSampleCandidate(
        sample_n=3,
        pts_time_s=12.0,
        fps=30.0,
        frame_idx=360,
        sampling_source="transnetv2",
    )

    merged = dedupe_samples([s1, s2, s3], tolerance_s=0.5)
    # s2 is within 0.3s of s1 (< 0.5s tolerance), so it gets dropped
    assert len(merged) == 2
    assert [s.frame_idx for s in merged] == [300, 360]


def test_extract_transnet_frames_pipeline(tmp_path: Path) -> None:
    """Verify extract_transnet_frames emits valid FrameRef JSONL, keyframes, and map CSV."""
    from scripts.extract_transnet_frames import main as extract_main
    from aic2026.frame_extraction.transnetv2 import TransNetV2InferenceResult
    from aic2026.frame_extraction.ffmpeg import VideoProbe

    video_path = tmp_path / "L21_V001.mp4"
    video_path.write_bytes(b"dummy")
    output_root = tmp_path / "artifacts"

    # Mock probe and transnetv2 inference
    mock_probe = VideoProbe(fps=25.0, width=1280, height=720, duration_s=10.0, frame_count=250)
    mock_shot = ShotRecord(
        video_id="L21_V001",
        shot_id="L21_V001:s00001",
        shot_start_idx=0,
        shot_end_idx=249,
        start_time_s=0.0,
        end_time_s=9.96,
        fps=25.0,
        source_video=str(video_path),
    )
    mock_result = TransNetV2InferenceResult(scenes=[(0, 249)], shots=[mock_shot])

    def mock_extract(video_path, outputs, jpeg_quality=2):
        for frame_idx, p in outputs:
            p.parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image
            img = Image.new("RGB", (1280, 720), color="blue")
            img.save(p, format="JPEG")

    with (
        patch("scripts.extract_transnet_frames.probe_video", return_value=mock_probe),
        patch("scripts.extract_transnet_frames.run_transnetv2_inference", return_value=mock_result),
        patch("scripts.extract_transnet_frames.extract_frames_by_index", side_effect=mock_extract),
    ):
        code = extract_main(
            [
                "--video-id", "L21_V001",
                "--video-path", str(video_path),
                "--output-root", str(output_root),
                "--limit", "3",
            ]
        )
        assert code == 0

    manifest_file = output_root / "frame_manifests" / "L21_V001.jsonl"
    map_csv_file = output_root / "map-keyframes" / "L21_V001.csv"

    assert manifest_file.is_file()
    assert map_csv_file.is_file()

    # Verify manifest JSONL
    with manifest_file.open("r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]
    assert len(lines) == 3
    assert lines[0]["video_id"] == "L21_V001"
    assert lines[0]["keyframe_n"] == 1
    assert lines[0]["width"] == 1280
    assert lines[0]["height"] == 720

    # Verify map CSV
    with map_csv_file.open("r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 3
    assert reader[0]["n"] == "1"
    assert reader[0]["fps"] == "25.00"
    assert "frame_idx" in reader[0]
