from __future__ import annotations

from pathlib import Path

from aic2026.contracts import ShotRecord
from aic2026.frame_extraction.discovery import find_support_file, find_video_file, locate_inputs
from aic2026.frame_extraction.sampling import (
    FrameSampleCandidate,
    adaptive_samples_from_shots,
    dedupe_samples,
    fallback_samples,
    map_keyframe_samples,
    sample_indices_for_shot,
)


def test_discovery_finds_kaggle_style_inputs(tmp_path: Path) -> None:
    video = tmp_path / "aic-26-video" / "videos" / "L21_V001.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"not a real video")
    map_csv = tmp_path / "aic-test-dataset" / "map-keyframes" / "L21_V001.csv"
    map_csv.parent.mkdir(parents=True)
    map_csv.write_text("n,pts_time,fps,frame_idx\n1,0,25,0\n", encoding="utf-8")
    media = tmp_path / "aic-test-dataset" / "media-info" / "L21_V001.json"
    media.parent.mkdir(parents=True)
    media.write_text("{}\n", encoding="utf-8")

    assert find_video_file(tmp_path, "L21_V001") == video.resolve()
    assert (
        find_support_file(
            tmp_path,
            "L21_V001",
            suffix=".csv",
            preferred_parent="map-keyframes",
        )
        == map_csv.resolve()
    )
    located = locate_inputs(video_id="L21_V001", search_root=tmp_path)

    assert located.video_path == video.resolve()
    assert located.map_csv == map_csv.resolve()
    assert located.media_info == media.resolve()


def test_map_sampling_preserves_organizer_frame_indices(tmp_path: Path) -> None:
    map_csv = tmp_path / "L21_V001.csv"
    map_csv.write_text(
        "n,pts_time,fps,frame_idx\n"
        "1,0.0,30.0,0\n"
        "2,11.7333,30.0,351\n"
        "3,12.0,30.0,360\n",
        encoding="utf-8",
    )

    samples = map_keyframe_samples(map_csv)

    assert [(sample.keyframe_n, sample.frame_idx) for sample in samples] == [
        (1, 0),
        (2, 351),
        (3, 360),
    ]
    assert all(sample.sampling_source == "map-keyframes" for sample in samples)


def test_fallback_sampling_builds_smoke_timestamps() -> None:
    samples = fallback_samples(fps=25.0, timestamps_s=[0, 1, 2, 3, 4])

    assert [sample.frame_idx for sample in samples] == [0, 25, 50, 75, 100]
    assert [sample.sample_n for sample in samples] == [1, 2, 3, 4, 5]
    assert all(sample.sampling_source == "fallback" for sample in samples)


def test_adaptive_sampling_short_long_and_cap_rules() -> None:
    assert sample_indices_for_shot(shot_start_idx=0, shot_end_idx=74, fps=25.0) == [37]

    long_indices = sample_indices_for_shot(shot_start_idx=0, shot_end_idx=249, fps=25.0)
    assert long_indices == [19, 56, 94, 131, 169, 206, 244]

    capped = sample_indices_for_shot(shot_start_idx=0, shot_end_idx=999, fps=25.0)
    assert len(capped) == 10
    assert capped[0] == 19
    assert capped[-1] <= 999


def test_adaptive_samples_from_shot_records() -> None:
    shot = ShotRecord(
        video_id="L21_V001",
        shot_id="L21_V001:s00001",
        shot_start_idx=0,
        shot_end_idx=74,
        start_time_s=0.0,
        end_time_s=2.96,
        fps=25.0,
        source_video="/kaggle/input/video.mp4",
    )

    samples = adaptive_samples_from_shots([shot])

    assert len(samples) == 1
    assert samples[0].shot_id == "L21_V001:s00001"
    assert samples[0].sampling_source == "transnetv2"


def test_dedupe_preserves_organizer_priority_across_time_order() -> None:
    adaptive = FrameSampleCandidate(
        sample_n=1,
        pts_time_s=10.0,
        fps=30.0,
        frame_idx=300,
        sampling_source="transnetv2",
    )
    organizer = FrameSampleCandidate(
        sample_n=1,
        pts_time_s=10.3,
        fps=30.0,
        frame_idx=309,
        sampling_source="map-keyframes",
        keyframe_n=1,
    )
    next_organizer = FrameSampleCandidate(
        sample_n=2,
        pts_time_s=10.4,
        fps=30.0,
        frame_idx=312,
        sampling_source="map-keyframes",
        keyframe_n=2,
    )

    merged = dedupe_samples([adaptive, next_organizer, organizer], tolerance_s=0.5)

    assert len(merged) == 2
    assert all(sample.sampling_source == "map-keyframes" for sample in merged)
    assert [sample.frame_idx for sample in merged] == [309, 312]
