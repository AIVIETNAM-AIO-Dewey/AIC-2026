"""Frame sampling policies for map-keyframes, fallback smoke, and shot records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from aic2026.common.frame_manifest import read_frame_map
from aic2026.contracts import ShotRecord


@dataclass(frozen=True, slots=True)
class FrameSampleCandidate:
    sample_n: int
    pts_time_s: float
    fps: float
    frame_idx: int
    sampling_source: str
    keyframe_n: int | None = None
    shot_id: str | None = None
    shot_start_idx: int | None = None
    shot_end_idx: int | None = None


def map_keyframe_samples(map_csv: Path, *, limit: int | None = None) -> list[FrameSampleCandidate]:
    rows = read_frame_map(map_csv)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    return [
        FrameSampleCandidate(
            sample_n=index,
            keyframe_n=row.keyframe_n,
            pts_time_s=row.pts_time_s,
            fps=row.fps,
            frame_idx=row.frame_idx,
            sampling_source="map-keyframes",
        )
        for index, row in enumerate(rows, start=1)
    ]


def fallback_samples(
    *,
    fps: float,
    timestamps_s: list[float] | tuple[float, ...] = (0, 1, 2, 3, 4),
    limit: int | None = None,
) -> list[FrameSampleCandidate]:
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError("fps must be positive and finite")
    values = list(timestamps_s)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        values = values[:limit]
    return [
        FrameSampleCandidate(
            sample_n=index,
            pts_time_s=float(timestamp),
            fps=fps,
            frame_idx=round(float(timestamp) * fps),
            sampling_source="fallback",
        )
        for index, timestamp in enumerate(values, start=1)
    ]


def _evenly_pick(values: list[int], count: int) -> list[int]:
    if count >= len(values):
        return values
    if count == 1:
        return [values[len(values) // 2]]
    positions = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return [values[position] for position in positions]


def sample_indices_for_shot(
    *,
    shot_start_idx: int,
    shot_end_idx: int,
    fps: float,
    short_shot_max_s: float = 3.0,
    cadence_s: float = 1.5,
    max_frames_per_shot: int = 10,
) -> list[int]:
    if shot_end_idx < shot_start_idx:
        raise ValueError("shot_end_idx must be >= shot_start_idx")
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError("fps must be positive and finite")
    if cadence_s <= 0 or max_frames_per_shot < 1:
        raise ValueError("cadence_s and max_frames_per_shot must be positive")

    duration_s = (shot_end_idx - shot_start_idx + 1) / fps
    if duration_s <= short_shot_max_s:
        return [round((shot_start_idx + shot_end_idx) / 2)]

    offsets: list[float] = []
    current = cadence_s / 2
    while current < duration_s:
        offsets.append(current)
        current += cadence_s
    if not offsets:
        offsets = [duration_s / 2]

    indices = sorted(
        {
            min(shot_end_idx, max(shot_start_idx, shot_start_idx + round(offset * fps)))
            for offset in offsets
        }
    )
    return _evenly_pick(indices, max_frames_per_shot)


def adaptive_samples_from_shots(
    shots: list[ShotRecord],
    *,
    sampling_source: str = "transnetv2",
    cadence_s: float = 1.5,
    max_frames_per_shot: int = 10,
) -> list[FrameSampleCandidate]:
    samples: list[FrameSampleCandidate] = []
    for shot in shots:
        indices = sample_indices_for_shot(
            shot_start_idx=shot.shot_start_idx,
            shot_end_idx=shot.shot_end_idx,
            fps=shot.fps,
            cadence_s=cadence_s,
            max_frames_per_shot=max_frames_per_shot,
        )
        for frame_idx in indices:
            samples.append(
                FrameSampleCandidate(
                    sample_n=len(samples) + 1,
                    pts_time_s=frame_idx / shot.fps,
                    fps=shot.fps,
                    frame_idx=frame_idx,
                    sampling_source=sampling_source,
                    shot_id=shot.shot_id,
                    shot_start_idx=shot.shot_start_idx,
                    shot_end_idx=shot.shot_end_idx,
                )
            )
    return samples


def dedupe_samples(
    samples: list[FrameSampleCandidate],
    *,
    tolerance_s: float = 0.5,
) -> list[FrameSampleCandidate]:
    priority = {
        "map-keyframes": 0,
        "organizer": 0,
        "transnetv2": 1,
        "interval": 2,
        "fallback": 3,
    }
    ordered = sorted(
        samples,
        key=lambda sample: (
            sample.pts_time_s,
            priority.get(sample.sampling_source, 99),
            sample.frame_idx,
        ),
    )
    kept: list[FrameSampleCandidate] = []
    for sample in ordered:
        if any(abs(sample.pts_time_s - previous.pts_time_s) <= tolerance_s for previous in kept):
            continue
        kept.append(sample)
    return [
        FrameSampleCandidate(
            sample_n=index,
            pts_time_s=sample.pts_time_s,
            fps=sample.fps,
            frame_idx=sample.frame_idx,
            sampling_source=sample.sampling_source,
            keyframe_n=sample.keyframe_n,
            shot_id=sample.shot_id,
            shot_start_idx=sample.shot_start_idx,
            shot_end_idx=sample.shot_end_idx,
        )
        for index, sample in enumerate(kept, start=1)
    ]
