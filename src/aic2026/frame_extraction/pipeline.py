"""Extract sampled frames from raw video into versioned JSONL records."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from aic2026.contracts import FrameSampleRecord

from .ffmpeg import extract_frames_by_index, probe_video
from .sampling import FrameSampleCandidate, fallback_samples, map_keyframe_samples


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Extracted frame is outside output root: {path}") from error


def choose_samples(
    *,
    map_csv: Path | None,
    fps: float,
    limit: int | None,
    fallback_timestamps_s: list[float],
) -> list[FrameSampleCandidate]:
    if map_csv is not None:
        return map_keyframe_samples(map_csv, limit=limit)
    return fallback_samples(fps=fps, timestamps_s=fallback_timestamps_s, limit=limit)


def extract_frame_samples(
    *,
    video_id: str,
    video_path: Path,
    output_root: Path,
    frames_dir: Path,
    map_csv: Path | None = None,
    limit: int | None = None,
    fallback_timestamps_s: list[float] | None = None,
    jpeg_quality: int = 2,
) -> list[FrameSampleRecord]:
    if fallback_timestamps_s is None:
        fallback_timestamps_s = [0, 1, 2, 3, 4]
    probe = probe_video(video_path)
    samples = choose_samples(
        map_csv=map_csv,
        fps=probe.fps,
        limit=limit,
        fallback_timestamps_s=fallback_timestamps_s,
    )
    output_root = output_root.expanduser().resolve()
    frames_dir = frames_dir.expanduser().resolve()
    frame_paths = {
        sample.frame_idx: frames_dir / f"{sample.sample_n:03d}.jpg"
        for sample in samples
    }
    extract_frames_by_index(
        video_path=video_path,
        outputs=[(sample.frame_idx, frame_paths[sample.frame_idx]) for sample in samples],
        jpeg_quality=jpeg_quality,
    )
    records: list[FrameSampleRecord] = []
    for sample in samples:
        frame_path = frame_paths[sample.frame_idx]
        with Image.open(frame_path) as image:
            width, height = image.size
            image.verify()
        records.append(
            FrameSampleRecord(
                video_id=video_id,
                frame_uid=f"{video_id}:{sample.frame_idx}",
                sample_n=sample.sample_n,
                keyframe_n=sample.keyframe_n,
                frame_idx=sample.frame_idx,
                pts_time_s=sample.pts_time_s,
                fps=sample.fps,
                frame_relpath=_relative_to_root(frame_path, output_root),
                width=width,
                height=height,
                source_video=str(video_path),
                sampling_source=sample.sampling_source,  # type: ignore[arg-type]
                extraction_method="frame-index-select",
                shot_id=sample.shot_id,
                shot_start_idx=sample.shot_start_idx,
                shot_end_idx=sample.shot_end_idx,
            )
        )
    return records
