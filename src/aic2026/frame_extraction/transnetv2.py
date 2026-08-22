"""Adapters for external TransNetV2 scene files and inference scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from aic2026.contracts import ShotRecord


def parse_scenes_txt(path: Path) -> list[tuple[int, int]]:
    scenes: list[tuple[int, int]] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) != 2:
                raise ValueError(f"Invalid scene row at {path}:{line_number}: {line!r}")
            start, end = int(parts[0]), int(parts[1])
            if start < 0 or end < start:
                raise ValueError(f"Invalid scene bounds at {path}:{line_number}")
            scenes.append((start, end))
    if not scenes:
        raise ValueError(f"Scene file is empty: {path}")
    return scenes


def build_shot_records(
    *,
    video_id: str,
    scenes: list[tuple[int, int]],
    fps: float,
    source_video: Path,
) -> list[ShotRecord]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    return [
        ShotRecord(
            video_id=video_id,
            shot_id=f"{video_id}:s{index:05d}",
            shot_start_idx=start,
            shot_end_idx=end,
            start_time_s=start / fps,
            end_time_s=end / fps,
            fps=fps,
            source_video=str(source_video),
        )
        for index, (start, end) in enumerate(scenes, start=1)
    ]


def run_transnetv2_inference(
    *,
    video_path: Path,
    entrypoint: Path,
    weights: Path | None,
    work_dir: Path,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    linked_video = work_dir / video_path.name
    if not linked_video.exists():
        os.symlink(video_path.resolve(), linked_video)

    command = [str(entrypoint.resolve()), str(linked_video)]
    if entrypoint.suffix == ".py":
        command = [sys.executable, *command]
    if weights is not None:
        command.extend(["--weights", str(weights.resolve())])

    result = subprocess.run(command, cwd=work_dir, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()[:2000]
        raise RuntimeError(f"TransNetV2 inference failed for {video_path}: {stderr}")
    scenes_path = Path(str(linked_video) + ".scenes.txt")
    if not scenes_path.is_file():
        raise FileNotFoundError(f"TransNetV2 did not create scenes file: {scenes_path}")
    return scenes_path
