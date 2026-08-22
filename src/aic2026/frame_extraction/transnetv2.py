"""Adapters for external TransNetV2 scene files and inference scripts."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from queue import Empty, Queue
from pathlib import Path
from threading import Thread

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
        command = [sys.executable, "-u", *command]
    if weights is not None:
        command.extend(["--weights", str(weights.resolve())])

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    print("$ " + " ".join(command), file=sys.stderr, flush=True)
    process = subprocess.Popen(
        command,
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    assert process.stdout is not None
    lines: Queue[str | None] = Queue()

    def forward_output() -> None:
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    Thread(target=forward_output, name="transnetv2-output", daemon=True).start()
    started_at = time.monotonic()
    print(
        f"[transnetv2] process_started pid={process.pid}; streaming output and heartbeat every 30s",
        file=sys.stderr,
        flush=True,
    )
    while True:
        try:
            line = lines.get(timeout=30)
        except Empty:
            if process.poll() is None:
                elapsed = time.monotonic() - started_at
                print(
                    f"[transnetv2] still_running elapsed_s={elapsed:.0f}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            break
        if line is None:
            break
        print(line, end="", file=sys.stderr, flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"TransNetV2 inference failed for {video_path} with code {return_code}")
    scenes_path = Path(str(linked_video) + ".scenes.txt")
    if not scenes_path.is_file():
        raise FileNotFoundError(f"TransNetV2 did not create scenes file: {scenes_path}")
    return scenes_path
