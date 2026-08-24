"""FFmpeg and FFprobe wrappers for video probing and high-quality frame extraction."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


class FFmpegError(RuntimeError):
    """Raised when ffmpeg or ffprobe cannot complete the requested operation."""


@dataclass(frozen=True, slots=True)
class VideoProbe:
    fps: float
    width: int | None
    height: int | None
    duration_s: float | None
    frame_count: int | None


def require_binary(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise FFmpegError(f"Required executable is not available on PATH: {name}")
    return resolved


def _parse_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


def _parse_optional_float(value: object) -> float | None:
    if value in (None, "N/A", ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_int(value: object) -> int | None:
    if value in (None, "N/A", ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def probe_video(path: Path) -> VideoProbe:
    ffprobe = require_binary("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,width,height,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe failed for {path}: {result.stderr.strip()[:1000]}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise FFmpegError(f"No video stream found: {path}")
    stream = streams[0]
    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    if fps is None or fps <= 0:
        raise FFmpegError(f"Unable to determine FPS for {path}")
    duration = _parse_optional_float(stream.get("duration"))
    if duration is None:
        duration = _parse_optional_float((payload.get("format") or {}).get("duration"))
    return VideoProbe(
        fps=fps,
        width=_parse_optional_int(stream.get("width")),
        height=_parse_optional_int(stream.get("height")),
        duration_s=duration,
        frame_count=_parse_optional_int(stream.get("nb_frames")),
    )


def extract_frame(
    *,
    video_path: Path,
    pts_time_s: float,
    output_path: Path,
    jpeg_quality: int = 2,
) -> None:
    ffmpeg = require_binary("ffmpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{pts_time_s:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        str(jpeg_quality),
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip()[:1000]
        raise FFmpegError(f"ffmpeg failed for {video_path} at {pts_time_s:.3f}s: {stderr}")
    if not output_path.is_file():
        raise FFmpegError(f"ffmpeg did not create expected frame: {output_path}")


def extract_frames_by_index(
    *,
    video_path: Path,
    outputs: list[tuple[int, Path]],
    jpeg_quality: int = 2,
) -> None:
    """Decode once and extract exact zero-based frame ordinals using FFmpeg select filter."""
    if not outputs:
        return
    ordered = sorted(outputs, key=lambda item: item[0])
    indices = [frame_idx for frame_idx, _ in ordered]
    if any(frame_idx < 0 for frame_idx in indices):
        raise ValueError("frame indices must be non-negative")
    if len(indices) != len(set(indices)):
        raise ValueError("frame indices must be unique")

    ffmpeg = require_binary("ffmpeg")
    for _, destination in ordered:
        destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = ordered[0][1].parent
    select_expression = "+".join(f"eq(n\\,{frame_idx})" for frame_idx in indices)
    started_at = time.monotonic()
    print(
        f"[frame_index_extract] requested={len(indices)} "
        f"first_idx={indices[0]} last_idx={indices[-1]}",
        file=sys.stderr,
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix=".frame-index-extract-", dir=temporary_parent) as raw:
        temporary_dir = Path(raw)
        pattern = temporary_dir / "%08d.jpg"
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-stats_period",
            "10",
            "-progress",
            "pipe:2",
            "-nostats",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"select={select_expression}",
            "-vsync",
            "0",
            "-q:v",
            str(jpeg_quality),
            str(pattern),
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stderr is not None
        stderr_tail: deque[str] = deque(maxlen=80)
        progress: dict[str, str] = {}
        for raw_line in process.stderr:
            line = raw_line.strip()
            if not line:
                continue
            stderr_tail.append(line)
            key, separator, value = line.partition("=")
            if separator:
                progress[key] = value
                if key == "progress":
                    details = " ".join(
                        f"{name}={progress[name]}"
                        for name in ("frame", "fps", "out_time", "speed", "progress")
                        if name in progress
                    )
                    print(f"[frame_index_extract] {details}", file=sys.stderr, flush=True)
                    progress.clear()
        return_code = process.wait()
        if return_code != 0:
            raise FFmpegError(
                f"ffmpeg exact-index extraction failed for {video_path}: "
                f"{' | '.join(stderr_tail)[-2000:]}"
            )
        extracted = sorted(temporary_dir.glob("*.jpg"))
        if len(extracted) != len(ordered):
            # Fallback to individual frame seeks if select filter dropped non-keyframes
            print(
                f"[frame_index_extract] fallback to individual timestamp seeks for {len(ordered)} frames",
                file=sys.stderr,
                flush=True,
            )
            probe = probe_video(video_path)
            for f_idx, out_p in ordered:
                pts = f_idx / probe.fps
                extract_frame(video_path=video_path, pts_time_s=pts, output_path=out_p, jpeg_quality=jpeg_quality)
            return

        for source, (_, destination) in zip(extracted, ordered, strict=True):
            os.replace(source, destination)
    print(
        f"[frame_index_extract] completed={len(ordered)} "
        f"elapsed_s={time.monotonic() - started_at:.3f}",
        file=sys.stderr,
        flush=True,
    )
