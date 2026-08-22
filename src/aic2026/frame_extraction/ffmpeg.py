"""Small ffmpeg/ffprobe wrappers used by offline frame extraction."""

from __future__ import annotations

import json
import shutil
import subprocess
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
