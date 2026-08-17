"""Timestamp-driven dense frame decoding for TRAKE."""

from __future__ import annotations

import math
from pathlib import Path

from aic2026.contracts import FrameRef


def advance_sampling_clock(
    pts_time_s: float, next_target_s: float, sampling_fps: float
) -> tuple[bool, float]:
    """Select the first decoded frame at/after each timestamp on the 5 FPS clock."""
    if sampling_fps <= 0 or not math.isfinite(sampling_fps):
        raise ValueError("sampling_fps must be positive and finite")
    if pts_time_s < 0 or not math.isfinite(pts_time_s):
        raise ValueError("pts_time_s must be non-negative and finite")
    interval = 1.0 / sampling_fps
    if pts_time_s + 1e-9 < next_target_s:
        return False, next_target_s
    while next_target_s <= pts_time_s + 1e-9:
        next_target_s += interval
    return True, next_target_s


def decode_dense_frames(
    *,
    video_path: Path,
    video_id: str,
    output_root: Path,
    sampling_fps: float = 5.0,
    jpeg_quality: int = 90,
    limit: int | None = None,
) -> list[FrameRef]:
    """Decode by real PTS while preserving the 0-based decoded frame index."""
    try:
        import av
    except ImportError as error:  # pragma: no cover - exercised in the cloud profile
        raise RuntimeError("PyAV is required; install requirements/siglip2.txt") from error

    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be within [1, 100]")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    image_dir = output_root / "dense_frames" / video_id
    image_dir.mkdir(parents=True, exist_ok=True)
    records: list[FrameRef] = []
    next_target_s = 0.0

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate
        source_fps = float(rate) if rate else 0.0
        if source_fps <= 0 or not math.isfinite(source_fps):
            raise ValueError(f"Video reports invalid FPS: {video_path}")
        decoded_index = 0
        for frame in container.decode(stream):
            current_index = decoded_index
            decoded_index += 1
            if frame.pts is None or frame.time_base is None:
                continue
            pts_time_s = float(frame.pts * frame.time_base)
            selected, next_target_s = advance_sampling_clock(
                pts_time_s, next_target_s, sampling_fps
            )
            if not selected:
                continue
            image = frame.to_image().convert("RGB")
            relative = Path("dense_frames") / video_id / f"{current_index}.jpg"
            target = output_root / relative
            temporary = target.with_suffix(".jpg.partial")
            image.save(temporary, format="JPEG", quality=jpeg_quality)
            temporary.replace(target)
            records.append(
                FrameRef(
                    video_id=video_id,
                    frame_uid=f"{video_id}:{current_index}",
                    keyframe_n=None,
                    frame_idx=current_index,
                    pts_time_s=pts_time_s,
                    fps=source_fps,
                    frame_relpath=relative.as_posix(),
                    width=image.width,
                    height=image.height,
                )
            )
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No timestamped video frames decoded from {video_path}")
    return records
