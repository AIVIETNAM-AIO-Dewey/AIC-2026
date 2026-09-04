"""Authoritative source-frame indexing for local video navigation and submission.

Retrieval operates on sparse organizer keyframes.  Submission, however, uses
the original video's zero-based frame index.  This module keeps those two
identities separate and provides one deterministic frame-index/timestamp
contract without importing or changing the retrieval pipeline.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any, Protocol


class FrameTimelineStore(Protocol):
    """Small metadata surface required by :class:`SourceFrameIndex`."""

    def video_frames(self, video_id: str) -> Sequence[Mapping[str, Any]]: ...


class SourceFrameIndex:
    """Resolve exact source-frame IDs and stable playback timestamps.

    Frame indices are always zero based.  Exact organizer keyframes retain
    their stored timestamps.  Frames between two keyframes use piecewise
    linear interpolation between those exact anchors; frames outside the
    anchor range use the nominal video FPS.  The nearest indexed keyframe is
    returned only as a clearly labelled preview and related-search seed.
    """

    schema_version = "video.frame-timeline.v1"
    frame_schema_version = "video.source-frame.v1"
    frame_index_base = 0

    def __init__(self, metadata: FrameTimelineStore, media_info_root: Path) -> None:
        self.metadata = metadata
        self.media_info_root = Path(media_info_root)

    @staticmethod
    def canonical_video_id(video_id: str) -> str:
        return str(video_id).strip().upper().replace("-", "_")

    @staticmethod
    def _positive_finite(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) and number > 0 else None

    def _media_duration(self, video_id: str) -> float | None:
        path = self.media_info_root / f"{video_id}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return self._positive_finite(payload.get("length"))

    @staticmethod
    def _fps(frames: Sequence[Mapping[str, Any]]) -> float | None:
        declared = [
            value
            for frame in frames
            if (value := SourceFrameIndex._positive_finite(frame.get("fps"))) is not None
        ]
        if declared:
            return float(median(declared))
        derived = [
            float(frame["frame_idx"]) / float(frame["pts_time_s"])
            for frame in frames
            if SourceFrameIndex._positive_finite(frame.get("pts_time_s")) is not None
            and float(frame.get("frame_idx", -1)) >= 0
        ]
        return float(median(derived)) if derived else None

    @staticmethod
    def _validated_frames(
        video_id: str,
        values: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        frames: list[dict[str, Any]] = []
        previous_idx = -1
        previous_time = -1.0
        for value in values:
            try:
                frame_idx = int(value["frame_idx"])
                pts_time_s = float(value["pts_time_s"])
                keyframe_n = int(value["keyframe_n"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid indexed-frame metadata for {video_id}") from exc
            if (
                frame_idx < 0
                or keyframe_n < 0
                or not math.isfinite(pts_time_s)
                or pts_time_s < 0
                or frame_idx <= previous_idx
                or pts_time_s <= previous_time
            ):
                raise ValueError(f"Indexed-frame timeline for {video_id} is not strictly ordered")
            frame = dict(value)
            frame["video_id"] = video_id
            frame["frame_idx"] = frame_idx
            frame["keyframe_n"] = keyframe_n
            frame["pts_time_s"] = pts_time_s
            frame["frame_uid"] = f"{video_id}:{frame_idx}"
            frame["image_relpath"] = str(
                frame.get("image_relpath") or frame.get("frame_relpath") or ""
            )
            frame["submission_string"] = f"{video_id}, {frame_idx}"
            frames.append(frame)
            previous_idx = frame_idx
            previous_time = pts_time_s
        return tuple(frames)

    @lru_cache(maxsize=64)  # noqa: B019 - one long-lived server instance
    def _contract(self, video_id: str) -> dict[str, Any] | None:
        canonical = self.canonical_video_id(video_id)
        frames = self._validated_frames(canonical, self.metadata.video_frames(canonical))
        if not frames:
            return None
        fps = self._fps(frames)
        if fps is None:
            raise ValueError(f"FPS is unavailable for {canonical}")

        media_duration = self._media_duration(canonical)
        final_anchor = frames[-1]
        duration_bound = (
            max(0, math.ceil(media_duration * fps) - 1)
            if media_duration is not None
            else 0
        )
        max_frame_idx = max(int(final_anchor["frame_idx"]), duration_bound)
        resolved_end_time = self._interpolate_time(max_frame_idx, frames, fps)
        duration_s = max(
            float(final_anchor["pts_time_s"]),
            resolved_end_time,
            media_duration or 0.0,
        )
        return {
            "schema_version": self.schema_version,
            "video_id": canonical,
            "frame_index_base": self.frame_index_base,
            "fps": fps,
            "duration_s": round(duration_s, 6),
            "media_duration_s": media_duration,
            "max_frame_idx": max_frame_idx,
            "frame_count": max_frame_idx + 1,
            "keyframe_count": len(frames),
            "timing_method": "exact-anchor-piecewise-linear-v1",
            "keyframes": frames,
        }

    def timeline(self, video_id: str) -> dict[str, Any] | None:
        """Return a copy-safe public timeline contract."""

        contract = self._contract(self.canonical_video_id(video_id))
        if contract is None:
            return None
        return {
            **contract,
            "keyframes": [dict(frame) for frame in contract["keyframes"]],
        }

    @staticmethod
    def _interpolate_time(
        frame_idx: int,
        frames: Sequence[Mapping[str, Any]],
        fps: float,
    ) -> float:
        indices = [int(frame["frame_idx"]) for frame in frames]
        position = bisect_left(indices, frame_idx)
        if position < len(frames) and indices[position] == frame_idx:
            return float(frames[position]["pts_time_s"])
        if position == 0:
            first_idx = indices[0]
            first_time = float(frames[0]["pts_time_s"])
            if first_idx > 0 and first_time > 0:
                return first_time * (frame_idx / first_idx)
            return frame_idx / fps
        if position < len(frames):
            left = frames[position - 1]
            right = frames[position]
            left_idx = int(left["frame_idx"])
            right_idx = int(right["frame_idx"])
            fraction = (frame_idx - left_idx) / (right_idx - left_idx)
            return float(left["pts_time_s"]) + fraction * (
                float(right["pts_time_s"]) - float(left["pts_time_s"])
            )
        last = frames[-1]
        return float(last["pts_time_s"]) + (frame_idx - int(last["frame_idx"])) / fps

    @staticmethod
    def _nearest_anchor(
        frame_idx: int,
        frames: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        indices = [int(frame["frame_idx"]) for frame in frames]
        position = bisect_left(indices, frame_idx)
        before = frames[max(0, position - 1)]
        after = frames[min(len(frames) - 1, position)]
        before_delta = abs(frame_idx - int(before["frame_idx"]))
        after_delta = abs(int(after["frame_idx"]) - frame_idx)
        return before if before_delta <= after_delta else after

    def resolve(self, video_id: str, frame_idx: int) -> dict[str, Any] | None:
        """Resolve an exact source-frame index without substituting its identity."""

        canonical = self.canonical_video_id(video_id)
        if isinstance(frame_idx, bool) or not isinstance(frame_idx, int) or frame_idx < 0:
            return None
        contract = self._contract(canonical)
        if contract is None or frame_idx > int(contract["max_frame_idx"]):
            return None

        frames = contract["keyframes"]
        indices = [int(frame["frame_idx"]) for frame in frames]
        position = bisect_left(indices, frame_idx)
        exact = (
            frames[position]
            if position < len(frames) and int(frames[position]["frame_idx"]) == frame_idx
            else None
        )
        preview = exact or self._nearest_anchor(frame_idx, frames)
        pts_time_s = (
            float(exact["pts_time_s"])
            if exact is not None
            else round(
                self._interpolate_time(frame_idx, frames, float(contract["fps"])),
                6,
            )
        )
        image_relpath = str(exact.get("image_relpath") or "") if exact is not None else ""
        preview_image_relpath = str(preview.get("image_relpath") or "")
        return {
            "video_id": canonical,
            "frame_idx": frame_idx,
            "frame_uid": f"{canonical}:{frame_idx}",
            "keyframe_n": int(exact["keyframe_n"]) if exact is not None else None,
            "pts_time_s": pts_time_s,
            "fps": float(contract["fps"]),
            "image_relpath": image_relpath,
            "indexed_keyframe": exact is not None,
            "validation": "canonical" if exact is not None else "source_timeline",
            "submission_string": f"{canonical}, {frame_idx}",
            "frame_index_base": self.frame_index_base,
            "max_frame_idx": int(contract["max_frame_idx"]),
            "duration_s": float(contract["duration_s"]),
            "timing_method": str(contract["timing_method"]),
            "preview_frame_idx": int(preview["frame_idx"]),
            "preview_keyframe_n": int(preview["keyframe_n"]),
            "preview_pts_time_s": float(preview["pts_time_s"]),
            "preview_image_relpath": preview_image_relpath,
            "related_seed_frame_idx": int(preview["frame_idx"]),
        }


__all__ = ["SourceFrameIndex"]
