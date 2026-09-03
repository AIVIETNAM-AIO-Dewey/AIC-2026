"""Canonical frame metadata resolver used by all online modalities."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..modalities.lexical import repair_mojibake


# Preserve the private helper name used by older metadata callers while
# sharing the conservative decoder used by both text retrieval modalities.
_repair_mojibake = repair_mojibake


class FrameMetadataStore:
    """Resolve canonical frame metadata without scanning the corpus at startup."""

    def __init__(self, data_root: Path, ocr: Any) -> None:
        self.video_metadata_dir = data_root / "visual_embeddings" / "metaclip2" / "video_metadata"
        self.unified_dir = data_root / "unified_metadata"
        self.ocr = ocr

    @lru_cache(maxsize=32)
    def video_frames(self, video_id: str) -> tuple[dict[str, Any], ...]:
        canonical = video_id.upper().replace("-", "_")
        path = self.video_metadata_dir / f"{canonical}.jsonl"
        if not path.is_file():
            return ()
        frames: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                item["video_id"] = canonical
                item["image_relpath"] = item.get("image_relpath") or item.get("frame_relpath", "")
                item["submission_string"] = f"{canonical}, {int(item['frame_idx'])}"
                frames.append(item)
        return tuple(frames)

    @lru_cache(maxsize=8)
    def unified_frames(self, video_id: str) -> dict[int, dict[str, Any]]:
        path = self.unified_dir / f"{video_id}.jsonl"
        if not path.is_file():
            return {}
        result: dict[int, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                item["ocr_text"] = _repair_mojibake(str(item.get("ocr_text", "")))
                result[int(item["keyframe_n"])] = item
        return result

    def detail(self, video_id: str, keyframe_n: int) -> dict[str, Any] | None:
        unified = self.unified_frames(video_id).get(keyframe_n)
        if unified is not None:
            return dict(unified)
        for frame in self.video_frames(video_id):
            if int(frame["keyframe_n"]) == keyframe_n:
                item = dict(frame)
                item["ocr_text"] = self._lookup_ocr_text(str(item["frame_uid"]))
                return item
        return None

    def frame_by_idx(self, video_id: str, frame_idx: int) -> dict[str, Any] | None:
        canonical = video_id.upper().replace("-", "_")
        for frame in self.video_frames(canonical):
            if int(frame["frame_idx"]) == int(frame_idx):
                item = dict(frame)
                item["video_id"] = canonical
                item["ocr_text"] = self._lookup_ocr_text(str(item["frame_uid"]))
                item["submission_string"] = f"{canonical}, {int(frame_idx)}"
                return item
        return None

    def _lookup_ocr_text(self, frame_uid: str) -> str:
        """Use the canonical bulk lookup API for optional OCR enrichment."""

        lookup_many = getattr(self.ocr, "lookup_many", None)
        if not callable(lookup_many):
            return ""
        try:
            return str(lookup_many([frame_uid]).get(frame_uid, "") or "")
        except Exception:
            # OCR is optional for metadata/detail routes; a stale OCR index
            # must not make canonical frame metadata unavailable.
            return ""

    def timeline(self, video_id: str) -> dict[str, Any] | None:
        canonical = video_id.upper().replace("-", "_")
        frames = [dict(frame) for frame in self.video_frames(canonical)]
        if not frames:
            return None
        for frame in frames:
            frame["video_id"] = canonical
            frame["submission_string"] = f"{canonical}, {int(frame['frame_idx'])}"
        fps_samples = [
            float(frame["frame_idx"]) / float(frame["pts_time_s"])
            for frame in frames
            if float(frame.get("pts_time_s", 0.0)) > 0
        ]
        fps_samples.sort()
        fps = round(fps_samples[len(fps_samples) // 2], 4) if fps_samples else None
        return {
            "video_id": canonical,
            "fps": fps,
            "keyframe_count": len(frames),
            "keyframes": frames,
        }


__all__ = ["FrameMetadataStore"]
