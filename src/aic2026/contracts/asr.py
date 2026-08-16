"""Pydantic contracts for ASR (speech-to-text) pipeline artifacts.

Schema version: aic26.asr_segments.v1

Each video produces one JSONL file where every line is a serialized
``AsrSegmentRecord``.  The companion ``AsrVideoManifest`` tracks
pipeline provenance and resume state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .models import StrictModel

# ──────────────────────────────────────────────────────────────────────
# Sub-models
# ──────────────────────────────────────────────────────────────────────


class AsrKeyframeRef(StrictModel):
    """A keyframe that falls within an ASR segment's time range."""

    keyframe_n: int = Field(ge=1)
    frame_idx: int = Field(ge=0)
    pts_time_s: float = Field(ge=0)
    frame_uid: str = Field(min_length=3)

    @model_validator(mode="after")
    def validate_uid_format(self) -> AsrKeyframeRef:
        # frame_uid must contain a colon separating video_id and frame_idx
        if ":" not in self.frame_uid:
            raise ValueError("frame_uid must be in 'video_id:frame_idx' format")
        parts = self.frame_uid.split(":", 1)
        if not parts[0] or not parts[1].isdigit():
            raise ValueError("frame_uid must be in 'video_id:frame_idx' format")
        if int(parts[1]) != self.frame_idx:
            raise ValueError(
                f"frame_uid suffix {parts[1]} does not match frame_idx {self.frame_idx}"
            )
        return self


# ──────────────────────────────────────────────────────────────────────
# Core record — one per ASR segment
# ──────────────────────────────────────────────────────────────────────


class AsrSegmentRecord(StrictModel):
    """A single contiguous speech segment transcribed from a video."""

    schema_version: Literal["aic26.asr_segments.v1"] = "aic26.asr_segments.v1"
    segment_id: str = Field(min_length=1)
    video_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    transcript_raw: str = Field(min_length=1)
    transcript_normalized: str = Field(min_length=1)
    language: str = Field(default="vi", min_length=2, max_length=5)
    keyframes: list[AsrKeyframeRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_time_range(self) -> AsrSegmentRecord:
        if self.end_ms <= self.start_ms:
            raise ValueError(
                f"end_ms ({self.end_ms}) must be greater than start_ms ({self.start_ms})"
            )
        return self

    @model_validator(mode="after")
    def validate_segment_id_prefix(self) -> AsrSegmentRecord:
        if not self.segment_id.startswith(self.video_id + ":"):
            raise ValueError(f"segment_id must be prefixed with '{self.video_id}:'")
        return self

    @model_validator(mode="after")
    def validate_keyframe_ordering(self) -> AsrSegmentRecord:
        for kf in self.keyframes:
            kf_ms = kf.pts_time_s * 1000
            if kf_ms < self.start_ms - 1 or kf_ms > self.end_ms + 1:
                raise ValueError(
                    f"keyframe {kf.frame_uid} at {kf.pts_time_s}s is outside "
                    f"segment [{self.start_ms}ms, {self.end_ms}ms]"
                )
        return self

    @field_validator("keyframes", mode="after")
    @classmethod
    def validate_keyframe_uniqueness(
        cls,
        keyframes: list[AsrKeyframeRef],
    ) -> list[AsrKeyframeRef]:
        uids = [kf.frame_uid for kf in keyframes]
        if len(uids) != len(set(uids)):
            raise ValueError("duplicate keyframe frame_uid values within a segment")
        return keyframes


# ──────────────────────────────────────────────────────────────────────
# Per-video manifest — provenance + resume state
# ──────────────────────────────────────────────────────────────────────


class AsrVideoManifest(StrictModel):
    """Pipeline provenance and status for one processed video."""

    schema_version: Literal["aic26.asr_manifest.v1"] = "aic26.asr_manifest.v1"
    video_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    status: Literal["running", "completed", "failed", "skipped"]
    segment_count: int = Field(ge=0)
    keyframe_count: int = Field(ge=0)
    audio_duration_s: float = Field(ge=0)
    model_id: str = Field(min_length=1)
    engine: str = Field(min_length=1)
    config: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    ended_at: datetime | None = None
    error_message: str | None = None

    @field_validator("started_at", "ended_at", mode="before")
    @classmethod
    def accept_iso_datetime(cls, value: Any) -> Any:
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    @model_validator(mode="after")
    def validate_completion(self) -> AsrVideoManifest:
        if self.status in {"completed", "failed"} and self.ended_at is None:
            raise ValueError("terminal manifests require ended_at")
        if self.status == "running" and self.ended_at is not None:
            raise ValueError("running manifests cannot have ended_at")
        if self.status == "failed" and self.error_message is None:
            raise ValueError("failed manifests require error_message")
        if self.status == "skipped" and self.segment_count != 0:
            raise ValueError("skipped manifests must have zero segments")
        return self
