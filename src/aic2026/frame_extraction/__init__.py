"""Offline frame extraction and shot-based sampling utilities."""

from .discovery import LocatedInputs, find_support_file, find_video_file, locate_inputs
from .pipeline import extract_frame_samples
from .sampling import (
    FrameSampleCandidate,
    adaptive_samples_from_shots,
    fallback_samples,
    map_keyframe_samples,
)
from .transnetv2 import build_shot_records, parse_scenes_txt

__all__ = [
    "FrameSampleCandidate",
    "LocatedInputs",
    "adaptive_samples_from_shots",
    "build_shot_records",
    "extract_frame_samples",
    "fallback_samples",
    "find_support_file",
    "find_video_file",
    "locate_inputs",
    "map_keyframe_samples",
    "parse_scenes_txt",
]
