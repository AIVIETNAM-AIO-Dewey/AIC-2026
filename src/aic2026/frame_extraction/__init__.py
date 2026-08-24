"""Offline frame extraction subsystem."""

from .ffmpeg import (
    FFmpegError,
    VideoProbe,
    extract_frame,
    extract_frames_by_index,
    probe_video,
)
from .sampling import (
    FrameSampleCandidate,
    adaptive_samples_from_shots,
    dedupe_samples,
    fallback_samples,
    map_keyframe_samples,
    sample_indices_for_shot,
)
from .transnetv2 import (
    TransNetV2InferenceResult,
    ensure_transnet_module,
    ensure_transnet_weights,
    load_transnetv2_model,
    run_transnetv2_inference,
)

__all__ = [
    "FFmpegError",
    "FrameSampleCandidate",
    "TransNetV2InferenceResult",
    "VideoProbe",
    "adaptive_samples_from_shots",
    "dedupe_samples",
    "ensure_transnet_module",
    "ensure_transnet_weights",
    "extract_frame",
    "extract_frames_by_index",
    "fallback_samples",
    "load_transnetv2_model",
    "map_keyframe_samples",
    "probe_video",
    "run_transnetv2_inference",
    "sample_indices_for_shot",
]
