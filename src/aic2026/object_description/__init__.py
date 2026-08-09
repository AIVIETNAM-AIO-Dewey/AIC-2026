"""Organizer object boxes, SAM segmentation, and DAM captioning."""

from .caption import DAM_PROMPT, normalize_caption
from .dam_backend import DamCaptioner, verify_installed_dam_revision
from .geometry import Detection, FilterConfig, filter_detections, normalized_to_pixels
from .organizer import load_organizer_detections, parse_organizer_detections
from .pipeline import prepare_masks, run_descriptions, validate_description_settings
from .sam_backend import SamMaskGenerator
from .validation import (
    validate_description_stage_inputs,
    validate_mask_stage_inputs,
    validate_published_object_artifact,
)

__all__ = [
    "DAM_PROMPT",
    "DamCaptioner",
    "Detection",
    "FilterConfig",
    "SamMaskGenerator",
    "filter_detections",
    "load_organizer_detections",
    "normalize_caption",
    "normalized_to_pixels",
    "parse_organizer_detections",
    "prepare_masks",
    "run_descriptions",
    "validate_description_stage_inputs",
    "validate_description_settings",
    "validate_mask_stage_inputs",
    "validate_published_object_artifact",
    "verify_installed_dam_revision",
]
