"""Branch-1 public model and query contracts."""

from __future__ import annotations

import math

MODEL_SPECS = {
    "siglip2": {
        "collection": "aic_frames",
        "vector": "siglip2",
        "languages": ("vi", "en"),
        "dimension": 768,
    },
    "metaclip2": {
        "collection": "aic_frames",
        "vector": "metaclip2",
        "languages": ("vi", "en"),
        "dimension": 1024,
    },
    "beit3": {
        "collection": "aic_beit3_frames",
        "vector": "beit3",
        "languages": ("en",),
        "dimension": 768,
    },
}
EXPECTED_FRAMES = 247_956
QUERY_ROLES = ("original", "entity", "action", "context", "synonym", "keyword")
BRANCH1_FINAL_TOP_K = 1_500


def normalize_model_weights(values: dict[str, float]) -> dict[str, float]:
    """Validate and normalize model-level fusion weights.

    The HTTP schema performs the same validation, but keeping the invariant at
    the service/fusion boundary prevents direct callers from accidentally
    applying percentages (``45/30/25``) as raw multipliers.
    """

    clean = {name: float(values.get(name, 0.0)) for name in MODEL_SPECS}
    if any(not math.isfinite(value) or value < 0 for value in clean.values()):
        raise ValueError("model weights must be finite and non-negative")
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("model weight sum must be greater than zero")
    return {name: value / total for name, value in clean.items()}


__all__ = [
    "BRANCH1_FINAL_TOP_K",
    "EXPECTED_FRAMES",
    "MODEL_SPECS",
    "normalize_model_weights",
    "QUERY_ROLES",
]
