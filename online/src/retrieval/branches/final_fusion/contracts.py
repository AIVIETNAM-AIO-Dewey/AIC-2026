"""Public and internal contracts for KIS cross-branch fusion."""

from __future__ import annotations

import math
from typing import Final

FINAL_FUSION_RESULT_SCHEMA_VERSION: Final[str] = "kis.fusion.result.v1"
RRF_K: Final[int] = 60
FINAL_TOP_K: Final[int] = 150
FINAL_RERANK_TOP_K: Final[int] = 100
DEFAULT_BRANCH_WEIGHTS: Final[dict[str, float]] = {
    "branch1": 0.40,
    "branch2": 0.30,
    "ocr": 0.15,
    "asr": 0.15,
}
BRANCH_POOL_LIMITS: Final[dict[str, int]] = {
    "branch1": 1_500,
    "branch2": 500,
    "ocr": 500,
    "asr": 500,
}
# Standalone Branch-2 gates are repeated here as immutable fusion policy so
# the final-fusion package does not depend on Branch-2 implementation modules.
BRANCH2_PER_STREAM_TOP_K: Final[int] = 2_000
BRANCH2_PRE_RERANK_TOP_K: Final[int] = 500
BRANCH2_RERANK_TOP_K: Final[int] = 100
BRANCH_SCHEMA_VERSIONS: Final[dict[str, str]] = {
    "branch1": "branch1.result.v1",
    "branch2": "branch2.result.v1",
    "ocr": "branch3.ocr.result.v1",
    "asr": "branch3.asr.result.v1",
}


def normalize_branch_weights(values: dict[str, float] | None) -> dict[str, float]:
    """Validate four positive branch weights and normalize to one."""

    supplied = DEFAULT_BRANCH_WEIGHTS if values is None else values
    if not isinstance(supplied, dict):
        raise ValueError("branch_weights must be an object")
    try:
        clean = {name: float(supplied.get(name, 0.0)) for name in DEFAULT_BRANCH_WEIGHTS}
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("branch weights must be numeric") from error
    if set(supplied) - set(DEFAULT_BRANCH_WEIGHTS):
        raise ValueError("branch_weights may only contain branch1, branch2, ocr and asr")
    if any(isinstance(supplied.get(name), bool) for name in DEFAULT_BRANCH_WEIGHTS):
        raise ValueError("branch weights must be numeric, not boolean")
    if any(not math.isfinite(value) or value <= 0.0 for value in clean.values()):
        raise ValueError("all branch weights must be finite and greater than zero")
    total = sum(clean.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("branch weight sum must be greater than zero")
    return {name: clean[name] / total for name in DEFAULT_BRANCH_WEIGHTS}


__all__ = [
    "BRANCH_POOL_LIMITS",
    "BRANCH2_PER_STREAM_TOP_K",
    "BRANCH2_PRE_RERANK_TOP_K",
    "BRANCH2_RERANK_TOP_K",
    "BRANCH_SCHEMA_VERSIONS",
    "DEFAULT_BRANCH_WEIGHTS",
    "FINAL_FUSION_RESULT_SCHEMA_VERSION",
    "FINAL_RERANK_TOP_K",
    "FINAL_TOP_K",
    "RRF_K",
    "normalize_branch_weights",
]
