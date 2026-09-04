"""Final KIS cross-branch fusion and bounded BEiT-3 validation."""

from .contracts import (
    DEFAULT_BRANCH_WEIGHTS,
    FINAL_FUSION_RESULT_SCHEMA_VERSION,
    FINAL_RERANK_TOP_K,
    FINAL_TOP_K,
    RRF_K,
)
from .health import fusion_health
from .provenance import compact_branch_evidence, materialize_fusion_candidate
from .service import KisFusionSearch

__all__ = [
    "DEFAULT_BRANCH_WEIGHTS",
    "FINAL_FUSION_RESULT_SCHEMA_VERSION",
    "FINAL_RERANK_TOP_K",
    "FINAL_TOP_K",
    "KisFusionSearch",
    "RRF_K",
    "fusion_health",
    "compact_branch_evidence",
    "materialize_fusion_candidate",
]
