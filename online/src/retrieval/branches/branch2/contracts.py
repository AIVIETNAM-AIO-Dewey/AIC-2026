"""Branch 2 constants and score utilities."""

from __future__ import annotations

from ...infrastructure.scoring import (
    normalize_scores as normalize_scores,
)
from ...infrastructure.scoring import (
    normalize_weights as normalize_weights,
)

QUERY_ROLES = ("original", "entity", "action", "context", "synonym", "keyword")
EXPECTED_DAM_REGIONS = 681_355
DEFAULT_PER_STREAM_TOP_K = 2_000
DEFAULT_PRE_RERANK_TOP_K = 500
DEFAULT_RERANK_TOP_K = 100

__all__ = [
    "DEFAULT_PER_STREAM_TOP_K",
    "DEFAULT_PRE_RERANK_TOP_K",
    "DEFAULT_RERANK_TOP_K",
    "EXPECTED_DAM_REGIONS",
    "QUERY_ROLES",
    "normalize_scores",
    "normalize_weights",
]
