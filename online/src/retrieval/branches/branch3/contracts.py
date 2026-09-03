"""Branch-3 ASR constants and response contract helpers."""

from __future__ import annotations

from typing import Final

QUERY_ROLES: Final[tuple[str, ...]] = (
    "original", "entity", "action", "context", "synonym", "keyword",
)
DEFAULT_PER_STREAM_TOP_K: Final[int] = 2_000
DEFAULT_FINAL_TOP_K: Final[int] = 500
MAX_PER_STREAM_TOP_K: Final[int] = 2_000
MAX_FINAL_TOP_K: Final[int] = 500
ASR_RESULT_SCHEMA_VERSION: Final[str] = "branch3.asr.result.v1"

__all__ = [
    "ASR_RESULT_SCHEMA_VERSION", "DEFAULT_FINAL_TOP_K", "DEFAULT_PER_STREAM_TOP_K",
    "MAX_FINAL_TOP_K", "MAX_PER_STREAM_TOP_K", "QUERY_ROLES",
]
