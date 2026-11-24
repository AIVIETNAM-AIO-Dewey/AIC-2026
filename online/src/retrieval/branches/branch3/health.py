"""Branch-3 ASR readiness adapter."""

from __future__ import annotations

from typing import Any

from ...modalities.asr import AsrFtsIndex


def branch3_asr_health(index: AsrFtsIndex) -> dict[str, Any]:
    return index.health()


__all__ = ["branch3_asr_health"]
