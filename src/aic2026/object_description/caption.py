"""DAM prompt and deterministic caption post-processing."""

from __future__ import annotations

import re

from aic2026.contracts import CaptionResult

DAM_PROMPT = """<image>\n<image>
Describe only the masked region in detail in English (at most 50 words)."""


def normalize_caption(text: str, maximum_words: int = 50) -> CaptionResult:
    cleaned = text.replace("▁", " ").replace(" ", " ")
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    normalized = re.sub(r"\s+", " ", cleaned).strip()
    if not normalized:
        raise ValueError("DAM returned an empty caption")
    words = normalized.split(" ")
    truncated = len(words) > maximum_words
    if truncated:
        normalized = " ".join(words[:maximum_words]).rstrip(" ,;:")
        words = normalized.split(" ")
    return CaptionResult(
        status="ok",
        description_en=normalized,
        word_count=len(words),
        truncated=truncated,
    )
