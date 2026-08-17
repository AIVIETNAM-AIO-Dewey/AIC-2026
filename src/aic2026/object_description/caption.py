"""DAM prompt and deterministic caption post-processing."""

from __future__ import annotations

import re

from aic2026.contracts import CaptionResult

DAM_PROMPT = """<image>
Describe only the masked region in detail in English (at most 50 words).
State the object or person, exact colors, visual attributes, appearance, surroundings, and visible action. Do not infer hidden facts."""


def normalize_caption(text: str, maximum_words: int = 50) -> CaptionResult:
    normalized = re.sub(r"\s+", " ", text).strip()
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
