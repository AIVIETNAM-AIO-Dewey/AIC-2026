"""DAM prompt and deterministic caption post-processing."""

from __future__ import annotations

import re

from aic2026.contracts import CaptionResult

DAM_PROMPT = (
    "<image>\n"
    "Describe only the masked region in one concise English sentence of at most 20 words.\n"
    "State the object or person, exact colors, visual attributes, and visible action. "
    "Do not infer hidden facts."
)


def normalize_caption(text: str, maximum_words: int = 20) -> CaptionResult:
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
