from __future__ import annotations

import pytest

from aic2026.object_description import DAM_PROMPT, normalize_caption


def test_prompt_contains_required_image_token_and_word_constraint() -> None:
    assert DAM_PROMPT.startswith("<image>")
    assert "at most 20 words" in DAM_PROMPT


def test_caption_is_normalized_and_hard_capped() -> None:
    caption = normalize_caption("  " + " ".join(f"word{i}" for i in range(25)) + "  ")

    assert caption.status == "ok"
    assert caption.word_count == 20
    assert caption.truncated is True
    assert caption.description_en.endswith("word19")


def test_empty_caption_fails() -> None:
    with pytest.raises(ValueError, match="empty"):
        normalize_caption(" \n ")
