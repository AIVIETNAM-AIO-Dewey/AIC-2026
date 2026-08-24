"""Vietnamese & English OCR text normalization for BM25 and lexical search."""

from __future__ import annotations

import re
import unicodedata


def normalize_vietnamese_text(raw_text: str) -> str:
    """Normalize OCR text into canonical NFC form with cleaned punctuation and whitespace."""
    if not raw_text or not raw_text.strip():
        return ""

    # 1. Unicode NFC normalization
    text = unicodedata.normalize("NFC", raw_text)

    # 2. Lowercase
    text = text.lower()

    # 3. Strip noise punctuation while preserving Vietnamese characters, digits, and spaces
    text = re.sub(r"[^\w\s\-_/.:]", " ", text, flags=re.UNICODE)

    # 4. Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
