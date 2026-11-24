"""Shared Unicode/lexical primitives for the local text modalities.

The ASR and OCR indexes deliberately use the same token contract.  Keeping the
implementation here avoids the two modalities slowly acquiring different
definitions of token coverage or adjacency while leaving the public scoring
formula in each modality unchanged.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable

# Keep both correctly decoded Vietnamese and the spellings that occur in older
# UTF-8-as-Latin-1 exports.  ``fold_text`` canonicalizes both forms before
# membership is checked.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "for",
        "from",
        "in",
        "of",
        "on",
        "the",
        "to",
        "with",
        # ``ở``, ``trên``, ``dưới`` and ``từ`` were not stopwords in the
        # established ASR contract; keep their folded forms searchable so
        # moving the primitive here does not silently change ASR rankings.
        # ``về`` is likewise intentionally searchable (``ve``).
        "các",
        "cho",
        "có",
        "của",
        "được",
        "là",
        "một",
        "những",
        "trong",
        "và",
        "với",
        # Historical mojibake forms retained for source compatibility.
        "cÃ¡c",
        "cÃ³",
        "cá»§a",
        "Ä‘Æ°á»£c",
        "lÃ ",
        "má»™t",
        "nhá»¯ng",
        "vÃ ",
        "vá»›i",
    }
)


def repair_mojibake(text: str) -> str:
    """Repair a UTF-8/Latin-1 round-trip when it is clearly present."""

    value = str(text or "")
    # Single-byte mojibake markers (C2/C3/C4 rendered as Latin-1) are common
    # in the exported OCR/ASR text.  Detect them by code point so ordinary
    # readable text is left untouched when the UTF-8 round-trip is invalid.
    if value and any(
        ord(character) in {0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xD0, 0xD1} for character in value
    ):
        try:
            repaired = value.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            repaired = value
        if "\u00ef\u00bf\u00bd" in repaired or "\ufffd" in repaired:
            return value
        if repaired != value and "ï¿½" not in repaired:
            return repaired
    if not value or not any(marker in value for marker in ("Ã", "Â", "Ä", "â", "ð")):
        return value
    try:
        repaired = value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    # Never replace a readable source with a lossy decode.
    return repaired if "�" not in repaired else value


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", repair_mojibake(str(value or ""))).casefold().strip()


def fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", normalize_text(value))
    without_marks = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return (
        without_marks.replace("\u0111", "d")
        .replace("\u0110", "D")
        # Vietnamese "đ" is a letter with a stroke, not a combining mark;
        # strip it explicitly so diacritic-insensitive matching treats both
        # valid Unicode text and repaired mojibake consistently.
        .replace("đ", "d")
        .replace("Đ", "D")
        .replace("đ", "d")
        .replace("Đ", "D")
        .replace("Ä‘", "d")
        .replace("Ã°", "d")
    )


def _folded_tokens(value: str) -> list[str]:
    return re.findall(r"[^\W_]+", fold_text(value), flags=re.UNICODE)


def _folded_stopwords() -> set[str]:
    return {fold_text(item) for item in STOPWORDS}


def ordered_lexical_tokens(query: str, limit: int | None = None) -> list[str]:
    """Return the complete folded token stream, preserving repeats/gaps."""

    values = _folded_tokens(query)
    return values if limit is None else values[:limit]


def query_tokens(query: str, limit: int | None = None) -> list[str]:
    """Return deduplicated searchable tokens in first-seen order."""

    stopwords = _folded_stopwords()
    useful = [
        token for token in _folded_tokens(query) if len(token) >= 2 and token not in stopwords
    ]
    values = list(dict.fromkeys(useful))
    return values if limit is None else values[:limit]


def _ordered_lexical_bigrams(query: str) -> list[tuple[str, str]]:
    """Return adjacent *searchable* pairs without bridging source gaps.

    Stopwords and short tokens remain in the original folded stream as
    boundaries.  A pair is emitted only when both original neighbours are
    useful lexical tokens, so ``one in two`` and ``xe A đỏ`` cannot create a
    synthetic ``one two``/``xe do`` pair.  Repeated tokens are retained.
    """

    values = _folded_tokens(query)
    stopwords = _folded_stopwords()

    def useful(token: str) -> bool:
        return len(token) >= 2 and token not in stopwords

    return [
        (left, right)
        for left, right in zip(values, values[1:], strict=False)
        if useful(left) and useful(right)
    ]


def _token_bigrams(tokens: Iterable[str]) -> list[tuple[str, str]]:
    values = [str(token) for token in tokens if str(token)]
    return list(zip(values, values[1:], strict=False))


def _stream_identity(stream: str) -> tuple[str, str | None]:
    role, _separator, language = str(stream).partition(":")
    return role, language or None


def sigmoid_zscore(values: list[float]) -> dict[int, float]:
    if not values:
        return {}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std < 1e-6:
        return {index: 0.5 for index in range(len(values))}
    result: dict[int, float] = {}
    for index, value in enumerate(values):
        z = max(-4.0, min(4.0, (value - mean) / std))
        result[index] = 1.0 / (1.0 + math.exp(-z))
    return result


# Private alias retained for ASR/OCR callers and existing tests.
_sigmoid_zscore = sigmoid_zscore
adjacent_bigrams = _ordered_lexical_bigrams


__all__ = [
    "STOPWORDS",
    "fold_text",
    "normalize_text",
    "ordered_lexical_tokens",
    "query_tokens",
    "_folded_tokens",
    "_ordered_lexical_bigrams",
    "adjacent_bigrams",
    "_token_bigrams",
    "_stream_identity",
    "sigmoid_zscore",
    "_sigmoid_zscore",
    "repair_mojibake",
]
