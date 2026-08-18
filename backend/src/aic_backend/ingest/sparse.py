"""Stable accent-tolerant and fuzzy lexical vectors for Qdrant retrieval."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
SPARSE_DIMENSION = 2**31 - 1
FOLDED_WORD_WEIGHT = 1.0
TRIGRAM_WEIGHT = 0.25


def fold_vietnamese(text: str) -> str:
    """Lowercase and remove combining marks, including Vietnamese đ/Đ."""

    normalized = unicodedata.normalize("NFKD", text).replace("đ", "d").replace("Đ", "D")
    return "".join(
        character for character in normalized if not unicodedata.combining(character)
    ).lower()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(unicodedata.normalize("NFC", text).lower())


def _trigrams(token: str) -> list[str]:
    if len(token) < 3:
        return []
    padded = f"^{token}$"
    return [padded[index : index + 3] for index in range(len(padded) - 2)]


def lexical_features(text: str) -> dict[str, float]:
    """Create generic exact, accent-folded, and character-trigram features."""

    exact_counts = Counter(tokenize(text))
    folded_counts = Counter(tokenize(fold_vietnamese(text)))
    features: defaultdict[str, float] = defaultdict(float)
    # Keep original token IDs/weights for compatibility with older collections.
    for token, count in exact_counts.items():
        features[token] += float(count)
    for token, count in folded_counts.items():
        features[f"fold:{token}"] += float(count) * FOLDED_WORD_WEIGHT
        for trigram in _trigrams(token):
            features[f"tri:{trigram}"] += float(count) * TRIGRAM_WEIGHT
    return dict(features)


def _feature_index(feature: str) -> int:
    return int(hashlib.sha1(feature.encode("utf-8")).hexdigest()[:8], 16) % SPARSE_DIMENSION


def sparse_vector(text: str) -> tuple[list[int], list[float]]:
    # Hash collisions are rare but must still produce one valid Qdrant index.
    hashed: defaultdict[int, float] = defaultdict(float)
    for feature, weight in lexical_features(text).items():
        hashed[_feature_index(feature)] += weight
    paired = sorted(hashed.items())
    return [index for index, _value in paired], [value for _index, value in paired]
