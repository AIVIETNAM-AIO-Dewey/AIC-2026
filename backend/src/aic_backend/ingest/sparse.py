"""Stable hashed lexical vectors for Qdrant sparse hybrid retrieval."""

from __future__ import annotations

import hashlib
import re
from collections import Counter

TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
SPARSE_DIMENSION = 2**31 - 1


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def sparse_vector(text: str) -> tuple[list[int], list[float]]:
    counts = Counter(tokenize(text))
    indexes = [
        int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16) % SPARSE_DIMENSION
        for token in counts
    ]
    paired = sorted(zip(indexes, (float(value) for value in counts.values()), strict=True))
    return [item[0] for item in paired], [item[1] for item in paired]
