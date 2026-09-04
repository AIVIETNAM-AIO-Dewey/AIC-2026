"""Small score/weight primitives shared by independent retrieval branches.

Keeping these helpers below ``infrastructure`` prevents reusable components
such as the BEiT-3 cosine reranker from depending on a particular branch's
package.  Branch-specific modules continue to re-export their historical
names for compatibility.
"""

from __future__ import annotations

import math
from typing import Any


def normalize_weights(values: dict[str, float], names: tuple[str, ...]) -> dict[str, float]:
    """Validate finite non-negative weights and normalize them to one."""

    if not isinstance(values, dict):
        raise ValueError("weights must be an object")
    clean: dict[str, float] = {}
    for name in names:
        if isinstance(values.get(name), bool):
            raise ValueError("weights must be numeric, not boolean")
        try:
            value = float(values.get(name, 0.0))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("weights must be numeric") from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("weights must be finite and non-negative")
        clean[name] = value
    total = sum(clean.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("weight sum must be greater than zero")
    return {name: clean[name] / total for name in names}


def normalize_scores(items: dict[str, dict[str, Any]], raw_field: str) -> None:
    """Attach clipped z-score sigmoid values, preserving missing as zero."""

    observed = [float(item[raw_field]) for item in items.values() if item.get("observed", True)]
    if any(not math.isfinite(value) for value in observed):
        raise ValueError(f"{raw_field} scores must be finite")
    if not observed:
        return
    mean = sum(observed) / len(observed)
    variance = sum((value - mean) ** 2 for value in observed) / len(observed)
    std = math.sqrt(variance)
    for item in items.values():
        if not item.get("observed", True):
            item["normalized_score"] = 0.0
            continue
        item["normalization_mean"] = mean
        item["normalization_std"] = std
        if std < 1e-6:
            item["normalized_score"] = 0.5
        else:
            z = max(-4.0, min(4.0, (float(item[raw_field]) - mean) / std))
            item["normalized_score"] = 1.0 / (1.0 + math.exp(-z))


__all__ = ["normalize_scores", "normalize_weights"]
