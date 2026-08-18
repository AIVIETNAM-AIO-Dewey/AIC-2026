"""Bounded, language-agnostic edit-distance reranking for lexical candidates."""

from __future__ import annotations

import re

from ..ingest.sparse import fold_vietnamese
from .models import FrameCandidate

NON_ALNUM = re.compile(r"[^a-z0-9]+")


def compact_fold(text: str) -> str:
    return NON_ALNUM.sub("", fold_vietnamese(text))


def substring_edit_similarity(pattern: str, text: str) -> float:
    """Return 1-normalized Levenshtein distance to the best document substring."""

    pattern = compact_fold(pattern)
    text = compact_fold(text)
    if not pattern or not text:
        return 0.0
    previous = [0] * (len(text) + 1)
    for pattern_index, pattern_character in enumerate(pattern, start=1):
        current = [pattern_index]
        for text_index, text_character in enumerate(text, start=1):
            current.append(
                min(
                    previous[text_index] + 1,
                    current[text_index - 1] + 1,
                    previous[text_index - 1] + (pattern_character != text_character),
                )
            )
        previous = current
    return max(0.0, 1.0 - min(previous) / len(pattern))


def rerank_fuzzy_candidates(
    query: str,
    candidates: list[FrameCandidate],
    *,
    limit: int,
    threshold: float = 0.68,
) -> list[FrameCandidate]:
    """Rerank a bounded Qdrant pool, then deduplicate candidates by frame."""

    if limit < 1:
        return []
    scored: list[tuple[int, float, int, FrameCandidate]] = []
    size = max(1, len(candidates))
    for rank, candidate in enumerate(candidates):
        text = candidate.evidence.text if candidate.evidence else None
        similarity = substring_edit_similarity(query, text or "")
        matched = int(similarity >= threshold)
        rank_score = 1.0 - rank / size
        blended = 0.75 * similarity + 0.25 * rank_score if matched else candidate.score
        scored.append((matched, similarity, rank, _with_score(candidate, blended)))
    scored.sort(key=lambda row: (-row[0], -row[1] if row[0] else row[2], row[2]))

    unique: list[FrameCandidate] = []
    seen: set[str] = set()
    for _matched, _similarity, _rank, candidate in scored:
        if candidate.frame_uid in seen:
            continue
        seen.add(candidate.frame_uid)
        unique.append(candidate)
        if len(unique) == limit:
            break
    return unique


def _with_score(candidate: FrameCandidate, score: float) -> FrameCandidate:
    return FrameCandidate(**{**candidate.__dict__, "score": score})
