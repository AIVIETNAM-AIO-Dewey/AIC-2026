"""Bounded, language-agnostic edit-distance reranking for lexical candidates."""

from __future__ import annotations

import re
from typing import Literal

from ..ingest.sparse import fold_vietnamese
from .models import FrameCandidate, OcrMatch

NON_ALNUM = re.compile(r"[^a-z0-9]+")
MatchType = Literal["exact", "accent_folded", "fuzzy", "trigram_candidate"]


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
    for rank, candidate in enumerate(candidates):
        text = candidate.evidence.text if candidate.evidence else None
        similarity = substring_edit_similarity(query, text or "")
        matched = int(similarity >= threshold)
        # Reciprocal lexical rank is independent from the requested result limit.
        rank_score = 1.0 / (rank + 1)
        blended = 0.75 * similarity + 0.25 * rank_score if matched else candidate.score
        scored.append(
            (
                matched,
                similarity,
                rank,
                _with_match(
                    candidate,
                    query=query,
                    final_score=blended,
                    similarity=similarity,
                    fuzzy_enabled=True,
                ),
            )
        )
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


def explain_ocr_candidates(
    query: str, candidates: list[FrameCandidate], *, limit: int
) -> list[FrameCandidate]:
    """Preserve lexical order while attaching a machine-readable match explanation."""

    explained = [
        _with_match(
            candidate,
            query=query,
            final_score=candidate.score,
            similarity=None,
            fuzzy_enabled=False,
        )
        for candidate in candidates
    ]
    return _deduplicate(explained, limit=limit)


def _match_type(query: str, text: str, similarity: float | None) -> MatchType:
    normalized_query = " ".join(query.lower().split())
    normalized_text = " ".join(text.lower().split())
    if normalized_query and normalized_query in normalized_text:
        return "exact"
    folded_query = " ".join(fold_vietnamese(query).split())
    folded_text = " ".join(fold_vietnamese(text).split())
    if folded_query and folded_query in folded_text:
        return "accent_folded"
    if similarity is not None and similarity >= 0.68:
        return "fuzzy"
    return "trigram_candidate"


def _with_match(
    candidate: FrameCandidate,
    *,
    query: str,
    final_score: float,
    similarity: float | None,
    fuzzy_enabled: bool,
) -> FrameCandidate:
    text = candidate.evidence.text if candidate.evidence and candidate.evidence.text else ""
    match = OcrMatch(
        query=query,
        normalized_query=" ".join(fold_vietnamese(query).split()),
        matched_text=text,
        lexical_score=candidate.score,
        fuzzy_similarity=similarity,
        final_score=final_score,
        match_type=_match_type(query, text, similarity),
        fuzzy_enabled=fuzzy_enabled,
    )
    return FrameCandidate(**{**candidate.__dict__, "score": final_score, "ocr_match": match})


def _deduplicate(candidates: list[FrameCandidate], *, limit: int) -> list[FrameCandidate]:
    unique: list[FrameCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.frame_uid in seen:
            continue
        seen.add(candidate.frame_uid)
        unique.append(candidate)
        if len(unique) == limit:
            break
    return unique
