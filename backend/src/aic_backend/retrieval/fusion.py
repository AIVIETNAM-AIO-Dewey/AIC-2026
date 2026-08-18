"""Deterministic ranking, reciprocal-rank fusion, and temporal diversification."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from .models import Evidence, FrameCandidate, SearchHit

DEFAULT_WEIGHTS = {"scene": 0.45, "object": 0.25, "ocr": 0.15, "asr": 0.15}


def normalized_weights(
    active: Iterable[str], weights: dict[str, float] | None = None
) -> dict[str, float]:
    base = weights or DEFAULT_WEIGHTS
    selected = {name: max(0.0, base.get(name, 0.0)) for name in active if base.get(name, 0.0) > 0}
    total = sum(selected.values())
    return {name: value / total for name, value in selected.items()} if total else {}


def weighted_rrf(
    ranked: dict[str, Sequence[FrameCandidate]], *, weights: dict[str, float], k: int = 60
) -> list[SearchHit]:
    """Fuse rank lists by canonical frame UID; no filename/keyframe fallback is possible."""
    score_by_uid: dict[str, float] = defaultdict(float)
    modality_by_uid: dict[str, dict[str, float]] = defaultdict(dict)
    sample_by_uid: dict[str, FrameCandidate] = {}
    evidence_by_uid: dict[str, list[Evidence]] = defaultdict(list)
    for modality, candidates in ranked.items():
        weight = weights.get(modality, 0.0)
        if not weight:
            continue
        seen_in_modality: set[str] = set()
        for rank, candidate in enumerate(candidates, start=1):
            value = weight / (k + rank)
            uid = candidate.frame_uid
            if uid in seen_in_modality:
                if candidate.evidence:
                    evidence_by_uid[uid].append(candidate.evidence)
                continue
            seen_in_modality.add(uid)
            score_by_uid[uid] += value
            modality_by_uid[uid][modality] = max(
                modality_by_uid[uid].get(modality, 0.0), candidate.score
            )
            sample_by_uid.setdefault(uid, candidate)
            if candidate.evidence:
                evidence_by_uid[uid].append(candidate.evidence)
    return sorted(
        (
            SearchHit(
                video_id=sample.video_id,
                frame_idx=sample.frame_idx,
                pts_time_s=sample.pts_time_s,
                keyframe_n=sample.keyframe_n,
                score=score_by_uid[uid],
                modality_scores=modality_by_uid[uid],
                evidence=tuple(evidence_by_uid[uid]),
                ocr=next(
                    (
                        candidate.ocr
                        for candidate in ranked.get("ocr", ())
                        if candidate.frame_uid == uid and candidate.ocr
                    ),
                    None,
                ),
                ocr_match=next(
                    (
                        candidate.ocr_match
                        for candidate in ranked.get("ocr", ())
                        if candidate.frame_uid == uid and candidate.ocr_match
                    ),
                    None,
                ),
            )
            for uid, sample in sample_by_uid.items()
        ),
        key=lambda hit: (-hit.score, hit.video_id, hit.frame_idx),
    )


def temporal_nms(
    hits: Sequence[SearchHit], *, seconds: float = 1.0, per_video: int = 20
) -> list[SearchHit]:
    kept: list[SearchHit] = []
    by_video: dict[str, list[SearchHit]] = defaultdict(list)
    for hit in hits:
        prior = by_video[hit.video_id]
        if len(prior) >= per_video or any(
            abs(other.pts_time_s - hit.pts_time_s) < seconds for other in prior
        ):
            continue
        prior.append(hit)
        kept.append(hit)
    return kept
