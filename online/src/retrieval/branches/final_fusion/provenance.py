"""Compact branch evidence retained in the final KIS response.

The standalone branch responses intentionally carry fairly large audit
objects (all query streams, normalization diagnostics and, for DAM, region
payloads).  Those objects are useful at the branch boundary but are not a
safe shape for the final cross-branch response.  This module is the single
allow-list boundary for KIS candidates.
"""

from __future__ import annotations

from typing import Any


def _pick(item: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: item[field] for field in fields if field in item}


def _compact_model_provenance(value: Any) -> dict[str, dict[str, Any]]:
    """Keep one small summary per observed Branch-1 model.

    Branch 1's raw ``query_scores`` contains six stream records per model and
    can be repeated for every final frame.  The final KIS audit only needs the
    winning score/stream and whether a model was observed.
    """

    if not isinstance(value, dict):
        return {}
    fields = (
        "observed",
        "raw_cosine",
        "normalized_score",
        "best_query_role",
        "best_query_language",
        "best_query_rank",
    )
    result: dict[str, dict[str, Any]] = {}
    for model, details in value.items():
        if not isinstance(details, dict):
            continue
        summary = _pick(details, fields)
        if details.get("observed") is not True:
            # An unobserved model contributes zero; do not expose a stale
            # winning role/language copied from a previous candidate.
            for field in ("best_query_role", "best_query_language", "best_query_rank"):
                summary.pop(field, None)
        result[str(model)] = summary
    return result


def _set_observed_language(
    result: dict[str, Any],
    item: dict[str, Any],
    *,
    observed_field: str,
    language_field: str,
    score_field: str,
) -> None:
    """Copy a language only for observed evidence.

    Branch 2 is English-only.  Older branch responses omitted the explicit
    language fields, so a present/observed evidence record gets the safe
    contract default ``en``.  Missing evidence must not acquire a fabricated
    language value.
    """

    observed = item.get(observed_field)
    # A few pre-language-field Branch-2 adapters did not emit the boolean
    # marker, but a non-null raw score is still unambiguous evidence.  Never
    # infer a language when the adapter explicitly marked the source absent.
    if observed is None:
        observed = item.get(score_field) is not None
    if observed is True:
        result[language_field] = str(item.get(language_field) or "en")


def _compact_region_evidence(value: Any) -> dict[str, Any] | None:
    """Reduce one DAM/sparse winning region to display/audit fields."""

    if not isinstance(value, dict):
        return None
    return _pick(
        value,
        (
            "point_id",
            "region_id",
            "class_entity",
            "description_en",
            "bbox",
            "query_role",
            "best_query_role",
            "query_language",
            "best_query_language",
            "rank",
            "bm25_raw",
            "lse_score",
        ),
    )


def compact_branch_evidence(branch: str, item: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    if branch == "branch1":
        result = _pick(
            item,
            (
                "final_score",
                "score",
                "score_type",
                "best_model",
                "best_stream_rank",
                "best_query_role",
                "best_query_language",
            ),
        )
        result["model_provenance"] = _compact_model_provenance(item.get("model_provenance"))
        return result
    if branch == "branch2":
        result = _pick(
            item,
            (
                "dense_raw",
                "dense_normalized",
                "dense_observed",
                "sparse_raw",
                "sparse_normalized",
                "sparse_observed",
                "sparse_bm25_raw",
                "hybrid_score",
                "reranked_score",
                "score",
                "score_type",
                "hybrid_rank",
                "pre_rerank_rank",
                "dam_winner",
                "dense_rank",
                "sparse_rank",
                "dense_best_query_role",
                "sparse_best_query_role",
                "beit3_raw_cosine",
                "beit3_normalized",
                "beit3_best_query_role",
                "sparse_winner",
            ),
        )
        _set_observed_language(
            result,
            item,
            observed_field="dense_observed",
            language_field="dense_best_query_language",
            score_field="dense_raw",
        )
        _set_observed_language(
            result,
            item,
            observed_field="sparse_observed",
            language_field="sparse_best_query_language",
            score_field="sparse_raw",
        )
        # The BEiT field is not explicitly marked by Branch 2, but its raw
        # cosine evidence is only present for the internal top-100.
        beit_observed = bool(
            item.get("beit3_raw_cosine") is not None
            or item.get("beit3_query_scores")
            or item.get("beit3_best_query_role")
        )
        if beit_observed:
            result["beit3_best_query_language"] = str(item.get("beit3_best_query_language") or "en")
        else:
            for field in (
                "beit3_raw_cosine",
                "beit3_normalized",
                "beit3_best_query_role",
                "beit3_query_scores",
                "beit3_best_query_language",
            ):
                result.pop(field, None)
        for field in ("dam_winner", "sparse_winner"):
            if field in result:
                compacted = _compact_region_evidence(result[field])
                if compacted is None:
                    result.pop(field, None)
                else:
                    result[field] = compacted
        return result
    if branch == "ocr":
        return _pick(
            item,
            (
                "ocr_text",
                "full_text",
                "ocr_raw_score",
                "ocr_combined_score",
                "ocr_normalized_score",
                "ocr_best_query_role",
                "ocr_best_query_language",
                "ocr_best_rank",
                "bm25_raw",
                "bm25_relevance",
                "token_coverage",
                "ngram_coverage",
                "adjacent_bigram_coverage",
                "matched_terms",
            ),
        )
    if branch == "asr":
        return _pick(
            item,
            (
                "asr_text",
                "asr_transcript",
                "transcript",
                "asr_raw_score",
                "asr_combined_score",
                "asr_normalized_score",
                "asr_best_query_role",
                "asr_best_query_language",
                "asr_best_rank",
                "bm25_raw",
                "bm25_relevance",
                "token_coverage",
                "ngram_coverage",
                "adjacent_bigram_coverage",
                "matched_terms",
                # Segment identity/time are part of the winning ASR evidence.
                "asr_segment_id",
                "segment_id",
                "asr_start_s",
                "asr_end_s",
                "segment_start_s",
                "segment_end_s",
                "start_s",
                "end_s",
            ),
        )
    raise ValueError(f"Unsupported fusion provenance branch: {branch}")


_CANONICAL_FIELDS = (
    "frame_uid",
    "point_id",
    "global_idx",
    "video_id",
    "frame_idx",
    "keyframe_n",
    "pts_time_s",
    "fps",
    "image_relpath",
    "submission_string",
)
_RRF_FIELDS = (
    "rank",
    "pre_rerank_rank",
    "rrf_score",
    "branch_agreement_count",
    "observed_branches",
    "branch_ranks",
    "rrf_contributions",
    "branch_normalized_scores",
    "weighted_normalized_score",
    "best_branch_rank",
)
_RERANK_FIELDS = (
    "score",
    "score_type",
    "final_score",
    "beit3_raw_cosine",
    "beit3_normalized",
    "rrf_normalized",
    "beit3_best_query_role",
    "beit3_best_query_language",
    "beit3_query_scores",
    "rank_delta",
    "rerank_formula",
)
_PROVENANCE_BRANCHES = ("branch1", "branch2", "ocr", "asr")


def materialize_fusion_candidate(
    item: dict[str, Any],
    *,
    include_rerank_fields: bool = True,
) -> dict[str, Any]:
    """Create the bounded public candidate shape from an internal RRF row.

    ``fuse_branch_pools`` deliberately retains the first branch record while
    it validates identity and computes rank-only contributions.  That is an
    internal convenience, not a public response contract.  Materialization
    must therefore happen before BEiT reranking so even tail candidates cannot
    leak a full standalone branch record into the final payload.
    """

    if not isinstance(item, dict):
        raise ValueError("fusion candidate must be an object")
    # Before the final rerank, deliberately omit BEiT fields that may have
    # come from Branch 2's *internal* top-100 rerank.  They are branch-local
    # evidence and must not look like final-fusion evidence, especially on
    # the final tail (ranks 101--150).  A second materialization after
    # reranking includes the canonical final fields for the selected rows.
    fields = (
        _CANONICAL_FIELDS
        + _RRF_FIELDS
        + (_RERANK_FIELDS if include_rerank_fields else ("score", "score_type", "final_score"))
    )
    result = _pick(item, fields)
    point_id = result.get("point_id", result.get("global_idx"))
    if point_id is not None:
        result["point_id"] = int(point_id)
        result.setdefault("global_idx", int(point_id))
    if "submission_string" not in result:
        video_id = result.get("video_id")
        frame_idx = result.get("frame_idx")
        if video_id is not None and frame_idx is not None:
            result["submission_string"] = f"{video_id}, {int(frame_idx)}"
    raw_provenance = item.get("branch_provenance")
    compacted: dict[str, dict[str, Any]] = {}
    if isinstance(raw_provenance, dict):
        for branch in _PROVENANCE_BRANCHES:
            evidence = raw_provenance.get(branch)
            if isinstance(evidence, dict):
                compacted[branch] = compact_branch_evidence(branch, evidence)
    result["branch_provenance"] = compacted
    return result


__all__ = ["compact_branch_evidence", "materialize_fusion_candidate"]
