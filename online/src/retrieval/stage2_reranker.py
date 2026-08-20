"""Stage 2: Precision Cross-Attention Re-Ranking & TRAKE Temporal Verification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import numpy as np

from online.src.contracts.query import (
    ParsedQuery,
    SearchResult,
    SpeechEvidence,
    MatchedObject,
)
from online.src.retrieval.embeddings import ModelRegistry

logger = logging.getLogger(__name__)


class Stage2Reranker:
    """Stage 2 High-Precision Cross-Attention Re-ranking and TRAKE Sequence Verifier."""

    def __init__(
        self,
        models: ModelRegistry,
        keyframes_root: Path = Path("/Users/khoale/Downloads/AIC_Challenger/data/keyframes"),
    ) -> None:
        self.models = models
        self.keyframes_root = keyframes_root

    def rerank_kis(
        self,
        parsed_query: ParsedQuery,
        candidates: list[dict],
        final_top_k: int = 50,
    ) -> list[SearchResult]:
        """Execute BGE-Reranker-v2-m3 Cross-Encoder on Top-50 candidate profiles."""
        if not candidates:
            return []

        user_query = parsed_query.original_query
        query_doc_pairs = []

        for cand in candidates:
            doc_parts = []
            if cand.get("dam_summary_en"):
                doc_parts.append(f"[DAM Objects] {cand['dam_summary_en']}")
            if cand.get("asr_transcript_vi"):
                doc_parts.append(f"[Audio Speech] {cand['asr_transcript_vi']}")
            if cand.get("ocr_text"):
                doc_parts.append(f"[Screen Text] {cand['ocr_text']}")

            doc_text = " ".join(doc_parts) if doc_parts else cand.get("image_relpath", "")
            query_doc_pairs.append((user_query, doc_text))

        # Cross-encoder inference on MPS/GPU
        rerank_scores = self.models.rerank_pairs(query_doc_pairs)

        # Normalize stage1 scores to [0, 1] for blending
        s1_scores = np.array([c["stage1_score"] for c in candidates], dtype=np.float32)
        if s1_scores.max() > s1_scores.min():
            s1_norm = (s1_scores - s1_scores.min()) / (s1_scores.max() - s1_scores.min())
        else:
            s1_norm = np.ones_like(s1_scores)

        # Final Blend: 40% Stage 1 Funnel + 60% Stage 2 Deep Cross-Attention
        final_scores = 0.40 * s1_norm + 0.60 * rerank_scores

        # Rank candidates by final blended score
        sorted_indices = np.argsort(final_scores)[::-1]
        results = []

        for rank_idx, idx in enumerate(sorted_indices[:final_top_k], 1):
            cand = candidates[idx]
            v_id = cand["video_id"]
            k_n = cand["keyframe_n"]
            f_idx = cand["frame_idx"]
            pts = cand["pts_time_s"]

            # Safe check image file existence
            img_path = self.keyframes_root / v_id / f"{k_n:03d}.jpg"
            img_path_alt = self.keyframes_root / v_id / f"{k_n}.jpg"
            image_available = img_path.exists() or img_path_alt.exists()

            # Speech evidence
            speech_ev = None
            if cand.get("has_speech") and cand.get("asr_transcript_vi"):
                speech_ev = SpeechEvidence(
                    start_s=max(0.0, pts - 2.5),
                    end_s=pts + 2.5,
                    transcript_raw=cand["asr_transcript_vi"],
                    score=cand.get("asr_score", 0.0),
                )

            # Adjacent keyframes for shot context
            adj_kfs = [max(1, k_n - 2), max(1, k_n - 1), k_n, k_n + 1, k_n + 2]

            res = SearchResult(
                rank=rank_idx,
                video_id=v_id,
                keyframe_n=k_n,
                frame_idx=f_idx,
                pts_time_s=pts,
                submission_string=f"{v_id}, {f_idx}",
                final_score=float(final_scores[idx]),
                stage1_score=float(cand["stage1_score"]),
                stage2_rerank_score=float(rerank_scores[idx]),
                visual_similarity=float(cand.get("visual_score", 0.0)),
                image_relpath=cand.get("image_relpath", f"keyframes/{v_id}/{k_n:03d}.jpg"),
                image_available=image_available,
                best_matching_objects=cand.get("matched_objects", []),
                dam_full_captions=[cand.get("dam_summary_en", "")] if cand.get("dam_summary_en") else [],
                has_speech=cand.get("has_speech", False),
                speech_evidence=speech_ev,
                ocr_text=cand.get("ocr_text", ""),
                adjacent_keyframes=adj_kfs,
            )
            results.append(res)

        return results

    def verify_trake_sequence(
        self,
        event_candidates_list: list[list[dict]],
        max_time_span_s: float = 90.0,
        final_top_k: int = 50,
    ) -> list[SearchResult]:
        """Verify strict chronological sequence ordering (t1 < t2 < t3) within same video."""
        if not event_candidates_list or not event_candidates_list[0]:
            return []

        if len(event_candidates_list) == 1:
            # Single event fallback
            return self.rerank_kis(
                ParsedQuery(task_type="TRAKE", original_query="TRAKE"),
                event_candidates_list[0],
                final_top_k=final_top_k,
            )

        # Multi-event sequence matching
        # Group candidates by video_id
        e1_by_video = {}
        for c in event_candidates_list[0]:
            e1_by_video.setdefault(c["video_id"], []).append(c)

        e2_by_video = {}
        for c in event_candidates_list[1]:
            e2_by_video.setdefault(c["video_id"], []).append(c)

        valid_sequences = []

        # Find videos appearing in both events
        common_videos = set(e1_by_video.keys()).intersection(e2_by_video.keys())

        for v_id in common_videos:
            for c1 in e1_by_video[v_id]:
                t1 = c1["pts_time_s"]
                for c2 in e2_by_video[v_id]:
                    t2 = c2["pts_time_s"]
                    # Constraint: t1 < t2 and time gap <= max_time_span
                    time_gap = t2 - t1
                    if 0.5 < time_gap <= max_time_span_s:
                        seq_score = (c1["stage1_score"] + c2["stage1_score"]) - 0.001 * time_gap
                        valid_sequences.append({
                            "video_id": v_id,
                            "start_keyframe_n": c1["keyframe_n"],
                            "end_keyframe_n": c2["keyframe_n"],
                            "start_frame_idx": c1["frame_idx"],
                            "end_frame_idx": c2["frame_idx"],
                            "start_pts": t1,
                            "end_pts": t2,
                            "seq_score": seq_score,
                            "c1": c1,
                            "c2": c2,
                        })

        valid_sequences.sort(key=lambda x: x["seq_score"], reverse=True)

        results = []
        for rank_idx, seq in enumerate(valid_sequences[:final_top_k], 1):
            v_id = seq["video_id"]
            k_n = seq["start_keyframe_n"]
            f_idx = seq["start_frame_idx"]
            f_end_idx = seq["end_frame_idx"]

            img_path = self.keyframes_root / v_id / f"{k_n:03d}.jpg"
            image_available = img_path.exists()

            res = SearchResult(
                rank=rank_idx,
                video_id=v_id,
                keyframe_n=k_n,
                frame_idx=f_idx,
                pts_time_s=seq["start_pts"],
                submission_string=f"{v_id}, {f_idx}, {f_end_idx}",
                final_score=float(seq["seq_score"]),
                stage1_score=float(seq["c1"]["stage1_score"]),
                stage2_rerank_score=float(seq["seq_score"]),
                visual_similarity=float(seq["c1"].get("visual_score", 0.0)),
                image_relpath=seq["c1"].get("image_relpath", f"keyframes/{v_id}/{k_n:03d}.jpg"),
                image_available=image_available,
                best_matching_objects=seq["c1"].get("matched_objects", []),
                dam_full_captions=[
                    f"Event 1 (t={seq['start_pts']:.1f}s): {seq['c1'].get('dam_summary_en', '')}",
                    f"Event 2 (t={seq['end_pts']:.1f}s): {seq['c2'].get('dam_summary_en', '')}",
                ],
                has_speech=seq["c1"].get("has_speech", False),
                adjacent_keyframes=[k_n, seq["end_keyframe_n"]],
            )
            results.append(res)

        return results
