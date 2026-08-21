"""Stage 2: Precision Cross-Attention Re-Ranking & TRAKE Dynamic Programming Path Finder.

Implements task-specific precision layers:
1. KIS: bge-reranker-v2-m3 Cross-Encoder on candidate dossiers (Score = 0.40 * Stage1 + 0.60 * Reranker).
2. VQA: Cross-Encoder ranking + Extractive LLM Reader for Top 1 evidence answer.
3. TRAKE: Dynamic Programming Monotonic Path Matching across events (t(f1) < t(f2) < ... < t(fN)).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from online.src.contracts.query import ParsedQuery
from online.src.retrieval.embeddings import ModelRegistry
from online.src.retrieval.vqa_reasoner import VQAReasoner

logger = logging.getLogger(__name__)


class Stage2Reranker:
    """Stage 2 Precision Re-Ranker and Reasoning Layer."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        vqa_reasoner: Optional[VQAReasoner] = None,
    ):
        self.registry = registry or ModelRegistry.get_instance()
        self.vqa_reasoner = vqa_reasoner or VQAReasoner()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. KIS Precision Re-Ranking (BGE Cross-Encoder)
    # ──────────────────────────────────────────────────────────────────────────
    def rerank_kis(
        self,
        parsed_query: ParsedQuery,
        candidates: list[dict[str, Any]],
        final_top_k: int = 20,
        top_k_rerank: int = 50,
    ) -> list[dict[str, Any]]:
        """Re-rank Stage 1 candidate pool using BGE-Reranker-v2-m3 Cross-Encoder.
        
        Args:
            parsed_query: Query with text descriptors
            candidates: Candidate pool from Stage 1 (e.g. 300 items)
            final_top_k: Number of final results to return
            top_k_rerank: Maximum candidates to evaluate with heavy cross-encoder (default 50)
        """
        if not candidates:
            return []

        user_query = parsed_query.original_query
        
        # Only send top_k_rerank candidates to the heavy cross-encoder
        to_eval = candidates[:top_k_rerank]
        dossiers = []

        for c in to_eval:
            parts = []
            if c.get("dam_summary"):
                parts.append(f"[Visual Objects] {c['dam_summary']}")
            if c.get("asr_transcript"):
                parts.append(f"[Spoken Speech] {c['asr_transcript']}")
            if c.get("ocr_text"):
                parts.append(f"[Screen Text] {c['ocr_text']}")
            
            # If no text payload, fallback to scene description
            dossier_text = " ".join(parts) if parts else f"[Scene] Video {c['video_id']} frame {c['frame_idx']}"
            dossiers.append(dossier_text)

        # Compute Cross-Encoder scores for the top candidates
        ce_scores = self.registry.compute_rerank_scores(user_query, dossiers)

        # Blend Stage 1 and Stage 2 scores: 0.40 * Stage1 + 0.60 * CrossEncoder
        reranked = []
        for i, c in enumerate(to_eval):
            c_copy = dict(c)
            ce_score = float(ce_scores[i])
            norm_s1 = c.get("normalized_score", 1.0)
            final_score = round(0.40 * norm_s1 + 0.60 * ce_score, 4)

            c_copy["cross_encoder_score"] = round(ce_score, 4)
            c_copy["final_score"] = final_score
            c_copy["submission_string"] = f"{c['video_id']}, {c['frame_idx']}"
            reranked.append(c_copy)

        reranked.sort(key=lambda x: x["final_score"], reverse=True)

        for rank, item in enumerate(reranked[:final_top_k], 1):
            item["final_rank"] = rank

        return reranked[:final_top_k]

    # ──────────────────────────────────────────────────────────────────────────
    # 2. VQA Precision Re-Ranking & Answer Extraction
    # ──────────────────────────────────────────────────────────────────────────
    def rerank_vqa(
        self,
        parsed_query: ParsedQuery,
        candidates: list[dict[str, Any]],
        final_top_k: int = 20,
        top_k_rerank: int = 50,
    ) -> list[dict[str, Any]]:
        """Re-rank candidates and extract concise answer for the Top Evidence Keyframe."""
        reranked = self.rerank_kis(parsed_query, candidates, final_top_k=final_top_k, top_k_rerank=top_k_rerank)
        if not reranked:
            return []

        # Run Extractive Reasoner on Top 1 Evidence Keyframe
        top_frame = reranked[0]
        q_text = parsed_query.vqa_question or parsed_query.original_query
        vqa_ans = self.vqa_reasoner.answer_question(q_text, top_frame, raw_query=parsed_query.original_query)

        top_frame["vqa_answer"] = vqa_ans
        top_frame["submission_string"] = f'{top_frame["video_id"]}, {top_frame["frame_idx"]}, "{vqa_ans}"'

        # Also populate default submission string for others
        for item in reranked[1:]:
            item["vqa_answer"] = vqa_ans  # Human can review / override in UI
            item["submission_string"] = f'{item["video_id"]}, {item["frame_idx"]}, "{vqa_ans}"'

        return reranked

    # ──────────────────────────────────────────────────────────────────────────
    # 3. TRAKE Dynamic Programming Monotonic Sequence Solver
    # ──────────────────────────────────────────────────────────────────────────
    def solve_trake_video_guided_dp(
        self,
        event_queries: list[ParsedQuery],
        candidate_pools: list[list[dict[str, Any]]],
        searcher: Any,
        top_n_videos: int = 10,
        final_top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Video-Level Timeline Dynamic Programming.
        
        1. Identifies top candidate videos from Stage 1 candidate pools.
        2. Slices all chronological keyframes for each top video.
        3. Computes event-to-frame similarity matrix (M x N).
        4. Runs Dynamic Programming enforcing strict monotonic order (f1 < f2 < ... < fN).
        """
        num_events = len(event_queries)
        if num_events == 0:
            return []

        # 1. Aggregate video scores across all event pools
        video_scores: dict[str, float] = defaultdict(float)
        for pool in candidate_pools:
            for cand in pool:
                video_scores[cand["video_id"]] += cand.get("stage1_score", 0.0)

        effective_top_n = max(top_n_videos, final_top_k, 50)
        top_videos = sorted(video_scores.items(), key=lambda x: x[1], reverse=True)[:effective_top_n]

        # 2. Build video-to-keyframes index map
        video_kfs = defaultdict(list)
        for idx, kf in enumerate(searcher.keyframe_metadata):
            video_kfs[kf["video_id"]].append((idx, kf))

        # 3. Encode visual query vectors for each event
        ev_vis_vecs = [self.registry.embed_siglip_text(q.global_scene_en) for q in event_queries]

        valid_sequences = []

        # 4. Run DP per candidate video
        for v_rank, (video_id, agg_score) in enumerate(top_videos):
            kfs_in_v = sorted(video_kfs[video_id], key=lambda x: x[1]["frame_idx"])
            M = len(kfs_in_v)
            if M < num_events:
                continue

            kf_indices = [k[0] for k in kfs_in_v]
            kf_vis_matrix = searcher.vis_matrix[kf_indices].astype(np.float32)

            # Similarity matrix: (M x 768) @ (768 x N) -> M x N
            sim_matrix = kf_vis_matrix @ np.column_stack(ev_vis_vecs)

            # DP Table: dp[ev_idx][kf_idx] = (max_score, prev_kf_idx)
            dp: list[list[tuple[float, int]]] = []
            dp.append([(float(sim_matrix[m, 0]), -1) for m in range(M)])

            for ev_idx in range(1, num_events):
                curr_dp = []
                prev_dp = dp[ev_idx - 1]
                for curr_m in range(M):
                    best_score = -float("inf")
                    best_prev = -1
                    curr_sim = float(sim_matrix[curr_m, ev_idx])
                    for prev_m in range(curr_m):
                        if prev_dp[prev_m][0] > best_score:
                            best_score = prev_dp[prev_m][0]
                            best_prev = prev_m
                    curr_dp.append((best_score + curr_sim, best_prev))
                dp.append(curr_dp)

            # For top 5 videos, extract top 2-3 K-best paths; for other videos, extract 1 path
            num_paths_to_take = 3 if v_rank < 5 else 1
            last_layer_scores = [d[0] for d in dp[-1]]
            sorted_last_indices = np.argsort(last_layer_scores)[::-1]

            for path_idx in range(min(num_paths_to_take, len(sorted_last_indices))):
                last_m = int(sorted_last_indices[path_idx])
                best_seq_score = float(last_layer_scores[last_m])
                if best_seq_score <= -1e5:
                    continue

                path = []
                curr_m = last_m
                for ev_idx in range(num_events - 1, -1, -1):
                    path.append(kfs_in_v[curr_m][1])
                    curr_m = dp[ev_idx][curr_m][1]
                path.reverse()

                frames = [p["frame_idx"] for p in path]
                times = [p.get("pts_time_s", 0.0) for p in path]
                frames_str = ", ".join(map(str, frames))

                # Build rich event dossiers
                event_dossiers = []
                for i, p in enumerate(path):
                    img_path = f"keyframes/{video_id}/{p.get('keyframe_n', 1):03d}.jpg"
                    event_dossiers.append(
                        {
                            "event_idx": i + 1,
                            "frame_idx": p["frame_idx"],
                            "pts_time_s": p.get("pts_time_s", 0.0),
                            "keyframe_n": p.get("keyframe_n", 1),
                            "image_relpath": img_path,
                            "score_vis": round(float(sim_matrix[frames.index(p['frame_idx']) if p['frame_idx'] in frames else 0, i]), 4),
                            "asr_transcript": p.get("asr_transcript", ""),
                        }
                    )

                valid_sequences.append(
                    {
                        "video_id": video_id,
                        "matched_frames": frames,
                        "timestamps": times,
                        "sequence_score": round(float(best_seq_score / num_events), 4),
                        "submission_string": f"{video_id}, {frames_str}",
                        "event_dossiers": event_dossiers,
                    }
                )

        valid_sequences.sort(key=lambda x: x["sequence_score"], reverse=True)
        for rank, s in enumerate(valid_sequences[:final_top_k], 1):
            s["rank"] = rank

        return valid_sequences[:final_top_k]

    # ──────────────────────────────────────────────────────────────────────────
    # 4. TRAKE Macro-Span Audio Narrative Reranker (Stage 3 & 4)
    # ──────────────────────────────────────────────────────────────────────────
    def _evaluate_narrative_alignment(
        self,
        events_desc: list[str],
        audio_span_text: str,
    ) -> tuple[float, str]:
        """Evaluate how well a continuous audio span matches the full multi-event sequence."""
        if not audio_span_text or len(audio_span_text.strip()) < 10:
            return 0.5, "No dialogue detected in action span"

        span_lower = audio_span_text.lower()

        # 1. High-speed Lexical Procedural Alignment Score
        matched_events = 0
        for ev in events_desc:
            words = [w for w in ev.lower().replace(".", "").replace(",", "").split() if len(w) > 2]
            hits = sum(1 for w in words if w in span_lower)
            if hits >= 2:
                matched_events += 1.0
            elif hits == 1:
                matched_events += 0.6
            else:
                matched_events += 0.2

        lexical_score = min(max(matched_events / max(len(events_desc), 1), 0.1), 1.0)
        return lexical_score, f"Lexical alignment score: {lexical_score:.2f}"

    def rerank_trake_sequences(
        self,
        event_descriptions: list[str],
        candidate_sequences: list[dict[str, Any]],
        searcher: Any,
        final_top_k: int = 100,
    ) -> list[dict[str, Any]]:
        """Stage 3 & 4: Re-rank candidate sequences using macro-span narrative alignment."""
        if not candidate_sequences:
            return []

        logger.info(f"⚡ Reranking {len(candidate_sequences)} TRAKE sequences via Macro-Span Audio Narrative...")

        reranked = []
        for i, seq in enumerate(candidate_sequences):
            vid = seq["video_id"]
            frames = seq.get("matched_frames", [])
            dp_score = float(seq.get("sequence_score", 0.5))

            if not frames:
                reranked.append(seq)
                continue

            start_f = min(frames)
            end_f = max(frames)

            # Stage 3: Extract macro-span audio transcript
            audio_span = searcher.get_video_audio_span(vid, start_f, end_f) if hasattr(searcher, "get_video_audio_span") else ""

            # Stage 4: Fast narrative alignment score
            if audio_span:
                narrative_score, reasoning = self._evaluate_narrative_alignment(event_descriptions, audio_span)
            else:
                narrative_score = dp_score
                reasoning = "No dialogue detected in action span"

            # Final blended score
            final_score = round(0.40 * dp_score + 0.60 * narrative_score, 4)

            seq_copy = dict(seq)
            seq_copy["dp_score"] = dp_score
            seq_copy["narrative_score"] = round(narrative_score, 4)
            seq_copy["final_score"] = final_score
            seq_copy["narrative_reasoning"] = reasoning
            seq_copy["audio_span"] = audio_span
            reranked.append(seq_copy)

        # Sort by final blended score
        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        for rank, s in enumerate(reranked[:final_top_k], 1):
            s["rank"] = rank

        return reranked[:final_top_k]

