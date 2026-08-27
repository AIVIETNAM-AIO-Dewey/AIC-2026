"""Stage 2: Precision Cross-Attention Re-Ranking & TRAKE Dynamic Programming Path Finder.

Implements task-specific precision layers:
1. KIS: bge-reranker-v2-m3 Cross-Encoder on candidate dossiers, then a Heuristic Scoring layer.
2. VQA: Cross-Encoder ranking + Extractive LLM Reader for Top 1 evidence answer.
3. TRAKE: Dynamic Programming Monotonic Path Matching across events (t(f1) < t(f2) < ... < t(fN)).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from online.src.contracts.query import ParsedQuery
from online.src.retrieval.embeddings import ModelRegistry
from online.src.retrieval.vqa_reasoner import VQAReasoner

logger = logging.getLogger(__name__)

_DIACRITIC_D = str.maketrans({"đ": "d", "Đ": "d"})
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize_text(text: str) -> str:
    """Fold text to lowercase, diacritic-free, punctuation-free for literal matching.

    Real OCR payloads are noisy and accented ("giây", "06:30:11"), while operators
    routinely type queries unaccented ("giay"). Matching the raw strings misses ~94%
    of accented rows, so both sides are folded before comparison.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFD", text.lower().translate(_DIACRITIC_D))
    folded = "".join(ch for ch in folded if unicodedata.category(ch) != "Mn")
    return " ".join(_PUNCT_RE.sub(" ", folded).split())


class Stage2Reranker:
    """Stage 2 Precision Re-Ranker and Reasoning Layer."""

    # Heuristic Scoring weights (tunable in one place).
    # Stage 1 already computes channel consensus, visual similarity and OCR hits, but
    # the cross-encoder only ever sees a flattened text dossier. These weights control
    # how much of that structured evidence is scored back in.
    HEURISTIC_WEIGHTS = {
        "stage1": 0.35,                 # RRF consensus evidence carried from Stage 1
        "cross_encoder": 0.65,          # BGE cross-encoder semantic relevance
        "consensus_per_channel": 0.04,  # per extra channel agreeing beyond the first
        "visual_confidence": 0.10,      # SigLIP probability, invisible to the dossier
        "ocr_exact": 0.08,              # queried OCR keyword literally on screen
        "ocr_folded_discount": 0.5,     # credit for a diacritic-folded-only OCR hit
        "dam_coverage": 0.06,           # fraction of queried objects detected in frame
        "textless_ce_trust": 0.25,      # CE trust when the dossier holds no real text
    }

    # Folded OCR matching is only allowed for tokens at least this long: short folded
    # tokens collide badly in Vietnamese ("ở" folds to "o" and matches ô/ổ/ộ/ố/ơ).
    MIN_FOLDED_MATCH_LEN = 4

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        vqa_reasoner: Optional[VQAReasoner] = None,
    ):
        self.registry = registry or ModelRegistry.get_instance()
        self.vqa_reasoner = vqa_reasoner or VQAReasoner()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. KIS Precision Re-Ranking (BGE Cross-Encoder -> Heuristic Scoring)
    # ──────────────────────────────────────────────────────────────────────────
    def _build_dossier(self, c: dict[str, Any]) -> tuple[str, dict[str, bool]]:
        """Flatten a candidate's multimodal payload into cross-encoder input text.

        Returns the dossier plus which modalities actually contributed text, so the
        scoring layer knows whether the cross-encoder had anything real to read.
        """
        parts = []
        coverage = {"dam": False, "asr": False, "ocr": False}

        if c.get("dam_summary"):
            parts.append(f"[Visual Objects] {c['dam_summary']}")
            coverage["dam"] = True
        if c.get("asr_transcript"):
            parts.append(f"[Spoken Speech] {c['asr_transcript']}")
            coverage["asr"] = True
        if c.get("ocr_text"):
            parts.append(f"[Screen Text] {c['ocr_text']}")
            coverage["ocr"] = True

        if parts:
            return " ".join(parts), coverage

        # No text payload at all: the cross-encoder will be scoring a placeholder.
        return f"[Scene] Video {c['video_id']} frame {c['frame_idx']}", coverage

    def compute_heuristic_score(
        self,
        parsed_query: ParsedQuery,
        c: dict[str, Any],
        ce_score: float,
        coverage: dict[str, bool],
    ) -> tuple[float, str]:
        """Heuristic Scoring: re-inject the structured signals the cross-encoder never saw.

        The cross-encoder only reads a flattened text dossier, so it is blind to channel
        consensus, visual similarity and exact OCR hits -- all of which Stage 1 already
        computed and stored on the candidate. This layer scores those explicitly instead
        of hiding them behind a single hard-coded Stage 1 weight.

        Returns:
            (final_score, human-readable breakdown of every term that fired)
        """
        w = self.HEURISTIC_WEIGHTS
        norm_s1 = float(c.get("normalized_score", 0.0) or 0.0)
        has_text = any(coverage.values())
        breakdown = []

        # Base blend. A dossier with no text makes ce_score meaningless (it scored a
        # placeholder string), so lean on Stage 1 visual evidence rather than let a
        # near-zero cross-encoder score bury an otherwise strong frame.
        if has_text:
            base = w["stage1"] * norm_s1 + w["cross_encoder"] * ce_score
            breakdown.append(
                f"{w['stage1']:.2f}*s1({norm_s1:.3f}) + {w['cross_encoder']:.2f}*ce({ce_score:.3f})"
            )
        else:
            trust = w["textless_ce_trust"]
            base = (1.0 - trust) * norm_s1 + trust * ce_score
            breakdown.append(
                f"textless {1.0 - trust:.2f}*s1({norm_s1:.3f}) + {trust:.2f}*ce({ce_score:.3f})"
            )

        bonus = 0.0

        # H1. Channel consensus: agreement across independent modalities is the single
        # strongest precision signal Stage 1 produces, and the dossier erases it.
        extra_channels = max(0, int(c.get("active_channels", 0) or 0) - 1)
        if extra_channels:
            term = w["consensus_per_channel"] * extra_channels
            bonus += term
            breakdown.append(f"+consensus({extra_channels}ch)={term:.3f}")

        # H2. Visual confidence: SigLIP similarity is never part of the text dossier.
        prob_vis = float(c.get("prob_vis", 0.0) or 0.0)
        if prob_vis > 0:
            term = w["visual_confidence"] * prob_vis
            bonus += term
            breakdown.append(f"+visual({prob_vis:.3f})={term:.3f}")

        # H3. OCR exact match: a literal on-screen string hit is near-decisive evidence,
        # but the cross-encoder treats it as ordinary prose.
        raw_ocr = (c.get("ocr_text") or "").lower()
        folded_ocr = _normalize_text(raw_ocr)
        keywords = [k.strip().lower() for k in (parsed_query.ocr_keywords or []) if k.strip()]
        if raw_ocr and keywords:
            # Two tiers. A diacritic-exact hit is trusted fully. A hit that only appears
            # after folding is credited at a discount: it recovers PPOCR mis-reading the
            # tone mark ("giây" -> "giày"/"giay"), but folding also merges genuinely
            # different Vietnamese words ("tai/tài/tại"), so it must not score as highly.
            exact = 0.0
            fuzzy = 0.0
            for kw in keywords:
                if kw in raw_ocr:
                    exact += 1.0
                    continue
                folded_kw = _normalize_text(kw)
                # Short tokens are far too collision-prone once folded ("ở" -> "o").
                if len(folded_kw) >= self.MIN_FOLDED_MATCH_LEN and folded_kw in folded_ocr:
                    fuzzy += 1.0

            if exact or fuzzy:
                credit = exact + w["ocr_folded_discount"] * fuzzy
                term = w["ocr_exact"] * (credit / len(keywords))
                bonus += term
                breakdown.append(
                    f"+ocr(exact {exact:.0f} fuzzy {fuzzy:.0f}/{len(keywords)})={term:.3f}"
                )

        # H4. DAM object coverage: reward frames whose detected objects literally cover
        # the objects the query asked for. Captions are English, so folding only strips
        # case and punctuation here -- no Vietnamese tone collapse to worry about.
        dam_summary = _normalize_text(c.get("dam_summary") or "")
        objects = [n for n in (_normalize_text(o) for o in (parsed_query.objects_en or [])) if n]
        if dam_summary and objects:
            hits = sum(1 for o in objects if o in dam_summary)
            if hits:
                term = w["dam_coverage"] * (hits / len(objects))
                bonus += term
                breakdown.append(f"+dam({hits}/{len(objects)})={term:.3f}")

        return max(0.0, base + bonus), " ".join(breakdown)

    def rerank_kis(
        self,
        parsed_query: ParsedQuery,
        candidates: list[dict[str, Any]],
        final_top_k: int = 20,
        top_k_rerank: int = 50,
    ) -> list[dict[str, Any]]:
        """RRF pool -> BGE cross-encoder -> heuristic scoring -> ranked evidence.

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
        dossiers: list[str] = []
        coverages: list[dict[str, bool]] = []
        for c in to_eval:
            dossier_text, coverage = self._build_dossier(c)
            dossiers.append(dossier_text)
            coverages.append(coverage)

        # Compute Cross-Encoder scores for the top candidates
        ce_scores = self.registry.compute_rerank_scores(user_query, dossiers)

        # Heuristic Scoring layer on top of the cross-encoder relevance
        reranked = []
        for i, c in enumerate(to_eval):
            c_copy = dict(c)
            ce_score = float(ce_scores[i])
            final_score, breakdown = self.compute_heuristic_score(
                parsed_query, c, ce_score, coverages[i]
            )

            c_copy["cross_encoder_score"] = round(ce_score, 4)
            c_copy["final_score"] = round(final_score, 4)
            c_copy["heuristic_breakdown"] = breakdown
            c_copy["dossier_coverage"] = coverages[i]
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

