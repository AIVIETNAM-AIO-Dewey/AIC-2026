"""Stage 1 Multimodal Fusion & Funnel (Weighted RRF + Synergy Multiplier).

Combines independent retrieval rankings from:
1. SigLIP-2 Visual (768-d)
2. DAM Objects Multi-Subject (1024-d)
3. Audio ASR Dialogue (1024-d)
4. OCR Text

Formula:
  RRF_Base(k) = sum_{c} [ w_c / (k_rrf + rank_c(k)) ]
  Synergy(k) = 1.0 + 0.20 * max(0, active_channels(k) - 1)
  Final_Stage1_Score(k) = RRF_Base(k) * Synergy(k)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

from online.src.contracts.query import ParsedQuery, SearchResult
from online.src.retrieval.embeddings import ModelRegistry
from online.src.retrieval.vector_search import FastVectorSearchEngine

logger = logging.getLogger(__name__)


class MultimodalFusionEngine:
    """Stage 1 Multimodal Funnel & Weighted Reciprocal Rank Fusion Engine."""

    def __init__(
        self,
        searcher: Optional[FastVectorSearchEngine] = None,
        registry: Optional[ModelRegistry] = None,
        k_rrf: int = 60,
    ):
        self.searcher = searcher or FastVectorSearchEngine()
        self.registry = registry or ModelRegistry.get_instance()
        self.k_rrf = k_rrf

    def retrieve_branches(
        self,
        parsed_query: ParsedQuery,
        branch_limit: int = 500,
    ) -> dict[str, list[dict[str, Any]]]:
        """Query all 4 retrieval channels independently and return raw hits for caching."""
        weights = parsed_query.weights or {"vis": 0.35, "dam": 0.30, "asr": 0.35, "ocr": 0.00}
        w_vis = weights.get("vis", 0.35)
        w_dam = weights.get("dam", 0.30)
        w_asr = weights.get("asr", 0.35)
        w_ocr = weights.get("ocr", 0.00)

        # Branch 1: Visual (SigLIP-2)
        vis_hits = []
        if parsed_query.global_scene_en and w_vis > 0:
            vis_vec = self.registry.embed_siglip_text(parsed_query.global_scene_en)
            vis_hits = self.searcher.search_visual(vis_vec, top_k=branch_limit)

        # Branch 2: DAM Objects
        dam_hits = []
        if parsed_query.objects_en and w_dam > 0:
            obj_vecs = [self.registry.embed_bge_text(obj) for obj in parsed_query.objects_en]
            dam_hits = self.searcher.search_dam(obj_vecs, parsed_query.objects_en, top_k=branch_limit)

        # Branch 3: Audio ASR
        asr_hits = []
        audio_text = parsed_query.speech_vi.strip() if parsed_query.speech_vi else parsed_query.original_query
        if audio_text and w_asr > 0:
            speech_vec = self.registry.embed_bge_text(audio_text)
            asr_hits = self.searcher.search_speech(speech_vec, top_k=branch_limit)

        # Branch 4: OCR Text
        ocr_hits = []
        if parsed_query.ocr_keywords and w_ocr > 0:
            ocr_hits = self.searcher.search_ocr(parsed_query.ocr_keywords, top_k=branch_limit)

        return {
            "vis": vis_hits,
            "dam": dam_hits,
            "asr": asr_hits,
            "ocr": ocr_hits,
        }

    def fuse_from_branch_hits(
        self,
        vis_hits: list[dict[str, Any]],
        dam_hits: list[dict[str, Any]],
        asr_hits: list[dict[str, Any]],
        ocr_hits: list[dict[str, Any]],
        weights: Optional[dict[str, float]] = None,
        top_k_pool: int = 300,
    ) -> list[dict[str, Any]]:
        """Instant CPU RRF fusion (< 5ms) on pre-computed branch hits with custom weights."""
        if weights is None:
            weights = {"vis": 0.35, "dam": 0.30, "asr": 0.35, "ocr": 0.00}
        
        w_vis = weights.get("vis", 0.35)
        w_dam = weights.get("dam", 0.30)
        w_asr = weights.get("asr", 0.35)
        w_ocr = weights.get("ocr", 0.00)

        candidates: dict[tuple[str, int], dict[str, Any]] = defaultdict(
            lambda: {
                "video_id": "",
                "frame_idx": 0,
                "keyframe_n": 0,
                "pts_time_s": 0.0,
                "image_relpath": "",
                "dam_summary": "",
                "asr_transcript": "",
                "matched_boxes": [],
                "rank_vis": None,
                "score_vis": 0.0,
                "prob_vis": 0.0,
                "rank_dam": None,
                "score_dam": 0.0,
                "rank_asr": None,
                "score_asr": 0.0,
                "rank_ocr": None,
                "score_ocr": 0.0,
                "active_channels": 0,
            }
        )

        for h in vis_hits:
            key = (h["video_id"], h["frame_idx"])
            c = candidates[key]
            c["video_id"] = h["video_id"]
            c["frame_idx"] = h["frame_idx"]
            c["keyframe_n"] = h["keyframe_n"]
            c["pts_time_s"] = h["pts_time_s"]
            c["image_relpath"] = h["image_relpath"]
            c["dam_summary"] = h.get("dam_summary", "")
            c["rank_vis"] = h["rank"]
            c["score_vis"] = h["score"]
            c["prob_vis"] = h.get("prob", 0.0)

        for h in dam_hits:
            key = (h["video_id"], h["frame_idx"])
            c = candidates[key]
            c["video_id"] = h["video_id"]
            c["frame_idx"] = h["frame_idx"]
            c["keyframe_n"] = h["keyframe_n"]
            c["rank_dam"] = h["rank"]
            c["score_dam"] = h["composite_score"]
            c["matched_boxes"] = h.get("matched_boxes", [])

        for h in asr_hits:
            key = (h["video_id"], h["frame_idx"])
            c = candidates[key]
            c["video_id"] = h["video_id"]
            c["frame_idx"] = h["frame_idx"]
            c["keyframe_n"] = h["keyframe_n"]
            c["pts_time_s"] = h["pts_time_s"]
            c["rank_asr"] = h["rank"]
            c["score_asr"] = h["score"]
            c["asr_transcript"] = h.get("transcript", "")

        for h in ocr_hits:
            key = (h["video_id"], h["frame_idx"])
            c = candidates[key]
            c["video_id"] = h["video_id"]
            c["frame_idx"] = h["frame_idx"]
            c["keyframe_n"] = h["keyframe_n"]
            c["rank_ocr"] = h["rank"]
            c["score_ocr"] = 1.0

        fused_pool = []
        for key, c in candidates.items():
            active_count = 0
            rrf_base = 0.0
            formula_terms = []

            if c["rank_vis"] is not None:
                active_count += 1
                vis_term = w_vis / (self.k_rrf + c["rank_vis"])
                rrf_base += vis_term
                formula_terms.append(f"{w_vis:.2f}/(60+{c['rank_vis']})")

            if c["rank_dam"] is not None:
                active_count += 1
                dam_term = w_dam / (self.k_rrf + c["rank_dam"])
                rrf_base += dam_term
                formula_terms.append(f"{w_dam:.2f}/(60+{c['rank_dam']})")

            if c["rank_asr"] is not None:
                active_count += 1
                asr_term = w_asr / (self.k_rrf + c["rank_asr"])
                rrf_base += asr_term
                formula_terms.append(f"{w_asr:.2f}/(60+{c['rank_asr']})")

            if c["rank_ocr"] is not None:
                active_count += 1
                ocr_term = w_ocr / (self.k_rrf + c["rank_ocr"])
                rrf_base += ocr_term
                formula_terms.append(f"{w_ocr:.2f}/(60+{c['rank_ocr']})")

            c["active_channels"] = active_count
            synergy = 1.0 + 0.20 * max(0, active_count - 1)
            c["synergy_multiplier"] = round(synergy, 2)
            c["rrf_base"] = round(rrf_base, 6)

            stage1_score = rrf_base * synergy
            c["stage1_score"] = round(stage1_score, 6)
            formula_str = f"({' + '.join(formula_terms)}) * {synergy:.2f}"
            c["calculation_breakdown"] = formula_str

            if not c["image_relpath"]:
                c["image_relpath"] = f"keyframes/{c['video_id']}/{c['keyframe_n']:03d}.jpg"

            fused_pool.append(c)

        fused_pool.sort(key=lambda x: x["stage1_score"], reverse=True)

        if fused_pool:
            max_score = fused_pool[0]["stage1_score"]
            for rank, item in enumerate(fused_pool[:top_k_pool], 1):
                item["rank"] = rank
                item["normalized_score"] = round(item["stage1_score"] / max(max_score, 1e-6), 4)

        return fused_pool[:top_k_pool]

    def retrieve_and_fuse(
        self,
        parsed_query: ParsedQuery,
        top_k_pool: int = 300,
        branch_limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Run all 4 search channels and fuse with Weighted RRF & Synergy."""
        branches = self.retrieve_branches(parsed_query, branch_limit=branch_limit)
        return self.fuse_from_branch_hits(
            vis_hits=branches["vis"],
            dam_hits=branches["dam"],
            asr_hits=branches["asr"],
            ocr_hits=branches["ocr"],
            weights=parsed_query.weights,
            top_k_pool=top_k_pool,
        )
