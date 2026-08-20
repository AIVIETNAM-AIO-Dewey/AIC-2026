"""Stage 1: High-Speed Multimodal Funnel across 4 Channels via Qdrant."""

from __future__ import annotations

import logging
from typing import Optional
from collections import defaultdict
import numpy as np
from qdrant_client import QdrantClient

from online.src.contracts.query import ParsedQuery, MatchedObject
from online.src.retrieval.embeddings import ModelRegistry

logger = logging.getLogger(__name__)


class Stage1Funnel:
    """Parallel 4-Channel Retrieval Funnel & Weighted Reciprocal Rank Fusion."""

    KEYFRAME_COLLECTION = "keyframes"
    DAM_COLLECTION = "dam_objects"

    def __init__(self, client: QdrantClient, models: ModelRegistry) -> None:
        self.client = client
        self.models = models

    def search_candidates(
        self,
        parsed_query: ParsedQuery,
        top_k: int = 50,
        candidate_pool_multiplier: int = 10,
    ) -> list[dict]:
        """Execute parallel 4-channel search and return top-K fused candidate frames."""
        weights = parsed_query.weights
        w_vis = weights.get("vis", 0.45)
        w_dam = weights.get("dam", 0.40)
        w_asr = weights.get("asr", 0.15)
        w_ocr = weights.get("ocr", 0.00)

        pool_size = top_k * candidate_pool_multiplier

        # Data structures to store per-channel rankings
        channel_ranks = defaultdict(lambda: {"vis": None, "dam": None, "asr": None, "ocr": None})
        channel_scores = defaultdict(lambda: {"vis": 0.0, "dam": 0.0, "asr": 0.0, "ocr": 0.0})
        frame_payloads = {}
        dam_matched_objects = defaultdict(list)

        # -------------------------------------------------------------
        # CHANNEL 1: Global Visual Search (SigLIP-2)
        # -------------------------------------------------------------
        if w_vis > 0 and parsed_query.global_scene_en:
            vis_query_vec = self.models.encode_siglip_text([parsed_query.global_scene_en])[0].tolist()
            vis_hits = self.client.query_points(
                collection_name=self.KEYFRAME_COLLECTION,
                query=vis_query_vec,
                using="visual",
                limit=pool_size,
            ).points

            for rank, hit in enumerate(vis_hits, 1):
                f_key = (hit.payload["video_id"], hit.payload["keyframe_n"])
                channel_ranks[f_key]["vis"] = rank
                channel_scores[f_key]["vis"] = float(hit.score)
                if f_key not in frame_payloads:
                    frame_payloads[f_key] = hit.payload

        # -------------------------------------------------------------
        # CHANNEL 2: Fine-Grained DAM Object Search (BGE-M3 Max-Pool)
        # -------------------------------------------------------------
        if w_dam > 0 and parsed_query.objects_en:
            obj_vectors = self.models.encode_bge_m3(parsed_query.objects_en)
            frame_obj_scores = defaultdict(lambda: defaultdict(float))
            frame_obj_records = defaultdict(list)

            for q_idx, q_vec in enumerate(obj_vectors):
                dam_hits = self.client.query_points(
                    collection_name=self.DAM_COLLECTION,
                    query=q_vec.tolist(),
                    limit=pool_size * 2,
                ).points

                for hit in dam_hits:
                    f_key = (hit.payload["video_id"], hit.payload["keyframe_n"])
                    score = float(hit.score)
                    # Track max score per query object
                    if score > frame_obj_scores[f_key][q_idx]:
                        frame_obj_scores[f_key][q_idx] = score

                    matched_obj = MatchedObject(
                        region_id=hit.payload.get("region_id", 1),
                        class_entity=hit.payload.get("class_entity", "Object"),
                        description_en=hit.payload.get("description_en", ""),
                        score=score,
                        bbox=hit.payload.get("bbox", [0.0, 0.0, 1.0, 1.0]),
                    )
                    frame_obj_records[f_key].append(matched_obj)

            # Max-pool across objects per frame
            dam_frame_totals = []
            for f_key, q_scores in frame_obj_scores.items():
                total_dam_score = sum(q_scores.values())
                dam_frame_totals.append((f_key, total_dam_score))

            dam_frame_totals.sort(key=lambda x: x[1], reverse=True)
            for rank, (f_key, d_score) in enumerate(dam_frame_totals[:pool_size], 1):
                channel_ranks[f_key]["dam"] = rank
                channel_scores[f_key]["dam"] = float(d_score)
                # Keep top 3 matching object boxes
                sorted_objs = sorted(frame_obj_records[f_key], key=lambda o: o.score, reverse=True)
                dam_matched_objects[f_key] = sorted_objs[:3]

        # -------------------------------------------------------------
        # CHANNEL 3: Spoken Speech Search (ASR BGE-M3)
        # -------------------------------------------------------------
        if w_asr > 0 and parsed_query.speech_vi:
            speech_query_vec = self.models.encode_bge_m3([parsed_query.speech_vi])[0].tolist()
            asr_hits = self.client.query_points(
                collection_name=self.KEYFRAME_COLLECTION,
                query=speech_query_vec,
                using="speech",
                limit=pool_size,
            ).points

            for rank, hit in enumerate(asr_hits, 1):
                # Filter out silence/non-verbal vectors (near 0)
                if hit.score > 0.10:
                    f_key = (hit.payload["video_id"], hit.payload["keyframe_n"])
                    channel_ranks[f_key]["asr"] = rank
                    channel_scores[f_key]["asr"] = float(hit.score)
                    if f_key not in frame_payloads:
                        frame_payloads[f_key] = hit.payload

        # -------------------------------------------------------------
        # CHANNEL 4: On-Screen Text (OCR BM25 Keyword Filter)
        # -------------------------------------------------------------
        if w_ocr > 0 and parsed_query.ocr_keywords:
            for f_key, payload in frame_payloads.items():
                ocr_text = payload.get("ocr_text", "").lower()
                matches = sum(1 for kw in parsed_query.ocr_keywords if kw.lower() in ocr_text)
                if matches > 0:
                    channel_scores[f_key]["ocr"] = float(matches)

        # -------------------------------------------------------------
        # RECIPROCAL RANK FUSION (RRF) & SYNERGY BOOST
        # -------------------------------------------------------------
        all_candidate_keys = set(channel_ranks.keys())
        fused_candidates = []

        for f_key in all_candidate_keys:
            ranks = channel_ranks[f_key]
            scores = channel_scores[f_key]

            # Standard RRF formula with k=60
            rrf_score = 0.0
            active_channels = 0

            if ranks["vis"] is not None:
                rrf_score += w_vis * (1.0 / (60.0 + ranks["vis"]))
                active_channels += 1
            if ranks["dam"] is not None:
                rrf_score += w_dam * (1.0 / (60.0 + ranks["dam"]))
                active_channels += 1
            if ranks["asr"] is not None:
                rrf_score += w_asr * (1.0 / (60.0 + ranks["asr"]))
                active_channels += 1
            if scores["ocr"] > 0:
                rrf_score += w_ocr * 0.05 * scores["ocr"]
                active_channels += 1

            # Multi-Modal Synergy Multiplier (boost frames matching 2+ modalities)
            synergy_multiplier = 1.0 + 0.20 * max(0, active_channels - 1)
            final_stage1_score = rrf_score * synergy_multiplier

            payload = frame_payloads.get(f_key, {})

            fused_candidates.append({
                "video_id": f_key[0],
                "keyframe_n": f_key[1],
                "frame_idx": payload.get("frame_idx", 0),
                "pts_time_s": payload.get("pts_time_s", 0.0),
                "frame_uid": payload.get("frame_uid", f"{f_key[0]}:{f_key[1]}"),
                "image_relpath": payload.get("image_relpath", f"keyframes/{f_key[0]}/{f_key[1]:03d}.jpg"),
                "stage1_score": final_stage1_score,
                "visual_score": scores["vis"],
                "dam_score": scores["dam"],
                "asr_score": scores["asr"],
                "ocr_score": scores["ocr"],
                "dam_summary_en": payload.get("dam_summary_en", ""),
                "asr_transcript_vi": payload.get("asr_transcript_vi", ""),
                "ocr_text": payload.get("ocr_text", ""),
                "has_speech": payload.get("has_speech", False),
                "matched_objects": dam_matched_objects[f_key],
            })

        fused_candidates.sort(key=lambda x: x["stage1_score"], reverse=True)
        return fused_candidates[:top_k]
