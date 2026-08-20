"""Batched Ingestion and Indexing into Embedded Qdrant Vector Database."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from online.src.retrieval.embeddings import ModelRegistry

logger = logging.getLogger(__name__)


class QdrantIndexer:
    """Manages creation and indexing of keyframes and dam_objects in Qdrant."""

    KEYFRAME_COLLECTION = "keyframes"
    DAM_COLLECTION = "dam_objects"

    def __init__(
        self,
        qdrant_db_path: str = "/Users/khoale/Downloads/AIC_HCM/qdrant_db",
        models: Optional[ModelRegistry] = None,
    ) -> None:
        self.qdrant_db_path = Path(qdrant_db_path)
        self.qdrant_db_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.qdrant_db_path))
        self.models = models or ModelRegistry()
        logger.info(f"Connected to embedded Qdrant at: {self.qdrant_db_path}")

    def init_collections(self, force_recreate: bool = False) -> None:
        """Create Qdrant collections with multi-vector configurations."""
        existing = [c.name for c in self.client.get_collections().collections]

        # 1. Keyframes Collection (Multi-vector: visual 768-d + speech 1024-d)
        if self.KEYFRAME_COLLECTION not in existing or force_recreate:
            if force_recreate and self.KEYFRAME_COLLECTION in existing:
                self.client.delete_collection(self.KEYFRAME_COLLECTION)
            self.client.create_collection(
                collection_name=self.KEYFRAME_COLLECTION,
                vectors_config={
                    "visual": VectorParams(size=768, distance=Distance.COSINE),
                    "speech": VectorParams(size=1024, distance=Distance.COSINE),
                },
            )
            logger.info(f"Created collection '{self.KEYFRAME_COLLECTION}'")

        # 2. DAM Objects Collection (1024-d BGE-M3 dense)
        if self.DAM_COLLECTION not in existing or force_recreate:
            if force_recreate and self.DAM_COLLECTION in existing:
                self.client.delete_collection(self.DAM_COLLECTION)
            self.client.create_collection(
                collection_name=self.DAM_COLLECTION,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            logger.info(f"Created collection '{self.DAM_COLLECTION}'")

    def index_all_videos(
        self,
        map_dir: Path,
        scene_dir: Path,
        dam_dir: Path,
        asr_dir: Path,
        ocr_dir: Optional[Path] = None,
        video_ids: Optional[list[str]] = None,
        batch_size: int = 500,
    ) -> dict[str, int]:
        """Index full corpus into Qdrant."""
        self.init_collections(force_recreate=False)

        if video_ids is None:
            video_ids = sorted([f.stem for f in map_dir.glob("*.csv")])

        logger.info(f"Starting Qdrant indexing across {len(video_ids)} videos...")

        total_kf_points = 0
        total_dam_points = 0
        point_id_counter = 1
        dam_point_id_counter = 1

        # Check existing indexed points count
        kf_info = self.client.get_collection(self.KEYFRAME_COLLECTION)
        dam_info = self.client.get_collection(self.DAM_COLLECTION)
        logger.info(f"Initial status -> keyframes: {kf_info.points_count}, dam_objects: {dam_info.points_count}")

        kf_points_buffer = []
        dam_points_buffer = []

        for video_id in tqdm(video_ids, desc="Indexing Videos into Qdrant"):
            map_file = map_dir / f"{video_id}.csv"
            scene_npy = scene_dir / f"{video_id}.f16.npy"
            scene_jsonl = scene_dir / f"{video_id}.jsonl"
            dam_file = dam_dir / f"{video_id}.jsonl"
            asr_file = asr_dir / f"{video_id}.jsonl"
            ocr_file = (ocr_dir / f"{video_id}.jsonl") if ocr_dir else None

            if not map_file.exists() or not scene_npy.exists() or not dam_file.exists():
                continue

            # Load Map & Scene visual vectors
            df_map = pd.read_csv(map_file)
            visual_arr = np.load(scene_npy).astype(np.float32)

            # Load ASR mapping: keyframe_n -> speech segment
            asr_by_kf = {}
            if asr_file.exists():
                with open(asr_file, "r", encoding="utf-8") as f:
                    for line in f:
                        d = json.loads(line)
                        for kf in d.get("keyframes", []):
                            asr_by_kf[kf["keyframe_n"]] = d

            # Load DAM descriptions: keyframe_n -> list of regions
            dam_by_kf = {}
            with open(dam_file, "r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    dam_by_kf[d["keyframe_n"]] = d.get("regions", [])

            # Load OCR if available
            ocr_by_kf = {}
            if ocr_file and ocr_file.exists():
                with open(ocr_file, "r", encoding="utf-8") as f:
                    for line in f:
                        d = json.loads(line)
                        ocr_by_kf[d["keyframe_n"]] = d.get("ocr_text", "")

            # 1. Prepare Keyframes Points
            speech_texts_to_embed = []
            speech_indices = []

            for row_idx, row in df_map.iterrows():
                k_n = int(row["n"])
                if k_n in asr_by_kf:
                    speech_texts_to_embed.append(asr_by_kf[k_n]["transcript_raw"])
                    speech_indices.append(row_idx)

            # Batch embed ASR speech on MPS/GPU
            if speech_texts_to_embed:
                speech_embeds = self.models.encode_bge_m3(speech_texts_to_embed)
            else:
                speech_embeds = np.empty((0, 1024), dtype=np.float32)

            speech_embed_map = {idx: speech_embeds[i] for i, idx in enumerate(speech_indices)}

            for row_idx, row in df_map.iterrows():
                k_n = int(row["n"])
                pts_time = float(row["pts_time"])
                fps = float(row["fps"])
                f_idx = int(row["frame_idx"])
                f_uid = f"{video_id}:{f_idx}"

                vis_vec = visual_arr[row_idx].tolist()
                speech_vec = (
                    speech_embed_map[row_idx].tolist()
                    if row_idx in speech_embed_map
                    else [0.0] * 1024
                )

                regions = dam_by_kf.get(k_n, [])
                dam_summary = " ".join([r.get("caption", {}).get("description_en", "") for r in regions])
                asr_text = asr_by_kf.get(k_n, {}).get("transcript_raw", "")
                ocr_text = ocr_by_kf.get(k_n, "")

                payload = {
                    "video_id": video_id,
                    "keyframe_n": k_n,
                    "frame_idx": f_idx,
                    "pts_time_s": pts_time,
                    "fps": fps,
                    "frame_uid": f_uid,
                    "image_relpath": f"keyframes/{video_id}/{k_n:03d}.jpg",
                    "dam_summary_en": dam_summary,
                    "asr_transcript_vi": asr_text,
                    "ocr_text": ocr_text,
                    "has_speech": k_n in asr_by_kf,
                    "num_objects": len(regions),
                }

                kf_points_buffer.append(
                    PointStruct(
                        id=point_id_counter,
                        vector={"visual": vis_vec, "speech": speech_vec},
                        payload=payload,
                    )
                )
                point_id_counter += 1
                total_kf_points += 1

                # 2. Prepare DAM Object Points
                if regions:
                    captions = [r.get("caption", {}).get("description_en", "") for r in regions]
                    dam_embeds = self.models.encode_bge_m3(captions)

                    for reg_idx, (reg, d_vec) in enumerate(zip(regions, dam_embeds)):
                        bbox = reg.get("box_2d", [0, 0, 1000, 1000])
                        # Normalize bbox to [0, 1]
                        norm_bbox = [b / 1000.0 for b in bbox]

                        dam_payload = {
                            "video_id": video_id,
                            "keyframe_n": k_n,
                            "frame_idx": f_idx,
                            "region_id": reg.get("region_id", reg_idx + 1),
                            "class_entity": reg.get("detector", {}).get("class_entity", "Object"),
                            "description_en": reg.get("caption", {}).get("description_en", ""),
                            "bbox": norm_bbox,
                        }

                        dam_points_buffer.append(
                            PointStruct(
                                id=dam_point_id_counter,
                                vector=d_vec.tolist(),
                                payload=dam_payload,
                            )
                        )
                        dam_point_id_counter += 1
                        total_dam_points += 1

            # Flush buffers if large
            if len(kf_points_buffer) >= batch_size:
                self.client.upsert(collection_name=self.KEYFRAME_COLLECTION, points=kf_points_buffer)
                kf_points_buffer.clear()

            if len(dam_points_buffer) >= batch_size:
                self.client.upsert(collection_name=self.DAM_COLLECTION, points=dam_points_buffer)
                dam_points_buffer.clear()

        # Final flush
        if kf_points_buffer:
            self.client.upsert(collection_name=self.KEYFRAME_COLLECTION, points=kf_points_buffer)
        if dam_points_buffer:
            self.client.upsert(collection_name=self.DAM_COLLECTION, points=dam_points_buffer)

        logger.info(f"✅ Qdrant Indexing Complete! Stored {total_kf_points} keyframes and {total_dam_points} objects.")
        return {"keyframes": total_kf_points, "dam_objects": total_dam_points}
