"""High-Throughput Local Ingestion into Embedded Qdrant Database.

Ingests the unified dataset into 2 specialized collections:
1. `keyframes`: 177,321 points with multi-vectors ("visual": 768-d, "speech": 1024-d)
2. `dam_objects`: 435,713 points with 1024-d BGE-M3 object vectors + bboxes

Target Directory: `/Users/khoale/Downloads/AIC_HCM/qdrant_db`
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Generator

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def chunk_list(items: list[Any], chunk_size: int = 5000) -> Generator[list[Any], None, None]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


class LocalQdrantIngester:
    KEYFRAMES_COLLECTION = "keyframes"
    DAM_COLLECTION = "dam_objects"

    def __init__(self, qdrant_db_path: Path):
        self.db_path = qdrant_db_path
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.db_path))
        logger.info(f"Connected to Qdrant at: {self.db_path}")

    def init_collections(self, force_recreate: bool = True):
        existing = [c.name for c in self.client.get_collections().collections]

        # 1. Keyframes Collection (Multi-vector: visual 768-d + speech 1024-d)
        if self.KEYFRAMES_COLLECTION in existing and force_recreate:
            logger.info(f"Recreating collection '{self.KEYFRAMES_COLLECTION}'...")
            self.client.delete_collection(self.KEYFRAMES_COLLECTION)

        if self.KEYFRAMES_COLLECTION not in existing or force_recreate:
            self.client.create_collection(
                collection_name=self.KEYFRAMES_COLLECTION,
                vectors_config={
                    "visual": VectorParams(size=768, distance=Distance.COSINE),
                    "speech": VectorParams(size=1024, distance=Distance.COSINE),
                },
            )
            logger.info(f"✅ Created collection '{self.KEYFRAMES_COLLECTION}' (Named vectors: visual=768-d, speech=1024-d)")

        # 2. DAM Objects Collection (1024-d BGE-M3)
        if self.DAM_COLLECTION in existing and force_recreate:
            logger.info(f"Recreating collection '{self.DAM_COLLECTION}'...")
            self.client.delete_collection(self.DAM_COLLECTION)

        if self.DAM_COLLECTION not in existing or force_recreate:
            self.client.create_collection(
                collection_name=self.DAM_COLLECTION,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            logger.info(f"✅ Created collection '{self.DAM_COLLECTION}' (Vector: 1024-d Cosine)")

    def ingest_keyframes(self, unified_dir: Path, batch_size: int = 5000) -> int:
        vis_npy = unified_dir / "keyframes_visual_vectors.f16.npy"
        speech_npy = unified_dir / "keyframes_speech_vectors.f16.npy"
        meta_jsonl = unified_dir / "keyframes_metadata.jsonl"

        logger.info(f"\n🖼️ Ingesting Keyframes from {unified_dir}...")
        vis_mmap = np.load(vis_npy, mmap_mode="r")
        speech_mmap = np.load(speech_npy, mmap_mode="r")
        total_kf = len(vis_mmap)
        logger.info(f"  • Total Keyframes to ingest: {total_kf:,}")

        # Read metadata
        meta_records = []
        with open(meta_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                meta_records.append(json.loads(line))

        points_buffer = []
        total_uploaded = 0

        t0 = time.time()
        for i in tqdm(range(total_kf), desc="Preparing & Ingesting Keyframes", unit="frame"):
            point_id = meta_records[i]["point_id"]
            vis_vec = vis_mmap[i].astype(np.float32).tolist()
            speech_vec = speech_mmap[i].astype(np.float32).tolist()
            payload = meta_records[i]

            points_buffer.append(
                PointStruct(
                    id=point_id,
                    vector={"visual": vis_vec, "speech": speech_vec},
                    payload=payload,
                )
            )

            if len(points_buffer) >= batch_size:
                self.client.upload_points(
                    collection_name=self.KEYFRAMES_COLLECTION,
                    points=points_buffer,
                    wait=True,
                )
                total_uploaded += len(points_buffer)
                points_buffer.clear()

        if points_buffer:
            self.client.upload_points(
                collection_name=self.KEYFRAMES_COLLECTION,
                points=points_buffer,
                wait=True,
            )
            total_uploaded += len(points_buffer)

        duration = time.time() - t0
        logger.info(f"  ✅ Keyframes Ingested: {total_uploaded:,} points in {duration:.1f}s ({total_uploaded/max(duration,0.01):.0f} pts/sec)")
        return total_uploaded

    def ingest_dam_objects(self, unified_dir: Path, batch_size: int = 5000) -> int:
        dam_npy = unified_dir / "dam_vectors.f16.npy"
        dam_meta = unified_dir / "dam_metadata.jsonl"

        logger.info(f"\n🔍 Ingesting DAM Objects from {unified_dir}...")
        dam_mmap = np.load(dam_npy, mmap_mode="r")
        total_dam = len(dam_mmap)
        logger.info(f"  • Total DAM Objects to ingest: {total_dam:,}")

        # Read metadata
        meta_records = []
        with open(dam_meta, "r", encoding="utf-8") as f:
            for line in f:
                meta_records.append(json.loads(line))

        points_buffer = []
        total_uploaded = 0

        t0 = time.time()
        for i in tqdm(range(total_dam), desc="Preparing & Ingesting DAM Objects", unit="object"):
            reg_meta = meta_records[i]
            point_id = abs(hash(f"{reg_meta['video_id']}_{reg_meta['frame_idx']}_{reg_meta['region_id']}")) % (10**16)
            vec = dam_mmap[i].astype(np.float32).tolist()

            points_buffer.append(
                PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=reg_meta,
                )
            )

            if len(points_buffer) >= batch_size:
                self.client.upload_points(
                    collection_name=self.DAM_COLLECTION,
                    points=points_buffer,
                    wait=True,
                )
                total_uploaded += len(points_buffer)
                points_buffer.clear()

        if points_buffer:
            self.client.upload_points(
                collection_name=self.DAM_COLLECTION,
                points=points_buffer,
                wait=True,
            )
            total_uploaded += len(points_buffer)

        duration = time.time() - t0
        logger.info(f"  ✅ DAM Objects Ingested: {total_uploaded:,} points in {duration:.1f}s ({total_uploaded/max(duration,0.01):.0f} pts/sec)")
        return total_uploaded


def main():
    parser = argparse.ArgumentParser(description="Ingest Unified Multimodal Dataset into Qdrant")
    parser.add_argument(
        "--unified-dir",
        type=str,
        default="/Users/khoale/Downloads/AIC_HCM/unified_index",
        help="Path to unified index directory",
    )
    parser.add_argument(
        "--qdrant-db",
        type=str,
        default="/Users/khoale/Downloads/AIC_HCM/qdrant_db",
        help="Target Qdrant database directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size for point uploads",
    )

    args = parser.parse_args()

    unified_dir = Path(args.unified_dir)
    qdrant_db = Path(args.qdrant_db)

    logger.info("=" * 80)
    logger.info("🚀 HIGH-THROUGHPUT QDRANT LOCAL INGESTION")
    logger.info(f"  • Source Directory: {unified_dir}")
    logger.info(f"  • Qdrant DB Path:   {qdrant_db}")
    logger.info(f"  • Batch Size:       {args.batch_size}")
    logger.info("=" * 80)

    t_start = time.time()
    ingester = LocalQdrantIngester(qdrant_db_path=qdrant_db)
    ingester.init_collections(force_recreate=True)

    # 1. Ingest Keyframes
    kf_count = ingester.ingest_keyframes(unified_dir=unified_dir, batch_size=args.batch_size)

    # 2. Ingest DAM Objects
    dam_count = ingester.ingest_dam_objects(unified_dir=unified_dir, batch_size=args.batch_size)

    total_time = time.time() - t_start
    logger.info("\n" + "=" * 80)
    logger.info("🎉 QDRANT INGESTION COMPLETE!")
    logger.info(f"  • Collection 'keyframes':   {kf_count:,} points stored")
    logger.info(f"  • Collection 'dam_objects': {dam_count:,} points stored")
    logger.info(f"  • Total Ingestion Time:     {total_time:.1f}s ({total_time / 60:.1f} min)")
    logger.info(f"  • Database Location:        {qdrant_db}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
