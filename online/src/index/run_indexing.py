"""Command-line script to run one-time Qdrant indexing for the AIC dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import yaml

from online.src.index.qdrant_indexer import QdrantIndexer
from online.src.retrieval.embeddings import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Index AIC-2026 Multimodal Dataset into Qdrant")
    parser.add_argument("--config", type=str, default="online/configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to index (for testing)")
    parser.add_argument("--videos", nargs="+", default=None, help="Specific video IDs to index (e.g. L21_V001 L21_V002)")
    parser.add_argument("--force-recreate", action="store_true", help="Delete and recreate Qdrant collections")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    map_dir = Path(paths["map_keyframes_dir"])
    scene_dir = Path(paths["scene_embeddings_dir"])
    dam_dir = Path(paths["dam_descriptions_dir"])
    asr_dir = Path(paths["asr_segments_dir"])
    ocr_dir = Path(paths["ocr_transcripts_dir"])
    qdrant_db_path = Path(paths["qdrant_db_path"])

    models = ModelRegistry(
        bge_model_id=config["models"]["bge_m3_model_id"],
        siglip_model_id=config["models"]["siglip_model_id"],
        reranker_model_id=config["models"]["bge_reranker_model_id"],
    )

    indexer = QdrantIndexer(qdrant_db_path=str(qdrant_db_path), models=models)
    indexer.init_collections(force_recreate=args.force_recreate)

    # Determine video list
    if args.videos:
        video_ids = args.videos
    else:
        video_ids = sorted([f.stem for f in map_dir.glob("*.csv")])
        if args.limit:
            video_ids = video_ids[: args.limit]

    logger.info(f"Target videos to index: {len(video_ids)}")
    stats = indexer.index_all_videos(
        map_dir=map_dir,
        scene_dir=scene_dir,
        dam_dir=dam_dir,
        asr_dir=asr_dir,
        ocr_dir=ocr_dir if ocr_dir.exists() else None,
        video_ids=video_ids,
    )

    logger.info(f"Indexing Complete: {stats}")


if __name__ == "__main__":
    main()
