#!/usr/bin/env python3
"""High-speed Ingestion Script: Merge OCR Shards into Unified Index Metadata.

Reads all OCR JSONL shards (e.g. shard-000002.jsonl ... shard-000008.jsonl)
and populates 'ocr_text' in keyframes_metadata.jsonl in < 5 seconds.
Can be re-run at any time as new shards arrive.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_ocr")


def ingest_ocr(
    ocr_dir: str | Path = "/Users/khoale/Downloads/OCR",
    unified_index_dir: str | Path = "/Users/khoale/Downloads/AIC_HCM/unified_index",
    backup: bool = True,
) -> None:
    ocr_path = Path(str(ocr_dir).strip().strip('"').strip("'")).expanduser().resolve()
    idx_path = Path(str(unified_index_dir).strip().strip('"').strip("'")).expanduser().resolve()

    metadata_file = idx_path / "keyframes_metadata.jsonl"
    if not metadata_file.exists():
        raise FileNotFoundError(f"keyframes_metadata.jsonl not found at {metadata_file}")

    if not ocr_path.exists():
        raise FileNotFoundError(f"OCR directory not found at {ocr_path}")

    # 1. Discover all OCR shards
    shard_files = sorted(glob.glob(str(ocr_path / "shard-*.jsonl")))
    if not shard_files:
        shard_files = sorted(glob.glob(str(ocr_path / "*.jsonl")))

    logger.info(f"Found {len(shard_files)} OCR shard files in {ocr_path}")

    # 2. Build In-Memory (video_id, frame_idx) -> full_text map
    t0 = time.perf_counter()
    ocr_map: dict[tuple[str, int], str] = {}

    for sf in shard_files:
        sf_name = os.path.basename(sf)
        sf_count = 0
        with open(sf, encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    v_id = data.get("video_id")
                    f_idx = data.get("frame_idx")
                    full_text = data.get("full_text", "").strip()

                    if v_id and f_idx is not None and full_text:
                        ocr_map[(v_id, int(f_idx))] = full_text
                        sf_count += 1
                except Exception:
                    continue
        logger.info(f"  Loaded {sf_count:,} text records from {sf_name}")

    t1 = time.perf_counter()
    logger.info(f"⚡ Ingested {len(ocr_map):,} total OCR frame texts in {(t1 - t0) * 1000:.1f}ms")

    # 3. Create Backup if requested
    if backup:
        backup_file = idx_path / "keyframes_metadata.jsonl.bak"
        if not backup_file.exists():
            logger.info(f"Creating backup at {backup_file}...")
            shutil.copyfile(metadata_file, backup_file)

    # 4. Stream & Merge into Temporary File
    temp_file = idx_path / "keyframes_metadata.jsonl.tmp"
    total_kf = 0
    matched_kf = 0

    t2 = time.perf_counter()
    with (
        open(metadata_file, encoding="utf-8") as fin,
        open(temp_file, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            total_kf += 1
            meta = json.loads(line)
            v_id = meta.get("video_id")
            f_idx = meta.get("frame_idx")

            key = (v_id, int(f_idx)) if v_id and f_idx is not None else None
            if key and key in ocr_map:
                meta["ocr_text"] = ocr_map[key]
                matched_kf += 1
            else:
                meta["ocr_text"] = meta.get("ocr_text", "")

            fout.write(json.dumps(meta, ensure_ascii=False) + "\n")

    # 5. Atomic Replace
    os.replace(temp_file, metadata_file)
    t3 = time.perf_counter()

    logger.info(
        f"✅ Merge complete in {(t3 - t2) * 1000:.1f}ms!\n"
        f"   Total Keyframes in Index: {total_kf:,}\n"
        f"   Keyframes with OCR Text: {matched_kf:,} ({matched_kf / max(1, total_kf) * 100:.1f}%)\n"
        f"   Updated File: {metadata_file}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Ingest OCR shards into unified keyframes metadata"
    )
    parser.add_argument(
        "--ocr-dir",
        default="/Users/khoale/Downloads/OCR",
        help="Directory containing OCR shard-*.jsonl",
    )
    parser.add_argument(
        "--unified-index-dir",
        default="/Users/khoale/Downloads/AIC_HCM/unified_index",
        help="Directory containing keyframes_metadata.jsonl",
    )
    args = parser.parse_args()

    ingest_ocr(ocr_dir=args.ocr_dir, unified_index_dir=args.unified_index_dir)


if __name__ == "__main__":
    main()
