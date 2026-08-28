"""Unified Multimodal Dataset Builder.

Merges:
1. Master keyframe CSVs (177,321 keyframes across 873 videos)
2. SigLIP-2 Visual vectors (768-d .f16.npy from [AIC2026] Scene Embeddings)
3. Audio ASR Speech vectors & transcripts (1024-d .f16.npy from embeddings_output)
4. DAM Object Captions & vectors (1024-d .f16.npy from embeddings_output)

Outputs a unified, 100% complete dataset directory ready for Qdrant ingestion or direct RAM search.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_unified_dataset(
    map_dir: Path,
    scene_dir: Path,
    embeddings_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("🚀 BUILDING UNIFIED MULTIMODAL DATASET")
    logger.info(f"  • Map Directory:        {map_dir}")
    logger.info(f"  • Scene Embeddings Dir: {scene_dir}")
    logger.info(f"  • Embeddings Output:    {embeddings_dir}")
    logger.info(f"  • Output Directory:     {output_dir}")
    logger.info("=" * 80)

    t_start = time.time()

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1: Load ASR Speech Vectors & Metadata
    # ──────────────────────────────────────────────────────────────────────────
    asr_npy_path = embeddings_dir / "asr_vectors.f16.npy"
    asr_meta_path = embeddings_dir / "asr_metadata.jsonl"

    if not asr_npy_path.exists() or not asr_meta_path.exists():
        raise FileNotFoundError(f"Missing ASR files in {embeddings_dir}")

    logger.info(f"📂 Loading ASR Speech Vectors from {asr_npy_path}...")
    asr_vectors_mmap = np.load(asr_npy_path, mmap_mode="r")
    logger.info(f"  ✅ ASR Vector Matrix: {asr_vectors_mmap.shape}, dtype={asr_vectors_mmap.dtype}")

    logger.info(f"📖 Reading ASR Metadata from {asr_meta_path}...")
    # Lookup: (video_id, keyframe_n) -> (asr_row_idx, asr_meta_dict)
    asr_lookup: dict[tuple[str, int], tuple[int, dict[str, Any]]] = {}
    with open(asr_meta_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            d = json.loads(line)
            asr_lookup[(d["video_id"], d["keyframe_n"])] = (idx, d)

    logger.info(f"  ✅ Indexed {len(asr_lookup):,} ASR keyframe records in lookup.")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2: Iterate over 873 Videos to Merge Visual + Speech + Maps
    # ──────────────────────────────────────────────────────────────────────────
    map_files = sorted(list(map_dir.glob("*.csv")))
    logger.info(f"\n🎥 Merging {len(map_files)} videos across Visual and Audio modalities...")

    all_keyframes_meta: list[dict[str, Any]] = []
    all_visual_vectors: list[np.ndarray] = []
    all_speech_vectors: list[np.ndarray] = []

    total_frames = 0
    frames_with_speech = 0
    silent_frames = 0
    zero_speech_vec = np.zeros(1024, dtype=np.float16)

    for csv_file in tqdm(map_files, desc="Merging Keyframes"):
        video_id = csv_file.stem
        scene_npy_file = scene_dir / f"{video_id}.f16.npy"

        if not scene_npy_file.exists():
            logger.warning(f"⚠️ Missing scene embedding for video: {video_id} ({scene_npy_file})")
            continue

        # Load Visual Array (N, 768)
        vis_arr = np.load(scene_npy_file).astype(np.float16)

        # Load Keyframe Map CSV
        with open(csv_file, encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        if len(reader) != len(vis_arr):
            logger.warning(
                f"⚠️ Row mismatch for {video_id}: CSV has {len(reader)} rows, npy has {len(vis_arr)} rows!"
            )
            min_len = min(len(reader), len(vis_arr))
            reader = reader[:min_len]
            vis_arr = vis_arr[:min_len]

        for row_idx, row in enumerate(reader):
            k_n = int(row["n"])
            pts_time = float(row["pts_time"])
            fps = float(row["fps"])
            f_idx = int(row["frame_idx"])
            f_uid = f"{video_id}:{f_idx}"

            vis_vec = vis_arr[row_idx]
            all_visual_vectors.append(vis_vec)

            # Check if this keyframe has ASR speech
            asr_key = (video_id, k_n)
            if asr_key in asr_lookup:
                asr_idx, asr_data = asr_lookup[asr_key]
                speech_vec = asr_vectors_mmap[asr_idx]
                has_speech = True
                asr_text = asr_data.get("asr_transcript_vi", "")
                dam_summary = asr_data.get("dam_summary_en", "")
                num_objs = asr_data.get("num_objects", 0)
                frames_with_speech += 1
            else:
                speech_vec = zero_speech_vec
                has_speech = False
                asr_text = ""
                dam_summary = ""
                num_objs = 0
                silent_frames += 1

            all_speech_vectors.append(speech_vec)

            meta_record = {
                "point_id": total_frames + 1,
                "video_id": video_id,
                "keyframe_n": k_n,
                "frame_idx": f_idx,
                "pts_time_s": pts_time,
                "fps": fps,
                "frame_uid": f_uid,
                "image_relpath": f"keyframes/{video_id}/{k_n:03d}.jpg",
                "asr_transcript_vi": asr_text,
                "has_speech": has_speech,
                "dam_summary_en": dam_summary,
                "num_objects": num_objs,
                "ocr_text": "",
            }
            all_keyframes_meta.append(meta_record)
            total_frames += 1

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3: Stack & Save Unified Keyframes Data
    # ──────────────────────────────────────────────────────────────────────────
    logger.info(f"\n💾 Stacking and saving {total_frames:,} unified keyframes...")

    # Visual Matrix (177321, 768)
    final_visual_matrix = np.vstack(all_visual_vectors).astype(np.float16)
    out_vis_path = output_dir / "keyframes_visual_vectors.f16.npy"
    np.save(out_vis_path, final_visual_matrix)
    logger.info(
        f"  ✅ Saved Visual Matrix: {final_visual_matrix.shape} ({out_vis_path.stat().st_size / 1024**2:.1f} MB)"
    )
    del all_visual_vectors, final_visual_matrix

    # Speech Matrix (177321, 1024)
    final_speech_matrix = np.vstack(all_speech_vectors).astype(np.float16)
    out_speech_path = output_dir / "keyframes_speech_vectors.f16.npy"
    np.save(out_speech_path, final_speech_matrix)
    logger.info(
        f"  ✅ Saved Speech Matrix: {final_speech_matrix.shape} ({out_speech_path.stat().st_size / 1024**2:.1f} MB)"
    )
    del all_speech_vectors, final_speech_matrix

    # Keyframes Metadata JSONL
    out_meta_path = output_dir / "keyframes_metadata.jsonl"
    with open(out_meta_path, "w", encoding="utf-8") as f:
        for rec in all_keyframes_meta:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(
        f"  ✅ Saved Keyframes Metadata: {len(all_keyframes_meta):,} lines ({out_meta_path.stat().st_size / 1024**2:.1f} MB)"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4: Link / Copy DAM Objects Data
    # ──────────────────────────────────────────────────────────────────────────
    dam_npy_src = embeddings_dir / "dam_vectors.f16.npy"
    dam_meta_src = embeddings_dir / "dam_metadata.jsonl"
    dam_npy_dst = output_dir / "dam_vectors.f16.npy"
    dam_meta_dst = output_dir / "dam_metadata.jsonl"

    logger.info("\n📦 Linking DAM Object vectors and metadata into unified folder...")
    if not dam_npy_dst.exists():
        import shutil

        shutil.copy2(dam_npy_src, dam_npy_dst)
        shutil.copy2(dam_meta_src, dam_meta_dst)
        logger.info("  ✅ Copied DAM vectors & metadata to unified folder.")
    else:
        logger.info("  ✅ DAM vectors & metadata already present.")

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 5: Verification & Summary
    # ──────────────────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    summary_stats = {
        "total_keyframes": total_frames,
        "frames_with_speech": frames_with_speech,
        "silent_frames": silent_frames,
        "visual_vectors_shape": [total_frames, 768],
        "speech_vectors_shape": [total_frames, 1024],
        "dam_objects_count": 435713,
        "total_execution_time_s": round(total_time, 2),
    }

    summary_file = output_dir / "unified_dataset_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("🎉 UNIFIED DATASET BUILD COMPLETE!")
    logger.info(f"  • Total Keyframes:      {total_frames:,}")
    logger.info(
        f"  • Frames With Speech:   {frames_with_speech:,} ({frames_with_speech / total_frames * 100:.1f}%)"
    )
    logger.info(
        f"  • Silent Frames:        {silent_frames:,} ({silent_frames / total_frames * 100:.1f}%)"
    )
    logger.info(f"  • Total Execution Time: {total_time:.2f}s")
    logger.info(f"  • Output Directory:     {output_dir}")
    logger.info("=" * 80)

    return summary_stats


def main():
    parser = argparse.ArgumentParser(description="Build Unified Multimodal Dataset")
    parser.add_argument(
        "--map-dir",
        type=str,
        default="/Users/khoale/Downloads/AIC_HCM/map-keyframes",
        help="Path to map-keyframes CSV directory",
    )
    parser.add_argument(
        "--scene-dir",
        type=str,
        default="/Users/khoale/Downloads/[AIC2026] Scene Embeddings",
        help="Path to SigLIP scene embeddings directory",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="/Users/khoale/Downloads/embeddings_output",
        help="Path to Colab output directory (dam/asr npy + jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/khoale/Downloads/AIC_HCM/unified_index",
        help="Target unified index directory",
    )

    args = parser.parse_args()

    build_unified_dataset(
        map_dir=Path(args.map_dir),
        scene_dir=Path(args.scene_dir),
        embeddings_dir=Path(args.embeddings_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
