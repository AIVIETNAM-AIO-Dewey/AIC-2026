#!/usr/bin/env python3
"""Stage 4: Multi-Modal Frame Metadata Fusion.

Joins Frame Manifest, YOLO+DAM Visual Descriptions, and OCR On-Screen Text
into a clean, unified, search-ready JSONL artifact per video.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aic2026.common.io import iter_jsonl, write_jsonl_atomic  # noqa: E402
from aic2026.contracts import FrameRef  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True, help="Video identifier (e.g. L21_V001)")
    parser.add_argument("--frame-manifest", type=Path, required=True, help="Path to frame manifest JSONL")
    parser.add_argument("--descriptions", type=Path, help="Path to DAM descriptions JSONL")
    parser.add_argument("--ocr-transcripts", type=Path, help="Path to OCR transcripts JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Path to output unified metadata JSONL")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_id = args.video_id
    manifest_path = args.frame_manifest.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frame manifest not found: {manifest_path}")

    # 1. Load Base Frame Manifest
    frames = [FrameRef.model_validate(val) for val in iter_jsonl(manifest_path)]

    # 2. Load Visual Descriptions (if available)
    desc_by_uid: dict[str, dict] = {}
    if args.descriptions and args.descriptions.expanduser().resolve().is_file():
        for rec in iter_jsonl(args.descriptions.expanduser().resolve()):
            uid = rec.get("frame_uid")
            if uid:
                desc_by_uid[uid] = rec

    # 3. Load OCR Transcripts (if available)
    ocr_by_uid: dict[str, dict] = {}
    if args.ocr_transcripts and args.ocr_transcripts.expanduser().resolve().is_file():
        for rec in iter_jsonl(args.ocr_transcripts.expanduser().resolve()):
            uid = rec.get("frame_uid")
            if uid:
                ocr_by_uid[uid] = rec

    # 4. Merge into Search-Ready Unified Records
    unified_records: list[dict] = []
    for idx, frame in enumerate(frames, start=1):
        uid = frame.frame_uid
        desc_entry = desc_by_uid.get(uid, {})
        ocr_entry = ocr_by_uid.get(uid, {})

        # Format visual descriptions
        raw_regions = desc_entry.get("regions", [])
        dam_regions: list[dict] = []
        dam_summary_parts: list[str] = []

        for r in raw_regions:
            entity = r.get("detector", {}).get("class_entity", "object")
            score = r.get("detector", {}).get("score", 0.0)
            bbox = r.get("bbox_xyxy_px")
            caption = r.get("caption", {}).get("description_en", "")
            if caption:
                dam_summary_parts.append(caption)
            dam_regions.append(
                {
                    "entity": entity,
                    "score": round(float(score), 4),
                    "bbox_xyxy_px": bbox,
                    "description_en": caption,
                }
            )

        dam_summary_en = " ".join(dam_summary_parts).strip()
        ocr_text = ocr_entry.get("full_text", "").strip()
        ocr_spans = ocr_entry.get("spans", [])

        unified_records.append(
            {
                "point_id": idx,
                "video_id": frame.video_id,
                "keyframe_n": frame.keyframe_n,
                "frame_idx": frame.frame_idx,
                "pts_time_s": round(float(frame.pts_time_s), 4),
                "fps": round(float(frame.fps), 2),
                "frame_uid": frame.frame_uid,
                "image_relpath": frame.frame_relpath,
                "width": frame.width,
                "height": frame.height,
                "dam_summary_en": dam_summary_en,
                "num_objects": len(dam_regions),
                "dam_regions": dam_regions,
                "ocr_text": ocr_text,
                "ocr_spans": ocr_spans,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_path, unified_records)

    print(
        json.dumps(
            {
                "status": "completed",
                "video_id": video_id,
                "frames": len(unified_records),
                "output": str(output_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
