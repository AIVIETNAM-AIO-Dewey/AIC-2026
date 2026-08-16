#!/usr/bin/env python3
"""Inspect DAM description results, draw bounding box overlays, and format visual output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aic2026.common import iter_jsonl  # noqa: E402
from aic2026.contracts import ObjectFrameRecord  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True, help="Path to description JSONL artifact")
    parser.add_argument("--data-root", type=Path, required=True, help="Data root containing keyframes/")
    parser.add_argument("--output-dir", type=Path, help="Directory to save visual output images")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of frames to process")
    return parser


def draw_region_overlay(
    image: Image.Image,
    bbox_xyxy: tuple[int, int, int, int],
    label: str,
    score: float,
    caption: str,
) -> Image.Image:
    """Draw a high-contrast bounding box and caption overlay onto the frame image."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    x1, y1, x2, y2 = bbox_xyxy
    
    # Draw thick bounding box
    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
    
    # Draw label badge
    header_text = f"{label} ({score:.2f})"
    draw.rectangle([x1, max(0, y1 - 20), x1 + len(header_text) * 8 + 10, y1], fill="red")
    draw.text((x1 + 5, max(0, y1 - 18)), header_text, fill="white")
    
    return canvas


def inspect_descriptions(
    artifact_path: Path,
    data_root: Path,
    output_dir: Path | None = None,
    limit: int | None = 5,
) -> list[dict[str, Any]]:
    """Process description records and print/save visual inspection cards."""
    records = list(iter_jsonl(artifact_path))
    if limit is not None:
        records = records[:limit]

    results = []
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f" DAM DESCRIPTION INSPECTION REPORT | File: {artifact_path.name}")
    print("=" * 80)

    for frame_idx, raw in enumerate(records, start=1):
        record = ObjectFrameRecord.model_validate(raw)
        image_path = (data_root / record.frame_relpath).resolve()
        
        print(f"\n📸 [Frame {frame_idx}/{len(records)}] UID: {record.frame_uid} | Path: {record.frame_relpath}")
        print("-" * 80)
        
        if not image_path.exists():
            print(f"   ⚠️ Image not found: {image_path}")
            continue

        with Image.open(image_path) as source_img:
            img = source_img.convert("RGB")
            
            for region_idx, region in enumerate(record.regions, start=1):
                detector = region.detector
                bbox = region.bbox_xyxy_px
                caption_info = region.caption
                desc = caption_info.description_en or f"[{caption_info.status}: {caption_info.error}]"
                
                print(f"   Region {region_idx}:")
                print(f"     • Label: {detector.class_name} ({detector.class_entity}) | Confidence: {detector.score:.4f}")
                print(f"     • Box (XYXY): [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
                print(f"     • Caption Status: {caption_info.status} | Words: {caption_info.word_count}")
                print(f"     • 📝 DAM Caption: \"{desc}\"")
                print()

                # Generate image overlay
                overlay = draw_region_overlay(
                    img,
                    bbox_xyxy=bbox,
                    label=detector.class_entity,
                    score=detector.score,
                    caption=desc,
                )

                if output_dir:
                    out_filename = f"{record.video_id}_frame{record.keyframe_n}_reg{region_idx}.jpg"
                    overlay.save(output_dir / out_filename)
                
                results.append({
                    "frame_uid": record.frame_uid,
                    "keyframe_n": record.keyframe_n,
                    "region_id": region.region_id,
                    "label": detector.class_entity,
                    "score": detector.score,
                    "bbox": bbox,
                    "caption": desc,
                    "overlay_img": overlay,
                })

    if output_dir:
        print(f"✅ Saved visual inspection images to: {output_dir}")

    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inspect_descriptions(
        artifact_path=args.artifact.expanduser().resolve(),
        data_root=args.data_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve() if args.output_dir else None,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
