#!/usr/bin/env python3
"""Run Unified Multi-Modal Pipeline Experiment: TransNetV2 -> Adaptive Keyframes -> SigLIP-2 -> OCR -> SAM -> DAM-3B."""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aic2026.unified import UnifiedFrameRecord, UnifiedVideoPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified Multi-Modal Pipeline Experiment")
    parser.add_argument("--video-id", type=str, default="L21_V003", help="Target video ID")
    parser.add_argument("--video-path", type=Path, default=None, help="Direct path to video .mp4 file")
    parser.add_argument("--video-root", type=Path, default=None, help="Root directory containing video files")
    parser.add_argument("--objects-root", type=Path, default=None, help="Root directory containing object detection JSONs")
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/multimodal_results"), help="Output directory")
    parser.add_argument("--max-frames", type=int, default=5, help="Max frames to extract and test")
    parser.add_argument("--score-threshold", type=float, default=0.30, help="Object detection score threshold")
    parser.add_argument("--max-regions", type=int, default=3, help="Max regions per frame")
    parser.add_argument("--max-words", type=int, default=50, help="Max words for DAM captions")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cuda/cpu)")
    parser.add_argument("--no-dam", action="store_true", help="Disable SAM/DAM stage")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR stage")
    parser.add_argument("--no-siglip", action="store_true", help="Disable SigLIP-2 stage")
    return parser


def find_video_file(video_id: str, direct_path: Path | None, video_root: Path | None) -> Path:
    if direct_path and direct_path.is_file():
        return direct_path

    roots_to_check = [
        video_root,
        Path("/kaggle/input"),
        Path("/kaggle/input/datasets/lyduchoang/aic-26-video/Videos"),
        REPO_ROOT / "data" / "videos",
    ]
    for root in roots_to_check:
        if root is None or not root.exists():
            continue
        candidates = [
            root / f"{video_id}.mp4",
            root / video_id.split("_")[0] / f"{video_id}.mp4",
            root / f"Videos_{video_id.split('_')[0]}" / f"{video_id}.mp4",
            root / "Videos" / f"Videos_{video_id.split('_')[0]}" / f"{video_id}.mp4",
        ]
        for cand in candidates:
            if cand.is_file():
                return cand
        for match in root.rglob(f"{video_id}.mp4"):
            if match.is_file():
                return match

    raise FileNotFoundError(f"Video file for {video_id} not found. Please provide --video-path")


def render_multimodal_card(
    record: UnifiedFrameRecord,
    image_path: Path,
    output_card_path: Path,
) -> None:
    """Render a comprehensive visual comparison card for a single frame."""
    with Image.open(image_path) as pil_img:
        orig_img = pil_img.convert("RGB")

    # Draw OCR polygons overlay
    ocr_overlay = orig_img.copy()
    draw = ImageDraw.Draw(ocr_overlay)
    for span in record.ocr.spans:
        if len(span.polygon_xy) >= 3:
            draw.polygon(span.polygon_xy, outline="lime", width=3)
            first_pt = span.polygon_xy[0]
            draw.text((first_pt[0] + 2, first_pt[1] + 2), span.normalized_text[:20], fill="yellow")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"wspace": 0.05})

    # Left: Original + OCR
    axes[0].imshow(ocr_overlay)
    axes[0].set_title(
        f"Frame #{record.keyframe_n} (idx:{record.frame_idx} @ {record.pts_time_s:.2f}s)\n"
        f"OCR: \"{record.ocr.full_text[:45]}{'...' if len(record.ocr.full_text)>45 else ''}\"",
        fontsize=11,
        weight="bold",
    )
    axes[0].axis("off")

    # Right: Summary & Captions Text
    axes[1].imshow(orig_img)
    caption_lines = []
    if record.dam_descriptions:
        for idx, cap in enumerate(record.dam_descriptions, start=1):
            caption_lines.append(f"[{idx}] {cap.class_label} (IoU {cap.sam_iou:.2f}):\n    \"{cap.caption_en}\" ({cap.word_count} words)")
    else:
        caption_lines.append("<No objects detected or segmented>")

    full_caption_text = "\n\n".join(caption_lines)
    axes[1].set_title(
        f"SigLIP-2: 768-dim vector (L2 norm: 1.00)\nDAM Dense Captions ({len(record.dam_descriptions)} regions)",
        fontsize=11,
        weight="bold",
    )
    axes[1].axis("off")

    plt.suptitle(
        f"Unified Multi-Modal Extraction: {record.frame_uid}",
        fontsize=14,
        weight="bold",
        y=0.98,
    )
    plt.tight_layout()
    output_card_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_card_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    video_path = find_video_file(args.video_id, args.video_path, args.video_root)
    print(f"🎬 Target Video: {video_path}")

    device = args.device
    pipeline = UnifiedVideoPipeline.load(
        device=device,
        load_transnet=True,
        load_siglip=not args.no_siglip,
        load_ocr=not args.no_ocr,
        load_sam_dam=not args.no_dam,
    )

    objects_dir = None
    if args.objects_root:
        cand_obj = args.objects_root / args.video_id
        if cand_obj.exists():
            objects_dir = cand_obj

    map_csv_path, unified_jsonl_path, records = pipeline.process_video(
        video_path=video_path,
        video_id=args.video_id,
        output_root=args.output_dir,
        objects_dir=objects_dir,
        max_frames=args.max_frames,
        max_regions_per_frame=args.max_regions,
        maximum_words=args.max_words,
        score_threshold=args.score_threshold,
    )

    print("🎨 Rendering visual inspection cards for generated frames...")
    vis_dir = args.output_dir / "visualizations" / args.video_id
    vis_dir.mkdir(parents=True, exist_ok=True)

    for rec in records:
        img_path = args.output_dir / rec.image_relpath
        if img_path.exists():
            card_p = vis_dir / f"{args.video_id}_f{rec.keyframe_n:04d}_multimodal.jpg"
            render_multimodal_card(rec, img_path, card_p)
            print(f"  ✓ Saved visual card: {card_p.name}")

    print("\n" + "=" * 75)
    print(f"🎉 SUCCESS! Unified Pipeline Experiment for {args.video_id} is complete!")
    print(f"  • Map-Keyframes CSV: {map_csv_path}")
    print(f"  • Unified JSONL:     {unified_jsonl_path}")
    print(f"  • Visual Cards:      {vis_dir}")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
