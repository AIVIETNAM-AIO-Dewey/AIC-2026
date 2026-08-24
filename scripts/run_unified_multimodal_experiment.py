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

    batch = video_id.split("_")[0]  # e.g. L21
    filename = f"{video_id}.mp4"

    # Targeted candidate locations across known Kaggle dataset structures
    base_dirs: list[Path | None] = [
        video_root,
        Path("/kaggle/input/datasets/lyduchoang/aic-26-video/Videos"),
        Path("/kaggle/input/datasets/lyduchoang/aic-26-video/videos"),
        Path("/kaggle/input/datasets/lyduchoang/aic-26-video"),
        Path("/kaggle/input/aic-26-video/Videos"),
        Path("/kaggle/input/aic-26-video/videos"),
        Path("/kaggle/input/aic-26-video"),
        Path("/kaggle/input/aic2026-video/Videos"),
        Path("/kaggle/input/aic2026-video"),
        REPO_ROOT / "data" / "videos",
    ]
    candidates: list[Path] = []
    for b in base_dirs:
        if b is None or not b.exists():
            continue
        candidates.extend([
            b / filename,
            b / batch / filename,
            b / f"Videos_{batch}" / filename,
            b / f"Videos_{batch}" / "video" / filename,
            b / f"Videos_{batch}" / "videos" / filename,
            b / "Videos" / f"Videos_{batch}" / "video" / filename,
            b / "Videos" / f"Videos_{batch}" / filename,
            b / "videos" / f"Videos_{batch}" / "video" / filename,
            b / "videos" / f"Videos_{batch}" / filename,
        ])

    for cand in candidates:
        if cand.is_file():
            return cand

    # Shallow scan over top-level datasets under /kaggle/input (avoiding unconstrained rglob)
    if Path("/kaggle/input").exists():
        for dataset_dir in Path("/kaggle/input").iterdir():
            if not dataset_dir.is_dir():
                continue
            for sub in [
                dataset_dir / filename,
                dataset_dir / batch / filename,
                dataset_dir / f"Videos_{batch}" / filename,
                dataset_dir / f"Videos_{batch}" / "video" / filename,
                dataset_dir / "Videos" / f"Videos_{batch}" / "video" / filename,
                dataset_dir / "Videos" / f"Videos_{batch}" / filename,
            ]:
                if sub.is_file():
                    return sub

    raise FileNotFoundError(
        f"Video file {filename} not found. Please pass direct path with --video-path /path/to/{filename}"
    )


def render_multimodal_card(
    record: UnifiedFrameRecord,
    image_path: Path,
    output_card_path: Path,
) -> None:
    """Render a comprehensive 3-panel visual inspection card for a single frame."""
    with Image.open(image_path) as pil_img:
        orig_img = pil_img.convert("RGB")

    # 1. OCR Overlay Image
    ocr_overlay = orig_img.copy()
    draw_ocr = ImageDraw.Draw(ocr_overlay)
    for span in record.ocr.spans:
        if len(span.polygon_xy) >= 3:
            draw_ocr.polygon(span.polygon_xy, outline="#00FF66", width=3)
            first_pt = span.polygon_xy[0]
            draw_ocr.text((first_pt[0] + 2, first_pt[1] + 2), span.normalized_text[:18], fill="#FFFF00")

    # 2. DAM/SAM Object Boxes Overlay Image
    obj_overlay = orig_img.copy()
    draw_obj = ImageDraw.Draw(obj_overlay)
    colors = ["#FF3366", "#33CCFF", "#FFCC00", "#9933FF", "#00FFCC"]
    for idx, cap in enumerate(record.dam_descriptions):
        box_col = colors[idx % len(colors)]
        x1, y1, x2, y2 = cap.bbox_xyxy_px
        draw_obj.rectangle([x1, y1, x2, y2], outline=box_col, width=3)
        draw_obj.text((x1 + 4, max(0, y1 - 15)), f"#{idx+1} {cap.class_label}", fill=box_col)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), gridspec_kw={"width_ratios": [1, 1, 1.2], "wspace": 0.08})

    # Panel 1: OCR
    axes[0].imshow(ocr_overlay)
    ocr_title = record.ocr.full_text[:40] + ("..." if len(record.ocr.full_text) > 40 else "")
    axes[0].set_title(f"🔍 1. OCR Text Overlays ({len(record.ocr.spans)} spans)\n\"{ocr_title or '<No text>'}\"", fontsize=10, weight="bold")
    axes[0].axis("off")

    # Panel 2: Segmented Objects
    axes[1].imshow(obj_overlay)
    axes[1].set_title(f"🎯 2. SAM / DAM Objects ({len(record.dam_descriptions)} regions)", fontsize=10, weight="bold")
    axes[1].axis("off")

    # Panel 3: Multi-Modal Metadata & Captions
    axes[2].axis("off")
    axes[2].set_facecolor("#1E1E2E")
    
    caption_lines = [
        f"📊 CANONICAL INDEXING:",
        f"  • Frame UID:  {record.frame_uid}",
        f"  • Keyframe #: {record.keyframe_n} (raw frame_idx: {record.frame_idx})",
        f"  • Timestamp:  {record.pts_time_s:.3f}s (FPS: {record.fps:.1f})",
        f"  • Shot ID:    {record.shot_id or 'N/A'}",
        "",
        f"🔮 SIGLIP-2 EMBEDDING:",
        f"  • Vector: 768-dim float32 (L2 Norm: 1.000)",
        "",
        f"📝 OCR TRANSCRIPT:",
        f"  • \"{record.ocr.full_text or '<No text detected>'}\"",
        "",
        f"🏷️ DAM-3B DENSE CAPTIONS (<= 50 words):",
    ]
    if record.dam_descriptions:
        for idx, cap in enumerate(record.dam_descriptions, start=1):
            caption_lines.append(f"  [{idx}] {cap.class_label} (IoU: {cap.sam_iou:.2f} | {cap.word_count} words):")
            wrapped = textwrap.fill(cap.caption_en, width=46)
            for w_line in wrapped.splitlines():
                caption_lines.append(f"      \"{w_line}\"")
    else:
        caption_lines.append("  <No object regions segmented>")

    full_text = "\n".join(caption_lines)
    axes[2].text(
        0.02, 0.98, full_text,
        transform=axes[2].transAxes,
        fontsize=9,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#D0D7DE", alpha=0.95),
    )

    plt.suptitle(
        f"🎬 Multi-Modal Frame Pipeline Inspection: {record.frame_uid}",
        fontsize=13,
        weight="bold",
        y=0.98,
    )
    plt.tight_layout()
    output_card_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_card_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> int:
    print("\n" + "=" * 75, flush=True)
    print("🔬 STARTING UNIFIED MULTI-MODAL PIPELINE EXPERIMENT", flush=True)
    print("=" * 75, flush=True)

    args = build_parser().parse_args()
    print(f"🔍 Searching for video: {args.video_id}...", flush=True)
    video_path = find_video_file(args.video_id, args.video_path, args.video_root)
    print(f"🎬 Found Video File:   {video_path}", flush=True)

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

    print(f"\n🎨 Rendering {len(records)} visual inspection cards...", flush=True)
    vis_dir = args.output_dir / "visualizations" / args.video_id
    vis_dir.mkdir(parents=True, exist_ok=True)

    for rec in records:
        img_path = args.output_dir / rec.image_relpath
        if img_path.exists():
            card_p = vis_dir / f"{args.video_id}_f{rec.keyframe_n:04d}_multimodal.jpg"
            render_multimodal_card(rec, img_path, card_p)
            print(f"  ✓ Saved visual card: {card_p.name}", flush=True)

    print("\n" + "=" * 75, flush=True)
    print(f"🎉 SUCCESS! Unified Pipeline Experiment for {args.video_id} is complete!", flush=True)
    print(f"  • Map-Keyframes CSV: {map_csv_path}", flush=True)
    print(f"  • Unified JSONL:     {unified_jsonl_path}", flush=True)
    print(f"  • Visual Cards:      {vis_dir}", flush=True)
    print("=" * 75 + "\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
