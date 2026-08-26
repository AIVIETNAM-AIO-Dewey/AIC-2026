#!/usr/bin/env python3
"""
Visualize keyframe images along with multi-modal metadata:
- Extracted image & bounding boxes (YOLO-World + DAM-3B)
- Fine-grained DAM-3B text descriptions
- EasyOCR Vietnamese transcripts
- SigLIP2 768-D dense embeddings
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from safetensors.numpy import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot keyframe image with DAM, OCR, and SigLIP2 metadata.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("/Users/khoale/Downloads/AIC_Challenger/downloaded_artifacts"),
        help="Path to directory containing downloaded artifact streams.",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default="L21_V001",
        help="Video ID to inspect (e.g. L21_V001).",
    )
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=0,
        help="Frame index within the video (0-indexed).",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Optional path to save figure image (e.g. preview.png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = args.artifacts_dir
    video_id = args.video_id
    idx = args.frame_idx

    # File paths
    unified_file = artifacts_dir / "unified_metadata" / f"{video_id}.jsonl"
    desc_file = artifacts_dir / "descriptions" / f"{video_id}.jsonl"
    safetensors_file = artifacts_dir / "scene_embeddings" / f"{video_id}.safetensors"
    zip_file = artifacts_dir / "keyframes_zips" / f"{video_id}.zip"

    # Assert existence
    for path, name in [
        (unified_file, "unified_metadata"),
        (desc_file, "descriptions"),
        (safetensors_file, "scene_embeddings"),
        (zip_file, "keyframes_zips"),
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} for {video_id} at {path}")

    # Load Unified Metadata
    with open(unified_file, "r", encoding="utf-8") as f:
        unified_records = [json.loads(line) for line in f]

    # Load Detailed DAM Bounding Boxes
    with open(desc_file, "r", encoding="utf-8") as f:
        desc_records = [json.loads(line) for line in f]

    # Load SigLIP2 Safetensors Matrix
    mat_dict = load_file(str(safetensors_file))
    embeddings_matrix = mat_dict["embeddings"]  # Shape: (N_frames, 768)

    if idx < 0 or idx >= len(unified_records):
        raise IndexError(f"Frame index {idx} out of range [0 .. {len(unified_records)-1}]")

    meta = unified_records[idx]
    desc = desc_records[idx] if idx < len(desc_records) else {}
    vec = embeddings_matrix[meta["embedding_row"]].astype(np.float32)

    # Load Image from Zip
    with zipfile.ZipFile(zip_file, "r") as zf:
        img_filename = Path(meta["image_relpath"]).name
        matched = [name for name in zf.namelist() if name.endswith(img_filename)]
        if not matched:
            raise FileNotFoundError(f"Image {img_filename} not found inside {zip_file}")
        with zf.open(matched[0]) as img_f:
            image = Image.open(img_f).convert("RGB")

    # Plot Layout
    fig = plt.figure(figsize=(18, 10), facecolor="#1a1a1a")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.2, 0.8], hspace=0.25, wspace=0.15)

    # --- PANEL 1: KEYFRAME IMAGE WITH BOUNDING BOXES ---
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(image)
    ax_img.set_title(
        f"Keyframe: {meta['frame_uid']} | PTS: {meta['pts_time_s']:.2f}s (Point ID: {meta['point_id']})",
        color="white", fontsize=14, pad=12, fontweight="bold"
    )
    ax_img.axis("off")

    W, H = image.size
    colors = ["#00ffcc", "#ff007f", "#ffcc00", "#0099ff", "#ff6600"]

    for r_idx, region in enumerate(desc.get("regions", [])):
        bbox = region.get("box_2d") or region.get("box")
        if bbox:
            y1, x1, y2, x2 = bbox
            if max(bbox) <= 1.0:
                x1, y1, x2, y2 = x1 * W, y1 * H, x2 * W, y2 * H
            
            color = colors[r_idx % len(colors)]
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2.5, edgecolor=color, facecolor="none", linestyle="-"
            )
            ax_img.add_patch(rect)
            label = region.get("label", f"Obj {r_idx+1}")
            ax_img.text(
                x1 + 4, y1 + 18, f"#{r_idx+1} {label}",
                color="black", fontsize=10, fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.85, edgecolor="none", boxstyle="round,pad=0.2")
            )

    # --- PANEL 2: MULTI-MODAL METADATA (DAM & OCR) ---
    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.set_facecolor("#222222")
    ax_text.axis("off")

    info_text = (
        f"--- VIDEO & FRAME INFO ---\n"
        f"• Video ID:      {meta['video_id']}\n"
        f"• Frame Index:   {meta['frame_idx']} (Keyframe #{meta['keyframe_n']})\n"
        f"• Timestamp:     {meta['pts_time_s']:.3f} seconds\n"
        f"• Resolution:    {W} x {H} px\n\n"
        f"--- DAM-3B VISUAL DESCRIPTION ---\n"
        f"{meta.get('dam_summary_en', 'N/A')}\n\n"
        f"--- EASYOCR VIETNAMESE TRANSCRIPT ---\n"
        f"\"{meta.get('ocr_text', 'None')}\"\n"
    )

    ax_text.text(
        0.04, 0.96, info_text, transform=ax_text.transAxes,
        fontsize=11, color="#f0f0f0", verticalalignment="top", fontfamily="sans-serif",
        bbox=dict(facecolor="#2b2b2b", edgecolor="#444444", boxstyle="round,pad=0.8")
    )

    # --- PANEL 3: SIGLIP2 EMBEDDING VECTOR ANALYSIS ---
    ax_vec = fig.add_subplot(gs[1, 1])
    ax_vec.set_facecolor("#222222")

    ax_vec.plot(vec, color="#00ffcc", linewidth=1.2, alpha=0.9)
    ax_vec.fill_between(range(len(vec)), vec, color="#00ffcc", alpha=0.25)
    ax_vec.axhline(0, color="#666666", linestyle="--", linewidth=0.8)

    ax_vec.set_title(
        f"⚡ SigLIP2 Dense Embedding (Dim: {len(vec)}, L2 Norm: {np.linalg.norm(vec):.4f})",
        color="white", fontsize=12, pad=8
    )
    ax_vec.set_xlabel("Vector Dimension [0 .. 767]", color="#aaaaaa", fontsize=9)
    ax_vec.set_ylabel("Activation", color="#aaaaaa", fontsize=9)
    ax_vec.tick_params(colors="#aaaaaa")
    for spine in ax_vec.spines.values():
        spine.set_color("#444444")

    plt.tight_layout()

    if args.save_path:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
        print(f"✓ Saved visualization to {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
