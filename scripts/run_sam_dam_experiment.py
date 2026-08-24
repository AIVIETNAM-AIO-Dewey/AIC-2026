#!/usr/bin/env python3
"""Interactive & CLI Step-by-Step SAM Segmentation -> DAM Dense Captioning Experiment Runner."""

from __future__ import annotations

import argparse
import math
import os
import sys
import textwrap
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Disable framework probing in transformers before importing DAM/PyTorch modules
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"
try:
    import transformers.utils.import_utils as _t_import

    _t_import._scipy_available = False
    _t_import.is_scipy_available = lambda: False
    _t_import._sklearn_available = False
    _t_import.is_sklearn_available = lambda: False
    _t_import._tf_available = False
    _t_import.is_tf_available = lambda: False
    import transformers.utils as _t_utils

    _t_utils._scipy_available = False
    _t_utils.is_scipy_available = lambda: False
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from aic2026.object_description import (  # noqa: E402
    FilterConfig,
    filter_detections,
    load_organizer_detections,
    normalize_caption,
    normalized_to_pixels,
)
from aic2026.object_description.dam_backend import DamCaptioner  # noqa: E402
from aic2026.object_description.rle import rectangle_mask  # noqa: E402

SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CHECKPOINT_FILENAME = "sam_vit_b_01ec64.pth"

COLOR_PALETTE = [
    (0.96, 0.26, 0.21, 0.55),  # Red
    (0.13, 0.59, 0.95, 0.55),  # Blue
    (0.30, 0.69, 0.31, 0.55),  # Green
    (1.00, 0.76, 0.03, 0.55),  # Amber
    (0.61, 0.15, 0.69, 0.55),  # Purple
    (0.00, 0.74, 0.83, 0.55),  # Cyan
]


@dataclass(frozen=True, slots=True)
class MaskResult:
    mask: np.ndarray
    source: str
    iou_score: float | None


class PathResolver:
    def __init__(
        self,
        keyframes_root: Path,
        objects_root: Path,
        map_keyframes_root: Path,
    ) -> None:
        self.keyframes_root = keyframes_root
        self.objects_root = objects_root
        self.map_keyframes_root = map_keyframes_root
        self._frame_dir_cache: dict[str, Path] = {}
        self._objects_dir_cache: dict[str, Path] = {}

    def resolve_frames_dir(self, video_id: str) -> Path:
        if video_id in self._frame_dir_cache:
            return self._frame_dir_cache[video_id]

        batch = video_id.split("_")[0]  # e.g. L21
        candidates = [
            self.keyframes_root / f"Keyframes_{batch}" / "keyframes" / video_id,
            self.keyframes_root / f"Keyframes_{batch}" / video_id,
            self.keyframes_root / "Keyframes" / f"Keyframes_{batch}" / "keyframes" / video_id,
            self.keyframes_root / "Keyframes" / "Keyframes" / f"Keyframes_{batch}" / "keyframes" / video_id,
            self.keyframes_root / "Keyframes" / f"Keyframes_{batch}" / video_id,
            self.keyframes_root / batch / video_id,
            self.keyframes_root / "keyframes" / video_id,
            self.keyframes_root / video_id,
            self.keyframes_root.parent / f"Keyframes_{batch}" / "keyframes" / video_id,
            self.keyframes_root.parent / f"Keyframes_{batch}" / video_id,
            self.keyframes_root.parent / "Keyframes" / f"Keyframes_{batch}" / "keyframes" / video_id,
        ]
        for candidate in candidates:
            if candidate.is_dir():
                self._frame_dir_cache[video_id] = candidate
                return candidate

        # Recursive search fallback
        if self.keyframes_root.exists():
            for match in self.keyframes_root.rglob(video_id):
                if match.is_dir():
                    self._frame_dir_cache[video_id] = match
                    return match
        if self.keyframes_root.parent.exists():
            for match in self.keyframes_root.parent.rglob(video_id):
                if match.is_dir():
                    self._frame_dir_cache[video_id] = match
                    return match

        return self.keyframes_root / video_id

    def resolve_objects_dir(self, video_id: str) -> Path:
        if video_id in self._objects_dir_cache:
            return self._objects_dir_cache[video_id]

        batch = video_id.split("_")[0]
        candidates = [
            self.objects_root / video_id,
            self.objects_root / "objects" / video_id,
            self.objects_root / "data" / "objects" / video_id,
            self.objects_root / batch / video_id,
        ]
        for candidate in candidates:
            if candidate.is_dir():
                self._objects_dir_cache[video_id] = candidate
                return candidate

        if self.objects_root.exists():
            for match in self.objects_root.rglob(video_id):
                if match.is_dir():
                    self._objects_dir_cache[video_id] = match
                    return match

        return self.objects_root / video_id


def find_keyframe_image(frames_dir: Path, keyframe_n: int) -> Path | None:
    """Find keyframe image supporting various zero-padded and extension schemes."""
    if not frames_dir.exists():
        return None
    patterns = [
        f"{keyframe_n:03d}.jpg",
        f"{keyframe_n:04d}.jpg",
        f"{keyframe_n:05d}.jpg",
        f"{keyframe_n:06d}.jpg",
        f"{keyframe_n}.jpg",
        f"{keyframe_n:03d}.png",
        f"{keyframe_n:04d}.png",
        f"{keyframe_n}.png",
    ]
    for p in patterns:
        cand = frames_dir / p
        if cand.is_file():
            return cand
    for file_path in frames_dir.iterdir():
        if file_path.is_file() and file_path.stem.isdigit() and int(file_path.stem) == keyframe_n:
            return file_path
    return None


def load_sam_predictor(checkpoint_dir: Path, device: str = "cuda") -> Any:
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            "segment_anything is required. Install via: pip install git+https://github.com/facebookresearch/segment-anything.git"
        ) from exc

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / SAM_CHECKPOINT_FILENAME
    if not checkpoint_file.exists() or checkpoint_file.stat().st_size < 100_000_000:
        print(f"📥 Downloading Meta SAM ViT-B checkpoint (375 MB) to {checkpoint_file} ...")
        urllib.request.urlretrieve(SAM_CHECKPOINT_URL, checkpoint_file)
        print("✓ SAM checkpoint downloaded!")

    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint_file)).to(device)
    sam.eval()
    return SamPredictor(sam)


def plot_sam_masks_overlay(
    image: Image.Image,
    detections: list[Any],
    sam_predictions: list[MaskResult],
    title: str = "SAM Segmented Masks",
    save_path: Path | None = None,
) -> None:
    """Render full image with translucent SAM masks, contours, bounding boxes, and IoU scores."""
    w, h = image.size
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(image)

    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for idx, (det, pred) in enumerate(zip(detections, sam_predictions)):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        overlay[pred.mask] = color

        box = normalized_to_pixels(det.bbox_yxyx_norm, w, h)
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2.5,
            edgecolor=(color[0], color[1], color[2], 1.0),
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

        iou_str = f"IoU: {pred.iou_score:.3f}" if pred.iou_score is not None else "BBox Fallback"
        label_text = f"#{idx+1}: {det.class_entity} ({det.score:.2f}) | {iou_str}"
        ax.text(
            x1 + 4,
            max(15, y1 - 6),
            label_text,
            color="white",
            fontsize=10,
            weight="bold",
            bbox=dict(
                facecolor=(color[0], color[1], color[2], 0.85),
                edgecolor="none",
                pad=3,
                boxstyle="round,pad=0.3",
            ),
        )

    ax.imshow(overlay)
    ax.set_title(title, fontsize=14, weight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"       📸 Saved SAM overlay plot -> {save_path}")
    plt.close(fig)


def plot_dam_region_cards(
    image: Image.Image,
    detections: list[Any],
    sam_predictions: list[MaskResult],
    dam_captions: list[Any],
    video_id: str,
    keyframe_n: int,
    output_dir: Path | None = None,
) -> None:
    """Display and optionally save region cutout cards with SAM masks & DAM descriptions."""
    w, h = image.size
    img_np = np.array(image.convert("RGB"))

    for idx, (det, pred, caption) in enumerate(zip(detections, sam_predictions, dam_captions)):
        box = normalized_to_pixels(det.bbox_yxyx_norm, w, h)
        x1, y1, x2, y2 = box

        masked_rgb = img_np.copy()
        masked_rgb[~pred.mask] = (20, 20, 20)

        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        crop_x1, crop_y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        crop_x2, crop_y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

        cropped_focus = img_np[crop_y1:crop_y2, crop_x1:crop_x2]
        cropped_mask = masked_rgb[crop_y1:crop_y2, crop_x1:crop_x2]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
        ax1.imshow(cropped_focus)
        ax1.set_title(f"Target Object #{idx+1}: {det.class_entity}", fontsize=11, weight="bold")
        ax1.axis("off")

        ax2.imshow(cropped_mask)
        iou_str = f"{pred.iou_score:.3f}" if pred.iou_score is not None else "BBox"
        ax2.set_title(f"SAM Mask Cutout (IoU: {iou_str})", fontsize=11, weight="bold")
        ax2.axis("off")

        plt.suptitle(
            f"Frame {keyframe_n} - Region #{idx+1} ({det.class_entity})",
            fontsize=13,
            weight="bold",
            y=1.02,
        )
        plt.tight_layout()

        if output_dir:
            card_path = output_dir / f"{video_id}_f{keyframe_n:04d}_reg{idx+1}_{det.class_entity}.jpg"
            plt.savefig(card_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

        # Print styled card box
        print("┌" + "─" * 78 + "┐")
        print(
            f"│ 📝 DAM-3B CAPTION [{caption.word_count}/50 words | status: {caption.status}]".ljust(
                79
            )
            + "│"
        )
        print("├" + "─" * 78 + "┤")
        for line in textwrap.wrap(caption.description_en, width=74):
            print(f"│   {line.ljust(74)} │")
        print("└" + "─" * 78 + "┘\n")


def run_sam_dam_experiment(
    video_id: str,
    target_frame_indices: list[int] | int | None = None,
    max_frames_to_test: int = 5,
    score_threshold: float = 0.30,
    min_area_ratio: float = 0.005,
    max_area_ratio: float = 0.85,
    class_nms_iou: float = 0.45,
    max_regions_per_frame: int = 5,
    maximum_words: int = 50,
    keyframes_root: Path | None = None,
    objects_root: Path | None = None,
    map_root: Path | None = None,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    device: str = "cuda",
) -> None:
    """Full step-by-step interactive SAM -> DAM execution and visualization."""
    if isinstance(target_frame_indices, int):
        target_frame_indices = [target_frame_indices]

    keyframes_root = keyframes_root or Path(
        os.environ.get(
            "KEYFRAMES_ROOT",
            "/kaggle/input/datasets/lyduchoang/aic-26-video/Keyframes/Keyframes",
        )
    )
    objects_root = objects_root or Path(
        os.environ.get(
            "OBJECTS_ROOT",
            "/kaggle/input/datasets/khoalequangminh/aic-test-dataset/data/objects",
        )
    )
    map_root = map_root or Path(
        os.environ.get(
            "MAP_ROOT",
            "/kaggle/input/datasets/khoalequangminh/aic-test-dataset/data/map-keyframes",
        )
    )
    cache_dir = cache_dir or Path("/kaggle/working/aic2026-model-cache")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f" 🔬 RUNNING SAM -> DAM STEP-BY-STEP PIPELINE EXPERIMENT FOR: {video_id}")
    if target_frame_indices is not None:
        print(f"    • Target Frame Indices: {target_frame_indices}")
    else:
        print(f"    • Max Frames to Test:  {max_frames_to_test}")
    print(f"    • Score Threshold:     {score_threshold}")
    print(f"    • Max Regions/Frame:   {max_regions_per_frame}")
    print(f"    • Max Caption Words:   {maximum_words} words (strict cap)")
    if output_dir:
        print(f"    • Output Directory:    {output_dir}")
    print("=" * 80)

    # 1. Load SAM
    print(f"🚀 [1/2] Loading Meta SAM (ViT-B) on {device}...")
    sam_predictor = load_sam_predictor(checkpoint_dir=cache_dir / "sam", device=device)
    print("✓ Meta SAM loaded successfully!")

    # 2. Load DAM-3B
    print(f"\n🚀 [2/2] Loading DAM-3B (nvidia/DAM-3B) on {device}...")
    dam_captioner = DamCaptioner.from_pretrained(
        model_id="nvidia/DAM-3B",
        revision="0797bedd98d645cd021379a4661ee233da279bba",
        code_revision="153ad3d33c29324e9197f565547c6bc8500da02d",
        cache_dir=cache_dir,
    )
    print("✓ DAM-3B loaded successfully!")
    print("=" * 80)

    filter_cfg = FilterConfig(
        minimum_score=score_threshold,
        minimum_area_ratio=min_area_ratio,
        maximum_area_ratio=max_area_ratio,
        same_class_iou=class_nms_iou,
        cross_label_duplicate_iou=0.60,
        maximum_regions=max_regions_per_frame,
    )

    resolver = PathResolver(
        keyframes_root=keyframes_root,
        objects_root=objects_root,
        map_keyframes_root=map_root,
    )
    frames_dir = resolver.resolve_frames_dir(video_id)
    objects_dir = resolver.resolve_objects_dir(video_id)

    print(f"📁 Keyframes Dir: {frames_dir}")
    print(f"📁 Objects Dir:   {objects_dir}")

    if not frames_dir.exists():
        print(f"❌ Frames directory not found: {frames_dir}")
        print(f"   Available paths under {keyframes_root}:")
        if keyframes_root.exists():
            for child in sorted(keyframes_root.iterdir())[:10]:
                print(f"     • {child.name}")
        return

    if not objects_dir.exists():
        print(f"❌ Objects directory not found: {objects_dir}")
        print(f"   Available paths under {objects_root}:")
        if objects_root.exists():
            for child in sorted(objects_root.iterdir())[:10]:
                print(f"     • {child.name}")
        return

    json_files = sorted(
        objects_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 999999
    )
    if not json_files:
        print(f"❌ No object JSON files found in {objects_dir}")
        return

    tested_count = 0
    total_target = len(target_frame_indices) if target_frame_indices else max_frames_to_test

    for json_path in json_files:
        keyframe_n = int(json_path.stem)

        if target_frame_indices is not None and keyframe_n not in target_frame_indices:
            continue

        img_path = find_keyframe_image(frames_dir, keyframe_n)
        if img_path is None:
            print(f"⚠️ Keyframe image #{keyframe_n} not found in {frames_dir}")
            continue

        raw_dets = load_organizer_detections(json_path)
        filtered_dets = filter_detections(raw_dets, filter_cfg)
        if not filtered_dets:
            print(
                f"ℹ️ Keyframe #{keyframe_n}: No detections passed score threshold >= {score_threshold}"
            )
            continue

        tested_count += 1
        print(
            f"\n🎥 [Frame {tested_count}/{total_target}] Keyframe #{keyframe_n} ({img_path.name})"
        )
        print(
            f"    Raw Boxes: {len(raw_dets)} -> Filtered Distinct Objects: {len(filtered_dets)} ({[d.class_entity for d in filtered_dets]})"
        )

        with Image.open(img_path) as src_img:
            image = src_img.convert("RGB")
            w, h = image.size
            rgb_np = np.array(image)

            boxes_xyxy = [normalized_to_pixels(d.bbox_yxyx_norm, w, h) for d in filtered_dets]

            # STEP 1: SAM Segmentation
            print(
                f"    ⏳ [SAM] Starting Meta SAM segmentation for {len(boxes_xyxy)} bounding boxes..."
            )
            sam_predictor.set_image(rgb_np)
            sam_preds = []
            for box in boxes_xyxy:
                fallback_mask = rectangle_mask(h, w, box)
                try:
                    masks, scores, _ = sam_predictor.predict(
                        box=np.array(box),
                        multimask_output=False,
                    )
                    mask = np.asarray(masks[0], dtype=bool)
                    score = float(scores[0])
                    if mask.shape != (h, w) or not mask.any() or not math.isfinite(score):
                        raise ValueError("Invalid mask")
                    sam_preds.append(MaskResult(mask=mask, source="sam", iou_score=score))
                except Exception:
                    sam_preds.append(
                        MaskResult(mask=fallback_mask, source="bbox_fallback", iou_score=None)
                    )

            iou_summary = [
                f"{p.iou_score:.3f}" if p.iou_score is not None else "fallback" for p in sam_preds
            ]
            print(
                f"    ✅ [SAM] Finished SAM! Generated {len(sam_preds)} masks (IoU Scores: {iou_summary})"
            )

            # Render & save SAM overlay plot
            overlay_save_path = (
                output_dir / f"{video_id}_f{keyframe_n:04d}_sam_overlay.jpg"
                if output_dir
                else None
            )
            plot_sam_masks_overlay(
                image=image,
                detections=filtered_dets,
                sam_predictions=sam_preds,
                title=f"{video_id} - Keyframe #{keyframe_n} | Meta SAM Segmented Mask Overlays",
                save_path=overlay_save_path,
            )

            # STEP 2: DAM Dense Captioning
            print(
                f"    ⏳ [DAM] Starting DAM-3B description for {len(sam_preds)} segmented regions (50-word cap)..."
            )
            dam_captions = []
            for reg_idx, (det, pred) in enumerate(zip(filtered_dets, sam_preds), start=1):
                mask_pil = Image.fromarray((np.asarray(pred.mask, dtype=bool) * 255).astype(np.uint8))
                raw_text = dam_captioner.describe(image, mask_pil, max_new_tokens=75)
                caption = normalize_caption(raw_text, maximum_words=maximum_words)
                dam_captions.append(caption)
                print(
                    f"       • Region #{reg_idx} [{det.class_entity}]: \"{caption.description_en}\" ({caption.word_count} words)"
                )
            print(f"    ✅ [DAM] Finished DAM descriptions for all {len(dam_captions)} regions!")

            # Render & save region cards
            plot_dam_region_cards(
                image,
                filtered_dets,
                sam_preds,
                dam_captions,
                video_id=video_id,
                keyframe_n=keyframe_n,
                output_dir=output_dir,
            )

        if target_frame_indices is None and tested_count >= max_frames_to_test:
            break

    print("=" * 80)
    print(f"🎉 EXPERIMENT COMPLETED FOR {video_id}! Processed {tested_count} keyframes.")
    if output_dir:
        print(f"📁 All visual inspection plots saved to: {output_dir}")
    print("=" * 80)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SAM -> DAM Step-by-Step Segmentation & Dense Captioning Experiment"
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default="L21_V003",
        help="Target video ID to test (default: L21_V003)",
    )
    parser.add_argument(
        "--frames",
        "--frame-indices",
        type=int,
        nargs="*",
        default=None,
        help="Specific keyframe indices to test (e.g. --frames 1 5 10 20)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=5,
        help="Maximum keyframes to test when specific frames are not given (default: 5)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.30,
        help="Detection confidence score threshold (default: 0.30)",
    )
    parser.add_argument(
        "--max-regions",
        type=int,
        default=5,
        help="Maximum distinct regions to caption per frame (default: 5)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=50,
        help="Maximum words per DAM caption (strict cap, default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/sam_dam_results"),
        help="Directory to save generated plot images (default: /kaggle/working/sam_dam_results)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument(
        "--keyframes-root",
        type=Path,
        default=None,
        help="Custom keyframes dataset directory",
    )
    parser.add_argument(
        "--objects-root",
        type=Path,
        default=None,
        help="Custom objects dataset directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    device = args.device if (torch.cuda.is_available() and args.device == "cuda") else "cpu"
    run_sam_dam_experiment(
        video_id=args.video_id,
        target_frame_indices=args.frames,
        max_frames_to_test=args.max_frames,
        score_threshold=args.score_threshold,
        max_regions_per_frame=args.max_regions,
        maximum_words=args.max_words,
        output_dir=args.output_dir.expanduser().resolve() if args.output_dir else None,
        keyframes_root=args.keyframes_root,
        objects_root=args.objects_root,
        device=device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
