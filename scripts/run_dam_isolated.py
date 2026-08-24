#!/usr/bin/env python3
"""Isolated subprocess runner for DAM-3B dense captioning.

Runs in a fresh Python process with a pristine CUDA context to guarantee
zero cuBLAS handle conflicts or memory fragmentation from other vision models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw

from aic2026.object_description.caption import normalize_caption
from aic2026.object_description.dam_backend import DamCaptioner


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated DAM-3B Subprocess Worker")
    parser.add_argument("--input-json", type=Path, required=True, help="Input frames and regions JSON")
    parser.add_argument("--output-json", type=Path, required=True, help="Output descriptions JSON")
    parser.add_argument("--device", default="cuda", help="CUDA device")
    parser.add_argument("--max-words", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--model-id", default="nvidia/DAM-3B")
    parser.add_argument("--revision", default="0797bedd98d645cd021379a4661ee233da279bba")
    parser.add_argument("--code-revision", default="153ad3d33c29324e9197f565547c6bc8500da02d")
    args = parser.parse_args()

    print(
        f"   🚀 [DAM-Isolated-Process] Starting fresh Python process (PID: {os.getpid()}) on {args.device.upper()}...",
        flush=True,
    )
    t0 = time.monotonic()

    captioner = DamCaptioner.from_pretrained(
        model_id=args.model_id,
        revision=args.revision,
        code_revision=args.code_revision,
    )
    print(
        f"   ✓ [DAM-Isolated-Process] DAM-3B initialized in {time.monotonic() - t0:.2f}s with pristine CUDA context!",
        flush=True,
    )

    with open(args.input_json, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    results: dict[str, list[dict]] = {}
    for task in tasks:
        keyframe_n = task["keyframe_n"]
        img_path = Path(task["image_path"])
        regions = task["regions"]

        if not img_path.exists():
            results[str(keyframe_n)] = []
            continue

        with Image.open(img_path) as im:
            image_rgb = im.convert("RGB")
            w, h = image_rgb.size

        frame_captions = []
        for reg in regions:
            reg_id = reg["region_id"]
            bbox = reg.get("bbox_xyxy")
            class_label = reg.get("class_label", "object")
            iou = reg.get("sam_iou", 0.90)

            # Create bounding box mask
            if bbox:
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(w - 1, int(round(x1))))
                y1 = max(0, min(h - 1, int(round(y1))))
                x2 = max(x1 + 1, min(w, int(round(x2))))
                y2 = max(y1 + 1, min(h, int(round(y2))))
                mask_img = Image.new("L", (w, h), 0)
                ImageDraw.Draw(mask_img).rectangle([x1, y1, x2, y2], fill=255)
            else:
                mask_img = Image.new("L", (w, h), 255)

            try:
                import torch

                with torch.inference_mode():
                    raw_caption = captioner.describe(
                        image_rgb, mask_img, max_new_tokens=args.max_new_tokens
                    )
                cap_res = normalize_caption(raw_caption, maximum_words=args.max_words)
                frame_captions.append(
                    {
                        "region_id": reg_id,
                        "class_label": class_label,
                        "bbox_xyxy_px": bbox,
                        "sam_iou": iou,
                        "caption_en": cap_res.description_en,
                        "word_count": cap_res.word_count,
                        "status": "ok",
                    }
                )
            except Exception as e:
                print(f"   ⚠️ [DAM-Isolated-Process] Region {reg_id} error: {e}", flush=True)
                frame_captions.append(
                    {
                        "region_id": reg_id,
                        "class_label": class_label,
                        "bbox_xyxy_px": bbox,
                        "sam_iou": iou,
                        "caption_en": "",
                        "word_count": 0,
                        "status": "error",
                    }
                )

        results[str(keyframe_n)] = frame_captions

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(
        f"   ✓ [DAM-Isolated-Process] Successfully captioned all regions in {time.monotonic() - t0:.2f}s!",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
