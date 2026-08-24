#!/usr/bin/env python3
"""Direct standalone test script for NVIDIA DAM-3B on a single image and mask.

Enforces strict single GPU execution (cuda:0) and tests various mask configurations
(synthetic box, keyframe image, full scene) to verify CUDA kernel execution.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Enforce strict single GPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from aic2026.object_description.caption import DAM_PROMPT, normalize_caption
from aic2026.object_description.dam_backend import DamCaptioner


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone DAM-3B Diagnostics and Test Runner")
    parser.add_argument("--image-path", type=Path, default=None, help="Path to test image file")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--max-words", type=int, default=50)
    args = parser.parse_args()

    print("=" * 70)
    print("🔬 RUNNING STANDALONE NVIDIA DAM-3B TEST HARNESS")
    print("=" * 70)

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  PyTorch Version: {torch.__version__}")
    print(f"🖥️  CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🖥️  GPU Device:      {torch.cuda.get_device_name(0)}")
        print(f"🖥️  GPU Count:       {torch.cuda.device_count()}")
        print(f"🖥️  Initial VRAM:    {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB allocated")

    # 1. Load DAM Captioner
    print("\n⏳ [1/2] Loading DAM-3B weights into GPU memory...")
    t0 = time.monotonic()
    captioner = DamCaptioner.from_pretrained()
    load_time = time.monotonic() - t0
    print(f"✓ DAM-3B loaded successfully in {load_time:.2f}s!")
    if torch.cuda.is_available():
        print(f"  VRAM after DAM load: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

    # 2. Prepare Test Image and Mask
    if args.image_path and args.image_path.is_file():
        print(f"\n🖼️ [2/2] Testing with provided image: {args.image_path}")
        with Image.open(args.image_path) as im:
            test_image = im.convert("RGB")
    else:
        # Check for extracted keyframes in output directory
        cand_keyframes = list(Path("/kaggle/working/multimodal_results").glob("**/*.jpg"))
        if cand_keyframes:
            test_path = cand_keyframes[0]
            print(f"\n🖼️ [2/2] Testing with discovered keyframe: {test_path}")
            with Image.open(test_path) as im:
                test_image = im.convert("RGB")
        else:
            print("\n🖼️ [2/2] Testing with synthetic 720x1280 image...")
            test_image = Image.new("RGB", (1280, 720), color=(100, 150, 200))
            draw = ImageDraw.Draw(test_image)
            draw.rectangle([400, 250, 880, 550], fill=(220, 50, 50))
            draw.ellipse([550, 150, 730, 330], fill=(240, 200, 50))

    w, h = test_image.size
    print(f"   Image Resolution: {w}x{h}")

    # Test Case 1: Centered Object Box Mask
    print("\n🧪 Test Case 1: Centered Object Region Mask (400, 200, 880, 550)...")
    mask_box = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask_box).rectangle([int(0.25 * w), int(0.25 * h), int(0.75 * w), int(0.75 * h)], fill=255)
    
    try:
        t_start = time.monotonic()
        caption_1 = captioner.describe(test_image, mask_box, max_new_tokens=48)
        norm_1 = normalize_caption(caption_1, maximum_words=args.max_words)
        elapsed_1 = time.monotonic() - t_start
        print(f"   ✅ Test Case 1 Succeeded in {elapsed_1:.2f}s!")
        print(f"   📝 Caption ({norm_1.word_count} words): \"{norm_1.description_en}\"")
    except Exception as e:
        print(f"   ❌ Test Case 1 Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test Case 2: Left Focal Object Mask
    print("\n🧪 Test Case 2: Left Sub-Region Mask (50, 100, 450, 600)...")
    mask_left = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask_left).rectangle([int(0.05 * w), int(0.15 * h), int(0.45 * w), int(0.85 * h)], fill=255)

    try:
        t_start = time.monotonic()
        caption_2 = captioner.describe(test_image, mask_left, max_new_tokens=48)
        norm_2 = normalize_caption(caption_2, maximum_words=args.max_words)
        elapsed_2 = time.monotonic() - t_start
        print(f"   ✅ Test Case 2 Succeeded in {elapsed_2:.2f}s!")
        print(f"   📝 Caption ({norm_2.word_count} words): \"{norm_2.description_en}\"")
    except Exception as e:
        print(f"   ❌ Test Case 2 Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test Case 3: Focal Scene Region
    print("\n🧪 Test Case 3: Scene Region Mask (5% - 95%)...")
    mask_scene = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask_scene).rectangle([int(0.05 * w), int(0.05 * h), int(0.95 * w), int(0.95 * h)], fill=255)

    try:
        t_start = time.monotonic()
        caption_3 = captioner.describe(test_image, mask_scene, max_new_tokens=48)
        norm_3 = normalize_caption(caption_3, maximum_words=args.max_words)
        elapsed_3 = time.monotonic() - t_start
        print(f"   ✅ Test Case 3 Succeeded in {elapsed_3:.2f}s!")
        print(f"   📝 Caption ({norm_3.word_count} words): \"{norm_3.description_en}\"")
    except Exception as e:
        print(f"   ❌ Test Case 3 Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("🎉 DAM-3B Standalone Diagnostics Complete!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
