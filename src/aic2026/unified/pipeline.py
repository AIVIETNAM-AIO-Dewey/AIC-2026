"""Unified Multi-Modal Offline Video Processing Pipeline.

Executes:
1. TransNetV2 Shot Boundary Detection
2. Adaptive Keyframe Sampling Policy
3. Canonical map-keyframes CSV Generation
4. High-Quality FFmpeg Frame Extraction (001.jpg, 002.jpg...)
5. SigLIP-2 Visual Embedding (768-dim unit vector)
6. OCR Text Extraction & Vietnamese Normalization
7. Meta SAM ViT-B Segmentation + NVIDIA DAM-3B Dense Captions (<= 50 words)
8. Output serialization to map-keyframes/<video_id>.csv and unified/<video_id>.jsonl
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from aic2026.frame_extraction import (
    adaptive_samples_from_shots,
    dedupe_samples,
    extract_frame,
    load_transnetv2_model,
    probe_video,
    run_transnetv2_inference,
)
from aic2026.object_description import (
    DamCaptioner,
    FilterConfig,
    SamMaskGenerator,
    filter_detections,
    load_organizer_detections,
)
from aic2026.ocr import OcrReader
from aic2026.scene_embedding import SiglipEncoder

from .contracts import (
    DamRegionCaption,
    UnifiedFrameRecord,
    UnifiedOcrResult,
    UnifiedOcrSpan,
)


class UnifiedVideoPipeline:
    """End-to-end multi-modal video extractor and enricher."""

    def __init__(
        self,
        transnet_model: Any | None = None,
        siglip_encoder: SiglipEncoder | None = None,
        ocr_reader: OcrReader | None = None,
        sam_generator: SamMaskGenerator | None = None,
        dam_captioner: DamCaptioner | None = None,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.transnet_model = transnet_model
        self.siglip_encoder = siglip_encoder
        self.ocr_reader = ocr_reader
        self.sam_generator = sam_generator
        self.dam_captioner = dam_captioner

    @classmethod
    def load(
        cls,
        device: str = "cuda",
        load_transnet: bool = True,
        load_siglip: bool = True,
        load_ocr: bool = True,
        load_sam_dam: bool = True,
        dam_model_id: str = "nvidia/DAM-3B",
        dam_revision: str = "0797bedd98d645cd021379a4661ee233da279bba",
        dam_code_revision: str = "153ad3d33c29324e9197f565547c6bc8500da02d",
    ) -> UnifiedVideoPipeline:
        """Load all enabled multi-modal models onto the specified device."""
        print(f"\n{'='*75}", flush=True)
        print(f"🚀 INITIALIZING MULTI-MODAL MODELS ON: {device.upper()}", flush=True)
        print(f"{'='*75}", flush=True)

        transnet = None
        if load_transnet:
            print("  ⏳ [1/4] Loading TransNetV2 (PyTorch Shot Boundary Detector)...", flush=True)
            transnet = load_transnetv2_model(device=device)
            print("     ✓ TransNetV2 loaded successfully!", flush=True)

        siglip = None
        if load_siglip:
            print("  ⏳ [2/4] Loading SigLIP-2 (google/siglip2-base-patch16-224)...", flush=True)
            siglip = SiglipEncoder.from_pretrained(device=device)
            print("     ✓ SigLIP-2 (768-dim) loaded successfully!", flush=True)

        ocr = None
        if load_ocr:
            print("  ⏳ [3/4] Loading OCR Reader (Vietnamese + English)...", flush=True)
            ocr = OcrReader.create(device=device)
            print(f"     ✓ OCR Reader ({ocr.backend_type.upper()}) loaded successfully!", flush=True)

        sam = None
        dam = None
        if load_sam_dam:
            print("  ⏳ [4/4] Loading Meta SAM (ViT-B) & NVIDIA DAM-3B...", flush=True)
            sam = SamMaskGenerator.from_pretrained(device=device)
            print("     ✓ Meta SAM loaded successfully!", flush=True)
            dam = DamCaptioner.from_pretrained(
                model_id=dam_model_id,
                revision=dam_revision,
                code_revision=dam_code_revision,
            )
            print("     ✓ DAM-3B loaded successfully!", flush=True)

        print(f"{'='*75}", flush=True)
        print("✓ All 5 multi-modal engines initialized into GPU memory!", flush=True)
        print(f"{'='*75}\n", flush=True)
        return cls(
            transnet_model=transnet,
            siglip_encoder=siglip,
            ocr_reader=ocr,
            sam_generator=sam,
            dam_captioner=dam,
            device=device,
        )

    def process_video(
        self,
        video_path: Path,
        video_id: str,
        output_root: Path,
        objects_dir: Path | None = None,
        max_frames: int | None = None,
        max_regions_per_frame: int = 3,
        maximum_words: int = 50,
        score_threshold: float = 0.30,
    ) -> tuple[Path, Path, list[UnifiedFrameRecord]]:
        """Run complete extraction and multi-modal enrichment for one video.

        Returns:
            (map_csv_path, unified_jsonl_path, records)
        """
        started_at = time.monotonic()
        video_path = video_path.resolve()
        probe = probe_video(video_path)
        fps = probe.fps

        print(f"🎬 Target Video: {video_id}", flush=True)
        print(f"   • Path:     {video_path}", flush=True)
        print(f"   • Duration: {probe.duration_s:.1f}s | FPS: {fps:.2f} | Resolution: {probe.width}x{probe.height}", flush=True)

        # 1. TransNetV2 Shot Detection
        print(f"\n⚡ [1/4] Running TransNetV2 Shot Boundary Detection...", flush=True)
        t_shot_start = time.monotonic()
        transnet_res = run_transnetv2_inference(
            video_path=video_path,
            video_id=video_id,
            fps=fps,
            model=self.transnet_model,
            device=self.device,
        )
        shots = transnet_res.shots
        print(f"   ✓ TransNetV2 completed in {time.monotonic() - t_shot_start:.2f}s (Detected {len(shots)} distinct scene shots)", flush=True)

        # 2. Adaptive Keyframe Sampling
        print("🎯 [2/4] Applying Adaptive Keyframe Sampling Policy...", flush=True)
        raw_samples = adaptive_samples_from_shots(shots)
        candidates = dedupe_samples(raw_samples, tolerance_s=0.5)
        if max_frames is not None and max_frames > 0:
            candidates = candidates[:max_frames]
        print(f"   ✓ Selected {len(candidates)} canonical keyframes for multi-modal processing", flush=True)

        # Setup directory paths
        frames_dir = output_root / "frames" / video_id
        map_csv_dir = output_root / "map-keyframes"
        unified_dir = output_root / "unified"
        frames_dir.mkdir(parents=True, exist_ok=True)
        map_csv_dir.mkdir(parents=True, exist_ok=True)
        unified_dir.mkdir(parents=True, exist_ok=True)

        map_csv_path = map_csv_dir / f"{video_id}.csv"
        unified_jsonl_path = unified_dir / f"{video_id}.jsonl"

        # 3. Write Canonical map-keyframes CSV
        with open(map_csv_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["n", "pts_time", "fps", "frame_idx"])
            for c in candidates:
                writer.writerow([c.keyframe_n, f"{c.pts_time_s:.3f}", f"{fps:.1f}", c.frame_idx])
        print(f"   ✓ Generated canonical map-keyframes CSV: {map_csv_path}", flush=True)

        # 4. Extract High-Quality JPEG Frames
        print(f"🖼️ [3/4] Extracting {len(candidates)} high-res JPEG keyframes via FFmpeg...", flush=True)
        extracted_frames: list[tuple[Any, Path]] = []
        for c in candidates:
            img_name = f"{c.keyframe_n:03d}.jpg"
            img_path = frames_dir / img_name
            if not img_path.exists():
                extract_frame(
                    video_path=video_path,
                    pts_time_s=c.pts_time_s,
                    output_path=img_path,
                    jpeg_quality=2,
                )
            extracted_frames.append((c, img_path))
        print("   ✓ Keyframe JPEG extraction completed!", flush=True)

        # 5. Multi-Modal Enrichment (SigLIP-2 + OCR + SAM + DAM)
        print(f"\n🔮 [4/4] Multi-Modal Enrichment for {len(extracted_frames)} Frames (SigLIP-2, OCR, SAM, DAM-3B)...", flush=True)
        filter_cfg = FilterConfig(
            minimum_score=score_threshold,
            minimum_area_ratio=0.005,
            maximum_area_ratio=0.85,
            same_class_iou=0.45,
            cross_label_duplicate_iou=0.60,
            maximum_regions=max_regions_per_frame,
        )

        records: list[UnifiedFrameRecord] = []
        for idx, (cand, img_path) in enumerate(extracted_frames, start=1):
            f_start = time.monotonic()
            with Image.open(img_path) as pil_img:
                image_rgb = pil_img.convert("RGB")
                img_w, img_h = image_rgb.size

            # A. SigLIP-2 Embedding
            siglip_vec: list[float] | None = None
            if self.siglip_encoder is not None:
                vec_np = self.siglip_encoder.encode_images([image_rgb])
                if len(vec_np) > 0:
                    siglip_vec = vec_np[0].tolist()

            # B. OCR Extraction
            ocr_res = UnifiedOcrResult(full_text="")
            if self.ocr_reader is not None:
                raw_ocr = self.ocr_reader.extract(image_rgb, image_path=img_path)
                ocr_spans = [
                    UnifiedOcrSpan(
                        line_id=s.line_id,
                        raw_text=s.raw_text,
                        normalized_text=s.normalized_text,
                        confidence=s.confidence,
                        polygon_xy=s.polygon_xy,
                        normalized_polygon_xy=s.normalized_polygon_xy,
                    )
                    for s in raw_ocr.spans
                ]
                ocr_res = UnifiedOcrResult(full_text=raw_ocr.full_text, spans=ocr_spans)

            # C. SAM Automatic Object Segmentation & DAM-3B Dense Descriptions (No bounding boxes needed)
            dam_captions: list[DamRegionCaption] = []
            if self.dam_captioner is not None and self.sam_generator is not None:
                auto_masks = self.sam_generator.generate_automatic_masks(
                    image=image_rgb,
                    max_regions=max_regions_per_frame,
                    min_area_ratio=0.005,
                    max_area_ratio=0.85,
                )

                # Explicitly flush temporary SAM tensors before DAM dense captioning
                try:
                    import gc
                    import torch

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                for r_idx, (mask_bool, bbox_xyxy, iou_score) in enumerate(auto_masks, start=1):
                    caption_result = self.dam_captioner.describe_region(
                        image=image_rgb,
                        mask=mask_bool,
                        bbox_xyxy_px=bbox_xyxy,
                        class_entity="object",
                        max_words=maximum_words,
                    )
                    if caption_result.status == "ok" and caption_result.description_en:
                        dam_captions.append(
                            DamRegionCaption(
                                region_id=f"reg_{r_idx:03d}",
                                class_label="object",
                                bbox_xyxy_px=bbox_xyxy,
                                sam_iou=float(iou_score) if iou_score is not None else 0.90,
                                caption_en=caption_result.description_en,
                                word_count=caption_result.word_count,
                            )
                        )

                # Light defragmentation after frame captioning completes
                try:
                    import gc
                    import torch

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            # Build Canonical Unified Frame Record
            frame_relpath = f"frames/{video_id}/{cand.keyframe_n:03d}.jpg"
            record = UnifiedFrameRecord(
                video_id=video_id,
                frame_uid=f"{video_id}:{cand.frame_idx}",
                keyframe_n=cand.keyframe_n,
                frame_idx=cand.frame_idx,
                pts_time_s=cand.pts_time_s,
                fps=fps,
                shot_id=cand.shot_id,
                image_relpath=frame_relpath,
                siglip_embedding=siglip_vec,
                ocr=ocr_res,
                dam_descriptions=dam_captions,
            )
            records.append(record)
            ocr_text_preview = ocr_res.full_text[:35] + "..." if len(ocr_res.full_text) > 35 else (ocr_res.full_text or "<no text>")
            f_elapsed = time.monotonic() - f_start
            print(
                f"  ▶ [{idx:02d}/{len(extracted_frames):02d}] Keyframe #{cand.keyframe_n} (idx:{cand.frame_idx:05d} @ {cand.pts_time_s:6.2f}s) "
                f"| OCR: {ocr_text_preview!r} | DAM: {len(dam_captions)} objects | {f_elapsed:.2f}s",
                flush=True,
            )

        # 6. Save Unified JSONL
        with open(unified_jsonl_path, "w", encoding="utf-8") as f_jsonl:
            for rec in records:
                f_jsonl.write(json.dumps(rec.model_dump(mode="json"), ensure_ascii=False) + "\n")

        total_elapsed = time.monotonic() - started_at
        print(f"\n💾 Saved {len(records)} records to {unified_jsonl_path}", flush=True)
        print(f"✅ Finished {video_id} in {total_elapsed:.2f}s (Avg {total_elapsed/max(1, len(records)):.2f}s/frame)\n", flush=True)
        return map_csv_path, unified_jsonl_path, records
