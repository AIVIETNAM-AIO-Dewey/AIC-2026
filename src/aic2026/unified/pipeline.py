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
    TransNetV2,
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
        transnet_model: TransNetV2 | None = None,
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
        transnet = None
        if load_transnet:
            print(f"[pipeline] Loading TransNetV2 on {device}...", file=sys.stderr, flush=True)
            transnet = load_transnetv2_model(device=device)

        siglip = None
        if load_siglip:
            print(f"[pipeline] Loading SigLIP-2 on {device}...", file=sys.stderr, flush=True)
            siglip = SiglipEncoder.from_pretrained(device=device)

        ocr = None
        if load_ocr:
            print(f"[pipeline] Loading OCR Reader on {device}...", file=sys.stderr, flush=True)
            ocr = OcrReader.create(device=device)

        sam = None
        dam = None
        if load_sam_dam:
            print(f"[pipeline] Loading Meta SAM ViT-B on {device}...", file=sys.stderr, flush=True)
            sam = SamMaskGenerator.from_pretrained(device=device)
            print(f"[pipeline] Loading DAM-3B on {device}...", file=sys.stderr, flush=True)
            dam = DamCaptioner.from_pretrained(
                model_id=dam_model_id,
                revision=dam_revision,
                code_revision=dam_code_revision,
            )

        print("✓ All requested multi-modal models initialized successfully!", file=sys.stderr, flush=True)
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

        print(f"\n{'='*75}", file=sys.stderr, flush=True)
        print(f"🎬 Processing Video: {video_id} ({probe.duration_s:.1f}s, {fps:.2f} fps, {probe.width}x{probe.height})", file=sys.stderr, flush=True)
        print(f"{'='*75}", file=sys.stderr, flush=True)

        # 1. TransNetV2 Shot Detection
        print(f"⚡ [1/5] Running TransNetV2 Shot Detection on {video_id}...", file=sys.stderr, flush=True)
        t_shot_start = time.monotonic()
        transnet_res = run_transnetv2_inference(
            video_path=video_path,
            video_id=video_id,
            fps=fps,
            model=self.transnet_model,
            device=self.device,
        )
        shots = transnet_res.shots
        print(f"✓ Detected {len(shots)} shots in {time.monotonic() - t_shot_start:.2f}s", file=sys.stderr, flush=True)

        # 2. Adaptive Keyframe Sampling
        print("🎯 [2/5] Applying Adaptive Keyframe Sampling Policy...", file=sys.stderr, flush=True)
        raw_samples = adaptive_samples_from_shots(shots)
        candidates = dedupe_samples(raw_samples, tolerance_s=0.5)
        if max_frames is not None and max_frames > 0:
            candidates = candidates[:max_frames]
        print(f"✓ Selected {len(candidates)} canonical keyframes for extraction", file=sys.stderr, flush=True)

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
        print(f"✓ Generated canonical map-keyframes CSV: {map_csv_path}", file=sys.stderr, flush=True)

        # 4. Extract High-Quality JPEG Frames
        print(f"🖼️ [3/5] Extracting {len(candidates)} JPEG keyframes via FFmpeg...", file=sys.stderr, flush=True)
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
        print("✓ Keyframe JPEG extraction completed!", file=sys.stderr, flush=True)

        # 5. Multi-Modal Enrichment (SigLIP-2 + OCR + SAM + DAM)
        print("🔮 [4/5] Running Multi-Modal Enrichment (SigLIP-2, OCR, SAM, DAM-3B)...", file=sys.stderr, flush=True)
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

            # C. SAM & DAM-3B Dense Descriptions
            dam_captions: list[DamRegionCaption] = []
            if self.dam_captioner is not None and self.sam_generator is not None:
                # Check for organizer detection JSON matching keyframe_n
                matched_dets = []
                if objects_dir and objects_dir.exists():
                    det_json = objects_dir / f"{cand.keyframe_n:06d}.json"
                    if not det_json.exists():
                        det_json = objects_dir / f"{cand.keyframe_n}.json"
                    if det_json.exists():
                        raw_dets = load_organizer_detections(det_json)
                        matched_dets = filter_detections(raw_dets, filter_cfg)

                for r_idx, det in enumerate(matched_dets, start=1):
                    # SAM segmentation
                    boxes = [list(det.bbox_xyxy_px)]
                    mask_preds = self.sam_generator.generate_masks(image_rgb, boxes)
                    if not mask_preds:
                        continue
                    pred = mask_preds[0]

                    # DAM caption
                    caption_result = self.dam_captioner.describe_region(
                        image=image_rgb,
                        mask=pred.mask_bool,
                        bbox_xyxy_px=det.bbox_xyxy_px,
                        class_entity=det.class_entity,
                        max_words=maximum_words,
                    )
                    if caption_result.status == "ok" and caption_result.description_en:
                        dam_captions.append(
                            DamRegionCaption(
                                region_id=f"reg_{r_idx:03d}",
                                class_label=det.class_entity,
                                bbox_xyxy_px=det.bbox_xyxy_px,
                                sam_iou=float(pred.iou_score) if pred.iou_score is not None else None,
                                caption_en=caption_result.description_en,
                                word_count=caption_result.word_count,
                            )
                        )

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
            print(
                f"  [{idx}/{len(extracted_frames)}] Frame #{cand.keyframe_n} (idx:{cand.frame_idx}, {cand.pts_time_s:.2f}s) "
                f"| OCR: {ocr_text_preview!r} | DAM Objects: {len(dam_captions)} | Elapsed: {time.monotonic() - f_start:.2f}s",
                file=sys.stderr,
                flush=True,
            )

        # 6. Save Unified JSONL
        print(f"💾 [5/5] Writing {len(records)} unified frame records to {unified_jsonl_path}...", file=sys.stderr, flush=True)
        with open(unified_jsonl_path, "w", encoding="utf-8") as f_jsonl:
            for rec in records:
                f_jsonl.write(json.dumps(rec.model_dump(mode="json"), ensure_ascii=False) + "\n")

        total_elapsed = time.monotonic() - started_at
        print(f"✅ COMPLETED {video_id} in {total_elapsed:.2f}s (Avg {total_elapsed/max(1, len(records)):.2f}s/frame)\n", file=sys.stderr, flush=True)
        return map_csv_path, unified_jsonl_path, records
