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
    """End-to-end multi-modal video extractor and enricher.

    Supports both:
    1. Staged mode (default): Loads models on-demand one by one and frees GPU memory between stages.
       This ensures maximum VRAM headroom (~15GB available for each model) and prevents CUDA context corruption.
    2. Preloaded mode: Uses explicitly provided in-memory model instances (useful for testing and fast single-frame pipelines).
    """

    def __init__(
        self,
        transnet_model: Any | None = None,
        siglip_encoder: SiglipEncoder | None = None,
        ocr_reader: OcrReader | None = None,
        sam_generator: SamMaskGenerator | None = None,
        dam_captioner: DamCaptioner | None = None,
        device: str = "cuda",
        staged: bool = True,
        load_transnet: bool = True,
        load_siglip: bool = True,
        load_ocr: bool = True,
        load_sam_dam: bool = True,
        dam_model_id: str = "nvidia/DAM-3B",
        dam_revision: str = "0797bedd98d645cd021379a4661ee233da279bba",
        dam_code_revision: str = "153ad3d33c29324e9197f565547c6bc8500da02d",
    ) -> None:
        self.device = device
        self.transnet_model = transnet_model
        self.siglip_encoder = siglip_encoder
        self.ocr_reader = ocr_reader
        self.sam_generator = sam_generator
        self.dam_captioner = dam_captioner
        self.staged = staged
        self.load_transnet = load_transnet
        self.load_siglip = load_siglip
        self.load_ocr = load_ocr
        self.load_sam_dam = load_sam_dam
        self.dam_model_id = dam_model_id
        self.dam_revision = dam_revision
        self.dam_code_revision = dam_code_revision

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
        staged: bool = True,
    ) -> UnifiedVideoPipeline:
        """Create the pipeline in staged execution mode (models loaded on-demand one-by-one)."""
        print(f"\n{'='*75}", flush=True)
        print(f"🚀 INITIALIZING UNIFIED MULTI-MODAL PIPELINE (MODE: {'STAGED / SEQUENTIAL' if staged else 'PRELOADED'}) ON: {device.upper()}", flush=True)
        print(f"{'='*75}", flush=True)
        print("  ✓ Stage 1: TransNetV2 Shot Detection (On-Demand)", flush=True)
        if load_siglip:
            print("  ✓ Stage 2: SigLIP-2 Visual Embedding (On-Demand)", flush=True)
        if load_ocr:
            print("  ✓ Stage 3: OCR Text Extraction & Normalization (On-Demand)", flush=True)
        if load_sam_dam:
            print("  ✓ Stage 4: Meta SAM Object Segmentation (On-Demand)", flush=True)
            print("  ✓ Stage 5: NVIDIA DAM-3B Dense Descriptions (On-Demand)", flush=True)
        print(f"{'='*75}\n", flush=True)

        return cls(
            device=device,
            staged=staged,
            load_transnet=load_transnet,
            load_siglip=load_siglip,
            load_ocr=load_ocr,
            load_sam_dam=load_sam_dam,
            dam_model_id=dam_model_id,
            dam_revision=dam_revision,
            dam_code_revision=dam_code_revision,
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
        import gc
        import torch

        def _cleanup_gpu() -> None:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        started_at = time.monotonic()
        video_path = video_path.resolve()
        probe = probe_video(video_path)
        fps = probe.fps

        print(f"🎬 Target Video: {video_id}", flush=True)
        print(f"   • Path:     {video_path}", flush=True)
        print(f"   • Duration: {probe.duration_s:.1f}s | FPS: {fps:.2f} | Resolution: {probe.width}x{probe.height}", flush=True)

        # ─────────────────────────────────────────────────────────────
        # STAGE 1: TransNetV2 Shot Detection
        # ─────────────────────────────────────────────────────────────
        print(f"\n⚡ [1/6] Running TransNetV2 Shot Boundary Detection...", flush=True)
        t_shot_start = time.monotonic()
        transnet = self.transnet_model
        if transnet is None and self.load_transnet:
            print("   ⏳ Loading TransNetV2 into GPU memory...", flush=True)
            transnet = load_transnetv2_model(device=self.device)

        transnet_res = run_transnetv2_inference(
            video_path=video_path,
            video_id=video_id,
            fps=fps,
            model=transnet,
            device=self.device,
        )
        shots = transnet_res.shots
        print(f"   ✓ TransNetV2 completed in {time.monotonic() - t_shot_start:.2f}s (Detected {len(shots)} distinct scene shots)", flush=True)

        if self.staged and self.transnet_model is None:
            del transnet
            _cleanup_gpu()

        # ─────────────────────────────────────────────────────────────
        # STAGE 2: Adaptive Keyframe Sampling & FFmpeg Extraction
        # ─────────────────────────────────────────────────────────────
        print("🎯 [2/6] Applying Adaptive Keyframe Sampling Policy...", flush=True)
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

        # Write Canonical map-keyframes CSV
        with open(map_csv_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["n", "pts_time", "fps", "frame_idx"])
            for c in candidates:
                writer.writerow([c.keyframe_n, f"{c.pts_time_s:.3f}", f"{fps:.1f}", c.frame_idx])
        print(f"   ✓ Generated canonical map-keyframes CSV: {map_csv_path}", flush=True)

        # Extract High-Quality JPEG Frames via FFmpeg
        print(f"🖼️ [3/6] Extracting {len(candidates)} high-res JPEG keyframes via FFmpeg...", flush=True)
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

        # Load extracted images into memory for multi-modal stages
        frame_images: dict[int, Image.Image] = {}
        for c, img_path in extracted_frames:
            with Image.open(img_path) as im:
                frame_images[c.keyframe_n] = im.convert("RGB")

        # ─────────────────────────────────────────────────────────────
        # STAGE 3: SigLIP-2 Scene Embeddings & OCR Text Extraction
        # ─────────────────────────────────────────────────────────────
        siglip_embeddings: dict[int, list[float] | None] = {}
        ocr_results: dict[int, UnifiedOcrResult] = {}

        if self.load_siglip:
            print(f"\n🔮 [4/6] Running SigLIP-2 Embeddings & OCR Text Extraction...", flush=True)
            siglip = self.siglip_encoder
            if siglip is None:
                print("   ⏳ Loading SigLIP-2 into GPU memory...", flush=True)
                siglip = SiglipEncoder.from_pretrained(device=self.device)

            print(f"   ▶ Encoding {len(extracted_frames)} keyframes with SigLIP-2...", flush=True)
            for c, img_path in extracted_frames:
                vecs = siglip.encode_images([frame_images[c.keyframe_n]])
                siglip_embeddings[c.keyframe_n] = vecs[0].tolist() if len(vecs) > 0 else None
            print("   ✓ SigLIP-2 embeddings generated!", flush=True)

            if self.staged and self.siglip_encoder is None:
                del siglip
                _cleanup_gpu()

        if self.load_ocr:
            ocr = self.ocr_reader
            if ocr is None:
                print("   ⏳ Loading OCR Reader into GPU memory...", flush=True)
                ocr = OcrReader.create(device=self.device)

            print(f"   ▶ Extracting OCR text from {len(extracted_frames)} keyframes...", flush=True)
            for c, img_path in extracted_frames:
                raw_ocr = ocr.extract(frame_images[c.keyframe_n], image_path=img_path)
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
                ocr_results[c.keyframe_n] = UnifiedOcrResult(full_text=raw_ocr.full_text, spans=ocr_spans)
            print("   ✓ OCR text extraction completed!", flush=True)

            if self.staged and self.ocr_reader is None:
                del ocr
                _cleanup_gpu()

        # ─────────────────────────────────────────────────────────────
        # STAGE 4: Meta SAM Object Segmentation
        # ─────────────────────────────────────────────────────────────
        all_masks: dict[int, list[tuple[np.ndarray, tuple[int, int, int, int], float]]] = {}
        if self.load_sam_dam:
            print(f"\n✂️ [5/6] Running Meta SAM ViT-B Object Segmentation...", flush=True)
            sam = self.sam_generator
            if sam is None:
                print("   ⏳ Loading Meta SAM ViT-B into GPU memory...", flush=True)
                sam = SamMaskGenerator.from_pretrained(device=self.device)

            for c, img_path in extracted_frames:
                masks = sam.generate_automatic_masks(
                    image=frame_images[c.keyframe_n],
                    max_regions=max_regions_per_frame,
                    min_area_ratio=0.005,
                    max_area_ratio=0.85,
                )
                all_masks[c.keyframe_n] = masks
            print(f"   ✓ SAM segmentation completed for {len(extracted_frames)} keyframes!", flush=True)

            if self.staged and self.sam_generator is None:
                del sam
                _cleanup_gpu()

        # ─────────────────────────────────────────────────────────────
        # STAGE 5: NVIDIA DAM-3B Dense Captioning (Isolated Subprocess)
        # ─────────────────────────────────────────────────────────────
        all_dam_captions: dict[int, list[DamRegionCaption]] = {}
        if self.load_sam_dam:
            print(f"\n🏷️ [6/6] Running NVIDIA DAM-3B Dense Descriptions (Isolated Process)...", flush=True)

            if self.dam_captioner is not None:
                # In-memory execution (for mock tests or preloaded instances)
                for c, img_path in extracted_frames:
                    frame_caps: list[DamRegionCaption] = []
                    frame_masks = all_masks.get(c.keyframe_n, [])
                    for r_idx, (mask_bool, bbox_xyxy, iou_score) in enumerate(frame_masks, start=1):
                        caption_result = self.dam_captioner.describe_region(
                            image=frame_images[c.keyframe_n],
                            mask=mask_bool,
                            bbox_xyxy_px=bbox_xyxy,
                            class_entity="object",
                            max_words=maximum_words,
                        )
                        if caption_result.status == "ok" and caption_result.description_en:
                            frame_caps.append(
                                DamRegionCaption(
                                    region_id=f"reg_{r_idx:03d}",
                                    class_label="object",
                                    bbox_xyxy_px=bbox_xyxy,
                                    sam_iou=float(iou_score) if iou_score is not None else 0.90,
                                    caption_en=caption_result.description_en,
                                    word_count=caption_result.word_count,
                                )
                            )
                    all_dam_captions[c.keyframe_n] = frame_caps
            else:
                # Execute in an isolated Python process to guarantee 100% fresh CUDA context
                import subprocess

                temp_dir = output_root / "_temp_dam"
                temp_dir.mkdir(parents=True, exist_ok=True)
                tasks_file = temp_dir / f"{video_id}_tasks.json"
                out_desc_file = temp_dir / f"{video_id}_descriptions.json"

                from aic2026.object_description.rle import encode_mask

                tasks_payload = []
                for c, img_path in extracted_frames:
                    frame_masks = all_masks.get(c.keyframe_n, [])
                    regions_payload = []
                    for r_idx, (mask_bool, bbox_xyxy, iou_score) in enumerate(frame_masks, start=1):
                        rle_data = encode_mask(mask_bool).model_dump(mode="json") if mask_bool is not None else None
                        regions_payload.append({
                            "region_id": f"reg_{r_idx:03d}",
                            "bbox_xyxy": list(bbox_xyxy) if bbox_xyxy else None,
                            "sam_iou": float(iou_score) if iou_score is not None else 0.90,
                            "class_label": "object",
                            "mask_rle": rle_data,
                        })
                    tasks_payload.append({
                        "keyframe_n": c.keyframe_n,
                        "image_path": str(img_path.resolve()),
                        "regions": regions_payload,
                    })

                with open(tasks_file, "w", encoding="utf-8") as f_tasks:
                    json.dump(tasks_payload, f_tasks, ensure_ascii=False)

                repo_root = Path(__file__).resolve().parents[3]
                cmd = [
                    sys.executable,
                    str(repo_root / "scripts/run_dam_isolated.py"),
                    "--input-json", str(tasks_file),
                    "--output-json", str(out_desc_file),
                    "--device", self.device,
                    "--max-words", str(maximum_words),
                    "--model-id", self.dam_model_id,
                    "--revision", self.dam_revision,
                    "--code-revision", self.dam_code_revision,
                ]

                sub_res = subprocess.run(cmd, capture_output=False, text=True)
                if sub_res.returncode != 0:
                    raise RuntimeError(f"Isolated DAM worker failed with exit code {sub_res.returncode}")

                if out_desc_file.exists():
                    with open(out_desc_file, "r", encoding="utf-8") as f_out:
                        raw_descs = json.load(f_out)
                    for key_str, caps_list in raw_descs.items():
                        kn = int(key_str)
                        all_dam_captions[kn] = [
                            DamRegionCaption(
                                region_id=cap["region_id"],
                                class_label=cap["class_label"],
                                bbox_xyxy_px=tuple(cap["bbox_xyxy_px"]) if cap.get("bbox_xyxy_px") else None,
                                sam_iou=cap.get("sam_iou", 0.90),
                                caption_en=cap["caption_en"],
                                word_count=cap.get("word_count", len(cap["caption_en"].split())),
                            )
                            for cap in caps_list
                            if cap.get("status") == "ok" and cap.get("caption_en")
                        ]

                # Cleanup temp files
                try:
                    if tasks_file.exists():
                        tasks_file.unlink()
                    if out_desc_file.exists():
                        out_desc_file.unlink()
                except Exception:
                    pass

            print(f"   ✓ DAM-3B descriptions generated for all {len(extracted_frames)} keyframes!", flush=True)

        # ─────────────────────────────────────────────────────────────
        # STAGE 6: Assemble Records & Serialize Output
        # ─────────────────────────────────────────────────────────────
        records: list[UnifiedFrameRecord] = []
        for cand, img_path in extracted_frames:
            frame_relpath = f"frames/{video_id}/{cand.keyframe_n:03d}.jpg"
            ocr_res = ocr_results.get(cand.keyframe_n, UnifiedOcrResult(full_text=""))
            dam_caps = all_dam_captions.get(cand.keyframe_n, [])
            sig_vec = siglip_embeddings.get(cand.keyframe_n, None)

            record = UnifiedFrameRecord(
                video_id=video_id,
                frame_uid=f"{video_id}:{cand.frame_idx}",
                keyframe_n=cand.keyframe_n,
                frame_idx=cand.frame_idx,
                pts_time_s=cand.pts_time_s,
                fps=fps,
                shot_id=cand.shot_id,
                image_relpath=frame_relpath,
                siglip_embedding=sig_vec,
                ocr=ocr_res,
                dam_descriptions=dam_caps,
            )
            records.append(record)

            ocr_preview = ocr_res.full_text[:35] + "..." if len(ocr_res.full_text) > 35 else (ocr_res.full_text or "<no text>")
            print(
                f"  ▶ Keyframe #{cand.keyframe_n:03d} (idx:{cand.frame_idx:05d} @ {cand.pts_time_s:6.2f}s) "
                f"| OCR: {ocr_preview!r} | DAM: {len(dam_caps)} objects",
                flush=True,
            )

        # Save Unified JSONL
        with open(unified_jsonl_path, "w", encoding="utf-8") as f_jsonl:
            for rec in records:
                f_jsonl.write(json.dumps(rec.model_dump(mode="json"), ensure_ascii=False) + "\n")

        total_elapsed = time.monotonic() - started_at
        print(f"\n💾 Saved {len(records)} records to {unified_jsonl_path}", flush=True)
        print(f"✅ Finished {video_id} in {total_elapsed:.2f}s (Avg {total_elapsed/max(1, len(records)):.2f}s/frame)\n", flush=True)
        return map_csv_path, unified_jsonl_path, records
