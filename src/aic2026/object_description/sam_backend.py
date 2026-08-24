"""Lazy Hugging Face SAM backend using one image encoding for all frame boxes."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .rle import rectangle_mask

SAM_MODEL_ID = "facebook/sam-vit-base"
SAM_REVISION = "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f"


@dataclass(frozen=True, slots=True)
class MaskPrediction:
    mask: np.ndarray
    source: str
    iou_score: float | None


class BboxMaskGenerator:
    """Direct, instant bounding box mask generator (zero GPU overhead)."""
    def generate(
        self, image: Image.Image, boxes_xyxy: list[tuple[int, int, int, int]]
    ) -> list[MaskPrediction]:
        width, height = image.size
        return [
            MaskPrediction(
                mask=rectangle_mask(height, width, box),
                source="bbox_fallback",
                iou_score=None,
            )
            for box in boxes_xyxy
        ]


SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CHECKPOINT_FILENAME = "sam_vit_b_01ec64.pth"


class SamMaskGenerator:
    def __init__(
        self,
        backend_type: str,
        predictor_or_processor: Any,
        model: Any = None,
        auto_mask_generator: Any = None,
        device: str = "cuda",
    ) -> None:
        self.backend_type = backend_type
        self.predictor = predictor_or_processor  # For segment_anything: SamPredictor; for HF: processor
        self.model = model
        self.auto_mask_generator = auto_mask_generator
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = SAM_MODEL_ID,
        revision: str = SAM_REVISION,
        cache_dir: Path | None = None,
        device: str = "cuda",
    ) -> SamMaskGenerator:
        # 1. Try Meta's official segment_anything library (Fastest, zero transformers/TF recursion)
        try:
            from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

            target_dir = Path(cache_dir or os.environ.get("AIC_MODEL_CACHE", "aic2026-model-cache/sam"))
            target_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_file = target_dir / SAM_CHECKPOINT_FILENAME
            if not checkpoint_file.exists() or checkpoint_file.stat().st_size < 100_000_000:
                print(f"📥 Downloading Meta SAM ViT-B checkpoint to {checkpoint_file} ...", flush=True)
                import urllib.request

                urllib.request.urlretrieve(SAM_CHECKPOINT_URL, checkpoint_file)
                print("✓ SAM checkpoint downloaded successfully!", flush=True)

            sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint_file)).to(device)
            sam.eval()
            predictor = SamPredictor(sam)
            return cls(
                backend_type="meta",
                predictor_or_processor=predictor,
                model=sam,
                device=device,
            )
        except ImportError:
            pass

        # 2. Fallback to Hugging Face transformers
        import os

        os.environ["USE_TF"] = "0"
        os.environ["USE_FLAX"] = "0"
        os.environ["USE_TORCH"] = "1"

        try:
            import transformers.utils.import_utils as _t_import

            _t_import._scipy_available = False
            _t_import.is_scipy_available = lambda: False
            _t_import._tf_available = False
            _t_import.is_tf_available = lambda: False
            _t_import._flax_available = False
            _t_import.is_flax_available = lambda: False
            _t_import._sklearn_available = False
            _t_import.is_sklearn_available = lambda: False
            _t_import._torchvision_available = False
            _t_import.is_torchvision_available = lambda: False
        except Exception:
            pass

        try:
            from transformers.models.sam.modeling_sam import SamModel as MaskModelClass
            from transformers.models.sam.processing_sam import SamProcessor as MaskProcessorClass
        except (ImportError, AttributeError):
            try:
                from transformers import AutoModelForMaskGeneration as MaskModelClass
                from transformers import AutoProcessor as MaskProcessorClass
            except ImportError as error:
                raise RuntimeError(
                    "segment_anything or transformers is required; run: pip install git+https://github.com/facebookresearch/segment-anything.git"
                ) from error
        local_files_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        processor = MaskProcessorClass.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        model = MaskModelClass.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            trust_remote_code=False,
            local_files_only=local_files_only,
        ).to(device)
        model.eval()
        return cls(backend_type="hf", predictor_or_processor=processor, model=model, device=device)

    def generate_automatic_masks(
        self,
        image: Image.Image,
        max_regions: int = 3,
        min_area_ratio: float = 0.005,
        max_area_ratio: float = 0.85,
    ) -> list[tuple[np.ndarray, tuple[int, int, int, int], float]]:
        """Discover salient objects via SamPredictor with candidate anchor boxes (100% CUDA safe)."""
        width, height = image.size
        img_area = float(width * height)

        # Generate salient anchor boxes covering center, left, right, and top focal regions
        anchor_boxes: list[tuple[int, int, int, int]] = [
            # 1. Center focal region (prominent subject)
            (int(0.12 * width), int(0.12 * height), int(0.88 * width), int(0.88 * height)),
            # 2. Left half subject
            (int(0.05 * width), int(0.15 * height), int(0.55 * width), int(0.85 * height)),
            # 3. Right half subject
            (int(0.45 * width), int(0.15 * height), int(0.95 * width), int(0.85 * height)),
            # 4. Upper center (faces, signage, upper body)
            (int(0.20 * width), int(0.05 * height), int(0.80 * width), int(0.60 * height)),
            # 5. Lower center (ground objects, vehicles, items)
            (int(0.15 * width), int(0.40 * height), int(0.85 * width), int(0.95 * height)),
        ]

        if self.backend_type == "meta":
            rgb_np = np.array(image.convert("RGB"))
            try:
                import torch

                with torch.inference_mode():
                    self.predictor.set_image(rgb_np)
                    candidates: list[tuple[np.ndarray, tuple[int, int, int, int], float, float]] = []
                    for box in anchor_boxes:
                        try:
                            masks, scores, _ = self.predictor.predict(
                                box=np.array(box),
                                multimask_output=False,
                            )
                            seg = np.asarray(masks[0], dtype=bool)
                            score = float(scores[0])
                            if not seg.any() or not math.isfinite(score):
                                continue

                            area = float(seg.sum())
                            ratio = area / img_area
                            if ratio < min_area_ratio or ratio > max_area_ratio:
                                continue

                            # Derive tight bounding box from the actual SAM silhouette
                            ys, xs = np.where(seg)
                            x1, y1 = int(xs.min()), int(ys.min())
                            x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
                            if x2 <= x1 or y2 <= y1:
                                continue

                            candidates.append((seg, (x1, y1, x2, y2), score, area))
                        except Exception:
                            continue

                # Sort by quality score descending
                candidates.sort(key=lambda item: item[2], reverse=True)
                selected: list[tuple[np.ndarray, tuple[int, int, int, int], float, float]] = []
                for seg, box, score, area in candidates:
                    overlap = False
                    for _, s_box, _, _ in selected:
                        ix1 = max(box[0], s_box[0])
                        iy1 = max(box[1], s_box[1])
                        ix2 = min(box[2], s_box[2])
                        iy2 = min(box[3], s_box[3])
                        if ix2 > ix1 and iy2 > iy1:
                            inter = (ix2 - ix1) * (iy2 - iy1)
                            b1_a = (box[2] - box[0]) * (box[3] - box[1])
                            b2_a = (s_box[2] - s_box[0]) * (s_box[3] - s_box[1])
                            iou = inter / float(b1_a + b2_a - inter)
                            if iou > 0.50:
                                overlap = True
                                break
                    if not overlap:
                        selected.append((seg, box, score, area))
                    if len(selected) >= max_regions:
                        break

                if selected:
                    return [(seg, box, score) for seg, box, score, _ in selected]
            except Exception:
                pass

        elif self.backend_type == "hf":
            preds = self.generate(image, anchor_boxes)
            selected_hf = []
            for pred, box in zip(preds, anchor_boxes):
                if pred.source == "sam" and pred.mask.any():
                    ys, xs = np.where(pred.mask)
                    x1, y1 = int(xs.min()), int(ys.min())
                    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
                    selected_hf.append((pred.mask, (x1, y1, x2, y2), pred.iou_score or 0.90))
                    if len(selected_hf) >= max_regions:
                        break
            if selected_hf:
                return selected_hf

        # Fallback: Focal scene region description if no salient sub-objects passed threshold
        full_box = (int(0.05 * width), int(0.05 * height), int(0.95 * width), int(0.95 * height))
        from .rle import rectangle_mask

        fallback_mask = rectangle_mask(height, width, full_box)
        return [(fallback_mask, full_box, 1.0)]

    def generate(
        self, image: Image.Image, boxes_xyxy: list[tuple[int, int, int, int]]
    ) -> list[MaskPrediction]:
        if not boxes_xyxy:
            return []
        
        width, height = image.size
        
        # Branch A: Meta official segment_anything (Fast & Direct)
        if self.backend_type == "meta":
            rgb_np = np.array(image.convert("RGB"))
            self.predictor.set_image(rgb_np)
            predictions: list[MaskPrediction] = []
            for box in boxes_xyxy:
                fallback = rectangle_mask(height, width, box)
                try:
                    masks, scores, _ = self.predictor.predict(
                        box=np.array(box),
                        multimask_output=False,
                    )
                    mask = np.asarray(masks[0], dtype=bool)
                    score = float(scores[0])
                    if mask.shape != (height, width) or not mask.any() or not math.isfinite(score):
                        raise ValueError("Invalid mask")
                    predictions.append(MaskPrediction(mask=mask, source="sam", iou_score=score))
                except Exception:
                    predictions.append(MaskPrediction(mask=fallback, source="bbox_fallback", iou_score=None))
            return predictions

        # Branch B: Hugging Face transformers
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required to run SAM") from error

        rgb = image.convert("RGB")
        inputs = self.predictor(
            images=rgb,
            input_boxes=[[list(box) for box in boxes_xyxy]],
            return_tensors="pt",
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        masks_by_image = self.predictor.image_processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
        )
        masks = masks_by_image[0]
        scores = outputs.iou_scores.detach().cpu()

        predictions: list[MaskPrediction] = []
        for index, box in enumerate(boxes_xyxy):
            fallback = rectangle_mask(height, width, box)
            try:
                box_masks = masks[index]
                box_scores = scores[0, index]
                best = int(torch.argmax(box_scores).item())
                mask = np.asarray(box_masks[best], dtype=bool)
                score = float(box_scores[best].item())
                if mask.shape != (height, width) or not mask.any() or not math.isfinite(score):
                    raise ValueError("SAM returned an invalid mask or quality score")
                predictions.append(MaskPrediction(mask=mask, source="sam", iou_score=score))
            except (IndexError, TypeError, ValueError):
                predictions.append(
                    MaskPrediction(mask=fallback, source="bbox_fallback", iou_score=None)
                )
        return predictions
