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


class SamMaskGenerator:
    def __init__(self, processor: Any, model: Any, device: str) -> None:
        self.processor = processor
        self.model = model
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
        # Cleanly disable sklearn check in transformers to avoid broken scipy recursion on Python 3.12
        import sys
        if sys.modules.get("sklearn") is None:
            sys.modules.pop("sklearn", None)
        if sys.modules.get("sklearn.metrics") is None:
            sys.modules.pop("sklearn.metrics", None)

        try:
            import transformers.utils.import_utils as _t_import
            _t_import.is_sklearn_available = lambda: False
            import transformers.utils as _t_utils
            _t_utils.is_sklearn_available = lambda: False
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
                    "transformers with SAM support is required; install requirements/runtime.txt"
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
        return cls(processor=processor, model=model, device=device)

    def generate(
        self, image: Image.Image, boxes_xyxy: list[tuple[int, int, int, int]]
    ) -> list[MaskPrediction]:
        if not boxes_xyxy:
            return []
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required to run SAM") from error

        rgb = image.convert("RGB")
        width, height = rgb.size
        inputs = self.processor(
            images=rgb,
            input_boxes=[[list(box) for box in boxes_xyxy]],
            return_tensors="pt",
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        masks_by_image = self.processor.image_processor.post_process_masks(
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
