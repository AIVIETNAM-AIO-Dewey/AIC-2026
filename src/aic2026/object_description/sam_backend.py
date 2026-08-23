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
    def __init__(self, backend_type: str, predictor_or_processor: Any, model: Any = None, device: str = "cuda") -> None:
        self.backend_type = backend_type
        self.predictor = predictor_or_processor  # For segment_anything: SamPredictor; for HF: processor
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
        # 1. Try Meta's official segment_anything library (Fastest, zero transformers/TF recursion)
        try:
            from segment_anything import sam_model_registry, SamPredictor
            target_dir = Path(cache_dir or "/kaggle/working/aic2026-model-cache/sam")
            target_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_file = target_dir / SAM_CHECKPOINT_FILENAME
            if not checkpoint_file.exists() or checkpoint_file.stat().st_size < 100_000_000:
                print(f"📥 Downloading Meta SAM ViT-B checkpoint to {checkpoint_file} ...")
                import urllib.request
                urllib.request.urlretrieve(SAM_CHECKPOINT_URL, checkpoint_file)
                print("✓ SAM checkpoint downloaded successfully!")
            
            sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint_file)).to(device)
            sam.eval()
            predictor = SamPredictor(sam)
            return cls(backend_type="meta", predictor_or_processor=predictor, device=device)
        except ImportError:
            pass

        # 2. Fallback to Hugging Face transformers
        import os
        os.environ["USE_TF"] = "0"
        os.environ["USE_FLAX"] = "0"
        os.environ["USE_TORCH"] = "1"

        try:
            import transformers.utils.import_utils as _t_import
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
