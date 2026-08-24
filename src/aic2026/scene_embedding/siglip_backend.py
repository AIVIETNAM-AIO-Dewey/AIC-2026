"""Hugging Face SigLIP-2 Backend producing 768-dim L2-normalized visual unit vectors."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SIGLIP_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"

TEXT_PADDING = "max_length"
COMPUTE_DTYPES = {"float32", "float16", "bfloat16"}


class SiglipEncoder:
    """SigLIP-2 vision and text embedding encoder."""

    def __init__(self, processor: Any, model: Any, device: str) -> None:
        self.processor = processor
        self.model = model
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = SIGLIP_MODEL_ID,
        revision: str = SIGLIP_REVISION,
        cache_dir: Path | None = None,
        device: str = "cuda",
        compute_dtype: str = "float32",
    ) -> SiglipEncoder:
        if compute_dtype not in COMPUTE_DTYPES:
            raise ValueError(f"Unsupported compute dtype: {compute_dtype!r}")
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "transformers (>=4.49) and PyTorch are required for SigLIP-2"
            ) from error

        local_files_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        target_device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"

        processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            trust_remote_code=False,
            local_files_only=local_files_only,
            use_fast=True,
        )
        model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        model = model.to(device=target_device, dtype=getattr(torch, compute_dtype if target_device != "cpu" else "float32"))
        model.eval()
        return cls(processor=processor, model=model, device=target_device)

    @property
    def embedding_dim(self) -> int:
        return int(self.model.config.vision_config.hidden_size)

    @staticmethod
    def _to_unit_rows(features: Any) -> np.ndarray:
        import torch

        values = features.detach().to("cpu", dtype=torch.float32).numpy()
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return values / norms

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Return `[len(images), 768]` float32 unit vectors."""
        if not images:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        import torch

        inputs = self.processor(
            images=[image.convert("RGB") for image in images], return_tensors="pt"
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            features = self.model.get_image_features(**inputs)
        return self._to_unit_rows(features)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Return `[len(texts), 768]` float32 unit vectors for queries."""
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        import torch

        inputs = self.processor(
            text=texts,
            padding=TEXT_PADDING,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        return self._to_unit_rows(features)
