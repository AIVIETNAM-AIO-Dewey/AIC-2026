"""Lazy Hugging Face SigLIP2 backend pairing the image and text towers of one revision."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SIGLIP_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"

# SigLIP was trained with the text tower padded to a fixed width. Dynamic padding
# silently shifts the text embedding, so the tokenizer call must pin it.
TEXT_PADDING = "max_length"

COMPUTE_DTYPES = {"float32", "float16", "bfloat16"}


class SiglipEncoder:
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
        device: str = "mps",
        compute_dtype: str = "float32",
    ) -> SiglipEncoder:
        if compute_dtype not in COMPUTE_DTYPES:
            raise ValueError(f"Unsupported compute dtype: {compute_dtype!r}")
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "transformers with SigLIP2 support (>=4.49) and PyTorch are required"
            ) from error
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device.startswith("mps") and compute_dtype == "bfloat16":
            raise ValueError("MPS does not support bfloat16; use float32 or float16")

        local_files_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            trust_remote_code=False,
            local_files_only=local_files_only,
            # Pinned rather than left to the transformers default, which has flipped
            # between releases and produces slightly different pixel values.
            use_fast=True,
        )
        model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        # Cast after loading rather than through from_pretrained: transformers renamed
        # that keyword (`torch_dtype` -> `dtype`) across 4.x/5.x, this spelling is stable.
        model = model.to(device=device, dtype=getattr(torch, compute_dtype))
        model.eval()
        return cls(processor=processor, model=model, device=device)

    @property
    def embedding_dim(self) -> int:
        return int(self.model.config.vision_config.hidden_size)

    @staticmethod
    def _to_unit_rows(features: Any) -> np.ndarray:
        # Cast before normalizing: fp16 norms lose precision that the index keeps forever.
        import torch

        values = features.detach().to("cpu", dtype=torch.float32).numpy()
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if not (norms > 0).all():
            raise ValueError("SigLIP2 returned a zero-length embedding")
        return values / norms

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Return `[len(images), D]` float32 unit vectors."""
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
        """Return `[len(texts), D]` float32 unit vectors for the online side."""
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
