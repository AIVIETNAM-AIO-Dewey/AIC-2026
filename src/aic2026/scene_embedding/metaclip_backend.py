"""MetaCLIP 2 visual scene embedding backend.

Produces 1024-dimensional L2-normalized visual unit vectors using
facebook/metaclip-2-worldwide-huge-quickgelu (or any MetaCLIP model).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

DEFAULT_METACLIP2_ID = "facebook/metaclip-2-worldwide-huge-quickgelu"


def unwrap_pooler_output(value: Any) -> Any:
    """Support both older and newer Transformers feature return types."""
    if hasattr(value, "pooler_output") and value.pooler_output is not None:
        return value.pooler_output
    if hasattr(value, "image_embeds") and value.image_embeds is not None:
        return value.image_embeds
    return value


class MetaClipEncoder:
    """MetaCLIP 2 vision embedding encoder."""

    def __init__(
        self,
        model_id: str = DEFAULT_METACLIP2_ID,
        device: str = "cuda",
        cache_dir: Path | None = None,
        compute_dtype: str = "float16",
    ) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.model_id = model_id

        local_files_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=local_files_only,
        )
        torch_dtype = getattr(torch, compute_dtype if self.device != "cpu" else "float32")
        self.model = AutoModel.from_pretrained(
            model_id,
            dtype=torch_dtype,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = DEFAULT_METACLIP2_ID,
        device: str = "cuda",
        cache_dir: Path | None = None,
        compute_dtype: str = "float16",
    ) -> MetaClipEncoder:
        return cls(
            model_id=model_id,
            device=device,
            cache_dir=cache_dir,
            compute_dtype=compute_dtype,
        )

    @property
    def embedding_dim(self) -> int:
        if hasattr(self.model.config, "projection_dim") and self.model.config.projection_dim:
            return int(self.model.config.projection_dim)
        if hasattr(self.model.config, "vision_config") and hasattr(self.model.config.vision_config, "hidden_size"):
            return int(self.model.config.vision_config.hidden_size)
        return 1024

    @property
    def preprocessing(self) -> dict[str, Any]:
        return {"source": "Hugging Face AutoProcessor", "model_id": self.model_id}

    def encode_images(self, images: list[Image.Image] | Sequence[Image.Image]) -> np.ndarray:
        """Return `[len(images), 1024]` float32 L2-normalized unit vectors."""
        if not images:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        torch = self.torch
        inputs = self.processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)
        if self.device != "cpu":
            pixel_values = pixel_values.half()

        with torch.inference_mode(), torch.autocast(
            device_type=self.device,
            dtype=torch.float16,
            enabled=self.device != "cpu",
        ):
            outputs = self.model.get_image_features(pixel_values=pixel_values)
            features = unwrap_pooler_output(outputs)
            features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)

        return features.cpu().numpy().astype(np.float32, copy=False)

    def encode(self, images: Sequence[Any]) -> np.ndarray:
        return self.encode_images(list(images))
