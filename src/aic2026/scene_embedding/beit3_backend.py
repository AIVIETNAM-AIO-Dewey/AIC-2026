"""BEiT-3 visual scene embedding backend.

Produces 768-dimensional L2-normalized visual unit vectors using
the official BEiT-3 Base COCO Retrieval 384x384 checkpoint.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import types
from collections import abc
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_BEIT3_CHECKPOINT = (
    "https://github.com/addf400/files/releases/download/beit3/"
    "beit3_base_patch16_384_coco_retrieval.pth"
)


def download_http_resumable(url: str, destination: Path) -> None:
    """Download a checkpoint via HTTP with resumable partial chunk support."""
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=60,
        allow_redirects=True,
    ) as response:
        if existing and response.status_code == 200:
            partial.unlink(missing_ok=True)
            existing = 0
        response.raise_for_status()
        mode = "ab" if existing and response.status_code == 206 else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(destination)


def ensure_unilm_repo(preferred_dir: Path | None = None) -> Path:
    """Ensure Microsoft unilm repo is cloned and accessible."""
    env_dir = Path(os.environ["UNILM_REPO_DIR"]).expanduser().resolve() if "UNILM_REPO_DIR" in os.environ else None
    candidates = [
        preferred_dir,
        env_dir,
        Path.cwd() / "unilm",
        Path.home() / ".cache" / "unilm",
    ]
    for c in candidates:
        if c is not None and (c / "beit3" / "modeling_finetune.py").is_file():
            return c / "beit3"
        if c is not None and (c / "modeling_finetune.py").is_file():
            return c

    # Clone to preferred_dir or cache
    target = preferred_dir or (Path.home() / ".cache" / "unilm")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        logger.info("Cloning Microsoft unilm repository to %s ...", target)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/microsoft/unilm.git", str(target)],
            check=True,
            capture_output=True,
        )

    beit3_dir = target / "beit3" if (target / "beit3").is_dir() else target
    if not (beit3_dir / "modeling_finetune.py").exists():
        raise FileNotFoundError(f"BEiT-3 source not found under {beit3_dir}")
    return beit3_dir


class Beit3Encoder:
    """BEiT-3 retrieval vision embedding encoder."""

    def __init__(
        self,
        checkpoint_url: str = DEFAULT_BEIT3_CHECKPOINT,
        checkpoint_dir: Path | None = None,
        repo_dir: Path | None = None,
        device: str = "cuda",
        compute_dtype: str = "float16",
    ) -> None:
        import torch
        import torch.nn as nn
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        self.torch = torch
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.checkpoint_url = checkpoint_url

        beit3_dir = ensure_unilm_repo(repo_dir)

        # timm 0.4.x / PyTorch 2.x compatibility shims
        torch_six_shim = types.ModuleType("torch._six")
        torch_six_shim.container_abcs = abc
        torch_six_shim.inf = float("inf")
        torch_six_shim.string_classes = (str,)
        sys.modules.setdefault("torch._six", torch_six_shim)

        utils_shim = types.ModuleType("utils")
        utils_shim.get_rank = lambda: 0
        utils_shim.get_world_size = lambda: 1

        class ClipLoss(nn.Module):
            def __init__(self, *args: Any, **kwargs: Any):
                del args, kwargs
                super().__init__()

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("ClipLoss is training-only and unavailable in inference mode")

        utils_shim.ClipLoss = ClipLoss
        sys.modules["utils"] = utils_shim

        if str(beit3_dir) not in sys.path:
            sys.path.insert(0, str(beit3_dir))

        from modeling_finetune import beit3_base_patch16_384_retrieval

        ckpt_dir = checkpoint_dir or (Path.home() / ".cache" / "beit3")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_filename = Path(checkpoint_url).name
        checkpoint_path = ckpt_dir / ckpt_filename

        if not checkpoint_path.is_file():
            logger.info("Downloading BEiT-3 retrieval checkpoint from %s ...", checkpoint_url)
            download_http_resumable(checkpoint_url, checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        model = beit3_base_patch16_384_retrieval()
        incompatible = model.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"BEiT-3 checkpoint mismatch: {incompatible}")

        self.model = model.to(self.device).eval()
        if self.device != "cpu":
            self.model.half()

        self.transform = transforms.Compose(
            [
                transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    @classmethod
    def from_pretrained(
        cls,
        *,
        checkpoint_url: str = DEFAULT_BEIT3_CHECKPOINT,
        checkpoint_dir: Path | None = None,
        repo_dir: Path | None = None,
        device: str = "cuda",
        compute_dtype: str = "float16",
    ) -> Beit3Encoder:
        return cls(
            checkpoint_url=checkpoint_url,
            checkpoint_dir=checkpoint_dir,
            repo_dir=repo_dir,
            device=device,
            compute_dtype=compute_dtype,
        )

    @property
    def embedding_dim(self) -> int:
        return 768

    @property
    def preprocessing(self) -> dict[str, Any]:
        return {
            "resize": [384, 384],
            "interpolation": "bicubic",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "checkpoint": self.checkpoint_url,
        }

    def encode_images(self, images: list[Image.Image] | Sequence[Image.Image]) -> np.ndarray:
        """Return `[len(images), 768]` float32 L2-normalized unit vectors."""
        if not images:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        torch = self.torch
        batch = torch.stack([self.transform(img.convert("RGB")) for img in images]).to(
            self.device, non_blocking=True
        )
        if self.device != "cpu":
            batch = batch.half()

        with torch.inference_mode(), torch.autocast(
            device_type=self.device,
            dtype=torch.float16,
            enabled=self.device != "cpu",
        ):
            vision_features, _ = self.model(image=batch, only_infer=True)
            vision_features = torch.nn.functional.normalize(vision_features.float(), p=2, dim=-1)

        return vision_features.cpu().numpy().astype(np.float32, copy=False)

    def encode(self, images: Sequence[Any]) -> np.ndarray:
        return self.encode_images(list(images))
