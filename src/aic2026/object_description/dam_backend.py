"""Lazy adapter around the pinned NVlabs Describe Anything implementation."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .caption import DAM_PROMPT, normalize_caption
from aic2026.contracts import CaptionResult

DAM_MODEL_ID = "nvidia/DAM-3B"
DAM_REVISION = "0797bedd98d645cd021379a4661ee233da279bba"
DAM_CODE_REVISION = "153ad3d33c29324e9197f565547c6bc8500da02d"


def verify_installed_dam_revision(expected_revision: str) -> str:
    """Fail closed unless PEP 610 metadata or the verified offline mirror pins DAM code."""
    try:
        direct_url_text = importlib.metadata.distribution("dam").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("The pinned dam package is not installed") from error
    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text)
            installed_revision = direct_url.get("vcs_info", {}).get("commit_id")
        except (AttributeError, json.JSONDecodeError):
            installed_revision = None
        if installed_revision:
            if installed_revision != expected_revision:
                raise RuntimeError(
                    f"Installed DAM revision {installed_revision!r} does not match "
                    f"{expected_revision!r}"
                )
            return "pep610"
    offline_revision = os.environ.get("AIC_DAM_CODE_REVISION")
    if offline_revision != expected_revision:
        raise RuntimeError(
            "Cannot verify installed DAM source revision. Online installs need PEP 610 VCS "
            "metadata; offline runs must verify the wheel checksum and set "
            "AIC_DAM_CODE_REVISION."
        )
    return "offline_mirror_checksum"


class DamCaptioner:
    def __init__(self, model: Any) -> None:
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str = DAM_MODEL_ID,
        revision: str = DAM_REVISION,
        code_revision: str = DAM_CODE_REVISION,
        cache_dir: Path | None = None,
    ) -> DamCaptioner:
        # Cleanly disable scipy/TF/Flax in transformers to prevent Python 3.12 wildcard recursion
        os.environ["USE_TF"] = "0"
        os.environ["USE_FLAX"] = "0"
        os.environ["USE_TORCH"] = "1"
        try:
            import transformers.utils.import_utils as _t_import

            _t_import._scipy_available = False
            _t_import.is_scipy_available = lambda: False
            _t_import._sklearn_available = False
            _t_import.is_sklearn_available = lambda: False
            _t_import._tf_available = False
            _t_import.is_tf_available = lambda: False
        except Exception:
            pass

        # Ensure transformers.modeling_utils compatibility for older LLaVA/DAM architectures
        try:
            import contextlib
            import transformers.modeling_utils as _t_mu

            if not hasattr(_t_mu, "no_init_weights"):

                @contextlib.contextmanager
                def _dummy_no_init_weights(*args: Any, **kwargs: Any) -> Any:
                    yield

                _t_mu.no_init_weights = _dummy_no_init_weights

            if not hasattr(_t_mu, "ContextManagers"):

                class _DummyContextManagers:
                    def __init__(self, context_managers: Any) -> None:
                        self.context_managers = list(context_managers) if context_managers else []

                    def __enter__(self) -> None:
                        for cm in self.context_managers:
                            if hasattr(cm, "__enter__"):
                                cm.__enter__()

                    def __exit__(self, *args: Any) -> None:
                        for cm in reversed(self.context_managers):
                            if hasattr(cm, "__exit__"):
                                cm.__exit__(*args)

                _t_mu.ContextManagers = _DummyContextManagers

            # Ensure PreTrainedModel handles older custom projectors missing all_tied_weights_keys
            if hasattr(_t_mu, "PreTrainedModel"):
                _orig_mark_tied = getattr(
                    _t_mu.PreTrainedModel, "mark_tied_weights_as_initialized", None
                )
                if _orig_mark_tied is not None:

                    def _safe_mark_tied(self: Any, *args: Any, **kwargs: Any) -> Any:
                        if not hasattr(self, "all_tied_weights_keys"):
                            self.all_tied_weights_keys = {}
                        try:
                            return _orig_mark_tied(self, *args, **kwargs)
                        except Exception:
                            return None

                    _t_mu.PreTrainedModel.mark_tied_weights_as_initialized = _safe_mark_tied
        except Exception:
            pass

        try:
            from dam import DescribeAnythingModel, disable_torch_init
        except ImportError as error:
            raise RuntimeError(
                "Pinned NVlabs/describe-anything is not installed. Follow docs/cloud-runbook.md."
            ) from error
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError("huggingface-hub is required to resolve DAM weights") from error
        model_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=os.environ.get("HF_HUB_OFFLINE", "0") == "1",
        )
        import torch

        disable_torch_init()
        target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            model = DescribeAnythingModel(
                model_path=model_path,
                conv_mode="v1",
                prompt_mode="full+focal_crop",
                device=target_device,
            )
        except TypeError:
            model = DescribeAnythingModel(
                model_path=model_path,
                conv_mode="v1",
                prompt_mode="full+focal_crop",
            )
        # Ensure SiglipVisionEmbeddings forward safely casts position_ids to CUDA device and handles 4-channel mask inputs
        try:
            from typing import Optional
            import dam.model.multimodal_encoder.siglip.modeling_siglip as _siglip_mod

            def _safe_siglip_emb_forward(
                self: Any,
                pixel_values: torch.FloatTensor,
                additional_position_embedding: Optional[torch.Tensor] = None,
                additional_embedding_mode: Optional[str] = None,
            ) -> torch.Tensor:
                target_dtype = self.patch_embedding.weight.dtype
                target_device = self.patch_embedding.weight.device

                if getattr(self, "mask_patch_embedding", None) is None:
                    patch_embeds = self.patch_embedding(pixel_values.to(device=target_device, dtype=target_dtype))
                else:
                    patch_embeds = self.patch_embedding(pixel_values[:, :3, ...].to(device=target_device, dtype=target_dtype))
                    if pixel_values.size(1) == 4:
                        patch_embeds = patch_embeds + self.mask_patch_embedding(pixel_values[:, 3:4, ...].to(device=target_device, dtype=target_dtype))

                embeddings = patch_embeds.flatten(2).transpose(1, 2)
                pos_ids = getattr(self, "position_ids", None)
                num_weights = self.position_embedding.weight.shape[0]
                if pos_ids is None or not isinstance(pos_ids, torch.Tensor) or pos_ids.device != target_device:
                    pos_ids = torch.arange(num_weights, dtype=torch.long, device=target_device).unsqueeze(0)
                    self.position_ids = pos_ids
                elif pos_ids.shape[-1] != num_weights:
                    pos_ids = torch.arange(num_weights, dtype=torch.long, device=target_device).unsqueeze(0)
                    self.position_ids = pos_ids
                else:
                    pos_ids = pos_ids.to(device=target_device, dtype=torch.long)

                if additional_position_embedding is not None:
                    if additional_embedding_mode == "add":
                        embeddings = embeddings + self.position_embedding(pos_ids)
                        embeddings = embeddings + additional_position_embedding
                    elif additional_embedding_mode == "replace":
                        embeddings = embeddings + self.position_embedding(pos_ids) * 0.0
                        embeddings = embeddings + additional_position_embedding
                    else:
                        raise ValueError(f"additional_embedding_mode should be either 'add' or 'replace', got {additional_embedding_mode}")
                else:
                    embeddings = embeddings + self.position_embedding(pos_ids)

                return embeddings

            _siglip_mod.SiglipVisionEmbeddings.forward = _safe_siglip_emb_forward
        except Exception:
            pass

        model.eval()
        # Ensure context provider treats all inputs as cimage so 0-batch cross-attention never occurs
        if hasattr(model, "model") and hasattr(model.model, "get_context_provider"):
            try:
                cp = model.model.get_context_provider()
                if cp is not None:
                    cp.treat_image_as_cimage = True
            except Exception:
                pass
        elif hasattr(model, "get_context_provider"):
            try:
                cp = model.get_context_provider()
                if cp is not None:
                    cp.treat_image_as_cimage = True
            except Exception:
                pass
        return cls(model)

    def describe(
        self,
        image: Image.Image,
        mask: Image.Image,
        *,
        max_new_tokens: int = 48,
    ) -> str:
        if mask.mode != "L":
            mask = mask.convert("L")
        if mask.size != image.size or mask.getbbox() is None:
            raise ValueError("DAM mask must be non-empty and match the image dimensions")

        image_rgb = image.convert("RGB")
        import torch

        with torch.inference_mode():
            output = self.model.get_description(
                image_rgb,
                mask,
                DAM_PROMPT,
                streaming=False,
                temperature=0,
                num_beams=1,
                max_new_tokens=max_new_tokens,
            )

        if not isinstance(output, str):
            try:
                output = next(iter(output))
            except Exception:
                output = str(output)

        if not isinstance(output, str):
            raise TypeError(f"DAM returned a non-string description: {type(output)}")
        return output

        if not isinstance(output, str):
            try:
                output = next(iter(output))
            except Exception:
                output = str(output)

        if not isinstance(output, str):
            raise TypeError(f"DAM returned a non-string description: {type(output)}")
        return output

    def describe_region(
        self,
        image: Image.Image,
        mask: np.ndarray | Image.Image | None = None,
        bbox_xyxy_px: tuple[int, int, int, int] | None = None,
        class_entity: str | None = None,
        max_words: int = 50,
        max_new_tokens: int = 48,
    ) -> CaptionResult:
        """Describe a specific segmented region directly via its bounding box/mask with normalized <=50 word caption."""
        image_rgb = image.convert("RGB")
        w, h = image_rgb.size

        # Directly generate the clean bounding-box rectangular mask
        if bbox_xyxy_px is not None:
            x1, y1, x2, y2 = bbox_xyxy_px
            x1 = max(0, min(w - 1, int(round(x1))))
            y1 = max(0, min(h - 1, int(round(y1))))
            x2 = max(x1 + 1, min(w, int(round(x2))))
            y2 = max(y1 + 1, min(h, int(round(y2))))
            mask_img = Image.new("L", (w, h), 0)
            ImageDraw.Draw(mask_img).rectangle([x1, y1, x2, y2], fill=255)
        elif isinstance(mask, np.ndarray):
            mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            if mask_img.size != (w, h):
                mask_img = mask_img.resize((w, h), Image.NEAREST)
        elif isinstance(mask, Image.Image):
            mask_img = mask.convert("L")
            if mask_img.size != (w, h):
                mask_img = mask_img.resize((w, h), Image.NEAREST)
        else:
            mask_img = Image.new("L", (w, h), 255)

        if mask_img.getbbox() is None:
            mask_img = Image.new("L", (w, h), 255)

        raw_caption = self.describe(image_rgb, mask_img, max_new_tokens=max_new_tokens)
        return normalize_caption(raw_caption, maximum_words=max_words)
