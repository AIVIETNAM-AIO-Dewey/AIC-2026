"""Lazy adapter around the pinned NVlabs Describe Anything implementation."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

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
        disable_torch_init()
        model = DescribeAnythingModel(
            model_path=model_path,
            conv_mode="v1",
            prompt_mode="full+focal_crop",
        )
        model.eval()

        # Fix: Ensure vision tower position_ids is explicitly on the exact target CUDA device
        try:
            import torch
            vt = getattr(model, "model", model)
            if hasattr(vt, "get_vision_tower"):
                vt_mod = vt.get_vision_tower()
            elif hasattr(vt, "vision_tower"):
                vt_mod = vt.vision_tower
            else:
                vt_mod = None

            if vt_mod is not None:
                vm = getattr(vt_mod, "vision_tower", vt_mod)
                vm_inner = getattr(vm, "vision_model", vm)
                emb = getattr(vm_inner, "embeddings", None)
                if emb is not None and hasattr(emb, "position_embedding") and hasattr(emb.position_embedding, "weight"):
                    target_device = emb.position_embedding.weight.device
                    num_pos = emb.position_embedding.weight.shape[0]
                    correct_pos_ids = torch.arange(num_pos, dtype=torch.long, device=target_device).unsqueeze(0)
                    emb.position_ids = correct_pos_ids
                    emb.register_buffer("position_ids", correct_pos_ids, persistent=False)
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
                temperature=0.2,
                top_p=0.5,
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

    def describe_region(
        self,
        image: Image.Image,
        mask_rle: Any,
        focal_box: tuple[int, int, int, int] | None = None,
        *,
        max_words: int = 50,
        max_new_tokens: int = 48,
    ) -> CaptionResult:
        from aic2026.object_description.rle import decode_mask, rectangle_mask

        w, h = image.size
        mask_array = None
        if mask_rle is not None:
            try:
                mask_array = decode_mask(mask_rle)
            except Exception:
                mask_array = None

        if mask_array is None:
            if focal_box is not None:
                mask_array = rectangle_mask(h, w, focal_box)
            else:
                x1, y1 = int(0.05 * w), int(0.05 * h)
                x2, y2 = int(0.95 * w), int(0.95 * h)
                mask_array = rectangle_mask(h, w, (x1, y1, x2, y2))

        mask_pil = Image.fromarray((mask_array.astype(bool) * 255).astype(np.uint8))
        raw_text = self.describe(image, mask_pil, max_new_tokens=max_new_tokens)
        return normalize_caption(raw_text, maximum_words=max_words)
