"""Lazy adapter around the pinned NVlabs Describe Anything implementation."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np

from .caption import DAM_PROMPT, normalize_caption

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
        import os
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
                _orig_mark_tied = getattr(_t_mu.PreTrainedModel, "mark_tied_weights_as_initialized", None)
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
        # DAM accepts a Hugging Face model id but has no explicit revision argument.
        # Resolve the pinned snapshot first and pass its immutable local path.
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
        import torch
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Try the current kwarg name first (`dtype`), fall back to deprecated (`torch_dtype`)
        model = None
        for kwarg_name in ("dtype", "torch_dtype"):
            if model is not None:
                break
            try:
                model = DescribeAnythingModel(
                    model_path=model_path,
                    conv_mode="v1",
                    prompt_mode="full+focal_crop",
                    **{kwarg_name: dtype},
                )
            except TypeError:
                continue
            except Exception:
                continue
        if model is None:
            model = DescribeAnythingModel(
                model_path=model_path,
                conv_mode="v1",
                prompt_mode="full+focal_crop",
            )

        # ── Force ALL submodules to float16 on CUDA (Tesla T4 has zero bfloat16 support) ──
        if torch.cuda.is_available():
            # The real nn.Module is model.model (LlavaLlamaModel).
            inner = getattr(model, "model", None)

            # Cast the inner LlavaLlamaModel
            if inner is not None and isinstance(inner, torch.nn.Module):
                inner.to(device="cuda", dtype=torch.float16)

            # Also cast the DescribeAnythingModel wrapper
            if isinstance(model, torch.nn.Module):
                model.to(device="cuda", dtype=torch.float16)

            # Hard verification: forcibly cast any remaining bfloat16 parameters
            target = inner if inner is not None else model
            if isinstance(target, torch.nn.Module):
                bf16_count = 0
                for name, param in target.named_parameters():
                    if param.is_floating_point() and param.dtype == torch.bfloat16:
                        param.data = param.data.to(torch.float16)
                        bf16_count += 1
                for name, buf in target.named_buffers():
                    if buf.is_floating_point() and buf.dtype == torch.bfloat16:
                        buf.data = buf.data.to(torch.float16)
                        bf16_count += 1
                if bf16_count > 0:
                    print(f"     ⚠️  Force-cast {bf16_count} remaining bfloat16 params/buffers → float16")

            # Override config.model_dtype so the forward pass uses float16
            if inner is not None and hasattr(inner, "config"):
                inner.config.model_dtype = "torch.float16"
                if hasattr(inner.config, "torch_dtype"):
                    inner.config.torch_dtype = torch.float16

            # Diagnostic: verify vision tower dtype
            if inner is not None:
                vt = getattr(inner, "vision_tower", None)
                if vt is not None and isinstance(vt, torch.nn.Module):
                    first_param = next(vt.parameters(), None)
                    if first_param is not None:
                        print(f"     ✓ Vision tower dtype after cast: {first_param.dtype}")

        model.eval()
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
        prompt = DAM_PROMPT
        pm = getattr(self.model, "prompt_mode", "full+focal_crop")
        if pm == "full+focal_crop" and prompt.count("<image>") < 2:
            prompt = "<image>\n" + prompt
        elif pm != "full+focal_crop" and prompt.count("<image>") > 1:
            prompt = prompt.replace("<image>\n<image>", "<image>")

        import torch
        with torch.inference_mode():
            try:
                output = self.model.get_description(
                    image.convert("RGB"),
                    mask,
                    prompt,
                    streaming=False,
                    temperature=0,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                )
            except ValueError as err:
                if "no <image> tag found" in str(err):
                    output = self.model.get_description(
                        image.convert("RGB"),
                        mask,
                        f"<image>\n{prompt}",
                        streaming=False,
                        temperature=0,
                        num_beams=1,
                        max_new_tokens=max_new_tokens,
                    )
                else:
                    raise
        if not isinstance(output, str):
            raise TypeError("DAM returned a non-string description")
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
        """Describe a specific segmented region directly via its bounding box mask with normalized <=50 word caption."""
        image_rgb = image.convert("RGB")
        w, h = image_rgb.size

        # Directly generate the clean bounding-box rectangular mask
        if bbox_xyxy_px is not None:
            x1, y1, x2, y2 = bbox_xyxy_px
            x1 = max(0, min(w - 1, int(round(x1))))
            y1 = max(0, min(h - 1, int(round(y1))))
            x2 = max(x1 + 1, min(w, int(round(x2))))
            y2 = max(y1 + 1, min(h, int(round(y2))))
            from PIL import ImageDraw

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


