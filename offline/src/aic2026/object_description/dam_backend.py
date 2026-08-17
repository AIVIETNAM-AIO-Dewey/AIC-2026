"""Lazy adapter around the pinned NVlabs Describe Anything implementation."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from .caption import DAM_PROMPT

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
        verify_installed_dam_revision(code_revision)
        try:
            from dam import DescribeAnythingModel, disable_torch_init
        except ImportError as error:
            raise RuntimeError(
                "Pinned NVlabs/describe-anything is not installed. Follow docs/runbook.md."
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
        model = DescribeAnythingModel(
            model_path=model_path,
            conv_mode="v1",
            prompt_mode="full+focal_crop",
        )
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
        output = self.model.get_description(
            image.convert("RGB"),
            mask,
            DAM_PROMPT,
            streaming=False,
            temperature=0,
            num_beams=1,
            max_new_tokens=max_new_tokens,
        )
        if not isinstance(output, str):
            raise TypeError("DAM returned a non-string description")
        return output
