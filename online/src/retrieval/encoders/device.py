"""Shared accelerator selection for native CPU and Apple Metal runtimes."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import suppress
from importlib.metadata import version
from typing import Any

# PyTorch reads this switch while initialising its MPS backend.  Native launchers
# also export it before Python starts, but setting the default here keeps direct
# module use safe as long as torch has not already been imported by the caller.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

VALID_DEVICES = {"auto", "mps", "cpu"}


def transformers_dtype_kwargs(
    dtype: Any,
    *,
    version_value: str | None = None,
) -> dict[str, Any]:
    """Use the non-deprecated dtype keyword without breaking 4.48 images."""

    release_text = version_value or version("transformers")
    try:
        major, minor = (int(part) for part in release_text.split(".", 2)[:2])
    except (TypeError, ValueError):
        # Older releases support only torch_dtype; it is the conservative
        # fallback for an unusual local version string as well.
        major, minor = 0, 0
    key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    return {key: dtype}


def requested_device(value: str | None = None) -> str:
    """Return a validated device preference.

    Library callers remain CPU-compatible by default.  The native macOS
    launcher explicitly exports ``AIC_DEVICE=auto`` so Metal is preferred.
    """

    selected = str(value or os.environ.get("AIC_DEVICE", "cpu")).strip().lower()
    if selected not in VALID_DEVICES:
        choices = ", ".join(sorted(VALID_DEVICES))
        raise ValueError(f"AIC_DEVICE must be one of {choices}; got {selected!r}")
    return selected


def cpu_fallback_allowed(value: bool | None = None) -> bool:
    if value is not None:
        return bool(value)
    return str(os.environ.get("AIC_ALLOW_CPU_FALLBACK", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(
        backend is not None
        and callable(getattr(backend, "is_built", None))
        and backend.is_built()
        and callable(getattr(backend, "is_available", None))
        and backend.is_available()
    )


def resolve_device(value: str | None = None, *, allow_fallback: bool | None = None) -> str:
    selected = requested_device(value)
    fallback = cpu_fallback_allowed(allow_fallback)
    if selected == "cpu":
        return "cpu"
    if mps_available():
        return "mps"
    if fallback or selected == "auto":
        return "cpu"
    raise RuntimeError("AIC_DEVICE=mps was requested, but PyTorch MPS is unavailable")


def move_tensors(value: Mapping[str, Any], device: str) -> dict[str, Any]:
    """Move tensor-valued model inputs while leaving metadata untouched."""

    return {
        key: item.to(device) if isinstance(item, torch.Tensor) else item
        for key, item in value.items()
    }


def clear_accelerator_cache(device: str | None) -> None:
    if device == "mps" and mps_available():
        with suppress(AttributeError, RuntimeError):
            torch.mps.empty_cache()


def is_mps_runtime_error(error: BaseException) -> bool:
    """Recognise failures for which retrying one worker on CPU is safe."""

    message = f"{type(error).__name__}: {error}".lower()
    markers = (
        "mps",
        "metal",
        "placeholder storage has not been allocated",
        "not implemented for",
        "not currently implemented",
        "out of memory",
    )
    return any(marker in message for marker in markers)


def device_contract(
    value: str | None = None, *, allow_fallback: bool | None = None
) -> dict[str, Any]:
    selected = requested_device(value)
    fallback = cpu_fallback_allowed(allow_fallback)
    return {
        "requested": selected,
        "preferred": resolve_device(selected, allow_fallback=fallback),
        "mps_built": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
        "mps_available": mps_available(),
        "cpu_fallback_allowed": fallback,
        "pytorch_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
    }


__all__ = [
    "VALID_DEVICES",
    "clear_accelerator_cache",
    "cpu_fallback_allowed",
    "device_contract",
    "is_mps_runtime_error",
    "move_tensors",
    "mps_available",
    "requested_device",
    "resolve_device",
    "transformers_dtype_kwargs",
]
