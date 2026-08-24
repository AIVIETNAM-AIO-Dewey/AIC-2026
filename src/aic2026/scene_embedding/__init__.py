"""Scene Embedding Subsystem using SigLIP-2."""

from .siglip_backend import (
    SIGLIP_MODEL_ID,
    SIGLIP_REVISION,
    SiglipEncoder,
)

__all__ = [
    "SIGLIP_MODEL_ID",
    "SIGLIP_REVISION",
    "SiglipEncoder",
]
