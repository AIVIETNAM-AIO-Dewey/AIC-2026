"""BGE-M3 model boundary for DAM description retrieval."""

from .cpu import CpuTextEncoders

MODEL_ID = "BAAI/bge-m3"
DIMENSION = 1024
POOLING = "cls"
NORMALIZATION = "l2"
MAX_TOKENS = 512


class BgeM3TextEncoder(CpuTextEncoders):
    """Named BGE-M3 boundary; DAM vectors stay on their offline space."""


__all__ = [
    "BgeM3TextEncoder",
    "DIMENSION",
    "MAX_TOKENS",
    "MODEL_ID",
    "NORMALIZATION",
    "POOLING",
]
