"""Canonical local encoder implementations and isolated worker adapters."""

import os

# This must be present before importing torch-backed child modules.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from .beit3_coco import Beit3CocoTextEncoder
from .bge_m3 import BgeM3TextEncoder
from .cpu import CpuTextEncoders
from .sequential_manager import SequentialBranch1Encoders
from .worker_manager import EncoderWorkerManager, ProcessBranch1Encoders, ProcessCpuTextEncoders

__all__ = [
    "CpuTextEncoders",
    "SequentialBranch1Encoders",
    "EncoderWorkerManager",
    "ProcessBranch1Encoders",
    "ProcessCpuTextEncoders",
    "Beit3CocoTextEncoder",
    "BgeM3TextEncoder",
]
