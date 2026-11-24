"""BEiT-3 COCO Retrieval identity and worker-backed text encoder boundary."""

from .sequential_manager import (
    BEIT3_CHECKPOINT_NAME,
    BEIT3_CHECKPOINT_SHA256,
    UNILM_REVISION,
    SequentialBranch1Encoders,
)

MODEL_ID = BEIT3_CHECKPOINT_NAME
DIMENSION = 768
MAX_TOKENS = 64
TEXT_OUTPUT = "language_head"
TASK = "COCO Retrieval"


class Beit3CocoTextEncoder(SequentialBranch1Encoders):
    """Named boundary for the COCO dual-encoder used by Branch 2 reranking."""


__all__ = [
    "BEIT3_CHECKPOINT_SHA256",
    "Beit3CocoTextEncoder",
    "DIMENSION",
    "MAX_TOKENS",
    "MODEL_ID",
    "TASK",
    "TEXT_OUTPUT",
    "UNILM_REVISION",
]
