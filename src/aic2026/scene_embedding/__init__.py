"""SigLIP2 whole-frame scene embeddings for coarse retrieval."""

from .pipeline import embed_frames
from .siglip_backend import SIGLIP_MODEL_ID, SIGLIP_REVISION, SiglipEncoder
from .store import l2_normalize, matrix_path_for, read_matrix, write_matrix_atomic
from .validation import validate_embedding_stage_inputs, validate_published_embeddings

__all__ = [
    "SIGLIP_MODEL_ID",
    "SIGLIP_REVISION",
    "SiglipEncoder",
    "embed_frames",
    "l2_normalize",
    "matrix_path_for",
    "read_matrix",
    "validate_embedding_stage_inputs",
    "validate_published_embeddings",
    "write_matrix_atomic",
]
