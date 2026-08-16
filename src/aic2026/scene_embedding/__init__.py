"""SigLIP2 whole-frame scene embeddings for coarse retrieval."""

from .pipeline import embed_frames
from .qdrant_index import ensure_collection, iter_shard_points, load_shard, point_id, shard_paths
from .siglip_backend import SIGLIP_MODEL_ID, SIGLIP_REVISION, SiglipEncoder
from .store import l2_normalize, matrix_path_for, read_matrix, write_matrix_atomic
from .validation import validate_embedding_stage_inputs, validate_published_embeddings

__all__ = [
    "SIGLIP_MODEL_ID",
    "SIGLIP_REVISION",
    "SiglipEncoder",
    "embed_frames",
    "ensure_collection",
    "iter_shard_points",
    "l2_normalize",
    "load_shard",
    "matrix_path_for",
    "point_id",
    "read_matrix",
    "shard_paths",
    "validate_embedding_stage_inputs",
    "validate_published_embeddings",
    "write_matrix_atomic",
]
