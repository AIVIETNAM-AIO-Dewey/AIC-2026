"""Pydantic model for the SigLIP2 scene-embedding index."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, PositiveInt, model_validator

from .models import FrameRef

SCENE_EMBEDDING_SCHEMA = "aic26.scene_embeddings.v1"


class SceneEmbeddingRecord(FrameRef):
    """One embedded keyframe, pointing at its row in the companion `.npy` matrix.

    Line order mirrors the upstream frame manifest and `row` must equal the line
    number, so the index and the matrix stay joinable by position alone. See
    `docs/architecture.md` for the full two-file contract.
    """

    schema_version: Literal["aic26.scene_embeddings.v1"] = SCENE_EMBEDDING_SCHEMA
    run_id: str = Field(min_length=1)
    row: int = Field(ge=0)
    embedding_dim: PositiveInt
    dtype: Literal["float16", "float32"]
    l2_normalized: bool

    @model_validator(mode="after")
    def validate_normalization(self) -> SceneEmbeddingRecord:
        # A non-normalized vector silently turns every downstream cosine into an
        # unnormalized dot product, so v1 refuses to describe one.
        if not self.l2_normalized:
            raise ValueError("scene embeddings must be L2-normalized in schema v1")
        return self
