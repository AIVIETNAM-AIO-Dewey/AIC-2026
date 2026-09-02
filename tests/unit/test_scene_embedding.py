"""Unit tests for SigLIP2 scene embedding pipeline, store, and validation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from aic2026.contracts import FrameRef, SceneEmbeddingRecord
from aic2026.scene_embedding.pipeline import embed_frames
from aic2026.scene_embedding.store import (
    l2_normalize,
    matrix_path_for,
    read_matrix,
    write_matrix_atomic,
)
from aic2026.scene_embedding.validation import validate_published_embeddings
from scripts.run_scene_embeddings import main as run_scene_embeddings_main


def test_l2_normalize() -> None:
    raw = np.array([[3.0, 4.0], [1.0, 1.0]], dtype=np.float32)
    normalized = l2_normalize(raw)
    norms = np.linalg.norm(normalized, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(normalized[0], [0.6, 0.8], atol=1e-6)


def test_matrix_io_roundtrip_npy(tmp_path: Path) -> None:
    matrix = np.array([[0.6, 0.8], [0.7071, 0.7071]], dtype=np.float32)
    path = tmp_path / "embeddings.f16.npy"
    write_matrix_atomic(path, matrix, dtype="float16")
    loaded = read_matrix(path)
    assert loaded.dtype == np.float16
    np.testing.assert_allclose(loaded, matrix, atol=1e-3)


def test_embed_frames_with_mock_backend(tmp_path: Path) -> None:
    # 1. Create dummy image
    img_dir = tmp_path / "keyframes" / "L21_V001"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_file = img_dir / "00000411.jpg"
    Image.new("RGB", (1280, 720), color="blue").save(img_file)

    # 2. Create manifest
    manifest_path = tmp_path / "frame_manifests" / "L21_V001.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    ref = FrameRef(
        video_id="L21_V001",
        frame_uid="L21_V001:411",
        keyframe_n=1,
        frame_idx=411,
        pts_time_s=13.7,
        fps=30.0,
        frame_relpath="keyframes/L21_V001/00000411.jpg",
        width=1280,
        height=720,
    )
    manifest_path.write_text(json.dumps(ref.model_dump()) + "\n", encoding="utf-8")

    # 3. Mock Backend
    mock_backend = MagicMock()
    mock_backend.encode_images.return_value = np.array([[0.6, 0.8]], dtype=np.float32)

    output_index = tmp_path / "scene_embeddings" / "L21_V001.jsonl"
    output_matrix = tmp_path / "scene_embeddings" / "L21_V001.f16.npy"

    counters = embed_frames(
        frame_manifest=manifest_path,
        data_root=tmp_path,
        output_index=output_index,
        output_matrix=output_matrix,
        run_id="test-run",
        backend=mock_backend,
        matrix_dtype="float16",
        batch_size=8,
    )

    assert counters["frames"] == 1
    assert counters["embedding_dim"] == 2
    assert output_index.is_file()
    assert output_matrix.is_file()

    summary = validate_published_embeddings(
        index_path=output_index,
        matrix_path=output_matrix,
        video_id="L21_V001",
        expected_frame_uids=["L21_V001:411"],
        expected_run_id="test-run",
    )
    assert summary["frames"] == 1


def test_run_scene_embeddings_parser_options():
    from scripts.run_scene_embeddings import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--video-id", "L21_V001",
        "--frame-manifest", "manifest.jsonl",
        "--data-root", "/data",
        "--output", "output.jsonl",
        "--model-family", "metaclip2",
    ])
    assert args.model_family == "metaclip2"

    args_beit = parser.parse_args([
        "--video-id", "L21_V001",
        "--frame-manifest", "manifest.jsonl",
        "--data-root", "/data",
        "--output", "output.jsonl",
        "--model-family", "beit3",
    ])
    assert args_beit.model_family == "beit3"

