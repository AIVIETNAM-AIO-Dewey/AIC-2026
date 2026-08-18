from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from aic2026.common.io import sha256_path
from aic2026.contracts.query import QuerySpec
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from aic_backend.api.app import create_app
from aic_backend.api.deps import (
    get_gpt,
    get_parser,
    get_repository,
    get_search_service,
    get_trake_service,
)
from aic_backend.ingest.artifacts import ArtifactFile, ingest
from aic_backend.retrieval.qdrant import QdrantRepository
from aic_backend.retrieval.search import SearchService
from aic_backend.retrieval.trake import TrakeService


class SceneEncoder:
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)


class DenseEncoder:
    def encode(self, texts: list[str], *, query: bool) -> np.ndarray:
        del query
        return np.zeros((len(texts), 768), dtype=np.float32)


class Parser:
    def parse(self, *, task_type: str, raw_query_vi: str) -> QuerySpec:
        return QuerySpec(task_type="kis", raw_query_vi=raw_query_vi, scene_en="scene")


class Gpt:
    client = None


@pytest.mark.integration
def test_artifact_ingest_to_api_preserves_canonical_frame_id(tmp_path: Path) -> None:
    path = tmp_path / "scene_embeddings" / "L21_V011.jsonl"
    path.parent.mkdir()
    row = {
        "video_id": "L21_V011",
        "frame_uid": "L21_V011:24925",
        "frame_idx": 24925,
        "keyframe_n": 262,
        "pts_time_s": 997.0,
        "frame_relpath": "keyframes/L21_V011/262.jpg",
        "row": 0,
        "embedding_dim": 3,
        "dtype": "float16",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    matrix_path = path.with_name("L21_V011.f16.npy")
    np.save(matrix_path, np.asarray([[1.0, 0.0, 0.0]], dtype=np.float16))
    manifest = path.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": "fixture-run",
                "counters": {"frames": 1},
                "models": [{"model_id": "siglip", "revision": "fixture"}],
                "outputs": [
                    {"sha256": sha256_path(path)},
                    {"sha256": sha256_path(matrix_path)},
                ],
            }
        ),
        encoding="utf-8",
    )
    qdrant = QdrantClient(":memory:")
    counts = ingest(
        qdrant,
        [ArtifactFile("frames_sparse", path, manifest)],
        dense_encoder=DenseEncoder(),
        activate=True,
    )
    assert counts == {"frames_sparse": 1}

    repository = QdrantRepository(
        qdrant,
        artifact_root=tmp_path,
        text_encoder=DenseEncoder(),
        scene_encoder=SceneEncoder(),
    )
    service = SearchService(repository)
    app = create_app()
    app.dependency_overrides = {
        get_repository: lambda: repository,
        get_parser: lambda: Parser(),
        get_search_service: lambda: service,
        get_trake_service: lambda: TrakeService(service),
        get_gpt: lambda: Gpt(),
    }
    response = TestClient(app).post(
        "/api/v1/search", json={"task_type": "kis", "raw_query_vi": "tìm cảnh"}
    )
    assert response.status_code == 200
    hit = response.json()["results"][0]
    assert (hit["video_id"], hit["frame_idx"]) == ("L21_V011", 24925)


@pytest.mark.integration
def test_ocr_query_uses_trigram_candidates_edit_distance_and_structured_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ocr" / "L21_V011.jsonl"
    path.parent.mkdir()
    texts = ["NON SÔNG LIỀN MỘT DẢI", "XE BUÝT TRÊN ĐƯỜNG PHỐ"]
    rows = []
    for offset, text in enumerate(texts):
        rows.append(
            {
                "video_id": "L21_V011",
                "frame_uid": f"L21_V011:{24925 + offset}",
                "frame_idx": 24925 + offset,
                "keyframe_n": 262 + offset,
                "pts_time_s": 997.0 + offset,
                "width": 1280,
                "height": 720,
                "source_image_sha256": f"{offset + 1:064x}",
                "terminal_status": "success",
                "full_text": text.lower(),
                "texts": [
                    {
                        "line_id": "line-0000",
                        "raw_text": text,
                        "normalized_text": text.lower(),
                        "confidence": 0.856,
                        "accepted": True,
                        "polygon_xy": [[100, 200], [600, 200], [600, 260], [100, 260]],
                        "reading_order": 0,
                    }
                ],
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = path.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": "ppocrv6-small-fixture",
                "counters": {"frames": 2},
                "models": [{"model_id": "PP-OCRv6-small", "revision": "fixture"}],
                "outputs": [{"sha256": sha256_path(path)}],
            }
        ),
        encoding="utf-8",
    )
    qdrant = QdrantClient(":memory:")
    ingest(
        qdrant,
        [ArtifactFile("ocr", path, manifest)],
        dense_encoder=DenseEncoder(),
        activate=True,
    )
    repository = QdrantRepository(
        qdrant,
        artifact_root=tmp_path,
        text_encoder=DenseEncoder(),
    )

    hits = repository.search_text("ocr", "non song cung mot dai", limit=2)

    assert hits[0].frame_idx == 24925
    assert hits[0].ocr is not None
    assert hits[0].ocr.lines[0].confidence == 0.856
    assert hits[0].ocr.model_revisions == ("PP-OCRv6-small@fixture",)
