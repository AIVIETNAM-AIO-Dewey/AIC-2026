from __future__ import annotations

from fastapi.testclient import TestClient

from aic_backend.api.app import create_app
from aic_backend.api.deps import get_gpt, get_ocr_search_service, get_repository
from aic_backend.retrieval.fuzzy import explain_ocr_candidates, rerank_fuzzy_candidates
from aic_backend.retrieval.models import Evidence, FrameCandidate, StructuredOcr
from aic_backend.retrieval.ocr_search import OcrSearchService


class Gpt:
    client = None


class OcrRepo:
    def __init__(self) -> None:
        self.fuzzy: bool | None = None

    def status(self):
        return {
            "qdrant_ready": True,
            "collections": {
                "frames_sparse": False,
                "frames_dense": False,
                "regions": False,
                "ocr": True,
                "asr": False,
            },
            "models": {"siglip2_text": False, "e5_text": False},
        }

    def search_text(self, modality, query, *, limit, fuzzy=True, **kwargs):
        del kwargs
        self.fuzzy = fuzzy
        candidate = FrameCandidate(
            video_id="L23_V001",
            frame_idx=7,
            keyframe_n=7,
            pts_time_s=7.0,
            score=8.5,
            modality="ocr",
            evidence=Evidence(modality="ocr", text="NON SÔNG LIỀN MỘT DẢI", score=8.5),
            ocr=StructuredOcr(
                terminal_status="success",
                full_text="non sông liền một dải",
                width=1280,
                height=720,
                run_id="fixture",
                model_revisions=("PP-OCRv6-small@fixture",),
                source_image_sha256="a" * 64,
            ),
        )
        candidates = [candidate]
        if fuzzy:
            return rerank_fuzzy_candidates(query, candidates, limit=limit)
        return explain_ocr_candidates(query, candidates, limit=limit)


def test_ocr_only_api_controls_fuzzy_and_explains_match() -> None:
    repo = OcrRepo()
    app = create_app()
    app.dependency_overrides = {
        get_repository: lambda: repo,
        get_ocr_search_service: lambda: OcrSearchService(repo),
        get_gpt: lambda: Gpt(),
    }

    response = TestClient(app).post(
        "/api/v1/ocr/search",
        json={"query": "non song lien mot dai", "top_k": 10, "fuzzy": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert repo.fuzzy is False
    assert body["task_type"] == "ocr"
    assert "levenshtein_rerank" not in body["strategies"]
    assert body["results"][0]["ocr_match"] == {
        "query": "non song lien mot dai",
        "normalized_query": "non song lien mot dai",
        "matched_text": "NON SÔNG LIỀN MỘT DẢI",
        "lexical_score": 8.5,
        "fuzzy_similarity": None,
        "final_score": 8.5,
        "match_type": "accent_folded",
        "fuzzy_enabled": False,
    }
    assert body["results"][0]["ocr"]["model_revisions"] == ["PP-OCRv6-small@fixture"]
