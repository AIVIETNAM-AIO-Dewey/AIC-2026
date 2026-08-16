from aic2026.contracts.query import QuerySpec
from fastapi.testclient import TestClient

from aic_backend.api.app import create_app
from aic_backend.api.deps import (
    get_gpt,
    get_parser,
    get_repository,
    get_search_service,
    get_trake_service,
)
from aic_backend.application.search_service import SearchService
from aic_backend.application.trake_service import TrakeService
from aic_backend.domain.models import FrameCandidate


class Repo:
    def ready(self):
        return True

    def search_scene(self, query, *, limit, video_id=None):
        return [
            FrameCandidate(
                video_id="L21_V011", frame_idx=24925, pts_time_s=997.0, keyframe_n=262, score=0.9
            )
        ]

    def search_text(self, *args, **kwargs):
        return []

    def frame_image_path(self, *args):
        return None

    def neighbors(self, *args, **kwargs):
        return []


class Parser:
    def parse(self, *, task_type, raw_query_vi):
        return QuerySpec(task_type="kis", raw_query_vi=raw_query_vi, scene_en="scene")


class Gpt:
    client = None


def test_kis_response_uses_video_id_and_frame_idx_without_filename_fallback():
    app = create_app()
    repo = Repo()
    search = SearchService(repo)
    app.dependency_overrides = {
        get_repository: lambda: repo,
        get_parser: lambda: Parser(),
        get_search_service: lambda: search,
        get_trake_service: lambda: TrakeService(search),
        get_gpt: lambda: Gpt(),
    }
    response = TestClient(app).post(
        "/api/v1/search", json={"task_type": "kis", "raw_query_vi": "tìm cảnh", "top_k": 100}
    )
    assert response.status_code == 200
    hit = response.json()["results"][0]
    assert hit["video_id"] == "L21_V011" and hit["frame_idx"] == 24925
