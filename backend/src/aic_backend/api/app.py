"""FastAPI factory with request IDs, no query/image logging, and explicit failures."""

from __future__ import annotations

import time
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ..application.query_parser import QueryParsingService
from ..application.search_service import SearchService
from ..application.trake_service import TrakeService
from ..domain.models import SearchHit
from ..infrastructure.openai.gpt4o import CapabilityUnavailable, GPT4oAdapter
from .deps import (
    get_gpt,
    get_parser,
    get_repository,
    get_search_service,
    get_settings,
    get_trake_service,
)
from .schemas import (
    CapabilitiesResponse,
    EvidenceResponse,
    FrameHitResponse,
    SearchRequest,
    SearchResponse,
    SubmissionRenderRequest,
    TrakeEventResponse,
    TrakeSequenceResponse,
)


def _hit(hit: SearchHit, rank: int) -> FrameHitResponse:
    return FrameHitResponse(
        rank=rank,
        score=hit.score,
        video_id=hit.video_id,
        frame_idx=hit.frame_idx,
        keyframe_n=hit.keyframe_n,
        pts_time_s=hit.pts_time_s,
        image_url=f"/api/v1/frames/{hit.video_id}/{hit.frame_idx}/image",
        modality_scores=hit.modality_scores,
        evidence=[EvidenceResponse(**item.__dict__) for item in hit.evidence],
    )


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="AIC 2026 MultiRetrieval API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(item) for item in settings.cors_origins],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        request.state.request_id = str(uuid4())
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.get("/api/v1/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/health/ready")
    def ready(repository=Depends(get_repository)) -> dict[str, str]:
        if not repository.ready():
            raise HTTPException(status_code=503, detail="qdrant_unavailable")
        return {"status": "ready"}

    @app.get("/api/v1/capabilities", response_model=CapabilitiesResponse)
    def capabilities(
        repository=Depends(get_repository), gpt: GPT4oAdapter = Depends(get_gpt)
    ) -> CapabilitiesResponse:
        return CapabilitiesResponse(
            qdrant_ready=repository.ready(),
            openai_configured=gpt.client is not None,
            image_answers_enabled=settings.enable_image_answers,
        )

    @app.post("/api/v1/search", response_model=SearchResponse)
    def search(
        request: Request,
        body: SearchRequest,
        parser: QueryParsingService = Depends(get_parser),
        service: SearchService = Depends(get_search_service),
        trake: TrakeService = Depends(get_trake_service),
        gpt: GPT4oAdapter = Depends(get_gpt),
    ) -> SearchResponse:
        started = time.perf_counter()
        stage: dict[str, float] = {}
        try:
            parsed = parser.parse(task_type=body.task_type, raw_query_vi=body.raw_query_vi)
        except CapabilityUnavailable as error:
            raise HTTPException(status_code=503, detail="capability_unavailable") from error
        stage["parse"] = (time.perf_counter() - started) * 1000
        if body.task_type == "trake":
            sequences = trake.retrieve(parsed, top_k=body.top_k)
            stage["retrieve"] = (time.perf_counter() - started) * 1000 - stage["parse"]
            return SearchResponse(
                request_id=request.state.request_id,
                task_type=body.task_type,
                latency_ms=(time.perf_counter() - started) * 1000,
                stage_latency_ms=stage,
                sequences=[
                    TrakeSequenceResponse(
                        rank=index,
                        video_id=sequence.video_id,
                        score=sequence.score,
                        events=[
                            TrakeEventResponse(
                                event_index=event.event_index,
                                frame=_hit(event.frame, event.event_index + 1),
                            )
                            for event in sequence.events
                        ],
                    )
                    for index, sequence in enumerate(sequences, 1)
                ],
            )
        hits = service.retrieve(parsed, top_k=body.top_k)
        stage["retrieve"] = (time.perf_counter() - started) * 1000 - stage["parse"]
        answer = None
        confidence = None
        evidence: list[str] = []
        if body.task_type == "qa":
            try:
                answer, confidence, evidence = gpt.answer(
                    query=parsed,
                    frames=hits[:20],
                    use_images=body.use_images_for_answer and settings.enable_image_answers,
                )
            except CapabilityUnavailable as error:
                raise HTTPException(status_code=503, detail="capability_unavailable") from error
            stage["answer"] = (
                (time.perf_counter() - started) * 1000 - stage["parse"] - stage["retrieve"]
            )
        return SearchResponse(
            request_id=request.state.request_id,
            task_type=body.task_type,
            latency_ms=(time.perf_counter() - started) * 1000,
            stage_latency_ms=stage,
            results=[_hit(hit, rank) for rank, hit in enumerate(hits, 1)],
            answer=answer,
            confidence=confidence,
            evidence_frame_uids=evidence,
            degraded=body.task_type == "kis" and gpt.client is None,
        )

    @app.get("/api/v1/frames/{video_id}/{frame_idx}/image")
    def frame_image(video_id: str, frame_idx: int, repository=Depends(get_repository)):
        path = repository.frame_image_path(video_id, frame_idx)
        if not path:
            raise HTTPException(status_code=404, detail="frame_image_not_found")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/v1/frames/{video_id}/{frame_idx}/neighbors")
    def neighbors(
        video_id: str, frame_idx: int, radius_s: float = 2.0, repository=Depends(get_repository)
    ) -> list[FrameHitResponse]:
        return [
            _hit(hit, rank)
            for rank, hit in enumerate(
                repository.neighbors(video_id, frame_idx, radius_s=radius_s), 1
            )
        ]

    @app.post("/api/v1/submissions/render")
    def render_submission(body: SubmissionRenderRequest) -> dict[str, object]:
        if body.task_type == "qa":
            return {
                "task_type": "qa",
                "answer": body.answer,
                "evidence": [item.frame_idx for item in body.frames],
            }
        if body.task_type == "trake":
            return {
                "task_type": "trake",
                "sequences": [
                    [event.frame.frame_idx for event in item.events] for item in body.sequences
                ],
            }
        return {"task_type": "kis", "frame_indices": [item.frame_idx for item in body.frames]}

    return app


app = create_app()
