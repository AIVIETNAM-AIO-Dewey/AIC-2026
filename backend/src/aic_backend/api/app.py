"""FastAPI factory with request IDs, no query/image logging, and explicit failures."""

from __future__ import annotations

import time
from dataclasses import asdict
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ..ingest.artifacts import ArtifactFile, ingest
from ..ingest.sparse import fold_vietnamese
from ..llm.gpt4o import CapabilityUnavailable, GPT4oAdapter
from ..llm.query_parser import QueryParsingService
from ..ocr import OcrJobManager
from ..retrieval.models import SearchHit
from ..retrieval.ocr_search import OcrSearchService
from ..retrieval.search import SearchService
from ..retrieval.trake import TrakeService
from .deps import (
    get_gpt,
    get_ocr_job_manager,
    get_ocr_search_service,
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
    OcrJobRunRequest,
    OcrJobsResponse,
    OcrMatchResponse,
    OcrSearchRequest,
    OcrSearchResponse,
    SearchRequest,
    SearchResponse,
    StructuredOcrResponse,
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
        ocr=StructuredOcrResponse.model_validate(asdict(hit.ocr)) if hit.ocr else None,
        ocr_match=(
            OcrMatchResponse.model_validate(asdict(hit.ocr_match)) if hit.ocr_match else None
        ),
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
        status = repository.status()
        collections = status["collections"]
        models = status["models"]
        qdrant_ready = bool(status["qdrant_ready"])
        openai_ready = gpt.client is not None
        kis_missing = []
        if not qdrant_ready:
            kis_missing.append("qdrant")
        visual_ready = bool(collections["frames_sparse"] and models["siglip2_text"])
        lexical_ready = bool(collections["regions"] or collections["ocr"] or collections["asr"])
        if not visual_ready and not lexical_ready:
            kis_missing.append("search_collection")
        kis_ready = qdrant_ready and (visual_ready or lexical_ready)
        qa_missing = [*kis_missing] + ([] if openai_ready else ["gpt4o"])
        trake_missing = [*qa_missing]
        if not collections["frames_dense"]:
            trake_missing.append("frames_dense_current")
        return CapabilitiesResponse(
            qdrant_ready=qdrant_ready,
            openai_configured=openai_ready,
            image_answers_enabled=settings.enable_image_answers,
            search_ready=kis_ready,
            collections=collections,
            models=models,
            tasks={
                "kis": {"ready": kis_ready, "missing": kis_missing},
                "qa": {"ready": not qa_missing, "missing": qa_missing},
                "trake": {"ready": not trake_missing, "missing": trake_missing},
                "ocr": {
                    "ready": qdrant_ready and bool(collections["ocr"]),
                    "missing": (
                        []
                        if qdrant_ready and collections["ocr"]
                        else ["qdrant" if not qdrant_ready else "ocr_current"]
                    ),
                },
            },
        )

    @app.post("/api/v1/ocr/search", response_model=OcrSearchResponse)
    def ocr_search(
        request: Request,
        body: OcrSearchRequest,
        repository=Depends(get_repository),
        service: OcrSearchService = Depends(get_ocr_search_service),
    ) -> OcrSearchResponse:
        status = repository.status()
        if not status["qdrant_ready"]:
            raise HTTPException(status_code=503, detail="qdrant_unavailable")
        if not status["collections"]["ocr"]:
            raise HTTPException(status_code=503, detail="ocr_collection_unavailable")
        started = time.perf_counter()
        query = " ".join(body.query.split())
        hits = service.retrieve(query, top_k=body.top_k, fuzzy=body.fuzzy)
        return OcrSearchResponse(
            request_id=request.state.request_id,
            query=query,
            normalized_query=" ".join(fold_vietnamese(query).split()),
            fuzzy_enabled=body.fuzzy,
            strategies=[
                "exact_tokens",
                "accent_folded_tokens",
                "character_trigrams",
                *(["levenshtein_rerank"] if body.fuzzy else []),
            ],
            latency_ms=(time.perf_counter() - started) * 1000,
            results=[_hit(hit, rank) for rank, hit in enumerate(hits, 1)],
        )

    @app.get("/api/v1/ocr/jobs", response_model=OcrJobsResponse)
    def ocr_jobs(manager: OcrJobManager = Depends(get_ocr_job_manager)) -> dict[str, object]:
        return manager.status()

    @app.post("/api/v1/ocr/jobs/run", response_model=OcrJobsResponse)
    def run_ocr_job(
        body: OcrJobRunRequest,
        manager: OcrJobManager = Depends(get_ocr_job_manager),
    ) -> dict[str, object]:
        try:
            return manager.start(body.manifest_id)
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except (RuntimeError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/v1/ocr/jobs/index")
    def index_ocr_job(
        body: OcrJobRunRequest,
        manager: OcrJobManager = Depends(get_ocr_job_manager),
        repository=Depends(get_repository),
    ) -> dict[str, object]:
        if not manager.settings.ocr_jobs_enabled:
            raise HTTPException(status_code=403, detail="ocr_jobs_disabled")
        try:
            output, manifest = manager.completed_artifact(body.manifest_id)
            counts = ingest(
                repository.client,
                [ArtifactFile("ocr", output, manifest)],
                dense_encoder=repository.text_encoder,
                activate=True,
            )
        except FileNotFoundError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"status": "indexed", "manifest_id": body.manifest_id, "counts": counts}

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
