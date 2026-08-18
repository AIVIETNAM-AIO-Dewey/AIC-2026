from __future__ import annotations

from types import SimpleNamespace

from aic_backend.api.app import _hit
from aic_backend.retrieval.models import OcrLine, SearchHit, StructuredOcr
from aic_backend.retrieval.qdrant import QdrantRepository


def _payload() -> dict[str, object]:
    return {
        "video_id": "L21_V011",
        "frame_idx": 24925,
        "keyframe_n": 262,
        "pts_time_s": 997.0,
        "text": "non sông liền một dải",
        "ocr_frame": {
            "terminal_status": "success",
            "full_text": "non sông liền một dải",
            "width": 1280,
            "height": 720,
            "run_id": "ocr-run",
            "model_revisions": ["PP-OCRv6-small@fixture"],
            "source_image_sha256": "a" * 64,
            "lines": [
                {
                    "line_id": "line-0001",
                    "raw_text": "NON SÔNG LIỀN MỘT DẢI",
                    "normalized_text": "non sông liền một dải",
                    "confidence": 0.856,
                    "accepted": True,
                    "polygon_xy": [[100, 200], [600, 200], [600, 260], [100, 260]],
                    "polygon_clamped": False,
                    "reading_order": 0,
                }
            ],
        },
    }


def test_qdrant_candidate_preserves_whole_structured_ocr_frame() -> None:
    candidate = QdrantRepository._candidate(
        SimpleNamespace(payload=_payload(), score=0.9, id="point-1"), "ocr"
    )
    assert candidate.ocr is not None
    assert candidate.ocr.width == 1280
    assert candidate.ocr.lines[0].polygon_xy == (
        (100.0, 200.0),
        (600.0, 200.0),
        (600.0, 260.0),
        (100.0, 260.0),
    )


def test_api_dto_exposes_confidence_polygon_and_model_provenance() -> None:
    ocr = StructuredOcr(
        terminal_status="success",
        full_text="non sông liền một dải",
        width=1280,
        height=720,
        run_id="ocr-run",
        model_revisions=("PP-OCRv6-small@fixture",),
        source_image_sha256="a" * 64,
        lines=(
            OcrLine(
                line_id="line-0001",
                raw_text="NON SÔNG LIỀN MỘT DẢI",
                normalized_text="non sông liền một dải",
                confidence=0.856,
                accepted=True,
                polygon_xy=((100.0, 200.0), (600.0, 200.0), (600.0, 260.0)),
            ),
        ),
    )
    response = _hit(
        SearchHit(
            video_id="L21_V011",
            frame_idx=24925,
            keyframe_n=262,
            pts_time_s=997.0,
            score=1.0,
            ocr=ocr,
        ),
        1,
    ).model_dump(mode="json")
    assert response["ocr"]["lines"][0]["confidence"] == 0.856
    assert response["ocr"]["lines"][0]["polygon_xy"][0] == [100.0, 200.0]
    assert response["ocr"]["model_revisions"] == ["PP-OCRv6-small@fixture"]
