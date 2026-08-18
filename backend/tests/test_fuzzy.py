from __future__ import annotations

from aic_backend.retrieval.fuzzy import rerank_fuzzy_candidates, substring_edit_similarity
from aic_backend.retrieval.models import Evidence, FrameCandidate


def _candidate(frame_idx: int, text: str, score: float) -> FrameCandidate:
    return FrameCandidate(
        video_id="L21_V001",
        frame_idx=frame_idx,
        pts_time_s=float(frame_idx),
        score=score,
        modality="ocr",
        evidence=Evidence(modality="ocr", text=text, score=score),
    )


def test_substring_edit_similarity_handles_accents_and_ocr_word_error() -> None:
    assert substring_edit_similarity("non song cung mot dai", "HTV NON SÔNG LIỀN MỘT DẢI") > 0.68
    assert substring_edit_similarity("non song cung mot dai", "xe buýt trên đường phố") < 0.4


def test_fuzzy_rerank_is_generic_bounded_and_deduplicates_frames() -> None:
    candidates = [
        _candidate(1, "một kết quả dense không liên quan", 0.9),
        _candidate(2, "NON SÔNG LIỀN MỘT DẢI", 0.5),
        _candidate(2, "logo HTV", 0.4),
        _candidate(3, "thanh pho ho chi minh", 0.3),
    ]
    ranked = rerank_fuzzy_candidates("non song cung mot dai", candidates, limit=3)
    assert [row.frame_idx for row in ranked] == [2, 1, 3]

    generic = rerank_fuzzy_candidates("thanh pho ho chi minhh", candidates, limit=2)
    assert generic[0].frame_idx == 3
