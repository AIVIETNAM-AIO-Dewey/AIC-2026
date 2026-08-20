from __future__ import annotations

from aic2026.contracts import OcrTrajectoryRecord, RepresentativeCropBinding, TrajectoryMember
from aic2026.ocr.representative_recognition import RepresentativeRecognitionResult
from aic2026.ocr.trajectory_consensus import _choose_consensus


def _result(
    *,
    rank: int,
    quality: float,
    transcript_raw: str,
    transcript_nfc: str,
    confidence: float,
) -> RepresentativeRecognitionResult:
    binding = RepresentativeCropBinding.model_construct(
        representative_rank=rank,
        quality_score=quality,
    )
    return RepresentativeRecognitionResult.model_construct(
        binding=binding,
        model_id="fixture-vietocr",
        model_revision="f" * 64,
        status="ok",
        transcript_raw=transcript_raw,
        transcript_nfc=transcript_nfc,
        confidence=confidence,
        latency_ms=1.0,
        error_type=None,
        error_message=None,
    )


def _trajectory() -> OcrTrajectoryRecord:
    return OcrTrajectoryRecord.model_construct(
        video_id="video",
        trajectory_id="video:trajectory-1",
        members=[TrajectoryMember.model_construct(frame_uid="video:0")],
    )


def _choose(results: list[RepresentativeRecognitionResult]):
    return _choose_consensus(
        _trajectory(),
        results,
        source_run_id="phase2-fixture",
        model_id="fixture-vietocr",
        model_revision="f" * 64,
    )


def test_consensus_decision_is_independent_of_native_confidence() -> None:
    low_rank_one = _choose(
        [
            _result(
                rank=1,
                quality=0.9,
                transcript_raw="alpha",
                transcript_nfc="alpha",
                confidence=0.001,
            ),
            _result(
                rank=2,
                quality=0.8,
                transcript_raw="beta",
                transcript_nfc="beta",
                confidence=0.999,
            ),
        ]
    )
    high_rank_one = _choose(
        [
            _result(
                rank=1,
                quality=0.9,
                transcript_raw="alpha",
                transcript_nfc="alpha",
                confidence=0.999,
            ),
            _result(
                rank=2,
                quality=0.8,
                transcript_raw="beta",
                transcript_nfc="beta",
                confidence=0.001,
            ),
        ]
    )

    decision_fields = (
        "transcript_nfc",
        "status",
        "supporting_ranks",
        "disagreeing_ranks",
        "method",
    )
    assert tuple(getattr(low_rank_one, field) for field in decision_fields) == tuple(
        getattr(high_rank_one, field) for field in decision_fields
    )
    assert low_rank_one.transcript_nfc == "alpha"
    assert low_rank_one.supporting_ranks == [1]
    assert low_rank_one.confidence == 0.001
    assert high_rank_one.confidence == 0.999

    decomposed = "e\N{COMBINING ACUTE ACCENT}"
    composed = "\N{LATIN SMALL LETTER E WITH ACUTE}"
    representative_low_confidence = _choose(
        [
            _result(
                rank=1,
                quality=0.9,
                transcript_raw=decomposed,
                transcript_nfc=composed,
                confidence=0.001,
            ),
            _result(
                rank=2,
                quality=0.8,
                transcript_raw=composed,
                transcript_nfc=composed,
                confidence=0.999,
            ),
        ]
    )
    representative_high_confidence = _choose(
        [
            _result(
                rank=1,
                quality=0.9,
                transcript_raw=decomposed,
                transcript_nfc=composed,
                confidence=0.999,
            ),
            _result(
                rank=2,
                quality=0.8,
                transcript_raw=composed,
                transcript_nfc=composed,
                confidence=0.001,
            ),
        ]
    )
    assert representative_low_confidence.transcript_raw == decomposed
    assert representative_high_confidence.transcript_raw == decomposed
