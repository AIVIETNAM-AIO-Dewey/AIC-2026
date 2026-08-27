"""Unit tests for the Stage 2 Heuristic Scoring layer.

Runs without Qdrant or GPU: the cross-encoder is stubbed so the tests isolate the
scoring maths rather than the model.
"""

from __future__ import annotations

from online.src.contracts.query import ParsedQuery
from online.src.retrieval.stage2_reranker import Stage2Reranker, _normalize_text


class StubRegistry:
    """Returns a fixed cross-encoder score per dossier, in call order."""

    def __init__(self, scores: list[float]):
        self.scores = scores
        self.seen_dossiers: list[str] = []

    def compute_rerank_scores(self, query: str, documents: list[str]) -> list[float]:
        self.seen_dossiers = list(documents)
        return self.scores[: len(documents)]


def make_reranker(scores: list[float]) -> Stage2Reranker:
    return Stage2Reranker(registry=StubRegistry(scores), vqa_reasoner=object())


def candidate(**kw):
    base = {
        "video_id": "L01_V001",
        "frame_idx": 100,
        "normalized_score": 0.5,
        "active_channels": 1,
        "prob_vis": 0.0,
        "dam_summary": "",
        "asr_transcript": "",
        "ocr_text": "",
    }
    base.update(kw)
    return base


def query(**kw):
    base = {"original_query": "nguoi dan ong mac ao do", "ocr_keywords": [], "objects_en": []}
    base.update(kw)
    return ParsedQuery(**base)


def test_empty_pool_returns_empty():
    assert make_reranker([]).rerank_kis(query(), []) == []


def test_textless_candidate_falls_back_to_stage1():
    """A frame with no text payload must not be buried by a near-zero CE score."""
    r = make_reranker([0.01])
    c = candidate(normalized_score=1.0)
    out = r.rerank_kis(query(), [c])

    # textless: 0.75*1.0 + 0.25*0.01 = 0.7525
    assert out[0]["final_score"] == 0.7525
    assert "textless" in out[0]["heuristic_breakdown"]
    assert out[0]["dossier_coverage"] == {"dam": False, "asr": False, "ocr": False}

    # Under the old hard blend this frame would have scored 0.40*1.0 + 0.60*0.01 = 0.406
    assert out[0]["final_score"] > 0.406


def test_text_candidate_uses_standard_blend():
    r = make_reranker([0.8])
    c = candidate(normalized_score=0.5, asr_transcript="xin chao cac ban")
    out = r.rerank_kis(query(), [c])

    # 0.35*0.5 + 0.65*0.8 = 0.695, no bonuses (1 channel, no prob_vis, no ocr/dam match)
    assert out[0]["final_score"] == 0.695
    assert out[0]["dossier_coverage"]["asr"] is True


def test_channel_consensus_bonus():
    """Same CE score: the frame agreed on by more channels must win."""
    r = make_reranker([0.7, 0.7])
    lonely = candidate(frame_idx=1, active_channels=1, asr_transcript="abc")
    agreed = candidate(frame_idx=2, active_channels=4, asr_transcript="abc")
    out = r.rerank_kis(query(), [lonely, agreed])

    assert out[0]["frame_idx"] == 2
    # 3 extra channels * 0.04 = 0.12
    assert round(out[0]["final_score"] - out[1]["final_score"], 4) == 0.12
    assert "+consensus(3ch)" in out[0]["heuristic_breakdown"]


def test_ocr_exact_match_bonus_is_proportional():
    r = make_reranker([0.5])
    c = candidate(ocr_text="GIAI VO DICH QUOC GIA 2024", asr_transcript="x")
    out = r.rerank_kis(query(ocr_keywords=["quoc gia", "khong co"]), [c])

    # 1 of 2 keywords hit exactly -> 0.08 * 0.5 = 0.04
    assert "+ocr(exact 1 fuzzy 0/2)=0.040" in out[0]["heuristic_breakdown"]


def test_dam_coverage_and_visual_confidence_stack():
    r = make_reranker([0.5])
    c = candidate(dam_summary="a man in a red shirt holding a microphone", prob_vis=0.9)
    out = r.rerank_kis(query(objects_en=["red shirt", "microphone"]), [c])

    bd = out[0]["heuristic_breakdown"]
    assert "+dam(2/2)=0.060" in bd       # both objects detected
    assert "+visual(0.900)=0.090" in bd  # 0.10 * 0.9
    # 0.35*0.5 + 0.65*0.5 + 0.06 + 0.09 = 0.65
    assert out[0]["final_score"] == 0.65


def test_score_never_negative():
    r = make_reranker([0.0])
    out = r.rerank_kis(query(), [candidate(normalized_score=0.0)])
    assert out[0]["final_score"] >= 0.0


def test_top_k_rerank_caps_cross_encoder_calls():
    r = make_reranker([0.5] * 100)
    pool = [candidate(frame_idx=i, asr_transcript="x") for i in range(80)]
    out = r.rerank_kis(query(), pool, final_top_k=10, top_k_rerank=25)

    assert len(r.registry.seen_dossiers) == 25
    assert len(out) == 10
    assert [x["final_rank"] for x in out] == list(range(1, 11))


def test_normalize_folds_vietnamese_diacritics():
    assert _normalize_text("giây") == "giay"
    assert _normalize_text("Đường") == "duong"
    assert _normalize_text("06:30:11 giây") == "06 30 11 giay"
    assert _normalize_text("") == ""


def test_accented_query_scores_full_credit_on_clean_ocr():
    """The normal case: query is written in Vietnamese, PPOCR read the tone correctly."""
    r = make_reranker([0.5])
    c = candidate(ocr_text="71 06:30:11 giây")
    out = r.rerank_kis(query(ocr_keywords=["giây"]), [c])
    assert "+ocr(exact 1 fuzzy 0/1)=0.080" in out[0]["heuristic_breakdown"]


def test_accented_query_gets_half_credit_when_ocr_mangled_the_tone():
    """Real data: 'giây' is also read as 'giày'/'giay'/'giấy' across frames."""
    r = make_reranker([0.5])
    c = candidate(ocr_text="06830816 malun giày")
    out = r.rerank_kis(query(ocr_keywords=["giây"]), [c])
    # folded hit only -> 0.08 * (0.5 / 1) = 0.04
    assert "+ocr(exact 0 fuzzy 1/1)=0.040" in out[0]["heuristic_breakdown"]


def test_short_token_never_matches_via_folding():
    """'ở' folds to 'o' and would otherwise match ô/ổ/ộ/ố across 8k+ real rows."""
    r = make_reranker([0.5])
    c = candidate(ocr_text="con số 15")
    out = r.rerank_kis(query(ocr_keywords=["ở"]), [c])
    assert "+ocr" not in out[0]["heuristic_breakdown"]


def test_exact_hit_outranks_folded_hit():
    r = make_reranker([0.5, 0.5])
    clean = candidate(frame_idx=1, ocr_text="thời gian 06:30:11 giây")
    mangled = candidate(frame_idx=2, ocr_text="thoi gian 06830816 giày")
    out = r.rerank_kis(query(ocr_keywords=["giây"]), [clean, mangled])
    assert out[0]["frame_idx"] == 1
    assert round(out[0]["final_score"] - out[1]["final_score"], 4) == 0.04


def test_dam_matching_still_works_after_folding():
    r = make_reranker([0.5])
    c = candidate(dam_summary="A tall SKYSCRAPER, with lit windows.")
    out = r.rerank_kis(query(objects_en=["skyscraper", "windows"]), [c])
    assert "+dam(2/2)=0.060" in out[0]["heuristic_breakdown"]
