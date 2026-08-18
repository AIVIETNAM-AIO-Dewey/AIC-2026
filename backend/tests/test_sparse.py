from __future__ import annotations

from aic_backend.ingest.sparse import fold_vietnamese, sparse_vector


def _dot(left: tuple[list[int], list[float]], right: tuple[list[int], list[float]]) -> float:
    left_values = dict(zip(*left, strict=True))
    right_values = dict(zip(*right, strict=True))
    return sum(
        left_values[index] * right_values[index] for index in left_values.keys() & right_values
    )


def test_fold_vietnamese_handles_diacritics_and_d_stroke() -> None:
    assert fold_vietnamese("Đất nước – NON SÔNG một dải") == "dat nuoc – non song mot dai"


def test_sparse_vector_is_sorted_unique_and_deterministic() -> None:
    first = sparse_vector("Thành phố Hồ Chí Minh")
    second = sparse_vector("Thành phố Hồ Chí Minh")
    assert first == second
    assert first[0] == sorted(set(first[0]))
    assert len(first[0]) == len(first[1])


def test_accentless_noisy_query_matches_vietnamese_ocr_phrase() -> None:
    document = sparse_vector("non sông liền một dải")
    noisy_query = sparse_vector("non song cung mot dai")
    unrelated_query = sparse_vector("xe buýt trên đường phố")

    assert _dot(document, noisy_query) > _dot(document, unrelated_query)


def test_trigrams_help_other_spelling_errors_without_phrase_rules() -> None:
    document = sparse_vector("thành phố hồ chí minh")
    noisy_query = sparse_vector("thanh pho ho chi minhh")
    unrelated_query = sparse_vector("bóng đá sân vận động")

    assert _dot(document, noisy_query) > _dot(document, unrelated_query)
