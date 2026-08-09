from __future__ import annotations

import numpy as np
import pytest

from aic2026.object_description.rle import decode_mask, encode_mask, rectangle_mask


@pytest.mark.parametrize(
    "bbox",
    [(0, 0, 1, 1), (2, 1, 5, 4), (0, 0, 7, 6)],
)
def test_compressed_coco_rle_round_trip(bbox: tuple[int, int, int, int]) -> None:
    mask = rectangle_mask(6, 7, bbox)

    encoded = encode_mask(mask)
    decoded = decode_mask(encoded)

    assert isinstance(encoded["counts"], str)
    np.testing.assert_array_equal(decoded, mask)


def test_empty_masks_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        encode_mask(np.zeros((4, 4), dtype=bool))


def test_truncated_compressed_rle_is_rejected_cleanly() -> None:
    with pytest.raises(ValueError, match="truncated"):
        decode_mask({"size": [1, 1], "counts": "P"})
