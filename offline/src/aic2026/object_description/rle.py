"""Pure-Python compatible COCO compressed RLE encoding helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def _counts_from_mask(mask: np.ndarray) -> list[int]:
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    binary = np.asarray(mask, dtype=np.uint8)
    if not np.array_equal(binary, binary.astype(bool)):
        raise ValueError("mask must be binary")
    pixels = binary.reshape(-1, order="F")
    counts: list[int] = []
    previous = 0
    run_length = 0
    for pixel in pixels:
        value = int(pixel)
        if value == previous:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            previous = value
    counts.append(run_length)
    return counts


def _compress_counts(counts: Iterable[int]) -> str:
    values = list(counts)
    encoded: list[str] = []
    for index, raw_value in enumerate(values):
        value = raw_value - values[index - 2] if index > 2 else raw_value
        more = True
        while more:
            character = value & 0x1F
            value >>= 5
            more = value != -1 if character & 0x10 else value != 0
            if more:
                character |= 0x20
            character += 48
            encoded.append(chr(character))
    return "".join(encoded)


def _decompress_counts(encoded: str) -> list[int]:
    if not encoded:
        raise ValueError("compressed COCO RLE cannot be empty")
    counts: list[int] = []
    position = 0
    while position < len(encoded):
        value = 0
        shift = 0
        more = True
        character = 0
        while more:
            if position >= len(encoded):
                raise ValueError("truncated compressed COCO RLE")
            character = ord(encoded[position]) - 48
            if not 0 <= character <= 0x3F:
                raise ValueError("invalid character in compressed COCO RLE")
            value |= (character & 0x1F) << (5 * shift)
            more = bool(character & 0x20)
            position += 1
            shift += 1
        if character & 0x10:
            value |= -1 << (5 * shift)
        if len(counts) > 2:
            value += counts[-2]
        if value < 0:
            raise ValueError("invalid negative COCO RLE count")
        counts.append(value)
    return counts


def encode_mask(mask: np.ndarray) -> dict[str, object]:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        raise ValueError("mask must be a non-empty two-dimensional array")
    return {
        "size": [int(binary.shape[0]), int(binary.shape[1])],
        "counts": _compress_counts(_counts_from_mask(binary)),
    }


def decode_mask(rle: dict[str, object]) -> np.ndarray:
    size = rle.get("size")
    counts_value = rle.get("counts")
    if (
        not isinstance(size, list | tuple)
        or len(size) != 2
        or not all(isinstance(value, int) and value > 0 for value in size)
        or not isinstance(counts_value, str)
    ):
        raise ValueError("invalid COCO RLE object")
    counts = _decompress_counts(counts_value)
    total = int(size[0]) * int(size[1])
    if sum(counts) != total:
        raise ValueError("COCO RLE count total does not match mask dimensions")
    pixels = np.empty(total, dtype=bool)
    offset = 0
    value = False
    for count in counts:
        pixels[offset : offset + count] = value
        offset += count
        value = not value
    return pixels.reshape((int(size[0]), int(size[1])), order="F")


def rectangle_mask(height: int, width: int, bbox_xyxy_px: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy_px
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("rectangle lies outside the image")
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask
