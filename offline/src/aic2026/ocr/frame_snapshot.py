"""Single-decode immutable frame snapshots shared by detector and crop geometry."""

from __future__ import annotations

import hashlib
import io
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from aic2026.contracts import FrameRef

CANONICAL_IMAGE_ALGORITHM = "aic26.exif_transpose.rgb8.v1"
SUPPORTED_SOURCE_MODES = frozenset({"RGB", "RGBA", "L"})


class CanonicalFrameError(ValueError):
    """Source bytes cannot satisfy the Phase 1 canonical image policy."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class CanonicalFrameSnapshot:
    image: Image.Image
    bgr: np.ndarray
    canonical_image_sha256: str


def _canonical_pixel_hash(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(CANONICAL_IMAGE_ALGORITHM.encode("ascii"))
    digest.update(b"\0")
    digest.update(struct.pack(">II", image.width, image.height))
    digest.update(image.tobytes())
    return digest.hexdigest()


def decode_canonical_frame(ref: FrameRef, path: Path) -> CanonicalFrameSnapshot:
    """Read bytes once, verify identity, normalize EXIF, and produce RGB/BGR8 views."""

    try:
        payload = path.read_bytes()
    except OSError as error:
        raise CanonicalFrameError(
            "source_unavailable", f"source image is unavailable: {ref.frame_uid}"
        ) from error
    source_hash = hashlib.sha256(payload).hexdigest()
    if source_hash != ref.source_image_sha256:
        raise CanonicalFrameError(
            "source_checksum_drift", f"source image checksum drift: {ref.frame_uid}"
        )
    try:
        with Image.open(io.BytesIO(payload)) as source:
            source.load()
            if source.mode not in SUPPORTED_SOURCE_MODES:
                raise CanonicalFrameError(
                    "unsupported_source_mode",
                    f"unsupported source image mode {source.mode!r}: {ref.frame_uid}",
                )
            image = ImageOps.exif_transpose(source).convert("RGB")
    except CanonicalFrameError:
        raise
    except OSError as error:
        raise CanonicalFrameError(
            "corrupt_source_image", f"source image is corrupt: {ref.frame_uid}"
        ) from error
    if image.size != (ref.width, ref.height):
        raise CanonicalFrameError(
            "canonical_dimension_mismatch",
            f"canonical image dimensions {image.size} differ from manifest "
            f"{(ref.width, ref.height)}: {ref.frame_uid}",
        )
    rgb = np.asarray(image, dtype=np.uint8)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    bgr.setflags(write=False)
    return CanonicalFrameSnapshot(
        image=image,
        bgr=bgr,
        canonical_image_sha256=_canonical_pixel_hash(image),
    )
