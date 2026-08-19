"""Deterministic quadrilateral normalization and lossless crop generation."""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

from aic2026.contracts import CropProvenance, QuadGeometry
from aic2026.contracts.ocr_phase1 import canonical_quad_points

Point = tuple[float, float]
Quad = tuple[Point, Point, Point, Point]


class CropGeometryError(ValueError):
    """A detector polygon cannot produce a trustworthy crop."""


@dataclass(frozen=True, slots=True)
class CropConfig:
    horizontal_padding_ratio: float = 0.08
    vertical_padding_ratio: float = 0.18
    minimum_height_px: int = 96
    perspective_resampling: str = "bicubic"
    visual_hash_resampling: str = "bilinear"
    png_compress_level: int = 9
    algorithm: str = "aic26.pil_quad_crop.v3"

    def __post_init__(self) -> None:
        if type(self.horizontal_padding_ratio) not in (int, float) or not math.isfinite(
            self.horizontal_padding_ratio
        ):
            raise ValueError("horizontal padding ratio must be a finite JSON number")
        if type(self.vertical_padding_ratio) not in (int, float) or not math.isfinite(
            self.vertical_padding_ratio
        ):
            raise ValueError("vertical padding ratio must be a finite JSON number")
        if not 0 <= self.horizontal_padding_ratio <= 1:
            raise ValueError("horizontal padding ratio must be inside [0, 1]")
        if not 0 <= self.vertical_padding_ratio <= 1:
            raise ValueError("vertical padding ratio must be inside [0, 1]")
        if type(self.minimum_height_px) is not int or self.minimum_height_px < 1:
            raise ValueError("minimum crop height must be positive")
        if type(self.png_compress_level) is not int or self.png_compress_level not in range(10):
            raise ValueError("PNG compression level must be inside [0, 9]")
        if self.perspective_resampling != "bicubic":
            raise ValueError("only pinned bicubic perspective resampling is supported")
        if self.visual_hash_resampling != "bilinear":
            raise ValueError("only pinned bilinear visual-hash resampling is supported")
        if self.algorithm != "aic26.pil_quad_crop.v3":
            raise ValueError("unsupported crop algorithm")

    @property
    def sha256(self) -> str:
        value = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EncodedCrop:
    png_bytes: bytes
    provenance: CropProvenance


def canonical_quad(points: Iterable[Point]) -> Quad:
    """Canonicalize a quad by convex-hull adjacency, winding, and start vertex."""

    values = [(float(x), float(y)) for x, y in points]
    if len(values) != 4 or any(not math.isfinite(v) for point in values for v in point):
        raise CropGeometryError("polygon must contain exactly four finite points")
    if len(set(values)) != 4:
        raise CropGeometryError("polygon contains duplicate points")

    try:
        canonical = canonical_quad_points(tuple(values))
    except ValueError as error:
        raise CropGeometryError("polygon is degenerate or non-convex") from error
    return validate_canonical_quad(canonical)


def validate_canonical_quad(points: Iterable[Point]) -> Quad:
    """Validate and preserve detector-native TL, TR, BR, BL ordering."""

    values = tuple((float(x), float(y)) for x, y in points)
    try:
        geometry = QuadGeometry(points=values)
    except ValueError as error:
        raise CropGeometryError(f"polygon violates TL/TR/BR/BL contract: {error}") from error
    return geometry.points


def clamp_quad(quad: Quad, *, width: int, height: int) -> Quad:
    if width < 2 or height < 2:
        raise CropGeometryError("source frame dimensions must be at least 2x2")
    clamped = tuple(
        (min(max(x, 0.0), width - 1.0), min(max(y, 0.0), height - 1.0)) for x, y in quad
    )
    return canonical_quad(clamped)


def padded_quad(quad: Quad, *, width: int, height: int, config: CropConfig) -> tuple[Quad, float]:
    tl, tr, br, bl = quad
    px = config.horizontal_padding_ratio
    py = config.vertical_padding_ratio

    def add(*vectors: Point) -> Point:
        return (sum(item[0] for item in vectors), sum(item[1] for item in vectors))

    def scale(vector: Point, factor: float) -> Point:
        return (vector[0] * factor, vector[1] * factor)

    def vector(first: Point, second: Point) -> Point:
        return (second[0] - first[0], second[1] - first[1])

    top = vector(tl, tr)
    bottom = vector(bl, br)
    left = vector(tl, bl)
    right = vector(tr, br)
    for attempt in range(26):
        factor = 2.0**-attempt if attempt < 25 else 0.0
        expanded: Quad = (
            add(tl, scale(top, -px * factor), scale(left, -py * factor)),
            add(tr, scale(top, px * factor), scale(right, -py * factor)),
            add(br, scale(bottom, px * factor), scale(right, py * factor)),
            add(bl, scale(bottom, -px * factor), scale(left, py * factor)),
        )
        clamped_values = tuple(
            (min(max(x, 0.0), width - 1.0), min(max(y, 0.0), height - 1.0)) for x, y in expanded
        )
        try:
            accepted = canonical_quad(clamped_values)
        except CropGeometryError:
            continue
        requested = sum(
            math.dist(before, original) for before, original in zip(expanded, quad, strict=True)
        )
        lost = sum(
            math.dist(before, after) for before, after in zip(expanded, clamped_values, strict=True)
        )
        penalty = min(1.0, lost / requested) if requested > 0 else 0.0
        return accepted, penalty
    raise CropGeometryError("padded polygon collapsed after frame-bounds clamp")


def _perspective_size(quad: Quad) -> tuple[int, int]:
    tl, tr, br, bl = quad
    width = max(1, round(max(math.dist(tl, tr), math.dist(bl, br))))
    height = max(1, round(max(math.dist(tl, bl), math.dist(tr, br))))
    return width, height


def _transform(image: Image.Image, quad: Quad, size: tuple[int, int]) -> Image.Image:
    tl, tr, br, bl = quad
    # Pillow QUAD expects source corners UL, LL, LR, UR.
    data = (*tl, *bl, *br, *tr)
    return image.transform(
        size,
        Image.Transform.QUAD,
        data,
        resample=Image.Resampling.BICUBIC,
    )


def visual_hash(image: Image.Image) -> str:
    pixels = np.asarray(
        image.convert("L").resize((9, 8), resample=Image.Resampling.BILINEAR),
        dtype=np.int16,
    )
    bits = pixels[:, 1:] >= pixels[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def sharpness_score(image: Image.Image) -> float:
    pixels = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    components: list[np.ndarray] = []
    if pixels.shape[1] >= 3:
        components.append(np.diff(pixels, n=2, axis=1).ravel())
    if pixels.shape[0] >= 3:
        components.append(np.diff(pixels, n=2, axis=0).ravel())
    if not components:
        return 0.0
    values = np.concatenate(components)
    return float(np.mean(values * values))


def encode_crop(
    image: Image.Image,
    polygon: Iterable[Point],
    *,
    config: CropConfig,
) -> EncodedCrop:
    ordered = clamp_quad(canonical_quad(polygon), width=image.width, height=image.height)
    padded, edge_penalty = padded_quad(
        ordered, width=image.width, height=image.height, config=config
    )
    perspective_size = _perspective_size(padded)
    crop = _transform(image.convert("RGB"), padded, perspective_size)
    rotation_quadrants_ccw = 0
    if crop.height / crop.width >= 1.5:
        _tl, tr, br, _bl = padded
        rotation_quadrants_ccw = 3 if br[0] < tr[0] - 1e-9 else 1
        crop = crop.transpose(
            Image.Transpose.ROTATE_270 if rotation_quadrants_ccw == 3 else Image.Transpose.ROTATE_90
        )
    if crop.height < config.minimum_height_px:
        scale = config.minimum_height_px / crop.height
        crop = crop.resize(
            (max(1, round(crop.width * scale)), config.minimum_height_px),
            resample=Image.Resampling.BICUBIC,
        )
    buffer = io.BytesIO()
    crop.save(buffer, format="PNG", optimize=False, compress_level=config.png_compress_level)
    payload = buffer.getvalue()
    provenance = CropProvenance(
        crop_config_sha256=config.sha256,
        png_compress_level=config.png_compress_level,
        padded_polygon_xy=QuadGeometry(points=padded),
        perspective_width=perspective_size[0],
        perspective_height=perspective_size[1],
        output_width=crop.width,
        output_height=crop.height,
        rotation_quadrants_ccw=rotation_quadrants_ccw,
        png_sha256=hashlib.sha256(payload).hexdigest(),
        visual_hash=visual_hash(crop),
        sharpness=sharpness_score(crop),
        edge_truncation_penalty=edge_penalty,
    )
    return EncodedCrop(png_bytes=payload, provenance=provenance)


def reconstruct_crop(image: Image.Image, provenance: CropProvenance) -> bytes:
    crop = _transform(
        image.convert("RGB"),
        provenance.padded_polygon_xy.points,
        (provenance.perspective_width, provenance.perspective_height),
    )
    if provenance.rotation_quadrants_ccw:
        crop = crop.transpose(
            Image.Transpose.ROTATE_270
            if provenance.rotation_quadrants_ccw == 3
            else Image.Transpose.ROTATE_90
        )
    if (crop.width, crop.height) != (provenance.output_width, provenance.output_height):
        crop = crop.resize(
            (provenance.output_width, provenance.output_height),
            resample=Image.Resampling.BICUBIC,
        )
    buffer = io.BytesIO()
    crop.save(
        buffer,
        format="PNG",
        optimize=False,
        compress_level=provenance.png_compress_level,
    )
    return buffer.getvalue()


def visual_hash_distance(first: str, second: str) -> float:
    return (int(first, 16) ^ int(second, 16)).bit_count() / 64.0
