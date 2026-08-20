from __future__ import annotations

import io
from pathlib import Path

import yaml
from aic2026.ocr.geometry import CropConfig, encode_crop, reconstruct_crop
from PIL import Image


def _textured_frame() -> Image.Image:
    width, height = 320, 180
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            pixels.extend(((3 * x + y) % 256, (x + 5 * y) % 256, (x * y // 7) % 256))
    return Image.frombytes("RGB", (width, height), bytes(pixels))


def test_fast_png_compression_preserves_all_crop_semantics() -> None:
    image = _textured_frame()
    polygon = ((17, 19), (299, 11), (307, 151), (13, 167))
    level_nine = encode_crop(image, polygon, config=CropConfig(png_compress_level=9))
    level_one = encode_crop(image, polygon, config=CropConfig(png_compress_level=1))

    assert level_nine.png_bytes != level_one.png_bytes
    assert level_nine.provenance.png_sha256 != level_one.provenance.png_sha256
    with (
        Image.open(io.BytesIO(level_nine.png_bytes)) as decoded_nine,
        Image.open(io.BytesIO(level_one.png_bytes)) as decoded_one,
    ):
        assert decoded_nine.mode == decoded_one.mode == "RGB"
        assert decoded_nine.size == decoded_one.size
        assert decoded_nine.tobytes() == decoded_one.tobytes()

    nine = level_nine.provenance.model_dump(mode="json")
    one = level_one.provenance.model_dump(mode="json")
    differing_fields = {key for key in nine if nine[key] != one[key]}
    assert differing_fields == {
        "crop_config_sha256",
        "png_compress_level",
        "png_sha256",
    }
    assert level_nine.provenance.padded_polygon_xy == level_one.provenance.padded_polygon_xy
    assert level_nine.provenance.visual_hash == level_one.provenance.visual_hash
    assert level_nine.provenance.sharpness == level_one.provenance.sharpness
    assert reconstruct_crop(image, level_nine.provenance) == level_nine.png_bytes
    assert reconstruct_crop(image, level_one.provenance) == level_one.png_bytes


def test_only_gpu_profile_uses_fast_lossless_png_compression() -> None:
    offline_root = Path(__file__).resolve().parents[2]
    cpu = yaml.safe_load(
        (offline_root / "configs" / "offline" / "ocr_phase1.yaml").read_text(encoding="utf-8")
    )
    gpu = yaml.safe_load(
        (offline_root / "configs" / "offline" / "ocr_phase1_kaggle_gpu.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert cpu["crop"]["png_compress_level"] == 9
    assert gpu["crop"]["png_compress_level"] == 1
