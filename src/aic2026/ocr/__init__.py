"""OCR Text Extraction Subsystem."""

from .ocr_backend import (
    OcrReader,
    OcrResult,
    OcrSpan,
)
from .text_normalizer import (
    normalize_vietnamese_text,
)

__all__ = [
    "OcrReader",
    "OcrResult",
    "OcrSpan",
    "normalize_vietnamese_text",
]
