"""OCR Text Extraction Subsystem."""

from .ocr_backend import (
    OcrReader,
    OcrResult,
    OcrSpan,
)
from .text_normalizer import (
    normalize_vietnamese_text,
)

# Alias for backward compatibility
EasyOcrReader = OcrReader

__all__ = [
    "OcrReader",
    "EasyOcrReader",
    "OcrResult",
    "OcrSpan",
    "normalize_vietnamese_text",
]
