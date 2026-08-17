"""PhoWhisper ASR subsystem for the AIC-2026 video retrieval pipeline.

Provides audio extraction, speech-to-text decoding, sliding window
deduplication, keyframe pre-indexing, and text normalization.
"""

from .audio import extract_audio_pcm
from .backend import create_asr_backend
from .normalizer import normalize_transcript

__all__ = [
    "create_asr_backend",
    "extract_audio_pcm",
    "normalize_transcript",
]
