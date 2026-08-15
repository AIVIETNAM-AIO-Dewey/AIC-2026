"""PhoWhisper model loader with faster-whisper (CTranslate2) primary and
HuggingFace Transformers fallback.

Mirrors the factory-with-fallback pattern from the existing SpeechToText
repository (see ``server/app/engines/factory.py``).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Return type
# ──────────────────────────────────────────────────────────────────────


@dataclass
class AsrSegmentRaw:
    """Raw segment output from the decoding backend (before dedup)."""

    start_s: float
    end_s: float
    text: str


# ──────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────


class AsrBackend(ABC):
    """Contract that every ASR inference backend must fulfil."""

    @abstractmethod
    def load(
        self,
        model_id: str,
        device: str,
        compute_type: str,
    ) -> None:
        """Download / initialise model weights and move to *device*."""

    @abstractmethod
    def decode(
        self,
        audio: np.ndarray,
        *,
        language: str = "vi",
        initial_prompt: str | None = None,
        vad_filter: bool = True,
        vad_min_silence_duration_ms: int = 500,
    ) -> list[AsrSegmentRaw]:
        """Transcribe a float32 audio array and return raw segments."""

    @property
    @abstractmethod
    def model_identifier(self) -> str:
        """Human-readable identifier for the loaded model."""


# ──────────────────────────────────────────────────────────────────────
# faster-whisper (CTranslate2) backend — primary
# ──────────────────────────────────────────────────────────────────────


class FasterWhisperBackend(AsrBackend):
    """CTranslate2 / faster-whisper backend — fastest inference path."""

    def __init__(self) -> None:
        self._model: Any = None
        self._model_id: str = ""

    def load(
        self,
        model_id: str,
        device: str,
        compute_type: str,
    ) -> None:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        # Map generic device names to faster-whisper device strings
        fw_device = device
        if device in ("mps", "auto"):
            fw_device = "cpu"  # faster-whisper does not support MPS
        logger.info(
            "Loading faster-whisper model %s on %s (%s)",
            model_id, fw_device, compute_type,
        )
        self._model = WhisperModel(
            model_id,
            device=fw_device,
            compute_type=compute_type,
        )
        self._model_id = model_id

    def decode(
        self,
        audio: np.ndarray,
        *,
        language: str = "vi",
        initial_prompt: str | None = None,
        vad_filter: bool = True,
        vad_min_silence_duration_ms: int = 500,
    ) -> list[AsrSegmentRaw]:
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")

        vad_params = {}
        if vad_filter:
            vad_params["min_silence_duration_ms"] = vad_min_silence_duration_ms

        segments_iter, _info = self._model.transcribe(
            audio,
            language=language,
            initial_prompt=initial_prompt,
            word_timestamps=False,
            vad_filter=vad_filter,
            vad_parameters=vad_params if vad_params else None,
        )

        results: list[AsrSegmentRaw] = []
        for seg in segments_iter:
            text = seg.text.strip()
            if text:
                results.append(AsrSegmentRaw(
                    start_s=seg.start,
                    end_s=seg.end,
                    text=text,
                ))
        return results

    @property
    def model_identifier(self) -> str:
        return f"faster-whisper:{self._model_id}"


# ──────────────────────────────────────────────────────────────────────
# HuggingFace Transformers backend — fallback
# ──────────────────────────────────────────────────────────────────────


class HuggingFaceBackend(AsrBackend):
    """Standard HuggingFace Transformers pipeline — universal fallback.

    Uses ``transformers.pipeline("automatic-speech-recognition")`` with
    ``return_timestamps=True`` to produce segment-level timestamps.
    """

    def __init__(self) -> None:
        self._pipe: Any = None
        self._model_id: str = ""

    def load(
        self,
        model_id: str,
        device: str,
        compute_type: str,
    ) -> None:
        import torch
        from transformers import pipeline  # type: ignore[import-untyped]

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                resolved = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                resolved = "mps"
            else:
                resolved = "cpu"
        else:
            resolved = device

        torch_dtype = torch.float16 if "16" in compute_type else torch.float32

        logger.info(
            "Loading HuggingFace ASR pipeline %s on %s (dtype=%s)",
            model_id, resolved, torch_dtype,
        )
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=resolved,
            torch_dtype=torch_dtype,
        )
        self._model_id = model_id

    def decode(
        self,
        audio: np.ndarray,
        *,
        language: str = "vi",
        initial_prompt: str | None = None,
        vad_filter: bool = True,
        vad_min_silence_duration_ms: int = 500,
    ) -> list[AsrSegmentRaw]:
        if self._pipe is None:
            raise RuntimeError("Model not loaded — call load() first")

        # HF pipeline expects {"raw": np.ndarray, "sampling_rate": int}
        hf_input = {"raw": audio, "sampling_rate": 16_000}
        generate_kwargs: dict[str, Any] = {"language": language}

        result = self._pipe(
            hf_input,
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )

        segments: list[AsrSegmentRaw] = []
        chunks = result.get("chunks", [])

        for chunk in chunks:
            text = chunk.get("text", "").strip()
            ts = chunk.get("timestamp", (None, None))
            if text and ts[0] is not None and ts[1] is not None:
                segments.append(AsrSegmentRaw(
                    start_s=ts[0],
                    end_s=ts[1],
                    text=text,
                ))

        return segments

    @property
    def model_identifier(self) -> str:
        return f"huggingface:{self._model_id}"


# ──────────────────────────────────────────────────────────────────────
# Factory with automatic fallback
# ──────────────────────────────────────────────────────────────────────


def create_asr_backend(
    engine: str,
    model_id: str,
    device: str = "auto",
    compute_type: str = "float16",
) -> AsrBackend:
    """Create and load an ASR backend with automatic fallback.

    Parameters
    ----------
    engine:
        ``"faster_whisper"`` (primary) or ``"huggingface"`` (fallback).
    model_id:
        HuggingFace model ID or CTranslate2 model path/ID.
    device:
        ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
    compute_type:
        Quantisation hint: ``"float16"``, ``"int8"``, ``"float32"``.

    Returns
    -------
    AsrBackend
        A loaded, ready-to-decode backend instance.
    """
    _ENGINE_MAP: dict[str, type[AsrBackend]] = {
        "faster_whisper": FasterWhisperBackend,
        "huggingface": HuggingFaceBackend,
    }

    backend_cls = _ENGINE_MAP.get(engine)
    if backend_cls is None:
        available = ", ".join(sorted(_ENGINE_MAP))
        raise ValueError(
            f"Unknown ASR engine '{engine}'. Available: {available}"
        )

    logger.info("Creating ASR backend: engine=%s, model=%s", engine, model_id)

    try:
        backend = backend_cls()
        backend.load(model_id, device, compute_type)
        return backend
    except Exception as exc:
        if engine == "faster_whisper":
            logger.warning(
                "faster-whisper failed (%s), falling back to HuggingFace...",
                exc,
            )
            fallback = HuggingFaceBackend()
            # Fall back to the base PhoWhisper model (not CT2 variant)
            fallback_model = model_id.replace("-ct2", "")
            if fallback_model == model_id:
                # Not a CT2 model ID — try the vinai original
                fallback_model = "vinai/PhoWhisper-large"
            fallback.load(fallback_model, device, compute_type)
            logger.info("HuggingFace fallback loaded successfully.")
            return fallback
        raise
