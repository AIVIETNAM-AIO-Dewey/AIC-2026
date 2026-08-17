"""Audio extraction from video files via ffmpeg subprocess.

Extracts audio as 16 kHz mono float32 numpy arrays entirely in-memory
(no temporary .wav files) to conserve Kaggle working disk.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# WAV header is exactly 44 bytes for standard PCM format
_WAV_HEADER_SIZE = 44


class AudioExtractionError(Exception):
    """Raised when ffmpeg fails to extract audio from a video."""


def extract_audio_pcm(
    video_path: str | Path,
    sample_rate: int = 16_000,
) -> np.ndarray:
    """Extract audio from a video file as a float32 numpy array.

    Uses ``ffmpeg`` to decode audio to 16-bit signed PCM, resampled to
    *sample_rate* Hz, mixed down to mono, and piped to stdout.  The
    44-byte WAV header is stripped to produce a raw PCM buffer that is
    then normalised to ``[-1.0, 1.0]`` float32.

    Parameters
    ----------
    video_path:
        Absolute or relative path to the ``.mp4`` (or other ffmpeg-
        supported container) file.
    sample_rate:
        Target sample rate in Hz.  PhoWhisper expects 16 000.

    Returns
    -------
    np.ndarray
        1-D float32 array of shape ``(num_samples,)``.

    Raises
    ------
    AudioExtractionError
        If ffmpeg exits with a non-zero code (e.g. the video contains
        no audio track).
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",  # discard video stream
        "-acodec",
        "pcm_s16le",  # 16-bit signed little-endian PCM
        "-ar",
        str(sample_rate),
        "-ac",
        "1",  # mono
        "-f",
        "wav",
        "pipe:1",  # write to stdout
    ]

    logger.debug("Running ffmpeg: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr_text = exc.stderr.decode("utf-8", errors="replace").strip()
        raise AudioExtractionError(f"ffmpeg failed for {video_path.name}: {stderr_text}") from exc

    raw_bytes = result.stdout
    if len(raw_bytes) <= _WAV_HEADER_SIZE:
        raise AudioExtractionError(
            f"No audio data extracted from {video_path.name} "
            f"(output was {len(raw_bytes)} bytes, expected > {_WAV_HEADER_SIZE})"
        )

    # Strip 44-byte WAV header → raw PCM int16 → normalise to float32
    pcm_int16 = np.frombuffer(raw_bytes[_WAV_HEADER_SIZE:], dtype=np.int16)
    audio_float32 = pcm_int16.astype(np.float32) / 32768.0

    duration_s = len(audio_float32) / sample_rate
    logger.info(
        "Extracted %.1fs audio from %s (%d samples @ %d Hz)",
        duration_s,
        video_path.name,
        len(audio_float32),
        sample_rate,
    )

    return audio_float32


def get_audio_duration_s(audio: np.ndarray, sample_rate: int = 16_000) -> float:
    """Return the duration in seconds of a PCM audio array."""
    return len(audio) / sample_rate
