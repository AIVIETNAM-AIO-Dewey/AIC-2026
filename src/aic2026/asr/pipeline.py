"""End-to-end ASR pipeline: audio extraction → sliding-window decoding →
deduplication → keyframe pre-indexing → atomic JSONL write.

Processes one video at a time for resumability on Kaggle.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from aic2026.contracts.asr import AsrKeyframeRef, AsrSegmentRecord, AsrVideoManifest

from .audio import AudioExtractionError, extract_audio_pcm, get_audio_duration_s
from .backend import AsrBackend, AsrSegmentRaw
from .normalizer import normalize_transcript

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Sliding-window segmentation
# ──────────────────────────────────────────────────────────────────────


def _generate_windows(
    total_duration_s: float,
    window_size_s: float,
    stride_s: float,
) -> list[tuple[float, float]]:
    """Return a list of ``(start_s, end_s)`` windows covering the audio."""
    if total_duration_s <= 0:
        return []

    # Short audio: single window
    if total_duration_s <= window_size_s:
        return [(0.0, total_duration_s)]

    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < total_duration_s:
        end = min(start + window_size_s, total_duration_s)
        # Skip windows shorter than 1 second (likely just tail silence)
        if end - start >= 1.0:
            windows.append((start, end))
        start += stride_s

    return windows


# ──────────────────────────────────────────────────────────────────────
# Deduplication of overlapping window segments
# ──────────────────────────────────────────────────────────────────────


def _text_similarity(a: str, b: str) -> float:
    """Normalised similarity ratio between two strings (0..1)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _time_overlap_ratio(
    s1_start: float, s1_end: float,
    s2_start: float, s2_end: float,
) -> float:
    """Fraction of the shorter segment's duration that overlaps."""
    overlap_start = max(s1_start, s2_start)
    overlap_end = min(s1_end, s2_end)
    overlap_dur = max(0.0, overlap_end - overlap_start)
    shorter_dur = min(s1_end - s1_start, s2_end - s2_start)
    if shorter_dur <= 0:
        return 0.0
    return overlap_dur / shorter_dur


def _center_score(seg_start: float, seg_end: float, win_start: float, win_end: float) -> float:
    """How centred is the segment within its decoding window?  Higher = better."""
    seg_mid = (seg_start + seg_end) / 2.0
    win_mid = (win_start + win_end) / 2.0
    win_half = (win_end - win_start) / 2.0
    if win_half <= 0:
        return 0.0
    return 1.0 - abs(seg_mid - win_mid) / win_half


def _deduplicate_segments(
    segments: list[dict],
    time_overlap_threshold: float = 0.80,
    text_similarity_threshold: float = 0.85,
    merge_gap_ms: int = 500,
) -> list[dict]:
    """Remove near-duplicate segments from overlapping windows.

    Each dict in *segments* must have keys: ``start_ms``, ``end_ms``,
    ``text``, ``window_start_s``, ``window_end_s``.

    Algorithm:
    1. Sort by ``start_ms``.
    2. Greedily merge: if two adjacent segments overlap ≥ *time_overlap_threshold*
       AND text similarity ≥ *text_similarity_threshold*, keep the one that
       is more centred in its decoding window.
    3. Merge adjacent segments with gap ≤ *merge_gap_ms* into one contiguous
       segment (concatenate text).
    """
    if not segments:
        return []

    # Sort by start time
    segments = sorted(segments, key=lambda s: s["start_ms"])

    # --- Pass 1: Drop near-duplicate overlaps ---
    kept: list[dict] = [segments[0]]
    for seg in segments[1:]:
        prev = kept[-1]
        overlap = _time_overlap_ratio(
            prev["start_ms"], prev["end_ms"],
            seg["start_ms"], seg["end_ms"],
        )
        if overlap >= time_overlap_threshold:
            sim = _text_similarity(prev["text"], seg["text"])
            if sim >= text_similarity_threshold:
                # Keep the one more centred in its window
                prev_center = _center_score(
                    prev["start_ms"], prev["end_ms"],
                    prev["window_start_s"] * 1000, prev["window_end_s"] * 1000,
                )
                seg_center = _center_score(
                    seg["start_ms"], seg["end_ms"],
                    seg["window_start_s"] * 1000, seg["window_end_s"] * 1000,
                )
                if seg_center > prev_center:
                    kept[-1] = seg
                # else: keep prev (already in kept)
                continue
        kept.append(seg)

    # --- Pass 2: Merge adjacent segments with small gaps ---
    merged: list[dict] = [kept[0]]
    for seg in kept[1:]:
        prev = merged[-1]
        gap = seg["start_ms"] - prev["end_ms"]
        if gap <= merge_gap_ms:
            # Extend previous segment
            prev["end_ms"] = max(prev["end_ms"], seg["end_ms"])
            prev["text"] = prev["text"].rstrip() + " " + seg["text"].lstrip()
        else:
            merged.append(seg)

    return merged


# ──────────────────────────────────────────────────────────────────────
# Keyframe pre-indexing
# ──────────────────────────────────────────────────────────────────────


def _load_keyframes(csv_path: str | Path | None) -> pd.DataFrame | None:
    """Load a map-keyframes CSV and return a sorted DataFrame, or None if not provided/found."""
    if csv_path is None:
        return None
    p = Path(csv_path)
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    # Normalise column names (strip whitespace, handle BOM)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    required = {"n", "pts_time", "fps", "frame_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Keyframe CSV missing columns: {missing}")
    return df.sort_values("pts_time").reset_index(drop=True)


def _index_keyframes(
    video_id: str,
    keyframe_df: pd.DataFrame | None,
    start_ms: int,
    end_ms: int,
) -> list[AsrKeyframeRef]:
    """Find all keyframes within ``[start_ms, end_ms]``."""
    if keyframe_df is None:
        return []
    refs: list[AsrKeyframeRef] = []
    seen_uids: set[str] = set()
    for row in keyframe_df.itertuples(index=False):
        kf_ms = row.pts_time * 1000.0
        if start_ms <= kf_ms <= end_ms:
            uid = f"{video_id}:{int(row.frame_idx)}"
            if uid not in seen_uids:
                seen_uids.add(uid)
                refs.append(AsrKeyframeRef(
                    keyframe_n=int(row.n),
                    frame_idx=int(row.frame_idx),
                    pts_time_s=float(row.pts_time),
                    frame_uid=uid,
                ))
    return refs


# ──────────────────────────────────────────────────────────────────────
# Atomic JSONL I/O
# ──────────────────────────────────────────────────────────────────────


def _write_jsonl_atomic(
    records: list[AsrSegmentRecord],
    output_path: Path,
) -> None:
    """Write records to a JSONL file atomically (write-to-tmp → rename)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")
    tmp_path.rename(output_path)
    logger.info("Wrote %d segments to %s", len(records), output_path.name)


def _write_manifest_atomic(
    manifest: AsrVideoManifest,
    output_path: Path,
) -> None:
    """Write manifest to a JSON file atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))
    tmp_path.rename(output_path)
    logger.info("Wrote manifest to %s", output_path.name)


# ──────────────────────────────────────────────────────────────────────
# Main pipeline entry point
# ──────────────────────────────────────────────────────────────────────


def process_video(
    *,
    video_id: str,
    video_path: str | Path,
    keyframe_csv_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    output_jsonl: str | Path | None = None,
    backend: AsrBackend,
    window_size_s: float = 15.0,
    stride_s: float = 7.5,
    sample_rate: int = 16_000,
    language: str = "vi",
    initial_prompt: str | None = None,
    vad_filter: bool = True,
    vad_min_silence_duration_ms: int = 500,
    dedup_time_overlap_threshold: float = 0.80,
    dedup_text_similarity_threshold: float = 0.85,
    merge_gap_ms: int = 500,
) -> AsrVideoManifest:
    """Run the full ASR pipeline for a single video.

    Parameters
    ----------
    video_id:
        Canonical video identifier (e.g. ``"L21_V001"``).
    video_path:
        Path to the ``.mp4`` video file.
    keyframe_csv_path:
        Path to ``map-keyframes/<VIDEO_ID>.csv``.
    output_dir:
        Directory to write ``<video_id>.jsonl`` and ``<video_id>.manifest.json``.
    backend:
        A loaded ``AsrBackend`` instance.
    window_size_s:
        Decoding window duration in seconds.
    stride_s:
        Window stride in seconds.
    sample_rate:
        Audio sample rate (must be 16000 for PhoWhisper).
    language:
        Target language code for the decoder.
    initial_prompt:
        Optional initial prompt to bias PhoWhisper output.
    vad_filter:
        Whether to enable VAD silence filtering.
    vad_min_silence_duration_ms:
        Minimum silence duration for VAD filtering.
    dedup_time_overlap_threshold:
        Time overlap threshold for deduplication (0..1).
    dedup_text_similarity_threshold:
        Text similarity threshold for deduplication (0..1).
    merge_gap_ms:
        Maximum gap in ms to merge adjacent segments.

    Returns
    -------
    AsrVideoManifest
        Pipeline run metadata for this video.
    """
    if output_jsonl is not None:
        jsonl_path = Path(output_jsonl).expanduser().resolve()
        manifest_path = jsonl_path.with_suffix(".manifest.json")
    elif output_dir is not None:
        output_dir = Path(output_dir).expanduser().resolve()
        jsonl_path = output_dir / f"{video_id}.jsonl"
        manifest_path = output_dir / f"{video_id}.manifest.json"
    else:
        raise ValueError("Either output_dir or output_jsonl must be provided")

    started_at = datetime.now(timezone.utc)

    # ── 1. Extract audio ──
    try:
        audio = extract_audio_pcm(video_path, sample_rate=sample_rate)
    except (AudioExtractionError, FileNotFoundError) as exc:
        logger.warning("Skipping %s: %s", video_id, exc)
        manifest = AsrVideoManifest(
            video_id=video_id,
            status="skipped",
            segment_count=0,
            keyframe_count=0,
            audio_duration_s=0.0,
            model_id=backend.model_identifier,
            engine=type(backend).__name__,
            started_at=started_at,
            ended_at=datetime.now(timezone.utc),
        )
        _write_jsonl_atomic([], jsonl_path)
        _write_manifest_atomic(manifest, manifest_path)
        return manifest

    audio_duration_s = get_audio_duration_s(audio, sample_rate)

    # ── 2. Generate sliding windows ──
    windows = _generate_windows(audio_duration_s, window_size_s, stride_s)
    logger.info(
        "%s: %.1fs audio → %d windows (%.0fs × %.0fs stride)",
        video_id, audio_duration_s, len(windows), window_size_s, stride_s,
    )

    # ── 3. Decode each window ──
    raw_segments: list[dict] = []
    for win_start, win_end in windows:
        # Slice the audio array for this window
        start_sample = int(win_start * sample_rate)
        end_sample = int(win_end * sample_rate)
        window_audio = audio[start_sample:end_sample]

        decoded = backend.decode(
            window_audio,
            language=language,
            initial_prompt=initial_prompt,
            vad_filter=vad_filter,
            vad_min_silence_duration_ms=vad_min_silence_duration_ms,
        )

        for seg in decoded:
            # Convert relative window timestamps to absolute video timestamps
            abs_start_ms = int((win_start + seg.start_s) * 1000)
            abs_end_ms = int((win_start + seg.end_s) * 1000)
            raw_segments.append({
                "start_ms": abs_start_ms,
                "end_ms": abs_end_ms,
                "text": seg.text,
                "window_start_s": win_start,
                "window_end_s": win_end,
            })

    logger.info(
        "%s: %d raw segments from %d windows",
        video_id, len(raw_segments), len(windows),
    )

    # ── 4. Deduplicate overlapping windows ──
    deduped = _deduplicate_segments(
        raw_segments,
        time_overlap_threshold=dedup_time_overlap_threshold,
        text_similarity_threshold=dedup_text_similarity_threshold,
        merge_gap_ms=merge_gap_ms,
    )
    logger.info(
        "%s: %d segments after deduplication (from %d raw)",
        video_id, len(deduped), len(raw_segments),
    )

    # ── 5. Load keyframes and pre-index ──
    keyframe_df = _load_keyframes(keyframe_csv_path)
    total_keyframes_indexed = 0

    records: list[AsrSegmentRecord] = []
    for i, seg in enumerate(deduped):
        transcript_raw = seg["text"].strip()
        transcript_norm = normalize_transcript(transcript_raw)
        if not transcript_raw or not transcript_norm:
            logger.debug("%s: Skipping empty or punctuation-only segment %d", video_id, i)
            continue
        keyframes = _index_keyframes(
            video_id, keyframe_df, seg["start_ms"], seg["end_ms"],
        )
        total_keyframes_indexed += len(keyframes)

        record = AsrSegmentRecord(
            segment_id=f"{video_id}:seg_{i}",
            video_id=video_id,
            start_ms=seg["start_ms"],
            end_ms=seg["end_ms"],
            transcript_raw=transcript_raw,
            transcript_normalized=transcript_norm,
            language=language,
            keyframes=keyframes,
        )
        records.append(record)

    # ── 6. Atomic write ──
    _write_jsonl_atomic(records, jsonl_path)

    manifest = AsrVideoManifest(
        video_id=video_id,
        status="completed",
        segment_count=len(records),
        keyframe_count=total_keyframes_indexed,
        audio_duration_s=round(audio_duration_s, 3),
        model_id=backend.model_identifier,
        engine=type(backend).__name__,
        started_at=started_at,
        ended_at=datetime.now(timezone.utc),
    )
    _write_manifest_atomic(manifest, manifest_path)

    logger.info(
        "%s: completed — %d segments, %d keyframes indexed",
        video_id, len(records), total_keyframes_indexed,
    )

    return manifest
