"""Build canonical frame references from organizer map-keyframes CSV files."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image

from aic2026.contracts import FrameRef

from .io import sha256_file

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True, slots=True)
class FrameMapRow:
    keyframe_n: int
    pts_time_s: float
    fps: float
    frame_idx: int
    # Keyframe numbers that shared this row's frame_idx and were not indexed.
    discarded_keyframe_ns: tuple[int, ...] = ()


def _drop_duplicate_frame_idx(rows: list[FrameMapRow]) -> list[FrameMapRow]:
    """Keep one keyframe per `frame_idx` instead of inventing a new index.

    Some organizer maps compute `frame_idx` as `floor(pts_time * fps)`, so a
    keyframe one frame after its predecessor (`pts_time` 1/fps, product 0.999...)
    inherits the predecessor's index and two keyframes claim one `frame_uid`.

    Rewriting the index would put a value in our submission that the organizer's
    own map never assigned. Dropping instead keeps every surviving `frame_idx`
    exactly as shipped; the discarded keyframe is simply not indexed.

    The survivor is the middle of the colliding run — deterministic, so a rerun
    with the same config reproduces the same manifest byte for byte.
    """
    kept: list[FrameMapRow] = []
    start = 0
    while start < len(rows):
        end = start
        while end + 1 < len(rows) and rows[end + 1].frame_idx == rows[start].frame_idx:
            end += 1
        run = rows[start : end + 1]
        survivor = run[len(run) // 2]
        if len(run) > 1:
            discarded = tuple(row.keyframe_n for row in run if row is not survivor)
            survivor = replace(survivor, discarded_keyframe_ns=discarded)
        kept.append(survivor)
        start = end + 1
    return kept


def read_frame_map(path: Path, *, drop_duplicate_frame_idx: bool = True) -> list[FrameMapRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"n", "pts_time", "fps", "frame_idx"}
        if set(reader.fieldnames or ()) != required:
            raise ValueError(
                f"{path} must contain exactly {sorted(required)}; got {reader.fieldnames}"
            )
        rows: list[FrameMapRow] = []
        for line_number, raw in enumerate(reader, start=2):
            try:
                row = FrameMapRow(
                    keyframe_n=int(raw["n"]),
                    pts_time_s=float(raw["pts_time"]),
                    fps=float(raw["fps"]),
                    frame_idx=int(raw["frame_idx"]),
                )
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid numeric value at {path}:{line_number}") from error
            if row.keyframe_n < 1 or row.frame_idx < 0 or row.pts_time_s < 0 or row.fps <= 0:
                raise ValueError(f"Out-of-range mapping value at {path}:{line_number}")
            if not all(math.isfinite(value) for value in (row.pts_time_s, row.fps)):
                raise ValueError(f"Non-finite mapping value at {path}:{line_number}")
            rows.append(row)

    if not rows:
        raise ValueError(f"Frame map is empty: {path}")
    if drop_duplicate_frame_idx:
        rows = _drop_duplicate_frame_idx(rows)
    for name, values in (
        ("n", [row.keyframe_n for row in rows]),
        ("frame_idx", [row.frame_idx for row in rows]),
        ("pts_time", [row.pts_time_s for row in rows]),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {name} in {path}")
        if any(current <= previous for previous, current in zip(values, values[1:], strict=False)):
            raise ValueError(f"{name} must be strictly increasing in {path}")
    return rows


def _index_frames(frames_dir: Path) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    for path in frames_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        try:
            keyframe_n = int(path.stem)
        except ValueError as error:
            raise ValueError(f"Frame filename stem must be numeric: {path.name}") from error
        if keyframe_n in indexed:
            raise ValueError(f"Multiple frame files resolve to keyframe n={keyframe_n}")
        indexed[keyframe_n] = path
    return indexed


def build_frame_refs(
    *,
    video_id: str,
    map_csv: Path,
    frames_dir: Path,
    data_root: Path,
    limit: int | None = None,
    drop_duplicate_frame_idx: bool = True,
) -> list[FrameRef]:
    rows = read_frame_map(map_csv, drop_duplicate_frame_idx=drop_duplicate_frame_idx)
    indexed = _index_frames(frames_dir)
    # Deliberately discarded keyframes still have an image on disk, so they must
    # count as mapped or the surplus check below rejects the whole video.
    expected_keyframes = {row.keyframe_n for row in rows}
    expected_keyframes.update(n for row in rows for n in row.discarded_keyframe_ns)
    if limit is None:
        surplus = sorted(set(indexed).difference(expected_keyframes))
        if surplus:
            raise ValueError(
                f"Frame directory contains n values absent from mapping: {surplus[:5]}"
            )
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = rows[:limit]

    refs: list[FrameRef] = []
    root = data_root.resolve()
    for row in rows:
        frame_path = indexed.get(row.keyframe_n)
        if frame_path is None:
            raise ValueError(f"Missing image for keyframe n={row.keyframe_n} in {frames_dir}")
        resolved = frame_path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"Frame is outside data root: {resolved}") from error
        with Image.open(resolved) as image:
            width, height = image.size
            image.verify()
        refs.append(
            FrameRef(
                video_id=video_id,
                frame_uid=f"{video_id}:{row.frame_idx}",
                keyframe_n=row.keyframe_n,
                frame_idx=row.frame_idx,
                pts_time_s=row.pts_time_s,
                fps=row.fps,
                frame_relpath=relative,
                source_image_sha256=sha256_file(resolved),
                width=width,
                height=height,
            )
        )
    return refs
