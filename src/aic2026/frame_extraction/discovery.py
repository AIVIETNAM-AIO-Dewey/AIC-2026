"""Resolve raw videos and metadata from Kaggle-style attached datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


@dataclass(frozen=True, slots=True)
class LocatedInputs:
    video_path: Path
    map_csv: Path | None
    media_info: Path | None


def _resolve_existing(path: Path, *, expected_file: bool = True) -> Path:
    resolved = path.expanduser().resolve()
    if expected_file and not resolved.is_file():
        raise FileNotFoundError(resolved)
    if not expected_file and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def _video_score(path: Path, video_id: str) -> tuple[int, int, str]:
    lowered_id = video_id.lower()
    lowered_stem = path.stem.lower()
    exact = 0 if lowered_stem == lowered_id else 1
    return (exact, len(path.parts), path.as_posix().lower())


def find_video_file(
    search_root: Path,
    video_id: str,
    *,
    suffixes: set[str] | None = None,
) -> Path | None:
    root = _resolve_existing(search_root, expected_file=False)
    allowed = {suffix.lower() for suffix in (suffixes or VIDEO_SUFFIXES)}
    lowered_id = video_id.lower()
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in allowed
        and (path.stem.lower() == lowered_id or lowered_id in path.name.lower())
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: _video_score(item, video_id))[0].resolve()


def list_video_candidates(search_root: Path, *, limit: int = 20) -> list[Path]:
    root = _resolve_existing(search_root, expected_file=False)
    candidates = [
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    ]
    return sorted(candidates, key=lambda item: item.as_posix().lower())[:limit]


def find_support_file(
    search_root: Path,
    video_id: str,
    *,
    suffix: str,
    preferred_parent: str,
) -> Path | None:
    root = _resolve_existing(search_root, expected_file=False)
    lowered_parent = preferred_parent.lower()
    lowered_suffix = suffix.lower()
    lowered_id = video_id.lower()
    candidates = [
        path
        for path in root.rglob(f"*{suffix}")
        if path.is_file()
        and path.suffix.lower() == lowered_suffix
        and path.stem.lower() == lowered_id
    ]
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, int, str]:
        parts = {part.lower() for part in path.parts}
        parent_score = 0 if lowered_parent in parts else 1
        return (parent_score, len(path.parts), path.as_posix().lower())

    return sorted(candidates, key=score)[0].resolve()


def locate_inputs(
    *,
    video_id: str,
    search_root: Path | None = None,
    video_path: Path | None = None,
    map_csv: Path | None = None,
    media_info: Path | None = None,
) -> LocatedInputs:
    if video_path is None:
        if search_root is None:
            raise ValueError("--search-root is required when --video-path is not provided")
        resolved_video = find_video_file(search_root, video_id)
        if resolved_video is None:
            candidates = [path.as_posix() for path in list_video_candidates(search_root)]
            message = f"No raw video found for {video_id!r} under {search_root}"
            if candidates:
                message += f"; first video candidates: {candidates}"
            raise FileNotFoundError(message)
    else:
        resolved_video = _resolve_existing(video_path)

    if map_csv is not None:
        resolved_map = _resolve_existing(map_csv)
    elif search_root is not None:
        resolved_map = find_support_file(
            search_root,
            video_id,
            suffix=".csv",
            preferred_parent="map-keyframes",
        )
    else:
        resolved_map = None

    if media_info is not None:
        resolved_media = _resolve_existing(media_info)
    elif search_root is not None:
        resolved_media = find_support_file(
            search_root,
            video_id,
            suffix=".json",
            preferred_parent="media-info",
        )
    else:
        resolved_media = None

    return LocatedInputs(
        video_path=resolved_video,
        map_csv=resolved_map,
        media_info=resolved_media,
    )
