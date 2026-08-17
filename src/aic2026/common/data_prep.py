"""Validated, selective extraction of organizer archives into one subset."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .io import atomic_write_json

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9]+_V[0-9]+$")


@dataclass(frozen=True, slots=True)
class ArchiveSpec:
    path: Path
    role: str
    expected_sha256: str | None
    ignored_by_policy: bool = False


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or normalized.startswith("/") or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe ZIP member path: {name!r}")
    if path.parts and re.match(r"^[A-Za-z]:$", path.parts[0]):
        raise ValueError(f"Unsafe ZIP member path: {name!r}")
    return path


def _role_from_filename(name: str) -> tuple[str, bool] | None:
    lower = name.lower()
    if lower.startswith("clip-features") and lower.endswith(".zip"):
        return "clip_legacy", True
    if lower.startswith("keyframes_") and lower.endswith(".zip"):
        return "keyframes", False
    if lower.startswith("videos_") and lower.endswith(".zip"):
        return "videos", False
    if lower.startswith("map-keyframes") and lower.endswith(".zip"):
        return "mappings", False
    if lower.startswith("objects") and lower.endswith(".zip"):
        return "objects", False
    if lower.startswith("media-info") and lower.endswith(".zip"):
        return "media_info", False
    return None


def discover_archives(raw_root: Path, configured: dict[str, Any]) -> list[ArchiveSpec]:
    specs: list[ArchiveSpec] = []
    known = configured.get("archives", {})
    for path in sorted(raw_root.glob("*.zip"), key=lambda item: item.name.lower()):
        detected = _role_from_filename(path.name)
        if detected is None:
            continue
        role, ignored = detected
        config = known.get(path.name, {})
        specs.append(
            ArchiveSpec(
                path=path,
                role=str(config.get("role", role)),
                expected_sha256=config.get("sha256"),
                ignored_by_policy=bool(config.get("ignored_by_policy", ignored)),
            )
        )
    return specs


def _entry_video_id(role: str, member: PurePosixPath) -> str | None:
    if role in {"videos", "mappings", "media_info"}:
        value = member.stem
    elif role in {"keyframes", "objects"} and len(member.parts) >= 3:
        value = member.parts[-2]
    else:
        return None
    return value if VIDEO_ID_RE.fullmatch(value) else None


def _output_relative(role: str, member: PurePosixPath) -> Path:
    if role == "videos":
        return Path("videos", member.name)
    if role == "keyframes":
        return Path("keyframes", member.parts[-2], member.name)
    if role == "mappings":
        return Path("map-keyframes", member.name)
    if role == "objects":
        return Path("objects", member.parts[-2], member.name)
    if role == "media_info":
        return Path("media-info", member.name)
    raise ValueError(f"Role cannot be extracted: {role}")


def _mapping_stats(archive: zipfile.ZipFile, name: str) -> tuple[int, list[dict[str, Any]]]:
    with archive.open(name) as binary:
        rows = list(csv.DictReader(line.decode("utf-8-sig") for line in binary))
    duplicates = [
        {
            "frame_idx": int(left["frame_idx"]),
            "discarded_keyframe_n": int(left["n"]),
            "kept_keyframe_n": int(right["n"]),
            "discarded_pts_time_s": float(left["pts_time"]),
            "kept_pts_time_s": float(right["pts_time"]),
        }
        for left, right in zip(rows, rows[1:], strict=False)
        if int(left["frame_idx"]) == int(right["frame_idx"])
    ]
    return len(rows), duplicates


def prepare_subset(
    *,
    specs: list[ArchiveSpec],
    prepared_root: Path,
    subset: str,
    resume: bool,
    expected_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    required_roles = {"videos", "keyframes", "mappings", "objects"}
    available_roles = {spec.role for spec in specs if not spec.ignored_by_policy}
    missing_roles = required_roles - available_roles
    if missing_roles:
        raise ValueError(f"Missing required source archive roles: {sorted(missing_roles)}")

    archive_reports: list[dict[str, Any]] = []
    coverage: dict[str, set[str]] = defaultdict(set)
    entries_by_spec: dict[Path, list[tuple[str, str, int]]] = defaultdict(list)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    mapping_rows: dict[str, int] = {}
    duplicate_rows: dict[str, list[dict[str, Any]]] = {}
    expected_extract_bytes = 0

    for spec in specs:
        actual_hash = sha256_file(spec.path)
        if spec.expected_sha256 and actual_hash != spec.expected_sha256.lower():
            raise ValueError(f"SHA-256 mismatch for {spec.path.name}")
        report = {
            "filename": spec.path.name,
            "role": spec.role,
            "sha256": actual_hash,
            "size_bytes": spec.path.stat().st_size,
            "ignored_by_policy": spec.ignored_by_policy,
        }
        archive_reports.append(report)
        if spec.ignored_by_policy:
            continue
        with zipfile.ZipFile(spec.path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                member = safe_member_path(info.filename)
                video_id = _entry_video_id(spec.role, member)
                if video_id is None or not video_id.startswith(f"{subset}_"):
                    continue
                coverage[spec.role].add(video_id)
                counts[spec.role][video_id] += 1
                entries_by_spec[spec.path].append((info.filename, video_id, info.file_size))
                if spec.role == "mappings":
                    row_count, duplicates = _mapping_stats(archive, info.filename)
                    mapping_rows[video_id] = row_count
                    duplicate_rows[video_id] = duplicates

    complete_ids = set.intersection(*(coverage[role] for role in required_roles))
    if not complete_ids:
        raise ValueError(f"No complete video IDs found for subset {subset!r}")
    for video_id in sorted(complete_ids):
        expected = mapping_rows[video_id]
        if counts["keyframes"][video_id] != expected or counts["objects"][video_id] != expected:
            raise ValueError(
                f"Coverage mismatch for {video_id}: map={expected}, "
                f"keyframes={counts['keyframes'][video_id]}, objects={counts['objects'][video_id]}"
            )

    final_root = (prepared_root / subset).resolve()
    prepared_root = prepared_root.resolve()
    if final_root.parent != prepared_root:
        raise ValueError("Prepared subset target escapes prepared_root")
    inventory_path = final_root / "inventory.json"
    source_fingerprint = {item["filename"]: item["sha256"] for item in archive_reports}
    if final_root.exists():
        if not resume:
            raise FileExistsError(f"Prepared subset exists; use --resume: {final_root}")
        if not inventory_path.is_file():
            raise ValueError(f"Prepared subset has no inventory: {final_root}")
        existing = json.loads(inventory_path.read_text(encoding="utf-8"))
        if existing.get("source_fingerprint") != source_fingerprint:
            raise ValueError("Prepared subset sources changed; refuse unsafe resume")
        existing["duplicate_frame_idx"] = {
            video: duplicate_rows[video] for video in sorted(complete_ids) if duplicate_rows[video]
        }
        _validate_expected_counts(existing.get("counts", {}), expected_counts)
        atomic_write_json(inventory_path, existing)
        return existing

    partial_root = prepared_root / f".{subset}.partial"
    if partial_root.exists():
        if not resume:
            raise FileExistsError(f"Partial preparation exists; use --resume: {partial_root}")
        shutil.rmtree(partial_root)
    partial_root.mkdir(parents=True, exist_ok=False)

    for spec in specs:
        if spec.ignored_by_policy:
            continue
        selected = [item for item in entries_by_spec[spec.path] if item[1] in complete_ids]
        expected_extract_bytes += sum(item[2] for item in selected)
    free_bytes = shutil.disk_usage(prepared_root).free
    if free_bytes < int(expected_extract_bytes * 1.2):
        shutil.rmtree(partial_root)
        raise OSError(
            f"Insufficient free disk: need 120% of {expected_extract_bytes} bytes, "
            f"have {free_bytes}"
        )

    extracted: set[Path] = set()
    try:
        for spec in specs:
            if spec.ignored_by_policy:
                continue
            selected = [item for item in entries_by_spec[spec.path] if item[1] in complete_ids]
            with zipfile.ZipFile(spec.path) as archive:
                for name, _, _ in selected:
                    relative = _output_relative(spec.role, safe_member_path(name))
                    if relative in extracted:
                        raise ValueError(f"Duplicate prepared output: {relative.as_posix()}")
                    extracted.add(relative)
                    destination = partial_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(destination.suffix + ".tmp")
                    with archive.open(name) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                        target.flush()
                        os.fsync(target.fileno())
                    temporary.replace(destination)

        inventory = {
            "schema_version": "aic26.data_inventory.v1",
            "subset": subset,
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_fingerprint": source_fingerprint,
            "archives": archive_reports,
            "video_ids": sorted(complete_ids),
            "counts": {
                "videos": len(complete_ids),
                "keyframes_raw": sum(counts["keyframes"][video] for video in complete_ids),
                "keyframes_canonical": sum(
                    mapping_rows[video] - len(duplicate_rows[video]) for video in complete_ids
                ),
                "objects": sum(counts["objects"][video] for video in complete_ids),
                "duplicate_frame_idx_rows": sum(
                    len(duplicate_rows[video]) for video in complete_ids
                ),
            },
            "duplicate_frame_idx": {
                video: duplicate_rows[video]
                for video in sorted(complete_ids)
                if duplicate_rows[video]
            },
        }
        _validate_expected_counts(inventory["counts"], expected_counts)
        atomic_write_json(partial_root / "inventory.json", inventory)
        partial_root.replace(final_root)
        return inventory
    except BaseException:
        if partial_root.exists():
            shutil.rmtree(partial_root)
        raise


def _validate_expected_counts(actual: dict[str, int], expected: dict[str, int] | None) -> None:
    for key, value in (expected or {}).items():
        if int(actual.get(key, -1)) != int(value):
            raise ValueError(
                f"Prepared subset regression for {key}: expected {value}, got {actual.get(key)}"
            )


def require_prepared_video(data_root: Path, video_id: str) -> None:
    """Enforce inventory coverage when data_root is a prepared subset."""
    inventory_path = data_root / "inventory.json"
    if not inventory_path.exists():
        return
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("status") != "completed":
        raise ValueError(f"Prepared inventory is not completed: {inventory_path}")
    if video_id not in inventory.get("video_ids", []):
        raise ValueError(f"{video_id} is not covered by {inventory_path}")
