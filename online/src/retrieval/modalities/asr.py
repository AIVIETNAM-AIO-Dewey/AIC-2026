"""Canonical CPU ASR retrieval and persistent FTS5 index.

The source ASR segments are Vietnamese transcripts.  Preparation resolves
each segment to one canonical keyframe (the keyframe nearest the segment
midpoint) and stores that identity in SQLite, so runtime search never has to
scan JSONL or infer frame metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import unicodedata
from collections.abc import Iterable
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..infrastructure.qdrant import base_frame
from .lexical import (
    _folded_tokens,
    _ordered_lexical_bigrams,
    _sigmoid_zscore,
    _stream_identity,
    _token_bigrams,
    fold_text,
    normalize_text,
    ordered_lexical_tokens,
    query_tokens,
)

EXPECTED_FRAMES = 247_956
EXPECTED_ASR_SEGMENTS = 55_168
EXPECTED_ASR_VIDEOS = 873
QUERY_ROLES = ("original", "entity", "action", "context", "synonym", "keyword")
ASR_INDEX_SCHEMA_VERSION = "branch3.asr-index.v2"
ASR_SQLITE_USER_VERSION = 4
ASR_SOURCE_SCHEMA_VERSION = "aic26.asr_segments.v1"
ASR_MAPPING_STRATEGY = "nearest_keyframe_to_segment_midpoint"
ASR_HEALTH_CACHE_TTL_S = 30.0


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _canonical_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def source_fingerprint(source_files: Iterable[dict[str, Any]]) -> str:
    """Fingerprint the immutable source-file records used to build SQLite."""

    records: list[dict[str, Any]] = []
    for record in source_files:
        if not isinstance(record, dict):
            raise ValueError("ASR source fingerprint records must be objects")
        records.append(
            {
                "path": str(record.get("path") or ""),
                "size": int(record.get("size", -1)),
                "sha256": str(record.get("sha256") or ""),
            }
        )
    records.sort(key=lambda record: record["path"])
    return _canonical_fingerprint(records)


def build_id_for(
    *,
    source_fingerprint_value: str,
    canonical_fingerprint_value: str,
    segment_count: int,
    video_count: int,
    mapping_strategy: str = ASR_MAPPING_STRATEGY,
) -> str:
    """Create a deterministic identity for one prepared ASR index."""

    return _canonical_fingerprint(
        {
            "schema_version": ASR_INDEX_SCHEMA_VERSION,
            "sqlite_user_version": ASR_SQLITE_USER_VERSION,
            "source_fingerprint": str(source_fingerprint_value),
            "canonical_fingerprint": str(canonical_fingerprint_value),
            "mapping_strategy": str(mapping_strategy),
            "segment_count": int(segment_count),
            "video_count": int(video_count),
        }
    )


def _fts_expression(tokens: Iterable[str]) -> str:
    values = []
    for token in tokens:
        clean = str(token).replace('"', " ").strip()
        if clean:
            values.append(f'"{clean}"')
    return " OR ".join(values)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(
    path: Path, *, relative_to: Path | None = None, include_hash: bool = True
) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "path": path.relative_to(relative_to).as_posix() if relative_to is not None else str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_hash:
        record["sha256"] = _sha256_file(path)
    return record


def load_canonical_frame_index(data_root: Path) -> dict[str, dict[str, Any]]:
    """Load the canonical frame identity map for preparation only."""

    path = data_root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Canonical frame metadata is missing: {path}")
    result: dict[str, dict[str, Any]] = {}
    expected_point_id = 1
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            frame_uid = str(item.get("frame_uid") or "")
            point_id = int(item.get("point_id", 0))
            video_id = str(item.get("video_id") or "").upper().replace("-", "_")
            frame_idx = int(item.get("frame_idx", -1))
            if not frame_uid or point_id < 1 or not video_id or frame_idx < 0:
                raise ValueError(f"Invalid canonical frame row {line_number}")
            if frame_uid != f"{video_id}:{frame_idx}":
                raise ValueError(f"Canonical frame_uid mismatch at row {line_number}")
            if point_id != expected_point_id:
                raise ValueError(
                    f"Canonical point order mismatch at row {line_number}: "
                    f"expected {expected_point_id}, got {point_id}"
                )
            if frame_uid in result:
                raise ValueError(f"Duplicate canonical frame_uid: {frame_uid}")
            image_relpath = str(item.get("image_relpath") or item.get("frame_relpath") or "")
            pts_time_s = float(item.get("pts_time_s", 0.0))
            fps = float(item.get("fps", 0.0))
            keyframe_n = int(item.get("keyframe_n", 1))
            if (
                not image_relpath
                or keyframe_n < 1
                or not math.isfinite(pts_time_s)
                or pts_time_s < 0
                or not math.isfinite(fps)
                or fps <= 0
            ):
                raise ValueError(f"Invalid canonical frame timing/path at row {line_number}")
            result[frame_uid] = {
                "point_id": point_id,
                "frame_uid": frame_uid,
                "video_id": video_id,
                "frame_idx": frame_idx,
                "keyframe_n": keyframe_n,
                "pts_time_s": pts_time_s,
                "fps": fps,
                "image_relpath": image_relpath,
            }
            expected_point_id += 1
    if len(result) != EXPECTED_FRAMES:
        raise ValueError(f"Expected {EXPECTED_FRAMES} canonical frames, found {len(result)}")
    return result


def _parse_source_segment(item: dict[str, Any], path: Path, line_number: int) -> dict[str, Any]:
    if item.get("schema_version") != ASR_SOURCE_SCHEMA_VERSION:
        raise ValueError(f"{path}:{line_number}: unsupported ASR schema")
    language = str(item.get("language") or "").strip().casefold()
    if language and language != "vi":
        raise ValueError(f"{path}:{line_number}: ASR language must be vi")
    segment_id = str(item.get("segment_id") or "").strip()
    video_id = str(item.get("video_id") or "").upper().replace("-", "_").strip()
    if not segment_id or not video_id:
        raise ValueError(f"{path}:{line_number}: segment_id/video_id is required")
    try:
        start_ms = int(item["start_ms"])
        end_ms = int(item["end_ms"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}:{line_number}: invalid segment timestamps") from error
    if start_ms < 0 or end_ms < start_ms:
        raise ValueError(f"{path}:{line_number}: invalid segment interval")
    transcript_display = unicodedata.normalize(
        "NFC", str(item.get("transcript_normalized") or item.get("transcript_raw") or "")
    ).strip()
    if not transcript_display:
        raise ValueError(f"{path}:{line_number}: transcript is empty")
    return {
        "segment_id": segment_id,
        "video_id": video_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "transcript": transcript_display,
        "transcript_search": fold_text(transcript_display),
    }


def validate_asr_sources(segments_dir: Path) -> dict[str, Any]:
    """Validate source rows/manifests and return immutable preparation facts."""

    if not segments_dir.is_dir():
        raise FileNotFoundError(f"ASR segment directory is missing: {segments_dir}")
    paths = sorted(segments_dir.glob("*.jsonl"))
    if not paths:
        raise ValueError(f"No ASR JSONL files found in {segments_dir}")
    if len(paths) != EXPECTED_ASR_VIDEOS:
        raise ValueError(f"Expected {EXPECTED_ASR_VIDEOS} ASR segment files, found {len(paths)}")
    source_files: list[dict[str, Any]] = []
    segment_ids: set[str] = set()
    source_video_ids: set[str] = set()
    indexed_video_ids: set[str] = set()
    empty_video_ids: list[str] = []
    total = 0
    source_segment_file_count = 0
    source_manifest_file_count = 0
    models: set[str] = set()
    engines: set[str] = set()
    for path in paths:
        manifest_path = path.with_suffix(".manifest.json")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            raise ValueError(f"Missing or invalid ASR manifest: {manifest_path}") from error
        if (
            manifest.get("schema_version") != "aic26.asr_manifest.v1"
            or manifest.get("status") != "completed"
        ):
            raise ValueError(f"ASR manifest is not completed: {manifest_path}")
        manifest_video_id = str(manifest.get("video_id") or "").upper().replace("-", "_").strip()
        expected_video_id = path.stem.upper().replace("-", "_").strip()
        if not manifest_video_id or manifest_video_id != expected_video_id:
            raise ValueError(
                f"ASR manifest video_id mismatch for {path.name}: "
                f"expected {expected_video_id}, got {manifest.get('video_id')}"
            )
        if manifest_video_id in source_video_ids:
            raise ValueError(f"Duplicate ASR source video_id: {manifest_video_id}")
        source_video_ids.add(manifest_video_id)
        file_count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                item = _parse_source_segment(json.loads(line), path, line_number)
                if item["video_id"] != manifest_video_id:
                    raise ValueError(
                        f"{path}:{line_number}: segment video_id does not match manifest video_id"
                    )
                if item["segment_id"] in segment_ids:
                    raise ValueError(f"Duplicate ASR segment_id: {item['segment_id']}")
                segment_ids.add(item["segment_id"])
                indexed_video_ids.add(item["video_id"])
                file_count += 1
        if int(manifest.get("segment_count", -1)) != file_count:
            raise ValueError(
                f"ASR manifest count mismatch for {path.name}: "
                f"manifest={manifest.get('segment_count')} rows={file_count}"
            )
        if file_count == 0:
            empty_video_ids.append(manifest_video_id)
        models.add(str(manifest.get("model_id") or ""))
        engines.add(str(manifest.get("engine") or ""))
        source_files.append(artifact_record(path, relative_to=segments_dir.parent))
        source_files.append(artifact_record(manifest_path, relative_to=segments_dir.parent))
        source_segment_file_count += 1
        source_manifest_file_count += 1
        total += file_count
    if total != EXPECTED_ASR_SEGMENTS:
        raise ValueError(f"Expected {EXPECTED_ASR_SEGMENTS} ASR segments, found {total}")
    if len(source_video_ids) != EXPECTED_ASR_VIDEOS:
        raise ValueError(
            f"Expected {EXPECTED_ASR_VIDEOS} ASR source videos, found {len(source_video_ids)}"
        )
    if not models or "" in models or not engines or "" in engines:
        raise ValueError("ASR model/engine identity is missing from a manifest")
    if len(models) != 1 or len(engines) != 1:
        raise ValueError("ASR source manifests must use one model and one engine")
    return {
        "source_files": source_files,
        "source_segment_file_count": source_segment_file_count,
        "source_manifest_file_count": source_manifest_file_count,
        "segment_count": total,
        "video_count": len(source_video_ids),
        "indexed_video_count": len(indexed_video_ids),
        "empty_video_count": len(empty_video_ids),
        "empty_video_ids": sorted(empty_video_ids),
        "video_ids": sorted(source_video_ids),
        "model_ids": sorted(models),
        "engines": sorted(engines),
        "source_fingerprint": source_fingerprint(source_files),
    }


class AsrFtsIndex:
    """Read-only runtime ASR index with six-role bilingual BM25/ngram retrieval."""

    def __init__(
        self,
        segments_dir: Path,
        database_path: Path,
        metadata: Any = None,
        *,
        manifest_path: Path | None = None,
        auto_prepare: bool = False,
        canonical_frame_index: dict[str, dict[str, Any]] | None = None,
        build_context: dict[str, Any] | None = None,
    ) -> None:
        self.segments_dir = segments_dir
        self.database_path = database_path
        self.metadata = metadata
        self.manifest_path = manifest_path or database_path.with_name("branch3_asr_manifest.json")
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        self._auto_prepare = bool(auto_prepare)
        self._connection_path: Path | None = None
        self._runtime_snapshot_path: Path | None = None
        self._opened_file_identity: tuple[int, int, int, int] | None = None
        self._opened_build_id: str | None = None
        self._connection_generation = 0
        self._connection_reopened = False
        self._connection_error: str | None = None
        self._health_cache: tuple[float, tuple[Any, ...], dict[str, Any]] | None = None
        self._fingerprint_cache: dict[str, tuple[tuple[int, int, int], str]] = {}
        self._artifact_diagnostics: dict[str, dict[str, Any]] = {}
        # (monotonic timestamp, manifest stat identity, immutable snapshot).
        # Keep the annotation aligned with the compact tuple actually stored;
        # the snapshot dictionary carries the detailed inventory diagnostics.
        self._source_inventory_cache: tuple[float, tuple[Any, ...], dict[str, Any]] | None = None
        if auto_prepare:
            database_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(database_path, check_same_thread=False)
            self._connection_path = database_path
        if self._connection is not None:
            self._connection.row_factory = sqlite3.Row
        if auto_prepare:
            self._ensure_index(canonical_frame_index or {}, build_context or {})

    def close(self) -> None:
        with self._lock:
            self._close_connection_locked()
            self._health_cache = None
            self._fingerprint_cache.clear()
            self._artifact_diagnostics.clear()
            self._source_inventory_cache = None
            self._opened_file_identity = None
            self._opened_build_id = None

    @property
    def ready(self) -> bool:
        return bool(self.health().get("ready"))

    def _runtime_ready_fast_locked(self) -> bool:
        """Validate the active DB/manifest pair without rescanning source files."""

        manifest = self._load_manifest()
        self._refresh_runtime_connection_locked(manifest)
        if self._connection is None:
            return False
        state = self._database_state()
        meta = state.get("meta") or {}
        db_record = manifest.get("database") or {}
        try:
            db_stat = self.database_path.stat()
            manifest_database_path = (
                (self.database_path.parent / str(db_record.get("path") or "")).resolve()
                if isinstance(db_record, dict)
                else None
            )
            db_stat_matches = (
                manifest_database_path == self.database_path.resolve()
                and int(db_record.get("size", -1)) == int(db_stat.st_size)
                and int(db_record.get("mtime_ns", -1)) == int(db_stat.st_mtime_ns)
            )
        except (OSError, TypeError, ValueError):
            db_stat_matches = False
        return bool(
            manifest.get("schema_version") == ASR_INDEX_SCHEMA_VERSION
            and manifest.get("status") == "ready"
            and manifest.get("passed") is True
            and manifest.get("build_id")
            and state.get("ready") is True
            and manifest.get("build_id") == meta.get("build_id")
            and manifest.get("source_fingerprint") == meta.get("source_fingerprint")
            and manifest.get("canonical_fingerprint") == meta.get("canonical_fingerprint")
            and db_stat_matches
        )

    def _health_error_payload(self, error: Exception) -> dict[str, Any]:
        """Return a fail-closed health response for unexpected validation errors."""

        return {
            "ready": False,
            "production_ready": False,
            "database": str(self.database_path),
            "manifest": str(self.manifest_path),
            "schema_version": None,
            "sqlite_user_version": None,
            "database_matches_manifest": False,
            "source_matches_manifest": False,
            "canonical_metadata_matches_manifest": False,
            "fingerprints_recorded": False,
            "internal_metadata_matches": False,
            "connection_generation": self._connection_generation,
            "connection_reopened": False,
            "opened_build_id": self._opened_build_id,
            "connection_error": str(error),
            "artifact_diagnostics": [
                {"path": "<health>", "error": str(error), "fingerprint_matches": False}
            ],
            "source_validation_age_s": None,
            "warnings": ["ASR health validation failed closed"],
        }

    def assert_ready(self) -> None:
        with self._lock:
            if not self._runtime_ready_fast_locked():
                raise RuntimeError("ASR FTS index is not prepared or its manifest is stale")

    def _table_columns(self, table: str) -> set[str]:
        if self._connection is None:
            return set()
        return {
            str(row[1])
            for row in self._connection.execute(f"PRAGMA table_info({table})").fetchall()
        }

    def _file_identity(self) -> tuple[int, int, int, int] | None:
        try:
            stat = self.database_path.stat()
        except OSError:
            return None
        return (
            int(getattr(stat, "st_dev", 0)),
            int(getattr(stat, "st_ino", 0)),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )

    def _load_manifest(self) -> dict[str, Any]:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        return manifest if isinstance(manifest, dict) else {}

    @staticmethod
    def _read_database_meta(connection: sqlite3.Connection) -> dict[str, Any]:
        rows = connection.execute(
            "SELECT schema_version, sqlite_user_version, build_id, "
            "source_fingerprint, canonical_fingerprint, mapping_strategy, "
            "segment_count, video_count, created_at FROM asr_meta LIMIT 2"
        ).fetchall()
        if len(rows) != 1:
            raise ValueError("ASR SQLite asr_meta must contain exactly one row")
        row = rows[0]
        # sqlite3.Row iterates values, unlike dict; its keys() API is required
        # here even though the generic mapping lint prefers direct iteration.
        return {str(key): row[key] for key in row.keys()}  # noqa: SIM118

    def _close_connection_locked(self) -> None:
        if self._connection is not None:
            with suppress(sqlite3.Error):
                self._connection.close()
        self._connection = None
        snapshot = self._runtime_snapshot_path
        self._runtime_snapshot_path = None
        self._connection_path = None
        if snapshot is not None:
            # A failed cleanup must not leave a stale connection state; the
            # next health call remains fail-closed and can retry.
            with suppress(OSError):
                snapshot.unlink(missing_ok=True)
        self._opened_file_identity = None
        self._opened_build_id = None

    def _refresh_runtime_connection_locked(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Open the current inode only after an atomic database/manifest swap."""

        if self._auto_prepare:
            return {
                "connection_generation": self._connection_generation,
                "connection_reopened": False,
                "opened_build_id": self._opened_build_id,
                "connection_error": self._connection_error,
            }
        expected_build_id = str(manifest.get("build_id") or "")
        file_identity = self._file_identity()
        needs_open = (
            self._connection is None
            or file_identity is None
            or file_identity != self._opened_file_identity
            or (expected_build_id and expected_build_id != self._opened_build_id)
        )
        self._connection_reopened = False
        if file_identity is None:
            self._close_connection_locked()
            self._connection_error = "ASR database file is missing"
            return {
                "connection_generation": self._connection_generation,
                "connection_reopened": False,
                "opened_build_id": None,
                "connection_error": self._connection_error,
            }
        if not needs_open:
            return {
                "connection_generation": self._connection_generation,
                "connection_reopened": False,
                "opened_build_id": self._opened_build_id,
                "connection_error": self._connection_error,
            }
        self._close_connection_locked()
        connection: sqlite3.Connection | None = None
        snapshot_path: Path | None = None
        try:
            open_path = self.database_path
            if os.name == "nt":
                # Windows keeps a sharing handle for read-only SQLite files
                # that can prevent the preparation process from atomically
                # replacing the pathname.  Copy only the currently published
                # inode, close that source handle, and keep the runtime
                # connection on the disposable snapshot.  The source stat
                # above remains the identity used for refresh detection.
                fd, snapshot_name = tempfile.mkstemp(
                    prefix=f".{self.database_path.name}.runtime-",
                    suffix=".sqlite3",
                    dir=str(self.database_path.parent),
                )
                os.close(fd)
                snapshot_path = Path(snapshot_name)
                shutil.copyfile(self.database_path, snapshot_path)
                open_path = snapshot_path
            connection = sqlite3.connect(
                # The published database is never modified in place.  Open
                # it as immutable so a preparation process can atomically
                # replace the pathname even on Windows.  Runtime still
                # stats the path and reopens this connection on every inode
                # or manifest-build change.
                f"file:{open_path.as_posix()}?mode=ro&immutable=1",
                uri=True,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            meta = self._read_database_meta(connection)
            if meta.get("schema_version") != ASR_INDEX_SCHEMA_VERSION:
                raise ValueError("ASR SQLite schema version is unsupported")
            if int(meta.get("sqlite_user_version", -1)) != ASR_SQLITE_USER_VERSION:
                raise ValueError("ASR SQLite user version is unsupported")
            if not str(meta.get("build_id") or ""):
                raise ValueError("ASR SQLite build_id is missing")
            if expected_build_id and str(meta.get("build_id")) != expected_build_id:
                raise ValueError("ASR SQLite build_id does not match manifest")
            self._connection = connection
            self._connection_path = open_path
            self._runtime_snapshot_path = snapshot_path
            self._opened_file_identity = file_identity
            self._opened_build_id = str(meta.get("build_id") or "")
            self._connection_generation += 1
            self._connection_reopened = True
            self._connection_error = None
        except (OSError, sqlite3.Error, ValueError, TypeError) as error:
            if connection is not None:
                with suppress(sqlite3.Error):
                    connection.close()
            if snapshot_path is not None:
                with suppress(OSError):
                    snapshot_path.unlink(missing_ok=True)
            self._close_connection_locked()
            self._connection_error = str(error)
        return {
            "connection_generation": self._connection_generation,
            "connection_reopened": self._connection_reopened,
            "opened_build_id": self._opened_build_id,
            "connection_error": self._connection_error,
        }

    def _database_state(self) -> dict[str, Any]:
        if self._connection is None:
            return {
                "ready": False,
                "segments": 0,
                "fts_segments": 0,
                "videos": 0,
                "mapped_segments": 0,
                "meta": {},
            }
        try:
            tables = {
                str(row[0])
                for row in self._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                ).fetchall()
            }
            required = {"asr_segments", "asr_fts", "asr_meta"}
            columns = self._table_columns("asr_segments")
            row = self._connection.execute(
                "SELECT COUNT(*) AS count, COUNT(DISTINCT video_id) AS videos, "
                "SUM(CASE WHEN frame_uid <> '' AND point_id > 0 THEN 1 ELSE 0 END) AS mapped "
                "FROM asr_segments"
            ).fetchone()
            fts_count = int(self._connection.execute("SELECT COUNT(*) FROM asr_fts").fetchone()[0])
            user_version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
            meta = self._read_database_meta(self._connection)
            count = int(row["count"] or 0) if row else 0
            videos = int(row["videos"] or 0) if row else 0
            mapped = int(row["mapped"] or 0) if row else 0
            expected_columns = {
                "segment_id",
                "video_id",
                "start_ms",
                "end_ms",
                "transcript",
                "transcript_search",
                "frame_uid",
                "point_id",
                "keyframe_n",
                "frame_idx",
                "pts_time_s",
                "fps",
                "image_relpath",
            }
            return {
                "ready": (
                    required.issubset(tables)
                    and expected_columns.issubset(columns)
                    and user_version == ASR_SQLITE_USER_VERSION
                    and meta.get("schema_version") == ASR_INDEX_SCHEMA_VERSION
                    and int(meta.get("sqlite_user_version", -1)) == ASR_SQLITE_USER_VERSION
                    and int(meta.get("segment_count", -1)) == count
                    and int(meta.get("video_count", -1)) == videos
                    and meta.get("mapping_strategy") == ASR_MAPPING_STRATEGY
                    and bool(meta.get("build_id"))
                    and bool(meta.get("source_fingerprint"))
                    and bool(meta.get("canonical_fingerprint"))
                    and count == EXPECTED_ASR_SEGMENTS
                    and fts_count == EXPECTED_ASR_SEGMENTS
                    and mapped == EXPECTED_ASR_SEGMENTS
                ),
                "segments": count,
                "fts_segments": fts_count,
                "videos": videos,
                "mapped_segments": mapped,
                "sqlite_user_version": user_version,
                "tables": sorted(tables),
                "meta": meta,
            }
        except sqlite3.Error as error:
            return {
                "ready": False,
                "segments": 0,
                "fts_segments": 0,
                "videos": 0,
                "mapped_segments": 0,
                "meta": {},
                "error": str(error),
            }
        except (TypeError, ValueError, KeyError) as error:
            return {
                "ready": False,
                "segments": 0,
                "fts_segments": 0,
                "videos": 0,
                "mapped_segments": 0,
                "meta": {},
                "error": str(error),
            }

    def health(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            try:
                manifest = self._load_manifest()
                connection_state = self._refresh_runtime_connection_locked(manifest)
                signature = self._health_signature(manifest)
                if (
                    self._health_cache is not None
                    and now - self._health_cache[0] < ASR_HEALTH_CACHE_TTL_S
                    and self._health_cache[1] == signature
                    and not connection_state.get("connection_error")
                    and (
                        not manifest.get("build_id")
                        or connection_state.get("opened_build_id") == manifest.get("build_id")
                    )
                ):
                    cached = dict(self._health_cache[2])
                    cached.update(connection_state)
                    cached["connection_reopened"] = bool(
                        connection_state.get("connection_reopened")
                    )
                    if self._source_inventory_cache is not None:
                        cached["source_validation_age_s"] = round(
                            max(0.0, now - self._source_inventory_cache[0]), 3
                        )
                    return cached
                payload = self._compute_health_locked(manifest, connection_state)
                self._health_cache = (now, signature, dict(payload))
                return payload
            except Exception as error:
                # A malformed manifest, artifact or SQLite schema must never
                # make health itself crash or leave a stale connection usable.
                self._close_connection_locked()
                self._health_cache = None
                return self._health_error_payload(error)

    @staticmethod
    def _path_stat_signature(path: Path) -> tuple[Any, ...]:
        try:
            stat = path.stat()
            return (
                int(stat.st_size),
                int(stat.st_mtime_ns),
                int(getattr(stat, "st_ino", 0)),
            )
        except OSError:
            return (None, None, None)

    def _source_inventory_snapshot(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Stat source artifacts at most once per ASR health-cache interval.

        Runtime search only needs the published SQLite database.  Source files
        are nevertheless checked for freshness, but their content hashes are
        recomputed only when the recorded stat changes.  The inventory is
        cached separately so every health/config/search call does not perform
        another 1,746-file walk.
        """

        now = time.monotonic()
        manifest_identity = self._path_stat_signature(self.manifest_path)
        source_directory_identity = self._path_stat_signature(self.segments_dir)
        cached = self._source_inventory_cache
        if (
            cached is not None
            and now - cached[0] < ASR_HEALTH_CACHE_TTL_S
            and cached[1] == manifest_identity
            and cached[2].get("source_directory_identity") == source_directory_identity
        ):
            snapshot = dict(cached[2])
            snapshot["age_s"] = max(0.0, now - cached[0])
            snapshot["cache_hit"] = True
            return snapshot

        source_files = manifest.get("source_files")
        inventory: list[tuple[Any, ...]] = []
        diagnostics: list[dict[str, Any]] = []
        source_matches = True
        source_fingerprints_present = True
        source_records_valid = isinstance(source_files, list) and bool(source_files)
        if not source_records_valid:
            source_matches = False
            source_fingerprints_present = False
            source_files = []

        for record in source_files:
            if isinstance(record, dict):
                try:
                    expected = (self.segments_dir.parent / str(record.get("path") or "")).resolve()
                    actual_identity = self._path_stat_signature(expected)
                    inventory.append((str(record.get("path") or ""), actual_identity))
                except (OSError, TypeError, ValueError):
                    inventory.append((str(record.get("path") or ""), (None, None, None)))
            else:
                inventory.append(("<invalid>", (None, None, None)))
            matched, stat_matches, fingerprint_present = self._artifact_status(
                record,
                relative_root=self.segments_dir.parent,
                verify_hash=False,
            )
            source_matches = source_matches and matched
            source_fingerprints_present = source_fingerprints_present and fingerprint_present
            if not matched or not stat_matches:
                diagnostic = self._artifact_diagnostics.get(
                    str(record.get("path") if isinstance(record, dict) else "<invalid>")
                )
                if diagnostic:
                    diagnostics.append(dict(diagnostic))

        snapshot = {
            "inventory_signature": tuple(inventory),
            "source_directory_identity": source_directory_identity,
            "source_matches": source_matches,
            "source_fingerprints_present": source_fingerprints_present,
            "source_records_valid": source_records_valid,
            "diagnostics": diagnostics,
            "validated_at": now,
            "age_s": 0.0,
            "cache_hit": False,
        }
        self._source_inventory_cache = (now, manifest_identity, dict(snapshot))
        return snapshot

    def _health_signature(self, manifest: dict[str, Any]) -> tuple[Any, ...]:
        source_inventory = self._source_inventory_snapshot(manifest)
        return (
            self._path_stat_signature(self.manifest_path),
            self._path_stat_signature(self.database_path),
            self._path_stat_signature(
                self.segments_dir.parent
                / "visual_embeddings"
                / "metaclip2"
                / "keyframes_metadata.jsonl"
            ),
            self._path_stat_signature(self.segments_dir),
            source_inventory.get("inventory_signature", ()),
            str(manifest.get("build_id") or ""),
        )

    def _artifact_status(
        self,
        record: Any,
        *,
        relative_root: Path,
        verify_hash: bool = True,
    ) -> tuple[bool, bool, bool]:
        """Validate one manifest artifact without hashing unchanged files.

        A matching stat is trusted through the immutable build-id chain.  A
        caller can request a direct content check with ``verify_hash=True``;
        changed stats always force that check automatically.
        """

        if not isinstance(record, dict):
            self._artifact_diagnostics["<invalid>"] = {
                "path": "<invalid>",
                "stat_matches": False,
                "hash_recomputed": False,
                "fingerprint_matches": False,
                "error": "artifact record must be an object",
            }
            return False, False, False
        diagnostic_key = str(record.get("path") or "<missing>")
        diagnostic: dict[str, Any] = {
            "path": diagnostic_key,
            "stat_matches": False,
            "hash_recomputed": False,
            "fingerprint_matches": False,
        }
        try:
            root = relative_root.resolve()
            expected_path = (relative_root / str(record["path"])).resolve()
            if expected_path != root and root not in expected_path.parents:
                diagnostic["error"] = "artifact path escapes configured root"
                self._artifact_diagnostics[diagnostic_key] = diagnostic
                return False, False, False
            stat = expected_path.stat()
            stat_matches = int(record.get("size", -1)) == int(stat.st_size) and int(
                record.get("mtime_ns", -1)
            ) == int(stat.st_mtime_ns)
            diagnostic["stat_matches"] = stat_matches
            expected_sha = str(record.get("sha256") or "")
            if len(expected_sha) != 64 or any(
                char not in "0123456789abcdefABCDEF" for char in expected_sha
            ):
                diagnostic["error"] = "artifact sha256 is missing or malformed"
                self._artifact_diagnostics[diagnostic_key] = diagnostic
                return False, stat_matches, False
            identity = (int(stat.st_size), int(stat.st_mtime_ns), int(getattr(stat, "st_ino", 0)))
            cache_key = str(expected_path.resolve())
            cached = self._fingerprint_cache.get(cache_key)
            hash_required = verify_hash or not stat_matches
            if hash_required and (cached is None or cached[0] != identity):
                cached = (identity, _sha256_file(expected_path))
                self._fingerprint_cache[cache_key] = cached
                diagnostic["hash_recomputed"] = True
            diagnostic["verification_mode"] = "content" if hash_required else "stat_chain"
            fingerprint_matches = (
                bool(cached is not None and cached[1] == expected_sha)
                if hash_required or cached is not None
                else True
            )
            diagnostic["fingerprint_matches"] = fingerprint_matches
            self._artifact_diagnostics[diagnostic_key] = diagnostic
            return fingerprint_matches, stat_matches, True
        except (KeyError, OSError, TypeError, ValueError) as error:
            diagnostic["error"] = str(error)
            self._artifact_diagnostics[diagnostic_key] = diagnostic
            return False, False, False

    def _compute_health_locked(
        self, manifest: dict[str, Any], connection_state: dict[str, Any]
    ) -> dict[str, Any]:
        self._artifact_diagnostics.clear()
        db_state = self._database_state()
        warnings: list[str] = []
        identity = manifest.get("offline_identity") or {}
        revision_verified = bool(
            isinstance(identity, dict) and identity.get("revision_verified") is True
        )
        if not revision_verified:
            warnings.append("ASR offline model revision is not cryptographically verified")
        db_record = manifest.get("database") or {}
        try:
            manifest_database_path = (
                (self.database_path.parent / str(db_record.get("path") or "")).resolve()
                if isinstance(db_record, dict)
                else None
            )
            database_path_matches = manifest_database_path == self.database_path.resolve()
        except (OSError, TypeError, ValueError):
            database_path_matches = False
        database_matches, database_stat_matches, database_fingerprint_present = (
            self._artifact_status(
                db_record, relative_root=self.database_path.parent, verify_hash=False
            )
        )
        source_inventory = self._source_inventory_snapshot(manifest)
        for diagnostic in source_inventory.get("diagnostics", []):
            if isinstance(diagnostic, dict) and diagnostic.get("path"):
                self._artifact_diagnostics[str(diagnostic["path"])] = dict(diagnostic)
        source_files = manifest.get("source_files") or []
        source_matches = bool(source_inventory.get("source_matches"))
        source_fingerprints_present = bool(source_inventory.get("source_fingerprints_present"))
        manifest_source_fingerprint = ""
        if not isinstance(source_files, list) or not source_files:
            source_matches = False
            source_fingerprints_present = False
        else:
            try:
                manifest_source_fingerprint = source_fingerprint(source_files)
            except (TypeError, ValueError, KeyError):
                manifest_source_fingerprint = ""
        canonical_path = (
            self.segments_dir.parent
            / "visual_embeddings"
            / "metaclip2"
            / "keyframes_metadata.jsonl"
        )
        canonical_record = manifest.get("canonical_metadata") or {}
        if not isinstance(canonical_record, dict):
            canonical_record = {}
        canonical_matches, canonical_stat_matches, canonical_fingerprint_present = (
            self._artifact_status(
                canonical_record, relative_root=canonical_path.parent, verify_hash=False
            )
        )
        expected_build_id = ""
        if manifest_source_fingerprint and canonical_fingerprint_present:
            expected_build_id = build_id_for(
                source_fingerprint_value=manifest_source_fingerprint,
                canonical_fingerprint_value=str(canonical_record.get("sha256") or ""),
                segment_count=_safe_int(manifest.get("segment_count", 0), 0),
                video_count=_safe_int(manifest.get("video_count", 0), 0),
            )
        mapping = manifest.get("mapping") or {}
        empty_video_ids = manifest.get("empty_video_ids") or []
        indexed_video_count = _safe_int(manifest.get("indexed_video_count", -1), -1)
        empty_video_count = _safe_int(manifest.get("empty_video_count", -1), -1)
        video_partition_matches = bool(
            isinstance(empty_video_ids, list)
            and all(isinstance(video_id, str) and video_id for video_id in empty_video_ids)
            and len(set(empty_video_ids)) == len(empty_video_ids)
            and len(empty_video_ids) == empty_video_count
            and indexed_video_count >= 0
            and indexed_video_count + empty_video_count == EXPECTED_ASR_VIDEOS
        )
        mapping_matches = (
            isinstance(mapping, dict)
            and _safe_int(mapping.get("mapped_segments", 0), 0) == EXPECTED_ASR_SEGMENTS
            and mapping.get("strategy") == ASR_MAPPING_STRATEGY
        )
        source_directory_matches = manifest.get("source_directory") == "asr_segments"
        source_file_count_matches = (
            isinstance(source_files, list)
            and len(source_files) == EXPECTED_ASR_VIDEOS * 2
            and _safe_int(manifest.get("source_file_count", 0), 0) == len(source_files)
            and _safe_int(manifest.get("source_segment_file_count", 0), 0) == EXPECTED_ASR_VIDEOS
            and _safe_int(manifest.get("source_manifest_file_count", 0), 0) == EXPECTED_ASR_VIDEOS
        )
        ready = bool(
            manifest.get("schema_version") == ASR_INDEX_SCHEMA_VERSION
            and manifest.get("status") == "ready"
            and manifest.get("passed") is True
            and _safe_int(manifest.get("segment_count", 0), 0) == EXPECTED_ASR_SEGMENTS
            and _safe_int(manifest.get("video_count", 0), 0) == EXPECTED_ASR_VIDEOS
            and video_partition_matches
            and mapping_matches
            and source_directory_matches
            and db_state.get("ready") is True
            and _safe_int(db_state.get("videos", 0), 0) == indexed_video_count
            and source_file_count_matches
            and database_path_matches
            and database_matches
            and source_matches
            and canonical_matches
            and database_fingerprint_present
            and source_fingerprints_present
            and canonical_fingerprint_present
            and manifest.get("build_id")
            and manifest.get("source_fingerprint") == manifest_source_fingerprint
            and manifest.get("canonical_fingerprint") == canonical_record.get("sha256")
            and manifest.get("build_id") == expected_build_id
            and manifest.get("source_fingerprint")
            == db_state.get("meta", {}).get("source_fingerprint")
            and manifest.get("canonical_fingerprint")
            == db_state.get("meta", {}).get("canonical_fingerprint")
            and manifest.get("build_id") == db_state.get("meta", {}).get("build_id")
            and db_state.get("meta", {}).get("mapping_strategy") == ASR_MAPPING_STRATEGY
        )
        if not database_stat_matches:
            warnings.append("ASR database stat differs from manifest")
        if not canonical_stat_matches:
            warnings.append("ASR canonical metadata stat differs from manifest")
        if not source_inventory.get("source_records_valid", False):
            warnings.append("ASR source artifact inventory is missing or malformed")
        artifact_diagnostics = [
            dict(item)
            for item in self._artifact_diagnostics.values()
            if item.get("error")
            or item.get("hash_recomputed")
            or item.get("fingerprint_matches") is False
            or item.get("stat_matches") is False
        ]
        return {
            "ready": ready,
            "production_ready": ready and revision_verified,
            "database": str(self.database_path),
            "manifest": str(self.manifest_path),
            "segments": int(db_state.get("segments", 0)),
            "fts_segments": int(db_state.get("fts_segments", 0)),
            "mapped_segments": int(db_state.get("mapped_segments", 0)),
            "videos": _safe_int(manifest.get("video_count", 0), 0),
            "indexed_videos": indexed_video_count,
            "empty_videos": empty_video_count,
            "empty_video_ids": list(empty_video_ids) if isinstance(empty_video_ids, list) else [],
            "video_partition_matches_manifest": video_partition_matches,
            "schema_version": manifest.get("schema_version"),
            "sqlite_user_version": db_state.get("sqlite_user_version"),
            "database_matches_manifest": database_matches,
            "database_stat_matches_manifest": database_stat_matches,
            "database_path_matches_manifest": database_path_matches,
            "source_matches_manifest": source_matches,
            "source_fingerprint_matches_manifest": manifest.get("source_fingerprint")
            == manifest_source_fingerprint,
            "source_file_count_matches_manifest": source_file_count_matches,
            "canonical_metadata_matches_manifest": canonical_matches,
            "canonical_metadata_stat_matches_manifest": canonical_stat_matches,
            "canonical_fingerprint_matches_manifest": manifest.get("canonical_fingerprint")
            == canonical_record.get("sha256"),
            "build_id_matches_expected": bool(
                manifest.get("build_id") and manifest.get("build_id") == expected_build_id
            ),
            "fingerprints_recorded": (
                database_fingerprint_present
                and source_fingerprints_present
                and canonical_fingerprint_present
            ),
            "build_id": manifest.get("build_id"),
            "manifest_build_id": manifest.get("build_id"),
            "internal_build_id": db_state.get("meta", {}).get("build_id"),
            "source_fingerprint_matches": manifest.get("source_fingerprint")
            == db_state.get("meta", {}).get("source_fingerprint"),
            "canonical_fingerprint_matches": manifest.get("canonical_fingerprint")
            == db_state.get("meta", {}).get("canonical_fingerprint"),
            "internal_metadata_matches": bool(
                manifest.get("build_id")
                and manifest.get("build_id") == db_state.get("meta", {}).get("build_id")
            ),
            "mapping_matches_manifest": mapping_matches,
            "source_directory_matches_manifest": source_directory_matches,
            "offline_identity": identity if isinstance(identity, dict) else {},
            "revision_verified": revision_verified,
            "model_ids": manifest.get("model_ids", []),
            "connection_generation": connection_state.get("connection_generation", 0),
            "connection_reopened": bool(connection_state.get("connection_reopened")),
            "opened_build_id": connection_state.get("opened_build_id"),
            "connection_error": connection_state.get("connection_error"),
            "source_validation_age_s": round(float(source_inventory.get("age_s", 0.0)), 3),
            "artifact_diagnostics": artifact_diagnostics,
            "warnings": warnings,
        }

    def _nearest_frame(
        self,
        video_id: str,
        midpoint_s: float,
        canonical_frame_index: dict[str, dict[str, Any]],
        frames_by_video: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        if frames_by_video is None:
            if self.metadata is None:
                raise ValueError("ASR preparation requires canonical frame metadata")
            frames = list(self.metadata.video_frames(video_id))
        else:
            frames = frames_by_video.get(video_id, [])
        if not frames:
            raise ValueError(f"No canonical frames found for ASR video {video_id}")
        selected = dict(
            min(
                frames,
                key=lambda item: (
                    abs(float(item.get("pts_time_s", 0.0)) - midpoint_s),
                    float(item.get("pts_time_s", 0.0)),
                    int(item.get("frame_idx", -1)),
                ),
            )
        )
        frame_uid = str(selected.get("frame_uid") or "")
        canonical = canonical_frame_index.get(frame_uid)
        if canonical is None:
            raise ValueError(f"ASR segment mapped to unknown canonical frame {frame_uid}")
        selected.update(canonical)
        return selected

    def _ensure_index(
        self,
        canonical_frame_index: dict[str, dict[str, Any]],
        build_context: dict[str, Any],
    ) -> None:
        if self._connection is None:
            raise RuntimeError("ASR SQLite connection is unavailable")
        if not canonical_frame_index:
            raise ValueError("ASR preparation requires a canonical frame index")
        required_context = {
            "build_id",
            "source_fingerprint",
            "canonical_fingerprint",
        }
        if not required_context.issubset(build_context):
            raise ValueError("ASR preparation requires deterministic build context")
        total = 0
        rows: list[tuple[Any, ...]] = []
        frames_by_video: dict[str, list[dict[str, Any]]] = {}
        for frame in canonical_frame_index.values():
            frames_by_video.setdefault(str(frame["video_id"]), []).append(frame)
        for frames in frames_by_video.values():
            frames.sort(key=lambda item: (float(item["pts_time_s"]), int(item["frame_idx"])))
        with self._connection:
            self._connection.execute("DROP TABLE IF EXISTS asr_meta")
            self._connection.execute("DROP TABLE IF EXISTS asr_fts")
            self._connection.execute("DROP TABLE IF EXISTS asr_segments")
            self._connection.execute(
                """
                CREATE TABLE asr_segments (
                    id INTEGER PRIMARY KEY,
                    segment_id TEXT NOT NULL UNIQUE,
                    video_id TEXT NOT NULL,
                    start_ms INTEGER NOT NULL,
                    end_ms INTEGER NOT NULL,
                    transcript TEXT NOT NULL,
                    transcript_search TEXT NOT NULL,
                    frame_uid TEXT NOT NULL,
                    point_id INTEGER NOT NULL,
                    keyframe_n INTEGER NOT NULL,
                    frame_idx INTEGER NOT NULL,
                    pts_time_s REAL NOT NULL,
                    fps REAL NOT NULL,
                    image_relpath TEXT NOT NULL
                )
                """
            )
            self._connection.execute("CREATE INDEX asr_video_idx ON asr_segments(video_id)")
            self._connection.execute("CREATE INDEX asr_frame_idx ON asr_segments(frame_uid)")
            self._connection.execute(
                "CREATE VIRTUAL TABLE asr_fts USING fts5(transcript_search, content='asr_segments', content_rowid='id', tokenize='unicode61 remove_diacritics 2')"
            )
            for path in sorted(self.segments_dir.glob("*.jsonl")):
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        parsed = _parse_source_segment(json.loads(line), path, line_number)
                        midpoint_s = (parsed["start_ms"] + parsed["end_ms"]) / 2000.0
                        frame = self._nearest_frame(
                            parsed["video_id"], midpoint_s, canonical_frame_index, frames_by_video
                        )
                        rows.append(
                            (
                                parsed["segment_id"],
                                parsed["video_id"],
                                parsed["start_ms"],
                                parsed["end_ms"],
                                parsed["transcript"],
                                parsed["transcript_search"],
                                str(frame["frame_uid"]),
                                int(frame["point_id"]),
                                int(frame["keyframe_n"]),
                                int(frame["frame_idx"]),
                                float(frame.get("pts_time_s", 0.0)),
                                float(frame.get("fps", 0.0)),
                                str(frame.get("image_relpath") or ""),
                            )
                        )
                        if len(rows) >= 2_000:
                            self._connection.executemany(
                                "INSERT INTO asr_segments(segment_id, video_id, start_ms, end_ms, transcript, transcript_search, frame_uid, point_id, keyframe_n, frame_idx, pts_time_s, fps, image_relpath) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                rows,
                            )
                            total += len(rows)
                            rows.clear()
            if rows:
                self._connection.executemany(
                    "INSERT INTO asr_segments(segment_id, video_id, start_ms, end_ms, transcript, transcript_search, frame_uid, point_id, keyframe_n, frame_idx, pts_time_s, fps, image_relpath) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                total += len(rows)
            if total != EXPECTED_ASR_SEGMENTS:
                raise ValueError(f"Expected {EXPECTED_ASR_SEGMENTS} ASR segments, found {total}")
            self._connection.execute("INSERT INTO asr_fts(asr_fts) VALUES ('rebuild')")
            self._connection.execute(
                """
                CREATE TABLE asr_meta (
                    schema_version TEXT NOT NULL,
                    sqlite_user_version INTEGER NOT NULL,
                    build_id TEXT NOT NULL UNIQUE,
                    source_fingerprint TEXT NOT NULL,
                    canonical_fingerprint TEXT NOT NULL,
                    mapping_strategy TEXT NOT NULL,
                    segment_count INTEGER NOT NULL,
                    video_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                INSERT INTO asr_meta(
                    schema_version, sqlite_user_version, build_id,
                    source_fingerprint, canonical_fingerprint, mapping_strategy,
                    segment_count, video_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ASR_INDEX_SCHEMA_VERSION,
                    ASR_SQLITE_USER_VERSION,
                    str(build_context["build_id"]),
                    str(build_context["source_fingerprint"]),
                    str(build_context["canonical_fingerprint"]),
                    ASR_MAPPING_STRATEGY,
                    total,
                    int(build_context.get("indexed_video_count", len(frames_by_video))),
                    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                ),
            )
            self._connection.execute(f"PRAGMA user_version = {ASR_SQLITE_USER_VERSION}")

    def validate_built_index(self, build_context: dict[str, Any]) -> dict[str, Any]:
        """Validate a staging database before its atomic publication."""

        with self._lock:
            if self._connection is None:
                raise RuntimeError("ASR SQLite connection is unavailable")
            integrity = str(self._connection.execute("PRAGMA integrity_check").fetchone()[0])
            state = self._database_state()
            meta = state.get("meta") or {}
            checks = {
                "integrity_ok": integrity.casefold() == "ok",
                "state_ready": state.get("ready") is True,
                "build_id_matches": meta.get("build_id") == build_context.get("build_id"),
                "source_fingerprint_matches": meta.get("source_fingerprint")
                == build_context.get("source_fingerprint"),
                "canonical_fingerprint_matches": meta.get("canonical_fingerprint")
                == build_context.get("canonical_fingerprint"),
            }
            if not all(checks.values()):
                raise ValueError(
                    f"ASR staging database validation failed: checks={checks}, state={state}"
                )
            return {"integrity": integrity, "state": state, "checks": checks}

    def _query_stream(self, query: str, limit: int) -> tuple[list[sqlite3.Row], list[str]]:
        tokens = query_tokens(query)
        if not tokens:
            return [], []
        expression = _fts_expression(tokens)
        if not expression or self._connection is None:
            return [], tokens
        rows = self._connection.execute(
            """
            SELECT s.segment_id, s.video_id, s.start_ms, s.end_ms,
                   s.transcript, s.transcript_search, s.frame_uid,
                   s.point_id, s.keyframe_n, s.frame_idx, s.pts_time_s,
                   s.fps, s.image_relpath, bm25(asr_fts) AS bm25_score
            FROM asr_fts
            JOIN asr_segments AS s ON s.id = asr_fts.rowid
            WHERE asr_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
            """,
            (expression, int(limit)),
        ).fetchall()
        return rows, tokens

    def search_many(
        self,
        query_by_role: dict[str, str],
        *,
        per_stream_top_k: int = 2_000,
        final_top_k: int = 500,
        _allow_single: bool = False,
    ) -> dict[str, Any]:
        if per_stream_top_k < 1 or per_stream_top_k > 2_000:
            raise ValueError("per_stream_top_k must be between 1 and 2000")
        if final_top_k < 1 or final_top_k > 500:
            raise ValueError("final_top_k must be between 1 and 500")
        started = time.perf_counter()
        segment_hits: dict[str, dict[str, Any]] = {}
        stream_counts: dict[str, int] = {}
        with self._lock:
            # ``health`` refreshes the current inode/manifest pair and is the
            # single readiness assertion for this locked search.  Calling the
            # full health path and a second fast check here duplicated all
            # SQLite/schema work on every request.
            if not self.health().get("ready"):
                raise RuntimeError("ASR FTS index is not prepared or its manifest is stale")
            expected_role_count = 1 if _allow_single else 6
            expected_stream_count = 1 if _allow_single else 12
            stream_queries = {
                str(stream): str(value).strip() for stream, value in query_by_role.items()
            }
            if _allow_single:
                valid_streams = len(stream_queries) == expected_stream_count
            else:
                identities = [_stream_identity(stream) for stream in stream_queries]
                valid_streams = (
                    len(stream_queries) == expected_stream_count
                    and len(set(role for role, _language in identities)) == expected_role_count
                    and all(role in QUERY_ROLES for role, _language in identities)
                    and all(language in {"vi", "en"} for _role, language in identities)
                    and len(set(identities)) == expected_stream_count
                )
            if not valid_streams or any(not value for value in stream_queries.values()):
                if _allow_single:
                    raise ValueError("ASR requires one non-empty query")
                raise ValueError("ASR requires six bilingual query variants")
            if any(not query_tokens(str(value)) for value in stream_queries.values()):
                raise ValueError("ASR queries must contain at least one searchable token")
            for stream, query in stream_queries.items():
                role, language = _stream_identity(stream)
                rows, tokens = self._query_stream(str(query), per_stream_top_k)
                stream_counts[stream] = len(rows)
                max_relevance = (
                    max((max(0.0, -float(row["bm25_score"])) for row in rows), default=0.0) or 1.0
                )
                ordered_tokens = ordered_lexical_tokens(str(query))
                query_bigrams = _ordered_lexical_bigrams(str(query))
                search_token_set = set(tokens)
                lexical_occurrence_count = sum(
                    1 for token in ordered_tokens if token in search_token_set
                )
                for rank, row in enumerate(rows, 1):
                    transcript_search = str(row["transcript_search"])
                    transcript_tokens = _folded_tokens(transcript_search)
                    token_set = set(transcript_tokens)
                    matched = [token for token in tokens if token in token_set]
                    token_coverage = len(matched) / len(tokens) if tokens else 0.0
                    transcript_bigrams = set(_token_bigrams(transcript_tokens))
                    if query_bigrams:
                        ngram_coverage = sum(
                            1 for value in query_bigrams if value in transcript_bigrams
                        ) / len(query_bigrams)
                    elif lexical_occurrence_count == 1:
                        ngram_coverage = token_coverage
                    else:
                        ngram_coverage = 0.0
                    bm25_raw = float(row["bm25_score"])
                    bm25_relevance = max(0.0, -bm25_raw) / max_relevance
                    combined = 0.55 * bm25_relevance + 0.30 * token_coverage + 0.15 * ngram_coverage
                    segment_id = str(row["segment_id"])
                    stream_evidence = {
                        "role": role,
                        "language": language,
                        "stream": stream,
                        "rank": rank,
                        "bm25_raw": round(bm25_raw, 8),
                        "bm25_relevance": round(bm25_relevance, 8),
                        "token_coverage": round(token_coverage, 8),
                        "ngram_coverage": round(ngram_coverage, 8),
                        "combined_score": round(combined, 8),
                        "matched_terms": matched,
                        "query_tokens": tokens,
                        "ordered_query_tokens": ordered_tokens,
                        "query_bigrams": [list(value) for value in query_bigrams],
                    }
                    segment = segment_hits.setdefault(
                        segment_id,
                        {
                            "segment_id": segment_id,
                            "video_id": str(row["video_id"]),
                            "start_ms": int(row["start_ms"]),
                            "end_ms": int(row["end_ms"]),
                            "transcript": str(row["transcript"]),
                            "frame_uid": str(row["frame_uid"]),
                            "point_id": int(row["point_id"]),
                            "keyframe_n": int(row["keyframe_n"]),
                            "frame_idx": int(row["frame_idx"]),
                            "pts_time_s": float(row["pts_time_s"]),
                            "fps": float(row["fps"]),
                            "image_relpath": str(row["image_relpath"]),
                            "query_scores": {},
                            "stream_provenance": {},
                        },
                    )
                    # Keep VI and EN evidence in separate stream slots.  A
                    # role-level dictionary would let the later language
                    # overwrite the earlier one instead of taking the
                    # bilingual maximum.
                    segment["query_scores"][stream] = combined
                    segment["stream_provenance"][stream] = stream_evidence
        if not segment_hits:
            return {
                "results": [],
                "candidate_segment_count": 0,
                "candidate_frame_count": 0,
                "stream_counts": stream_counts,
                "timing": {"total_ms": round((time.perf_counter() - started) * 1000.0, 2)},
            }
        frame_hits: dict[str, dict[str, Any]] = {}
        for segment in segment_hits.values():
            best_stream, best_score = max(
                segment["query_scores"].items(),
                key=lambda pair: (
                    float(pair[1]),
                    -int(segment["stream_provenance"][pair[0]]["rank"]),
                    pair[0],
                ),
            )
            best_role, best_language = _stream_identity(best_stream)
            segment["best_query_role"] = best_role
            segment["best_query_language"] = best_language
            segment["best_query_stream"] = best_stream
            segment["raw_score"] = float(best_score)
            segment["best_rank"] = int(segment["stream_provenance"][best_stream]["rank"])
            frame_uid = segment["frame_uid"]
            current = frame_hits.get(frame_uid)
            if current is None or (
                segment["raw_score"],
                -segment["best_rank"],
                segment["segment_id"],
            ) > (current["raw_score"], -current["best_rank"], current["segment_id"]):
                frame_hits[frame_uid] = segment
        values = [float(item["raw_score"]) for item in frame_hits.values()]
        normalized = _sigmoid_zscore(values)
        ranked_frames = []
        for index, segment in enumerate(frame_hits.values()):
            normalized_score = float(normalized[index])
            frame = base_frame(segment, score=normalized_score, rank=1, score_type="asr_bm25_ngram")
            winner = segment["stream_provenance"][segment["best_query_stream"]]
            role_scores: dict[str, float] = {}
            language_scores: dict[str, dict[str, float]] = {}
            for stream, score in segment["query_scores"].items():
                role, language = _stream_identity(stream)
                role_scores[role] = max(role_scores.get(role, 0.0), float(score))
                if language is not None:
                    language_scores.setdefault(role, {})[language] = round(float(score), 8)
            frame.update(
                {
                    "frame_uid": segment["frame_uid"],
                    "point_id": int(segment["point_id"]),
                    "global_idx": int(segment["point_id"]),
                    "asr_raw_score": round(float(segment["raw_score"]), 8),
                    "asr_normalized_score": round(normalized_score, 8),
                    "asr_best_query_role": segment["best_query_role"],
                    "asr_best_query_language": segment["best_query_language"],
                    "asr_best_query_stream": segment["best_query_stream"],
                    "asr_best_rank": int(segment["best_rank"]),
                    "asr_segment_id": segment["segment_id"],
                    "asr_start_s": segment["start_ms"] / 1000.0,
                    "asr_end_s": segment["end_ms"] / 1000.0,
                    "asr_transcript": segment["transcript"],
                    "matched_terms": winner["matched_terms"],
                    "bm25_raw": winner["bm25_raw"],
                    "bm25_relevance": winner["bm25_relevance"],
                    "token_coverage": winner["token_coverage"],
                    "ngram_coverage": winner["ngram_coverage"],
                    "asr_query_scores": {
                        role: round(float(role_scores.get(role, 0.0)), 8) for role in QUERY_ROLES
                    },
                    "asr_language_scores": language_scores,
                    "asr_stream_provenance": {
                        stream: segment["stream_provenance"].get(stream)
                        for stream in stream_queries
                    },
                    "asr_segment_candidate_count": len(segment_hits),
                }
            )
            ranked_frames.append(frame)
        ranked_frames.sort(
            key=lambda item: (
                -float(item["asr_normalized_score"]),
                -float(item["asr_raw_score"]),
                int(item.get("asr_best_rank", 2_000_001)),
                str(item["video_id"]),
                int(item["frame_idx"]),
                str(item.get("asr_segment_id", "")),
            )
        )
        for rank, frame in enumerate(ranked_frames[:final_top_k], 1):
            frame["rank"] = rank
        return {
            "results": ranked_frames[:final_top_k],
            "candidate_segment_count": len(segment_hits),
            "candidate_frame_count": len(frame_hits),
            "stream_counts": stream_counts,
            "timing": {"total_ms": round((time.perf_counter() - started) * 1000.0, 2)},
        }

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Compatibility wrapper for the existing single-speech endpoint."""

        result = self.search_many(
            {"original": query},
            per_stream_top_k=min(2_000, max(1, top_k * 30)),
            final_top_k=top_k,
            _allow_single=True,
        )
        return result["results"]

    def audio_span(self, video_id: str, frame_idx: int, window_frames: int = 450) -> str:
        with self._lock:
            if not self.health().get("ready") or self.metadata is None or self._connection is None:
                return ""
            frames = self.metadata.video_frames(video_id)
            if not frames:
                return ""
            target = next(
                (frame for frame in frames if int(frame.get("frame_idx", -1)) == int(frame_idx)),
                None,
            )
            if target is None:
                return ""
            fps = float(target.get("fps") or 0.0) or 30.0
            center_ms = float(target.get("pts_time_s", frame_idx / fps)) * 1000.0
            half_window_ms = max(1, int(round(abs(window_frames) / fps * 1000.0)))
            rows = self._connection.execute(
                "SELECT transcript FROM asr_segments WHERE video_id = ? AND end_ms >= ? AND start_ms <= ? ORDER BY start_ms",
                (
                    str(video_id).upper().replace("-", "_"),
                    int(center_ms - half_window_ms),
                    int(center_ms + half_window_ms),
                ),
            ).fetchall()
        return " ".join(
            dict.fromkeys(
                str(row["transcript"]).strip() for row in rows if str(row["transcript"]).strip()
            )
        )


def build_asr_manifest(
    *,
    data_root: Path,
    state_root: Path,
    database_path: Path,
    source_facts: dict[str, Any],
    build_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    canonical_path = data_root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl"
    if build_context is None:
        canonical_record = artifact_record(canonical_path)
        source_fingerprint_value = source_fingerprint(source_facts["source_files"])
        build_context = {
            "source_fingerprint": source_fingerprint_value,
            "canonical_fingerprint": canonical_record["sha256"],
            "build_id": build_id_for(
                source_fingerprint_value=source_fingerprint_value,
                canonical_fingerprint_value=canonical_record["sha256"],
                segment_count=source_facts["segment_count"],
                video_count=source_facts["video_count"],
            ),
        }
    return {
        "schema_version": ASR_INDEX_SCHEMA_VERSION,
        "status": "ready",
        "passed": True,
        "build_id": str(build_context["build_id"]),
        "source_fingerprint": str(build_context["source_fingerprint"]),
        "canonical_fingerprint": str(build_context["canonical_fingerprint"]),
        "database": artifact_record(database_path, relative_to=state_root),
        "source_directory": "asr_segments",
        "source_files": source_facts["source_files"],
        "segment_count": int(source_facts["segment_count"]),
        "video_count": int(source_facts["video_count"]),
        "indexed_video_count": int(source_facts["indexed_video_count"]),
        "empty_video_count": int(source_facts["empty_video_count"]),
        "empty_video_ids": list(source_facts["empty_video_ids"]),
        "source_file_count": len(source_facts["source_files"]),
        "source_segment_file_count": int(source_facts["source_segment_file_count"]),
        "source_manifest_file_count": int(source_facts["source_manifest_file_count"]),
        "model_ids": source_facts["model_ids"],
        "engines": source_facts["engines"],
        "canonical_metadata": artifact_record(canonical_path),
        "mapping": {
            "mapped_segments": int(source_facts["segment_count"]),
            "strategy": ASR_MAPPING_STRATEGY,
        },
        "offline_identity": {
            "model_id": source_facts["model_ids"][0],
            "engine": source_facts["engines"][0],
            "revision_verified": False,
            "evidence": "per-video ASR manifests; immutable checkpoint revision is not recorded",
        },
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


__all__ = [
    "ASR_INDEX_SCHEMA_VERSION",
    "ASR_SQLITE_USER_VERSION",
    "EXPECTED_ASR_SEGMENTS",
    "EXPECTED_ASR_VIDEOS",
    "ASR_MAPPING_STRATEGY",
    "AsrFtsIndex",
    "artifact_record",
    "build_asr_manifest",
    "build_id_for",
    "fold_text",
    "load_canonical_frame_index",
    "normalize_text",
    "ordered_lexical_tokens",
    "query_tokens",
    "source_fingerprint",
    "validate_asr_sources",
    "_ordered_lexical_bigrams",
]
