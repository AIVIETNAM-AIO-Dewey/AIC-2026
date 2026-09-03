"""Canonical OCR FTS5 preparation and read-only runtime retrieval.

OCR is an optional Branch-3 capability.  The preparation command publishes a
SQLite/manifest pair atomically; the runtime never rebuilds the index and
never falls back to a stale connection or to a non-canonical frame identity.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from ..infrastructure.qdrant import base_frame
from .lexical import (
    _folded_tokens,
    _ordered_lexical_bigrams,
    _stream_identity,
    _token_bigrams,
    fold_text,
    ordered_lexical_tokens,
    query_tokens,
    sigmoid_zscore,
)


LOGGER = logging.getLogger(__name__)

OCR_INDEX_SCHEMA_VERSION = "branch3.ocr-index.v3"
OCR_SQLITE_USER_VERSION = 3
OCR_MAPPING_STRATEGY = "canonical_frame_uid_exact_v1"
OCR_LEXICAL_CONTRACT_VERSION = "branch3.ocr-lexical.v3"
OCR_HEALTH_CACHE_TTL_S = 30.0
OCR_SOURCE_AUDIT_TTL_S = 300.0
OCR_EXPECTED_SOURCE_FILES = 873
OCR_EXPECTED_FRAMES = 247_956
OCR_RESULT_SCHEMA_VERSION = "branch3.ocr.result.v1"
OCR_SOURCE_DIR_NAME = "ocr_transcripts"
OCR_CANONICAL_RELATIVE_PATH = "visual_embeddings/metaclip2/keyframes_metadata.jsonl"

QUERY_ROLES = ("original", "entity", "action", "context", "synonym", "keyword")


def repair_mojibake(value: str, *, max_passes: int = 3) -> str:
    """Repair up to three *lossless* OCR encoding round trips.

    OCR exports have appeared both as Latin-1 and Windows-1252 renderings of
    UTF-8.  The latter is important for strings such as ``"ÃƒÂ "``: the
    intermediate byte ``0x83`` is represented by ``ƒ`` and therefore cannot be
    encoded with strict Latin-1.  Try both single-byte codecs, choose only a
    candidate with fewer mojibake markers, and stop on any lossy replacement.
    The shared lexical helper remains deliberately one-pass for ASR; this
    deeper repair is OCR-specific.
    """

    current = str(value or "")

    def marker_count(text: str) -> int:
        # Include the Windows-1252 ``ƒ`` introduced by a second round trip,
        # along with the common UTF-8-as-single-byte marker characters.
        return sum(
            text.count(marker)
            for marker in ("Ã", "Â", "Ä", "Å", "Æ", "Ð", "Ñ", "â", "ð", "ƒ", "ï¿½")
        )

    for _ in range(max(1, int(max_passes))):
        before_markers = marker_count(current)
        if before_markers == 0 or "\ufffd" in current:
            break
        candidates: list[str] = []
        for encoding in ("cp1252", "latin-1"):
            try:
                candidate = current.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if "\ufffd" in candidate or "ï¿½" in candidate:
                continue
            if candidate != current and marker_count(candidate) < before_markers:
                candidates.append(candidate)
        if not candidates:
            # Preserve the original rather than applying a lossy or
            # ambiguous conversion to otherwise valid Unicode text.
            break
        current = min(candidates, key=marker_count)
    return current


def _strict_int(value: Any, field: str, *, context: str = "OCR row") -> int:
    """Parse an integer identity field without silently truncating floats."""

    if isinstance(value, bool):
        raise ValueError(f"{context}: {field} must be an integer")
    if isinstance(value, float):
        if not math.isfinite(value) or value != math.trunc(value):
            raise ValueError(f"{context}: {field} must be an integer")
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{context}: {field} must be an integer") from error


def _safe_relative_path(value: Any) -> bool:
    """Return whether a published path is relative and cannot traverse out.

    Metadata is exchanged between Linux containers and the Windows host, so
    validate both slash conventions rather than relying on the platform's
    ``Path`` parser alone.
    """

    text = str(value or "")
    normalized = text.replace("\\", "/")
    pure = PurePosixPath(normalized)
    return bool(
        text
        and "\x00" not in text
        and normalized not in {".", "./"}
        and not pure.is_absolute()
        and not (len(normalized) >= 2 and normalized[1] == ":")
        and ".." not in pure.parts
    )


def _ocr_query_score(
    bm25_relevance: float,
    token_coverage: float,
    adjacent_bigram_coverage: float,
) -> float:
    """Combine one OCR stream's normalized evidence using the locked weights."""

    return (
        0.55 * float(bm25_relevance)
        + 0.30 * float(token_coverage)
        + 0.15 * float(adjacent_bigram_coverage)
    )


def _canonical_json_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (
        int(getattr(stat, "st_dev", 0)),
        int(getattr(stat, "st_ino", 0)),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _artifact_record(
    path: Path,
    *,
    relative_to: Path,
    include_hash: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "path": path.resolve().relative_to(relative_to.resolve()).as_posix(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_hash:
        record["sha256"] = _sha256_file(path)
    record.update(extra)
    return record


def source_fingerprint(source_files: Iterable[dict[str, Any]]) -> str:
    records = []
    for item in source_files:
        if not isinstance(item, dict):
            raise ValueError("OCR source fingerprint records must be objects")
        records.append(
            {
                "path": str(item.get("path") or ""),
                "size": int(item.get("size", -1)),
                "sha256": str(item.get("sha256") or ""),
                "row_count": int(item.get("row_count", -1)),
                "video_id": str(item.get("video_id") or ""),
            }
        )
    return _canonical_json_digest(sorted(records, key=lambda value: value["path"]))


def _fts_content_fingerprint_for_connection(
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    """Verify FTS row identity/content and fingerprint the expected rows.

    This is deliberately a build-time/staging check.  Runtime does not scan
    all 247,956 FTS rows on every search; it relies on the validated database
    digest recorded in the v3 manifest and ``ocr_meta``.
    """

    previous_row_factory = connection.row_factory
    connection.row_factory = sqlite3.Row
    try:
        digest = hashlib.sha256()
        mismatch_count = 0
        rows = 0
        cursor = connection.execute(
            "SELECT f.id, f.frame_uid, f.full_text_search, "
            "x.rowid AS fts_rowid, x.full_text_search AS fts_text "
            "FROM ocr_frames AS f "
            "LEFT JOIN ocr_fts AS x ON x.rowid = f.id "
            "ORDER BY f.id"
        )
        for row in cursor:
            rows += 1
            expected = {
                "id": int(row["id"]),
                "frame_uid": str(row["frame_uid"] or ""),
                "full_text_search": str(row["full_text_search"] or ""),
            }
            encoded = json.dumps(
                expected,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(encoded)
            digest.update(b"\n")
            if (
                row["fts_rowid"] is None
                or int(row["fts_rowid"]) != expected["id"]
                or str(row["fts_text"] or "") != expected["full_text_search"]
            ):
                mismatch_count += 1
        orphaned = int(
            connection.execute(
                "SELECT COUNT(*) FROM ocr_fts AS x "
                "LEFT JOIN ocr_frames AS f ON f.id = x.rowid "
                "WHERE f.id IS NULL"
            ).fetchone()[0]
        )
        fts_count = int(connection.execute("SELECT COUNT(*) FROM ocr_fts").fetchone()[0])
        return {
            "fingerprint": digest.hexdigest(),
            "rows": rows,
            "fts_count": fts_count,
            "mismatch_count": mismatch_count,
            "orphaned": orphaned,
            "verified": bool(mismatch_count == 0 and orphaned == 0 and rows == fts_count),
        }
    finally:
        connection.row_factory = previous_row_factory


def build_id_for(
    *,
    source_fingerprint_value: str,
    canonical_fingerprint_value: str,
    frame_count: int,
    video_count: int,
    fts_content_fingerprint_value: str,
    lexical_contract_version: str = OCR_LEXICAL_CONTRACT_VERSION,
    mapping_strategy: str = OCR_MAPPING_STRATEGY,
) -> str:
    return _canonical_json_digest(
        {
            "schema_version": OCR_INDEX_SCHEMA_VERSION,
            "sqlite_user_version": OCR_SQLITE_USER_VERSION,
            "source_fingerprint": str(source_fingerprint_value),
            "canonical_fingerprint": str(canonical_fingerprint_value),
            "mapping_strategy": str(mapping_strategy),
            "frame_count": int(frame_count),
            "video_count": int(video_count),
            "fts_content_fingerprint": str(fts_content_fingerprint_value),
            "lexical_contract_version": str(lexical_contract_version),
        }
    )


def _manifest_build_identity_valid(manifest: dict[str, Any]) -> bool:
    """Validate the deterministic identity chain without touching artifacts."""

    if not isinstance(manifest, dict):
        return False
    try:
        counts = manifest.get("counts") or {}
        if not isinstance(counts, dict):
            return False
        source_fingerprint_value = str(manifest.get("source_fingerprint") or "")
        canonical_fingerprint_value = str(manifest.get("canonical_fingerprint") or "")
        fts_content_fingerprint_value = str(
            manifest.get("fts_content_fingerprint") or ""
        )
        lexical_contract_version = str(
            manifest.get("lexical_contract_version") or ""
        )
        frame_count = int(counts.get("frame_count", manifest.get("frame_count", -1)))
        fts_frame_count = int(counts.get("fts_frame_count", -1))
        mapped_frame_count = int(counts.get("mapped_frame_count", -1))
        video_count = int(counts.get("video_count", manifest.get("video_count", -1)))
        mapping = manifest.get("mapping") or {}
        if not isinstance(mapping, dict):
            return False
        mapping_strategy = str(mapping.get("strategy") or "")
        expected_build_id = str(manifest.get("build_id") or "")
        source_files = manifest.get("source_files")
        canonical = manifest.get("canonical_metadata") or {}
        database = manifest.get("database") or {}
        if not isinstance(canonical, dict) or not isinstance(database, dict):
            return False
        source_paths: list[str] = []
        source_video_ids: list[str] = []
        source_row_count = 0
        if not isinstance(source_files, list):
            return False
        for record in source_files:
            if not isinstance(record, dict):
                return False
            path = str(record.get("path") or "")
            path_parts = PurePosixPath(path).parts
            sha256 = str(record.get("sha256") or "")
            video_id = str(record.get("video_id") or "")
            try:
                record_size = _strict_int(record.get("size", -1), "size", context="OCR manifest")
                record_mtime_ns = _strict_int(
                    record.get("mtime_ns", -1),
                    "mtime_ns",
                    context="OCR manifest",
                )
                record_row_count = _strict_int(
                    record.get("row_count", 0),
                    "row_count",
                    context="OCR manifest",
                )
            except ValueError:
                return False
            if (
                not path.startswith(f"{OCR_SOURCE_DIR_NAME}/")
                or PurePosixPath(path).is_absolute()
                or ".." in path_parts
                or not path.endswith(".jsonl")
                or len(sha256) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in sha256)
                or record_size < 0
                or record_mtime_ns < 0
                or record_row_count <= 0
                or not video_id
                or PurePosixPath(path).stem.upper().replace("-", "_") != video_id
            ):
                return False
            source_paths.append(path)
            source_video_ids.append(video_id)
            source_row_count += record_row_count
        # The top-level source fingerprint is part of the manifest's
        # deterministic identity chain.  Merely checking that every record has
        # a non-empty SHA is insufficient: a damaged/hand-edited manifest could
        # otherwise pass the fast readiness path while its records disagree
        # with the aggregate used by ``build_id``.
        if source_fingerprint(source_files) != source_fingerprint_value:
            return False
        # The SQLite user version is part of the identity chain.  Do not
        # silently supply the current version when an older/incomplete
        # manifest omits it: an omitted field is indistinguishable from a
        # legacy manifest and must fail closed until the index is rebuilt.
        sqlite_user_version = manifest.get("sqlite_user_version")
        return bool(
            str(manifest.get("schema_version") or "") == OCR_INDEX_SCHEMA_VERSION
            and sqlite_user_version is not None
            and int(sqlite_user_version) == OCR_SQLITE_USER_VERSION
            and expected_build_id
            and source_fingerprint_value
            and canonical_fingerprint_value
            and len(fts_content_fingerprint_value) == 64
            and all(
                character in "0123456789abcdefABCDEF"
                for character in fts_content_fingerprint_value
            )
            and lexical_contract_version == OCR_LEXICAL_CONTRACT_VERSION
            and mapping_strategy == OCR_MAPPING_STRATEGY
            and int(mapping.get("frame_count", -1)) == OCR_EXPECTED_FRAMES
            and str(manifest.get("source_directory") or "") == OCR_SOURCE_DIR_NAME
            and frame_count == OCR_EXPECTED_FRAMES
            and fts_frame_count == OCR_EXPECTED_FRAMES
            and mapped_frame_count == OCR_EXPECTED_FRAMES
            and video_count == OCR_EXPECTED_SOURCE_FILES
            and source_row_count == OCR_EXPECTED_FRAMES
            and isinstance(source_files, list)
            and len(source_files) == OCR_EXPECTED_SOURCE_FILES
            and len(set(source_paths)) == OCR_EXPECTED_SOURCE_FILES
            and len(set(source_video_ids)) == OCR_EXPECTED_SOURCE_FILES
            and int(manifest.get("source_file_count", -1)) == OCR_EXPECTED_SOURCE_FILES
            and int(manifest.get("frame_count", -1)) == OCR_EXPECTED_FRAMES
            and int(manifest.get("video_count", -1)) == OCR_EXPECTED_SOURCE_FILES
            and str(database.get("path") or "") == "ocr.sqlite3"
            and int(database.get("size", -1)) >= 0
            and int(database.get("mtime_ns", -1)) >= 0
            and len(str(database.get("sha256") or "")) == 64
            and all(
                character in "0123456789abcdefABCDEF"
                for character in str(database.get("sha256") or "")
            )
            and str(canonical.get("path") or "") == OCR_CANONICAL_RELATIVE_PATH
            and int(canonical.get("size", -1)) >= 0
            and int(canonical.get("mtime_ns", -1)) >= 0
            and str(canonical.get("sha256") or "") == canonical_fingerprint_value
            and expected_build_id
            == build_id_for(
                source_fingerprint_value=source_fingerprint_value,
                canonical_fingerprint_value=canonical_fingerprint_value,
                frame_count=frame_count,
                video_count=video_count,
                fts_content_fingerprint_value=fts_content_fingerprint_value,
                lexical_contract_version=lexical_contract_version,
                mapping_strategy=mapping_strategy,
            )
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _load_canonical_frame_index(data_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load and validate canonical frame identity for preparation only."""

    path = data_root / OCR_CANONICAL_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Canonical frame metadata is missing: {path}")
    frames: dict[str, dict[str, Any]] = {}
    expected_point_id = 1
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Canonical metadata row {line_number} must be an object")
            frame_uid = str(item.get("frame_uid") or "")
            video_id = str(item.get("video_id") or "").upper().replace("-", "_")
            frame_idx = _strict_int(
                item.get("frame_idx", -1),
                "frame_idx",
                context=f"Canonical row {line_number}",
            )
            point_id = _strict_int(
                item.get("point_id", 0),
                "point_id",
                context=f"Canonical row {line_number}",
            )
            keyframe_n = _strict_int(
                item.get("keyframe_n", 0),
                "keyframe_n",
                context=f"Canonical row {line_number}",
            )
            pts_time_s = float(item.get("pts_time_s", 0.0))
            fps = float(item.get("fps", 0.0))
            image_relpath = str(item.get("image_relpath") or item.get("frame_relpath") or "")
            if not video_id or not frame_uid or frame_uid != f"{video_id}:{frame_idx}":
                raise ValueError(f"Canonical frame_uid mismatch at row {line_number}")
            if point_id != expected_point_id or frame_uid in frames:
                raise ValueError(f"Canonical point/frame ordering mismatch at row {line_number}")
            if (
                point_id < 1
                or frame_idx < 0
                or keyframe_n < 1
                or not _safe_relative_path(image_relpath)
                or not math.isfinite(pts_time_s)
                or pts_time_s < 0
                or not math.isfinite(fps)
                or fps <= 0
            ):
                raise ValueError(f"Invalid canonical frame row {line_number}")
            frames[frame_uid] = {
                "frame_uid": frame_uid,
                "point_id": point_id,
                "video_id": video_id,
                "frame_idx": frame_idx,
                "keyframe_n": keyframe_n,
                "pts_time_s": pts_time_s,
                "fps": fps,
                "image_relpath": image_relpath,
            }
            expected_point_id += 1
    if len(frames) != OCR_EXPECTED_FRAMES:
        raise ValueError(f"Expected {OCR_EXPECTED_FRAMES} canonical frames, found {len(frames)}")
    return frames, _artifact_record(path, relative_to=data_root)


def _parse_ocr_row(item: dict[str, Any], path: Path, line_number: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"{path}:{line_number}: OCR row must be an object")
    required = ("video_id", "frame_uid", "keyframe_n", "frame_idx", "pts_time_s", "full_text")
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(f"{path}:{line_number}: missing OCR fields {missing}")
    video_id = str(item.get("video_id") or "").upper().replace("-", "_").strip()
    frame_uid = str(item.get("frame_uid") or "").strip()
    frame_idx = _strict_int(item.get("frame_idx", -1), "frame_idx", context=f"{path}:{line_number}")
    keyframe_n = _strict_int(item.get("keyframe_n", 0), "keyframe_n", context=f"{path}:{line_number}")
    pts_time_s = float(item.get("pts_time_s", 0.0))
    if not video_id or frame_uid != f"{video_id}:{frame_idx}":
        raise ValueError(f"{path}:{line_number}: invalid video/frame identity")
    if keyframe_n < 1 or frame_idx < 0 or not math.isfinite(pts_time_s) or pts_time_s < 0:
        raise ValueError(f"{path}:{line_number}: invalid OCR frame metadata")
    raw_full_text = item.get("full_text")
    if not isinstance(raw_full_text, str):
        raise ValueError(f"{path}:{line_number}: full_text must be a string")
    full_text = repair_mojibake(raw_full_text)
    if "\ufffd" in full_text or "ï¿½" in full_text:
        raise ValueError(f"{path}:{line_number}: full_text contains a replacement character")
    return {
        "video_id": video_id,
        "frame_uid": frame_uid,
        "keyframe_n": keyframe_n,
        "frame_idx": frame_idx,
        "pts_time_s": pts_time_s,
        "full_text": full_text,
        "full_text_search": fold_text(full_text),
    }


def validate_ocr_sources(transcripts_dir: Path) -> dict[str, Any]:
    """Validate source file identity/counts and return preparation facts."""

    if not transcripts_dir.is_dir():
        raise FileNotFoundError(f"OCR transcript directory is missing: {transcripts_dir}")
    paths = sorted(transcripts_dir.glob("*.jsonl"))
    if len(paths) != OCR_EXPECTED_SOURCE_FILES:
        raise ValueError(
            f"Expected {OCR_EXPECTED_SOURCE_FILES} OCR source files, found {len(paths)}"
        )
    frame_uids: set[str] = set()
    video_ids: set[str] = set()
    source_files: list[dict[str, Any]] = []
    total = 0
    data_root = transcripts_dir.parent
    for path in paths:
        expected_video_id = path.stem.upper().replace("-", "_")
        rows = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                parsed = _parse_ocr_row(json.loads(line), path, line_number)
                if parsed["video_id"] != expected_video_id:
                    raise ValueError(f"{path}:{line_number}: video_id does not match filename")
                if parsed["frame_uid"] in frame_uids:
                    raise ValueError(f"Duplicate OCR frame_uid: {parsed['frame_uid']}")
                frame_uids.add(parsed["frame_uid"])
                video_ids.add(parsed["video_id"])
                rows += 1
        if rows == 0:
            raise ValueError(f"OCR source file is empty: {path}")
        source_files.append(
            _artifact_record(
                path,
                relative_to=data_root,
                row_count=rows,
                video_id=expected_video_id,
            )
        )
        total += rows
    if total != OCR_EXPECTED_FRAMES:
        raise ValueError(f"Expected {OCR_EXPECTED_FRAMES} OCR rows, found {total}")
    if len(video_ids) != OCR_EXPECTED_SOURCE_FILES:
        raise ValueError(
            f"Expected {OCR_EXPECTED_SOURCE_FILES} OCR videos, found {len(video_ids)}"
        )
    return {
        "source_files": source_files,
        "source_file_count": len(source_files),
        "frame_count": total,
        "video_count": len(video_ids),
        "video_ids": sorted(video_ids),
        "source_fingerprint": source_fingerprint(source_files),
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.staging")
    _write_json_file(staging, payload)
    os.replace(staging, path)


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    """Write one complete JSON artifact without publishing it.

    Preparation uses this helper for the manifest staging path so that the
    database and its manifest can be published as a pair.  Callers must use
    ``os.replace`` only after the payload has passed validation.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_ocr_index(
    transcripts_dir: Path,
    database_path: Path,
    *,
    data_root: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Build and atomically publish one OCR SQLite/manifest pair."""

    transcripts_dir = transcripts_dir.resolve()
    data_root = (data_root or transcripts_dir.parent).resolve()
    database_path = database_path.resolve()
    manifest_path = (manifest_path or database_path.with_name("branch3_ocr_manifest.json")).resolve()
    manifest_staging = manifest_path.with_name(f".{manifest_path.name}.staging")
    source_facts = validate_ocr_sources(transcripts_dir)
    canonical_frames, canonical_record = _load_canonical_frame_index(data_root)
    canonical_fingerprint = str(canonical_record["sha256"])
    database_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = database_path.with_name(
        f".{database_path.name}.staging.{os.getpid()}"
    )
    try:
        staging_path.unlink(missing_ok=True)
        manifest_staging.unlink(missing_ok=True)
    except OSError:
        pass
    try:
        connection = sqlite3.connect(staging_path)
    except Exception:
        try:
            staging_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            manifest_staging.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            connection.execute(f"PRAGMA user_version = {OCR_SQLITE_USER_VERSION}")
            connection.execute(
                """
                CREATE TABLE ocr_frames (
                    id INTEGER PRIMARY KEY,
                    frame_uid TEXT NOT NULL UNIQUE,
                    point_id INTEGER NOT NULL UNIQUE,
                    video_id TEXT NOT NULL,
                    keyframe_n INTEGER NOT NULL,
                    frame_idx INTEGER NOT NULL,
                    pts_time_s REAL NOT NULL,
                    fps REAL NOT NULL,
                    image_relpath TEXT NOT NULL,
                    full_text TEXT NOT NULL,
                    full_text_search TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX ocr_video_idx ON ocr_frames(video_id)")
            connection.execute("CREATE INDEX ocr_frame_uid_idx ON ocr_frames(frame_uid)")
            connection.execute(
                # Keep a regular FTS5 content table rather than an external
                # content table.  OCR rows with empty text are valid and must
                # still be represented in the FTS row count/rowid mapping;
                # external-content indexes may omit empty documents from
                # their visible row set.
                "CREATE VIRTUAL TABLE ocr_fts USING fts5(full_text_search, tokenize='unicode61 remove_diacritics 2')"
            )
            insert_sql = (
                "INSERT INTO ocr_frames(frame_uid, point_id, video_id, keyframe_n, frame_idx, "
                "pts_time_s, fps, image_relpath, full_text, full_text_search) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            rows: list[tuple[Any, ...]] = []
            inserted = 0
            observed_uids: set[str] = set()
            for path in sorted(transcripts_dir.glob("*.jsonl")):
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        parsed = _parse_ocr_row(json.loads(line), path, line_number)
                        frame = canonical_frames.get(parsed["frame_uid"])
                        if frame is None:
                            raise ValueError(
                                f"{path}:{line_number}: OCR frame is missing from canonical metadata"
                            )
                        if (
                            parsed["video_id"] != frame["video_id"]
                            or parsed["frame_idx"] != frame["frame_idx"]
                            or parsed["keyframe_n"] != frame["keyframe_n"]
                        ):
                            raise ValueError(
                                f"{path}:{line_number}: OCR/canonical identity mismatch"
                            )
                        if parsed["frame_uid"] in observed_uids:
                            raise ValueError(f"Duplicate OCR frame_uid: {parsed['frame_uid']}")
                        observed_uids.add(parsed["frame_uid"])
                        rows.append(
                            (
                                frame["frame_uid"],
                                int(frame["point_id"]),
                                frame["video_id"],
                                int(frame["keyframe_n"]),
                                int(frame["frame_idx"]),
                                float(frame["pts_time_s"]),
                                float(frame["fps"]),
                                frame["image_relpath"],
                                parsed["full_text"],
                                parsed["full_text_search"],
                            )
                        )
                        if len(rows) >= 2_000:
                            connection.executemany(insert_sql, rows)
                            inserted += len(rows)
                            rows.clear()
            if rows:
                connection.executemany(insert_sql, rows)
                inserted += len(rows)
            if inserted != OCR_EXPECTED_FRAMES or len(observed_uids) != OCR_EXPECTED_FRAMES:
                raise ValueError(
                    f"Expected {OCR_EXPECTED_FRAMES} OCR rows, found {inserted}"
                )
            if set(canonical_frames) != observed_uids:
                raise ValueError("OCR/canonical frame coverage mismatch")
            connection.execute(
                "INSERT INTO ocr_fts(rowid, full_text_search) "
                "SELECT id, full_text_search FROM ocr_frames ORDER BY id"
            )
            fts_state = _fts_content_fingerprint_for_connection(connection)
            if fts_state.get("verified") is not True:
                raise ValueError(
                    "OCR staging FTS content validation failed: "
                    f"{fts_state}"
                )
            fts_content_fingerprint = str(fts_state["fingerprint"])
            build_id = build_id_for(
                source_fingerprint_value=source_facts["source_fingerprint"],
                canonical_fingerprint_value=canonical_fingerprint,
                frame_count=source_facts["frame_count"],
                video_count=source_facts["video_count"],
                fts_content_fingerprint_value=fts_content_fingerprint,
            )
            connection.execute(
                """
                CREATE TABLE ocr_meta (
                    schema_version TEXT NOT NULL,
                    sqlite_user_version INTEGER NOT NULL,
                    build_id TEXT NOT NULL UNIQUE,
                    source_fingerprint TEXT NOT NULL,
                    canonical_fingerprint TEXT NOT NULL,
                    mapping_strategy TEXT NOT NULL,
                    frame_count INTEGER NOT NULL,
                    video_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    fts_content_fingerprint TEXT NOT NULL,
                    lexical_contract_version TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "INSERT INTO ocr_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    OCR_INDEX_SCHEMA_VERSION,
                    OCR_SQLITE_USER_VERSION,
                    build_id,
                    source_facts["source_fingerprint"],
                    canonical_fingerprint,
                    OCR_MAPPING_STRATEGY,
                    OCR_EXPECTED_FRAMES,
                    OCR_EXPECTED_SOURCE_FILES,
                    datetime.now(timezone.utc).isoformat(),
                    fts_content_fingerprint,
                    OCR_LEXICAL_CONTRACT_VERSION,
                ),
            )
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            if integrity != "ok":
                raise ValueError(f"OCR staging integrity check failed: {integrity}")
            db_count = int(connection.execute("SELECT COUNT(*) FROM ocr_frames").fetchone()[0])
            fts_count = int(connection.execute("SELECT COUNT(*) FROM ocr_fts").fetchone()[0])
            if db_count != OCR_EXPECTED_FRAMES or fts_count != OCR_EXPECTED_FRAMES:
                raise ValueError("OCR staging frame/FTS count mismatch")
            staging_state = _database_state_for_connection(connection)
            staging_meta = staging_state.get("meta") or {}
            # Reuse the runtime's streaming canonical-identity inspector for
            # the staging file as well.  This catches a malformed write even
            # when row counts, FTS rowids and SQLite constraints still look
            # correct, before the live pair is replaced.
            staging_probe = object.__new__(OcrFtsIndex)
            staging_probe.canonical_metadata_path = data_root / OCR_CANONICAL_RELATIVE_PATH
            staging_canonical_state = staging_probe._canonical_identity_state_for_connection(
                connection
            )
            staging_identity_matches = (
                staging_meta.get("schema_version") == OCR_INDEX_SCHEMA_VERSION
                and staging_meta.get("sqlite_user_version") == OCR_SQLITE_USER_VERSION
                and staging_meta.get("build_id") == build_id
                and staging_meta.get("source_fingerprint") == source_facts["source_fingerprint"]
                and staging_meta.get("canonical_fingerprint") == canonical_fingerprint
                and staging_meta.get("mapping_strategy") == OCR_MAPPING_STRATEGY
                and int(staging_meta.get("frame_count", -1)) == OCR_EXPECTED_FRAMES
                and int(staging_meta.get("video_count", -1)) == OCR_EXPECTED_SOURCE_FILES
                and staging_meta.get("fts_content_fingerprint") == fts_content_fingerprint
                and staging_meta.get("lexical_contract_version") == OCR_LEXICAL_CONTRACT_VERSION
            )
            if (
                staging_state.get("ready") is not True
                or not staging_identity_matches
                or staging_canonical_state.get("ready") is not True
            ):
                raise ValueError(
                    "OCR staging database validation failed: "
                    f"{staging_state.get('diagnostics') or staging_canonical_state or staging_state}"
                )
    except Exception:
        # A failed staging build must not leave a database that a later run
        # could accidentally publish.  Close before unlinking so this is safe
        # on Windows as well as POSIX.
        try:
            connection.close()
        except sqlite3.Error:
            pass
        try:
            staging_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        connection.close()

    # Refuse to publish an index assembled from a moving source tree.  The
    # source/canonical fingerprints were captured before the build; checking
    # publication stats here prevents a newer artifact from being paired with
    # the staged rows and leaves the live database/manifest untouched.
    try:
        canonical_after = _artifact_record(
            data_root / OCR_CANONICAL_RELATIVE_PATH,
            relative_to=data_root,
            include_hash=False,
        )
        if any(
            int(canonical_after.get(field, -1)) != int(canonical_record.get(field, -2))
            for field in ("size", "mtime_ns")
        ):
            raise RuntimeError("Canonical OCR metadata changed during preparation")
        for source_record in source_facts["source_files"]:
            source_path = data_root / str(source_record.get("path") or "")
            source_stat = source_path.stat()
            if (
                int(source_record.get("size", -1)) != int(source_stat.st_size)
                or int(source_record.get("mtime_ns", -1)) != int(source_stat.st_mtime_ns)
            ):
                raise RuntimeError(f"OCR source changed during preparation: {source_path}")
    except Exception:
        try:
            staging_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            manifest_staging.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    # Prepare the manifest before replacing the live database.  The record is
    # named after the eventual database path, while size/mtime/hash are taken
    # from the validated staging file (rename preserves those values on the
    # supported filesystems).  If a reader observes the database replacement
    # before the manifest replacement it will fail closed on build_id/stat
    # mismatch rather than querying a mixed pair.
    try:
        database_record = _artifact_record(staging_path, relative_to=database_path.parent)
    except Exception:
        try:
            staging_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    database_record["path"] = database_path.name
    manifest = {
        "schema_version": OCR_INDEX_SCHEMA_VERSION,
        "sqlite_user_version": OCR_SQLITE_USER_VERSION,
        "status": "ready",
        "passed": True,
        "database": database_record,
        "source_directory": OCR_SOURCE_DIR_NAME,
        "source_files": source_facts["source_files"],
        "source_file_count": source_facts["source_file_count"],
        "source_fingerprint": source_facts["source_fingerprint"],
        "fts_content_fingerprint": fts_content_fingerprint,
        "lexical_contract_version": OCR_LEXICAL_CONTRACT_VERSION,
        "canonical_metadata": canonical_record,
        "canonical_fingerprint": canonical_fingerprint,
        "mapping": {
            "strategy": OCR_MAPPING_STRATEGY,
            "frame_count": OCR_EXPECTED_FRAMES,
        },
        "counts": {
            "frame_count": OCR_EXPECTED_FRAMES,
            "fts_frame_count": OCR_EXPECTED_FRAMES,
            "mapped_frame_count": OCR_EXPECTED_FRAMES,
            "video_count": OCR_EXPECTED_SOURCE_FILES,
        },
        "frame_count": OCR_EXPECTED_FRAMES,
        "video_count": OCR_EXPECTED_SOURCE_FILES,
        "offline_identity": {
            "model_id": None,
            "engine": "OCR source export",
            "revision_verified": False,
            "evidence": "OCR source files do not record an immutable checkpoint revision",
        },
        "revision_verified": False,
        "production_ready": False,
        "build_id": build_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _write_json_file(manifest_staging, manifest)
        os.replace(staging_path, database_path)

        # Re-stat the published artifact before the manifest becomes visible.
        # This also covers filesystems that update metadata during replace.
        published_record = _artifact_record(
            database_path,
            relative_to=database_path.parent,
        )
        published_record["path"] = database_path.name
        manifest["database"] = published_record
        _write_json_file(manifest_staging, manifest)
        os.replace(manifest_staging, manifest_path)
    except Exception:
        try:
            manifest_staging.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            staging_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return manifest


class OcrFtsIndex:
    """Read-only OCR runtime with manifest-bound connection refresh."""

    def __init__(
        self,
        transcripts_dir: Path,
        database_path: Path,
        metadata: Any = None,
        *,
        manifest_path: Path | None = None,
        canonical_metadata_path: Path | None = None,
        auto_prepare: bool = False,
    ) -> None:
        if auto_prepare:
            raise ValueError("OCR runtime cannot prepare indexes; use build_ocr_index()")
        self.transcripts_dir = transcripts_dir.resolve()
        self.data_root = self.transcripts_dir.parent
        self.database_path = database_path.resolve()
        self.manifest_path = (
            manifest_path or self.database_path.with_name("branch3_ocr_manifest.json")
        ).resolve()
        self.canonical_metadata_path = (
            canonical_metadata_path
            or self.data_root / OCR_CANONICAL_RELATIVE_PATH
        ).resolve()
        self.metadata = metadata
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        self._connection_path: Path | None = None
        self._runtime_snapshot_path: Path | None = None
        self._opened_file_identity: tuple[int, int, int, int] | None = None
        self._opened_build_id: str | None = None
        self._connection_generation = 0
        self._connection_reopened = False
        self._connection_error: str | None = None
        self._validated_connection_generation = 0
        self._validated_build_id: str | None = None
        self._validated_database_identity: tuple[int, int, int, int] | None = None
        self._validated_database_state: dict[str, Any] | None = None
        # The database inspector validates SQLite/FTS structure.  This second
        # snapshot is populated only when a new connection generation is
        # opened and streams the complete database against canonical metadata
        # so a same-shaped but wrong frame identity can never be published.
        self._validated_canonical_state: dict[str, Any] | None = None
        self._health_cache: tuple[float, tuple[Any, ...], dict[str, Any]] | None = None
        self._source_inventory_cache: tuple[float, tuple[Any, ...], dict[str, Any]] | None = None
        self._source_audit_cache: tuple[
            float,
            tuple[Any, ...],
            bool,
            bool,
            list[dict[str, Any]],
        ] | None = None
        self._artifact_hash_cache: dict[str, tuple[tuple[int, int, int], str]] = {}

    def close(self) -> None:
        with self._lock:
            self._close_connection_locked()
            self._health_cache = None
            self._source_inventory_cache = None
            self._source_audit_cache = None
            self._artifact_hash_cache.clear()
            self._opened_file_identity = None
            self._opened_build_id = None
            self._validated_connection_generation = 0
            self._validated_build_id = None
            self._validated_database_identity = None
            self._validated_database_state = None
            self._validated_canonical_state = None

    @property
    def ready(self) -> bool:
        return bool(self.health().get("ready"))

    def _close_connection_locked(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except sqlite3.Error:
                pass
        self._connection = None
        self._connection_path = None
        if self._runtime_snapshot_path is not None:
            try:
                self._runtime_snapshot_path.unlink(missing_ok=True)
            except OSError:
                pass
        self._runtime_snapshot_path = None
        self._opened_file_identity = None
        self._opened_build_id = None
        self._validated_connection_generation = 0
        self._validated_build_id = None
        self._validated_database_identity = None
        self._validated_database_state = None
        self._validated_canonical_state = None

    def _load_manifest_locked(self) -> dict[str, Any]:
        value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("OCR manifest must be a JSON object")
        return value

    @staticmethod
    def _read_meta(connection: sqlite3.Connection) -> dict[str, Any]:
        row = connection.execute(
            "SELECT schema_version, sqlite_user_version, build_id, source_fingerprint, "
            "canonical_fingerprint, mapping_strategy, frame_count, video_count, "
            "fts_content_fingerprint, lexical_contract_version "
            "FROM ocr_meta LIMIT 1"
        ).fetchone()
        if row is None:
            raise ValueError("OCR SQLite ocr_meta row is missing")
        return {
            "schema_version": str(row[0]),
            "sqlite_user_version": int(row[1]),
            "build_id": str(row[2]),
            "source_fingerprint": str(row[3]),
            "canonical_fingerprint": str(row[4]),
            "mapping_strategy": str(row[5]),
            "frame_count": int(row[6]),
            "video_count": int(row[7]),
            "fts_content_fingerprint": str(row[8]),
            "lexical_contract_version": str(row[9]),
        }

    def _canonical_identity_state_for_connection(
        self,
        connection: sqlite3.Connection,
    ) -> dict[str, Any]:
        """Compare every published OCR row with canonical frame metadata.

        The SQLite schema checks that identity columns are well formed, but a
        corrupted database could still contain a *different* valid frame with
        the same shape/counts.  This streaming comparison is therefore part
        of connection validation.  It runs once per newly opened database
        generation and keeps only compact diagnostics in memory.
        """

        result: dict[str, Any] = {
            "ready": False,
            "canonical_rows": 0,
            "database_rows": 0,
            "mismatch_count": 0,
            "mismatches": [],
            "path": str(self.canonical_metadata_path),
        }
        try:
            if not self.canonical_metadata_path.is_file():
                result["error"] = "canonical metadata file is missing"
                return result
            cursor = connection.execute(
                "SELECT frame_uid, point_id, video_id, keyframe_n, frame_idx, "
                "pts_time_s, fps, image_relpath FROM ocr_frames ORDER BY point_id"
            )
            database_row = cursor.fetchone()
            database_row_count = 0
            canonical_rows = 0
            mismatch_samples: list[dict[str, Any]] = []
            expected_point_id = 1
            with self.canonical_metadata_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    canonical_item = json.loads(line)
                    if not isinstance(canonical_item, dict):
                        raise ValueError(
                            f"canonical metadata row {line_number} must be an object"
                        )
                    video_id = str(canonical_item.get("video_id") or "").upper().replace(
                        "-", "_"
                    )
                    frame_idx = _strict_int(
                        canonical_item.get("frame_idx", -1),
                        "frame_idx",
                        context=f"Canonical row {line_number}",
                    )
                    point_id = _strict_int(
                        canonical_item.get("point_id", 0),
                        "point_id",
                        context=f"Canonical row {line_number}",
                    )
                    keyframe_n = _strict_int(
                        canonical_item.get("keyframe_n", 0),
                        "keyframe_n",
                        context=f"Canonical row {line_number}",
                    )
                    pts_time_s = float(canonical_item.get("pts_time_s", 0.0))
                    fps = float(canonical_item.get("fps", 0.0))
                    image_relpath = str(
                        canonical_item.get("image_relpath")
                        or canonical_item.get("frame_relpath")
                        or ""
                    )
                    frame_uid = str(canonical_item.get("frame_uid") or "")
                    if (
                        not video_id
                        or not frame_uid
                        or frame_uid != f"{video_id}:{frame_idx}"
                        or point_id != expected_point_id
                        or frame_idx < 0
                        or keyframe_n < 1
                        or not _safe_relative_path(image_relpath)
                        or not math.isfinite(pts_time_s)
                        or pts_time_s < 0
                        or not math.isfinite(fps)
                        or fps <= 0
                    ):
                        raise ValueError(
                            f"invalid canonical identity at row {line_number}"
                        )
                    canonical_rows += 1
                    expected = (
                        frame_uid,
                        point_id,
                        video_id,
                        keyframe_n,
                        frame_idx,
                        pts_time_s,
                        fps,
                        image_relpath,
                    )
                    actual = tuple(database_row) if database_row is not None else None
                    if database_row is not None:
                        database_row_count += 1
                    matches = bool(
                        actual is not None
                        and actual[0] == expected[0]
                        and int(actual[1]) == expected[1]
                        and actual[2] == expected[2]
                        and int(actual[3]) == expected[3]
                        and int(actual[4]) == expected[4]
                        and math.isclose(float(actual[5]), expected[5], rel_tol=0.0, abs_tol=1e-9)
                        and math.isclose(float(actual[6]), expected[6], rel_tol=0.0, abs_tol=1e-9)
                        and actual[7] == expected[7]
                    )
                    if not matches:
                        result["mismatch_count"] += 1
                        if len(mismatch_samples) < 8:
                            mismatch_samples.append(
                                {
                                    "line": line_number,
                                    "frame_uid": frame_uid,
                                    "expected_point_id": point_id,
                                    "actual": list(actual) if actual is not None else None,
                                }
                            )
                    if database_row is not None:
                        database_row = cursor.fetchone()
                    expected_point_id += 1
            while database_row is not None:
                result["mismatch_count"] += 1
                database_row_count += 1
                if len(mismatch_samples) < 8:
                    mismatch_samples.append(
                        {
                            "line": None,
                            "frame_uid": str(database_row[0] or ""),
                            "reason": "database has extra row",
                        }
                    )
                database_row = cursor.fetchone()
            result["canonical_rows"] = canonical_rows
            result["database_rows"] = database_row_count
            result["mismatches"] = mismatch_samples
            result["ready"] = bool(
                canonical_rows == OCR_EXPECTED_FRAMES
                and result["mismatch_count"] == 0
                and result["database_rows"] == OCR_EXPECTED_FRAMES
            )
            if not result["ready"]:
                result["error"] = "canonical metadata and OCR database identity mismatch"
            return result
        except (OSError, json.JSONDecodeError, sqlite3.Error, TypeError, ValueError, OverflowError) as error:
            result["error"] = str(error)
            result["mismatches"] = mismatch_samples if "mismatch_samples" in locals() else []
            result["mismatch_count"] = max(
                int(result.get("mismatch_count", 0)),
                1,
            )
            return result


    def _refresh_connection_locked(self, manifest: dict[str, Any]) -> dict[str, Any]:
        file_identity = _file_identity(self.database_path)
        expected_build_id = str(manifest.get("build_id") or "")
        previous_file_identity = self._opened_file_identity
        needs_open = (
            self._connection is None
            or file_identity is None
            or file_identity != self._opened_file_identity
            or (expected_build_id and expected_build_id != self._opened_build_id)
            or self._validated_database_identity != file_identity
            or self._validated_build_id != self._opened_build_id
        )
        self._connection_reopened = False
        if file_identity is None:
            self._close_connection_locked()
            self._connection_error = "OCR database file is missing"
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
            database_record = manifest.get("database") or {}
            database_artifact = self._artifact_status_locked(
                database_record,
                root=self.database_path.parent,
                label="database",
                # Establish the database digest on initial open and after any
                # identity change.  Subsequent requests reuse the cached hash.
                verify_hash=previous_file_identity is None
                or file_identity != previous_file_identity,
            )
            if database_artifact.get("fingerprint_matches") is not True:
                raise ValueError(
                    "OCR database fingerprint does not match manifest: "
                    f"{database_artifact.get('error') or database_artifact}"
                )
            canonical_record = manifest.get("canonical_metadata") or {}
            canonical_artifact = self._artifact_status_locked(
                canonical_record,
                root=self.data_root,
                label="canonical_metadata",
                # The canonical file is also hashed once per observed file
                # identity; the artifact cache prevents repeated hashing.
                verify_hash=True,
            )
            if canonical_artifact.get("fingerprint_matches") is not True:
                raise ValueError(
                    "OCR canonical metadata fingerprint does not match manifest: "
                    f"{canonical_artifact.get('error') or canonical_artifact}"
                )
            open_path = self.database_path
            if os.name == "nt":
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
                f"file:{open_path.as_posix()}?mode=ro&immutable=1",
                uri=True,
                check_same_thread=False,
            )
            connection.row_factory = sqlite3.Row
            meta = self._read_meta(connection)
            user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if meta["schema_version"] != OCR_INDEX_SCHEMA_VERSION:
                raise ValueError("OCR SQLite schema version is unsupported")
            if user_version != OCR_SQLITE_USER_VERSION or meta["sqlite_user_version"] != OCR_SQLITE_USER_VERSION:
                raise ValueError("OCR SQLite user version is unsupported")
            if not meta["build_id"] or (expected_build_id and meta["build_id"] != expected_build_id):
                raise ValueError("OCR SQLite build_id does not match manifest")
            if meta["source_fingerprint"] != str(manifest.get("source_fingerprint") or ""):
                raise ValueError("OCR SQLite source fingerprint does not match manifest")
            if meta["canonical_fingerprint"] != str(manifest.get("canonical_fingerprint") or ""):
                raise ValueError("OCR SQLite canonical fingerprint does not match manifest")
            if meta["fts_content_fingerprint"] != str(
                manifest.get("fts_content_fingerprint") or ""
            ):
                raise ValueError("OCR SQLite FTS content fingerprint does not match manifest")
            if meta["lexical_contract_version"] != OCR_LEXICAL_CONTRACT_VERSION:
                raise ValueError("OCR SQLite lexical contract is unsupported")
            database_state = _database_state_for_connection(connection)
            if database_state.get("ready") is not True:
                raise ValueError(
                    "OCR SQLite database validation failed: "
                    f"{database_state.get('diagnostics') or database_state}"
                )
            canonical_state = self._canonical_identity_state_for_connection(connection)
            if canonical_state.get("ready") is not True:
                raise ValueError(
                    "OCR canonical identity validation failed: "
                    f"{canonical_state.get('error') or canonical_state}"
                )
            database_state["canonical_identity"] = canonical_state
            self._connection = connection
            self._connection_path = open_path
            self._runtime_snapshot_path = snapshot_path
            self._opened_file_identity = file_identity
            self._opened_build_id = meta["build_id"]
            self._connection_generation += 1
            self._connection_reopened = True
            self._connection_error = None
            self._validated_connection_generation = self._connection_generation
            self._validated_build_id = meta["build_id"]
            self._validated_database_identity = file_identity
            self._validated_database_state = dict(database_state)
            self._validated_canonical_state = dict(canonical_state)
        except (OSError, sqlite3.Error, ValueError, TypeError) as error:
            try:
                if connection is not None:
                    connection.close()
            except sqlite3.Error:
                pass
            if snapshot_path is not None:
                try:
                    snapshot_path.unlink(missing_ok=True)
                except OSError:
                    pass
            self._close_connection_locked()
            self._connection_error = str(error)
        return {
            "connection_generation": self._connection_generation,
            "connection_reopened": self._connection_reopened,
            "opened_build_id": self._opened_build_id,
            "connection_error": self._connection_error,
        }

    def _database_state_locked(self) -> dict[str, Any]:
        if (
            self._validated_database_state is not None
            and self._validated_connection_generation == self._connection_generation
            and self._validated_database_identity == self._opened_file_identity
            and self._validated_build_id == self._opened_build_id
        ):
            return dict(self._validated_database_state)
        if self._connection is None:
            return {
                "ready": False,
                "integrity": "missing",
                "frames": 0,
                "fts_frames": 0,
                "mapped_frames": 0,
                "fts_orphaned": 0,
                "invalid_paths": 0,
                "meta_rows": 0,
                "videos": 0,
                "meta": {},
                "tables": [],
            }
        try:
            tables = {
                str(row[0])
                for row in self._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                ).fetchall()
            }
            required = {"ocr_frames", "ocr_fts", "ocr_meta"}
            columns = {
                str(row[1])
                for row in self._connection.execute("PRAGMA table_info(ocr_frames)").fetchall()
            }
            fts_columns = {
                str(row[1])
                for row in self._connection.execute("PRAGMA table_info(ocr_fts)").fetchall()
            }
            row = self._connection.execute(
                "SELECT COUNT(*) AS frames, COUNT(DISTINCT video_id) AS videos, "
                "SUM(CASE WHEN frame_uid <> '' AND point_id > 0 AND fps > 0 "
                "AND image_relpath <> '' THEN 1 ELSE 0 END) AS mapped FROM ocr_frames"
            ).fetchone()
            frames = int(row["frames"] or 0) if row else 0
            videos = int(row["videos"] or 0) if row else 0
            mapped = int(row["mapped"] or 0) if row else 0
            point_stats = self._connection.execute(
                "SELECT COUNT(DISTINCT point_id) AS distinct_points, "
                "MIN(point_id) AS min_point_id, MAX(point_id) AS max_point_id "
                "FROM ocr_frames"
            ).fetchone()
            distinct_points = int(point_stats["distinct_points"] or 0) if point_stats else 0
            min_point_id = int(point_stats["min_point_id"] or 0) if point_stats else 0
            max_point_id = int(point_stats["max_point_id"] or 0) if point_stats else 0
            invalid_identity = int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM ocr_frames WHERE "
                    "video_id IS NULL OR video_id = '' OR frame_uid IS NULL OR frame_uid = '' "
                    "OR frame_uid <> video_id || ':' || frame_idx "
                    "OR point_id IS NULL OR point_id < 1 "
                    "OR point_id != CAST(point_id AS INTEGER) "
                    "OR keyframe_n IS NULL OR keyframe_n < 1 "
                    "OR keyframe_n != CAST(keyframe_n AS INTEGER) "
                    "OR frame_idx IS NULL OR frame_idx < 0 "
                    "OR frame_idx != CAST(frame_idx AS INTEGER) "
                    "OR pts_time_s IS NULL OR pts_time_s != pts_time_s "
                    "OR NOT (pts_time_s >= 0 AND pts_time_s < 1.0e308) "
                    "OR fps IS NULL OR fps != fps OR NOT (fps > 0 AND fps < 1.0e308) "
                    "OR image_relpath IS NULL OR image_relpath = '' "
                ).fetchone()[0]
            )
            invalid_paths = int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM ocr_frames WHERE "
                    "image_relpath = '' OR image_relpath IN ('.', './') "
                    "OR substr(image_relpath, 1, 1) IN ('/', char(92)) "
                    "OR substr(image_relpath, 2, 1) = ':' "
                    "OR image_relpath LIKE '%..%'"
                ).fetchone()[0]
            )
            fts_frames = int(self._connection.execute("SELECT COUNT(*) FROM ocr_fts").fetchone()[0])
            fts_unmapped = int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM ocr_frames AS f "
                    "LEFT JOIN ocr_fts AS x ON x.rowid = f.id "
                    "WHERE x.rowid IS NULL"
                ).fetchone()[0]
            )
            fts_orphaned = int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM ocr_fts AS x "
                    "LEFT JOIN ocr_frames AS f ON f.id = x.rowid "
                    "WHERE f.id IS NULL"
                ).fetchone()[0]
            )
            meta_rows = int(
                self._connection.execute("SELECT COUNT(*) FROM ocr_meta").fetchone()[0]
            ) if "ocr_meta" in tables else 0
            user_version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
            meta = self._read_meta(self._connection) if "ocr_meta" in tables else {}
            expected_columns = {
                "id", "frame_uid", "point_id", "video_id", "keyframe_n", "frame_idx",
                "pts_time_s", "fps", "image_relpath", "full_text", "full_text_search",
            }
            integrity = str(self._connection.execute("PRAGMA integrity_check").fetchone()[0])
            ready = bool(
                integrity == "ok"
                and
                required.issubset(tables)
                and expected_columns.issubset(columns)
                and "full_text_search" in fts_columns
                and meta_rows == 1
                and user_version == OCR_SQLITE_USER_VERSION
                and meta.get("schema_version") == OCR_INDEX_SCHEMA_VERSION
                and int(meta.get("sqlite_user_version", -1)) == OCR_SQLITE_USER_VERSION
                and int(meta.get("frame_count", -1)) == frames == OCR_EXPECTED_FRAMES
                and int(meta.get("video_count", -1)) == videos == OCR_EXPECTED_SOURCE_FILES
                and meta.get("mapping_strategy") == OCR_MAPPING_STRATEGY
                and len(str(meta.get("fts_content_fingerprint") or "")) == 64
                and all(
                    character in "0123456789abcdefABCDEF"
                    for character in str(meta.get("fts_content_fingerprint") or "")
                )
                and meta.get("lexical_contract_version") == OCR_LEXICAL_CONTRACT_VERSION
                and frames == fts_frames == mapped == OCR_EXPECTED_FRAMES
                and fts_unmapped == 0
                and fts_orphaned == 0
                and distinct_points == frames
                and min_point_id == 1
                and max_point_id == OCR_EXPECTED_FRAMES
                and invalid_identity == 0
                and invalid_paths == 0
            )
            return {
                "ready": ready,
                "integrity": integrity,
                "frames": frames,
                "fts_frames": fts_frames,
                "mapped_frames": mapped,
                "videos": videos,
                "distinct_points": distinct_points,
                "min_point_id": min_point_id,
                "max_point_id": max_point_id,
                "fts_unmapped": fts_unmapped,
                "fts_orphaned": fts_orphaned,
                "invalid_identity": invalid_identity,
                "invalid_paths": invalid_paths,
                "meta_rows": meta_rows,
                "sqlite_user_version": user_version,
                "meta": meta,
                "tables": sorted(tables),
            }
        except (sqlite3.Error, ValueError, TypeError, OverflowError) as error:
            return {
                "ready": False,
                "integrity": "error",
                "frames": 0,
                "fts_frames": 0,
                "mapped_frames": 0,
                "videos": 0,
                "distinct_points": 0,
                "min_point_id": 0,
                "max_point_id": 0,
                "fts_unmapped": 0,
                "fts_orphaned": 0,
                "invalid_identity": 0,
                "invalid_paths": 0,
                "meta_rows": 0,
                "meta": {},
                "tables": [],
                "diagnostics": str(error),
            }

    def _resolve_under(self, root: Path, relative_path: str) -> Path:
        # Normalize host-provided Windows separators before resolving inside a
        # Linux container (and vice versa).  The containment check below is
        # still authoritative after resolution.
        normalized = str(relative_path).replace("\\", "/")
        if not _safe_relative_path(normalized):
            raise ValueError(f"OCR artifact path is not a safe relative path: {relative_path}")
        candidate = (root / Path(*PurePosixPath(normalized).parts)).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"OCR artifact escapes its root: {relative_path}") from error
        return candidate

    def _artifact_status_locked(
        self,
        record: dict[str, Any],
        *,
        root: Path,
        label: str,
        verify_hash: bool = True,
    ) -> dict[str, Any]:
        path_value = str(record.get("path") or "") if isinstance(record, dict) else ""
        result: dict[str, Any] = {
            "label": label,
            "path": path_value,
            "stat_matches": False,
            "hash_recomputed": False,
            "fingerprint_matches": False,
        }
        if not isinstance(record, dict):
            result["path"] = "<invalid>"
            result["error"] = "artifact record must be an object"
            return result
        try:
            path = self._resolve_under(root, path_value)
            result["resolved_path"] = str(path)
            stat = path.stat()
            stat_matches = (
                int(record.get("size", -1)) == int(stat.st_size)
                and int(record.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
            )
            result["stat_matches"] = stat_matches
            expected_sha = str(record.get("sha256") or "")
            if len(expected_sha) != 64 or any(
                character not in "0123456789abcdefABCDEF"
                for character in expected_sha
            ):
                result["error"] = "sha256 is missing or malformed"
                return result
            cache_key = str(path)
            cache_identity = (int(stat.st_size), int(stat.st_mtime_ns), int(getattr(stat, "st_ino", 0)))
            cached = self._artifact_hash_cache.get(cache_key)
            hash_required = bool(verify_hash or not stat_matches)
            if cached is not None and cached[0] == cache_identity:
                actual_sha = cached[1]
            elif not hash_required:
                # A matching stat is evidence from preparation; do not hash a
                # large artifact during every health request.  Hashing is done
                # once when the artifact is first observed or when stat changes.
                actual_sha = expected_sha
            else:
                actual_sha = _sha256_file(path)
                self._artifact_hash_cache[cache_key] = (cache_identity, actual_sha)
                result["hash_recomputed"] = True
            # Keep the observed digest so callers can build aggregate
            # diagnostics without hashing the same changed artifact twice.
            result["observed_sha256"] = actual_sha
            result["fingerprint_matches"] = actual_sha == expected_sha
            if not result["fingerprint_matches"]:
                result["error"] = "sha256 mismatch"
            return result
        except (OSError, ValueError, TypeError, OverflowError) as error:
            result["error"] = str(error)
            return result

    def _artifact_status(
        self,
        record: Any,
        *,
        relative_root: Path | None = None,
        root: Path | None = None,
        verify_hash: bool = True,
    ) -> tuple[bool, bool, bool]:
        """Validate one artifact for diagnostics and compatibility tests.

        The tuple is ``(content_matches, stat_matches, fingerprint_present)``.
        Runtime health uses the richer locked diagnostic, while this small
        adapter keeps the established ASR/OCR test contract available.
        """

        artifact_root = (root or relative_root or self.data_root).resolve()
        with self._lock:
            status = self._artifact_status_locked(
                record if isinstance(record, dict) else {},
                root=artifact_root,
                label=str(record.get("path") or "<invalid>")
                if isinstance(record, dict)
                else "<invalid>",
                verify_hash=verify_hash,
            )
        expected_sha = str(record.get("sha256") or "") if isinstance(record, dict) else ""
        fingerprint_present = len(expected_sha) == 64 and all(
            character in "0123456789abcdefABCDEF" for character in expected_sha
        )
        return (
            bool(status.get("fingerprint_matches") is True),
            bool(status.get("stat_matches") is True),
            fingerprint_present,
        )

    def _source_inventory_locked(self, manifest: dict[str, Any]) -> dict[str, Any]:
        now = time.monotonic()
        records = manifest.get("source_files")
        if not isinstance(records, list):
            raise ValueError("OCR manifest source_files must be a list")
        manifest_identity = _file_identity(self.manifest_path)
        directory_identity = _file_identity(self.transcripts_dir)
        manifest_record_paths = tuple(
            str(record.get("path") or "")
            if isinstance(record, dict)
            else "<invalid>"
            for record in records
        )
        cache_key = (manifest_identity, directory_identity, manifest_record_paths)
        cached = self._source_inventory_cache
        if (
            cached is not None
            and cached[1] == cache_key
            and now - cached[0] < OCR_SOURCE_AUDIT_TTL_S
        ):
            payload = dict(cached[2])
            payload["age_s"] = max(0.0, now - cached[0])
            return payload
        stat_records: list[dict[str, Any]] = []
        signature: list[Any] = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("OCR source record must be an object")
            path = self._resolve_under(self.data_root, str(record.get("path") or ""))
            try:
                stat = path.stat()
                item = {
                    "path": str(record.get("path") or ""),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "inode": int(getattr(stat, "st_ino", 0)),
                    "exists": True,
                }
            except OSError:
                item = {
                    "path": str(record.get("path") or ""),
                    "size": -1,
                    "mtime_ns": -1,
                    "inode": -1,
                    "exists": False,
                }
            stat_records.append(item)
            signature.append(tuple(item.values()))
        manifest_paths = {str(item.get("path") or "") for item in records if isinstance(item, dict)}
        try:
            actual_paths = {
                path.resolve().relative_to(self.data_root).as_posix()
                for path in self.transcripts_dir.glob("*.jsonl")
            }
        except OSError:
            actual_paths = set()
        directory_paths_match = (
            len(actual_paths) == OCR_EXPECTED_SOURCE_FILES
            and actual_paths == manifest_paths
        )
        payload = {
            "records": stat_records,
            "signature": tuple(signature),
            "directory_file_count": len(actual_paths),
            "directory_paths_match": directory_paths_match,
            "age_s": 0.0,
        }
        self._source_inventory_cache = (now, cache_key, payload)
        return payload

    def _validate_source_artifacts_locked(
        self,
        manifest: dict[str, Any],
        inventory: dict[str, Any],
    ) -> tuple[bool, bool, list[dict[str, Any]]]:
        records = manifest.get("source_files") or []
        inventory_by_path = {str(item["path"]): item for item in inventory.get("records", [])}
        actual_records: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        all_match = True
        for record in records:
            path_value = str(record.get("path") or "")
            observed = inventory_by_path.get(path_value) or {}
            expected_sha = str(record.get("sha256") or "")
            expected_sha_valid = len(expected_sha) == 64 and all(
                character in "0123456789abcdefABCDEF" for character in expected_sha
            )
            observed_stat_matches = bool(
                observed.get("exists") is True
                and int(observed.get("size", -1)) == int(record.get("size", -2))
                and int(observed.get("mtime_ns", -1)) == int(record.get("mtime_ns", -2))
            )
            # ``_source_inventory_locked`` already performed the complete
            # source stat pass (and is TTL-cached).  Do not stat all 873 files
            # again on every health call; only an absent/changed artifact
            # needs the richer resolver/hash path below.
            if observed_stat_matches and expected_sha_valid:
                try:
                    resolved = self._resolve_under(self.data_root, path_value)
                    cached_hash = self._artifact_hash_cache.get(str(resolved))
                    cache_identity = (
                        int(observed.get("size", -1)),
                        int(observed.get("mtime_ns", -1)),
                        int(observed.get("inode", 0)),
                    )
                    if cached_hash is not None and cached_hash[0] == cache_identity:
                        observed_sha = str(cached_hash[1])
                        fingerprint_matches = observed_sha == expected_sha
                    else:
                        # Matching publication stats are the preparation
                        # evidence used to avoid hashing the full corpus on
                        # every request.  A previously cached digest, when
                        # available, always wins over that shortcut.
                        observed_sha = expected_sha
                        fingerprint_matches = True
                    status = {
                        "label": path_value,
                        "path": path_value,
                        "resolved_path": str(resolved),
                        "stat_matches": True,
                        "hash_recomputed": False,
                        "fingerprint_matches": fingerprint_matches,
                        "observed_sha256": observed_sha,
                    }
                    if not fingerprint_matches:
                        status["error"] = "sha256 mismatch"
                except (OSError, ValueError, TypeError) as error:
                    status = {
                        "label": path_value,
                        "path": path_value,
                        "stat_matches": False,
                        "hash_recomputed": False,
                        "fingerprint_matches": False,
                        "error": str(error),
                    }
            else:
                status = self._artifact_status_locked(
                    record,
                    root=self.data_root,
                    label=path_value,
                    verify_hash=False,
                )
            diagnostics.append(status)
            current = dict(record)
            if observed.get("exists") is not True:
                all_match = False
            # The manifest binds both content and the publication stat.  A
            # source file whose bytes still hash to the same value but whose
            # size/mtime no longer matches is nevertheless a stale
            # publication and must not make the capability ready.
            if status.get("stat_matches") is not True:
                all_match = False
            if status.get("fingerprint_matches") is not True:
                all_match = False
            if (
                status.get("stat_matches") is not True
                or status.get("fingerprint_matches") is not True
            ):
                current["sha256"] = str(status.get("observed_sha256") or "")
            actual_records.append(current)
        aggregate_matches = source_fingerprint(actual_records) == str(
            manifest.get("source_fingerprint") or ""
        )
        directory_paths_match = inventory.get("directory_paths_match") is True
        return (
            all_match and aggregate_matches and directory_paths_match,
            aggregate_matches and directory_paths_match,
            diagnostics,
        )

    def _fast_ready_locked(self, manifest: dict[str, Any]) -> bool:
        if (
            manifest.get("schema_version") != OCR_INDEX_SCHEMA_VERSION
            or manifest.get("status") != "ready"
            or manifest.get("passed") is not True
        ):
            return False
        canonical = manifest.get("canonical_metadata") or {}
        build_identity_valid = _manifest_build_identity_valid(manifest)
        connection_state = self._refresh_connection_locked(manifest)
        if connection_state.get("connection_error") or self._connection is None:
            return False
        state = self._database_state_locked()
        meta = state.get("meta") or {}
        canonical_identity = state.get("canonical_identity") or self._validated_canonical_state or {}
        database = manifest.get("database") or {}
        database_path_matches = str(database.get("path") or "") == self.database_path.name
        database_stat_matches = False
        try:
            stat = self.database_path.stat()
            database_stat_matches = (
                database_path_matches
                and int(database.get("size", -1)) == int(stat.st_size)
                and int(database.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
            )
        except (OSError, TypeError, ValueError):
            pass
        canonical = manifest.get("canonical_metadata") or {}
        canonical_stat_matches = False
        try:
            canonical_stat = self.canonical_metadata_path.stat()
            canonical_stat_matches = (
                int(canonical.get("size", -1)) == int(canonical_stat.st_size)
                and int(canonical.get("mtime_ns", -1)) == int(canonical_stat.st_mtime_ns)
            )
        except (OSError, TypeError, ValueError):
            pass
        database_diag = self._artifact_status_locked(
            database,
            root=self.database_path.parent,
            label="database",
            verify_hash=False,
        )
        canonical_diag = self._artifact_status_locked(
            canonical,
            root=self.data_root,
            label="canonical_metadata",
            verify_hash=False,
        )
        return bool(
            state.get("ready") is True
            and canonical_identity.get("ready") is True
            and build_identity_valid
            and database_stat_matches
            and database_diag.get("stat_matches") is True
            and database_diag.get("fingerprint_matches") is True
            and canonical_stat_matches
            and canonical_diag.get("stat_matches") is True
            and canonical_diag.get("fingerprint_matches") is True
            and meta.get("build_id") == manifest.get("build_id")
            and meta.get("source_fingerprint") == manifest.get("source_fingerprint")
            and meta.get("canonical_fingerprint") == manifest.get("canonical_fingerprint")
            and meta.get("fts_content_fingerprint")
            == manifest.get("fts_content_fingerprint")
            and meta.get("lexical_contract_version")
            == manifest.get("lexical_contract_version")
        )

    def assert_ready(self) -> None:
        with self._lock:
            try:
                manifest = self._load_manifest_locked()
                if not self._fast_ready_locked(manifest):
                    raise RuntimeError("OCR index is stale, incomplete, or not prepared")
            except Exception as error:
                self._close_connection_locked()
                raise RuntimeError(f"OCR index readiness validation failed: {error}") from error

    def health(self, audit_sources: bool = False) -> dict[str, Any]:
        with self._lock:
            started = time.monotonic()
            try:
                manifest = self._load_manifest_locked()
                # Raw OCR sources are immutable inputs to the published
                # SQLite artifact.  Do not scan 873 JSONL files on the hot
                # health/config/search path; the dedicated OCR health route
                # can request the cached source audit explicitly.
                # A legacy/invalid manifest cannot be made healthy by auditing
                # its raw sources.  Reject it before touching the 873-file
                # directory so an old v2 index fails quickly and the
                # dedicated health route remains a bounded diagnostic.
                inventory = (
                    self._source_inventory_locked(manifest)
                    if audit_sources
                    and manifest.get("schema_version") == OCR_INDEX_SCHEMA_VERSION
                    else None
                )
                manifest_stat = _file_identity(self.manifest_path)
                database_stat = _file_identity(self.database_path)
                canonical_stat = _file_identity(self.canonical_metadata_path)
                signature = (
                    bool(audit_sources),
                    manifest_stat,
                    database_stat,
                    canonical_stat,
                    inventory.get("signature") if inventory is not None else None,
                    self._connection_generation,
                    self._opened_file_identity,
                    self._opened_build_id,
                    self._connection_error,
                )
                if self._health_cache is not None:
                    cached_at, cached_signature, cached_payload = self._health_cache
                    if (
                        cached_signature == signature
                        and started - cached_at < OCR_HEALTH_CACHE_TTL_S
                        and self._connection is not None
                        and self._connection_error is None
                    ):
                        payload = dict(cached_payload)
                        if audit_sources:
                            audit_cached_at = (
                                self._source_audit_cache[0]
                                if self._source_audit_cache is not None
                                else cached_at
                            )
                            payload["source_validation_age_s"] = round(
                                started - audit_cached_at, 3
                            )
                            payload["source_audit_age_s"] = round(
                                started - audit_cached_at, 3
                            )
                        payload["connection_reopened"] = False
                        return payload

                connection_state = self._refresh_connection_locked(manifest)
                database_state = self._database_state_locked()
                database_record = manifest.get("database") or {}
                canonical_record = manifest.get("canonical_metadata") or {}
                database_diag = self._artifact_status_locked(
                    database_record,
                    root=self.database_path.parent,
                    label="database",
                    verify_hash=False,
                )
                canonical_diag = self._artifact_status_locked(
                    canonical_record,
                    root=self.data_root,
                    label="canonical_metadata",
                    verify_hash=False,
                )
                if audit_sources and inventory is not None:
                    source_audit_key = (
                        _file_identity(self.manifest_path),
                        inventory.get("signature"),
                    )
                    cached_source_audit = self._source_audit_cache
                    if (
                        cached_source_audit is not None
                        and cached_source_audit[1] == source_audit_key
                        and started - cached_source_audit[0] < OCR_SOURCE_AUDIT_TTL_S
                    ):
                        source_ready = cached_source_audit[2]
                        source_fingerprint_matches = cached_source_audit[3]
                        source_diags = [dict(item) for item in cached_source_audit[4]]
                    else:
                        (
                            source_ready,
                            source_fingerprint_matches,
                            source_diags,
                        ) = self._validate_source_artifacts_locked(manifest, inventory)
                        self._source_audit_cache = (
                            started,
                            source_audit_key,
                            source_ready,
                            source_fingerprint_matches,
                            [dict(item) for item in source_diags],
                        )
                    source_stat_matches = bool(
                        all(item.get("stat_matches") is True for item in source_diags)
                        and len(source_diags) == OCR_EXPECTED_SOURCE_FILES
                    )
                else:
                    source_ready = True
                    source_fingerprint_matches = None
                    source_diags = []
                    source_stat_matches = None
                source_audit_timestamp = (
                    self._source_audit_cache[0]
                    if audit_sources and self._source_audit_cache is not None
                    else None
                )
                meta = database_state.get("meta") or {}
                canonical_identity = (
                    database_state.get("canonical_identity")
                    or self._validated_canonical_state
                    or {}
                )
                build_identity_valid = _manifest_build_identity_valid(manifest)
                database_matches = bool(
                    database_state.get("ready") is True
                    and meta.get("build_id") == manifest.get("build_id")
                    and meta.get("source_fingerprint") == manifest.get("source_fingerprint")
                    and meta.get("canonical_fingerprint") == manifest.get("canonical_fingerprint")
                    and meta.get("fts_content_fingerprint")
                    == manifest.get("fts_content_fingerprint")
                    and meta.get("lexical_contract_version")
                    == manifest.get("lexical_contract_version")
                )
                canonical_fingerprint_matches = bool(
                    canonical_diag.get("fingerprint_matches") is True
                    and str(manifest.get("canonical_fingerprint") or "")
                    == str(canonical_record.get("sha256") or "")
                )
                source_stale = (
                    None
                    if not audit_sources
                    else bool(
                        source_ready is not True
                        or source_stat_matches is not True
                        or source_fingerprint_matches is not True
                    )
                )
                ready = bool(
                    manifest.get("schema_version") == OCR_INDEX_SCHEMA_VERSION
                    and manifest.get("status") == "ready"
                    and manifest.get("passed") is True
                    and database_matches
                    and database_diag.get("stat_matches") is True
                    and canonical_fingerprint_matches
                    and canonical_diag.get("stat_matches") is True
                    and database_diag.get("fingerprint_matches") is True
                    and build_identity_valid
                    and canonical_identity.get("ready") is True
                    and connection_state.get("connection_error") is None
                )
                warnings: list[str] = []
                offline_identity = manifest.get("offline_identity") or {}
                if offline_identity.get("revision_verified") is not True:
                    warnings.append("OCR offline model revision is not cryptographically verified")
                if source_stale is True:
                    warnings.append(
                        "OCR raw source differs from the published index; rebuild is recommended"
                    )
                failed_source_diags = [
                    item
                    for item in source_diags
                    if item.get("stat_matches") is not True
                    or item.get("fingerprint_matches") is not True
                    or item.get("error")
                ]
                diagnostics = [database_diag, canonical_diag, *failed_source_diags]
                artifact_summary = {
                    "source_total": (
                        len(source_diags)
                        if audit_sources
                        else len(manifest.get("source_files") or [])
                    ),
                    "source_verified": (
                        len(source_diags) - len(failed_source_diags)
                        if audit_sources
                        else None
                    ),
                    "source_failed": (
                        len(failed_source_diags) if audit_sources else None
                    ),
                    "hash_recomputed": sum(
                        1
                        for item in (database_diag, canonical_diag, *source_diags)
                        if item.get("hash_recomputed") is True
                    ),
                }
                payload = {
                    "status": "ready" if ready else "not_ready",
                    "ready": ready,
                    "production_ready": False,
                    "required": False,
                    "fail_closed": not ready,
                    "database": str(self.database_path),
                    "manifest": str(self.manifest_path),
                    "schema_version": manifest.get("schema_version"),
                    "sqlite_user_version": database_state.get("sqlite_user_version"),
                    "frames": int(database_state.get("frames", 0)),
                    "fts_frames": int(database_state.get("fts_frames", 0)),
                    "fts_unmapped_rows": int(database_state.get("fts_unmapped", 0)),
                    "fts_orphaned_rows": int(database_state.get("fts_orphaned", 0)),
                    "mapped_frames": int(database_state.get("mapped_frames", 0)),
                    "videos": int(database_state.get("videos", 0)),
                    "distinct_point_ids": int(database_state.get("distinct_points", 0)),
                    "min_point_id": int(database_state.get("min_point_id", 0)),
                    "max_point_id": int(database_state.get("max_point_id", 0)),
                    "invalid_identity_rows": int(database_state.get("invalid_identity", 0)),
                    "invalid_path_rows": int(database_state.get("invalid_paths", 0)),
                    "meta_rows": int(database_state.get("meta_rows", 0)),
                    "integrity": database_state.get("integrity"),
                    "manifest_build_id": manifest.get("build_id"),
                    "internal_build_id": meta.get("build_id"),
                    "opened_build_id": connection_state.get("opened_build_id"),
                    "database_matches_manifest": database_matches,
                    "database_path_matches_manifest": database_diag.get("path")
                    == self.database_path.name,
                    "database_stat_matches_manifest": database_diag.get("stat_matches", False),
                    "canonical_metadata_matches_manifest": canonical_fingerprint_matches,
                    "canonical_metadata_stat_matches_manifest": canonical_diag.get(
                        "stat_matches", False
                    ),
                    "source_matches_manifest": source_ready if audit_sources else None,
                    "source_stat_matches_manifest": source_stat_matches,
                    "source_fingerprint_matches_manifest": source_fingerprint_matches,
                    "source_directory_file_count": int(
                        inventory.get("directory_file_count", 0)
                        if inventory is not None
                        else 0
                    ),
                    "source_file_count_matches_manifest": bool(
                        audit_sources
                        and len(manifest.get("source_files") or [])
                        == OCR_EXPECTED_SOURCE_FILES
                        and inventory is not None
                        and inventory.get("directory_file_count", 0)
                        == OCR_EXPECTED_SOURCE_FILES
                    ),
                    "source_directory_matches_manifest": bool(
                        audit_sources
                        and inventory is not None
                        and inventory.get("directory_paths_match") is True
                    ),
                    "canonical_fingerprint_matches": canonical_fingerprint_matches,
                    "canonical_identity_matches": bool(
                        canonical_identity.get("ready") is True
                    ),
                    "canonical_identity_rows": int(
                        canonical_identity.get("canonical_rows", 0)
                    ),
                    "canonical_identity_mismatch_count": int(
                        canonical_identity.get("mismatch_count", 0)
                    ),
                    "build_id_matches": bool(
                        build_identity_valid
                        and meta.get("build_id") == manifest.get("build_id")
                    ),
                    "internal_metadata_matches": database_matches,
                    "connection_generation": connection_state.get("connection_generation", 0),
                    "connection_reopened": bool(connection_state.get("connection_reopened")),
                    "connection_error": connection_state.get("connection_error"),
                    "source_validation_age_s": (
                        round(float(inventory.get("age_s", 0.0)), 3)
                        if inventory is not None
                        else None
                    ),
                    "source_audit_performed": bool(audit_sources),
                    "source_stale": source_stale,
                    "source_drift_is_blocking": False,
                    "source_audit_age_s": (
                        round(started - source_audit_timestamp, 3)
                        if source_audit_timestamp is not None
                        else None
                    ),
                    "fts_content_verified": bool(
                        database_state.get("meta", {}).get("fts_content_fingerprint")
                        == manifest.get("fts_content_fingerprint")
                        and database_state.get("meta", {}).get("lexical_contract_version")
                        == manifest.get("lexical_contract_version")
                    ),
                    "fts_content_fingerprint": manifest.get("fts_content_fingerprint"),
                    "lexical_contract_version": manifest.get("lexical_contract_version"),
                    "artifact_diagnostics": diagnostics,
                    "artifact_summary": artifact_summary,
                    "offline_identity": offline_identity,
                    "revision_verified": bool(offline_identity.get("revision_verified") is True),
                    "warnings": warnings,
                    "timing": {
                        "health_ms": round((time.monotonic() - started) * 1000.0, 2),
                    },
                }
                self._health_cache = (started, signature, dict(payload))
                return payload
            except Exception as error:  # health must never crash the API
                self._close_connection_locked()
                self._health_cache = None
                return {
                    "status": "not_ready",
                    "ready": False,
                    "production_ready": False,
                    "required": False,
                    "database": str(self.database_path),
                    "manifest": str(self.manifest_path),
                    "schema_version": None,
                    "sqlite_user_version": None,
                    "frames": 0,
                    "fts_frames": 0,
                    "fts_unmapped_rows": 0,
                    "fts_orphaned_rows": 0,
                    "mapped_frames": 0,
                    "videos": 0,
                    "distinct_point_ids": 0,
                    "min_point_id": 0,
                    "max_point_id": 0,
                    "invalid_identity_rows": 0,
                    "invalid_path_rows": 0,
                    "meta_rows": 0,
                    "integrity": "error",
                    "manifest_build_id": None,
                    "internal_build_id": None,
                    "database_matches_manifest": False,
                    "database_path_matches_manifest": False,
                    "database_stat_matches_manifest": False,
                    "canonical_metadata_matches_manifest": False,
                    "canonical_metadata_stat_matches_manifest": False,
                    "source_matches_manifest": False,
                    "source_stat_matches_manifest": False,
                    "source_fingerprint_matches_manifest": False,
                    "source_directory_file_count": 0,
                    "source_directory_matches_manifest": False,
                    "canonical_fingerprint_matches": False,
                    "canonical_identity_matches": False,
                    "canonical_identity_rows": 0,
                    "canonical_identity_mismatch_count": 0,
                    "build_id_matches": False,
                    "internal_metadata_matches": False,
                    "error": str(error),
                    "fail_closed": True,
                    "connection_generation": self._connection_generation,
                    "connection_reopened": False,
                    "opened_build_id": None,
                    "connection_error": str(error),
                    "artifact_diagnostics": [
                        {
                            "label": "health",
                            "path": "<health>",
                            "stat_matches": False,
                            "hash_recomputed": False,
                            "fingerprint_matches": False,
                            "error": str(error),
                        }
                    ],
                    "artifact_summary": {
                        "source_total": 0,
                        "source_verified": 0,
                        "source_failed": 0,
                        "hash_recomputed": 0,
                    },
                    "fts_content_verified": False,
                    "fts_content_fingerprint": None,
                    "lexical_contract_version": OCR_LEXICAL_CONTRACT_VERSION,
                    "source_audit_performed": bool(audit_sources),
                    "source_stale": None,
                    "source_drift_is_blocking": False,
                    "source_audit_age_s": None,
                    "offline_identity": {},
                    "revision_verified": False,
                    "warnings": ["OCR health validation failed closed"],
                    "source_file_count_matches_manifest": False,
                    "source_validation_age_s": None,
                }

    @staticmethod
    def _match_expression(tokens: Iterable[str]) -> str:
        terms: list[str] = []
        for token in tokens:
            clean = " ".join(str(token).replace('"', " ").split())
            if clean:
                terms.append(f'"{clean}"')
        return " OR ".join(terms)

    def _query_stream_locked(self, query: str, limit: int) -> tuple[list[sqlite3.Row], list[str]]:
        tokens = query_tokens(query)
        if not tokens or self._connection is None:
            return [], tokens
        expression = self._match_expression(tokens)
        rows = self._connection.execute(
            """
            SELECT f.id, f.frame_uid, f.point_id, f.video_id, f.keyframe_n,
                   f.frame_idx, f.pts_time_s, f.fps, f.image_relpath,
                   f.full_text, bm25(ocr_fts) AS bm25_score
            FROM ocr_fts
            JOIN ocr_frames AS f ON f.id = ocr_fts.rowid
            WHERE ocr_fts MATCH ?
            ORDER BY bm25_score, f.id
            LIMIT ?
            """,
            (expression, int(limit)),
        ).fetchall()
        return rows, tokens

    def _search_many_locked(
        self,
        query_by_stream: dict[str, str],
        *,
        per_stream_top_k: int,
        final_top_k: int,
        allow_single: bool = False,
    ) -> dict[str, Any]:
        streams = {str(key): str(value).strip() for key, value in query_by_stream.items()}
        if allow_single:
            valid = len(streams) == 1
        else:
            identities = [_stream_identity(key) for key in streams]
            valid = (
                len(streams) == 12
                and {role for role, _language in identities} == set(QUERY_ROLES)
                and all(language in {"vi", "en"} for _role, language in identities)
                and len(set(identities)) == 12
            )
        if not valid or any(not value for value in streams.values()):
            raise ValueError(
                "OCR requires six bilingual query variants"
                if not allow_single
                else "OCR requires one non-empty query"
            )
        if any(not query_tokens(value) for value in streams.values()):
            raise ValueError("OCR queries must contain at least one searchable token")

        started = time.perf_counter()
        frame_hits: dict[str, dict[str, Any]] = {}
        stream_counts: dict[str, int] = {}
        stream_order = list(streams)
        for stream, query in streams.items():
            role, language = _stream_identity(stream)
            rows, tokens = self._query_stream_locked(query, per_stream_top_k)
            stream_counts[stream] = len(rows)
            max_relevance = max(
                (max(0.0, -float(row["bm25_score"])) for row in rows),
                default=0.0,
            ) or 1.0
            ordered_tokens = ordered_lexical_tokens(query)
            query_bigrams = _ordered_lexical_bigrams(query)
            search_token_set = set(tokens)
            lexical_occurrence_count = sum(
                1 for token in ordered_tokens if token in search_token_set
            )
            for rank, row in enumerate(rows, 1):
                frame_uid = str(row["frame_uid"] or "")
                video_id = str(row["video_id"] or "")
                try:
                    frame_idx = int(row["frame_idx"])
                    point_id = int(row["point_id"])
                    keyframe_n = int(row["keyframe_n"])
                    pts_time_s = float(row["pts_time_s"])
                    fps = float(row["fps"])
                except (TypeError, ValueError, OverflowError) as error:
                    raise RuntimeError(
                        f"OCR row has invalid canonical identity: {frame_uid}"
                    ) from error
                image_relpath = str(row["image_relpath"] or "")
                if (
                    not frame_uid
                    or frame_uid != f"{video_id}:{frame_idx}"
                    or point_id < 1
                    or keyframe_n < 1
                    or not math.isfinite(pts_time_s)
                    or pts_time_s < 0
                    or not image_relpath
                    or not math.isfinite(fps)
                    or fps <= 0
                ):
                    raise RuntimeError(f"OCR row has invalid canonical identity: {frame_uid}")
                full_text = str(row["full_text"] or "")
                transcript_tokens = _folded_tokens(full_text)
                token_set = set(transcript_tokens)
                matched = [token for token in tokens if token in token_set]
                token_coverage = len(matched) / len(tokens) if tokens else 0.0
                transcript_bigrams = set(_token_bigrams(transcript_tokens))
                if query_bigrams:
                    ngram_coverage = sum(
                        1 for pair in query_bigrams if pair in transcript_bigrams
                    ) / len(query_bigrams)
                elif lexical_occurrence_count == 1:
                    # A single lexical token has no pair to score, so retain
                    # the established token-coverage fallback.  Multiple
                    # lexical tokens separated by a stopword/short token have
                    # no adjacent query bigram; do not manufacture one by
                    # falling back to token coverage.
                    ngram_coverage = token_coverage
                else:
                    ngram_coverage = 0.0
                bm25_raw = float(row["bm25_score"])
                if not math.isfinite(bm25_raw):
                    raise RuntimeError("SQLite OCR BM25 returned a non-finite score")
                bm25_relevance = max(0.0, -bm25_raw) / max_relevance
                combined = _ocr_query_score(
                    bm25_relevance,
                    token_coverage,
                    ngram_coverage,
                )
                evidence = {
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
                    "query_bigrams": [list(pair) for pair in query_bigrams],
                }
                candidate = frame_hits.setdefault(
                    frame_uid,
                    {
                        "frame_uid": frame_uid,
                        "point_id": point_id,
                        "source_row_id": int(row["id"]),
                        "video_id": video_id,
                        "keyframe_n": keyframe_n,
                        "frame_idx": frame_idx,
                        "pts_time_s": pts_time_s,
                        "fps": fps,
                        "image_relpath": image_relpath,
                        "full_text": full_text,
                        "query_scores": {},
                        "stream_provenance": {},
                    },
                )
                if (
                    candidate["point_id"] != point_id
                    or candidate["video_id"] != video_id
                    or candidate["frame_idx"] != frame_idx
                    or candidate["keyframe_n"] != keyframe_n
                    or candidate["image_relpath"] != image_relpath
                    or float(candidate["pts_time_s"]) != pts_time_s
                    or float(candidate["fps"]) != fps
                ):
                    raise RuntimeError(
                        f"OCR frame maps to conflicting canonical identities: {frame_uid}"
                    )
                candidate["query_scores"][stream] = combined
                candidate["stream_provenance"][stream] = evidence

        if not frame_hits:
            return {
                "results": [],
                "candidate_frame_count": 0,
                "stream_counts": stream_counts,
                "timing": {"total_ms": round((time.perf_counter() - started) * 1000.0, 2)},
            }
        values = [max(float(value) for value in item["query_scores"].values()) for item in frame_hits.values()]
        normalized = sigmoid_zscore(values)
        ranked: list[dict[str, Any]] = []
        for index, candidate in enumerate(frame_hits.values()):
            best_stream, raw_score = max(
                candidate["query_scores"].items(),
                key=lambda pair: (
                    float(pair[1]),
                    -int(candidate["stream_provenance"][pair[0]]["rank"]),
                    -stream_order.index(pair[0]),
                ),
            )
            role, language = _stream_identity(best_stream)
            winner = candidate["stream_provenance"][best_stream]
            frame = base_frame(
                {
                    "frame_uid": candidate["frame_uid"],
                    "video_id": candidate["video_id"],
                    "keyframe_n": candidate["keyframe_n"],
                    "frame_idx": candidate["frame_idx"],
                    "pts_time_s": candidate["pts_time_s"],
                    "fps": candidate["fps"],
                    "image_relpath": candidate["image_relpath"],
                },
                score=float(normalized[index]),
                rank=1,
                score_type="ocr_bm25_ngram",
            )
            frame.update(
                {
                    "frame_uid": candidate["frame_uid"],
                    "point_id": candidate["point_id"],
                    "global_idx": candidate["point_id"],
                    "ocr_text": candidate["full_text"],
                    "full_text": candidate["full_text"],
                    "ocr_raw_score": round(float(raw_score), 8),
                    "ocr_combined_score": round(float(raw_score), 8),
                    "ocr_normalized_score": round(float(normalized[index]), 8),
                    # Generic aliases mirror the ASR compatibility result
                    # shape while the OCR-prefixed fields remain canonical.
                    "raw_score": round(float(raw_score), 8),
                    "normalized_score": round(float(normalized[index]), 8),
                    "best_query_role": role,
                    "best_query_language": language,
                    "best_query_stream": best_stream,
                    "ocr_best_query_role": role,
                    "ocr_best_query_language": language,
                    "ocr_best_query_stream": best_stream,
                    "ocr_best_rank": int(winner["rank"]),
                    "ocr_query_scores": {
                        stream: round(float(candidate["query_scores"].get(stream, 0.0)), 8)
                        for stream in streams
                    },
                    "ocr_stream_provenance": {
                        stream: candidate["stream_provenance"].get(stream)
                        for stream in streams
                    },
                    "matched_keywords": winner["matched_terms"],
                    "matched_terms": winner["matched_terms"],
                    "bm25_raw": winner["bm25_raw"],
                    "ocr_bm25_raw": winner["bm25_raw"],
                    "bm25_relevance": winner["bm25_relevance"],
                    "ocr_bm25_relevance": winner["bm25_relevance"],
                    "token_coverage": winner["token_coverage"],
                    "ocr_token_coverage": winner["token_coverage"],
                    "ngram_coverage": winner["ngram_coverage"],
                    "adjacent_bigram_coverage": winner["ngram_coverage"],
                }
            )
            ranked.append(frame)
        ranked.sort(
            key=lambda item: (
                -float(item["ocr_normalized_score"]),
                int(item.get("ocr_best_rank", 2_000_001)),
                str(item["frame_uid"]),
            )
        )
        selected = ranked[:final_top_k]
        for rank, frame in enumerate(selected, 1):
            frame["rank"] = rank
        return {
            "results": selected,
            "candidate_frame_count": len(frame_hits),
            "stream_counts": stream_counts,
            "timing": {"total_ms": round((time.perf_counter() - started) * 1000.0, 2)},
        }

    def search_many(
        self,
        query_by_stream: dict[str, str],
        *,
        per_stream_top_k: int = 2_000,
        final_top_k: int = 500,
        _allow_single: bool = False,
    ) -> dict[str, Any]:
        if not 1 <= int(per_stream_top_k) <= 2_000:
            raise ValueError("per_stream_top_k must be between 1 and 2000")
        if not 1 <= int(final_top_k) <= 500:
            raise ValueError("final_top_k must be between 1 and 500")
        with self._lock:
            try:
                manifest = self._load_manifest_locked()
                ready = self._fast_ready_locked(manifest)
            except Exception as error:
                # Manifest/database publication and readiness failures are
                # dependency failures, not malformed query contracts.  Keep
                # them on the 503 path instead of misreporting them as 422.
                raise RuntimeError(
                    f"OCR index readiness validation failed: {error}"
                ) from error
            if not ready:
                raise RuntimeError("OCR index is stale, incomplete, or not prepared")
            if self._connection is None:
                raise RuntimeError("OCR index connection is unavailable")
            try:
                return self._search_many_locked(
                    query_by_stream,
                    per_stream_top_k=int(per_stream_top_k),
                    final_top_k=int(final_top_k),
                    allow_single=_allow_single,
                )
            except RuntimeError:
                raise
            except sqlite3.Error as error:
                raise RuntimeError(f"OCR FTS query failed: {error}") from error

    def lookup_many(self, frame_uids: Iterable[str]) -> dict[str, str]:
        """Resolve OCR text for many canonical frames with one readiness pass."""

        ordered: list[str] = []
        seen: set[str] = set()
        for value in frame_uids:
            frame_uid = str(value or "")
            if frame_uid and frame_uid not in seen:
                seen.add(frame_uid)
                ordered.append(frame_uid)
        if not ordered:
            return {}
        with self._lock:
            try:
                manifest = self._load_manifest_locked()
                if not self._fast_ready_locked(manifest) or self._connection is None:
                    return {}
                result: dict[str, str] = {}
                for offset in range(0, len(ordered), 500):
                    batch = ordered[offset : offset + 500]
                    placeholders = ",".join("?" for _ in batch)
                    rows = self._connection.execute(
                        "SELECT frame_uid, full_text FROM ocr_frames "
                        f"WHERE frame_uid IN ({placeholders})",
                        batch,
                    ).fetchall()
                    result.update({str(row[0]): str(row[1] or "") for row in rows})
                return result
            except (OSError, ValueError, TypeError, sqlite3.Error):
                return {}

    def lookup(self, frame_uid: str) -> str:
        with self._lock:
            # Keep the compatibility method's historical fail-soft contract,
            # while delegating the actual lookup to the bulk path.
            return self.lookup_many([frame_uid]).get(str(frame_uid), "")

    def search(self, keywords: list[str], top_k: int) -> list[dict[str, Any]]:
        """Compatibility one-query search; the service owns its heavy lock."""

        top_k = int(top_k)
        if not 1 <= top_k <= 500:
            raise ValueError("OCR top_k must be between 1 and 500")
        query = " ".join(str(item) for item in keywords if str(item).strip()).strip()
        payload = self.search_many(
            {"legacy": query},
            per_stream_top_k=min(2_000, max(500, top_k * 20)),
            final_top_k=top_k,
            _allow_single=True,
        )
        return payload["results"]

def _database_state_for_connection(connection: sqlite3.Connection) -> dict[str, Any]:
    """Inspect a connection without publishing it to a runtime index.

    The runtime inspector is reused by the preparation command so a malformed
    staging database is rejected before it can replace the live artifact.
    ``_database_state_locked`` only depends on ``_connection`` and therefore a
    small probe object is sufficient here; no runtime state is mutated.
    """

    previous_row_factory = connection.row_factory
    connection.row_factory = sqlite3.Row
    try:
        probe = object.__new__(OcrFtsIndex)
        probe._connection = connection
        probe._validated_database_state = None
        probe._validated_connection_generation = 0
        probe._connection_generation = 0
        probe._validated_database_identity = None
        probe._opened_file_identity = None
        probe._validated_build_id = None
        probe._opened_build_id = None
        return OcrFtsIndex._database_state_locked(probe)
    finally:
        connection.row_factory = previous_row_factory


__all__ = [
    "OCR_CANONICAL_RELATIVE_PATH",
    "OCR_EXPECTED_FRAMES",
    "OCR_EXPECTED_SOURCE_FILES",
    "OCR_LEXICAL_CONTRACT_VERSION",
    "OCR_INDEX_SCHEMA_VERSION",
    "OCR_MAPPING_STRATEGY",
    "OCR_RESULT_SCHEMA_VERSION",
    "OCR_SQLITE_USER_VERSION",
    "OcrFtsIndex",
    "_database_state_for_connection",
    "_fts_content_fingerprint_for_connection",
    "_manifest_build_identity_valid",
    "build_id_for",
    "build_ocr_index",
    "repair_mojibake",
    "source_fingerprint",
    "validate_ocr_sources",
]
