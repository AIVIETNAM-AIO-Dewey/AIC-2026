"""Persistent SQLite FTS5 BM25 index over DAM descriptions."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from ...infrastructure.qdrant import base_frame
from ..branch1.contracts import EXPECTED_FRAMES
from .contracts import EXPECTED_DAM_REGIONS


class DamBm25Index:
    # v2 stores the canonical frame record in the index. A sparse-only hit
    # must remain usable by the workbench without reconstructing metadata.
    VERSION = 2
    SCHEMA_VERSION = "branch2.dam-bm25.v2"

    def __init__(self, data_root: Path, state_root: Path) -> None:
        self.metadata_path = data_root / "dense_text_embeddings" / "dam_metadata.jsonl"
        self.state_root = state_root
        self.database_path = state_root / "branch2_dam_bm25.sqlite3"
        self.manifest_path = state_root / "branch2_dam_manifest.json"
        self._db: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    def _fingerprint(self) -> str:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                return ""
            if manifest.get("passed") is not True or manifest.get("status") != "ready":
                return ""
            return str(manifest["metadata_sha256"])
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return ""

    def _frame_fingerprint(self) -> str:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                return ""
            return str(manifest["frame_metadata_sha256"])
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return ""

    def _connection(self) -> sqlite3.Connection:
        with self._lock:
            if self._db is None:
                if not self.database_path.is_file():
                    raise FileNotFoundError(
                        f"Branch-2 BM25 index is missing: {self.database_path}; run the preparation command"
                    )
                # The BM25 database is published by atomic replacement and
                # is never mutated at runtime.  Immutable mode prevents the
                # old connection from blocking replacement on Windows;
                # manifest/fingerprint checks still gate every use.
                self._db = sqlite3.connect(
                    f"file:{self.database_path.as_posix()}?mode=ro&immutable=1",
                    uri=True,
                    check_same_thread=False,
                )
                self._db.row_factory = sqlite3.Row
            return self._db

    def _ready(self, db: sqlite3.Connection) -> bool:
        try:
            tables = {
                str(row[0])
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                ).fetchall()
            }
            if not {"dam_regions", "dam_fts", "branch2_meta"}.issubset(tables):
                return False
            count = int(db.execute("SELECT COUNT(*) FROM dam_regions").fetchone()[0])
            fts_count = int(db.execute("SELECT COUNT(*) FROM dam_fts").fetchone()[0])
            fingerprint = db.execute(
                "SELECT value FROM branch2_meta WHERE key='fingerprint'"
            ).fetchone()
            version = db.execute("SELECT value FROM branch2_meta WHERE key='version'").fetchone()
            document_count = db.execute(
                "SELECT value FROM branch2_meta WHERE key='document_count'"
            ).fetchone()
            schema = db.execute(
                "SELECT value FROM branch2_meta WHERE key='schema_version'"
            ).fetchone()
            frame_fingerprint = db.execute(
                "SELECT value FROM branch2_meta WHERE key='frame_metadata_fingerprint'"
            ).fetchone()
            return (
                count == EXPECTED_DAM_REGIONS
                and fts_count == EXPECTED_DAM_REGIONS
                and fingerprint is not None
                and bool(str(fingerprint[0] or ""))
                and fingerprint[0] == self._fingerprint()
                and version is not None
                and int(version[0]) == self.VERSION
                and document_count is not None
                and int(document_count[0]) == EXPECTED_DAM_REGIONS
                and schema is not None
                and schema[0] == self.SCHEMA_VERSION
                and frame_fingerprint is not None
                and bool(str(frame_fingerprint[0] or ""))
                and frame_fingerprint[0] == self._frame_fingerprint()
            )
        except (sqlite3.Error, TypeError, ValueError):
            return False

    @classmethod
    def prepare(
        cls,
        data_root: Path,
        state_root: Path,
        metadata_sha256: str,
        frame_metadata_sha256: str,
    ) -> Path:
        if not metadata_sha256 or not frame_metadata_sha256:
            raise ValueError(
                "DAM and canonical metadata fingerprints are required for BM25 preparation"
            )
        metadata_path = data_root / "dense_text_embeddings" / "dam_metadata.jsonl"
        frame_metadata_path = (
            data_root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl"
        )
        database_path = state_root / "branch2_dam_bm25.sqlite3"
        staging = database_path.with_suffix(".staging.sqlite3")
        state_root.mkdir(parents=True, exist_ok=True)
        if staging.exists():
            staging.unlink()
        frame_records: dict[str, dict[str, Any]] = {}
        with frame_metadata_path.open("r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, 1):
                frame = json.loads(line)
                frame_uid = str(frame.get("frame_uid") or "")
                derived_uid = f"{frame.get('video_id')}:{int(frame.get('frame_idx', -1))}"
                if (
                    not frame_uid
                    or frame_uid != derived_uid
                    or frame_uid in frame_records
                    or int(frame.get("point_id", 0)) != row_number
                ):
                    raise ValueError(
                        f"Invalid canonical frame metadata identity at row {row_number}"
                    )
                frame_records[frame_uid] = frame
        if len(frame_records) != EXPECTED_FRAMES:
            raise ValueError(
                f"Expected {EXPECTED_FRAMES} canonical frames, found {len(frame_records)}"
            )
        db = sqlite3.connect(staging)
        try:
            with db:
                db.execute(
                    "CREATE TABLE dam_regions ("
                    "id INTEGER PRIMARY KEY, "
                    "parent_point_id INTEGER NOT NULL, "
                    "frame_uid TEXT NOT NULL, "
                    "video_id TEXT NOT NULL, "
                    "frame_idx INTEGER NOT NULL, "
                    "keyframe_n INTEGER NOT NULL, "
                    "pts_time_s REAL NOT NULL, "
                    "fps REAL NOT NULL, "
                    "image_relpath TEXT NOT NULL, "
                    "description_en TEXT NOT NULL, "
                    "class_entity TEXT NOT NULL, "
                    "region_id TEXT NOT NULL, "
                    "bbox_json TEXT NOT NULL)"
                )
                db.execute(
                    "CREATE VIRTUAL TABLE dam_fts USING fts5(description_en, class_entity, content='dam_regions', content_rowid='id', tokenize='unicode61 remove_diacritics 2')"
                )
            batch: list[tuple[Any, ...]] = []
            total = 0
            with metadata_path.open("r", encoding="utf-8") as handle:
                for row_id, line in enumerate(handle, 1):
                    item = json.loads(line)
                    video_id = str(item["video_id"])
                    frame_idx = int(item["frame_idx"])
                    frame_uid = f"{video_id}:{frame_idx}"
                    frame_record = frame_records.get(frame_uid)
                    if frame_record is None:
                        raise ValueError(
                            f"DAM row {row_id} maps to unknown canonical frame {frame_uid}"
                        )
                    exported_keyframe_n = item.get("keyframe_n")
                    canonical_keyframe_n = int(frame_record["keyframe_n"])
                    if (
                        exported_keyframe_n is not None
                        and int(exported_keyframe_n) != canonical_keyframe_n
                    ):
                        raise ValueError(
                            f"DAM keyframe_n disagrees with canonical metadata for {frame_uid}"
                        )
                    batch.append(
                        (
                            row_id,
                            int(frame_record["point_id"]),
                            frame_uid,
                            video_id,
                            frame_idx,
                            canonical_keyframe_n,
                            float(frame_record["pts_time_s"]),
                            float(frame_record["fps"]),
                            str(frame_record["image_relpath"]),
                            str(item.get("description_en", "")),
                            str(item.get("class_entity", "")),
                            str(item.get("region_id", "")),
                            json.dumps(item.get("bbox", [])),
                        )
                    )
                    if len(batch) >= 5000:
                        db.executemany(
                            "INSERT INTO dam_regions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", batch
                        )
                        total += len(batch)
                        batch.clear()
            if batch:
                db.executemany("INSERT INTO dam_regions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                total += len(batch)
            if total != EXPECTED_DAM_REGIONS:
                raise ValueError(f"Expected {EXPECTED_DAM_REGIONS} DAM documents, found {total}")
            db.execute("INSERT INTO dam_fts(dam_fts) VALUES ('rebuild')")
            db.execute("CREATE INDEX dam_regions_frame_idx ON dam_regions(frame_uid)")
            db.execute("CREATE TABLE branch2_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.executemany(
                "INSERT INTO branch2_meta(key,value) VALUES (?,?)",
                [
                    ("fingerprint", metadata_sha256),
                    ("version", str(cls.VERSION)),
                    ("document_count", str(total)),
                    ("schema_version", cls.SCHEMA_VERSION),
                    ("frame_metadata_fingerprint", frame_metadata_sha256),
                ],
            )
            db.execute("PRAGMA optimize")
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()
        os.replace(staging, database_path)
        return database_path

    @staticmethod
    def _tokens(query: str) -> list[str]:
        return list(dict.fromkeys(re.findall(r"[\w]+", query.casefold(), flags=re.UNICODE)))[:48]

    def health(self) -> dict[str, Any]:
        try:
            db = self._connection()
            count = int(db.execute("SELECT COUNT(*) FROM dam_regions").fetchone()[0])
            ready = self._ready(db)
        except (OSError, sqlite3.Error):
            count = 0
            ready = False
        return {
            "ready": ready,
            "documents": count,
            "database": str(self.database_path),
            "version": self.VERSION,
            "schema_version": self.SCHEMA_VERSION,
            "fingerprint": self._fingerprint(),
            "frame_metadata_fingerprint": self._frame_fingerprint(),
        }

    def search(self, queries: list[str], top_k: int) -> dict[str, dict[str, Any]]:
        if len(queries) != 6 or not 1 <= int(top_k) <= 2_000:
            raise ValueError("DAM BM25 retrieval requires six queries and top_k in 1..2000")
        db = self._connection()
        if not self._ready(db):
            raise RuntimeError("Branch-2 BM25 index is not ready; rerun prepare_branch2.py")
        hits: dict[str, dict[str, Any]] = {}
        for role, query in zip(
            ("original", "entity", "action", "context", "synonym", "keyword"), queries, strict=True
        ):
            tokens = self._tokens(query)
            if not tokens:
                continue
            expression = " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens)
            with self._lock:
                rows = db.execute(
                    "SELECT r.*, bm25(dam_fts) AS bm25_raw FROM dam_fts JOIN dam_regions r ON r.id=dam_fts.rowid WHERE dam_fts MATCH ? ORDER BY bm25_raw LIMIT ?",
                    (expression, top_k),
                ).fetchall()
            for rank, row in enumerate(rows, 1):
                frame_uid = str(row["frame_uid"])
                score = 1.0 / (60.0 + rank)
                bm25_raw = float(row["bm25_raw"])
                if not math.isfinite(bm25_raw):
                    raise ValueError("SQLite BM25 returned a non-finite score")
                current = hits.get(frame_uid)
                evidence = {
                    "rank": rank,
                    "rank_score": score,
                    "bm25_raw": bm25_raw,
                    "region_id": row["region_id"],
                    "language": "en",
                    "observed": True,
                }
                if current is not None:
                    role_scores = current.setdefault("sparse_query_scores", {})
                    previous_evidence = role_scores.get(role)
                    if previous_evidence is None or float(previous_evidence["rank_score"]) < score:
                        role_scores[role] = evidence
                if current is not None and current["sparse_raw"] >= score:
                    continue
                query_scores = (
                    dict(current.get("sparse_query_scores", {})) if current is not None else {}
                )
                query_scores[role] = evidence
                payload = {
                    "point_id": int(row["parent_point_id"]),
                    "video_id": row["video_id"],
                    "frame_idx": int(row["frame_idx"]),
                    "keyframe_n": int(row["keyframe_n"]),
                    "pts_time_s": float(row["pts_time_s"]),
                    "fps": float(row["fps"]),
                    "frame_uid": frame_uid,
                    "image_relpath": str(row["image_relpath"]),
                }
                frame = base_frame(payload, score=score, rank=rank, score_type="bm25_rank")
                frame.update(
                    {
                        "frame_uid": frame_uid,
                        "global_idx": int(row["parent_point_id"]),
                        "sparse_raw": score,
                        "sparse_observed": True,
                        "sparse_rank": rank,
                        "sparse_bm25_raw": bm25_raw,
                        "sparse_best_query_role": role,
                        "sparse_best_query_language": "en",
                        "sparse_query_scores": query_scores,
                        "sparse_winner": {
                            "region_id": row["region_id"],
                            "class_entity": row["class_entity"],
                            "description_en": row["description_en"],
                            "bbox": json.loads(row["bbox_json"]),
                            "query_role": role,
                            "query_language": "en",
                            "rank": rank,
                            "bm25_raw": bm25_raw,
                        },
                    }
                )
                hits[frame_uid] = frame
        for frame in hits.values():
            scores = frame.setdefault("sparse_query_scores", {})
            for role in ("original", "entity", "action", "context", "synonym", "keyword"):
                scores.setdefault(
                    role,
                    {
                        "rank": None,
                        "rank_score": None,
                        "bm25_raw": None,
                        "region_id": None,
                        "language": "en",
                        "observed": False,
                    },
                )
        return hits
