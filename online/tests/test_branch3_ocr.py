"""Branch-3 OCR routing and output-gate contracts."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sqlite3
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from online.src.retrieval.branches.branch3.ocr import Branch3OcrSearch
from online.src.retrieval.modalities import ocr as ocr_module
from online.src.retrieval.modalities.lexical import (
    _ordered_lexical_bigrams,
    fold_text,
    ordered_lexical_tokens,
    query_tokens,
)

ROLES = ("original", "entity", "action", "context", "synonym", "keyword")


def bundle() -> dict[str, object]:
    return {
        "schema_version": "branch1.query.v1",
        "queries": [{"role": role, "vi": f"vi {role}", "en": f"en {role}"} for role in ROLES],
    }


class _FakeOcrIndex:
    def __init__(self) -> None:
        self.received: dict[str, object] | None = None

    def health(self):
        return {"ready": True, "production_ready": False}

    def assert_ready(self):
        return None

    def search_many(self, query_by_stream, *, per_stream_top_k, final_top_k):
        self.received = {
            "streams": query_by_stream,
            "per_stream_top_k": per_stream_top_k,
            "final_top_k": final_top_k,
        }
        return {
            "candidate_frame_count": 2,
            "stream_counts": {stream: 1 for stream in query_by_stream},
            "results": [
                {"frame_uid": "L21_V001:4", "ocr_normalized_score": 0.8},
                {"frame_uid": "L21_V001:8", "ocr_normalized_score": 0.7},
            ],
            "timing": {"total_ms": 1.0},
        }


class _SingleStreamOcrIndex:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def assert_ready(self) -> None:
        return None

    def search_many(self, query_by_stream, *, per_stream_top_k, final_top_k, _allow_single=False):
        self.calls.append(
            {
                "streams": query_by_stream,
                "per_stream_top_k": per_stream_top_k,
                "final_top_k": final_top_k,
                "allow_single": _allow_single,
            }
        )
        return {"results": [{"frame_uid": "L21_V001:4"}]}


class Branch3OcrContractTests(unittest.TestCase):
    def test_search_many_signature_has_no_readiness_bypass(self) -> None:
        self.assertNotIn(
            "_ready_asserted",
            inspect.signature(ocr_module.OcrFtsIndex.search_many).parameters,
        )

    def test_missing_connection_cannot_return_empty_success(self) -> None:
        index = object.__new__(ocr_module.OcrFtsIndex)
        index._lock = threading.RLock()
        index._connection = None
        index._load_manifest_locked = Mock(return_value={})
        index._fast_ready_locked = Mock(return_value=True)
        with self.assertRaisesRegex(RuntimeError, "connection is unavailable"):
            index.search_many(
                {"legacy": "visible title"},
                per_stream_top_k=10,
                final_top_k=10,
                _allow_single=True,
            )

    def test_search_fast_gate_does_not_scan_raw_ocr_sources(self) -> None:
        """Search uses the published SQLite proof, not the 873 JSONL audit."""

        index = object.__new__(ocr_module.OcrFtsIndex)
        index._lock = threading.RLock()
        index._connection = object()
        index._load_manifest_locked = Mock(return_value={})
        index._fast_ready_locked = Mock(return_value=True)
        index._source_inventory_locked = Mock(
            side_effect=AssertionError("raw source audit must not run during search")
        )
        index._search_many_locked = Mock(return_value={"results": [], "candidate_frame_count": 0})
        result = index.search_many(
            {"legacy": "visible title"},
            per_stream_top_k=10,
            final_top_k=10,
            _allow_single=True,
        )
        self.assertEqual(result["results"], [])
        index._source_inventory_locked.assert_not_called()

    def test_query_validation_error_is_not_relabelled_as_readiness_failure(self) -> None:
        index = object.__new__(ocr_module.OcrFtsIndex)
        index._lock = threading.RLock()
        index._connection = object()
        index._load_manifest_locked = Mock(return_value={})
        index._fast_ready_locked = Mock(return_value=True)
        index._search_many_locked = Mock(side_effect=ValueError("bad query"))
        with self.assertRaisesRegex(ValueError, "bad query"):
            index.search_many(
                {"legacy": "visible title"},
                per_stream_top_k=10,
                final_top_k=10,
                _allow_single=True,
            )

    def test_ocr_score_weights_are_locked_to_55_30_15(self) -> None:
        score = ocr_module._ocr_query_score(0.8, 0.5, 0.25)
        self.assertAlmostEqual(score, 0.55 * 0.8 + 0.30 * 0.5 + 0.15 * 0.25)

    def test_lookup_many_deduplicates_and_runs_one_readiness_pass(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.execute(
            "CREATE TABLE ocr_frames(frame_uid TEXT PRIMARY KEY, full_text TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO ocr_frames(frame_uid, full_text) VALUES (?, ?)",
            [("L21_V001:1", "one"), ("L21_V001:2", "two")],
        )
        index = object.__new__(ocr_module.OcrFtsIndex)
        index._lock = threading.RLock()
        index._connection = connection
        index._load_manifest_locked = Mock(return_value={})
        index._fast_ready_locked = Mock(return_value=True)
        try:
            result = index.lookup_many(["L21_V001:1", "L21_V001:1", "L21_V001:2"])
            self.assertEqual(result, {"L21_V001:1": "one", "L21_V001:2": "two"})
            index._fast_ready_locked.assert_called_once()
        finally:
            connection.close()

    def test_database_inspector_rejects_orphan_fts_rows(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA user_version = {ocr_module.OCR_SQLITE_USER_VERSION}")
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
        connection.execute(
            "INSERT INTO ocr_frames VALUES (1, 'L21_V001:0', 1, 'L21_V001', 1, 0, 0.0, 25.0, 'frame.jpg', 'hello', 'hello')"
        )
        connection.execute("CREATE VIRTUAL TABLE ocr_fts USING fts5(full_text_search)")
        connection.execute("INSERT INTO ocr_fts(rowid, full_text_search) VALUES (1, 'hello')")
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
        fts_state = ocr_module._fts_content_fingerprint_for_connection(connection)
        connection.execute(
            "INSERT INTO ocr_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ocr_module.OCR_INDEX_SCHEMA_VERSION,
                ocr_module.OCR_SQLITE_USER_VERSION,
                "build",
                "source",
                "canonical",
                ocr_module.OCR_MAPPING_STRATEGY,
                1,
                1,
                "now",
                fts_state["fingerprint"],
                ocr_module.OCR_LEXICAL_CONTRACT_VERSION,
            ),
        )
        try:
            with (
                patch.object(ocr_module, "OCR_EXPECTED_FRAMES", 1),
                patch.object(ocr_module, "OCR_EXPECTED_SOURCE_FILES", 1),
            ):
                self.assertTrue(ocr_module._database_state_for_connection(connection)["ready"])
                connection.execute(
                    "INSERT INTO ocr_fts(rowid, full_text_search) VALUES (99, 'orphan')"
                )
                state = ocr_module._database_state_for_connection(connection)
                self.assertFalse(state["ready"])
                self.assertEqual(state["fts_orphaned"], 1)
                connection.execute("DELETE FROM ocr_fts WHERE rowid = 99")
                connection.execute(
                    "INSERT INTO ocr_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        ocr_module.OCR_INDEX_SCHEMA_VERSION,
                        ocr_module.OCR_SQLITE_USER_VERSION,
                        "build-two",
                        "source",
                        "canonical",
                        ocr_module.OCR_MAPPING_STRATEGY,
                        1,
                        1,
                        "now",
                        fts_state["fingerprint"],
                        ocr_module.OCR_LEXICAL_CONTRACT_VERSION,
                    ),
                )
                state = ocr_module._database_state_for_connection(connection)
                self.assertFalse(state["ready"])
                self.assertEqual(state["meta_rows"], 2)
                connection.execute("DELETE FROM ocr_meta WHERE build_id = 'build-two'")
                connection.execute(
                    "UPDATE ocr_frames SET pts_time_s = ?, image_relpath = ? WHERE id = 1",
                    (float("inf"), "C:\\frames\\outside.jpg"),
                )
                state = ocr_module._database_state_for_connection(connection)
                self.assertFalse(state["ready"])
                self.assertGreater(state["invalid_identity"], 0)
                self.assertGreater(state["invalid_paths"], 0)
        finally:
            connection.close()

    def test_fts_content_validator_rejects_same_rowid_with_wrong_text(self) -> None:
        connection = sqlite3.connect(":memory:")
        connection.execute(
            "CREATE TABLE ocr_frames(id INTEGER PRIMARY KEY, frame_uid TEXT, full_text_search TEXT)"
        )
        connection.execute("INSERT INTO ocr_frames VALUES (1, 'L21_V001:0', 'xin chao')")
        connection.execute("CREATE VIRTUAL TABLE ocr_fts USING fts5(full_text_search)")
        connection.execute("INSERT INTO ocr_fts(rowid, full_text_search) VALUES (1, 'xin chao')")
        try:
            valid = ocr_module._fts_content_fingerprint_for_connection(connection)
            self.assertTrue(valid["verified"])
            connection.execute("UPDATE ocr_fts SET full_text_search = 'wrong text' WHERE rowid = 1")
            invalid = ocr_module._fts_content_fingerprint_for_connection(connection)
            self.assertFalse(invalid["verified"])
            self.assertEqual(invalid["mismatch_count"], 1)
        finally:
            connection.close()

    def test_service_sends_twelve_bilingual_streams_and_fixed_gate(self) -> None:
        index = _FakeOcrIndex()
        service = Branch3OcrSearch(index, threading.Lock())
        response = service.execute(bundle(), 2000, 500)
        assert index.received is not None
        streams = index.received["streams"]
        self.assertEqual(len(streams), 12)
        self.assertEqual(
            set(streams),
            {f"{role}:{language}" for role in ROLES for language in ("vi", "en")},
        )
        self.assertEqual(index.received["per_stream_top_k"], 2000)
        self.assertEqual(index.received["final_top_k"], 500)
        self.assertEqual(response["stream_count"], 12)
        self.assertEqual(response["gate_top_k"], 500)
        self.assertTrue(response["future_fusion_eligible"])

    def test_service_rejects_output_above_the_500_frame_gate(self) -> None:
        service = Branch3OcrSearch(_FakeOcrIndex(), threading.Lock())
        with self.assertRaisesRegex(ValueError, "between 1 and 500"):
            service.execute(bundle(), 2000, 501)

    def test_service_defensively_slices_an_oversized_adapter_response(self) -> None:
        class _OversizedIndex(_FakeOcrIndex):
            def search_many(self, query_by_stream, *, per_stream_top_k, final_top_k):
                return {
                    "candidate_frame_count": 600,
                    "stream_counts": {},
                    "results": [{"frame_uid": f"L21_V001:{index}"} for index in range(600)],
                }

        response = Branch3OcrSearch(_OversizedIndex(), threading.Lock()).execute(
            bundle(), 2000, 500
        )
        self.assertEqual(response["result_count"], 500)
        self.assertEqual(len(response["results"]), 500)
        self.assertEqual(response["candidate_count_before_gate"], 600)

    def test_lexical_helper_preserves_boundaries_and_repeated_tokens(self) -> None:
        self.assertEqual(fold_text("Đặng Văn A"), "dang van a")
        self.assertEqual(ordered_lexical_tokens("one in two"), ["one", "in", "two"])
        self.assertEqual(query_tokens("one in two"), ["one", "two"])
        self.assertEqual(_ordered_lexical_bigrams("one in two"), [])
        self.assertEqual(_ordered_lexical_bigrams("xe A đỏ"), [])
        self.assertEqual(
            _ordered_lexical_bigrams("alpha alpha beta"),
            [("alpha", "alpha"), ("alpha", "beta")],
        )

    def test_empty_ocr_text_remains_a_valid_canonical_row(self) -> None:
        # One, two and three successive UTF-8/single-byte decoding mistakes
        # must all converge to the original Vietnamese text.
        self.assertEqual(
            ocr_module.repair_mojibake(
                "Xin ch" + chr(0xC3) + chr(0x83) + chr(0xC2) + chr(0xA0) + "o"
            ),
            "Xin chào",
        )
        self.assertEqual(
            ocr_module.repair_mojibake("Xin ch" + chr(0xC3) + chr(0xA0) + "o"),
            "Xin ch" + chr(0xE0) + "o",
        )
        self.assertEqual(
            ocr_module.repair_mojibake(
                "Xin ch"
                + chr(0xC3)
                + chr(0x0192)
                + chr(0xC6)
                + chr(0x2019)
                + chr(0xC3)
                + chr(0x201A)
                + chr(0xC2)
                + chr(0xA0)
                + "o"
            ),
            "Xin chào",
        )
        self.assertEqual(ocr_module.repair_mojibake("Xin chào"), "Xin chào")
        parsed = ocr_module._parse_ocr_row(
            {
                "video_id": "L21_V001",
                "frame_uid": "L21_V001:4",
                "keyframe_n": 1,
                "frame_idx": 4,
                "pts_time_s": 0.1,
                "full_text": "",
            },
            Path("L21_V001.jsonl"),
            1,
        )
        self.assertEqual(parsed["full_text"], "")
        self.assertEqual(parsed["full_text_search"], "")
        with self.assertRaisesRegex(ValueError, "full_text"):
            ocr_module._parse_ocr_row(
                {
                    "video_id": "L21_V001",
                    "frame_uid": "L21_V001:4",
                    "keyframe_n": 1,
                    "frame_idx": 4,
                    "pts_time_s": 0.1,
                    "full_text": None,
                },
                Path("L21_V001.jsonl"),
                2,
            )
        with self.assertRaisesRegex(ValueError, "replacement character"):
            ocr_module._parse_ocr_row(
                {
                    "video_id": "L21_V001",
                    "frame_uid": "L21_V001:4",
                    "keyframe_n": 1,
                    "frame_idx": 4,
                    "pts_time_s": 0.1,
                    "full_text": "Xin ch\ufffdo",
                },
                Path("L21_V001.jsonl"),
                3,
            )

    def test_source_gate_rejects_wrong_file_count(self) -> None:
        with TemporaryDirectory() as directory:
            transcripts = Path(directory) / "ocr_transcripts"
            transcripts.mkdir()
            transcripts.joinpath("L21_V001.jsonl").write_text(
                json.dumps(
                    {
                        "video_id": "L21_V001",
                        "frame_uid": "L21_V001:4",
                        "keyframe_n": 1,
                        "frame_idx": 4,
                        "pts_time_s": 0.1,
                        "full_text": "text",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with (
                patch.object(ocr_module, "OCR_EXPECTED_SOURCE_FILES", 2),
                patch.object(ocr_module, "OCR_EXPECTED_FRAMES", 1),
                self.assertRaisesRegex(ValueError, "Expected 2 OCR source files"),
            ):
                ocr_module.validate_ocr_sources(transcripts)

    def test_source_gate_rejects_duplicate_frame_uid(self) -> None:
        with TemporaryDirectory() as directory:
            transcripts = Path(directory) / "ocr_transcripts"
            transcripts.mkdir()
            row = {
                "video_id": "L21_V001",
                "frame_uid": "L21_V001:4",
                "keyframe_n": 1,
                "frame_idx": 4,
                "pts_time_s": 0.1,
                "full_text": "text",
            }
            transcripts.joinpath("L21_V001.jsonl").write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            with (
                patch.object(ocr_module, "OCR_EXPECTED_SOURCE_FILES", 1),
                patch.object(ocr_module, "OCR_EXPECTED_FRAMES", 2),
                self.assertRaisesRegex(ValueError, "Duplicate OCR frame_uid"),
            ):
                ocr_module.validate_ocr_sources(transcripts)

    def test_artifact_status_is_bound_to_instance_and_detects_content_mismatch(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "ocr.sqlite3"
            artifact.write_bytes(b"actual")
            index = ocr_module.OcrFtsIndex(root / "ocr_transcripts", artifact)
            matched, stat_matches, fingerprint_present = index._artifact_status(
                {
                    "path": artifact.name,
                    "size": artifact.stat().st_size,
                    "mtime_ns": artifact.stat().st_mtime_ns,
                    "sha256": "0" * 64,
                },
                relative_root=root,
            )
            self.assertFalse(matched)
            self.assertTrue(stat_matches)
            self.assertTrue(fingerprint_present)
            escaped, escaped_stat, escaped_fingerprint = index._artifact_status(
                {
                    "path": "../outside.sqlite3",
                    "size": 1,
                    "mtime_ns": 1,
                    "sha256": "0" * 64,
                },
                relative_root=root,
            )
            self.assertFalse(escaped)
            self.assertFalse(escaped_stat)
            self.assertTrue(escaped_fingerprint)
            index.close()

    def test_stat_matching_artifact_skips_hash_and_changed_artifact_rehashes_once(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "ocr.sqlite3"
            artifact.write_bytes(b"actual")
            record = {
                "path": artifact.name,
                "size": artifact.stat().st_size,
                "mtime_ns": artifact.stat().st_mtime_ns,
                "sha256": hashlib.sha256(b"actual").hexdigest(),
            }
            index = ocr_module.OcrFtsIndex(root / "ocr_transcripts", artifact)
            try:
                with patch.object(ocr_module, "_sha256_file", side_effect=AssertionError):
                    matched, stat_matches, _ = index._artifact_status(
                        record, relative_root=root, verify_hash=False
                    )
                self.assertTrue(matched)
                self.assertTrue(stat_matches)

                artifact.write_bytes(b"changed")
                real_sha256 = hashlib.sha256(b"changed").hexdigest()
                with patch.object(
                    ocr_module,
                    "_sha256_file",
                    return_value=real_sha256,
                ) as digest:
                    matched, stat_matches, _ = index._artifact_status(
                        record, relative_root=root, verify_hash=False
                    )
                self.assertFalse(matched)
                self.assertFalse(stat_matches)
                # macOS resolves the /var compatibility symlink to /private/var.
                digest.assert_called_once_with(artifact.resolve())
            finally:
                index.close()

    def test_health_malformed_or_missing_artifacts_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            index = ocr_module.OcrFtsIndex(root / "ocr_transcripts", root / "ocr.sqlite3")
            payload = index.health()
            self.assertFalse(payload["ready"])
            self.assertFalse(payload["production_ready"])
            self.assertTrue(payload["fail_closed"])
            self.assertIn("error", payload)
            index.close()

    def test_manifest_identity_is_deterministic_and_rejects_old_schema(self) -> None:
        source_records = [
            {
                "path": f"ocr_transcripts/L21_V{index:03d}.jsonl",
                "size": index + 1,
                "mtime_ns": index + 10,
                "sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                "row_count": 247084 if index == 1 else 1,
                "video_id": f"L21_V{index:03d}",
            }
            for index in range(1, ocr_module.OCR_EXPECTED_SOURCE_FILES + 1)
        ]
        source_fp = ocr_module.source_fingerprint(source_records)
        canonical_fp = "c" * 64
        fts_fp = "a" * 64
        manifest = {
            "schema_version": ocr_module.OCR_INDEX_SCHEMA_VERSION,
            "sqlite_user_version": ocr_module.OCR_SQLITE_USER_VERSION,
            "status": "ready",
            "passed": True,
            "source_files": source_records,
            "source_file_count": ocr_module.OCR_EXPECTED_SOURCE_FILES,
            "source_directory": ocr_module.OCR_SOURCE_DIR_NAME,
            "source_fingerprint": source_fp,
            "fts_content_fingerprint": fts_fp,
            "lexical_contract_version": ocr_module.OCR_LEXICAL_CONTRACT_VERSION,
            "database": {
                "path": "ocr.sqlite3",
                "size": 1,
                "mtime_ns": 1,
                "sha256": "d" * 64,
            },
            "canonical_metadata": {
                "path": ocr_module.OCR_CANONICAL_RELATIVE_PATH,
                "size": 1,
                "mtime_ns": 1,
                "sha256": canonical_fp,
            },
            "canonical_fingerprint": canonical_fp,
            "mapping": {
                "strategy": ocr_module.OCR_MAPPING_STRATEGY,
                "frame_count": ocr_module.OCR_EXPECTED_FRAMES,
            },
            "counts": {
                "frame_count": ocr_module.OCR_EXPECTED_FRAMES,
                "fts_frame_count": ocr_module.OCR_EXPECTED_FRAMES,
                "mapped_frame_count": ocr_module.OCR_EXPECTED_FRAMES,
                "video_count": ocr_module.OCR_EXPECTED_SOURCE_FILES,
            },
            "frame_count": ocr_module.OCR_EXPECTED_FRAMES,
            "video_count": ocr_module.OCR_EXPECTED_SOURCE_FILES,
            "build_id": ocr_module.build_id_for(
                source_fingerprint_value=source_fp,
                canonical_fingerprint_value=canonical_fp,
                frame_count=ocr_module.OCR_EXPECTED_FRAMES,
                video_count=ocr_module.OCR_EXPECTED_SOURCE_FILES,
                fts_content_fingerprint_value=fts_fp,
            ),
        }
        self.assertTrue(ocr_module._manifest_build_identity_valid(manifest))
        original_sha = manifest["source_files"][0]["sha256"]
        manifest["source_files"][0]["sha256"] = "f" * 64
        # The top-level aggregate/build id are intentionally unchanged: the
        # record-level edit alone must invalidate the identity chain.
        self.assertFalse(ocr_module._manifest_build_identity_valid(manifest))
        manifest["source_files"][0]["sha256"] = original_sha
        original_video_id = manifest["source_files"][0]["video_id"]
        manifest["source_files"][0]["video_id"] = "WRONG_VIDEO"
        self.assertFalse(ocr_module._manifest_build_identity_valid(manifest))
        manifest["source_files"][0]["video_id"] = original_video_id
        manifest["source_fingerprint"] = "e" * 64
        self.assertFalse(ocr_module._manifest_build_identity_valid(manifest))
        manifest["source_fingerprint"] = source_fp
        manifest["canonical_metadata"]["sha256"] = "f" * 64
        self.assertFalse(ocr_module._manifest_build_identity_valid(manifest))
        manifest["canonical_fingerprint"] = canonical_fp
        manifest["canonical_metadata"]["sha256"] = canonical_fp
        manifest["sqlite_user_version"] = 1
        self.assertFalse(ocr_module._manifest_build_identity_valid(manifest))
        manifest["sqlite_user_version"] = ocr_module.OCR_SQLITE_USER_VERSION
        manifest["schema_version"] = "branch3.ocr-index.v1"
        self.assertFalse(ocr_module._manifest_build_identity_valid(manifest))
        manifest["schema_version"] = "branch3.ocr-index.v2"
        self.assertFalse(ocr_module._manifest_build_identity_valid(manifest))
        manifest["schema_version"] = ocr_module.OCR_INDEX_SCHEMA_VERSION
        manifest["sqlite_user_version"] = 2
        self.assertFalse(ocr_module._manifest_build_identity_valid(manifest))

    def test_compatibility_search_uses_one_stream_and_shared_lock(self) -> None:
        index = _SingleStreamOcrIndex()
        service = Branch3OcrSearch(index, threading.Lock())
        result = service.execute_single("visible title", 20)
        self.assertEqual(result[0]["frame_uid"], "L21_V001:4")
        self.assertEqual(len(index.calls), 1)
        self.assertEqual(index.calls[0]["streams"], {"legacy": "visible title"})
        self.assertEqual(index.calls[0]["per_stream_top_k"], 500)
        self.assertTrue(index.calls[0]["allow_single"])
        self.assertNotIn("ready_asserted", index.calls[0])

        # The compatibility path must also cap its internal request when the
        # caller asks for the full 500-frame OCR pool.
        service.execute_single("visible title", 500)
        self.assertEqual(index.calls[-1]["per_stream_top_k"], 2_000)

    def test_health_never_promotes_unverified_ocr_to_production(self) -> None:
        class _HealthIndex:
            def health(self):
                return {"ready": True, "production_ready": True, "revision_verified": False}

        payload = Branch3OcrSearch(_HealthIndex(), threading.Lock()).health()
        self.assertTrue(payload["ready"])
        self.assertFalse(payload["production_ready"])

    def test_v3_fixture_reopens_after_atomic_database_publication(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            transcripts = root / "ocr_transcripts"
            canonical_path = root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl"
            transcripts.mkdir()
            canonical_path.parent.mkdir(parents=True)
            canonical = {
                "video_id": "L21_V001",
                "keyframe_n": 1,
                "frame_idx": 4,
                "pts_time_s": 0.1333,
                "fps": 30.0,
                "frame_uid": "L21_V001:4",
                "image_relpath": "keyframes/L21_V001/00000004.jpg",
                "point_id": 1,
            }
            canonical_path.write_text(json.dumps(canonical) + "\n", encoding="utf-8")
            transcripts.joinpath("L21_V001.jsonl").write_text(
                json.dumps(
                    {
                        "video_id": "L21_V001",
                        "frame_uid": "L21_V001:4",
                        "keyframe_n": 1,
                        "frame_idx": 4,
                        "pts_time_s": 0.1333,
                        "full_text": "Xin chào",
                        **{
                            "full_text": "Xin ch"
                            + chr(0xC3)
                            + chr(0x83)
                            + chr(0xC2)
                            + chr(0xA0)
                            + "o",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            database = root / "ocr.sqlite3"
            manifest = root / "branch3_ocr_manifest.json"
            with (
                patch.object(ocr_module, "OCR_EXPECTED_SOURCE_FILES", 1),
                patch.object(ocr_module, "OCR_EXPECTED_FRAMES", 1),
            ):
                ocr_module.build_ocr_index(
                    transcripts,
                    database,
                    data_root=root,
                    manifest_path=manifest,
                )
                index = ocr_module.OcrFtsIndex(
                    transcripts,
                    database,
                    manifest_path=manifest,
                    canonical_metadata_path=canonical_path,
                )
                try:
                    first = index.health()
                    self.assertTrue(first["ready"])
                    self.assertEqual(first["frames"], 1)
                    self.assertEqual(first["manifest_build_id"], first["internal_build_id"])
                    self.assertTrue(first["canonical_identity_matches"])
                    self.assertEqual(first["canonical_identity_rows"], 1)
                    self.assertTrue(first["fts_content_verified"])
                    self.assertEqual(
                        first["lexical_contract_version"],
                        ocr_module.OCR_LEXICAL_CONTRACT_VERSION,
                    )
                    source_audit = index.health(audit_sources=True)
                    self.assertFalse(source_audit["source_stale"])
                    source_file = transcripts.joinpath("L21_V001.jsonl")
                    source_file.write_text(
                        source_file.read_text(encoding="utf-8").replace(
                            "Xin ch", "Xin ch altered", 1
                        ),
                        encoding="utf-8",
                    )
                    # A source audit is intentionally TTL-cached; invalidate
                    # the fixture cache to model the next scheduled audit.
                    index._source_inventory_cache = None
                    drift = index.health(audit_sources=True)
                    self.assertTrue(drift["ready"])
                    self.assertTrue(drift["source_stale"])
                    self.assertFalse(drift["source_drift_is_blocking"])

                    streams = {
                        f"{role}:{language}": "xin chao"
                        for role in ROLES
                        for language in ("vi", "en")
                    }
                    search = index.search_many(
                        streams,
                        per_stream_top_k=2_000,
                        final_top_k=500,
                    )
                    self.assertEqual(search["candidate_frame_count"], 1)
                    result = search["results"][0]
                    self.assertEqual(result["frame_uid"], "L21_V001:4")
                    self.assertEqual(result["full_text"], "Xin chào")
                    self.assertEqual(len(result["ocr_stream_provenance"]), 12)

                    # A database row can remain schema-valid while pointing
                    # at a different canonical frame.  The generation-time
                    # streaming inspector must reject that content mismatch,
                    # not merely rely on shape/count checks.
                    writer = sqlite3.connect(database)
                    try:
                        writer.execute(
                            "UPDATE ocr_frames SET image_relpath = ? WHERE id = 1",
                            ("keyframes/L21_V001/00000099.jpg",),
                        )
                        writer.commit()
                    finally:
                        writer.close()
                    mismatched = index.health()
                    self.assertFalse(mismatched["ready"])
                    self.assertIsNone(mismatched["opened_build_id"])

                    ocr_module.build_ocr_index(
                        transcripts,
                        database,
                        data_root=root,
                        manifest_path=manifest,
                    )
                    second = index.health()
                    self.assertTrue(second["ready"])
                    self.assertGreaterEqual(second["connection_generation"], 2)
                    self.assertTrue(second["connection_reopened"])

                    invalid_replacement = root / "ocr.invalid.staging"
                    invalid_connection = sqlite3.connect(invalid_replacement)
                    invalid_connection.execute("CREATE TABLE unrelated(value TEXT)")
                    invalid_connection.commit()
                    invalid_connection.close()
                    os.replace(invalid_replacement, database)
                    invalid = index.health()
                    self.assertFalse(invalid["ready"])
                    self.assertIsNone(invalid["opened_build_id"])

                    live_database = database.read_bytes()
                    live_manifest = manifest.read_bytes()
                    with (
                        patch.object(
                            ocr_module,
                            "_database_state_for_connection",
                            side_effect=ValueError("synthetic staging validation failure"),
                        ),
                        self.assertRaisesRegex(ValueError, "synthetic staging"),
                    ):
                        ocr_module.build_ocr_index(
                            transcripts,
                            database,
                            data_root=root,
                            manifest_path=manifest,
                        )
                    self.assertEqual(database.read_bytes(), live_database)
                    self.assertEqual(manifest.read_bytes(), live_manifest)
                    self.assertFalse(list(root.glob(".ocr.sqlite3.staging.*")))
                    self.assertFalse(list(root.glob(".branch3_ocr_manifest.json.staging")))
                finally:
                    # Always release the SQLite/snapshot handle before
                    # TemporaryDirectory cleanup on Windows, including when
                    # an assertion or publication step fails.
                    index.close()


if __name__ == "__main__":
    unittest.main()
