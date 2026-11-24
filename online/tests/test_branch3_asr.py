"""ASR Branch-3 behavior tests (not executed as part of implementation handoff)."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from online.src.retrieval.branches.branch3.asr import Branch3AsrSearch
from online.src.retrieval.modalities.asr import (
    ASR_INDEX_SCHEMA_VERSION,
    ASR_SQLITE_USER_VERSION,
    AsrFtsIndex,
    _ordered_lexical_bigrams,
    build_id_for,
    fold_text,
    ordered_lexical_tokens,
    query_tokens,
    source_fingerprint,
    validate_asr_sources,
)
from online.src.retrieval.modalities.local import CpuQdrantSearch


class _FakeAsrIndex:
    def __init__(self) -> None:
        self.received = None

    def health(self):
        return {"ready": True, "production_ready": False}

    def search_many(self, query_by_role, *, per_stream_top_k, final_top_k):
        self.received = (query_by_role, per_stream_top_k, final_top_k)
        return {
            "candidate_segment_count": 1,
            "candidate_frame_count": 1,
            "stream_counts": {role: 1 for role in query_by_role},
            "timing": {"total_ms": 1.0},
            "results": [{"frame_uid": "L21_V001:4", "score": 0.5}],
        }


class Branch3AsrContractTests(unittest.TestCase):
    def test_source_gate_accepts_a_completed_video_with_zero_segments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            segments_dir = Path(directory) / "asr_segments"
            segments_dir.mkdir()

            populated = {
                "schema_version": "aic26.asr_segments.v1",
                "segment_id": "L00_V001:seg_0",
                "video_id": "L00_V001",
                "start_ms": 0,
                "end_ms": 1000,
                "transcript_normalized": "xin chao",
                "language": "vi",
            }
            (segments_dir / "L00_V001.jsonl").write_text(
                json.dumps(populated, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            (segments_dir / "L00_V002.jsonl").write_text("", encoding="utf-8")
            for video_id, segment_count in (("L00_V001", 1), ("L00_V002", 0)):
                (segments_dir / f"{video_id}.manifest.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "aic26.asr_manifest.v1",
                            "video_id": video_id,
                            "status": "completed",
                            "segment_count": segment_count,
                            "model_id": "test-model",
                            "engine": "test-engine",
                        }
                    ),
                    encoding="utf-8",
                )

            with (
                patch("online.src.retrieval.modalities.asr.EXPECTED_ASR_VIDEOS", 2),
                patch("online.src.retrieval.modalities.asr.EXPECTED_ASR_SEGMENTS", 1),
            ):
                facts = validate_asr_sources(segments_dir)

            self.assertEqual(facts["video_count"], 2)
            self.assertEqual(facts["indexed_video_count"], 1)
            self.assertEqual(facts["empty_video_count"], 1)
            self.assertEqual(facts["empty_video_ids"], ["L00_V002"])

    def test_vietnamese_fold_and_tokens_are_diacritic_insensitive(self) -> None:
        self.assertEqual(fold_text("Đặng Văn A"), "dang van a")
        self.assertEqual(
            query_tokens("các thông tin về Đà Nẵng"), ["thong", "tin", "ve", "da", "nang"]
        )

    def test_service_sends_six_vietnamese_streams_and_returns_contract(self) -> None:
        index = _FakeAsrIndex()
        service = Branch3AsrSearch(index, threading.Lock())
        bundle = {
            "schema_version": "branch1.query.v1",
            "queries": [
                {"role": role, "vi": f"câu hỏi {role}", "en": f"query {role}"}
                for role in ("original", "entity", "action", "context", "synonym", "keyword")
            ],
        }
        response = service.execute(bundle, 2000, 500)
        self.assertEqual(response["schema_version"], "branch3.asr.result.v1")
        self.assertEqual(len(index.received[0]), 12)
        self.assertEqual(
            set(index.received[0]),
            {
                f"{role}:{language}"
                for role in ("original", "entity", "action", "context", "synonym", "keyword")
                for language in ("vi", "en")
            },
        )
        self.assertEqual(index.received[1:], (2000, 500))
        self.assertEqual(response["result_count"], 1)

    def test_index_accepts_all_twelve_bilingual_stream_roles(self) -> None:
        index = AsrFtsIndex(Path("/missing/asr_segments"), Path("/missing/asr.sqlite3"))
        index.health = lambda: {"ready": True}  # type: ignore[method-assign]
        index._query_stream = lambda _query, _limit: ([], ["searchable"])  # type: ignore[method-assign]
        streams = {
            f"{role}:{language}": f"searchable {role} {language}"
            for role in ("original", "entity", "action", "context", "synonym", "keyword")
            for language in ("vi", "en")
        }

        result = index.search_many(streams)

        self.assertEqual(set(result["stream_counts"]), set(streams))
        self.assertEqual(result["results"], [])

    def test_ordered_tokens_preserve_real_adjacency_and_duplicates(self) -> None:
        self.assertEqual(ordered_lexical_tokens("one in two"), ["one", "in", "two"])
        self.assertEqual(ordered_lexical_tokens("alpha alpha beta"), ["alpha", "alpha", "beta"])
        self.assertEqual(ordered_lexical_tokens("xe A đỏ"), ["xe", "a", "do"])
        self.assertEqual(query_tokens("one in two"), ["one", "two"])
        self.assertEqual(_ordered_lexical_bigrams("one in two"), [])
        self.assertEqual(_ordered_lexical_bigrams("xe A đỏ"), [])

    def test_bigram_does_not_bridge_removed_stopwords(self) -> None:
        index = AsrFtsIndex(Path("/missing/asr_segments"), Path("/missing/asr.sqlite3"))
        index._runtime_ready_fast_locked = lambda: True  # type: ignore[method-assign]
        index.health = lambda: {"ready": True}  # type: ignore[method-assign]
        index._query_stream = lambda _query, _limit: (  # type: ignore[method-assign]
            [
                {
                    "segment_id": "seg-1",
                    "video_id": "L00_V001",
                    "start_ms": 0,
                    "end_ms": 1000,
                    "transcript": "one two",
                    "transcript_search": "one two",
                    "frame_uid": "L00_V001:0",
                    "point_id": 1,
                    "keyframe_n": 1,
                    "frame_idx": 0,
                    "pts_time_s": 0.0,
                    "fps": 25.0,
                    "image_relpath": "L00_V001/00000000.jpg",
                    "bm25_score": -1.0,
                }
            ],
            query_tokens("one in two"),
        )
        result = index.search_many({"original": "one in two"}, _allow_single=True)
        evidence = result["results"][0]["asr_stream_provenance"]["original"]
        self.assertEqual(evidence["query_bigrams"], [])
        self.assertEqual(evidence["ngram_coverage"], 0.0)

    def test_repeated_query_token_keeps_adjacent_bigram(self) -> None:
        index = AsrFtsIndex(Path("/missing/asr_segments"), Path("/missing/asr.sqlite3"))
        index._runtime_ready_fast_locked = lambda: True  # type: ignore[method-assign]
        index.health = lambda: {"ready": True}  # type: ignore[method-assign]
        index._query_stream = lambda _query, _limit: (  # type: ignore[method-assign]
            [
                {
                    "segment_id": "seg-1",
                    "video_id": "L00_V001",
                    "start_ms": 0,
                    "end_ms": 1000,
                    "transcript": "alpha beta",
                    "transcript_search": "alpha beta",
                    "frame_uid": "L00_V001:0",
                    "point_id": 1,
                    "keyframe_n": 1,
                    "frame_idx": 0,
                    "pts_time_s": 0.0,
                    "fps": 25.0,
                    "image_relpath": "L00_V001/00000000.jpg",
                    "bm25_score": -1.0,
                }
            ],
            query_tokens("alpha alpha beta"),
        )
        result = index.search_many({"original": "alpha alpha beta"}, _allow_single=True)
        evidence = result["results"][0]["asr_stream_provenance"]["original"]
        self.assertEqual(evidence["query_bigrams"], [["alpha", "alpha"], ["alpha", "beta"]])
        self.assertEqual(evidence["ngram_coverage"], 0.5)

    def test_score_formula_and_max_segment_to_frame_are_auditable(self) -> None:
        index = AsrFtsIndex(Path("/missing/asr_segments"), Path("/missing/asr.sqlite3"))
        index._runtime_ready_fast_locked = lambda: True  # type: ignore[method-assign]
        index.health = lambda: {"ready": True}  # type: ignore[method-assign]
        rows = [
            {
                "segment_id": "seg-best",
                "video_id": "L00_V001",
                "start_ms": 0,
                "end_ms": 1000,
                "transcript": "alpha beta",
                "transcript_search": "alpha beta",
                "frame_uid": "L00_V001:0",
                "point_id": 1,
                "keyframe_n": 1,
                "frame_idx": 0,
                "pts_time_s": 0.0,
                "fps": 25.0,
                "image_relpath": "L00_V001/00000000.jpg",
                "bm25_score": -2.0,
            },
            {
                "segment_id": "seg-weaker",
                "video_id": "L00_V001",
                "start_ms": 1000,
                "end_ms": 2000,
                "transcript": "alpha",
                "transcript_search": "alpha",
                "frame_uid": "L00_V001:0",
                "point_id": 1,
                "keyframe_n": 1,
                "frame_idx": 0,
                "pts_time_s": 0.0,
                "fps": 25.0,
                "image_relpath": "L00_V001/00000000.jpg",
                "bm25_score": -1.0,
            },
        ]
        index._query_stream = lambda _query, _limit: (rows, ["alpha", "beta"])  # type: ignore[method-assign]
        result = index.search_many({"original": "alpha beta"}, _allow_single=True)
        self.assertEqual(result["candidate_segment_count"], 2)
        self.assertEqual(result["candidate_frame_count"], 1)
        winner = result["results"][0]
        self.assertEqual(winner["asr_segment_id"], "seg-best")
        evidence = winner["asr_stream_provenance"]["original"]
        self.assertAlmostEqual(evidence["combined_score"], 1.0, places=7)

    def test_deterministic_build_identity_contains_schema_and_fingerprints(self) -> None:
        records = [
            {"path": "asr_segments/b.jsonl", "size": 2, "sha256": "b"},
            {"path": "asr_segments/a.jsonl", "size": 1, "sha256": "a"},
        ]
        first = source_fingerprint(records)
        second = source_fingerprint(list(reversed(records)))
        self.assertEqual(first, second)
        build_id = build_id_for(
            source_fingerprint_value=first,
            canonical_fingerprint_value="canonical",
            segment_count=55168,
            video_count=873,
        )
        self.assertEqual(len(build_id), 64)
        self.assertEqual(ASR_INDEX_SCHEMA_VERSION, "branch3.asr-index.v2")
        self.assertEqual(ASR_SQLITE_USER_VERSION, 4)

    def test_single_query_compatibility_path_uses_one_stream(self) -> None:
        index = _FakeAsrIndex()
        calls = []

        def search(query, top_k):
            calls.append((query, top_k))
            return [{"frame_uid": "L21_V001:4", "score": 0.5}]

        index.search = search
        service = Branch3AsrSearch(index, threading.Lock())
        result = service.execute_single("một câu hỏi", 20)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result[0]["frame_uid"], "L21_V001:4")

    def test_compatibility_path_reports_shared_busy_lock(self) -> None:
        lock = threading.Lock()
        lock.acquire()
        try:
            service = Branch3AsrSearch(_FakeAsrIndex(), lock)
            with self.assertRaisesRegex(RuntimeError, "BRANCH3_ASR_SEARCH_BUSY"):
                service.execute_single("query", 20)
        finally:
            lock.release()

    def test_visual_searcher_delegates_compatibility_asr_to_canonical_service(self) -> None:
        calls = []

        class _Service:
            def execute_single(self, query, top_k):
                calls.append((query, top_k))
                return [{"frame_uid": "L00_V001:0"}]

        searcher = CpuQdrantSearch(object(), object(), object(), asr_service=_Service())
        result = searcher.search_speech("một câu hỏi", top_k=20)
        self.assertEqual(result, [{"frame_uid": "L00_V001:0"}])
        self.assertEqual(calls, [("một câu hỏi", 20)])

    def test_visual_searcher_fails_closed_when_asr_service_is_missing(self) -> None:
        searcher = CpuQdrantSearch(object(), object(), object())
        with self.assertRaisesRegex(RuntimeError, "ASR service is not ready"):
            searcher.search_speech("một câu hỏi", top_k=20)

    def test_runtime_reopens_after_atomic_database_and_manifest_swap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "asr.sqlite3"
            manifest = root / "branch3_asr_manifest.json"

            def write_database(path: Path, build_id: str) -> None:
                connection = sqlite3.connect(path)
                connection.execute(
                    "CREATE TABLE asr_meta (schema_version TEXT, sqlite_user_version INTEGER, build_id TEXT, source_fingerprint TEXT, canonical_fingerprint TEXT, mapping_strategy TEXT, segment_count INTEGER, video_count INTEGER, created_at TEXT)"
                )
                connection.execute(
                    "INSERT INTO asr_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "branch3.asr-index.v2",
                        4,
                        build_id,
                        "source",
                        "canonical",
                        "nearest_keyframe_to_segment_midpoint",
                        0,
                        0,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                connection.commit()
                connection.close()

            def write_manifest(build_id: str, path: Path) -> None:
                manifest.write_text(json.dumps({"build_id": build_id}), encoding="utf-8")

            write_database(database, "build-one")
            write_manifest("build-one", database)
            index = AsrFtsIndex(root / "asr_segments", database, manifest_path=manifest)
            first = index.health()
            self.assertEqual(first["connection_generation"], 1)
            replacement = root / "asr.sqlite3.staging"
            write_database(replacement, "build-two")
            os.replace(replacement, database)
            transitional = index.health()
            self.assertFalse(transitional["ready"])
            self.assertIsNone(transitional["opened_build_id"])
            write_manifest("build-two", database)
            second = index.health()
            self.assertEqual(second["connection_generation"], 2)
            self.assertEqual(second["opened_build_id"], "build-two")
            index.close()

    def test_nonempty_wrong_artifact_hash_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"actual")
            index = AsrFtsIndex(root / "asr_segments", root / "asr.sqlite3")
            matched, stat_matches, fingerprint_present = index._artifact_status(  # type: ignore[protected-access]
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

    def test_unchanged_artifact_uses_stat_chain_without_rehashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"actual")
            record = {
                "path": artifact.name,
                "size": artifact.stat().st_size,
                "mtime_ns": artifact.stat().st_mtime_ns,
                "sha256": hashlib.sha256(b"actual").hexdigest(),
            }
            index = AsrFtsIndex(root / "asr_segments", root / "asr.sqlite3")
            with patch(
                "online.src.retrieval.modalities.asr._sha256_file", side_effect=AssertionError
            ):
                matched, stat_matches, fingerprint_present = index._artifact_status(
                    record,
                    relative_root=root,
                    verify_hash=False,
                )
            self.assertTrue(matched)
            self.assertTrue(stat_matches)
            self.assertTrue(fingerprint_present)

    def test_artifact_path_traversal_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root.parent / "outside.bin"
            outside.write_bytes(b"outside")
            try:
                index = AsrFtsIndex(root / "asr_segments", root / "asr.sqlite3")
                matched, stat_matches, fingerprint_present = index._artifact_status(
                    {
                        "path": "../outside.bin",
                        "size": outside.stat().st_size,
                        "mtime_ns": outside.stat().st_mtime_ns,
                        "sha256": "0" * 64,
                    },
                    relative_root=root,
                )
                self.assertFalse(matched)
                self.assertFalse(stat_matches)
                self.assertFalse(fingerprint_present)
            finally:
                outside.unlink(missing_ok=True)

    def test_health_converts_unexpected_validation_error_to_fail_closed_payload(self) -> None:
        index = AsrFtsIndex(Path("/missing/asr_segments"), Path("/missing/asr.sqlite3"))
        with patch.object(index, "_health_signature", side_effect=ValueError("broken validation")):
            payload = index.health()
        self.assertFalse(payload["ready"])
        self.assertFalse(payload["production_ready"])
        self.assertIn("broken validation", payload["connection_error"])


if __name__ == "__main__":
    unittest.main()
