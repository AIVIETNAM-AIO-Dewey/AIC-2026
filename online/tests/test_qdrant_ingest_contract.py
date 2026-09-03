"""Behavioural tests for the non-destructive Qdrant repair tool."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.qdrant import ingest


class _BatchClient:
    """Small fake that records the data passed to the Qdrant batch adapter."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def upsert(self, **kwargs):
        self.calls.append(kwargs)


class _RepairClient:
    def __init__(self, record: dict) -> None:
        self.points = {int(record["id"]): record}
        self.upserts = 0

    def count(self, *, collection_name: str, exact: bool) -> int:
        del collection_name, exact
        return len(self.points)

    def scroll(self, **kwargs):
        del kwargs
        return ([{"id": point_id} for point_id in sorted(self.points)], None)

    def retrieve(
        self,
        *,
        collection_name: str,
        ids: list[int],
        with_payload: bool,
        with_vectors: bool,
    ):
        del collection_name, with_payload, with_vectors
        return [self.points[point_id] for point_id in ids if point_id in self.points]


def _source(vector: np.ndarray | None = None) -> dict:
    return {
        "id": 1,
        "vectors": {
            "dam": vector
            if vector is not None
            else np.asarray([1, 0, 0, 0], dtype=np.float16)
        },
        "payload": {
            "point_id": 1,
            "frame_uid": "L01_V001:1",
            "video_id": "L01_V001",
            "frame_idx": 1,
            "ingest_schema_version": ingest.POINT_SCHEMA_VERSION,
        },
    }


class QdrantIngestBehaviourTests(unittest.TestCase):
    def test_payload_match_tolerates_nested_qdrant_float_roundoff(self) -> None:
        expected = {"bbox": [0.0, 0.9963802761501737, 1.0]}
        actual = {"bbox": [0.0, 0.9963802761501735, 1.0]}

        self.assertTrue(ingest._payload_matches(actual, expected))

    def test_upsert_source_batch_stacks_named_vectors_in_source_order(self) -> None:
        client = _BatchClient()
        first = _source(np.asarray([1, 0, 0, 0], dtype=np.float16))
        second = {
            **_source(np.asarray([0, 1, 0, 0], dtype=np.float16)),
            "id": 2,
            "payload": {
                **_source()["payload"],
                "point_id": 2,
                "frame_uid": "L01_V001:2",
                "frame_idx": 2,
            },
        }
        captured: dict = {}

        def fake_upsert(_client, _collection, ids, vectors, payloads):
            captured.update(ids=ids, vectors=vectors, payloads=payloads)

        with patch.object(ingest, "upsert_with_retry", fake_upsert):
            ingest._upsert_source_batch(
                client,
                ingest.DAM_COLLECTION,
                [first, second],
                {"dam": 4},
            )

        self.assertEqual(captured["ids"], [1, 2])
        self.assertEqual(
            captured["vectors"]["dam"].tolist(),
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        )

    def test_upsert_rejects_missing_named_vector_and_bad_source_ids(self) -> None:
        client = _BatchClient()
        with self.assertRaises(ValueError):
            ingest._upsert_source_batch(
                client,
                ingest.DAM_COLLECTION,
                [{**_source(), "vectors": {}}],
                {"dam": 4},
            )
        with self.assertRaises(ValueError):
            ingest._upsert_source_batch(
                client,
                ingest.DAM_COLLECTION,
                [{**_source(), "id": 0}],
                {"dam": 4},
            )

    def test_vector_content_mismatch_is_not_shape_valid(self) -> None:
        expected = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        same_direction = np.asarray([2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        wrong_direction = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        self.assertTrue(ingest._vector_matches_source(same_direction, expected, 4)[0])
        self.assertFalse(ingest._vector_matches_source(wrong_direction, expected, 4)[0])

    def test_record_verification_checks_payload_and_vector_content(self) -> None:
        source = _source()
        valid_record = {
            "id": 1,
            "payload": dict(source["payload"]),
            "vector": {
                "dam": np.asarray(source["vectors"]["dam"], dtype=np.float32)
            },
        }
        self.assertTrue(
            ingest._record_matches_source(valid_record, source, {"dam": 4})[0]
        )
        wrong_vector = {
            **valid_record,
            "vector": {"dam": np.asarray([0, 1, 0, 0], dtype=np.float32)},
        }
        self.assertFalse(
            ingest._record_matches_source(wrong_vector, source, {"dam": 4})[0]
        )
        wrong_payload = {
            **valid_record,
            "payload": {**valid_record["payload"], "frame_idx": 99},
        }
        self.assertFalse(
            ingest._record_matches_source(wrong_payload, source, {"dam": 4})[0]
        )

    def test_verify_only_detects_same_dimension_wrong_vector_without_mutating(self) -> None:
        source = _source()
        existing = {
            "id": 1,
            "payload": dict(source["payload"]),
            "vector": {"dam": np.asarray([0, 1, 0, 0], dtype=np.float32)},
        }
        client = _RepairClient(existing)
        with patch.object(ingest, "validate_collection_definition"), self.assertRaises(
            ValueError
        ):
            ingest.reconcile_collection(
                client,
                "http://qdrant",
                ingest.DAM_COLLECTION,
                1,
                {"dam": 4},
                [source],
                1,
                repair=False,
            )
        self.assertEqual(client.upserts, 0)
        self.assertEqual(
            ingest.RECONCILIATION_REPORTS[ingest.DAM_COLLECTION]["mismatch_counts"][
                "vector_mismatch"
            ],
            1,
        )

    def test_repair_readback_verifies_payload_and_vector(self) -> None:
        source = _source()
        existing = {
            "id": 1,
            "payload": dict(source["payload"]),
            "vector": {"dam": np.asarray([0, 1, 0, 0], dtype=np.float32)},
        }
        client = _RepairClient(existing)

        def fake_upsert(_client, _collection, ids, vectors, payloads):
            client.upserts += len(ids)
            for index, point_id in enumerate(ids):
                client.points[point_id] = {
                    "id": point_id,
                    "payload": payloads[index],
                    "vector": {"dam": vectors["dam"][index]},
                }

        with patch.object(ingest, "validate_collection_definition"), patch.object(
            ingest, "upsert_with_retry", fake_upsert
        ):
            result = ingest.reconcile_collection(
                client,
                "http://qdrant",
                ingest.DAM_COLLECTION,
                1,
                {"dam": 4},
                [source],
                1,
                repair=True,
            )
        self.assertEqual(result, 1)
        self.assertEqual(client.upserts, 1)
        self.assertEqual(
            ingest.RECONCILIATION_REPORTS[ingest.DAM_COLLECTION]["repaired_count"],
            1,
        )

    def test_manifest_is_v3_and_written_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_root = Path(directory) / "state"
            data_root = Path(directory) / "data"
            artifact = data_root / "dense_text_embeddings" / "dam_vectors.f16.npy"
            metadata = data_root / "dense_text_embeddings" / "dam_metadata.jsonl"
            frame_metadata = (
                data_root
                / "visual_embeddings"
                / "metaclip2"
                / "keyframes_metadata.jsonl"
            )
            for path in (artifact, metadata, frame_metadata):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            ingest.write_ingestion_manifest(
                state_root,
                data_root,
                {ingest.DAM_COLLECTION: ingest.EXPECTED_DAM_REGIONS},
                ("dam",),
                status="ready",
                verification={
                    ingest.DAM_COLLECTION: {
                        "expected_count": ingest.EXPECTED_DAM_REGIONS,
                        "verified_count": ingest.EXPECTED_DAM_REGIONS,
                        "repaired_count": 0,
                        "payload_verified": True,
                        "vector_content_verified": True,
                        "verification_threshold": {
                            "cosine_min": ingest.VECTOR_DIRECTION_COSINE_MIN,
                            "max_abs_error": ingest.VECTOR_DIRECTION_MAX_ABS_ERROR,
                        },
                        "completed_at": "2026-08-30T00:00:00+00:00",
                    }
                },
            )
            report = json.loads(
                (state_root / "qdrant_ingestion_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                report["schema_version"], ingest.INGEST_MANIFEST_SCHEMA_VERSION
            )
            self.assertTrue(
                report["verification"][ingest.DAM_COLLECTION][
                    "vector_content_verified"
                ]
            )

    def test_ready_manifest_requires_verification_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "dense_text_embeddings" / "dam_vectors.f16.npy"
            metadata = root / "dense_text_embeddings" / "dam_metadata.jsonl"
            frame_metadata = root / "visual_embeddings" / "metaclip2" / "keyframes_metadata.jsonl"
            for path in (artifact, metadata, frame_metadata):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            with self.assertRaises(ValueError):
                ingest.write_ingestion_manifest(
                    root / "state",
                    root,
                    {ingest.DAM_COLLECTION: ingest.EXPECTED_DAM_REGIONS},
                    ("dam",),
                    status="ready",
                )


if __name__ == "__main__":
    unittest.main()
