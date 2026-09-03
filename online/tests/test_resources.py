"""Fail-closed resource qualification checks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from online.src.retrieval.infrastructure.resources import (
    MAX_PRODUCTION_RSS_BYTES,
    RESOURCE_QUALIFICATION_SCHEMA,
    resource_qualification,
)


def write_runtime_fingerprint(root: Path) -> str:
    material = {"compose": {"sha256": "a" * 64}, "images": {"api": "sha256:api", "qdrant": "sha256:qdrant"}}
    canonical = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    (root / "runtime_fingerprint.json").write_text(
        json.dumps(
            {
                "schema_version": "aic.runtime-fingerprint.v1",
                "fingerprint": fingerprint,
                "material": material,
            }
        ),
        encoding="utf-8",
    )
    return fingerprint


class ResourceQualificationTests(unittest.TestCase):
    def test_missing_report_is_not_production_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = resource_qualification(Path(directory))
            self.assertFalse(result["ready"])
            self.assertFalse(result["production_ready"])

    def test_report_requires_both_branch_measurements_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fingerprint = write_runtime_fingerprint(root)
            report = {
                "schema_version": RESOURCE_QUALIFICATION_SCHEMA,
                "passed": True,
                "production_ready": True,
                "stack_peak_rss_bytes": MAX_PRODUCTION_RSS_BYTES,
                "api_peak_rss_bytes": 1,
                "worker_peak_rss_bytes": 1,
                "qdrant_peak_rss_bytes": 1,
                "coverage": {
                    "branch1": True,
                    "branch2": True,
                    "siglip2": True,
                    "metaclip2": True,
                    "bge_m3": True,
                    "beit3": True,
                },
                "compose_fingerprint": fingerprint,
                "measured_at": "2026-08-30T00:00:00+00:00",
            }
            (root / "resource_qualification.json").write_text(
                json.dumps(report), encoding="utf-8"
            )
            with patch.dict(os.environ, {"AIC_COMPOSE_FINGERPRINT": fingerprint}, clear=False):
                self.assertTrue(resource_qualification(root)["production_ready"])
                report["stack_peak_rss_bytes"] = MAX_PRODUCTION_RSS_BYTES + 1
                (root / "resource_qualification.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )
                self.assertFalse(resource_qualification(root)["production_ready"])

    def test_missing_or_stale_runtime_fingerprint_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fingerprint = write_runtime_fingerprint(root)
            report = {
                "schema_version": RESOURCE_QUALIFICATION_SCHEMA,
                "passed": True,
                "production_ready": True,
                "stack_peak_rss_bytes": 1,
                "api_peak_rss_bytes": 1,
                "worker_peak_rss_bytes": 0,
                "qdrant_peak_rss_bytes": 0,
                "coverage": {name: True for name in ("branch1", "branch2", "siglip2", "metaclip2", "bge_m3", "beit3")},
                "compose_fingerprint": fingerprint,
                "measured_at": "2026-08-30T00:00:00+00:00",
            }
            (root / "resource_qualification.json").write_text(json.dumps(report), encoding="utf-8")
            with patch.dict(os.environ, {"AIC_COMPOSE_FINGERPRINT": ""}, clear=False):
                self.assertFalse(resource_qualification(root)["production_ready"])
            with patch.dict(os.environ, {"AIC_COMPOSE_FINGERPRINT": "b" * 64}, clear=False):
                self.assertFalse(resource_qualification(root)["production_ready"])

    def test_report_without_measurement_metadata_is_not_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {
                "schema_version": RESOURCE_QUALIFICATION_SCHEMA,
                "passed": True,
                "production_ready": True,
                "stack_peak_rss_bytes": 1,
                "api_peak_rss_bytes": 1,
                "worker_peak_rss_bytes": 0,
                "qdrant_peak_rss_bytes": 0,
                "coverage": {name: True for name in ("branch1", "branch2", "siglip2", "metaclip2", "bge_m3", "beit3")},
                "compose_fingerprint": "test-image-config",
            }
            (root / "resource_qualification.json").write_text(json.dumps(report), encoding="utf-8")
            self.assertFalse(resource_qualification(root)["production_ready"])


if __name__ == "__main__":
    unittest.main()
