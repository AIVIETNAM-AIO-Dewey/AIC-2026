from __future__ import annotations

import json
from pathlib import Path

import pytest

from aic_backend.ocr.jobs import OcrJobManager
from aic_backend.settings import Settings


def test_job_status_is_filesystem_derived_and_runner_is_disabled_by_default(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    frame_manifests = artifacts / "frame_manifests"
    outputs = artifacts / "ocr"
    frame_manifests.mkdir(parents=True)
    outputs.mkdir()
    (frame_manifests / "L23_TEST.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (outputs / "L23_TEST.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (outputs / "L23_TEST.manifest.json").write_text(
        json.dumps({"status": "completed", "counters": {"frames": 2, "success": 2}}),
        encoding="utf-8",
    )
    manager = OcrJobManager(Settings(artifact_root=artifacts, ocr_jobs_enabled=False))

    status = manager.status()

    assert status["enabled"] is False
    assert status["datasets"] == [
        {
            "manifest_id": "L23_TEST",
            "status": "completed",
            "total_frames": 2,
            "processed_frames": 2,
            "remaining_frames": 0,
            "counters": {"frames": 2, "success": 2},
            "output_exists": True,
        }
    ]
    with pytest.raises(PermissionError, match="ocr_jobs_disabled"):
        manager.start("L23_TEST")
    with pytest.raises(ValueError, match="invalid_manifest_id"):
        manager.completed_artifact("../escape")
