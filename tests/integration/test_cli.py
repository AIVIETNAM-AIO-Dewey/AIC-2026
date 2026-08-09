from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

from aic2026.contracts import RunManifest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_frame_builder_and_validator_cli(tmp_path: Path) -> None:
    frames_dir = tmp_path / "keyframes" / "L21_V011"
    frames_dir.mkdir(parents=True)
    for keyframe_n in (1, 2, 3):
        Image.new("RGB", (16, 9), color=(keyframe_n, 0, 0)).save(
            frames_dir / f"{keyframe_n:06d}.jpg"
        )
    map_csv = tmp_path / "L21_V011.csv"
    shutil.copy(FIXTURES / "frame_map.csv", map_csv)
    artifact_root = tmp_path / "artifacts"
    output = artifact_root / "frame_manifests" / "L21_V011.jsonl"

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_frame_manifest.py"),
        "--video-id",
        "L21_V011",
        "--data-root",
        str(tmp_path),
        "--output-root",
        str(artifact_root),
        "--map-csv",
        str(map_csv),
        "--frames-dir",
        str(frames_dir),
        "--output",
        str(output),
        "--resume",
    ]
    first = subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    assert json.loads(first.stdout)["frames"] == 3

    manifest_path = output.with_suffix(".manifest.json")
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert manifest.status == "completed"
    assert manifest.outputs[0].sha256 is not None

    interrupted = manifest.model_copy(
        update={"status": "running", "ended_at": None, "outputs": [], "counters": {}}
    )
    manifest_path.write_text(
        interrupted.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    recovered = subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    recovery_report = json.loads(recovered.stdout)
    assert recovery_report["status"] == "completed"
    assert recovery_report["recovered_final"] == 1

    second = subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    assert json.loads(second.stdout)["status"] == "already_complete"

    validator = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "validate_object_artifacts.py"),
            "--artifact",
            str(output),
            "--manifest",
            str(manifest_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    validation_report = json.loads(validator.stdout)
    assert validation_report["ok"] is True
    assert validation_report["artifacts"][0]["checksum_verified"] is True

    # Blank JSONL lines are semantically harmless but still change the artifact checksum.
    output.write_text(output.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    mismatch = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "validate_object_artifacts.py"),
            "--artifact",
            str(output),
            "--manifest",
            str(manifest_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert mismatch.returncode != 0
    assert "checksum does not match" in mismatch.stderr


def test_environment_preflight_rejects_missing_data_and_non_directory_output(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    bad_output = tmp_path / "not-a-directory"
    bad_output.write_text("file", encoding="utf-8")
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "verify_environment.py"),
        "--device",
        "cpu",
        "--data-root",
        str(tmp_path / "missing-data"),
        "--output-root",
        str(bad_output),
        "--cache-root",
        str(cache_root),
    ]

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["ok"] is False
    checks = {item["name"]: item["ok"] for item in report["checks"]}
    assert checks["data_root"] is False
    assert checks["output_root"] is False
