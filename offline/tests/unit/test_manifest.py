from __future__ import annotations

from pathlib import Path

import pytest
from aic2026.common.manifest import (
    complete_manifest,
    create_manifest,
    prepare_resume,
    write_manifest,
)
from aic2026.contracts import RunManifest


def test_manifest_records_output_checksum_and_validates_resume(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "L21_V011.jsonl"
    output.write_text('{"frame_uid":"L21_V011:0"}\n', encoding="utf-8")
    manifest_path = output.with_suffix(".manifest.json")

    proposed = create_manifest(
        run_id="test-run",
        stage="sam_masks",
        config={"threshold": 0.1},
        seed=2026,
        input_paths=[("objects", source)],
        repo_root=Path(__file__).resolve().parents[2],
    )
    completed = complete_manifest(
        proposed,
        counters={"frames": 1},
        shard="L21_V011",
        output_paths=[("mask_artifact", output)],
    )
    write_manifest(manifest_path, completed)

    parsed = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert parsed.status == "completed"
    assert parsed.outputs[0].sha256 is not None

    existing, is_complete = prepare_resume(
        manifest_path=manifest_path,
        output_path=output,
        proposed=proposed,
        resume=True,
    )
    assert is_complete is True
    assert existing.run_id == "test-run"

    incompatible = create_manifest(
        run_id="test-run",
        stage="sam_masks",
        config={"threshold": 0.2},
        seed=2026,
        input_paths=[("objects", source)],
        repo_root=Path(__file__).resolve().parents[2],
    )
    with pytest.raises(ValueError, match="config differs"):
        prepare_resume(
            manifest_path=manifest_path,
            output_path=output,
            proposed=incompatible,
            resume=True,
        )


def test_resume_allows_stage_to_repair_final_output_with_running_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "artifact.jsonl"
    output.write_text('{"frame_uid":"L21_V011:0"}\n', encoding="utf-8")
    manifest_path = output.with_suffix(".manifest.json")
    proposed = create_manifest(
        run_id="test-run",
        stage="sam_masks",
        config={"threshold": 0.1},
        seed=2026,
        input_paths=[("objects", source)],
        repo_root=Path(__file__).resolve().parents[2],
    )
    write_manifest(manifest_path, proposed)

    recovered, is_complete = prepare_resume(
        manifest_path=manifest_path,
        output_path=output,
        proposed=proposed,
        resume=True,
    )

    assert is_complete is False
    assert recovered.status == "running"


def test_orphan_partial_without_sidecar_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "artifact.jsonl"
    output.with_suffix(".jsonl.partial").write_text("{}\n", encoding="utf-8")
    proposed = create_manifest(
        run_id="test-run",
        stage="sam_masks",
        config={"threshold": 0.1},
        seed=2026,
        input_paths=[("objects", source)],
        repo_root=Path(__file__).resolve().parents[2],
    )

    with pytest.raises(FileExistsError, match="without a run manifest"):
        prepare_resume(
            manifest_path=output.with_suffix(".manifest.json"),
            output_path=output,
            proposed=proposed,
            resume=True,
        )
