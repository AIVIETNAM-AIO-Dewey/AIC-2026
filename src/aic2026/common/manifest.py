"""Reproducible run-manifest construction and status transitions."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aic2026.contracts import RunManifest
from aic2026.contracts.models import ArtifactInput, ModelRevision

from .io import atomic_write_json, sha256_path


def _git_state(repo_root: Path) -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _runtime_platform() -> dict[str, Any]:
    report: dict[str, Any] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": {},
    }
    for name in (
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "huggingface-hub",
        "dam",
        "pydantic",
        "numpy",
        "pillow",
        "pyyaml",
        "jsonschema",
        "tqdm",
    ):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    try:
        import torch

        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_build"] = torch.version.cuda
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            report["gpu"] = {
                "index": index,
                "name": properties.name,
                "total_vram_bytes": properties.total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
            }
    except ImportError:
        report["cuda_available"] = False
    return report


def create_manifest(
    *,
    run_id: str,
    stage: str,
    config: dict[str, Any],
    seed: int,
    input_paths: list[tuple[str, Path]],
    models: list[dict[str, str]] | None = None,
    repo_root: Path,
) -> RunManifest:
    git_sha, git_dirty = _git_state(repo_root)
    inputs = [
        ArtifactInput(name=name, source_id=path.name, sha256=sha256_path(path))
        for name, path in input_paths
    ]
    return RunManifest(
        run_id=run_id,
        stage=stage,
        status="running",
        git_sha=git_sha,
        git_dirty=git_dirty,
        platform=_runtime_platform(),
        resolved_config=config,
        config_sha256=_config_hash(config),
        seed=seed,
        inputs=inputs,
        models=[ModelRevision.model_validate(model) for model in (models or [])],
        started_at=datetime.now(timezone.utc),
    )


def complete_manifest(
    manifest: RunManifest,
    *,
    counters: dict[str, int],
    shard: str,
    output_paths: list[tuple[str, Path]] | None = None,
) -> RunManifest:
    outputs = [
        ArtifactInput(name=name, source_id=path.name, sha256=sha256_path(path))
        for name, path in (output_paths or [])
    ]
    return manifest.model_copy(
        update={
            "status": "completed",
            "ended_at": datetime.now(timezone.utc),
            "counters": counters,
            "completed_shards": sorted(set([*manifest.completed_shards, shard])),
            "outputs": outputs,
        }
    )


def fail_manifest(manifest: RunManifest, error: BaseException) -> RunManifest:
    return manifest.model_copy(
        update={
            "status": "failed",
            "ended_at": datetime.now(timezone.utc),
            "errors": [
                *manifest.errors,
                {"type": error.__class__.__name__, "message": str(error)},
            ],
        }
    )


def write_manifest(path: Path, manifest: RunManifest) -> None:
    atomic_write_json(path, manifest.model_dump(mode="json"))


def prepare_resume(
    *,
    manifest_path: Path,
    output_path: Path,
    proposed: RunManifest,
    resume: bool,
) -> tuple[RunManifest, bool]:
    """Validate resume compatibility and report whether the stage is already complete."""
    if not manifest_path.exists():
        orphaned = [
            candidate
            for candidate in (
                output_path,
                output_path.with_suffix(output_path.suffix + ".partial"),
                output_path.with_suffix(output_path.suffix + ".tmp"),
            )
            if candidate.exists()
        ]
        if orphaned:
            raise FileExistsError(f"Artifact exists without a run manifest: {orphaned[0]}")
        return proposed, False
    if not resume:
        raise FileExistsError(f"Run manifest already exists; use --resume: {manifest_path}")
    existing = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if existing.run_id != proposed.run_id or existing.stage != proposed.stage:
        raise ValueError("Resume run_id/stage does not match the existing manifest")
    if existing.config_sha256 != proposed.config_sha256:
        raise ValueError("Resume config differs from the existing manifest")
    existing_inputs = [(item.name, item.sha256) for item in existing.inputs]
    proposed_inputs = [(item.name, item.sha256) for item in proposed.inputs]
    if existing_inputs != proposed_inputs:
        raise ValueError("Resume inputs differ from the existing manifest")
    existing_models = [(item.model_id, item.revision) for item in existing.models]
    proposed_models = [(item.model_id, item.revision) for item in proposed.models]
    if existing_models != proposed_models:
        raise ValueError("Resume model revisions differ from the existing manifest")
    if output_path.exists():
        if existing.status != "completed":
            partial_path = output_path.with_suffix(output_path.suffix + ".partial")
            if partial_path.exists():
                raise ValueError("Both final and partial outputs exist; recovery is ambiguous")
            # The process may have crashed after the atomic partial->final rename and
            # before publishing its completed sidecar. Let the stage validate the
            # final artifact and repair the sidecar without rerunning a GPU model.
            return proposed.model_copy(
                update={"started_at": existing.started_at, "errors": existing.errors}
            ), False
        matching_outputs = [item for item in existing.outputs if item.sha256 is not None]
        if not matching_outputs:
            raise ValueError("Completed run manifest does not record an output checksum")
        actual_output_hash = sha256_path(output_path)
        if actual_output_hash not in {item.sha256 for item in matching_outputs}:
            raise ValueError("Final output checksum differs from the completed manifest")
        return existing, True
    return proposed.model_copy(
        update={"started_at": existing.started_at, "errors": existing.errors}
    ), False


def validate_upstream_manifest(
    *, manifest_path: Path, artifact_path: Path, expected_stage: str
) -> RunManifest:
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if manifest.stage != expected_stage or manifest.status != "completed":
        raise ValueError(
            f"Upstream manifest must be completed stage {expected_stage!r}: {manifest_path}"
        )
    actual_hash = sha256_path(artifact_path)
    if actual_hash not in {item.sha256 for item in manifest.outputs}:
        raise ValueError(f"Upstream artifact checksum does not match {manifest_path}")
    return manifest
