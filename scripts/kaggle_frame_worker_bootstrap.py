"""Kaggle bootstrap for production frame-extraction worker notebooks."""

from __future__ import annotations

import base64
import binascii
import configparser
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


WEIGHTS_SHA256 = "a313d0b3bebfa9a71914b375bfdf918a30b5c3b1e6be51972d35dd8078b442de"
WEIGHTS_URL = (
    "https://huggingface.co/ByteDance/shot2story/resolve/"
    "ff853c571fd92eb4e0c5713e27f2a323ac903f67/"
    "transnetv2-pytorch-weights.pth?download=true"
)


def _log(message: str) -> None:
    print(f"[kaggle_worker] {message}", flush=True)


def _run_streamed(
    command: list[str],
    *,
    prefix: str = "",
    env: dict[str, str] | None = None,
) -> list[str]:
    print("$", " ".join(map(str, command)), flush=True)
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    lines: list[str] = []
    for line in process.stdout:
        lines.append(line.rstrip())
        print(f"{prefix}{line}", end="", flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Command failed with code {return_code}: {' '.join(map(str, command))}\n"
            + "\n".join(lines[-80:])
        )
    return lines


def _json_report(lines: list[str]) -> dict[str, Any]:
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Command produced no JSON report")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(video_ids: list[str]) -> dict[str, Path]:
    video_root = Path("/kaggle/input/datasets/lyduchoang/aic-26-video/Video")
    _log(f"phase=inventory result=attempting root={video_root} count={len(video_ids)}")
    if not video_root.is_dir():
        raise FileNotFoundError(f"Raw-video dataset root is missing: {video_root}")
    wanted = set(video_ids)
    matches: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in video_root.rglob("*.mp4"):
        if path.stem not in wanted:
            continue
        if path.stem in matches:
            duplicates.setdefault(path.stem, [matches[path.stem]]).append(path)
        else:
            matches[path.stem] = path
    missing = [video_id for video_id in video_ids if video_id not in matches]
    if missing or duplicates:
        raise RuntimeError(f"Inventory mismatch missing={missing} duplicates={duplicates}")
    for index, video_id in enumerate(video_ids, start=1):
        _log(
            f"phase=inventory item={index}/{len(video_ids)} "
            f"video_id={video_id} path={matches[video_id]}"
        )
    _log(f"phase=inventory result=success count={len(matches)}")
    return matches


def _runtime_preflight() -> None:
    for binary in ("git", "ffmpeg", "ffprobe", "nvidia-smi"):
        resolved = shutil.which(binary)
        _log(f"phase=runtime binary={binary} path={resolved or '<missing>'}")
        if not resolved:
            raise RuntimeError(f"Required binary is missing: {binary}")
    import torch

    _log(
        f"phase=runtime torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"gpu_count={torch.cuda.device_count()}"
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Frame worker requires Kaggle Accelerator = GPU T4 x2")
    for index in range(2):
        name = torch.cuda.get_device_name(index)
        _log(f"phase=runtime gpu={index} name={name}")
        if "T4" not in name.upper():
            raise RuntimeError(f"GPU {index} is not a Tesla T4: {name}")


def _rclone_setup() -> tuple[Path, Path, str, str]:
    from kaggle_secrets import UserSecretsClient

    _log("phase=kaggle_secrets result=attempting")
    secrets = UserSecretsClient()
    config_secret = (secrets.get_secret("AIC_RCLONE_CONFIG") or "").strip()
    folder_id = (secrets.get_secret("AIC_GDRIVE_FOLDER_ID") or "").strip()
    if not config_secret or not folder_id:
        raise ValueError("Enable AIC_RCLONE_CONFIG and AIC_GDRIVE_FOLDER_ID secrets")
    _log("phase=kaggle_secrets result=success values=<redacted>")
    if config_secret.lstrip().startswith("["):
        config_text = config_secret
    else:
        encoded = config_secret.removeprefix("base64:")
        try:
            config_text = base64.b64decode(
                "".join(encoded.split()), validate=True
            ).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError) as error:
            raise ValueError(
                "AIC_RCLONE_CONFIG must be rclone.conf text or its base64 payload"
            ) from error
    remote = os.environ.get("AIC_RCLONE_REMOTE", "gdrive")
    parsed = configparser.RawConfigParser()
    parsed.read_string(config_text)
    if not parsed.has_section(remote) or parsed.get(remote, "type", fallback="") != "drive":
        raise ValueError(f"rclone config must contain a [{remote}] drive remote")
    config_path = Path("/tmp/aic2026-rclone/rclone.conf")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_text, encoding="utf-8")
    config_path.chmod(0o600)
    rclone_bin = Path("/kaggle/working/bin/rclone")
    if not rclone_bin.is_file():
        archive = Path("/tmp/rclone.zip")
        _log("phase=rclone_install result=attempting")
        urllib.request.urlretrieve(
            "https://downloads.rclone.org/rclone-current-linux-amd64.zip", archive
        )
        with zipfile.ZipFile(archive) as bundle:
            member = next(item for item in bundle.infolist() if item.filename.endswith("/rclone"))
            rclone_bin.parent.mkdir(parents=True, exist_ok=True)
            rclone_bin.write_bytes(bundle.read(member))
        rclone_bin.chmod(0o755)
    _log(f"phase=rclone_install result=success path={rclone_bin}")
    return rclone_bin, config_path, remote, folder_id


def _sync_base(
    *,
    rclone_bin: Path,
    config_path: Path,
    remote: str,
    folder_id: str,
    package_name: str,
    worker_id: str,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "scripts/sync_frame_video_with_rclone.py",
        "--rclone-bin",
        str(rclone_bin),
        "--config",
        str(config_path),
        "--remote",
        remote,
        "--root-folder-id",
        folder_id,
        "--remote-root-name",
        package_name,
        "--worker-id",
        worker_id,
    ]


def _scan(
    sync_base: list[str], video_ids: list[str], identity_dir: Path
) -> dict[str, Any]:
    command = sync_base[:3] + ["scan"] + sync_base[3:]
    command.extend(["--identity-dir", str(identity_dir), "--attempts", "5"])
    for video_id in video_ids:
        command.extend(["--video-id", video_id])
    return _json_report(_run_streamed(command, prefix="[drive_scan] "))


def _transnet_setup() -> tuple[Path, Path]:
    source = Path("/kaggle/working/TransNetV2-source")
    module = source / "inference-pytorch" / "transnetv2_pytorch.py"
    weights = Path("/kaggle/working/transnetv2-pytorch/transnetv2-pytorch-weights.pth")
    if not module.is_file():
        if source.exists():
            raise RuntimeError(f"TransNet source exists but PyTorch module is missing: {source}")
        environment = os.environ.copy()
        environment["GIT_LFS_SKIP_SMUDGE"] = "1"
        _run_streamed(
            ["git", "clone", "--depth=1", "https://github.com/soCzech/TransNetV2.git", str(source)],
            env=environment,
        )
    if not weights.is_file() or _sha256(weights) != WEIGHTS_SHA256:
        weights.parent.mkdir(parents=True, exist_ok=True)
        partial = weights.with_suffix(".partial")
        _log("phase=checkpoint result=downloading")
        urllib.request.urlretrieve(WEIGHTS_URL, partial)
        if _sha256(partial) != WEIGHTS_SHA256:
            raise ValueError("Downloaded TransNetV2 checkpoint SHA-256 mismatch")
        os.replace(partial, weights)
    _log(f"phase=transnet_setup result=success module={module} checkpoint=verified")
    return module, weights


def run_kaggle_worker(
    *,
    worker_id: str,
    owned_video_ids: list[str],
    resume_probe_video_ids: list[str],
    session_started_at: float,
) -> dict[str, Any]:
    if len(owned_video_ids) != 73 or len(set(owned_video_ids)) != 73:
        raise ValueError("Production worker must own exactly 73 unique videos")
    if set(owned_video_ids) & set(resume_probe_video_ids):
        raise ValueError("Resume probes cannot also be owned by this worker")
    all_video_ids = list(dict.fromkeys([*resume_probe_video_ids, *owned_video_ids]))
    _log(
        f"phase=assignment worker_id={worker_id} owned={len(owned_video_ids)} "
        f"resume_probes={resume_probe_video_ids}"
    )
    _runtime_preflight()
    video_paths = _inventory(all_video_ids)
    artifact_root = Path("/kaggle/working/aic2026-artifacts")
    package_name = "self-cut-btc-compatible"
    package_root = artifact_root / "exports" / package_name
    config_path = Path("configs/offline/frame_extraction.yaml").resolve()
    runtime_dir = Path("/kaggle/working") / worker_id
    identity_dir = runtime_dir / "identities"
    identity_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = runtime_dir / "assignment.json"
    video_index_path = runtime_dir / "video_index.json"
    assignment_path.write_text(
        json.dumps({"worker_id": worker_id, "video_ids": owned_video_ids}, indent=2) + "\n",
        encoding="utf-8",
    )
    video_index_path.write_text(
        json.dumps({key: str(video_paths[key]) for key in owned_video_ids}, indent=2) + "\n",
        encoding="utf-8",
    )
    config_sha256 = _sha256(config_path)
    for video_id in all_video_ids:
        identity = {
            "contract_version": "frame-extraction-worker-v1",
            "video_id": video_id,
            "source_size_bytes": video_paths[video_id].stat().st_size,
            "config_sha256": config_sha256,
            "checkpoint_sha256": WEIGHTS_SHA256,
            "batch_size": 16,
            "sampling_policy": "lt2-midpoint__2-lt4-quarter-pair__gte4-1.5s__gte7-cap10",
            "extraction_method": "frame-index-select",
        }
        (identity_dir / f"{video_id}.json").write_text(
            json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8"
        )

    rclone_bin, rclone_config, remote, folder_id = _rclone_setup()
    sync_base = _sync_base(
        rclone_bin=rclone_bin,
        config_path=rclone_config,
        remote=remote,
        folder_id=folder_id,
        package_name=package_name,
        worker_id=worker_id,
    )
    preflight = sync_base[:3] + ["preflight"] + sync_base[3:] + ["--attempts", "5"]
    preflight_report = _json_report(_run_streamed(preflight, prefix="[drive] "))
    initial_scan = _scan(sync_base, all_video_ids, identity_dir)
    incomplete_probes = [
        video_id for video_id in resume_probe_video_ids if video_id in initial_scan["pending"]
    ]
    if incomplete_probes:
        raise RuntimeError(
            "Resume probe is not complete; wait for its owning worker to publish a valid marker: "
            f"{incomplete_probes}"
        )
    _log(
        f"phase=resume_probe result=success completed={resume_probe_video_ids or '<none>'}"
    )
    pending_owned = [
        video_id for video_id in owned_video_ids if video_id in initial_scan["pending"]
    ]
    if not pending_owned:
        report = {
            "status": "completed",
            "worker_id": worker_id,
            "remote_completed": owned_video_ids,
            "remaining": [],
        }
    else:
        module, weights = _transnet_setup()
        command = [
            sys.executable,
            "-u",
            "scripts/run_frame_extraction_worker.py",
            "--worker-id",
            worker_id,
            "--assignment",
            str(assignment_path),
            "--video-index",
            str(video_index_path),
            "--identity-dir",
            str(identity_dir),
            "--config",
            str(config_path),
            "--output-root",
            str(artifact_root),
            "--package-root",
            str(package_root),
            "--entrypoint",
            str(module),
            "--weights",
            str(weights),
            "--batch-size",
            "16",
            "--rclone-bin",
            str(rclone_bin),
            "--rclone-config",
            str(rclone_config),
            "--rclone-remote",
            remote,
            "--drive-root-folder-id",
            folder_id,
            "--remote-root-name",
            package_name,
            "--session-start-epoch",
            str(session_started_at),
            "--accept-new-work-seconds",
            str(11 * 3600 + 15 * 60),
            "--scan-attempts",
            "5",
            "--upload-attempts",
            "10",
        ]
        report = _json_report(_run_streamed(command, prefix=f"[{worker_id}] "))
    report["resume_probes"] = resume_probe_video_ids
    report["drive_url"] = preflight_report.get("folder_url")
    print(json.dumps(report, indent=2), flush=True)
    return report
