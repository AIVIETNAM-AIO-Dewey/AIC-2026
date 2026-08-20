#!/usr/bin/env python3
"""Continue completed Kaggle OCR Phase 1 shards through local VietOCR Phase 2.

The launcher is intentionally outside the Phase 1 implementation.  It reuses a
dedicated environment under ``/kaggle/working``, verifies or acquires the pinned
VietOCR checkpoint, resumes representative recognition per shard, then publishes
a content-addressed bundle for a later, separate upload step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE2_RUNNER = REPO_ROOT / "offline/scripts/run_ocr_phase2_local.py"
DEFAULT_PHASE1_CONFIG = REPO_ROOT / "offline/configs/offline/ocr_phase1_kaggle_gpu.yaml"
DEFAULT_MODEL_CONFIG = REPO_ROOT / "offline/configs/offline/vietocr_vgg_seq2seq.yaml"

WEIGHTS_URL = "https://vocr.vn/data/vietocr/vgg_seq2seq.pth"
WEIGHTS_SHA256 = "0921503a41375a0584268e23ef3d414ea478a8fe8777865c7745d38f2d0bc5db"
WEIGHTS_BYTES = 89_575_371
MODEL_ID = "pbcquoc/vietocr-vgg-seq2seq"

ENV_REQUIREMENTS = (
    "einops==0.8.1",
    "gdown==5.2.0",
    "numpy==1.26.4",
    "Pillow==11.1.0",
    "pydantic==2.10.6",
    "PyYAML==6.0.2",
    "requests==2.32.3",
    "tqdm==4.67.1",
    "wrapt==1.17.3",
)
ENV_EXPECTED_VERSIONS = {
    "einops": "0.8.1",
    "gdown": "5.2.0",
    "numpy": "1.26.4",
    "Pillow": "11.1.0",
    "pydantic": "2.10.6",
    "PyYAML": "6.0.2",
    "requests": "2.32.3",
    "tqdm": "4.67.1",
    "vietocr": "0.3.13",
    "wrapt": "1.17.3",
}


@dataclass(frozen=True)
class Phase1Shard:
    shard_id: str
    frame_count: int
    frame_manifest: Path
    detections: Path
    trajectories: Path
    representatives: Path


class IncompletePhase1Shard(Exception):
    """A planned Phase 1 shard has not reached its completed receipt yet."""


class SoftStop(Exception):
    """The Kaggle durability deadline was reached."""


def log(message: str) -> None:
    print(f"[phase2-e2e] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def receipt_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".receipt.json")


def find_named_paths(
    root: Path, name: str, *, directories: bool, maximum_depth: int = 7
) -> list[Path]:
    """Search attached outputs without descending into the 177k-frame collections."""

    if not root.is_dir():
        return []
    found: list[Path] = []
    pending = [(root, 0)]
    skipped = {"keyframes", "map-keyframes", "crops", "state", "runtime-cache"}
    while pending:
        parent, depth = pending.pop()
        if depth >= maximum_depth:
            continue
        try:
            children = sorted(parent.iterdir())
        except OSError:
            continue
        for child in children:
            if child.name == name and child.is_dir() == directories:
                found.append(child.resolve())
                if directories:
                    continue
            if child.is_dir() and child.name not in skipped:
                pending.append((child, depth + 1))
    return sorted(set(found))


def _verify_weight(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != WEIGHTS_BYTES:
        raise ValueError(f"VietOCR weight byte count mismatch: {path}")
    if sha256_file(path) != WEIGHTS_SHA256:
        raise ValueError(f"VietOCR weight SHA-256 mismatch: {path}")


def _copy_weight(source: Path, destination: Path) -> Path:
    _verify_weight(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    with source.open("rb") as reader, temporary.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    _verify_weight(temporary)
    os.replace(temporary, destination)
    return destination


def acquire_weights(
    *, input_root: Path, cache_path: Path, explicit: Path | None, allow_download: bool
) -> Path:
    """Reuse a verified checkpoint, then an attached one, then the pinned URL."""

    if cache_path.is_file():
        _verify_weight(cache_path)
        log(f"reusing verified weights: {cache_path}")
        return cache_path
    if explicit is not None:
        return _copy_weight(explicit.expanduser().resolve(), cache_path)
    for candidate in find_named_paths(input_root, "vgg_seq2seq.pth", directories=False):
        try:
            _verify_weight(candidate)
        except (OSError, ValueError):
            continue
        log(f"copying verified attached weights: {candidate}")
        return _copy_weight(candidate, cache_path)
    if not allow_download:
        raise FileNotFoundError("pinned vgg_seq2seq.pth is not attached and download is disabled")

    log(f"downloading pinned VietOCR weights: {WEIGHTS_URL}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(WEIGHTS_URL, headers={"User-Agent": "aic2026-phase2/1"})
    with urllib.request.urlopen(request, timeout=60) as reader, temporary.open("xb") as writer:
        while chunk := reader.read(1024 * 1024):
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    try:
        _verify_weight(temporary)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, cache_path)
    return cache_path


def _environment_report(python: Path, device: str) -> dict[str, Any] | None:
    if not python.is_file():
        return None
    code = (
        "import importlib.metadata as m,json,torch,torchvision;"
        "from vietocr.tool.config import Cfg;"
        "from vietocr.tool.predictor import Predictor;"
        f"names={list(ENV_EXPECTED_VERSIONS)!r};"
        "print(json.dumps({'versions':{n:m.version(n) for n in names},"
        "'cuda':torch.cuda.is_available(),'cuda_count':torch.cuda.device_count(),"
        "'cuda_devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],"
        "'torch':m.version('torch'),"
        "'torchvision':m.version('torchvision')}))"
    )
    result = subprocess.run([str(python), "-c", code], text=True, capture_output=True)
    if result.returncode != 0:
        return None
    try:
        report = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return None
    if report.get("versions") != ENV_EXPECTED_VERSIONS:
        return None
    if device.startswith("cuda") and report.get("cuda") is not True:
        return None
    return report


def setup_environment(env_root: Path, device: str, input_root: Path) -> Path:
    """Create once and thereafter reuse a Phase-2-only venv."""

    python = env_root / "bin/python"
    report = _environment_report(python, device)
    if report is not None:
        log(f"reusing verified Phase 2 environment: {env_root}")
        return python
    if env_root.exists():
        shutil.rmtree(env_root)

    candidates = [
        item
        for item in find_named_paths(input_root, env_root.name, directories=True)
        if _environment_report(item / "bin/python", device) is not None
    ]
    if candidates:
        source = candidates[0]
        temporary = env_root.with_name(f".{env_root.name}.restoring-{os.getpid()}")
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(source, temporary, symlinks=True)
        os.replace(temporary, env_root)
        if _environment_report(python, device) is not None:
            log(f"restored verified Phase 2 environment: {source}")
            return python
        shutil.rmtree(env_root)

    # Kaggle's injected sitecustomize can hang ``python -m venv``.  Construct
    # the minimal PEP 405 prefix directly, while retaining the image's matched
    # torch/torchvision CUDA pair via system site-packages.
    executable = Path(sys.executable).resolve()
    (env_root / "bin").mkdir(parents=True)
    os.symlink(executable, python)
    (env_root / "pyvenv.cfg").write_text(
        "\n".join(
            (
                f"home = {executable.parent}",
                "include-system-site-packages = true",
                "version = "
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                f"executable = {executable}",
                "",
            )
        ),
        encoding="utf-8",
    )
    site_packages = (
        env_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    site_packages.mkdir(parents=True)
    pip_environment = os.environ.copy()
    pip_environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    pip_environment["PIP_DEFAULT_TIMEOUT"] = "60"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(site_packages),
            "--ignore-installed",
            "--no-cache-dir",
            *ENV_REQUIREMENTS,
        ],
        env=pip_environment,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(site_packages),
            "--ignore-installed",
            "--no-cache-dir",
            "--no-deps",
            "vietocr==0.3.13",
        ],
        env=pip_environment,
        check=True,
    )
    report = _environment_report(python, device)
    if report is None:
        raise RuntimeError(
            "Phase 2 environment preflight failed; Kaggle must supply compatible torch/torchvision"
        )
    atomic_json(env_root / ".aic-phase2-ready.json", report)
    return python


def discover_phase1_root(input_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        candidates = [explicit.expanduser().resolve()]
    else:
        manifests = find_named_paths(input_root, "global-shards.json", directories=False)
        manifests = [
            item
            for item in manifests
            if item.parent.name == "shards" and item.parent.parent.name == "manifests"
        ]
        working = Path("/kaggle/working")
        if working.is_dir():
            working_manifests = find_named_paths(
                working, "global-shards.json", directories=False
            )
            manifests.extend(
                item
                for item in working_manifests
                if item.parent.name == "shards" and item.parent.parent.name == "manifests"
            )
        candidates = sorted({item.parents[2].resolve() for item in manifests})
    valid = [
        item
        for item in candidates
        if (item / "manifests/shards/global-shards.json").is_file()
        and (item / "phase1").is_dir()
    ]
    if len(valid) != 1:
        raise ValueError(
            f"expected exactly one Phase 1 output root, found {len(valid)}; pass --phase1-root"
        )
    return valid[0]


def _completed_phase1_artifact(path: Path, *, stage: str, shard_id: str) -> dict[str, Any]:
    marker = receipt_path(path)
    if not path.is_file() or not marker.is_file():
        raise IncompletePhase1Shard(path)
    value = read_json(marker)
    if value.get("status") != "completed" or value.get("stage") != stage:
        raise IncompletePhase1Shard(f"Phase 1 {stage} is not completed: {shard_id}")
    if value.get("shard_id") != shard_id or value.get("output_sha256") != sha256_file(path):
        raise ValueError(f"Phase 1 {stage} identity/checksum mismatch: {shard_id}")
    return value


def discover_completed_shards(
    phase1_root: Path,
    *,
    phase1_run_id: str | None,
    selected: set[str],
) -> tuple[str, str, int, int, list[Phase1Shard]]:
    global_path = phase1_root / "manifests/shards/global-shards.json"
    global_receipt = read_json(receipt_path(global_path))
    if (
        global_receipt.get("status") != "completed"
        or global_receipt.get("global_manifest_sha256") != sha256_file(global_path)
    ):
        raise ValueError("global Phase 1 shard manifest is not completed or has drifted")
    global_value = read_json(global_path)
    shard_values = global_value.get("shards")
    if not isinstance(shard_values, list) or not shard_values:
        raise ValueError("global Phase 1 manifest contains no shards")

    phase1_dir = phase1_root / "phase1"
    if phase1_run_id is None:
        run_dirs = sorted(item for item in phase1_dir.iterdir() if item.is_dir())
        if len(run_dirs) != 1:
            raise ValueError("cannot infer a unique Phase 1 run; pass --phase1-run-id")
        run_dir = run_dirs[0]
        phase1_run_id = run_dir.name
    else:
        run_dir = phase1_dir / phase1_run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    known = {str(item.get("shard_id")) for item in shard_values}
    unknown = selected - known
    if unknown:
        raise ValueError(f"unknown requested shard(s): {sorted(unknown)}")
    completed: list[Phase1Shard] = []
    config_hash: str | None = None
    for item in shard_values:
        shard_id = str(item.get("shard_id"))
        if selected and shard_id not in selected:
            continue
        raw_relpath = item.get("manifest_relpath")
        if not isinstance(raw_relpath, str):
            raise ValueError(f"invalid shard manifest path: {shard_id}")
        frame_manifest = (global_path.parent / raw_relpath).resolve()
        frame_manifest.relative_to(global_path.parent.resolve())
        if (
            not frame_manifest.is_file()
            or sha256_file(frame_manifest) != item.get("manifest_sha256")
        ):
            raise ValueError(f"Phase 1 shard manifest checksum mismatch: {shard_id}")
        root = run_dir / shard_id
        shard = Phase1Shard(
            shard_id=shard_id,
            frame_count=int(item.get("frame_count", 0)),
            frame_manifest=frame_manifest,
            detections=root / "detections.jsonl",
            trajectories=root / "trajectories.jsonl",
            representatives=root / "representatives.jsonl",
        )
        try:
            receipts = (
                _completed_phase1_artifact(
                    shard.detections, stage="detect_crop", shard_id=shard_id
                ),
                _completed_phase1_artifact(shard.trajectories, stage="track", shard_id=shard_id),
                _completed_phase1_artifact(
                    shard.representatives, stage="select_representatives", shard_id=shard_id
                ),
            )
        except IncompletePhase1Shard:
            if selected:
                raise ValueError(f"requested Phase 1 shard is incomplete: {shard_id}") from None
            log(f"skipping incomplete Phase 1 shard: {shard_id}")
            continue
        hashes = {str(value.get("config_sha256")) for value in receipts}
        run_ids = {str(value.get("run_id")) for value in receipts}
        if hashes != {str(global_value.get("config_sha256"))} or run_ids != {phase1_run_id}:
            raise ValueError(f"Phase 1 run/config identity mismatch: {shard_id}")
        config_hash = hashes.pop()
        completed.append(shard)
    if not completed or config_hash is None:
        raise ValueError("no completed Phase 1 shards are available")
    total_frames = int(global_receipt.get("frame_count", 0))
    return phase1_run_id, config_hash, len(shard_values), total_frames, completed


def discover_resume_root(input_root: Path, explicit: Path | None) -> Path | None:
    """Find at most one prior saved Phase 2 work root mounted read-only."""

    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not (root / "stages").is_dir():
            raise FileNotFoundError(f"prior Phase 2 root has no stages directory: {root}")
        return root
    candidates = [
        item
        for item in find_named_paths(input_root, "ocr-phase2-e2e-v1", directories=True)
        if (item / "stages").is_dir()
    ]
    if len(candidates) > 1:
        raise ValueError("multiple prior Phase 2 roots found; pass --resume-from")
    return candidates[0] if candidates else None


def _copy_resume_file(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_file() and sha256_file(destination) == sha256_file(source):
            return
        raise ValueError(f"current Phase 2 work conflicts with prior saved work: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".restore-tmp")
    temporary.unlink(missing_ok=True)
    shutil.copyfile(source, temporary)
    if sha256_file(temporary) != sha256_file(source):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"prior Phase 2 artifact changed while copying: {source}")
    os.replace(temporary, destination)


def restore_prior_shard(prior_root: Path, work_root: Path, shard_id: str) -> bool:
    """Restore only receipt-bound Phase 2 files; the core re-verifies them on resume."""

    source_root = prior_root / "stages" / shard_id
    if not source_root.is_dir():
        return False
    destination_root = work_root / "stages" / shard_id
    recognition_receipt = source_root / "recognition.jsonl.receipt.json"
    restored: list[tuple[Path, Path]] = []
    if recognition_receipt.is_file():
        marker = read_json(recognition_receipt)
        if marker.get("model_weights_sha256") != WEIGHTS_SHA256:
            raise ValueError(f"prior recognition uses a different model: {shard_id}")
        status = marker.get("status")
        payload_name = "recognition.jsonl" if status == "completed" else "recognition.jsonl.partial"
        payload = source_root / payload_name
        if status not in {"running", "completed"} or not payload.is_file():
            raise ValueError(f"prior recognition receipt/payload is incomplete: {shard_id}")
        committed_bytes = int(marker.get("committed_bytes", -1))
        if committed_bytes < 0 or payload.stat().st_size < committed_bytes:
            raise ValueError(f"prior recognition committed byte count drift: {shard_id}")
        digest = hashlib.sha256()
        with payload.open("rb") as stream:
            remaining = committed_bytes
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
        if remaining or digest.hexdigest() != marker.get("committed_sha256"):
            raise ValueError(f"prior recognition committed prefix checksum drift: {shard_id}")
        restored.extend(
            [
                (payload, destination_root / payload.name),
                (recognition_receipt, destination_root / recognition_receipt.name),
            ]
        )
    for payload_name, marker_name in (
        ("consensus.jsonl", "consensus.jsonl.receipt.json"),
        ("final.jsonl", "final.manifest.json"),
    ):
        payload = source_root / payload_name
        marker = source_root / marker_name
        if payload.is_file() != marker.is_file():
            raise ValueError(
                f"prior downstream Phase 2 pair is incomplete: {shard_id}/{payload_name}"
            )
        if payload.is_file():
            restored.extend(
                [
                    (payload, destination_root / payload.name),
                    (marker, destination_root / marker.name),
                ]
            )
    # Payloads are copied before commit markers so an interrupted restore never
    # advertises bytes which have not reached writable storage.
    restored.sort(key=lambda pair: pair[0].name.endswith(("receipt.json", "manifest.json")))
    for source, destination in restored:
        _copy_resume_file(source, destination)
    return bool(restored)


def run(command: list[str], *, deadline: float | None = None) -> None:
    log("RUN " + " ".join(command))
    if deadline is not None and time.monotonic() >= deadline:
        raise SoftStop
    process = subprocess.Popen(command, cwd=REPO_ROOT, text=True)
    while True:
        try:
            returncode = process.wait(timeout=5)
            break
        except subprocess.TimeoutExpired:
            if deadline is None or time.monotonic() < deadline:
                continue
            log(f"soft-stop reached; interrupting pid={process.pid}")
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=30)
            raise SoftStop from None
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def smoke_model(
    *,
    python: Path,
    model_config: Path,
    weights: Path,
    device: str,
    environment: dict[str, Any],
    source_commit_sha: str,
    marker: Path,
    deadline: float,
) -> None:
    """Load the pinned model and run one real image before starting four workers."""

    identity = {
        "schema_version": "aic26.ocr_phase2_model_smoke.v1",
        "status": "completed",
        "source_commit_sha": source_commit_sha,
        "model_config_sha256": sha256_file(model_config),
        "model_weights_sha256": sha256_file(weights),
        "device": device,
        "environment": environment,
    }
    if marker.is_file():
        try:
            if read_json(marker) == identity:
                log("reusing completed VietOCR model/inference smoke")
                return
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    code = (
        "import sys; from pathlib import Path; from PIL import Image;"
        "sys.path.insert(0, sys.argv[1]);"
        "from aic2026.ocr.local_recognition import VietOcrRecognizer;"
        "recognizer=VietOcrRecognizer.create(config_path=Path(sys.argv[2]),"
        "weights_path=Path(sys.argv[3]),device=sys.argv[4],"
        "expected_weights_sha256=sys.argv[5]);"
        "prediction=recognizer.predict(Image.new('RGB',(320,48),'white'));"
        "assert isinstance(prediction.transcript_raw,str);"
        "print('VietOCR smoke passed')"
    )
    run(
        [
            str(python),
            "-c",
            code,
            str(REPO_ROOT / "offline/src"),
            str(model_config),
            str(weights),
            device,
            WEIGHTS_SHA256,
        ],
        deadline=deadline,
    )
    atomic_json(marker, identity)


def _verified_output(path: Path, marker: Path, *, kind: str, run_id: str) -> bool:
    if not path.exists() and not marker.exists():
        return False
    if not path.is_file() or not marker.is_file():
        # Downstream consensus/final stages are deterministic and publish atomically;
        # an unpaired file is an uncommitted crash remnant owned by this launcher.
        path.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
        return False
    value = read_json(marker)
    if value.get("status") != "completed" or value.get("run_id") != run_id:
        raise ValueError(f"existing {kind} marker has incompatible identity: {marker}")
    expected = value.get("output_sha256")
    if kind == "final":
        outputs = value.get("outputs")
        expected = outputs[0].get("sha256") if isinstance(outputs, list) and outputs else None
    if expected != sha256_file(path):
        raise ValueError(f"existing {kind} output checksum mismatch: {path}")
    return True


def process_shard(
    *,
    shard: Phase1Shard,
    python: Path,
    data_root: Path,
    work_root: Path,
    phase1_config: Path,
    model_config: Path,
    weights: Path,
    device: str,
    source_commit_sha: str,
    phase1_run_id: str,
    phase1_config_sha256: str,
    consensus_run_id: str,
    final_run_id: str,
    batch_size: int,
    commit_interval: int,
    deadline: float | None,
) -> dict[str, Path]:
    output_root = work_root / "stages" / shard.shard_id
    output_root.mkdir(parents=True, exist_ok=True)
    recognition = output_root / "recognition.jsonl"
    recognition_receipt = receipt_path(recognition)
    recognition_done = False
    if recognition.is_file() and recognition_receipt.is_file():
        marker = read_json(recognition_receipt)
        recognition_done = (
            marker.get("status") == "completed"
            and marker.get("run_id") == phase1_run_id
            and marker.get("source_commit_sha") == source_commit_sha
            and marker.get("phase1_config_sha256") == phase1_config_sha256
            and marker.get("model_weights_sha256") == WEIGHTS_SHA256
            and marker.get("model_config_sha256") == sha256_file(model_config)
            and marker.get("output_sha256") == sha256_file(recognition)
            and marker.get("frame_manifest_sha256") == sha256_file(shard.frame_manifest)
            and marker.get("detections_sha256") == sha256_file(shard.detections)
            and marker.get("trajectories_sha256") == sha256_file(shard.trajectories)
            and marker.get("representatives_sha256") == sha256_file(shard.representatives)
            and marker.get("committed_bytes") == recognition.stat().st_size
            and marker.get("batch_size") == batch_size
            and marker.get("commit_interval_records") == commit_interval
        )
        if marker.get("status") == "completed" and not recognition_done:
            raise ValueError(f"completed recognition identity drift: {shard.shard_id}")
    if not recognition_done:
        run(
            [
                str(python),
                str(PHASE2_RUNNER),
                "run-representatives",
                "--phase1-config",
                str(phase1_config),
                "--frame-manifest",
                str(shard.frame_manifest),
                "--data-root",
                str(data_root),
                "--detections",
                str(shard.detections),
                "--trajectories",
                str(shard.trajectories),
                "--representatives",
                str(shard.representatives),
                "--output",
                str(recognition),
                "--model-config",
                str(model_config),
                "--weights",
                str(weights),
                "--weights-sha256",
                WEIGHTS_SHA256,
                "--device",
                device,
                "--source-commit-sha",
                source_commit_sha,
                "--batch-size",
                str(batch_size),
                "--commit-interval-records",
                str(commit_interval),
                "--resume",
            ],
            deadline=deadline,
        )

    consensus = output_root / "consensus.jsonl"
    if not _verified_output(
        consensus,
        receipt_path(consensus),
        kind="consensus",
        run_id=consensus_run_id,
    ):
        run(
            [
                str(python),
                str(PHASE2_RUNNER),
                "consensus",
                "--trajectories",
                str(shard.trajectories),
                "--representatives",
                str(shard.representatives),
                "--recognition-output",
                str(recognition),
                "--output",
                str(consensus),
                "--run-id",
                consensus_run_id,
            ],
            deadline=deadline,
        )
    consensus_marker = read_json(receipt_path(consensus))
    if (
        consensus_marker.get("model_revision") != WEIGHTS_SHA256
        or consensus_marker.get("trajectories_sha256") != sha256_file(shard.trajectories)
        or consensus_marker.get("representatives_sha256") != sha256_file(shard.representatives)
        or consensus_marker.get("recognition_output_sha256") != sha256_file(recognition)
        or consensus_marker.get("recognition_receipt_sha256")
        != sha256_file(recognition_receipt)
    ):
        raise ValueError(f"consensus provenance drift: {shard.shard_id}")

    final = output_root / "final.jsonl"
    final_manifest = final.with_suffix(".manifest.json")
    if not _verified_output(final, final_manifest, kind="final", run_id=final_run_id):
        run(
            [
                str(python),
                str(PHASE2_RUNNER),
                "build-final",
                "--trajectories",
                str(shard.trajectories),
                "--consensus",
                str(consensus),
                "--output",
                str(final),
                "--run-id",
                final_run_id,
            ],
            deadline=deadline,
        )
    final_value = read_json(final_manifest)
    final_input_hashes = {
        item.get("sha256") for item in final_value.get("inputs", []) if isinstance(item, dict)
    }
    if final_input_hashes != {
        sha256_file(shard.trajectories),
        sha256_file(consensus),
        sha256_file(receipt_path(consensus)),
    }:
        raise ValueError(f"final OCR provenance drift: {shard.shard_id}")
    return {
        "recognition": recognition,
        "recognition_receipt": recognition_receipt,
        "consensus": consensus,
        "consensus_receipt": receipt_path(consensus),
        "final": final,
        "final_manifest": final_manifest,
    }


def _file_entry(path: Path, relative: str) -> dict[str, Any]:
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_bundle(bundle: Path) -> dict[str, Any]:
    index_path = bundle / "bundle-index.json"
    sums_path = bundle / "SHA256SUMS"
    index = read_json(index_path)
    files = index.get("files")
    if not isinstance(files, list):
        raise ValueError("bundle index has no file list")
    expected_paths: set[str] = set()
    expected_lines: list[str] = []
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("invalid bundle file entry")
        relative = entry["path"]
        path = (bundle / relative).resolve()
        path.relative_to(bundle.resolve())
        if relative in expected_paths or not path.is_file():
            raise ValueError("bundle has a duplicate or missing member")
        expected_paths.add(relative)
        if entry.get("bytes") != path.stat().st_size or entry.get("sha256") != sha256_file(path):
            raise ValueError(f"bundle member checksum drift: {relative}")
        expected_lines.append(f"{entry['sha256']}  {relative}\n")
    index_hash = sha256_file(index_path)
    expected_lines.append(f"{index_hash}  bundle-index.json\n")
    if sums_path.read_text(encoding="utf-8") != "".join(sorted(expected_lines)):
        raise ValueError("bundle SHA256SUMS drift")
    actual = {
        item.relative_to(bundle).as_posix() for item in bundle.rglob("*") if item.is_file()
    }
    if actual != expected_paths | {"bundle-index.json", "SHA256SUMS"}:
        raise ValueError("bundle contains missing or undeclared files")
    return index


def publish_bundle(
    *,
    bundle_root: Path,
    shards: list[Phase1Shard],
    outputs: dict[str, dict[str, Path]],
    source_commit_sha: str,
    phase1_run_id: str,
    phase1_config_sha256: str,
    total_phase1_shards: int,
    total_source_frames: int,
    model_config: Path,
    consensus_run_id: str,
    final_run_id: str,
) -> Path:
    bundle_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".publishing-", dir=bundle_root))
    try:
        files: list[dict[str, Any]] = []
        shard_entries: list[dict[str, Any]] = []
        for shard in sorted(shards, key=lambda item: item.shard_id):
            output_entries: list[dict[str, Any]] = []
            for role, source in sorted(outputs[shard.shard_id].items()):
                if role == "final":
                    relative = f"ocr/{final_run_id}/{shard.shard_id}.jsonl"
                elif role == "final_manifest":
                    relative = f"ocr/{final_run_id}/{shard.shard_id}.manifest.json"
                else:
                    relative = f"evidence/shards/{shard.shard_id}/{source.name}"
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                entry = _file_entry(destination, relative)
                entry["role"] = role
                files.append(entry)
                output_entries.append(entry)
            inputs = []
            for role, source in (
                ("frame_manifest", shard.frame_manifest),
                ("detections", shard.detections),
                ("detections_receipt", receipt_path(shard.detections)),
                ("trajectories", shard.trajectories),
                ("trajectories_receipt", receipt_path(shard.trajectories)),
                ("representatives", shard.representatives),
                ("representatives_receipt", receipt_path(shard.representatives)),
            ):
                entry = _file_entry(source, role)
                entry["role"] = role
                entry.pop("path")
                inputs.append(entry)
            shard_entries.append(
                {
                    "shard_id": shard.shard_id,
                    "inputs": inputs,
                    "outputs": output_entries,
                }
            )
        files.sort(key=lambda item: item["path"])
        index = {
            "schema_version": "aic26.ocr_phase2_kaggle_bundle.v1",
            "status": "completed",
            "source_commit_sha": source_commit_sha,
            "phase1": {
                "run_id": phase1_run_id,
                "config_sha256": phase1_config_sha256,
                "total_shards": total_phase1_shards,
                "total_source_frames": total_source_frames,
                "processed_source_frames": sum(item.frame_count for item in shards),
                "processed_shards": [
                    item.shard_id for item in sorted(shards, key=lambda x: x.shard_id)
                ],
            },
            "phase2": {
                "consensus_run_id": consensus_run_id,
                "final_run_id": final_run_id,
                "emitted_ocr_frames": sum(
                    int(read_json(outputs[item.shard_id]["final_manifest"])["counters"]["frames"])
                    for item in shards
                ),
                "ingest_root": ".",
            },
            "model": {
                "model_id": MODEL_ID,
                "revision": WEIGHTS_SHA256,
                "weights_bytes": WEIGHTS_BYTES,
                "weights_url": WEIGHTS_URL,
                "config_sha256": sha256_file(model_config),
            },
            "shards": shard_entries,
            "files": files,
            "trust_boundary": "integrity_metadata_not_a_signature",
        }
        atomic_json(staging / "bundle-index.json", index)
        checksum_lines = [f"{item['sha256']}  {item['path']}\n" for item in files]
        checksum_lines.append(f"{sha256_file(staging / 'bundle-index.json')}  bundle-index.json\n")
        sums = staging / "SHA256SUMS"
        with sums.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("".join(sorted(checksum_lines)))
            stream.flush()
            os.fsync(stream.fileno())
        index_hash = sha256_file(staging / "bundle-index.json")
        target = bundle_root / f"bundle-{index_hash[:16]}"
        if target.exists():
            verify_bundle(target)
            if sha256_file(target / "bundle-index.json") != index_hash:
                raise FileExistsError(f"content-addressed bundle collision: {target}")
            return target
        os.replace(staging, target)
        verify_bundle(target)
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _device_for_shard(shard_id: str, devices: list[str], workers_per_gpu: int) -> str:
    try:
        ordinal = int(shard_id.rsplit("-", 1)[1]) - 1
    except (IndexError, ValueError):
        ordinal = int(hashlib.sha256(shard_id.encode("utf-8")).hexdigest(), 16)
    slots = [device for _ in range(workers_per_gpu) for device in devices]
    return slots[ordinal % len(slots)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--phase1-root", type=Path)
    parser.add_argument("--phase1-run-id")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Prior saved /kaggle/input/.../ocr-phase2-e2e-v1 root; auto-detected if unique.",
    )
    parser.add_argument("--phase1-config", type=Path, default=DEFAULT_PHASE1_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--no-weight-download", action="store_true")
    parser.add_argument(
        "--work-root", type=Path, default=Path("/kaggle/working/ocr-phase2-e2e-v1")
    )
    parser.add_argument(
        "--env-root", type=Path, default=Path("/kaggle/working/phase2-vietocr-env")
    )
    parser.add_argument("--phase2-python", type=Path)
    parser.add_argument(
        "--devices",
        default="cuda:0,cuda:1",
        help="Comma-separated CUDA devices; defaults to both Kaggle T4 GPUs.",
    )
    parser.add_argument("--device", help="Compatibility shortcut for a single device.")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument(
        "--soft-stop-seconds",
        type=int,
        default=34_200,
        help="Interrupt inference after this launcher runtime so Kaggle can save output.",
    )
    parser.add_argument("--source-commit-sha")
    parser.add_argument("--consensus-run-id", default="vietocr-local-consensus-v1")
    parser.add_argument("--final-run-id", default="vietocr-local-v1")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--commit-interval-records", type=int, default=256)
    parser.add_argument("--shard", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = build_parser().parse_args(argv)
    working = Path("/kaggle/working").resolve()
    if not working.is_dir():
        raise RuntimeError("this launcher is intended for Kaggle Save & Run All")
    work_root = args.work_root.expanduser().resolve()
    work_root.relative_to(working)
    input_root = args.input_root.expanduser().resolve()
    data_root = (args.data_root or input_root).expanduser().resolve()
    phase1_root = discover_phase1_root(input_root, args.phase1_root)
    phase1_run_id, config_hash, total_shards, total_frames, shards = discover_completed_shards(
        phase1_root,
        phase1_run_id=args.phase1_run_id,
        selected=set(args.shard),
    )
    log(f"found {len(shards)}/{total_shards} completed Phase 1 shard(s)")
    source_commit_sha = args.source_commit_sha or _git_head()
    if not (7 <= len(source_commit_sha) <= 64) or any(
        character not in "0123456789abcdef" for character in source_commit_sha
    ):
        raise ValueError("source commit must be 7..64 lowercase hexadecimal characters")

    devices = [args.device] if args.device else [item.strip() for item in args.devices.split(",")]
    if not devices or any(not item for item in devices):
        raise ValueError("--devices must contain at least one non-empty device")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicates")
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be positive")
    if args.soft_stop_seconds < 300:
        raise ValueError("--soft-stop-seconds must leave at least five minutes of useful runtime")
    deadline = started + args.soft_stop_seconds

    if args.phase2_python is None:
        env_root = args.env_root.expanduser().resolve()
        env_root.relative_to(working)
        python = setup_environment(env_root, devices[0], input_root)
    else:
        python = args.phase2_python.expanduser().resolve()
        if _environment_report(python, devices[0]) is None:
            raise RuntimeError("--phase2-python does not identify a verified Phase 2 environment")
    environment = _environment_report(python, devices[0])
    if environment is None:
        raise RuntimeError("Phase 2 environment changed after setup")
    cuda_indexes = [
        int(device.split(":", 1)[1])
        for device in devices
        if device.startswith("cuda:") and device.split(":", 1)[1].isdigit()
    ]
    if cuda_indexes and max(cuda_indexes) >= int(environment["cuda_count"]):
        raise RuntimeError(
            f"requested {devices}, but only {environment['cuda_count']} CUDA device(s) are visible"
        )
    weights = acquire_weights(
        input_root=input_root,
        cache_path=work_root / "models/vgg_seq2seq.pth",
        explicit=args.weights,
        allow_download=not args.no_weight_download,
    )
    smoke_model(
        python=python,
        model_config=args.model_config.expanduser().resolve(),
        weights=weights,
        device=devices[0],
        environment=environment,
        source_commit_sha=source_commit_sha,
        marker=work_root / "model-smoke.json",
        deadline=deadline,
    )
    resume_root = discover_resume_root(input_root, args.resume_from)
    if resume_root is not None:
        restored = sum(
            restore_prior_shard(resume_root, work_root, item.shard_id) for item in shards
        )
        log(f"restored prior Phase 2 state for {restored} shard(s) from {resume_root}")

    outputs: dict[str, dict[str, Path]] = {}
    executors = {
        device: ThreadPoolExecutor(
            max_workers=args.workers_per_gpu,
            thread_name_prefix=f"phase2-{device.replace(':', '-')}",
        )
        for device in devices
    }
    futures: dict[Future[dict[str, Path]], Phase1Shard] = {}
    try:
        for shard in shards:
            device = _device_for_shard(shard.shard_id, devices, args.workers_per_gpu)
            log(f"queueing {shard.shard_id} on {device}")
            futures[
                executors[device].submit(
                    process_shard,
                    shard=shard,
                    python=python,
                    data_root=data_root,
                    work_root=work_root,
                    phase1_config=args.phase1_config.expanduser().resolve(),
                    model_config=args.model_config.expanduser().resolve(),
                    weights=weights,
                    device=device,
                    source_commit_sha=source_commit_sha,
                    phase1_run_id=phase1_run_id,
                    phase1_config_sha256=config_hash,
                    consensus_run_id=args.consensus_run_id,
                    final_run_id=args.final_run_id,
                    batch_size=args.batch_size,
                    commit_interval=args.commit_interval_records,
                    deadline=deadline,
                )
            ] = shard
        soft_stopped = False
        for future in as_completed(futures):
            shard = futures[future]
            try:
                outputs[shard.shard_id] = future.result()
            except SoftStop:
                soft_stopped = True
                log(f"checkpointed {shard.shard_id} at the Kaggle soft-stop")
            else:
                log(f"completed {shard.shard_id} ({len(outputs)}/{len(shards)})")
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)
    if soft_stopped:
        state = {
            "schema_version": "aic26.ocr_phase2_continuation_state.v1",
            "status": "soft_stopped",
            "completed_shards": sorted(outputs),
            "remaining_shards": sorted(set(item.shard_id for item in shards) - set(outputs)),
            "upload_performed": False,
        }
        atomic_json(work_root / "continuation-state.json", state)
        print(json.dumps(state, sort_keys=True))
        return 0
    bundle = publish_bundle(
        bundle_root=work_root / "final-bundles",
        shards=shards,
        outputs=outputs,
        source_commit_sha=source_commit_sha,
        phase1_run_id=phase1_run_id,
        phase1_config_sha256=config_hash,
        total_phase1_shards=total_shards,
        total_source_frames=total_frames,
        model_config=args.model_config.expanduser().resolve(),
        consensus_run_id=args.consensus_run_id,
        final_run_id=args.final_run_id,
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "processed_shards": len(shards),
                "total_phase1_shards": total_shards,
                "bundle": str(bundle),
                "bundle_index_sha256": sha256_file(bundle / "bundle-index.json"),
                "upload_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
