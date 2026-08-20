#!/usr/bin/env python3
"""Run resumable OCR Phase 1 inside a Kaggle Save & Run All version.

This bootstrap intentionally uses only the Python standard library.  It creates an
isolated pinned Paddle GPU environment, rebuilds the canonical manifest from the
mounted Kaggle datasets, and runs two independent shards on the two T4 devices.
The foreground process stops early enough to publish durable checkpoint bundles
and let Kaggle save ``/kaggle/working`` as a successful notebook output.
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
import textwrap
import time
from pathlib import Path

EXPECTED_FRAMES = 176_707
EXPECTED_VIDEOS = 873
SOFT_STOP_SECONDS = 9.5 * 60 * 60
OFFLINE_INSTALL_TIMEOUT_SECONDS = 15 * 60
PADDLE_INSTALL_TIMEOUT_SECONDS = 20 * 60
ENV_COPY_TIMEOUT_SECONDS = 15 * 60
GPU_PROBE_TIMEOUT_SECONDS = 120
SMOKE_TIMEOUT_SECONDS = 10 * 60
MANIFEST_TIMEOUT_SECONDS = 90 * 60
PLANNER_TIMEOUT_SECONDS = 5 * 60
MAX_WORKER_FAILURES = 2
RUN_ID = "ppocrv6-small-det-gpt4o-mini-high-v1-phase1"

REPO = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
).strip()
INPUT_ROOT = Path("/kaggle/input").resolve()
WORK_ROOT = Path("/kaggle/working/ocr-production-bg-v1").resolve()
ENV_ROOT = Path("/kaggle/working/phase1-gpu-prod-env").resolve()
ENV_PYTHON = ENV_ROOT / "bin/python"
CONFIG = REPO / "offline/configs/offline/ocr_phase1_kaggle_gpu.yaml"
RUNNER = REPO / "offline/scripts/run_ocr_phase1.py"
PLANNER = REPO / "offline/scripts/plan_ocr_phase1_shards.py"
MANIFEST = WORK_ROOT / "manifests/source.frames.jsonl"
SHARD_ROOT = WORK_ROOT / "manifests/shards"
GLOBAL_MANIFEST = SHARD_ROOT / "global-shards.json"
STATE = WORK_ROOT / "background-state.json"
SMOKE_ROOT = Path("/kaggle/working/ocr-phase1-smoke").resolve()
ENV_MARKER = ".aic-phase1-gpu-ready"
ENV_MARKER_VALUE = "aic26.phase1-gpu-env.ppocr-3.7.0-paddle-3.3.1-pydantic-2.10.6.v3"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    log("RUN " + " ".join(command))
    return subprocess.run(command, text=True, check=True, **kwargs)


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def find_input_directories(
    name: str,
    *,
    maximum_depth: int = 5,
    roots: tuple[Path, ...] | None = None,
) -> list[Path]:
    """Find attached notebook outputs without descending into frame collections."""

    if not INPUT_ROOT.is_dir():
        return []
    found: list[Path] = []
    pending = [(root, 0) for root in (roots or (INPUT_ROOT,))]
    skipped = {"keyframes", "map-keyframes", "crops", "state"}
    while pending:
        parent, depth = pending.pop()
        if depth >= maximum_depth:
            continue
        try:
            children = sorted(item for item in parent.iterdir() if item.is_dir())
        except OSError:
            continue
        for child in children:
            if child.name == name:
                found.append(child.resolve())
            elif child.name not in skipped:
                pending.append((child, depth + 1))
    return sorted(set(found))


def find_unique_input_directory(name: str) -> Path:
    matches = find_input_directories(name)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one Kaggle input directory named {name!r}, found {len(matches)}"
        )
    return matches[0]


def find_dataset_mount(slug: str) -> Path:
    """Resolve one Kaggle dataset slug without recursively scanning its files."""

    candidates = [INPUT_ROOT / slug]
    namespaced_root = INPUT_ROOT / "datasets"
    if namespaced_root.is_dir():
        try:
            owners = sorted(item for item in namespaced_root.iterdir() if item.is_dir())
        except OSError:
            owners = []
        candidates.extend(owner / slug for owner in owners)
    matches = sorted({item.resolve() for item in candidates if item.is_dir()})
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one mounted Kaggle dataset with slug {slug!r}, found {matches}"
        )
    return matches[0]


def environment_marker_matches(root: Path) -> bool:
    marker = root / ENV_MARKER
    python = root / "bin/python"
    try:
        value = marker.read_text(encoding="utf-8").strip()
        legacy_commit_marker = len(value) == 40 and all(
            character in "0123456789abcdef" for character in value
        )
        return python.is_file() and (value == ENV_MARKER_VALUE or legacy_commit_marker)
    except OSError:
        return False


def probe_environment() -> None:
    run(
        [
            str(ENV_PYTHON),
            "-c",
            (
                "import importlib.metadata as m; import numpy, paddle, wrapt; "
                "expected={'paddlepaddle-gpu':'3.3.1','paddleocr':'3.7.0',"
                "'paddlex':'3.7.2','pyclipper':'1.4.0',"
                "'opencv-contrib-python':'4.10.0.84','Pillow':'11.1.0',"
                "'numpy':'1.26.4','PyYAML':'6.0.2','pydantic':'2.10.6',"
                "'wrapt':'1.17.3','protobuf':'5.29.3',"
                "'opt-einsum':'3.3.0','networkx':'3.6.1',"
                "'safetensors':'0.8.0'}; "
                "assert all(m.version(k)==v for k,v in expected.items()), "
                "{k:m.version(k) for k in expected}; "
                "assert numpy.__version__=='1.26.4', numpy.__version__; "
                "assert paddle.device.cuda.device_count()==2; "
                "print(paddle.__version__, paddle.device.cuda.device_count())"
            ),
        ],
        timeout=GPU_PROBE_TIMEOUT_SECONDS,
    )


def restore_environment() -> bool:
    candidates = [
        item for item in find_input_directories(ENV_ROOT.name) if environment_marker_matches(item)
    ]
    if not candidates:
        return False
    source = candidates[0]
    temporary = ENV_ROOT.with_name(f".{ENV_ROOT.name}.restoring-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        run(
            ["cp", "-a", f"{source}/.", str(temporary)],
            timeout=ENV_COPY_TIMEOUT_SECONDS,
        )
        if not environment_marker_matches(temporary):
            raise RuntimeError("restored Phase 1 environment marker is invalid")
        os.replace(temporary, ENV_ROOT)
        probe_environment()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        if temporary.exists():
            shutil.rmtree(temporary)
        if ENV_ROOT.exists():
            shutil.rmtree(ENV_ROOT)
        log(f"prior environment restore failed; building fresh: {error}")
        return False
    log(f"restored compatible environment from {source}")
    return True


def setup_environment() -> None:
    marker = ENV_ROOT / ENV_MARKER
    if environment_marker_matches(ENV_ROOT):
        try:
            probe_environment()
            return
        except subprocess.SubprocessError:
            log("existing Phase 1 environment failed validation; rebuilding")
    if ENV_ROOT.exists():
        shutil.rmtree(ENV_ROOT)
    if restore_environment():
        return
    # Kaggle's ``venv`` bootstrap can hang inside its injected sitecustomize,
    # even with ``-S`` and ``--without-pip``.  Build the minimal PEP 405 prefix
    # directly instead: the system interpreter supplies the standard library,
    # while all third-party packages are installed into this private prefix.
    executable = Path(sys.executable).resolve()
    (ENV_ROOT / "bin").mkdir(parents=True)
    os.symlink(executable, ENV_PYTHON)
    (ENV_ROOT / "pyvenv.cfg").write_text(
        "\n".join(
            (
                f"home = {executable.parent}",
                "include-system-site-packages = false",
                "version = "
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                f"executable = {executable}",
                "",
            )
        ),
        encoding="utf-8",
    )
    site_packages_path = (
        ENV_ROOT
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    site_packages_path.mkdir(parents=True, exist_ok=True)
    site_packages = str(site_packages_path)
    wheelhouse = find_unique_input_directory("aic-ocr-phase1-wheelhouse")
    packages = [
        "paddleocr==3.7.0",
        "paddlex==3.7.2",
        "pyclipper==1.4.0",
        "opencv-contrib-python==4.10.0.84",
        "Pillow==11.1.0",
        "numpy==1.26.4",
        "pyyaml==6.0.2",
        # The attached wheelhouse currently carries 2.13.4.  Install it only to
        # satisfy offline dependency resolution, then replace it with the repo pin.
        "pydantic==2.13.4",
    ]
    pip_environment = os.environ.copy()
    pip_environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    pip_environment["PIP_DEFAULT_TIMEOUT"] = "60"
    # Install wrapt first so the next normal venv interpreter startup cannot be
    # broken by Kaggle's injected sitecustomize.
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            site_packages,
            "--ignore-installed",
            "--no-deps",
            "--no-cache-dir",
            "--only-binary=:all:",
            "--index-url",
            "https://pypi.org/simple",
            "wrapt==1.17.3",
        ],
        env=pip_environment,
        timeout=OFFLINE_INSTALL_TIMEOUT_SECONDS,
    )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            site_packages,
            "--ignore-installed",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            *packages,
        ],
        env=pip_environment,
        timeout=OFFLINE_INSTALL_TIMEOUT_SECONDS,
    )
    for pattern in (
        "pydantic",
        "pydantic-*.dist-info",
        "pydantic_core",
        "pydantic_core-*.dist-info",
    ):
        for stale in site_packages_path.glob(pattern):
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            site_packages,
            "--ignore-installed",
            "--upgrade",
            "--no-cache-dir",
            "--only-binary=:all:",
            "--index-url",
            "https://pypi.org/simple",
            "pydantic==2.10.6",
        ],
        env=pip_environment,
        timeout=OFFLINE_INSTALL_TIMEOUT_SECONDS,
    )
    # Paddle is installed with ``--no-deps`` below so its large wheel cannot
    # perturb the locked environment. Install its remaining direct runtime
    # requirements explicitly, also without dependency resolution: all of
    # their required shared packages are already supplied by the wheelhouse.
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            site_packages,
            "--ignore-installed",
            "--no-deps",
            "--no-cache-dir",
            "--only-binary=:all:",
            "--index-url",
            "https://pypi.org/simple",
            "protobuf==5.29.3",
            "opt_einsum==3.3.0",
            "networkx==3.6.1",
            "safetensors==0.8.0",
        ],
        env=pip_environment,
        timeout=OFFLINE_INSTALL_TIMEOUT_SECONDS,
    )
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            site_packages,
            "--ignore-installed",
            "--no-deps",
            "--no-cache-dir",
            "--index-url",
            "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
            "paddlepaddle-gpu==3.3.1",
        ],
        env=pip_environment,
        timeout=PADDLE_INSTALL_TIMEOUT_SECONDS,
    )
    probe_environment()
    temporary_marker = marker.with_suffix(".tmp")
    temporary_marker.write_text(ENV_MARKER_VALUE + "\n", encoding="utf-8")
    os.replace(temporary_marker, marker)


def build_manifest() -> None:
    if MANIFEST.is_file() and sum(1 for _ in MANIFEST.open("rb")) == EXPECTED_FRAMES:
        return
    dataset_roots = tuple(find_dataset_mount(slug) for slug in ("aic-test-dataset", "aic-26-video"))
    log("resolved dataset roots: " + ", ".join(map(str, dataset_roots)))
    log("discovering map-keyframes only inside mounted dataset roots")
    map_directories = find_input_directories("map-keyframes", maximum_depth=8, roots=dataset_roots)
    if not map_directories:
        raise FileNotFoundError("no mounted map-keyframes directory was found")
    log(f"found {len(map_directories)} map-keyframes directories")
    helper = WORK_ROOT / "build_source_manifest.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text(
        textwrap.dedent(
            f"""
            import os
            import re
            import sys
            from concurrent.futures import ThreadPoolExecutor
            from pathlib import Path

            repo = Path({str(REPO)!r})
            sys.path.insert(0, str(repo / "offline/src"))
            from aic2026.common import write_jsonl_atomic
            from aic2026.common.frame_manifest import build_frame_refs
            from aic2026.ocr.tracking import natural_key

            root = Path("/kaggle/input").resolve()
            output = Path({str(MANIFEST)!r})
            map_directories = [Path(value) for value in {list(map(str, map_directories))!r}]
            video_re = re.compile(r"^[A-Za-z0-9]+_V[0-9]+$")
            candidates = {{}}
            for map_dir in map_directories:
                if not map_dir.is_dir():
                    continue
                for map_csv in map_dir.glob("*.csv"):
                    if not video_re.fullmatch(map_csv.stem):
                        continue
                    frames = map_dir.parent / "keyframes" / map_csv.stem
                    if frames.is_dir():
                        candidates.setdefault(map_csv.stem, []).append((map_csv, frames))
            if len(candidates) != {EXPECTED_VIDEOS}:
                raise SystemExit(f"expected {EXPECTED_VIDEOS} videos, found {{len(candidates)}}")

            choices_by_video = []
            for video_id in sorted(candidates, key=natural_key):
                choices = sorted(
                    candidates[video_id],
                    key=lambda pair: (
                        0 if "aic-test-dataset" in pair[0].as_posix() else 1,
                        pair[0].as_posix(),
                    ),
                )
                map_csv, frames = choices[0]
                choices_by_video.append((video_id, map_csv, frames))

            def build_one(choice):
                video_id, map_csv, frames = choice
                return video_id, build_frame_refs(
                    video_id=video_id,
                    map_csv=map_csv,
                    frames_dir=frames,
                    data_root=root,
                )

            records = []
            worker_count = min(8, max(2, (os.cpu_count() or 2) * 2))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                for video_id, video_records in executor.map(build_one, choices_by_video):
                    records.extend(video_records)
                    print(f"manifest {{video_id}}: {{len(records)}}", flush=True)
            records.sort(key=lambda item: (natural_key(item.video_id), item.frame_idx))
            if len(records) != {EXPECTED_FRAMES}:
                raise SystemExit(
                    f"expected {EXPECTED_FRAMES} frames, built {{len(records)}}"
                )
            write_jsonl_atomic(output, records)
            print(f"manifest complete: {{len(records)}}", flush=True)
            """
        ),
        encoding="utf-8",
    )
    run([str(ENV_PYTHON), str(helper)], timeout=MANIFEST_TIMEOUT_SECONDS)


def plan_shards() -> list[Path]:
    receipt = SHARD_ROOT / "global-shards.json.receipt.json"
    if not receipt.is_file():
        if SHARD_ROOT.exists():
            shutil.move(SHARD_ROOT, SHARD_ROOT.with_name(f"shards.incomplete.{int(time.time())}"))
        run(
            [
                str(ENV_PYTHON),
                str(PLANNER),
                "plan",
                "--config",
                str(CONFIG),
                "--source-manifest",
                str(MANIFEST),
                "--output-dir",
                str(SHARD_ROOT),
            ],
            env=phase1_env(),
            timeout=PLANNER_TIMEOUT_SECONDS,
        )
    manifests = sorted(SHARD_ROOT.glob("shard-*.frames.jsonl"))
    if not manifests:
        raise RuntimeError("shard planner produced no frame manifests")
    return manifests


def run_one_frame_smoke() -> None:
    """Run real GPU detection, tracking, and selection on one canonical frame."""

    if SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)
    manifest_root = SMOKE_ROOT / "manifests"
    shard_root = manifest_root / "shards"
    source_manifest = manifest_root / "source.frames.jsonl"
    manifest_root.mkdir(parents=True)
    with MANIFEST.open("rb") as source:
        first_record = source.readline()
    if not first_record:
        raise RuntimeError("canonical source manifest is empty")
    with source_manifest.open("wb") as target:
        target.write(first_record)
        target.flush()
        os.fsync(target.fileno())
    run(
        [
            str(ENV_PYTHON),
            str(PLANNER),
            "plan",
            "--config",
            str(CONFIG),
            "--source-manifest",
            str(source_manifest),
            "--output-dir",
            str(shard_root),
        ],
        env=phase1_env(),
        timeout=PLANNER_TIMEOUT_SECONDS,
    )
    shards = sorted(shard_root.glob("shard-*.frames.jsonl"))
    if len(shards) != 1:
        raise RuntimeError(f"one-frame smoke planned {len(shards)} shards")
    shard = shards[0]
    shard_id = shard.name.removesuffix(".frames.jsonl")
    output = SMOKE_ROOT / "phase1" / RUN_ID / shard_id
    runtime = SMOKE_ROOT / "runtime" / shard_id
    output.mkdir(parents=True)
    runtime.mkdir(parents=True)
    run(
        [
            str(ENV_PYTHON),
            str(RUNNER),
            "run",
            "--config",
            str(CONFIG),
            "--cache-root",
            str(model_root()),
            "--runtime-cache-root",
            str(runtime),
            "--output-root",
            str(output),
            "--source-manifest",
            str(source_manifest),
            "--global-manifest",
            str(shard_root / "global-shards.json"),
            "--frame-manifest",
            str(shard),
            "--data-root",
            str(INPUT_ROOT),
            "--shard-id",
            shard_id,
            "--git-commit-sha",
            SOURCE_COMMIT,
        ],
        env=phase1_env(0),
        timeout=SMOKE_TIMEOUT_SECONDS,
    )
    receipt = output / "representatives.jsonl.receipt.json"
    try:
        status = json.loads(receipt.read_text(encoding="utf-8"))["status"]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError("one-frame smoke did not publish a valid receipt") from error
    if status != "completed":
        raise RuntimeError(f"one-frame smoke status is {status!r}")
    atomic_json(
        SMOKE_ROOT / "smoke-state.json",
        {"source_commit": SOURCE_COMMIT, "frames": 1, "status": "completed"},
    )
    log("one-frame GPU smoke completed")


def phase1_env(gpu: int | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "offline/src"), str(REPO / "offline/scripts")]
    )
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return environment


def model_root() -> Path:
    relative = Path("ocr/ppocrv6-small/detector/inference.pdiparams")
    candidates = [
        root
        for root in find_input_directories("aic-ocr-phase1-model-root")
        if (root / relative).is_file()
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            "expected one attached aic-ocr-phase1-model-root with pinned detector, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def checkpoint_chain_summary(root: Path, shard_id: str) -> tuple[int, tuple[str, ...]] | None:
    """Validate enough marker metadata to select a restore source.

    The Phase 1 runner performs the full hash and semantic verification before
    copying any checkpoint payload into writable storage.
    """

    if not root.is_dir():
        return None
    bundles = sorted(
        item for item in root.iterdir() if item.is_dir() and item.name.startswith("checkpoint-")
    )
    if not bundles:
        return None
    parsed: list[tuple[int, str, str | None]] = []
    for bundle in bundles:
        marker = bundle / "checkpoint.json"
        if not marker.is_file():
            raise ValueError(f"checkpoint bundle is missing commit marker: {bundle}")
        payload = marker.read_bytes()
        value = json.loads(payload)
        identity = (value.get("run_id"), value.get("git_commit_sha"), value.get("shard_id"))
        if identity != (RUN_ID, SOURCE_COMMIT, shard_id):
            return None
        sequence = value.get("checkpoint_sequence")
        previous = value.get("previous_checkpoint_sha256")
        if not isinstance(sequence, int) or sequence < 1:
            raise ValueError(f"invalid checkpoint sequence: {marker}")
        if previous is not None and (
            not isinstance(previous, str)
            or len(previous) != 64
            or any(character not in "0123456789abcdef" for character in previous)
        ):
            raise ValueError(f"invalid checkpoint predecessor: {marker}")
        parsed.append((sequence, hashlib.sha256(payload).hexdigest(), previous))
    parsed.sort()
    if [item[0] for item in parsed] != list(range(1, len(parsed) + 1)):
        raise ValueError(f"checkpoint sequence is not contiguous: {root}")
    if parsed[0][2] is not None:
        raise ValueError(f"checkpoint history does not start at sequence 1: {root}")
    for previous, current in zip(parsed, parsed[1:], strict=False):
        if current[2] != previous[1]:
            raise ValueError(f"checkpoint hash chain is broken: {root}")
    return parsed[-1][0], tuple(item[1] for item in parsed)


def discover_prior_checkpoints(shards: list[Path]) -> dict[str, Path]:
    prior_roots = find_input_directories(WORK_ROOT.name)
    selected: dict[str, Path] = {}
    for shard in shards:
        shard_id = shard.name.removesuffix(".frames.jsonl")
        receipt = WORK_ROOT / "phase1" / RUN_ID / shard_id / "detections.jsonl.receipt.json"
        if receipt.is_file():
            continue
        candidates: list[tuple[int, tuple[str, ...], Path]] = []
        for prior_root in prior_roots:
            history = prior_root / "checkpoints" / RUN_ID / shard_id
            try:
                summary = checkpoint_chain_summary(history, shard_id)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                log(f"ignoring invalid prior checkpoint {history}: {error}")
                continue
            if summary is not None:
                candidates.append((*summary, history))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], item[2].as_posix()), reverse=True)
        best = candidates[0]
        conflicts = [item for item in candidates if item[0] == best[0] and item[1] != best[1]]
        if conflicts:
            raise RuntimeError(f"ambiguous checkpoint histories for {shard_id}")
        selected[shard_id] = best[2]
        log(f"will restore {shard_id} checkpoint sequence {best[0]} from {best[2]}")
    return selected


def command_for(
    shard: Path, gpu: int, *, resume_from: Path | None = None
) -> tuple[str, list[str], Path]:
    shard_id = shard.name.removesuffix(".frames.jsonl")
    output = WORK_ROOT / "phase1" / RUN_ID / shard_id
    checkpoint = WORK_ROOT / "checkpoints" / RUN_ID / shard_id
    runtime = WORK_ROOT / "runtime" / shard_id
    for path in (output, checkpoint, runtime, WORK_ROOT / "logs"):
        path.mkdir(parents=True, exist_ok=True)
    command = [
        str(ENV_PYTHON),
        str(RUNNER),
        "run",
        "--config",
        str(CONFIG),
        "--cache-root",
        str(model_root()),
        "--runtime-cache-root",
        str(runtime),
        "--output-root",
        str(output),
        "--checkpoint-root",
        str(checkpoint),
        "--source-manifest",
        str(MANIFEST),
        "--global-manifest",
        str(GLOBAL_MANIFEST),
        "--frame-manifest",
        str(shard),
        "--data-root",
        str(INPUT_ROOT),
        "--shard-id",
        shard_id,
        "--git-commit-sha",
        SOURCE_COMMIT,
        "--resume",
    ]
    if resume_from is not None:
        command.extend(["--resume-from", str(resume_from)])
    return shard_id, command, WORK_ROOT / "logs" / f"{shard_id}.log"


def completed(shard_id: str) -> bool:
    receipt = WORK_ROOT / "phase1" / RUN_ID / shard_id / "representatives.jsonl.receipt.json"
    if not receipt.is_file():
        return False
    try:
        return json.loads(receipt.read_text(encoding="utf-8"))["status"] == "completed"
    except (KeyError, json.JSONDecodeError):
        return False


def publish_checkpoint(shard: Path) -> None:
    shard_id = shard.name.removesuffix(".frames.jsonl")
    output = WORK_ROOT / "phase1" / RUN_ID / shard_id
    receipt = output / "detections.jsonl.receipt.json"
    if not receipt.is_file():
        log(f"{shard_id} has no committed detection prefix yet")
        return
    command = [
        str(ENV_PYTHON),
        str(RUNNER),
        "checkpoint",
        "--config",
        str(CONFIG),
        "--source-manifest",
        str(MANIFEST),
        "--global-manifest",
        str(GLOBAL_MANIFEST),
        "--frame-manifest",
        str(shard),
        "--data-root",
        str(INPUT_ROOT),
        "--output-root",
        str(output),
        "--checkpoint-root",
        str(WORK_ROOT / "checkpoints" / RUN_ID / shard_id),
        "--shard-id",
        shard_id,
        "--git-commit-sha",
        SOURCE_COMMIT,
    ]
    try:
        run(command, env=phase1_env())
    except subprocess.CalledProcessError as error:
        log(f"checkpoint failed for {shard_id}: {error}")


def run_workers(shards: list[Path], started: float, prior_checkpoints: dict[str, Path]) -> None:
    pending = [item for item in shards if not completed(item.name.removesuffix(".frames.jsonl"))]
    status: dict[str, object] = {"source_commit": SOURCE_COMMIT, "total_shards": len(shards)}
    failures: dict[str, int] = {}
    while pending and time.monotonic() - started < SOFT_STOP_SECONDS:
        group = pending[:2]
        processes: list[tuple[Path, str, subprocess.Popen[bytes], object]] = []
        for gpu, shard in enumerate(group):
            shard_id = shard.name.removesuffix(".frames.jsonl")
            detection_receipt = (
                WORK_ROOT / "phase1" / RUN_ID / shard_id / "detections.jsonl.receipt.json"
            )
            resume_from = None if detection_receipt.is_file() else prior_checkpoints.get(shard_id)
            shard_id, command, log_path = command_for(shard, gpu, resume_from=resume_from)
            handle = log_path.open("ab", buffering=0)
            process = subprocess.Popen(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=phase1_env(gpu),
            )
            processes.append((shard, shard_id, process, handle))
            log(f"started {shard_id} on T4:{gpu} pid={process.pid}")

        stopping = False
        while any(process.poll() is None for _, _, process, _ in processes):
            if time.monotonic() - started >= SOFT_STOP_SECONDS:
                stopping = True
                for _, _, process, _ in processes:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline and any(
                    process.poll() is None for _, _, process, _ in processes
                ):
                    time.sleep(2)
                for _, _, process, _ in processes:
                    if process.poll() is None:
                        process.terminate()
                break
            time.sleep(15)

        for _, _, process, handle in processes:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
            handle.close()
        exhausted: list[str] = []
        for shard, shard_id, process, _ in processes:
            is_completed = completed(shard_id)
            if is_completed:
                pending.remove(shard)
                failures.pop(shard_id, None)
            else:
                publish_checkpoint(shard)
                if not stopping:
                    failures[shard_id] = failures.get(shard_id, 0) + 1
                    if failures[shard_id] >= MAX_WORKER_FAILURES:
                        exhausted.append(shard_id)
            status[shard_id] = {
                "returncode": process.returncode,
                "completed": is_completed,
                "failures": failures.get(shard_id, 0),
            }
        atomic_json(STATE, status)
        if exhausted:
            raise RuntimeError(
                "Phase 1 worker failure limit reached: " + ", ".join(sorted(exhausted))
            )
        if stopping:
            break

    status["elapsed_seconds"] = round(time.monotonic() - started, 3)
    status["completed_shards"] = sum(
        completed(item.name.removesuffix(".frames.jsonl")) for item in shards
    )
    status["remaining_shards"] = len(shards) - int(status["completed_shards"])
    atomic_json(STATE, status)
    log(json.dumps(status, sort_keys=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("setup", "smoke", "full"),
        default="full",
        help="setup dependencies only, run one real GPU frame, or launch all shards",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.monotonic()
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("this launcher is intended for Kaggle Save & Run All")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    setup_environment()
    if args.mode == "setup":
        log("Phase 1 environment setup completed")
        return
    build_manifest()
    if args.mode == "smoke":
        run_one_frame_smoke()
        return
    shards = plan_shards()
    prior_checkpoints = discover_prior_checkpoints(shards)
    run_workers(shards, started, prior_checkpoints)


if __name__ == "__main__":
    main()
