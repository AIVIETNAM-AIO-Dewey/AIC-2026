#!/usr/bin/env python3
"""Run resumable OCR Phase 1 inside a Kaggle Save & Run All version.

This bootstrap intentionally uses only the Python standard library.  It creates an
isolated pinned Paddle GPU environment, rebuilds the canonical manifest from the
mounted Kaggle datasets, and runs two independent shards on the two T4 devices.
The foreground process stops early enough to publish durable checkpoint bundles
and let Kaggle save ``/kaggle/working`` as a successful notebook output.
"""

from __future__ import annotations

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


def find_unique(pattern: str) -> Path:
    matches = sorted(INPUT_ROOT.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"required Kaggle input is missing: {pattern}")
    return matches[0]


def setup_environment() -> None:
    marker = ENV_ROOT / ".aic-phase1-gpu-ready"
    if marker.is_file():
        return
    if ENV_ROOT.exists():
        shutil.rmtree(ENV_ROOT)
    run([sys.executable, "-m", "venv", "--without-pip", str(ENV_ROOT)])
    site_packages = subprocess.check_output(
        [
            str(ENV_PYTHON),
            "-c",
            "import sysconfig; print(sysconfig.get_paths()['purelib'])",
        ],
        text=True,
    ).strip()
    wheelhouse = find_unique("aic-ocr-phase1-wheelhouse")
    packages = [
        "paddleocr==3.7.0",
        "paddlex==3.7.2",
        "pyclipper==1.4.0",
        "opencv-contrib-python==4.10.0.84",
        "Pillow==11.1.0",
        "numpy==1.26.4",
        "pyyaml==6.0.2",
        "pydantic==2.10.6",
        "wrapt==1.17.3",
    ]
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
        ]
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
        ]
    )
    run(
        [
            str(ENV_PYTHON),
            "-c",
            (
                "import paddle; "
                "assert paddle.device.cuda.device_count()==2; "
                "print(paddle.__version__, paddle.device.cuda.device_count())"
            ),
        ]
    )
    marker.write_text(SOURCE_COMMIT + "\n", encoding="utf-8")


def build_manifest() -> None:
    if MANIFEST.is_file() and sum(1 for _ in MANIFEST.open("rb")) == EXPECTED_FRAMES:
        return
    helper = WORK_ROOT / "build_source_manifest.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text(
        textwrap.dedent(
            f"""
            import re
            import sys
            from pathlib import Path

            repo = Path({str(REPO)!r})
            sys.path.insert(0, str(repo / "offline/src"))
            from aic2026.common import write_jsonl_atomic
            from aic2026.common.frame_manifest import build_frame_refs
            from aic2026.ocr.tracking import natural_key

            root = Path("/kaggle/input").resolve()
            output = Path({str(MANIFEST)!r})
            video_re = re.compile(r"^[A-Za-z0-9]+_V[0-9]+$")
            candidates = {{}}
            for map_dir in root.rglob("map-keyframes"):
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

            records = []
            for video_id in sorted(candidates, key=natural_key):
                choices = sorted(
                    candidates[video_id],
                    key=lambda pair: (
                        0 if "aic-test-dataset" in pair[0].as_posix() else 1,
                        pair[0].as_posix(),
                    ),
                )
                map_csv, frames = choices[0]
                records.extend(
                    build_frame_refs(
                        video_id=video_id,
                        map_csv=map_csv,
                        frames_dir=frames,
                        data_root=root,
                    )
                )
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
    run([str(ENV_PYTHON), str(helper)])


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
        )
    manifests = sorted(SHARD_ROOT.glob("shard-*.frames.jsonl"))
    if not manifests:
        raise RuntimeError("shard planner produced no frame manifests")
    return manifests


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
    model_file = find_unique(relative.as_posix())
    return model_file.parents[3]


def command_for(shard: Path, gpu: int) -> tuple[str, list[str], Path]:
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


def run_workers(shards: list[Path], started: float) -> None:
    pending = [item for item in shards if not completed(item.name.removesuffix(".frames.jsonl"))]
    status: dict[str, object] = {"source_commit": SOURCE_COMMIT, "total_shards": len(shards)}
    while pending and time.monotonic() - started < SOFT_STOP_SECONDS:
        group = pending[:2]
        processes: list[tuple[Path, str, subprocess.Popen[bytes], object]] = []
        for gpu, shard in enumerate(group):
            shard_id, command, log_path = command_for(shard, gpu)
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
            process.wait(timeout=60)
            handle.close()
        for shard, shard_id, process, _ in processes:
            status[shard_id] = {"returncode": process.returncode, "completed": completed(shard_id)}
            if completed(shard_id):
                pending.remove(shard)
            else:
                publish_checkpoint(shard)
        atomic_json(STATE, status)
        if stopping:
            break

    status["elapsed_seconds"] = round(time.monotonic() - started, 3)
    status["completed_shards"] = sum(
        completed(item.name.removesuffix(".frames.jsonl")) for item in shards
    )
    status["remaining_shards"] = len(shards) - int(status["completed_shards"])
    atomic_json(STATE, status)
    log(json.dumps(status, sort_keys=True))


def main() -> None:
    started = time.monotonic()
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("this launcher is intended for Kaggle Save & Run All")
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    setup_environment()
    build_manifest()
    shards = plan_shards()
    run_workers(shards, started)


if __name__ == "__main__":
    main()
