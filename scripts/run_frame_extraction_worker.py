#!/usr/bin/env python3
"""Run a resumable dual-GPU frame-extraction worker assignment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _log(message: str) -> None:
    print(f"[frame_worker] {message}", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--video-index", type=Path, required=True)
    parser.add_argument("--identity-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--entrypoint", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rclone-bin", required=True)
    parser.add_argument("--rclone-config", type=Path, required=True)
    parser.add_argument("--rclone-remote", default="gdrive")
    parser.add_argument("--drive-root-folder-id", required=True)
    parser.add_argument("--remote-root-name", default="self-cut-btc-compatible")
    parser.add_argument("--session-start-epoch", type=float, required=True)
    parser.add_argument("--accept-new-work-seconds", type=float, default=40500.0)
    parser.add_argument("--upload-attempts", type=int, default=10)
    parser.add_argument("--scan-attempts", type=int, default=5)
    parser.add_argument("--keep-local", action="store_true")
    return parser


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_streamed(
    command: list[str],
    *,
    prefix: str = "",
    env: dict[str, str] | None = None,
) -> list[str]:
    _log(f"command={' '.join(map(str, command))}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
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


def _sync_base(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-u",
        "scripts/sync_frame_video_with_rclone.py",
        "--rclone-bin",
        args.rclone_bin,
        "--config",
        str(args.rclone_config),
        "--remote",
        args.rclone_remote,
        "--root-folder-id",
        args.drive_root_folder_id,
        "--remote-root-name",
        args.remote_root_name,
        "--worker-id",
        args.worker_id,
    ]


def _scan(args: argparse.Namespace, video_ids: list[str]) -> dict[str, Any]:
    command = _sync_base(args)
    command.insert(3, "scan")
    command.extend(
        ["--identity-dir", str(args.identity_dir), "--attempts", str(args.scan_attempts)]
    )
    for video_id in video_ids:
        command.extend(["--video-id", video_id])
    return _json_report(_run_streamed(command, prefix="[inventory] "))


def _publish(args: argparse.Namespace, video_id: str) -> dict[str, Any]:
    command = _sync_base(args)
    command.insert(3, "publish")
    command.extend(
        [
            "--video-id",
            video_id,
            "--identity-file",
            str(args.identity_dir / f"{video_id}.json"),
            "--package-root",
            str(args.package_root),
            "--attempts",
            str(args.upload_attempts),
        ]
    )
    return _json_report(_run_streamed(command, prefix=f"[{video_id}:upload] "))


def _shot_command(
    args: argparse.Namespace, video_id: str, video_path: Path
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "scripts/run_transnetv2_shots.py",
        "--config",
        str(args.config),
        "--backend",
        "pytorch",
        "--batch-size",
        str(args.batch_size),
        "--video-id",
        video_id,
        "--output-root",
        str(args.output_root),
        "--resume",
        "--video-path",
        str(video_path),
        "--entrypoint",
        str(args.entrypoint),
        "--weights",
        str(args.weights),
    ]


def _monitor(processes: dict[str, subprocess.Popen[str]], stop: threading.Event) -> None:
    while not stop.wait(10):
        active = [video_id for video_id, process in processes.items() if process.poll() is None]
        report = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in report.stdout.splitlines():
            _log(f"phase=gpu_monitor active={','.join(active) or 'none'} values={line}")


def _run_pair(
    args: argparse.Namespace,
    pair: list[str],
    video_paths: dict[str, Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    processes: dict[str, subprocess.Popen[str]] = {}
    lines: dict[str, list[str]] = {video_id: [] for video_id in pair}
    started: dict[str, float] = {}
    completed: dict[str, float] = {}
    for gpu_index, video_id in enumerate(pair):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        command = _shot_command(args, video_id, video_paths[video_id])
        _log(f"phase=gpu_pair video_id={video_id} physical_gpu={gpu_index} result=starting")
        started[video_id] = time.monotonic()
        processes[video_id] = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )

    def forward(video_id: str) -> None:
        process = processes[video_id]
        assert process.stdout is not None
        for line in process.stdout:
            lines[video_id].append(line.rstrip())
            print(f"[{video_id}:gpu] {line}", end="", flush=True)
        process.wait()
        completed[video_id] = time.monotonic()

    threads = [threading.Thread(target=forward, args=(video_id,), daemon=True) for video_id in pair]
    for thread in threads:
        thread.start()
    stop = threading.Event()
    monitor = threading.Thread(target=_monitor, args=(processes, stop), daemon=True)
    monitor.start()
    for thread in threads:
        thread.join()
    stop.set()
    monitor.join(timeout=2)

    reports: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for video_id in pair:
        process = processes[video_id]
        if process.returncode != 0:
            failures[video_id] = "\n".join(lines[video_id][-80:])
            _log(
                f"phase=shot_detection video_id={video_id} "
                f"result=failed code={process.returncode}"
            )
            continue
        report = _json_report(lines[video_id])
        report["worker_wall_s"] = completed[video_id] - started[video_id]
        reports[video_id] = report
        _log(f"phase=shot_detection video_id={video_id} result=success")
    return reports, failures


def _build_video(
    args: argparse.Namespace, video_id: str, video_path: Path
) -> dict[str, Any]:
    shots = args.output_root / "shot_detection" / f"{video_id}.jsonl"
    candidates = (
        args.output_root / "frame_extraction" / "adaptive_candidates" / f"{video_id}.jsonl"
    )
    commands = [
        [
            sys.executable,
            "-u",
            "scripts/build_adaptive_frame_candidates.py",
            "--config",
            str(args.config),
            "--video-id",
            video_id,
            "--shots",
            str(shots),
            "--output-root",
            str(args.output_root),
        ],
        [
            sys.executable,
            "-u",
            "scripts/extract_adaptive_frames.py",
            "--video-id",
            video_id,
            "--video-path",
            str(video_path),
            "--candidates",
            str(candidates),
            "--output-root",
            str(args.output_root),
        ],
        [
            sys.executable,
            "-u",
            "scripts/build_transnet_keyframe_package.py",
            "--video-id",
            video_id,
            "--output-root",
            str(args.output_root),
            "--package-root",
            str(args.package_root),
        ],
    ]
    reports = [
        _json_report(_run_streamed(command, prefix=f"[{video_id}:build] "))
        for command in commands
    ]
    upload = _publish(args, video_id)
    return {
        "candidates": reports[0],
        "extraction": reports[1],
        "package": reports[2],
        "upload": upload,
    }


def _safe_remove(path: Path, root: Path) -> None:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    if resolved.is_dir():
        shutil.rmtree(resolved)
    elif resolved.exists():
        resolved.unlink()


def _cleanup(args: argparse.Namespace, video_id: str) -> None:
    if args.keep_local:
        return
    batch = video_id.split("_", maxsplit=1)[0]
    paths = [
        args.output_root / "adaptive_keyframes" / video_id,
        args.output_root / "shot_detection" / "transnetv2_work" / video_id,
        args.package_root / f"Keyframes_{batch}" / "keyframes" / video_id,
        args.package_root / "map-keyframes" / f"{video_id}.csv",
        args.package_root / "manifests" / f"{video_id}.jsonl",
    ]
    for path in paths:
        _safe_remove(path, args.output_root)
    _log(f"phase=cleanup video_id={video_id} result=success")


def _copy_benchmark(args: argparse.Namespace, benchmark_path: Path) -> None:
    remote_path = (
        f"{args.rclone_remote}:{args.remote_root_name}/benchmark/{args.worker_id}.json"
    )
    command = [
        args.rclone_bin,
        "copyto",
        str(benchmark_path),
        remote_path,
        "--config",
        str(args.rclone_config),
        "--drive-root-folder-id",
        args.drive_root_folder_id,
        "--checksum",
        "--retries",
        str(args.upload_attempts),
        "--low-level-retries",
        "10",
        "-v",
    ]
    _run_streamed(command, prefix="[benchmark:upload] ")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    assignment = _load_json(args.assignment)
    video_ids = assignment.get("video_ids") if isinstance(assignment, dict) else None
    if not isinstance(video_ids, list) or not video_ids or len(video_ids) != len(set(video_ids)):
        raise ValueError("Assignment must contain unique video_ids")
    video_index = _load_json(args.video_index)
    video_paths = {video_id: Path(video_index[video_id]) for video_id in video_ids}
    for video_id, path in video_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Assigned video is missing: {video_id} -> {path}")
    for path in (args.config, args.entrypoint, args.weights, args.rclone_config):
        if not path.is_file():
            raise FileNotFoundError(path)

    initial = _scan(args, video_ids)
    pending = list(initial["pending"])
    benchmark_path = args.package_root / "benchmark" / f"{args.worker_id}.json"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "worker_id": args.worker_id,
        "assignment_count": len(video_ids),
        "batch_size": args.batch_size,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "initial_completed": initial["completed"],
        "processed": {},
        "failures": {},
        "sampling_policy": {
            "duration_lt_2s": "midpoint",
            "duration_2_to_lt_4s": "quarter_and_three_quarter",
            "duration_gte_4s": "1.5s_interval_centers",
            "duration_gte_7s": "cap_10_evenly_spaced",
        },
    }
    _log(
        f"phase=queue assignment={len(video_ids)} completed={len(initial['completed'])} "
        f"pending={len(pending)}"
    )
    index = 0
    while index < len(pending):
        elapsed = time.time() - args.session_start_epoch
        if elapsed >= args.accept_new_work_seconds:
            _log(
                f"phase=deadline result=stop_accepting elapsed_s={elapsed:.1f} "
                f"threshold_s={args.accept_new_work_seconds:.1f}"
            )
            break
        pair = pending[index : index + 2]
        index += len(pair)
        _log(f"phase=gpu_pair result=accepted videos={','.join(pair)} elapsed_s={elapsed:.1f}")
        shot_reports, shot_failures = _run_pair(args, pair, video_paths)
        summary["failures"].update(shot_failures)
        for video_id in pair:
            if video_id not in shot_reports:
                continue
            try:
                result = _build_video(args, video_id, video_paths[video_id])
                metrics_path = (
                    args.output_root
                    / "shot_detection"
                    / "transnetv2_work"
                    / video_id
                    / "transnetv2_metrics.json"
                )
                result["shot_detection"] = shot_reports[video_id]
                if metrics_path.is_file():
                    result["metrics"] = _load_json(metrics_path)
                summary["processed"][video_id] = result
                _log(f"phase=checkpoint video_id={video_id} result=completed")
                _cleanup(args, video_id)
            except BaseException as error:
                summary["failures"][video_id] = f"{type(error).__name__}: {error}"
                _log(
                    f"phase=checkpoint video_id={video_id} result=failed "
                    f"error={type(error).__name__}"
                )
        summary["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        benchmark_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        _copy_benchmark(args, benchmark_path)

    final = _scan(args, video_ids)
    summary.update(
        {
            "status": (
                "completed"
                if not final["pending"]
                else "completed_with_failures"
                if summary["failures"]
                else "checkpointed"
            ),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "session_elapsed_s": time.time() - args.session_start_epoch,
            "remote_completed": final["completed"],
            "remaining": final["pending"],
        }
    )
    benchmark_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _copy_benchmark(args, benchmark_path)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 1 if summary["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
