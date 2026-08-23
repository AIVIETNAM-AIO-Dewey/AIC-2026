#!/usr/bin/env python3
"""Preflight, inspect, and publish one frame-extraction video with rclone."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar


T = TypeVar("T")
IMAGE_PATTERN = re.compile(r"[0-9]{6}\.jpg")


def _log(message: str) -> None:
    print(f"[rclone_checkpoint] {message}", file=sys.stderr, flush=True)


def _display(command: list[str]) -> str:
    redacted: list[str] = []
    hide_next = False
    for value in command:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
        elif value == "--config":
            redacted.append(value)
            hide_next = True
        else:
            redacted.append(value)
    return shlex.join(redacted)


def _run(command: list[str], *, capture: bool = True) -> str:
    _log(f"command={_display(command)}")
    completed = subprocess.run(command, capture_output=capture, text=True, check=False)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr, flush=True)
    if completed.returncode != 0:
        stdout = completed.stdout[-2000:] if completed.stdout else ""
        raise RuntimeError(
            f"Command failed with code {completed.returncode}: {_display(command)}\n{stdout}"
        )
    return completed.stdout if capture else ""


def _retry(label: str, attempts: int, callback: Callable[[], T]) -> T:
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        _log(f"operation={label} attempt={attempt}/{attempts} result=attempting")
        try:
            result = callback()
            _log(f"operation={label} attempt={attempt}/{attempts} result=success")
            return result
        except BaseException as error:
            last_error = error
            _log(
                f"operation={label} attempt={attempt}/{attempts} result=failed "
                f"error={type(error).__name__} detail={str(error)[:500]!r}"
            )
            if attempt < attempts:
                delay = min(30, 2 ** (attempt - 1))
                _log(f"operation={label} retry_in_s={delay}")
                time.sleep(delay)
    assert last_error is not None
    raise RuntimeError(f"{label} failed after {attempts} attempts") from last_error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "scan", "publish"))
    parser.add_argument("--rclone-bin", default="rclone")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remote", default="gdrive")
    parser.add_argument("--root-folder-id", required=True)
    parser.add_argument("--remote-root-name", default="self-cut-btc-compatible")
    parser.add_argument("--worker-id")
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--identity-dir", type=Path)
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--attempts", type=int)
    return parser


def _validate(args: argparse.Namespace) -> None:
    if not args.config.is_file():
        raise FileNotFoundError(f"rclone config does not exist: {args.config}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.remote):
        raise ValueError("--remote contains unsupported characters")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.root_folder_id):
        raise ValueError("--root-folder-id must be an ID, not a URL")
    if not re.fullmatch(r"[^/\\]+", args.remote_root_name):
        raise ValueError("--remote-root-name must be one path component")
    if args.attempts is not None and args.attempts < 1:
        raise ValueError("--attempts must be positive")


def _rclone(args: argparse.Namespace, command: str, *values: str) -> list[str]:
    return [
        str(args.rclone_bin),
        command,
        *values,
        "--config",
        str(args.config),
        "--drive-root-folder-id",
        args.root_folder_id,
    ]


def _remote(args: argparse.Namespace, *parts: str) -> str:
    suffix = "/".join(part.strip("/") for part in parts if part)
    path = args.remote_root_name if not suffix else f"{args.remote_root_name}/{suffix}"
    return f"{args.remote}:{path}"


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_set_digest(rows: list[tuple[str, int, str]]) -> str:
    digest = hashlib.sha256()
    for name, size, checksum in sorted(rows):
        digest.update(f"{name}\t{size}\t{checksum.lower()}\n".encode("utf-8"))
    return digest.hexdigest()


def _expected_names(count: int) -> list[str]:
    return [f"{index:06d}.jpg" for index in range(1, count + 1)]


def _local_descriptor(package_root: Path, video_id: str) -> dict[str, Any]:
    batch = video_id.split("_", maxsplit=1)[0]
    frames_dir = package_root / f"Keyframes_{batch}" / "keyframes" / video_id
    map_path = package_root / "map-keyframes" / f"{video_id}.csv"
    manifest_path = package_root / "manifests" / f"{video_id}.jsonl"
    for path in (frames_dir, map_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Package artifact is missing: {path}")
    images = sorted(frames_dir.glob("*.jpg"))
    names = [path.name for path in images]
    if not images or names != _expected_names(len(images)):
        raise ValueError(f"Image numbering is not contiguous for {video_id}")
    rows = [(path.name, path.stat().st_size, _md5(path)) for path in images]
    return {
        "video_id": video_id,
        "image_count": len(images),
        "image_set_sha256": _image_set_digest(rows),
        "map_md5": _md5(map_path),
        "manifest_md5": _md5(manifest_path),
        "frames_dir": frames_dir,
        "map_path": map_path,
        "manifest_path": manifest_path,
    }


def _remote_md5(args: argparse.Namespace, path: str) -> str:
    output = _run(_rclone(args, "md5sum", path))
    line = next((line for line in output.splitlines() if line.strip()), "")
    checksum = line.split(maxsplit=1)[0].lower() if line else ""
    if not re.fullmatch(r"[0-9a-f]{32}", checksum):
        raise ValueError(f"No valid remote MD5 for {path}")
    return checksum


def _remote_descriptor(
    args: argparse.Namespace, video_id: str, expected_count: int
) -> dict[str, Any]:
    batch = video_id.split("_", maxsplit=1)[0]
    frames_path = _remote(args, f"Keyframes_{batch}", "keyframes", video_id)
    output = _run(
        _rclone(args, "lsjson", frames_path)
        + ["--files-only", "--recursive", "--hash"]
    )
    records = json.loads(output)
    rows: list[tuple[str, int, str]] = []
    for record in records:
        name = str(record.get("Path") or record.get("Name") or "")
        hashes = record.get("Hashes") or {}
        checksum = str(hashes.get("MD5") or hashes.get("md5") or "").lower()
        if not IMAGE_PATTERN.fullmatch(name) or not re.fullmatch(r"[0-9a-f]{32}", checksum):
            raise ValueError(f"Unexpected frame record on Drive: {record}")
        rows.append((name, int(record["Size"]), checksum))
    names = sorted(name for name, _, _ in rows)
    if names != _expected_names(expected_count):
        raise ValueError(
            f"Remote image set mismatch for {video_id}: "
            f"expected={expected_count} actual={len(rows)}"
        )
    return {
        "image_count": len(rows),
        "image_set_sha256": _image_set_digest(rows),
        "map_md5": _remote_md5(args, _remote(args, "map-keyframes", f"{video_id}.csv")),
        "manifest_md5": _remote_md5(
            args, _remote(args, "manifests", f"{video_id}.jsonl")
        ),
    }


def _read_identity(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Identity must be a JSON object: {path}")
    return value


def _marker_path(args: argparse.Namespace, video_id: str) -> str:
    return _remote(args, "_completed", f"{video_id}.json")


def _read_marker(args: argparse.Namespace, video_id: str) -> dict[str, Any]:
    value = json.loads(_run(_rclone(args, "cat", _marker_path(args, video_id))))
    if not isinstance(value, dict):
        raise ValueError(f"Completion marker is not an object for {video_id}")
    return value


def _verify_marker(
    args: argparse.Namespace,
    video_id: str,
    identity: dict[str, Any],
    *,
    attempts: int,
) -> dict[str, Any]:
    def verify() -> dict[str, Any]:
        marker = _read_marker(args, video_id)
        if marker.get("status") != "completed" or marker.get("video_id") != video_id:
            raise ValueError("Marker status or video_id is invalid")
        if marker.get("identity") != identity:
            raise ValueError("Marker identity does not match this worker revision")
        expected_count = int(marker.get("image_count", 0))
        if expected_count < 1:
            raise ValueError("Marker has no images")
        remote = _remote_descriptor(args, video_id, expected_count)
        for key in ("image_count", "image_set_sha256", "map_md5", "manifest_md5"):
            if marker.get(key) != remote.get(key):
                raise ValueError(f"Marker checksum mismatch: {key}")
        return marker

    return _retry(f"verify_{video_id}", attempts, verify)


def _folder_report(args: argparse.Namespace) -> dict[str, Any]:
    records = json.loads(
        _run(_rclone(args, "lsjson", f"{args.remote}:") + ["--dirs-only", "--max-depth", "1"])
    )
    for record in records:
        if record.get("Name") == args.remote_root_name and record.get("IsDir"):
            folder_id = record.get("ID")
            return {
                "folder_id": folder_id,
                "folder_url": (
                    f"https://drive.google.com/drive/folders/{folder_id}" if folder_id else None
                ),
            }
    raise RuntimeError(f"Drive root is not visible: {args.remote_root_name}")


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    attempts = args.attempts or 5
    _log("phase=version result=attempting")
    version = _run([str(args.rclone_bin), "version"]).splitlines()[0]
    _log(f"phase=version result=success version={version}")
    remotes = {
        line.strip().removesuffix(":")
        for line in _run(
            [str(args.rclone_bin), "listremotes", "--config", str(args.config)]
        ).splitlines()
        if line.strip()
    }
    if args.remote not in remotes:
        raise ValueError(f"rclone config has no [{args.remote}] remote")
    _log(f"phase=config result=success remote={args.remote} values=<redacted>")

    def probe() -> dict[str, Any]:
        probe_name = f"probe-{int(time.time() * 1000)}.json"
        remote_probe = _remote(args, "_preflight", probe_name)
        payload = json.dumps({"probe": probe_name, "worker_id": args.worker_id}).encode()
        with tempfile.TemporaryDirectory(prefix="aic-rclone-probe-") as directory:
            local = Path(directory) / probe_name
            local.write_bytes(payload)
            _run(_rclone(args, "mkdir", _remote(args)))
            _run(_rclone(args, "mkdir", _remote(args, "_completed")))
            _run(_rclone(args, "copyto", str(local), remote_probe) + ["--checksum"])
            downloaded = _run(_rclone(args, "cat", remote_probe)).encode()
            if downloaded != payload:
                raise ValueError("Drive read-back differs from uploaded preflight payload")
            remote_checksum = _remote_md5(args, remote_probe)
            if remote_checksum != _md5(local):
                raise ValueError("Drive preflight checksum mismatch")
            _run(_rclone(args, "deletefile", remote_probe))
        return _folder_report(args)

    report = _retry("drive_write_read_checksum_delete_probe", attempts, probe)
    return {"status": "ready", **report}


def _scan(args: argparse.Namespace) -> dict[str, Any]:
    if not args.video_id or args.identity_dir is None:
        raise ValueError("scan requires --video-id and --identity-dir")
    attempts = args.attempts or 5
    def list_markers() -> set[str]:
        output = _run(
            _rclone(args, "lsf", _remote(args, "_completed"))
            + ["--files-only", "--max-depth", "1"]
        )
        return {line.strip() for line in output.splitlines() if line.strip()}

    marker_names = _retry("list_completion_markers", attempts, list_markers)
    results: list[dict[str, Any]] = []
    for video_id in args.video_id:
        identity = _read_identity(args.identity_dir / f"{video_id}.json")
        if f"{video_id}.json" not in marker_names:
            results.append(
                {"video_id": video_id, "status": "pending", "reason": "marker_missing"}
            )
            _log(f"phase=inventory video_id={video_id} result=pending reason=marker_missing")
            continue
        try:
            marker = _verify_marker(args, video_id, identity, attempts=attempts)
            results.append(
                {"video_id": video_id, "status": "completed", "image_count": marker["image_count"]}
            )
            _log(f"phase=inventory video_id={video_id} result=completed")
        except BaseException as error:
            results.append(
                {
                    "video_id": video_id,
                    "status": "pending",
                    "reason": type(error).__name__,
                    "detail": str(error)[:500],
                }
            )
            _log(
                f"phase=inventory video_id={video_id} result=pending "
                f"reason={type(error).__name__}"
            )
    completed = [item["video_id"] for item in results if item["status"] == "completed"]
    pending = [item["video_id"] for item in results if item["status"] != "completed"]
    return {"status": "completed", "completed": completed, "pending": pending, "results": results}


def _delete_marker_best_effort(args: argparse.Namespace, video_id: str) -> None:
    command = _rclone(args, "deletefile", _marker_path(args, video_id))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    _log(
        f"phase=marker_remove video_id={video_id} result="
        f"{'success' if completed.returncode == 0 else 'not_present'}"
    )


def _publish(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.video_id) != 1 or args.identity_file is None or args.package_root is None:
        raise ValueError("publish requires one --video-id, --identity-file, and --package-root")
    if not args.worker_id:
        raise ValueError("publish requires --worker-id")
    video_id = args.video_id[0]
    package_root = args.package_root.resolve()
    identity = _read_identity(args.identity_file)
    local = _local_descriptor(package_root, video_id)
    attempts = args.attempts or 10
    batch = video_id.split("_", maxsplit=1)[0]

    def upload() -> dict[str, Any]:
        _delete_marker_best_effort(args, video_id)
        common = [
            "--checksum",
            "--retries",
            "1",
            "--low-level-retries",
            "10",
            "--stats",
            "10s",
            "--stats-one-line",
            "-v",
        ]
        _run(
            _rclone(
                args,
                "sync",
                str(local["frames_dir"]),
                _remote(args, f"Keyframes_{batch}", "keyframes", video_id),
            )
            + common,
            capture=False,
        )
        _run(
            _rclone(
                args,
                "copyto",
                str(local["map_path"]),
                _remote(args, "map-keyframes", f"{video_id}.csv"),
            )
            + common,
            capture=False,
        )
        _run(
            _rclone(
                args,
                "copyto",
                str(local["manifest_path"]),
                _remote(args, "manifests", f"{video_id}.jsonl"),
            )
            + common,
            capture=False,
        )
        remote = _remote_descriptor(args, video_id, local["image_count"])
        for key in ("image_count", "image_set_sha256", "map_md5", "manifest_md5"):
            if local[key] != remote[key]:
                raise ValueError(f"Remote verification failed for {key}")
        marker = {
            "schema_version": "1.0",
            "status": "completed",
            "video_id": video_id,
            "worker_id": args.worker_id,
            "identity": identity,
            "image_count": local["image_count"],
            "image_set_sha256": local["image_set_sha256"],
            "map_md5": local["map_md5"],
            "manifest_md5": local["manifest_md5"],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        with tempfile.TemporaryDirectory(prefix="aic-marker-") as directory:
            marker_file = Path(directory) / f"{video_id}.json"
            marker_file.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
            _run(
                _rclone(args, "copyto", str(marker_file), _marker_path(args, video_id))
                + ["--checksum"]
            )
        verified = _verify_marker(args, video_id, identity, attempts=1)
        return verified

    marker = _retry(f"publish_{video_id}", attempts, upload)
    return {
        "status": "completed",
        "video_id": video_id,
        "image_count": marker["image_count"],
        "remote_marker": _marker_path(args, video_id),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate(args)
    if args.action == "preflight":
        report = _preflight(args)
    elif args.action == "scan":
        report = _scan(args)
    else:
        report = _publish(args)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
