#!/usr/bin/env python3
"""Copy an organizer-compatible keyframe package to Google Drive with rclone."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rclone-bin", default="rclone")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--remote", default="gdrive")
    parser.add_argument("--root-folder-id", required=True)
    parser.add_argument("--remote-root-name", default="transnetv2-only")
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _log(message: str) -> None:
    print(f"[rclone_export] {message}", file=sys.stderr, flush=True)


def _display_command(command: list[str]) -> str:
    return shlex.join(command)


def _run_streamed(command: list[str]) -> None:
    print(f"$ {_display_command(command)}", file=sys.stderr, flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", file=sys.stderr, flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"rclone command failed with code {return_code}: {_display_command(command)}"
        )


def _run_captured(command: list[str]) -> str:
    print(f"$ {_display_command(command)}", file=sys.stderr, flush=True)
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if completed.stderr.strip():
        print(completed.stderr, end="", file=sys.stderr, flush=True)
    return completed.stdout


def _rclone_command(
    args: argparse.Namespace,
    command: str,
    *values: str,
) -> list[str]:
    return [
        str(args.rclone_bin),
        command,
        *values,
        "--config",
        str(args.config),
        "--drive-root-folder-id",
        args.root_folder_id,
    ]


def _validate_args(args: argparse.Namespace) -> None:
    if not args.config.is_file():
        raise FileNotFoundError(f"rclone config does not exist: {args.config}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.remote):
        raise ValueError("--remote may contain only letters, digits, underscore, and hyphen")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.root_folder_id):
        raise ValueError("--root-folder-id must be a Drive folder ID, not a URL")
    if not re.fullmatch(r"[^/\\]+", args.remote_root_name):
        raise ValueError("--remote-root-name must be a single path component")


def _remote_folder(args: argparse.Namespace) -> str:
    return f"{args.remote}:{args.remote_root_name}"


def _folder_report(args: argparse.Namespace) -> dict[str, Any]:
    output = _run_captured(
        _rclone_command(args, "lsjson", f"{args.remote}:")
        + ["--dirs-only", "--max-depth", "1"]
    )
    records = json.loads(output)
    for record in records:
        if record.get("Name") == args.remote_root_name and record.get("IsDir"):
            folder_id = record.get("ID")
            return {
                "folder_id": folder_id,
                "folder_url": (
                    f"https://drive.google.com/drive/folders/{folder_id}"
                    if folder_id
                    else None
                ),
            }
    raise RuntimeError(f"rclone created no visible folder named {args.remote_root_name}")


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    _log("phase=version result=attempting")
    version = _run_captured([str(args.rclone_bin), "version"]).splitlines()[0]
    _log(f"phase=version result=success version={version}")

    _log("phase=config result=attempting values=<redacted>")
    remotes = {
        value.strip().removesuffix(":")
        for value in _run_captured(
            [str(args.rclone_bin), "listremotes", "--config", str(args.config)]
        ).splitlines()
        if value.strip()
    }
    if args.remote not in remotes:
        raise ValueError(
            f"rclone config has no [{args.remote}] remote; available={sorted(remotes)}"
        )
    _log(f"phase=config result=success remote={args.remote} values=<redacted>")

    _log("phase=drive_connection result=attempting retries=3")
    _run_streamed(
        _rclone_command(args, "mkdir", _remote_folder(args))
        + [
            "--retries",
            "3",
            "--low-level-retries",
            "10",
            "--retries-sleep",
            "2s",
            "-v",
        ]
    )
    report = _folder_report(args)
    _log("phase=drive_connection result=success writable=true")
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    folder_report = _preflight(args)
    if args.preflight_only:
        print(json.dumps({"status": "ready", **folder_report}))
        return 0

    if args.package_root is None:
        raise ValueError("--package-root is required unless --preflight-only is used")
    package_root = args.package_root.expanduser().resolve()
    if not package_root.is_dir():
        raise FileNotFoundError(f"Package root does not exist: {package_root}")
    local_files = sorted(path for path in package_root.rglob("*") if path.is_file())
    if not local_files:
        raise ValueError(f"Package root has no files: {package_root}")

    _log(
        f"phase=copy result=attempting files={len(local_files)} "
        f"destination={_remote_folder(args)}"
    )
    _run_streamed(
        _rclone_command(
            args,
            "copy",
            str(package_root),
            _remote_folder(args),
        )
        + [
            "--checksum",
            "--check-first",
            "--retries",
            "3",
            "--low-level-retries",
            "10",
            "--retries-sleep",
            "2s",
            "--transfers",
            "4",
            "--checkers",
            "8",
            "--stats",
            "10s",
            "--stats-one-line",
            "--stats-one-line-date",
            "--stats-log-level",
            "NOTICE",
            "-v",
        ]
    )
    _log(f"phase=copy result=success files={len(local_files)}")
    print(
        json.dumps(
            {
                "status": "completed",
                "files": len(local_files),
                "remote_path": _remote_folder(args),
                **folder_report,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
