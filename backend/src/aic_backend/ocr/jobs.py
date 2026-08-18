"""One-at-a-time local OCR jobs selected from trusted artifact manifests."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from ..settings import Settings

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as stream:
        return sum(bool(line.strip()) for line in stream)


class OcrJobManager:
    """Launch the pinned CLI only; browser input never becomes a command or path."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._process: subprocess.Popen[str] | None = None
        self._log_stream: IO[str] | None = None
        self._active_id: str | None = None
        self._started_at: str | None = None
        self._last_exit_code: int | None = None

    @property
    def manifest_root(self) -> Path:
        return (self.settings.artifact_root / "frame_manifests").resolve()

    @property
    def output_root(self) -> Path:
        return (self.settings.artifact_root / "ocr").resolve()

    def _safe_path(self, root: Path, manifest_id: str, suffix: str) -> Path:
        if not SAFE_ID.fullmatch(manifest_id):
            raise ValueError("invalid_manifest_id")
        path = (root / f"{manifest_id}{suffix}").resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("invalid_manifest_id") from error
        return path

    def _refresh(self) -> None:
        if self._process is None:
            return
        exit_code = self._process.poll()
        if exit_code is None:
            return
        self._last_exit_code = exit_code
        self._process = None
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None

    def _dataset(self, manifest: Path) -> dict[str, Any]:
        manifest_id = manifest.stem
        output = self._safe_path(self.output_root, manifest_id, ".jsonl")
        partial = output.with_suffix(output.suffix + ".partial")
        run_manifest = output.with_suffix(".manifest.json")
        total = _line_count(manifest)
        processed = _line_count(output if output.is_file() else partial)
        status = "not_started"
        if output.is_file():
            status = "completed"
        elif partial.is_file():
            status = "interrupted"
        if self._active_id == manifest_id and self._process is not None:
            status = "running"
        if (
            self._active_id == manifest_id
            and self._process is None
            and self._last_exit_code not in (None, 0)
        ):
            status = "failed"
        counters: dict[str, int] = {}
        if run_manifest.is_file():
            try:
                payload = json.loads(run_manifest.read_text(encoding="utf-8"))
                counters = {key: int(value) for key, value in payload.get("counters", {}).items()}
            except (OSError, ValueError, TypeError):
                counters = {}
        return {
            "manifest_id": manifest_id,
            "status": status,
            "total_frames": total,
            "processed_frames": processed,
            "remaining_frames": max(0, total - processed),
            "counters": counters,
            "output_exists": output.is_file(),
        }

    def status(self) -> dict[str, Any]:
        self._refresh()
        manifests = (
            sorted(self.manifest_root.glob("*.jsonl")) if self.manifest_root.is_dir() else []
        )
        return {
            "enabled": self.settings.ocr_jobs_enabled,
            "model_id": "ppocrv6-small",
            "active_manifest_id": self._active_id if self._process is not None else None,
            "started_at": self._started_at if self._process is not None else None,
            "last_exit_code": self._last_exit_code,
            "datasets": [self._dataset(path) for path in manifests],
        }

    def completed_artifact(self, manifest_id: str) -> tuple[Path, Path]:
        output = self._safe_path(self.output_root, manifest_id, ".jsonl")
        manifest = output.with_suffix(".manifest.json")
        if not output.is_file() or not manifest.is_file():
            raise FileNotFoundError("ocr_output_not_completed")
        return output, manifest

    def start(self, manifest_id: str) -> dict[str, Any]:
        self._refresh()
        if not self.settings.ocr_jobs_enabled:
            raise PermissionError("ocr_jobs_disabled")
        if self._process is not None:
            raise RuntimeError("ocr_job_already_running")

        manifest = self._safe_path(self.manifest_root, manifest_id, ".jsonl")
        if not manifest.is_file():
            raise FileNotFoundError("ocr_manifest_not_found")
        output = self._safe_path(self.output_root, manifest_id, ".jsonl")
        if output.is_file():
            self._active_id = manifest_id
            self._last_exit_code = 0
            return self.status()

        repo_root = Path(__file__).resolve().parents[4]
        script = repo_root / "offline" / "scripts" / "run_ppocrv6.py"
        config = self.settings.ocr_config_path.expanduser().resolve()
        data_root = self.settings.ocr_data_root.expanduser().resolve()
        cache_root = self.settings.ocr_cache_root.expanduser().resolve()
        for required, code in (
            (script, "ocr_runner_not_found"),
            (config, "ocr_config_not_found"),
            (data_root, "ocr_data_root_not_found"),
            (cache_root, "ocr_cache_root_not_found"),
        ):
            if not required.exists():
                raise FileNotFoundError(code)

        self.output_root.mkdir(parents=True, exist_ok=True)
        logs = self.output_root / "jobs"
        logs.mkdir(parents=True, exist_ok=True)
        log_path = self._safe_path(logs.resolve(), manifest_id, ".log")
        self._log_stream = log_path.open("a", encoding="utf-8")
        command = [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--data-root",
            str(data_root),
            "--output-root",
            str(self.settings.artifact_root.expanduser().resolve()),
            "--cache-root",
            str(cache_root),
            "--video-id",
            manifest_id,
            "--frame-manifest",
            str(manifest),
            "--output",
            str(output),
            "--resume",
        ]
        self._process = subprocess.Popen(  # noqa: S603 - fixed executable and trusted paths only
            command,
            cwd=repo_root,
            stdout=self._log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._active_id = manifest_id
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._last_exit_code = None
        return self.status()
