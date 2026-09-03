"""Short-lived CPU encoder workers with deterministic RAM reclamation."""

from __future__ import annotations

import io
import math
import multiprocessing as mp
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .cpu import CpuTextEncoders
from .sequential_manager import SequentialBranch1Encoders
from ..infrastructure.resources import (
    MAX_PRODUCTION_RSS_BYTES,
    current_process_rss_bytes,
    peak_process_rss_bytes,
)

MAX_IDLE_TIMEOUT_SECONDS = 30.0


def _worker_identity(request: dict[str, Any]) -> tuple[str, ...]:
    """Return the immutable model identity allowed to stay resident."""
    kind = str(request["kind"])
    # The identity is deliberately stricter than the model name.  A worker
    # may be reused only when the checkpoint/revision and the preprocessing
    # contract are identical; otherwise a cached model could silently serve
    # vectors from a different tokenizer or projection head.
    tokenizer_config = str(request.get("tokenizer_config") or "default")
    if kind == "branch1_text":
        return (
            kind,
            str(request["model_name"]),
            str(request["model_root"]),
            str(request.get("model_revision") or "unknown-revision"),
            tokenizer_config,
        )
    if kind == "bge_text":
        return (
            "bge_text",
            str(request["bge_id"]),
            str(request.get("bge_revision") or "local-cache"),
            tokenizer_config,
        )
    if kind in {"siglip_text", "siglip_image"}:
        # Text and image requests intentionally share one identity because
        # both use the same SigLIP checkpoint and projection.  Callers pass
        # the same tokenizer/processor contract for the two modalities.
        return (
            "siglip",
            str(request["siglip_id"]),
            str(request["siglip_revision"]),
            tokenizer_config,
        )
    raise ValueError(f"Unknown encoder worker request: {kind}")


def _load_worker_encoder(request: dict[str, Any]) -> tuple[Any, float]:
    load_started = time.perf_counter()
    kind = str(request["kind"])
    if kind == "branch1_text":
        encoder = SequentialBranch1Encoders(Path(request["model_root"]))
        encoder._load(str(request["model_name"]))
    else:
        encoder = CpuTextEncoders(
            siglip_id=str(request["siglip_id"]),
            siglip_revision=str(request["siglip_revision"]),
            bge_id=str(request["bge_id"]),
            bge_revision=(
                str(request["bge_revision"])
                if request.get("bge_revision")
                else None
            ),
        )
        if kind == "bge_text":
            encoder._load_bge()
        elif kind in {"siglip_text", "siglip_image"}:
            encoder._load_siglip()
    return encoder, (time.perf_counter() - load_started) * 1000.0


def _run_worker_inference(encoder: Any, request: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    kind = str(request["kind"])
    if kind == "branch1_text":
        return encoder.encode(str(request["model_name"]), list(request["texts"]))
    if kind == "bge_text":
        return encoder.encode_bge_text(list(request["texts"]))
    if kind == "siglip_text":
        text = str(request["text"])
        return encoder.embed_siglip_text(text), [encoder.siglip_text_diagnostics(text)]
    if kind == "siglip_image":
        with Image.open(io.BytesIO(request["image_bytes"])) as source:
            return encoder.embed_siglip_image(source.convert("RGB")), []
    raise ValueError(f"Unknown encoder worker request: {kind}")


def _worker_main(connection: Any, request: dict[str, Any], idle_timeout_seconds: float) -> None:
    """Serve one model identity and exit after a bounded idle period."""
    encoder: Any | None = None
    try:
        identity = _worker_identity(request)
        encoder, first_load_ms = _load_worker_encoder(request)
        pending = request
        load_ms = first_load_ms
        while True:
            inference_started = time.perf_counter()
            vectors, diagnostics = _run_worker_inference(encoder, pending)
            connection.send(
                {
                    "ok": True,
                    "vectors": np.asarray(vectors, dtype=np.float32),
                    "diagnostics": diagnostics,
                    "timing": {
                        "model_loading_ms": round(load_ms, 2),
                        "inference_ms": round(
                            (time.perf_counter() - inference_started) * 1000.0, 2
                        ),
                        "worker_reused": load_ms == 0.0,
                        "worker_pid": os.getpid(),
                        "worker_load_count": 1,
                    },
                    "peak_rss_bytes": peak_process_rss_bytes(),
                }
            )
            load_ms = 0.0
            if not connection.poll(idle_timeout_seconds):
                break
            pending = connection.recv()
            if _worker_identity(pending) != identity:
                raise RuntimeError("Encoder worker received a different model identity")
    except BaseException as error:
        try:
            connection.send(
                {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "peak_rss_bytes": peak_process_rss_bytes(),
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if encoder is not None:
            try:
                if hasattr(encoder, "unload"):
                    encoder.unload()
                elif hasattr(encoder, "unload_all"):
                    encoder.unload_all()
            except (AttributeError, RuntimeError):
                pass
        connection.close()


class EncoderWorkerManager:
    """Serialize heavy inference and reclaim model memory by process exit."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 600.0,
        idle_timeout_seconds: float = 30.0,
        worker_target: Callable[[Any, dict[str, Any], float], None] | None = None,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.idle_timeout_seconds = float(idle_timeout_seconds)
        if (
            not math.isfinite(self.idle_timeout_seconds)
            or self.idle_timeout_seconds <= 0
            or self.idle_timeout_seconds > MAX_IDLE_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "idle_timeout_seconds must be finite, positive, and no greater "
                f"than {MAX_IDLE_TIMEOUT_SECONDS:g}"
            )
        self._context = mp.get_context("spawn")
        # Test-only targets must be module-level callables so they remain
        # pickleable under Windows/Linux spawn. Production always uses the
        # concrete model worker above.
        self._worker_target = worker_target or _worker_main
        self._lock = threading.Lock()
        self._process: mp.Process | None = None
        self._connection: Any | None = None
        self._identity: tuple[str, ...] | None = None
        self.peak_worker_rss_bytes = 0
        self.last_timing: dict[str, Any] = {}
        self.last_worker_reused = False
        self.last_worker_spawned = False
        self.last_worker_pid: int | None = None
        self.last_worker_load_count = 0

    @property
    def production_ready(self) -> bool:
        return self.estimated_peak_total_rss_bytes <= MAX_PRODUCTION_RSS_BYTES

    @property
    def estimated_peak_total_rss_bytes(self) -> int:
        return current_process_rss_bytes() + self.peak_worker_rss_bytes

    def _stop_active(self) -> None:
        process, connection = self._process, self._connection
        self._process = None
        self._connection = None
        self._identity = None
        self.last_worker_pid = None
        self.last_worker_load_count = 0
        if connection is not None:
            try:
                connection.close()
            except (OSError, EOFError):
                pass
        if process is not None:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    def _start(self, request: dict[str, Any]) -> None:
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=self._worker_target,
            args=(child, request, self.idle_timeout_seconds),
            daemon=True,
        )
        process.start()
        child.close()
        self._process = process
        self._connection = parent
        self._identity = _worker_identity(request)

    def close_active(self) -> None:
        with self._lock:
            self._stop_active()

    def _receive(self) -> dict[str, Any]:
        if self._connection is None:
            raise RuntimeError("Encoder worker connection is unavailable")
        if not self._connection.poll(self.timeout_seconds):
            raise TimeoutError(f"Encoder worker exceeded {self.timeout_seconds:.0f}s")
        try:
            return dict(self._connection.recv())
        except (EOFError, OSError) as error:
            exit_code = None if self._process is None else self._process.exitcode
            raise RuntimeError(
                f"Encoder worker exited without a response (exit_code={exit_code})"
            ) from error

    def execute(self, request: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        with self._lock:
            identity = _worker_identity(request)
            for attempt in range(2):
                reusable = (
                    self._process is not None
                    and self._process.is_alive()
                    and self._identity == identity
                    and self._connection is not None
                )
                if not reusable:
                    self._stop_active()
                    self._start(request)
                self.last_worker_reused = reusable
                self.last_worker_spawned = not reusable
                try:
                    if reusable:
                        self._connection.send(request)
                    result = self._receive()
                except (BrokenPipeError, EOFError, OSError, TimeoutError, RuntimeError):
                    # A worker can exit in the small race between the
                    # liveness check and sending a request after its idle
                    # timeout.  Retry once with a fresh process; errors from
                    # the fresh worker are reported without a hidden fallback.
                    self._stop_active()
                    if reusable and attempt == 0:
                        continue
                    raise
                peak = result.get("peak_rss_bytes")
                if peak is not None:
                    self.peak_worker_rss_bytes = max(self.peak_worker_rss_bytes, int(peak))
                self.last_timing = dict(result.get("timing") or {})
                self.last_timing["worker_reused"] = float(bool(reusable))
                self.last_timing["worker_spawned"] = float(not reusable)
                worker_pid = result.get("timing", {}).get("worker_pid")
                self.last_worker_pid = int(worker_pid) if worker_pid is not None else None
                self.last_worker_load_count = int(
                    result.get("timing", {}).get("worker_load_count", 0)
                )
                if not result.get("ok"):
                    self._stop_active()
                    raise RuntimeError(
                        f"Encoder worker failed: {result.get('error')}\n{result.get('traceback', '')}"
                    )
                return np.asarray(result["vectors"], dtype=np.float32), list(result["diagnostics"])
            raise RuntimeError("Encoder worker retry loop exhausted")


class ProcessBranch1Encoders:
    """Branch-1 encoder API backed by short-lived worker processes."""

    def __init__(self, manager: EncoderWorkerManager, model_root: Path | None = None) -> None:
        self.manager = manager
        self.inspector = SequentialBranch1Encoders(model_root)
        self.model_root = self.inspector.model_root
        self.revisions = self.inspector.revisions

    def health(self) -> dict[str, dict[str, Any]]:
        return self.inspector.health()

    def encode(self, model_name: str, texts: list[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if model_name in {"siglip2", "metaclip2"} and len(texts) == 12:
            language_contract = "languages=vi,en"
        elif model_name == "siglip2":
            language_contract = "languages=vi"
        else:
            language_contract = "languages=en"
        return self.manager.execute(
            {
                "kind": "branch1_text",
                "model_name": model_name,
                "texts": texts,
                "model_root": str(self.model_root),
                "model_revision": self.revisions[model_name],
                "tokenizer_config": (
                    f"{language_contract};max_tokens=64;normalization=l2"
                    if model_name == "siglip2"
                    else f"{language_contract};max_tokens=77;normalization=l2"
                    if model_name == "metaclip2"
                    else f"{language_contract};max_tokens=64;output=language_head;normalization=l2"
                ),
            }
        )

    def unload(self) -> None:
        # Keep the worker alive for the bounded idle period.  A subsequent
        # request with another identity causes the manager to stop it first.
        return None


class ProcessCpuTextEncoders:
    """Existing workbench encoder contract backed by isolated workers."""

    def __init__(
        self,
        manager: EncoderWorkerManager,
        *,
        siglip_id: str = "google/siglip2-base-patch16-224",
        siglip_revision: str = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        bge_id: str = "BAAI/bge-m3",
        bge_revision: str | None = None,
    ) -> None:
        self.manager = manager
        self.siglip_id = siglip_id
        self.siglip_revision = siglip_revision
        self.bge_id = bge_id
        self.inspector = CpuTextEncoders(
            siglip_id=siglip_id,
            siglip_revision=siglip_revision,
            bge_id=bge_id,
            bge_revision=bge_revision,
        )
        self.bge_revision = self.inspector.bge_revision

    def _base(self) -> dict[str, Any]:
        return {
            "siglip_id": self.siglip_id,
            "siglip_revision": self.siglip_revision,
            "bge_id": self.bge_id,
            "bge_revision": self.bge_revision,
        }

    def health(self) -> dict[str, object]:
        return self.inspector.health()

    def encode_bge_text(self, texts: list[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        return self.manager.execute(
            {
                **self._base(),
                "kind": "bge_text",
                "texts": texts,
                "tokenizer_config": "max_tokens=512;pooling=cls;normalization=l2",
            }
        )

    def embed_bge_text(self, texts: list[str] | str) -> np.ndarray:
        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        vectors, _ = self.encode_bge_text(values)
        return vectors[0] if single else vectors

    def embed_siglip_text(self, text: str) -> np.ndarray:
        vectors, _ = self.manager.execute(
            {
                **self._base(),
                "kind": "siglip_text",
                "text": text,
                "tokenizer_config": "max_tokens=64;normalization=l2",
            }
        )
        return vectors

    def siglip_text_diagnostics(self, text: str) -> dict[str, Any]:
        """Return tokenizer diagnostics from the same isolated SigLIP worker."""
        _vector, diagnostics = self.manager.execute(
            {
                **self._base(),
                "kind": "siglip_text",
                "text": text,
                "tokenizer_config": "max_tokens=64;normalization=l2",
            }
        )
        if diagnostics:
            return diagnostics[0]
        return {
            "token_count": None,
            "max_tokens": 64,
            "truncated": False,
            "effective_query": text,
        }

    def embed_siglip_image(self, image: Image.Image) -> np.ndarray:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        vectors, _ = self.manager.execute(
            {
                **self._base(),
                "kind": "siglip_image",
                "image_bytes": buffer.getvalue(),
                "tokenizer_config": "max_tokens=64;normalization=l2",
            }
        )
        return vectors

    def warm(self) -> None:
        # Deliberately no-op: model processes are demand-loaded only.
        return None

    def unload_all(self) -> None:
        # Explicitly release the active worker when a route asks for a clean
        # boundary between unrelated model families.
        self.manager.close_active()
