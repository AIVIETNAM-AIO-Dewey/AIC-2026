"""Short-lived accelerator workers with deterministic memory reclamation."""

from __future__ import annotations

import io
import math
import multiprocessing as mp
import os
import threading
import time
import traceback
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..infrastructure.resources import (
    MAX_PRODUCTION_RSS_BYTES,
    current_process_rss_bytes,
    peak_process_rss_bytes,
)
from .cpu import CpuTextEncoders
from .device import (
    clear_accelerator_cache,
    cpu_fallback_allowed,
    device_contract,
    is_mps_runtime_error,
    requested_device,
    resolve_device,
)
from .sequential_manager import SequentialBranch1Encoders

MAX_IDLE_TIMEOUT_SECONDS = 30.0


def _worker_identity(request: dict[str, Any]) -> tuple[str, ...]:
    """Return the immutable model identity allowed to stay resident."""
    kind = str(request["kind"])
    # The identity is deliberately stricter than the model name.  A worker
    # may be reused only when the checkpoint/revision and the preprocessing
    # contract are identical; otherwise a cached model could silently serve
    # vectors from a different tokenizer or projection head.
    tokenizer_config = str(request.get("tokenizer_config") or "default")
    accelerator_contract = (
        str(request.get("device") or "cpu"),
        str(bool(request.get("allow_cpu_fallback", True))),
    )
    if kind == "branch1_text":
        return (
            kind,
            str(request["model_name"]),
            str(request["model_root"]),
            str(request.get("model_revision") or "unknown-revision"),
            tokenizer_config,
            *accelerator_contract,
        )
    if kind == "bge_text":
        return (
            "bge_text",
            str(request["bge_id"]),
            str(request.get("bge_revision") or "local-cache"),
            tokenizer_config,
            *accelerator_contract,
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
            *accelerator_contract,
        )
    raise ValueError(f"Unknown encoder worker request: {kind}")


def _device_state_key(identity: tuple[str, ...]) -> str:
    return ":".join(identity[:2])


def _load_worker_encoder(request: dict[str, Any]) -> tuple[Any, float]:
    load_started = time.perf_counter()
    kind = str(request["kind"])
    if kind == "branch1_text":
        encoder = SequentialBranch1Encoders(
            Path(request["model_root"]),
            device=str(request.get("device") or "cpu"),
            allow_cpu_fallback=bool(request.get("allow_cpu_fallback", True)),
        )
        encoder._load(str(request["model_name"]))
    else:
        encoder = CpuTextEncoders(
            siglip_id=str(request["siglip_id"]),
            siglip_revision=str(request["siglip_revision"]),
            bge_id=str(request["bge_id"]),
            bge_revision=(str(request["bge_revision"]) if request.get("bge_revision") else None),
            device=str(request.get("device") or "cpu"),
            allow_cpu_fallback=bool(request.get("allow_cpu_fallback", True)),
        )
        if kind == "bge_text":
            encoder._load_bge()
        elif kind in {"siglip_text", "siglip_image"}:
            encoder._load_siglip()
    return encoder, (time.perf_counter() - load_started) * 1000.0


def _unload_worker_encoder(encoder: Any | None) -> None:
    if encoder is None:
        return
    try:
        if hasattr(encoder, "unload"):
            encoder.unload()
        elif hasattr(encoder, "unload_all"):
            encoder.unload_all()
    except (AttributeError, RuntimeError):
        pass


def _can_fallback_to_cpu(request: dict[str, Any], error: BaseException) -> bool:
    return (
        bool(request.get("allow_cpu_fallback", True))
        and str(request.get("device") or "cpu") != "cpu"
        and is_mps_runtime_error(error)
    )


def _cpu_request(request: dict[str, Any]) -> dict[str, Any]:
    return {**request, "device": "cpu", "allow_cpu_fallback": False}


def _run_worker_inference(
    encoder: Any, request: dict[str, Any]
) -> tuple[Any, list[dict[str, Any]]]:
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
    fallback_reason: str | None = None
    try:
        identity = _worker_identity(request)
        try:
            encoder, first_load_ms = _load_worker_encoder(request)
        except BaseException as error:
            if not _can_fallback_to_cpu(request, error):
                raise
            fallback_reason = f"{type(error).__name__}: {error}"
            clear_accelerator_cache("mps")
            encoder, first_load_ms = _load_worker_encoder(_cpu_request(request))
        pending = request
        load_ms = first_load_ms
        while True:
            inference_started = time.perf_counter()
            try:
                vectors, diagnostics = _run_worker_inference(encoder, pending)
            except BaseException as error:
                if getattr(encoder, "device", "cpu") != "mps" or not _can_fallback_to_cpu(
                    pending, error
                ):
                    raise
                fallback_reason = f"{type(error).__name__}: {error}"
                _unload_worker_encoder(encoder)
                encoder = None
                clear_accelerator_cache("mps")
                encoder, fallback_load_ms = _load_worker_encoder(_cpu_request(pending))
                load_ms += fallback_load_ms
                vectors, diagnostics = _run_worker_inference(encoder, _cpu_request(pending))
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
                        "requested_device": str(request.get("device") or "cpu"),
                        "execution_device": str(getattr(encoder, "device", "cpu")),
                        "cpu_fallback": fallback_reason is not None,
                    },
                    "device": {
                        "requested": str(request.get("device") or "cpu"),
                        "actual": str(getattr(encoder, "device", "cpu")),
                        "fallback_reason": fallback_reason,
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
        with suppress(BrokenPipeError, EOFError, OSError):
            connection.send(
                {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "peak_rss_bytes": peak_process_rss_bytes(),
                }
            )
    finally:
        _unload_worker_encoder(encoder)
        connection.close()


class EncoderWorkerManager:
    """Serialize heavy inference and reclaim model memory by process exit."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 600.0,
        idle_timeout_seconds: float = 30.0,
        worker_target: Callable[[Any, dict[str, Any], float], None] | None = None,
        device: str | None = None,
        allow_cpu_fallback: bool | None = None,
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
        self.requested_device = requested_device(device)
        self.allow_cpu_fallback = cpu_fallback_allowed(allow_cpu_fallback)
        self.preferred_device = resolve_device(
            self.requested_device,
            allow_fallback=self.allow_cpu_fallback,
        )
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
        self.last_device = self.preferred_device
        self.last_fallback_reason: str | None = None
        self._device_state_lock = threading.RLock()
        self._device_states: dict[str, dict[str, Any]] = {}

    @property
    def cache_device(self) -> str:
        return self.preferred_device

    def cache_device_for(self, state_key: str) -> str:
        """Return the device that actually serves one model identity."""

        with self._device_state_lock:
            state = self._device_states.get(state_key) or {}
            return str(state.get("actual") or self.preferred_device)

    def device_health(self) -> dict[str, Any]:
        with self._device_state_lock:
            actual_workers = {name: dict(state) for name, state in self._device_states.items()}
        return {
            **device_contract(
                self.requested_device,
                allow_fallback=self.allow_cpu_fallback,
            ),
            "actual_workers": actual_workers,
            "last_execution_device": self.last_device,
            "last_fallback_reason": self.last_fallback_reason,
        }

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
            with suppress(OSError, EOFError):
                connection.close()
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
            request = dict(request)
            request.setdefault("device", self.requested_device)
            request.setdefault("allow_cpu_fallback", self.allow_cpu_fallback)
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
                raw_device = result.get("device") or {}
                device_state = raw_device if isinstance(raw_device, dict) else {}
                self.last_device = str(
                    device_state.get("actual")
                    or result.get("timing", {}).get("execution_device")
                    or self.preferred_device
                )
                fallback_reason = device_state.get("fallback_reason")
                self.last_fallback_reason = str(fallback_reason) if fallback_reason else None
                state_key = _device_state_key(identity)
                with self._device_state_lock:
                    self._device_states[state_key] = {
                        "requested": str(device_state.get("requested") or self.requested_device),
                        "actual": self.last_device,
                        "fallback_reason": self.last_fallback_reason,
                    }
                self.last_timing.setdefault("execution_device", self.last_device)
                self.last_timing.setdefault("cpu_fallback", self.last_fallback_reason is not None)
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
        self.inspector = SequentialBranch1Encoders(
            model_root,
            device=manager.requested_device,
            allow_cpu_fallback=manager.allow_cpu_fallback,
        )
        self.model_root = self.inspector.model_root
        self.revisions = self.inspector.revisions
        self.cache_device = manager.cache_device

    def cache_device_for_model(self, model_name: str) -> str:
        return self.manager.cache_device_for(f"branch1_text:{model_name}")

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
            device=manager.requested_device,
            allow_cpu_fallback=manager.allow_cpu_fallback,
        )
        self.bge_revision = self.inspector.bge_revision
        self.cache_device = manager.cache_device

    def cache_device_for_bge(self) -> str:
        return self.manager.cache_device_for(f"bge_text:{self.bge_id}")

    def _base(self) -> dict[str, Any]:
        return {
            "siglip_id": self.siglip_id,
            "siglip_revision": self.siglip_revision,
            "bge_id": self.bge_id,
            "bge_revision": self.bge_revision,
            "device": self.manager.requested_device,
            "allow_cpu_fallback": self.manager.allow_cpu_fallback,
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
