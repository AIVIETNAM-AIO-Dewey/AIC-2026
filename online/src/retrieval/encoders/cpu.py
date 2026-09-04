"""Lazy text encoders used by the local Qdrant search server."""

from __future__ import annotations

import gc
import json
import logging
import os
import threading
from contextlib import suppress
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

from .device import (
    clear_accelerator_cache,
    cpu_fallback_allowed,
    device_contract,
    move_tensors,
    requested_device,
    resolve_device,
    transformers_dtype_kwargs,
)

LOGGER = logging.getLogger(__name__)


class CpuTextEncoders:
    def __init__(
        self,
        *,
        siglip_id: str = "google/siglip2-base-patch16-224",
        siglip_revision: str = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        bge_id: str = "BAAI/bge-m3",
        bge_revision: str | None = None,
        device: str | None = None,
        allow_cpu_fallback: bool | None = None,
    ) -> None:
        threads = max(1, int(os.environ.get("AIC_CPU_THREADS", "8")))
        torch.set_num_threads(threads)
        with suppress(RuntimeError):
            torch.set_num_interop_threads(1)
        self.siglip_id = siglip_id
        self.siglip_revision = siglip_revision
        self.bge_id = bge_id
        self.bge_revision = (
            bge_revision or os.environ.get("AIC_BGE_REVISION") or self._manifest_revision("bge_m3")
        )
        self.requested_device = requested_device(device)
        self.allow_cpu_fallback = cpu_fallback_allowed(allow_cpu_fallback)
        self.device = resolve_device(
            self.requested_device,
            allow_fallback=self.allow_cpu_fallback,
        )
        self._siglip_tokenizer = None
        self._siglip_processor = None
        self._siglip_model = None
        self._bge_tokenizer = None
        self._bge_model = None
        self._lock = threading.RLock()
        self._health_bge_probe: dict[str, object] | None = None

    @staticmethod
    def _manifest_revision(model_name: str) -> str | None:
        path = Path(os.environ.get("AIC_QUERY_MODEL_MANIFEST", "/models/query_models.json"))
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return str(value["models"][model_name]["revision"])
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def _manifest_assets(
        self, model_name: str, model_id: str, revision: str | None, tokenizer_config: str
    ) -> dict[str, object]:
        path = Path(os.environ.get("AIC_QUERY_MODEL_MANIFEST", "/models/query_models.json"))
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            model = (manifest.get("models") or {}).get(model_name) or {}
            files = model.get("files") or []
            if (
                manifest.get("schema_version") != "query.models.v2"
                or model.get("model_id") != model_id
                or model.get("revision") != revision
                or model.get("tokenizer_config") != tokenizer_config
                or not isinstance(files, list)
                or not files
            ):
                raise ValueError("query model manifest identity or schema mismatch")
            for record in files:
                file_path = Path(str(record["path"]))
                stat = file_path.stat()
                if (
                    stat.st_size != int(record["size"])
                    or stat.st_mtime_ns != int(record["mtime_ns"])
                    or not str(record.get("sha256") or "")
                ):
                    raise ValueError(f"query model asset changed: {file_path}")
            return {"ready": True, "manifest": str(path), "file_count": len(files)}
        except (OSError, ValueError, TypeError, KeyError) as error:
            return {"ready": False, "manifest": str(path), "error": str(error)}

    def health(self) -> dict[str, object]:
        """Report whether the configured BGE-M3 config is available locally."""
        try:
            with self._lock:
                probe = self._health_bge_probe
                if probe is None:
                    kwargs = {"revision": self.bge_revision} if self.bge_revision else {}
                    config = AutoConfig.from_pretrained(
                        self.bge_id,
                        local_files_only=True,
                        **kwargs,
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.bge_id,
                        trust_remote_code=False,
                        local_files_only=True,
                        **kwargs,
                    )
                    probe = {
                        "dimension": int(getattr(config, "hidden_size", 0)),
                        "tokenizer_ready": tokenizer is not None,
                    }
                    if probe["dimension"] == 1024 and probe["tokenizer_ready"] is True:
                        self._health_bge_probe = probe
                dimension = int(probe["dimension"])
                tokenizer_ready = probe["tokenizer_ready"] is True
            asset_health = self._manifest_assets(
                "bge_m3",
                self.bge_id,
                self.bge_revision,
                "max_tokens=512;pooling=cls;normalization=l2",
            )
            return {
                "ready": dimension == 1024 and tokenizer_ready and asset_health["ready"] is True,
                "model_id": self.bge_id,
                "revision": self.bge_revision or "local-cache",
                "dimension": dimension,
                "local_files_only": True,
                "snapshot_assets": asset_health,
                "execution_device": self.device,
                "device": device_contract(
                    self.requested_device,
                    allow_fallback=self.allow_cpu_fallback,
                ),
            }
        except Exception as error:  # transformers raises several cache-specific exception types
            return {
                "ready": False,
                "model_id": self.bge_id,
                "revision": self.bge_revision or "local-cache",
                "dimension": 1024,
                "local_files_only": True,
                "execution_device": self.device,
                "device": device_contract(
                    self.requested_device,
                    allow_fallback=self.allow_cpu_fallback,
                ),
                "error": str(error),
            }

    def _load_siglip(self) -> None:
        if self._siglip_model is not None:
            return
        LOGGER.info("Loading SigLIP2 encoder on %s: %s", self.device, self.siglip_id)
        self._siglip_tokenizer = AutoTokenizer.from_pretrained(
            self.siglip_id,
            revision=self.siglip_revision,
            trust_remote_code=False,
            local_files_only=True,
        )
        self._siglip_processor = AutoProcessor.from_pretrained(
            self.siglip_id,
            revision=self.siglip_revision,
            trust_remote_code=False,
            local_files_only=True,
            use_fast=False,
        )
        self._siglip_model = AutoModel.from_pretrained(
            self.siglip_id,
            revision=self.siglip_revision,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            local_files_only=True,
            **transformers_dtype_kwargs(torch.float32),
        )
        self._siglip_model.to(self.device).eval()
        LOGGER.info("SigLIP2 %s encoder ready", self.device)

    def _load_bge(self) -> None:
        if self._bge_model is not None:
            return
        LOGGER.info("Loading BGE-M3 dense encoder on %s: %s", self.device, self.bge_id)
        kwargs = {"revision": self.bge_revision} if self.bge_revision else {}
        self._bge_tokenizer = AutoTokenizer.from_pretrained(
            self.bge_id,
            trust_remote_code=False,
            local_files_only=True,
            **kwargs,
        )
        self._bge_model = AutoModel.from_pretrained(
            self.bge_id,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            local_files_only=True,
            **transformers_dtype_kwargs(torch.float32),
            **kwargs,
        )
        self._bge_model.to(self.device).eval()
        LOGGER.info("BGE-M3 %s encoder ready", self.device)

    def warm(self) -> None:
        self.embed_siglip_text("warmup")
        self.embed_bge_text("warmup")

    def unload_all(self) -> None:
        """Release the general workbench encoders before a heavy Branch-1 search."""
        with self._lock:
            self._siglip_tokenizer = None
            self._siglip_processor = None
            self._siglip_model = None
            self._bge_tokenizer = None
            self._bge_model = None
        gc.collect()
        clear_accelerator_cache(self.device)

    def siglip_text_diagnostics(self, text: str) -> dict[str, object]:
        with self._lock:
            self._load_siglip()
            token_ids = self._siglip_tokenizer(text, add_special_tokens=True)["input_ids"]
        return {
            "token_count": len(token_ids),
            "max_tokens": 64,
            "truncated": len(token_ids) > 64,
            "effective_query": text,
        }

    @torch.inference_mode()
    def embed_siglip_text(self, text: str) -> np.ndarray:
        with self._lock:
            self._load_siglip()
            inputs = self._siglip_tokenizer(
                [text],
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            inputs = move_tensors(inputs, self.device)
            output = self._siglip_model.get_text_features(**inputs)
            if isinstance(output, torch.Tensor):
                features = output
            elif getattr(output, "pooler_output", None) is not None:
                features = output.pooler_output
            else:
                features = output[0]
            vector = F.normalize(features, p=2, dim=-1)[0]
            result = vector.to(torch.float32).cpu().numpy()
        if result.shape != (768,):
            raise ValueError(f"SigLIP2 returned unexpected shape {result.shape}")
        return result

    @torch.inference_mode()
    def embed_siglip_image(self, image: Image.Image) -> np.ndarray:
        with self._lock:
            self._load_siglip()
            inputs = self._siglip_processor(images=image.convert("RGB"), return_tensors="pt")
            inputs = move_tensors(inputs, self.device)
            output = self._siglip_model.get_image_features(**inputs)
            if isinstance(output, torch.Tensor):
                features = output
            elif getattr(output, "pooler_output", None) is not None:
                features = output.pooler_output
            else:
                features = output[0]
            vector = F.normalize(features, p=2, dim=-1)[0]
            result = vector.to(torch.float32).cpu().numpy()
        if result.shape != (768,):
            raise ValueError(f"SigLIP2 image encoder returned unexpected shape {result.shape}")
        return result

    @torch.inference_mode()
    def embed_bge_text(self, texts: list[str] | str) -> np.ndarray:
        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        clean = [value.strip() or " " for value in values]
        with self._lock:
            self._load_bge()
            inputs = self._bge_tokenizer(
                clean,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = move_tensors(inputs, self.device)
            output = self._bge_model(**inputs)
            vectors = F.normalize(output.last_hidden_state[:, 0], p=2, dim=-1)
            result = vectors.to(torch.float32).cpu().numpy()
        if result.ndim != 2 or result.shape[1] != 1024:
            raise ValueError(f"BGE-M3 returned unexpected shape {result.shape}")
        return result[0] if single else result

    @torch.inference_mode()
    def encode_bge_text(self, texts: list[str]) -> tuple[np.ndarray, list[dict[str, object]]]:
        clean = [value.strip() or " " for value in texts]
        with self._lock:
            self._load_bge()
            raw_counts = [
                len(self._bge_tokenizer(value, add_special_tokens=True)["input_ids"])
                for value in clean
            ]
            inputs = self._bge_tokenizer(
                clean,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = move_tensors(inputs, self.device)
            output = self._bge_model(**inputs)
            vectors = F.normalize(output.last_hidden_state[:, 0], p=2, dim=-1)
            result = vectors.to(torch.float32).cpu().numpy()
        if result.ndim != 2 or result.shape[1] != 1024:
            raise ValueError(f"BGE-M3 returned unexpected shape {result.shape}")
        diagnostics = [
            {"token_count": count, "max_tokens": 512, "truncated": count > 512}
            for count in raw_counts
        ]
        return result, diagnostics
