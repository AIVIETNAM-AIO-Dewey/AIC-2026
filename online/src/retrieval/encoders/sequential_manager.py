"""Sequential CPU text encoders for Branch-1 visual retrieval."""

from __future__ import annotations

import abc
import gc
import json
import logging
import os
import sys
import threading
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, XLMRobertaTokenizer


LOGGER = logging.getLogger(__name__)
SIGLIP_ID = "google/siglip2-base-patch16-224"
SIGLIP_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
METACLIP2_ID = "facebook/metaclip-2-worldwide-huge-quickgelu"
METACLIP2_REVISION = "2431b607fc8e05dd43b73797ba1a7a042514bcf4"
BEIT3_CHECKPOINT_NAME = "beit3_base_patch16_384_coco_retrieval.pth"
UNILM_REVISION = "ca43e4cd19445a536f133bf2bc25b573b2f0c7c5"
BEIT3_CHECKPOINT_SHA256 = "df39666a88508ccd356567616582bc62cd56fa86ad6a8f8e50471b35217c8629"
BEIT3_SENTENCEPIECE_SHA256 = "6f5e2fefcf793761a76a6bfb8ad35489f9c203b25557673284b6d032f41043f4"
UNILM_SOURCE_SHA256 = "e12617e2dcbae818f051b74ad146253ee406889715c451f345a5fcb88fe41d81"


def _query_snapshot_assets(
    model_name: str, model_id: str, revision: str, tokenizer_config: str
) -> dict[str, Any]:
    """Validate setup-time snapshot inventory using stat only at runtime."""
    manifest_path = Path(os.environ.get("AIC_QUERY_MODEL_MANIFEST", "/models/query_models.json"))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = (manifest.get("models") or {}).get(model_name) or {}
        files = model.get("files") or []
        if (
            manifest.get("schema_version") != "query.models.v2"
            or model.get("model_id") != model_id
            or model.get("revision") != revision
            or model.get("tokenizer_config") != tokenizer_config
            or not isinstance(files, list)
        ):
            raise ValueError("query model manifest identity or schema mismatch")
        for record in files:
            path = Path(str(record["path"]))
            stat = path.stat()
            if (
                stat.st_size != int(record["size"])
                or stat.st_mtime_ns != int(record["mtime_ns"])
                or not str(record.get("sha256") or "")
            ):
                raise ValueError(f"query model asset changed: {path}")
        return {
            "ready": bool(files),
            "manifest": str(manifest_path),
            "file_count": len(files),
        }
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {"ready": False, "manifest": str(manifest_path), "error": str(error)}


class SequentialBranch1Encoders:
    """Loads exactly one text encoder at a time and unloads it after use."""

    def __init__(self, model_root: Path | None = None) -> None:
        threads = max(1, int(os.environ.get("AIC_CPU_THREADS", "8")))
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch only allows the inter-op pool to be configured before
            # the first parallel operation.  A worker may already have
            # initialized it while importing a sibling encoder.
            pass
        self.model_root = model_root or Path(os.environ.get("AIC_BRANCH1_MODEL_ROOT", "/models/branch1"))
        self.beit3_source = self.model_root / "unilm" / "beit3"
        self.beit3_source_archive = self.model_root / "unilm.zip"
        self.beit3_checkpoint = self.model_root / BEIT3_CHECKPOINT_NAME
        self.beit3_sentencepiece = self.model_root / "beit3.spm"
        self.revisions = {
            "siglip2": SIGLIP_REVISION,
            "metaclip2": METACLIP2_REVISION,
            # Include every immutable BEiT-3 input in the cache namespace:
            # changing the checkpoint, tokenizer model, or source code must
            # never reuse an embedding produced by a different runtime.
            "beit3": ":".join(
                (
                    os.environ.get("AIC_BEIT3_CHECKPOINT_SHA256", BEIT3_CHECKPOINT_SHA256),
                    BEIT3_SENTENCEPIECE_SHA256,
                    UNILM_REVISION,
                )
            ),
        }
        self._lock = threading.RLock()
        self._loaded_name: str | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def health(self) -> dict[str, dict[str, Any]]:
        try:
            from transformers import AutoConfig, AutoTokenizer
        except ImportError as error:
            return {
                "siglip2": {"ready": False, "error": str(error)},
                "metaclip2": {"ready": False, "error": str(error)},
                "beit3": {"ready": False, "error": str(error)},
            }
        try:
            from transformers import AutoProcessor
        except ImportError:
            AutoProcessor = None  # type: ignore[assignment,misc]
        try:
            from transformers import MetaClip2TextModelWithProjection  # noqa: F401

            metaclip_supported = True
        except ImportError:
            metaclip_supported = False
        def local_text_assets(model_id: str, revision: str) -> dict[str, Any]:
            try:
                kwargs = {"revision": revision} if revision else {}
                config = AutoConfig.from_pretrained(
                    model_id, trust_remote_code=False, local_files_only=True, **kwargs
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, trust_remote_code=False, local_files_only=True, **kwargs
                )
                return {
                    "ready": config is not None and tokenizer is not None,
                    "config": True,
                    "tokenizer": True,
                }
            except Exception as error:
                return {"ready": False, "config": False, "tokenizer": False, "error": str(error)}

        siglip_assets = local_text_assets(SIGLIP_ID, SIGLIP_REVISION)
        siglip_snapshot = _query_snapshot_assets(
            "siglip2", SIGLIP_ID, SIGLIP_REVISION, "max_tokens=64;normalization=l2"
        )
        if AutoProcessor is None:
            siglip_image_assets = {"ready": False, "error": "AutoProcessor is unavailable"}
        else:
            try:
                siglip_processor = AutoProcessor.from_pretrained(
                    SIGLIP_ID,
                    revision=SIGLIP_REVISION,
                    trust_remote_code=False,
                    local_files_only=True,
                )
                siglip_image_assets = {"ready": siglip_processor is not None}
            except Exception as error:
                siglip_image_assets = {"ready": False, "error": str(error)}
        metaclip_assets = local_text_assets(METACLIP2_ID, METACLIP2_REVISION)
        metaclip_snapshot = _query_snapshot_assets(
            "metaclip2", METACLIP2_ID, METACLIP2_REVISION, "max_tokens=77;normalization=l2"
        )
        beit_files = {
            "source": (self.beit3_source / "modeling_finetune.py").is_file(),
            "source_archive": self.beit3_source_archive.is_file(),
            "checkpoint": self.beit3_checkpoint.is_file(),
            "sentencepiece": self.beit3_sentencepiece.is_file(),
        }
        manifest_path = self.model_root / "manifest.json"
        manifest_ok = False
        manifest_hashes: dict[str, str] = {}
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_hashes = dict(manifest.get("sha256") or {})
            assets = manifest.get("assets") or {}
            manifest_ok = (
                manifest.get("schema_version") == "branch1.models.v2"
                and manifest.get("unilm_revision") == UNILM_REVISION
                and manifest_hashes.get("unilm.zip") == UNILM_SOURCE_SHA256
                and manifest_hashes.get(BEIT3_CHECKPOINT_NAME) == BEIT3_CHECKPOINT_SHA256
                and manifest_hashes.get("beit3.spm") == BEIT3_SENTENCEPIECE_SHA256
                and all(
                    isinstance(assets.get(name), dict)
                    and int(assets[name].get("size", -1)) == path.stat().st_size
                    and int(assets[name].get("mtime_ns", -1)) == path.stat().st_mtime_ns
                    and assets[name].get("sha256") == manifest_hashes.get(name)
                    for name, path in (
                        ("unilm.zip", self.beit3_source_archive),
                        (BEIT3_CHECKPOINT_NAME, self.beit3_checkpoint),
                        ("beit3.spm", self.beit3_sentencepiece),
                    )
                )
            )
        except (OSError, ValueError, TypeError, AttributeError):
            pass
        return {
            "siglip2": {
                "ready": bool(siglip_assets["ready"]) and bool(siglip_snapshot["ready"]),
                "model_id": SIGLIP_ID,
                "revision": SIGLIP_REVISION,
                "dimension": 768,
                "max_tokens": 64,
                "local_assets": siglip_assets,
                "snapshot_assets": siglip_snapshot,
                "image_ready": bool(siglip_image_assets["ready"]),
                "image_assets": siglip_image_assets,
            },
            "metaclip2": {
                "ready": metaclip_supported and bool(metaclip_assets["ready"]) and bool(metaclip_snapshot["ready"]),
                "model_id": METACLIP2_ID,
                "revision": METACLIP2_REVISION,
                "dimension": 1024,
                "max_tokens": 77,
                "transformers_support": metaclip_supported,
                "local_assets": metaclip_assets,
                "snapshot_assets": metaclip_snapshot,
            },
            "beit3": {
                "ready": all(beit_files.values()) and manifest_ok,
                "checkpoint": str(self.beit3_checkpoint),
                "source_revision": UNILM_REVISION,
                "dimension": 768,
                "max_tokens": 64,
                "text_output": "language_head",
                "checkpoint_task": "COCO Retrieval",
                "files": beit_files,
                "manifest": str(manifest_path),
                "manifest_hashes": manifest_hashes,
                "hashes_verified": manifest_ok,
            },
        }

    def _load_siglip2(self) -> None:
        from transformers import AutoModel

        self._tokenizer = AutoTokenizer.from_pretrained(
            SIGLIP_ID, revision=SIGLIP_REVISION, trust_remote_code=False, local_files_only=True
        )
        self._model = AutoModel.from_pretrained(
            SIGLIP_ID,
            revision=SIGLIP_REVISION,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            local_files_only=True,
        ).eval()

    def _load_metaclip2(self) -> None:
        from transformers import AutoConfig, MetaClip2TextModelWithProjection

        self._tokenizer = AutoTokenizer.from_pretrained(
            METACLIP2_ID, revision=METACLIP2_REVISION, trust_remote_code=False, local_files_only=True
        )
        # The published worldwide-huge checkpoint stores the shared retrieval
        # projection dimension on the parent MetaCLIP2 config (1024), while its
        # nested text config still carries the architecture default (512).
        # Loading the text-only class from the nested value creates a 512x1024
        # projection and then fails against the checkpoint's 1024x1024 weight.
        # Copy the parent retrieval dimension into the text config so we can
        # load only the text tower without materialising the vision tower.
        full_config = AutoConfig.from_pretrained(
            METACLIP2_ID,
            revision=METACLIP2_REVISION,
            trust_remote_code=False,
            local_files_only=True,
        )
        text_config = full_config.text_config
        text_config.projection_dim = int(full_config.projection_dim)
        self._model = MetaClip2TextModelWithProjection.from_pretrained(
            METACLIP2_ID,
            revision=METACLIP2_REVISION,
            config=text_config,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).eval()

    @staticmethod
    def _install_beit3_compatibility_shims() -> None:
        torch_six = types.ModuleType("torch._six")
        torch_six.container_abcs = abc
        torch_six.inf = float("inf")
        torch_six.string_classes = (str,)
        sys.modules.setdefault("torch._six", torch_six)

        utils = types.ModuleType("utils")
        utils.get_rank = lambda: 0
        utils.get_world_size = lambda: 1

        class ClipLoss(nn.Module):
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                super().__init__()

            def forward(self, *_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError("ClipLoss is unavailable in inference mode")

        utils.ClipLoss = ClipLoss
        sys.modules["utils"] = utils

    def _load_beit3(self) -> None:
        missing = [
            path
            for path in (
                self.beit3_source / "modeling_finetune.py",
                self.beit3_checkpoint,
                self.beit3_sentencepiece,
            )
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"BEiT-3 local model files are missing: {missing}")
        self._install_beit3_compatibility_shims()
        sys.path.insert(0, str(self.beit3_source))
        try:
            from modeling_finetune import beit3_base_patch16_384_retrieval

            model = beit3_base_patch16_384_retrieval()
            # The official fine-tuning artifact is a checkpoint dictionary
            # containing optimizer/args metadata in addition to ``model``.
            # ``weights_only=True`` rejects those trusted metadata objects on
            # newer PyTorch releases, so load the hash-verified local file in
            # its documented full-checkpoint form.
            checkpoint = torch.load(self.beit3_checkpoint, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model", checkpoint)
            # Microsoft’s loader intentionally ignores the non-persistent
            # relative-position index buffers.  Keep that exact allowance,
            # while still failing on every real parameter mismatch so a
            # different task/checkpoint cannot silently enter retrieval.
            incompatible = model.load_state_dict(state_dict, strict=False)
            missing = [
                key
                for key in incompatible.missing_keys
                if "relative_position_index" not in key
            ]
            unexpected = list(incompatible.unexpected_keys)
            if missing or unexpected:
                raise RuntimeError(
                    "BEiT-3 checkpoint mismatch: "
                    f"missing={missing[:10]}, unexpected={unexpected[:10]}"
                )
        finally:
            if sys.path and sys.path[0] == str(self.beit3_source):
                sys.path.pop(0)
        self._tokenizer = XLMRobertaTokenizer(str(self.beit3_sentencepiece))
        self._model = model.eval()

    def _load(self, model_name: str) -> None:
        if self._loaded_name == model_name and self._model is not None:
            return
        self.unload()
        LOGGER.info("Loading Branch-1 %s text encoder on CPU", model_name)
        if model_name == "siglip2":
            self._load_siglip2()
        elif model_name == "metaclip2":
            self._load_metaclip2()
        elif model_name == "beit3":
            self._load_beit3()
        else:
            raise ValueError(f"Unknown Branch-1 model: {model_name}")
        self._loaded_name = model_name

    @staticmethod
    def _diagnostic(token_counts: list[int], max_tokens: int) -> list[dict[str, Any]]:
        return [
            {"token_count": count, "max_tokens": max_tokens, "truncated": count > max_tokens}
            for count in token_counts
        ]

    @torch.inference_mode()
    def encode(self, model_name: str, texts: list[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
        expected_rows = 6 if model_name == "beit3" else 12
        if len(texts) != expected_rows:
            raise ValueError(
                f"Branch-1 {model_name} encoder requires exactly {expected_rows} queries"
            )
        with self._lock:
            self._load(model_name)
            if model_name == "siglip2":
                raw_counts = [len(self._tokenizer(text, add_special_tokens=True)["input_ids"]) for text in texts]
                inputs = self._tokenizer(
                    texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt"
                )
                output = self._model.get_text_features(**inputs)
                features = output if isinstance(output, torch.Tensor) else output.pooler_output
                vectors = F.normalize(features.float(), p=2, dim=-1)
                diagnostics = self._diagnostic(raw_counts, 64)
            elif model_name == "metaclip2":
                raw_counts = [len(self._tokenizer(text, add_special_tokens=True)["input_ids"]) for text in texts]
                inputs = self._tokenizer(
                    texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
                )
                output = self._model(**inputs)
                vectors = F.normalize(output.text_embeds.float(), p=2, dim=-1)
                diagnostics = self._diagnostic(raw_counts, 77)
            else:
                encoded: list[list[int]] = []
                padding_masks: list[list[int]] = []
                raw_counts = []
                for text in texts:
                    token_ids = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(text))
                    raw_counts.append(len(token_ids) + 2)
                    token_ids = token_ids[:62]
                    row = [self._tokenizer.bos_token_id, *token_ids, self._tokenizer.eos_token_id]
                    padding = [0] * len(row) + [1] * (64 - len(row))
                    encoded.append(row + [self._tokenizer.pad_token_id] * (64 - len(row)))
                    padding_masks.append(padding)
                _, language = self._model(
                    text_description=torch.tensor(encoded, dtype=torch.long),
                    padding_mask=torch.tensor(padding_masks, dtype=torch.bool),
                    only_infer=True,
                )
                vectors = F.normalize(language.float(), p=2, dim=-1)
                diagnostics = self._diagnostic(raw_counts, 64)
            result = vectors.cpu().numpy().astype(np.float32, copy=False)
        return result, diagnostics

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._loaded_name = None
        gc.collect()
