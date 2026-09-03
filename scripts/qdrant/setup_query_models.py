"""Resolve and cache immutable Hugging Face snapshots used by CPU query workers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


MODELS = {
    "siglip2": ("google/siglip2-base-patch16-224", "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"),
    "metaclip2": ("facebook/metaclip-2-worldwide-huge-quickgelu", "2431b607fc8e05dd43b73797ba1a7a042514bcf4"),
    "bge_m3": ("BAAI/bge-m3", None),
}
TOKENIZER_CONTRACTS = {
    "siglip2": "max_tokens=64;normalization=l2",
    "metaclip2": "max_tokens=77;normalization=l2",
    "bge_m3": "max_tokens=512;pooling=cls;normalization=l2",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_inventory(path: Path) -> list[dict[str, object]]:
    """Hash the materialized snapshot once during setup, never at runtime."""
    files: list[dict[str, object]] = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        stat = candidate.stat()
        files.append(
            {
                "path": str(candidate),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(candidate),
            }
        )
    if not files:
        raise RuntimeError(f"Model snapshot is empty: {path}")
    return files


def main() -> int:
    model_root = Path(os.environ.get("AIC_MODEL_ROOT", "/models"))
    hf_home = Path(os.environ.get("HF_HOME", str(model_root / "huggingface")))
    cache_root = hf_home / "hub"
    cache_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    models: dict[str, dict[str, str]] = {}
    manifest: dict[str, object] = {"schema_version": "query.models.v2", "models": models}
    for name, (model_id, configured_revision) in MODELS.items():
        revision = configured_revision or api.model_info(model_id, revision="main").sha
        if not revision:
            raise RuntimeError(f"Could not resolve immutable revision for {model_id}")
        snapshot = Path(snapshot_download(repo_id=model_id, revision=revision, cache_dir=cache_root))
        models[name] = {
            "model_id": model_id,
            "revision": revision,
            "snapshot_path": str(snapshot),
            "tokenizer_config": TOKENIZER_CONTRACTS[name],
            "files": snapshot_inventory(snapshot),
        }
    target = model_root / "query_models.json"
    staging = target.with_suffix(".staging.json")
    staging.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(staging, target)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
