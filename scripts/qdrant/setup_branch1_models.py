#!/usr/bin/env python3
"""Download pinned BEiT-3 runtime assets into the persistent model volume."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


UNILM_REVISION = "ca43e4cd19445a536f133bf2bc25b573b2f0c7c5"
UNILM_URL = f"https://github.com/microsoft/unilm/archive/{UNILM_REVISION}.zip"
CHECKPOINT_URL = (
    "https://github.com/addf400/files/releases/download/beit3/"
    "beit3_base_patch16_384_coco_retrieval.pth"
)
SENTENCEPIECE_URL = "https://github.com/addf400/files/releases/download/beit3/beit3.spm"

# Populated after a one-time controlled hash-discovery download, then enforced on every setup.
EXPECTED_SHA256 = {
    "unilm.zip": "e12617e2dcbae818f051b74ad146253ee406889715c451f345a5fcb88fe41d81",
    "beit3_base_patch16_384_coco_retrieval.pth": "df39666a88508ccd356567616582bc62cd56fa86ad6a8f8e50471b35217c8629",
    "beit3.spm": "6f5e2fefcf793761a76a6bfb8ad35489f9c203b25557673284b6d032f41043f4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "AIC-2026-Branch1-Setup/1"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    partial.replace(destination)


def verify(path: Path, expected: str, allow_discovery: bool) -> str:
    actual = sha256(path)
    if not expected:
        if not allow_discovery:
            raise RuntimeError(f"No trusted SHA-256 is pinned for {path.name}")
    elif actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path.name}: {actual} != {expected}")
    return actual


def asset_record(path: Path, digest: str) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}


def safe_extract_beit3(archive: Path, destination: Path) -> None:
    prefix = f"unilm-{UNILM_REVISION}/beit3/"
    temporary = destination.with_name(destination.name + ".staging")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            if not info.filename.startswith(prefix) or info.is_dir():
                continue
            relative = Path(info.filename[len(prefix) :])
            target = (temporary / relative).resolve()
            if temporary.resolve() not in target.parents:
                raise RuntimeError(f"Unsafe UNILM archive member: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    if not (temporary / "modeling_finetune.py").is_file():
        raise RuntimeError("Pinned UNILM archive does not contain BEiT-3 source")
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)


def main() -> int:
    root = Path(os.environ.get("AIC_BRANCH1_MODEL_ROOT", "/models/branch1"))
    root.mkdir(parents=True, exist_ok=True)
    allow_discovery = os.environ.get("AIC_ALLOW_HASH_DISCOVERY") == "1"
    downloads = {
        "unilm.zip": UNILM_URL,
        "beit3_base_patch16_384_coco_retrieval.pth": CHECKPOINT_URL,
        "beit3.spm": SENTENCEPIECE_URL,
    }
    observed: dict[str, str] = {}
    with tempfile.TemporaryDirectory(dir=root) as temp_name:
        temp = Path(temp_name)
        for name, url in downloads.items():
            destination = temp / name
            persistent = root / name
            if persistent.is_file():
                destination = persistent
            else:
                print(f"Downloading {url}", flush=True)
                download(url, destination)
            observed[name] = verify(destination, EXPECTED_SHA256[name], allow_discovery)
            if destination != persistent:
                destination.replace(persistent)
        archive = root / "unilm.zip"
        safe_extract_beit3(archive, root / "unilm" / "beit3")
    manifest = {
        "schema_version": "branch1.models.v2",
        "unilm_revision": UNILM_REVISION,
        "sha256": observed,
        "urls": downloads,
        "assets": {
            name: asset_record(root / name, digest)
            for name, digest in observed.items()
        },
    }
    manifest_path = root / "manifest.json"
    staging = manifest_path.with_suffix(".staging.json")
    staging.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(staging, manifest_path)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
