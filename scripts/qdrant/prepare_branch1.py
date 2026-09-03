"""Validate Branch-1 artifacts and publish fail-closed readiness manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow this command to be invoked either as a module or by file path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.qdrant.validate_branch1_data import (  # noqa: E402
    DATA_GATE_SCHEMA_VERSION,
    build_data_gate_report,
)
from online.src.retrieval.encoders.sequential_manager import SequentialBranch1Encoders  # noqa: E402


SIGLIP_ID = "google/siglip2-base-patch16-224"
SIGLIP_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
METACLIP_ID = "facebook/metaclip-2-worldwide-huge-quickgelu"
METACLIP_REVISION = "2431b607fc8e05dd43b73797ba1a7a042514bcf4"
BEIT3_ID = (
    "https://github.com/addf400/files/releases/download/beit3/"
    "beit3_base_patch16_384_coco_retrieval.pth"
)
UNILM_REVISION = "ca43e4cd19445a536f133bf2bc25b573b2f0c7c5"
BEIT3_CHECKPOINT = "beit3_base_patch16_384_coco_retrieval.pth"
UNILM_SOURCE_SHA256 = "e12617e2dcbae818f051b74ad146253ee406889715c451f345a5fcb88fe41d81"
BEIT3_CHECKPOINT_SHA256 = "df39666a88508ccd356567616582bc62cd56fa86ad6a8f8e50471b35217c8629"
BEIT3_SENTENCEPIECE_SHA256 = "6f5e2fefcf793761a76a6bfb8ad35489f9c203b25557673284b6d032f41043f4"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".staging")
    staging.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(staging, path)


def validate_data(data_root: Path, beit3_dir: Path) -> dict[str, Any]:
    return build_data_gate_report(data_root, beit3_dir)


def validate_encoder_compatibility(model_root: Path, query_manifest_path: Path) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    try:
        query = _read_json(query_manifest_path)
        models = query.get("models", {})
        checks["query_model_manifest_schema"] = query.get("schema_version") == "query.models.v2"
        checks["siglip2_checkpoint"] = (
            models.get("siglip2", {}).get("model_id") == SIGLIP_ID
            and models.get("siglip2", {}).get("revision") == SIGLIP_REVISION
            and models.get("siglip2", {}).get("tokenizer_config") == "max_tokens=64;normalization=l2"
            and bool(models.get("siglip2", {}).get("files"))
        )
        checks["metaclip2_checkpoint"] = (
            models.get("metaclip2", {}).get("model_id") == METACLIP_ID
            and models.get("metaclip2", {}).get("revision") == METACLIP_REVISION
            and models.get("metaclip2", {}).get("tokenizer_config") == "max_tokens=77;normalization=l2"
            and bool(models.get("metaclip2", {}).get("files"))
        )
        details["query_models"] = models
    except (OSError, ValueError, TypeError, KeyError) as error:
        checks["query_model_manifest"] = False
        details["query_model_manifest_error"] = str(error)

    setup_manifest_path = model_root / "manifest.json"
    try:
        setup = _read_json(setup_manifest_path)
        hashes = setup.get("sha256", {})
        checks["beit3_setup_manifest"] = (
            setup.get("schema_version") == "branch1.models.v2"
            and setup.get("unilm_revision") == UNILM_REVISION
            and hashes.get("unilm.zip") == UNILM_SOURCE_SHA256
            and hashes.get(BEIT3_CHECKPOINT) == BEIT3_CHECKPOINT_SHA256
            and hashes.get("beit3.spm") == BEIT3_SENTENCEPIECE_SHA256
            and bool(setup.get("assets"))
        )
        details["beit3_setup"] = setup
    except (OSError, ValueError, TypeError, KeyError) as error:
        checks["beit3_setup_manifest"] = False
        details["beit3_setup_manifest_error"] = str(error)

    probe_roles = [
        "a person near a red car",
        "a cyclist crosses a bridge",
        "an indoor market scene",
        "a sunny city street",
        "vehicle bicycle pedestrian",
        "red car bridge market",
    ]
    probe_roles_vi = [
        "một người gần ô tô đỏ",
        "một người đi xe đạp qua cầu",
        "khung cảnh chợ trong nhà",
        "đường phố thành phố nắng",
        "xe người đi xe đạp người đi bộ",
        "ô tô đỏ cầu khu chợ",
    ]
    probe_texts_by_model = {
        "siglip2": probe_roles_vi + probe_roles,
        "metaclip2": probe_roles_vi + probe_roles,
        "beit3": probe_roles,
    }
    expected_probe_rows = {"siglip2": 12, "metaclip2": 12, "beit3": 6}
    probe_results: dict[str, Any] = {}
    try:
        encoder = SequentialBranch1Encoders(model_root)
        for model_name, dimension in (("siglip2", 768), ("metaclip2", 1024), ("beit3", 768)):
            probe_texts = probe_texts_by_model[model_name]
            vectors, diagnostics = encoder.encode(model_name, probe_texts)
            vectors = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(vectors, axis=1) if vectors.ndim == 2 else np.asarray([])
            expected_rows = expected_probe_rows[model_name]
            passed = (
                vectors.shape == (expected_rows, dimension)
                and bool(np.isfinite(vectors).all())
                and bool(np.all(norms > 0.999))
                and bool(np.all(norms < 1.001))
                and len(diagnostics) == expected_rows
                and all("token_count" in item and "truncated" in item for item in diagnostics)
            )
            checks[f"{model_name}_runtime_probe"] = passed
            probe_results[model_name] = {
                "shape": list(vectors.shape),
                "finite": bool(np.isfinite(vectors).all()),
                "min_norm": float(norms.min()) if norms.size else None,
                "max_norm": float(norms.max()) if norms.size else None,
                "tokenizer_diagnostics": diagnostics,
            }
            encoder.unload()
    except Exception as error:
        checks["runtime_probe"] = False
        details["runtime_probe_error"] = f"{type(error).__name__}: {error}"
    details["runtime_probe"] = probe_results
    report = {
        "schema_version": "branch1.encoder-compatibility.v2",
        "passed": all(checks.values()),
        "checks": checks,
        "details": details,
        "text_encoder_contract": {
            "siglip2": {"model_id": SIGLIP_ID, "revision": SIGLIP_REVISION, "languages": ["vi", "en"], "dimension": 768},
            "metaclip2": {"model_id": METACLIP_ID, "revision": METACLIP_REVISION, "languages": ["vi", "en"], "dimension": 1024},
            "beit3": {"checkpoint": BEIT3_CHECKPOINT, "source_revision": UNILM_REVISION, "languages": ["en"], "dimension": 768},
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("AIC_DATA_ROOT", "/data")))
    parser.add_argument("--state-root", type=Path, default=Path(os.environ.get("AIC_STATE_ROOT", "/state")))
    parser.add_argument("--model-root", type=Path, default=Path(os.environ.get("AIC_BRANCH1_MODEL_ROOT", "/models/branch1")))
    parser.add_argument("--query-manifest", type=Path, default=Path(os.environ.get("AIC_QUERY_MODEL_MANIFEST", "/models/query_models.json")))
    args = parser.parse_args()
    beit3_dir = args.data_root / "visual_embeddings" / "beit3"
    data_gate_path = args.state_root / "branch1_data_gate.json"
    compatibility_path = args.state_root / "branch1_encoder_compatibility.json"
    # Invalidate any previous gate before touching the artifacts.  If the
    # process is interrupted halfway through validation, health must not keep
    # serving an old ``passed=true`` report for potentially changed data.
    _write_atomic(
        data_gate_path,
        {"schema_version": DATA_GATE_SCHEMA_VERSION, "passed": False, "status": "validating"},
    )
    _write_atomic(
        compatibility_path,
        {
            "schema_version": "branch1.encoder-compatibility.v2",
            "passed": False,
            "status": "validating",
        },
    )
    try:
        data_report = validate_data(args.data_root, beit3_dir)
        compatibility = validate_encoder_compatibility(args.model_root, args.query_manifest)
    except Exception as error:
        _write_atomic(
            data_gate_path,
            {
                "schema_version": DATA_GATE_SCHEMA_VERSION,
                "passed": False,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        _write_atomic(
            compatibility_path,
            {
            "schema_version": "branch1.encoder-compatibility.v2",
                "passed": False,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    _write_atomic(data_gate_path, data_report)
    _write_atomic(compatibility_path, compatibility)
    print(json.dumps({"data_gate": data_report, "encoder_compatibility": compatibility}, ensure_ascii=False, indent=2))
    if not compatibility.get("passed"):
        raise RuntimeError(
            "Branch-1 encoder compatibility gate failed; inspect "
            f"{args.state_root / 'branch1_encoder_compatibility.json'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
