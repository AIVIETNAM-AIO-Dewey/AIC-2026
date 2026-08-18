"""Ingest completed offline artifacts into versioned Qdrant collections."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..retrieval.e5 import E5OnnxEncoder
from .artifacts import discover_artifacts, ingest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--e5-model-path", type=Path)
    parser.add_argument(
        "--lexical-only",
        action="store_true",
        help="Ingest text collections with sparse lexical/trigram vectors and no E5 model.",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.all:
        parser.error("--all is required in v1 to prevent accidental partial index activation")
    from qdrant_client import QdrantClient

    artifacts = discover_artifacts(args.artifact_root)
    if not artifacts:
        parser.error("No completed artifacts discovered")
    if args.check_only:
        from .artifacts import validate_artifact

        report = [validate_artifact(item) for item in artifacts]
        collections = sorted({item.source.collection for item in report})
        print({"artifacts": len(report), "collections": collections})
        return 0
    if args.e5_model_path is None and not args.lexical_only:
        parser.error("--e5-model-path or --lexical-only is required unless --check-only is used")
    encoder = (
        E5OnnxEncoder.from_pretrained(model_path=args.e5_model_path)
        if args.e5_model_path is not None
        else None
    )
    print(
        ingest(
            QdrantClient(url=args.qdrant_url),
            artifacts,
            dense_encoder=encoder,
            activate=args.activate,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
