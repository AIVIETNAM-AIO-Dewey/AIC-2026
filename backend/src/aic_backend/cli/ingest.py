"""Ingest completed offline artifacts into versioned Qdrant collections."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..infrastructure.encoders.e5 import E5OnnxEncoder
from ..infrastructure.qdrant.ingest import discover_artifacts, ingest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--e5-model-path", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.all:
        parser.error("--all is required in v1 to prevent accidental partial index activation")
    from qdrant_client import QdrantClient

    artifacts = discover_artifacts(args.artifact_root)
    if not artifacts:
        parser.error("No completed artifacts discovered")
    encoder = E5OnnxEncoder.from_pretrained(model_path=args.e5_model_path)
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
