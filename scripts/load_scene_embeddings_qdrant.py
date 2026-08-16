#!/usr/bin/env python3
"""Load published scene-embedding shards into a local Qdrant collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _common import add_common_arguments, read_config, runtime_roots  # noqa: E402

from aic2026.common import iter_jsonl  # noqa: E402
from aic2026.contracts import SceneEmbeddingRecord  # noqa: E402
from aic2026.scene_embedding.qdrant_index import (  # noqa: E402
    ensure_collection,
    load_shard,
    shard_paths,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--embeddings-dir", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the collection first. Without it, shards upsert onto stable ids.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = read_config(args.config)
    collection = args.collection or str(config.get("qdrant_collection", "aic26_scene_siglip2"))

    if args.embeddings_dir:
        embeddings_dir = args.embeddings_dir.expanduser().resolve()
    else:
        roots = runtime_roots(args, config, required=("output_root",))
        embeddings_dir = roots["output_root"] / "scene_embeddings"
    shards = shard_paths(embeddings_dir)
    if not shards:
        raise SystemExit(f"No embedding shards found in {embeddings_dir}")

    try:
        from qdrant_client import QdrantClient
    except ImportError as error:
        raise SystemExit(
            "qdrant-client is required; install requirements/runtime-base.txt"
        ) from error

    # Read the dimension off the data rather than trusting a config value.
    first = SceneEmbeddingRecord.model_validate(next(iter(iter_jsonl(shards[0]))))
    client = QdrantClient(url=args.url)
    ensure_collection(client, collection, first.embedding_dim, recreate=args.recreate)

    total = 0
    for shard in shards:
        loaded = load_shard(client, collection, shard, batch_size=args.batch_size)
        total += loaded
        print(json.dumps({"shard": shard.stem, "points": loaded}))

    counted = client.count(collection_name=collection, exact=True).count
    print(
        json.dumps(
            {
                "status": "completed",
                "collection": collection,
                "shards": len(shards),
                "points_sent": total,
                "points_in_collection": counted,
                "dim": first.embedding_dim,
            }
        )
    )
    return 0 if counted == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
