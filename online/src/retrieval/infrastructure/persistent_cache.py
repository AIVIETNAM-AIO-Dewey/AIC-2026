"""Persistent LRU cache for small query-embedding batches."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


class PersistentQueryEmbeddingCache:
    def __init__(self, path: Path, max_entries: int = 128) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS branch1_embeddings (
                cache_key TEXT PRIMARY KEY, model_name TEXT NOT NULL,
                dimension INTEGER NOT NULL, rows INTEGER NOT NULL,
                vector_blob BLOB NOT NULL, diagnostics_json TEXT NOT NULL,
                accessed_at REAL NOT NULL
            )"""
        )
        self._db.commit()

    @staticmethod
    def key(
        model_name: str,
        revision: str,
        texts: list[str],
        tokenizer_config: str | None = None,
        stream_contract: list[dict[str, Any]] | None = None,
        device: str = "cpu",
    ) -> str:
        canonical = json.dumps(
            {
                "model": model_name,
                "revision": revision,
                "tokenizer_config": tokenizer_config or "default",
                "device": str(device or "cpu"),
                "streams": stream_contract or [{"text": text} for text in texts],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> tuple[np.ndarray, list[dict[str, Any]]] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT dimension, rows, vector_blob, diagnostics_json FROM branch1_embeddings WHERE cache_key=?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            self._db.execute(
                "UPDATE branch1_embeddings SET accessed_at=? WHERE cache_key=?",
                (time.time(), cache_key),
            )
            self._db.commit()
        dimension, rows, blob, diagnostics_json = row
        matrix = np.frombuffer(blob, dtype=np.float32).reshape(int(rows), int(dimension)).copy()
        return matrix, json.loads(diagnostics_json)

    def put(
        self, cache_key: str, model_name: str, matrix: np.ndarray, diagnostics: list[dict[str, Any]]
    ) -> None:
        value = np.asarray(matrix, dtype=np.float32)
        with self._lock:
            self._db.execute(
                """INSERT OR REPLACE INTO branch1_embeddings
                (cache_key, model_name, dimension, rows, vector_blob, diagnostics_json, accessed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    cache_key,
                    model_name,
                    value.shape[1],
                    value.shape[0],
                    value.tobytes(),
                    json.dumps(diagnostics, ensure_ascii=False),
                    time.time(),
                ),
            )
            self._db.execute(
                """DELETE FROM branch1_embeddings WHERE cache_key IN (
                    SELECT cache_key FROM branch1_embeddings
                    ORDER BY accessed_at DESC LIMIT -1 OFFSET ?
                )""",
                (self.max_entries,),
            )
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()


__all__ = ["PersistentQueryEmbeddingCache"]
