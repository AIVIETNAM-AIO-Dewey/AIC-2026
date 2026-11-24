"""Native accelerator routing contracts that never load model weights."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from online.src.retrieval.encoders import device as runtime_device
from online.src.retrieval.infrastructure.persistent_cache import (
    PersistentQueryEmbeddingCache,
)


class DeviceRoutingTests(unittest.TestCase):
    def test_auto_prefers_mps_when_available(self) -> None:
        with patch.object(runtime_device, "mps_available", return_value=True):
            self.assertEqual(runtime_device.resolve_device("auto"), "mps")

    def test_explicit_mps_can_fail_closed(self) -> None:
        with (
            patch.object(runtime_device, "mps_available", return_value=False),
            self.assertRaises(RuntimeError),
        ):
            runtime_device.resolve_device("mps", allow_fallback=False)

    def test_explicit_mps_falls_back_when_allowed(self) -> None:
        with patch.object(runtime_device, "mps_available", return_value=False):
            self.assertEqual(
                runtime_device.resolve_device("mps", allow_fallback=True),
                "cpu",
            )

    def test_embedding_cache_is_device_namespaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = PersistentQueryEmbeddingCache(Path(temporary) / "cache.sqlite3")
            try:
                cpu = cache.key("model", "revision", ["query"], device="cpu")
                mps = cache.key("model", "revision", ["query"], device="mps")
            finally:
                cache.close()
        self.assertNotEqual(cpu, mps)

    def test_transformers_dtype_keyword_is_version_compatible(self) -> None:
        self.assertEqual(
            runtime_device.transformers_dtype_kwargs(
                "float32",
                version_value="4.48.3",
            ),
            {"torch_dtype": "float32"},
        )
        self.assertEqual(
            runtime_device.transformers_dtype_kwargs(
                "float32",
                version_value="4.57.1",
            ),
            {"dtype": "float32"},
        )


if __name__ == "__main__":
    unittest.main()
