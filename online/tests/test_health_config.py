"""Fail-closed API health and capability tests."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import online.cpu_server as server


class _BrokenQdrant:
    def collection(self, _name: str):
        raise ConnectionError("qdrant is offline")


class _NotReadyBranch2:
    def health(self):
        raise RuntimeError("branch-2 dependencies are unavailable")


class HealthConfigTests(unittest.TestCase):
    def test_qdrant_disconnect_is_not_ready(self) -> None:
        with patch.object(server, "qdrant_client", _BrokenQdrant()):
            result = server._qdrant_health()
        self.assertFalse(result["ready"])

    def test_config_disables_capabilities_when_dependencies_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.multiple(
                server,
                searcher=object(),
                qdrant_client=_BrokenQdrant(),
                branch1_qdrant=object(),
                branch1_encoders=object(),
                branch2_searcher=_NotReadyBranch2(),
                metadata_store=None,
                ocr_index=None,
                asr_index=None,
                encoder_workers=None,
                STATE_ROOT=Path(directory),
                _health_cache=None,
            ), patch.object(
                server,
                "branch1_health",
                return_value={"ready": False, "models": {}},
            ):
                health = asyncio.run(server._dependency_health(force=True))
                config = asyncio.run(server.config())
        self.assertNotEqual(health["status"], "ready")
        self.assertFalse(health["production_ready"])
        self.assertFalse(config["capabilities"]["image_search"])
        self.assertFalse(config["capabilities"]["branch1_three_model"])
        self.assertFalse(config["capabilities"]["branch2_dam_hybrid"])


if __name__ == "__main__":
    unittest.main()
