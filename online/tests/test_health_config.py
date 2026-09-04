"""Fail-closed API health and capability tests."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import online.cpu_server as server


class _BrokenQdrant:
    def collection(self, _name: str):
        raise ConnectionError("qdrant is offline")


class _NotReadyBranch2:
    def health(self):
        raise RuntimeError("branch-2 dependencies are unavailable")


class HealthConfigTests(unittest.TestCase):
    def test_media_info_root_prefers_canonical_hyphenated_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "media-info"
            legacy = root / "media_info"
            canonical.mkdir()
            legacy.mkdir()
            self.assertEqual(server._resolve_media_info_root(root), canonical)

    def test_media_info_root_supports_legacy_underscore_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "media_info"
            legacy.mkdir()
            self.assertEqual(server._resolve_media_info_root(root), legacy)

    def test_qdrant_disconnect_is_not_ready(self) -> None:
        with patch.object(server, "qdrant_client", _BrokenQdrant()):
            result = server._qdrant_health()
        self.assertFalse(result["ready"])

    def test_concurrent_health_cache_misses_share_one_build(self) -> None:
        signature = (("fixture", 1, 1),)
        payload = {"status": "ready", "ready": True}

        async def delayed_build(_signature):
            await asyncio.sleep(0.01)
            return payload

        builder = AsyncMock(side_effect=delayed_build)

        async def request_together():
            return await asyncio.gather(
                server._dependency_health(),
                server._dependency_health(),
            )

        with (
            patch.object(server, "_health_cache_signature", return_value=signature),
            patch.object(server, "_health_cache", None),
            patch.object(server, "_build_dependency_health", builder),
        ):
            first, second = asyncio.run(request_together())

        self.assertIs(first, payload)
        self.assertIs(second, payload)
        builder.assert_awaited_once_with(signature)

    def test_config_disables_capabilities_when_dependencies_fail(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.multiple(
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
            ),
            patch.object(
                server,
                "branch1_health",
                return_value={"ready": False, "models": {}},
            ),
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
