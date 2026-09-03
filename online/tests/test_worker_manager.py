"""CPU worker lifecycle contracts that do not load model weights."""

from __future__ import annotations

import os
import time
import unittest

import numpy as np

from online.src.retrieval.encoders.worker_manager import EncoderWorkerManager, _worker_identity


def echo_worker(connection, request, idle_timeout_seconds: float) -> None:
    """Pickleable spawn-worker fixture; it never loads a real model."""
    first = True
    identity = _worker_identity(request)
    while True:
        connection.send(
            {
                "ok": True,
                "vectors": np.ones((6, 2), dtype=np.float32),
                "diagnostics": [],
                "timing": {
                    "model_loading_ms": 1.0 if first else 0.0,
                    "inference_ms": 0.0,
                    "worker_reused": not first,
                    "worker_pid": os.getpid(),
                    "worker_load_count": 1,
                },
                "peak_rss_bytes": 0,
            }
        )
        first = False
        if not connection.poll(idle_timeout_seconds):
            break
        pending = connection.recv()
        if _worker_identity(pending) != identity:
            raise RuntimeError("identity switch was sent to an existing worker")


def worker_request(model_name: str = "siglip2") -> dict[str, object]:
    return {
        "kind": "branch1_text",
        "model_name": model_name,
        "model_root": "/models",
        "model_revision": "revision",
        "tokenizer_config": "max_tokens=64",
        "texts": ["x"] * 6,
    }


class WorkerManagerContractTests(unittest.TestCase):
    def test_siglip_text_and_image_share_checkpoint_identity(self) -> None:
        base = {
            "siglip_id": "siglip",
            "siglip_revision": "revision",
            "bge_id": "bge",
            "bge_revision": "bge-revision",
        }
        text = {**base, "kind": "siglip_text", "text": "hello"}
        image = {**base, "kind": "siglip_image", "image_bytes": b""}
        self.assertEqual(_worker_identity(text), _worker_identity(image))

    def test_identity_changes_when_revision_or_tokenizer_contract_changes(self) -> None:
        request = {
            "kind": "branch1_text",
            "model_name": "beit3",
            "model_root": "/models",
            "model_revision": "checkpoint-a",
            "tokenizer_config": "max_tokens=64;output=language_head",
        }
        self.assertNotEqual(
            _worker_identity(request),
            _worker_identity({**request, "model_revision": "checkpoint-b"}),
        )
        self.assertNotEqual(
            _worker_identity(request),
            _worker_identity({**request, "tokenizer_config": "max_tokens=32"}),
        )

    def test_different_models_do_not_share_worker_identity(self) -> None:
        branch1 = {"kind": "branch1_text", "model_name": "beit3", "model_root": "/models"}
        bge = {
            "kind": "bge_text",
            "bge_id": "bge",
            "bge_revision": "revision",
        }
        self.assertNotEqual(_worker_identity(branch1), _worker_identity(bge))

    def test_idle_timeout_is_bounded_and_positive(self) -> None:
        manager = EncoderWorkerManager(idle_timeout_seconds=30.0)
        self.assertEqual(manager.idle_timeout_seconds, 30.0)
        with self.assertRaises(ValueError):
            EncoderWorkerManager(idle_timeout_seconds=0.0)
        with self.assertRaises(ValueError):
            EncoderWorkerManager(idle_timeout_seconds=30.1)

    def test_spawn_reuse_switch_and_idle_exit_are_observable(self) -> None:
        manager = EncoderWorkerManager(
            timeout_seconds=5.0,
            idle_timeout_seconds=0.1,
            worker_target=echo_worker,
        )
        try:
            manager.execute(worker_request())
            first_pid = manager.last_worker_pid
            self.assertTrue(manager.last_worker_spawned)
            self.assertEqual(manager.last_worker_load_count, 1)

            manager.execute(worker_request())
            self.assertTrue(manager.last_worker_reused)
            self.assertEqual(manager.last_worker_pid, first_pid)
            self.assertEqual(manager.last_worker_load_count, 1)

            manager.execute(worker_request("metaclip2"))
            self.assertTrue(manager.last_worker_spawned)
            self.assertFalse(manager.last_worker_reused)

            time.sleep(0.2)
            manager.execute(worker_request("metaclip2"))
            self.assertTrue(manager.last_worker_spawned)
        finally:
            manager.close_active()


if __name__ == "__main__":
    unittest.main()
