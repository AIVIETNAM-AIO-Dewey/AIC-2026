"""Focused API contracts for the additive image/video workbench features."""

from __future__ import annotations

import io
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image

import online.server as server
from online.src.contracts.query import ParsedQuery
from online.src.retrieval.embeddings import ModelRegistry
from online.src.retrieval.query_parser import QueryParser


def _frame_result(video_id: str = "L00_V001") -> dict:
    return {
        "video_id": video_id,
        "keyframe_n": 1,
        "frame_idx": 25,
        "pts_time_s": 1.0,
        "image_relpath": f"keyframes/{video_id}/00000025.jpg",
        "submission_string": f"{video_id}, 25",
        "dam_summary": "",
        "asr_transcript": "",
        "ocr_text": "",
        "rank": 1,
        "global_idx": 0,
        "score": 1.0,
        "score_type": "cosine",
    }


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (12, 8), (10, 20, 30)).save(output, format="PNG")
    return output.getvalue()


class FakeRegistry:
    def __init__(self) -> None:
        self.received_mode: str | None = None

    def embed_siglip_image(self, image: Image.Image) -> np.ndarray:
        self.received_mode = image.mode
        vector = np.zeros(768, dtype=np.float32)
        vector[0] = 1.0
        return vector


class FakeSearcher:
    def __init__(self) -> None:
        self.timeline_video_id: str | None = None
        self.scoped_video_id: str | None = None
        self.frames = [
            {
                "video_id": "L00_V001",
                "keyframe_n": keyframe_n,
                "frame_idx": keyframe_n * 25,
                "pts_time_s": float(keyframe_n),
                "image_relpath": f"keyframes/L00_V001/{keyframe_n * 25:08d}.jpg",
            }
            for keyframe_n in range(1, 6)
        ]
        self.frame_lookup = {
            (frame["video_id"], frame["frame_idx"]): frame for frame in self.frames
        }

    def get_video_timeline(self, video_id: str) -> dict | None:
        self.timeline_video_id = video_id
        if video_id != "L00_V001":
            return None
        return {
            "video_id": video_id,
            "fps": 25.0,
            "keyframe_count": 1,
            "keyframes": [
                {
                    "keyframe_n": 1,
                    "frame_idx": 25,
                    "pts_time_s": 1.0,
                    "image_relpath": "keyframes/L00_V001/00000025.jpg",
                }
            ],
        }

    def get_total_frame_count(self) -> int:
        return 247_956

    def get_video_keyframe_list(self, video_id: str) -> list[dict]:
        return [dict(frame) for frame in self.frames if frame["video_id"] == video_id]

    def get_video_frame_count(self, video_id: str) -> int:
        return 1 if video_id == "L00_V001" else 0

    def search_visual(self, vector: np.ndarray, top_k: int) -> list[dict]:
        assert vector.shape == (768,)
        assert top_k == 7
        return [_frame_result()]

    def search_visual_in_video(
        self,
        vector: np.ndarray,
        video_id: str,
        top_k: int,
    ) -> list[dict]:
        self.scoped_video_id = video_id
        result = _frame_result(video_id)
        result.update({"scope": "video", "scope_video_id": video_id})
        return [result]


class FakeModalitySearch:
    def __init__(self) -> None:
        self.temporal_kwargs: dict | None = None

    def search_temporal_intersection(self, **kwargs) -> dict:
        self.temporal_kwargs = kwargs
        return {
            "operation": "ordered_siglip_intersection",
            "sequences": [],
            "reserve_sequences": [],
            "paths_per_video": kwargs["paths_per_video"],
            "sequence_reservoir_size": kwargs["sequence_reservoir_size"],
        }


class WorkbenchApiTests(unittest.TestCase):
    def setUp(self) -> None:
        # TestClient only runs FastAPI lifespan hooks when used as a context
        # manager. Keeping it un-entered ensures these API tests never load the
        # real models or dataset.
        self.client = TestClient(server.app)
        self.registry = FakeRegistry()
        self.searcher = FakeSearcher()
        self.modality_search = FakeModalitySearch()
        self.registry_patch = patch.object(server, "_registry", self.registry)
        self.searcher_patch = patch.object(server, "_searcher", self.searcher)
        self.modality_patch = patch.object(server, "_modality_search", self.modality_search)
        self.registry_patch.start()
        self.searcher_patch.start()
        self.modality_patch.start()

    def tearDown(self) -> None:
        self.modality_patch.stop()
        self.searcher_patch.stop()
        self.registry_patch.stop()
        self.client.close()

    def test_timeline_normalizes_id_and_returns_minimal_contract(self) -> None:
        response = self.client.get("/api/video/l00-v001/timeline")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(self.searcher.timeline_video_id, "L00_V001")
        self.assertEqual(payload["video_id"], "L00_V001")
        self.assertEqual(payload["fps"], 25.0)
        self.assertEqual(payload["keyframe_count"], 1)
        self.assertEqual(
            set(payload["keyframes"][0]),
            {"keyframe_n", "frame_idx", "pts_time_s", "image_relpath"},
        )

    def test_config_advertises_additive_workbench_capabilities(self) -> None:
        response = self.client.get("/api/config")
        self.assertEqual(response.status_code, 200)
        config = response.json()
        capabilities = config["capabilities"]
        self.assertTrue(capabilities["video_timeline"])
        self.assertTrue(capabilities["image_search"])
        self.assertTrue(capabilities["qwen_fallback_control"])
        self.assertTrue(capabilities["submission_prepare"])
        self.assertTrue(capabilities["ordered_search_expansion"])
        self.assertIn("direct", capabilities["parser_modes"])
        submission = config["submission"]
        self.assertFalse(submission["csv_has_header"])
        self.assertEqual(submission["vqa_answer_max_characters"], 100)
        self.assertEqual(submission["trake_max_sequence_rows"], 100)
        self.assertEqual(submission["trake_max_events_per_submission_row"], 100)
        self.assertEqual(submission["minimum_valid_rows"], 1)
        self.assertTrue(submission["trake_canonical_neighbor_fill"])

    def test_submission_prepare_canonicalizes_and_completes_verified_rows(self) -> None:
        response = self.client.post(
            "/api/submission/prepare",
            json={
                "task_type": "KIS",
                "query_id": "query-1",
                "target_rows": 3,
                "manual_selections": [{"video_id": "l00-v001", "frame_idx": 50}],
                "candidate_reservoir": [{"video_id": "L00_V001", "frame_idx": 100}],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["complete"])
        self.assertTrue(payload["valid_for_download"])
        self.assertTrue(payload["official_csv"]["valid"])
        self.assertTrue(payload["server_verified"])
        self.assertEqual(payload["row_count"], 3)
        self.assertEqual([row["frame_idx"] for row in payload["rows"][:2]], [50, 100])
        self.assertEqual(payload["rows"][0]["video_id"], "L00_V001")
        self.assertTrue(payload["rows"][0]["manual"])
        self.assertEqual(payload["rows"][2]["selection_origin"], "canonical_neighbor")
        self.assertEqual(payload["official_csv"]["content"].split("\r\n")[0], "L00_V001,50")

    def test_submission_prepare_reports_vqa_answer_error_without_fabricating_rows(self) -> None:
        response = self.client.post(
            "/api/submission/prepare",
            json={
                "task_type": "VQA",
                "query_id": "query-2",
                "target_rows": 1,
                "manual_selections": [{"video_id": "L00_V001", "frame_idx": 25}],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertFalse(payload["ok"])
        self.assertTrue(payload["server_verified"])
        self.assertEqual(payload["rows"], [])
        self.assertIn("human-provided VQA answer", payload["errors"][0])

    def test_submission_prepare_rejects_vqa_answer_over_official_limit(self) -> None:
        response = self.client.post(
            "/api/submission/prepare",
            json={
                "task_type": "VQA",
                "query_id": "query-2",
                "target_rows": 1,
                "manual_selections": [{"video_id": "L00_V001", "frame_idx": 25}],
                "vqa_answer": "x" * 101,
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_submission_prepare_fills_multiple_official_trake_rows(self) -> None:
        response = self.client.post(
            "/api/submission/prepare",
            json={
                "task_type": "TRAKE",
                "query_id": "query-3",
                "target_rows": 5,
                "event_count": 2,
                "manual_sequences": [
                    [
                        {"video_id": "L00_V001", "frame_idx": 25},
                        {"video_id": "L00_V001", "frame_idx": 100},
                    ]
                ],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["complete"])
        self.assertEqual(payload["row_count"], 5)
        self.assertEqual(payload["rows"][0]["matched_frames"], [25, 100])
        self.assertEqual(payload["rows"][0]["csv_line"], "L00_V001,25,100")
        identities = {
            (row["video_id"], tuple(row["matched_frames"])) for row in payload["rows"]
        }
        self.assertEqual(len(identities), 5)
        self.assertTrue(
            all(row["matched_frames"][0] < row["matched_frames"][1] for row in payload["rows"])
        )

    def test_temporal_endpoint_forwards_opt_in_expansion_without_changing_defaults(self) -> None:
        response = self.client.post(
            "/api/search/temporal-intersection",
            json={
                "events": [
                    {"order": 1, "description": "first", "global_scene_en": "first scene"},
                    {"order": 2, "description": "second", "global_scene_en": "second scene"},
                ],
                "top_k_per_event": 500,
                "top_k_sequences": 50,
                "paths_per_video": 3,
                "sequence_reservoir_size": 150,
                "path_beam_width": 256,
                "path_diversity_min_events": 2,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["paths_per_video"], 3)
        forwarded = self.modality_search.temporal_kwargs
        self.assertIsNotNone(forwarded)
        self.assertEqual(forwarded["top_k_per_event"], 500)
        self.assertEqual(forwarded["top_k_sequences"], 50)
        self.assertEqual(forwarded["sequence_reservoir_size"], 150)
        self.assertEqual(forwarded["path_beam_width"], 256)
        self.assertEqual(forwarded["path_diversity_min_events"], 2)

    def test_timeline_rejects_path_like_video_id(self) -> None:
        response = self.client.get("/api/video/L00_V001.bad/timeline")
        self.assertEqual(response.status_code, 400)

    def test_image_search_returns_one_explicit_no_fusion_siglip_pool(self) -> None:
        response = self.client.post(
            "/api/search/image",
            data={"top_k": "7"},
            files={"file": ("query.png", _png_bytes(), "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(self.registry.received_mode, "RGB")
        self.assertEqual(payload["operation"], "image_query")
        self.assertEqual(payload["scope"], "global")
        self.assertEqual(payload["evaluated_frames"], 247_956)
        self.assertFalse(payload["fusion_applied"])
        self.assertFalse(payload["reranking_applied"])
        pool = payload["modality_result"]
        self.assertEqual(pool["modality"], "siglip")
        self.assertEqual(pool["query_source"], "uploaded_image")
        self.assertEqual(pool["score_type"], "cosine")
        self.assertFalse(pool["provenance"]["fusion_applied"])
        self.assertEqual(pool["results"][0]["frame_idx"], 25)

    def test_image_search_can_be_scoped_to_a_normalized_video_id(self) -> None:
        response = self.client.post(
            "/api/search/image",
            data={"top_k": "3", "video_id": "l00-v001"},
            files={"file": ("query.png", _png_bytes(), "image/png")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["scope"], "video")
        self.assertEqual(payload["video_id"], "L00_V001")
        self.assertEqual(self.searcher.scoped_video_id, "L00_V001")

    def test_image_search_rejects_non_image_upload_before_inference(self) -> None:
        response = self.client.post(
            "/api/search/image",
            files={"file": ("query.txt", b"not an image", "text/plain")},
        )
        self.assertEqual(response.status_code, 415)
        self.assertIsNone(self.registry.received_mode)

    def test_image_search_rejects_payload_above_hard_limit(self) -> None:
        response = self.client.post(
            "/api/search/image",
            files={
                "file": (
                    "query.png",
                    b"x" * (server.MAX_IMAGE_UPLOAD_BYTES + 1),
                    "image/png",
                )
            },
        )
        self.assertEqual(response.status_code, 413)


class ParserFallbackControlTests(unittest.TestCase):
    @staticmethod
    def _parser_without_initialization() -> QueryParser:
        parser = QueryParser.__new__(QueryParser)
        parser.gemini_model_id = "fake-gemini"
        parser.qwen_model_id = "fake-qwen"
        parser.ollama_url = "http://invalid"
        parser._gemini_client = object()
        return parser

    def test_disabling_qwen_uses_rule_parser_after_gemini_failure(self) -> None:
        parser = self._parser_without_initialization()
        with (
            patch.object(parser, "_parse_with_gemini", side_effect=RuntimeError("offline")),
            patch.object(parser, "_parse_with_qwen") as qwen,
        ):
            parsed = parser.parse(
                "A cyclist crosses the finish line",
                task_type="KIS",
                engine="gemini",
                allow_qwen_fallback=False,
            )
        qwen.assert_not_called()
        self.assertEqual(parsed.global_scene_en, "A cyclist crosses the finish line")

    def test_default_keeps_existing_gemini_to_qwen_fallback(self) -> None:
        parser = self._parser_without_initialization()
        expected = ParsedQuery(
            task_type="KIS",
            original_query="query",
            global_scene_en="qwen result",
        )
        with (
            patch.object(parser, "_parse_with_gemini", side_effect=RuntimeError("offline")),
            patch.object(parser, "_parse_with_qwen", return_value=expected) as qwen,
        ):
            parsed = parser.parse("query", task_type="KIS", engine="gemini")
        qwen.assert_called_once()
        self.assertIs(parsed, expected)


class SiglipImageEmbeddingContractTests(unittest.TestCase):
    def test_image_tower_returns_float32_unit_vector_without_loading_weights(self) -> None:
        class FakeProcessor:
            def __call__(self, *, images, return_tensors):
                self.image_modes = [image.mode for image in images]
                self.return_tensors = return_tensors
                return {"pixel_values": torch.zeros((1, 3, 2, 2), dtype=torch.float32)}

        class FakeModel:
            def get_image_features(self, **inputs):
                self.inputs = inputs
                values = torch.zeros((1, 768), dtype=torch.float32)
                values[0, 0] = 3.0
                values[0, 1] = 4.0
                return values

        registry = ModelRegistry.__new__(ModelRegistry)
        registry.device = "cpu"
        registry._siglip_inference_lock = threading.RLock()
        registry._siglip_image_processor = FakeProcessor()
        registry._siglip_model = FakeModel()

        vector = registry.embed_siglip_image(Image.new("RGB", (2, 2)))

        self.assertEqual(vector.shape, (768,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)
        self.assertAlmostEqual(float(vector[0]), 0.6, places=6)
        self.assertAlmostEqual(float(vector[1]), 0.8, places=6)


if __name__ == "__main__":
    unittest.main()
