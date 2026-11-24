"""Executed HTTP contracts for the canonical CPU workbench routes."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import online.cpu_server as server


def query_bundle() -> dict[str, object]:
    roles = ("original", "entity", "action", "context", "synonym", "keyword")
    return {
        "schema_version": "branch1.query.v1",
        "queries": [{"role": role, "vi": f"vi {role}", "en": f"en {role}"} for role in roles],
    }


class CanonicalHttpContractTests(unittest.TestCase):
    def setUp(self) -> None:
        # Do not enter TestClient as a context manager: that would run the
        # real lifespan and attempt to open Qdrant/models during unit tests.
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        self.client.close()

    def test_health_and_config_are_real_http_contracts(self) -> None:
        dependency = {
            "status": "degraded",
            "production_ready": False,
            "components": {
                "branch1": {"ready": False},
                "branch2": {"ready": False, "components": {}},
                "image_search": {"ready": False},
                "siglip_text": {"ready": False},
                "metadata": {"ready": False},
            },
        }
        with patch.object(server, "_dependency_health", AsyncMock(return_value=dependency)):
            health = self.client.get("/api/health")
            config = self.client.get("/api/config")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "degraded")
        self.assertEqual(config.status_code, 200)
        self.assertFalse(config.json()["capabilities"]["branch1_three_model"])
        self.assertFalse(config.json()["capabilities"]["branch2_dam_hybrid"])
        self.assertTrue(config.json()["capabilities"]["kis_query_planning"])

    def test_compact_health_keeps_ui_gates_and_omits_heavy_diagnostics(self) -> None:
        dependency = {
            "status": "ready",
            "ready": True,
            "device": "mps",
            "large_diagnostic": ["unused"] * 10_000,
            "components": {
                "branch1": {
                    "status": "ready",
                    "ready": True,
                    "large_diagnostic": ["unused"] * 10_000,
                }
            },
        }
        with patch.object(server, "_dependency_health", AsyncMock(return_value=dependency)):
            response = self.client.get("/api/health?compact=true")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["device"], "mps")
        self.assertTrue(payload["components"]["branch1"]["ready"])
        self.assertNotIn("large_diagnostic", payload)
        self.assertNotIn("large_diagnostic", payload["components"]["branch1"])

    def test_parse_is_local_and_never_advertises_external_llm(self) -> None:
        response = self.client.post("/api/parse", json={"query": "xe dap tren cau"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["external_llm_used"])
        self.assertFalse(payload["qwen_fallback_allowed"])
        self.assertIn("parsed_query", payload)

    def test_kis_query_plan_links_bundle_and_events_without_retrieval(self) -> None:
        response = self.client.post(
            "/api/query/kis/plan",
            json={
                "query": "Video trong vườn. E1: Có sầu riêng. E2: Có măng cụt.",
                "task_type": "TRAKE",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["schema_version"], "kis.query-plan.v1")
        self.assertEqual(payload["event_count"], 2)
        self.assertFalse(payload["retrieval_invoked"])
        self.assertFalse(payload["external_llm_used"])
        self.assertEqual(payload["query_bundle"]["schema_version"], "branch1.query.v1")

    def test_kis_query_plan_preserves_a_matching_manual_bundle(self) -> None:
        bundle = query_bundle()
        bundle["queries"][0]["vi"] = "truy van goc"
        response = self.client.post(
            "/api/query/kis/plan",
            json={"query": "truy van goc", "task_type": "KIS", "query_bundle": bundle},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["bundle_preserved"])
        self.assertEqual(response.json()["query_bundle"], bundle)

    def test_branch2_rejects_any_rerank_over_top_100_at_http_boundary(self) -> None:
        response = self.client.post(
            "/api/search/branch2",
            json={
                "query_bundle": query_bundle(),
                "pre_rerank_top_k": 500,
                "rerank_top_k": 101,
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_branch_routes_fail_closed_when_services_are_not_initialized(self) -> None:
        with (
            patch.object(server, "branch1_searcher", None),
            patch.object(server, "branch2_searcher", None),
            patch.object(server, "branch3_asr_searcher", None),
        ):
            branch1 = self.client.post("/api/search/branch1", json={"query_bundle": query_bundle()})
            branch2 = self.client.post("/api/search/branch2", json={"query_bundle": query_bundle()})
            branch3 = self.client.post(
                "/api/search/branch3/asr", json={"query_bundle": query_bundle()}
            )
        self.assertEqual(branch1.status_code, 503)
        self.assertEqual(branch2.status_code, 503)
        self.assertEqual(branch3.status_code, 503)

    def test_branch3_rejects_duplicate_query_roles_at_http_boundary(self) -> None:
        bundle = query_bundle()
        bundle["queries"][1]["role"] = "original"
        response = self.client.post("/api/search/branch3/asr", json={"query_bundle": bundle})
        self.assertEqual(response.status_code, 422)

    def test_branch3_route_uses_service_contract_and_maps_busy_to_429(self) -> None:
        class _ReadyBranch3:
            def execute(self, *_args):
                return {
                    "schema_version": "branch3.asr.result.v1",
                    "result_count": 0,
                    "results": [],
                }

        with patch.object(server, "branch3_asr_searcher", _ReadyBranch3()):
            response = self.client.post(
                "/api/search/branch3/asr", json={"query_bundle": query_bundle()}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result_count"], 0)

        class _BusyBranch3:
            def execute(self, *_args):
                raise RuntimeError("BRANCH3_ASR_SEARCH_BUSY")

        with patch.object(server, "branch3_asr_searcher", _BusyBranch3()):
            response = self.client.post(
                "/api/search/branch3/asr", json={"query_bundle": query_bundle()}
            )
        self.assertEqual(response.status_code, 429)

    def test_branch3_ocr_route_maps_contract_and_failures(self) -> None:
        class _ReadyOcr:
            def execute(self, *_args):
                return {
                    "schema_version": "branch3.ocr.result.v1",
                    "result_count": 0,
                    "results": [],
                }

            def health(self, _audit_sources=False):
                return {"status": "ready", "ready": True, "production_ready": False}

        with patch.object(server, "branch3_ocr_searcher", _ReadyOcr()):
            response = self.client.post(
                "/api/search/branch3/ocr",
                json={"query_bundle": query_bundle()},
            )
            health = self.client.get("/api/branch3/ocr/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["schema_version"], "branch3.ocr.result.v1")
        self.assertEqual(health.status_code, 200)
        self.assertFalse(health.json()["production_ready"])

        class _BusyOcr:
            def execute(self, *_args):
                raise RuntimeError("BRANCH3_OCR_SEARCH_BUSY")

        with patch.object(server, "branch3_ocr_searcher", _BusyOcr()):
            response = self.client.post(
                "/api/search/branch3/ocr",
                json={"query_bundle": query_bundle()},
            )
        self.assertEqual(response.status_code, 429)

        class _NotReadyOcr:
            def execute(self, *_args):
                raise RuntimeError("OCR index is stale, incomplete, or not prepared")

        with patch.object(server, "branch3_ocr_searcher", _NotReadyOcr()):
            response = self.client.post(
                "/api/search/branch3/ocr",
                json={"query_bundle": query_bundle()},
            )
        self.assertEqual(response.status_code, 503)

        oversized = self.client.post(
            "/api/search/branch3/ocr",
            json={"query_bundle": query_bundle(), "final_top_k": 501},
        )
        self.assertEqual(oversized.status_code, 422)

    def test_legacy_search_maps_ocr_value_error_to_422(self) -> None:
        dependency = {
            "status": "ready",
            "components": {"qdrant": {"ready": True}},
        }
        parsed = {"task_type": "KIS", "original_query": "text"}
        with (
            patch.object(server, "searcher", object()),
            patch.object(server, "_dependency_health", AsyncMock(return_value=dependency)),
            patch.object(
                server,
                "_execute_search",
                side_effect=ValueError("OCR query has no searchable token"),
            ),
        ):
            response = self.client.post("/api/search", json={"parsed_query": parsed})
        self.assertEqual(response.status_code, 422)
        self.assertIn("no searchable token", response.json()["detail"])

    def test_branch3_ocr_health_runs_the_cached_source_audit(self) -> None:
        service = type(
            "_OcrHealthService",
            (),
            {
                "health": lambda self, audit_sources=False: {
                    "status": "ready",
                    "ready": True,
                    "production_ready": False,
                    "required": False,
                    "schema_version": "branch3.ocr-index.v3",
                    "source_audit_performed": audit_sources,
                },
            },
        )()
        with patch.object(server, "branch3_ocr_searcher", service):
            response = self.client.get("/api/branch3/ocr/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ready"])
        self.assertFalse(payload["production_ready"])
        self.assertEqual(payload["schema_version"], "branch3.ocr-index.v3")
        self.assertTrue(payload["source_audit_performed"])
        self.assertFalse(payload["required"])

    def test_kis_fusion_route_keeps_phase_specific_error_codes(self) -> None:
        class _FailingFusion:
            def __init__(self, error: str) -> None:
                self.error = error

            def execute(self, *_args):
                raise RuntimeError(self.error)

        cases = (
            ("KIS_FUSION_SEARCH_BUSY", 429),
            ("KIS_FUSION_NOT_READY", 503),
            ("KIS_FUSION_BRANCH_FAILED: branch-2", 503),
            ("KIS_FUSION_RRF_FAILED: duplicate frame", 503),
            ("KIS_FUSION_BEIT3_FAILED: missing point", 503),
            ("unexpected failure", 503),
        )
        for error, expected_status in cases:
            with (
                self.subTest(error=error),
                patch.object(server, "kis_fusion_searcher", _FailingFusion(error)),
                patch.object(server, "_require_component", AsyncMock()),
            ):
                response = self.client.post(
                    "/api/search/fusion/kis",
                    json={"query_bundle": query_bundle()},
                )
            self.assertEqual(response.status_code, expected_status)
            detail = response.json()["detail"]
            if expected_status == 429 or isinstance(detail, dict):
                expected_code = (
                    "KIS_FUSION_SEARCH_BUSY"
                    if error == "KIS_FUSION_SEARCH_BUSY"
                    else error.split(":", 1)[0]
                    if error.startswith("KIS_FUSION_")
                    else "KIS_FUSION_EXECUTION_FAILED"
                )
                self.assertEqual(detail["code"], expected_code)

    def test_kis_fusion_health_does_not_require_legacy_searcher_object(self) -> None:
        dependency = {
            "kis_fusion": {
                "schema_version": "kis.fusion.health.v1",
                "branch": "final_fusion",
                "task_type": "KIS",
                "status": "ready",
                "ready": True,
                "production_ready": False,
                "required": False,
            }
        }
        with (
            patch.object(server, "kis_fusion_searcher", object()),
            patch.object(server, "searcher", None),
            patch.object(server, "_dependency_health", AsyncMock(return_value=dependency)),
        ):
            response = self.client.get("/api/fusion/kis/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ready"])

    def test_kis_health_and_config_share_a_non_production_ready_snapshot(self) -> None:
        fusion = {
            "schema_version": "kis.fusion.health.v1",
            "branch": "final_fusion",
            "task_type": "KIS",
            "status": "ready",
            "ready": True,
            "production_ready": False,
            "required": False,
            "resource_qualification": {
                "ready": False,
                "production_ready": False,
                "fail_closed": True,
                "error": "resource report is malformed",
            },
        }
        dependency = {
            "status": "ready",
            "ready": True,
            "production_ready": False,
            "kis_fusion": fusion,
            "components": {
                "branch1": {"ready": True},
                "branch2": {"ready": True, "components": {}},
                "branch3_asr": {"ready": True},
                "branch3_ocr": {"ready": True},
                "kis_fusion": fusion,
                "image_search": {"ready": True},
                "siglip_text": {"ready": True},
                "metadata": {"ready": True},
            },
        }
        with (
            patch.object(server, "kis_fusion_searcher", object()),
            patch.object(server, "_dependency_health", AsyncMock(return_value=dependency)),
        ):
            health = self.client.get("/api/health")
            config = self.client.get("/api/config")
            fusion_health = self.client.get("/api/fusion/kis/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["kis_fusion"]["ready"])
        self.assertFalse(health.json()["kis_fusion"]["production_ready"])
        self.assertTrue(config.json()["capabilities"]["kis_fusion"])
        self.assertTrue(config.json()["capabilities"]["kis_ordered_events"])
        self.assertTrue(config.json()["capabilities"]["video_visual_fusion"])
        self.assertFalse(fusion_health.json()["production_ready"])

    def test_kis_fusion_empty_query_contract_is_rejected_before_execution(self) -> None:
        class _ShouldNotRun:
            def execute(self, *_args):
                raise AssertionError("invalid query must be rejected by the request model")

        bundle = query_bundle()
        bundle["queries"][0]["en"] = ""
        with patch.object(server, "kis_fusion_searcher", _ShouldNotRun()):
            response = self.client.post(
                "/api/search/fusion/kis",
                json={"query_bundle": bundle},
            )
        self.assertEqual(response.status_code, 422)

    def test_ordered_kis_route_runs_one_complete_fusion_batch(self) -> None:
        class _Fusion:
            def __init__(self) -> None:
                self.bundles = []

            def execute_batch(self, bundles, _weights, *, _health_already_checked=False):
                self.bundles = bundles

                def result(frame_idx: int, score: float):
                    return {
                        "frame_uid": f"L00_V001:{frame_idx}",
                        "video_id": "L00_V001",
                        "frame_idx": frame_idx,
                        "keyframe_n": frame_idx,
                        "pts_time_s": frame_idx / 10,
                        "fps": 10.0,
                        "image_relpath": f"keyframes/L00_V001/{frame_idx:08d}.jpg",
                        "submission_string": f"L00_V001, {frame_idx}",
                        "rank": 1,
                        "score": score,
                        "final_score": score,
                    }

                return [
                    {
                        "fusion_applied": True,
                        "final_top_k": 150,
                        "branch_pool_counts": {},
                        "timing": {"total_ms": 1},
                        "results": [result(10, 0.9)],
                    },
                    {
                        "fusion_applied": True,
                        "final_top_k": 150,
                        "branch_pool_counts": {},
                        "timing": {"total_ms": 1},
                        "results": [result(20, 0.8)],
                    },
                ]

        fusion = _Fusion()
        with (
            patch.object(server, "kis_fusion_searcher", fusion),
            patch.object(server, "_require_component", AsyncMock()),
        ):
            response = self.client.post(
                "/api/search/fusion/kis/temporal",
                json={
                    "task_type": "VQA",
                    "query_bundle": query_bundle(),
                    "events": [
                        {"order": 1, "description": "first", "vi": "dau", "en": "first"},
                        {"order": 2, "description": "second", "vi": "sau", "en": "second"},
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["task_type"], "VQA")
        self.assertEqual(payload["operation"], "ordered_kis_fusion")
        self.assertTrue(payload["event_fusion_applied"])
        self.assertEqual(payload["sequences"][0]["matched_frames"], [10, 20])
        self.assertEqual(len(fusion.bundles), 2)
        self.assertEqual(fusion.bundles[0]["queries"][0]["en"], "first")

    def test_ordered_kis_route_rejects_noncontiguous_events_before_execution(self) -> None:
        response = self.client.post(
            "/api/search/fusion/kis/temporal",
            json={
                "query_bundle": query_bundle(),
                "events": [
                    {"order": 1, "description": "first"},
                    {"order": 3, "description": "third"},
                ],
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_video_visual_fusion_route_uses_only_branch1_models_and_exact_scope(self) -> None:
        class _Branch1:
            def __init__(self) -> None:
                self.arguments = None

            def execute_in_video(self, bundle, video_id, top_k):
                self.arguments = (bundle, video_id, top_k)
                return {
                    "weights": {"siglip2": 0.45, "metaclip2": 0.30, "beit3": 0.25},
                    "scope": "video",
                    "video_id": video_id,
                    "result_count": 0,
                    "results": [],
                }

        class _VideoIndex:
            @staticmethod
            def get_video_frame_count(video_id):
                return 42 if video_id == "L00_V001" else 0

        branch1 = _Branch1()
        with (
            patch.object(server, "branch1_searcher", branch1),
            patch.object(server, "searcher", _VideoIndex()),
            patch.object(server, "_require_component", AsyncMock()),
        ):
            response = self.client.post(
                "/api/video/l00-v001/search/visual-fusion",
                json={
                    "query": "person opens a door",
                    "query_bundle": query_bundle(),
                    "top_k": 25,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["models"], ["siglip2", "metaclip2", "beit3"])
        self.assertFalse(payload["cross_modal_fusion_applied"])
        self.assertEqual(payload["evaluated_frames"], 42)
        bundle, video_id, top_k = branch1.arguments
        self.assertEqual(video_id, "L00_V001")
        self.assertEqual(top_k, 25)
        self.assertEqual(bundle["queries"][0]["en"], "person opens a door")

    def test_branch3_optional_failure_does_not_hide_required_server_readiness(self) -> None:
        dependency = {
            "status": "ready",
            "production_ready": False,
            "components": {
                "api_process": {"ready": True},
                "qdrant": {"ready": True},
                "metadata": {"ready": True},
                "branch1": {"ready": True},
                "branch2": {"ready": True, "components": {}},
                "branch3_asr": {"ready": False, "required": False},
                "image_search": {"ready": True},
                "siglip_text": {"ready": True},
            },
        }
        with patch.object(server, "_dependency_health", AsyncMock(return_value=dependency)):
            health = self.client.get("/api/health")
            config = self.client.get("/api/config")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ready")
        self.assertFalse(config.json()["capabilities"]["branch3_asr"])

    def test_metadata_and_search_endpoints_fail_closed_without_dependencies(self) -> None:
        with patch.object(server, "metadata_store", None), patch.object(server, "searcher", None):
            timeline = self.client.get("/api/video/L00_V001/timeline")
            search = self.client.post(
                "/api/search",
                json={"parsed_query": {"task_type": "KIS", "original_query": "test"}},
            )
        self.assertEqual(timeline.status_code, 503)
        self.assertEqual(search.status_code, 503)

    def test_exact_frame_route_never_substitutes_a_neighbour(self) -> None:
        class _Metadata:
            @staticmethod
            def frame_by_idx(video_id, frame_idx):
                if (video_id, frame_idx) != ("L00_V001", 120):
                    return None
                return {
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "keyframe_n": 7,
                    "frame_uid": f"{video_id}:{frame_idx}",
                    "pts_time_s": 4.8,
                    "fps": 25.0,
                    "image_relpath": f"keyframes/{video_id}/{frame_idx:08d}.jpg",
                }

        with patch.object(server, "metadata_store", _Metadata()):
            exact = self.client.get("/api/frame/L00-V001/120")
            missing = self.client.get("/api/frame/L00_V001/121")
        self.assertEqual(exact.status_code, 200)
        self.assertTrue(exact.json()["exact_match"])
        self.assertEqual(exact.json()["keyframe"]["frame_uid"], "L00_V001:120")
        self.assertEqual(missing.status_code, 404)

    def test_source_frame_route_preserves_arbitrary_zero_based_identity(self) -> None:
        class _SourceIndex:
            @staticmethod
            def resolve(video_id, frame_idx):
                if frame_idx > 249:
                    return None
                return {
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "frame_uid": f"{video_id}:{frame_idx}",
                    "keyframe_n": None,
                    "pts_time_s": frame_idx / 25,
                    "fps": 25.0,
                    "indexed_keyframe": False,
                    "validation": "source_timeline",
                    "frame_index_base": 0,
                    "max_frame_idx": 249,
                    "related_seed_frame_idx": 120,
                }

            @staticmethod
            def timeline(video_id):
                return {"video_id": video_id, "frame_index_base": 0, "max_frame_idx": 249}

        with patch.object(server, "source_frame_index", _SourceIndex()):
            exact = self.client.get("/api/video/L00-V001/source-frame/121")
            out_of_range = self.client.get("/api/video/L00_V001/source-frame/250")
        self.assertEqual(exact.status_code, 200)
        self.assertTrue(exact.json()["exact_match"])
        self.assertEqual(exact.json()["source_frame"]["frame_uid"], "L00_V001:121")
        self.assertEqual(exact.json()["source_frame"]["frame_idx"], 121)
        self.assertEqual(out_of_range.status_code, 404)
        self.assertIn("between 0 and 249", out_of_range.json()["detail"])

    def test_related_frame_route_is_a_separate_encoder_free_contract(self) -> None:
        class _Related:
            @staticmethod
            def execute(video_id, frame_idx, limit):
                return {
                    "schema_version": "submission.related-frames.v1",
                    "query_pipeline_invoked": False,
                    "seed": {"video_id": video_id, "frame_idx": frame_idx},
                    "result_count": limit,
                    "results": [],
                }

        class _SourceIndex:
            @staticmethod
            def resolve(video_id, frame_idx):
                return {
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "indexed_keyframe": False,
                    "related_seed_frame_idx": 120,
                }

        with (
            patch.object(server, "related_frame_searcher", _Related()),
            patch.object(server, "source_frame_index", _SourceIndex()),
        ):
            response = self.client.post(
                "/api/submission/related-frames",
                json={"video_id": "l00-v001", "frame_idx": 121, "limit": 5},
            )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["query_pipeline_invoked"])
        self.assertEqual(response.json()["seed"]["video_id"], "L00_V001")
        self.assertEqual(response.json()["requested_seed"]["frame_idx"], 121)
        self.assertEqual(response.json()["embedding_seed"]["frame_idx"], 120)
        self.assertEqual(response.json()["result_count"], 5)

    def test_remaining_ui_routes_are_executed_and_fail_closed_before_inference(self) -> None:
        parsed = {"task_type": "KIS", "original_query": "test"}
        responses = [
            self.client.get("/api/branch1/health"),
            self.client.get("/api/branch2/health"),
            self.client.post(
                "/api/search/image",
                files={"file": ("not-used.jpg", b"not decoded", "image/jpeg")},
            ),
            self.client.post("/api/video/L00_V001/search/siglip", json={"parsed_query": parsed}),
            self.client.post("/api/discover/dam-to-siglip", json={"parsed_query": parsed}),
            self.client.post(
                "/api/search/temporal-intersection",
                json={
                    "events": [
                        {"order": 1, "description": "first", "global_scene_en": "first"},
                        {"order": 2, "description": "second", "global_scene_en": "second"},
                    ]
                },
            ),
            self.client.post("/api/submission/prepare", json={}),
            self.client.get("/api/keyframe/L00_V001/1"),
            self.client.get("/api/video/L00_V001/keyframes"),
            self.client.get("/api/video/L00_V001/timeline"),
            self.client.get("/api/video/L00_V001/media-info"),
        ]
        self.assertEqual(responses[0].status_code, 200)
        self.assertEqual(responses[1].status_code, 200)
        self.assertEqual(responses[2].status_code, 503)
        self.assertEqual(responses[3].status_code, 503)
        self.assertEqual(responses[4].status_code, 503)
        self.assertEqual(responses[5].status_code, 503)
        self.assertEqual(responses[6].status_code, 503)
        self.assertEqual(responses[7].status_code, 503)
        self.assertEqual(responses[8].status_code, 503)
        self.assertEqual(responses[9].status_code, 503)
        self.assertEqual(responses[10].status_code, 404)


if __name__ == "__main__":
    unittest.main()
