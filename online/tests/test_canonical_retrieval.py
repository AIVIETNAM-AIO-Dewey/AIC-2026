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

    def test_parse_is_local_and_never_advertises_external_llm(self) -> None:
        response = self.client.post("/api/parse", json={"query": "xe dap tren cau"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["external_llm_used"])
        self.assertFalse(payload["qwen_fallback_allowed"])
        self.assertIn("parsed_query", payload)

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
        with patch.object(server, "branch1_searcher", None), patch.object(server, "branch2_searcher", None), patch.object(server, "branch3_asr_searcher", None):
            branch1 = self.client.post("/api/search/branch1", json={"query_bundle": query_bundle()})
            branch2 = self.client.post("/api/search/branch2", json={"query_bundle": query_bundle()})
            branch3 = self.client.post("/api/search/branch3/asr", json={"query_bundle": query_bundle()})
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
            response = self.client.post("/api/search/branch3/asr", json={"query_bundle": query_bundle()})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["result_count"], 0)

        class _BusyBranch3:
            def execute(self, *_args):
                raise RuntimeError("BRANCH3_ASR_SEARCH_BUSY")

        with patch.object(server, "branch3_asr_searcher", _BusyBranch3()):
            response = self.client.post("/api/search/branch3/asr", json={"query_bundle": query_bundle()})
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
        with patch.object(server, "searcher", object()), patch.object(
            server, "_dependency_health", AsyncMock(return_value=dependency)
        ), patch.object(
            server, "_execute_search", side_effect=ValueError("OCR query has no searchable token")
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
            with self.subTest(error=error), patch.object(
                server, "kis_fusion_searcher", _FailingFusion(error)
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
        with patch.object(server, "kis_fusion_searcher", object()), patch.object(
            server, "searcher", None
        ), patch.object(server, "_dependency_health", AsyncMock(return_value=dependency)):
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
        with patch.object(server, "kis_fusion_searcher", object()), patch.object(
            server, "_dependency_health", AsyncMock(return_value=dependency)
        ):
            health = self.client.get("/api/health")
            config = self.client.get("/api/config")
            fusion_health = self.client.get("/api/fusion/kis/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["kis_fusion"]["ready"])
        self.assertFalse(health.json()["kis_fusion"]["production_ready"])
        self.assertTrue(config.json()["capabilities"]["kis_fusion"])
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
                json={"events": [{"order": 1, "description": "first", "global_scene_en": "first"}, {"order": 2, "description": "second", "global_scene_en": "second"}]},
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
