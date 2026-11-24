"""Tests for no-inference KIS query authoring plans."""

from __future__ import annotations

import unittest

from online.src.retrieval.branches.branch1.contracts import QUERY_ROLES
from online.src.retrieval.branches.final_fusion.query_plan import build_kis_query_plan
from online.src.retrieval.infrastructure.query_parser import LocalQueryParser


def bilingual_bundle() -> dict[str, object]:
    return {
        "schema_version": "branch1.query.v1",
        "queries": [
            {
                "role": role,
                "vi": (
                    "Vườn trái cây. E1: Có sầu riêng. E2: Có măng cụt."
                    if role == "original"
                    else f"ngữ cảnh thủ công {role}"
                ),
                "en": (
                    "Fruit orchard. E1: Durian appears. E2: Mangosteen appears."
                    if role == "original"
                    else f"manual context {role}"
                ),
            }
            for role in QUERY_ROLES
        ],
    }


class KisQueryPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = LocalQueryParser()

    def test_explicit_event_labels_create_one_linked_plan_without_retrieval(self) -> None:
        query = "Video về vườn trái cây. E1: Có sầu riêng. E2: Có măng cụt."
        result = build_kis_query_plan(
            query=query,
            task_type="TRAKE",
            parser=self.parser,
        )

        self.assertEqual(result["schema_version"], "kis.query-plan.v1")
        self.assertEqual(result["task_type"], "TRAKE")
        self.assertEqual(result["event_count"], 2)
        self.assertEqual([event["order"] for event in result["events"]], [1, 2])
        self.assertFalse(result["retrieval_invoked"])
        self.assertFalse(result["external_llm_used"])
        self.assertFalse(result["translation_generated"])
        roles = [query["role"] for query in result["query_bundle"]["queries"]]
        self.assertEqual(tuple(roles), QUERY_ROLES)
        self.assertNotIn("sầu riêng", result["shared_context"]["vi"].casefold())

    def test_matching_manual_bilingual_bundle_is_preserved_and_events_are_paired(self) -> None:
        bundle = bilingual_bundle()
        result = build_kis_query_plan(
            query="Vườn trái cây. E1: Có sầu riêng. E2: Có măng cụt.",
            task_type="TRAKE",
            parser=self.parser,
            query_bundle=bundle,
        )

        self.assertTrue(result["bundle_preserved"])
        self.assertEqual(result["bundle_source"], "preserved_matching_bundle")
        self.assertEqual(result["query_bundle"], bundle)
        self.assertEqual(result["events"][0]["vi"], "Có sầu riêng")
        self.assertEqual(result["events"][0]["en"], "Durian appears")
        self.assertEqual(result["events"][1]["en"], "Mangosteen appears")

    def test_unrelated_manual_bundle_is_not_reused(self) -> None:
        result = build_kis_query_plan(
            query="A chef cuts shrimp, then grills the shrimp",
            task_type="KIS",
            parser=self.parser,
            query_bundle=bilingual_bundle(),
        )

        self.assertFalse(result["bundle_preserved"])
        original = next(
            query for query in result["query_bundle"]["queries"] if query["role"] == "original"
        )
        self.assertEqual(original["en"], "A chef cuts shrimp, then grills the shrimp")
        self.assertEqual(result["event_count"], 2)

    def test_semicolon_is_temporal_only_when_sequence_language_is_explicit(self) -> None:
        temporal = self.parser.parse(
            "Chuỗi chuyển cảnh: hải sản thứ nhất; hải sản thứ hai; toàn bộ nguyên liệu"
        )
        ordinary = self.parser.parse("Có tôm; bánh mì; rau xanh")

        self.assertEqual(len(temporal.trake_events), 3)
        self.assertEqual(ordinary.trake_events, [])


if __name__ == "__main__":
    unittest.main()
