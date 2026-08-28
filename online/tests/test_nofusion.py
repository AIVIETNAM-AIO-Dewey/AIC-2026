"""Focused tests for the KIS no-fusion experiment."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydantic import ValidationError
from safetensors.numpy import save_file

from online.src.contracts.query import ParsedQuery
from online.src.index.build_nofusion_index import build_nofusion_index
from online.src.retrieval.modality_search import IndependentModalitySearch
from online.src.retrieval.query_parser import QueryParser
from online.src.retrieval.vector_search import FastVectorSearchEngine


def _unit(dim: int, index: int) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[index] = 1.0
    return vector


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


class FakeRegistry:
    def embed_siglip_text(self, text: str) -> np.ndarray:
        return _unit(768, 0 if "first" in text else 1)

    def embed_bge_text(self, texts: list[str] | str) -> np.ndarray:
        if isinstance(texts, str):
            return _unit(1024, 0 if "alpha" in texts else 1)
        return np.stack([_unit(1024, 0 if "alpha" in text else 1) for text in texts])


def _temporal_hit(
    video_id: str,
    frame_idx: int,
    keyframe_n: int,
    pts_time_s: float,
    rank: int,
    score: float,
    global_idx: int,
) -> dict:
    return {
        "video_id": video_id,
        "keyframe_n": keyframe_n,
        "frame_idx": frame_idx,
        "pts_time_s": pts_time_s,
        "image_relpath": f"keyframes/{video_id}/{frame_idx:08d}.jpg",
        "submission_string": f"{video_id}, {frame_idx}",
        "dam_summary": "",
        "asr_transcript": "",
        "ocr_text": "",
        "rank": rank,
        "global_idx": global_idx,
        "score": score,
        "score_type": "cosine",
    }


class FakeTemporalRegistry:
    _query_indices = {
        "event one": 0,
        "event two": 1,
        "event three": 2,
        "anchor query": 3,
    }

    def embed_siglip_text(self, text: str) -> np.ndarray:
        return _unit(768, self._query_indices[text])

    def siglip_text_diagnostics(self, text: str) -> dict[str, object]:
        return {
            "token_count": len(text.split()),
            "max_tokens": 64,
            "truncated": False,
            "effective_query": text,
        }


class FakeTemporalSearcher:
    def __init__(self, pools: list[list[dict]]) -> None:
        self.pools = pools

    def search_visual(self, query_vector: np.ndarray, top_k: int = 100) -> list[dict]:
        event_index = int(np.argmax(query_vector[: len(self.pools)]))
        return [dict(result) for result in self.pools[event_index][:top_k]]


class NoFusionSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.index_dir = Path(self.temp_dir.name)

        visual = np.stack(
            [
                _unit(768, 0),
                _unit(768, 1),
                (_unit(768, 0) + _unit(768, 1)) / np.sqrt(2.0),
            ]
        ).astype(np.float16)
        speech = np.stack([np.zeros(1024), _unit(1024, 0), _unit(1024, 1)]).astype(np.float16)
        dam = np.stack(
            [
                _unit(1024, 0),
                _unit(1024, 1),
                (_unit(1024, 0) + _unit(1024, 1)) / np.sqrt(2.0),
                _unit(1024, 2),
                -_unit(1024, 0),
                -_unit(1024, 1),
            ]
        ).astype(np.float16)
        np.save(self.index_dir / "keyframes_visual_vectors.f16.npy", visual)
        np.save(self.index_dir / "keyframes_speech_vectors.f16.npy", speech)
        np.save(self.index_dir / "dam_vectors.f16.npy", dam)

        video_ids = ["L00_V001", "L00_V001", "L00_V002"]
        frames = []
        for index in range(3):
            video_id = video_ids[index]
            frames.append(
                {
                    "point_id": index + 1,
                    "video_id": video_id,
                    "keyframe_n": index + 1,
                    "frame_idx": (index + 1) * 10,
                    "pts_time_s": float(index),
                    "image_relpath": f"keyframes/{video_id}/{(index + 1) * 10:08d}.jpg",
                    "visual_vector_row": index,
                    "speech_vector_row": index,
                    "has_speech": index > 0,
                    "asr_transcript_vi": "alpha speech" if index == 1 else "beta speech",
                    "ocr_text": ["alpha beta", "alpha", "beta"][index],
                    "dam_summary_en": f"frame {index}",
                }
            )
        _write_jsonl(self.index_dir / "keyframes_metadata.jsonl", frames)

        dam_rows = []
        for dam_row in range(6):
            frame = dam_row // 2
            dam_rows.append(
                {
                    "video_id": video_ids[frame],
                    "frame_idx": (frame + 1) * 10,
                    "keyframe_n": frame + 1,
                    "region_id": f"region-{dam_row}",
                    "class_entity": f"object-{dam_row}",
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                    "description_en": f"region {dam_row}",
                }
            )
        _write_jsonl(self.index_dir / "dam_metadata.jsonl", dam_rows)
        self.searcher = FastVectorSearchEngine(self.index_dir, block_rows=2)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_visual_returns_raw_cosine_order(self) -> None:
        results = self.searcher.search_visual(_unit(768, 0), top_k=3)
        self.assertEqual([result["frame_idx"] for result in results], [10, 30, 20])
        self.assertEqual(results[0]["score_type"], "cosine")
        self.assertNotIn("stage1_score", results[0])
        self.assertNotIn("final_score", results[0])

    def test_visual_can_be_restricted_to_one_video_without_score_fusion(self) -> None:
        results = self.searcher.search_visual_in_video(
            _unit(768, 0),
            "l00-v001",
            top_k=10,
        )
        self.assertEqual([result["frame_idx"] for result in results], [10, 20])
        self.assertTrue(all(result["video_id"] == "L00_V001" for result in results))
        self.assertTrue(all(result["score_type"] == "cosine" for result in results))
        self.assertTrue(all(result["scope"] == "video" for result in results))
        self.assertEqual(self.searcher.get_video_frame_count("L00-V001"), 2)
        self.assertEqual(
            self.searcher.search_visual_in_video(_unit(768, 0), "L99_V999"),
            [],
        )

    def test_speech_excludes_silent_zero_vector_rows(self) -> None:
        results = self.searcher.search_speech(-_unit(1024, 0), top_k=3)
        self.assertEqual({result["frame_idx"] for result in results}, {20, 30})
        self.assertNotIn(10, [result["frame_idx"] for result in results])

    def test_dam_uses_mean_best_region_cosine_without_bonus(self) -> None:
        results = self.searcher.search_dam(
            [_unit(1024, 0), _unit(1024, 1)],
            ["alpha", "beta"],
            top_k=3,
        )
        self.assertEqual(results[0]["frame_idx"], 10)
        self.assertAlmostEqual(results[0]["score"], 1.0, places=5)
        self.assertEqual(results[0]["score"], results[0]["composite_score"])
        self.assertEqual(len(results[0]["subject_scores"]), 2)
        self.assertNotIn("synergy_multiplier", results[0])

    def test_dam_threshold_labels_unsupported_regions_without_changing_raw_rank(self) -> None:
        results = self.searcher.search_dam(
            [_unit(1024, 0), _unit(1024, 1)],
            ["alpha", "beta"],
            top_k=3,
            match_threshold=0.5,
        )
        self.assertEqual([result["frame_idx"] for result in results], [10, 20, 30])
        unsupported = results[-1]
        self.assertEqual(unsupported["subjects_matched"], "0/2")
        self.assertEqual(unsupported["coverage_ratio"], 0.0)
        self.assertEqual(unsupported["matched_boxes"], [])
        self.assertEqual(len(unsupported["best_region_candidates"]), 2)
        self.assertTrue(all(not item["matched"] for item in unsupported["subject_scores"]))

    def test_ocr_is_labeled_keyword_coverage_not_cosine(self) -> None:
        results = self.searcher.search_ocr(["alpha", "beta"], top_k=3)
        self.assertEqual(results[0]["frame_idx"], 10)
        self.assertEqual(results[0]["score"], 1.0)
        self.assertEqual(results[0]["score_type"], "keyword_match_ratio")

    def test_orchestrator_ignores_weights_and_keeps_four_pools(self) -> None:
        parsed = ParsedQuery(
            task_type="KIS",
            original_query="alpha original",
            global_scene_en="first scene",
            objects_en=["alpha", "beta"],
            speech_vi="alpha speech",
            ocr_keywords=["alpha", "beta"],
            weights={"vis": 0.0, "dam": 0.0, "asr": 0.0, "ocr": 0.0},
        )
        orchestrator = IndependentModalitySearch(
            searcher=self.searcher,
            registry=FakeRegistry(),
        )
        pools = orchestrator.search(parsed, top_k=2)
        self.assertEqual(set(pools), {"siglip", "dam", "ocr", "asr"})
        self.assertTrue(all(pool["status"] == "ok" for pool in pools.values()))
        self.assertTrue(all(pool["result_count"] == 2 for pool in pools.values()))
        for pool in pools.values():
            for result in pool["results"]:
                self.assertNotIn("stage1_score", result)
                self.assertNotIn("final_score", result)

    def test_orchestrator_does_not_search_asr_with_visual_query_fallback(self) -> None:
        parsed = ParsedQuery(
            task_type="KIS",
            original_query="a purely visual description",
            global_scene_en="first scene",
            objects_en=["alpha"],
            speech_vi="",
            ocr_keywords=[],
        )
        orchestrator = IndependentModalitySearch(
            searcher=self.searcher,
            registry=FakeRegistry(),
        )
        pools = orchestrator.search(parsed, top_k=2)
        self.assertEqual(pools["asr"]["status"], "not_run")
        self.assertEqual(pools["asr"]["query_source"], "speech_vi")
        self.assertIn("no explicit speech", pools["asr"]["reason"])

    def test_orchestrator_video_drilldown_remains_siglip_only_and_auditable(self) -> None:
        parsed = ParsedQuery(
            task_type="KIS",
            original_query="first scene",
            global_scene_en="first scene",
        )
        orchestrator = IndependentModalitySearch(
            searcher=self.searcher,
            registry=FakeRegistry(),
        )
        pool = orchestrator.search_visual_in_video(
            parsed,
            video_id="L00-V001",
            top_k=10,
        )
        self.assertEqual(pool["modality"], "siglip")
        self.assertEqual(pool["scope"], "video")
        self.assertEqual(pool["video_id"], "L00_V001")
        self.assertEqual(pool["evaluated_frames"], 2)
        self.assertFalse(pool["fusion_applied"])
        self.assertFalse(pool["reranking_applied"])
        self.assertEqual([result["frame_idx"] for result in pool["results"]], [10, 20])

    def test_explicit_discovery_cascade_keeps_dam_gating_out_of_final_score(self) -> None:
        parsed = ParsedQuery(
            task_type="KIS",
            original_query="first scene with alpha object",
            global_scene_en="first scene",
            objects_en=["alpha"],
        )
        orchestrator = IndependentModalitySearch(
            searcher=self.searcher,
            registry=FakeRegistry(),
        )
        discovery = orchestrator.discover_dam_to_siglip(
            parsed,
            dam_top_frames_per_object=3,
            siglip_top_frames_per_video=2,
        )
        self.assertEqual(discovery["operation"], "dam_to_siglip_discovery_cascade")
        self.assertTrue(discovery["cross_modal_gating_applied"])
        self.assertFalse(discovery["fusion_applied"])
        self.assertFalse(discovery["dam_score_used_in_final_rank"])
        self.assertEqual(discovery["unique_candidate_video_count"], 2)

        cascade = discovery["cascades"][0]
        self.assertEqual(cascade["object_query"], "alpha")
        self.assertEqual(cascade["candidate_video_count"], 2)
        self.assertEqual([result["frame_idx"] for result in cascade["results"]], [10, 30, 20])
        self.assertTrue(all(result["score_type"] == "cosine" for result in cascade["results"]))
        self.assertTrue(
            all(result["scope"] == "dam_to_siglip_cascade" for result in cascade["results"])
        )
        self.assertTrue(all("dam_discovery_rank" in result for result in cascade["results"]))


class TemporalIntersectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events = [
            {
                "order": 1,
                "description": "First cyclist",
                "global_scene_en": "event one",
            },
            {
                "order": 2,
                "description": "Second cyclist",
                "global_scene_en": "event two",
            },
            {
                "order": 3,
                "description": "Third cyclist",
                "global_scene_en": "event three",
            },
        ]
        # Event one's global Top-1 is the same physical frame as event three.
        # A valid path therefore has to retain the lower-ranked frame at t=1.
        self.pools = [
            [
                _temporal_hit("L00_V001", 300, 3, 3.0, 1, 0.99, 2),
                _temporal_hit("L00_V002", 400, 4, 4.0, 2, 0.90, 3),
                _temporal_hit("L00_V001", 100, 1, 1.0, 3, 0.80, 0),
            ],
            [
                _temporal_hit("L00_V001", 200, 2, 2.0, 1, 0.95, 1),
                _temporal_hit("L00_V002", 500, 5, 5.0, 2, 0.85, 4),
            ],
            [
                _temporal_hit("L00_V001", 300, 3, 3.0, 1, 0.90, 2),
                _temporal_hit("L00_V002", 450, 5, 4.5, 2, 0.84, 5),
            ],
        ]

    def _search(self, pools: list[list[dict]] | None = None, **kwargs) -> dict:
        orchestrator = IndependentModalitySearch(
            searcher=FakeTemporalSearcher(pools or self.pools),
            registry=FakeTemporalRegistry(),
        )
        return orchestrator.search_temporal_intersection(
            events=self.events,
            top_k_per_event=300,
            top_k_sequences=20,
            max_gap_seconds=kwargs.pop("max_gap_seconds", 30.0),
            **kwargs,
        )

    def test_temporal_intersection_uses_dp_instead_of_naive_event_top_ones(self) -> None:
        response = self._search()
        self.assertEqual(response["intersection_video_count"], 2)
        self.assertEqual(response["monotonic_video_count"], 1)
        self.assertEqual(response["result_count"], 1)
        self.assertTrue(response["same_modality_event_aggregation_applied"])
        self.assertFalse(response["cross_modal_fusion_applied"])
        self.assertFalse(response["reranking_applied"])

        sequence = response["sequences"][0]
        self.assertEqual(sequence["video_id"], "L00_V001")
        self.assertEqual(sequence["matched_frames"], [100, 200, 300])
        self.assertEqual(
            [event["rank"] for event in sequence["matched_events"]],
            [3, 1, 1],
        )
        self.assertEqual(sequence["global_rank_sum"], 5)
        self.assertEqual(sequence["minimum_event_score"], 0.8)
        self.assertEqual(sequence["mean_event_score"], 0.883333)
        self.assertEqual(sequence["sequence_score"], 0.8)
        self.assertEqual(sequence["gaps_seconds"], [1.0, 1.0])
        self.assertEqual(sequence["span_seconds"], 2.0)
        self.assertEqual(sequence["anchor_frame"]["frame_idx"], 100)
        self.assertEqual(sequence["submission_string"], "L00_V001, 100")
        self.assertEqual(
            [event["event_order"] for event in sequence["matched_events"]],
            [1, 2, 3],
        )
        self.assertEqual(
            [event["score_type"] for event in sequence["matched_events"]],
            ["cosine", "cosine", "cosine"],
        )
        self.assertEqual(
            response["same_modality_event_aggregation"],
            "bottleneck_minimum_then_arithmetic_mean",
        )

    def test_temporal_intersection_prefers_balanced_event_evidence(self) -> None:
        pools = [
            [
                _temporal_hit("BALANCED", 10, 1, 1.0, 1, 0.80, 0),
                _temporal_hit("UNBALANCED", 10, 1, 1.0, 2, 0.99, 3),
            ],
            [
                _temporal_hit("BALANCED", 20, 2, 2.0, 1, 0.80, 1),
                _temporal_hit("UNBALANCED", 20, 2, 2.0, 2, 0.99, 4),
            ],
            [
                _temporal_hit("BALANCED", 30, 3, 3.0, 1, 0.80, 2),
                _temporal_hit("UNBALANCED", 30, 3, 3.0, 2, 0.70, 5),
            ],
        ]
        response = self._search(pools)
        self.assertEqual(response["sequences"][0]["video_id"], "BALANCED")
        self.assertEqual(response["sequences"][0]["minimum_event_score"], 0.8)
        self.assertGreater(
            response["sequences"][1]["mean_event_score"],
            response["sequences"][0]["mean_event_score"],
        )

    def test_temporal_intersection_uses_optional_shared_scene_anchor(self) -> None:
        anchor_pool = [
            _temporal_hit("L00_V001", 90, 1, 0.5, 2, 0.72, 6),
            _temporal_hit("L00_V002", 390, 3, 3.5, 1, 0.96, 7),
        ]
        response = self._search([*self.pools, anchor_pool], anchor_query="anchor query")
        self.assertTrue(response["anchor_query_applied"])
        self.assertEqual(
            response["same_modality_event_aggregation"],
            "mean_context_anchor_and_minimum_event_then_event_mean",
        )
        # Only L00_V001 has a valid monotonic event path, so the anchor may
        # narrow/rank candidates but cannot manufacture a temporal match.
        self.assertEqual(response["sequences"][0]["video_id"], "L00_V001")
        self.assertEqual(response["sequences"][0]["context_anchor_score"], 0.72)
        self.assertEqual(response["sequences"][0]["sequence_score"], 0.76)

    def test_temporal_intersection_applies_gap_as_filter_not_score(self) -> None:
        response = self._search(max_gap_seconds=0.5)
        self.assertEqual(response["intersection_video_count"], 2)
        self.assertEqual(response["monotonic_video_count"], 0)
        self.assertEqual(response["sequences"], [])
        self.assertIn("filter and never changes a score", response["score_policy"])

    def test_temporal_intersection_returns_empty_when_event_video_sets_do_not_overlap(self) -> None:
        disjoint_pools = [list(pool) for pool in self.pools]
        disjoint_pools[2] = [
            _temporal_hit("L00_V003", 600, 6, 6.0, 1, 0.91, 6),
        ]
        response = self._search(disjoint_pools)
        self.assertEqual(response["intersection_video_count"], 0)
        self.assertEqual(response["sequences"], [])

    def test_temporal_intersection_validates_event_contract(self) -> None:
        orchestrator = IndependentModalitySearch(
            searcher=FakeTemporalSearcher(self.pools),
            registry=FakeTemporalRegistry(),
        )
        with self.assertRaisesRegex(ValueError, "between 2 and 6"):
            orchestrator.search_temporal_intersection(events=self.events[:1])
        with self.assertRaisesRegex(ValueError, "description and global_scene_en"):
            orchestrator.search_temporal_intersection(
                events=[self.events[0], {**self.events[1], "global_scene_en": ""}]
            )
        with self.assertRaisesRegex(ValueError, "unique positive integers"):
            orchestrator.search_temporal_intersection(
                events=[self.events[0], {**self.events[1], "order": 1}]
            )


class QueryParserPurityTests(unittest.TestCase):
    def setUp(self) -> None:
        # These unit tests exercise deterministic post-processing only; no API
        # client or local LLM is needed.
        self.parser = QueryParser.__new__(QueryParser)

    def test_model_output_is_capped_for_existing_embedding_constraints(self) -> None:
        parsed = self.parser._build_parsed_query(
            {
                "global_scene_en": " ".join(f"word{index}" for index in range(60)),
                "objects_en": [
                    " ".join(f"object{item}_{word}" for word in range(20)) for item in range(6)
                ],
                "speech_vi": "",
                "ocr_keywords": [],
            },
            "Mô tả hình ảnh",
            "KIS",
        )
        self.assertLessEqual(len(parsed.global_scene_en.split()), 40)
        self.assertEqual(len(parsed.objects_en), 3)
        self.assertTrue(all(len(item.split()) <= 14 for item in parsed.objects_en))

    def test_direct_json_contract_does_not_require_original_query(self) -> None:
        parsed = ParsedQuery(global_scene_en="a compact visual query")
        self.assertEqual(parsed.original_query, "")

    def test_direct_json_contract_rejects_silent_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError):
            ParsedQuery(global_scene_en="a visual query", layout_en="silently ignored before")

    def test_ordered_finish_description_is_detected_for_event_decomposition(self) -> None:
        query = (
            "Ba tay đua về đích lần lượt là áo vàng, "
            "áo xanh quần đen và áo xanh quần đỏ"
        )
        self.assertEqual(self.parser._detect_task_type(query), "TRAKE")

    def test_temporal_event_queries_are_normalized_and_capped(self) -> None:
        parsed = self.parser._build_parsed_query(
            {
                "global_scene_en": "A bicycle race finish line",
                "objects_en": ["cyclists"],
                "trake_events": [
                    {
                        "order": 1,
                        "description": "  Yellow rider finishes first  ",
                        "scene_en": " ".join(f"eventword{index}" for index in range(60)),
                        "objects_en": ["yellow cyclist", "yellow cyclist"],
                    },
                    {
                        "order": 2,
                        "description": "Blue rider follows",
                        "scene_en": "A blue cyclist crosses the finish line",
                    },
                ],
            },
            "Các tay đua lần lượt về đích",
            "TRAKE",
        )
        self.assertEqual(parsed.trake_events[0].description, "Yellow rider finishes first")
        self.assertLessEqual(len(parsed.trake_events[0].scene_en.split()), 40)
        self.assertEqual(parsed.trake_events[0].objects_en, ["yellow cyclist"])
        self.assertTrue(parsed.is_temporal_trake)

    def test_rule_parser_splits_vietnamese_respectively_list_without_llm(self) -> None:
        query = (
            "Góc máy sát mặt đường tại vạch đích, theo thứ tự nhất nhì ba lần lượt là "
            "tay đua áo vàng quần đen, tay đua áo xanh quần đen và "
            "tay đua áo xanh quần đỏ."
        )
        parsed = self.parser._parse_local(query, "TRAKE")
        self.assertEqual(len(parsed.trake_events), 3)
        self.assertEqual(
            [event.description for event in parsed.trake_events],
            [
                "tay đua áo vàng quần đen",
                "tay đua áo xanh quần đen",
                "tay đua áo xanh quần đỏ",
            ],
        )
        self.assertTrue(all(event.scene_en for event in parsed.trake_events))

    def test_rule_parser_keeps_pure_visual_query_out_of_asr(self) -> None:
        parsed = self.parser._parse_local(
            "Giáo viên đứng cạnh một slide có ba tầng hộp màu",
            "KIS",
        )
        self.assertEqual(parsed.speech_vi, "")

    def test_rule_parser_keeps_explicit_spoken_topic_for_asr(self) -> None:
        query = "Người giáo viên nói về nguồn lao động Việt Nam"
        parsed = self.parser._parse_local(query, "KIS")
        self.assertEqual(parsed.speech_vi, query)


class NoFusionBuilderTests(unittest.TestCase):
    def test_builder_merges_rows_and_never_reuses_per_video_point_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            artifacts = root / "artifacts"
            for name in (
                "map-keyframes",
                "unified_metadata",
                "scene_embeddings",
                "asr_aligned",
                "dense_text_embeddings",
                "media-info",
            ):
                (artifacts / name).mkdir(parents=True, exist_ok=True)

            video_id = "L00_V001"
            (artifacts / "media-info" / f"{video_id}.json").write_text("{}\n", encoding="utf-8")
            (artifacts / "keyframes" / video_id).mkdir(parents=True)
            for frame_idx in (3, 33):
                (artifacts / "keyframes" / video_id / f"{frame_idx:08d}.jpg").touch()
            with (artifacts / "map-keyframes" / f"{video_id}.csv").open(
                "w", encoding="utf-8", newline=""
            ) as file:
                writer = csv.DictWriter(file, fieldnames=["n", "pts_time", "fps", "frame_idx"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"n": 1, "pts_time": 0.1, "fps": 30, "frame_idx": 3},
                        {"n": 2, "pts_time": 1.1, "fps": 30, "frame_idx": 33},
                    ]
                )

            source_rows = [
                {
                    "point_id": 1,
                    "video_id": video_id,
                    "keyframe_n": index + 1,
                    "frame_idx": [3, 33][index],
                    "pts_time_s": [0.1, 1.1][index],
                    "fps": 30.0,
                    "frame_uid": f"{video_id}:{[3, 33][index]}",
                    "image_relpath": f"keyframes/{video_id}/{[3, 33][index]:08d}.jpg",
                    "embedding_row": index,
                    "ocr_text": "text" if index == 0 else "",
                }
                for index in range(2)
            ]
            _write_jsonl(artifacts / "unified_metadata" / f"{video_id}.jsonl", source_rows)
            save_file(
                {"embeddings": np.stack([_unit(768, 0), _unit(768, 1)]).astype(np.float16)},
                artifacts / "scene_embeddings" / f"{video_id}.safetensors",
            )

            asr_rows = []
            for index, source in enumerate(source_rows):
                asr_rows.append(
                    {
                        **source,
                        "point_id": index + 1,
                        "speech_vector_row": index,
                        "asr_transcript_vi": "speech" if index == 1 else "",
                        "has_speech": index == 1,
                    }
                )
            _write_jsonl(artifacts / "asr_aligned" / "keyframes_asr_metadata.jsonl", asr_rows)
            np.save(
                artifacts / "asr_aligned" / "keyframes_speech_vectors.f16.npy",
                np.stack([np.zeros(1024), _unit(1024, 0)]).astype(np.float16),
            )

            dam_rows = [
                {
                    "video_id": video_id,
                    "frame_idx": source["frame_idx"],
                    "keyframe_n": source["keyframe_n"],
                    "region_id": f"region-{index}",
                    "class_entity": "object",
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                    "description_en": "object",
                }
                for index, source in enumerate(source_rows)
            ]
            _write_jsonl(artifacts / "dense_text_embeddings" / "dam_metadata.jsonl", dam_rows)
            np.save(
                artifacts / "dense_text_embeddings" / "dam_vectors.f16.npy",
                np.stack([_unit(1024, 0), _unit(1024, 1)]).astype(np.float16),
            )

            output = root / "unified_index"
            summary = build_nofusion_index(
                root,
                output,
                asset_mode="hardlink",
                compute_checksums=False,
                expected_videos=1,
                expected_frames=2,
                expected_dam_regions=2,
            )
            built_rows = list(
                json.loads(line)
                for line in (output / "keyframes_metadata.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertEqual([row["point_id"] for row in built_rows], [1, 2])
            self.assertEqual([row["visual_vector_row"] for row in built_rows], [0, 1])
            self.assertEqual(built_rows[1]["asr_transcript_vi"], "speech")
            self.assertEqual(summary["total_keyframes"], 2)
            self.assertTrue((output / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
