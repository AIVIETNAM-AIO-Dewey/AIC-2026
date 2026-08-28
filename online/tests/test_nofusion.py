"""Focused tests for the KIS no-fusion experiment."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from online.src.contracts.query import ParsedQuery
from online.src.index.build_nofusion_index import build_nofusion_index
from online.src.retrieval.modality_search import IndependentModalitySearch
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

        frames = []
        for index in range(3):
            frames.append(
                {
                    "point_id": index + 1,
                    "video_id": "L00_V001",
                    "keyframe_n": index + 1,
                    "frame_idx": (index + 1) * 10,
                    "pts_time_s": float(index),
                    "image_relpath": f"keyframes/L00_V001/{(index + 1) * 10:08d}.jpg",
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
                    "video_id": "L00_V001",
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
