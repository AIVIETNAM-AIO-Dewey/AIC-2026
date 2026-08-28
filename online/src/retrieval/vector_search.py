"""Independent, auditable searches over the four dataset modalities.

SigLIP and ASR return raw cosine similarity. DAM compares each object query
against region vectors and ranks a frame by the mean of its best region cosine
for each query object. OCR is deliberately lexical because this dataset does
not contain OCR embeddings.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


class FastVectorSearchEngine:
    """Memory-mapped search with bounded float32 working memory."""

    def __init__(
        self,
        unified_index_dir: str | Path,
        *,
        block_rows: int = 32_768,
    ) -> None:
        clean_path = str(unified_index_dir).strip().strip('"').strip("'")
        self.index_dir = Path(clean_path).expanduser().resolve()
        if block_rows < 1:
            raise ValueError("block_rows must be positive")
        self.block_rows = block_rows
        self._load_matrices()
        self._load_metadata()

    def _load_matrices(self) -> None:
        LOGGER.info("Loading memory-mapped matrices from %s", self.index_dir)
        started = time.perf_counter()
        self.vis_matrix = np.load(
            self.index_dir / "keyframes_visual_vectors.f16.npy", mmap_mode="r"
        )
        self.speech_matrix = np.load(
            self.index_dir / "keyframes_speech_vectors.f16.npy", mmap_mode="r"
        )
        self.dam_matrix = np.load(self.index_dir / "dam_vectors.f16.npy", mmap_mode="r")
        if self.vis_matrix.ndim != 2 or self.vis_matrix.shape[1] != 768:
            raise ValueError(f"Invalid visual matrix shape: {self.vis_matrix.shape}")
        if self.speech_matrix.ndim != 2 or self.speech_matrix.shape[1] != 1024:
            raise ValueError(f"Invalid speech matrix shape: {self.speech_matrix.shape}")
        if self.dam_matrix.ndim != 2 or self.dam_matrix.shape[1] != 1024:
            raise ValueError(f"Invalid DAM matrix shape: {self.dam_matrix.shape}")
        if self.vis_matrix.shape[0] != self.speech_matrix.shape[0]:
            raise ValueError("Visual and speech matrices must have the same frame rows")
        LOGGER.info(
            "Matrices ready in %.1fms: visual=%s speech=%s DAM=%s",
            (time.perf_counter() - started) * 1000.0,
            self.vis_matrix.shape,
            self.speech_matrix.shape,
            self.dam_matrix.shape,
        )

    def _load_metadata(self) -> None:
        LOGGER.info("Loading frame and DAM metadata")
        started = time.perf_counter()
        self.keyframe_metadata: list[dict[str, Any]] = []
        self.video_keyframes_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.keyframe_lookup: dict[tuple[str, int], dict[str, Any]] = {}
        self.frame_lookup: dict[tuple[str, int], dict[str, Any]] = {}

        keyframe_path = self.index_dir / "keyframes_metadata.jsonl"
        with keyframe_path.open("r", encoding="utf-8") as file:
            for expected_row, line in enumerate(file):
                item = json.loads(line)
                visual_row = int(item.get("visual_vector_row", expected_row))
                speech_row = int(item.get("speech_vector_row", expected_row))
                if visual_row != expected_row or speech_row != expected_row:
                    raise ValueError(f"Metadata/vector row mismatch at keyframe row {expected_row}")
                self.keyframe_metadata.append(item)
                video_id = item["video_id"]
                keyframe_n = int(item["keyframe_n"])
                frame_idx = int(item["frame_idx"])
                self.video_keyframes_map[video_id].append(item)
                self.keyframe_lookup[(video_id, keyframe_n)] = item
                self.frame_lookup[(video_id, frame_idx)] = item

        if len(self.keyframe_metadata) != self.vis_matrix.shape[0]:
            raise ValueError(
                "Keyframe metadata count does not match visual/speech matrix rows: "
                f"{len(self.keyframe_metadata)} != {self.vis_matrix.shape[0]}"
            )
        self.speech_active_mask = np.fromiter(
            (bool(item.get("has_speech")) for item in self.keyframe_metadata),
            dtype=bool,
            count=len(self.keyframe_metadata),
        )

        self.dam_metadata: list[dict[str, Any]] = []
        self.frame_dam_map: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        self.frame_dam_rows: list[list[int]] = [[] for _ in range(len(self.keyframe_metadata))]
        frame_global_rows = {
            (item["video_id"], int(item["frame_idx"])): row
            for row, item in enumerate(self.keyframe_metadata)
        }
        dam_parent_rows: list[int] = []
        dam_path = self.index_dir / "dam_metadata.jsonl"
        with dam_path.open("r", encoding="utf-8") as file:
            for dam_row, line in enumerate(file):
                item = json.loads(line)
                key = (item["video_id"], int(item["frame_idx"]))
                parent_row = frame_global_rows.get(key)
                if parent_row is None:
                    raise ValueError(f"DAM row {dam_row} has no parent keyframe: {key}")
                self.dam_metadata.append(item)
                self.frame_dam_map[key].append(item)
                self.frame_dam_rows[parent_row].append(dam_row)
                dam_parent_rows.append(parent_row)
        if len(self.dam_metadata) != self.dam_matrix.shape[0]:
            raise ValueError(
                "DAM metadata count does not match vector rows: "
                f"{len(self.dam_metadata)} != {self.dam_matrix.shape[0]}"
            )
        if any(not rows for rows in self.frame_dam_rows):
            raise ValueError("Every keyframe must have at least one DAM region")
        self.dam_parent_rows = np.asarray(dam_parent_rows, dtype=np.int32)

        LOGGER.info(
            "Metadata ready in %.1fms: %d frames, %d videos, %d DAM regions, %d frames with speech",
            (time.perf_counter() - started) * 1000.0,
            len(self.keyframe_metadata),
            len(self.video_keyframes_map),
            len(self.dam_metadata),
            int(self.speech_active_mask.sum()),
        )

    @staticmethod
    def _normalized_query(query_vector: np.ndarray, expected_dim: int) -> np.ndarray:
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape != (expected_dim,):
            raise ValueError(f"Expected query shape ({expected_dim},), got {query.shape}")
        norm = float(np.linalg.norm(query))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("Query vector must be finite and non-zero")
        return query / norm

    def _dot_blocks(
        self,
        matrix: np.ndarray,
        query: np.ndarray,
    ) -> Iterator[tuple[int, np.ndarray]]:
        for start in range(0, matrix.shape[0], self.block_rows):
            end = min(start + self.block_rows, matrix.shape[0])
            block = np.asarray(matrix[start:end], dtype=np.float32)
            yield start, block @ query

    def _top_k_dot(
        self,
        matrix: np.ndarray,
        query: np.ndarray,
        top_k: int,
        *,
        valid_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if top_k < 1:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        best_indices = np.empty(0, dtype=np.int64)
        best_scores = np.empty(0, dtype=np.float32)

        for start, scores in self._dot_blocks(matrix, query):
            if valid_mask is not None:
                block_mask = valid_mask[start : start + len(scores)]
                scores = scores.copy()
                scores[~block_mask] = -np.inf
            candidate_count = min(top_k, len(scores))
            split = len(scores) - candidate_count
            local = np.argpartition(scores, split)[split:]
            finite = np.isfinite(scores[local])
            local = local[finite]
            if not len(local):
                continue
            combined_indices = np.concatenate((best_indices, local.astype(np.int64) + start))
            combined_scores = np.concatenate((best_scores, scores[local].astype(np.float32)))
            keep_count = min(top_k, len(combined_scores))
            keep_split = len(combined_scores) - keep_count
            keep = np.argpartition(combined_scores, keep_split)[keep_split:]
            best_indices = combined_indices[keep]
            best_scores = combined_scores[keep]

        order = np.lexsort((best_indices, -best_scores))
        return best_indices[order], best_scores[order]

    @staticmethod
    def _base_result(meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "video_id": meta["video_id"],
            "keyframe_n": int(meta["keyframe_n"]),
            "frame_idx": int(meta["frame_idx"]),
            "pts_time_s": float(meta.get("pts_time_s", 0.0)),
            "image_relpath": meta.get("image_relpath", ""),
            "submission_string": f"{meta['video_id']}, {int(meta['frame_idx'])}",
            "dam_summary": meta.get("dam_summary_en", ""),
            "asr_transcript": meta.get("asr_transcript_vi", ""),
            "ocr_text": meta.get("ocr_text", ""),
        }

    def get_keyframe_by_video_and_n(self, video_id: str, keyframe_n: int) -> dict[str, Any] | None:
        return self.keyframe_lookup.get((video_id, keyframe_n))

    def get_video_keyframe_list(self, video_id: str) -> list[dict[str, Any]]:
        return sorted(
            self.video_keyframes_map.get(video_id, []),
            key=lambda item: int(item["keyframe_n"]),
        )

    def get_dam_objects_for_frame(self, video_id: str, frame_idx: int) -> list[dict[str, Any]]:
        return self.frame_dam_map.get((video_id, frame_idx), [])

    def get_video_audio_span(self, video_id: str, start_frame_idx: int, end_frame_idx: int) -> str:
        transcripts: list[str] = []
        last_text = ""
        for keyframe in self.video_keyframes_map.get(video_id, []):
            frame_idx = int(keyframe["frame_idx"])
            if start_frame_idx <= frame_idx <= end_frame_idx:
                text = str(keyframe.get("asr_transcript_vi", "")).strip()
                if text and text != last_text:
                    transcripts.append(text)
                    last_text = text
        return " ".join(transcripts)

    def search_visual(self, query_vector: np.ndarray, top_k: int = 100) -> list[dict[str, Any]]:
        """Rank keyframes by raw SigLIP image/text cosine similarity."""
        query = self._normalized_query(query_vector, 768)
        indices, scores = self._top_k_dot(self.vis_matrix, query, top_k)
        results: list[dict[str, Any]] = []
        for rank, (index, score) in enumerate(zip(indices, scores, strict=True), 1):
            result = self._base_result(self.keyframe_metadata[int(index)])
            result.update(
                {
                    "rank": rank,
                    "global_idx": int(index),
                    "score": round(float(score), 6),
                    "score_type": "cosine",
                }
            )
            results.append(result)
        return results

    def search_dam(
        self,
        query_vectors: list[np.ndarray],
        subject_names: list[str],
        top_k: int = 100,
    ) -> list[dict[str, Any]]:
        """Rank frames using transparent mean-best-region cosine aggregation."""
        if not query_vectors:
            return []
        if len(query_vectors) != len(subject_names):
            raise ValueError("DAM query vectors and subject names must have equal length")

        queries = [self._normalized_query(vector, 1024) for vector in query_vectors]
        per_subject_best: list[np.ndarray] = []
        frame_count = len(self.keyframe_metadata)
        for query in queries:
            frame_best = np.full(frame_count, -np.inf, dtype=np.float32)
            for start, scores in self._dot_blocks(self.dam_matrix, query):
                parents = self.dam_parent_rows[start : start + len(scores)]
                np.maximum.at(frame_best, parents, scores)
            if not np.isfinite(frame_best).all():
                raise ValueError("A keyframe had no DAM region score")
            per_subject_best.append(frame_best)

        subject_score_matrix = np.stack(per_subject_best)
        frame_scores = subject_score_matrix.mean(axis=0, dtype=np.float32)
        result_count = min(top_k, frame_count)
        split = frame_count - result_count
        top_frames = np.argpartition(frame_scores, split)[split:]
        order = np.lexsort((top_frames, -frame_scores[top_frames]))
        top_frames = top_frames[order]

        results: list[dict[str, Any]] = []
        for rank, frame_row_raw in enumerate(top_frames, 1):
            frame_row = int(frame_row_raw)
            dam_rows = self.frame_dam_rows[frame_row]
            region_matrix = np.asarray(self.dam_matrix[dam_rows], dtype=np.float32)
            matched_boxes: list[dict[str, Any]] = []
            subject_scores: list[dict[str, Any]] = []
            for subject, query in zip(subject_names, queries, strict=True):
                region_scores = region_matrix @ query
                best_local = int(np.argmax(region_scores))
                dam_row = dam_rows[best_local]
                object_meta = self.dam_metadata[dam_row]
                cosine = float(region_scores[best_local])
                subject_scores.append({"subject": subject, "cosine": round(cosine, 6)})
                matched_boxes.append(
                    {
                        "query_subject": subject,
                        "region_id": object_meta.get("region_id"),
                        "class_entity": object_meta.get("class_entity", "Object"),
                        "bbox": object_meta.get("bbox", []),
                        "score": round(cosine, 6),
                        "caption": object_meta.get("description_en", ""),
                    }
                )

            score = float(frame_scores[frame_row])
            result = self._base_result(self.keyframe_metadata[frame_row])
            result.update(
                {
                    "rank": rank,
                    "global_idx": frame_row,
                    "score": round(score, 6),
                    "score_type": "mean_best_region_cosine",
                    "aggregation": "mean(best region cosine per object query)",
                    "subject_scores": subject_scores,
                    "subjects_matched": f"{len(subject_names)}/{len(subject_names)}",
                    "matched_boxes": matched_boxes,
                    # Compatibility aliases; no coverage or synergy bonus is applied.
                    "composite_score": round(score, 6),
                    "avg_score": round(score, 6),
                }
            )
            results.append(result)
        return results

    def search_speech(self, query_vector: np.ndarray, top_k: int = 100) -> list[dict[str, Any]]:
        """Rank only speech-bearing frames by raw BGE-M3 cosine similarity."""
        query = self._normalized_query(query_vector, 1024)
        indices, scores = self._top_k_dot(
            self.speech_matrix,
            query,
            top_k,
            valid_mask=self.speech_active_mask,
        )
        results: list[dict[str, Any]] = []
        for rank, (index, score) in enumerate(zip(indices, scores, strict=True), 1):
            meta = self.keyframe_metadata[int(index)]
            result = self._base_result(meta)
            result.update(
                {
                    "rank": rank,
                    "global_idx": int(index),
                    "score": round(float(score), 6),
                    "score_type": "cosine",
                    "transcript": meta.get("asr_transcript_vi", ""),
                }
            )
            results.append(result)
        return results

    def search_ocr(self, keywords: list[str], top_k: int = 100) -> list[dict[str, Any]]:
        """Rank OCR text by exact case-insensitive keyword coverage."""
        original_keywords = list(
            dict.fromkeys(keyword.strip() for keyword in keywords if keyword.strip())
        )
        folded_keywords = [keyword.casefold() for keyword in original_keywords]
        if not folded_keywords:
            return []

        candidates: list[tuple[float, int, list[str]]] = []
        for index, meta in enumerate(self.keyframe_metadata):
            text = str(meta.get("ocr_text", "")).strip()
            if not text:
                continue
            folded_text = text.casefold()
            matched = [
                original
                for original, folded in zip(original_keywords, folded_keywords, strict=True)
                if folded in folded_text
            ]
            if matched:
                score = len(matched) / len(original_keywords)
                candidates.append((score, index, matched))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        results: list[dict[str, Any]] = []
        for rank, (score, index, matched) in enumerate(candidates[:top_k], 1):
            result = self._base_result(self.keyframe_metadata[index])
            result.update(
                {
                    "rank": rank,
                    "global_idx": index,
                    "score": round(float(score), 6),
                    "score_type": "keyword_match_ratio",
                    "matched_keywords": matched,
                }
            )
            results.append(result)
        return results
