"""Phase 2 Validation: 4-Channel Independent Vector Search Benchmark.

Runs parsed queries (Gemini & Qwen) across all 4 search branches:
- Table 1: SigLIP Visual Search (177,321 keyframes)
- Table 2: DAM Object Multi-Subject Composite Search (435,713 objects)
- Table 3: Audio ASR Spoken Dialogue Search (177,321 speech vectors)
- Table 4: On-Screen Text Search (OCR)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from online.src.contracts.query import ParsedQuery
from online.src.retrieval.embeddings import ModelRegistry
from online.src.retrieval.query_parser import QueryParser
from online.src.retrieval.vector_search import FastVectorSearchEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_channel_searches(
    searcher: FastVectorSearchEngine,
    registry: ModelRegistry,
    parsed: ParsedQuery,
    top_k: int = 10,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    results: dict[str, list[dict[str, Any]]] = {
        "visual": [],
        "dam": [],
        "audio": [],
        "ocr": [],
    }
    latencies: dict[str, float] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # 1. SigLIP Visual Search
    # ──────────────────────────────────────────────────────────────────────────
    if parsed.global_scene_en:
        t0 = time.perf_counter()
        vis_vec = registry.embed_siglip_text(parsed.global_scene_en)
        vis_results = searcher.search_visual(vis_vec, top_k=top_k)
        latencies["visual"] = (time.perf_counter() - t0) * 1000.0
        results["visual"] = vis_results

    # ──────────────────────────────────────────────────────────────────────────
    # 2. DAM Multi-Subject Search & Composite Pooling
    # ──────────────────────────────────────────────────────────────────────────
    if parsed.objects_en:
        t0 = time.perf_counter()
        obj_vecs = [registry.embed_bge_text(obj) for obj in parsed.objects_en]
        dam_results = searcher.search_dam(obj_vecs, parsed.objects_en, top_k=top_k)
        latencies["dam"] = (time.perf_counter() - t0) * 1000.0
        results["dam"] = dam_results

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Audio ASR Search (Always searches: uses speech_vi or fallback to original query)
    # ──────────────────────────────────────────────────────────────────────────
    audio_query = parsed.speech_vi.strip() if parsed.speech_vi else parsed.original_query
    if audio_query:
        t0 = time.perf_counter()
        speech_vec = registry.embed_bge_text(audio_query)
        audio_results = searcher.search_speech(speech_vec, top_k=top_k)
        latencies["audio"] = (time.perf_counter() - t0) * 1000.0
        results["audio"] = audio_results

    # ──────────────────────────────────────────────────────────────────────────
    # 4. OCR Search
    # ──────────────────────────────────────────────────────────────────────────
    if parsed.ocr_keywords:
        t0 = time.perf_counter()
        ocr_results = searcher.search_ocr(parsed.ocr_keywords, top_k=top_k)
        latencies["ocr"] = (time.perf_counter() - t0) * 1000.0
        results["ocr"] = ocr_results

    return results, latencies


def print_tables_for_query(query_title: str, results: dict[str, list[dict[str, Any]]], latencies: dict[str, float]):
    print(f"\n{'='*95}")
    print(f"📊 PHASE 2 RESULTS: {query_title}")
    print(f"{'='*95}")

    # ── Table 1: SigLIP ──
    lat_vis = latencies.get("visual", 0.0)
    print(f"\n### 🖼️ Table 1: SigLIP-2 Visual Search Results (Top 10) — [Latency: {lat_vis:.1f}ms]")
    if results["visual"]:
        print("| Rank | Video ID | Frame Index | Time (s) | Cosine Sim | SigLIP Prob | Image Path (AIC Relative) |")
        print("|:---:|:---:|:---:|:---:|:---:|:---:|:---|")
        for r in results["visual"]:
            prob_pct = f"{r.get('prob', 0.0)*100.0:.1f}%"
            kf_str = f"{r['keyframe_n']:03d}.jpg"
            rel_path = f"keyframes/{r['video_id']}/{kf_str}"
            print(f"| {r['rank']} | `{r['video_id']}` | `{r['frame_idx']}` | {r['pts_time_s']}s | `{r['score']}` | **{prob_pct}** | `{rel_path}` |")
    else:
        print("*(No visual sub-query provided)*")

    # ── Table 2: DAM ──
    lat_dam = latencies.get("dam", 0.0)
    print(f"\n### 🔍 Table 2: DAM Multi-Subject Composite Search Results (Top 10) — [Latency: {lat_dam:.1f}ms]")
    if results["dam"]:
        print("| Rank | Video ID | Frame Index | Subjects Matched | Composite Score | Best Matched Description & BBox |")
        print("|:---:|:---:|:---:|:---:|:---:|:---|")
        for r in results["dam"]:
            boxes_str = "; ".join([f"**{b['class_entity']}** (*\"{b['caption'][:55]}...\"*, Sim: `{b['score']}`)" for b in r['matched_boxes']])
            print(f"| {r['rank']} | `{r['video_id']}` | `{r['frame_idx']}` | {r['subjects_matched']} | **{r['composite_score']}** | {boxes_str} |")
    else:
        print("*(No object sub-query provided)*")

    # ── Table 3: Audio ASR ──
    lat_aud = latencies.get("audio", 0.0)
    print(f"\n### 🎙️ Table 3: Audio ASR Speech Search Results (Top 10) — [Latency: {lat_aud:.1f}ms]")
    if results["audio"]:
        print("| Rank | Video ID | Frame Index | Time (s) | Cosine Sim | Spoken Transcript Snippet |")
        print("|:---:|:---:|:---:|:---:|:---:|:---|")
        for r in results["audio"]:
            print(f"| {r['rank']} | `{r['video_id']}` | `{r['frame_idx']}` | {r['pts_time_s']}s | **{r['score']}** | *\"{r['transcript']}\"* |")
    else:
        print("*(No spoken audio sub-query or silent scene)*")

    # ── Table 4: OCR ──
    lat_ocr = latencies.get("ocr", 0.0)
    print(f"\n### 📝 Table 4: On-Screen Text (OCR) Search Results (Top 10) — [Latency: {lat_ocr:.1f}ms]")
    if results["ocr"]:
        print("| Rank | Video ID | Frame Index | Matched Text |")
        print("|:---:|:---:|:---:|:---|")
        for r in results["ocr"]:
            print(f"| {r['rank']} | `{r['video_id']}` | `{r['frame_idx']}` | {r['text']} |")
    else:
        print("*(OCR text extraction currently running/pending — table is currently empty)*")


def main():
    searcher = FastVectorSearchEngine()
    registry = ModelRegistry.get_instance()
    parser = QueryParser(gemini_model_id="gemini-3.6-flash", qwen_model_id="qwen2.5:1.5b")

    queries = [
        ("Test 1 (VQA)", "VQA", "Đoạn video về một người phụ nữ dạy nấu ăn cho những người khác. Trong đoạn video có thể thấy một người đang cầm công thức món ăn với nguyên liệu chính là 200g thịt nạc xay. Hỏi tiêu đề của công thức nấu ăn (tên món ăn) này là gì?"),
        ("Test 2 (KIS)", "KIS", "Tìm một đoạn video đua xe đạp, góc quay từ flycam trên cao, một vận động viên mặc áo xanh dương, trắng đang vượt ba vận động viên khác và lên vị trí dẫn đầu. Biết sau đó vận động viên này dẫn đầu suốt đoạn đường còn lại đến đích."),
    ]

    for title, task, raw_q in queries:
        # 1. Gemini query evaluation
        parsed_gemini = parser.parse(raw_q, task_type=task, engine="gemini")
        res_gemini, lat_gemini = run_channel_searches(searcher, registry, parsed_gemini, top_k=10)
        print_tables_for_query(f"{title} — [GEMINI 3.6 FLASH DECOMPOSITION]", res_gemini, lat_gemini)

        # 2. Qwen query evaluation
        parsed_qwen = parser.parse(raw_q, task_type=task, engine="qwen")
        res_qwen, lat_qwen = run_channel_searches(searcher, registry, parsed_qwen, top_k=10)
        print_tables_for_query(f"{title} — [LOCAL QWEN 2.5 DECOMPOSITION]", res_qwen, lat_qwen)


if __name__ == "__main__":
    main()
