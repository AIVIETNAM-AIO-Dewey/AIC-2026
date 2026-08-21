"""Phase 3 Test: Stage 1 Multimodal Fusion (Weighted RRF + Synergy Bonus).

Evaluates queries and displays the complete step-by-step calculation breakdown
for each top-10 fused keyframe across all modalities.
"""

from __future__ import annotations

import logging
from typing import Any

from online.src.retrieval.embeddings import ModelRegistry
from online.src.retrieval.fusion import MultimodalFusionEngine
from online.src.retrieval.query_parser import QueryParser
from online.src.retrieval.vector_search import FastVectorSearchEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def print_fusion_table(query_title: str, fused_results: list[dict[str, Any]], weights: dict[str, float]):
    print(f"\n{'='*105}")
    print(f"🏆 PHASE 3 STAGE-1 FUSED RESULTS: {query_title}")
    print(f"⚖️ Active Channel Weights: Vis={weights.get('vis', 0.40):.2f}, DAM={weights.get('dam', 0.40):.2f}, ASR={weights.get('asr', 0.20):.2f}, OCR={weights.get('ocr', 0.00):.2f} | RRF Constant k=60")
    print(f"{'='*105}")

    print("\n### 🥇 Top 10 Stage-1 Multimodal Fused Keyframes")
    print("| Rank | Video ID | Frame Index | Time (s) | Modalities (Vis / DAM / ASR) | Active Ch | Synergy | Step-by-Step RRF Calculation Formula | Fused Score | Rel. Path |")
    print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|:---|")

    for r in fused_results[:10]:
        vis_info = f"r={r['rank_vis']} ({r['score_vis']:.3f})" if r['rank_vis'] else "-"
        dam_info = f"r={r['rank_dam']} ({r['score_dam']:.3f})" if r['rank_dam'] else "-"
        asr_info = f"r={r['rank_asr']} ({r['score_asr']:.3f})" if r['rank_asr'] else "-"
        mod_summary = f"{vis_info} \| {dam_info} \| {asr_info}"

        print(
            f"| **{r['rank']}** | `{r['video_id']}` | `{r['frame_idx']}` | {r['pts_time_s']}s | "
            f"{mod_summary} | {r['active_channels']}/3 | **{r['synergy_multiplier']}x** | "
            f"`{r['calculation_breakdown']}` | **{r['stage1_score']:.6f}** (Norm: `{r['normalized_score']*100:.1f}%`) | `{r['image_relpath']}` |"
        )


def main():
    searcher = FastVectorSearchEngine()
    registry = ModelRegistry.get_instance()
    fusion_engine = MultimodalFusionEngine(searcher=searcher, registry=registry, k_rrf=60)
    parser = QueryParser(gemini_model_id="gemini-3.6-flash")

    queries = [
        ("Test 1 (VQA - Cooking Class / Recipe)", "VQA", "Đoạn video về một người phụ nữ dạy nấu ăn cho những người khác. Trong đoạn video có thể thấy một người đang cầm công thức món ăn với nguyên liệu chính là 200g thịt nạc xay. Hỏi tiêu đề của công thức nấu ăn (tên món ăn) này là gì?"),
        ("Test 2 (KIS - Bicycle Road Race Flycam)", "KIS", "Tìm một đoạn video đua xe đạp, góc quay từ flycam trên cao, một vận động viên mặc áo xanh dương, trắng đang vượt ba vận động viên khác và lên vị trí dẫn đầu. Biết sau đó vận động viên này dẫn đầu suốt đoạn đường còn lại đến đích."),
    ]

    for title, task, raw_q in queries:
        parsed = parser.parse(raw_q, task_type=task, engine="gemini")
        fused_pool = fusion_engine.retrieve_and_fuse(parsed, top_k_pool=50, branch_limit=500)
        print_fusion_table(title, fused_pool, parsed.weights)


if __name__ == "__main__":
    main()
