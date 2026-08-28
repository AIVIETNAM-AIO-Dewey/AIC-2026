"""Phase 3 Test: TRAKE Multi-Event Stage 1 Funnel & Fusion.

Runs Stage 1 Multimodal Retrieval for each individual sub-event (E1, E2, E3, E4),
producing the Top 10 candidate frames per event with step-by-step calculations.
"""

from __future__ import annotations

import logging

from online.src.contracts.query import ParsedQuery
from online.src.retrieval.embeddings import ModelRegistry
from online.src.retrieval.fusion import MultimodalFusionEngine
from online.src.retrieval.query_parser import QueryParser
from online.src.retrieval.vector_search import FastVectorSearchEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    searcher = FastVectorSearchEngine()
    registry = ModelRegistry.get_instance()
    fusion_engine = MultimodalFusionEngine(searcher=searcher, registry=registry, k_rrf=60)
    parser = QueryParser(gemini_model_id="gemini-3.6-flash")

    raw_trake = """E1: Khoảnh khắc đầu tiên bột được bỏ vào tô măng tây.
E2: Khoảnh khắc đầu tiên thấy miến măng tây đầu tiên tiếp xúc với dầu trong chảo.
E3: Khoảnh khắc miếng măng tây đầu tiên rời khỏi chảo dầu.
E4: Khoảng khắc miếng măng tây cuối cùng rời chảo dầu và nằm hoàn toàn trên dĩa."""

    parsed = parser.parse(raw_trake, task_type="TRAKE", engine="gemini")
    print(f"\n{'=' * 105}")
    print(f"🎬 EVALUATING TRAKE QUERY: {len(parsed.trake_events)} SUB-EVENTS")
    print(f"{'=' * 105}")

    event_results = {}

    for ev in parsed.trake_events:
        print(f"\n{'─' * 105}")
        print(f'📍 EVENT E{ev.order}: "{ev.description}"')
        print(f'   • Scene (SigLIP): "{ev.scene_en}"')
        print(f"   • Objects (DAM): {ev.objects_en}")
        print(f'   • Speech (ASR): "{ev.speech_vi}"')
        print(f"{'─' * 105}")

        # Create sub-query for this specific event
        sub_parsed = ParsedQuery(
            task_type="KIS",
            original_query=ev.description,
            global_scene_en=ev.scene_en,
            objects_en=ev.objects_en,
            speech_vi=ev.speech_vi,
            ocr_keywords=ev.ocr_keywords,
            weights={"vis": 0.45, "dam": 0.40, "asr": 0.15, "ocr": 0.00},
        )

        fused_pool = fusion_engine.retrieve_and_fuse(sub_parsed, top_k_pool=10, branch_limit=500)
        event_results[f"E{ev.order}"] = fused_pool

        print(f"\n### 🏆 Top 10 Stage-1 Keyframes for Event E{ev.order}")
        print(
            "| Rank | Video ID | Frame Index | Time (s) | Modalities (Vis / DAM / ASR) | Active Ch | Synergy | Step-by-Step RRF Calculation Formula | Fused Score | Rel. Path |"
        )
        print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|:---|")

        for r in fused_pool:
            vis_info = f"r={r['rank_vis']} ({r['score_vis']:.3f})" if r["rank_vis"] else "-"
            dam_info = f"r={r['rank_dam']} ({r['score_dam']:.3f})" if r["rank_dam"] else "-"
            asr_info = f"r={r['rank_asr']} ({r['score_asr']:.3f})" if r["rank_asr"] else "-"
            mod_summary = f"{vis_info} \\| {dam_info} \\| {asr_info}"

            print(
                f"| **{r['rank']}** | `{r['video_id']}` | `{r['frame_idx']}` | {r['pts_time_s']}s | "
                f"{mod_summary} | {r['active_channels']}/3 | **{r['synergy_multiplier']}x** | "
                f"`{r['calculation_breakdown']}` | **{r['stage1_score']:.6f}** (Norm: `{r['normalized_score'] * 100:.1f}%`) | `{r['image_relpath']}` |"
            )


if __name__ == "__main__":
    main()
