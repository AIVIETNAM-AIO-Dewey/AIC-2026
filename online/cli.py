"""Terminal-based Interactive Query Interface for AIC Online Retrieval Pipeline.

Usage:
    python -m online.cli
    python -m online.cli --query "..." --task KIS
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

from online.src.contracts.query import ParsedQuery
from online.src.retrieval.embeddings import ModelRegistry
from online.src.retrieval.fusion import MultimodalFusionEngine
from online.src.retrieval.query_parser import QueryParser
from online.src.retrieval.stage2_reranker import Stage2Reranker
from online.src.retrieval.vector_search import FastVectorSearchEngine
from online.src.retrieval.vqa_reasoner import VQAReasoner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Global singletons (loaded once, reused across interactive queries)
# ──────────────────────────────────────────────────────────────────────────────
_searcher: FastVectorSearchEngine | None = None
_registry: ModelRegistry | None = None
_fusion: MultimodalFusionEngine | None = None
_reranker: Stage2Reranker | None = None
_parser: QueryParser | None = None


def _init_engine():
    global _searcher, _registry, _fusion, _reranker, _parser
    if _searcher is not None:
        return

    print("\n⚡ Loading retrieval engine (first-time only)...")
    t0 = time.perf_counter()

    _searcher = FastVectorSearchEngine()
    _registry = ModelRegistry.get_instance()
    _fusion = MultimodalFusionEngine(searcher=_searcher, registry=_registry, k_rrf=60)
    _reranker = Stage2Reranker(registry=_registry, vqa_reasoner=VQAReasoner())
    _parser = QueryParser()

    dt = time.perf_counter() - t0
    print(f"✅ Engine ready in {dt:.1f}s\n")


# ──────────────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────────────
def _print_kis_results(results: list[dict[str, Any]], explain_count: int = 1):
    print(f"\n{'─' * 90}")
    print(f"🎯 KIS RESULTS — Top {len(results)} Candidates")
    print(f"{'─' * 90}")
    print(
        f"{'Rank':<5} {'Submission String':<30} {'Time(s)':<10} {'S1 Norm':<10} {'CE Score':<10} {'Final':<10} {'Image Path'}"
    )
    print(f"{'─' * 90}")
    for r in results:
        print(
            f"{r['final_rank']:<5} {r['submission_string']:<30} "
            f"{r['pts_time_s']:<10.1f} {r['normalized_score'] * 100:<9.1f}% {r['cross_encoder_score']:<10.4f} "
            f"{r['final_score']:<10.4f} {r['image_relpath']}"
        )

    cards_to_show = results[:explain_count]
    for r in cards_to_show:
        rank_label = (
            f"TOP {r['final_rank']}" if r["final_rank"] == 1 else f"RANK #{r['final_rank']}"
        )
        print(f"\n{'═' * 90}")
        print(f"🌟 {rank_label} EXPLAINABILITY CARD")
        print(f"{'═' * 90}")
        print(f"  Video:       {r['video_id']}")
        print(
            f"  Frame:       {r['frame_idx']}  ({r['pts_time_s']:.1f}s | Keyframe #{r.get('keyframe_n', 1)})"
        )
        print(f"  Submit:      {r['submission_string']}")
        print(f"  Image:       {r['image_relpath']}")
        print(
            f"  Final Score: {r['final_score']:.4f}  (S1: {r['normalized_score'] * 100:.1f}%  CE: {r['cross_encoder_score']:.4f})"
        )
        print(
            f"  Modalities:  {r['active_channels']}/4 channels, synergy {r['synergy_multiplier']}x"
        )

        # Full ASR Speech (Priority)
        asr_txt = r.get("asr_transcript", "").strip()
        if asr_txt and asr_txt not in ("[Silent Frame]", ""):
            print(f'\n  🎙️ FULL ASR SPEECH TRANSCRIPT:\n     "{asr_txt}"')
        else:
            print("\n  🎙️ FULL ASR SPEECH TRANSCRIPT:\n     (No speech / background audio)")

        # Full DAM Visual Description
        dam_txt = r.get("dam_summary", "").strip()
        if dam_txt:
            print(f'\n  🖼️ FULL DAM VISUAL DESCRIPTION:\n     "{dam_txt}"')

        if r.get("matched_boxes"):
            print("\n  📦 MATCHED OBJECT BOUNDING BOXES:")
            for b in r["matched_boxes"]:
                print(
                    f'     - [{b.get("class_entity", "Object")}] (Sim: {b.get("score", 0.0):.3f}, BBox: {b.get("bbox", [])}): "{b.get("caption", "")}"'
                )
        print(f"{'═' * 90}")
    print()


def _print_vqa_results(results: list[dict[str, Any]], explain_count: int = 10):
    print(f"\n{'─' * 90}")
    print(f"❓ VQA RESULTS — Top {len(results)} Candidates")
    print(f"{'─' * 90}")
    print(f"{'Rank':<5} {'Submission String':<50} {'CE Score':<10} {'Final':<10} {'Image Path'}")
    print(f"{'─' * 90}")
    for r in results:
        sub = r["submission_string"]
        if len(sub) > 48:
            sub = sub[:45] + "..."
        print(
            f"{r['final_rank']:<5} {sub:<50} "
            f"{r['cross_encoder_score']:<10.4f} {r['final_score']:<10.4f} {r['image_relpath']}"
        )

    cards_to_show = results[:explain_count]
    for r in cards_to_show:
        rank_label = (
            f"TOP {r['final_rank']}" if r["final_rank"] == 1 else f"RANK #{r['final_rank']}"
        )
        print(f"\n{'═' * 90}")
        print(f"🌟 {rank_label} EXPLAINABILITY CARD (VQA EVIDENCE)")
        print(f"{'═' * 90}")
        print(f'  Extracted Answer: "{r.get("vqa_answer", "N/A")}"')
        print(f"  Video:            {r['video_id']}")
        print(
            f"  Frame:            {r['frame_idx']}  ({r['pts_time_s']:.1f}s | Keyframe #{r.get('keyframe_n', 1)})"
        )
        print(f"  Submit:           {r['submission_string']}")
        print(f"  Image:            {r['image_relpath']}")
        print(f"  Final Score:      {r['final_score']:.4f}")

        # Full ASR Speech
        asr_txt = r.get("asr_transcript", "").strip()
        if asr_txt and asr_txt not in ("[Silent Frame]", ""):
            print(f'\n  🎙️ FULL ASR SPEECH TRANSCRIPT:\n     "{asr_txt}"')
        else:
            print("\n  🎙️ FULL ASR SPEECH TRANSCRIPT:\n     (No speech / background audio)")

        # Full DAM Visual Description
        dam_txt = r.get("dam_summary", "").strip()
        if dam_txt:
            print(f'\n  🖼️ FULL DAM VISUAL DESCRIPTION:\n     "{dam_txt}"')

        if r.get("matched_boxes"):
            print("\n  📦 MATCHED OBJECT BOUNDING BOXES:")
            for b in r["matched_boxes"]:
                print(
                    f'     - [{b.get("class_entity", "Object")}] (Sim: {b.get("score", 0.0):.3f}, BBox: {b.get("bbox", [])}): "{b.get("caption", "")}"'
                )
        print(f"{'═' * 90}")
    print()


def _print_trake_results(sequences: list[dict[str, Any]], explain_count: int = 1):
    print(f"\n{'─' * 95}")
    print(f"⏱️ TRAKE RESULTS — Top {len(sequences)} Monotonic Sequences (DP + Narrative Reranked)")
    print(f"{'─' * 95}")
    print(
        f"{'Rank':<5} {'Submission String':<45} {'Final':<9} {'DP':<8} {'Narrative':<11} {'Mono?'}"
    )
    print(f"{'─' * 95}")
    for s in sequences:
        times = s["timestamps"]
        is_mono = all(times[i] < times[i + 1] for i in range(len(times) - 1))
        f_score = s.get("final_score", s.get("sequence_score", 0.0))
        dp_sc = s.get("dp_score", s.get("sequence_score", 0.0))
        narr_sc = s.get("narrative_score", dp_sc)
        img_paths = " -> ".join(
            f"keyframes/{s['video_id']}/{ev.get('keyframe_n', 1):03d}.jpg"
            for ev in s.get("event_dossiers", [])
        )
        print(
            f"{s['rank']:<5} {s['submission_string']:<45} "
            f"{f_score:<9.4f} {dp_sc:<8.4f} {narr_sc:<11.4f} {'✅' if is_mono else '❌'}"
        )
        if img_paths:
            print(f"      📸 Paths: {img_paths}")

    cards_to_show = sequences[:explain_count]
    for top in cards_to_show:
        rank_lbl = f"TOP {top['rank']}" if top["rank"] == 1 else f"RANK #{top['rank']}"
        print(f"\n{'═' * 95}")
        print(f"🌟 {rank_lbl} TRAKE EXPLAINABILITY CARD")
        print(f"{'═' * 95}")
        print(f"  Video:             {top['video_id']}")
        print(f"  Submit:            {top['submission_string']}")
        print(
            f"  Final Score:       {top.get('final_score', top.get('sequence_score', 0.0)):.4f}  (DP: {top.get('dp_score', top.get('sequence_score', 0.0)):.4f} | Narrative: {top.get('narrative_score', 0.0):.4f})"
        )
        if top.get("narrative_reasoning"):
            print(f'  Narrative Judge:   "{top["narrative_reasoning"]}"')
        if top.get("audio_span"):
            print(
                f'\n  🎙️ MACRO-SPAN AUDIO TRANSCRIPT ({min(top["timestamps"]):.1f}s → {max(top["timestamps"]):.1f}s):\n     "{top["audio_span"]}"'
            )

        print("\n  🎬 CHRONOLOGICAL EVENT FRAMES:")
        for ev in top.get("event_dossiers", []):
            print(
                f"     - Event E{ev['event_idx']}: Frame {ev['frame_idx']} "
                f"({ev['pts_time_s']:.1f}s | #{ev.get('keyframe_n', 1):03d}.jpg)  "
                f"(vis_sim={ev.get('score_vis', 0.0):.3f})"
            )
        print(f"{'═' * 95}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Core pipeline runner
# ──────────────────────────────────────────────────────────────────────────────
def run_query(query: str, task: str, top_k: int = 10, explain_all: bool = False):
    _init_engine()

    task = task.upper()
    if task not in ("KIS", "VQA", "TRAKE"):
        print(f"❌ Unknown task type: {task}. Must be KIS, VQA, or TRAKE.")
        return

    t0 = time.perf_counter()
    print(f'\n🔍 Query: "{query}"')
    print(f"📋 Task:  {task}")

    # Phase 1: Parse
    parsed = _parser.parse(query, task_type=task, engine="gemini")
    print(f"   ✓ Query parsed ({parsed.task_type}, scene_en={parsed.global_scene_en[:60]}...)")

    explain_count = top_k

    if task in ("KIS", "VQA"):
        # Phase 3: Fusion
        pool = _fusion.retrieve_and_fuse(parsed, top_k_pool=300, branch_limit=500)
        print(f"   ✓ Stage 1 fusion complete ({len(pool)} candidates)")

        # Phase 4: Stage 2
        if task == "KIS":
            results = _reranker.rerank_kis(parsed, pool, final_top_k=top_k)
            dt = time.perf_counter() - t0
            print(f"   ✓ Stage 2 re-ranking complete in {dt:.1f}s")
            _print_kis_results(results, explain_count=explain_count)
        else:
            results = _reranker.rerank_vqa(parsed, pool, final_top_k=top_k)
            dt = time.perf_counter() - t0
            print(f"   ✓ Stage 2 re-ranking + VQA extraction complete in {dt:.1f}s")
            _print_vqa_results(results, explain_count=explain_count)

    elif task == "TRAKE":
        # Decompose into per-event queries
        if not parsed.trake_events:
            print("❌ No TRAKE events detected in query. Use 'E1: ... E2: ...' format.")
            return

        event_queries = []
        event_pools = []
        for i, ev in enumerate(parsed.trake_events, 1):
            sub = ParsedQuery(
                task_type="KIS",
                original_query=ev.description,
                global_scene_en=ev.scene_en,
                objects_en=ev.objects_en,
                speech_vi=ev.speech_vi,
                ocr_keywords=ev.ocr_keywords,
                weights={"vis": 0.35, "dam": 0.30, "asr": 0.35, "ocr": 0.00},
            )
            event_queries.append(sub)
            pool = _fusion.retrieve_and_fuse(sub, top_k_pool=100, branch_limit=500)
            event_pools.append(pool)
            print(f"   ✓ Event E{i} fused ({len(pool)} candidates)")

        raw_sequences = _reranker.solve_trake_video_guided_dp(
            event_queries=event_queries,
            candidate_pools=event_pools,
            searcher=_searcher,
            top_n_videos=10,
            final_top_k=top_k,
        )
        print(f"   ✓ TRAKE DP pathfinding complete ({len(raw_sequences)} candidate sequences)")

        # Stage 3 & 4: Macro-Span Audio Narrative Reranker
        event_descs = [ev.description for ev in parsed.trake_events]
        sequences = _reranker.rerank_trake_sequences(
            event_descriptions=event_descs,
            candidate_sequences=raw_sequences,
            searcher=_searcher,
            final_top_k=top_k,
        )
        dt = time.perf_counter() - t0
        print(f"   ✓ TRAKE Narrative Reranking complete in {dt:.1f}s")
        _print_trake_results(sequences, explain_count=explain_count)


def interactive_loop():
    print("=" * 60)
    print("  AIC Online Retrieval Engine — Interactive Terminal")
    print("=" * 60)
    print("Commands:")
    print("  Type a query and press Enter")
    print("  /task KIS|VQA|TRAKE  — switch task type (default: KIS)")
    print("  /top N               — set number of results (default: 10)")
    print("  /explain on|off      — toggle full explainability for all top results")
    print("  /quit                — exit")
    print()

    current_task = "KIS"
    current_top_k = 10
    explain_all = False

    while True:
        try:
            user_input = input(f"[{current_task}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Bye!")
            break

        if user_input.lower().startswith("/task"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2 and parts[1].upper() in ("KIS", "VQA", "TRAKE"):
                current_task = parts[1].upper()
                print(f"✅ Task set to: {current_task}")
            else:
                print("Usage: /task KIS|VQA|TRAKE")
            continue

        if user_input.lower().startswith("/top"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2 and parts[1].isdigit():
                current_top_k = int(parts[1])
                print(f"✅ Top-K set to: {current_top_k}")
            else:
                print("Usage: /top N")
            continue

        if user_input.lower().startswith("/explain"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2 and parts[1].lower() in ("on", "all", "true", "1"):
                explain_all = True
                print("✅ Full explainability enabled for ALL top candidates")
            else:
                explain_all = False
                print("✅ Explainability set to Top 1 only")
            continue

        run_query(user_input, current_task, current_top_k, explain_all=explain_all)


def main():
    ap = argparse.ArgumentParser(description="AIC Online Retrieval Engine CLI")
    ap.add_argument(
        "--query", "-q", type=str, default=None, help="Query text (runs once and exits)"
    )
    ap.add_argument(
        "--task", "-t", type=str, default="KIS", choices=["KIS", "VQA", "TRAKE"], help="Task type"
    )
    ap.add_argument("--top", "-k", type=int, default=10, help="Number of results")
    ap.add_argument(
        "--explain-all",
        "-e",
        action="store_true",
        help="Print full explainability cards for all top results",
    )
    args = ap.parse_args()

    if args.query:
        run_query(args.query, args.task, args.top, explain_all=args.explain_all)
    else:
        interactive_loop()


if __name__ == "__main__":
    main()
