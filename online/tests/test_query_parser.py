"""Comprehensive comparison test: Gemini 3.6 Flash vs. Local Qwen 2.5 1.5B."""

from __future__ import annotations

import json
import time

from online.src.retrieval.query_parser import QueryParser

TEST_QUERIES = [
    {
        "id": "Test 1 (VQA)",
        "task_type": "VQA",
        "query": "Đoạn video về một người phụ nữ dạy nấu ăn cho những người khác. Trong đoạn video có thể thấy một người đang cầm công thức món ăn với nguyên liệu chính là 200g thịt nạc xay. Hỏi tiêu đề của công thức nấu ăn (tên món ăn) này là gì?",
    },
    {
        "id": "Test 2 (KIS)",
        "task_type": "KIS",
        "query": "Tìm một đoạn video đua xe đạp, góc quay từ flycam trên cao, một vận động viên mặc áo xanh dương, trắng đang vượt ba vận động viên khác và lên vị trí dẫn đầu. Biết sau đó vận động viên này dẫn đầu suốt đoạn đường còn lại đến đích.",
    },
    {
        "id": "Test 3 (TRAKE)",
        "task_type": "TRAKE",
        "query": """E1: Khoảnh khắc đầu tiên bột được bỏ vào tô măng tây.
E2: Khoảnh khắc đầu tiên thấy miến măng tây đầu tiên tiếp xúc với dầu trong chảo.
E3: Khoảnh khắc miếng măng tây đầu tiên rời khỏi chảo dầu.
E4: Khoảng khắc miếng măng tây cuối cùng rời chảo dầu và nằm hoàn toàn trên dĩa.""",
    },
]


def run_comparison():
    parser = QueryParser(gemini_model_id="gemini-3.6-flash", qwen_model_id="qwen2.5:1.5b")

    print("=" * 80)
    print("🚀 COMPARISON BENCHMARK: GEMINI 3.6 FLASH vs. LOCAL QWEN 2.5 (1.5B)")
    print("=" * 80)

    for item in TEST_QUERIES:
        print("\n================================================================================")
        print(f"📌 {item['id']}")
        print(f"📝 Raw Query: {item['query'][:80]}...")
        print("================================================================================")

        # 1. Test Gemini 3.6 Flash
        t0 = time.perf_counter()
        gemini_parsed = parser.parse(item["query"], task_type=item["task_type"], engine="gemini")
        gemini_time = (time.perf_counter() - t0) * 1000.0

        print(f"\n🔹 [GEMINI 3.6 FLASH] Latency: {gemini_time:.1f}ms")
        print(json.dumps(gemini_parsed.model_dump(), indent=2, ensure_ascii=False))

        # 2. Test Local Qwen 2.5 1.5B
        t0 = time.perf_counter()
        qwen_parsed = parser.parse(item["query"], task_type=item["task_type"], engine="qwen")
        qwen_time = (time.perf_counter() - t0) * 1000.0

        print(f"\n🔸 [LOCAL QWEN 2.5 1.5B] Latency: {qwen_time:.1f}ms")
        print(json.dumps(qwen_parsed.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_comparison()
