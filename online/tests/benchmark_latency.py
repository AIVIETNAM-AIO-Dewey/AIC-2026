"""Benchmark search latency across multiple query executions."""

from __future__ import annotations

import time
import numpy as np
from online.src.retrieval.pipeline import VideoRetrievalEngine


def main():
    print("=" * 80)
    print(" ⚡ BENCHMARKING MULTIMODAL SEARCH ENGINE LATENCY")
    print("=" * 80)

    engine = VideoRetrievalEngine(
        qdrant_db_path="/tmp/qdrant_test_db",
        keyframes_root="/Users/khoale/Downloads/AIC_Challenger/data/keyframes",
    )

    test_queries = [
        "người đàn ông mặc áo sơ mi xanh trong trường quay",
        "quả bóng lớn màu vàng cam phản chiếu ánh sáng",
        "ngôi nhà có mái màu xanh lá cây và hàng rào màu trắng",
        "xe ô tô màu trắng biển số xanh",
        "bản tin sáu mươi giây của đài truyền hình",
    ]

    latencies = []

    # Warmup
    engine.search(test_queries[0], task_type="KIS", top_k=20)

    for i in range(15):
        q = test_queries[i % len(test_queries)]
        t0 = time.perf_counter()
        res = engine.search(q, task_type="KIS", top_k=50)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
        print(f"Query {i+1:02d}: {dt:.2f} ms (Found {len(res.results)} results)")

    latencies = np.array(latencies)
    print("=" * 80)
    print(f"📊 Latency Statistics (50 Candidates Returned):")
    print(f"   • Mean Latency:  {latencies.mean():.2f} ms")
    print(f"   • Median (P50):  {np.median(latencies):.2f} ms")
    print(f"   • P95 Latency:   {np.percentile(latencies, 95):.2f} ms")
    print(f"   • Min Latency:   {latencies.min():.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
