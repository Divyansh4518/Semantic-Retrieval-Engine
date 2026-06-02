"""Phase 2 Micro-Benchmark — FaissHNSWIndex vs FaissFlatIndex.

Metrics reported:
    1. Flat build time (s)
    2. HNSW build time (s)
    3. Flat p50 query latency (ms)
    4. HNSW p50 query latency (ms)
    5. Recall@5
    6. Recall@10
    7. HNSW speedup factor (Flat_p50 / HNSW_p50)

Invariants:
    - HNSW latency > 0 (no silent skips)
    - Recall@5  >= 0.90
    - Recall@10 >= 0.90

Run:
    uv run tests/test_hnsw_compare.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.index.faiss_flat import FaissFlatIndex  # noqa: E402
from src.index.faiss_hnsw import FaissHNSWIndex  # noqa: E402
from src.models import Document  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class PerQueryResult(NamedTuple):
    latency_s: float
    results: list[tuple[Document, float]]


def _make_documents(vectors: np.ndarray, prefix: str = "doc") -> list[Document]:
    return [
        Document(id=f"{prefix}-{i}", text="", embedding=vectors[i])
        for i in range(vectors.shape[0])
    ]


def _timed_build(index, documents: list[Document]) -> float:
    """Return wall-clock build time in seconds."""
    t0 = time.perf_counter()
    index.add_documents(documents)
    return time.perf_counter() - t0


def _sequential_queries(
    index,
    query_vectors: np.ndarray,
    k: int,
) -> tuple[list[list[tuple[Document, float]]], list[float]]:
    """Run sequential queries; return (all_results, latencies_seconds)."""
    all_results: list[list[tuple[Document, float]]] = []
    latencies: list[float] = []
    for i in range(query_vectors.shape[0]):
        t0 = time.perf_counter()
        results = index.search(query_vectors[i], k=k)
        latencies.append(time.perf_counter() - t0)
        all_results.append(results)
    return all_results, latencies


def _recall_at_k(
    ann_results: list[tuple[Document, float]],
    exact_results: list[tuple[Document, float]],
    k: int,
) -> float:
    """Fraction of exact top-k doc IDs present in ANN top-k."""
    exact_ids = {doc.id for doc, _ in exact_results[:k]}
    if not exact_ids:
        return 1.0
    ann_ids = {doc.id for doc, _ in ann_results[:k]}
    return len(exact_ids & ann_ids) / len(exact_ids)


def _percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(np.array(values, dtype=float), pct))


def _run_invariant(label: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main() -> None:
    N_CORPUS   = 1000
    N_QUERIES  = 10
    DIM        = 128
    K_SMALL    = 5
    K_LARGE    = 10
    M          = 32
    EF_CONSTR  = 200
    EF_SEARCH  = 64
    SEED       = 42

    rng = np.random.default_rng(SEED)
    corpus_vectors = rng.standard_normal((N_CORPUS, DIM)).astype(np.float32)
    query_vectors  = rng.standard_normal((N_QUERIES, DIM)).astype(np.float32)
    documents      = _make_documents(corpus_vectors)

    print("=" * 64)
    print("PHASE 2 MICRO-BENCHMARK — FaissHNSWIndex vs FaissFlatIndex")
    print("=" * 64)
    print(f"  Corpus : N={N_CORPUS}, dim={DIM}")
    print(f"  Queries: {N_QUERIES}")
    print(f"  HNSW   : M={M}, ef_construction={EF_CONSTR}, ef_search={EF_SEARCH}")

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    print("\n--- BUILD ---")

    flat_idx  = FaissFlatIndex()
    flat_build_s = _timed_build(flat_idx, documents)
    print(f"  Flat  build time : {flat_build_s * 1000:.3f} ms")

    hnsw_idx  = FaissHNSWIndex(M=M, ef_construction=EF_CONSTR, ef_search=EF_SEARCH)
    hnsw_build_s = _timed_build(hnsw_idx, documents)
    print(f"  HNSW  build time : {hnsw_build_s * 1000:.3f} ms")
    print(f"  Build overhead   : {hnsw_build_s / flat_build_s:.1f}x slower to build (expected for ANN)")

    # ------------------------------------------------------------------
    # Query — k=10 (superset; we'll slice for k=5 recall too)
    # ------------------------------------------------------------------
    print("\n--- QUERY (sequential) ---")

    flat_results_10, flat_latencies = _sequential_queries(flat_idx, query_vectors, k=K_LARGE)
    hnsw_results_10, hnsw_latencies = _sequential_queries(hnsw_idx, query_vectors, k=K_LARGE)

    flat_p50_ms = _percentile(flat_latencies, 50) * 1000
    flat_p95_ms = _percentile(flat_latencies, 95) * 1000
    flat_avg_ms = float(np.mean(flat_latencies)) * 1000

    hnsw_p50_ms = _percentile(hnsw_latencies, 50) * 1000
    hnsw_p95_ms = _percentile(hnsw_latencies, 95) * 1000
    hnsw_avg_ms = float(np.mean(hnsw_latencies)) * 1000

    speedup = flat_p50_ms / hnsw_p50_ms if hnsw_p50_ms > 0 else float("inf")

    print(f"  {'Metric':<28} {'Flat':>12} {'HNSW':>12}")
    print(f"  {'-'*52}")
    print(f"  {'p50 latency (ms)':<28} {flat_p50_ms:>12.4f} {hnsw_p50_ms:>12.4f}")
    print(f"  {'p95 latency (ms)':<28} {flat_p95_ms:>12.4f} {hnsw_p95_ms:>12.4f}")
    print(f"  {'avg latency (ms)':<28} {flat_avg_ms:>12.4f} {hnsw_avg_ms:>12.4f}")
    print(f"\n  HNSW speedup factor (Flat_p50 / HNSW_p50) : {speedup:.3f}x")

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------
    print("\n--- RECALL (treating FaissFlatIndex as ground truth) ---")

    recalls_5  = [
        _recall_at_k(hnsw_results_10[i], flat_results_10[i], K_SMALL)
        for i in range(N_QUERIES)
    ]
    recalls_10 = [
        _recall_at_k(hnsw_results_10[i], flat_results_10[i], K_LARGE)
        for i in range(N_QUERIES)
    ]

    mean_recall_5  = float(np.mean(recalls_5))
    mean_recall_10 = float(np.mean(recalls_10))

    print(f"  Per-query Recall@{K_SMALL} : {[round(r, 3) for r in recalls_5]}")
    print(f"  Per-query Recall@{K_LARGE}: {[round(r, 3) for r in recalls_10]}")
    print(f"  Mean Recall@{K_SMALL}      : {mean_recall_5:.4f}")
    print(f"  Mean Recall@{K_LARGE}     : {mean_recall_10:.4f}")

    # ------------------------------------------------------------------
    # Latency direction analysis
    # ------------------------------------------------------------------
    print("\n--- LATENCY DIRECTION ANALYSIS ---")
    if hnsw_p50_ms >= flat_p50_ms:
        print(
            f"  NOTE: HNSW p50 ({hnsw_p50_ms:.4f} ms) >= Flat p50 ({flat_p50_ms:.4f} ms).")
        print( "  This is EXPECTED at N=1000.")
        print( "  Reason: At small corpus sizes the HNSW graph traversal overhead")
        print( "  (pointer-chasing through C++ heap-allocated layer structures) is")
        print( "  NOT amortized over enough candidates to beat a single BLAS SGEMM")
        print( "  call on the full 1000-vector matrix. The crossover where HNSW wins")
        print( "  on latency typically occurs between N=5k and N=50k for dim=128.")
        print( "  Sweep H will quantify the exact crossover point.")
    else:
        print(
            f"  HNSW p50 ({hnsw_p50_ms:.4f} ms) < Flat p50 ({flat_p50_ms:.4f} ms). "
            f"  Speedup = {speedup:.3f}x."
        )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("PHASE 2 SUMMARY TABLE")
    print("=" * 64)
    print(f"  1. Flat  build time   : {flat_build_s * 1000:.3f} ms")
    print(f"  2. HNSW  build time   : {hnsw_build_s * 1000:.3f} ms")
    print(f"  3. Flat  p50 latency  : {flat_p50_ms:.4f} ms")
    print(f"  4. HNSW  p50 latency  : {hnsw_p50_ms:.4f} ms")
    print(f"  5. Recall@5           : {mean_recall_5:.4f}")
    print(f"  6. Recall@10          : {mean_recall_10:.4f}")
    print(f"  7. HNSW speedup       : {speedup:.3f}x")

    # ------------------------------------------------------------------
    # Invariant checks
    # ------------------------------------------------------------------
    print("\n--- INVARIANT CHECKS ---")
    all_ok = True

    all_ok &= _run_invariant(
        "HNSW latency > 0 (no silent skips)",
        hnsw_p50_ms > 0,
    )
    all_ok &= _run_invariant(
        f"Recall@5  >= 90%  (got {mean_recall_5:.1%})",
        mean_recall_5 >= 0.90,
    )
    all_ok &= _run_invariant(
        f"Recall@10 >= 90%  (got {mean_recall_10:.1%})",
        mean_recall_10 >= 0.90,
    )

    print("\n" + "=" * 64)
    if all_ok:
        print("RESULT: ALL PHASE 2 INVARIANTS PASSED (OK)")
        sys.exit(0)
    else:
        print("RESULT: PHASE 2 INVARIANTS FAILED — INVESTIGATE BEFORE PROCEEDING")
        sys.exit(1)


if __name__ == "__main__":
    main()
