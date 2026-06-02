"""Orchestrator and CLI for NSW benchmark sweeps.

Usage::

    python -m benchmarks.runner sweep-a
    python -m benchmarks.runner sweep-b --seed 123
    python -m benchmarks.runner all

Every output JSON is stamped with ``timestamp``, ``git_commit``,
``seed``, and full hyperparameters.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ------------------------------------------------------------------
# Ensure project root is on sys.path so both ``src`` and
# ``benchmarks`` packages resolve correctly.
# ------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.index.exact import ExactIndex  # noqa: E402
from src.index.faiss_flat import FaissFlatIndex  # noqa: E402
from src.index.faiss_hnsw import FaissHNSWIndex  # noqa: E402
from src.index.graph import GraphIndex  # noqa: E402
from src.models import Document  # noqa: E402

# ------------------------------------------------------------------
# Fixed HNSW hyperparameters used in Sweeps A, C, D, H.
# Sweeps B/C sweep M / ef_search independently.
# ------------------------------------------------------------------
_HNSW_M = 32
_HNSW_EF_CONSTRUCTION = 200
_HNSW_EF_SEARCH = 64

from benchmarks.datasets import (  # noqa: E402
    generate_at_dimension,
    generate_clustered,
    generate_uniform,
)
from benchmarks.telemetry import (  # noqa: E402
    analyze_graph,
    capture_search_diagnostics,
    compute_match_rate,
    compute_query_percentiles,
    compute_recall,
    measure_memory,
)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
_BENCHMARKS_DIR = Path(__file__).resolve().parent
_RAW_DIR = _BENCHMARKS_DIR / "outputs" / "raw"
_AGG_DIR = _BENCHMARKS_DIR / "outputs" / "aggregated"

_RAW_DIR.mkdir(parents=True, exist_ok=True)
_AGG_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _git_commit() -> str:
    """Return the current short git commit hash, or ``'unknown'``."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(_PROJECT_ROOT),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _metadata(seed: int, **hyperparams: Any) -> dict[str, Any]:
    """Build the metadata block stamped on every output JSON."""
    return {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "git_commit": _git_commit(),
        "seed": seed,
        "hyperparameters": hyperparams,
    }


def _vectors_to_documents(vectors: np.ndarray) -> list[Document]:
    """Wrap an ``(N, dim)`` array into a list of ``Document`` objects."""
    return [
        Document(id=f"doc-{i}", text="", embedding=vectors[i])
        for i in range(vectors.shape[0])
    ]


def _build_indices(
    vectors: np.ndarray,
    M: int,
    ef_construction: int,
    ef_search: int,
) -> tuple[GraphIndex, ExactIndex, float]:
    """Build both indices.  Returns ``(graph_idx, exact_idx, build_time_s)``."""
    documents = _vectors_to_documents(vectors)

    exact_idx = ExactIndex()
    exact_idx.add_documents(documents)

    graph_idx = GraphIndex(M=M, ef_construction=ef_construction, ef_search=ef_search)

    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        graph_idx.add_documents(documents)
    build_time = time.perf_counter() - t0

    return graph_idx, exact_idx, build_time


def _build_all_indices(
    vectors: np.ndarray,
    M: int,
    ef_construction: int,
    ef_search: int,
) -> tuple[GraphIndex, ExactIndex, FaissFlatIndex, float, float, float]:
    """Build all three indices.

    Returns ``(graph_idx, exact_idx, faiss_idx,
               graph_build_s, exact_build_s, faiss_build_s)``.
    """
    documents = _vectors_to_documents(vectors)

    exact_idx = ExactIndex()
    t0 = time.perf_counter()
    exact_idx.add_documents(documents)
    exact_build = time.perf_counter() - t0

    faiss_idx = FaissFlatIndex()
    t0 = time.perf_counter()
    faiss_idx.add_documents(documents)
    faiss_build = time.perf_counter() - t0

    graph_idx = GraphIndex(M=M, ef_construction=ef_construction, ef_search=ef_search)
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        graph_idx.add_documents(documents)
    graph_build = time.perf_counter() - t0

    return graph_idx, exact_idx, faiss_idx, graph_build, exact_build, faiss_build


def _build_flat_indices(
    vectors: np.ndarray,
) -> tuple[ExactIndex, FaissFlatIndex, float, float]:
    """Build only ExactIndex and FaissFlatIndex (no GraphIndex).

    Returns ``(exact_idx, faiss_idx, exact_build_s, faiss_build_s)``.
    """
    documents = _vectors_to_documents(vectors)

    exact_idx = ExactIndex()
    t0 = time.perf_counter()
    exact_idx.add_documents(documents)
    exact_build = time.perf_counter() - t0

    faiss_idx = FaissFlatIndex()
    t0 = time.perf_counter()
    faiss_idx.add_documents(documents)
    faiss_build = time.perf_counter() - t0

    return exact_idx, faiss_idx, exact_build, faiss_build


def _build_all_indices_with_hnsw(
    vectors: np.ndarray,
    graph_M: int,
    graph_ef_construction: int,
    graph_ef_search: int,
    hnsw_M: int = _HNSW_M,
    hnsw_ef_construction: int = _HNSW_EF_CONSTRUCTION,
    hnsw_ef_search: int = _HNSW_EF_SEARCH,
) -> tuple[
    GraphIndex, ExactIndex, FaissFlatIndex, FaissHNSWIndex,
    float, float, float, float,
]:
    """Build all four indices: Graph, Exact, FaissFlat, FaissHNSW.

    Returns
    -------
    tuple
        ``(graph_idx, exact_idx, faiss_idx, hnsw_idx,
           graph_build_s, exact_build_s, faiss_build_s, hnsw_build_s)``
    """
    documents = _vectors_to_documents(vectors)

    exact_idx = ExactIndex()
    t0 = time.perf_counter()
    exact_idx.add_documents(documents)
    exact_build = time.perf_counter() - t0

    faiss_idx = FaissFlatIndex()
    t0 = time.perf_counter()
    faiss_idx.add_documents(documents)
    faiss_build = time.perf_counter() - t0

    hnsw_idx = FaissHNSWIndex(
        M=hnsw_M,
        ef_construction=hnsw_ef_construction,
        ef_search=hnsw_ef_search,
    )
    t0 = time.perf_counter()
    hnsw_idx.add_documents(documents)
    hnsw_build = time.perf_counter() - t0

    graph_idx = GraphIndex(
        M=graph_M,
        ef_construction=graph_ef_construction,
        ef_search=graph_ef_search,
    )
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        graph_idx.add_documents(documents)
    graph_build = time.perf_counter() - t0

    return (
        graph_idx, exact_idx, faiss_idx, hnsw_idx,
        graph_build, exact_build, faiss_build, hnsw_build,
    )


def _query_flat_index(
    index: ExactIndex | FaissFlatIndex,
    query_vectors: np.ndarray,
    k: int = 10,
) -> tuple[list[list[tuple]], list[float]]:
    """Run sequential queries against a flat index.

    Returns ``(all_results, latencies)``.
    """
    all_results: list[list[tuple]] = []
    latencies: list[float] = []
    for qi in range(query_vectors.shape[0]):
        qvec = query_vectors[qi]
        t0 = time.perf_counter()
        results = index.search(qvec, k=k)
        latency = time.perf_counter() - t0
        all_results.append(results)
        latencies.append(latency)
    return all_results, latencies


def _run_queries(
    graph_idx: GraphIndex,
    exact_idx: ExactIndex,
    query_vectors: np.ndarray,
    k: int = 10,
) -> tuple[list[dict], list[float], list[float]]:
    """Execute *query_vectors* against both indices.

    Returns ``(per_query_logs, latencies, recalls)``.
    """
    per_query_logs: list[dict] = []
    latencies: list[float] = []
    recalls: list[float] = []

    for qi in range(query_vectors.shape[0]):
        qvec = query_vectors[qi]

        t0 = time.perf_counter()
        diag = capture_search_diagnostics(graph_idx, qvec, k=k)
        latency = time.perf_counter() - t0

        graph_results = diag["results"]

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exact_results = exact_idx.search(qvec, k=k)

        recall = compute_recall(graph_results, exact_results, k)

        latencies.append(latency)
        recalls.append(recall)

        per_query_logs.append(
            {
                "query_index": qi,
                "latency_s": round(latency, 6),
                "recall": round(recall, 4),
                "nodes_evaluated": diag.get("nodes_evaluated"),
                "entry_node": diag.get("entry_node"),
            }
        )

    return per_query_logs, latencies, recalls


def _save_outputs(
    sweep_name: str,
    raw_logs: list[dict],
    summary: dict[str, Any],
) -> None:
    """Write per-query JSONL to ``raw/`` and summary JSON to ``aggregated/``."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_path = _RAW_DIR / f"{sweep_name}_{ts}.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for entry in raw_logs:
            f.write(json.dumps(entry) + "\n")

    agg_path = _AGG_DIR / f"{sweep_name}_{ts}.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"  -> Raw logs: {raw_path}")
    print(f"  -> Summary:  {agg_path}")


# ------------------------------------------------------------------
# Sweep A — Scaling (N sweep)
# ------------------------------------------------------------------


def sweep_a(seed: int = 42) -> None:
    """Fix M=8, ef_search=64.  Sweep N over [100, 500, 1500, 5000].

    FaissHNSWIndex is built with fixed hyperparameters (_HNSW_M=32,
    _HNSW_EF_CONSTRUCTION=200, _HNSW_EF_SEARCH=64) alongside the
    existing GraphIndex / ExactIndex / FaissFlatIndex.
    """
    print("\n" + "=" * 60)
    print("SWEEP A -- Scaling (N sweep)")
    print("=" * 60)

    M, ef_construction, ef_search, dim = 8, 32, 64, 128
    n_values = [100, 500, 1500, 5000]
    n_queries = 100
    k = 10

    all_raw: list[dict] = []
    sweep_results: list[dict] = []

    for N in n_values:
        print(f"\n  N={N} ...")
        vectors = generate_uniform(N, dim, seed=seed)
        query_vectors = generate_uniform(n_queries, dim, seed=seed + 1)

        graph_idx: GraphIndex | None = None
        exact_idx: ExactIndex | None = None
        faiss_idx: FaissFlatIndex | None = None
        hnsw_idx: FaissHNSWIndex | None = None
        graph_build = 0.0
        exact_build = 0.0
        faiss_build = 0.0
        hnsw_build = 0.0

        def _do_build() -> None:
            nonlocal graph_idx, exact_idx, faiss_idx, hnsw_idx
            nonlocal graph_build, exact_build, faiss_build, hnsw_build
            (
                graph_idx, exact_idx, faiss_idx, hnsw_idx,
                graph_build, exact_build, faiss_build, hnsw_build,
            ) = _build_all_indices_with_hnsw(vectors, M, ef_construction, ef_search)

        mem = measure_memory(_do_build)

        assert (
            graph_idx is not None
            and exact_idx is not None
            and faiss_idx is not None
            and hnsw_idx is not None
        )

        graph_health = analyze_graph(graph_idx)
        per_query, latencies, recalls = _run_queries(
            graph_idx, exact_idx, query_vectors, k=k
        )
        percentiles = compute_query_percentiles(latencies)

        # FaissFlatIndex sequential queries
        faiss_results_list, faiss_latencies = _query_flat_index(
            faiss_idx, query_vectors, k=k
        )
        faiss_percentiles = compute_query_percentiles(faiss_latencies)

        # ExactIndex sequential queries (timing comparison baseline)
        exact_results_list, exact_latencies = _query_flat_index(
            exact_idx, query_vectors, k=k
        )
        exact_percentiles = compute_query_percentiles(exact_latencies)

        # FaissHNSWIndex sequential queries
        hnsw_results_list, hnsw_latencies = _query_flat_index(
            hnsw_idx, query_vectors, k=k
        )
        hnsw_percentiles = compute_query_percentiles(hnsw_latencies)

        # Match rate: ExactIndex vs FaissFlatIndex (unchanged)
        match_rates = [
            compute_match_rate(er, fr)
            for er, fr in zip(exact_results_list, faiss_results_list)
        ]
        mean_match_rate = round(float(np.mean(match_rates)), 4)

        # HNSW recall: compare against FaissFlatIndex ground truth
        hnsw_recalls = [
            compute_recall(hnsw_results_list[qi], faiss_results_list[qi], k)
            for qi in range(n_queries)
        ]
        hnsw_recall_mean = round(float(np.mean(hnsw_recalls)), 4)

        meta = _metadata(
            seed, M=M, ef_construction=ef_construction,
            ef_search=ef_search, N=N, dim=dim,
            hnsw_M=_HNSW_M, hnsw_ef_construction=_HNSW_EF_CONSTRUCTION,
            hnsw_ef_search=_HNSW_EF_SEARCH,
        )

        for entry in per_query:
            entry.update(meta)
            entry["N"] = N
        all_raw.extend(per_query)

        point = {
            **meta,
            "N": N,
            # --- build times ---
            "graph_build_time_s": round(graph_build, 4),
            "exact_build_time_s": round(exact_build, 4),
            "faiss_build_time_s": round(faiss_build, 4),
            "faiss_hnsw_build_time_s": round(hnsw_build, 4),
            "build_time_s": round(graph_build, 4),   # back-compat alias
            # --- memory (whole build closure) ---
            "memory": mem,
            # --- graph topology ---
            "graph_health": graph_health,
            # --- query latencies ---
            "query_latency": percentiles,
            "exact_query_latency": exact_percentiles,
            "faiss_query_latency": faiss_percentiles,
            "faiss_hnsw_query_latency": hnsw_percentiles,
            # --- recall / match ---
            "recall_mean": round(float(np.mean(recalls)), 4),
            "faiss_hnsw_recall_mean": hnsw_recall_mean,
            "exact_vs_faiss_match_rate": mean_match_rate,
        }
        sweep_results.append(point)
        print(
            f"    Graph Build: {graph_build:.3f}s | "
            f"Exact Build: {exact_build:.4f}s | "
            f"FAISS Build: {faiss_build:.4f}s | "
            f"HNSW Build: {hnsw_build:.4f}s"
        )
        print(
            f"    Graph Recall: {np.mean(recalls):.4f} | "
            f"HNSW Recall: {hnsw_recall_mean:.4f} | "
            f"Match Rate: {mean_match_rate:.4f} | "
            f"HNSW QPS: {hnsw_percentiles['qps']:.0f}"
        )

    summary = {
        **_metadata(
            seed, M=M, ef_construction=ef_construction,
            ef_search=ef_search, dim=dim,
            hnsw_M=_HNSW_M, hnsw_ef_construction=_HNSW_EF_CONSTRUCTION,
            hnsw_ef_search=_HNSW_EF_SEARCH,
        ),
        "sweep": "A",
        "variable": "N",
        "results": sweep_results,
    }
    _save_outputs("sweep_a", all_raw, summary)


# ------------------------------------------------------------------
# Sweep B — Topology (M sweep)
# ------------------------------------------------------------------


def sweep_b(seed: int = 42) -> None:
    """Fix N=5000, ef_search=64.  Sweep M over [2, 4, 8, 16, 32].

    Both GraphIndex and FaissHNSWIndex use the swept M value so the
    effect of graph connectivity is directly comparable across
    implementations.  ef_construction=200 and ef_search=64 are fixed
    for HNSW; ef_construction=32 is kept for GraphIndex.
    """
    print("\n" + "=" * 60)
    print("SWEEP B -- Topology (M sweep)")
    print("=" * 60)

    N, ef_construction, ef_search, dim = 5000, 32, 64, 128
    hnsw_ef_construction = 200
    m_values = [2, 4, 8, 16, 32]
    n_queries = 100
    k = 10

    vectors = generate_uniform(N, dim, seed=seed)
    query_vectors = generate_uniform(n_queries, dim, seed=seed + 1)

    all_raw: list[dict] = []
    sweep_results: list[dict] = []

    for M in m_values:
        print(f"\n  M={M} ...")

        # --- GraphIndex ---
        graph_idx, exact_idx, graph_build = _build_indices(
            vectors, M, ef_construction, ef_search
        )
        graph_health = analyze_graph(graph_idx)
        per_query, latencies, recalls = _run_queries(
            graph_idx, exact_idx, query_vectors, k=k
        )
        graph_percentiles = compute_query_percentiles(latencies)

        # --- FaissHNSWIndex (same M, fixed ef params) ---
        documents = _vectors_to_documents(vectors)
        hnsw_idx = FaissHNSWIndex(
            M=M,
            ef_construction=hnsw_ef_construction,
            ef_search=ef_search,
        )
        t0 = time.perf_counter()
        hnsw_idx.add_documents(documents)
        hnsw_build = time.perf_counter() - t0

        # Ground truth for HNSW recall
        flat_results_list, _ = _query_flat_index(exact_idx, query_vectors, k=k)
        hnsw_results_list, hnsw_latencies = _query_flat_index(hnsw_idx, query_vectors, k=k)
        hnsw_percentiles = compute_query_percentiles(hnsw_latencies)
        hnsw_recalls = [
            compute_recall(hnsw_results_list[qi], flat_results_list[qi], k)
            for qi in range(n_queries)
        ]
        hnsw_recall_mean = round(float(np.mean(hnsw_recalls)), 4)

        meta = _metadata(
            seed, M=M, ef_construction=ef_construction,
            ef_search=ef_search, N=N, dim=dim,
            hnsw_ef_construction=hnsw_ef_construction,
        )

        for entry in per_query:
            entry.update(meta)
            entry["M"] = M
        all_raw.extend(per_query)

        point = {
            **meta,
            "M": M,
            "build_time_s": round(graph_build, 4),
            "faiss_hnsw_build_time_s": round(hnsw_build, 4),
            "graph_health": graph_health,
            "query_latency": graph_percentiles,
            "faiss_hnsw_query_latency": hnsw_percentiles,
            "recall_mean": round(float(np.mean(recalls)), 4),
            "faiss_hnsw_recall_mean": hnsw_recall_mean,
        }
        sweep_results.append(point)
        print(
            f"    Graph  -> Recall: {np.mean(recalls):.4f} | "
            f"Build: {graph_build:.3f}s | "
            f"p50: {graph_percentiles['p50']*1000:.3f}ms"
        )
        print(
            f"    HNSW   -> Recall: {hnsw_recall_mean:.4f} | "
            f"Build: {hnsw_build:.3f}s | "
            f"p50: {hnsw_percentiles['p50']*1000:.3f}ms"
        )

    summary = {
        **_metadata(
            seed, N=N, ef_construction=ef_construction,
            ef_search=ef_search, dim=dim,
            hnsw_ef_construction=hnsw_ef_construction,
        ),
        "sweep": "B",
        "variable": "M",
        "results": sweep_results,
    }
    _save_outputs("sweep_b", all_raw, summary)


# ------------------------------------------------------------------
# Sweep C — Traversal (ef_search sweep)
# ------------------------------------------------------------------


def sweep_c(seed: int = 42) -> None:
    """Fix N=5000, M=8.  Sweep ef_search over [8..512].

    Optimization: both GraphIndex and FaissHNSWIndex are built *once*
    at the maximum ef_search.  Each iteration mutates the search
    parameter in-place:
        * ``graph_idx.ef_search = ef_search``
        * ``hnsw_idx.set_ef_search(ef_search)``
    This avoids redundant O(N) graph construction per ef_search value.
    """
    print("\n" + "=" * 60)
    print("SWEEP C -- Traversal (ef_search sweep)")
    print("=" * 60)

    N, M, ef_construction, dim = 5000, 8, 32, 128
    ef_values = [8, 16, 32, 64, 128, 256, 512]
    n_queries = 100
    k = 10

    vectors = generate_uniform(N, dim, seed=seed)
    query_vectors = generate_uniform(n_queries, dim, seed=seed + 1)
    documents = _vectors_to_documents(vectors)

    # --- Build both indices once at max ef_search ---
    graph_idx, exact_idx, graph_build = _build_indices(
        vectors, M, ef_construction, max(ef_values)
    )

    hnsw_idx = FaissHNSWIndex(
        M=_HNSW_M,
        ef_construction=_HNSW_EF_CONSTRUCTION,
        ef_search=max(ef_values),
    )
    t0 = time.perf_counter()
    hnsw_idx.add_documents(documents)
    hnsw_build = time.perf_counter() - t0
    print(f"  Graph build: {graph_build:.3f}s | HNSW build: {hnsw_build:.3f}s")

    # Exact ground truth (used for HNSW recall computation)
    exact_results_list, _ = _query_flat_index(exact_idx, query_vectors, k=k)

    all_raw: list[dict] = []
    sweep_results: list[dict] = []

    for ef_search in ef_values:
        print(f"\n  ef_search={ef_search} ...")

        # Mutate search parameters in-place
        graph_idx.ef_search = ef_search
        hnsw_idx.set_ef_search(ef_search)

        # Graph queries
        per_query, graph_latencies, graph_recalls = _run_queries(
            graph_idx, exact_idx, query_vectors, k=k
        )
        graph_percentiles = compute_query_percentiles(graph_latencies)

        # HNSW queries
        hnsw_results_list, hnsw_latencies = _query_flat_index(hnsw_idx, query_vectors, k=k)
        hnsw_percentiles = compute_query_percentiles(hnsw_latencies)
        hnsw_recalls = [
            compute_recall(hnsw_results_list[qi], exact_results_list[qi], k)
            for qi in range(n_queries)
        ]
        hnsw_recall_mean = round(float(np.mean(hnsw_recalls)), 4)

        meta = _metadata(
            seed, M=M, ef_construction=ef_construction,
            ef_search=ef_search, N=N, dim=dim,
            hnsw_M=_HNSW_M, hnsw_ef_construction=_HNSW_EF_CONSTRUCTION,
        )

        for entry in per_query:
            entry.update(meta)
            entry["ef_search"] = ef_search
        all_raw.extend(per_query)

        point = {
            **meta,
            "ef_search": ef_search,
            "build_time_s": round(graph_build, 4),
            "faiss_hnsw_build_time_s": round(hnsw_build, 4),
            "query_latency": graph_percentiles,
            "faiss_hnsw_query_latency": hnsw_percentiles,
            "recall_mean": round(float(np.mean(graph_recalls)), 4),
            "faiss_hnsw_recall_mean": hnsw_recall_mean,
        }
        sweep_results.append(point)
        print(
            f"    Graph  -> Recall: {np.mean(graph_recalls):.4f} | "
            f"p50: {graph_percentiles['p50']*1000:.3f}ms"
        )
        print(
            f"    HNSW   -> Recall: {hnsw_recall_mean:.4f} | "
            f"p50: {hnsw_percentiles['p50']*1000:.3f}ms"
        )

    summary = {
        **_metadata(
            seed, M=M, ef_construction=ef_construction, N=N, dim=dim,
            hnsw_M=_HNSW_M, hnsw_ef_construction=_HNSW_EF_CONSTRUCTION,
        ),
        "sweep": "C",
        "variable": "ef_search",
        "results": sweep_results,
    }
    _save_outputs("sweep_c", all_raw, summary)


# ------------------------------------------------------------------
# Sweep D — Escape Success Matrix
# ------------------------------------------------------------------


def sweep_d(seed: int = 42) -> None:
    """5000 vectors, 10 clusters.  Two 10x10 escape-success matrices.

    Runs the cross-cluster escape logic for both GraphIndex and
    FaissHNSWIndex, producing:
        * ``graph_escape_matrix``      (10x10)
        * ``faiss_hnsw_escape_matrix`` (10x10)
    """
    print("\n" + "=" * 60)
    print("SWEEP D -- Escape Success Matrix")
    print("=" * 60)

    N, M, ef_construction, ef_search, dim = 5000, 8, 32, 64, 128
    n_clusters = 10
    queries_per_source = 5
    k = 10

    vectors, assignments = generate_clustered(
        N, dim, n_clusters=n_clusters, seed=seed
    )
    documents = _vectors_to_documents(vectors)

    # --- Build Graph index ---
    graph_idx, exact_idx, graph_build = _build_indices(
        vectors, M, ef_construction, ef_search
    )

    # --- Build HNSW index (same corpus) ---
    hnsw_idx = FaissHNSWIndex(
        M=_HNSW_M,
        ef_construction=_HNSW_EF_CONSTRUCTION,
        ef_search=_HNSW_EF_SEARCH,
    )
    t0 = time.perf_counter()
    hnsw_idx.add_documents(documents)
    hnsw_build = time.perf_counter() - t0

    print(f"  Graph build: {graph_build:.3f}s | HNSW build: {hnsw_build:.3f}s")

    # Group document indices by cluster
    cluster_indices: dict[int, list[int]] = {}
    for i, c in enumerate(assignments):
        cluster_indices.setdefault(int(c), []).append(i)

    graph_escape_matrix = np.zeros((n_clusters, n_clusters), dtype=float)
    hnsw_escape_matrix  = np.zeros((n_clusters, n_clusters), dtype=float)
    all_raw: list[dict] = []

    rng = np.random.default_rng(seed + 100)

    for src_cluster in range(n_clusters):
        src_indices = cluster_indices.get(src_cluster, [])
        if not src_indices:
            continue

        sample_count = min(queries_per_source, len(src_indices))
        sampled = rng.choice(src_indices, size=sample_count, replace=False)

        for tgt_cluster in range(n_clusters):
            tgt_set = set(cluster_indices.get(tgt_cluster, []))
            graph_successes = 0
            hnsw_successes  = 0

            for src_idx in sampled:
                query_vec = vectors[src_idx]

                # --- Graph escape ---
                diag = capture_search_diagnostics(graph_idx, query_vec, k=k)
                graph_result_indices = [
                    int(doc.id.split("-")[1]) for doc, _ in diag["results"]
                ]
                graph_escaped = any(idx in tgt_set for idx in graph_result_indices)
                if graph_escaped:
                    graph_successes += 1

                # --- HNSW escape ---
                hnsw_results = hnsw_idx.search(query_vec, k=k)
                hnsw_result_indices = [
                    int(doc.id.split("-")[1]) for doc, _ in hnsw_results
                ]
                hnsw_escaped = any(idx in tgt_set for idx in hnsw_result_indices)
                if hnsw_escaped:
                    hnsw_successes += 1

                all_raw.append(
                    {
                        "source_cluster": src_cluster,
                        "target_cluster": tgt_cluster,
                        "query_doc_index": int(src_idx),
                        "graph_escaped": graph_escaped,
                        "hnsw_escaped": hnsw_escaped,
                        "graph_result_indices": graph_result_indices,
                        "hnsw_result_indices": hnsw_result_indices,
                    }
                )

            rate = lambda s: s / sample_count if sample_count > 0 else 0.0
            graph_escape_matrix[src_cluster, tgt_cluster] = rate(graph_successes)
            hnsw_escape_matrix[src_cluster, tgt_cluster]  = rate(hnsw_successes)

    def _print_matrix(label: str, mat: np.ndarray) -> None:
        print(f"\n  {label} (rows=source, cols=target):")
        header = "       " + "  ".join(f"C{c:02d}" for c in range(n_clusters))
        print(header)
        for src in range(n_clusters):
            row = f"  C{src:02d}  " + "  ".join(
                f"{mat[src, tgt]:.2f}" for tgt in range(n_clusters)
            )
            print(row)

    _print_matrix("Graph Escape Matrix", graph_escape_matrix)
    _print_matrix("HNSW  Escape Matrix", hnsw_escape_matrix)

    meta = _metadata(
        seed, M=M, ef_construction=ef_construction,
        ef_search=ef_search, N=N, dim=dim,
        hnsw_M=_HNSW_M, hnsw_ef_construction=_HNSW_EF_CONSTRUCTION,
        hnsw_ef_search=_HNSW_EF_SEARCH,
    )
    summary = {
        **meta,
        "sweep": "D",
        "variable": "cluster_escape",
        "n_clusters": n_clusters,
        "queries_per_source": queries_per_source,
        "k": k,
        "graph_build_time_s": round(graph_build, 4),
        "faiss_hnsw_build_time_s": round(hnsw_build, 4),
        # Legacy alias kept for backward compatibility
        "build_time_s": round(graph_build, 4),
        "graph_escape_matrix": graph_escape_matrix.tolist(),
        "faiss_hnsw_escape_matrix": hnsw_escape_matrix.tolist(),
        # Legacy alias — same as graph_escape_matrix
        "escape_matrix": graph_escape_matrix.tolist(),
    }
    _save_outputs("sweep_d", all_raw, summary)


# ------------------------------------------------------------------
# Sweep E — Dimensionality Curse
# ------------------------------------------------------------------


def sweep_e(seed: int = 42) -> None:
    """Fix N=5000, M=8, ef_search=64.  Sweep dim over [128..1536]."""
    print("\n" + "=" * 60)
    print("SWEEP E -- Dimensionality Curse")
    print("=" * 60)

    N, M, ef_construction, ef_search = 5000, 8, 32, 64
    dim_values = [128, 384, 768, 1536]
    n_queries = 100
    k = 10

    all_raw: list[dict] = []
    sweep_results: list[dict] = []

    for dim in dim_values:
        print(f"\n  dim={dim} ...")
        vectors = generate_at_dimension(N, dim, seed=seed)
        query_vectors = generate_at_dimension(n_queries, dim, seed=seed + 1)

        graph_idx, exact_idx, faiss_idx, graph_build, exact_build, faiss_build = (
            _build_all_indices(vectors, M, ef_construction, ef_search)
        )

        per_query, latencies, recalls = _run_queries(
            graph_idx, exact_idx, query_vectors, k=k
        )
        percentiles = compute_query_percentiles(latencies)

        # FAISS queries
        faiss_results_list, faiss_latencies = _query_flat_index(
            faiss_idx, query_vectors, k=k
        )
        faiss_percentiles = compute_query_percentiles(faiss_latencies)

        # Exact queries (timed)
        exact_results_list, exact_latencies = _query_flat_index(
            exact_idx, query_vectors, k=k
        )
        exact_percentiles = compute_query_percentiles(exact_latencies)

        match_rates = [
            compute_match_rate(er, fr)
            for er, fr in zip(exact_results_list, faiss_results_list)
        ]
        mean_match_rate = round(float(np.mean(match_rates)), 4)

        meta = _metadata(
            seed, M=M, ef_construction=ef_construction,
            ef_search=ef_search, N=N, dim=dim,
        )

        for entry in per_query:
            entry.update(meta)
            entry["dim"] = dim
        all_raw.extend(per_query)

        point = {
            **meta,
            "dim": dim,
            "build_time_s": round(graph_build, 4),
            "exact_build_time_s": round(exact_build, 4),
            "faiss_build_time_s": round(faiss_build, 4),
            "query_latency": percentiles,
            "exact_query_latency": exact_percentiles,
            "faiss_query_latency": faiss_percentiles,
            "recall_mean": round(float(np.mean(recalls)), 4),
            "exact_vs_faiss_match_rate": mean_match_rate,
        }
        sweep_results.append(point)
        print(
            f"    Graph Recall: {np.mean(recalls):.4f} | "
            f"Match Rate: {mean_match_rate:.4f} | "
            f"FAISS QPS: {faiss_percentiles['qps']:.0f}"
        )

    summary = {
        **_metadata(seed, M=M, ef_construction=ef_construction, ef_search=ef_search, N=N),
        "sweep": "E",
        "variable": "dim",
        "results": sweep_results,
    }
    _save_outputs("sweep_e", all_raw, summary)


# ------------------------------------------------------------------
# Sweep F — Implementation Scale (Exact vs FAISS)
# ------------------------------------------------------------------


def sweep_f(seed: int = 42) -> None:
    """Compare ExactIndex vs FaissFlatIndex.  Sweep N over [5k..50k]."""
    print("\n" + "=" * 60)
    print("SWEEP F -- Implementation Scale (Exact vs FAISS)")
    print("=" * 60)

    dim = 128
    n_values = [5000, 10000, 20000, 50000]
    n_queries = 100
    k = 10

    all_raw: list[dict] = []
    sweep_results: list[dict] = []

    for N in n_values:
        print(f"\n  N={N} ...")
        vectors = generate_uniform(N, dim, seed=seed)
        query_vectors = generate_uniform(n_queries, dim, seed=seed + 1)

        exact_idx, faiss_idx, exact_build, faiss_build = _build_flat_indices(vectors)

        # Sequential queries — Exact
        exact_results_list, exact_latencies = _query_flat_index(
            exact_idx, query_vectors, k=k
        )
        exact_percentiles = compute_query_percentiles(exact_latencies)

        # Sequential queries — FAISS
        faiss_results_list, faiss_latencies = _query_flat_index(
            faiss_idx, query_vectors, k=k
        )
        faiss_percentiles = compute_query_percentiles(faiss_latencies)

        # Match rate
        match_rates = [
            compute_match_rate(er, fr)
            for er, fr in zip(exact_results_list, faiss_results_list)
        ]
        mean_match_rate = round(float(np.mean(match_rates)), 4)

        meta = _metadata(seed, N=N, dim=dim)

        for qi in range(n_queries):
            all_raw.append({
                **meta,
                "query_index": qi,
                "N": N,
                "exact_latency_s": round(exact_latencies[qi], 6),
                "faiss_latency_s": round(faiss_latencies[qi], 6),
                "match_rate": round(match_rates[qi], 4),
            })

        point = {
            **meta,
            "N": N,
            "exact_build_time_s": round(exact_build, 4),
            "faiss_build_time_s": round(faiss_build, 4),
            "exact_query_latency": exact_percentiles,
            "faiss_query_latency": faiss_percentiles,
            "exact_vs_faiss_match_rate": mean_match_rate,
        }
        sweep_results.append(point)
        print(
            f"    Exact Build: {exact_build:.4f}s | "
            f"FAISS Build: {faiss_build:.4f}s"
        )
        print(
            f"    Exact QPS: {exact_percentiles['qps']:.0f} | "
            f"FAISS QPS: {faiss_percentiles['qps']:.0f} | "
            f"Match: {mean_match_rate:.4f}"
        )

    summary = {
        **_metadata(seed, dim=dim),
        "sweep": "F",
        "variable": "N",
        "results": sweep_results,
    }
    _save_outputs("sweep_f", all_raw, summary)


# ------------------------------------------------------------------
# Sweep G — Batch Throughput (Sequential vs Batched Matrix Query)
# ------------------------------------------------------------------


def sweep_g(seed: int = 42) -> None:
    """Fix N=50000.  Compare sequential vs batched matrix query throughput."""
    print("\n" + "=" * 60)
    print("SWEEP G -- Batch Throughput")
    print("=" * 60)

    N, dim = 50000, 128
    batch_sizes = [1, 10, 100, 1000]
    k = 10

    vectors = generate_uniform(N, dim, seed=seed)
    query_pool = generate_uniform(max(batch_sizes), dim, seed=seed + 1)

    exact_idx, faiss_idx, exact_build, faiss_build = _build_flat_indices(vectors)

    all_raw: list[dict] = []
    sweep_results: list[dict] = []

    for batch_size in batch_sizes:
        print(f"\n  batch_size={batch_size} ...")
        batch_queries = query_pool[:batch_size]

        # --- Sequential querying (via .search wrapper) ---
        _, exact_seq_latencies = _query_flat_index(exact_idx, batch_queries, k=k)
        exact_seq_total = sum(exact_seq_latencies)
        exact_seq_qps = batch_size / exact_seq_total if exact_seq_total > 0 else 0.0

        _, faiss_seq_latencies = _query_flat_index(faiss_idx, batch_queries, k=k)
        faiss_seq_total = sum(faiss_seq_latencies)
        faiss_seq_qps = batch_size / faiss_seq_total if faiss_seq_total > 0 else 0.0

        # --- Batched matrix querying (bypass .search wrapper) ---
        # FAISS: use the raw _index.search with a batch matrix
        batch_matrix_f32 = np.ascontiguousarray(
            batch_queries.astype(np.float32)
        )
        # Normalize for inner product
        norms = np.linalg.norm(batch_matrix_f32, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        batch_matrix_f32 = batch_matrix_f32 / norms

        t0 = time.perf_counter()
        faiss_idx._index.search(batch_matrix_f32, k)
        faiss_batch_time = time.perf_counter() - t0
        faiss_batch_qps = batch_size / faiss_batch_time if faiss_batch_time > 0 else 0.0

        # Exact: raw NumPy dot product (matrix @ matrix.T)
        batch_matrix_f64 = batch_queries.astype(float)
        t0 = time.perf_counter()
        exact_idx._embeddings @ batch_matrix_f64.T
        exact_batch_time = time.perf_counter() - t0
        exact_batch_qps = batch_size / exact_batch_time if exact_batch_time > 0 else 0.0

        meta = _metadata(seed, N=N, dim=dim, batch_size=batch_size)

        all_raw.append({
            **meta,
            "batch_size": batch_size,
            "exact_seq_qps": round(exact_seq_qps, 2),
            "faiss_seq_qps": round(faiss_seq_qps, 2),
            "exact_batch_qps": round(exact_batch_qps, 2),
            "faiss_batch_qps": round(faiss_batch_qps, 2),
        })

        point = {
            **meta,
            "batch_size": batch_size,
            "exact_seq_qps": round(exact_seq_qps, 2),
            "faiss_seq_qps": round(faiss_seq_qps, 2),
            "exact_batch_qps": round(exact_batch_qps, 2),
            "faiss_batch_qps": round(faiss_batch_qps, 2),
            "exact_seq_total_s": round(exact_seq_total, 6),
            "faiss_seq_total_s": round(faiss_seq_total, 6),
            "exact_batch_total_s": round(exact_batch_time, 6),
            "faiss_batch_total_s": round(faiss_batch_time, 6),
        }
        sweep_results.append(point)
        print(
            f"    Exact: seq={exact_seq_qps:.0f} QPS, batch={exact_batch_qps:.0f} QPS"
        )
        print(
            f"    FAISS: seq={faiss_seq_qps:.0f} QPS, batch={faiss_batch_qps:.0f} QPS"
        )

    summary = {
        **_metadata(seed, N=N, dim=dim),
        "sweep": "G",
        "variable": "batch_size",
        "results": sweep_results,
    }
    _save_outputs("sweep_g", all_raw, summary)


# ------------------------------------------------------------------
# Sweep H — Industrial Scale: FaissFlatIndex vs FaissHNSWIndex
# ------------------------------------------------------------------


def sweep_h(seed: int = 42) -> None:
    """Compare FaissFlatIndex vs FaissHNSWIndex at industrial scale.

    Fixes M=32, ef_construction=200, ef_search=64 for HNSW.
    Sweeps N over [5000, 10000, 20000, 50000].
    Runs 100 sequential queries per N.
    Recall is computed using FaissFlatIndex as 100% ground truth.
    """
    print("\n" + "=" * 60)
    print("SWEEP H -- Industrial Scale: Exact vs ANN")
    print("=" * 60)

    dim = 128
    n_values = [5000, 10000, 20000, 50000]
    n_queries = 100
    k = 10

    all_raw: list[dict] = []
    sweep_results: list[dict] = []

    for N in n_values:
        print(f"\n  N={N} ...")
        vectors = generate_uniform(N, dim, seed=seed)
        query_vectors = generate_uniform(n_queries, dim, seed=seed + 1)
        documents = _vectors_to_documents(vectors)

        # --- Build FaissFlatIndex ---
        flat_idx = FaissFlatIndex()
        t0 = time.perf_counter()
        flat_idx.add_documents(documents)
        flat_build = time.perf_counter() - t0

        # --- Build FaissHNSWIndex ---
        hnsw_idx = FaissHNSWIndex(
            M=_HNSW_M,
            ef_construction=_HNSW_EF_CONSTRUCTION,
            ef_search=_HNSW_EF_SEARCH,
        )
        t0 = time.perf_counter()
        hnsw_idx.add_documents(documents)
        hnsw_build = time.perf_counter() - t0

        # --- Sequential queries: FaissFlatIndex (ground truth) ---
        flat_results_list, flat_latencies = _query_flat_index(
            flat_idx, query_vectors, k=k
        )
        flat_percentiles = compute_query_percentiles(flat_latencies)

        # --- Sequential queries: FaissHNSWIndex ---
        hnsw_results_list, hnsw_latencies = _query_flat_index(
            hnsw_idx, query_vectors, k=k
        )
        hnsw_percentiles = compute_query_percentiles(hnsw_latencies)

        # --- Recall: HNSW vs Flat ground truth ---
        hnsw_recalls = [
            compute_recall(hnsw_results_list[qi], flat_results_list[qi], k)
            for qi in range(n_queries)
        ]
        hnsw_recall_mean = round(float(np.mean(hnsw_recalls)), 4)

        # Flat recall against itself is trivially 1.0 by definition
        flat_recall_mean = 1.0

        meta = _metadata(
            seed, N=N, dim=dim,
            hnsw_M=_HNSW_M,
            hnsw_ef_construction=_HNSW_EF_CONSTRUCTION,
            hnsw_ef_search=_HNSW_EF_SEARCH,
        )

        for qi in range(n_queries):
            all_raw.append({
                **meta,
                "query_index": qi,
                "N": N,
                "flat_latency_s": round(flat_latencies[qi], 8),
                "hnsw_latency_s": round(hnsw_latencies[qi], 8),
                "hnsw_recall": round(hnsw_recalls[qi], 4),
            })

        point = {
            **meta,
            "N": N,
            # Build times
            "flat_build_time_s": round(flat_build, 6),
            "hnsw_build_time_s": round(hnsw_build, 6),
            # Latency percentiles
            "flat_query_latency": flat_percentiles,
            "hnsw_query_latency": hnsw_percentiles,
            # Recall
            "flat_recall_mean": flat_recall_mean,
            "hnsw_recall_mean": hnsw_recall_mean,
        }
        sweep_results.append(point)
        print(
            f"    Flat  -> Build: {flat_build:.4f}s | "
            f"QPS: {flat_percentiles['qps']:>8.0f} | "
            f"p50: {flat_percentiles['p50']*1000:.4f}ms | "
            f"Recall: {flat_recall_mean:.4f}"
        )
        print(
            f"    HNSW  -> Build: {hnsw_build:.4f}s | "
            f"QPS: {hnsw_percentiles['qps']:>8.0f} | "
            f"p50: {hnsw_percentiles['p50']*1000:.4f}ms | "
            f"Recall: {hnsw_recall_mean:.4f}"
        )
        crossover = "HNSW faster" if hnsw_percentiles["qps"] > flat_percentiles["qps"] else "Flat faster"
        print(f"    QPS winner: {crossover}")

    summary = {
        **_metadata(
            seed, dim=dim,
            hnsw_M=_HNSW_M,
            hnsw_ef_construction=_HNSW_EF_CONSTRUCTION,
            hnsw_ef_search=_HNSW_EF_SEARCH,
        ),
        "sweep": "H",
        "variable": "N",
        "results": sweep_results,
    }
    _save_outputs("sweep_h", all_raw, summary)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

_DISPATCH: dict[str, Any] = {
    "sweep-a": sweep_a,
    "sweep-b": sweep_b,
    "sweep-c": sweep_c,
    "sweep-d": sweep_d,
    "sweep-e": sweep_e,
    "sweep-f": sweep_f,
    "sweep-g": sweep_g,
    "sweep-h": sweep_h,
}


def main() -> None:
    """Entry point for the benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="NSW Benchmark Sweep Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "sweep",
        choices=[*_DISPATCH.keys(), "all"],
        help="Which sweep to run (or 'all' for the full suite).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()

    if args.sweep == "all":
        for name, fn in _DISPATCH.items():
            fn(seed=args.seed)
    else:
        _DISPATCH[args.sweep](seed=args.seed)

    print("\n[OK] Sweep(s) complete. Outputs written to benchmarks/outputs/")


if __name__ == "__main__":
    main()
