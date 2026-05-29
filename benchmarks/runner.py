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
from src.index.graph import GraphIndex  # noqa: E402
from src.models import Document  # noqa: E402

from benchmarks.datasets import (  # noqa: E402
    generate_at_dimension,
    generate_clustered,
    generate_uniform,
)
from benchmarks.telemetry import (  # noqa: E402
    analyze_graph,
    capture_search_diagnostics,
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
    """Fix M=8, ef_search=64.  Sweep N over [100, 500, 1500, 5000]."""
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
        build_time = 0.0

        def _do_build() -> None:
            nonlocal graph_idx, exact_idx, build_time
            graph_idx, exact_idx, build_time = _build_indices(
                vectors, M, ef_construction, ef_search
            )

        mem = measure_memory(_do_build)

        assert graph_idx is not None and exact_idx is not None

        graph_health = analyze_graph(graph_idx)
        per_query, latencies, recalls = _run_queries(
            graph_idx, exact_idx, query_vectors, k=k
        )
        percentiles = compute_query_percentiles(latencies)

        meta = _metadata(
            seed, M=M, ef_construction=ef_construction,
            ef_search=ef_search, N=N, dim=dim,
        )

        for entry in per_query:
            entry.update(meta)
            entry["N"] = N
        all_raw.extend(per_query)

        point = {
            **meta,
            "N": N,
            "build_time_s": round(build_time, 4),
            "memory": mem,
            "graph_health": graph_health,
            "query_latency": percentiles,
            "recall_mean": round(float(np.mean(recalls)), 4),
        }
        sweep_results.append(point)
        print(
            f"    Build: {build_time:.3f}s | "
            f"Recall: {np.mean(recalls):.4f} | "
            f"p95: {percentiles['p95'] * 1000:.1f}ms"
        )

    summary = {
        **_metadata(
            seed, M=M, ef_construction=ef_construction,
            ef_search=ef_search, dim=dim,
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
    """Fix N=5000, ef_search=64.  Sweep M over [2, 4, 8, 16, 32]."""
    print("\n" + "=" * 60)
    print("SWEEP B -- Topology (M sweep)")
    print("=" * 60)

    N, ef_construction, ef_search, dim = 5000, 32, 64, 128
    m_values = [2, 4, 8, 16, 32]
    n_queries = 100
    k = 10

    vectors = generate_uniform(N, dim, seed=seed)
    query_vectors = generate_uniform(n_queries, dim, seed=seed + 1)

    all_raw: list[dict] = []
    sweep_results: list[dict] = []

    for M in m_values:
        print(f"\n  M={M} ...")
        graph_idx, exact_idx, build_time = _build_indices(
            vectors, M, ef_construction, ef_search
        )
        graph_health = analyze_graph(graph_idx)
        per_query, latencies, recalls = _run_queries(
            graph_idx, exact_idx, query_vectors, k=k
        )
        percentiles = compute_query_percentiles(latencies)

        meta = _metadata(
            seed, M=M, ef_construction=ef_construction,
            ef_search=ef_search, N=N, dim=dim,
        )

        for entry in per_query:
            entry.update(meta)
            entry["M"] = M
        all_raw.extend(per_query)

        point = {
            **meta,
            "M": M,
            "build_time_s": round(build_time, 4),
            "graph_health": graph_health,
            "query_latency": percentiles,
            "recall_mean": round(float(np.mean(recalls)), 4),
        }
        sweep_results.append(point)
        print(
            f"    Components: {graph_health['num_components']} | "
            f"Asymmetric: {graph_health['asymmetric_edge_count']} | "
            f"Recall: {np.mean(recalls):.4f}"
        )

    summary = {
        **_metadata(
            seed, N=N, ef_construction=ef_construction,
            ef_search=ef_search, dim=dim,
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
    """Fix N=5000, M=8.  Sweep ef_search over [8..512]."""
    print("\n" + "=" * 60)
    print("SWEEP C -- Traversal (ef_search sweep)")
    print("=" * 60)

    N, M, ef_construction, dim = 5000, 8, 32, 128
    ef_values = [8, 16, 32, 64, 128, 256, 512]
    n_queries = 100
    k = 10

    vectors = generate_uniform(N, dim, seed=seed)
    query_vectors = generate_uniform(n_queries, dim, seed=seed + 1)

    # Build once — ef_search only affects queries, not construction.
    graph_idx, exact_idx, build_time = _build_indices(
        vectors, M, ef_construction, max(ef_values)
    )

    all_raw: list[dict] = []
    sweep_results: list[dict] = []

    for ef_search in ef_values:
        print(f"\n  ef_search={ef_search} ...")
        graph_idx.ef_search = ef_search

        per_query, latencies, recalls = _run_queries(
            graph_idx, exact_idx, query_vectors, k=k
        )
        percentiles = compute_query_percentiles(latencies)

        meta = _metadata(
            seed, M=M, ef_construction=ef_construction,
            ef_search=ef_search, N=N, dim=dim,
        )

        for entry in per_query:
            entry.update(meta)
            entry["ef_search"] = ef_search
        all_raw.extend(per_query)

        point = {
            **meta,
            "ef_search": ef_search,
            "build_time_s": round(build_time, 4),
            "query_latency": percentiles,
            "recall_mean": round(float(np.mean(recalls)), 4),
        }
        sweep_results.append(point)
        print(
            f"    Recall: {np.mean(recalls):.4f} | "
            f"p95: {percentiles['p95'] * 1000:.1f}ms"
        )

    summary = {
        **_metadata(seed, M=M, ef_construction=ef_construction, N=N, dim=dim),
        "sweep": "C",
        "variable": "ef_search",
        "results": sweep_results,
    }
    _save_outputs("sweep_c", all_raw, summary)


# ------------------------------------------------------------------
# Sweep D — Escape Success Matrix
# ------------------------------------------------------------------


def sweep_d(seed: int = 42) -> None:
    """5000 vectors, 10 clusters.  10x10 escape-success matrix."""
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

    graph_idx, exact_idx, build_time = _build_indices(
        vectors, M, ef_construction, ef_search
    )

    # Group document indices by cluster
    cluster_indices: dict[int, list[int]] = {}
    for i, c in enumerate(assignments):
        cluster_indices.setdefault(int(c), []).append(i)

    escape_matrix = np.zeros((n_clusters, n_clusters), dtype=float)
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
            successes = 0

            for src_idx in sampled:
                query_vec = vectors[src_idx]
                diag = capture_search_diagnostics(graph_idx, query_vec, k=k)
                results = diag["results"]

                result_indices = [
                    int(doc.id.split("-")[1]) for doc, _ in results
                ]
                escaped = any(idx in tgt_set for idx in result_indices)
                if escaped:
                    successes += 1

                all_raw.append(
                    {
                        "source_cluster": src_cluster,
                        "target_cluster": tgt_cluster,
                        "query_doc_index": int(src_idx),
                        "escaped": escaped,
                        "result_indices": result_indices,
                    }
                )

            escape_matrix[src_cluster, tgt_cluster] = (
                successes / sample_count if sample_count > 0 else 0.0
            )

    # Pretty-print the matrix
    print("\n  Escape Success Matrix (rows=source, cols=target):")
    header = "       " + "  ".join(f"C{c:02d}" for c in range(n_clusters))
    print(header)
    for src in range(n_clusters):
        row = f"  C{src:02d}  " + "  ".join(
            f"{escape_matrix[src, tgt]:.2f}" for tgt in range(n_clusters)
        )
        print(row)

    meta = _metadata(
        seed, M=M, ef_construction=ef_construction,
        ef_search=ef_search, N=N, dim=dim,
    )
    summary = {
        **meta,
        "sweep": "D",
        "variable": "cluster_escape",
        "n_clusters": n_clusters,
        "queries_per_source": queries_per_source,
        "k": k,
        "build_time_s": round(build_time, 4),
        "escape_matrix": escape_matrix.tolist(),
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

        graph_idx, exact_idx, build_time = _build_indices(
            vectors, M, ef_construction, ef_search
        )

        per_query, latencies, recalls = _run_queries(
            graph_idx, exact_idx, query_vectors, k=k
        )
        percentiles = compute_query_percentiles(latencies)

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
            "build_time_s": round(build_time, 4),
            "query_latency": percentiles,
            "recall_mean": round(float(np.mean(recalls)), 4),
        }
        sweep_results.append(point)
        print(
            f"    Recall: {np.mean(recalls):.4f} | "
            f"p95: {percentiles['p95'] * 1000:.1f}ms"
        )

    summary = {
        **_metadata(seed, M=M, ef_construction=ef_construction, ef_search=ef_search, N=N),
        "sweep": "E",
        "variable": "dim",
        "results": sweep_results,
    }
    _save_outputs("sweep_e", all_raw, summary)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

_DISPATCH: dict[str, Any] = {
    "sweep-a": sweep_a,
    "sweep-b": sweep_b,
    "sweep-c": sweep_c,
    "sweep-d": sweep_d,
    "sweep-e": sweep_e,
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
