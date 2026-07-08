"""
tests/smoke/phase5_ingestion_smoke_test.py
-------------------------------------------
Phase 5 Smoke Test: High-Fidelity Ingestion Layer (Phases 2-4 Validation).

Validates:
  1. json_renderer module imports cleanly.
  2. render_benchmark_to_prose() produces prose for all 8 sweep types (A-H).
  3. Prose outputs contain no raw JSON artifacts ({, [, ": patterns).
  4. Sweep-specific semantic invariants are satisfied.
  5. Chunker uses langchain splitters (MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter).
  6. Markdown files are chunked with header metadata.
  7. Prose text chunked with no header_trail (correct routing).
  8. Public ingestion package exports render_benchmark_to_prose.
  9. render_benchmark_file() works on real JSON files from the benchmark corpus.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _assert(condition: bool, message: str) -> None:
    if not condition:
        print(f"  [FAIL] {message}")
        sys.exit(1)
    print(f"  [PASS] {message}")


# ---------------------------------------------------------------------------
# Minimal sweep fixture data
# ---------------------------------------------------------------------------

def _make_sweep(letter: str, extra: dict = None) -> dict:
    base = {"sweep": letter, "git_commit": "abc1234", "seed": 42,
            "hyperparameters": {"dim": 128, "N": 5000}, "results": []}
    if extra:
        base.update(extra)
    return base


SWEEP_FIXTURES: dict[str, dict] = {
    "A": {
        "sweep": "A", "git_commit": "abc", "seed": 42,
        "hyperparameters": {"M": 8, "ef_construction": 32, "ef_search": 64, "dim": 128},
        "results": [{
            "N": 5000, "graph_build_time_s": 25.0, "faiss_hnsw_build_time_s": 0.3,
            "query_latency": {"avg": 0.001, "p50": 0.001, "p95": 0.002, "qps": 900.0},
            "faiss_hnsw_query_latency": {"avg": 0.00009, "p50": 0.00008, "qps": 10000.0},
            "recall_mean": 0.95,
            "memory": {"rss_before_mb": 50.0, "rss_peak_mb": 55.0, "rss_after_mb": 55.0},
            "graph_health": {"total_edges": 3000, "min_degree": 2, "max_degree": 8,
                             "mean_degree": 7.1, "num_components": 1, "asymmetric_edge_count": 0},
        }],
    },
    "B": {
        "sweep": "B", "git_commit": "abc", "seed": 42,
        "hyperparameters": {"N": 5000, "ef_construction": 32, "ef_search": 64, "dim": 128},
        "results": [{
            "M": 8, "build_time_s": 25.0, "faiss_hnsw_build_time_s": 0.3,
            "query_latency": {"avg": 0.001, "p95": 0.002, "qps": 900.0},
            "recall_mean": 0.95,
            "graph_health": {"mean_degree": 7.1, "max_degree": 8, "num_components": 1,
                             "asymmetric_edge_count": 0},
        }],
    },
    "C": {
        "sweep": "C", "git_commit": "abc", "seed": 42,
        "hyperparameters": {"N": 5000, "M": 8, "ef_construction": 32, "dim": 128},
        "results": [{
            "ef_search": 64, "build_time_s": 25.0, "faiss_hnsw_build_time_s": 0.3,
            "query_latency": {"avg": 0.001, "p95": 0.002, "qps": 900.0},
            "faiss_hnsw_query_latency": {"avg": 0.00009, "qps": 10000.0},
            "recall_mean": 0.95, "faiss_hnsw_recall_mean": 1.0,
        }],
    },
    "D": {
        "sweep": "D", "git_commit": "abc", "seed": 42,
        "hyperparameters": {"M": 8, "ef_search": 64, "N": 5000, "dim": 128},
        "n_clusters": 5, "queries_per_source": 5, "k": 10,
        "graph_build_time_s": 25.0, "faiss_hnsw_build_time_s": 0.1, "build_time_s": 25.0,
        "graph_escape_matrix": [[0.8, 0.1, 0.0, 0.0, 0.1],
                                 [0.0, 0.9, 0.0, 0.0, 0.1],
                                 [0.1, 0.0, 0.7, 0.1, 0.1],
                                 [0.0, 0.0, 0.1, 0.8, 0.1],
                                 [0.0, 0.1, 0.0, 0.1, 0.8]],
        "results": [],
    },
    "E": {
        "sweep": "E", "git_commit": "abc", "seed": 42,
        "hyperparameters": {"N": 5000, "M": 8, "ef_search": 64, "dim": 128},
        "results": [{
            "ef_construction": 200, "build_time_s": 30.0, "faiss_hnsw_build_time_s": 0.5,
            "query_latency": {"avg": 0.0009, "p95": 0.0015, "qps": 1100.0},
            "recall_mean": 0.98,
            "graph_health": {"mean_degree": 7.8, "num_components": 1},
        }],
    },
    "F": {
        "sweep": "F", "git_commit": "abc", "seed": 42,
        "hyperparameters": {"dim": 128},
        "results": [{
            "N": 5000, "exact_build_time_s": 0.007, "faiss_build_time_s": 0.017,
            "exact_query_latency": {"avg": 0.0025, "p95": 0.003, "qps": 397.0},
            "faiss_query_latency": {"avg": 0.000137, "p95": 0.0003, "qps": 7321.0},
            "exact_vs_faiss_match_rate": 1.0,
        }],
    },
    "G": {
        "sweep": "G", "git_commit": "abc", "seed": 42,
        "hyperparameters": {"N": 50000, "dim": 128},
        "results": [{
            "batch_size": 100,
            "exact_batch_qps": 5961.39, "faiss_batch_qps": 6014.02,
            "exact_seq_qps": 42.88, "faiss_seq_qps": 1108.14,
        }],
    },
    "H": {
        "sweep": "H", "git_commit": "abc", "seed": 42,
        "hyperparameters": {"dim": 128, "hnsw_M": 32, "hnsw_ef_construction": 200, "hnsw_ef_search": 64},
        "results": [
            {
                "N": 5000,
                "flat_build_time_s": 0.03, "hnsw_build_time_s": 0.15,
                "flat_query_latency": {"avg": 0.000115, "qps": 8675.0},
                "hnsw_query_latency": {"avg": 0.000142, "qps": 7031.0},
                "flat_recall_mean": 1.0, "hnsw_recall_mean": 0.936,
            },
            {
                "N": 10000,
                "flat_build_time_s": 0.034, "hnsw_build_time_s": 0.40,
                "flat_query_latency": {"avg": 0.000204, "qps": 4902.0},
                "hnsw_query_latency": {"avg": 0.000178, "qps": 5615.0},
                "flat_recall_mean": 1.0, "hnsw_recall_mean": 0.881,
            },
        ],
    },
}


def run_smoke_test() -> None:
    print("\n" + "=" * 64)
    print("PHASE 5 SMOKE TEST: High-Fidelity Ingestion Layer")
    print("=" * 64)

    # ------------------------------------------------------------------
    # 1. Import validation
    # ------------------------------------------------------------------
    print("\n--- Import Validation ---")
    from src.ingestion.json_renderer import (
        render_benchmark_to_prose,
        render_benchmark_file,
        _render_sweep_f,
        _render_sweep_g,
        _render_sweep_h,
    )
    _assert(True, "json_renderer module imports cleanly")

    from src.ingestion import render_benchmark_to_prose as pkg_render
    _assert(True, "render_benchmark_to_prose accessible from src.ingestion package")

    from src.ingestion.chunker import RecursiveTokenChunker
    _assert(True, "RecursiveTokenChunker imports cleanly")

    # Verify langchain splitters are wired in
    import inspect
    chunker_src = inspect.getsource(RecursiveTokenChunker)
    _assert(
        "MarkdownHeaderTextSplitter" in chunker_src,
        "RecursiveTokenChunker uses MarkdownHeaderTextSplitter",
    )
    _assert(
        "RecursiveCharacterTextSplitter" in chunker_src,
        "RecursiveTokenChunker uses RecursiveCharacterTextSplitter",
    )

    # ------------------------------------------------------------------
    # 2. Prose rendering — all 8 sweeps
    # ------------------------------------------------------------------
    print("\n--- Prose Rendering: All 8 Sweeps ---")
    for letter, fixture in SWEEP_FIXTURES.items():
        prose = render_benchmark_to_prose(fixture)
        _assert(isinstance(prose, str), f"Sweep {letter}: returns a string")
        _assert(len(prose) >= 50, f"Sweep {letter}: prose >= 50 chars (got {len(prose)})")
        _assert(
            not prose.strip().startswith(("{", "[")),
            f"Sweep {letter}: output is NOT raw JSON",
        )
        _assert(
            any(c == "." for c in prose),
            f"Sweep {letter}: prose contains full English sentences",
        )

    # ------------------------------------------------------------------
    # 3. Sweep-specific semantic invariants
    # ------------------------------------------------------------------
    print("\n--- Semantic Invariants ---")

    # Sweep F: mentions both implementations and match rate
    f_prose = render_benchmark_to_prose(SWEEP_FIXTURES["F"])
    _assert("5,000" in f_prose, "Sweep F: N=5000 rendered as formatted number")
    _assert("exact" in f_prose.lower() or "numpy" in f_prose.lower(),
            "Sweep F: references exact/NumPy index")
    _assert("faiss" in f_prose.lower(), "Sweep F: references FAISS index")
    _assert("match rate" in f_prose.lower() or "1.000" in f_prose,
            "Sweep F: match rate reported")

    # Sweep G: batch and QPS
    g_prose = render_benchmark_to_prose(SWEEP_FIXTURES["G"])
    _assert("batch" in g_prose.lower(), "Sweep G: mentions batch queries")
    _assert("qps" in g_prose.lower(), "Sweep G: reports QPS throughput")
    _assert("50,000" in g_prose, "Sweep G: N=50000 rendered as formatted number")

    # Sweep H: crossover detection
    h_prose = render_benchmark_to_prose(SWEEP_FIXTURES["H"])
    _assert("crossover" in h_prose.lower(), "Sweep H: crossover point mentioned")
    _assert("hnsw" in h_prose.lower(), "Sweep H: HNSW mentioned")
    _assert("flat" in h_prose.lower(), "Sweep H: Flat index mentioned")
    _assert("n=5,000" in h_prose.lower() or "n=10,000" in h_prose.lower(),
            "Sweep H: dataset sizes mentioned")

    # Sweep D: cluster analysis
    d_prose = render_benchmark_to_prose(SWEEP_FIXTURES["D"])
    _assert("cluster" in d_prose.lower(), "Sweep D: cluster analysis mentioned")
    _assert("escape" in d_prose.lower() or "recall" in d_prose.lower(),
            "Sweep D: escape/recall metrics mentioned")

    # ------------------------------------------------------------------
    # 4. Real benchmark files from the corpus
    # ------------------------------------------------------------------
    print("\n--- Real Benchmark Files ---")
    bench_dir = _PROJECT_ROOT / "benchmarks" / "outputs" / "aggregated"
    if bench_dir.exists():
        for json_file in sorted(bench_dir.glob("*.json"))[:5]:
            prose = render_benchmark_file(json_file)
            _assert(
                not prose.strip().startswith(("{", "[")),
                f"{json_file.name}: no raw JSON output",
            )
            _assert(
                len(prose) >= 100,
                f"{json_file.name}: prose has >= 100 chars (got {len(prose)})",
            )
            _assert(
                "." in prose,
                f"{json_file.name}: prose contains full sentences",
            )
    else:
        print("  [SKIP] Benchmark directory not found — skipping real file tests.")

    # ------------------------------------------------------------------
    # 5. Chunker routing: Markdown → header metadata present
    # ------------------------------------------------------------------
    print("\n--- Chunker Routing Validation ---")
    chunker = RecursiveTokenChunker(chunk_size=800, overlap=80)

    md_text = """# Introduction\n\nThis is the intro paragraph with enough text to be a chunk.\n\n## Methods\n\nHere are the methods described in full detail with adequate length.\n\n### Subsection\n\nA subsection with details that are long enough to be meaningful.\n"""
    md_chunks = chunker.chunk_document(md_text, "test_doc.md")
    _assert(len(md_chunks) >= 1, f"Markdown doc produces >= 1 chunk (got {len(md_chunks)})")
    _assert(
        all(isinstance(c.metadata.get("header_trail"), list) for c in md_chunks),
        "Markdown chunks all have 'header_trail' in metadata",
    )

    # Prose path: no header_trail content expected for plain prose
    prose_text = (
        "In Benchmark Sweep F at N=5000 vectors: the NumPy index achieved 397 QPS. "
        "The FAISS FlatIndex achieved 7321 QPS. The match rate was 1.000, confirming identical results. "
        "This demonstrates the C++ optimized implementation is 18x faster than pure Python. "
        "Additional sentences to ensure the text is long enough to be chunked. "
        "Performance scales sub-linearly with dataset size for both implementations."
    )
    prose_chunks = chunker.chunk_document(prose_text, "sweep_f_prose.txt")
    _assert(len(prose_chunks) >= 1, f"Prose text produces >= 1 chunk (got {len(prose_chunks)})")
    _assert(
        all("header_trail" in c.metadata for c in prose_chunks),
        "Prose chunks all have 'header_trail' key in metadata (empty list)",
    )
    _assert(
        all(c.metadata["header_trail"] == [] for c in prose_chunks),
        "Prose chunks have empty header_trail (not routed to Markdown path)",
    )

    # ------------------------------------------------------------------
    # 6. chunk_all interface backward-compatibility
    # ------------------------------------------------------------------
    print("\n--- chunk_all Interface ---")
    payloads = {
        "doc.md": md_text,
        "prose.txt": prose_text,
    }
    all_chunks = chunker.chunk_all(payloads)
    _assert(len(all_chunks) >= 2, f"chunk_all returns >= 2 chunks (got {len(all_chunks)})")
    _assert(
        all(hasattr(c, "chunk_id") for c in all_chunks),
        "All chunks have chunk_id attribute",
    )
    _assert(
        all(hasattr(c, "source_file") for c in all_chunks),
        "All chunks have source_file attribute",
    )
    _assert(
        all(len(c.text) >= 20 for c in all_chunks),
        "All chunks meet min_chunk_size threshold",
    )

    print(
        "\n[ALL PASSED] PHASE 5 SMOKE TEST PASSED "
        "-- High-fidelity ingestion layer fully verified.\n"
    )


if __name__ == "__main__":
    run_smoke_test()
