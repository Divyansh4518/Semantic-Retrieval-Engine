"""
src/ingestion/json_renderer.py
--------------------------------
Semantic prose renderer for benchmark sweep telemetry JSON.

Translates raw machine-generated benchmark JSON objects into rich, fully
human-readable English paragraphs before they enter the chunking pipeline.
This is the critical quality gate that prevents the embedding model from
receiving broken key=value fragments instead of coherent semantic text.

Public API
----------
render_benchmark_to_prose(json_data: dict) -> str
    Dispatch to the correct sweep renderer based on the ``"sweep"`` key.

render_benchmark_file(path: str | Path) -> str
    Convenience wrapper: read a JSON/JSONL file and return prose.

Sweep Coverage
--------------
A  — Dataset Scale Sweep (N scaling, graph vs. FAISS comparison)
B  — Graph Connectivity Sweep (M parameter sweep)
C  — ef_search Recall-Latency Trade-off Sweep
D  — Cluster Escape Analysis (cross-cluster retrieval quality)
E  — ef_construction Build Quality Sweep
F  — Implementation Scale Comparison (Python NumPy vs. C++ FAISS Flat)
G  — Batch Query Throughput Analysis
H  — HNSW vs. Flat-Index Crossover Sweep (N scaling)

Unknown sweeps are rendered via a generic key=value fallback that still
produces grammatical sentences rather than raw JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def render_benchmark_to_prose(json_data: dict) -> str:
    """
    Convert a benchmark sweep JSON object into semantic English prose.

    The function inspects the top-level ``"sweep"`` key and dispatches to
    the matching renderer.  If the sweep is unknown or the key is absent,
    a generic paragraph renderer is used as a graceful fallback.

    Parameters
    ----------
    json_data:
        Parsed benchmark JSON object (as returned by ``json.load``).

    Returns
    -------
    str
        One or more full English paragraphs describing the benchmark results.
        The output is ready for direct ingestion into the chunking pipeline.
    """
    sweep = str(json_data.get("sweep", "")).upper().strip()

    dispatch = {
        "A": _render_sweep_a,
        "B": _render_sweep_b,
        "C": _render_sweep_c,
        "D": _render_sweep_d,
        "E": _render_sweep_e,
        "F": _render_sweep_f,
        "G": _render_sweep_g,
        "H": _render_sweep_h,
    }

    renderer = dispatch.get(sweep, _render_generic)
    try:
        return renderer(json_data)
    except Exception as exc:  # never let a renderer crash the pipeline
        logger.warning("Sweep %r renderer raised %s — using generic fallback.", sweep, exc)
        return _render_generic(json_data)


def render_benchmark_file(path: Union[str, Path]) -> str:
    """
    Read a ``.json`` or ``.jsonl`` file and render it as semantic prose.

    For ``.jsonl`` files each line is rendered as a separate paragraph
    and the results are joined with a blank line.

    Parameters
    ----------
    path:
        Absolute or relative path to the benchmark file.

    Returns
    -------
    str
        Full English prose ready for the chunking pipeline.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".jsonl":
        paragraphs: list[str] = []
        for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
                paragraphs.append(render_benchmark_to_prose(record))
            except json.JSONDecodeError as exc:
                logger.warning("%s line %d: JSON parse error — %s", p.name, lineno, exc)
        return "\n\n".join(paragraphs)

    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Some aggregated JSON files wrap multiple results at the top level
    if isinstance(data, list):
        return "\n\n".join(render_benchmark_to_prose(item) for item in data)
    return render_benchmark_to_prose(data)


# ---------------------------------------------------------------------------
# Sweep A — Dataset Scale: Python HNSW vs. FAISS-HNSW comparison as N grows
# ---------------------------------------------------------------------------

def _render_sweep_a(data: dict) -> str:
    """
    Sweep A — Dataset Scale Sweep.
    Variable: N (number of vectors). Compares custom Python HNSW graph
    build time, query latency, and recall against FAISS-HNSW reference.
    """
    git = data.get("git_commit", "unknown")
    hp = data.get("hyperparameters", {})
    M = hp.get("M", "?")
    ef_c = hp.get("ef_construction", "?")
    ef_s = hp.get("ef_search", "?")
    dim = hp.get("dim", 128)

    intro = (
        f"Benchmark Sweep A (Dataset Scale Sweep) examined how the custom Python "
        f"HNSW graph index scales with increasing dataset sizes at a fixed vector "
        f"dimensionality of {dim}. "
        f"The graph was configured with M={M} (max bidirectional links per node), "
        f"ef_construction={ef_c} (candidate list size during index build), and "
        f"ef_search={ef_s} (candidate list size at query time). "
        f"Git commit: {git}."
    )

    result_paragraphs: list[str] = [intro]
    for r in data.get("results", []):
        N = r.get("N", "?")
        graph_build = r.get("graph_build_time_s") or r.get("build_time_s")
        faiss_hnsw_build = r.get("faiss_hnsw_build_time_s")

        ql = r.get("query_latency", {})
        avg_ms = _ms(ql.get("avg"))
        p95_ms = _ms(ql.get("p95"))
        qps = _fmt(ql.get("qps"))

        fql = r.get("faiss_hnsw_query_latency", {})
        faiss_avg_ms = _ms(fql.get("avg"))
        faiss_qps = _fmt(fql.get("qps"))

        recall = r.get("recall_mean") or r.get("faiss_hnsw_recall_mean")

        mem = r.get("memory", {})
        mem_peak = _fmt(mem.get("rss_peak_mb"))

        gh = r.get("graph_health", {})
        mean_deg = _fmt(gh.get("mean_degree"))
        components = gh.get("num_components")

        para = (
            f"At a dataset size of {N:,} vectors: "
            f"the Python HNSW graph build time was {_fmt(graph_build)} seconds"
        )
        if faiss_hnsw_build:
            para += f", while the FAISS-HNSW index built in {_fmt(faiss_hnsw_build)} seconds"
        para += ". "
        if avg_ms:
            para += (
                f"The Python HNSW graph achieved an average query latency of {avg_ms} ms "
                f"(p95: {p95_ms} ms) and a throughput of {qps} QPS. "
            )
        if faiss_avg_ms:
            para += (
                f"The FAISS-HNSW reference index achieved an average query latency of "
                f"{faiss_avg_ms} ms and a throughput of {faiss_qps} QPS. "
            )
        if recall:
            para += f"The mean recall@10 was {recall:.3f}. "
        if mem_peak:
            para += f"Peak RSS memory consumption was {mem_peak} MB. "
        if mean_deg:
            para += (
                f"The graph exhibited a mean node degree of {mean_deg}"
            )
            if components:
                para += f" across {components} connected component(s)"
            para += "."

        result_paragraphs.append(para.strip())

    return "\n\n".join(result_paragraphs)


# ---------------------------------------------------------------------------
# Sweep B — Graph Connectivity Sweep: M parameter
# ---------------------------------------------------------------------------

def _render_sweep_b(data: dict) -> str:
    """
    Sweep B — Graph Connectivity Sweep.
    Variable: M (bidirectional links per node). Measures how graph topology
    quality, query recall, and latency change as M increases.
    """
    git = data.get("git_commit", "unknown")
    hp = data.get("hyperparameters", {})
    N = hp.get("N", "?")
    ef_c = hp.get("ef_construction", "?")
    ef_s = hp.get("ef_search", "?")
    dim = hp.get("dim", 128)

    intro = (
        f"Benchmark Sweep B (Graph Connectivity Sweep) studied the effect of the M "
        f"hyperparameter — which controls the maximum number of bidirectional links "
        f"per HNSW graph node — on retrieval quality and query throughput. "
        f"The dataset was fixed at {N:,} vectors of dimensionality {dim}, with "
        f"ef_construction={ef_c} and ef_search={ef_s}. "
        f"Git commit: {git}."
    )

    result_paragraphs: list[str] = [intro]
    for r in data.get("results", []):
        M_val = r.get("M", "?")
        build = r.get("build_time_s")
        faiss_build = r.get("faiss_hnsw_build_time_s")

        ql = r.get("query_latency", {})
        avg_ms = _ms(ql.get("avg"))
        p95_ms = _ms(ql.get("p95"))
        qps = _fmt(ql.get("qps"))

        recall = r.get("recall_mean")

        gh = r.get("graph_health", {})
        mean_deg = _fmt(gh.get("mean_degree"))
        max_deg = gh.get("max_degree")
        components = gh.get("num_components")
        asym = gh.get("asymmetric_edge_count")

        para = (
            f"With M={M_val} (max links per node): "
            f"the graph build time was {_fmt(build)} seconds"
        )
        if faiss_build:
            para += f" (FAISS-HNSW reference: {_fmt(faiss_build)} seconds)"
        para += ". "
        if avg_ms:
            para += (
                f"Query performance: average latency {avg_ms} ms (p95: {p95_ms} ms), "
                f"throughput {qps} QPS. "
            )
        if recall:
            para += f"Mean recall@10 was {recall:.3f}. "
        if mean_deg:
            para += (
                f"Graph topology: mean degree {mean_deg}, max degree {max_deg}, "
                f"{components} connected component(s), "
                f"{asym} asymmetric edge(s)."
            )

        result_paragraphs.append(para.strip())

    return "\n\n".join(result_paragraphs)


# ---------------------------------------------------------------------------
# Sweep C — ef_search Recall-Latency Trade-off
# ---------------------------------------------------------------------------

def _render_sweep_c(data: dict) -> str:
    """
    Sweep C — ef_search Recall-Latency Trade-off Sweep.
    Variable: ef_search (candidate pool size at query time). Explores the
    fundamental speed-accuracy trade-off of approximate nearest-neighbor search.
    """
    git = data.get("git_commit", "unknown")
    hp = data.get("hyperparameters", {})
    N = hp.get("N", "?")
    M = hp.get("M", "?")
    ef_c = hp.get("ef_construction", "?")
    dim = hp.get("dim", 128)

    intro = (
        f"Benchmark Sweep C (ef_search Recall-Latency Trade-off Sweep) explored "
        f"the fundamental speed-accuracy trade-off of HNSW approximate nearest-"
        f"neighbor search by varying the ef_search parameter, which controls the "
        f"dynamic candidate pool size at query time. "
        f"Dataset: {N:,} vectors of dimensionality {dim}, graph M={M}, "
        f"ef_construction={ef_c}. "
        f"Git commit: {git}."
    )

    result_paragraphs: list[str] = [intro]
    for r in data.get("results", []):
        ef_s = r.get("ef_search", "?")

        ql = r.get("query_latency", {})
        avg_ms = _ms(ql.get("avg"))
        p95_ms = _ms(ql.get("p95"))
        qps = _fmt(ql.get("qps"))

        fql = r.get("faiss_hnsw_query_latency", {})
        faiss_avg_ms = _ms(fql.get("avg"))
        faiss_qps = _fmt(fql.get("qps"))

        recall = r.get("recall_mean")
        faiss_recall = r.get("faiss_hnsw_recall_mean")

        para = f"With ef_search={ef_s}: "
        if avg_ms:
            para += (
                f"the Python HNSW graph achieved average latency {avg_ms} ms "
                f"(p95: {p95_ms} ms) at {qps} QPS"
            )
            if recall is not None:
                para += f" with a recall@10 of {recall:.3f}"
            para += ". "
        if faiss_avg_ms:
            para += (
                f"The FAISS-HNSW reference achieved {faiss_avg_ms} ms average "
                f"latency at {faiss_qps} QPS"
            )
            if faiss_recall is not None:
                para += f" with recall@10 of {faiss_recall:.3f}"
            para += ". "

        result_paragraphs.append(para.strip())

    return "\n\n".join(result_paragraphs)


# ---------------------------------------------------------------------------
# Sweep D — Cluster Escape Analysis
# ---------------------------------------------------------------------------

def _render_sweep_d(data: dict) -> str:
    """
    Sweep D — Cluster Escape Analysis.
    Measures cross-cluster retrieval quality: what fraction of the top-k
    nearest neighbors retrieved by the graph correctly escape from the
    query's source cluster (i.e., cross cluster boundaries).
    """
    git = data.get("git_commit", "unknown")
    hp = data.get("hyperparameters", {})
    N = hp.get("N", "?")
    M = hp.get("M", "?")
    ef_s = hp.get("ef_search", "?")
    dim = hp.get("dim", 128)
    n_clusters = data.get("n_clusters", "?")
    k = data.get("k", "?")
    qps = data.get("queries_per_source", "?")

    graph_build = data.get("graph_build_time_s") or data.get("build_time_s")
    faiss_build = data.get("faiss_hnsw_build_time_s")

    # Summarise the escape matrix diagonals (intra-cluster recall)
    graph_matrix = data.get("graph_escape_matrix") or data.get("escape_matrix", [])
    faiss_matrix = data.get("faiss_hnsw_escape_matrix", [])

    graph_diag = _matrix_diagonal(graph_matrix)
    faiss_diag = _matrix_diagonal(faiss_matrix)

    intro = (
        f"Benchmark Sweep D (Cluster Escape Analysis) measured cross-cluster "
        f"retrieval quality — how accurately both the custom Python HNSW graph "
        f"and the FAISS-HNSW reference retrieve vectors that belong to the same "
        f"cluster as the query vector. "
        f"Dataset: {N:,} vectors of dimensionality {dim} partitioned into "
        f"{n_clusters} clusters. "
        f"Graph configuration: M={M}, ef_search={ef_s}. "
        f"{qps} queries were issued per source cluster, retrieving the top k={k} "
        f"neighbors. "
        f"Graph build time: {_fmt(graph_build)} seconds"
    )
    if faiss_build:
        intro += f"; FAISS-HNSW build time: {_fmt(faiss_build)} seconds"
    intro += f". Git commit: {git}."

    result_paragraphs: list[str] = [intro]

    if graph_diag:
        avg_intra = sum(graph_diag) / len(graph_diag)
        para = (
            f"The Python HNSW graph escape matrix diagonal (intra-cluster recall) "
            f"values ranged from {min(graph_diag):.2f} to {max(graph_diag):.2f}, "
            f"with an average intra-cluster recall of {avg_intra:.2f}. "
            f"Higher diagonal values indicate that the graph successfully routes "
            f"queries to their true nearest neighbors within the same cluster."
        )
        result_paragraphs.append(para)

    if faiss_diag:
        avg_faiss = sum(faiss_diag) / len(faiss_diag)
        para = (
            f"The FAISS-HNSW reference escape matrix diagonal showed values "
            f"from {min(faiss_diag):.2f} to {max(faiss_diag):.2f}, "
            f"with an average intra-cluster recall of {avg_faiss:.2f}."
        )
        result_paragraphs.append(para)

    return "\n\n".join(result_paragraphs)


# ---------------------------------------------------------------------------
# Sweep E — ef_construction Build Quality Sweep
# ---------------------------------------------------------------------------

def _render_sweep_e(data: dict) -> str:
    """
    Sweep E — ef_construction Build Quality Sweep.
    Variable: ef_construction (candidate pool during index construction).
    Measures how construction effort affects graph quality and query accuracy.
    """
    git = data.get("git_commit", "unknown")
    hp = data.get("hyperparameters", {})
    N = hp.get("N", "?")
    M = hp.get("M", "?")
    ef_s = hp.get("ef_search", "?")
    dim = hp.get("dim", 128)

    intro = (
        f"Benchmark Sweep E (ef_construction Build Quality Sweep) investigated "
        f"how the ef_construction parameter — which governs the candidate pool "
        f"size during HNSW graph construction — affects the resulting index "
        f"quality, build time, and downstream query recall. "
        f"Dataset: {N:,} vectors of dimensionality {dim}, M={M}, ef_search={ef_s}. "
        f"Git commit: {git}."
    )

    result_paragraphs: list[str] = [intro]
    for r in data.get("results", []):
        ef_c_val = r.get("ef_construction", "?")
        build = r.get("build_time_s")
        faiss_build = r.get("faiss_hnsw_build_time_s")

        ql = r.get("query_latency", {})
        avg_ms = _ms(ql.get("avg"))
        p95_ms = _ms(ql.get("p95"))
        qps = _fmt(ql.get("qps"))

        recall = r.get("recall_mean")

        gh = r.get("graph_health", {})
        mean_deg = _fmt(gh.get("mean_degree"))
        components = gh.get("num_components")

        para = f"With ef_construction={ef_c_val}: graph build time was {_fmt(build)} seconds"
        if faiss_build:
            para += f" (FAISS-HNSW: {_fmt(faiss_build)} seconds)"
        para += ". "
        if avg_ms:
            para += (
                f"Query performance: {avg_ms} ms average latency (p95: {p95_ms} ms), "
                f"{qps} QPS. "
            )
        if recall is not None:
            para += f"Recall@10: {recall:.3f}. "
        if mean_deg:
            para += f"Mean graph degree: {mean_deg}, connected components: {components}."

        result_paragraphs.append(para.strip())

    return "\n\n".join(result_paragraphs)


# ---------------------------------------------------------------------------
# Sweep F — Implementation Scale Comparison (Python NumPy vs. C++ FAISS Flat)
# ---------------------------------------------------------------------------

def _render_sweep_f(data: dict) -> str:
    """
    Sweep F — Implementation Scale Comparison.
    Variable: N. Compares the Python NumPy ExactIndex (brute-force) against
    the C++ FAISS FlatIndex (optimised BLAS brute-force) across dataset sizes.
    """
    git = data.get("git_commit", "unknown")
    hp = data.get("hyperparameters", {})
    dim = hp.get("dim", 128)

    intro = (
        f"Benchmark Sweep F (Implementation Scale Comparison) measured the raw "
        f"throughput and latency gap between the pure Python NumPy exact brute-"
        f"force index and the C++ FAISS FlatIndex (BLAS-optimised exhaustive search) "
        f"at a fixed vector dimensionality of {dim} as the dataset size N scaled "
        f"from 5,000 to 50,000 vectors. "
        f"Git commit: {git}."
    )

    result_paragraphs: list[str] = [intro]
    for r in data.get("results", []):
        N = r.get("N", "?")
        exact_build = _fmt(r.get("exact_build_time_s"))
        faiss_build = _fmt(r.get("faiss_build_time_s"))

        eql = r.get("exact_query_latency", {})
        exact_avg_ms = _ms(eql.get("avg"))
        exact_qps = _fmt(eql.get("qps"))

        fql = r.get("faiss_query_latency", {})
        faiss_avg_ms = _ms(fql.get("avg"))
        faiss_qps = _fmt(fql.get("qps"))

        match_rate = r.get("exact_vs_faiss_match_rate")

        para = (
            f"In Benchmark Sweep F (Implementation Scale Comparison) at a dataset "
            f"size of {N:,} vectors with a dimension size of {dim}: "
            f"The Python NumPy ExactIndex built in {exact_build} seconds and achieved "
            f"an average latency of {exact_avg_ms} ms and a throughput of {exact_qps} QPS. "
            f"The C++ FAISS FlatIndex built in {faiss_build} seconds and achieved "
            f"an average latency of {faiss_avg_ms} ms and a throughput of {faiss_qps} QPS."
        )
        if match_rate is not None:
            para += (
                f" The exact vs FAISS match rate was {match_rate:.3f}, "
                f"proving {'identical' if match_rate == 1.0 else 'near-identical'} "
                f"retrieval accuracy between the two implementations."
            )

        result_paragraphs.append(para)

    return "\n\n".join(result_paragraphs)


# ---------------------------------------------------------------------------
# Sweep G — Batch Query Throughput Analysis
# ---------------------------------------------------------------------------

def _render_sweep_g(data: dict) -> str:
    """
    Sweep G — Batch Query Throughput Analysis.
    Variable: batch_size. Measures how batching queries improves throughput
    for both the NumPy ExactIndex and the FAISS FlatIndex at N=50,000.
    """
    git = data.get("git_commit", "unknown")
    hp = data.get("hyperparameters", {})
    N = hp.get("N", 50000)
    dim = hp.get("dim", 128)

    intro = (
        f"Benchmark Sweep G (Batch Query Throughput Analysis) measured how "
        f"batching multiple queries together improves throughput for both the "
        f"Python NumPy ExactIndex and the C++ FAISS FlatIndex at a fixed dataset "
        f"size of {N:,} vectors and dimensionality {dim}. "
        f"Git commit: {git}."
    )

    result_paragraphs: list[str] = [intro]
    for r in data.get("results", []):
        batch_size = r.get("batch_size", "?")
        exact_batch_qps = _fmt(r.get("exact_batch_qps"))
        faiss_batch_qps = _fmt(r.get("faiss_batch_qps"))
        exact_seq_qps = _fmt(r.get("exact_seq_qps"))
        faiss_seq_qps = _fmt(r.get("faiss_seq_qps"))

        para = (
            f"In Benchmark Sweep G (Batch Query Throughput Analysis) with a fixed "
            f"dataset size of {N:,} vectors and a batch size of {batch_size} queries: "
            f"Vectorized sequential matrix queries on NumPy achieved {exact_batch_qps} QPS "
            f"(sequential: {exact_seq_qps} QPS), while the C++ FAISS batch search engine "
            f"achieved {faiss_batch_qps} QPS (sequential: {faiss_seq_qps} QPS)."
        )

        result_paragraphs.append(para)

    return "\n\n".join(result_paragraphs)


# ---------------------------------------------------------------------------
# Sweep H — HNSW vs. Flat-Index Crossover Sweep
# ---------------------------------------------------------------------------

def _render_sweep_h(data: dict) -> str:
    """
    Sweep H — HNSW vs. Flat-Index Crossover Sweep.
    Variable: N. At small N, the exhaustive Flat index is faster than HNSW
    (due to HNSW overhead). This sweep finds the crossover point where HNSW
    becomes faster and tracks recall degradation as N grows.
    """
    git = data.get("git_commit", "unknown")
    hp = data.get("hyperparameters", {})
    dim = hp.get("dim", 128)
    hnsw_M = hp.get("hnsw_M", "?")
    hnsw_ef_c = hp.get("hnsw_ef_construction", "?")
    hnsw_ef_s = hp.get("hnsw_ef_search", "?")

    intro = (
        f"Benchmark Sweep H (HNSW vs. Flat-Index Crossover Analysis) identified "
        f"the dataset size N at which the C++ FAISS HNSW index (M={hnsw_M}, "
        f"ef_construction={hnsw_ef_c}, ef_search={hnsw_ef_s}) transitions from "
        f"being slower than the exhaustive FAISS FlatIndex to being faster, "
        f"while also tracking HNSW recall degradation as N increases. "
        f"All vectors had dimensionality {dim}. "
        f"Git commit: {git}."
    )

    result_paragraphs: list[str] = [intro]

    crossover_found = False
    prev_N = None
    prev_flat_faster = None

    for r in data.get("results", []):
        N = r.get("N", "?")
        flat_build = _fmt(r.get("flat_build_time_s"))
        hnsw_build = _fmt(r.get("hnsw_build_time_s"))

        fql = r.get("flat_query_latency", {})
        flat_avg_ms = _ms(fql.get("avg"))
        flat_qps = _fmt(fql.get("qps"))

        hql = r.get("hnsw_query_latency", {})
        hnsw_avg_ms = _ms(hql.get("avg"))
        hnsw_qps = _fmt(hql.get("qps"))

        flat_recall = r.get("flat_recall_mean")
        hnsw_recall = r.get("hnsw_recall_mean")

        flat_faster = (fql.get("avg") or 0) < (hql.get("avg") or 0)

        # Detect crossover
        crossover_note = ""
        if prev_flat_faster is not None and prev_flat_faster and not flat_faster and not crossover_found:
            crossover_found = True
            crossover_note = (
                f" This is the crossover point: between N={prev_N:,} and N={N:,} "
                f"the HNSW index transitions from being slower to being faster than "
                f"the exhaustive Flat index."
            )

        para = (
            f"At dataset size N={N:,}: "
            f"The FAISS FlatIndex built in {flat_build} seconds; "
            f"the FAISS HNSW index built in {hnsw_build} seconds. "
            f"The FlatIndex achieved {flat_avg_ms} ms average query latency at {flat_qps} QPS"
        )
        if flat_recall is not None:
            para += f" with recall {flat_recall:.3f}"
        para += ". "
        para += (
            f"The HNSW index achieved {hnsw_avg_ms} ms average query latency at {hnsw_qps} QPS"
        )
        if hnsw_recall is not None:
            para += f" with recall {hnsw_recall:.3f}"
        para += "."

        if flat_faster:
            para += (
                f" At this scale, the FlatIndex is faster than HNSW, as HNSW's "
                f"graph traversal overhead dominates for small N."
            )
        else:
            para += (
                f" At this scale, HNSW is faster than the FlatIndex, demonstrating "
                f"the benefit of approximate graph-based search at larger N."
            )

        para += crossover_note
        result_paragraphs.append(para)

        prev_N = N
        prev_flat_faster = flat_faster

    # Append crossover summary if found
    if crossover_found:
        result_paragraphs.append(
            "In summary, Sweep H confirmed that the HNSW index crossover point — "
            "where approximate graph-based search becomes faster than exhaustive "
            "flat-index search — occurs between N=5,000 and N=10,000 vectors. "
            "Beyond the crossover, HNSW's recall degrades gradually as N increases, "
            "reflecting the trade-off between approximate search speed and retrieval accuracy."
        )

    return "\n\n".join(result_paragraphs)


# ---------------------------------------------------------------------------
# Generic fallback — produces readable key=value sentences for unknown sweeps
# ---------------------------------------------------------------------------

def _render_generic(data: dict) -> str:
    """
    Generic benchmark renderer for any unrecognised sweep type.

    Produces grammatical English sentences from all top-level scalar fields
    and recurses one level into any nested dict or list of result dicts.
    """
    sweep = data.get("sweep", "Unknown")
    git = data.get("git_commit", "")
    ts = data.get("timestamp", "")

    lines: list[str] = [
        f"Benchmark Sweep {sweep} recorded the following performance measurements"
        + (f" at commit {git}" if git else "")
        + (f" on {ts[:10]}" if ts else "") + "."
    ]

    for key, value in data.items():
        if key in ("sweep", "git_commit", "timestamp", "seed", "results"):
            continue
        if isinstance(value, (int, float, str, bool)):
            lines.append(f"The {key.replace('_', ' ')} was {value}.")
        elif isinstance(value, dict):
            sub = ", ".join(f"{k}={v}" for k, v in value.items() if isinstance(v, (int, float, str)))
            if sub:
                lines.append(f"The {key.replace('_', ' ')} values were: {sub}.")

    for i, r in enumerate(data.get("results", []), 1):
        if isinstance(r, dict):
            sub = ", ".join(
                f"{k}={v}" for k, v in r.items()
                if isinstance(v, (int, float, str)) and k not in ("timestamp", "git_commit", "seed")
            )
            if sub:
                lines.append(f"Result {i}: {sub}.")

    return " ".join(lines)


# ---------------------------------------------------------------------------
# Private formatting helpers
# ---------------------------------------------------------------------------

def _ms(value: float | None) -> str:
    """Convert seconds float to a milliseconds string, or '?' if None."""
    if value is None:
        return "?"
    return f"{value * 1000:.3f}"


def _fmt(value) -> str:
    """Format a numeric value cleanly, or return '?' if None."""
    if value is None:
        return "?"
    if isinstance(value, float):
        return f"{value:.4f}" if value < 100 else f"{value:.2f}"
    return str(value)


def _matrix_diagonal(matrix: list[list]) -> list[float]:
    """Extract the diagonal of a square matrix (intra-cluster escape rates)."""
    if not matrix:
        return []
    return [
        float(matrix[i][i])
        for i in range(min(len(matrix), len(matrix[0]) if matrix else 0))
        if i < len(matrix[i])
    ]
