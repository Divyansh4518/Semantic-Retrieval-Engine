"""Advanced diagnostics: memory profiling, graph health, query percentiles, recall."""

from __future__ import annotations

import contextlib
import io
import os
import re
import threading
from collections import deque
from typing import Any, Callable

import numpy as np

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ---------------------------------------------------------------------------
# Memory Profiling
# ---------------------------------------------------------------------------


def measure_memory(build_fn: Callable[[], Any]) -> dict[str, float]:
    """Measure RSS memory before, peak during, and after *build_fn*.

    A background thread polls ``psutil.Process().memory_info().rss`` at
    10 ms intervals to approximate peak memory.

    Returns
    -------
    dict
        Keys: ``rss_before_mb``, ``rss_peak_mb``, ``rss_after_mb``.
        All values are ``-1.0`` when *psutil* is not installed.
    """
    if not _HAS_PSUTIL:
        build_fn()
        return {"rss_before_mb": -1.0, "rss_peak_mb": -1.0, "rss_after_mb": -1.0}

    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss

    peak_rss = rss_before
    stop_event = threading.Event()

    def _poll_peak() -> None:
        nonlocal peak_rss
        while not stop_event.is_set():
            try:
                current = process.memory_info().rss
                if current > peak_rss:
                    peak_rss = current
            except Exception:
                pass
            stop_event.wait(0.01)

    monitor = threading.Thread(target=_poll_peak, daemon=True)
    monitor.start()

    try:
        build_fn()
    finally:
        stop_event.set()
        monitor.join(timeout=2.0)

    rss_after = process.memory_info().rss
    if rss_after > peak_rss:
        peak_rss = rss_after

    return {
        "rss_before_mb": round(rss_before / (1024 * 1024), 2),
        "rss_peak_mb": round(peak_rss / (1024 * 1024), 2),
        "rss_after_mb": round(rss_after / (1024 * 1024), 2),
    }


# ---------------------------------------------------------------------------
# Graph Health Analysis
# ---------------------------------------------------------------------------


def analyze_graph(graph_index: Any) -> dict[str, Any]:
    """Analyse the topology of a ``GraphIndex._graph`` adjacency dict.

    Returns
    -------
    dict
        ``total_edges`` (undirected count), ``min_degree``, ``max_degree``,
        ``mean_degree``, ``num_components``, ``largest_component_size``,
        ``asymmetric_edge_count``.
    """
    graph: dict[int, set[int]] = graph_index._graph

    if not graph:
        return {
            "total_edges": 0,
            "min_degree": 0,
            "max_degree": 0,
            "mean_degree": 0.0,
            "num_components": 0,
            "largest_component_size": 0,
            "asymmetric_edge_count": 0,
        }

    # Degree statistics
    degrees = [len(neighbors) for neighbors in graph.values()]
    total_directed_edges = sum(degrees)
    total_edges = total_directed_edges // 2

    min_degree = min(degrees)
    max_degree = max(degrees)
    mean_degree = total_directed_edges / len(degrees)

    # Asymmetric edge count — undirected integrity check
    asymmetric_count = 0
    for node_id, neighbors in graph.items():
        for neighbor_id in neighbors:
            if neighbor_id not in graph or node_id not in graph[neighbor_id]:
                asymmetric_count += 1

    # Connected components via BFS
    visited: set[int] = set()
    num_components = 0
    largest_component_size = 0

    for node_id in graph:
        if node_id in visited:
            continue
        num_components += 1
        component_size = 0
        queue = deque([node_id])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            component_size += 1
            for neighbor in graph.get(current, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        largest_component_size = max(largest_component_size, component_size)

    return {
        "total_edges": total_edges,
        "min_degree": min_degree,
        "max_degree": max_degree,
        "mean_degree": round(mean_degree, 4),
        "num_components": num_components,
        "largest_component_size": largest_component_size,
        "asymmetric_edge_count": asymmetric_count,
    }


# ---------------------------------------------------------------------------
# Query Latency Percentiles
# ---------------------------------------------------------------------------


def compute_query_percentiles(latencies: list[float]) -> dict[str, float]:
    """Compute avg, p50, p95, and max latency from per-query timings.

    All values are in seconds.
    """
    if not latencies:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}

    arr = np.array(latencies, dtype=float)
    return {
        "avg": round(float(np.mean(arr)), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "max": round(float(np.max(arr)), 6),
    }


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------


def compute_recall(
    graph_results: list[tuple],
    exact_results: list[tuple],
    k: int,
) -> float:
    """Fraction of exact top-K document IDs found in graph top-K.

    Both *graph_results* and *exact_results* are lists of
    ``(Document, score)`` tuples as returned by ``search()``.
    """
    exact_ids = {doc.id for doc, _ in exact_results[:k]}
    if not exact_ids:
        return 1.0  # vacuous truth

    graph_ids = {doc.id for doc, _ in graph_results[:k]}
    return len(exact_ids & graph_ids) / len(exact_ids)


# ---------------------------------------------------------------------------
# Stdout Capture for Search Diagnostics
# ---------------------------------------------------------------------------


def capture_search_diagnostics(
    index: Any,
    query_embedding: Any,
    k: int = 5,
) -> dict[str, Any]:
    """Run ``index.search()`` while capturing stdout to extract diagnostics.

    ``GraphIndex.search`` prints debug information including
    ``Nodes Evaluated``, ``Entry Node``, ``Entry Similarity``, etc.
    This function intercepts that output and parses it into a dict
    **without** modifying the production source code.

    Returns
    -------
    dict
        Always contains ``results`` (the search return value).
        May also contain ``nodes_evaluated``, ``entry_node``,
        ``entry_similarity``, and ``explored_percentage`` when the
        debug output is available.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = index.search(query_embedding, k=k)

    output = buf.getvalue()
    diagnostics: dict[str, Any] = {"results": results}

    for line in output.splitlines():
        if "Nodes Evaluated:" in line:
            match = re.search(r"Nodes Evaluated:\s*(\d+)", line)
            if match:
                diagnostics["nodes_evaluated"] = int(match.group(1))
        elif "Entry Node:" in line:
            match = re.search(r"Entry Node:\s*(\d+)", line)
            if match:
                diagnostics["entry_node"] = int(match.group(1))
        elif "Entry Similarity:" in line:
            match = re.search(r"Entry Similarity:\s*([\d.eE+-]+)", line)
            if match:
                diagnostics["entry_similarity"] = float(match.group(1))
        elif "%" in line:
            match = re.search(r"([\d.]+)%", line)
            if match:
                diagnostics["explored_percentage"] = float(match.group(1))

    return diagnostics
