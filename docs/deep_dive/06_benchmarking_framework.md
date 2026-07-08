# 06 — Benchmarking Framework Deep Dive

> **Scope:** The complete benchmarking infrastructure — sweep architecture,
> synthetic data generators, telemetry instrumentation, output schemas,
> and chart generation.

---

## 1. Framework Philosophy

The benchmarking suite is designed as a **research-grade harness** that answers three fundamental questions about retrieval systems:

1. **Topology:** How does graph structure (M, connectivity) affect recall?
2. **Scaling:** How does performance change with dataset size and dimensionality?
3. **Implementation:** How much performance comes from algorithm design vs. hardware optimisation?

**Critical constraint:** The suite is a *read-only consumer* of the `src/` APIs. Zero modifications to production code. All telemetry is captured via `contextlib.redirect_stdout()` rather than instrumentation hooks injected into the index implementations.

---

## 2. Architecture

```
benchmarks/
├── runner.py              # CLI orchestrator (1,311 lines)
│   ├── sweep_a()          # Scaling (N sweep)
│   ├── sweep_b()          # Topology (M sweep)
│   ├── sweep_c()          # Traversal (ef_search sweep)
│   ├── sweep_d()          # Escape Success Matrix
│   ├── sweep_e()          # Dimensionality Curse
│   ├── sweep_f()          # Implementation Scale (Exact vs FAISS)
│   ├── sweep_g()          # Batch Throughput
│   └── sweep_h()          # Industrial Scale (Flat vs HNSW)
├── telemetry.py           # Metric computation functions
├── plots.py               # 10 matplotlib chart generators
├── datasets/              # Synthetic data generators
│   ├── uniform.py         # L2-normalised random vectors
│   ├── clustered.py       # Gaussian cluster generator
│   └── dimensionality.py  # Dimension-variable wrapper
└── outputs/
    ├── raw/               # Per-query JSONL logs
    ├── aggregated/        # Sweep summary JSON files
    └── figures/           # Generated PNG charts
```

### CLI Dispatch

```bash
python -m benchmarks.runner sweep-a           # Single sweep
python -m benchmarks.runner sweep-h --seed 99 # Custom seed
python -m benchmarks.runner all               # All 8 sweeps
```

The `_DISPATCH` dict maps sweep names to functions:

```python
_DISPATCH = {
    "sweep-a": sweep_a,
    "sweep-b": sweep_b,
    ...
    "sweep-h": sweep_h,
}
```

---

## 3. Sweep Specifications

### Sweep A — Scaling (N Sweep)

| Parameter | Value |
|---|---|
| **Variable** | N ∈ [100, 500, 1500, 5000] |
| **Fixed** | M=8, ef_construction=32, ef_search=64, dim=128 |
| **Indices** | GraphIndex, ExactIndex, FaissFlatIndex, FaissHNSWIndex |
| **Queries** | 100 per N value |

**Purpose:** Profile how all four index implementations scale with corpus size.

**Metrics:** Build time per engine, RSS memory (before/peak/after), graph health, query latency percentiles, recall (Graph vs Exact), HNSW recall (vs Flat), Exact vs FAISS match rate.

### Sweep B — Topology (M Sweep)

| Parameter | Value |
|---|---|
| **Variable** | M ∈ [2, 4, 8, 16, 32] |
| **Fixed** | N=5000, ef_construction=32, ef_search=64, dim=128 |
| **Indices** | GraphIndex + FaissHNSWIndex (same M for both) |

**Purpose:** Map the exact threshold between graph fragmentation and memory overhead.

**Key finding:** At M=2, the graph fragments into 1,136 disconnected components with 0.3% recall.

### Sweep C — Traversal (ef_search Sweep)

| Parameter | Value |
|---|---|
| **Variable** | ef_search ∈ [8, 16, 32, 64, 128, 256, 512] |
| **Fixed** | N=5000, M=8, ef_construction=32, dim=128 |

**Optimisation:** Both indices are built *once* at maximum `ef_search`. Each iteration mutates the parameter in-place:
```python
graph_idx.ef_search = ef_search
hnsw_idx.set_ef_search(ef_search)
```

This avoids redundant O(N) graph construction per sweep point.

### Sweep D — Escape Success Matrix

| Parameter | Value |
|---|---|
| **Configuration** | N=5000, 10 clusters, 5 queries per source cluster |
| **Output** | Two 10×10 escape success matrices (Graph + HNSW) |

**Definition:** For a query from cluster `src`, "Escape Success" to cluster `tgt` = 1 if at least one of the top-K results belongs to `tgt`, else 0.

**Key finding:** The flat NSW graph shows near 0% cross-cluster escape, demonstrating local-minima trapping. HNSW achieves significantly better escape rates due to hierarchical shortcuts.

### Sweep E — Dimensionality Curse

| Parameter | Value |
|---|---|
| **Variable** | dim ∈ [128, 384, 768, 1536] |
| **Fixed** | N=5000, M=8, ef_construction=32, ef_search=64 |

**Purpose:** Quantify how increasing dimensionality impacts recall and latency for the flat NSW graph.

### Sweep F — Implementation Scale

| Parameter | Value |
|---|---|
| **Variable** | N ∈ [5000, 10000, 20000, 50000] |
| **Indices** | ExactIndex vs FaissFlatIndex only |

**Purpose:** Isolate the performance cost of the Python interpreter by comparing two implementations of the *same algorithm* (exhaustive brute-force).

**Key finding:** 100% match rate (identical results), 18–24x speedup from FAISS.

### Sweep G — Batch Throughput

| Parameter | Value |
|---|---|
| **Variable** | batch_size ∈ [1, 10, 100, 1000] |
| **Fixed** | N=50000, dim=128, k=10 |

**Technique:** Compares four execution strategies:
1. Exact sequential (`for` loop + `.search()`)
2. Exact batched (`embeddings @ batch_matrix.T`)
3. FAISS sequential (`for` loop + `.search()`)
4. FAISS batched (`_index.search(batch_matrix, k)`)

**Key finding:** FAISS batched achieves 10,117 QPS — a 297x improvement over sequential Python.

### Sweep H — Industrial Scale

| Parameter | Value |
|---|---|
| **Variable** | N ∈ [5000, 10000, 20000, 50000] |
| **Indices** | FaissFlatIndex vs FaissHNSWIndex |

**Purpose:** Determine the exact corpus size at which HNSW outperforms brute-force.

**Key finding:** Crossover at N≈10,000 vectors. At N=50,000, HNSW delivers 2x higher throughput.

---

## 4. Telemetry Module

**File:** `benchmarks/telemetry.py` (276 lines)

### `measure_memory(build_fn)`

Measures RSS memory before, peak during, and after index construction:

```python
def measure_memory(build_fn):
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    
    # Background thread polls RSS at 10ms intervals
    monitor = threading.Thread(target=_poll_peak, daemon=True)
    monitor.start()
    
    build_fn()
    stop_event.set()
    
    return {"rss_before_mb": ..., "rss_peak_mb": ..., "rss_after_mb": ...}
```

Falls back gracefully when `psutil` is not installed (returns -1.0 for all values).

### `analyze_graph(graph_index)`

Analyses the topology of a `GraphIndex._graph` adjacency dict:

- **Degree statistics:** min, max, mean degree
- **Edge count:** total undirected edges
- **Asymmetric edge audit:** counts edges where `A→B` exists but `B→A` does not (integrity check)
- **Connected components:** BFS traversal to count isolated subgraphs

### `compute_recall(graph_results, exact_results, k)`

```python
def compute_recall(graph_results, exact_results, k):
    exact_ids = {doc.id for doc, _ in exact_results[:k]}
    graph_ids = {doc.id for doc, _ in graph_results[:k]}
    return len(exact_ids & graph_ids) / len(exact_ids)
```

Standard Recall@k: fraction of exact top-K document IDs found in graph top-K.

### `capture_search_diagnostics(index, query, k)`

Captures stdout from `GraphIndex.search()` to extract debug output:

```python
def capture_search_diagnostics(index, query, k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = index.search(query, k=k)
    # Parse stdout for "Nodes Evaluated:", "Entry Node:", etc.
```

This technique instruments the graph index without modifying its source code.

---

## 5. Synthetic Dataset Generators

### `generate_uniform(n, dim, seed)`

L2-normalised random vectors on the unit hypersphere:

```python
rng = np.random.default_rng(seed)
vectors = rng.standard_normal((n, dim))
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
return vectors / np.maximum(norms, 1e-12)
```

### `generate_clustered(n, dim, n_clusters, cluster_std, seed)`

Gaussian clusters around random unit-norm centroids:

1. Generate `n_clusters` random unit-norm centroids
2. Assign vectors to clusters via round-robin (then shuffle)
3. Add Gaussian noise (σ = `cluster_std`) to each centroid
4. Re-normalise to unit length

The `cluster_std` parameter (default 0.05) controls cluster tightness. Smaller values create denser clusters that are harder for greedy search to escape.

### `generate_at_dimension(n, dim, seed)`

A thin wrapper around `generate_uniform` provided for semantic clarity in dimensionality sweep code.

---

## 6. Output Schema

### Raw Logs (`outputs/raw/`)

JSONL format — one JSON object per query:

```json
{
  "timestamp": "2026-06-01T18:05:15+05:30",
  "git_commit": "52c2bd7",
  "seed": 42,
  "hyperparameters": {"M": 8, "ef_construction": 32, "ef_search": 64, "N": 5000, "dim": 128},
  "query_index": 0,
  "latency_s": 0.004521,
  "recall": 0.3,
  "nodes_evaluated": 64,
  "entry_node": 2741
}
```

### Aggregated Summaries (`outputs/aggregated/`)

JSON format — one file per sweep:

```json
{
  "timestamp": "...",
  "git_commit": "52c2bd7",
  "seed": 42,
  "sweep": "A",
  "variable": "N",
  "results": [
    {
      "N": 5000,
      "graph_build_time_s": 36.51,
      "exact_build_time_s": 0.0069,
      "faiss_build_time_s": 0.0146,
      "faiss_hnsw_build_time_s": 0.85,
      "memory": {"rss_before_mb": 85.2, "rss_peak_mb": 112.4, "rss_after_mb": 108.1},
      "graph_health": {"total_edges": 19842, "num_components": 1, ...},
      "query_latency": {"avg": 0.005, "p50": 0.004, "p95": 0.007, "qps": 200},
      "recall_mean": 0.297,
      "faiss_hnsw_recall_mean": 0.936,
      "exact_vs_faiss_match_rate": 1.0
    }
  ]
}
```

Every output JSON is stamped with `timestamp`, `git_commit`, `seed`, and `hyperparameters` for full reproducibility.

---

## 7. Chart Generation

**File:** `benchmarks/plots.py` (650 lines)

10 matplotlib chart generators with a unified dark theme:

| Chart | Source | Description |
|---|---|---|
| `recall_vs_ef_search.png` | Sweep C | Dual-series: Graph vs HNSW recall |
| `recall_vs_M.png` | Sweep B | Dual-series: Graph vs HNSW recall |
| `build_time_vs_N.png` | Sweep A | Log-scale: 4-engine build times |
| `query_latency_vs_N.png` | Sweep A | p50 latency: Graph vs Flat vs HNSW |
| `components_vs_N.png` | Sweep A | Bar chart: connected components |
| `sweep_f_qps_vs_N.png` | Sweep F | Exact vs FAISS QPS |
| `sweep_g_batch_qps.png` | Sweep G | Grouped bar: sequential vs batched |
| `sweep_h_qps_vs_N.png` | Sweep H | Flat vs HNSW with crossover line |
| `sweep_h_latency_vs_N.png` | Sweep H | Dual-panel: p50 and p95 |
| `sweep_h_recall_vs_N.png` | Sweep H | HNSW recall decay vs N |

### Theme

```python
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#1A1A2E",
    "axes.facecolor": "#16213E",
    "figure.dpi": 300,
    ...
})
```

All charts use 300 DPI resolution, suitable for publication.

### Colour Palette

```python
_COLORS = {
    "primary": "#6C63FF",    # GraphIndex series
    "secondary": "#FF6584",  # FaissFlatIndex series
    "accent": "#43E8D8",     # GraphIndex series (alt)
    "hnsw": "#00C896",       # FaissHNSWIndex series (teal-green)
    "warn": "#FFD93D",       # Annotations
    "muted": "#8B8B9E",      # Fill areas
}
```

### Graceful Degradation

Each plot function checks for data availability before proceeding:

```python
def plot_recall_vs_ef_search():
    path = _find_latest("sweep_c")
    if not path:
        print("  [SKIP] No sweep_c data found")
        return
```

This allows `generate_all()` to run even when only some sweeps have been executed.
