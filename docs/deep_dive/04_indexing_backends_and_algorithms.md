# 04 — Indexing Backends & Algorithms

> **Scope:** Deep analysis of all four index implementations — their algorithms,
> data structures, complexity characteristics, and the design decisions that
> differentiate them.

---

## 1. The Index Hierarchy

All index backends implement the `BaseIndex` abstract contract:

```python
class BaseIndex(ABC):
    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None: ...

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[Document, float]]: ...
```

The return type `list[tuple[Document, float]]` pairs each result with its similarity score, ordered by descending relevance.

```
                         BaseIndex (ABC)
                             │
         ┌───────────┬───────┴────────┬──────────────┐
    ExactIndex   GraphIndex   FaissFlatIndex   FaissHNSWIndex
    (NumPy)      (Manual NSW) (FAISS C++)      (FAISS C++)
```

---

## 2. `ExactIndex` — NumPy Brute-Force

**File:** `src/index/exact.py` (103 lines)

### Algorithm

1. **Build:** Stack all document embeddings into a single `np.ndarray` matrix of shape `(N, dim)`.
2. **Search:** Compute cosine similarity via matrix-vector dot product, sort, return top-k.

### Implementation Details

```python
class ExactIndex(BaseIndex):
    def add_documents(self, documents: list[Document]) -> None:
        self._documents = list(documents)
        embeddings = np.array([doc.embedding for doc in documents])
        # L2-normalise for cosine similarity via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        self._embeddings = embeddings / norms

    def search(self, query_embedding: np.ndarray, k: int = 5):
        query_norm = query_embedding / max(np.linalg.norm(query_embedding), 1e-10)
        similarities = self._embeddings @ query_norm
        top_k_indices = np.argsort(similarities)[::-1][:k]
        return [(self._documents[i], float(similarities[i])) for i in top_k_indices]
```

### Complexity

| Operation | Time | Space |
|---|---|---|
| `add_documents` | O(N·d) | O(N·d) for the normalised matrix |
| `search` | O(N·d) | O(N) for the similarity vector |

### Design Notes

- **Normalisation at build time:** L2-normalising all vectors during `add_documents()` converts cosine similarity to a simple dot product at query time, avoiding per-query normalisation overhead.
- **Zero-vector guard:** `np.maximum(norms, 1e-10)` prevents division by zero for degenerate vectors, returning zero similarity instead of NaN.
- **No BLAS explicit call:** NumPy's `@` operator automatically dispatches to optimised BLAS routines (SGEMM for float32, DGEMM for float64), meaning this "pure Python" implementation actually leverages C-level linear algebra.

### Role in the System

`ExactIndex` serves as **ground truth** for recall computation in the benchmarking suite. Every ANN index's results are compared against `ExactIndex` to compute Recall@k.

---

## 3. `GraphIndex` — Manual NSW Skeleton

**File:** `src/index/graph.py` (160 lines)

### Algorithm

A flat Navigable Small World (NSW) graph with greedy best-first search:

1. **Build (insertion):** For each new document:
   - Compute cosine similarity to all existing nodes
   - Connect to the M most similar nodes (bidirectional edges)
   - This creates an O(N²) insertion bottleneck

2. **Search (greedy traversal):**
   - Start at a random entry point
   - Maintain a candidate set of size `ef_search`
   - Greedily explore neighbours, always moving toward higher similarity
   - Return top-k from the explored set

### Internal State

```python
class GraphIndex(BaseIndex):
    def __init__(self, M=8, ef_construction=32, ef_search=64):
        self._graph: dict[int, set[int]] = {}  # adjacency list
        self._documents: list[Document] = []
        self._embeddings: np.ndarray  # stacked embedding matrix
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
```

### The O(N²) Build Bottleneck

The insertion algorithm performs a full brute-force similarity scan for every new node:

```python
for i, doc in enumerate(documents):
    # For each new node, scan ALL existing nodes to find M nearest
    similarities = self._compute_all_similarities(doc.embedding, existing_embeddings)
    nearest_M = top_M_indices(similarities)
    self._graph[i] = set(nearest_M)
    for j in nearest_M:
        self._graph[j].add(i)  # bidirectional edge
```

This causes build time to scale quadratically:
- N=100: 0.012s
- N=1,500: 7.2s
- N=8,000: **186s**

### Search Diagnostics

The `search()` method prints diagnostic information to stdout:
- `Nodes Evaluated: N` — number of nodes visited during traversal
- `Entry Node: X` — the randomly chosen starting node
- `Explored X.XX% of graph` — fraction of the graph visited

This stdout-based diagnostics approach is captured by the benchmarking suite via `contextlib.redirect_stdout()` — a clever technique that avoids modifying the production code.

### Why This Index Exists

`GraphIndex` is deliberately *not* production-grade. It exists as an **educational implementation** that makes the NSW algorithm's mechanics visible:
- The O(N²) build bottleneck motivates hierarchical approaches (HNSW)
- The graph fragmentation at low M values demonstrates why connectivity matters
- The local-minima trapping in cross-cluster queries shows why hierarchical skip-lists are needed

The benchmarking suite's Sweeps A–E systematically prove these limitations.

---

## 4. `FaissFlatIndex` — FAISS Exact Search

**File:** `src/index/faiss_flat.py` (126 lines)

### Algorithm

Uses `faiss.IndexFlatIP` (Flat Index with Inner Product metric) for exact brute-force search in C++:

```python
class FaissFlatIndex(BaseIndex):
    def add_documents(self, documents: list[Document]) -> None:
        self._documents = list(documents)
        embeddings = np.array([doc.embedding for doc in documents], dtype=np.float32)
        # L2-normalise → Inner Product = Cosine Similarity
        faiss.normalize_L2(embeddings)
        self._embedding_dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self._embedding_dim)
        self._index.add(embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 5):
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)
        distances, indices = self._index.search(query, k)
        # distances are inner product scores (= cosine after normalisation)
```

### The Normalisation Trick

This is a critical design pattern:

```
L2-normalise all vectors → Inner Product ≡ Cosine Similarity
```

By normalising vectors to unit length before adding them to a `faiss.IndexFlatIP`, the inner product between any two vectors equals their cosine similarity. This avoids the need for `faiss.IndexFlatL2` (which computes L2 distance) and the subsequent conversion to cosine scores.

**Why `IndexFlatIP` instead of `IndexFlatL2`?**
- Inner product returns similarity scores in [−1, 1] (higher = more similar)
- L2 distance returns distances in [0, ∞) (lower = more similar)
- The RAG pipeline expects similarity scores, not distances
- No conversion step needed — scores are directly usable

### Performance Characteristics

| Operation | Time | Why It's Fast |
|---|---|---|
| `add_documents` | O(N·d) | Single contiguous memory allocation + BLAS normalisation |
| `search` | O(N·d) | SIMD-accelerated inner product via BLAS SGEMM |

FAISS achieves an **18–24x speedup** over `ExactIndex` at the same algorithmic complexity, purely from:
- SIMD vectorisation (SSE/AVX)
- Cache-friendly memory layout (contiguous float32 arrays)
- C++ elimination of Python interpreter overhead

### Role in the System

`FaissFlatIndex` serves as the **production-grade exact baseline**:
- 100% recall (provably identical results to `ExactIndex`, verified by `exact_vs_faiss_match_rate = 1.0`)
- Ground truth for HNSW recall computation in Sweep H
- Demonstrates the performance gap between algorithm and implementation

---

## 5. `FaissHNSWIndex` — Production ANN Index

**File:** `src/index/faiss_hnsw.py` (188 lines)

### Algorithm

Hierarchical Navigable Small World (HNSW) — a multi-layer skip-list graph structure:

```
Layer L  ─── sparse entry points (long-range shortcuts)
Layer ...
Layer 1  ─── denser connections
Layer 0  ─── full graph (all N vectors connected)
```

Search starts at the top layer and greedily descends, using long-range connections to quickly navigate to the relevant region before refining at the bottom layer.

### Implementation

```python
class FaissHNSWIndex(BaseIndex):
    def __init__(self, M: int = 32, ef_construction: int = 200, ef_search: int = 64):
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._index: faiss.IndexHNSWFlat | None = None
        self._documents: list[Document] = []
        self._embedding_dim: int = 0

    def add_documents(self, documents: list[Document]) -> None:
        self._documents = list(documents)
        embeddings = np.array([doc.embedding for doc in documents], dtype=np.float32)
        faiss.normalize_L2(embeddings)
        
        dim = embeddings.shape[1]
        self._embedding_dim = dim
        
        # Create HNSW index with Inner Product metric
        self._index = faiss.IndexHNSWFlat(dim, self.M, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = self.ef_construction
        self._index.hnsw.efSearch = self.ef_search
        self._index.add(embeddings)
```

### Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `M` | 32 | Max edges per node per layer. Higher → better recall, more memory, slower build |
| `ef_construction` | 200 | Build-time beam width. Higher → better graph quality, slower build |
| `ef_search` | 64 | Query-time beam width. Higher → better recall, slower search |

### The `set_ef_search()` Method

```python
def set_ef_search(self, ef_search: int) -> None:
    """Mutate the search beam width without rebuilding the index."""
    self.ef_search = ef_search
    if self._index is not None:
        self._index.hnsw.efSearch = ef_search
```

This enables Sweep C to test multiple `ef_search` values without rebuilding the graph — a significant optimisation since HNSW build times can be 93x slower than flat indices.

### Search Implementation

```python
def search(self, query_embedding: np.ndarray, k: int = 5):
    query = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query)
    distances, indices = self._index.search(query, k)
    
    results = []
    for j in range(k):
        idx = int(indices[0][j])
        if idx < 0:  # FAISS returns -1 for missing results
            continue
        results.append((self._documents[idx], float(distances[0][j])))
    return results
```

**Guard against `-1` indices:** FAISS returns `-1` for unfilled slots when `k` exceeds the number of indexed vectors. The guard prevents `IndexError` on the `self._documents` list.

### Complexity

| Operation | Time | Space |
|---|---|---|
| `add_documents` | O(N · M · log(N) · ef_construction) | O(N · M · layers) |
| `search` | O(log(N) · ef_search) | O(ef_search) for the candidate set |

The key insight: **search is O(log N)**, not O(N). This gives HNSW its fundamental advantage at large scale.

### Persistence Hooks

`FaissHNSWIndex` integrates with the persistence module:

```python
# persistence.py
def save_index_to_disk(index: FaissHNSWIndex, directory: str) -> None:
    faiss.write_index(index._index, str(faiss_path))
    with open(docs_path, "wb") as f:
        pickle.dump(index._documents, f)
    with open(meta_path, "w") as f:
        json.dump(metadata, f)

def load_index_from_disk(directory: str) -> FaissHNSWIndex:
    raw_index = faiss.read_index(str(faiss_path))
    documents = pickle.load(open(docs_path, "rb"))
    # Reconstruct FaissHNSWIndex with loaded state
    index = FaissHNSWIndex.__new__(FaissHNSWIndex)
    index._index = raw_index
    index._documents = documents
    # ... restore M, ef_construction, ef_search, _embedding_dim
```

The three-file persistence protocol splits state across:
- `faiss.index` — binary HNSW graph (FAISS-native format)
- `documents.pkl` — Python document objects with text and metadata
- `metadata.json` — configuration snapshot for validation

---

## 6. Comparative Analysis

### Build Time Scaling

| Index | N=100 | N=1,500 | N=5,000 | N=50,000 |
|---|---|---|---|---|
| `ExactIndex` | 0.0002s | 0.003s | 0.007s | — |
| `FaissFlatIndex` | ~0.001s | ~0.005s | 0.015s | 0.18s |
| `GraphIndex` | 0.012s | 7.2s | 36.5s | — (impractical) |
| `FaissHNSWIndex` | — | — | — | 17.01s |

### Query Latency (p50)

| Index | N=5,000 | N=50,000 |
|---|---|---|
| `ExactIndex` | 4.15ms | — |
| `FaissFlatIndex` | 0.218ms | 2.09ms |
| `FaissHNSWIndex` | 0.134ms | 1.008ms |

### Recall@10 (vs Flat Ground Truth)

| Index | N=5,000 | N=10,000 | N=50,000 |
|---|---|---|---|
| `FaissFlatIndex` | 1.0000 | 1.0000 | 1.0000 |
| `FaissHNSWIndex` | 0.9360 | 0.8810 | 0.6020 |
| `GraphIndex` | 0.297 (vs Exact) | — | — |

### The Crossover Point

Sweep H empirically determined that **HNSW first outperforms FaissFlatIndex at N ≈ 10,000 vectors** (128-d, M=32, ef_construction=200, ef_search=64). Below this threshold, FAISS's BLAS-accelerated matrix multiplication is faster than HNSW's graph traversal overhead.

---

## 7. Design Lessons

### Lesson 1: Algorithm vs Implementation

`ExactIndex` and `FaissFlatIndex` perform the *same algorithm* (exhaustive brute-force search). The 18–24x performance gap is purely from implementation quality: SIMD, C++, cache-friendly memory layout. This proves that algorithmic analysis alone is insufficient — systems engineering matters.

### Lesson 2: HNSW Is a Read-Heavy Architecture

The 93x build-time overhead (vs flat) is the "investment" that buys O(log N) query time. This trade-off makes HNSW ideal for read-heavy workloads (many queries, infrequent updates) but problematic for write-heavy scenarios.

### Lesson 3: Fixed ef_search Creates Recall Decay

As the dataset grows, a fixed `ef_search=64` budget explores a shrinking fraction of the corpus (from ~1.3% at N=5k to ~0.13% at N=50k). Production systems must either:
- Scale `ef_search` with N
- Accept the recall-latency trade-off as a business decision
- Use hybrid approaches (pre-filter + ANN)
