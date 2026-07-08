# 03 — Ingestion Pipeline Deep Dive

> **Scope:** File loading, file-type-aware chunking, embedding vectorisation,
> and the JSON-to-prose rendering subsystem.

---

## 1. Pipeline Overview

The ingestion pipeline transforms raw files into searchable vector documents through four stages:

```
Files on Disk → RepositoryLoader → SmartRepositoryChunker → EmbeddingService → Document[]
```

Each stage is independently swappable and testable. The pipeline enforces a **frozen chunking policy** (documented in `docs/ingestion_policy.md`) that prohibits universal text splitting.

---

## 2. Stage 1: File Loading — `RepositoryLoader`

**File:** `src/ingestion/loaders.py` (76 lines)

### Implementation

```python
class RepositoryLoader:
    def __init__(self, root_dir: str, extensions: set[str] | None = None):
        self._root = Path(root_dir)
        self._extensions = extensions or {".txt", ".md", ".py", ".json", ".jsonl"}

    def load(self) -> dict[str, str]:
        # Recursively walks root_dir, reads matching files,
        # returns {relative_path: file_content}
```

### Key Behaviours

1. **Extension filtering:** Only files matching `self._extensions` are loaded. Default set: `.txt`, `.md`, `.py`, `.json`, `.jsonl`.

2. **Exclusion patterns:** Skips directories starting with `.` (e.g., `.venv`, `.git`) and `__pycache__`.

3. **Encoding fallback:** Attempts UTF-8 first, falls back to `latin-1` with error replacement. This prevents ingestion crashes on binary files that slip through the extension filter.

4. **Return format:** `dict[str, str]` where keys are relative paths. This format is consumed directly by the chunker's `chunk_all()` method.

### Design Trade-off

The loader performs no content validation — it trusts the extension filter and reads everything as text. This is fast but means a `.py` file containing binary data would produce garbage text. In practice, this trade-off is acceptable because the repository contains only source-authored files.

---

## 3. Stage 2: Smart Chunking — `SmartRepositoryChunker`

**File:** `src/ingestion/chunker.py` (190 lines)

### Architecture: File-Type-Aware Routing

The chunker inspects each file's extension and dispatches to a specialised strategy:

```
.json / .jsonl  →  Whole-Document (1 file = 1 chunk)
.md             →  Markdown Header-Aware Splitting
.py             →  Python AST/Function-Based Splitting
.txt / other    →  Paragraph-Based Fallback
```

### Strategy 1: JSON/JSONL — Whole-Document

```python
if ext in (".json", ".jsonl"):
    chunks.append(TextChunk(
        text=text,
        source_file=source_file,
        chunk_index=0,
        metadata={},
    ))
```

**Rationale:** Benchmark JSON files are pre-rendered into prose by the `json_renderer` module *before* reaching the chunker. Splitting this prose would fragment experiment context (hyperparameters, metrics, conclusions) across multiple chunks, destroying retrieval quality.

### Strategy 2: Markdown — Header-Aware Splitting

```python
if ext == ".md":
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
    )
    md_chunks = md_splitter.split_text(text)
    # Then apply RecursiveCharacterTextSplitter as a size guard
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
```

**Two-pass approach:**
1. **First pass:** Split on ATX headers (`#`, `##`, `###`), producing semantically coherent sections.
2. **Second pass:** Apply a character-length guard to prevent any single section from exceeding the embedding model's context window.

The header metadata (`h1`, `h2`, `h3`) is propagated into each chunk's `metadata["header_trail"]`, enabling the RAG pipeline to display section context alongside retrieved text.

### Strategy 3: Python — Language-Aware Splitting

```python
if ext == ".py":
    py_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.PYTHON,
        chunk_size=800,
        chunk_overlap=100,
    )
```

Uses LangChain's `Language.PYTHON` mode which respects class and function boundaries. This prevents splitting a function definition across two chunks, preserving logical completeness.

### Strategy 4: Fallback — Paragraph-Based

```python
# Default for .txt and all other extensions
fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

Priority order: paragraph boundaries (`\n\n`) → line breaks (`\n`) → sentence-ending periods (`. `) → word boundaries (` `) → character-level (``). This hierarchy ensures the cleanest possible splits.

### `RecursiveTokenChunker` — Legacy / Simplified Interface

```python
class RecursiveTokenChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        ...
    def chunk_document(self, text: str, source_file: str = "unknown") -> list[TextChunk]
    def chunk_all(self, payloads: dict[str, str]) -> list[TextChunk]
```

A simpler chunker used in tests and the Streamlit upload flow. Does not perform file-type routing — applies a single `RecursiveCharacterTextSplitter` to all inputs. Retained for backward compatibility and rapid prototyping.

---

## 4. Stage 3: Embedding — `EmbeddingService`

**File:** `src/ingestion/embeddings.py` (182 lines)

### `MockEmbeddingService`

```python
class MockEmbeddingService:
    def __init__(self, dim: int = 128):
        self._dim = dim

    def embed_chunks(self, chunks: list[TextChunk]) -> list[Document]:
        # Generates deterministic random embeddings via SHA-256 seeding
        # Each chunk's text is hashed to produce a reproducible seed
        # Vector is L2-normalised to unit length
```

**Key design:** Uses `hashlib.sha256(chunk.text.encode())` to derive a deterministic seed for each chunk. This ensures:
- Same text always produces the same embedding (test reproducibility)
- Different texts produce different embeddings (avoids degenerate search)
- No model download required (offline development)

The `embed_query()` method follows the same SHA-256 seeding pattern, ensuring consistency between ingestion-time and query-time embeddings.

### `MiniLMEmbeddingService`

```python
class MiniLMEmbeddingService:
    _MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._MODEL_NAME)

    def embed_chunks(self, chunks: list[TextChunk]) -> list[Document]:
        texts = [chunk.text for chunk in chunks]
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        # Returns 384-dimensional vectors
```

**Key design decisions:**

1. **Lazy import:** `SentenceTransformer` is imported inside `__init__`, not at module level. This prevents import errors when `sentence-transformers` is not installed but other parts of the system need the module's other symbols.

2. **Batch encoding:** All chunks are encoded in a single `self._model.encode()` call, leveraging PyTorch's batched inference for GPU/CPU efficiency.

3. **384-dimensional output:** The `all-MiniLM-L6-v2` model produces 384-d vectors, which are significantly smaller than larger models (768-d or 1536-d), trading some semantic resolution for speed and memory efficiency.

4. **No explicit normalisation:** The MiniLM model produces embeddings that are already approximately normalised. The FAISS index applies its own L2-normalisation during `add_documents()`, making this safe.

### `embed_query()` Contract

Both services expose `embed_query(query: str) -> np.ndarray` for query-time embedding. This is the interface the RAGPipeline relies on (duck-typed, not formally abstract).

---

## 5. JSON-to-Prose Rendering — `json_renderer.py`

**File:** `src/ingestion/json_renderer.py` (248 lines)

### Problem Statement

Benchmark sweep outputs are structured JSON containing nested metrics, hyperparameters, and arrays of numbers. Feeding raw JSON to an LLM produces poor retrieval quality because:
- JSON syntax (`{`, `}`, `:`) wastes embedding capacity
- Semantic meaning is implicit in key names rather than explicit in natural language
- The embedding model was trained on English text, not JSON

### Solution: Structured Prose Rendering

```python
def render_benchmark_to_prose(record: dict) -> str:
    """Convert a single benchmark JSON record into rich English prose."""
```

The renderer translates structured data into natural-language paragraphs:

```python
# Input JSON:
{"sweep": "H", "N": 50000, "hnsw_recall_mean": 0.602, ...}

# Output prose:
"Sweep H — Industrial Scale: Exact vs ANN. At a corpus size of 50,000 vectors
in 128-dimensional space, the FaissHNSWIndex achieved a mean Recall@10 of 60.2%
against the FaissFlatIndex ground truth. The HNSW index reached a crossover..."
```

### Rendering Strategies

The renderer handles multiple sweep types through conditional logic:

| Sweep Type | Key Prose Elements |
|---|---|
| **Sweep A** (Scaling) | Build times per engine, memory RSS, recall, match rate |
| **Sweep B** (Topology) | M parameter effect, graph health, component count |
| **Sweep C** (Traversal) | ef_search vs recall trade-off |
| **Sweep D** (Escape) | Cross-cluster escape success rates |
| **Sweep E** (Dimensionality) | Recall degradation across dimensions |
| **Sweep F** (Implementation) | Exact vs FAISS QPS comparison, match rate |
| **Sweep G** (Batching) | Sequential vs batched throughput |
| **Sweep H** (Industrial) | Crossover point, tail latency, recall decay |

### `render_benchmark_file()`

```python
def render_benchmark_file(filepath: Path) -> str:
    """Render an entire JSON or JSONL file into prose."""
```

Handles both single-object `.json` files and multi-record `.jsonl` files. For aggregated sweep files, renders the top-level metadata plus each result entry as separate paragraphs.

### Validation in Demo Flow

The `examples/demo_app_flow.py` script includes an inline prose quality validation sweep that verifies:
1. No payload starts with `{` or `[` (i.e., no raw JSON leaked through)
2. All JSON-sourced payloads contain English sentences (period-terminated text)
3. Sweep-specific vocabulary is present (e.g., "crossover" in Sweep H, "batch" in Sweep G)

---

## 6. The Streamlit Upload Path

The Streamlit app has its own ingestion path that bypasses `RepositoryLoader` and `SmartRepositoryChunker`:

```python
# app/streamlit_app.py
def _read_uploaded_file_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    raw_bytes = uploaded_file.read()
    
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(raw_bytes))
        # Extract text from all pages
    
    if suffix == ".jsonl":
        # Parse each line, render via json_renderer
    
    if suffix == ".json":
        # Parse, render via json_renderer
    
    return text  # Plain text for .txt, .md

def _ingest_and_build_index(uploaded_files):
    embed_svc = _get_embedding_service()
    chunker = RecursiveTokenChunker(chunk_size=800, overlap=80)
    # Uses the simpler RecursiveTokenChunker, not SmartRepositoryChunker
```

**Notable difference:** The Streamlit upload path uses `RecursiveTokenChunker` (simple, no file-type routing) rather than `SmartRepositoryChunker`. This means uploaded `.py` files won't get Python-aware splitting, and uploaded `.md` files won't get header-aware splitting. The JSON/JSONL rendering is still applied at the `_read_uploaded_file_text` level.

---

## 7. Engineering Insights

### Insight 1: Chunking Strategy Matters More Than Chunk Size

The system's adoption of file-type-aware routing (rather than a universal splitter) reflects a key RAG engineering principle: **structural boundaries in source documents carry semantic signal**. A function boundary in Python or a header boundary in Markdown is not just a formatting convention — it's a signal about topic coherence that should be preserved in chunks.

### Insight 2: Pre-Rendering JSON Is a RAG Technique

The JSON-to-prose rendering pipeline is a technique for making structured data accessible to embedding models trained on natural language. This is analogous to how production RAG systems handle tables, code, and structured records — by translating them into the representation space where the embedding model has the strongest signal.

### Insight 3: The Mock/Real Duality

Every service in the ingestion pipeline has a mock counterpart:
- `MockEmbeddingService` vs `MiniLMEmbeddingService`
- `MockLLM` vs `OpenRouterLLM`

This enables the entire pipeline to run end-to-end without network access, API keys, or model downloads — essential for testing and CI.
