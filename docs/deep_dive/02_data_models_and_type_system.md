# 02 — Data Models & Type System

> **Scope:** Deep analysis of `TextChunk`, `Document`, and every dataclass
> in the system — their fields, invariants, serialisation behaviour, and the
> design philosophy behind the two-stage data lifecycle.

---

## 1. The Two-Stage Data Lifecycle

The system enforces a strict boundary between pre-embedding and post-embedding data through two separate dataclasses:

```
Raw File → RepositoryLoader → SmartRepositoryChunker → TextChunk[]
                                                           │
                                              EmbeddingService.embed_chunks()
                                                           │
                                                       Document[]
                                                           │
                                              BaseIndex.add_documents()
```

This separation is a deliberate architectural choice: **`TextChunk` is the contract between ingestion and embedding; `Document` is the contract between embedding and everything downstream (indexing, retrieval, persistence, UI).**

---

## 2. `TextChunk` — The Pre-Embedding Unit

**File:** `src/models.py`

```python
@dataclass
class TextChunk:
    text: str
    source_file: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)
```

### Field Semantics

| Field | Type | Invariant | Purpose |
|---|---|---|---|
| `text` | `str` | Non-empty | The raw text content of the chunk |
| `source_file` | `str` | Relative path | Provenance trace back to the originating file |
| `chunk_index` | `int` | `>= 0` | Positional index within the parent document |
| `metadata` | `dict` | Arbitrary | Extensible metadata bag (e.g., `header_trail` for Markdown) |

### Design Notes

1. **No `id` field:** TextChunks are anonymous — identity is assigned later by the embedding service when creating a `Document`. This prevents premature ID generation before the chunk's final content is determined.

2. **`metadata` as extensible bag:** Different chunking strategies inject different metadata:
   - Markdown chunker: `{"header_trail": "## Section > ### Subsection"}`
   - JSON chunker: no extra metadata (whole-document strategy)
   - Python chunker: LangChain's language-aware metadata

3. **Immutability not enforced:** The dataclass does not use `frozen=True`, allowing metadata to be mutated after creation. This is a pragmatic choice — the chunker builds chunks incrementally.

---

## 3. `Document` — The Post-Embedding Unit

**File:** `src/models.py`

```python
@dataclass
class Document:
    id: str
    text: str
    embedding: np.ndarray
    metadata: dict = field(default_factory=dict)
```

### Field Semantics

| Field | Type | Invariant | Purpose |
|---|---|---|---|
| `id` | `str` | UUID v4 string | Stable unique identifier for the chunk |
| `text` | `str` | Non-empty | Preserved raw text for display and prompt building |
| `embedding` | `np.ndarray` | Shape `(dim,)`, dtype `float32` or `float64` | Dense vector representation |
| `metadata` | `dict` | Must contain `source_file` | Provenance and chunk-level metadata |

### Design Notes

1. **UUID-based identity:** The `EmbeddingService.embed_chunks()` method generates `id` as a UUID v4 string, ensuring global uniqueness without coordination. This is critical because documents may be loaded from multiple files across multiple ingestion runs.

2. **Embedding as `np.ndarray`:** Storing the vector as a raw NumPy array enables:
   - Zero-copy handoff to FAISS (`faiss.IndexFlatIP.add()`)
   - Direct NumPy dot product in `ExactIndex`
   - Efficient pickle serialisation in `documents.pkl`

3. **Text preservation:** The `text` field is carried forward into the index and retrieved during search. This is essential because the RAG pipeline needs the original text to build the context prompt — vectors alone are insufficient.

4. **Metadata propagation:** The embedding service copies `source_file` and other metadata from `TextChunk.metadata` into `Document.metadata`, maintaining provenance through the entire pipeline.

---

## 4. Supporting Dataclasses

### `GenerationConfig` (`src/llm/config.py`)

```python
@dataclass(slots=True)
class GenerationConfig:
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    seed: int | None = None
```

**Design decisions:**

- **`slots=True`:** Memory-optimised — no `__dict__` per instance.
- **Deterministic defaults:** `temperature=0.0` ensures reproducible outputs by default. This is a conscious departure from typical LLM defaults (usually 0.7–1.0) because the system prioritises deterministic RAG retrieval over creative generation.
- **`seed` field:** Supports reproducible sampling at the provider level. `None` defers to the provider's default seeding.

### `LLMResponse` (`src/llm/response.py`)

```python
@dataclass(slots=True)
class LLMResponse:
    text: str
    model_name: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
```

**Key property:**

```python
@property
def total_tokens(self) -> int | None:
    if self.prompt_tokens is None or self.completion_tokens is None:
        return None
    return self.prompt_tokens + self.completion_tokens
```

**Design decisions:**

- **Nullable token counts:** Not all providers surface usage data. The `None` sentinel prevents silently incorrect zeros from propagating into cost-tracking or rate-limit calculations.
- **`total_tokens` as derived property:** Avoids data staleness — the total is always computed live from its components.
- **`model_name` as string:** Avoids coupling to any provider's model enum. OpenRouter models use `"google/gemini-2.5-flash"` format; mocks use `"mock/stable-simulator-v1"`.

### `RetrievalResult` (`src/rag/response.py`)

```python
@dataclass
class RetrievalResult:
    document_id: str
    score: float
    source_file: str
    text: str
```

A flattened, serialisation-friendly projection of a `(Document, float)` search result. Used in the UI to display source references without carrying the full embedding vector.

### `RAGResponse` (`src/rag/response.py`)

```python
@dataclass
class RAGResponse:
    answer: str
    query: str
    context: str
    retrieved_documents: list[RetrievalResult] = field(default_factory=list)
    llm_response: LLMResponse = field(
        default_factory=lambda: LLMResponse(text="", model_name="unknown")
    )
```

**Design decisions:**

- **`context` preserved:** The full compiled prompt is stored, enabling the UI to show exactly what was sent to the LLM (transparency/debuggability).
- **Default `llm_response`:** The lambda factory creates a placeholder so the dataclass can be partially constructed during error paths.
- **`retrieved_documents` as `list[RetrievalResult]`:** Decoupled from the index's internal `(Document, float)` representation, making RAGResponse UI-friendly.

---

## 5. Type Flow Through the System

```
str (raw text)
  │
  ├─→ TextChunk (source_file, chunk_index, metadata)
  │       │
  │       ├─→ Document (id, embedding, metadata)
  │       │       │
  │       │       ├─→ BaseIndex._documents[i]
  │       │       │
  │       │       └─→ (Document, float) ← index.search()
  │       │               │
  │       │               └─→ RetrievalResult (flattened for UI)
  │       │
  │       └─→ embed_chunks() is the ONLY TextChunk→Document gateway
  │
  └─→ str (user query)
        │
        ├─→ np.ndarray ← embed_query()
        │       │
        │       └─→ index.search(vector, k) → [(Document, float)]
        │
        ├─→ PromptBuilder.build_prompt() → str (context prompt)
        │       │
        │       └─→ LLM.generate(prompt) → LLMResponse
        │
        └─→ RAGResponse (answer + context + retrieved_documents + llm_response)
```

---

## 6. Serialisation Behaviour

### Pickle (`documents.pkl`)

The `Document` list is serialised via `pickle.dump()`. This works because:
- `str`, `dict`, and `np.ndarray` are all pickle-safe
- No lambda or closure references in the dataclass fields

### FAISS Binary (`faiss.index`)

The FAISS index is serialised via `faiss.write_index()`. This contains only the vector data and graph structure — not the `Document` objects. The `persistence.py` module reconstructs the `FaissHNSWIndex` object by loading both files and re-wiring the internal state.

### JSON (`metadata.json`)

A minimal configuration snapshot containing `embedding_dim`, `document_count`, and `index_type`. Used as a sanity check during `load_index_from_disk()`.

---

## 7. Invariants & Contracts

| Invariant | Enforcement Location | Consequence of Violation |
|---|---|---|
| `Document.embedding.shape == (dim,)` | `FaissHNSWIndex.add_documents()` auto-detects dim from first document | Dimension mismatch → FAISS segfault |
| `TextChunk.text` is non-empty | Not enforced (implicit) | Empty text → empty embedding → degenerate search results |
| `Document.id` is unique | UUID v4 generation in `embed_chunks()` | Collision → incorrect document retrieval |
| `LLMResponse.total_tokens` returns `None` if either component is `None` | Property guard in `total_tokens` | Prevents silent arithmetic on `None` values |
| `RetrievalResult.score ∈ [-1, 1]` for cosine | FAISS inner product on L2-normalised vectors | Scores outside range indicate normalisation failure |
