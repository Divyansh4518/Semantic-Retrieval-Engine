# 01 — Repository Census & Architecture

> **Scope:** Directory topology, dependency graph, data-flow pipeline, and
> high-level system design decisions.

---

## 1. Directory Topology

```
vector-search-engine/
├── pyproject.toml              # PEP 621 manifest (uv-managed)
├── .gitignore
├── .env                        # OpenRouter API key
├── README.md                   # Project-level thesis (≈18 KB)
│
├── src/                        # Core library — all production logic
│   ├── models.py               # TextChunk / Document dataclasses
│   ├── ingestion/              # Load → Chunk → Embed pipeline
│   │   ├── loaders.py          # RepositoryLoader (filesystem scanner)
│   │   ├── chunker.py          # Smart file-type-aware routing chunker
│   │   ├── embeddings.py       # Mock + MiniLM embedding services
│   │   ├── json_renderer.py    # Benchmark JSON → English prose translator
│   │   └── benchmarking.py     # Back-compat alias
│   ├── index/                  # Vector index backends
│   │   ├── base.py             # BaseIndex ABC
│   │   ├── exact.py            # NumPy brute-force cosine search
│   │   ├── graph.py            # Manual NSW skeleton (educational)
│   │   ├── faiss_flat.py       # FAISS exact search wrapper
│   │   ├── faiss_hnsw.py       # FAISS HNSW wrapper (production)
│   │   └── persistence.py      # Index serialisation (3-file protocol)
│   ├── llm/                    # LLM abstraction layer
│   │   ├── base.py             # BaseLLM ABC
│   │   ├── config.py           # GenerationConfig dataclass
│   │   ├── mock.py             # MockLLM (offline simulator)
│   │   ├── openrouter.py       # OpenRouterLLM (production gateway)
│   │   └── response.py         # LLMResponse carrier
│   └── rag/                    # Retrieval-Augmented Generation
│       ├── pipeline.py         # RAGPipeline orchestrator
│       ├── prompt_builder.py   # Context assembly + grounding instruction
│       └── response.py         # RAGResponse + RetrievalResult
│
├── app/                        # Streamlit web application
│   ├── streamlit_app.py        # Root entry point (549 lines)
│   ├── ui_components.py        # Modular rendering layer (671 lines)
│   └── session_manager.py      # Session state lifecycle
│
├── benchmarks/                 # Research-grade benchmarking suite
│   ├── runner.py               # CLI orchestrator (1,311 lines, 8 sweeps)
│   ├── telemetry.py            # Metrics: memory, graph health, recall
│   ├── plots.py                # 10 matplotlib chart generators
│   ├── datasets/               # Synthetic data generators
│   └── outputs/                # Raw logs, aggregated JSON, PNG figures
│
├── tests/                      # Test suite
│   ├── test_*.py               # Unit/integration tests (pytest)
│   ├── smoke/                  # Per-phase smoke tests
│   └── diagnostics/            # Dimension audit + diagnostic runner
│
├── examples/
│   └── demo_app_flow.py        # End-to-end RAG demo script
│
├── data/
│   └── index/                  # Persisted FAISS index state
│
└── docs/
    ├── ingestion_policy.md     # Frozen chunking policy document
    └── deep_dive/              # This thesis series
```

---

## 2. Architectural Layers

The system is structured into five clean, dependency-ordered layers:

```
┌─────────────────────────────────────────────────────┐
│                  Presentation Layer                  │
│          app/streamlit_app.py + ui_components        │
├─────────────────────────────────────────────────────┤
│                 Orchestration Layer                   │
│            src/rag/pipeline.py (RAGPipeline)          │
├───────────────┬──────────────┬──────────────────────┤
│  Embedding    │   Indexing   │     Generation        │
│  Service      │   Backend    │     Backend           │
│  (ingestion/) │  (index/)    │     (llm/)            │
├───────────────┴──────────────┴──────────────────────┤
│                   Data Model Layer                   │
│          src/models.py (TextChunk, Document)          │
└─────────────────────────────────────────────────────┘
```

### Layer 1: Data Models (`src/models.py`)

Two frozen dataclasses define the system's universal vocabulary:

- **`TextChunk`** — Pre-embedding: raw text with source provenance and positional metadata
- **`Document`** — Post-embedding: text + `np.ndarray` embedding vector + stable UUID

This split enforces a clear boundary: upstream code produces `TextChunk`s; the embedding service is the *only* gateway that transforms them into `Document`s.

### Layer 2: Service Modules

Three independent service domains, each behind its own abstract interface:

| Domain | ABC | Implementations | Swap Strategy |
|---|---|---|---|
| **Embedding** | (duck-typed) | `MockEmbeddingService`, `MiniLMEmbeddingService` | Constructor injection |
| **Indexing** | `BaseIndex` | `ExactIndex`, `GraphIndex`, `FaissFlatIndex`, `FaissHNSWIndex` | Constructor injection |
| **Generation** | `BaseLLM` | `MockLLM`, `OpenRouterLLM` | Constructor injection |

### Layer 3: Orchestration (`src/rag/pipeline.py`)

`RAGPipeline` is a stateless orchestrator that wires the three services into a single `query()` call:

```
query(text) → embed_query() → index.search() → PromptBuilder.build_prompt() → llm.generate() → RAGResponse
```

### Layer 4: Presentation (`app/`)

The Streamlit application is a thin reactive shell over the orchestration layer. It manages session state, renders a glassmorphism dark UI, and provides sidebar controls for model selection, embedding provider, and generation parameters.

### Layer 5: Benchmarking (`benchmarks/`)

A standalone research harness that is a *read-only consumer* of the `src/` APIs. Zero modifications to production code. Sweeps A–H systematically explore scaling, topology, traversal, escape dynamics, dimensionality, implementation quality, batching, and industrial-scale crossover.

---

## 3. End-to-End Data Flow

```
                   ┌──────────────┐
                   │  File System  │
                   │  (.md/.py/.   │
                   │   json/.txt)  │
                   └──────┬───────┘
                          │ RepositoryLoader.load()
                          ▼
                   ┌──────────────┐
                   │  Raw Text    │
                   │  Payloads    │
                   │  dict[str,   │
                   │     str]     │
                   └──────┬───────┘
                          │ SmartRepositoryChunker
                          │ (file-type routing)
                          ▼
                   ┌──────────────┐
                   │  TextChunk[] │
                   │  (text +     │
                   │   metadata)  │
                   └──────┬───────┘
                          │ EmbeddingService.embed_chunks()
                          ▼
                   ┌──────────────┐
                   │  Document[]  │
                   │  (text +     │
                   │   embedding) │
                   └──────┬───────┘
                          │ FaissHNSWIndex.add_documents()
                          ▼
                   ┌──────────────┐
                   │  FAISS Index │
                   │  (in-memory  │
                   │   HNSW graph)│
                   └──────┬───────┘
                          │ persistence.save_index_to_disk()
                          ▼
              ┌───────────┴───────────┐
              │  data/index/          │
              │  ├── faiss.index      │
              │  ├── documents.pkl    │
              │  └── metadata.json    │
              └───────────────────────┘
```

**Query-time flow:**

```
User Query → embed_query() → index.search(vector, k)
                                    │
                          ┌─────────┴──────────┐
                          │ [(Document, score)] │
                          └─────────┬──────────┘
                                    │
                    PromptBuilder.build_prompt()
                                    │
                          ┌─────────┴──────────┐
                          │  Structured Prompt  │
                          │  with numbered      │
                          │  context blocks     │
                          └─────────┬──────────┘
                                    │
                         LLM.generate(prompt)
                                    │
                          ┌─────────┴──────────┐
                          │    RAGResponse      │
                          │  (answer + sources  │
                          │   + token telemetry)│
                          └─────────────────────┘
```

---

## 4. Dependency Graph

External package dependencies form a clean DAG with no circular imports:

```
pyproject.toml
    ├── faiss-cpu        ← src/index/faiss_*.py
    ├── numpy            ← everywhere (vectors, similarity math)
    ├── sentence-transformers ← src/ingestion/embeddings.py (MiniLM)
    ├── langchain-text-splitters ← src/ingestion/chunker.py
    ├── streamlit        ← app/*.py
    ├── openai           ← src/llm/openrouter.py (lazy import)
    ├── matplotlib       ← benchmarks/plots.py
    ├── psutil           ← benchmarks/telemetry.py (optional)
    ├── pypdf            ← app/streamlit_app.py (PDF upload)
    ├── python-dotenv    ← app/session_manager.py
    └── pytest           ← tests/*.py
```

**Key design choice:** The `openai` package is imported lazily inside `OpenRouterLLM.generate()` and `_build_client()`, so the package is only required at call time — not at import time. This allows the rest of the system to function without the `openai` SDK installed.

---

## 5. Configuration & Environment

| Mechanism | File | Variables |
|---|---|---|
| PEP 621 metadata | `pyproject.toml` | `name`, `version`, `requires-python`, `dependencies` |
| Environment secrets | `.env` | `OPENROUTER_API_KEY` |
| Pytest config | `pyproject.toml` | `pythonpath = ["."]`, `testpaths = ["tests"]` |
| Runtime defaults | `session_manager.py` | 14 session state keys with defaults |
| Index persistence | `data/index/` | `faiss.index`, `documents.pkl`, `metadata.json` |

---

## 6. Key Architectural Decisions

### Decision 1: Dataclass-Based Models Over ORM

Using Python `@dataclass` (with `slots=True` where applicable) instead of Pydantic or SQLAlchemy keeps the data layer zero-dependency and fast. The `Document.embedding` field stores raw `np.ndarray` rather than serialised bytes, enabling zero-copy handoff to FAISS.

### Decision 2: Abstract Base Classes for Swappability

`BaseIndex` and `BaseLLM` enforce interface contracts via `abc.ABC`, enabling transparent backend swaps:
- Development: `MockLLM` + `MockEmbeddingService` + `ExactIndex`
- Production: `OpenRouterLLM` + `MiniLMEmbeddingService` + `FaissHNSWIndex`

### Decision 3: Streamlit Over FastAPI

The project chose Streamlit for the frontend, trading API-first design for rapid interactive prototyping. The entire UI is a single-page reactive loop with `st.session_state` as the state store.

### Decision 4: Benchmarking as a Separate Package

The `benchmarks/` directory is a standalone consumer of `src/` APIs, enforcing the principle that observability infrastructure should never modify the system it measures. All telemetry is captured via stdout redirection (`contextlib.redirect_stdout`) rather than instrumentation hooks.

### Decision 5: Three-File Persistence Protocol

Index state is split across three files (`faiss.index`, `documents.pkl`, `metadata.json`) rather than a single archive, enabling independent inspection and partial updates.
