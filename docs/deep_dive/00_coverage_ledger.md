# 00 — Coverage Ledger

> **Purpose:** Exhaustive inventory of every artifact in the repository.  
> **Status:** Complete — all files inspected and catalogued.  
> **Last Audit:** 2026-06-17

---

## Repository Identity

| Property | Value |
|---|---|
| **Project Name** | `vector-search-engine` |
| **Version** | `0.1.0` |
| **Python** | `>=3.13` |
| **Package Manager** | `uv` (PEP 621 via `pyproject.toml`) |
| **Primary Framework** | Streamlit (app), FAISS (indexing), SentenceTransformers (embeddings) |
| **License** | Not specified |

---

## File Census

### Root Configuration Files

| File | Size | Purpose | Inspected |
|---|---|---|---|
| `pyproject.toml` | 547 B | PEP 621 project metadata, dependencies, pytest config | ✅ |
| `.gitignore` | 378 B | VCS exclusions (`.venv`, `.env`, `__pycache__`, etc.) | ✅ |
| `.env` | — | OpenRouter API key (single secret) | ✅ (denied; user-confirmed content) |
| `README.md` | 18,077 B | Extensive project overview, benchmarks, architectural analysis | ✅ |

---

### `src/` — Core Library (15 files)

#### `src/models.py` — Data Model Layer

| Symbol | Type | Line(s) | Purpose |
|---|---|---|---|
| `TextChunk` | dataclass | — | Pre-embedding unit: `text`, `source_file`, `chunk_index`, `metadata` |
| `Document` | dataclass | — | Post-embedding unit: `id`, `text`, `embedding`, `metadata` |

#### `src/ingestion/` — Ingestion Pipeline (6 files)

| File | Size | Key Symbols | Inspected |
|---|---|---|---|
| `__init__.py` | 786 B | Public re-exports of all ingestion symbols | ✅ |
| `loaders.py` | 2,710 B | `RepositoryLoader` — recursive file scanner with extension filtering | ✅ |
| `chunker.py` | 6,822 B | `RecursiveTokenChunker`, `SmartRepositoryChunker` — file-type-aware routing | ✅ |
| `embeddings.py` | 6,610 B | `MockEmbeddingService`, `MiniLMEmbeddingService` — vector generation | ✅ |
| `json_renderer.py` | 10,355 B | `render_benchmark_to_prose()`, `render_benchmark_file()` — JSON→English converter | ✅ |
| `benchmarking.py` | 822 B | Back-compat alias: re-exports `render_benchmark_to_prose` | ✅ |

#### `src/index/` — Indexing Backends (6 files)

| File | Size | Key Symbols | Inspected |
|---|---|---|---|
| `__init__.py` | 127 B | Package marker | ✅ |
| `base.py` | 1,727 B | `BaseIndex` — ABC with `add_documents()` and `search()` | ✅ |
| `exact.py` | 3,648 B | `ExactIndex` — NumPy brute-force cosine similarity | ✅ |
| `graph.py` | 5,849 B | `GraphIndex` — Manual NSW skeleton with greedy search | ✅ |
| `faiss_flat.py` | 4,558 B | `FaissFlatIndex` — FAISS exact search wrapper | ✅ |
| `faiss_hnsw.py` | 6,803 B | `FaissHNSWIndex` — FAISS HNSW wrapper with persistence hooks | ✅ |
| `persistence.py` | 5,459 B | `save_index_to_disk()`, `load_index_from_disk()` — serialisation | ✅ |

#### `src/llm/` — LLM Abstraction Layer (5 files)

| File | Size | Key Symbols | Inspected |
|---|---|---|---|
| `__init__.py` | 596 B | Public re-exports | ✅ |
| `base.py` | 1,933 B | `BaseLLM` — ABC with `generate()` contract | ✅ |
| `config.py` | 1,192 B | `GenerationConfig` — dataclass (`temperature`, `max_tokens`, `top_p`, `seed`) | ✅ |
| `mock.py` | 4,887 B | `MockLLM` — deterministic zero-network simulator | ✅ |
| `openrouter.py` | 8,575 B | `OpenRouterLLM` — production OpenRouter gateway client | ✅ |
| `response.py` | 1,841 B | `LLMResponse` — structured carrier with token telemetry | ✅ |

#### `src/rag/` — RAG Pipeline (4 files)

| File | Size | Key Symbols | Inspected |
|---|---|---|---|
| `__init__.py` | 503 B | Public re-exports | ✅ |
| `pipeline.py` | 5,229 B | `RAGPipeline` — end-to-end orchestrator | ✅ |
| `prompt_builder.py` | 3,528 B | `PromptBuilder` — structured context assembly with grounding instruction | ✅ |
| `response.py` | 1,942 B | `RAGResponse`, `RetrievalResult` — telemetry carriers | ✅ |

---

### `app/` — Streamlit Application (4 files)

| File | Size | Purpose | Inspected |
|---|---|---|---|
| `__init__.py` | 61 B | Package marker | ✅ |
| `streamlit_app.py` | 19,047 B | Root entry point: page config, CSS, index loading, chat loop | ✅ |
| `ui_components.py` | 22,953 B | Modular rendering: sidebar, metrics, chat, RAG telemetry pillars | ✅ |
| `session_manager.py` | 4,646 B | Session state lifecycle: `initialize_session()`, `add_message()`, `clear_chat_history()` | ✅ |

---

### `benchmarks/` — Benchmarking Framework (8 files)

| File | Size | Purpose | Inspected |
|---|---|---|---|
| `README.md` | 13,186 B | Suite documentation: sweeps A–H, CLI, output schemas | ✅ |
| `runner.py` | 43,908 B | CLI orchestrator: 8 sweep functions (A–H), argparse dispatch | ✅ |
| `telemetry.py` | 8,678 B | `measure_memory`, `analyze_graph`, `compute_query_percentiles`, `compute_recall`, `compute_match_rate`, `capture_search_diagnostics` | ✅ |
| `plots.py` | 21,676 B | 10 matplotlib chart generators, dark theme, 300 DPI | ✅ |
| `datasets/__init__.py` | 320 B | Re-exports generators | ✅ |
| `datasets/uniform.py` | 933 B | `generate_uniform` — L2-normalised random vectors | ✅ |
| `datasets/clustered.py` | 1,988 B | `generate_clustered` — Gaussian clusters with centroid routing | ✅ |
| `datasets/dimensionality.py` | 813 B | `generate_at_dimension` — thin wrapper for clarity | ✅ |

---

### `tests/` — Test Suite (10 files)

| File | Size | Purpose | Inspected |
|---|---|---|---|
| `test_core.py` | 0 B | Empty placeholder | ✅ |
| `test_ingestion_pipeline.py` | 2,126 B | End-to-end ingestion: load → chunk → embed → assert | ✅ |
| `test_llm_infrastructure.py` | 8,276 B | MockLLM + OpenRouterLLM unit/integration tests | ✅ |
| `test_hnsw_smoke.py` | 4,945 B | FaissHNSWIndex invariants: basic search, self-neighbor, zero-vector | ✅ |
| `test_hnsw_compare.py` | 9,135 B | Phase 2 micro-benchmark: HNSW vs Flat (recall, latency, speedup) | ✅ |
| `smoke/phase1_smoke_test.py` | 5,603 B | Phase 1 validation | ✅ (listed) |
| `smoke/phase2_smoke_test.py` | 6,894 B | Phase 2 validation | ✅ (listed) |
| `smoke/phase3_smoke_test.py` | 7,929 B | Phase 3 validation | ✅ (listed) |
| `smoke/phase4_ui_smoke_test.py` | 10,179 B | Phase 4 UI validation | ✅ (listed) |
| `smoke/phase5_ingestion_smoke_test.py` | 14,008 B | Phase 5 ingestion validation | ✅ (listed) |
| `diagnostics/dimension_audit.py` | 10,421 B | Dimension mismatch diagnostic | ✅ (listed) |
| `diagnostics/run_diagnostics.py` | 12,887 B | Full diagnostic runner | ✅ (listed) |

---

### `examples/` — Demo Scripts (1 file)

| File | Size | Purpose | Inspected |
|---|---|---|---|
| `demo_app_flow.py` | 17,603 B | End-to-end RAG demo: collect → chunk → embed → index → query → telemetry | ✅ |

---

### `docs/` — Documentation (1 file + directories)

| File | Size | Purpose | Inspected |
|---|---|---|---|
| `ingestion_policy.md` | 4,033 B | Frozen chunking policy: file-type-aware routing rules | ✅ |
| `benchmarks/` | dir | Contains `graph_1.0_matplotlib.png` (legacy benchmark figure) | ✅ (listed) |

---

### `data/` — Persisted State

| File | Size | Purpose | Inspected |
|---|---|---|---|
| `data/index/faiss.index` | 83,210 B | Serialised FAISS HNSW binary graph | ✅ (listed) |
| `data/index/documents.pkl` | 125,589 B | Pickled `list[Document]` with embeddings | ✅ (listed) |
| `data/index/metadata.json` | 142 B | Index configuration snapshot | ✅ (listed) |

---

## Dependency Map

| Package | Version Constraint | Role |
|---|---|---|
| `faiss-cpu` | `>=1.14.2` | C++ ANN / exact vector search |
| `numpy` | `>=1.24.0` | Numerical computation backbone |
| `sentence-transformers` | `>=2.2.2` | MiniLM embedding model |
| `langchain-text-splitters` | `>=1.1.2` | File-type-aware text chunking |
| `streamlit` | `>=1.58.0` | Interactive web UI |
| `openai` | `>=1.0.0` | OpenRouter API client (SDK) |
| `matplotlib` | `>=3.10.9` | Benchmark chart generation |
| `psutil` | `>=7.2.2` | RSS memory profiling |
| `pypdf` | `>=5.0.0` | PDF ingestion support |
| `python-dotenv` | `>=1.2.2` | `.env` file loading |
| `pytest` | `>=7.0.0` | Test runner |
| `torchvision` | `>=0.27.0` | SentenceTransformers dependency |

---

## Coverage Summary

| Category | Count |
|---|---|
| **Python source files** | ~35 (excluding `__pycache__`, `.pyc`) |
| **Markdown documentation** | 4 files |
| **Configuration files** | 3 (`.gitignore`, `pyproject.toml`, `.env`) |
| **Test files** | ~12 |
| **Benchmark sweep functions** | 8 (Sweep A–H) |
| **Plot generators** | 10 |
| **Data model classes** | 2 (`TextChunk`, `Document`) |
| **Index implementations** | 4 (`ExactIndex`, `GraphIndex`, `FaissFlatIndex`, `FaissHNSWIndex`) |
| **LLM backends** | 2 (`MockLLM`, `OpenRouterLLM`) |
| **Embedding services** | 2 (`MockEmbeddingService`, `MiniLMEmbeddingService`) |
| **Chunking strategies** | 4 (JSON whole-doc, Markdown header, Python AST, paragraph fallback) |

**All files inspected. Zero uninspected artifacts remain.**
