# 08 — Testing, Quality & CI Strategy

> **Scope:** Test suite architecture, coverage analysis, smoke test phases,
> diagnostic tools, and quality assurance patterns.

---

## 1. Test Infrastructure

### Test Runner Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths   = ["tests"]
```

**Execution:**
```bash
uv run pytest tests/ -v           # Full suite
uv run pytest tests/test_*.py -v  # Top-level tests only
uv run python tests/test_hnsw_smoke.py  # Standalone mode
```

### Test File Inventory

| File | Lines | Tests | Coverage Area |
|---|---|---|---|
| `test_core.py` | 0 | 0 | Empty placeholder |
| `test_ingestion_pipeline.py` | 57 | 1 | End-to-end ingestion flow |
| `test_llm_infrastructure.py` | 228 | 9 | LLM abstraction layer |
| `test_hnsw_smoke.py` | 164 | 3 | FaissHNSWIndex invariants |
| `test_hnsw_compare.py` | 253 | 0 (script) | HNSW vs Flat micro-benchmark |

---

## 2. Test Categories

### 2.1 Unit Tests — LLM Infrastructure

**File:** `tests/test_llm_infrastructure.py`

**`GenerationConfig` tests:**
```python
def test_generation_config_defaults():
    cfg = GenerationConfig()
    assert cfg.temperature == 0.0
    assert cfg.max_tokens == 1024
    assert cfg.top_p == 1.0
    assert cfg.seed is None

def test_generation_config_custom():
    cfg = GenerationConfig(temperature=0.2, max_tokens=100, top_p=0.9, seed=42)
    assert cfg.temperature == 0.2
```

**`LLMResponse` tests:**
```python
def test_llm_response_total_tokens():
    resp = LLMResponse(text="hello", model_name="test/model",
                       prompt_tokens=10, completion_tokens=5)
    assert resp.total_tokens == 15

def test_llm_response_total_tokens_none_when_partial():
    # None when either component is missing
    assert LLMResponse(text="x", model_name="y").total_tokens is None
    assert LLMResponse(text="x", model_name="y", prompt_tokens=5).total_tokens is None
```

**`MockLLM` tests:**
```python
def test_mock_llm_dynamic_response_structure():
    resp = MockLLM().generate("What is HNSW?", cfg)
    assert isinstance(resp, LLMResponse)
    assert "[MOCK RESPONSE]" in resp.text
    assert resp.model_name == "mock/stable-simulator-v1"

def test_mock_llm_static_mode():
    mock = MockLLM(static_response="Fixed answer.")
    for _ in range(3):
        assert mock.generate("Any prompt").text == "Fixed answer."

def test_mock_llm_respects_max_tokens():
    cfg = GenerationConfig(max_tokens=1)
    resp = MockLLM().generate("A" * 1000, cfg)
    assert resp.completion_tokens <= 1
```

**`OpenRouterLLM` construction tests:**
```python
def test_openrouter_raises_without_api_key():
    # Temporarily clear env
    os.environ.pop("OPENROUTER_API_KEY", None)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterLLM()

def test_openrouter_accepts_explicit_api_key():
    llm = OpenRouterLLM(api_key="sk-or-test-dummy-key")
    assert llm is not None
```

**Live OpenRouter tests (conditionally skipped):**
```python
_LIVE_SKIP = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set"
)

@_LIVE_SKIP
def test_openrouter_live_generate_returns_llm_response():
    llm = OpenRouterLLM(default_model="google/gemini-2.5-flash")
    resp = llm.generate("Respond with only 'ACK'.", cfg)
    assert resp.text.strip() != ""
```

### 2.2 Integration Test — Ingestion Pipeline

**File:** `tests/test_ingestion_pipeline.py`

```python
def test_ingestion():
    loader = RepositoryLoader(".", extensions={".md"})
    payloads = loader.load()
    readme_payloads = {k: v for k, v in payloads.items() if k == "README.md"}
    assert readme_payloads, "README.md not found"
    
    chunker = RecursiveTokenChunker(chunk_size=1000, overlap=100)
    chunks = chunker.chunk_all(readme_payloads)
    
    embedder = MockEmbeddingService(dim=128)
    vectorized_docs = embedder.embed_chunks(chunks)
    
    assert len(vectorized_docs) > 0
    assert len(vectorized_docs[0].embedding) == 128
```

This test exercises the full pipeline: Load → Chunk → Embed → Assert. Uses real filesystem I/O (reads the actual `README.md`) but mock embeddings.

### 2.3 Smoke Tests — FaissHNSWIndex

**File:** `tests/test_hnsw_smoke.py`

**Test 1: Basic Search**
```python
def test_basic_search():
    vectors = rng.standard_normal((100, 128)).astype(np.float32)
    index = FaissHNSWIndex(M=32, ef_construction=200, ef_search=64)
    index.add_documents(_make_documents(vectors))
    results = index.search(query, k=5)
    
    assert len(results) == 5
    assert all(isinstance(s, float) for _, s in results)
    assert all(s <= 1.0 + 1e-5 for _, s in results)
    assert all(not np.isnan(s) for _, s in results)
```

**Test 2: Self-Neighbor**
```python
def test_self_neighbor():
    # Querying with vector V should return V as the top result
    assert results[0][0].id == "doc-0"
    assert results[0][1] >= 0.99  # Self-similarity ≈ 1.0
```

**Test 3: Zero-Vector Safety**
```python
def test_zero_vector_safety():
    vectors[25] = 0.0  # Inject zero vector
    results = index.search(query, k=5)
    assert all(not np.isnan(s) for _, s in results)
```

### 2.4 Micro-Benchmark — HNSW vs Flat

**File:** `tests/test_hnsw_compare.py`

A standalone benchmark script (not a pytest test) that compares:
- Build time: Flat vs HNSW
- p50/p95 query latency
- Recall@5 and Recall@10 (HNSW vs Flat ground truth)
- Speedup factor

**Invariant checks:**
```python
assert hnsw_p50_ms > 0, "HNSW latency > 0 (no silent skips)"
assert mean_recall_5 >= 0.90, "Recall@5 >= 90%"
assert mean_recall_10 >= 0.90, "Recall@10 >= 90%"
```

---

## 3. Phase-Based Smoke Tests

**Directory:** `tests/smoke/`

Five comprehensive smoke tests corresponding to the project's development phases:

| Phase | File | Lines | Focus |
|---|---|---|---|
| Phase 1 | `phase1_smoke_test.py` | 5,603 | Basic index construction + search |
| Phase 2 | `phase2_smoke_test.py` | 6,894 | HNSW vs Flat comparison |
| Phase 3 | `phase3_smoke_test.py` | 7,929 | LLM infrastructure + RAG pipeline |
| Phase 4 | `phase4_ui_smoke_test.py` | 10,179 | Streamlit UI components |
| Phase 5 | `phase5_ingestion_smoke_test.py` | 14,008 | Smart routing ingestion |

These smoke tests are designed to be run as standalone scripts during development, not as part of the automated pytest suite.

---

## 4. Diagnostic Tools

**Directory:** `tests/diagnostics/`

### `dimension_audit.py` (10,421 lines)

A diagnostic tool that audits dimension consistency across the pipeline:
- Checks embedding service output dimensions
- Verifies FAISS index dimension matches embedding dimensions
- Detects dimension mismatches between query embeddings and index

### `run_diagnostics.py` (12,887 lines)

A comprehensive diagnostic runner that exercises all major system components and reports on:
- Module import health
- Configuration validity
- Index state
- Pipeline construction

---

## 5. Test Design Patterns

### Pattern 1: Dual-Mode Tests

Many test files support both `pytest` and standalone execution:

```python
# pytest mode
def test_basic_search() -> bool:
    ...
    return ok

# Standalone mode
if __name__ == "__main__":
    results = [test_basic_search(), test_self_neighbor(), ...]
    passed = sum(results)
    sys.exit(0 if passed == len(results) else 1)
```

### Pattern 2: Conditional Skipping

Live API tests use `pytest.mark.skipif` to avoid failures in CI:

```python
_LIVE_SKIP = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set"
)
```

### Pattern 3: Rich Standalone Output

Standalone test scripts print formatted output with visual indicators:

```python
def _run_invariant(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition
```

### Pattern 4: Deterministic Seeding

All tests and benchmarks use `np.random.default_rng(seed)` with explicit seeds for strict reproducibility:

```python
rng = np.random.default_rng(42)
vectors = rng.standard_normal((100, 128)).astype(np.float32)
```

---

## 6. Coverage Gaps

| Area | Current Coverage | Gap |
|---|---|---|
| **Data models** | No dedicated tests | `TextChunk` and `Document` are exercised indirectly |
| **Chunking strategies** | No unit tests | Smart routing logic is only tested via smoke tests |
| **Persistence** | No unit tests | `save_index_to_disk` / `load_index_from_disk` tested only in integration |
| **Prompt builder** | No unit tests | `PromptBuilder.build_prompt()` tested only through the pipeline |
| **RAG pipeline** | No unit tests | Pipeline logic tested only via demo script and smoke tests |
| **JSON renderer** | No unit tests | Prose quality validated only in `demo_app_flow.py` |
| **Session manager** | No unit tests | Session state lifecycle tested only via UI |
| **UI components** | No unit tests | Rendering functions tested only via phase 4 smoke test |

### Observations

1. **Test focus is on invariants, not coverage.** The test suite prioritises verifying critical behavioural invariants (search returns correct results, scores are valid, self-neighbor works) over achieving line coverage metrics.

2. **The benchmarking suite doubles as an integration test.** Sweeps A–H exercise the full index stack under stress conditions that go beyond what typical unit tests would cover.

3. **The demo script is a comprehensive integration test.** `examples/demo_app_flow.py` runs the complete pipeline end-to-end with inline validation sweeps, effectively serving as a manual acceptance test.

4. **`test_core.py` is empty.** This suggests the project evolved organically, with tests added for specific components as they were developed rather than following a test-first methodology.
