# 05 — LLM Abstraction & RAG Pipeline

> **Scope:** The LLM abstraction layer, the RAG pipeline orchestrator,
> prompt construction strategy, and the telemetry system that ties them together.

---

## 1. LLM Abstraction Layer

### 1.1 `BaseLLM` — The Abstract Contract

**File:** `src/llm/base.py`

```python
class BaseLLM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse: ...
```

The contract is minimal by design — a single method with two parameters. This narrowness ensures that *any* LLM backend can satisfy the interface, from a zero-cost mock to a production gateway routing through hundreds of providers.

### 1.2 `MockLLM` — Deterministic Simulator

**File:** `src/llm/mock.py` (138 lines)

Two operating modes:

**Dynamic mode** (default):
```python
def _build_dynamic_response(self, prompt, config):
    text = (
        f"[MOCK RESPONSE] Captured prompt length: {len(prompt)} characters."
        " System is nominal."
    )
    prompt_tokens = max(1, len(prompt) // 4)  # 1 token ≈ 4 chars
    completion_tokens = min(max(1, len(text) // 4), config.max_tokens)
    return LLMResponse(text=text, model_name="mock/stable-simulator-v1", ...)
```

**Static mode** (for tests):
```python
llm = MockLLM(static_response="Fixed answer")
resp = llm.generate("Any prompt")  # Always returns "Fixed answer"
```

**Design insights:**
- Token counts are estimated using the GPT-tokeniser heuristic (1 token ≈ 4 characters). While approximate, this provides realistic telemetry values for the UI and benchmarking pipeline.
- `config.max_tokens` is respected as a ceiling on `completion_tokens`, simulating the behaviour of real providers.
- The model name `"mock/stable-simulator-v1"` follows OpenRouter's `provider/model` naming convention, allowing mock responses to flow through the same telemetry pipeline as real responses.

### 1.3 `OpenRouterLLM` — Production Gateway

**File:** `src/llm/openrouter.py` (254 lines)

Uses the `openai` Python SDK pointed at OpenRouter's base URL:

```python
class OpenRouterLLM(BaseLLM):
    def __init__(self, default_model="google/gemini-2.5-flash", api_key=None):
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ValueError("No OpenRouter API key found.")
        self._client = openai.OpenAI(api_key=resolved_key, base_url="https://openrouter.ai/api/v1")
```

**Key design decisions:**

1. **Fail-fast construction:** The `ValueError` is raised at `__init__`, not at the first `generate()` call. This surfaces misconfiguration immediately rather than failing silently during a user query.

2. **Lazy `openai` import:** The `import openai` statement is inside `generate()` and `_build_client()`, not at module level. This means:
   - The module can be imported without `openai` installed
   - The `openai` package is only required when actually calling `generate()`
   - Other symbols (`MockLLM`, etc.) remain available regardless

3. **Structured error handling:** Three distinct exception types are caught and logged:
   - `openai.APITimeoutError` — request exceeded 60s timeout
   - `openai.APIStatusError` — non-2xx HTTP status (auth failure, rate limit, etc.)
   - `openai.APIConnectionError` — network-level failure

4. **Defensive response parsing:**
   ```python
   usage = getattr(completion, "usage", None)
   prompt_tokens = getattr(usage, "prompt_tokens", None)
   ```
   Uses `getattr` with `None` defaults rather than direct attribute access, handling providers that omit usage data.

5. **Default model:** `google/gemini-2.5-flash` — a cost-effective, fast model chosen for the default RAG use case.

---

## 2. The RAG Pipeline

### 2.1 `RAGPipeline` — Orchestrator

**File:** `src/rag/pipeline.py` (157 lines)

```python
class RAGPipeline:
    def __init__(self, index, embedding_service, llm, top_k=5):
        self._index = index
        self._embedding_service = embedding_service
        self._llm = llm
        self._top_k = top_k
        self._prompt_builder = PromptBuilder()
```

The pipeline is **stateless with respect to conversation history** — each `query()` call is an isolated, atomic operation. Multi-turn conversation is handled at the Streamlit session layer, not inside the pipeline.

### 2.2 Query Execution Flow

```python
def query(self, user_query: str, config: GenerationConfig | None = None) -> RAGResponse:
    # Step 1: Vectorize the query
    query_embedding = self._embedding_service.embed_query(user_query)

    # Step 2: Search the index
    search_results = self._index.search(query_embedding, k=self._top_k)

    # Step 3: Map into RetrievalResult objects
    retrieval_results = [
        RetrievalResult(
            document_id=doc.id,
            score=score,
            source_file=doc.metadata.get("source_file", doc.id),
            text=doc.text,
        )
        for doc, score in search_results
    ]

    # Step 4: Compile the structured prompt
    context = PromptBuilder.build_prompt(
        query=user_query,
        retrieved_documents=[doc for doc, _ in search_results],
        scores=[score for _, score in search_results],
    )

    # Step 5: Submit to the LLM backend
    llm_response = self._llm.generate(prompt=context, config=config)

    # Step 6: Bundle into RAGResponse
    return RAGResponse(
        answer=llm_response.text,
        query=user_query,
        context=context,
        retrieved_documents=retrieval_results,
        llm_response=llm_response,
    )
```

### 2.3 Empty Result Handling

```python
if not search_results:
    logger.warning(
        "RAGPipeline: index returned no results for query '%s'. "
        "The index may be empty.",
        user_query,
    )
```

When no results are found, the pipeline does **not** short-circuit. It still sends the prompt to the LLM, which will see the `(No relevant documents were found in the index.)` message from `PromptBuilder` and is instructed to state that the context is insufficient.

---

## 3. Prompt Construction — `PromptBuilder`

**File:** `src/rag/prompt_builder.py` (105 lines)

### Prompt Template

```
=== RETRIEVED CONTEXT ===

[1]
Source: filename.md
Score:  0.8542
Text:   The actual chunk text...

[2]
Source: another_file.py
Score:  0.7891
Text:   Another chunk...

========================================

Question: What is HNSW?

Instructions: Using ONLY the context documents listed above, provide
a comprehensive and accurate answer to the question. If the answer
cannot be found in the provided context, explicitly state: 'The
provided context does not contain sufficient information to answer
this question.' Do not speculate or introduce information outside
the given context.
```

### Design Decisions

1. **Numbered context blocks:** Each retrieved document gets a sequential number, source filename, similarity score, and full text. This makes it easy for both humans and LLMs to reference specific sources.

2. **Source filename extraction:** Uses `Path(doc.metadata.get("source_file", doc.id)).name` to show just the filename (not the full path), keeping the prompt compact.

3. **Score transparency:** Including the similarity score in the prompt serves two purposes:
   - **For the LLM:** Allows the model to weight higher-scoring documents more heavily
   - **For the user:** Provides auditability when viewing the constructed context in the UI

4. **Grounding instruction:** The system instruction at the end explicitly constrains the LLM:
   - "Using ONLY the context documents listed above" — prevents hallucination
   - "explicitly state: 'The provided context does not contain sufficient information'" — honesty when context is insufficient
   - "Do not speculate or introduce information outside the given context" — reinforces grounding

5. **Length validation:**
   ```python
   if len(retrieved_documents) != len(scores):
       raise ValueError(...)
   ```
   Strict positional alignment between documents and scores is enforced.

---

## 4. Telemetry Architecture

The system provides end-to-end telemetry through a chain of carrier dataclasses:

```
LLMResponse (model_name, prompt_tokens, completion_tokens, total_tokens)
     │
     └─→ RAGResponse.llm_response
              │
              ├─→ RAGResponse.retrieved_documents → list[RetrievalResult]
              │       │
              │       └─→ (document_id, score, source_file, text)
              │
              ├─→ RAGResponse.context → full prompt string
              │
              └─→ RAGResponse.answer → generated text
```

### Token Tracking

```python
# MockLLM: estimated via character heuristic
prompt_tokens = max(1, len(prompt) // 4)

# OpenRouterLLM: extracted from API response
usage = getattr(completion, "usage", None)
prompt_tokens = getattr(usage, "prompt_tokens", None)
```

### UI Telemetry Pillars (Phase 4)

The Streamlit UI renders four structured telemetry pillars for each query:

| Pillar | Content |
|---|---|
| **1. Documents** | Indexed chunk count + retrieved count + total tokens (as metric cards) |
| **2. Context** | Expandable code block showing the full compiled prompt |
| **3. Query** | Styled echo of the user's original question |
| **4. Result** | Markdown answer + per-model token breakdown |

Below the result, expandable source cards show each retrieved document with filename, score, and text snippet.

### Session-Level Aggregation

```python
session_tokens = sum(t.get("tokens") or 0 for t in st.session_state.chat_history)
queries_run = sum(1 for t in st.session_state.chat_history if t.get("role") == "assistant")
```

Cumulative metrics are computed from the chat history and displayed in the top metrics bar.

---

## 5. Dynamic Backend Selection

The Streamlit app implements a dynamic LLM/embedding backend factory:

### LLM Selection

```python
def _build_llm(model_slug, api_key=""):
    if model_slug == "mock":
        return MockLLM()
    
    resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not resolved_key:
        st.warning("No API key provided. Falling back to Mock Simulator.")
        return MockLLM()
    
    return OpenRouterLLM(default_model=model_slug, api_key=resolved_key)
```

Three-tier resolution: explicit key → environment variable → graceful fallback to mock.

### Embedding Selection

```python
def _get_embedding_service():
    provider = st.session_state.get("embedding_provider", "MiniLM (Local - 384d)")
    if provider.startswith("MiniLM"):
        if _minilm_service_cache is None:
            _minilm_service_cache = MiniLMEmbeddingService()  # Process-level singleton
        return _minilm_service_cache
    return MockEmbeddingService(dim=128)
```

The MiniLM model is cached as a **process-level singleton** to avoid reloading the 80MB model weights on every Streamlit rerun.

### Pipeline Reconstruction

The pipeline is rebuilt whenever the model or API key changes:

```python
if selected_slug != _prev_model or ui_api_key != _prev_key or pipeline is None:
    pipeline = _build_pipeline(index, selected_slug, api_key=ui_api_key)
```

This ensures the pipeline always reflects the UI's current configuration without full page reload.

---

## 6. Dynamic Model Discovery

**File:** `app/ui_components.py`

```python
def fetch_openrouter_models(api_key="") -> dict[str, str]:
    resp = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=8)
    models = {}
    for entry in resp.json().get("data", []):
        models[entry["name"]] = entry["id"]
    models.update({"Mock Simulator Mode": "mock"})  # Always available
    return models
```

The sidebar offers:
- **Static fallback:** 6 pre-configured models (Gemini, Claude, DeepSeek, Mock)
- **Live fetch:** Pulls hundreds of models from OpenRouter's catalogue
- **Free-tier filter:** Shows only models with `:free` suffix

A "Test Connection" button verifies the selected model + API key with a minimal `"Reply with only the word 'ACK'."` probe request.
