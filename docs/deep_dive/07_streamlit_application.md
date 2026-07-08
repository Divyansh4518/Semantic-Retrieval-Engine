# 07 — Streamlit Application & UI Architecture

> **Scope:** The Streamlit web application — page lifecycle, session management,
> CSS design system, component decomposition, and the reactive state model.

---

## 1. Application Entry Point

**File:** `app/streamlit_app.py` (549 lines)

### Boot Sequence

```python
# 1. Page configuration (MUST be first Streamlit call)
st.set_page_config(
    page_title="RAG Engine | Semantic Retrieval Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Global CSS injection (glassmorphism design system)
st.markdown("<style>...</style>", unsafe_allow_html=True)

# 3. Session state initialization
initialize_session()  # from session_manager.py

# 4. Auto-load existing index from disk
if st.session_state.loaded_index is None:
    # Check data/index/ for persisted FAISS state
    if (Path(index_dir) / "faiss.index").exists():
        idx = load_index_from_disk(index_dir)
        st.session_state.loaded_index = idx
        st.session_state.pipeline = _build_pipeline(idx, ...)

# 5. Render sidebar → returns config dict
sidebar_cfg = render_sidebar(MODELS)

# 6. Pipeline hot-swap on model/key change
if model_changed or key_changed or pipeline is None:
    pipeline = _build_pipeline(index, selected_slug, api_key)

# 7. Handle index rebuild from uploaded files
if sidebar_cfg["rebuild_clicked"] and sidebar_cfg["uploaded_files"]:
    new_index, doc_count, chunk_count = _ingest_and_build_index(files)
    save_index_to_disk(new_index, directory=index_dir)

# 8. Render main content area (header + metrics + chat + input)
```

### Key Design: Hot-Swap Pipeline

The pipeline is reconstructed whenever the LLM model or API key changes, without requiring a full page reload:

```python
_prev_model = st.session_state.get("_prev_model", "")
_prev_key = st.session_state.get("_prev_api_key", "")

if selected_slug != _prev_model or ui_api_key != _prev_key or pipeline is None:
    st.session_state["_prev_model"] = selected_slug
    st.session_state["_prev_api_key"] = ui_api_key
    pipeline = _build_pipeline(index, selected_slug, api_key=ui_api_key)
```

---

## 2. Session State Management

**File:** `app/session_manager.py` (148 lines)

### State Keys

| Key | Type | Default | Purpose |
|---|---|---|---|
| `chat_history` | `list[dict]` | `[]` | Conversation turns |
| `active_model` | `str` | `"mock"` | Current LLM model slug |
| `loaded_index` | `FaissHNSWIndex | None` | `None` | Active vector index |
| `document_count` | `int` | `0` | Source documents ingested |
| `chunk_count` | `int` | `0` | Text chunks indexed |
| `pipeline` | `RAGPipeline | None` | `None` | Active pipeline instance |
| `upload_key` | `int` | `0` | File uploader reset counter |
| `index_dir` | `str` | `"data/index/"` | Persistence directory |
| `openrouter_api_key` | `str` | `os.getenv(...)` | API key (from `.env` or sidebar) |
| `available_models` | `dict[str, str]` | Static fallback | Model catalogue |
| `models_fetched` | `bool` | `False` | Live fetch completed flag |
| `last_rag_response` | `RAGResponse | None` | `None` | Most recent response for telemetry |
| `processing` | `bool` | `False` | Processing state flag |
| `last_error` | `str | None` | `None` | Last error traceback |

### Initialisation Strategy

```python
def initialize_session():
    for key, default in _DEFAULTS.items():
        st.session_state.setdefault(key, default)
```

Uses `setdefault()` to populate missing keys without overwriting existing values. This is safe to call on every Streamlit rerun.

### Chat History Format

Each message in `chat_history` is a dict:

```python
{
    "role": "user" | "assistant",
    "content": str,              # Message text
    "sources": list | None,      # RetrievalResult objects (assistant only)
    "tokens": int | None,        # Total tokens consumed
    "rag_response": RAGResponse | None,  # Full response for telemetry replay
}
```

Storing the full `RAGResponse` in the chat history enables the UI to re-render the four telemetry pillars (documents, context, query, result) during chat replay.

---

## 3. CSS Design System

**File:** `app/streamlit_app.py` (lines 80–188)

### Design Language: Dark Glassmorphism

```css
/* Background gradient */
.stApp {
    background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #0f172a 100%);
}

/* Sidebar with translucent border */
[data-testid="stSidebar"] {
    background: rgba(15, 17, 26, 0.95) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.2) !important;
}

/* Metric cards with glass effect */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 10px;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(99,102,241,0.4);
}
```

### Colour Palette

| Colour | Usage |
|---|---|
| `#6366f1` (Indigo) | Primary accent, gradient starts, borders |
| `#8b5cf6` (Purple) | Gradient midpoints |
| `#06b6d4` (Cyan) | Gradient endpoints |
| `#10b981` (Emerald) | Status: ready/success |
| `#f59e0b` (Amber) | Status: warning/empty |
| `#e2e8f0` (Slate-200) | Text colour |
| `#a5b4fc` (Indigo-300) | Query echo text |

### Typography

```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
```

Uses Inter for UI text and JetBrains Mono for code/technical displays.

### Custom Components

| CSS Class | Purpose |
|---|---|
| `.rag-header` | Gradient header bar with title and status badge |
| `.status-badge.status-ready` | Green pill: "● Index Ready" |
| `.status-badge.status-empty` | Amber pill: "● No Index" |
| `.telemetry-query` | Indigo-bordered query echo block |

---

## 4. Component Decomposition

**File:** `app/ui_components.py` (671 lines)

### `render_sidebar(models_dict)` → `dict`

The sidebar is the primary control surface, containing:

1. **API Key Input** — password-masked text field with "Fetch Live Models" button
2. **Free-tier filter** — checkbox to show only `:free` models
3. **Model selector** — dropdown populated from static fallback or live fetch
4. **Test Connection** — probes the selected model with a micro-request
5. **Embedding provider** — MiniLM (384-d) or Mock (128-d) selector
6. **Generation parameters** — temperature (0.0–2.0) and max_tokens (128–4096) sliders
7. **Document management** — file uploader + Rebuild Index button
8. **Index status** — ready/empty indicator + chunk/document counts

Returns a config dict consumed by the main app:

```python
return {
    "selected_model_slug": str,
    "api_key": str,
    "temperature": float,
    "max_tokens": int,
    "uploaded_files": list | None,
    "rebuild_clicked": bool,
}
```

### `render_rag_answer(rag_response)` — Four Telemetry Pillars

Renders the Phase 4 telemetry layout inside a chat message:

```
┌──────────────┬──────────────┬──────────────┐
│ 📦 Indexed   │ 📎 Retrieved │ 🔤 Total     │
│    Chunks    │             │    Tokens     │
└──────────────┴──────────────┴──────────────┘

📝 View Constructed Context Block [▶ expandable]

┌─────────────────────────────────────────────┐
│ Query: What is HNSW?                         │ (indigo accent border)
└─────────────────────────────────────────────┘

[Generated Answer in Markdown]

📊 model/name · prompt: X tok · completion: Y tok · total: Z tok

📚 Retrieved Sources
  [1] filename.md — Score: 0.8542 [▶ expandable]
  [2] another.py — Score: 0.7891
```

### `render_sources(rag_response)` — Source Document Cards

Each retrieved document gets an expandable card with:
- Source filename and similarity score as metric
- Score percentage conversion
- Text snippet (first 500 chars)
- Chunk ID and character count caption

### `render_chat(chat_history)` — Chat Replay

Iterates through chat history and renders each turn:
- **User turns:** Simple markdown rendering
- **Assistant turns with RAG response:** Full four-pillar telemetry layout
- **Assistant turns without RAG response:** Plain markdown
- Compact inline source references
- Token count caption

---

## 5. File Upload & Ingestion Flow

### Supported Formats

```python
st.file_uploader(
    "Upload Documents",
    type=["txt", "md", "pdf", "json", "jsonl"],
    accept_multiple_files=True,
)
```

### Upload Processing Pipeline

```python
def _read_uploaded_file_text(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    raw_bytes = uploaded_file.read()
    
    if suffix == ".pdf":
        # pypdf: extract text from all pages
        reader = PdfReader(BytesIO(raw_bytes))
        pages = [page.extract_text() for page in reader.pages]
        return "\n\n".join(pages)
    
    if suffix == ".jsonl":
        # Parse each line, render to prose via json_renderer
        for raw_line in text.splitlines():
            record = json.loads(raw_line)
            paragraphs.append(render_benchmark_to_prose(record))
        return "\n\n".join(paragraphs)
    
    if suffix == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            return "\n\n".join(render_benchmark_to_prose(item) for item in data)
        return render_benchmark_to_prose(data)
    
    return text  # .txt, .md
```

### Index Construction

```python
def _ingest_and_build_index(uploaded_files):
    embed_svc = _get_embedding_service()  # MiniLM or Mock
    chunker = RecursiveTokenChunker(chunk_size=800, overlap=80)
    
    all_chunks = []
    for uploaded_file in uploaded_files:
        text = _read_uploaded_file_text(uploaded_file)
        file_chunks = chunker.chunk_document(text, source_file=uploaded_file.name)
        all_chunks.extend(file_chunks)
    
    documents = embed_svc.embed_chunks(all_chunks)
    
    index = FaissHNSWIndex(M=32, ef_construction=200, ef_search=64)
    index.add_documents(documents)
    return index, doc_count, len(all_chunks)
```

After construction, the index is persisted to disk and the file uploader is reset by incrementing `upload_key`.

---

## 6. Dynamic Model Discovery

### Live Fetch

```python
def fetch_openrouter_models(api_key=""):
    resp = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        timeout=8,
    )
    models = {entry["name"]: entry["id"] for entry in resp.json()["data"]}
    models["Mock Simulator Mode"] = "mock"  # Always available
    return models
```

### Free-Tier Filter

```python
def _filter_free_models(models):
    return {name: slug for name, slug in models.items()
            if slug.endswith(":free") or slug == "mock"}
```

### Connection Test

```python
def _run_connection_test(model_slug, api_key):
    llm = OpenRouterLLM(default_model=model_slug, api_key=api_key)
    probe_cfg = GenerationConfig(temperature=0.0, max_tokens=8)
    t0 = time.perf_counter()
    resp = llm.generate("Reply with only the word 'ACK'.", config=probe_cfg)
    latency_ms = (time.perf_counter() - t0) * 1000
    # Display success/failure with latency
```

---

## 7. State Machine

The application follows a Streamlit reactive state machine:

```
┌─────────────────┐
│  INITIAL STATE   │ (No index loaded)
│  pipeline=None   │
└────────┬────────┘
         │ Upload + Rebuild Index
         ▼
┌─────────────────┐
│  INDEX LOADED    │ (pipeline constructed)
│  pipeline=active │
└────────┬────────┘
         │ User types query
         ▼
┌─────────────────┐
│  QUERY ACTIVE    │ (spinner shown)
│  pipeline.query()│
└────────┬────────┘
         │ Response received
         ▼
┌─────────────────┐
│  RESULTS SHOWN   │ (telemetry rendered)
│  chat updated    │
└────────┬────────┘
         │ st.rerun()
         ▼
         └──→ Back to INDEX LOADED (chat history preserved)
```

State transitions are triggered by:
- File upload + rebuild button → index construction
- Model/key change → pipeline hot-swap
- Chat input → RAG query execution
- Clear Chat button → history reset + rerun
