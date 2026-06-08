# 🔍 RAG Engine — Streamlit UI

A production-ready chat interface for the Vector Search RAG Platform. Upload documents, build a semantic index on-device using FAISS HNSW, and query them through a conversational AI — all running locally, no external vector database required.

---

## What It Is

This UI sits on top of a modular **Retrieval-Augmented Generation (RAG)** pipeline built from three independently swappable components:

| Component | Role | Default |
|---|---|---|
| **Embedding Service** | Converts text chunks and queries into dense vectors | `MockEmbeddingService` (dim=128) |
| **Vector Index** | Approximate nearest-neighbor search over embedded chunks | `FaissHNSWIndex` (M=32, ef=64) |
| **Language Model** | Generates answers grounded in the retrieved context | `MockLLM` (offline) or any OpenRouter model |

The app handles the entire lifecycle: ingest → chunk → embed → index → persist → query — with a real-time chat interface and per-response telemetry.

---

## What It Can Do

- **Upload and index any `.txt`, `.md`, or `.pdf` document** in the sidebar. Documents are chunked, embedded, and stored in a local FAISS HNSW index that is saved to disk automatically.
- **Ask natural language questions** against your indexed documents. The pipeline retrieves the top-5 most semantically relevant chunks and feeds them as grounded context to the language model.
- **Switch language models** on the fly between 6 options — from offline mock mode to Gemini, Claude, and DeepSeek via OpenRouter.
- **Control generation parameters** (temperature, max tokens) with sliders.
- **Inspect retrieved sources** — every assistant response shows expandable cards with the source filename, similarity score, and a text snippet for each retrieved chunk.
- **Track session-level telemetry** in the metrics bar: indexed chunks, source documents, tokens consumed, and queries run.
- **Persist the index across sessions** — the index is automatically saved to `data/index/` after every rebuild and reloaded on the next app start.
- **Clear chat history** without losing the index.

---

## Quick Start

### 1. Install Dependencies

From the project root using `uv` (primary for this codebase):

```bash
uv sync
uv add streamlit      # if not already installed
```

Or with pip (alternative):

```bash
pip install -r requirements.txt
pip install streamlit
```

### 2. (Optional) Set Your API Key

To use a real language model, set your OpenRouter API key as an environment variable:

```bash
# Windows PowerShell
$env:OPENROUTER_API_KEY = "sk-or-..."

# Windows Command Prompt
set OPENROUTER_API_KEY=sk-or-...

# macOS / Linux
export OPENROUTER_API_KEY=sk-or-...
```

If no key is set, the app automatically uses **Mock Simulator Mode** — fully functional offline with deterministic responses.

### 3. Launch the App

From the project root using `uv` (recommended):

```bash
uv run streamlit run app/streamlit_app.py
```

Or directly with `streamlit`:

```bash
streamlit run app/streamlit_app.py
```

The app opens in your browser at `http://localhost:8501`.

### 4. Index Your Documents

1. In the **sidebar**, click **"Upload Documents"** and select one or more `.txt`, `.md`, or `.pdf` files.
2. Click **"🔄 Rebuild Index"**.
3. Wait for the success message — the index is built and saved to `data/index/` automatically.

### 5. Ask a Question

Type your question in the chat input at the bottom and press **Enter**. The pipeline retrieves relevant chunks, compiles a grounded prompt, and streams the answer back into the chat.

---

## Using the Pre-Built Demo Index

The end-to-end demo script generates a ready-to-use index from the project's own README and benchmark artifacts:

```bash
python examples/demo_app_flow.py
```

This creates `data/index/` with 177 vectors. Launch the app afterwards — it will auto-load the index on startup with no documents to upload.

---

## Sidebar — Inputs

### 🤖 Language Model

A dropdown selector. Changing the model rebuilds the pipeline immediately — no restart needed.

| Display Name | Model Slug | Notes |
|---|---|---|
| Gemini 2.5 Flash | `google/gemini-2.5-flash` | Requires `OPENROUTER_API_KEY` |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` | Requires `OPENROUTER_API_KEY` |
| Claude 3.5 Sonnet | `anthropic/claude-sonnet-4` | Requires `OPENROUTER_API_KEY` |
| Claude 3.1 Opus | `anthropic/claude-opus-4.1` | Requires `OPENROUTER_API_KEY` |
| DeepSeek V3 | `deepseek/deepseek-chat` | Requires `OPENROUTER_API_KEY` |
| **Mock Simulator Mode** | `mock` | **Offline. No API key needed.** |

> If `OPENROUTER_API_KEY` is not set and a real model is selected, the app silently falls back to Mock Simulator Mode and shows a warning banner.

---

### ⚙️ Generation Parameters

| Control | Type | Range | Default | Effect |
|---|---|---|---|---|
| **Temperature** | Slider | 0.0 – 2.0 | 0.3 | Controls output randomness. `0.0` = deterministic, `2.0` = highly creative. |
| **Max Tokens** | Slider | 128 – 4096 | 1024 | Hard ceiling on tokens generated per response. |

---

### 📄 Document Management

| Control | Description |
|---|---|
| **Upload Documents** | Multi-file uploader. Accepts `.txt`, `.md`, `.pdf`. Files are read, chunked (800 char target, 80 char overlap), and embedded. |
| **🔄 Rebuild Index** | Triggers full ingestion of all uploaded files. Builds a new HNSW index and saves it to `data/index/`. |
| **🗑️ Clear Chat** | Wipes the current conversation history. Does not affect the loaded index. |

---

### Index Status (Read-Only)

| Indicator | Description |
|---|---|
| **● Index Ready** (green) | An index is loaded and the pipeline is active. |
| **● No Index** (amber) | No index has been loaded yet. Upload documents and rebuild. |
| **Chunks** | Total text chunks currently in the vector index. |
| **Docs** | Total source files loaded before chunking. |

---

## Main Area — Outputs

### Metrics Bar

Displayed at the top of the page. Updates after every query.

| Metric | Description |
|---|---|
| **📦 Indexed Chunks** | Total chunks stored in the HNSW index. |
| **📁 Source Documents** | Total source files loaded this session. |
| **🔤 Tokens Used** | Cumulative prompt + completion tokens across all queries this session. |
| **💬 Queries Run** | Number of RAG pipeline queries executed. |

---

### Chat Interface

Standard two-role conversation view:

| Role | Description |
|---|---|
| **User** | Your question, shown in a right-aligned bubble. |
| **Assistant** | The model's answer, grounded in retrieved context. |

Each **assistant message** includes:

- The generated answer text.
- A **compact source reference line** directly under the answer: `Sources: filename.md (0.312) · other.txt (0.287)` — showing filenames and cosine similarity scores.
- A **token count caption** for that turn (when telemetry is available).

---

### Retrieved Sources (Expandable)

Below every assistant message, up to 5 source cards are shown. Each card is an expandable block containing:

| Field | Description |
|---|---|
| **Header** | `[N] filename — Score: X.XXXX` |
| **Source path** | Full relative path to the original file. |
| **Similarity** | Cosine similarity score and percentage (e.g. `0.3124 (31.2%)`). The first result is expanded by default. |
| **Text snippet** | Up to 500 characters of the retrieved chunk, styled in a left-bordered callout block. |
| **Chunk ID** | The internal chunk identifier (e.g. `README:12`) and character count. |

---

## Session State Keys

These keys persist across Streamlit reruns within a single browser session:

| Key | Type | Description |
|---|---|---|
| `chat_history` | `list[dict]` | All conversation turns with role, content, sources, and tokens. |
| `active_model` | `str` | Currently selected model slug. |
| `loaded_index` | `FaissHNSWIndex \| None` | The active vector index instance. |
| `document_count` | `int` | Source document count (pre-chunking). |
| `chunk_count` | `int` | Text chunk count (post-chunking). |
| `pipeline` | `RAGPipeline \| None` | The wired-up pipeline for the current index + model. |
| `index_dir` | `str` | Disk path for index persistence (default: `data/index/`). |
| `upload_key` | `int` | Incremented on each rebuild to reset the file uploader widget. |

---

## Disk Persistence

Every successful index rebuild writes three files to `data/index/`:

| File | Format | Contents |
|---|---|---|
| `faiss.index` | FAISS binary | The full C++ HNSW graph — vectors and topology. |
| `documents.pkl` | Python pickle | `list[Document]` with text, IDs, and metadata. |
| `metadata.json` | JSON | Index config: `M`, `ef_construction`, `ef_search`, `embedding_dim`, `document_count`. |

On the **next app start**, these files are detected automatically and the index is restored without any user action.

---

## Pipeline Internals (What Happens Per Query)

```
User types question
        │
        ▼
embed_query(question)  →  128-dim unit vector
        │
        ▼
FaissHNSWIndex.search(vector, k=5)  →  top-5 (Document, score) pairs
        │
        ▼
Map to RetrievalResult objects  →  [document_id, score, source_file, text]
        │
        ▼
PromptBuilder.build_prompt()  →  numbered context block + grounding instruction
        │
        ▼
LLM.generate(prompt, config)  →  LLMResponse [text + token telemetry]
        │
        ▼
RAGResponse  →  [answer, query, context, retrieved_documents, llm_response]
        │
        ▼
Streamlit renders answer + source cards + token caption
```

---

## File Structure

```
app/
├── README.md              ← You are here
├── __init__.py
├── session_manager.py     ← st.session_state lifecycle
├── streamlit_app.py       ← Root entry point (run this)
└── ui_components.py       ← Rendering functions (sidebar, chat, metrics, sources)
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No index loaded` warning on start | Upload documents and click Rebuild Index, or run `python examples/demo_app_flow.py` first. |
| Model returns mock responses even after selecting a real model | Set `OPENROUTER_API_KEY` in your environment before launching the app. |
| `UnicodeDecodeError` on upload | The app tries UTF-8 then falls back to Latin-1. Binary files (images, etc.) are not supported. |
| Index rebuild is slow | Normal for large document sets — chunking and embedding all run in-process. Use fewer/smaller files for faster iteration. |
| App won't start | Ensure you run `streamlit run app/streamlit_app.py` from the **project root**, not from inside `app/`. |
