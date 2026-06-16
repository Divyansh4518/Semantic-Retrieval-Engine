"""
app/streamlit_app.py
---------------------
Root entry point for the Vector Search RAG Platform.

Run with:
    streamlit run app/streamlit_app.py

Architecture
------------
This module wires together:
  - ``session_manager``  — persistent state across reruns
  - ``ui_components``    — modular rendering functions
  - ``RAGPipeline``      — end-to-end retrieval-augmented generation
  - ``FaissHNSWIndex``   — HNSW vector index (load from disk or rebuild)
  - ``persistence``      — disk save/load for the HNSW index

Phase 3 Update: LLM selection is now driven entirely by the UI.
  - If model == "mock"              → MockLLM (no key needed)
  - If model is real + key present  → OpenRouterLLM(api_key=ui_key)
  - If model is real + key missing  → warning + fallback to MockLLM
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from io import BytesIO
from pathlib import Path

import streamlit as st

# Ensure project root is on path when launched from any directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Internal imports — all from the existing working core components
from app.session_manager import initialize_session, add_message, clear_chat_history
from app.ui_components import (
    MODELS,
    render_chat,
    render_metrics,
    render_sidebar,
    render_sources,
    render_rag_answer,
)
from src.index.faiss_hnsw import FaissHNSWIndex
from src.index.persistence import load_index_from_disk, save_index_to_disk
from src.ingestion import (
    MockEmbeddingService,
    MiniLMEmbeddingService,
    RecursiveTokenChunker,
    render_benchmark_to_prose,
)
from src.llm.config import GenerationConfig
from src.llm.mock import MockLLM
from src.rag.pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Page configuration (must be the FIRST Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Engine | Semantic Retrieval Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Vector Search RAG Platform — FAISS HNSW + Streamlit",
    },
)

# ---------------------------------------------------------------------------
# Global CSS — premium dark glassmorphism design system
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Global Reset ── */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #0f172a 100%); }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(15, 17, 26, 0.95) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.2) !important;
    }
    [data-testid="stSidebar"] .stMarkdown h2 { color: #e2e8f0 !important; }

    /* ── Header gradient bar ── */
    .rag-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(99, 102, 241, 0.3);
    }
    .rag-header h1 {
        color: #fff !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }
    .rag-header p { color: rgba(255,255,255,0.85) !important; margin: 0.25rem 0 0 0 !important; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 10px;
        padding: 0.75rem 1rem !important;
        transition: border-color 0.2s;
    }
    [data-testid="stMetric"]:hover { border-color: rgba(99,102,241,0.4); }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {
        background: rgba(255,255,255,0.06) !important;
        border-color: rgba(99,102,241,0.3) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.25) !important;
    }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: transform 0.15s, box-shadow 0.15s !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(99,102,241,0.4) !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
    }

    /* ── Divider ── */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* ── Status indicators ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-ready { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
    .status-empty { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }

    /* ── Telemetry pillars ── */
    .telemetry-query {
        background: rgba(99,102,241,0.08);
        border-left: 3px solid #6366f1;
        border-radius: 6px;
        padding: 0.5rem 0.9rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        color: #a5b4fc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session initialization
# ---------------------------------------------------------------------------

initialize_session()

# ---------------------------------------------------------------------------
# Phase 3: Dynamic embedding service factory — driven by sidebar selector
# ---------------------------------------------------------------------------

_MINILM_DIM = 384
_MOCK_DIM = 128

# Module-level cache so the MiniLM model is only loaded once per process
_minilm_service_cache: MiniLMEmbeddingService | None = None


def _get_embedding_service():
    """
    Return the embedding service matching the current sidebar selection.

    MiniLM is cached as a process-level singleton to avoid reloading the
    model weights on every Streamlit rerun.  Mock is cheap to construct.
    """
    global _minilm_service_cache
    provider = st.session_state.get("embedding_provider", "MiniLM (Local - 384d)")
    if provider.startswith("MiniLM"):
        if _minilm_service_cache is None:
            _minilm_service_cache = MiniLMEmbeddingService()
        return _minilm_service_cache
    return MockEmbeddingService(dim=_MOCK_DIM)


def _get_faiss_dim() -> int:
    """Return the FAISS index dimension matching the active embedding provider."""
    provider = st.session_state.get("embedding_provider", "MiniLM (Local - 384d)")
    return _MINILM_DIM if provider.startswith("MiniLM") else _MOCK_DIM


# ---------------------------------------------------------------------------
# Phase 3: Dynamic LLM builder — driven by UI key, not env variable
# ---------------------------------------------------------------------------

def _build_llm(model_slug: str, api_key: str = ""):
    """
    Construct the LLM backend based on the UI-selected model and the
    API key entered in the sidebar.

    Logic
    -----
    - ``model_slug == "mock"``         → MockLLM (offline, no key needed)
    - real model + api_key present     → OpenRouterLLM(api_key=api_key)
    - real model + api_key missing     → warning banner + fallback MockLLM
    """
    if model_slug == "mock":
        return MockLLM()

    # Resolve key: UI input takes priority, then env fallback
    resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")

    if not resolved_key:
        st.warning(
            "No API key provided. Enter your OpenRouter key in the sidebar "
            "or set OPENROUTER_API_KEY. Falling back to Mock Simulator.",
            icon="⚠️",
        )
        return MockLLM()

    from src.llm.openrouter import OpenRouterLLM
    return OpenRouterLLM(default_model=model_slug, api_key=resolved_key)


def _build_pipeline(
    index: FaissHNSWIndex,
    model_slug: str,
    api_key: str = "",
) -> RAGPipeline:
    """Wire up a fresh RAGPipeline for the given index, model, and API key."""
    llm = _build_llm(model_slug, api_key=api_key)
    return RAGPipeline(
        index=index,
        embedding_service=_get_embedding_service(),
        llm=llm,
        top_k=10,
    )


def _read_uploaded_file_text(uploaded_file) -> str:
    """Convert an uploaded file into ingestible text based on its suffix."""
    suffix = Path(uploaded_file.name).suffix.lower()
    raw_bytes = uploaded_file.read()

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "The 'pypdf' package is required to ingest PDF uploads. "
                "Install it with: uv add pypdf"
            ) from exc

        reader = PdfReader(BytesIO(raw_bytes))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text)
        return "\n\n".join(pages)

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1", errors="replace")

    if suffix == ".jsonl":
        paragraphs: list[str] = []
        for raw_line in text.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                paragraphs.append(raw_line)
                continue
            if isinstance(record, dict):
                paragraphs.append(render_benchmark_to_prose(record))
            else:
                paragraphs.append(json.dumps(record, ensure_ascii=False))
        return "\n\n".join(paragraphs)

    if suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text

        if isinstance(data, list):
            return "\n\n".join(
                render_benchmark_to_prose(item) if isinstance(item, dict) else json.dumps(item, ensure_ascii=False)
                for item in data
            )
        if isinstance(data, dict):
            return render_benchmark_to_prose(data)
        return json.dumps(data, ensure_ascii=False)

    return text


def _ingest_and_build_index(uploaded_files) -> tuple[FaissHNSWIndex, int, int]:
    """
    Run the full ingestion pipeline on uploaded files and build a new index.

    The FAISS index dimension is automatically matched to the active
    embedding provider (384-d for MiniLM, 128-d for Mock).

    Returns (index, document_count, chunk_count).
    """
    embed_svc = _get_embedding_service()
    chunker = RecursiveTokenChunker(chunk_size=800, overlap=80)
    all_chunks = []
    doc_count = 0

    for uploaded_file in uploaded_files:
        text = _read_uploaded_file_text(uploaded_file)
        file_chunks = chunker.chunk_document(text, source_file=uploaded_file.name)
        all_chunks.extend(file_chunks)
        doc_count += 1

    documents = embed_svc.embed_chunks(all_chunks)

    if documents:
        print(type(embed_svc).__name__)
        print(documents[0].embedding.shape)

    index = FaissHNSWIndex(M=32, ef_construction=200, ef_search=64)
    if documents:
        index.add_documents(documents)
        print(index._embedding_dim)

    return index, doc_count, len(all_chunks)


# ---------------------------------------------------------------------------
# Auto-load existing index from disk on first run
# ---------------------------------------------------------------------------

if st.session_state.loaded_index is None:
    index_dir = st.session_state.index_dir
    faiss_path = Path(index_dir) / "faiss.index"
    if faiss_path.exists():
        try:
            with st.spinner("Loading existing index from disk..."):
                idx = load_index_from_disk(index_dir)
                st.session_state.loaded_index = idx
                st.session_state.chunk_count = len(idx._documents)
                st.session_state.pipeline = _build_pipeline(
                    idx,
                    st.session_state.active_model,
                    api_key=st.session_state.get("openrouter_api_key", ""),
                )
        except Exception:
            pass  # Start fresh if loading fails

# ---------------------------------------------------------------------------
# Sidebar — renders controls, returns config dict
# ---------------------------------------------------------------------------

sidebar_cfg = render_sidebar(MODELS)
selected_slug = sidebar_cfg["selected_model_slug"]
ui_api_key = sidebar_cfg["api_key"]

# Rebuild pipeline if model or API key changed
_prev_model = st.session_state.get("_prev_model", "")
_prev_key   = st.session_state.get("_prev_api_key", "")

if (
    selected_slug != _prev_model
    or ui_api_key != _prev_key
    or st.session_state.pipeline is None
):
    st.session_state["_prev_model"]   = selected_slug
    st.session_state["_prev_api_key"] = ui_api_key
    st.session_state.active_model = selected_slug

    if st.session_state.loaded_index is not None:
        st.session_state.pipeline = _build_pipeline(
            st.session_state.loaded_index,
            selected_slug,
            api_key=ui_api_key,
        )

# ---------------------------------------------------------------------------
# Handle index rebuild from uploaded files
# ---------------------------------------------------------------------------

if sidebar_cfg["rebuild_clicked"] and sidebar_cfg["uploaded_files"]:
    with st.spinner("Chunking, embedding, and indexing documents..."):
        try:
            new_index, doc_count, chunk_count = _ingest_and_build_index(
                sidebar_cfg["uploaded_files"]
            )
            st.session_state.loaded_index = new_index
            st.session_state.document_count = doc_count
            st.session_state.chunk_count = chunk_count
            st.session_state.pipeline = _build_pipeline(
                new_index, selected_slug, api_key=ui_api_key
            )

            # Persist index to disk
            save_index_to_disk(new_index, directory=st.session_state.index_dir)
            st.session_state.upload_key += 1
            st.success(
                f"Index built: {chunk_count} chunks from {doc_count} document(s).",
                icon="✅",
            )
        except Exception as exc:
            st.error(f"Index build failed: {exc}", icon="❌")
            st.session_state.last_error = traceback.format_exc()

elif sidebar_cfg["rebuild_clicked"] and not sidebar_cfg["uploaded_files"]:
    st.sidebar.warning("Upload at least one file before rebuilding.", icon="⚠️")

# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------

# ── Header ──
status = st.session_state.loaded_index
status_html = (
    '<span class="status-badge status-ready">● Index Ready</span>'
    if status is not None
    else '<span class="status-badge status-empty">● No Index</span>'
)

model_label = selected_slug if selected_slug != "mock" else "Mock Simulator"

st.markdown(
    f"""
    <div class="rag-header">
        <h1>🔍 Semantic RAG Platform</h1>
        <p>FAISS HNSW · Retrieval-Augmented Generation · Real-time Telemetry
           &nbsp;&nbsp;{status_html}
           &nbsp;&nbsp;<span style="font-size:0.75rem; opacity:0.8;">Model: {model_label}</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Session-level metrics row ──
session_tokens = sum(
    t.get("tokens") or 0 for t in st.session_state.chat_history
)
queries_run = sum(
    1 for t in st.session_state.chat_history if t.get("role") == "assistant"
)
render_metrics(
    chunk_count=st.session_state.chunk_count,
    document_count=st.session_state.document_count,
    total_tokens=session_tokens,
    queries_run=queries_run,
)

st.divider()

# ── Chat history ──
chat_container = st.container()
with chat_container:
    render_chat(st.session_state.chat_history)

# ── User input ──
if query := st.chat_input(
    "Ask a question about your documents...",
    key="chat_input",
):
    add_message("user", query)

    if st.session_state.pipeline is None:
        add_message(
            "assistant",
            "No index is loaded yet. Please upload documents and click **Rebuild Index** in the sidebar.",
        )
        st.rerun()
    else:
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                cfg = GenerationConfig(
                    temperature=sidebar_cfg["temperature"],
                    max_tokens=sidebar_cfg["max_tokens"],
                )
                try:
                    rag_resp = st.session_state.pipeline.query(query, config=cfg)

                    # Phase 4: render 4-pillar telemetry layout
                    render_rag_answer(rag_resp)

                    # Store for session replay
                    st.session_state.last_rag_response = rag_resp

                    tokens = rag_resp.llm_response.total_tokens

                    add_message(
                        "assistant",
                        rag_resp.answer,
                        sources=rag_resp.retrieved_documents,
                        tokens=tokens,
                        rag_response=rag_resp,
                    )

                except Exception as exc:
                    error_msg = f"Pipeline error: {exc}"
                    st.error(error_msg, icon="❌")
                    add_message("assistant", f"*Error:* {error_msg}")

        st.rerun()
