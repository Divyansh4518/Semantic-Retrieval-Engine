"""
app/ui_components.py
---------------------
Modular, reusable Streamlit UI presentation layer.

All rendering functions are pure side-effects: they write directly to the
Streamlit frame and return None (or control values via st.session_state).
Each function is independently testable via static import validation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Static fallback model registry (used when offline)
# ---------------------------------------------------------------------------

MODELS: dict[str, str] = {
    "Gemini 2.5 Flash": "google/gemini-2.5-flash",
    "Gemini 2.5 Pro": "google/gemini-2.5-pro",
    "Claude 3.5 Sonnet": "anthropic/claude-sonnet-4",
    "Claude 3.1 Opus": "anthropic/claude-opus-4.1",
    "DeepSeek V3": "deepseek/deepseek-chat",
    "Mock Simulator Mode": "mock",
}

# Always-present mock entry appended at the end of dynamic lists
_MOCK_ENTRY: dict[str, str] = {"Mock Simulator Mode": "mock"}

# Reverse lookup: slug -> friendly name (for static fallback)
_SLUG_TO_NAME: dict[str, str] = {v: k for k, v in MODELS.items()}


# ---------------------------------------------------------------------------
# Phase 1: Dynamic OpenRouter model discovery
# ---------------------------------------------------------------------------

def fetch_openrouter_models(api_key: str = "") -> dict[str, str]:
    """
    Fetch the live model catalogue from OpenRouter and return a
    ``{friendly_name: model_id}`` mapping.

    The request is issued to ``https://openrouter.ai/api/v1/models`` using a
    standard GET with an optional Bearer token.  If the call fails for any
    reason (network error, timeout, bad JSON), the static fallback ``MODELS``
    dict is returned so the app remains functional offline.

    Parameters
    ----------
    api_key:
        Optional OpenRouter API key.  Improves rate limits but is not
        required for the public catalogue endpoint.

    Returns
    -------
    dict[str, str]
        ``{display_name: model_id}`` — may include dozens of entries when
        the live fetch succeeds, or 6 static entries on failure.
    """
    try:
        import requests

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()

        models: dict[str, str] = {}
        for entry in data.get("data", []):
            model_id: str = entry.get("id", "")
            model_name: str = entry.get("name", model_id)
            if model_id and model_name:
                # De-duplicate display names by appending id suffix when needed
                display = model_name if model_name not in models else f"{model_name} ({model_id})"
                models[display] = model_id

        if models:
            # Always guarantee Mock is available at the bottom
            models.update(_MOCK_ENTRY)
            return models

    except Exception:
        pass  # Any failure falls through to the static fallback

    return dict(MODELS)  # return a copy of the static fallback


def _filter_free_models(models: dict[str, str]) -> dict[str, str]:
    """
    Return only models whose slug ends with ``:free``, plus Mock.

    This mirrors OpenRouter's free-tier convention where free variants
    carry the ``:free`` suffix on their model ID.
    """
    filtered = {
        name: slug
        for name, slug in models.items()
        if slug.endswith(":free") or slug == "mock"
    }
    # Guarantee Mock is always present
    if "Mock Simulator Mode" not in filtered:
        filtered.update(_MOCK_ENTRY)
    return filtered if len(filtered) > 1 else dict(MODELS)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar(models_dict: dict[str, str] = MODELS) -> dict[str, Any]:
    """
    Render the full application sidebar and return the selected configuration.

    Sections
    --------
    1. API Key input (password-masked) + Fetch Models button
    2. Free-tier filter toggle
    3. Dynamic model selector dropdown
    4. Test Connection button
    5. Generation parameter sliders (temperature, max_tokens)
    6. Document uploader with Rebuild Index action
    7. Index status metrics

    Parameters
    ----------
    models_dict:
        Fallback mapping of {friendly_name: model_slug}.  The sidebar will
        prefer ``st.session_state.available_models`` when populated.

    Returns
    -------
    dict with keys:
        selected_model_slug: str
        api_key: str
        temperature: float
        max_tokens: int
        uploaded_files: list | None
        rebuild_clicked: bool
    """
    with st.sidebar:
        # ── Brand header ──
        st.markdown(
            """
            <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
                <div style='font-size: 2rem;'>🔍</div>
                <h2 style='margin:0; font-weight:700; color:#e2e8f0;'>RAG Engine</h2>
                <p style='margin:0; font-size:0.75rem; color:#94a3b8;'>
                    Semantic Retrieval · Augmented Generation
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Phase 1: API Key Input ──
        st.markdown("#### 🔑 OpenRouter API Key")
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=st.session_state.get("openrouter_api_key", ""),
            placeholder="sk-or-...",
            label_visibility="collapsed",
            key="api_key_input",
            help="Paste your OpenRouter key to unlock live models. Leave blank for Mock mode.",
        )
        st.session_state["openrouter_api_key"] = api_key

        fetch_col, _ = st.columns([3, 1])
        with fetch_col:
            if st.button(
                "🔄 Fetch Live Models",
                use_container_width=True,
                key="fetch_models_btn",
                help="Pull the latest model list from OpenRouter.",
            ):
                with st.spinner("Fetching models from OpenRouter..."):
                    fetched = fetch_openrouter_models(api_key=api_key)
                    st.session_state["available_models"] = fetched
                    st.session_state["models_fetched"] = True
                n = len(fetched) - 1  # exclude Mock from count
                st.success(f"Loaded {n} models.", icon="✅")

        st.divider()

        # ── Phase 1: Dynamic Model Dropdown ──
        st.markdown("#### 🤖 Language Model")

        # Free-tier filter
        show_only_free = st.checkbox(
            "Show only free models",
            value=False,
            key="free_tier_filter",
            help="Filter to models with the :free suffix on OpenRouter.",
        )

        # Resolve the working model dict for this render
        raw_models: dict[str, str] = st.session_state.get(
            "available_models", models_dict
        )
        working_models = (
            _filter_free_models(raw_models) if show_only_free else raw_models
        )

        friendly_names = list(working_models.keys())
        current_slug = st.session_state.get("active_model", "mock")

        # Find display name for current slug
        reverse = {v: k for k, v in working_models.items()}
        current_name = reverse.get(current_slug, friendly_names[-1])

        try:
            default_idx = friendly_names.index(current_name)
        except ValueError:
            default_idx = len(friendly_names) - 1

        selected_name = st.selectbox(
            "Select Model",
            options=friendly_names,
            index=default_idx,
            label_visibility="collapsed",
            key="model_selector",
        )
        selected_slug = working_models[selected_name]
        st.session_state.active_model = selected_slug

        if selected_slug == "mock":
            st.info("Mock Simulator Mode: No API key required.", icon="🧪")
        else:
            st.caption(f"`{selected_slug}`")
            if not api_key:
                st.warning("API key required for this model.", icon="⚠️")

        # ── Phase 2: Test Connection Button ──
        if st.button(
            "⚡ Test Connection",
            use_container_width=True,
            key="test_conn_btn",
            help="Verify the current model + API key works end-to-end.",
        ):
            _run_connection_test(selected_slug, api_key)

        st.divider()

        # ── Embedding Provider Selector ──
        st.markdown("#### 🧠 Embedding Provider")
        embedding_options = ["MiniLM (Local - 384d)", "Mock Simulator (Testing - 128d)"]
        current_provider = st.session_state.get("embedding_provider", embedding_options[0])
        try:
            provider_idx = embedding_options.index(current_provider)
        except ValueError:
            provider_idx = 0

        selected_provider = st.selectbox(
            "Embedding Provider",
            options=embedding_options,
            index=provider_idx,
            label_visibility="collapsed",
            key="embedding_provider_selector",
            help=(
                "MiniLM: real semantic 384-d vectors (requires first-run model download). "
                "Mock: deterministic random 128-d vectors for offline testing."
            ),
        )
        st.session_state["embedding_provider"] = selected_provider

        if selected_provider.startswith("MiniLM"):
            st.caption("🟢 `all-MiniLM-L6-v2` · 384-d · local inference")
        else:
            st.caption("🧪 Mock vectors · 128-d · no model required")

        st.divider()

        # ── Generation Parameters ──
        st.markdown("#### ⚙️ Generation Parameters")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.3,
            step=0.05,
            help="Controls randomness. 0 = deterministic, 2 = highly creative.",
            key="temperature_slider",
        )
        max_tokens = st.slider(
            "Max Tokens",
            min_value=128,
            max_value=4096,
            value=1024,
            step=128,
            help="Maximum tokens the model may generate per response.",
            key="max_tokens_slider",
        )

        st.divider()

        # ── Document Management ──
        st.markdown("#### 📄 Document Management")
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["txt", "md", "pdf", "json", "jsonl", "py"],
            accept_multiple_files=True,
            key=f"file_uploader_{st.session_state.get('upload_key', 0)}",
            help="Upload .txt, .md, .pdf, .json, .jsonl, or .py files to index.",
        )

        rebuild_clicked = st.button(
            "🔄 Rebuild Index",
            use_container_width=True,
            type="primary",
            key="rebuild_btn",
            help="Re-chunk and re-embed all uploaded documents.",
        )

        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True,
            key="clear_chat_btn",
        ):
            st.session_state.chat_history = []
            st.session_state.last_rag_response = None
            st.rerun()

        st.divider()

        # ── Index Status ──
        idx_status = st.session_state.get("loaded_index")
        if idx_status is not None:
            st.success("Index loaded", icon="✅")
        else:
            st.warning("No index loaded", icon="⚠️")

        col1, col2 = st.columns(2)
        col1.metric("Chunks", st.session_state.get("chunk_count", 0))
        col2.metric("Docs", st.session_state.get("document_count", 0))

        st.divider()
        st.caption(
            "Vector Search Engine · RAG Platform\n"
            "Built with FAISS HNSW + Streamlit"
        )

    return {
        "selected_model_slug": selected_slug,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "uploaded_files": uploaded_files,
        "rebuild_clicked": rebuild_clicked,
    }


# ---------------------------------------------------------------------------
# Phase 2: Connection test helper
# ---------------------------------------------------------------------------

def _run_connection_test(model_slug: str, api_key: str) -> None:
    """
    Execute a minimal live test against the selected model and display
    the result inline in the sidebar.

    Mock mode is validated instantly (no network call).  Real models require
    a valid API key and make a single micro-request returning 'ACK'.
    """
    if model_slug == "mock":
        st.sidebar.success("Mock environment fully operational.", icon="✅")
        return

    if not api_key:
        st.sidebar.error(
            "Missing API Key. Enter your OpenRouter key above before testing.",
            icon="❌",
        )
        return

    with st.sidebar:
        with st.spinner(f"Testing `{model_slug}`..."):
            try:
                from src.llm.openrouter import OpenRouterLLM
                from src.llm.config import GenerationConfig

                llm = OpenRouterLLM(default_model=model_slug, api_key=api_key)
                probe_cfg = GenerationConfig(temperature=0.0, max_tokens=8)
                t0 = time.perf_counter()
                resp = llm.generate("Reply with only the word 'ACK'.", config=probe_cfg)
                latency_ms = (time.perf_counter() - t0) * 1000

                st.success(
                    f"Connection OK — model responded in {latency_ms:.0f} ms",
                    icon="✅",
                )
                st.info(
                    f"**Model reply:** `{resp.text.strip()}`  \n"
                    f"**Prompt tokens:** {resp.prompt_tokens}  \n"
                    f"**Completion tokens:** {resp.completion_tokens}  \n"
                    f"**Total tokens:** {resp.total_tokens}",
                )

            except Exception as exc:
                st.error(
                    f"Connection failed: {type(exc).__name__}: {exc}",
                    icon="❌",
                )


# ---------------------------------------------------------------------------
# Chat history renderer
# ---------------------------------------------------------------------------


def render_chat(chat_history: list[dict]) -> None:
    """
    Render the full conversation history using native st.chat_message bubbles.

    Each assistant turn renders the answer, Phase 4 telemetry pillars
    (doc count, context expander, query echo, result + token stats),
    and a compact source reference line.

    Parameters
    ----------
    chat_history:
        List of message dicts with keys: role, content, sources, tokens,
        rag_response.
    """
    for turn in chat_history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        sources = turn.get("sources")
        tokens = turn.get("tokens")
        rag_response = turn.get("rag_response")

        with st.chat_message(role):
            if role == "assistant" and rag_response is not None:
                # Full Phase 4 telemetry pillars for assistant turns
                _render_rag_telemetry(rag_response)
            else:
                st.markdown(content)

            if role == "assistant" and sources:
                render_sources_inline(sources, compact=True)

            if tokens:
                st.caption(f"Tokens used: {tokens:,}")


# ---------------------------------------------------------------------------
# Phase 4: RAG Telemetry Pillars
# ---------------------------------------------------------------------------

def render_rag_answer(rag_response) -> None:
    """
    Render the four Phase 4 telemetry pillars for a fresh query response
    inside an active ``st.chat_message`` context.

    Pillars
    -------
    1. Number of Documents — metric showing active chunk count.
    2. Context Created     — expandable block showing the compiled prompt.
    3. Query              — the user's original question.
    4. Result             — markdown answer + token caption.
    """
    _render_rag_telemetry(rag_response)


def _render_rag_telemetry(rag_response) -> None:
    """Internal implementation of the 4-pillar telemetry layout."""
    llm_resp = rag_response.llm_response
    retrieved = rag_response.retrieved_documents

    # ── Pillar 1: Number of Documents ──
    chunk_count = st.session_state.get("chunk_count", 0)
    retrieved_count = len(retrieved)

    p1_col, p2_col, p3_col = st.columns(3)
    p1_col.metric(
        label="📦 Indexed Chunks",
        value=f"{chunk_count:,}",
        help="Total vectors stored in the active FaissHNSWIndex.",
    )
    p2_col.metric(
        label="📎 Retrieved",
        value=f"{retrieved_count}",
        help="Chunks retrieved as context for this query.",
    )
    p3_col.metric(
        label="🔤 Total Tokens",
        value=f"{llm_resp.total_tokens or 0:,}",
        help="Prompt + completion tokens consumed for this response.",
    )

    # ── Pillar 2: Context Created ──
    with st.expander("📝 View Constructed Context Block", expanded=False):
        st.code(rag_response.context, language="markdown")

    # ── Pillar 3: Query ──
    st.markdown(
        f"""
        <div style="
            background: rgba(99,102,241,0.08);
            border-left: 3px solid #6366f1;
            border-radius: 6px;
            padding: 0.5rem 0.9rem;
            margin: 0.5rem 0;
            font-size: 0.88rem;
            color: #a5b4fc;
        ">
            <strong>Query:</strong> {rag_response.query}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Pillar 4: Result ──
    st.markdown(rag_response.answer)

    # Token stat caption
    prompt_tok = llm_resp.prompt_tokens
    compl_tok = llm_resp.completion_tokens
    if prompt_tok is not None and compl_tok is not None:
        st.caption(
            f"📊 `{llm_resp.model_name}` · "
            f"prompt: **{prompt_tok:,}** tok · "
            f"completion: **{compl_tok:,}** tok · "
            f"total: **{(prompt_tok + compl_tok):,}** tok"
        )

    # Retrieved sources below the answer
    if retrieved:
        render_sources(rag_response)


# ---------------------------------------------------------------------------
# Metrics bar
# ---------------------------------------------------------------------------


def render_metrics(
    chunk_count: int,
    document_count: int,
    total_tokens: int = 0,
    queries_run: int = 0,
) -> None:
    """
    Render a real-time metrics bar at the top of the main area.

    Parameters
    ----------
    chunk_count:
        Total text chunks currently in the index.
    document_count:
        Total source documents ingested (before chunking).
    total_tokens:
        Cumulative tokens consumed across all LLM calls in this session.
    queries_run:
        Number of queries executed in this session.
    """
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        label="📦 Indexed Chunks",
        value=f"{chunk_count:,}",
        help="Total text chunks stored in the HNSW vector index.",
    )
    col2.metric(
        label="📁 Source Documents",
        value=f"{document_count:,}",
        help="Total source files loaded before chunking.",
    )
    col3.metric(
        label="🔤 Tokens Used",
        value=f"{total_tokens:,}",
        help="Cumulative prompt + completion tokens consumed this session.",
    )
    col4.metric(
        label="💬 Queries Run",
        value=f"{queries_run:,}",
        help="Total RAG pipeline queries executed this session.",
    )


# ---------------------------------------------------------------------------
# Source document expanders
# ---------------------------------------------------------------------------


def render_sources(rag_response) -> None:
    """
    Render expandable source document blocks for a ``RAGResponse``.

    Each block shows the file name, alignment score (cosine similarity),
    and a text snippet preview.

    Parameters
    ----------
    rag_response:
        A ``RAGResponse`` instance with a populated ``retrieved_documents`` list.
    """
    retrieved = getattr(rag_response, "retrieved_documents", [])
    if not retrieved:
        st.caption("No source documents were retrieved for this query.")
        return

    st.markdown("##### 📚 Retrieved Sources")
    for i, result in enumerate(retrieved, start=1):
        source_name = Path(result.source_file).name
        score_pct = max(0.0, result.score) * 100
        score_label = f"{result.score:.4f} ({score_pct:.1f}%)"

        with st.expander(
            f"[{i}] {source_name} — Score: {result.score:.4f}",
            expanded=i == 1,  # expand the top result by default
        ):
            col_a, col_b = st.columns([3, 1])
            col_a.markdown(f"**Source:** `{result.source_file}`")
            col_b.metric("Similarity", score_label)

            snippet = result.text[:500] + ("..." if len(result.text) > 500 else "")
            st.markdown(
                f"""
                <div style="
                    background: rgba(255,255,255,0.04);
                    border-left: 3px solid #6366f1;
                    border-radius: 6px;
                    padding: 0.75rem 1rem;
                    font-size: 0.88rem;
                    line-height: 1.6;
                    color: #cbd5e1;
                    font-family: 'Inter', sans-serif;
                ">
                    {snippet}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"Chunk ID: `{result.document_id}` · {len(result.text)} chars")


def render_sources_inline(sources: list, compact: bool = False) -> None:
    """
    Render a compact source reference list (used inside chat bubbles).

    Parameters
    ----------
    sources:
        List of ``RetrievalResult`` objects.
    compact:
        When True, renders a minimal reference list without full expanders.
    """
    if not sources:
        return

    if compact:
        refs = " · ".join(
            f"`{Path(r.source_file).name}` ({r.score:.3f})" for r in sources
        )
        st.caption(f"Sources: {refs}")
    else:
        render_sources(type("_FakeResponse", (), {"retrieved_documents": sources})())
