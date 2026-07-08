"""
app/session_manager.py
-----------------------
Streamlit session state lifecycle management.

Handles initialization and validation of all persistent state keys across
Streamlit reruns.  Each key is checked and initialized on first load, then
preserved as-is across subsequent reruns.

State Keys
----------
chat_history:
    list[dict]  — Ordered conversation turns: {role: "user"|"assistant", content: str,
                   sources: list[RetrievalResult]|None, tokens: int|None}.
active_model:
    str — The model slug currently selected (e.g. "google/gemini-2.5-flash").
loaded_index:
    FaissHNSWIndex | None — The currently active vector index instance.
document_count:
    int — Total number of source documents (pre-chunking) ingested.
chunk_count:
    int — Total number of text chunks indexed.
pipeline:
    RAGPipeline | None — The active pipeline wired to the current index and LLM.
upload_key:
    int — Incremented to reset the file uploader widget between sessions.
index_dir:
    str — Directory where the index is persisted on disk.
openrouter_api_key:
    str — API key entered by the user in the sidebar. Defaults to empty string.
available_models:
    dict[str, str] — Mapping of {friendly_name: model_slug} fetched from
    OpenRouter /models endpoint. Falls back to a static minimal dict offline.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # Load .env file before any Streamlit state is touched

import streamlit as st

# ---------------------------------------------------------------------------
# Static fallback model list — used when offline or before fetch completes
# ---------------------------------------------------------------------------

FALLBACK_MODELS: dict[str, str] = {
    "Gemini 2.5 Flash": "google/gemini-2.5-flash",
    "Gemini 2.5 Pro": "google/gemini-2.5-pro",
    "Claude 3.5 Sonnet": "anthropic/claude-sonnet-4",
    "Claude 3.1 Opus": "anthropic/claude-opus-4.1",
    "DeepSeek V3": "deepseek/deepseek-chat",
    "Mock Simulator Mode": "mock",
}

# Default values for each session key
_DEFAULTS: dict = {
    "chat_history": [],
    "active_model": "mock",
    "loaded_index": None,
    "document_count": 0,
    "chunk_count": 0,
    "pipeline": None,
    "upload_key": 0,
    "index_dir": "data/index/",
    "processing": False,
    "last_error": None,
    # Phase 1 additions — default from .env, overridable via sidebar
    "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
    "available_models": FALLBACK_MODELS,
    "models_fetched": False,   # True once a live fetch has been attempted
    "last_rag_response": None, # Store most recent RAGResponse for telemetry display
}


def initialize_session() -> None:
    """
    Ensure all required session state keys exist with their default values.

    Safe to call on every Streamlit rerun — existing values are never
    overwritten, only missing keys are populated.
    """
    for key, default in _DEFAULTS.items():
        st.session_state.setdefault(key, default)

    # Always seed the API key from env if session doesn't already have one
    st.session_state.setdefault(
        "openrouter_api_key", os.getenv("OPENROUTER_API_KEY", "")
    )


def clear_chat_history() -> None:
    """Reset the conversation history to an empty list."""
    st.session_state.chat_history = []


def add_message(
    role: str,
    content: str,
    sources=None,
    tokens: int | None = None,
    rag_response=None,
) -> None:
    """
    Append a new message to the chat history.

    Parameters
    ----------
    role:
        Either ``"user"`` or ``"assistant"``.
    content:
        The message text.
    sources:
        Optional list of ``RetrievalResult`` objects (assistant turns only).
    tokens:
        Optional total token count for this turn.
    rag_response:
        Optional full ``RAGResponse`` object for telemetry replay in history.
    """
    st.session_state.chat_history.append(
        {
            "role": role,
            "content": content,
            "sources": sources,
            "tokens": tokens,
            "rag_response": rag_response,
        }
    )


def get_index_status() -> dict:
    """
    Return a snapshot of the current index state for display.

    Returns
    -------
    dict with keys: is_loaded, document_count, chunk_count, active_model.
    """
    return {
        "is_loaded": st.session_state.loaded_index is not None,
        "document_count": st.session_state.document_count,
        "chunk_count": st.session_state.chunk_count,
        "active_model": st.session_state.active_model,
    }
