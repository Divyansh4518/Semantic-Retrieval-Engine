"""
src/rag/__init__.py
--------------------
Public API for the Retrieval-Augmented Generation pipeline.

Consumers can do:

    from src.rag import RAGPipeline, RAGResponse, RetrievalResult, PromptBuilder
"""

from __future__ import annotations

from src.rag.pipeline import RAGPipeline
from src.rag.prompt_builder import PromptBuilder
from src.rag.response import RAGResponse, RetrievalResult

__all__: list[str] = [
    "RAGPipeline",
    "PromptBuilder",
    "RAGResponse",
    "RetrievalResult",
]
