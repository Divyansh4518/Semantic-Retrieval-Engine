"""
src/rag/response.py
-------------------
Telemetry data structures for the RAG pipeline.

``RetrievalResult`` captures a single retrieved context fragment with its
provenance and alignment score.  ``RAGResponse`` is the canonical top-level
carrier returned by ``RAGPipeline.query``, bundling the generated answer,
the full prompt context, and per-document retrieval telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.llm.response import LLMResponse


@dataclass
class RetrievalResult:
    """
    A single document retrieved from the vector index.

    Attributes
    ----------
    document_id:
        Stable unique identifier for the source chunk (``Document.id``).
    score:
        Cosine similarity score in [-1, 1].  Higher values indicate a
        closer semantic match to the query.
    source_file:
        Original file path from which this chunk was extracted.
    text:
        The raw chunk text that was retrieved and included in the context.
    """

    document_id: str
    score: float
    source_file: str
    text: str


@dataclass
class RAGResponse:
    """
    Complete response carrier returned by ``RAGPipeline.query``.

    Attributes
    ----------
    answer:
        The generated text from the LLM, grounded in the retrieved context.
    query:
        The original user query string that triggered this pipeline run.
    context:
        The full compiled context string that was submitted to the LLM.
    retrieved_documents:
        Ordered list of ``RetrievalResult`` objects, one per retrieved chunk.
    llm_response:
        The raw ``LLMResponse`` from the backend, including token telemetry.
    """

    answer: str
    query: str
    context: str
    retrieved_documents: list[RetrievalResult] = field(default_factory=list)
    llm_response: LLMResponse = field(
        default_factory=lambda: LLMResponse(text="", model_name="unknown")
    )
