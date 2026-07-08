"""
tests/smoke/phase1_smoke_test.py
---------------------------------
Phase 1 Smoke Test: RAG Core & Telemetry Data Structures.

Invariants verified:
  1. RAGResponse.retrieved_documents length == 1
  2. RAGResponse.context contains the document text
  3. RAGResponse.answer is a non-empty string
  4. RetrievalResult fields are correctly populated
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from src.index.faiss_hnsw import FaissHNSWIndex
from src.llm.mock import MockLLM
from src.models import Document
from src.rag.pipeline import RAGPipeline
from src.rag.response import RAGResponse, RetrievalResult


# ---------------------------------------------------------------------------
# Minimal embedding service adapter that satisfies both embed_chunks and
# embed_query contracts for the pipeline test.
# ---------------------------------------------------------------------------

class _FixedVectorEmbeddingService:
    """
    A test-only embedding service that returns a fixed unit vector for any
    input.  This guarantees the query and the document share identical
    directions, so cosine similarity == 1.0 and the document is always
    retrieved as the top result.
    """

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim
        # Fixed unit vector: [1/sqrt(dim), 1/sqrt(dim), ...]
        self._vector = np.ones(dim, dtype=np.float32) / np.sqrt(dim)

    def embed_query(self, query: str) -> np.ndarray:
        return self._vector.copy()

    def embed_chunks(self, chunks):
        """Not used in this smoke test."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _assert(condition: bool, message: str) -> None:
    if not condition:
        print(f"  [FAIL] {message}")
        sys.exit(1)
    print(f"  [PASS] {message}")


# ---------------------------------------------------------------------------
# Phase 1 Smoke Test
# ---------------------------------------------------------------------------

def run_phase1_smoke_test() -> None:
    print("\n" + "=" * 60)
    print("PHASE 1 SMOKE TEST: RAG Core & Telemetry Data Structures")
    print("=" * 60)

    dim = 16
    embedding_service = _FixedVectorEmbeddingService(dim=dim)
    llm = MockLLM()
    index = FaissHNSWIndex(M=16, ef_construction=40, ef_search=16)

    # Create a single document with a known embedding identical to the
    # query vector so it will always be the top result.
    vector = embedding_service.embed_query("test")
    doc = Document(
        id="smoke-doc-001",
        text="The quick brown fox jumps over the lazy dog.",
        metadata={"source_file": "smoke_test.txt"},
        embedding=vector,
    )
    index.add_documents([doc])

    # Construct and run the pipeline
    pipeline = RAGPipeline(
        index=index,
        embedding_service=embedding_service,
        llm=llm,
        top_k=1,
    )

    response = pipeline.query("What does the fox do?")

    print("\n--- Invariant Checks ---")

    # Invariant 1: Correct return type
    _assert(
        isinstance(response, RAGResponse),
        f"pipeline.query() returns RAGResponse (got {type(response).__name__})",
    )

    # Invariant 2: retrieved_documents length == 1
    _assert(
        len(response.retrieved_documents) == 1,
        f"retrieved_documents length == 1 (got {len(response.retrieved_documents)})",
    )

    # Invariant 3: context contains the document text
    _assert(
        doc.text in response.context,
        f"context contains document text: '{doc.text[:40]}...'",
    )

    # Invariant 4: answer is non-empty
    _assert(
        len(response.answer) > 0,
        f"answer is non-empty (length={len(response.answer)})",
    )

    # Invariant 5: RetrievalResult fields are correctly set
    result: RetrievalResult = response.retrieved_documents[0]
    _assert(
        result.document_id == "smoke-doc-001",
        f"RetrievalResult.document_id == 'smoke-doc-001' (got '{result.document_id}')",
    )
    _assert(
        result.text == doc.text,
        f"RetrievalResult.text matches document text",
    )
    _assert(
        result.source_file == "smoke_test.txt",
        f"RetrievalResult.source_file == 'smoke_test.txt' (got '{result.source_file}')",
    )
    _assert(
        isinstance(result.score, float) and result.score > 0.0,
        f"RetrievalResult.score is positive float (got {result.score:.4f})",
    )

    # Invariant 6: LLMResponse is wired correctly
    _assert(
        response.llm_response.model_name == "mock/stable-simulator-v1",
        f"llm_response.model_name == 'mock/stable-simulator-v1'",
    )

    # Invariant 7: query is preserved
    _assert(
        response.query == "What does the fox do?",
        f"response.query preserved correctly",
    )

    print("\n--- Telemetry Preview ---")
    print(f"  Query: {response.query}")
    print(f"  Documents retrieved: {len(response.retrieved_documents)}")
    print(f"  Top score: {response.retrieved_documents[0].score:.4f}")
    print(f"  Answer preview: {response.answer[:80]}...")
    print(f"  Prompt tokens: {response.llm_response.prompt_tokens}")
    print(f"  Completion tokens: {response.llm_response.completion_tokens}")

    print("\n[ALL PASSED] PHASE 1 SMOKE TEST PASSED -- All invariants satisfied.\n")


if __name__ == "__main__":
    run_phase1_smoke_test()
