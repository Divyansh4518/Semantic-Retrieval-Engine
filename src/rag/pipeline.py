"""
src/rag/pipeline.py
--------------------
Orchestrator for the full Retrieval-Augmented Generation workflow.

``RAGPipeline`` wires together three independently swappable components:

1. **Embedding service** — vectorizes the user query into a dense array.
2. **Index** — searches for the k-nearest neighbors in the vector space.
3. **LLM backend** — generates a grounded answer from the compiled context.

The pipeline is intentionally stateless with respect to conversation history;
each call to ``query`` is an isolated, atomic operation.
"""

from __future__ import annotations

import logging

from src.index.base import BaseIndex
from src.llm.base import BaseLLM
from src.llm.config import GenerationConfig
from src.rag.prompt_builder import PromptBuilder
from src.rag.response import RAGResponse, RetrievalResult

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation orchestrator.

    Parameters
    ----------
    index:
        Any ``BaseIndex`` implementation (e.g. ``FaissHNSWIndex``).
    embedding_service:
        Any service exposing an ``embed_query(query: str) -> np.ndarray``
        method.  Both ``EmbeddingService`` and ``MockEmbeddingService`` satisfy
        this if they also expose ``embed_query``.  The pipeline also accepts
        any object with a callable ``embed_query`` attribute, including a raw
        lambda.
    llm:
        Any ``BaseLLM`` implementation (e.g. ``MockLLM``, ``OpenRouterLLM``).
    top_k:
        Number of nearest-neighbor documents to retrieve per query.
        Defaults to 5.
    """

    def __init__(
        self,
        index: BaseIndex,
        embedding_service,
        llm: BaseLLM,
        top_k: int = 5,
    ) -> None:
        self._index = index
        self._embedding_service = embedding_service
        self._llm = llm
        self._top_k = top_k
        self._prompt_builder = PromptBuilder()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def query(
        self,
        user_query: str,
        config: GenerationConfig | None = None,
    ) -> RAGResponse:
        """
        Execute the full RAG pipeline for a single user query.

        Workflow
        --------
        1. Vectorize the query via the embedding service.
        2. Search the index for the ``top_k`` most similar documents.
        3. Map ``(Document, float)`` pairs into ``RetrievalResult`` objects.
        4. Compile the structured prompt via ``PromptBuilder``.
        5. Submit the prompt to the LLM backend.
        6. Bundle everything into a ``RAGResponse`` and return.

        Parameters
        ----------
        user_query:
            The raw natural language question from the user.
        config:
            Optional ``GenerationConfig`` forwarded to the LLM.  When ``None``,
            each LLM backend applies its own defaults.

        Returns
        -------
        RAGResponse
            Fully populated response including generated answer, compiled
            context, and per-document retrieval telemetry.
        """
        logger.info("RAGPipeline.query: '%s' (top_k=%d)", user_query, self._top_k)

        # Step 1: Vectorize the query
        query_embedding = self._embedding_service.embed_query(user_query)
        print(type(self._embedding_service).__name__)
        print(query_embedding.shape)

        # Step 2: Search the index
        search_results = self._index.search(query_embedding, k=self._top_k)
        # search_results: list[tuple[Document, float]]

        if not search_results:
            logger.warning(
                "RAGPipeline: index returned no results for query '%s'. "
                "The index may be empty.",
                user_query,
            )

        # Step 3: Map into RetrievalResult objects
        documents = [doc for doc, _ in search_results]
        scores = [score for _, score in search_results]

        retrieval_results: list[RetrievalResult] = []
        for doc, score in search_results:
            retrieval_results.append(
                RetrievalResult(
                    document_id=doc.id,
                    score=score,
                    source_file=doc.metadata.get("source_file", doc.id),
                    text=doc.text,
                )
            )

        # Step 4: Compile the structured prompt
        context = PromptBuilder.build_prompt(
            query=user_query,
            retrieved_documents=documents,
            scores=scores,
        )

        # Step 5: Submit to the LLM backend
        llm_response = self._llm.generate(prompt=context, config=config)

        # Step 6: Bundle into RAGResponse
        rag_response = RAGResponse(
            answer=llm_response.text,
            query=user_query,
            context=context,
            retrieved_documents=retrieval_results,
            llm_response=llm_response,
        )

        logger.info(
            "RAGPipeline: generated answer (%d chars) from %d retrieved documents.",
            len(rag_response.answer),
            len(retrieval_results),
        )

        return rag_response
