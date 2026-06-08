"""
src/ingestion/embeddings.py
----------------------------
Embedding generation service clients for the ingestion pipeline.

This module provides three concrete embedding backends that share a common
interface:

``EmbeddingService``
    Production client that calls the OpenRouter / OpenAI embeddings endpoint.
    Converts ``TextChunk`` objects into ``Document`` objects with dense 1536-d
    (or model-native-d) numpy embedding arrays ready for ``BaseIndex.add_documents``.

``MockEmbeddingService``
    Offline, deterministic backend for unit tests and CI.  Generates
    reproducible unit-normalised random vectors seeded from ``hash(chunk_id)``
    so the same input always produces the same vector.

``MiniLMEmbeddingService``
    Local, real semantic embedding backend powered by
    ``sentence-transformers/all-MiniLM-L6-v2``.  Produces 384-dimensional
    L2-normalised ``float32`` vectors with genuine semantic meaning.

``EmbeddingService``
    Production client that calls the OpenRouter / OpenAI embeddings endpoint.
    Converts ``TextChunk`` objects into ``Document`` objects with dense 1536-d
    (or model-native-d) numpy embedding arrays ready for ``BaseIndex.add_documents``.

``MockEmbeddingService``
    Offline, deterministic backend for unit tests and CI.  Generates
    reproducible unit-normalised random vectors seeded from ``hash(chunk_id)``
    so the same input always produces the same vector.

Both services accept a ``List[TextChunk]`` and return a ``List[Document]``,
making them drop-in substitutable.

Usage — production
------------------
    import os
    from src.ingestion import EmbeddingService, RecursiveTokenChunker, RepositoryLoader

    payloads = RepositoryLoader("docs/").load()
    chunks   = RecursiveTokenChunker().chunk_all(payloads)

    svc  = EmbeddingService()            # reads OPENROUTER_API_KEY from env
    docs = svc.embed_chunks(chunks)      # List[Document] ready for indexing

Usage — offline / test
----------------------
    from src.ingestion import MockEmbeddingService

    svc  = MockEmbeddingService(dim=128)
    docs = svc.embed_chunks(chunks)
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import List

import numpy as np

from src.ingestion.models import TextChunk
from src.models import Document

logger = logging.getLogger(__name__)

# Default model served by OpenRouter's embeddings endpoint.
_DEFAULT_EMBED_MODEL: str = "openai/text-embedding-3-small"
_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
_REQUEST_TIMEOUT: float = 60.0

# Maximum number of texts per API batch request.
# OpenAI / OpenRouter limit is 2 048 inputs per call; we stay conservative.
_BATCH_SIZE: int = 128


class EmbeddingService:
    """
    Production embedding client backed by the OpenRouter / OpenAI API.

    Parameters
    ----------
    model:
        The embedding model slug to request.
        Defaults to ``"openai/text-embedding-3-small"``.
    api_key:
        Your API key.  When ``None``, the constructor resolves the key from
        environment variables in priority order:

        1. ``OPENROUTER_API_KEY``
        2. ``OPENAI_API_KEY``

        A ``ValueError`` is raised at construction time if no key is found.
    batch_size:
        Number of texts sent per API call.  Defaults to 128.

    Raises
    ------
    ValueError
        Raised immediately when no API key can be resolved.
    ImportError
        Raised when the ``openai`` package is not installed.
    """

    def __init__(
        self,
        model: str = _DEFAULT_EMBED_MODEL,
        api_key: str | None = None,
        batch_size: int = _BATCH_SIZE,
    ) -> None:
        resolved_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not resolved_key:
            raise ValueError(
                "No API key found for EmbeddingService.  "
                "Pass 'api_key' to the constructor, or set the "
                "'OPENROUTER_API_KEY' or 'OPENAI_API_KEY' environment variable."
            )

        self._model = model
        self._batch_size = batch_size
        self._client = self._build_client(resolved_key)
        logger.debug(
            "EmbeddingService initialised: model=%r, batch_size=%d",
            self._model,
            self._batch_size,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: List[TextChunk]) -> List[Document]:
        """
        Embed a list of ``TextChunk`` objects and return ``Document`` objects.

        The method batches API calls to avoid exceeding provider rate limits
        and token-per-request ceilings.  On failure the offending batch is
        skipped and a warning is logged; other batches continue normally.

        Parameters
        ----------
        chunks:
            The chunks to embed.  Empty lists are handled gracefully.

        Returns
        -------
        List[Document]
            One ``Document`` per successfully embedded chunk, with the
            ``embedding`` field populated as a ``numpy.ndarray`` of
            ``float64``.  The ``Document.id`` equals ``chunk.chunk_id``,
            ``Document.text`` equals ``chunk.text``, and
            ``Document.metadata`` is a shallow copy of ``chunk.metadata``
            augmented with ``"source_file"`` and ``"embed_model"`` keys.
        """
        if not chunks:
            return []

        documents: List[Document] = []

        # Process in batches
        for batch_start in range(0, len(chunks), self._batch_size):
            batch = chunks[batch_start : batch_start + self._batch_size]
            texts = [c.text for c in batch]

            try:
                vectors = self._call_api(texts)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Embedding batch [%d:%d] failed: %s — skipping.",
                    batch_start,
                    batch_start + len(batch),
                    exc,
                )
                continue

            for chunk, vector in zip(batch, vectors):
                doc = self._build_document(chunk, vector)
                documents.append(doc)

        logger.info(
            "EmbeddingService: embedded %d/%d chunks.", len(documents), len(chunks)
        )
        return documents

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_api(self, texts: list[str]) -> list[np.ndarray]:
        """
        Send a batch of texts to the embeddings endpoint and return vectors.

        Raises
        ------
        openai.APIError
            Any provider-level error (timeout, status, connection).  Let it
            propagate so the caller can decide whether to skip or re-raise.
        """
        import openai  # local import — required only when this class is used

        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
            timeout=_REQUEST_TIMEOUT,
        )
        # The response.data list is ordered to match the input texts.
        return [
            np.asarray(item.embedding, dtype=np.float64)
            for item in response.data
        ]

    def _build_document(self, chunk: TextChunk, vector: np.ndarray) -> Document:
        """Assemble a ``Document`` from a ``TextChunk`` and its embedding vector."""
        metadata = dict(chunk.metadata)
        metadata["source_file"] = chunk.source_file
        metadata["embed_model"] = self._model

        return Document(
            id=chunk.chunk_id,
            text=chunk.text,
            metadata=metadata,
            embedding=vector,
        )

    @staticmethod
    def _build_client(api_key: str):
        """
        Construct an ``openai.OpenAI`` client pointed at the OpenRouter gateway.

        Raises ``ImportError`` with an actionable message if ``openai`` is
        not installed.
        """
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for EmbeddingService.  "
                "Install it with:  uv add openai"
            ) from exc

        return openai.OpenAI(
            api_key=api_key,
            base_url=_OPENROUTER_BASE_URL,
        )


class MockEmbeddingService:
    """
    Offline, deterministic embedding backend for unit tests and CI.

    Vectors are generated by seeding ``numpy``'s random number generator
    from a stable integer derived from ``hash(chunk.chunk_id)``.  The same
    input chunk always produces the same unit-normalised vector, regardless
    of the order in which chunks are processed.

    Parameters
    ----------
    dim:
        Dimensionality of the generated embedding vectors.  Defaults to 128
        to match our FAISS index configuration.
    model_name:
        Reported in ``Document.metadata["embed_model"]`` for traceability.
    """

    def __init__(
        self,
        dim: int = 128,
        model_name: str = "mock/unit-vector-v1",
    ) -> None:
        self._dim = dim
        self._model_name = model_name

    # ------------------------------------------------------------------
    # Public interface (mirrors EmbeddingService)
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: List[TextChunk]) -> List[Document]:
        """
        Generate deterministic unit-normalised vectors for each chunk.

        Parameters
        ----------
        chunks:
            Input chunks.  Empty lists are handled gracefully.

        Returns
        -------
        List[Document]
            One ``Document`` per chunk.  ``Document.embedding`` is a
            ``float64`` numpy array of shape ``(dim,)`` with unit L2 norm.
        """
        documents: List[Document] = []
        for chunk in chunks:
            vector = self._deterministic_unit_vector(chunk.chunk_id, self._dim)
            metadata = dict(chunk.metadata)
            metadata["source_file"] = chunk.source_file
            metadata["embed_model"] = self._model_name

            documents.append(
                Document(
                    id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=metadata,
                    embedding=vector,
                )
            )
        logger.debug(
            "MockEmbeddingService: generated %d vector(s) (dim=%d).",
            len(documents),
            self._dim,
        )
        return documents

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deterministic_unit_vector(chunk_id: str, dim: int) -> np.ndarray:
        """
        Generate a reproducible unit-normalised random vector.

        The seed is derived from a stable SHA-256 hash of ``chunk_id`` so
        vector identity is independent of Python's built-in ``hash()``
        (which is randomised across interpreter sessions by PYTHONHASHSEED).

        Parameters
        ----------
        chunk_id:
            The chunk identifier to seed from.
        dim:
            Output dimensionality.

        Returns
        -------
        numpy.ndarray
            Shape ``(dim,)``, dtype ``float64``, L2-normalised.
        """
        # Use first 4 bytes of SHA-256 digest as a 32-bit seed
        digest = hashlib.sha256(chunk_id.encode()).digest()
        seed = int.from_bytes(digest[:4], byteorder="little")

        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim).astype(np.float64)

        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            # Astronomically unlikely, but guard against the zero vector
            vec = np.ones(dim, dtype=np.float64)
            norm = np.sqrt(dim)

        return vec / norm


class MiniLMEmbeddingService:
    """
    Local semantic embedding backend using ``sentence-transformers``.

    Loads ``all-MiniLM-L6-v2`` (384-dimensional) from the HuggingFace Hub
    on first instantiation and caches it in memory for the lifetime of the
    process.  All vectors are L2-normalised to ``float32`` so that inner-
    product search in ``FaissHNSWIndex`` is equivalent to cosine similarity.

    The class shares the same public interface as ``EmbeddingService`` and
    ``MockEmbeddingService``, making it a drop-in replacement in both the
    ingestion pipeline and the RAGPipeline query path.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Defaults to ``"all-MiniLM-L6-v2"``.
    """

    _MODEL_NAME: str = "all-MiniLM-L6-v2"
    _DIM: int = 384

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer  # local import

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        logger.info("MiniLMEmbeddingService: loaded model %r (dim=%d).", model_name, self._DIM)

    # ------------------------------------------------------------------
    # Public interface (mirrors EmbeddingService / MockEmbeddingService)
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string and return a 384-d ``float32`` array.

        The vector is L2-normalised so it is ready for inner-product
        cosine-similarity search against the FAISS index.

        Parameters
        ----------
        query:
            The natural-language query to encode.

        Returns
        -------
        numpy.ndarray
            Shape ``(384,)``, dtype ``float32``, L2-normalised.
        """
        vec = self._model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        return vec.astype(np.float32).reshape(-1)

    def embed_chunks(self, chunks: List[TextChunk]) -> List[Document]:
        """
        Embed a list of ``TextChunk`` objects and return ``Document`` objects.

        All texts are encoded in a single batched forward pass for efficiency.
        Each resulting vector is L2-normalised to ``float32``.

        Parameters
        ----------
        chunks:
            Input chunks.  Empty lists are handled gracefully.

        Returns
        -------
        List[Document]
            One ``Document`` per chunk with a 384-d ``float32`` embedding.
        """
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        documents: List[Document] = []
        for chunk, vec in zip(chunks, vectors):
            metadata = dict(chunk.metadata)
            metadata["source_file"] = chunk.source_file
            metadata["embed_model"] = self._model_name

            documents.append(
                Document(
                    id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=metadata,
                    embedding=vec,
                )
            )

        logger.info(
            "MiniLMEmbeddingService: embedded %d chunk(s) (dim=%d).",
            len(documents),
            self._DIM,
        )
        return documents
