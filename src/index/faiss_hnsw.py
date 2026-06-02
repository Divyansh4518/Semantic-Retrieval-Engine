"""FAISS-backed HNSW approximate nearest-neighbor index.

Uses ``faiss.IndexHNSWFlat`` with inner-product metric on L2-normalized
vectors, so inner product == cosine similarity.
"""

from __future__ import annotations

import faiss
import numpy as np

from src.index.base import BaseIndex
from src.models import Document


class FaissHNSWIndex(BaseIndex):
    """Approximate nearest-neighbor index backed by ``faiss.IndexHNSWFlat``.

    Embeddings are L2-normalized to ``float32`` before insertion so that
    inner-product scoring is equivalent to cosine similarity.

    Parameters
    ----------
    M:
        Maximum number of bi-directional links per node in the HNSW graph.
        Higher values improve recall at the cost of memory and build time.
    ef_construction:
        Size of the dynamic candidate list during graph construction.
        Higher values yield a higher-quality graph but slower build.
    ef_search:
        Size of the dynamic candidate list at query time.
        Higher values improve recall at the cost of query latency.
    """

    def __init__(
        self,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        """Initialize the HNSW index configuration and backing storage."""
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self._documents: list[Document] = []
        self._index: faiss.IndexHNSWFlat | None = None
        self._embedding_dim: int | None = None

    # ------------------------------------------------------------------
    # Encapsulated ef_search setter
    # ------------------------------------------------------------------

    def set_ef_search(self, ef_search: int) -> None:
        """Update ``ef_search`` on both this object and the FAISS index.

        Safe to call before the index has been initialized (no-op on the
        FAISS side until after ``add_documents`` has been called).

        Parameters
        ----------
        ef_search:
            New candidate list size to use during search traversal.
        """
        self.ef_search = ef_search
        if self._index is not None:
            self._index.hnsw.efSearch = ef_search

    # ------------------------------------------------------------------
    # BaseIndex interface
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents and their embeddings to the HNSW index.

        Embeddings are cast to ``float32``, safely L2-normalized (zero
        vectors are left as zero rather than producing NaN), and stored in
        a contiguous C-layout array before being handed to FAISS.

        Parameters
        ----------
        documents:
            Batch of documents, each carrying a non-``None`` embedding.
        """
        if not documents:
            return

        embeddings: list[np.ndarray] = []
        for document in documents:
            embedding = document.embedding
            if embedding is None:
                raise ValueError("All documents must include an embedding")

            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            if self._embedding_dim is None:
                self._embedding_dim = int(vector.shape[0])
            elif vector.shape[0] != self._embedding_dim:
                raise ValueError(
                    "Document embedding dimensionality does not match the index"
                )
            embeddings.append(vector)

        # Stack → float32 → safe L2 normalize → contiguous
        batch = np.stack(embeddings).astype(np.float32, copy=False)
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        normalized = np.divide(
            batch,
            norms,
            out=np.zeros_like(batch, dtype=np.float32),
            where=norms > 0.0,
        )
        normalized = np.ascontiguousarray(normalized, dtype=np.float32)

        # Lazy initialization of the FAISS index
        if self._index is None:
            if self._embedding_dim is None:
                raise ValueError(
                    "Cannot initialize FAISS HNSW index without an embedding dimension"
                )
            self._index = faiss.IndexHNSWFlat(
                self._embedding_dim,
                self.M,
                faiss.METRIC_INNER_PRODUCT,
            )

        # CRITICAL: set efConstruction BEFORE adding vectors
        self._index.hnsw.efConstruction = self.ef_construction
        self._index.add(normalized)
        self._documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> list[tuple[Document, float]]:
        """Return the top-``k`` approximate nearest neighbors.

        Parameters
        ----------
        query_embedding:
            1-D array of floats representing the query vector.
        k:
            Number of nearest neighbors to retrieve.

        Returns
        -------
        list of ``(Document, float)`` tuples ordered by descending
        cosine similarity.
        """
        if self._index is None or not self._documents or k <= 0:
            return []

        if self._embedding_dim is None:
            raise ValueError(
                "FaissHNSWIndex has not been initialized with embeddings"
            )

        # Cast + reshape + normalize
        query_vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query_vector.shape[1] != self._embedding_dim:
            raise ValueError(
                "query_embedding dimensionality does not match the index"
            )

        query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        normalized_query = np.divide(
            query_vector,
            query_norm,
            out=np.zeros_like(query_vector, dtype=np.float32),
            where=query_norm > 0.0,
        )
        normalized_query = np.ascontiguousarray(normalized_query, dtype=np.float32)

        # CRITICAL: set efSearch BEFORE querying
        self.set_ef_search(self.ef_search)

        distances, indices = self._index.search(normalized_query, int(k))

        results: list[tuple[Document, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            results.append((self._documents[int(idx)], float(dist)))

        return results
