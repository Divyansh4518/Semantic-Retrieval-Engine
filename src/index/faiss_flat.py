"""FAISS-backed exact vector index using inner product search."""

from __future__ import annotations

import faiss
import numpy as np

from src.index.base import BaseIndex
from src.models import Document


class FaissFlatIndex(BaseIndex):
	"""Exact search index backed by ``faiss.IndexFlatIP``.

	The index stores embeddings as L2-normalized ``float32`` vectors so inner
	product scoring is equivalent to cosine similarity.
	"""

	def __init__(self) -> None:
		"""Initialize the in-memory document store and FAISS index handle."""
		self._documents: list[Document] = []
		self._index = None
		self._embedding_dim: int | None = None

	def add_documents(self, documents: list[Document]) -> None:
		"""Add documents and their embeddings to the FAISS index."""
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
				raise ValueError("Document embedding dimensionality does not match the index")

			embeddings.append(vector)

		batch_embeddings = np.stack(embeddings).astype(np.float32, copy=False)
		norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
		normalized_embeddings = np.divide(
			batch_embeddings,
			norms,
			out=np.zeros_like(batch_embeddings, dtype=np.float32),
			where=norms > 0.0,
		)
		normalized_embeddings = np.ascontiguousarray(normalized_embeddings, dtype=np.float32)

		if self._index is None:
			if self._embedding_dim is None:
				raise ValueError("Cannot initialize FAISS index without an embedding dimension")
			self._index = faiss.IndexFlatIP(self._embedding_dim)

		self._index.add(normalized_embeddings)
		self._documents.extend(documents)

	def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[Document, float]]:
		"""Return the top-``k`` documents ranked by cosine similarity."""
		if self._index is None or not self._documents or k <= 0:
			return []

		if self._embedding_dim is None:
			raise ValueError("FaissFlatIndex has not been initialized with embeddings")

		query_vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
		if query_vector.shape[1] != self._embedding_dim:
			raise ValueError("query_embedding dimensionality does not match the index")

		query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
		normalized_query = np.divide(
			query_vector,
			query_norm,
			out=np.zeros_like(query_vector, dtype=np.float32),
			where=query_norm > 0.0,
		)
		normalized_query = np.ascontiguousarray(normalized_query, dtype=np.float32)

		distances, indices = self._index.search(normalized_query, int(k))
		results: list[tuple[Document, float]] = []
		for index, distance in zip(indices[0], distances[0]):
			if index == -1:
				continue
			results.append((self._documents[int(index)], float(distance)))

		return results