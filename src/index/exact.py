"""Exact vector search index implementation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.index.base import BaseIndex
from src.models import Document


class ExactIndex(BaseIndex):
	"""Brute-force cosine similarity index over a dense embedding matrix."""

	def __init__(self) -> None:
		self._documents: list[Document] = []
		self._embeddings: np.ndarray = np.empty((0, 0), dtype=float)
		self._embedding_dim: int | None = None

	def add_documents(self, documents: List[Document]) -> None:
		if not documents:
			return

		batch_documents: list[Document] = []
		batch_embeddings: list[np.ndarray] = []

		for document in documents:
			if document.embedding is None:
				raise ValueError("All documents must have an embedding before indexing.")

			embedding = np.asarray(document.embedding, dtype=float).ravel()
			if embedding.ndim != 1:
				raise ValueError("Document embeddings must be one-dimensional.")

			if self._embedding_dim is None:
				self._embedding_dim = embedding.shape[0]
			elif embedding.shape[0] != self._embedding_dim:
				raise ValueError("All embeddings must share the same dimensionality.")

			batch_documents.append(document)
			batch_embeddings.append(embedding)

		batch_matrix = np.vstack(batch_embeddings)
		if self._embeddings.size == 0:
			self._embeddings = batch_matrix
		else:
			self._embeddings = np.vstack((self._embeddings, batch_matrix))

		self._documents.extend(batch_documents)

	def search(
		self,
		query_embedding: np.ndarray,
		k: int = 5,
	) -> List[Tuple[Document, float]]:
		if not self._documents:
			return []

		query = np.asarray(query_embedding, dtype=float).ravel()
		if self._embedding_dim is not None and query.shape[0] != self._embedding_dim:
			raise ValueError("Query embedding dimensionality must match the indexed embeddings.")

		query_norm = float(np.linalg.norm(query))
		document_norms = np.linalg.norm(self._embeddings, axis=1)

		if query_norm == 0.0:
			similarities = np.zeros(len(self._documents), dtype=float)
		else:
			denominator = document_norms * query_norm
			numerator = self._embeddings @ query
			similarities = np.divide(
				numerator,
				denominator,
				out=np.zeros_like(numerator, dtype=float),
				where=denominator != 0,
			)

		limit = min(max(k, 0), len(self._documents))
		if limit == 0:
			return []

		if limit == len(self._documents):
			indices = np.argsort(similarities)[::-1]
		else:
			indices = np.argpartition(similarities, -limit)[-limit:]
			indices = indices[np.argsort(similarities[indices])[::-1]]

		return [(self._documents[index], float(similarities[index])) for index in indices[:limit]]
