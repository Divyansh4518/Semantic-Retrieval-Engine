"""Embedding helpers for vectorizing documents and queries."""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.models import Document


class EmbeddingPipeline:
	"""Thin wrapper around SentenceTransformer for document embeddings."""

	def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
		self.model = SentenceTransformer(model_name)

	def embed_documents(self, documents: List[Document]) -> List[Document]:
		"""Embed documents in place and return the same list."""
		if not documents:
			return documents

		embeddings = self.model.encode(
			[document.text for document in documents],
			convert_to_numpy=True,
		)

		for document, embedding in zip(documents, embeddings, strict=True):
			document.embedding = np.asarray(embedding, dtype=float)

		return documents

	def embed_query(self, query: str) -> np.ndarray:
		"""Return the embedding for a single query string."""
		embedding = self.model.encode(query, convert_to_numpy=True)
		return np.asarray(embedding, dtype=float)
