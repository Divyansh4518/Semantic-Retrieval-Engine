"""Base interfaces for vector search index implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from src.models import Document


class BaseIndex(ABC):
	"""Abstract interface for all index implementations."""

	@abstractmethod
	def add_documents(self, documents: List[Document]) -> None:
		"""Add a batch of documents to the index."""

	@abstractmethod
	def search(
		self,
		query_embedding: np.ndarray,
		k: int = 5,
	) -> List[Tuple[Document, float]]:
		"""Return the top-k matching documents with similarity scores."""
