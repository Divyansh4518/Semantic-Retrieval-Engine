"""Core data models for the vector search engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Document:
	"""A text document with optional vector embedding and metadata."""

	id: str
	text: str
	metadata: dict[str, Any] = field(default_factory=dict)
	embedding: np.ndarray | None = None

	def __post_init__(self) -> None:
		self.metadata = dict(self.metadata)

		if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
			self.embedding = np.asarray(self.embedding, dtype=float)
