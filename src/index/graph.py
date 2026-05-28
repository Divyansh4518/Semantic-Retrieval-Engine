"""Flat Navigable Small World graph index skeleton."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.index.base import BaseIndex
from src.models import Document


class GraphIndex(BaseIndex):
	"""Flat NSW graph index placeholder.

	The implementation currently stores documents, embeddings, and adjacency
	state, but leaves insertion and graph search behavior for later phases.
	"""

	__slots__ = (
		"M",
		"ef_construction",
		"_documents",
		"_embeddings",
		"_graph",
		"_embedding_dim",
	)

	def __init__(self, M: int = 5, ef_construction: int = 32) -> None:
		"""Initialize the graph index configuration and backing storage."""
		self.M = M
		self.ef_construction = ef_construction
		self._documents: list[Document] = []
		self._embeddings: np.ndarray = np.empty((0, 0), dtype=float)
		self._graph: dict[int, list[int]] = {}
		self._embedding_dim: int | None = None

	def add_documents(self, documents: List[Document]) -> None:
		"""Add a batch of documents to the index.

		Insertion is deferred to a later phase.
		"""
		if not documents:
			return

		new_embeddings: list[np.ndarray] = []
		for document in documents:
			embedding: np.ndarray | None = document.embedding
			if embedding is None:
				raise ValueError("All documents must include an embedding")

			embedding_array: np.ndarray = np.asarray(embedding, dtype=float).reshape(-1)
			if self._embedding_dim is None:
				self._embedding_dim = int(embedding_array.shape[0])
			elif embedding_array.shape[0] != self._embedding_dim:
				raise ValueError("Document embedding dimensionality does not match the index")

			new_embeddings.append(embedding_array)

		batch_embeddings: np.ndarray = np.vstack(new_embeddings)
		start_idx: int = len(self._documents)

		self._documents.extend(documents)
		if self._embeddings.size == 0:
			self._embeddings = batch_embeddings
		else:
			self._embeddings = np.vstack((self._embeddings, batch_embeddings))

		for offset, document in enumerate(documents):
			new_idx: int = start_idx + offset
			if new_idx == 0:
				self._graph[new_idx] = []
				continue

			embedding_array = np.asarray(document.embedding, dtype=float).reshape(-1)
			neighbor_indices: list[int] = self._exact_search(embedding_array, k=self.M)
			filtered_neighbors: list[int] = [
				neighbor_idx
				for neighbor_idx in neighbor_indices
				if neighbor_idx != new_idx
			]

			self._graph.setdefault(new_idx, [])
			self._graph[new_idx].extend(filtered_neighbors)

			for neighbor_idx in filtered_neighbors:
				self._graph.setdefault(neighbor_idx, [])
				self._graph[neighbor_idx].append(new_idx)

	def search(
		self,
		query_embedding: np.ndarray,
		k: int = 5,
	) -> List[Tuple[Document, float]]:
		"""Return the top-k matching documents with similarity scores.

		Performs a greedy best-first traversal over the graph structure.
		"""
		if not self._documents:
			return []

		query_vector: np.ndarray = np.asarray(query_embedding, dtype=float).reshape(-1)
		if self._embedding_dim is None:
			raise ValueError("GraphIndex has not been initialized with embeddings")
		if query_vector.shape[0] != self._embedding_dim:
			raise ValueError("query_embedding dimensionality does not match the index")

		query_norm: float = float(np.linalg.norm(query_vector))
		if query_norm == 0.0:
			return []

		embedding_norms: np.ndarray = np.linalg.norm(self._embeddings, axis=1)
		valid_mask: np.ndarray = embedding_norms > 0.0

		visited: set[int] = set()
		candidates: list[Tuple[float, int]] = []
		top_k_results: list[Tuple[float, int]] = []

		def cosine_similarity(node_idx: int) -> float:
			"""Compute cosine similarity between the query and a stored node."""
			if not valid_mask[node_idx]:
				return float("-inf")
			return float(
				(self._embeddings[node_idx] @ query_vector)
				/ (float(embedding_norms[node_idx]) * query_norm)
			)

		start_idx: int = 0
		visited.add(start_idx)
		start_score: float = cosine_similarity(start_idx)
		candidates.append((start_score, start_idx))
		top_k_results.append((start_score, start_idx))

		while candidates and len(visited) < max(self.ef_construction, 1):
			best_position: int = max(range(len(candidates)), key=lambda idx: candidates[idx][0])
			current_score, current_idx = candidates.pop(best_position)

			neighbors: list[int] = self._graph.get(current_idx, [])
			if not neighbors:
				continue

			neighbor_array: np.ndarray = np.asarray(neighbors, dtype=int)
			unvisited_mask: np.ndarray = np.fromiter(
				(neighbor_idx not in visited for neighbor_idx in neighbor_array),
				dtype=bool,
				count=neighbor_array.shape[0],
			)
			if not np.any(unvisited_mask):
				continue

			new_neighbor_indices: np.ndarray = neighbor_array[unvisited_mask]
			visited.update(int(neighbor_idx) for neighbor_idx in new_neighbor_indices)

			neighbor_embeddings: np.ndarray = self._embeddings[new_neighbor_indices]
			neighbor_norms: np.ndarray = embedding_norms[new_neighbor_indices]
			neighbor_scores: np.ndarray = (neighbor_embeddings @ query_vector) / (
				neighbor_norms * query_norm
			)

			for neighbor_score, neighbor_idx in zip(
				neighbor_scores.tolist(),
				new_neighbor_indices.tolist(),
			):
				candidates.append((float(neighbor_score), int(neighbor_idx)))
				top_k_results.append((float(neighbor_score), int(neighbor_idx)))

			top_k_results.sort(key=lambda item: item[0], reverse=True)
			top_k_results = top_k_results[: self.ef_construction]

		final_results: list[Tuple[Document, float]] = []
		for score, node_idx in top_k_results[:k]:
			final_results.append((self._documents[node_idx], score))

		return final_results

	def _exact_search(self, query: np.ndarray, k: int) -> list[int]:
		"""Return the indices of the top-k nodes by cosine similarity."""
		if k <= 0 or self._embeddings.size == 0:
			return []

		query_vector = np.asarray(query, dtype=float).reshape(-1)
		embeddings = np.asarray(self._embeddings, dtype=float)

		if embeddings.ndim != 2:
			raise ValueError("_embeddings must be a 2D array")

		if embeddings.shape[1] != query_vector.shape[0]:
			raise ValueError("query dimensionality does not match stored embeddings")

		query_norm = np.linalg.norm(query_vector)
		if query_norm == 0.0:
			return []

		embedding_norms = np.linalg.norm(embeddings, axis=1)
		valid_mask = embedding_norms > 0.0
		if not np.any(valid_mask):
			return []

		cosine_scores = np.full(embeddings.shape[0], -np.inf, dtype=float)
		cosine_scores[valid_mask] = (
			embeddings[valid_mask] @ query_vector
		) / (embedding_norms[valid_mask] * query_norm)

		top_k = min(k, cosine_scores.shape[0])
		if top_k == cosine_scores.shape[0]:
			return np.argsort(-cosine_scores).tolist()

		partition = np.argpartition(-cosine_scores, top_k - 1)[:top_k]
		ordered = partition[np.argsort(-cosine_scores[partition])]
		return ordered.tolist()
