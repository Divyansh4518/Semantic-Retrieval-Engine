"""Flat Navigable Small World graph index skeleton."""

from __future__ import annotations

import heapq
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
		"ef_search",
		"ef_construction",
		"_documents",
		"_embeddings",
		"_graph",
		"_embedding_dim",
	)

	def __init__(self, M: int = 16, ef_construction: int = 32, ef_search: int = 256) -> None:
		"""Initialize the graph index configuration and backing storage."""
		self.M = M
		self.ef_construction = ef_construction
		self.ef_search = ef_search
		self._documents: list[Document] = []
		self._embeddings: np.ndarray = np.empty((0, 0), dtype=float)
		self._graph: dict[int, set[int]] = {}
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

		for offset in range(len(documents)):
			self._graph[start_idx + offset] = set()

		for offset in range(len(documents)):
			new_idx: int = start_idx + offset
			embedding_array = new_embeddings[offset]
			neighbor_indices: list[int] = self._exact_search(embedding_array, k=self.M + 1)
			filtered_neighbors: list[int] = [
				neighbor_idx
				for neighbor_idx in neighbor_indices
				if neighbor_idx != new_idx
			]
			filtered_neighbors = filtered_neighbors[: max(self.M, 0)]

			new_neighbors = self._graph[new_idx]

			for neighbor_idx in filtered_neighbors:
				new_neighbors.add(neighbor_idx)
				self._graph[neighbor_idx].add(new_idx)
				if len(self._graph[neighbor_idx]) > self.M:
					self._prune_overloaded_node(neighbor_idx)

			if len(new_neighbors) > self.M:
				self._prune_overloaded_node(new_idx)

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
		if not np.any(valid_mask):
			return []

		entry_idx: int | None = self._select_entry_point(query_vector, query_norm)
		if entry_idx is None:
			return []

		target_results: int = max(int(k), max(self.ef_search, 1))
		visited: set[int] = set()
		candidates: list[Tuple[float, int]] = []
		best_visited: list[Tuple[float, int]] = []

		entry_score: float = self._cosine_similarity(
			entry_idx,
			query_vector,
			query_norm,
			embedding_norms,
			valid_mask,
		)
		heappush_candidate = heapq.heappush
		heappop_candidate = heapq.heappop
		heappush_best = heapq.heappush
		heapreplace_best = heapq.heapreplace

		heappush_candidate(candidates, (-entry_score, entry_idx))

		while candidates and len(visited) < max(self.ef_search, 1):
			negative_score, current_idx = heappop_candidate(candidates)
			current_score = -negative_score
			if current_idx in visited:
				continue

			if best_visited and len(best_visited) >= target_results and current_score < best_visited[0][0]:
				break

			visited.add(current_idx)
			if len(best_visited) < target_results:
				heappush_best(best_visited, (current_score, current_idx))
			elif current_score > best_visited[0][0]:
				heapreplace_best(best_visited, (current_score, current_idx))

			neighbors: set[int] = self._graph.get(current_idx, set())
			if not neighbors:
				continue

			neighbor_array: np.ndarray = np.fromiter(
				neighbors,
				dtype=int,
				count=len(neighbors),
			)
			unvisited_mask: np.ndarray = np.fromiter(
				(neighbor_idx not in visited for neighbor_idx in neighbor_array),
				dtype=bool,
				count=neighbor_array.shape[0],
			)
			if not np.any(unvisited_mask):
				continue

			new_neighbor_indices: np.ndarray = neighbor_array[unvisited_mask]

			neighbor_embeddings: np.ndarray = self._embeddings[new_neighbor_indices]
			neighbor_norms: np.ndarray = embedding_norms[new_neighbor_indices]
			neighbor_scores: np.ndarray = np.full(new_neighbor_indices.shape[0], -np.inf, dtype=float)
			neighbor_valid_mask: np.ndarray = neighbor_norms > 0.0
			if np.any(neighbor_valid_mask):
				neighbor_scores[neighbor_valid_mask] = (
					neighbor_embeddings[neighbor_valid_mask] @ query_vector
				) / (neighbor_norms[neighbor_valid_mask] * query_norm)

			for neighbor_score, neighbor_idx in zip(
				neighbor_scores.tolist(),
				new_neighbor_indices.tolist(),
			):
				if int(neighbor_idx) in visited:
					continue
				heappush_candidate(candidates, (-float(neighbor_score), int(neighbor_idx)))

		final_results: list[Tuple[Document, float]] = []
		ordered_results = sorted(best_visited, key=lambda item: item[0], reverse=True)
		for score, node_idx in ordered_results[:k]:
			final_results.append((self._documents[node_idx], score))

		evaluated_count: int = len(visited)
		total_nodes: int = len(self._documents)
		explored_percentage: float = (evaluated_count / total_nodes * 100.0) if total_nodes > 0 else 0.0
		best_similarity_found: float = ordered_results[0][0] if ordered_results else float("-inf")
		worst_returned_similarity: float = (
			min(score for _, score in final_results) if final_results else float("-inf")
		)
		returned_ids: list[str] = [document.id for document, _ in final_results]

		print("# === Search Debug ===")
		print(f"Entry Node: {entry_idx}")
		print(f"Entry Similarity: {entry_score:.6f}")
		print(f"Nodes Evaluated: {evaluated_count} / {total_nodes} ({explored_percentage:.2f}%)")
		print(f"Best Similarity Found: {best_similarity_found:.6f}")
		print(f"Worst Returned Similarity: {worst_returned_similarity:.6f}")
		print(f"Returned IDs: {returned_ids}")
		return final_results

	def debug_graph_stats(self) -> None:
		"""Print a diagnostic summary of the current graph structure."""
		node_count: int = len(self._documents)
		edge_count: int = sum(len(neighbors) for neighbors in self._graph.values()) // 2
		average_degree: float = (2.0 * edge_count / node_count) if node_count > 0 else 0.0
		isolated_nodes: int = sum(1 for node_idx in range(node_count) if len(self._graph.get(node_idx, set())) == 0)

		reachable_count: int = 0
		if node_count > 0:
			start_node: int = next((node_idx for node_idx in range(node_count) if self._graph.get(node_idx)), 0)
			visited: set[int] = set()
			stack: list[int] = [start_node]

			while stack:
				node_idx = stack.pop()
				if node_idx in visited:
					continue
				visited.add(node_idx)
				stack.extend(
					neighbor_idx
					for neighbor_idx in self._graph.get(node_idx, set())
					if neighbor_idx not in visited
				)

			reachable_count = len(visited)

		print("\n=== GraphIndex Debug Stats ===")
		print(f"Nodes: {node_count}")
		print(f"Edges: {edge_count}")
		print(f"Average degree: {average_degree:.2f}")
		print(f"Isolated nodes: {isolated_nodes}")
		print(f"Reachable nodes from DFS entry: {reachable_count}")
		print("==============================\n")

	def _select_entry_point(self, query_vector: np.ndarray, query_norm: float) -> int | None:
		"""Choose an entry point by scoring a small random pool of nodes."""
		if not self._documents:
			return None

		pool_size: int = min(8, len(self._documents))
		candidate_indices = np.random.choice(len(self._documents), size=pool_size, replace=False)
		candidate_embeddings: np.ndarray = self._embeddings[candidate_indices]
		candidate_norms: np.ndarray = np.linalg.norm(candidate_embeddings, axis=1)
		candidate_scores: np.ndarray = np.full(candidate_indices.shape[0], -np.inf, dtype=float)
		valid_mask: np.ndarray = candidate_norms > 0.0

		if np.any(valid_mask):
			candidate_scores[valid_mask] = (
				candidate_embeddings[valid_mask] @ query_vector
			) / (candidate_norms[valid_mask] * query_norm)

		best_position: int = int(np.argmax(candidate_scores))
		best_score: float = float(candidate_scores[best_position])
		if not np.isfinite(best_score):
			fallback = np.flatnonzero(np.linalg.norm(self._embeddings, axis=1) > 0.0)
			if fallback.size == 0:
				return None
			return int(fallback[0])

		return int(candidate_indices[best_position])

	def _cosine_similarity(
		self,
		node_idx: int,
		query_vector: np.ndarray,
		query_norm: float,
		embedding_norms: np.ndarray,
		valid_mask: np.ndarray,
	) -> float:
		"""Compute cosine similarity for a single node with zero-norm protection."""
		if not valid_mask[node_idx]:
			return float("-inf")
		return float(
			(self._embeddings[node_idx] @ query_vector)
			/ (float(embedding_norms[node_idx]) * query_norm)
		)

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

	def _prune_overloaded_node(self, node_idx: int) -> None:
		"""Trim a node's connections to the strongest cosine neighbors."""
		neighbors = self._graph.get(node_idx)
		if not neighbors:
			return

		neighbors.discard(node_idx)
		if len(neighbors) <= self.M:
			return

		connected_indices: np.ndarray = np.fromiter(
			neighbors,
			dtype=int,
			count=len(neighbors),
		)
		node_embedding: np.ndarray = np.asarray(self._embeddings[node_idx], dtype=float).reshape(-1)
		connected_embeddings: np.ndarray = self._embeddings[connected_indices]

		node_norm: float = float(np.linalg.norm(node_embedding))
		connected_norms: np.ndarray = np.linalg.norm(connected_embeddings, axis=1)
		valid_mask: np.ndarray = connected_norms > 0.0
		cosine_scores: np.ndarray = np.full(connected_indices.shape[0], -np.inf, dtype=float)

		if node_norm > 0.0 and np.any(valid_mask):
			cosine_scores[valid_mask] = (
				connected_embeddings[valid_mask] @ node_embedding
			) / (connected_norms[valid_mask] * node_norm)

		keep_count: int = min(max(self.M, 0), cosine_scores.shape[0])
		if keep_count <= 0:
			kept_indices = np.empty(0, dtype=int)
		else:
			if keep_count == cosine_scores.shape[0]:
				return
			keep_partition: np.ndarray = np.argpartition(-cosine_scores, keep_count - 1)[:keep_count]
			keep_partition = keep_partition[np.argsort(-cosine_scores[keep_partition])]
			kept_indices = connected_indices[keep_partition]

		kept_neighbor_ids: set[int] = set(kept_indices.tolist())
		dropped_indices = [
			neighbor_idx
			for neighbor_idx in connected_indices.tolist()
			if neighbor_idx not in kept_neighbor_ids
		]

		neighbors.intersection_update(kept_neighbor_ids)
		for dropped_idx in dropped_indices:
			dropped_neighbors = self._graph.get(dropped_idx)
			if dropped_neighbors is not None:
				dropped_neighbors.discard(node_idx)
