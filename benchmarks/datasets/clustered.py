"""Generate vectors tightly grouped around distinct centroids."""

from __future__ import annotations

import numpy as np


def generate_clustered(
    n: int,
    dim: int,
    n_clusters: int,
    cluster_std: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate vectors tightly grouped around distinct centroids.

    Centroids are random unit-norm vectors.  Points are generated around
    each centroid with additive Gaussian noise (``std=cluster_std``) and
    re-normalized to unit length.

    Parameters
    ----------
    n : int
        Total number of vectors to generate.
    dim : int
        Dimensionality of each vector.
    n_clusters : int
        Number of distinct clusters.
    cluster_std : float
        Standard deviation of the Gaussian noise added to each centroid.
    seed : int
        Random seed for strict reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(vectors, assignments)`` where *vectors* is ``(n, dim)`` and
        *assignments* is ``(n,)`` with ``assignments[i]`` giving the
        cluster id of vector *i*.
    """
    rng = np.random.default_rng(seed)

    # Random unit-norm centroids
    centroids = rng.standard_normal((n_clusters, dim))
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroid_norms = np.maximum(centroid_norms, 1e-12)
    centroids = centroids / centroid_norms

    # Round-robin assignment then shuffle for balance
    assignments = np.array([i % n_clusters for i in range(n)], dtype=int)
    rng.shuffle(assignments)

    # Generate points around their assigned centroid
    vectors = np.empty((n, dim), dtype=float)
    for i in range(n):
        cluster_id = assignments[i]
        noise = rng.standard_normal(dim) * cluster_std
        vec = centroids[cluster_id] + noise
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        vectors[i] = vec

    return vectors, assignments
