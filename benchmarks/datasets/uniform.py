"""Generate uniformly distributed, L2-normalized random vectors."""

from __future__ import annotations

import numpy as np


def generate_uniform(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Return an (n, dim) array of L2-normalized random vectors.

    Each vector is drawn from a standard normal distribution and then
    normalized to unit length, producing a uniform distribution on the
    unit hypersphere.

    Parameters
    ----------
    n : int
        Number of vectors to generate.
    dim : int
        Dimensionality of each vector.
    seed : int
        Random seed for strict reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape ``(n, dim)`` with unit-norm rows.
    """
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms
