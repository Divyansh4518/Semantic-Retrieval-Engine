"""Helper wrapper to generate uniform vectors at variable dimensions."""

from __future__ import annotations

import numpy as np

from benchmarks.datasets.uniform import generate_uniform


def generate_at_dimension(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate *n* uniform unit-norm vectors of the given dimensionality.

    This is a thin wrapper around :func:`generate_uniform` provided for
    semantic clarity in dimensionality sweep code.

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
    return generate_uniform(n, dim, seed=seed)
