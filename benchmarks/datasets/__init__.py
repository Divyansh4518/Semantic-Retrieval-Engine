"""Synthetic dataset generators for benchmarking."""

from benchmarks.datasets.clustered import generate_clustered
from benchmarks.datasets.dimensionality import generate_at_dimension
from benchmarks.datasets.uniform import generate_uniform

__all__ = ["generate_clustered", "generate_at_dimension", "generate_uniform"]
