"""Phase 1 Smoke Test — FaissHNSWIndex.

Invariants verified:
    1. search() returns exactly 5 results.
    2. All cosine scores are valid floats <= 1.0.
    3. No NaN scores.
    4. Self-neighbor test: querying with vector V returns V as the top result.

Run:
    python tests/test_hnsw_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.index.faiss_hnsw import FaissHNSWIndex  # noqa: E402
from src.models import Document  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_documents(vectors: np.ndarray, id_prefix: str = "doc") -> list[Document]:
    return [
        Document(id=f"{id_prefix}-{i}", text="", embedding=vectors[i])
        for i in range(vectors.shape[0])
    ]


def _run_invariant(label: str, condition: bool) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_search() -> bool:
    """100 random 128-D vectors; query with a fresh vector; expect 5 results."""
    print("\n=== TEST 1: Basic Search (100 docs, k=5) ===")
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((100, 128)).astype(np.float32)
    query = rng.standard_normal((128,)).astype(np.float32)

    index = FaissHNSWIndex(M=32, ef_construction=200, ef_search=64)
    index.add_documents(_make_documents(vectors))

    results = index.search(query, k=5)

    print(f"  results returned: {len(results)}")
    for rank, (doc, score) in enumerate(results):
        print(f"    rank {rank}: doc_id={doc.id}  score={score:.6f}")

    ok = True
    ok &= _run_invariant("returns exactly 5 results", len(results) == 5)
    ok &= _run_invariant(
        "all scores are floats",
        all(isinstance(s, float) for _, s in results),
    )
    ok &= _run_invariant(
        "all scores <= 1.0",
        all(s <= 1.0 + 1e-5 for _, s in results),  # tiny float tolerance
    )
    ok &= _run_invariant(
        "no NaN scores",
        all(not np.isnan(s) for _, s in results),
    )
    return ok


def test_self_neighbor() -> bool:
    """Self-neighbor: inserting V and querying with V should return V first."""
    print("\n=== TEST 2: Self-Neighbor ===")
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((100, 128)).astype(np.float32)

    # Use the 0th vector as our probe
    probe_vector = vectors[0].copy()
    probe_id = "doc-0"

    index = FaissHNSWIndex(M=32, ef_construction=200, ef_search=64)
    index.add_documents(_make_documents(vectors))

    results = index.search(probe_vector, k=5)

    print(f"  top result: doc_id={results[0][0].id}  score={results[0][1]:.6f}")

    ok = True
    ok &= _run_invariant(
        f"top result is self ({probe_id})",
        len(results) > 0 and results[0][0].id == probe_id,
    )
    ok &= _run_invariant(
        "self-similarity score ~ 1.0 (>= 0.99)",
        len(results) > 0 and results[0][1] >= 0.99,
    )
    return ok


def test_zero_vector_safety() -> bool:
    """Zero-vector in corpus must not produce NaN scores or crash."""
    print("\n=== TEST 3: Zero-Vector Safety ===")
    rng = np.random.default_rng(13)
    vectors = rng.standard_normal((50, 128)).astype(np.float32)

    # Inject a zero vector at position 25
    vectors[25] = 0.0

    index = FaissHNSWIndex(M=32, ef_construction=200, ef_search=64)
    index.add_documents(_make_documents(vectors))

    query = rng.standard_normal((128,)).astype(np.float32)
    results = index.search(query, k=5)

    ok = True
    ok &= _run_invariant("search does not crash with zero vector in corpus", True)
    ok &= _run_invariant(
        "no NaN scores with zero vector present",
        all(not np.isnan(s) for _, s in results),
    )
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1 SMOKE TEST — FaissHNSWIndex")
    print("=" * 60)

    results_all = [
        test_basic_search(),
        test_self_neighbor(),
        test_zero_vector_safety(),
    ]

    passed = sum(results_all)
    total = len(results_all)
    print(f"\n{'=' * 60}")
    print(f"RESULT: {passed}/{total} test groups passed")
    print("=" * 60)

    if passed == total:
        print("\n[PHASE 1 INVARIANTS: ALL PASSED] (OK)")
        sys.exit(0)
    else:
        print("\n[PHASE 1 INVARIANTS: FAILED] (FAIL)")
        sys.exit(1)
