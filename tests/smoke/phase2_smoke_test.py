"""
tests/smoke/phase2_smoke_test.py
---------------------------------
Phase 2 Smoke Test: Storage Serialization & Persistence Engine.

Invariants verified:
  1. Index is saved without error.
  2. Index is loaded from disk into a fresh instance.
  3. Document IDs returned by the memory index and disk-reloaded index are identical.
  4. Raw float scores returned by both instances are bitwise identical (within float32 precision).
  5. Loaded index has the same document count as the original.
"""

from __future__ import annotations

import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from src.index.faiss_hnsw import FaissHNSWIndex
from src.index.persistence import save_index_to_disk, load_index_from_disk
from src.models import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert(condition: bool, message: str) -> None:
    if not condition:
        print(f"  [FAIL] {message}")
        sys.exit(1)
    print(f"  [PASS] {message}")


def _make_deterministic_vector(seed: int, dim: int = 32) -> np.ndarray:
    """Return a reproducible unit-normalised float32 vector."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-10 else vec


# ---------------------------------------------------------------------------
# Phase 2 Smoke Test
# ---------------------------------------------------------------------------

def run_phase2_smoke_test() -> None:
    print("\n" + "=" * 60)
    print("PHASE 2 SMOKE TEST: Storage Serialization & Persistence")
    print("=" * 60)

    dim = 32
    n_docs = 5
    tmp_dir = tempfile.mkdtemp(prefix="rag_phase2_")

    try:
        # ------------------------------------------------------------------
        # Build the in-memory index with 5 distinct documents
        # ------------------------------------------------------------------
        memory_index = FaissHNSWIndex(M=16, ef_construction=40, ef_search=16)
        documents = []
        for i in range(n_docs):
            vec = _make_deterministic_vector(seed=i, dim=dim)
            doc = Document(
                id=f"doc-{i:03d}",
                text=f"Document number {i}. This is test content for persistence validation.",
                metadata={"source_file": f"file_{i}.txt"},
                embedding=vec,
            )
            documents.append(doc)

        memory_index.add_documents(documents)
        print(f"\n  Built memory index: {n_docs} documents, dim={dim}")

        # ------------------------------------------------------------------
        # Save to disk
        # ------------------------------------------------------------------
        save_index_to_disk(memory_index, directory=tmp_dir)

        # Verify all three files were created
        import os as _os
        files_created = _os.listdir(tmp_dir)
        _assert(
            "faiss.index" in files_created,
            "faiss.index created on disk",
        )
        _assert(
            "documents.pkl" in files_created,
            "documents.pkl created on disk",
        )
        _assert(
            "metadata.json" in files_created,
            "metadata.json created on disk",
        )

        # ------------------------------------------------------------------
        # Load from disk into a fresh index instance
        # ------------------------------------------------------------------
        disk_index = load_index_from_disk(directory=tmp_dir)

        _assert(
            len(disk_index._documents) == n_docs,
            f"Loaded index has {n_docs} documents (got {len(disk_index._documents)})",
        )

        _assert(
            disk_index._embedding_dim == dim,
            f"Loaded index embedding_dim == {dim} (got {disk_index._embedding_dim})",
        )

        # ------------------------------------------------------------------
        # Run identical query against both indices and compare results
        # ------------------------------------------------------------------
        # Use the embedding of document 0 as the query (should always be top result)
        query_vector = _make_deterministic_vector(seed=0, dim=dim)

        memory_results = memory_index.search(query_vector, k=n_docs)
        disk_results   = disk_index.search(query_vector, k=n_docs)

        _assert(
            len(memory_results) == len(disk_results),
            f"Both indices return same number of results "
            f"(memory={len(memory_results)}, disk={len(disk_results)})",
        )

        # Compare document IDs — must be completely identical
        memory_ids = [doc.id for doc, _ in memory_results]
        disk_ids   = [doc.id for doc, _ in disk_results]
        _assert(
            memory_ids == disk_ids,
            f"Document ID order is identical: {memory_ids}",
        )

        # Compare raw float scores — must match to float32 precision
        memory_scores = [s for _, s in memory_results]
        disk_scores   = [s for _, s in disk_results]
        scores_match = all(
            abs(ms - ds) < 1e-5
            for ms, ds in zip(memory_scores, disk_scores)
        )
        _assert(
            scores_match,
            f"All scores match within 1e-5 tolerance "
            f"(memory top score={memory_scores[0]:.6f}, "
            f"disk top score={disk_scores[0]:.6f})",
        )

        # Verify top result is document 0 (query is its exact vector)
        _assert(
            memory_ids[0] == "doc-000",
            f"Top result for query seeded from doc-000 is 'doc-000' "
            f"(got '{memory_ids[0]}')",
        )

        # ------------------------------------------------------------------
        # Verify document text is preserved through serialization
        # ------------------------------------------------------------------
        loaded_doc_texts = {doc.id: doc.text for doc in disk_index._documents}
        for orig_doc in documents:
            _assert(
                loaded_doc_texts.get(orig_doc.id) == orig_doc.text,
                f"Text preserved for '{orig_doc.id}'",
            )

        print("\n--- Score Comparison Table ---")
        print(f"  {'Doc ID':<12} {'Memory Score':>14} {'Disk Score':>14} {'Delta':>12}")
        print("  " + "-" * 54)
        for (mdoc, ms), (ddoc, ds) in zip(memory_results, disk_results):
            delta = abs(ms - ds)
            print(f"  {mdoc.id:<12} {ms:>14.6f} {ds:>14.6f} {delta:>12.2e}")

        print("\n[ALL PASSED] PHASE 2 SMOKE TEST PASSED -- All invariants satisfied.\n")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_phase2_smoke_test()
