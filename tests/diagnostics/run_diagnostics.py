"""
tests/diagnostics/run_diagnostics.py
--------------------------------------
Read-only forensic diagnostic script for the RAG retrieval pipeline.

Performs four sequential checks:
  1. Index Demographics — storage audit of indexed documents
  2. Embedding Integrity — MiniLM embed_query vector validation
  3. Raw Retrieval Profiling — bypasses RAG/LLM, raw FAISS top-20 search
  4. Corpus Inspection — ground-truth scan of Sweep H chunks

Run with:
    uv run python tests/diagnostics/run_diagnostics.py
"""

from __future__ import annotations

import sys
import io

# Force UTF-8 output on Windows to prevent cp1252 encoding crashes
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import os
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import faiss
import numpy as np

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.models import Document

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
INDEX_DIR = _PROJECT_ROOT / "data" / "index"
DOCS_PKL  = INDEX_DIR / "documents.pkl"
FAISS_IDX = INDEX_DIR / "faiss.index"
META_JSON = INDEX_DIR / "metadata.json"

SEPARATOR = "=" * 80


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 1: Index Demographics
# ──────────────────────────────────────────────────────────────────────────────
def check_index_demographics() -> list[Document]:
    section("CHECK 1: INDEX DEMOGRAPHICS (Storage Audit)")

    # Load documents.pkl
    if not DOCS_PKL.exists():
        print(f"[FATAL] {DOCS_PKL} does not exist. Index has not been built.")
        return []

    with open(DOCS_PKL, "rb") as fh:
        documents: list[Document] = pickle.load(fh)

    print(f"Total indexed chunks: {len(documents)}")

    # Unique source files
    source_files = [d.metadata.get("source_file", d.id) for d in documents]
    unique_sources = set(source_files)
    print(f"Unique source files:  {len(unique_sources)}")
    for sf in sorted(unique_sources):
        count = source_files.count(sf)
        print(f"  {sf:50s}  → {count} chunks")

    # Sweep H specific count
    sweep_h_chunks = [d for d in documents if "sweep_h" in (d.metadata.get("source_file", d.id)).lower()]
    print(f"\nChunks from files containing 'sweep_h': {len(sweep_h_chunks)}")
    for i, ch in enumerate(sweep_h_chunks):
        sf = ch.metadata.get("source_file", ch.id)
        print(f"  [{i}] source_file={sf!r}, chunk_id={ch.id!r}")

    # Embedding dimensionality audit
    print("\n--- Embedding Dimensionality Audit ---")
    dim_counts: Counter = Counter()
    dtype_counts: Counter = Counter()
    null_count = 0
    for d in documents:
        if d.embedding is None:
            null_count += 1
        else:
            dim_counts[d.embedding.shape] += 1
            dtype_counts[str(d.embedding.dtype)] += 1

    print(f"Documents with None embedding: {null_count}")
    for shape, count in dim_counts.items():
        print(f"Embedding shape {shape}: {count} documents")
    for dt, count in dtype_counts.items():
        print(f"Embedding dtype {dt}: {count} documents")

    # FAISS index dimensionality
    if FAISS_IDX.exists():
        raw_index = faiss.read_index(str(FAISS_IDX))
        faiss_dim = raw_index.d
        faiss_ntotal = raw_index.ntotal
        print(f"\nFAISS index dimensionality (d):  {faiss_dim}")
        print(f"FAISS index total vectors:       {faiss_ntotal}")
    else:
        print(f"\n[WARN] {FAISS_IDX} not found.")
        faiss_dim = None

    # metadata.json
    if META_JSON.exists():
        with open(META_JSON) as fh:
            meta = json.load(fh)
        print(f"\nmetadata.json contents: {json.dumps(meta, indent=2)}")
    else:
        print(f"\n[WARN] {META_JSON} not found.")

    # Consistency check
    doc_dims = set(dim_counts.keys())
    if faiss_dim is not None:
        all_match = all(shape == (faiss_dim,) for shape in doc_dims)
        print(f"\n✓ Doc embedding dims match FAISS d={faiss_dim}? → {all_match}")
    else:
        print("\n⚠ Cannot verify dim consistency without FAISS index.")

    return documents


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 2: Embedding Integrity
# ──────────────────────────────────────────────────────────────────────────────
def check_embedding_integrity() -> np.ndarray:
    section("CHECK 2: EMBEDDING INTEGRITY (MiniLM Math Check)")

    from src.ingestion.embeddings import MiniLMEmbeddingService

    svc = MiniLMEmbeddingService()
    test_query = "What was the crossover point discovered in Sweep H?"
    vec = svc.embed_query(test_query)

    print(f"Query:           {test_query!r}")
    print(f"Vector dtype:    {vec.dtype}")
    print(f"Vector shape:    {vec.shape}")
    print(f"Dimensionality:  {vec.shape[0]}")

    l2_norm = float(np.linalg.norm(vec))
    print(f"L2 norm:         {l2_norm:.10f}")
    print(f"L2 norm == 1.0?  {np.isclose(l2_norm, 1.0)}")

    # Verify it matches FAISS index dim
    if FAISS_IDX.exists():
        raw_index = faiss.read_index(str(FAISS_IDX))
        print(f"\nQuery vector dim ({vec.shape[0]}) == FAISS index dim ({raw_index.d})?  "
              f"→ {vec.shape[0] == raw_index.d}")

    return vec


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 3: Raw Retrieval Profiling
# ──────────────────────────────────────────────────────────────────────────────
def check_raw_retrieval(query_vec: np.ndarray, documents: list[Document]) -> None:
    section("CHECK 3: RAW RETRIEVAL PROFILING (FAISS Top-20 Search)")

    if not FAISS_IDX.exists():
        print("[FATAL] No FAISS index file. Cannot run retrieval check.")
        return

    from src.index.persistence import load_index_from_disk

    index = load_index_from_disk(str(INDEX_DIR))
    results = index.search(query_vec, k=20)

    print(f"Results returned: {len(results)}\n")
    print(f"{'Rank':<6} {'Score':>8}  {'Source File':<50}  Chunk Snippet (first 60 chars)")
    print("-" * 130)

    for rank, (doc, score) in enumerate(results, 1):
        sf = doc.metadata.get("source_file", doc.id)
        snippet = doc.text[:60].replace("\n", " ").strip()
        print(f"  {rank:<4} {score:>8.5f}  {sf:<50}  {snippet}")

    # Score distribution analysis
    scores = [s for _, s in results]
    if scores:
        print(f"\n--- Score Distribution ---")
        print(f"  Max score:  {max(scores):.5f}")
        print(f"  Min score:  {min(scores):.5f}")
        print(f"  Mean score: {sum(scores)/len(scores):.5f}")
        print(f"  Spread (max-min): {max(scores)-min(scores):.5f}")

        # Check if scores are tightly clustered
        spread = max(scores) - min(scores)
        if spread < 0.15:
            print(f"  ⚠ Scores are TIGHTLY CLUSTERED (spread={spread:.5f} < 0.15)")
        else:
            print(f"  ✓ Scores have meaningful separation (spread={spread:.5f})")


# ──────────────────────────────────────────────────────────────────────────────
# CHECK 4: Corpus Inspection (Ground Truth)
# ──────────────────────────────────────────────────────────────────────────────
def check_corpus_inspection(documents: list[Document]) -> None:
    section("CHECK 4: CORPUS INSPECTION (Ground Truth for Sweep H)")

    sweep_h_chunks = [
        d for d in documents
        if "sweep_h" in (d.metadata.get("source_file", d.id)).lower()
    ]

    if not sweep_h_chunks:
        print("[FATAL] No chunks found with 'sweep_h' in source_file.")
        print("The Sweep H benchmark data is NOT in the indexed database.")
        return

    print(f"Total Sweep H chunks found: {len(sweep_h_chunks)}\n")

    # Print every chunk's full raw text
    for i, doc in enumerate(sweep_h_chunks):
        sf = doc.metadata.get("source_file", doc.id)
        print(f"--- Sweep H Chunk [{i}] (source: {sf}, id: {doc.id}) ---")
        print(f"    Length: {len(doc.text)} chars")
        print(doc.text)
        print()

    # Search for target answer strings
    print("=" * 80)
    print("  GROUND TRUTH SEARCH: Looking for target answer content")
    print("=" * 80)

    target_patterns = [
        ("10000", "exact string '10000'"),
        ("10,000", "formatted string '10,000'"),
        ("crossover", "string 'crossover'"),
        ("5000", "exact string '5000' (crossover lower bound)"),
        ("5,000", "formatted string '5,000'"),
    ]

    for pattern, desc in target_patterns:
        found_in = []
        for i, doc in enumerate(sweep_h_chunks):
            if pattern.lower() in doc.text.lower():
                # Find exact line
                for lineno, line in enumerate(doc.text.splitlines(), 1):
                    if pattern.lower() in line.lower():
                        snippet = line.strip()[:100]
                        found_in.append((i, lineno, snippet))
        
        found = len(found_in) > 0
        status = "✓ FOUND" if found else "✗ NOT FOUND"
        print(f"\n  {status}: {desc}")
        if found_in:
            for chunk_idx, lineno, snippet in found_in:
                print(f"    → chunk [{chunk_idx}], line {lineno}: {snippet!r}")

    # Also search for key semantic phrases
    print(f"\n--- Semantic phrase search across ALL chunks ---")
    semantic_queries = [
        "crossover point",
        "HNSW becomes faster",
        "transitions from being slower",
        "Flat index is faster",
        "N=10,000",
        "N=10000",
    ]
    for phrase in semantic_queries:
        matches = []
        for i, doc in enumerate(documents):
            if phrase.lower() in doc.text.lower():
                sf = doc.metadata.get("source_file", doc.id)
                matches.append((i, sf))
        status = "✓ FOUND" if matches else "✗ NOT FOUND"
        print(f"  {status}: {phrase!r}")
        for idx, sf in matches:
            snippet = documents[idx].text[:80].replace("\n", " ")
            print(f"    → document[{idx}] from {sf}: {snippet!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("  RAG RETRIEVAL PIPELINE -- FORENSIC DIAGNOSTIC REPORT")
    print("#" * 80)

    documents = check_index_demographics()
    query_vec = check_embedding_integrity()
    check_raw_retrieval(query_vec, documents)
    check_corpus_inspection(documents)

    print(f"\n{'#' * 80}")
    print("  DIAGNOSTIC COMPLETE")
    print("#" * 80)
