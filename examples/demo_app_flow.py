"""
examples/demo_app_flow.py
--------------------------
End-to-end integration demo for the RAG platform (Phase 5 — Smart Routing Ingestion).

This script demonstrates the complete upgraded pipeline:
  1. Scans the repository using RepositoryLoader (Markdown + text files).
  2. Scans benchmark JSON/JSONL files and converts them to semantic English
     prose via json_renderer.render_benchmark_to_prose() BEFORE chunking.
  3. Passes all combined payloads into the SmartRepositoryChunker which
     routes each file to a specialised strategy based on extension:
       - .json/.jsonl → single atomic chunk
       - .md → header-aware splitting
       - .py → AST/function-based splitting
       - .txt/other → paragraph-based fallback
  4. Generates deterministic embeddings via MockEmbeddingService (dim=128).
  5. Purges data/index/ and rebuilds a FaissHNSWIndex.
  6. Runs a RAGPipeline query: "What was the crossover point discovered in Sweep H?"
  7. Prints four structured telemetry blocks + per-extension chunk summary.
  8. Runs an inline validation sweep confirming prose quality.

Run with:
    python examples/demo_app_flow.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import textwrap
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is importable regardless of invocation directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.index.faiss_hnsw import FaissHNSWIndex
from src.index.persistence import save_index_to_disk
from src.ingestion import (
    MockEmbeddingService,
    SmartRepositoryChunker,
    RepositoryLoader,
    render_benchmark_file,
    render_benchmark_to_prose,
)
from src.llm.mock import MockLLM
from src.rag.pipeline import RAGPipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDEX_DIR = "data/index/"
TARGET_QUERY = "What was the crossover point discovered in Sweep H?"

# Directories to ingest via RepositoryLoader (Markdown + text only)
_MD_ROOTS: list[Path] = [
    _PROJECT_ROOT,                                   # root README.md
    _PROJECT_ROOT / "benchmarks",                    # benchmarks/README.md
    _PROJECT_ROOT / "docs" if (_PROJECT_ROOT / "docs").exists() else None,
]

# Benchmark JSON directories whose files will be prose-rendered
_BENCH_DIRS: list[Path] = [
    _PROJECT_ROOT / "benchmarks" / "outputs" / "aggregated",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    width = 64
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _block(label: str, content: str, indent: int = 4) -> None:
    """Print a formatted telemetry block."""
    print(f"\n[{label}]")
    wrapper = textwrap.TextWrapper(
        width=76,
        initial_indent=" " * indent,
        subsequent_indent=" " * indent,
    )
    for line in content.strip().splitlines():
        if line.strip():
            safe_line = line.encode("ascii", errors="replace").decode("ascii")
            print(wrapper.fill(safe_line))
        else:
            print()


# ---------------------------------------------------------------------------
# Step 1: Collect — RepositoryLoader (MD/txt) + JSON prose renderer
# ---------------------------------------------------------------------------

def collect_payloads(project_root: Path) -> dict[str, str]:
    """
    Collect all ingestion payloads as {source_key: text}.

    Strategy
    --------
    * Markdown / .txt files  → loaded verbatim via RepositoryLoader.
    * JSON / JSONL benchmarks → converted to rich English prose via
      json_renderer before entering the chunking pipeline.
    """
    payloads: dict[str, str] = {}

    # ── Project-authored Markdown files only (exclude .venv, dist-info, tests) ──
    # We explicitly load from the small set of known documentation roots so the
    # RepositoryLoader does not sweep installed package files into the corpus.
    _MD_ROOTS_RESOLVED = [
        project_root / "README.md",
        project_root / "app" / "README.md",
        project_root / "benchmarks" / "README.md",
    ]
    for md_path in _MD_ROOTS_RESOLVED:
        if md_path.exists():
            try:
                text = md_path.read_text(encoding="utf-8")
                rel = str(md_path.relative_to(project_root))
                payloads[rel] = text
                print(f"  [MD] {rel} ({len(text):,} chars)")
            except Exception as exc:
                print(f"  [SKIP] {md_path.name}: {exc}")

    # ── Benchmark JSON/JSONL → semantic prose ───────────────────────────
    bench_dir = project_root / "benchmarks" / "outputs" / "aggregated"
    if bench_dir.exists():
        for json_file in sorted(bench_dir.glob("*.json")):
            try:
                prose = render_benchmark_file(json_file)
                key = str(json_file.relative_to(project_root))
                payloads[key] = prose
                print(f"  [JSON→prose] {json_file.name} → {len(prose):,} chars")
            except Exception as exc:
                print(f"  [SKIP] {json_file.name}: {exc}")

        for jsonl_file in sorted(bench_dir.glob("*.jsonl")):
            try:
                prose = render_benchmark_file(jsonl_file)
                key = str(jsonl_file.relative_to(project_root))
                payloads[key] = prose
                print(f"  [JSONL→prose] {jsonl_file.name} → {len(prose):,} chars")
            except Exception as exc:
                print(f"  [SKIP] {jsonl_file.name}: {exc}")

    return payloads


# ---------------------------------------------------------------------------
# Step 2: Chunk — SmartRepositoryChunker with file-type-aware routing
# ---------------------------------------------------------------------------

def chunk_payloads(payloads: dict[str, str]):
    """
    Run the SmartRepositoryChunker over all payloads.

    Each file is routed to its optimal strategy:
      - .json/.jsonl → whole-document (1 chunk each)
      - .md → header-aware → character guard
      - .py → AST/function-based
      - .txt/other → paragraph fallback
    """
    chunker = SmartRepositoryChunker()
    chunks = chunker.chunk_all(payloads)
    print(f"  Produced {len(chunks)} chunks from {len(payloads)} source(s).")

    # ── Per-extension summary ──
    ext_file_count: Counter = Counter()
    ext_chunk_count: Counter = Counter()
    for source_file in payloads:
        ext = Path(source_file).suffix.lower() or "(no ext)"
        ext_file_count[ext] += 1
    for chunk in chunks:
        ext = Path(chunk.source_file).suffix.lower() or "(no ext)"
        ext_chunk_count[ext] += 1

    print("\n  Per-Extension Routing Summary:")
    print(f"  {'Extension':<12} {'Files':>6} {'Chunks':>8}  Route")
    print(f"  {'-' * 12} {'-' * 6} {'-' * 8}  {'-' * 22}")
    route_labels = {
        ".json": "json_whole_document",
        ".jsonl": "json_whole_document",
        ".md": "markdown_header",
        ".py": "python_ast",
        ".txt": "fallback_paragraph",
    }
    for ext in sorted(set(ext_file_count) | set(ext_chunk_count)):
        files = ext_file_count.get(ext, 0)
        ccount = ext_chunk_count.get(ext, 0)
        route = route_labels.get(ext, "fallback_paragraph")
        print(f"  {ext:<12} {files:>6} {ccount:>8}  {route}")
    print()
    return chunks


# ---------------------------------------------------------------------------
# Step 3: Embed
# ---------------------------------------------------------------------------

def embed_chunks(chunks):
    svc = MockEmbeddingService(dim=128)
    documents = svc.embed_chunks(chunks)
    print(f"  Generated {len(documents)} embedding vectors (dim=128).")
    return documents, svc


# ---------------------------------------------------------------------------
# Step 4: Build HNSW index
# ---------------------------------------------------------------------------

def build_index(documents) -> FaissHNSWIndex:
    index = FaissHNSWIndex(M=32, ef_construction=200, ef_search=64)
    if documents:
        index.add_documents(documents)
    print(
        f"  Built FaissHNSWIndex: {len(index._documents)} vectors, "
        f"dim={index._embedding_dim}."
    )
    return index


# ---------------------------------------------------------------------------
# Step 5: Purge old index & Persist
# ---------------------------------------------------------------------------

def persist_index(index: FaissHNSWIndex) -> None:
    # Purge stale index before rebuilding
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
        print(f"  Purged old index at '{INDEX_DIR}'.")
    os.makedirs(INDEX_DIR, exist_ok=True)
    save_index_to_disk(index, directory=INDEX_DIR)
    print(f"  Index serialized to '{INDEX_DIR}'.")


# ---------------------------------------------------------------------------
# Step 6: Query embedder
# ---------------------------------------------------------------------------

class _MockQueryEmbedder:
    """Wraps MockEmbeddingService to expose embed_query for RAGPipeline."""

    def __init__(self, dim: int = 128) -> None:
        self._svc = MockEmbeddingService(dim=dim)
        self._dim = dim

    def embed_query(self, query: str):
        import hashlib
        import numpy as np
        digest = hashlib.sha256(query.encode()).digest()
        seed = int.from_bytes(digest[:4], byteorder="little")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-10 else vec


# ---------------------------------------------------------------------------
# Inline validation: prose quality checks
# ---------------------------------------------------------------------------

def validate_prose_quality(payloads: dict[str, str]) -> None:
    """
    Inline validation sweep to confirm JSON files were converted to prose.

    Checks
    ------
    1. At least one JSON-sourced payload exists.
    2. No payload is raw JSON (starts with '{' or '[').
    3. All JSON-sourced payloads contain complete English sentences
       (contain at least one period-terminated sentence).
    4. Sweep H prose mentions the crossover point.
    5. Sweep F prose mentions both NumPy and FAISS implementations.
    6. Sweep G prose mentions batch size and QPS.
    """
    _banner("Inline Validation: Prose Quality Checks")

    json_payloads = {
        k: v for k, v in payloads.items()
        if k.endswith(".json") or k.endswith(".jsonl")
    }

    def _check(condition: bool, label: str) -> None:
        tag = "[PASS]" if condition else "[FAIL]"
        print(f"  {tag} {label}")
        if not condition:
            sys.exit(1)

    _check(len(json_payloads) >= 1, f"At least 1 JSON payload found (got {len(json_payloads)})")

    for key, text in json_payloads.items():
        name = Path(key).name
        _check(
            not text.strip().startswith(("{", "[")),
            f"{name}: payload is NOT raw JSON (prose conversion succeeded)",
        )
        _check(
            ". " in text or text.strip().endswith("."),
            f"{name}: payload contains English sentences (has period-terminated text)",
        )
        _check(
            any(word in text.lower() for word in ["benchmark", "sweep", "latency", "vector", "index", "recall", "qps"]),
            f"{name}: payload contains benchmark domain vocabulary",
        )

    # Sweep-specific semantic checks
    sweep_h_keys = [k for k in json_payloads if "sweep_h" in k.lower()]
    if sweep_h_keys:
        h_text = json_payloads[sweep_h_keys[0]]
        _check("crossover" in h_text.lower(), "Sweep H prose mentions the crossover point")
        _check("hnsw" in h_text.lower(), "Sweep H prose mentions HNSW index")
        _check("flat" in h_text.lower(), "Sweep H prose mentions Flat index comparison")

    sweep_f_keys = [k for k in json_payloads if "sweep_f" in k.lower()]
    if sweep_f_keys:
        f_text = json_payloads[sweep_f_keys[0]]
        _check("numpy" in f_text.lower() or "exact" in f_text.lower(),
               "Sweep F prose mentions NumPy/Exact implementation")
        _check("faiss" in f_text.lower(), "Sweep F prose mentions FAISS implementation")
        _check("match rate" in f_text.lower(), "Sweep F prose reports exact-vs-FAISS match rate")

    sweep_g_keys = [k for k in json_payloads if "sweep_g" in k.lower()]
    if sweep_g_keys:
        g_text = json_payloads[sweep_g_keys[0]]
        _check("batch" in g_text.lower(), "Sweep G prose mentions batch queries")
        _check("qps" in g_text.lower(), "Sweep G prose reports QPS throughput")

    print("\n  All prose quality checks passed.")


# ---------------------------------------------------------------------------
# Main demo flow
# ---------------------------------------------------------------------------

def main() -> None:
    _banner("RAG PLATFORM — Smart Routing Ingestion Demo (Phase 5)")
    print(f"\n  Project root: {_PROJECT_ROOT}")
    print(f"  Target query: \"{TARGET_QUERY}\"")

    # ── Step 1: Collect ──
    _banner("Step 1: Collecting Source Material (RepositoryLoader + JSON Prose Renderer)")
    payloads = collect_payloads(_PROJECT_ROOT)

    if not payloads:
        print("  ERROR: No source files found. Aborting.")
        sys.exit(1)

    print(f"\n  Total payloads: {len(payloads)} source(s)")

    # ── Inline validation of prose quality ──
    validate_prose_quality(payloads)

    # ── Step 2: Chunk ──
    _banner("Step 2: Chunking (SmartRepositoryChunker — file-type-aware routing)")
    chunks = chunk_payloads(payloads)

    if not chunks:
        print("  ERROR: No chunks produced. Aborting.")
        sys.exit(1)

    # Validate chunks contain sentence-like text (no raw JSON fragments)
    _banner("Inline Validation: Chunk Quality Checks")
    raw_json_chunks = [
        c for c in chunks
        if c.text.strip().startswith(("{", "[")) or '": ' in c.text[:60]
    ]
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Raw JSON fragments (should be 0): {len(raw_json_chunks)}")
    if raw_json_chunks:
        print("  [WARN] Some raw JSON fragments found — check prose renderer coverage.")
        for c in raw_json_chunks[:3]:
            print(f"    Source: {c.source_file!r} | Preview: {c.text[:80]!r}")
    else:
        print("  [PASS] Zero raw JSON fragments — all chunks are clean prose or Markdown.")

    # ── Step 3: Embed ──
    _banner("Step 3: Generating Embeddings (MockEmbeddingService, dim=128)")
    documents, _ = embed_chunks(chunks)

    # ── Step 4: Build Index ──
    _banner("Step 4: Building FaissHNSWIndex")
    index = build_index(documents)

    # ── Step 5: Persist ──
    _banner("Step 5: Serializing Index to Disk")
    persist_index(index)

    # ── Step 6: Build RAGPipeline ──
    _banner("Step 6: Initializing RAGPipeline")
    embedder = _MockQueryEmbedder(dim=128)
    llm = MockLLM()
    pipeline = RAGPipeline(
        index=index,
        embedding_service=embedder,
        llm=llm,
        top_k=5,
    )
    print(f"  RAGPipeline ready (MockLLM, top_k=5).")

    # ── Step 7: Execute query ──
    _banner("Step 7: Executing Target Query")
    print(f"\n  Query: \"{TARGET_QUERY}\"")
    print("  Running pipeline...")
    response = pipeline.query(TARGET_QUERY)

    # ── Step 8: Print structured telemetry blocks ──
    _banner("TELEMETRY OUTPUT — 4 Structured Blocks")

    # Block 1: Number of Documents
    doc_summary = (
        f"Total source payloads loaded:  {len(payloads)}\n"
        f"Total chunks produced:         {len(chunks)}\n"
        f"Total vectors in HNSW index:   {len(documents)}\n"
        f"Chunks retrieved for query:    {len(response.retrieved_documents)}"
    )
    _block("1  NUMBER OF DOCUMENTS", doc_summary)

    # Block 2: Context Created
    context_preview = response.context
    if len(context_preview) > 1500:
        context_preview = context_preview[:1500] + "\n  ... [truncated for display]"
    _block("2  CONTEXT CREATED", context_preview)

    # Block 3: Query
    _block("3  QUERY", response.query)

    # Block 4: Result
    result_text = (
        f"Answer:\n    {response.answer}\n\n"
        f"Retrieved Sources:\n"
    )
    for i, r in enumerate(response.retrieved_documents, 1):
        source_name = Path(r.source_file).name
        result_text += (
            f"    [{i}] {source_name} | "
            f"Score: {r.score:.4f} | "
            f"ID: {r.document_id}\n"
        )
    result_text += (
        f"\nLLM Backend:       {response.llm_response.model_name}\n"
        f"Prompt Tokens:     {response.llm_response.prompt_tokens}\n"
        f"Completion Tokens: {response.llm_response.completion_tokens}\n"
        f"Total Tokens:      {response.llm_response.total_tokens}"
    )
    _block("4  RESULT", result_text)

    _banner("DEMO COMPLETE -- Phase 5 smart routing ingestion pipeline verified.")
    print()


if __name__ == "__main__":
    main()
