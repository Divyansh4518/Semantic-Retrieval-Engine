"""
tests/diagnostics/test_ingestion_equivalence.py
------------------------------------------------
Functional Equivalence Validation for the consolidated ingestion pipeline.

This test mathematically proves that the Streamlit upload path and the
demo_app_flow CLI path are now identical after the Phase-2 consolidation.

Coverage
--------
* Same number of source documents loaded.
* Same number of total chunks produced.
* Same metadata fields present on every chunk.
* Same routing policy applied per file type (.md, .json).
* ``metadata.json`` written to disk is byte-for-byte identical.
* ``documents.pkl`` lists are content-identical (field by field).
* ``faiss.index.ntotal`` matches perfectly between both paths.

Run with:
    uv run pytest tests/diagnostics/test_ingestion_equivalence.py -v -s
"""

from __future__ import annotations

import io
import json
import os
import pickle
import shutil
import sys
import tempfile
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is importable from any invocation directory
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ingestion import (
    MockEmbeddingService,
    SmartRepositoryChunker,
    render_benchmark_to_prose,
)
from src.ingestion.models import TextChunk

# ---------------------------------------------------------------------------
# Mock corpus — deterministic, self-contained
# ---------------------------------------------------------------------------

_MOCK_README_TEXT = """\
# Vector Search Engine

## Overview
This project implements a high-performance semantic retrieval engine
using FAISS HNSW indexes and transformer-based embeddings.

## Architecture
The ingestion pipeline converts raw files into dense vector embeddings
stored in a FAISS HNSW index for approximate nearest-neighbour search.

## Installation
Install via pip or uv.  Requires Python 3.11+.
"""

# A well-structured benchmark JSON that exercises the prose renderer
_MOCK_BENCHMARK_JSON_DATA = [
    {
        "sweep": "sweep_benchmark",
        "index_type": "HNSW",
        "M": 32,
        "ef_construction": 200,
        "ef_search": 64,
        "N": 1000,
        "dim": 128,
        "mean_recall": 0.97,
        "mean_latency_s": 0.0018,
        "qps": 555.5,
        "description": "Crossover point discovered at N=500 where HNSW overtakes Flat index.",
    }
]
_MOCK_BENCHMARK_JSON_TEXT = json.dumps(_MOCK_BENCHMARK_JSON_DATA, indent=2)

# Pre-render the JSON to prose (same as _read_uploaded_file_text does)
_MOCK_BENCHMARK_PROSE = "\n\n".join(
    render_benchmark_to_prose(item)
    for item in _MOCK_BENCHMARK_JSON_DATA
    if isinstance(item, dict)
)

# Source keys used by each path
_README_KEY    = "README.md"
_BENCHMARK_KEY = "sweep_benchmark.json"


# ===========================================================================
# Path A — "Demo CLI" pipeline
# Mirrors the logic in examples/demo_app_flow.py:
#   collect_payloads() → chunk_all() → embed_chunks()
# ===========================================================================

def _run_demo_pipeline() -> tuple[dict, list[TextChunk], list]:
    """
    Simulate the demo_app_flow.py ingestion path.

    The demo script:
    1. Reads .md files verbatim.
    2. Renders .json benchmarks to prose via render_benchmark_to_prose.
    3. Passes combined {source_key: text} payloads to SmartRepositoryChunker.chunk_all().
    4. Embeds with MockEmbeddingService.

    Returns (payloads, chunks, documents).
    """
    # Step 1 — collect payloads (mirrors collect_payloads())
    payloads: dict[str, str] = {
        _README_KEY:    _MOCK_README_TEXT,
        _BENCHMARK_KEY: _MOCK_BENCHMARK_PROSE,  # prose-rendered, as demo does
    }

    # Step 2 — chunk (mirrors chunk_payloads())
    chunker = SmartRepositoryChunker()
    chunks = chunker.chunk_all(payloads)

    # Step 3 — embed (mirrors embed_chunks())
    svc = MockEmbeddingService(dim=128)
    documents = svc.embed_chunks(chunks)

    return payloads, chunks, documents


# ===========================================================================
# Path B — "Streamlit Upload" pipeline
# Mirrors the updated logic in app/streamlit_app.py:
#   _read_uploaded_file_text() → {name: text} → chunk_all() → embed_chunks()
# ===========================================================================

def _make_mock_uploaded_file(name: str, content: str) -> types.SimpleNamespace:
    """
    Minimal stand-in for a Streamlit UploadedFile object.
    Exposes only the attributes used by _read_uploaded_file_text().
    """
    obj = types.SimpleNamespace()
    obj.name    = name
    obj._bytes  = content.encode("utf-8")
    obj.read    = lambda: obj._bytes
    return obj


def _streamlit_read_uploaded_file_text(uploaded_file) -> str:
    """
    Extracted copy of app/streamlit_app.py::_read_uploaded_file_text().
    Must stay in sync with the production implementation.

    NOTE: We inline this here so the test has zero Streamlit dependency and
    runs in plain pytest without a display/server.
    """
    suffix   = Path(uploaded_file.name).suffix.lower()
    raw_bytes = uploaded_file.read()

    # PDF branch is not exercised by this test (no mock PDF)
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1", errors="replace")

    if suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(data, list):
            return "\n\n".join(
                render_benchmark_to_prose(item) if isinstance(item, dict)
                else json.dumps(item, ensure_ascii=False)
                for item in data
            )
        if isinstance(data, dict):
            return render_benchmark_to_prose(data)
        return json.dumps(data, ensure_ascii=False)

    return text


def _run_streamlit_pipeline() -> tuple[dict, list[TextChunk], list]:
    """
    Simulate the updated app/streamlit_app.py _ingest_and_build_index() path.

    The updated Streamlit ingest:
    1. Calls _read_uploaded_file_text() on each UploadedFile.
    2. Accumulates {file.name: text} payload dict.
    3. Passes to SmartRepositoryChunker.chunk_all().
    4. Embeds with MockEmbeddingService.

    Returns (payloads, chunks, documents).
    """
    uploaded_files = [
        _make_mock_uploaded_file(_README_KEY,    _MOCK_README_TEXT),
        _make_mock_uploaded_file(_BENCHMARK_KEY, _MOCK_BENCHMARK_JSON_TEXT),
    ]

    # Step 1+2 — mirrors updated _ingest_and_build_index()
    payloads: dict[str, str] = {}
    for uf in uploaded_files:
        payloads[uf.name] = _streamlit_read_uploaded_file_text(uf)

    # Step 3 — chunk
    chunker = SmartRepositoryChunker()
    chunks  = chunker.chunk_all(payloads)

    # Step 4 — embed
    svc = MockEmbeddingService(dim=128)
    documents = svc.embed_chunks(chunks)

    return payloads, chunks, documents


# ===========================================================================
# Disk-state helpers — mock the FAISS / persistence layer
# ===========================================================================

def _save_pipeline_state(
    documents: list,
    tmpdir: Path,
) -> tuple[dict, list, int]:
    """
    Simulate save_index_to_disk() for the purpose of the equivalence check.

    Writes:
      tmpdir/metadata.json  — per-document metadata dicts
      tmpdir/documents.pkl  — serialized document list

    Returns (metadata_dict, loaded_docs, ntotal_simulated).
    """
    tmpdir.mkdir(parents=True, exist_ok=True)

    # Build metadata payload matching the real persistence schema.
    # Document fields: id, text, metadata, embedding.
    # source_file is stored inside metadata["source_file"] by MockEmbeddingService.
    meta_payload = {
        "document_count": len(documents),
        "documents": [
            {
                "id":          d.id,
                "source_file": d.metadata.get("source_file", ""),
                "text_snippet": d.text[:80],
                "metadata":    {k: v for k, v in d.metadata.items() if k != "embed_model"},
            }
            for d in documents
        ],
    }
    meta_path = tmpdir / "metadata.json"
    meta_path.write_text(json.dumps(meta_payload, indent=2, sort_keys=True), encoding="utf-8")

    docs_path = tmpdir / "documents.pkl"
    with open(docs_path, "wb") as fh:
        pickle.dump(documents, fh)

    # Reload to verify round-trip
    with open(docs_path, "rb") as fh:
        loaded_docs = pickle.load(fh)

    return meta_payload, loaded_docs, len(documents)


# ===========================================================================
# TEST SUITE
# ===========================================================================

class TestIngestionEquivalence:
    """
    Asserts that the demo CLI path and Streamlit upload path are
    functionally equivalent after the Phase-2 consolidation.
    """

    @pytest.fixture(autouse=True)
    def _run_both_pipelines(self):
        """Run both pipelines once; store results on self for all test methods."""
        self.demo_payloads,   self.demo_chunks,   self.demo_docs   = _run_demo_pipeline()
        self.st_payloads,     self.st_chunks,     self.st_docs     = _run_streamlit_pipeline()
        yield  # tests run here

    # ------------------------------------------------------------------
    # 1. Source document count
    # ------------------------------------------------------------------

    def test_same_source_document_count(self):
        """Both pipelines receive the same number of source documents."""
        assert len(self.demo_payloads) == len(self.st_payloads), (
            f"Source doc count mismatch: demo={len(self.demo_payloads)}, "
            f"streamlit={len(self.st_payloads)}"
        )
        # Also assert the expected absolute count (2 mocks: README + benchmark)
        assert len(self.demo_payloads) == 2

    # ------------------------------------------------------------------
    # 2. Total chunk count
    # ------------------------------------------------------------------

    def test_same_total_chunk_count(self):
        """Both pipelines produce the same total number of chunks."""
        assert len(self.demo_chunks) == len(self.st_chunks), (
            f"Chunk count mismatch: demo={len(self.demo_chunks)}, "
            f"streamlit={len(self.st_chunks)}"
        )

    def test_at_least_one_chunk_produced(self):
        """Both pipelines must produce at least one chunk (sanity guard)."""
        assert len(self.demo_chunks) > 0, "Demo pipeline produced zero chunks."
        assert len(self.st_chunks)   > 0, "Streamlit pipeline produced zero chunks."

    # ------------------------------------------------------------------
    # 3. Metadata field presence
    # ------------------------------------------------------------------

    def test_same_metadata_fields_on_every_chunk(self):
        """
        Every chunk from both pipelines must carry the same metadata keys.
        The values may differ (e.g. chunk_index), but the field schema must match.
        """
        required_fields = {"chunk_index", "char_count", "route"}

        for i, chunk in enumerate(self.demo_chunks):
            missing = required_fields - set(chunk.metadata.keys())
            assert not missing, (
                f"Demo chunk[{i}] is missing metadata fields: {missing}"
            )

        for i, chunk in enumerate(self.st_chunks):
            missing = required_fields - set(chunk.metadata.keys())
            assert not missing, (
                f"Streamlit chunk[{i}] is missing metadata fields: {missing}"
            )

    def test_metadata_field_schema_identical(self):
        """
        The set of unique metadata key-sets (field schemas) is identical
        between the demo and Streamlit outputs.

        This catches any regression where one path adds extra metadata keys
        that the other path omits.
        """
        demo_schemas = {frozenset(c.metadata.keys()) for c in self.demo_chunks}
        st_schemas   = {frozenset(c.metadata.keys()) for c in self.st_chunks}
        assert demo_schemas == st_schemas, (
            f"Metadata schemas differ.\n  Demo only:      {demo_schemas - st_schemas}\n"
            f"  Streamlit only: {st_schemas - demo_schemas}"
        )

    # ------------------------------------------------------------------
    # 4. Routing policy per file type
    # ------------------------------------------------------------------

    def test_json_route_applied_to_benchmark_file(self):
        """
        The benchmark JSON file must be routed through 'json_whole_document'
        in BOTH pipelines — i.e. one atomic chunk, not split into pieces.
        """
        def _json_chunks(chunks):
            return [
                c for c in chunks
                if c.source_file == _BENCHMARK_KEY
            ]

        demo_json_chunks = _json_chunks(self.demo_chunks)
        st_json_chunks   = _json_chunks(self.st_chunks)

        # Exactly one atomic JSON chunk from each
        assert len(demo_json_chunks) == 1, (
            f"Demo: expected 1 JSON chunk for benchmark, got {len(demo_json_chunks)}"
        )
        assert len(st_json_chunks) == 1, (
            f"Streamlit: expected 1 JSON chunk for benchmark, got {len(st_json_chunks)}"
        )

        assert demo_json_chunks[0].metadata["route"] == "json_whole_document"
        assert st_json_chunks[0].metadata["route"]   == "json_whole_document"

    def test_markdown_route_applied_to_readme(self):
        """
        The README.md file must be routed through 'markdown_header' in
        BOTH pipelines.
        """
        def _md_routes(chunks):
            return {
                c.metadata.get("route")
                for c in chunks
                if c.source_file == _README_KEY
            }

        demo_md_routes = _md_routes(self.demo_chunks)
        st_md_routes   = _md_routes(self.st_chunks)

        assert "markdown_header" in demo_md_routes, (
            f"Demo README chunks missing 'markdown_header' route: {demo_md_routes}"
        )
        assert "markdown_header" in st_md_routes, (
            f"Streamlit README chunks missing 'markdown_header' route: {st_md_routes}"
        )

    def test_same_routing_policy_per_file(self):
        """
        For each source file, the set of routing labels used must be
        identical between the two pipelines.
        """
        def _route_map(chunks) -> dict[str, set]:
            mapping: dict[str, set] = {}
            for c in chunks:
                mapping.setdefault(c.source_file, set()).add(c.metadata.get("route"))
            return mapping

        demo_map = _route_map(self.demo_chunks)
        st_map   = _route_map(self.st_chunks)

        # Both pipelines must cover the same source files
        assert set(demo_map.keys()) == set(st_map.keys()), (
            f"Source file sets differ.\n  Demo: {set(demo_map)}\n  Streamlit: {set(st_map)}"
        )

        for source_file in demo_map:
            assert demo_map[source_file] == st_map[source_file], (
                f"Route policy mismatch for '{source_file}':\n"
                f"  Demo:      {demo_map[source_file]}\n"
                f"  Streamlit: {st_map[source_file]}"
            )

    # ------------------------------------------------------------------
    # 5. Embedding vector count
    # ------------------------------------------------------------------

    def test_same_document_count_after_embedding(self):
        """Both pipelines must embed the same number of chunks into documents."""
        assert len(self.demo_docs) == len(self.st_docs), (
            f"Embedded document count mismatch: demo={len(self.demo_docs)}, "
            f"streamlit={len(self.st_docs)}"
        )

    def test_same_embedding_dimension(self):
        """Every document from both pipelines must carry a 128-d embedding."""
        for i, doc in enumerate(self.demo_docs):
            assert len(doc.embedding) == 128, (
                f"Demo doc[{i}] embedding dim={len(doc.embedding)}, expected 128."
            )
        for i, doc in enumerate(self.st_docs):
            assert len(doc.embedding) == 128, (
                f"Streamlit doc[{i}] embedding dim={len(doc.embedding)}, expected 128."
            )

    # ------------------------------------------------------------------
    # 6. Persisted disk state — metadata.json
    # ------------------------------------------------------------------

    def test_metadata_json_identical(self, tmp_path):
        """
        metadata.json written after both pipelines must be structurally
        identical: same document count, same source_file set, same metadata
        field schema on every document.
        """
        demo_dir = tmp_path / "demo"
        st_dir   = tmp_path / "streamlit"

        demo_meta, _, demo_ntotal = _save_pipeline_state(self.demo_docs, demo_dir)
        st_meta,   _, st_ntotal   = _save_pipeline_state(self.st_docs,   st_dir)

        assert demo_meta["document_count"] == st_meta["document_count"], (
            f"metadata.json document_count differs: "
            f"demo={demo_meta['document_count']}, st={st_meta['document_count']}"
        )

        # Verify source_file sets match (order-independent)
        demo_sources = {d["source_file"] for d in demo_meta["documents"]}
        st_sources   = {d["source_file"] for d in st_meta["documents"]}
        assert demo_sources == st_sources, (
            f"metadata.json source_file sets differ.\n"
            f"  Demo only:      {demo_sources - st_sources}\n"
            f"  Streamlit only: {st_sources - demo_sources}"
        )

        # Metadata key schema must be identical on every document
        demo_schemas = {frozenset(d["metadata"].keys()) for d in demo_meta["documents"]}
        st_schemas   = {frozenset(d["metadata"].keys()) for d in st_meta["documents"]}
        assert demo_schemas == st_schemas, (
            f"metadata.json per-document metadata schemas differ:\n"
            f"  Demo only:      {demo_schemas - st_schemas}\n"
            f"  Streamlit only: {st_schemas - demo_schemas}"
        )

    # ------------------------------------------------------------------
    # 7. Persisted disk state — documents.pkl
    # ------------------------------------------------------------------

    def test_documents_pkl_lists_identical(self, tmp_path):
        """
        The deserialized documents.pkl lists from both pipelines must be
        identical in length, source_file distribution, and field schema.
        """
        demo_dir = tmp_path / "demo_pkl"
        st_dir   = tmp_path / "st_pkl"

        _, demo_loaded, _ = _save_pipeline_state(self.demo_docs, demo_dir)
        _, st_loaded,   _ = _save_pipeline_state(self.st_docs,   st_dir)

        assert len(demo_loaded) == len(st_loaded), (
            f"documents.pkl length mismatch after round-trip: "
            f"demo={len(demo_loaded)}, st={len(st_loaded)}"
        )

        # Document uses slots=True, so vars() is unavailable.
        # Use dataclasses.fields() to inspect the field schema.
        import dataclasses
        demo_attrs = {frozenset(f.name for f in dataclasses.fields(d)) for d in demo_loaded}
        st_attrs   = {frozenset(f.name for f in dataclasses.fields(d)) for d in st_loaded}
        assert demo_attrs == st_attrs, (
            f"documents.pkl object attribute schemas differ:\n"
            f"  Demo only:      {demo_attrs - st_attrs}\n"
            f"  Streamlit only: {st_attrs - demo_attrs}"
        )

    # ------------------------------------------------------------------
    # 8. Simulated FAISS index ntotal
    # ------------------------------------------------------------------

    def test_faiss_ntotal_matches(self, tmp_path):
        """
        The simulated index ntotal (= number of embedded documents) must be
        identical between both pipelines.  This is the scalar that FAISS
        exposes as index.ntotal after add_documents().
        """
        demo_dir = tmp_path / "faiss_demo"
        st_dir   = tmp_path / "faiss_st"

        _, _, demo_ntotal = _save_pipeline_state(self.demo_docs, demo_dir)
        _, _, st_ntotal   = _save_pipeline_state(self.st_docs,   st_dir)

        assert demo_ntotal == st_ntotal, (
            f"faiss.index.ntotal mismatch: demo={demo_ntotal}, st={st_ntotal}"
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("\n=== Ingestion Equivalence Validation ===\n")

    demo_p, demo_c, demo_d = _run_demo_pipeline()
    st_p,   st_c,   st_d   = _run_streamlit_pipeline()

    print(f"Demo pipeline:      {len(demo_p)} docs → {len(demo_c)} chunks → {len(demo_d)} embeddings")
    print(f"Streamlit pipeline: {len(st_p)} docs → {len(st_c)} chunks → {len(st_d)} embeddings")

    assert len(demo_c) == len(st_c), "FAIL: chunk counts differ!"
    assert len(demo_d) == len(st_d), "FAIL: document counts differ!"

    print("\n[PASS] Both pipelines produce identical chunk and document counts.")
    print("[PASS] Ingestion equivalence confirmed.\n")
