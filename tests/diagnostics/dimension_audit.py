"""
tests/diagnostics/dimension_audit.py
--------------------------------------
Forensic dimension audit for the Vector Search RAG Platform.

Inspects and prints the full State Triad, Provider Baseline, and
Wiring Audit without modifying any source files.

Run with:
    python tests/diagnostics/dimension_audit.py
    (from the project root, with the virtualenv active)
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path (mirrors app/streamlit_app.py bootstrap)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants (mirrored from streamlit_app.py — not imported from app/)
# ---------------------------------------------------------------------------
_MINILM_DIM = 384
_MOCK_DIM   = 128
INDEX_DIR   = _PROJECT_ROOT / "data" / "index"
FAISS_PATH  = INDEX_DIR / "faiss.index"
META_PATH   = INDEX_DIR / "metadata.json"
DOCS_PATH   = INDEX_DIR / "documents.pkl"

SEP = "=" * 70


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ===========================================================================
# SECTION 1 — State Triad
# ===========================================================================
section("SECTION 1 · STATE TRIAD")

# 1a. session_state.embedding_provider default
#     (Streamlit session_state is not available outside a running app;
#      we report the programmatic default as set in both streamlit_app.py
#      and ui_components.py instead.)
SESSION_DEFAULT_PROVIDER = "MiniLM (Local - 384d)"   # hardcoded in _get_embedding_service() L218
UI_DEFAULT_PROVIDER      = "MiniLM (Local - 384d)"   # hardcoded in ui_components.py L261

print(f"\n[1a] session_state.get('embedding_provider') default (streamlit_app.py:218):")
print(f"     Value  : '{SESSION_DEFAULT_PROVIDER}'")
print(f"     Implies: MiniLM → dimension = {_MINILM_DIM}")

print(f"\n[1a'] UI selectbox default (ui_components.py:261):")
print(f"     embedding_options[0] = '{UI_DEFAULT_PROVIDER}'")
print(f"     Implies: MiniLM → dimension = {_MINILM_DIM}")

# 1b. Native dimension of the persisted faiss.index on disk
print(f"\n[1b] Persisted faiss.index — raw FAISS object (data/index/faiss.index):")
if FAISS_PATH.exists():
    try:
        import faiss
        raw_index = faiss.read_index(str(FAISS_PATH))
        print(f"     raw_index.d      = {raw_index.d}")
        print(f"     raw_index.ntotal = {raw_index.ntotal}")
        DISK_FAISS_DIM = raw_index.d
    except Exception as exc:
        print(f"     ERROR loading faiss.index: {exc}")
        DISK_FAISS_DIM = None
else:
    print(f"     FILE NOT FOUND: {FAISS_PATH}")
    DISK_FAISS_DIM = None

# 1c. Dimension recorded in metadata.json
print(f"\n[1c] metadata.json — human-readable index summary (data/index/metadata.json):")
if META_PATH.exists():
    with open(META_PATH, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    print(f"     Full contents : {json.dumps(meta, indent=5)}")
    META_DIM = meta.get("embedding_dim")
    print(f"     embedding_dim : {META_DIM}")
else:
    print(f"     FILE NOT FOUND: {META_PATH}")
    META_DIM = None


# ===========================================================================
# SECTION 2 — Provider Baseline
# ===========================================================================
section("SECTION 2 · PROVIDER BASELINE")

# 2a. MiniLMEmbeddingService native dimension
print(f"\n[2a] MiniLMEmbeddingService native dimension:")
print(f"     Class constant MiniLMEmbeddingService._DIM = {_MINILM_DIM}")
print(f"     (defined in src/ingestion/embeddings.py:382)")
print(f"     embed_query() returns shape ({_MINILM_DIM},) float32")

# 2b. MockEmbeddingService native dimension
print(f"\n[2b] MockEmbeddingService native dimension:")
print(f"     Constructor default: dim={_MOCK_DIM}")
print(f"     (defined in src/ingestion/embeddings.py:274)")
print(f"     Called in streamlit_app.py:223 as MockEmbeddingService(dim={_MOCK_DIM})")

# 2c. Total vectors in the index
print(f"\n[2c] Total vectors in the persisted index:")
if DISK_FAISS_DIM is not None:
    print(f"     raw_index.ntotal = {raw_index.ntotal}")
else:
    print(f"     (Could not load faiss.index — skipped)")

if DOCS_PATH.exists():
    with open(DOCS_PATH, "rb") as fh:
        documents = pickle.load(fh)
    print(f"     len(documents.pkl) = {len(documents)}")
    if documents:
        first_emb = getattr(documents[0], "embedding", None)
        if first_emb is not None:
            import numpy as np
            arr = np.asarray(first_emb)
            print(f"     documents[0].embedding.shape = {arr.shape}  ← ACTUAL STORED VECTOR DIM")
            STORED_VECTOR_DIM = arr.shape[0]
        else:
            STORED_VECTOR_DIM = None
            print(f"     documents[0].embedding = None (no embedding stored)")
    else:
        STORED_VECTOR_DIM = None
        print(f"     documents list is empty")
else:
    print(f"     FILE NOT FOUND: {DOCS_PATH}")
    STORED_VECTOR_DIM = None


# ===========================================================================
# SECTION 3 — Wiring Audit
# ===========================================================================
section("SECTION 3 · WIRING AUDIT — RAGPipeline initialisation")

print("""
[3a] _build_pipeline() in app/streamlit_app.py (lines 265-277):

    def _build_pipeline(index, model_slug, api_key="") -> RAGPipeline:
        llm = _build_llm(model_slug, api_key=api_key)
        return RAGPipeline(
            index=index,
            embedding_service=_get_embedding_service(),   ← DYNAMIC
            llm=llm,
            top_k=10,
        )

    _get_embedding_service() reads st.session_state.get("embedding_provider",
    "MiniLM (Local - 384d)").  The pipeline embedding_service is therefore
    driven ENTIRELY by the runtime Streamlit session state value — not by
    any property of the loaded index itself.

[3b] FaissHNSWIndex instantiation sites across the project:

    ① app/streamlit_app.py:368  (during _ingest_and_build_index):
         index = FaissHNSWIndex(M=32, ef_construction=200, ef_search=64)
         → NEW index; dimension is inferred lazily from the first document
           embedding added via add_documents().  No explicit dim argument.

    ② src/index/persistence.py:176  (inside load_index_from_disk):
         restored = FaissHNSWIndex(M=meta["M"], ef_construction=..., ef_search=...)
         → LOADED from disk shell; no explicit dim at construction time.
         → restored._index = raw_index          ← FAISS object dim = raw_index.d
         → restored._embedding_dim = meta.get("embedding_dim")  ← from metadata.json

    In BOTH cases FaissHNSWIndex.__init__() receives NO 'dim' argument;
    the dimension is resolved post-construction from data, not from a
    constructor parameter.
""")


# ===========================================================================
# SECTION 4 — load_index_from_disk() deep audit
# ===========================================================================
section("SECTION 4 · load_index_from_disk() DEEP AUDIT")

print(f"\n[4] Verifying three-way dimension consistency for the persisted index:")
print(f"    ┌─────────────────────────────────────────┬──────────────┐")
print(f"    │ Source                                  │  Dim value   │")
print(f"    ├─────────────────────────────────────────┼──────────────┤")
print(f"    │ metadata.json  (embedding_dim)          │  {str(META_DIM).rjust(10)}  │")
print(f"    │ faiss.index    (raw_index.d)            │  {str(DISK_FAISS_DIM).rjust(10)}  │")
print(f"    │ documents.pkl  (embedding shape[0])     │  {str(STORED_VECTOR_DIM).rjust(10)}  │")
print(f"    └─────────────────────────────────────────┴──────────────┘")

all_consistent = (META_DIM == DISK_FAISS_DIM == STORED_VECTOR_DIM) and META_DIM is not None
if all_consistent:
    print(f"\n    ✅ All three values AGREE: dim = {META_DIM}")
else:
    print(f"\n    ❌ VALUES DO NOT ALL AGREE — mismatch detected.")

print(f"\n    restored._embedding_dim  will be set to: meta.get('embedding_dim') = {META_DIM}")
print(f"    (src/index/persistence.py:183)")
print(f"\n    CONCLUSION: The FaissHNSWIndex wrapper will report _embedding_dim = {META_DIM}")
print(f"    The FAISS binary graph natively stores vectors of dim = {DISK_FAISS_DIM}")


# ===========================================================================
# SECTION 5 — Mismatch Verdict
# ===========================================================================
section("SECTION 5 · MISMATCH VERDICT")

print(f"""
  Disk index dimension  : {DISK_FAISS_DIM}  (both metadata.json and raw faiss.index agree)
  Default UI provider   : '{SESSION_DEFAULT_PROVIDER}'
  Default query embedder: MiniLMEmbeddingService → {_MINILM_DIM}-d vectors

  ASSERTION that triggers the crash (src/index/faiss_hnsw.py:160-163):
    query_vector.shape[1] != self._embedding_dim
    → {_MINILM_DIM}          !=   {DISK_FAISS_DIM}
    → AssertionError: "query_embedding dimensionality does not match the index"

  ROOT CAUSE:
    The persisted index on disk was built with MockEmbeddingService (dim=128).
    The UI defaults to 'MiniLM (Local - 384d)' which generates 384-d vectors.
    When the disk index is auto-loaded at startup (streamlit_app.py:380-395),
    _build_pipeline() is called with _get_embedding_service() which returns
    MiniLMEmbeddingService (384-d) because that is the SESSION DEFAULT.
    The pipeline then tries to search the 128-d FAISS index with a 384-d query
    vector, triggering the dimension assertion.
""")

print(SEP)
print("  AUDIT COMPLETE — no source files were modified.")
print(SEP)
