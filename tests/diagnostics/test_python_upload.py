"""
tests/diagnostics/test_python_upload.py
-----------------------------------------
End-to-end diagnostic verifying that `.py` files are fully supported through
the entire ingestion pipeline: reading → chunking → embedding → indexing.

Run:
    python -m tests.diagnostics.test_python_upload
"""

from __future__ import annotations

import sys
import textwrap
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

# ── Ensure project root is on sys.path ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Mock Python source code (realistic enough to exercise the AST splitter)
# ---------------------------------------------------------------------------

MOCK_PYTHON_SOURCE = textwrap.dedent("""\
    \"\"\"mock_code.py — A sample Python module for ingestion testing.\"\"\"

    from __future__ import annotations

    import math
    from typing import List


    class Vector:
        \"\"\"A simple 2D vector class.\"\"\"

        def __init__(self, x: float, y: float) -> None:
            self.x = x
            self.y = y

        def magnitude(self) -> float:
            \"\"\"Return the Euclidean magnitude of this vector.\"\"\"
            return math.sqrt(self.x ** 2 + self.y ** 2)

        def normalize(self) -> 'Vector':
            \"\"\"Return a unit vector in the same direction.\"\"\"
            mag = self.magnitude()
            if mag == 0:
                raise ValueError("Cannot normalize a zero-length vector.")
            return Vector(self.x / mag, self.y / mag)


    def dot_product(a: Vector, b: Vector) -> float:
        \"\"\"Compute the dot product of two vectors.\"\"\"
        return a.x * b.x + a.y * b.y


    def cosine_similarity(a: Vector, b: Vector) -> float:
        \"\"\"Compute cosine similarity between two vectors.\"\"\"
        denom = a.magnitude() * b.magnitude()
        if denom == 0:
            return 0.0
        return dot_product(a, b) / denom


    def batch_similarities(vectors: List[Vector], query: Vector) -> List[float]:
        \"\"\"Return cosine similarities of each vector against a query.\"\"\"
        return [cosine_similarity(v, query) for v in vectors]
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_uploaded_file(name: str, content: str) -> SimpleNamespace:
    """Create a minimal mock of Streamlit's UploadedFile for testing."""
    raw = content.encode("utf-8")
    buf = BytesIO(raw)
    buf.name = name
    buf.read = buf.read  # noqa: ensure .read() works
    # Reset to start so .read() returns bytes
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Diagnostic test runner
# ---------------------------------------------------------------------------

def run_diagnostics() -> bool:
    """Execute all diagnostic checks. Returns True if all pass."""
    results: list[tuple[str, bool, str]] = []

    # ------------------------------------------------------------------
    # Test 1: _read_uploaded_file_text can read a .py file
    # ------------------------------------------------------------------
    try:
        from app.streamlit_app import _read_uploaded_file_text  # type: ignore[import]

        uploaded = _make_uploaded_file("mock_code.py", MOCK_PYTHON_SOURCE)
        text = _read_uploaded_file_text(uploaded)
        ok = len(text) > 0 and "class Vector" in text
        results.append(("1. _read_uploaded_file_text reads .py", ok,
                         f"{len(text)} chars extracted" if ok else "EMPTY or missing content"))
    except Exception as exc:
        results.append(("1. _read_uploaded_file_text reads .py", False, str(exc)))

    # ------------------------------------------------------------------
    # Test 2: SmartRepositoryChunker receives the text
    # ------------------------------------------------------------------
    try:
        from src.ingestion import SmartRepositoryChunker

        chunker = SmartRepositoryChunker()
        chunks = chunker.chunk_document(text, "mock_code.py")  # type: ignore[possibly-unbound]
        ok = len(chunks) > 0
        results.append(("2. Chunker receives & processes .py text", ok,
                         f"{len(chunks)} chunk(s)" if ok else "NO chunks produced"))
    except Exception as exc:
        results.append(("2. Chunker receives & processes .py text", False, str(exc)))

    # ------------------------------------------------------------------
    # Test 3: Python routing policy is selected
    # ------------------------------------------------------------------
    try:
        routes = {c.metadata.get("route") for c in chunks}  # type: ignore[possibly-unbound]
        ok = "python_ast" in routes
        results.append(("3. Python AST routing policy selected", ok,
                         f"routes={routes}" if ok else f"expected 'python_ast', got {routes}"))
    except Exception as exc:
        results.append(("3. Python AST routing policy selected", False, str(exc)))

    # ------------------------------------------------------------------
    # Test 4: At least one valid chunk is produced
    # ------------------------------------------------------------------
    try:
        first = chunks[0]  # type: ignore[possibly-unbound]
        ok = len(first.text.strip()) >= 20 and first.chunk_id.startswith("mock_code:")
        results.append(("4. Valid chunk produced", ok,
                         f"chunk_id={first.chunk_id!r}, chars={first.char_count}"))
    except Exception as exc:
        results.append(("4. Valid chunk produced", False, str(exc)))

    # ------------------------------------------------------------------
    # Test 5: Metadata is correctly preserved
    # ------------------------------------------------------------------
    try:
        meta = first.metadata  # type: ignore[possibly-unbound]
        ok = (
            "chunk_index" in meta
            and "char_count" in meta
            and "route" in meta
            and meta["route"] == "python_ast"
        )
        results.append(("5. Metadata correctly preserved", ok,
                         f"keys={sorted(meta.keys())}"))
    except Exception as exc:
        results.append(("5. Metadata correctly preserved", False, str(exc)))

    # ------------------------------------------------------------------
    # Test 6: Embeddings generated without dimensionality errors
    # ------------------------------------------------------------------
    try:
        from src.ingestion import MockEmbeddingService

        embed_svc = MockEmbeddingService(dim=128)
        documents = embed_svc.embed_chunks(chunks)  # type: ignore[possibly-unbound]
        ok = (
            len(documents) > 0
            and documents[0].embedding is not None
            and documents[0].embedding.shape == (128,)
        )
        results.append(("6. Embeddings generated (128-d, no errors)", ok,
                         f"{len(documents)} doc(s), shape={documents[0].embedding.shape}" if ok
                         else "embedding failure"))
    except Exception as exc:
        results.append(("6. Embeddings generated (128-d, no errors)", False, str(exc)))

    # ------------------------------------------------------------------
    # Test 7: Document reaches FAISS build simulation
    # ------------------------------------------------------------------
    try:
        from src.index.faiss_hnsw import FaissHNSWIndex

        index = FaissHNSWIndex(M=16, ef_construction=64, ef_search=32)
        index.add_documents(documents)  # type: ignore[possibly-unbound]
        ok = (
            index._embedding_dim == 128
            and len(index._documents) == len(documents)  # type: ignore[possibly-unbound]
        )
        results.append(("7. FAISS index built successfully", ok,
                         f"dim={index._embedding_dim}, docs={len(index._documents)}"))
    except Exception as exc:
        results.append(("7. FAISS index built successfully", False, str(exc)))

    # ------------------------------------------------------------------
    # Bonus: Verify RepositoryLoader allowlist includes .py
    # ------------------------------------------------------------------
    try:
        from src.ingestion.loaders import _DEFAULT_EXTENSIONS

        ok = ".py" in _DEFAULT_EXTENSIONS
        results.append(("8. RepositoryLoader allowlist includes .py", ok,
                         f"extensions={sorted(_DEFAULT_EXTENSIONS)}"))
    except Exception as exc:
        results.append(("8. RepositoryLoader allowlist includes .py", False, str(exc)))

    # ------------------------------------------------------------------
    # Bonus: Verify st.file_uploader type list includes "py"
    # ------------------------------------------------------------------
    try:
        import ast

        ui_path = _PROJECT_ROOT / "app" / "ui_components.py"
        source = ui_path.read_text(encoding="utf-8")
        # Parse and walk the AST to find the file_uploader call's type= kwarg
        tree = ast.parse(source)
        found_py = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "type" and isinstance(kw.value, ast.List):
                        elts = [e.value for e in kw.value.elts  # type: ignore
                                if isinstance(e, (ast.Constant,))]
                        if "py" in elts:
                            found_py = True
        ok = found_py
        results.append(("9. st.file_uploader type list includes 'py'", ok,
                         "verified via AST parse" if ok else "'py' NOT found in type= list"))
    except Exception as exc:
        results.append(("9. st.file_uploader type list includes 'py'", False, str(exc)))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PYTHON UPLOAD DIAGNOSTIC -- PASS / FAIL SUMMARY")
    print("=" * 70)

    all_passed = True
    for label, passed, detail in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {label}")
        print(f"          {detail}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("  ALL CHECKS PASSED -- .py upload pipeline is fully operational.")
    else:
        print("  SOME CHECKS FAILED -- review details above.")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)
