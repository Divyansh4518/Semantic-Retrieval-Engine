"""
tests/smoke/phase3_smoke_test.py
---------------------------------
Phase 3 Smoke Test: Streamlit Application Architecture.

Validates:
  1. All app modules import without syntax or runtime errors.
  2. Key symbols are exported correctly from each module.
  3. MODELS dict is correctly structured (friendly->slug mapping).
  4. Session state defaults are complete.
  5. PromptBuilder integration is accessible from the UI layer path.
"""

from __future__ import annotations

import sys
import os
import ast
import importlib
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        print(f"  [FAIL] {message}")
        sys.exit(1)
    print(f"  [PASS] {message}")


def _check_syntax(filepath: str) -> None:
    """Parse the file with ast.parse to catch syntax errors without executing it."""
    with open(filepath, "r", encoding="utf-8") as fh:
        source = fh.read()
    try:
        ast.parse(source)
    except SyntaxError as exc:
        print(f"  [FAIL] Syntax error in '{filepath}': {exc}")
        sys.exit(1)
    print(f"  [PASS] Syntax OK: {os.path.basename(filepath)}")


def run_phase3_smoke_test() -> None:
    print("\n" + "=" * 60)
    print("PHASE 3 SMOKE TEST: Streamlit Application Architecture")
    print("=" * 60)

    project_root = os.path.join(os.path.dirname(__file__), "..", "..")

    # ------------------------------------------------------------------
    # 1. Syntax verification for all app modules (no Streamlit execution)
    # ------------------------------------------------------------------
    print("\n--- Syntax Verification ---")
    app_files = [
        os.path.join(project_root, "app", "__init__.py"),
        os.path.join(project_root, "app", "session_manager.py"),
        os.path.join(project_root, "app", "ui_components.py"),
        os.path.join(project_root, "app", "streamlit_app.py"),
    ]
    for filepath in app_files:
        _assert(os.path.exists(filepath), f"File exists: {os.path.basename(filepath)}")
        _check_syntax(filepath)

    # ------------------------------------------------------------------
    # 2. Import validation for non-Streamlit modules
    # ------------------------------------------------------------------
    print("\n--- Import Validation ---")

    # RAG core modules
    from src.rag.response import RAGResponse, RetrievalResult
    _assert(True, "src.rag.response imports: RAGResponse, RetrievalResult")

    from src.rag.prompt_builder import PromptBuilder
    _assert(True, "src.rag.prompt_builder imports: PromptBuilder")

    from src.rag.pipeline import RAGPipeline
    _assert(True, "src.rag.pipeline imports: RAGPipeline")

    # Index persistence
    from src.index.persistence import save_index_to_disk, load_index_from_disk
    _assert(True, "src.index.persistence imports: save_index_to_disk, load_index_from_disk")

    # ------------------------------------------------------------------
    # 3. MODELS dict structure validation (parsed statically from ui_components.py)
    # ------------------------------------------------------------------
    print("\n--- MODELS Dict Validation (static parse) ---")

    ui_path = os.path.join(project_root, "app", "ui_components.py")
    with open(ui_path, "r", encoding="utf-8") as fh:
        source = fh.read()

    tree = ast.parse(source)

    # Find the MODELS dict assignment (handles both Assign and AnnAssign)
    models_node = None
    for node in ast.walk(tree):
        # plain assignment: MODELS = {...}
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODELS":
                    models_node = node.value
                    break
        # annotated assignment: MODELS: dict[str, str] = {...}
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "MODELS":
                models_node = node.value
        if models_node is not None:
            break

    _assert(models_node is not None, "MODELS dict found in ui_components.py")
    _assert(isinstance(models_node, ast.Dict), "MODELS is a dict literal")

    # Extract key-value pairs statically
    static_models: dict[str, str] = {}
    for k, v in zip(models_node.keys, models_node.values):
        if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
            static_models[k.value] = v.value

    required_slugs = [
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
        "anthropic/claude-sonnet-4",
        "anthropic/claude-opus-4.1",
        "deepseek/deepseek-chat",
        "mock",
    ]
    _assert(len(static_models) == 6, f"MODELS has 6 entries (got {len(static_models)})")
    for slug in required_slugs:
        _assert(slug in static_models.values(), f"Model slug present: '{slug}'")

    _assert("mock" in static_models.values(), "Mock Simulator Mode slug 'mock' present")

    # ------------------------------------------------------------------
    # 4. Session state defaults validation
    # ------------------------------------------------------------------
    print("\n--- Session Manager Defaults Validation ---")

    sm_path = os.path.join(project_root, "app", "session_manager.py")
    with open(sm_path, "r", encoding="utf-8") as fh:
        sm_source = fh.read()
    sm_tree = ast.parse(sm_source)

    # Find _DEFAULTS dict (handles both Assign and AnnAssign)
    defaults_node = None
    for node in ast.walk(sm_tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_DEFAULTS":
                    defaults_node = node.value
                    break
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "_DEFAULTS":
                defaults_node = node.value
        if defaults_node is not None:
            break

    _assert(defaults_node is not None, "_DEFAULTS dict found in session_manager.py")

    required_keys = [
        "chat_history", "active_model", "loaded_index",
        "document_count", "chunk_count", "pipeline",
    ]
    if isinstance(defaults_node, ast.Dict):
        static_keys = [
            k.value for k in defaults_node.keys if isinstance(k, ast.Constant)
        ]
        for key in required_keys:
            _assert(key in static_keys, f"Session key '{key}' in _DEFAULTS")

    # ------------------------------------------------------------------
    # 5. RAG module __init__ exports validation
    # ------------------------------------------------------------------
    print("\n--- RAG Package Export Validation ---")

    from src.rag import RAGPipeline, RAGResponse, RetrievalResult, PromptBuilder
    _assert(True, "src.rag package exports: RAGPipeline, RAGResponse, RetrievalResult, PromptBuilder")

    from src.rag import __all__ as rag_all
    for sym in ["RAGPipeline", "RAGResponse", "RetrievalResult", "PromptBuilder"]:
        _assert(sym in rag_all, f"'{sym}' in src.rag.__all__")

    # ------------------------------------------------------------------
    # 6. UI component function signatures validation
    # ------------------------------------------------------------------
    print("\n--- UI Component Function Signature Validation ---")

    ui_tree = ast.parse(source)
    function_names = {
        node.name
        for node in ast.walk(ui_tree)
        if isinstance(node, ast.FunctionDef)
    }

    for fn_name in ["render_sidebar", "render_chat", "render_metrics", "render_sources"]:
        _assert(fn_name in function_names, f"Function '{fn_name}' defined in ui_components.py")

    print("\n[ALL PASSED] PHASE 3 SMOKE TEST PASSED -- All structural invariants satisfied.\n")


if __name__ == "__main__":
    run_phase3_smoke_test()
