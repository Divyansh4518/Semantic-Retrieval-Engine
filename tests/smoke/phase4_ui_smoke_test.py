"""
tests/smoke/phase4_ui_smoke_test.py
------------------------------------
Validation sweep for Phase 1-4 UI updates.

Checks:
  1. New session state keys (openrouter_api_key, available_models, etc.)
  2. fetch_openrouter_models() returns a dict and includes 'mock' on failure
  3. _filter_free_models() only passes :free slugs + mock
  4. render_rag_answer / _render_rag_telemetry / render_rag_answer are all defined
  5. Phase 3: _build_llm logic — reads UI key, falls back to MockLLM
  6. Phase 4: 4-pillar function symbols exist in ui_components
  7. FALLBACK_MODELS exported from session_manager
  8. fetch_openrouter_models graceful fallback on bad URL
  9. API key in sidebar cfg return dict
  10. sidebar_cfg return dict contains 'api_key' key
"""

from __future__ import annotations

import ast
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _assert(condition: bool, message: str) -> None:
    if not condition:
        print(f"  [FAIL] {message}")
        sys.exit(1)
    print(f"  [PASS] {message}")


def run_phase4_ui_smoke_test() -> None:
    print("\n" + "=" * 60)
    print("PHASE 4 UI SMOKE TEST: Phase 1-4 UI Update Validation")
    print("=" * 60)

    project_root = os.path.join(os.path.dirname(__file__), "..", "..")

    # ------------------------------------------------------------------
    # 1. Syntax verification
    # ------------------------------------------------------------------
    print("\n--- Syntax Verification ---")
    for fname in ["session_manager.py", "ui_components.py", "streamlit_app.py"]:
        fpath = os.path.join(project_root, "app", fname)
        _assert(os.path.exists(fpath), f"File exists: {fname}")
        with open(fpath, "r", encoding="utf-8") as fh:
            source = fh.read()
        try:
            ast.parse(source)
            print(f"  [PASS] Syntax OK: {fname}")
        except SyntaxError as e:
            print(f"  [FAIL] Syntax error in {fname}: {e}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # 2. New session state keys in _DEFAULTS
    # ------------------------------------------------------------------
    print("\n--- Session Manager New Keys ---")
    from app.session_manager import FALLBACK_MODELS

    sm_path = os.path.join(project_root, "app", "session_manager.py")
    with open(sm_path, "r", encoding="utf-8") as fh:
        sm_source = fh.read()
    sm_tree = ast.parse(sm_source)

    defaults_node = None
    for node in ast.walk(sm_tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "_DEFAULTS":
                    defaults_node = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "_DEFAULTS":
                defaults_node = node.value
        if defaults_node is not None:
            break

    _assert(defaults_node is not None, "_DEFAULTS found in session_manager.py")

    new_keys = ["openrouter_api_key", "available_models", "models_fetched", "last_rag_response"]
    if isinstance(defaults_node, ast.Dict):
        static_keys = [k.value for k in defaults_node.keys if isinstance(k, ast.Constant)]
        for key in new_keys:
            _assert(key in static_keys, f"New session key '{key}' in _DEFAULTS")

    _assert(
        isinstance(FALLBACK_MODELS, dict),
        "FALLBACK_MODELS exported from session_manager as dict",
    )
    _assert("mock" in FALLBACK_MODELS.values(), "FALLBACK_MODELS contains 'mock' slug")
    _assert(
        len(FALLBACK_MODELS) >= 2,
        f"FALLBACK_MODELS has at least 2 entries (got {len(FALLBACK_MODELS)})",
    )

    # ------------------------------------------------------------------
    # 3. fetch_openrouter_models fallback behaviour
    # ------------------------------------------------------------------
    print("\n--- fetch_openrouter_models Fallback ---")
    from app.ui_components import fetch_openrouter_models, _filter_free_models

    # Offline call — must not raise; must return dict with 'mock'
    result = fetch_openrouter_models(api_key="")
    _assert(isinstance(result, dict), "fetch_openrouter_models returns dict")
    _assert(len(result) >= 2, f"Returns >= 2 entries (got {len(result)})")
    _assert("mock" in result.values(), "Result always contains 'mock' slug")

    # Bad key call — must also not raise
    result_bad = fetch_openrouter_models(api_key="sk-invalid-key-test")
    _assert(isinstance(result_bad, dict), "fetch_openrouter_models with bad key returns dict")
    _assert("mock" in result_bad.values(), "Bad-key result contains 'mock'")

    # ------------------------------------------------------------------
    # 4. _filter_free_models
    # ------------------------------------------------------------------
    print("\n--- _filter_free_models ---")
    mixed_models = {
        "Free Gemini": "google/gemini-flash-1.5:free",
        "Paid Model": "openai/gpt-4o",
        "Another Free": "meta-llama/llama-3:free",
        "Mock": "mock",
    }
    free_only = _filter_free_models(mixed_models)
    _assert(isinstance(free_only, dict), "_filter_free_models returns dict")
    _assert(
        all(v.endswith(":free") or v == "mock" for v in free_only.values()),
        "All slugs in filtered result end with :free or are 'mock'",
    )
    _assert("mock" in free_only.values(), "Filtered result always contains 'mock'")

    # Edge: all-paid dict (only mock remains — function falls back to MODELS)
    all_paid = {"Paid A": "openai/gpt-4o", "Paid B": "anthropic/claude-3"}
    paid_result = _filter_free_models(all_paid)
    _assert(
        "mock" in paid_result.values(),
        "_filter_free_models with no free models still provides 'mock'",
    )

    # ------------------------------------------------------------------
    # 5. New UI component functions exist
    # ------------------------------------------------------------------
    print("\n--- UI Component New Functions ---")
    ui_path = os.path.join(project_root, "app", "ui_components.py")
    with open(ui_path, "r", encoding="utf-8") as fh:
        ui_source = fh.read()
    ui_tree = ast.parse(ui_source)

    fn_names = {n.name for n in ast.walk(ui_tree) if isinstance(n, ast.FunctionDef)}
    new_fns = [
        "fetch_openrouter_models",
        "_filter_free_models",
        "_run_connection_test",
        "render_rag_answer",
        "_render_rag_telemetry",
        "render_sources_inline",
    ]
    for fn in new_fns:
        _assert(fn in fn_names, f"Function '{fn}' defined in ui_components.py")

    # ------------------------------------------------------------------
    # 6. sidebar cfg return dict contains 'api_key'
    # ------------------------------------------------------------------
    print("\n--- Sidebar Return Dict Structure ---")
    # Parse statically: find the return dict in render_sidebar
    return_keys = []
    for node in ast.walk(ui_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "render_sidebar":
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
                    return_keys = [
                        k.value
                        for k in child.value.keys
                        if isinstance(k, ast.Constant)
                    ]
            break

    _assert("api_key" in return_keys, "'api_key' in render_sidebar return dict")
    _assert("selected_model_slug" in return_keys, "'selected_model_slug' in return dict")
    _assert("temperature" in return_keys, "'temperature' in return dict")
    _assert("max_tokens" in return_keys, "'max_tokens' in return dict")
    _assert("uploaded_files" in return_keys, "'uploaded_files' in return dict")
    _assert("rebuild_clicked" in return_keys, "'rebuild_clicked' in return dict")

    # ------------------------------------------------------------------
    # 7. streamlit_app.py uses ui_api_key in _build_pipeline calls
    # ------------------------------------------------------------------
    print("\n--- streamlit_app.py Phase 3 Wiring ---")
    app_path = os.path.join(project_root, "app", "streamlit_app.py")
    with open(app_path, "r", encoding="utf-8") as fh:
        app_source = fh.read()

    _assert(
        "ui_api_key" in app_source,
        "streamlit_app.py references 'ui_api_key' variable",
    )
    _assert(
        "api_key=ui_api_key" in app_source,
        "streamlit_app.py passes api_key=ui_api_key to _build_pipeline",
    )
    _assert(
        "render_rag_answer" in app_source,
        "streamlit_app.py calls render_rag_answer for Phase 4 telemetry",
    )
    _assert(
        "rag_response=rag_resp" in app_source,
        "streamlit_app.py stores rag_response in add_message call",
    )
    _assert(
        "OpenRouterLLM" in app_source,
        "streamlit_app.py imports/references OpenRouterLLM",
    )
    _assert(
        "default_model=model_slug" in app_source,
        "OpenRouterLLM called with correct 'default_model=' kwarg",
    )

    # ------------------------------------------------------------------
    # 8. Phase 4 context expander present in ui_components
    # ------------------------------------------------------------------
    print("\n--- Phase 4 Context Expander ---")
    _assert(
        "View Constructed Context Block" in ui_source,
        "'View Constructed Context Block' expander label present",
    )
    _assert(
        "st.expander" in ui_source,
        "st.expander used for context display",
    )
    _assert(
        "rag_response.context" in ui_source,
        "rag_response.context displayed in telemetry",
    )
    _assert(
        "prompt_tokens" in ui_source,
        "prompt_tokens referenced in token caption",
    )
    _assert(
        "completion_tokens" in ui_source,
        "completion_tokens referenced in token caption",
    )

    print(
        "\n[ALL PASSED] PHASE 4 UI SMOKE TEST PASSED "
        "-- All Phase 1-4 invariants satisfied.\n"
    )


if __name__ == "__main__":
    run_phase4_ui_smoke_test()
