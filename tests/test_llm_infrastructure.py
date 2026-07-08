"""
tests/test_llm_infrastructure.py
---------------------------------
Integration test for the telemetry-aware LLM abstraction layer (src/llm/).

Running modes
-------------
1. Standalone script (prints rich output):

       uv run python tests/test_llm_infrastructure.py

2. pytest suite (CI / automated):

       uv run pytest tests/test_llm_infrastructure.py -v

The OpenRouter tests are skipped automatically when OPENROUTER_API_KEY is not
set in the environment, so the suite always passes in offline / CI contexts.
"""

from __future__ import annotations

import os

import pytest

from src.llm.config import GenerationConfig
from src.llm.mock import MockLLM
from src.llm.openrouter import OpenRouterLLM
from src.llm.response import LLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LIVE_SKIP = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set — skipping live OpenRouter tests",
)


# ---------------------------------------------------------------------------
# GenerationConfig tests
# ---------------------------------------------------------------------------


def test_generation_config_defaults() -> None:
    """Defaults must match the documented values exactly."""
    cfg = GenerationConfig()
    assert cfg.temperature == 0.0
    assert cfg.max_tokens == 1024
    assert cfg.top_p == 1.0
    assert cfg.seed is None


def test_generation_config_custom() -> None:
    """All fields should be independently overridable."""
    cfg = GenerationConfig(temperature=0.2, max_tokens=100, top_p=0.9, seed=42)
    assert cfg.temperature == 0.2
    assert cfg.max_tokens == 100
    assert cfg.top_p == 0.9
    assert cfg.seed == 42


# ---------------------------------------------------------------------------
# LLMResponse tests
# ---------------------------------------------------------------------------


def test_llm_response_total_tokens() -> None:
    """total_tokens must sum prompt and completion tokens."""
    resp = LLMResponse(text="hello", model_name="test/model", prompt_tokens=10, completion_tokens=5)
    assert resp.total_tokens == 15


def test_llm_response_total_tokens_none_when_partial() -> None:
    """total_tokens must be None when either component is missing."""
    assert LLMResponse(text="x", model_name="y").total_tokens is None
    assert LLMResponse(text="x", model_name="y", prompt_tokens=5).total_tokens is None
    assert LLMResponse(text="x", model_name="y", completion_tokens=5).total_tokens is None


# ---------------------------------------------------------------------------
# MockLLM tests
# ---------------------------------------------------------------------------


def test_mock_llm_dynamic_response_structure() -> None:
    """Dynamic mode must return a well-formed LLMResponse."""
    cfg = GenerationConfig(temperature=0.2, max_tokens=100, seed=42)
    mock = MockLLM()
    resp = mock.generate("What is HNSW?", cfg)

    assert isinstance(resp, LLMResponse)
    assert "[MOCK RESPONSE]" in resp.text
    assert resp.model_name == "mock/stable-simulator-v1"
    assert isinstance(resp.prompt_tokens, int) and resp.prompt_tokens > 0
    assert isinstance(resp.completion_tokens, int) and resp.completion_tokens > 0


def test_mock_llm_dynamic_prompt_length_in_text() -> None:
    """Dynamic response text must reflect the actual prompt character count."""
    prompt = "What is HNSW?"
    resp = MockLLM().generate(prompt)
    assert f"{len(prompt)} characters" in resp.text


def test_mock_llm_static_mode() -> None:
    """Static mode must return exactly the canned text on every call."""
    canned = "Hierarchical Navigable Small World is an ANN algorithm."
    mock = MockLLM(static_response=canned)
    for _ in range(3):
        resp = mock.generate("What is HNSW?")
        assert resp.text == canned
        assert resp.model_name == "mock/stable-simulator-v1"
        assert resp.prompt_tokens is not None


def test_mock_llm_respects_max_tokens() -> None:
    """completion_tokens must not exceed GenerationConfig.max_tokens."""
    cfg = GenerationConfig(max_tokens=1)
    resp = MockLLM().generate("A" * 1000, cfg)
    assert resp.completion_tokens <= 1


def test_mock_llm_no_config_uses_defaults() -> None:
    """generate() without a config must still succeed using internal defaults."""
    resp = MockLLM().generate("Hello")
    assert resp.completion_tokens is not None
    assert resp.prompt_tokens is not None


# ---------------------------------------------------------------------------
# OpenRouterLLM construction tests (no API key required)
# ---------------------------------------------------------------------------


def test_openrouter_raises_without_api_key() -> None:
    """Constructor must raise ValueError immediately when no key is available."""
    original = os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterLLM()
    finally:
        if original is not None:
            os.environ["OPENROUTER_API_KEY"] = original


def test_openrouter_accepts_explicit_api_key() -> None:
    """
    Constructor must NOT raise when an api_key is passed directly,
    even if the env variable is absent.
    Verified without making a network call — we only check instantiation.
    """
    os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        llm = OpenRouterLLM(api_key="sk-or-test-dummy-key-for-construction")
        assert llm is not None
    except ImportError:
        pytest.skip("openai package not installed")
    finally:
        pass  # env already cleared; nothing to restore


# ---------------------------------------------------------------------------
# OpenRouterLLM live tests (require OPENROUTER_API_KEY)
# ---------------------------------------------------------------------------


@_LIVE_SKIP
def test_openrouter_live_generate_returns_llm_response() -> None:
    """Live call must return a valid LLMResponse with non-empty text."""
    cfg = GenerationConfig(temperature=0.2, max_tokens=100, seed=42)
    llm = OpenRouterLLM(default_model="google/gemini-2.5-flash")
    resp = llm.generate("Respond with only 'ACK'.", cfg)

    assert isinstance(resp, LLMResponse)
    assert resp.text.strip() != ""
    assert resp.model_name != ""


@_LIVE_SKIP
def test_openrouter_live_token_telemetry() -> None:
    """Live call must populate both prompt_tokens and completion_tokens."""
    cfg = GenerationConfig(temperature=0.0, max_tokens=50, seed=1)
    llm = OpenRouterLLM(default_model="google/gemini-2.5-flash")
    resp = llm.generate("Say hello.", cfg)

    assert resp.prompt_tokens is not None and resp.prompt_tokens > 0
    assert resp.completion_tokens is not None and resp.completion_tokens > 0


# ---------------------------------------------------------------------------
# Standalone runner — rich printed output (mirrors the original test script)
# ---------------------------------------------------------------------------


def _run_interactive() -> None:
    config = GenerationConfig(temperature=0.2, max_tokens=100, seed=42)

    print("=" * 60)
    print("  LLM Infrastructure Integration Test")
    print("=" * 60)

    print("\n>>  Testing MockLLM structured telemetry...")
    mock = MockLLM()
    r = mock.generate("What is HNSW?", config)
    print(f"  Model  : {r.model_name}")
    print(f"  Tokens : P={r.prompt_tokens}, C={r.completion_tokens}, Total={r.total_tokens}")
    print(f"  Text   : {r.text}")

    if os.getenv("OPENROUTER_API_KEY"):
        print("\n>>  Testing OpenRouterLLM upstream endpoint...")
        llm = OpenRouterLLM(default_model="google/gemini-2.5-flash")
        r2 = llm.generate("Respond with only 'ACK'.", config)
        print(f"  Model  : {r2.model_name}")
        print(f"  Tokens : P={r2.prompt_tokens}, C={r2.completion_tokens}, Total={r2.total_tokens}")
        print(f"  Text   : {r2.text}")
    else:
        print("\n!!  OPENROUTER_API_KEY not set -- skipping live OpenRouter test.")

    print("\n" + "=" * 60)
    print("  All checks complete.")
    print("=" * 60)


if __name__ == "__main__":
    _run_interactive()
