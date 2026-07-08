"""
src/llm/mock.py
---------------
Zero-cost, zero-network LLM simulator for local development and unit tests.

``MockLLM`` satisfies the ``BaseLLM`` contract without making any network
calls.  It is the recommended backend for:

* Unit and integration tests that should not incur API costs.
* Local development where no API key is available.
* Benchmarking the surrounding infrastructure independent of model latency.

Usage
-----
    # Dynamic mode — mirrors a real payload, tokens derived from prompt length:
    llm = MockLLM()
    response = llm.generate("What is FAISS?")

    # Static mode — always returns the same canned text:
    llm = MockLLM(static_response="This is a fixed answer.")
    response = llm.generate("Anything at all")
    assert response.text == "This is a fixed answer."
"""

from __future__ import annotations

from src.llm.base import BaseLLM
from src.llm.config import GenerationConfig
from src.llm.response import LLMResponse

# Rough approximation: 1 token ≈ 4 characters (GPT-tokeniser heuristic).
_CHARS_PER_TOKEN: int = 4

_MOCK_MODEL_NAME: str = "mock/stable-simulator-v1"


class MockLLM(BaseLLM):
    """
    A deterministic, in-process LLM backend for testing and local development.

    Parameters
    ----------
    static_response:
        When provided, *every* call to ``generate`` returns this exact text
        string wrapped in an ``LLMResponse``.  When ``None`` (default), each
        call constructs a dynamic response whose content reflects the actual
        input prompt.
    """

    def __init__(self, static_response: str | None = None) -> None:
        self._static_response: str | None = static_response

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        """
        Return a simulated completion without any network I/O.

        In *static* mode the ``text`` field is always ``self._static_response``.
        In *dynamic* mode the response mirrors a realistic production payload:

        * ``text`` encodes the prompt length so tests can assert on it.
        * Token counts are calculated from character-length heuristics so
          downstream telemetry pipelines receive valid integers to process.

        Parameters
        ----------
        prompt:
            Input text — inspected in dynamic mode to compute simulated tokens.
        config:
            Ignored in the mock but accepted for interface compatibility.
            Defaults to ``GenerationConfig()`` when ``None``.

        Returns
        -------
        LLMResponse
            A fully-populated response matching the contract of a real backend.
        """
        _cfg = config or GenerationConfig()

        if self._static_response is not None:
            return self._build_static_response(prompt)

        return self._build_dynamic_response(prompt, _cfg)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_static_response(self, prompt: str) -> LLMResponse:
        """Wrap the pre-configured static text in a telemetry-ready LLMResponse."""
        assert self._static_response is not None  # narrowing for type-checker

        prompt_tokens = max(1, len(prompt) // _CHARS_PER_TOKEN)
        completion_tokens = max(1, len(self._static_response) // _CHARS_PER_TOKEN)

        return LLMResponse(
            text=self._static_response,
            model_name=_MOCK_MODEL_NAME,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _build_dynamic_response(
        self,
        prompt: str,
        config: GenerationConfig,
    ) -> LLMResponse:
        """
        Construct a dynamic response that reflects real prompt characteristics.

        Token counts are capped by ``config.max_tokens`` so the simulated
        payload is consistent with what a real provider would return.
        """
        text = (
            f"[MOCK RESPONSE] Captured prompt length: {len(prompt)} characters."
            " System is nominal."
        )

        prompt_tokens = max(1, len(prompt) // _CHARS_PER_TOKEN)
        # Completion tokens: use the generated text length, but respect the
        # configured ceiling so the mock does not misrepresent capacity.
        raw_completion_tokens = max(1, len(text) // _CHARS_PER_TOKEN)
        completion_tokens = min(raw_completion_tokens, config.max_tokens)

        return LLMResponse(
            text=text,
            model_name=_MOCK_MODEL_NAME,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
