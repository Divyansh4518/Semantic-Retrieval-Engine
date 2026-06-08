"""
src/llm/response.py
-------------------
Structured carrier for LLM completions.

``LLMResponse`` is the single canonical return type for every backend
implementation.  Capturing token-usage metadata alongside the text payload
enables upstream telemetry, cost-tracking, and rate-limit monitoring without
callers needing to parse raw provider payloads.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LLMResponse:
    """
    A provider-agnostic completion response enriched with telemetry metadata.

    Attributes
    ----------
    text:
        The raw text content returned by the model.
    model_name:
        The canonical model identifier as reported by (or inferred from) the
        provider — e.g. ``"google/gemini-2.5-flash"`` or
        ``"mock/stable-simulator-v1"``.
    prompt_tokens:
        Number of tokens consumed by the input prompt.  ``None`` when the
        provider does not surface usage data.
    completion_tokens:
        Number of tokens generated in the completion.  ``None`` when the
        provider does not surface usage data.
    """

    text: str
    model_name: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def total_tokens(self) -> int | None:
        """
        Sum of prompt and completion tokens.

        Returns ``None`` if either component is unavailable so callers get a
        clean sentinel rather than a silently incorrect integer.
        """
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens
