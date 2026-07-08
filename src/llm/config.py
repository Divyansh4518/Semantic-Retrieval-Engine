"""
src/llm/config.py
-----------------
Structured generation-time hyper-parameters for any LLM backend.

Keeping configuration in a dedicated dataclass (rather than ad-hoc **kwargs)
makes call-sites type-safe, self-documenting, and trivially serialisable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class GenerationConfig:
    """
    Encapsulates model generation hyper-parameters.

    All fields carry sensible defaults so callers only need to override what
    they care about:

        cfg = GenerationConfig(temperature=0.7, max_tokens=512)

    Attributes
    ----------
    temperature:
        Sampling temperature in [0, 2].  0.0 → greedy / deterministic output.
    max_tokens:
        Hard ceiling on the number of completion tokens the model may emit.
    top_p:
        Nucleus-sampling probability mass threshold.  1.0 disables nucleus
        filtering (equivalent to top-p off).
    seed:
        Optional integer seed for reproducible sampling.  ``None`` leaves
        seeding up to the provider.
    """

    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    seed: int | None = None
