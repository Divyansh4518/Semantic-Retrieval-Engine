"""
src/llm/base.py
---------------
Abstract contract for every LLM backend in this system.

All concrete implementations (``MockLLM``, ``OpenRouterLLM``, …) must
subclass ``BaseLLM`` and satisfy the ``generate`` interface.  This guarantees
that callers coded against the abstract type can swap backends transparently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.llm.config import GenerationConfig
from src.llm.response import LLMResponse


class BaseLLM(ABC):
    """
    Abstract base class that defines the minimal contract for all LLM backends.

    Implementors are free to maintain internal state (e.g. API clients,
    thread-pools, caches) inside ``__init__``, but the *only* public method
    every backend must expose is ``generate``.

    Example
    -------
    >>> class MyBackend(BaseLLM):
    ...     def generate(self, prompt: str, config: GenerationConfig | None = None) -> LLMResponse:
    ...         ...
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        """
        Generate a structured response using the provided prompt and configuration.

        Parameters
        ----------
        prompt:
            The input text to send to the language model.
        config:
            Optional generation hyper-parameters.  When ``None``, each backend
            should apply its own defaults (typically ``GenerationConfig()``).

        Returns
        -------
        LLMResponse
            A fully-populated response object carrying both the generated text
            and provider-level telemetry metadata.

        Raises
        ------
        Exception
            Concrete subclasses may raise backend-specific exceptions (e.g.
            network errors, authentication failures).  Callers should be
            prepared to handle them.
        """
