"""
src/llm/openrouter.py
---------------------
Production-grade LLM backend powered by the OpenRouter gateway.

OpenRouter provides a unified OpenAI-compatible API surface over hundreds of
models from different providers.  This module uses the ``openai`` Python SDK
(≥ 1.x) pointed at the OpenRouter base URL so the implementation remains
idiomatic and benefits from automatic retry logic built into the SDK.

Authentication
--------------
Pass ``api_key`` directly to the constructor, or export the environment
variable ``OPENROUTER_API_KEY`` before your process starts.  If neither is
present a ``ValueError`` is raised at construction time — not at the first
``generate`` call — so misconfiguration surfaces immediately.

Usage
-----
    import os
    from src.llm import OpenRouterLLM, GenerationConfig

    llm = OpenRouterLLM()  # reads OPENROUTER_API_KEY from env

    cfg = GenerationConfig(temperature=0.3, max_tokens=256, seed=42)
    response = llm.generate("Explain HNSW indexing in two sentences.", cfg)

    print(response.text)
    print(f"Tokens used — prompt: {response.prompt_tokens}, "
          f"completion: {response.completion_tokens}")
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from src.llm.base import BaseLLM
from src.llm.config import GenerationConfig
from src.llm.response import LLMResponse

if TYPE_CHECKING:
    import openai  # noqa: F401 — guarded import for type-checking only

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL: str = "google/gemini-2.5-flash"
_REQUEST_TIMEOUT: float = 60.0  # seconds


class OpenRouterLLM(BaseLLM):
    """
    LLM backend that routes completions through `OpenRouter <https://openrouter.ai>`_.

    The backend uses the ``openai`` Python SDK (≥ 1.x) with a custom
    ``base_url`` so all model-routing, auth, and retries are handled
    transparently by the SDK's internals.

    Parameters
    ----------
    default_model:
        The OpenRouter model slug to use when ``generate`` is called without
        an override.  Defaults to ``"google/gemini-2.5-flash"``.
    api_key:
        Your OpenRouter API key.  If ``None``, the constructor reads the
        ``OPENROUTER_API_KEY`` environment variable.  Raises ``ValueError``
        if no key can be found via either path.

    Raises
    ------
    ValueError
        Raised immediately at construction time when no API key is available.
    ImportError
        Raised when the ``openai`` package is not installed.  Install it with
        ``pip install openai``.
    """

    def __init__(
        self,
        default_model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")

        if not resolved_key:
            raise ValueError(
                "No OpenRouter API key found.  "
                "Pass 'api_key' to the constructor or set the "
                "'OPENROUTER_API_KEY' environment variable."
            )

        self._default_model: str = default_model
        self._client = self._build_client(resolved_key)

        logger.debug(
            "OpenRouterLLM initialised with model=%r, base_url=%r",
            self._default_model,
            _OPENROUTER_BASE_URL,
        )

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> LLMResponse:
        """
        Send ``prompt`` to OpenRouter and return a structured ``LLMResponse``.

        The ``GenerationConfig`` fields are mapped 1-to-1 into the OpenAI
        chat-completions payload.  Token usage returned by the API is
        extracted and surfaced in the ``LLMResponse`` telemetry fields.

        Parameters
        ----------
        prompt:
            The user message to send to the model.
        config:
            Optional generation hyper-parameters.  Defaults to
            ``GenerationConfig()`` (deterministic, 1 024-token ceiling).

        Returns
        -------
        LLMResponse
            Populated with the model's reply text and provider-reported token
            usage.

        Raises
        ------
        openai.APITimeoutError
            When the request exceeds ``_REQUEST_TIMEOUT`` seconds.
        openai.APIStatusError
            When the provider returns a non-2xx HTTP status code.  The
            exception message includes the status code and response body.
        openai.APIConnectionError
            When a network-level error prevents the request from reaching the
            provider.
        """
        import openai  # local import so the package is only required at call time

        cfg = config or GenerationConfig()

        # Build the completions payload from the structured config.
        payload: dict = {
            "model": self._default_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "top_p": cfg.top_p,
        }
        if cfg.seed is not None:
            payload["seed"] = cfg.seed

        logger.debug("Dispatching completion request: model=%r", self._default_model)

        try:
            completion = self._client.chat.completions.create(
                **payload,
                timeout=_REQUEST_TIMEOUT,
            )
        except openai.APITimeoutError:
            logger.error(
                "OpenRouter request timed out after %.1f s (model=%r, "
                "prompt_len=%d chars)",
                _REQUEST_TIMEOUT,
                self._default_model,
                len(prompt),
            )
            raise
        except openai.APIStatusError as exc:
            logger.error(
                "OpenRouter returned HTTP %d: %s",
                exc.status_code,
                exc.message,
            )
            raise
        except openai.APIConnectionError:
            logger.error(
                "Failed to reach OpenRouter — check your network connection."
            )
            raise

        return self._parse_response(completion)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_client(api_key: str):  # type: ignore[return]
        """
        Construct an ``openai.OpenAI`` client pointed at the OpenRouter gateway.

        Raises ``ImportError`` with an actionable message if the ``openai``
        package is absent so users know exactly what to install.
        """
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenRouterLLM.  "
                "Install it with:  pip install openai"
            ) from exc

        return openai.OpenAI(
            api_key=api_key,
            base_url=_OPENROUTER_BASE_URL,
        )

    @staticmethod
    def _parse_response(completion) -> LLMResponse:
        """
        Extract the text payload and token-usage telemetry from an API response.

        Both ``usage.prompt_tokens`` and ``usage.completion_tokens`` are read
        defensively — the fields are ``None`` when the provider omits usage
        information rather than raising an ``AttributeError``.

        Parameters
        ----------
        completion:
            A ``openai.types.chat.ChatCompletion`` object returned by the SDK.

        Returns
        -------
        LLMResponse
            Fully populated carrier with text and telemetry fields.
        """
        text: str = completion.choices[0].message.content or ""
        model_name: str = completion.model or "unknown"

        usage = getattr(completion, "usage", None)
        prompt_tokens: int | None = getattr(usage, "prompt_tokens", None)
        completion_tokens: int | None = getattr(usage, "completion_tokens", None)

        logger.debug(
            "Received completion: model=%r, prompt_tokens=%s, completion_tokens=%s",
            model_name,
            prompt_tokens,
            completion_tokens,
        )

        return LLMResponse(
            text=text,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
