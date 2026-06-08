"""
src/llm/__init__.py
-------------------
Public API for the telemetry-aware LLM abstraction layer.

Re-exports all key symbols so consumers can do:

    from src.llm import BaseLLM, GenerationConfig, LLMResponse, MockLLM, OpenRouterLLM
"""

from __future__ import annotations

from src.llm.base import BaseLLM
from src.llm.config import GenerationConfig
from src.llm.mock import MockLLM
from src.llm.openrouter import OpenRouterLLM
from src.llm.response import LLMResponse

__all__: list[str] = [
    "BaseLLM",
    "GenerationConfig",
    "LLMResponse",
    "MockLLM",
    "OpenRouterLLM",
]
