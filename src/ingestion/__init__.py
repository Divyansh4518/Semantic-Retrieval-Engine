"""
src/ingestion/__init__.py
--------------------------
Public API for the document ingestion, chunking, and embedding pipeline.

Consumers can do:

    from src.ingestion import (
        TextChunk,
        RepositoryLoader,
        SmartRepositoryChunker,
        RecursiveTokenChunker,
        EmbeddingService,
        MockEmbeddingService,
        render_benchmark_to_prose,
        render_benchmark_file,
    )
"""

from __future__ import annotations

from src.ingestion.chunker import RecursiveTokenChunker, SmartRepositoryChunker
from src.ingestion.embeddings import EmbeddingService, MockEmbeddingService, MiniLMEmbeddingService
from src.ingestion.json_renderer import render_benchmark_file, render_benchmark_to_prose
from src.ingestion.loaders import RepositoryLoader
from src.ingestion.models import TextChunk

__all__: list[str] = [
    "TextChunk",
    "RepositoryLoader",
    "SmartRepositoryChunker",
    "RecursiveTokenChunker",
    "EmbeddingService",
    "MockEmbeddingService",
    "MiniLMEmbeddingService",
    "render_benchmark_to_prose",
    "render_benchmark_file",
]
