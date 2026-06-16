"""
src/ingestion/chunker.py
-------------------------
Production-grade, file-type-aware smart routing chunker for the ingestion
pipeline.

This module implements a ``SmartRepositoryChunker`` that inspects the file
extension of each incoming document and applies a specialised chunking
strategy:

Routing Table
-------------
+-------------------+-----------------------------------------------------+
| Extension         | Strategy                                            |
+===================+=====================================================+
| ``.json``,        | **Whole-document** — one file = one chunk.  Assumes |
| ``.jsonl``        | the loader has already rendered benchmark data into |
|                   | English prose.                                      |
+-------------------+-----------------------------------------------------+
| ``.md``           | **Header-aware** — ``MarkdownHeaderTextSplitter``   |
|                   | at H1/H2/H3 boundaries, followed by a              |
|                   | ``RecursiveCharacterTextSplitter(1000, 100)``       |
|                   | guard for oversized sections.                       |
+-------------------+-----------------------------------------------------+
| ``.py``           | **AST/function-based** —                            |
|                   | ``RecursiveCharacterTextSplitter.from_language``    |
|                   | with ``Language.PYTHON`` (800, 100).                |
+-------------------+-----------------------------------------------------+
| ``.txt`` / other  | **Paragraph fallback** —                            |
|                   | ``RecursiveCharacterTextSplitter(800, 100)``        |
|                   | prioritising ``\\n\\n`` paragraph boundaries.       |
+-------------------+-----------------------------------------------------+

Public API
----------
SmartRepositoryChunker(min_chunk_size)
    .chunk_document(text, source_file) -> list[TextChunk]
    .chunk_all(payloads)               -> list[TextChunk]

RecursiveTokenChunker
    Backward-compatible alias for ``SmartRepositoryChunker``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.ingestion.models import TextChunk

logger = logging.getLogger(__name__)

# Header levels that MarkdownHeaderTextSplitter will split on.
_MD_HEADERS: list[tuple[str, str]] = [
    ("#",   "h1"),
    ("##",  "h2"),
    ("###", "h3"),
]


class SmartRepositoryChunker:
    """
    A production-grade, file-type-aware smart routing chunker.

    Inspects the file extension of each document and dispatches to the
    optimal splitting strategy.  See the module docstring for the full
    routing table.

    Parameters
    ----------
    min_chunk_size:
        Chunks shorter than this threshold (after stripping whitespace)
        are silently discarded.  Default: 20.
    """

    def __init__(self, min_chunk_size: int = 20, **kwargs) -> None:
        # Accept (and ignore) legacy keyword arguments for backward compat
        # with callers that pass chunk_size= / overlap= to RecursiveTokenChunker.
        self._min_chunk_size = min_chunk_size

        # ── Route 2: Markdown pipeline ────────────────────────────────
        # Step 1 — split at ATX headers (preserves hierarchy in metadata)
        self._md_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_MD_HEADERS,
            strip_headers=False,   # keep headers in chunk text for context
        )
        # Step 2 — sub-divide oversized header sections
        self._md_char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # ── Route 3: Python source-code pipeline ─────────────────────
        self._python_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=800,
            chunk_overlap=100,
        )

        # ── Route 4: Fallback / plain-text pipeline ──────────────────
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chunk_all(self, payloads: dict[str, str]) -> list[TextChunk]:
        """
        Chunk every document in ``payloads`` and return a flat list of chunks.

        Parameters
        ----------
        payloads:
            Mapping of ``source_file -> raw_text`` as produced by
            ``RepositoryLoader.load()`` or assembled manually.

        Returns
        -------
        list[TextChunk]
            All chunks from all documents in document order.
        """
        all_chunks: list[TextChunk] = []
        for source_file, text in payloads.items():
            doc_chunks = self.chunk_document(text, source_file)
            all_chunks.extend(doc_chunks)
            logger.debug("Chunked %s into %d chunk(s)", source_file, len(doc_chunks))

        logger.info(
            "SmartRepositoryChunker: produced %d total chunk(s) from %d file(s)",
            len(all_chunks),
            len(payloads),
        )
        return all_chunks

    def chunk_document(self, text: str, source_file: str) -> List[TextChunk]:
        """
        Route a single document to the optimal chunking strategy based on
        its file extension.

        Parameters
        ----------
        text:
            Full raw text of the document.
        source_file:
            Relative or absolute path — used for extension detection and to
            populate ``TextChunk.source_file``.

        Returns
        -------
        List[TextChunk]
            One or more chunks produced by the selected strategy, or an
            empty list if the document is blank.
        """
        if not text.strip():
            return []

        lower_source = source_file.lower()

        # Route 1 — JSON / JSONL benchmark files: keep as one atomic chunk
        if lower_source.endswith((".json", ".jsonl")):
            return self._chunk_json(text, source_file)

        # Route 2 — Markdown: header-aware + character guard
        if lower_source.endswith(".md"):
            return self._chunk_markdown(text, source_file)

        # Route 3 — Python source code: AST/function-based splitting
        if lower_source.endswith(".py"):
            return self._chunk_python(text, source_file)

        # Route 4 — Fallback (txt, or anything else)
        return self._chunk_fallback(text, source_file)

    # ------------------------------------------------------------------
    # Route 1 — JSON / JSONL (whole-document, single chunk)
    # ------------------------------------------------------------------

    def _chunk_json(self, text: str, source_file: str) -> list[TextChunk]:
        """Return the entire document as one atomic chunk."""
        stem = Path(source_file).stem
        return [
            TextChunk(
                text=text.strip(),
                source_file=source_file,
                chunk_id=f"{stem}:0",
                metadata={
                    "chunk_index": 0,
                    "char_count": len(text.strip()),
                    "route": "json_whole_document",
                },
            )
        ]

    # ------------------------------------------------------------------
    # Route 2 — Markdown (header-aware + character guard)
    # ------------------------------------------------------------------

    def _chunk_markdown(self, text: str, source_file: str) -> list[TextChunk]:
        """
        Two-phase Markdown splitting:
          1. MarkdownHeaderTextSplitter -> header-aware fragments with metadata.
          2. RecursiveCharacterTextSplitter -> sub-divide oversized fragments.
        """
        stem = Path(source_file).stem
        results: list[TextChunk] = []
        global_index = 0

        try:
            md_docs = self._md_header_splitter.split_text(text)
        except Exception as exc:
            logger.warning(
                "MarkdownHeaderTextSplitter failed for %s (%s). "
                "Falling back to paragraph splitter.",
                source_file, exc,
            )
            return self._chunk_fallback(text, source_file)

        for lc_doc in md_docs:
            section_text: str = lc_doc.page_content
            header_meta: dict = lc_doc.metadata  # e.g. {"h1": "Intro", ...}

            # Build a human-readable header trail from metadata
            header_trail = [v for v in header_meta.values() if v]

            # Sub-divide by character budget
            sub_pieces = self._md_char_splitter.split_text(section_text)
            if not sub_pieces:
                sub_pieces = [section_text]

            for piece in sub_pieces:
                text_stripped = piece.strip()
                if len(text_stripped) < self._min_chunk_size:
                    continue

                results.append(
                    TextChunk(
                        text=text_stripped,
                        source_file=source_file,
                        chunk_id=f"{stem}:{global_index}",
                        metadata={
                            "header_trail": header_trail,
                            "header_meta": header_meta,
                            "chunk_index": global_index,
                            "char_count": len(text_stripped),
                            "route": "markdown_header",
                        },
                    )
                )
                global_index += 1

        return results

    # ------------------------------------------------------------------
    # Route 3 — Python source code (AST/function-based)
    # ------------------------------------------------------------------

    def _chunk_python(self, text: str, source_file: str) -> list[TextChunk]:
        """
        Split Python source code using LangChain's Language.PYTHON splitter
        which respects class and function boundaries.
        """
        stem = Path(source_file).stem
        results: list[TextChunk] = []

        pieces = self._python_splitter.split_text(text)

        for idx, piece in enumerate(pieces):
            text_stripped = piece.strip()
            if len(text_stripped) < self._min_chunk_size:
                continue

            results.append(
                TextChunk(
                    text=text_stripped,
                    source_file=source_file,
                    chunk_id=f"{stem}:{idx}",
                    metadata={
                        "header_trail": [],
                        "chunk_index": idx,
                        "char_count": len(text_stripped),
                        "route": "python_ast",
                    },
                )
            )

        return results

    # ------------------------------------------------------------------
    # Route 4 — Fallback / plain-text (paragraph-based)
    # ------------------------------------------------------------------

    def _chunk_fallback(self, text: str, source_file: str) -> list[TextChunk]:
        """
        Paragraph-based splitting with RecursiveCharacterTextSplitter.
        Respects sentence and paragraph boundaries; never chops a word in half.
        """
        stem = Path(source_file).stem
        results: list[TextChunk] = []

        pieces = self._fallback_splitter.split_text(text)

        for idx, piece in enumerate(pieces):
            text_stripped = piece.strip()
            if len(text_stripped) < self._min_chunk_size:
                continue

            results.append(
                TextChunk(
                    text=text_stripped,
                    source_file=source_file,
                    chunk_id=f"{stem}:{idx}",
                    metadata={
                        "header_trail": [],
                        "chunk_index": idx,
                        "char_count": len(text_stripped),
                        "route": "fallback_paragraph",
                    },
                )
            )

        return results


# ======================================================================
# Backward-compatible alias
# ======================================================================

class RecursiveTokenChunker(SmartRepositoryChunker):
    """
    Legacy alias for ``SmartRepositoryChunker``.

    Preserves backward compatibility for all existing call sites that
    instantiate ``RecursiveTokenChunker(chunk_size=..., overlap=...)``.
    The ``chunk_size`` and ``overlap`` parameters are accepted but ignored
    since routing now uses fixed, per-type-optimal budgets.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 80,
        min_chunk_size: int = 20,
    ) -> None:
        super().__init__(min_chunk_size=min_chunk_size)
