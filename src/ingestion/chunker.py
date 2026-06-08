"""
src/ingestion/chunker.py
-------------------------
Production-grade hybrid text splitter for the ingestion pipeline.

This module replaces the hand-rolled recursive separator logic with
industry-standard splitters from ``langchain-text-splitters``, while
keeping the public API (``RecursiveTokenChunker``) fully backward-compatible
with the rest of the codebase.

Splitting Strategy
------------------
Two distinct paths depending on the detected file type:

**Markdown path** (``.md`` files or text that contains ATX headers):
    1. ``MarkdownHeaderTextSplitter`` splits the document at header
       boundaries (H1/H2/H3), preserving header context in metadata.
    2. Each resulting fragment is then passed through
       ``RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)``
       so that long sections are further sub-divided while retaining their
       parent header context inside the ``metadata`` blocks.

**Prose / plain-text path** (all other files, including rendered JSON prose):
    ``RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)``
    honours sentence and paragraph boundaries without chopping words
    mid-token — the critical fix for semantic JSON telemetry text.

Both paths emit native ``TextChunk`` staging objects so downstream
components (``MockEmbeddingService``, ``FaissHNSWIndex``) are unaffected.

Public API
----------
RecursiveTokenChunker(chunk_size, overlap, min_chunk_size)
    .chunk_document(text, source_file) -> list[TextChunk]
    .chunk_all(payloads)              -> list[TextChunk]
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.ingestion.models import TextChunk

logger = logging.getLogger(__name__)

# Markdown ATX header detection regex (used to route files to the MD path).
_HEADER_RE = re.compile(r"^#{1,6}\s+.+", re.MULTILINE)

# Header levels that MarkdownHeaderTextSplitter will split on.
_MD_HEADERS: list[tuple[str, str]] = [
    ("#",   "Header 1"),
    ("##",  "Header 2"),
    ("###", "Header 3"),
]


class RecursiveTokenChunker:
    """
    A production-grade, hybrid-strategy text chunker that produces
    ``TextChunk`` instances with clean sentence boundaries and configurable
    size budgets.

    Markdown files are split header-aware first (preserving hierarchy in
    chunk metadata), then sub-divided by character budget.  All other
    content — including semantically rendered benchmark JSON prose — is
    split directly with the recursive character splitter, which respects
    paragraph and sentence boundaries.

    Parameters
    ----------
    chunk_size:
        Target maximum characters per chunk.  Default: 800.
        Markdown sub-splitter uses ``min(chunk_size, 600)`` to keep
        header-scoped chunks compact.
    overlap:
        Characters of overlap between consecutive chunks.  Default: 80.
        Markdown sub-splitter uses ``min(overlap, 50)``.
    min_chunk_size:
        Chunks shorter than this threshold (after stripping whitespace)
        are discarded.  Default: 20.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        overlap: int = 80,
        min_chunk_size: int = 20,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})."
            )
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._min_chunk_size = min_chunk_size

        # ── Markdown pipeline ──────────────────────────────────────────
        # Step 1: split at headers (preserves hierarchy in metadata)
        self._md_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=_MD_HEADERS,
            strip_headers=False,   # keep headers in chunk text for context
        )
        # Step 2: sub-divide oversized header sections by character budget
        md_sub_chunk = min(chunk_size, 600)
        md_sub_overlap = min(overlap, 50)
        self._md_char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=md_sub_chunk,
            chunk_overlap=md_sub_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # ── Prose / plain-text pipeline ────────────────────────────────
        self._prose_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------
    # Public interface (backward-compatible with the original API)
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
            "RecursiveTokenChunker: produced %d total chunk(s) from %d file(s)",
            len(all_chunks),
            len(payloads),
        )
        return all_chunks

    def chunk_document(self, text: str, source_file: str) -> list[TextChunk]:
        """
        Chunk a single document into a list of ``TextChunk`` objects.

        Routing logic
        -------------
        * If ``source_file`` ends with ``.md`` **or** the text contains
          ATX Markdown headers → Markdown-aware split pipeline.
        * Otherwise (plain text, rendered JSON prose, etc.) → prose
          ``RecursiveCharacterTextSplitter`` pipeline.

        Parameters
        ----------
        text:
            Full raw text of the document.
        source_file:
            Used to populate ``TextChunk.source_file`` and derive chunk IDs.
            Also used to detect Markdown files by extension.

        Returns
        -------
        list[TextChunk]
            Ordered list of non-empty chunks for this document.
        """
        if not text.strip():
            return []

        is_markdown = (
            Path(source_file).suffix.lower() == ".md"
            or bool(_HEADER_RE.search(text))
        )

        if is_markdown:
            return self._chunk_markdown(text, source_file)
        return self._chunk_prose(text, source_file)

    # ------------------------------------------------------------------
    # Private — Markdown pipeline
    # ------------------------------------------------------------------

    def _chunk_markdown(self, text: str, source_file: str) -> list[TextChunk]:
        """
        Two-phase Markdown splitting:
          1. MarkdownHeaderTextSplitter → header-aware fragments with metadata.
          2. RecursiveCharacterTextSplitter → sub-divide oversized fragments.
        """
        stem = Path(source_file).stem
        results: list[TextChunk] = []
        global_index = 0

        try:
            md_docs = self._md_header_splitter.split_text(text)
        except Exception as exc:
            logger.warning(
                "MarkdownHeaderTextSplitter failed for %s (%s). "
                "Falling back to prose splitter.",
                source_file, exc,
            )
            return self._chunk_prose(text, source_file)

        for lc_doc in md_docs:
            section_text: str = lc_doc.page_content
            header_meta: dict = lc_doc.metadata  # e.g. {"Header 1": "Intro", ...}

            # Build the human-readable trail for metadata
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
                        },
                    )
                )
                global_index += 1

        return results

    # ------------------------------------------------------------------
    # Private — Prose / plain-text pipeline
    # ------------------------------------------------------------------

    def _chunk_prose(self, text: str, source_file: str) -> list[TextChunk]:
        """
        Single-phase prose splitting with RecursiveCharacterTextSplitter.
        Respects sentence and paragraph boundaries; never chops a word in half.
        """
        stem = Path(source_file).stem
        results: list[TextChunk] = []
        global_index = 0

        pieces = self._prose_splitter.split_text(text)

        for piece in pieces:
            text_stripped = piece.strip()
            if len(text_stripped) < self._min_chunk_size:
                continue

            results.append(
                TextChunk(
                    text=text_stripped,
                    source_file=source_file,
                    chunk_id=f"{stem}:{global_index}",
                    metadata={
                        "header_trail": [],
                        "chunk_index": global_index,
                        "char_count": len(text_stripped),
                    },
                )
            )
            global_index += 1

        return results
