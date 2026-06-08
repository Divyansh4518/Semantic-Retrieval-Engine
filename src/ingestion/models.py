"""
src/ingestion/models.py
------------------------
Local staging data structures for the ingestion pipeline.

``TextChunk`` is the intermediate representation that travels through the
pipeline *before* embedding.  Once embedding vectors are attached, chunks are
promoted into native ``src.models.Document`` objects suitable for indexing.

Keeping the two types distinct enforces a clean separation of concerns:

* ``TextChunk``  — pure text + provenance metadata (no vectors, no numpy).
* ``Document``   — text + dense embedding ready for ``BaseIndex.add_documents``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TextChunk:
    """
    A contiguous slice of text extracted from a source file.

    Attributes
    ----------
    text:
        The raw textual content of this chunk.
    source_file:
        Absolute or relative path of the originating file so provenance can
        be traced back after retrieval.
    chunk_id:
        A stable, unique identifier for this chunk.  Convention:
        ``"<stem>:<chunk_index>"`` — e.g. ``"README:0"``, ``"README:1"``.
    metadata:
        Arbitrary key/value pairs for downstream enrichment.  Common fields
        written by the chunker:

        * ``"header_trail"``  — list of Markdown headers that scope the chunk.
        * ``"char_start"``    — character offset of the first character within
          the source file.
        * ``"chunk_index"``   — zero-based position within the document.
    """

    text: str
    source_file: str
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Defensive copy so callers cannot mutate the dict externally.
        self.metadata = dict(self.metadata)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def char_count(self) -> int:
        """Number of characters in this chunk's text."""
        return len(self.text)

    def __repr__(self) -> str:  # pragma: no cover
        snippet = self.text[:60].replace("\n", " ")
        return (
            f"TextChunk(chunk_id={self.chunk_id!r}, "
            f"chars={self.char_count}, text={snippet!r}...)"
        )
