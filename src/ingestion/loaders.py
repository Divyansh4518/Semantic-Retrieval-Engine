"""
src/ingestion/loaders.py
-------------------------
Repository file readers for the ingestion pipeline.

``RepositoryLoader`` recursively scans a directory tree and reads every
supported file into a unified ``{relative_path: raw_text}`` mapping.

Supported formats
-----------------
* ``.md``    — Markdown documents (full text preserved; code-block content
               optionally stripped).
* ``.txt``   — Plain-text files read verbatim.
* ``.pdf``   — PDF files extracted to text using ``pypdf``.
* ``.jsonl`` — JSONL files where each line is a JSON object (e.g. benchmark
               sweep outputs). Each record is rendered as a human-readable
               one-line text summary.
* ``.json``  — Single JSON object or array; rendered as formatted text.

Usage
-----
    loader = RepositoryLoader(".")
    payloads = loader.load()
    # payloads == {"README.md": "# My Project\n...", "data.jsonl": "..."}

    # Restrict to Markdown only:
    loader = RepositoryLoader("docs/", extensions={".md"})
    payloads = loader.load()
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# Default set of file extensions the loader will process.
_DEFAULT_EXTENSIONS: frozenset[str] = frozenset({".md", ".txt", ".pdf", ".jsonl", ".json"})

# Regex that matches a fenced Markdown code block (``` ... ```).
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)


class RepositoryLoader:
    """
    Recursively scans a directory and reads every supported text file.

    Parameters
    ----------
    root:
        Path to the directory (or single file) to scan.  Accepts both
        ``str`` and ``pathlib.Path``.
    extensions:
        File extensions to include.  Defaults to ``{".md", ".txt", ".pdf",
        ".jsonl", ".json"}``.  Pass a custom set to restrict or expand coverage.
    strip_code_blocks:
        When ``True`` (default), fenced code blocks are removed from Markdown
        files before the text is returned.  This avoids polluting the chunk
        corpus with raw source code.
    encoding:
        Character encoding for reading files.  Defaults to ``"utf-8"``.
    """

    def __init__(
        self,
        root: str | Path,
        extensions: frozenset[str] | set[str] | None = None,
        strip_code_blocks: bool = True,
        encoding: str = "utf-8",
    ) -> None:
        self._root = Path(root).resolve()
        self._extensions: frozenset[str] = (
            frozenset(extensions) if extensions is not None else _DEFAULT_EXTENSIONS
        )
        self._strip_code_blocks = strip_code_blocks
        self._encoding = encoding

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> dict[str, str]:
        """
        Walk the root path and load every matching file.

        Returns
        -------
        dict[str, str]
            Mapping of ``relative_path_string -> raw_text``.  Paths are
            expressed relative to ``self._root`` so they are portable across
            machines.  Files that cannot be decoded are skipped with a warning.
        """
        payloads: dict[str, str] = {}

        for path in self._iter_files():
            rel = str(path.relative_to(self._root))
            try:
                text = self._read(path)
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("Skipping %s — could not read: %s", rel, exc)
                continue

            if not text.strip():
                logger.debug("Skipping %s — empty after reading.", rel)
                continue

            payloads[rel] = text
            logger.debug("Loaded %s (%d chars)", rel, len(text))

        logger.info(
            "RepositoryLoader: loaded %d files from %s", len(payloads), self._root
        )
        return payloads

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _iter_files(self) -> Iterator[Path]:
        """Yield every file under ``_root`` whose suffix is in ``_extensions``."""
        if self._root.is_file():
            if self._root.suffix.lower() in self._extensions:
                yield self._root
            return

        for path in sorted(self._root.rglob("*")):
            if path.is_file() and path.suffix.lower() in self._extensions:
                yield path

    def _read(self, path: Path) -> str:
        """Dispatch to the correct format reader based on file suffix."""
        suffix = path.suffix.lower()

        if suffix == ".md":
            return self._read_markdown(path)
        if suffix == ".pdf":
            return self._read_pdf(path)
        if suffix == ".jsonl":
            return self._read_jsonl(path)
        if suffix == ".json":
            return self._read_json(path)
        # .txt and any other included extensions → verbatim
        return path.read_text(encoding=self._encoding)

    def _read_markdown(self, path: Path) -> str:
        """
        Read a Markdown file, optionally stripping fenced code blocks.

        Stripping code blocks keeps the chunk corpus focused on prose and
        prevents code tokens (imports, variable names, etc.) from diluting
        semantic search quality.
        """
        text = path.read_text(encoding=self._encoding)
        if self._strip_code_blocks:
            text = _CODE_BLOCK_RE.sub("", text)
        return text

    def _read_pdf(self, path: Path) -> str:
        """Extract text from a PDF file using ``pypdf``."""
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "The 'pypdf' package is required to read PDF files. "
                "Install it with: uv add pypdf"
            ) from exc

        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text)
        return "\n\n".join(pages)

    def _read_jsonl(self, path: Path) -> str:
        """
        Parse a JSONL file and render each record as a human-readable summary.

        The format is tailored to our benchmark sweep outputs where each line
        contains fields like ``query_index``, ``latency_s``, ``recall``,
        ``hyperparameters``, etc.  Unknown fields are still included so the
        loader is forward-compatible with future schema changes.
        """
        lines: list[str] = []

        for lineno, raw in enumerate(
            path.read_text(encoding=self._encoding).splitlines(), start=1
        ):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record: dict = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "%s line %d: JSON parse error — %s", path.name, lineno, exc
                )
                continue
            lines.append(self._render_jsonl_record(record))

        return "\n".join(lines)

    @staticmethod
    def _render_jsonl_record(record: dict) -> str:
        """
        Convert a benchmark record dict into a compact, readable text summary.

        Example output::

            Query 5 | latency=0.0023s | recall=1.000 | nodes_evaluated=64 |
            M=16 ef_construction=64 ef_search=128 N=500 dim=128 |
            git=52c2bd7 seed=42

        Fields present in the benchmark schema are rendered with friendly
        labels; any additional / unknown fields are appended as ``key=value``.
        """
        parts: list[str] = []

        # Known high-value fields with friendly labels
        if "query_index" in record:
            parts.append(f"Query {record['query_index']}")
        if "latency_s" in record:
            parts.append(f"latency={record['latency_s']:.4f}s")
        if "recall" in record:
            parts.append(f"recall={record['recall']:.3f}")
        if "nodes_evaluated" in record:
            parts.append(f"nodes_evaluated={record['nodes_evaluated']}")
        if "entry_node" in record:
            parts.append(f"entry_node={record['entry_node']}")

        # Hyperparameters sub-dict
        hp: dict = record.get("hyperparameters", {})
        if hp:
            hp_str = " ".join(f"{k}={v}" for k, v in hp.items())
            parts.append(f"hyperparams=[{hp_str}]")

        # Remaining top-level scalar fields not yet handled
        _handled = {
            "query_index", "latency_s", "recall", "nodes_evaluated",
            "entry_node", "hyperparameters", "N",
        }
        extras = {k: v for k, v in record.items() if k not in _handled}
        for k, v in extras.items():
            parts.append(f"{k}={v}")

        return " | ".join(parts)

    def _read_json(self, path: Path) -> str:
        """
        Parse a JSON file and return a pretty-printed text representation.

        Arrays are unwrapped and each element rendered on its own line.
        """
        raw = path.read_text(encoding=self._encoding)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("%s: JSON parse error — %s. Returning raw text.", path.name, exc)
            return raw

        if isinstance(obj, list):
            return "\n".join(
                json.dumps(item, ensure_ascii=False) for item in obj
            )
        return json.dumps(obj, indent=2, ensure_ascii=False)
