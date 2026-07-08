"""
src/index/persistence.py
-------------------------
Disk serialization and deserialization for FaissHNSWIndex.

This module provides two public functions for persisting an HNSW index to disk
and reloading it into a fresh instance:

``save_index_to_disk``
    Writes the raw FAISS C++ vector graph to ``faiss.index`` and the Python-side
    document metadata list to ``documents.pkl``, plus a human-readable summary
    to ``metadata.json``.

``load_index_from_disk``
    Reconstructs a fully operational ``FaissHNSWIndex`` from the files written
    by ``save_index_to_disk``, restoring both the C++ vector structure and the
    document identity mapping.

Mechanics
---------
* The FAISS C++ graph is serialized using ``faiss.write_index`` / ``faiss.read_index``
  — the native binary format that preserves the full HNSW topology.
* The ``Document`` list is serialized via ``pickle`` which handles numpy arrays
  in the ``embedding`` field without lossy conversion.
* A supplementary ``metadata.json`` file records the index configuration and
  document count for human inspection without requiring Python.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

import faiss
import numpy as np

from src.index.faiss_hnsw import FaissHNSWIndex
from src.models import Document

logger = logging.getLogger(__name__)

# File names written inside the target directory.
_FAISS_FILE = "faiss.index"
_DOCS_FILE = "documents.pkl"
_META_FILE = "metadata.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_index_to_disk(
    index: FaissHNSWIndex,
    directory: str = "data/index/",
) -> None:
    """
    Serialize a ``FaissHNSWIndex`` to disk.

    Three files are written to ``directory``:

    * ``faiss.index``    — FAISS binary graph (C++ native format).
    * ``documents.pkl``  — Python pickle of the ``list[Document]`` metadata.
    * ``metadata.json``  — Human-readable index configuration summary.

    Parameters
    ----------
    index:
        A populated ``FaissHNSWIndex`` instance.  The function is a no-op if
        the underlying FAISS index has not been initialised (i.e. no documents
        have been added).
    directory:
        Target directory path.  Created automatically if it does not exist.

    Raises
    ------
    ValueError
        If the index has not been initialized (no documents added).
    """
    if index._index is None:
        raise ValueError(
            "Cannot save an uninitialised FaissHNSWIndex.  "
            "Call add_documents() before saving."
        )

    dirpath = Path(directory)
    dirpath.mkdir(parents=True, exist_ok=True)

    # 1. Persist the raw FAISS C++ vector graph
    faiss_path = str(dirpath / _FAISS_FILE)
    faiss.write_index(index._index, faiss_path)
    logger.info("Saved FAISS index to '%s'", faiss_path)

    # 2. Persist the Document metadata list via pickle
    docs_path = dirpath / _DOCS_FILE
    with open(docs_path, "wb") as fh:
        pickle.dump(index._documents, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved %d documents to '%s'", len(index._documents), docs_path)

    # 3. Write a human-readable JSON summary
    meta = {
        "document_count": len(index._documents),
        "embedding_dim": index._embedding_dim,
        "M": index.M,
        "ef_construction": index.ef_construction,
        "ef_search": index.ef_search,
        "faiss_total_vectors": int(index._index.ntotal),
    }
    meta_path = dirpath / _META_FILE
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Saved index metadata to '%s'", meta_path)

    print(
        f"[save_index_to_disk] Saved index: {len(index._documents)} docs, "
        f"dim={index._embedding_dim}, directory='{directory}'"
    )


def load_index_from_disk(
    directory: str = "data/index/",
) -> FaissHNSWIndex:
    """
    Deserialize a ``FaissHNSWIndex`` from disk.

    Reconstructs the full index from files previously written by
    ``save_index_to_disk``.  The returned instance is immediately queryable.

    Parameters
    ----------
    directory:
        Directory containing ``faiss.index``, ``documents.pkl``, and
        ``metadata.json``.

    Returns
    -------
    FaissHNSWIndex
        A fully populated, queryable index instance.

    Raises
    ------
    FileNotFoundError
        If any required file is missing from ``directory``.
    """
    dirpath = Path(directory)
    faiss_path = dirpath / _FAISS_FILE
    docs_path = dirpath / _DOCS_FILE
    meta_path = dirpath / _META_FILE

    for path in (faiss_path, docs_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required index file not found: '{path}'.  "
                f"Run save_index_to_disk() first."
            )

    # 1. Load the human-readable configuration
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta: dict = json.load(fh)

    # 2. Restore the FAISS C++ vector graph
    raw_index = faiss.read_index(str(faiss_path))
    logger.info(
        "Loaded FAISS index from '%s' (%d vectors)", faiss_path, raw_index.ntotal
    )

    # 3. Restore the Document metadata list
    with open(docs_path, "rb") as fh:
        documents: list[Document] = pickle.load(fh)
    logger.info("Loaded %d documents from '%s'", len(documents), docs_path)

    # 4. Reconstruct the Python-side FaissHNSWIndex shell
    restored = FaissHNSWIndex(
        M=meta.get("M", 32),
        ef_construction=meta.get("ef_construction", 200),
        ef_search=meta.get("ef_search", 64),
    )
    restored._index = raw_index
    restored._documents = documents
    restored._embedding_dim = meta.get("embedding_dim")

    # Apply ef_search to the restored FAISS index
    if restored._index is not None:
        restored._index.hnsw.efSearch = restored.ef_search

    print(
        f"[load_index_from_disk] Loaded index: {len(documents)} docs, "
        f"dim={restored._embedding_dim}, directory='{directory}'"
    )
    return restored
