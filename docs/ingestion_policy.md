# Ingestion Pipeline — Chunking Policy

> **Status:** FROZEN — Do NOT revert to a universal text splitter.
> **Effective:** 2026-06-09
> **Owner:** AI Infrastructure Team

---

## Overview

Our ingestion pipeline uses a **file-type-aware smart routing architecture**
implemented in `src/ingestion/chunker.py`. Every incoming document is inspected
by file extension and dispatched to a specialised chunking strategy that
preserves the maximum semantic context for retrieval.

This policy is **mandatory**. Universal "one-size-fits-all" text splitting is
prohibited because it destroys the structural signals (headers, function
boundaries, experiment context) that downstream RAG retrieval depends on.

---

## Chunking Routes

### 1. Benchmark JSON / JSONL (`.json`, `.jsonl`)

| Property          | Value                                   |
| ----------------- | --------------------------------------- |
| **Strategy**      | **Whole-Document** — one file = one chunk |
| **Splitter**      | None (bypass)                           |
| **Rationale**     | The loader renders benchmark data into coherent English prose *before* chunking. Splitting this prose would fragment the experiment context (hyperparameters, metrics, conclusions) across multiple chunks, degrading retrieval quality. |

### 2. Markdown (`.md`)

| Property          | Value                                               |
| ----------------- | --------------------------------------------------- |
| **Strategy**      | **Header-Aware** — split on ATX headers, then guard |
| **Splitter**      | `MarkdownHeaderTextSplitter` (H1/H2/H3) → `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)` |
| **Rationale**     | Markdown sections are semantically meaningful units. Splitting at header boundaries preserves section context. The character guard prevents any single section from exceeding the embedding model's context window. Header metadata (`h1`, `h2`, `h3`) is propagated into each chunk's `header_trail`. |

### 3. Source Code — Python (`.py`)

| Property          | Value                                              |
| ----------------- | -------------------------------------------------- |
| **Strategy**      | **AST / Function-Based**                           |
| **Splitter**      | `RecursiveCharacterTextSplitter.from_language(Language.PYTHON, chunk_size=800, chunk_overlap=100)` |
| **Rationale**     | LangChain's Python-aware splitter respects class and function boundaries, producing chunks that are logically complete units of code. This prevents splitting a function definition across two chunks. |

### 4. Plain Text / Fallback (`.txt`, all others)

| Property          | Value                                              |
| ----------------- | -------------------------------------------------- |
| **Strategy**      | **Paragraph-Based Fallback**                       |
| **Splitter**      | `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)` with separators `["\n\n", "\n", ". ", " ", ""]` |
| **Rationale**     | Prioritises paragraph boundaries (`\n\n`), then line breaks, then sentence-ending periods. Ensures clean, word-boundary-respecting splits for any generic text content. |

---

## Extension Guide

When adding support for a new file type:

1. **Add a new route** in `SmartRepositoryChunker.chunk_document()` before the
   fallback route.
2. **Choose the right splitter** — use LangChain's `Language` enum if the file
   type is a programming language (e.g., `Language.JS`, `Language.JAVA`).
3. **Update this policy** — document the new route, its splitter, and the
   rationale.
4. **Rebuild the index** — purge `data/index/` and re-run the indexing script.

---

## Anti-Patterns (DO NOT)

- ❌ Apply a single `RecursiveCharacterTextSplitter` to all file types.
- ❌ Split JSON/JSONL benchmark files — they must remain atomic.
- ❌ Use character-only splitting on Markdown without header-aware pre-pass.
- ❌ Split Python code without language-aware boundaries.
