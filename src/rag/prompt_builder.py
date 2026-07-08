"""
src/rag/prompt_builder.py
--------------------------
Context prompt assembly for the RAG pipeline.

``PromptBuilder`` takes retrieved documents and their alignment scores and
assembles a structured, numbered context block for submission to the LLM.
The system instruction appended at the end anchors the model to the provided
context and prevents hallucination outside the retrieved material.
"""

from __future__ import annotations

from pathlib import Path

from src.models import Document


class PromptBuilder:
    """
    Constructs LLM prompts from retrieved documents and a user query.

    The assembled prompt follows a structured format that makes the context
    easy for both humans and models to parse:

    1. A numbered context section with source file, similarity score, and text.
    2. A separator line.
    3. The original user question.
    4. A grounding instruction requiring the model to rely exclusively on the
       provided context or honestly declare when the answer is not available.
    """

    # System instruction appended after all context blocks.
    _SYSTEM_INSTRUCTION: str = (
        "Instructions: Using ONLY the context documents listed above, provide "
        "a comprehensive and accurate answer to the question. "
        "If the answer cannot be found in the provided context, explicitly state: "
        "'The provided context does not contain sufficient information to answer "
        "this question.' Do not speculate or introduce information outside the "
        "given context."
    )

    @staticmethod
    def build_prompt(
        query: str,
        retrieved_documents: list[Document],
        scores: list[float],
    ) -> str:
        """
        Compile a fully structured RAG prompt.

        Parameters
        ----------
        query:
            The user's natural language question.
        retrieved_documents:
            Ordered list of retrieved ``Document`` objects (most relevant first).
        scores:
            Cosine similarity scores aligned positionally with
            ``retrieved_documents``.  Must have the same length.

        Returns
        -------
        str
            A formatted prompt string ready for submission to any ``BaseLLM``
            backend.

        Raises
        ------
        ValueError
            If ``retrieved_documents`` and ``scores`` differ in length.
        """
        if len(retrieved_documents) != len(scores):
            raise ValueError(
                f"retrieved_documents length ({len(retrieved_documents)}) must "
                f"match scores length ({len(scores)})."
            )

        lines: list[str] = ["=== RETRIEVED CONTEXT ===", ""]

        if not retrieved_documents:
            lines.append("(No relevant documents were found in the index.)")
            lines.append("")
        else:
            for idx, (doc, score) in enumerate(
                zip(retrieved_documents, scores), start=1
            ):
                source_name = Path(
                    doc.metadata.get("source_file", doc.id)
                ).name

                lines.append(f"[{idx}]")
                lines.append(f"Source: {source_name}")
                lines.append(f"Score:  {score:.4f}")
                lines.append(f"Text:   {doc.text}")
                lines.append("")

        lines.append("=" * 40)
        lines.append("")
        lines.append(f"Question: {query}")
        lines.append("")
        lines.append(PromptBuilder._SYSTEM_INSTRUCTION)

        return "\n".join(lines)
