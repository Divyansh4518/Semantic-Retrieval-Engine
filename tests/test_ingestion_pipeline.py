"""
tests/test_ingestion_pipeline.py
---------------------------------
End-to-end offline integration test for the ingestion pipeline.

Running modes
-------------
1. pytest suite (CI / automated):
       uv run pytest tests/test_ingestion_pipeline.py -v -s

2. Standalone script (prints rich output):
       uv run python tests/test_ingestion_pipeline.py
"""
from __future__ import annotations

from src.ingestion.loaders import RepositoryLoader
from src.ingestion.chunker import RecursiveTokenChunker
from src.ingestion.embeddings import MockEmbeddingService


def test_ingestion() -> None:
    print("\n📁 Scanning repository files...")
    # Load all markdown files in the root directory
    loader = RepositoryLoader(".", extensions={".md"})
    payloads = loader.load()
    
    # Target our root README.md
    readme_payloads = {k: v for k, v in payloads.items() if k == "README.md"}
    assert readme_payloads, "README.md not found in the root directory"
    
    print("✂️ Chunking text structurally...")
    chunker = RecursiveTokenChunker(chunk_size=1000, overlap=100)
    chunks = chunker.chunk_all(readme_payloads)
    print(f"Generated {len(chunks)} structural text chunks.")
    
    print("🧬 Vectorizing chunks via MockEmbeddingService...")
    embedder = MockEmbeddingService(dim=128)
    vectorized_docs = embedder.embed_chunks(chunks)
    
    print("\n✅ INVARIANT VERIFICATION:")
    print(f"- Total Documents Built: {len(vectorized_docs)}")
    print(f"- Document 0 ID: {vectorized_docs[0].id}")
    print(f"- Document 0 Vector Dimensions: {len(vectorized_docs[0].embedding)}")
    print(f"- Document 0 Snippet: {vectorized_docs[0].text[:60].replace('\n', ' ')}...")
    
    assert len(vectorized_docs) > 0, "No documents were generated!"
    assert len(vectorized_docs[0].embedding) == 128, "Vector dimensions do not match index expectations!"


if __name__ == "__main__":
    # Workaround for Windows terminals that default to CP-1252 encoding when printing emojis
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    test_ingestion()
