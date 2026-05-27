# 🚀 Vector Search Engine

An AI-native semantic retrieval engine built from scratch in Python.

This project explores the core infrastructure behind modern Retrieval-Augmented Generation (RAG) systems by implementing vector embeddings, cosine similarity search, and Approximate Nearest Neighbor (ANN) retrieval without relying on external vector databases.

Currently implemented:
- Batch embedding pipeline using SentenceTransformers
- Exact cosine similarity search using NumPy
- Modular indexing architecture
- Top-k semantic document retrieval

Planned:
- Graph-based ANN index (NSW/HNSW-inspired)
- FastAPI retrieval API
- Benchmarking suite
- RAG integration

## Example Retrieval

Query:
> "How do I bake bread?"

Top Results:
1. "To get a crispy crust on homemade bread..."
2. "Sourdough bread requires a healthy starter..."
