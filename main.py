"""Entry point to test the Day 1 Core Retrieval Engine."""

import time

from src.models import Document
from src.embeddings import EmbeddingPipeline
from src.index.exact import ExactIndex


def main() -> None:
    print("🚀 Initializing Vector Search Engine (Day 1 Prototype)...")

    # 1. Create Sample Data (Mixing cooking, space, and tech)
    raw_data = [
        ("doc_1", "The James Webb Space Telescope captured new images of the Orion Nebula.", {"category": "space"}),
        ("doc_2", "Sourdough bread requires a healthy starter, flour, water, and a long fermentation time.", {"category": "cooking"}),
        ("doc_3", "Quantum computing uses qubits to perform calculations exponentially faster than classical computers.", {"category": "tech"}),
        ("doc_4", "To get a crispy crust on homemade bread, bake it in a preheated cast-iron Dutch oven.", {"category": "cooking"}),
        ("doc_5", "Supermassive black holes have a gravitational pull so strong that not even light can escape.", {"category": "space"}),
    ]

    documents = [
        Document(id=doc_id, text=text, metadata=meta) 
        for doc_id, text, meta in raw_data
    ]

    # 2. Load the Embedding Model
    print("\n[1/4] Loading Embedding Model (all-MiniLM-L6-v2)...")
    pipeline = EmbeddingPipeline()

    # 3. Vectorize the Documents
    print("[2/4] Embedding documents using batch processing...")
    pipeline.embed_documents(documents)

    # 4. Load into the Database
    print("[3/4] Indexing documents into ExactIndex...")
    index = ExactIndex()
    index.add_documents(documents)

    # 5. Execute a Search
    query_text = "How do I bake bread?"
    print(f"\n[4/4] 🔍 Querying the engine: '{query_text}'")
    
    start_time = time.perf_counter()
    query_embedding = pipeline.embed_query(query_text)
    
    # Let's grab the top 2 results
    results = index.search(query_embedding, k=2)
    end_time = time.perf_counter()

    # 6. Display the Output
    print(f"\n✅ Search executed in {(end_time - start_time) * 1000:.2f} ms")
    print("-" * 60)
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"Rank {rank} | Score: {score:.4f} | Category: {doc.metadata.get('category')}")
        print(f"Text: {doc.text}\n")


if __name__ == "__main__":
    main()