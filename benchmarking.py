"""Comprehensive stress-testing and bottleneck analysis script."""
from typing import Type, Any

import time
import numpy as np
from src.models import Document
from src.index.exact import ExactIndex
from src.index.graph import GraphIndex


def generate_synthetic_data(num_docs: int, dim: int = 384) -> list[Document]:
    """Generate high-dimensional synthetic documents with normalized vectors."""
    print(f"📦 Generating {num_docs} synthetic vectors ({dim}-dimensional)...")
    
    # Generate random numbers and normalize them to unit length
    raw_vectors = np.random.randn(num_docs, dim)
    norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
    # Handle potential zero division safely
    normalized_vectors = np.divide(raw_vectors, norms, out=np.zeros_like(raw_vectors), where=norms != 0)

    documents = []
    for i in range(num_docs):
        documents.append(
            Document(
                id=f"syn_{i}",
                text=f"Synthetic document text placeholder for item {i}.",
                metadata={"index_num": i},
                embedding=normalized_vectors[i]
            )
        )
    return documents


def evaluate_index(index_class, name: str, documents: list[Document], query_vector: np.ndarray, k: int = 5):
    """Profile build time and search performance of an index."""
    idx = index_class()
    
    # 1. Profile Build Time (Insertion Latency)
    start_build = time.perf_counter()
    try:
        idx.add_documents(documents)
        build_time = time.perf_counter() - start_build
        print(f"  🟢 [{name}] Build Time: {build_time:.4f} seconds")
        if hasattr(idx, "debug_graph_stats"):
            idx.debug_graph_stats()
    except Exception as e:
        print(f"  🔴 [{name}] Build Failed: {e}")
        return None, None

    # 2. Profile Query Latency
    start_search = time.perf_counter()
    results = idx.search(query_vector, k=k)
    search_time = (time.perf_counter() - start_search) * 1000  # Convert to ms
    print(f"  🔍 [{name}] Query Time: {search_time:.2f} ms")
    
    return results, search_time


def calculate_recall(exact_results, graph_results) -> float:
    """Calculate what percentage of true nearest neighbors the graph index found."""
    if not exact_results or not graph_results:
        return 0.0
    exact_ids = {doc.id for doc, _ in exact_results}
    graph_ids = {doc.id for doc, _ in graph_results}
    matches = exact_ids.intersection(graph_ids)
    return len(matches) / len(exact_ids)


def run_test_tier(tier_name: str, num_docs: int):
    print(f"\n==================== 🔥 RUNNING {tier_name.upper()} TEST ({num_docs} vectors) ====================")
    
    docs = generate_synthetic_data(num_docs)
    # Generate a random search query vector and normalize it
    query = np.random.randn(384)
    query /= np.linalg.norm(query)

    print("\n⚡ Running Baseline (Exact Brute Force)...")
    exact_res, exact_time = evaluate_index(ExactIndex, "ExactIndex", docs, query)

    print("\n⚡ Running Graph Index Traversal...")
    graph_res, graph_time = evaluate_index(GraphIndex, "GraphIndex", docs, query)

    if exact_res and graph_res:
        exact_ids = [doc.id for doc, _ in exact_res]
        graph_ids = [doc.id for doc, _ in graph_res]
        print("\n# === Recall Debug ===")
        print(f"Exact IDs: {exact_ids}")
        print(f"Graph IDs: {graph_ids}")

        recall = calculate_recall(exact_res, graph_res)
        print(f"\n📊 Performance Metrics for {tier_name.upper()}:")
        print(f"  -> Recall Rate (Accuracy vs Ground Truth): {recall * 100:.1f}%")
        if graph_time and exact_time:
            speedup = exact_time / graph_time if graph_time > 0 else 0
            # Note: For ultra-small scales, exact matrix math might beat graph loops due to python overhead
            print(f"  -> Search Latency Comparison: Graph took {graph_time:.2f}ms vs Exact {exact_time:.2f}ms")


if __name__ == "__main__":
    # TIER 1: Simple Test (Verification)
    run_test_tier("Simple", num_docs=100)

    # TIER 2: Medium Test (Real-world scale comparison)
    run_test_tier("Medium", num_docs=1500)

    # TIER 3: The Breakpoint Test (Hostile Scale)
    # WARNING: This tier is explicitly designed to show where our unoptimized Python graph engine fails.
    # It will either take a very long time during the build phase or reveal severe execution bottlenecks.
    run_test_tier("Hard", num_docs=8000)