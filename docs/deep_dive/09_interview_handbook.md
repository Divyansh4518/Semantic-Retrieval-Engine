# 09 — Interview Handbook

> **Purpose:** 50+ interview questions derived from the codebase, ranging from
> foundational concepts to deep systems engineering. Organised by difficulty
> tier and topic domain.

---

## Tier 1 — Foundational (Junior / Mid-Level)

### Data Models & Python

**Q1.** Why does the system use two separate dataclasses (`TextChunk` and `Document`) instead of a single class? What invariant does this boundary enforce?

**Q2.** Explain why `TextChunk` does not have an `id` field, but `Document` does. When and where is the `id` generated?

**Q3.** What does `@dataclass(slots=True)` do on `GenerationConfig` and `LLMResponse`? What advantage does it provide over a regular dataclass?

**Q4.** Why does `LLMResponse.total_tokens` return `None` instead of `0` when either component token count is missing? What bug would returning `0` introduce?

**Q5.** The `Document.embedding` field stores a raw `np.ndarray` rather than a serialised representation. Name two advantages of this choice.

### Ingestion Pipeline

**Q6.** Describe the four chunking strategies implemented in `SmartRepositoryChunker`. For each, name the file extension it handles and the rationale for the strategy choice.

**Q7.** Why are `.json` and `.jsonl` benchmark files converted to English prose *before* being chunked? What would happen to retrieval quality if they were chunked as raw JSON?

**Q8.** The `RepositoryLoader` has an encoding fallback: `raw_bytes.decode("utf-8")` first, then `raw_bytes.decode("latin-1", errors="replace")`. Why is this necessary?

**Q9.** What is the purpose of the `chunk_overlap` parameter in `RecursiveCharacterTextSplitter`? What problem does it solve?

**Q10.** The `MiniLMEmbeddingService` uses `SentenceTransformer` with a lazy import inside `__init__`. Why not import at module level?

### Indexing

**Q11.** Explain the "L2-normalise then use Inner Product" trick used by both FAISS indices. Why does this make Inner Product equivalent to Cosine Similarity?

**Q12.** What does FAISS return as the index value when `k` exceeds the number of indexed vectors? How does `FaissHNSWIndex.search()` handle this?

**Q13.** The `ExactIndex` uses `np.maximum(norms, 1e-10)` during normalisation. What specific failure does this guard prevent?

**Q14.** What are the three files produced by `save_index_to_disk()`? What does each file contain?

---

## Tier 2 — Intermediate (Mid-Level / Senior)

### Algorithm Analysis

**Q15.** The `GraphIndex` has an O(N²) build time. Trace through the code and explain exactly *why* it's quadratic. What specific operation causes it?

**Q16.** Compare the search complexity of `ExactIndex` (O(N·d)) vs `FaissHNSWIndex` (O(log(N)·ef_search)). At what dataset size does the logarithmic advantage overcome the constant-factor overhead? Cite the empirical finding from Sweep H.

**Q17.** Sweep B shows that at M=2, the graph fractures into 1,136 components with 0.3% recall. Explain the *mechanism* by which low M causes graph fragmentation.

**Q18.** Why does HNSW Recall@10 decrease from 93.6% to 60.2% as N scales from 5,000 to 50,000 with a fixed `ef_search=64`? Express the answer in terms of the fraction of the corpus explored.

**Q19.** Sweep C allows mutating `ef_search` in-place without rebuilding the index. Why is this optimisation correct — i.e., why does changing `ef_search` not invalidate the graph structure?

### Systems Engineering

**Q20.** The benchmarks show that `ExactIndex` and `FaissFlatIndex` return identical results (`match_rate = 1.0`) but FAISS is 18–24x faster. List three specific systems-level reasons for this speedup.

**Q21.** Sweep G shows FAISS batched throughput jumping to 10,117 QPS (vs 681 QPS sequential). Explain *why* batching provides such dramatic improvement from a hardware perspective.

**Q22.** The `capture_search_diagnostics()` function uses `contextlib.redirect_stdout()` to capture GraphIndex debug output. What is the advantage of this approach over adding a return value or callback parameter to `search()`?

**Q23.** The `measure_memory()` function uses a background thread polling at 10ms intervals. Why might this miss the true peak RSS? What alternative approach would be more accurate?

**Q24.** The persistence module uses `pickle.dump()` for `documents.pkl`. Name two security risks of using pickle for serialisation and suggest an alternative.

### RAG Pipeline

**Q25.** The `RAGPipeline.query()` method does not short-circuit when `search_results` is empty. It still sends the prompt to the LLM. Why is this the correct behaviour?

**Q26.** The `PromptBuilder` includes a grounding instruction: "Using ONLY the context documents listed above..." Why is this instruction positioned *after* the context and question, not before?

**Q27.** The RAGPipeline stores both `context` (the full prompt) and `answer` (the LLM output) in the `RAGResponse`. Why store the full context if the user only sees the answer?

**Q28.** The Streamlit app caches the MiniLM embedding service as a process-level singleton. What would happen if it were stored in `st.session_state` instead?

### Architecture

**Q29.** The system uses constructor injection to swap backends (e.g., `MockLLM` vs `OpenRouterLLM`). Compare this to an alternative approach using a factory pattern with configuration-driven instantiation. What are the trade-offs?

**Q30.** The `openai` package is imported lazily inside `OpenRouterLLM.generate()`. What problem does this solve for users who only use the `MockLLM` backend?

**Q31.** Why does the Streamlit upload path use `RecursiveTokenChunker` (simple) instead of `SmartRepositoryChunker` (file-type-aware)? Is this a bug or a deliberate design choice?

---

## Tier 3 — Advanced (Senior / Staff)

### Scaling & Performance

**Q32.** The README identifies the HNSW crossover point at N≈10,000 and attributes it to the CPU L3 cache boundary. Evaluate this hypothesis: is it the only explanation, or could other factors contribute?

**Q33.** At N=50,000, HNSW's p95 latency (1.329ms) is significantly lower than Flat's p95 (2.598ms). Explain why HNSW's tail latency is more *predictable* than Flat's, using the algorithmic structure of each approach.

**Q34.** Design a strategy to maintain >90% Recall@10 as N scales to 1M vectors with the existing `FaissHNSWIndex` implementation. What parameters would you tune, and what trade-offs would you accept?

**Q35.** The benchmarking suite generates synthetic data using `np.random.default_rng()`. Critique this approach: how might synthetic data distributions differ from real-world embedding distributions, and what impact would this have on benchmark validity?

### Architecture & Design

**Q36.** The system stores all `Document` objects (including text and metadata) in `documents.pkl` alongside the FAISS index. At N=1M documents, this would consume significant memory. Design an alternative persistence architecture that supports lazy loading.

**Q37.** The `RAGPipeline` is stateless per-query with no conversation history. Design an extension that supports multi-turn conversation while preserving the current single-query interface. What state needs to be maintained?

**Q38.** The current system embeds queries and documents with the *same* model (MiniLM). Some production systems use separate query and document encoders (bi-encoder architecture). What advantage would this provide, and what complexity would it add to this codebase?

**Q39.** The Streamlit app uses `st.session_state` as its only state store. What happens when the Streamlit server restarts? Design a persistence strategy that survives server restarts without a database.

**Q40.** The benchmarking suite captures diagnostics via stdout redirection. Design an alternative instrumentation approach that provides structured telemetry without modifying the production code and without relying on stdout.

### Information Retrieval Theory

**Q41.** The system uses cosine similarity as its sole ranking signal. In what scenarios would BM25 (term-based scoring) outperform dense vector retrieval? Design a hybrid retrieval strategy that combines both.

**Q42.** The `PromptBuilder` includes similarity scores in the prompt. Evaluate whether this actually helps the LLM produce better answers, or whether it's purely informational. Design an experiment to test this.

**Q43.** The current system uses a fixed `top_k=5` or `top_k=10` for retrieval. Design an adaptive retrieval strategy that dynamically adjusts the number of retrieved documents based on the query and the score distribution.

**Q44.** The JSON-to-prose rendering approach converts structured data into natural language before embedding. Compare this to an alternative: embedding the JSON directly with a model fine-tuned on structured data. What are the trade-offs?

### Production Engineering

**Q45.** The system has no rate limiting on the OpenRouter API calls. Design a rate-limiting and retry strategy that handles API quotas, transient errors, and backpressure.

**Q46.** The current FAISS index must be rebuilt from scratch when new documents are added. Design an incremental indexing strategy that supports online updates without full rebuilds.

**Q47.** The `documents.pkl` file uses Python pickle, which is version-sensitive and insecure. Design a migration path to a safer serialisation format while maintaining backward compatibility with existing persisted indices.

**Q48.** The Streamlit app runs as a single-process server. Design a deployment architecture that supports multiple concurrent users with shared index state.

**Q49.** The system currently has no observability beyond the benchmark suite. Design a production monitoring strategy that tracks query latency, recall, token usage, and error rates in real-time.

**Q50.** The benchmarking suite runs all sweeps sequentially. Design a parallelisation strategy that can run independent sweeps concurrently while maintaining reproducibility.

---

## Tier 4 — Expert (Staff / Principal)

### System Design

**Q51.** You need to scale this system to serve 10,000 QPS with 10M documents and p99 latency under 50ms. Outline the full system architecture, including compute, storage, networking, and caching layers.

**Q52.** The current system has a "retrieve then generate" pipeline. Design a "retrieve, re-rank, then generate" pipeline. Where would you insert the re-ranker, what model would you use, and how would you handle the latency budget?

**Q53.** Design a continuous evaluation system that automatically detects retrieval quality degradation as the document corpus evolves. What metrics would you track, and what would trigger an alert?

**Q54.** The system currently supports English only. Design a multilingual RAG system that handles queries and documents in multiple languages. What changes are needed at each layer of the current architecture?

**Q55.** Design a system that allows users to provide feedback on RAG answers (thumbs up/down, corrections) and uses that feedback to improve retrieval quality over time. What training signal would you extract from the feedback, and what components would need to change?

---

## Quick-Fire Questions

**Q56.** What is the default dimensionality of MiniLM embeddings?  
**→** 384

**Q57.** What model does the `MockLLM` report in `LLMResponse.model_name`?  
**→** `"mock/stable-simulator-v1"`

**Q58.** What three FAISS exception types does `OpenRouterLLM.generate()` catch?  
**→** `APITimeoutError`, `APIStatusError`, `APIConnectionError`

**Q59.** What value does `FaissHNSWIndex.search()` return for FAISS's `-1` sentinel?  
**→** The result is skipped (filtered out via `if idx < 0: continue`)

**Q60.** What seed is used across all benchmarks by default?  
**→** `42`

**Q61.** How does the `MockEmbeddingService` generate deterministic embeddings?  
**→** SHA-256 hash of the chunk text → seed for `np.random.default_rng`

**Q62.** What is the `_CHARS_PER_TOKEN` constant used for in `MockLLM`?  
**→** `4` — approximates GPT-tokeniser heuristic (1 token ≈ 4 characters)

**Q63.** What metric does `compute_match_rate()` compute?  
**→** Jaccard overlap of Document IDs between two result sets

**Q64.** What DPI resolution are benchmark charts generated at?  
**→** 300 DPI

**Q65.** What is the timeout for the OpenRouter API connection test?  
**→** The model fetch endpoint has an 8-second timeout; the `generate()` call uses a 60-second timeout
