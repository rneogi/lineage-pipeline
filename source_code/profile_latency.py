#!/usr/bin/env python3
"""
profile_latency.py - Detailed latency breakdown for the RAG pipeline

Measures time spent in each stage:
1. OpenAI embedding generation
2. ChromaDB vector search
3. ZeroEntropy reranking
4. End-to-end query
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List

# Set up paths
BASE_DIR = Path(__file__).resolve().parent

# Ensure API keys are set
def check_keys():
    keys = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "ZEROENTROPY_API_KEY": os.environ.get("ZEROENTROPY_API_KEY"),
    }
    missing = [k for k, v in keys.items() if not v]
    if missing:
        print(f"[WARN] Missing keys: {missing}")
        print("Run: venv\\Scripts\\activate first")
        return False
    return True


def profile_embedding(query: str, n_runs: int = 3) -> Dict[str, float]:
    """Profile OpenAI embedding generation."""
    from openai import OpenAI

    client = OpenAI()
    times = []

    for i in range(n_runs):
        start = time.perf_counter()
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Embedding run {i+1}: {elapsed*1000:.1f}ms")

    return {
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "avg_ms": (sum(times) / len(times)) * 1000,
        "dimensions": len(resp.data[0].embedding)
    }


def profile_chromadb_search(query: str, n_results: int = 100, n_runs: int = 3) -> Dict[str, float]:
    """Profile ChromaDB vector search (excluding embedding time)."""
    import chromadb
    from openai import OpenAI

    # First, get the embedding
    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-3-small", input=[query])
    query_embedding = resp.data[0].embedding

    # Connect to ChromaDB
    chroma_path = BASE_DIR.parent / "chroma_db"
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    collection = chroma_client.get_collection("lineage_collection")

    times = []
    for i in range(n_runs):
        start = time.perf_counter()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  ChromaDB run {i+1}: {elapsed*1000:.1f}ms (returned {len(results['ids'][0])} docs)")

    return {
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "avg_ms": (sum(times) / len(times)) * 1000,
        "docs_returned": len(results['ids'][0])
    }


def profile_zeroentropy_rerank(query: str, documents: List[str], top_n: int = 20, n_runs: int = 3) -> Dict[str, float]:
    """Profile ZeroEntropy reranking."""
    from zeroentropy import ZeroEntropy

    zclient = ZeroEntropy()
    times = []

    for i in range(n_runs):
        start = time.perf_counter()
        response = zclient.models.rerank(
            model="zerank-2",
            query=query,
            documents=documents,
            top_n=top_n
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  ZeroEntropy run {i+1}: {elapsed*1000:.1f}ms (reranked {len(documents)} -> top {top_n})")

    return {
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "avg_ms": (sum(times) / len(times)) * 1000,
        "input_docs": len(documents),
        "output_docs": top_n
    }


def profile_full_query(query: str, k: int = 20, n_runs: int = 3) -> Dict[str, Any]:
    """Profile full RAG query (embedding + search + rerank)."""
    from demo import LineageRAG

    rag = LineageRAG()
    times = []

    for i in range(n_runs):
        start = time.perf_counter()
        results = rag.query(query, k=k)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Full query run {i+1}: {elapsed*1000:.1f}ms (returned {len(results)} results)")

    return {
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "avg_ms": (sum(times) / len(times)) * 1000,
        "results_returned": len(results)
    }


def main():
    if not check_keys():
        return

    print("\n" + "="*70)
    print("LATENCY PROFILER - RAG Pipeline Breakdown")
    print("="*70)

    query = "Which procedures write to tlprdets table and what columns are affected?"
    print(f"\nTest query: {query}\n")

    # 1. Profile embedding
    print("\n[1] OpenAI Embedding (text-embedding-3-small)")
    print("-" * 50)
    emb_stats = profile_embedding(query)

    # 2. Profile ChromaDB search
    print("\n[2] ChromaDB Vector Search (k=100)")
    print("-" * 50)
    chroma_stats = profile_chromadb_search(query, n_results=100)

    # 3. Get documents for reranking
    import chromadb
    from openai import OpenAI

    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-3-small", input=[query])
    query_embedding = resp.data[0].embedding

    chroma_path = BASE_DIR.parent / "chroma_db"
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    collection = chroma_client.get_collection("lineage_collection")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=100,
        include=["documents"]
    )
    documents = results['documents'][0]

    # 4. Profile ZeroEntropy rerank
    print("\n[3] ZeroEntropy Rerank (zerank-2, top_n=20)")
    print("-" * 50)
    ze_stats = profile_zeroentropy_rerank(query, documents, top_n=20)

    # 5. Profile full query
    print("\n[4] Full RAG Query (embedding + search + rerank)")
    print("-" * 50)
    full_stats = profile_full_query(query, k=20)

    # Summary
    print("\n" + "="*70)
    print("LATENCY SUMMARY")
    print("="*70)
    print(f"""
┌─────────────────────────────┬───────────┬───────────┬───────────┐
│ Stage                       │  Min (ms) │  Avg (ms) │  Max (ms) │
├─────────────────────────────┼───────────┼───────────┼───────────┤
│ OpenAI Embedding            │ {emb_stats['avg_ms']:>9.1f} │ {emb_stats['avg_ms']:>9.1f} │ {emb_stats['max_ms']:>9.1f} │
│ ChromaDB Search (k=100)     │ {chroma_stats['min_ms']:>9.1f} │ {chroma_stats['avg_ms']:>9.1f} │ {chroma_stats['max_ms']:>9.1f} │
│ ZeroEntropy Rerank (top=20) │ {ze_stats['min_ms']:>9.1f} │ {ze_stats['avg_ms']:>9.1f} │ {ze_stats['max_ms']:>9.1f} │
├─────────────────────────────┼───────────┼───────────┼───────────┤
│ TOTAL (Full Query)          │ {full_stats['min_ms']:>9.1f} │ {full_stats['avg_ms']:>9.1f} │ {full_stats['max_ms']:>9.1f} │
└─────────────────────────────┴───────────┴───────────┴───────────┘
    """)

    # Bottleneck analysis
    total_component = emb_stats['avg_ms'] + chroma_stats['avg_ms'] + ze_stats['avg_ms']
    print("\nBOTTLENECK ANALYSIS:")
    print("-" * 50)
    print(f"  OpenAI Embedding:    {emb_stats['avg_ms']:>7.1f}ms ({emb_stats['avg_ms']/total_component*100:>5.1f}%)")
    print(f"  ChromaDB Search:     {chroma_stats['avg_ms']:>7.1f}ms ({chroma_stats['avg_ms']/total_component*100:>5.1f}%)")
    print(f"  ZeroEntropy Rerank:  {ze_stats['avg_ms']:>7.1f}ms ({ze_stats['avg_ms']/total_component*100:>5.1f}%)")
    print(f"  ─────────────────────────────────────────")
    print(f"  Component Total:     {total_component:>7.1f}ms")
    print(f"  Measured Total:      {full_stats['avg_ms']:>7.1f}ms")
    print(f"  Overhead:            {full_stats['avg_ms'] - total_component:>7.1f}ms")

    # Recommendations
    print("\n" + "="*70)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*70)

    if emb_stats['avg_ms'] > 500:
        print("[!] OpenAI Embedding is slow (>500ms)")
        print("  -> Consider caching embeddings for repeated queries")
        print("  -> Consider batching multiple queries")

    if chroma_stats['avg_ms'] > 100:
        print("[!] ChromaDB search is slow (>100ms)")
        print("  -> Check index type (HNSW should be <50ms)")
        print("  -> Consider reducing k if not all results needed")

    if ze_stats['avg_ms'] > 1000:
        print("[!] ZeroEntropy rerank is slow (>1000ms)")
        print("  -> This is network latency to their API")
        print("  -> Consider reducing input document count")
        print("  -> Consider local reranker (cross-encoder)")

    if ze_stats['avg_ms'] > emb_stats['avg_ms'] + chroma_stats['avg_ms']:
        print("\n[*] ZeroEntropy reranking is your biggest bottleneck!")
        print("   Options:")
        print("   1. Reduce docs sent to reranker (currently 100 -> try 50)")
        print("   2. Skip reranking for simple queries")
        print("   3. Use local cross-encoder model (slower first load, faster queries)")


if __name__ == "__main__":
    main()
