#!/usr/bin/env python3
"""
Demonstrates the exact blocker: LLM generation requires Ollama.
This script shows the error and what's needed to fix it.
"""

import sys
import time
sys.path.insert(0, '.')

print("\n" + "="*70)
print("FINDING THE BLOCKER - LLM GENERATION TEST")
print("="*70)

# Step 1: Retrieval works
print("\n[Step 1] RETRIEVAL - Testing if we can get chunks from corpus")
from src.utils import load_documents, clean_documents
from src.chunking import compare_strategies
from src.embeddings import EmbeddingModel
from src.retrieval import HybridRetriever
from src.reranker import CrossEncoderReranker
from pathlib import Path

CORPUS_DIR = Path('data/corpus')
documents = load_documents(CORPUS_DIR)
clean_docs = clean_documents(documents)
best_chunks, _ = compare_strategies(clean_docs)
embed_model = EmbeddingModel('all-MiniLM-L6-v2', batch_size=32)

print(f"  ✓ Loaded {len(best_chunks)} chunks")

embeddings = embed_model.embed_chunks(best_chunks, show_progress=False)
print(f"  ✓ Generated {embeddings.shape[0]} embeddings")

from src.retrieval import FAISSRetriever, BM25Retriever
faiss_ret = FAISSRetriever()
faiss_ret.build(best_chunks, embeddings)
bm25_ret = BM25Retriever()
bm25_ret.build(best_chunks)
hybrid_ret = HybridRetriever(faiss_ret, bm25_ret, embed_model)
reranker = CrossEncoderReranker()

print(f"  ✓ Retrieval pipeline ready")

# Step 2: Retrieve for a query
print("\n[Step 2] RETRIEVE - Getting top chunks for query")
test_query = "What is the main contribution of transformer models to NLP?"
hybrid_results = hybrid_ret.search(test_query, top_k=5)
reranked = reranker.rerank(test_query, hybrid_results, top_k=3)

print(f"  ✓ Retrieved {len(reranked)} chunks")
print(f"\n  Top chunk:")
chunk, score = reranked[0]
print(f"    Title: {chunk['title'][:70]}")
print(f"    Score: {score:.4f}")
print(f"    Preview: {chunk['text'][:150]}...")

# Step 3: Build prompt
print("\n[Step 3] PROMPT - Formatting context for LLM")
from src.prompt import build_prompt
prompt = build_prompt(test_query, reranked)
print(f"  ✓ Prompt built")
print(f"\n  Prompt preview (first 200 chars):")
print(f"    {prompt[:200]}...")

# Step 4: LLM call - THIS IS WHERE IT FAILS
print("\n[Step 4] LLM CALL - Sending to Ollama at localhost:11434")
print(f"\n  Attempting connection...")

import requests
import json

try:
    # This is exactly what the LLM generation phase does
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama2",
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7
        },
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        answer = result.get('response', '')
        print(f"  ✓ LLM generated answer!")
        print(f"\n  Answer (first 200 chars):")
        print(f"    {answer[:200]}...")
    else:
        print(f"  ✗ Unexpected status: {response.status_code}")
        
except requests.exceptions.ConnectionError as e:
    print(f"  ✗ CONNECTION REFUSED")
    print(f"\n  Error: {e}")
    print(f"\n  Root cause: Ollama is not running")
    print(f"\n  Fix required:")
    print(f"    1. Install: brew install ollama")
    print(f"    2. Run: ollama serve  (in separate terminal)")
    print(f"    3. Pull: ollama pull llama2")
    print(f"    4. Try again")
    
except requests.exceptions.Timeout:
    print(f"  ✗ TIMEOUT - Ollama took too long to respond")
    print(f"  Likely causes:")
    print(f"    - First query (model loading)")
    print(f"    - System is slow")
    print(f"    - Model is too large")
    
except Exception as e:
    print(f"  ✗ Unexpected error: {type(e).__name__}: {e}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("Pipeline phases that WORK:")
print("  ✓ Phase 2: Document Loading")
print("  ✓ Phase 3: Text Cleaning")
print("  ✓ Phase 4: Semantic Chunking")
print("  ✓ Phase 5: Embeddings")
print("  ✓ Phase 6: FAISS Indexing")
print("  ✓ Phase 7: BM25 Indexing")
print("  ✓ Phase 8: Hybrid Retrieval")
print("  ✓ Phase 9: Re-ranking")
print("  ✓ Phase 10: Prompt Building")
print("\nPipeline phase that FAILS:")
print("  ✗ Phase 11: LLM Generation (BLOCKED - no Ollama)")
print("\nTo proceed:")
print("  1. brew install ollama")
print("  2. In terminal: ollama serve")
print("  3. In terminal: ollama pull llama2")
print("  4. Re-run this script")
print("="*70 + "\n")
