"""
main.py
End-to-end RAG pipeline orchestrator for Group138 – LLM Assignment 2B.

Run:
    python main.py

Environment variables (optional – set in .env or shell):
    LLM_PROVIDER   : "ollama" | "hf"  (default: ollama)
    OLLAMA_MODEL   : model name served by Ollama  (default: llama3)
    OLLAMA_URL     : Ollama API endpoint  (default: http://localhost:11434)
    HF_MODEL       : HuggingFace model id  (default: google/gemma-2b-it)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# ── Allow imports from src/ regardless of working directory ────────────────
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.utils import load_documents, clean_documents
from src.chunking import compare_strategies
from src.embeddings import EmbeddingModel
from src.retrieval import FAISSRetriever, BM25Retriever, HybridRetriever
from src.reranker import CrossEncoderReranker
from src.prompt import build_prompt
from src.evaluation import evaluate_rag, measure_latency

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CORPUS_DIR  = BASE_DIR / "data" / "corpus"
INDEX_DIR   = BASE_DIR / "indexes"
EVAL_DIR    = BASE_DIR / "evaluation"
OUTPUT_DIR  = BASE_DIR / "outputs"

# ── Configuration ────────────────────────────────────────────────────────────
LLM_PROVIDER  = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")
HF_MODEL      = os.getenv("HF_MODEL", "google/gemma-2b-it")

EMBED_MODEL   = "all-MiniLM-L6-v2"
TOP_K_HYBRID  = 20
TOP_K_RERANK  = 5


# ════════════════════════════════════════════════════════════════════════════
# LLM wrappers
# ════════════════════════════════════════════════════════════════════════════

def _llm_ollama(prompt: str) -> str:
    import requests  # type: ignore

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as exc:
        return f"[LLM ERROR] {exc}"


def _llm_hf(prompt: str) -> str:
    from transformers import pipeline  # type: ignore

    pipe = pipeline("text-generation", model=HF_MODEL, max_new_tokens=256)
    output = pipe(prompt)[0]["generated_text"]
    # Strip the prompt from the output if it was echoed
    return output[len(prompt):].strip() if output.startswith(prompt) else output.strip()


def get_llm():
    if LLM_PROVIDER == "hf":
        print(f"[main] Using HuggingFace model: {HF_MODEL}")
        return _llm_hf
    print(f"[main] Using Ollama model: {OLLAMA_MODEL} at {OLLAMA_URL}")
    return _llm_ollama


# ════════════════════════════════════════════════════════════════════════════
# Pipeline setup (run once, index is reused on subsequent runs)
# ════════════════════════════════════════════════════════════════════════════

def build_pipeline():
    """Load corpus, chunk, embed, and build indexes.  Returns retriever objects."""

    # Phase 2 – Load
    documents = load_documents(CORPUS_DIR)

    # Phase 3 – Clean
    clean_docs = clean_documents(documents)

    # Phase 4 – Chunking comparison → select best strategy
    best_chunks, chunking_report = compare_strategies(clean_docs)

    # Save chunking report
    EVAL_DIR.mkdir(exist_ok=True)
    with open(EVAL_DIR / "chunking_report.json", "w") as f:
        json.dump(chunking_report, f, indent=2)
    print(f"[main] Chunking report saved → {EVAL_DIR / 'chunking_report.json'}")

    # Phase 5 – Embeddings
    embed_model = EmbeddingModel(EMBED_MODEL)
    embeddings = embed_model.embed_chunks(best_chunks)

    # Phase 6 – FAISS
    faiss_ret = FAISSRetriever()
    faiss_ret.build(best_chunks, embeddings)
    faiss_ret.save(INDEX_DIR / "faiss")

    # Phase 7 – BM25
    bm25_ret = BM25Retriever()
    bm25_ret.build(best_chunks)
    bm25_ret.save(INDEX_DIR / "bm25")

    # Phase 8 – Hybrid
    hybrid_ret = HybridRetriever(faiss_ret, bm25_ret, embed_model)

    # Phase 9 – Cross-Encoder
    reranker = CrossEncoderReranker()

    return embed_model, faiss_ret, bm25_ret, hybrid_ret, reranker


def load_pipeline():
    """Load pre-built indexes from disk (faster on subsequent runs)."""
    embed_model = EmbeddingModel(EMBED_MODEL)
    faiss_ret = FAISSRetriever()
    faiss_ret.load(INDEX_DIR / "faiss")
    bm25_ret = BM25Retriever()
    bm25_ret.load(INDEX_DIR / "bm25")
    hybrid_ret = HybridRetriever(faiss_ret, bm25_ret, embed_model)
    reranker = CrossEncoderReranker()
    return embed_model, faiss_ret, bm25_ret, hybrid_ret, reranker


# ════════════════════════════════════════════════════════════════════════════
# Single-query RAG function
# ════════════════════════════════════════════════════════════════════════════

def rag_query(
    question: str,
    hybrid_ret: HybridRetriever,
    reranker: CrossEncoderReranker,
    llm_fn,
) -> tuple[str, list, float]:
    """
    Run the full RAG pipeline for one question.

    Returns (answer, retrieved_chunks, latency_seconds).
    """
    t0 = time.perf_counter()

    # Retrieve
    candidates = hybrid_ret.search(question, top_k=TOP_K_HYBRID)

    # Re-rank
    ranked = reranker.rerank(question, candidates, top_k=TOP_K_RERANK)

    # Build prompt
    prompt = build_prompt(question, ranked)

    # Generate
    answer = llm_fn(prompt)

    latency = time.perf_counter() - t0
    return answer, [c for c, _ in ranked], latency


# ════════════════════════════════════════════════════════════════════════════
# Demo evaluation set
# ════════════════════════════════════════════════════════════════════════════

DEMO_QUESTIONS = [
    "What is the main contribution of transformer models to NLP?",
    "How does QLoRA reduce memory usage during fine-tuning?",
    "What are the challenges of NLP for low-resource languages?",
    "Describe the Bangla NLP tasks covered in the survey.",
    "How is NLP taught interactively to young students?",
]


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 65)
    print("  Group138 – LLM Assignment 2B  |  RAG Pipeline")
    print("=" * 65 + "\n")

    # ── Build or load indexes ────────────────────────────────────────────
    faiss_index_path = INDEX_DIR / "faiss" / "faiss.index"
    if faiss_index_path.exists():
        print("[main] Existing indexes found – loading from disk …")
        embed_model, faiss_ret, bm25_ret, hybrid_ret, reranker = load_pipeline()
    else:
        print("[main] No indexes found – building from scratch …")
        embed_model, faiss_ret, bm25_ret, hybrid_ret, reranker = build_pipeline()

    # ── LLM ─────────────────────────────────────────────────────────────
    llm_fn = get_llm()

    # ── Evaluation ──────────────────────────────────────────────────────
    def _rag_fn(question):
        return rag_query(question, hybrid_ret, reranker, llm_fn)

    results = evaluate_rag(
        questions=DEMO_QUESTIONS,
        references=None,
        relevant_doc_ids=None,
        rag_fn=_rag_fn,
    )

    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[main] Evaluation results saved → {out_path}")

    # ── Interactive loop ─────────────────────────────────────────────────
    print("\n[main] Entering interactive query mode. Type 'quit' to exit.\n")
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        answer, chunks, latency = rag_query(question, hybrid_ret, reranker, llm_fn)
        print(f"\nAnswer ({latency:.2f}s):\n{answer}\n")
        print(f"Sources: {', '.join(set(c['title'] for c in chunks))}\n")


if __name__ == "__main__":
    main()
