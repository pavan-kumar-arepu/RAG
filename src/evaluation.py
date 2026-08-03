"""
src/evaluation.py
Phase 12 – Evaluation metrics for the RAG pipeline.

Metrics
-------
- Latency            : seconds per query
- Retrieval Recall   : fraction of relevant chunks retrieved
- Retrieval Precision: fraction of retrieved chunks that are relevant
- Answer Coverage    : % of question words found in the generated answer
- Answer Relevance   : ROUGE-L F1 between answer and ground-truth (if available)
"""
from __future__ import annotations

import time
from typing import List, Dict, Tuple, Callable, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────
# Latency
# ─────────────────────────────────────────────────────────────

def measure_latency(fn: Callable, *args, **kwargs) -> Tuple:
    """
    Call *fn(*args, **kwargs)* and return ``(result, elapsed_seconds)``.
    """
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return result, elapsed


# ─────────────────────────────────────────────────────────────
# Retrieval precision / recall
# ─────────────────────────────────────────────────────────────

def retrieval_precision_recall(
    retrieved_chunks: List[Dict],
    relevant_doc_ids: List[int],
) -> Dict[str, float]:
    """
    Parameters
    ----------
    retrieved_chunks : chunks returned by retriever / reranker
    relevant_doc_ids : ground-truth doc IDs considered relevant for the query

    Returns
    -------
    Dict with keys 'precision', 'recall', 'f1'
    """
    relevant_set = set(relevant_doc_ids)
    retrieved_doc_ids = [c["doc_id"] for c in retrieved_chunks]

    if not retrieved_doc_ids:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = sum(1 for d in retrieved_doc_ids if d in relevant_set)
    precision = tp / len(retrieved_doc_ids)
    recall = tp / len(relevant_set) if relevant_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ─────────────────────────────────────────────────────────────
# Coverage (question → answer)
# ─────────────────────────────────────────────────────────────

def answer_coverage(question: str, answer: str) -> float:
    """
    Fraction of *content* words in the question that also appear in the answer.
    Simple token-overlap heuristic.
    """
    _stopwords = {
        "what", "which", "who", "how", "where", "when", "why",
        "is", "are", "was", "were", "the", "a", "an", "of", "in",
        "to", "and", "or", "for", "with", "on", "at", "by",
    }
    q_words = {w.lower() for w in question.split() if w.lower() not in _stopwords}
    a_words = {w.lower() for w in answer.split()}
    if not q_words:
        return 1.0
    return round(len(q_words & a_words) / len(q_words), 4)


# ─────────────────────────────────────────────────────────────
# ROUGE-L relevance (optional, needs rouge-score package)
# ─────────────────────────────────────────────────────────────

def rouge_l_score(prediction: str, reference: str) -> float:
    """Return ROUGE-L F1. Returns 0.0 if rouge-score is not installed."""
    try:
        from rouge_score import rouge_scorer  # type: ignore

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        result = scorer.score(reference, prediction)
        return round(result["rougeL"].fmeasure, 4)
    except ImportError:
        return 0.0


# ─────────────────────────────────────────────────────────────
# Full evaluate_rag helper
# ─────────────────────────────────────────────────────────────

def evaluate_rag(
    questions: List[str],
    references: Optional[List[str]],       # ground-truth answers; None = skip ROUGE
    relevant_doc_ids: Optional[List[List[int]]],  # per-question; None = skip P/R
    rag_fn: Callable[[str], Tuple[str, List[Dict], float]],
    # rag_fn(question) → (answer, retrieved_chunks, latency_secs)
) -> List[Dict]:
    """
    Run *rag_fn* for every question and aggregate all metrics.

    Returns
    -------
    List of per-question result dicts.
    """
    results = []
    for i, question in enumerate(questions):
        answer, chunks, latency = rag_fn(question)

        result: Dict = {
            "question": question,
            "answer": answer,
            "latency_s": round(latency, 3),
            "coverage": answer_coverage(question, answer),
        }

        if references and i < len(references) and references[i]:
            result["rouge_l"] = rouge_l_score(answer, references[i])

        if relevant_doc_ids and i < len(relevant_doc_ids):
            result.update(retrieval_precision_recall(chunks, relevant_doc_ids[i]))

        results.append(result)

    _print_summary(results)
    return results


def _print_summary(results: List[Dict]) -> None:
    keys = ["latency_s", "coverage", "rouge_l", "precision", "recall", "f1"]
    print("\n[evaluation] Summary")
    print("-" * 55)
    for k in keys:
        vals = [r[k] for r in results if k in r]
        if vals:
            print(f"  {k:<20} avg={np.mean(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}")
    print("-" * 55 + "\n")
