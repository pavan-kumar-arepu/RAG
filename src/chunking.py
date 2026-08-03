"""
src/chunking.py
Phase 4 – Three chunking strategies with comparison metrics.
"""
from __future__ import annotations

import re
import statistics
from typing import List, Dict, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────
# Strategy 1 – Fixed-size chunking
# ─────────────────────────────────────────────────────────────

def fixed_chunking(
    documents: List[Dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict]:
    """
    Split each document into chunks of exactly *chunk_size* words,
    with *overlap* words of context overlap between consecutive chunks.
    """
    chunks: List[Dict] = []
    chunk_id = 0

    for doc in documents:
        words = doc["text"].split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "text": chunk_text,
                    "strategy": "fixed",
                }
            )
            chunk_id += 1
            if end == len(words):
                break
            start += chunk_size - overlap

    print(f"[chunking] fixed_chunking → {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────────────────────
# Strategy 2 – Sliding-window chunking (sentence-aware)
# ─────────────────────────────────────────────────────────────

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]


def sliding_window_chunking(
    documents: List[Dict],
    window_sentences: int = 10,
    step_sentences: int = 5,
) -> List[Dict]:
    """
    Slide a window of *window_sentences* sentences over the document,
    advancing *step_sentences* at a time.
    """
    chunks: List[Dict] = []
    chunk_id = 0

    for doc in documents:
        sentences = _split_sentences(doc["text"])
        start = 0
        while start < len(sentences):
            end = min(start + window_sentences, len(sentences))
            chunk_text = " ".join(sentences[start:end])
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "text": chunk_text,
                    "strategy": "sliding_window",
                }
            )
            chunk_id += 1
            if end == len(sentences):
                break
            start += step_sentences

    print(f"[chunking] sliding_window_chunking → {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────────────────────
# Strategy 3 – Semantic chunking (paragraph-based)
# ─────────────────────────────────────────────────────────────

def semantic_chunking(
    documents: List[Dict],
    max_words: int = 400,
) -> List[Dict]:
    """
    Split on double-newline paragraph boundaries, then merge small
    paragraphs until *max_words* is reached.  Keeps semantically
    coherent paragraphs together.
    """
    chunks: List[Dict] = []
    chunk_id = 0

    for doc in documents:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", doc["text"]) if p.strip()]
        buffer: List[str] = []
        buffer_words = 0

        def _flush():
            nonlocal chunk_id, buffer_words
            if buffer:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc["doc_id"],
                        "title": doc["title"],
                        "text": " ".join(buffer),
                        "strategy": "semantic",
                    }
                )
                chunk_id += 1
            buffer.clear()
            buffer_words = 0

        for para in paragraphs:
            word_count = len(para.split())
            if buffer_words + word_count > max_words and buffer:
                _flush()
            buffer.append(para)
            buffer_words += word_count

        _flush()

    print(f"[chunking] semantic_chunking → {len(chunks)} chunks")
    return chunks


# ─────────────────────────────────────────────────────────────
# Comparison helper
# ─────────────────────────────────────────────────────────────

def _broken_sentence_pct(chunks: List[Dict]) -> float:
    """Fraction of chunks that do NOT start with a capital letter."""
    if not chunks:
        return 0.0
    broken = sum(1 for c in chunks if c["text"] and not c["text"][0].isupper())
    return round(broken / len(chunks) * 100, 2)


def compare_strategies(
    documents: List[Dict],
) -> Tuple[List[Dict], Dict]:
    """
    Run all three strategies and return:
    - ``best_chunks`` : chunks from the best strategy (lowest broken-sentence %)
    - ``report``      : dict with per-strategy stats for display
    """
    strategies = {
        "fixed": fixed_chunking(documents),
        "sliding_window": sliding_window_chunking(documents),
        "semantic": semantic_chunking(documents),
    }

    report: Dict = {}
    for name, chunks in strategies.items():
        sizes = [len(c["text"].split()) for c in chunks]
        report[name] = {
            "num_chunks": len(chunks),
            "avg_chunk_words": round(statistics.mean(sizes), 1) if sizes else 0,
            "std_chunk_words": round(statistics.stdev(sizes), 1) if len(sizes) > 1 else 0,
            "broken_sentence_pct": _broken_sentence_pct(chunks),
        }

    # Best = fewest broken sentences, ties broken by lower std
    best_name = min(
        report,
        key=lambda k: (report[k]["broken_sentence_pct"], report[k]["std_chunk_words"]),
    )
    print(f"[chunking] Best strategy: '{best_name}'")
    for name, stats in report.items():
        marker = " ← SELECTED" if name == best_name else ""
        print(f"  {name}: {stats}{marker}")

    return strategies[best_name], report
