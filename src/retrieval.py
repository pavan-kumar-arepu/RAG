"""
src/retrieval.py
Phase 6 (FAISS), Phase 7 (BM25), Phase 8 (Hybrid RRF).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────
# Phase 6 – Dense Retrieval with FAISS
# ─────────────────────────────────────────────────────────────

class FAISSRetriever:
    """
    Builds and queries a FAISS flat-L2 (or IVF) index over chunk embeddings.
    Since embeddings are L2-normalised, inner-product ≡ cosine similarity.
    """

    def __init__(self) -> None:
        self._index = None
        self._chunks: List[Dict] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: List[Dict], embeddings: np.ndarray) -> None:
        import faiss  # type: ignore

        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings must have the same length")

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)   # Inner Product (cosine after L2-norm)
        self._index.add(embeddings)
        self._chunks = list(chunks)
        print(f"[retrieval] FAISS index built with {self._index.ntotal} vectors (dim={dim})")

    # ------------------------------------------------------------------
    # Persist / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        import faiss  # type: ignore

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "faiss.index"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        print(f"[retrieval] FAISS index saved to '{path}'")

    def load(self, path: str | Path) -> None:
        import faiss  # type: ignore

        path = Path(path)
        self._index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "chunks.pkl", "rb") as f:
            self._chunks = pickle.load(f)
        print(f"[retrieval] FAISS index loaded from '{path}' ({self._index.ntotal} vectors)")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[Dict, float]]:
        """
        Returns list of ``(chunk_dict, score)`` sorted by descending similarity.
        ``query_embedding`` should be shape ``(1, dim)``.
        """
        scores, indices = self._index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._chunks[idx], float(score)))
        return results


# ─────────────────────────────────────────────────────────────
# Phase 7 – Sparse Retrieval with BM25
# ─────────────────────────────────────────────────────────────

class BM25Retriever:
    """Keyword-based retrieval using the rank-bm25 library."""

    def __init__(self) -> None:
        self._bm25 = None
        self._chunks: List[Dict] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: List[Dict]) -> None:
        from rank_bm25 import BM25Okapi  # type: ignore

        tokenised = [c["text"].lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenised)
        self._chunks = list(chunks)
        print(f"[retrieval] BM25 index built over {len(chunks)} chunks")

    # ------------------------------------------------------------------
    # Persist / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump((self._bm25, self._chunks), f)
        print(f"[retrieval] BM25 index saved to '{path}'")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        with open(path / "bm25.pkl", "rb") as f:
            self._bm25, self._chunks = pickle.load(f)
        print(f"[retrieval] BM25 index loaded from '{path}'")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Returns list of ``(chunk_dict, bm25_score)`` sorted descending."""
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._chunks[i], float(scores[i])) for i in top_indices]


# ─────────────────────────────────────────────────────────────
# Phase 8 – Hybrid Retrieval (Reciprocal Rank Fusion)
# ─────────────────────────────────────────────────────────────

def _rrf_score(rank: int, k: int = 60) -> float:
    """Standard RRF formula: 1 / (k + rank)."""
    return 1.0 / (k + rank + 1)


class HybridRetriever:
    """
    Combines FAISS and BM25 results using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        faiss_retriever: FAISSRetriever,
        bm25_retriever: BM25Retriever,
        embed_model,                      # EmbeddingModel instance
        rrf_k: int = 60,
    ) -> None:
        self._faiss = faiss_retriever
        self._bm25 = bm25_retriever
        self._embed = embed_model
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 50,
    ) -> List[Tuple[Dict, float]]:
        """
        1. Retrieve *candidate_k* results from each index.
        2. Apply RRF to merge ranked lists.
        3. Return top *top_k* chunks with their fused scores.
        """
        # Dense results
        q_emb = self._embed.embed_query(query)
        dense_results = self._faiss.search(q_emb, top_k=candidate_k)

        # Sparse results
        sparse_results = self._bm25.search(query, top_k=candidate_k)

        # Accumulate RRF scores keyed by chunk_id
        rrf_scores: Dict[int, float] = {}
        chunk_map: Dict[int, Dict] = {}

        for rank, (chunk, _) in enumerate(dense_results):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + _rrf_score(rank, self._rrf_k)
            chunk_map[cid] = chunk

        for rank, (chunk, _) in enumerate(sparse_results):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + _rrf_score(rank, self._rrf_k)
            chunk_map[cid] = chunk

        # Sort by fused score
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [(chunk_map[cid], rrf_scores[cid]) for cid in sorted_ids[:top_k]]
