"""
src/reranker.py
Phase 9 – Cross-Encoder re-ranking of retrieved chunks.
"""
from __future__ import annotations

from typing import List, Tuple, Dict


class CrossEncoderReranker:
    """
    Re-ranks a list of (chunk, score) pairs using a cross-encoder model.

    Default model: 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    (fast, small, good for passage ranking).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        from sentence_transformers import CrossEncoder  # type: ignore

        print(f"[reranker] Loading cross-encoder '{model_name}' …")
        self._model = CrossEncoder(model_name)
        self._model_name = model_name
        print("[reranker] Cross-encoder loaded.")

    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Dict, float]],
        top_k: int | None = None,
    ) -> List[Tuple[Dict, float]]:
        """
        Parameters
        ----------
        query      : user query string
        candidates : output of HybridRetriever.search() – list of (chunk, score)
        top_k      : keep only the top-k results after re-ranking; None = keep all

        Returns
        -------
        Re-ranked list of (chunk, cross_encoder_score), highest score first.
        """
        if not candidates:
            return []

        pairs = [(query, chunk["text"]) for chunk, _ in candidates]
        ce_scores = self._model.predict(pairs)

        reranked = sorted(
            zip([c for c, _ in candidates], ce_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        if top_k is not None:
            reranked = reranked[:top_k]

        return [(chunk, float(score)) for chunk, score in reranked]

    # ------------------------------------------------------------------

    @staticmethod
    def compare(
        before: List[Tuple[Dict, float]],
        after: List[Tuple[Dict, float]],
        n: int = 5,
    ) -> None:
        """Print a side-by-side comparison of the top-n results before/after."""
        print("\n" + "=" * 70)
        print(f"{'BEFORE re-rank':35s} | {'AFTER re-rank':35s}")
        print("=" * 70)
        for i in range(min(n, max(len(before), len(after)))):
            b_title = before[i][0]["title"][:30] if i < len(before) else ""
            a_title = after[i][0]["title"][:30] if i < len(after) else ""
            b_score = f"{before[i][1]:.4f}" if i < len(before) else ""
            a_score = f"{after[i][1]:.4f}" if i < len(after) else ""
            print(f"  {i+1}. {b_title:<28} {b_score:<6} | {i+1}. {a_title:<28} {a_score}")
        print("=" * 70 + "\n")
