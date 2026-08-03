"""
src/embeddings.py
Phase 5 – Generate embeddings for every chunk using sentence-transformers.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np


class EmbeddingModel:
    """
    Wrapper around sentence-transformers for generating chunk embeddings.

    Parameters
    ----------
    model_name : str
        Any model from https://www.sbert.net/docs/pretrained_models.html.
        Default: 'all-MiniLM-L6-v2'  (fast, small, good quality)
    batch_size : int
        Batch size passed to ``encode()``.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
    ) -> None:
        # Lazy import so the module is importable even before install
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.model_name = model_name
        self.batch_size = batch_size
        print(f"[embeddings] Loading model '{model_name}' …")
        self._model = SentenceTransformer(model_name)
        print(f"[embeddings] Model loaded. Embedding dim = {self.embedding_dim}")

    @property
    def embedding_dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Core encode method
    # ------------------------------------------------------------------

    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of strings and return a float32 numpy array
        of shape ``(len(texts), embedding_dim)``.
        """
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalise → cosine similarity = dot product
        )
        return embeddings.astype(np.float32)

    # ------------------------------------------------------------------
    # Convenience: encode chunks list in-place
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: List[Dict], show_progress: bool = True) -> np.ndarray:
        """
        Extract the ``text`` field from every chunk and generate embeddings.

        Returns
        -------
        np.ndarray of shape ``(N, dim)`` in the same order as *chunks*.
        """
        texts = [c["text"] for c in chunks]
        print(f"[embeddings] Encoding {len(texts)} chunks …")
        embeddings = self.encode(texts, show_progress=show_progress)
        print(f"[embeddings] Done. Shape: {embeddings.shape}")
        return embeddings

    # ------------------------------------------------------------------
    # Convenience: encode a single query
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        """Return shape ``(1, dim)`` embedding for a query string."""
        return self.encode([query], show_progress=False)
