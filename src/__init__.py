"""
src/__init__.py
Exposes the public API of every module for convenient imports.
"""
from .utils import load_documents, clean_text
from .chunking import fixed_chunking, sliding_window_chunking, semantic_chunking, compare_strategies
from .embeddings import EmbeddingModel
from .retrieval import FAISSRetriever, BM25Retriever, HybridRetriever
from .reranker import CrossEncoderReranker
from .prompt import build_prompt
from .evaluation import evaluate_rag
from .tabular_rag import TabularRAG

__all__ = [
    "load_documents", "clean_text",
    "fixed_chunking", "sliding_window_chunking", "semantic_chunking", "compare_strategies",
    "EmbeddingModel",
    "FAISSRetriever", "BM25Retriever", "HybridRetriever",
    "CrossEncoderReranker",
    "build_prompt",
    "evaluate_rag",
    "TabularRAG",
]
