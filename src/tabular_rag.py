"""
src/tabular_rag.py
Phase 13 – Tabular RAG: extract tables from PDFs, convert rows to chunks,
embed, index, and answer table-based questions.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Optional


class TabularRAG:
    """
    Full Tabular-RAG pipeline:
    1. Extract tables from PDFs using pdfplumber.
    2. Convert each table row → text chunk.
    3. Embed with an EmbeddingModel.
    4. Build a FAISS index over table chunks.
    5. Retrieve and answer.
    """

    def __init__(self, embed_model) -> None:
        """
        Parameters
        ----------
        embed_model : EmbeddingModel instance (already initialised)
        """
        self._embed = embed_model
        self._table_chunks: List[Dict] = []
        self._index = None

    # ------------------------------------------------------------------
    # Step 1 – Extract tables from all PDFs in a directory
    # ------------------------------------------------------------------

    def extract_tables_from_pdf(self, pdf_path: str | Path) -> List[Dict]:
        """
        Extract all tables from a single PDF and return a list of
        row-level dicts with keys: ``source``, ``table_idx``, ``row_idx``,
        ``headers``, ``row_data``, ``text``.
        """
        try:
            import pdfplumber  # type: ignore
        except ImportError:
            print("[tabular_rag] pdfplumber not installed. Run: pip install pdfplumber")
            return []

        pdf_path = Path(pdf_path)
        rows: List[Dict] = []
        chunk_id = len(self._table_chunks)

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue
                    headers = [str(h).strip() if h else f"col{i}" for i, h in enumerate(table[0])]
                    for r_idx, row in enumerate(table[1:], start=1):
                        cells = [str(c).strip() if c else "" for c in row]
                        text = "; ".join(
                            f"{h}: {v}" for h, v in zip(headers, cells) if v
                        )
                        if not text:
                            continue
                        rows.append(
                            {
                                "chunk_id": chunk_id,
                                "doc_id": -1,           # tabular chunks use -1
                                "title": pdf_path.stem,
                                "source": pdf_path.name,
                                "page": page_num + 1,
                                "table_idx": t_idx,
                                "row_idx": r_idx,
                                "headers": headers,
                                "text": text,
                                "strategy": "tabular",
                            }
                        )
                        chunk_id += 1

        print(f"[tabular_rag] Extracted {len(rows)} table rows from '{pdf_path.name}'")
        return rows

    def extract_tables_from_dir(self, pdf_dir: str | Path) -> List[Dict]:
        """Extract tables from all PDFs in *pdf_dir*."""
        pdf_dir = Path(pdf_dir)
        all_rows: List[Dict] = []
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            all_rows.extend(self.extract_tables_from_pdf(pdf_path))
        print(f"[tabular_rag] Total table chunks extracted: {len(all_rows)}")
        return all_rows

    # ------------------------------------------------------------------
    # Step 2–4 – Embed and index table chunks
    # ------------------------------------------------------------------

    def build_index(self, table_chunks: List[Dict]) -> None:
        """Embed *table_chunks* and build an in-memory FAISS index."""
        import faiss  # type: ignore
        import numpy as np

        self._table_chunks = table_chunks
        embeddings = self._embed.embed_chunks(table_chunks)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        print(f"[tabular_rag] FAISS table index built with {self._index.ntotal} vectors")

    # ------------------------------------------------------------------
    # Step 5 – Retrieve relevant table rows
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Return top-k table chunks most relevant to *query*."""
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        q_emb = self._embed.embed_query(query)
        scores, indices = self._index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._table_chunks[idx], float(score)))
        return results

    # ------------------------------------------------------------------
    # Step 6 – Answer (passes context to LLM or returns context string)
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        llm_fn: Optional = None,
        top_k: int = 5,
    ) -> str:
        """
        Parameters
        ----------
        query  : user question
        llm_fn : callable(prompt: str) → str  — any LLM wrapper.
                 If None, returns the raw context string.
        top_k  : number of table rows to retrieve
        """
        results = self.search(query, top_k=top_k)
        context_parts = [f"  [{c['source']} | page {c['page']}]\n  {c['text']}" for c, _ in results]
        context = "\n\n".join(context_parts) or "No relevant table data found."

        prompt = (
            f"Answer the question using only the table data below.\n\n"
            f"Table Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )

        if llm_fn is not None:
            return llm_fn(prompt)
        return context
