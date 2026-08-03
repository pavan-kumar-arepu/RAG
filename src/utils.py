"""
src/utils.py
Phase 2 & 3 – Document loading and cleaning.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Dict


# ─────────────────────────────────────────────────────────────
# Phase 2 – Document Loading
# ─────────────────────────────────────────────────────────────

def load_documents(corpus_dir: str | Path) -> List[Dict]:
    """
    Load all .txt files from *corpus_dir* into a list of document dicts.

    Returns
    -------
    List[Dict] with keys:
        - ``doc_id``   : zero-based integer
        - ``filename`` : original file name
        - ``title``    : file stem (human-readable title)
        - ``text``     : raw UTF-8 content
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    documents: List[Dict] = []
    for idx, path in enumerate(sorted(corpus_dir.glob("*.txt"))):
        text = path.read_text(encoding="utf-8", errors="replace")
        documents.append(
            {
                "doc_id": idx,
                "filename": path.name,
                "title": path.stem.replace("_", " "),
                "text": text,
            }
        )

    print(f"[utils] Loaded {len(documents)} documents from '{corpus_dir}'")
    return documents


# ─────────────────────────────────────────────────────────────
# Phase 3 – Text Cleaning
# ─────────────────────────────────────────────────────────────

# Patterns for artefacts commonly found in PDF-extracted .txt files
_PAGE_NUMBER_RE = re.compile(r"(?m)^\s*\d+\s*$")           # lone digits on a line
_HEADER_FOOTER_RE = re.compile(                             # repeated short lines
    r"(?m)^(.{1,80})\n(?:.*\n)*?\1\n", re.MULTILINE
)
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")               # broken words at line end
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")                   # 3+ blank lines → 2
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")                  # multiple spaces


def clean_text(text: str) -> str:
    """
    Clean a single document string.

    Steps
    -----
    1. Remove page-number-only lines.
    2. Re-join words broken by a hyphen at line end.
    3. Collapse excessive whitespace / blank lines.
    """
    text = _PAGE_NUMBER_RE.sub("", text)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = text.strip()
    return text


def clean_documents(documents: List[Dict]) -> List[Dict]:
    """Apply *clean_text* to every document and return a new list."""
    cleaned = []
    for doc in documents:
        cleaned.append({**doc, "text": clean_text(doc["text"])})
    print(f"[utils] Cleaned {len(cleaned)} documents")
    return cleaned
