"""
src/prompt.py
Phase 10 – Prompt construction from retrieved chunks.
"""
from __future__ import annotations

from typing import List, Tuple, Dict

# Maximum number of context tokens (rough word-based limit)
DEFAULT_CONTEXT_WORD_LIMIT = 1200

PROMPT_TEMPLATE = """\
You are a knowledgeable assistant specialising in Natural Language Processing
and related research topics.

Answer the question **only** using the information in the context below.
If the answer is not present in the context, respond with:
"I cannot find a relevant answer in the provided documents."

### Context
{context}

### Question
{question}

### Answer
"""


def build_prompt(
    question: str,
    ranked_chunks: List[Tuple[Dict, float]],
    context_word_limit: int = DEFAULT_CONTEXT_WORD_LIMIT,
    template: str = PROMPT_TEMPLATE,
) -> str:
    """
    Construct the full prompt by injecting the top retrieved chunks as context.

    Parameters
    ----------
    question         : user's query string
    ranked_chunks    : output of reranker.rerank() – [(chunk_dict, score), ...]
    context_word_limit : stop adding chunks once this word count is reached
    template         : jinja-style template with {context} and {question}

    Returns
    -------
    Ready-to-send prompt string.
    """
    context_parts: List[str] = []
    word_count = 0

    for chunk, _ in ranked_chunks:
        text = chunk["text"].strip()
        words = len(text.split())
        if word_count + words > context_word_limit:
            # Include a truncated version of this chunk if nothing was added yet
            if not context_parts:
                remaining = context_word_limit
                text = " ".join(text.split()[:remaining])
                context_parts.append(f"[{chunk['title']}]\n{text}")
            break
        context_parts.append(f"[{chunk['title']}]\n{text}")
        word_count += words

    context = "\n\n".join(context_parts) if context_parts else "No relevant context found."
    prompt = template.format(context=context, question=question)
    return prompt
