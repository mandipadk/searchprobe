"""Query-focused passage selection.

Long pages are not truncated blindly to their first N characters (which hides
the sentence that settles a constraint). Instead the text is split into
overlapping windows, scored with BM25 against the query and constraint terms,
and the lead window plus the best-scoring windows are kept, in page order.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "this",
        "that",
        "as",
        "at",
        "from",
        "not",
        "no",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in (m.lower() for m in _TOKEN.findall(text)) if t not in _STOP]


def windows(text: str, size: int = 120, stride: int = 90) -> list[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    out = []
    for start in range(0, len(words), stride):
        chunk = words[start : start + size]
        if chunk:
            out.append(" ".join(chunk))
        if start + size >= len(words):
            break
    return out


def bm25_scores(query_terms: list[str], docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
    """BM25 score of each tokenized doc against the query terms."""
    n = len(docs)
    if n == 0:
        return []
    avgdl = sum(len(d) for d in docs) / n or 1.0
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    scores = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for term in set(query_terms):
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(d) / avgdl))
        scores.append(s)
    return scores


def select_passages(text: str, query: str, max_chars: int = 6000, extra_terms: str = "") -> str:
    """Return the lead window plus the most query-relevant windows, within ``max_chars``."""
    if len(text) <= max_chars:
        return text
    wins = windows(text)
    if not wins:
        return ""
    terms = tokenize(f"{query} {extra_terms}")
    scores = bm25_scores(terms, [tokenize(w) for w in wins])
    chosen = {0}
    budget = max_chars - len(wins[0])
    for idx in sorted(range(len(wins)), key=lambda i: scores[i], reverse=True):
        if idx in chosen or scores[idx] <= 0:
            continue
        if len(wins[idx]) + 5 > budget:
            continue
        chosen.add(idx)
        budget -= len(wins[idx]) + 5
    return "\n[...]\n".join(wins[i] for i in sorted(chosen))
