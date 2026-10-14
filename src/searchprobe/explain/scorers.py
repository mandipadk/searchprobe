"""Relevance scorers used to probe triplets.

Every scorer maps (query, documents) -> scores (higher = more relevant).
Embedding models are used with the query/passage prompts they were trained
with; a model scored without its prompts under-performs and would misattribute
a failure to "embeddings in general".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from searchprobe.evidence.passages import bm25_scores, tokenize


class Scorer(Protocol):
    name: str

    def score(self, query: str, docs: list[str]) -> list[float]: ...


@dataclass
class BM25Scorer:
    """Lexical baseline. IDF is computed over the documents being scored."""

    name: str = "bm25"

    def score(self, query: str, docs: list[str]) -> list[float]:
        return bm25_scores(tokenize(query), [tokenize(d) for d in docs])


# Query/document prompts per model family (prefix match on the model id).
EMBEDDING_PROMPTS: dict[str, tuple[str, str]] = {
    "intfloat/e5": ("query: ", "passage: "),
    "intfloat/multilingual-e5": ("query: ", "passage: "),
    "BAAI/bge-small-en": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-base-en": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-large-en": ("Represent this sentence for searching relevant passages: ", ""),
    "nomic-ai/nomic-embed-text": ("search_query: ", "search_document: "),
    "mixedbread-ai/mxbai-embed": ("Represent this sentence for searching relevant passages: ", ""),
    "Snowflake/snowflake-arctic-embed": (
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
}

# Models that ship named prompts inside their sentence-transformers config.
NAMED_PROMPT_MODELS = ("Qwen/Qwen3-Embedding", "google/embeddinggemma", "jinaai/jina-embeddings-v3")


def prompts_for(model: str) -> tuple[str, str]:
    for prefix, prompts in EMBEDDING_PROMPTS.items():
        if model.startswith(prefix):
            return prompts
    return "", ""


class SentenceTransformerScorer:
    """Bi-encoder cosine similarity with model-appropriate prompts."""

    def __init__(self, model: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = f"st:{model}"
        self.model_id = model
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if device:
            kwargs["device"] = device
        self.model = SentenceTransformer(model, **kwargs)
        self.q_prompt, self.d_prompt = prompts_for(model)
        self.named = model.startswith(NAMED_PROMPT_MODELS)

    def score(self, query: str, docs: list[str]) -> list[float]:
        if self.named and getattr(self.model, "prompts", None):
            q = self.model.encode([query], prompt_name="query", normalize_embeddings=True)
            d = self.model.encode(docs, normalize_embeddings=True)
        else:
            q = self.model.encode([self.q_prompt + query], normalize_embeddings=True)
            d = self.model.encode([self.d_prompt + x for x in docs], normalize_embeddings=True)
        return [float(x) for x in (np.asarray(d) @ np.asarray(q)[0])]


class CrossEncoderScorer:
    """Cross-encoder relevance (logits mapped through a sigmoid)."""

    def __init__(self, model: str, device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.name = f"ce:{model}"
        kwargs: dict[str, Any] = {}
        if device:
            kwargs["device"] = device
        self.model = CrossEncoder(model, **kwargs)

    def score(self, query: str, docs: list[str]) -> list[float]:
        from searchprobe.validation.cross_encoder import to_probability

        raw = np.asarray(self.model.predict([(query, d) for d in docs]), dtype=float)
        return [float(x) for x in to_probability(raw)]


class OpenAIEmbeddingScorer:
    """OpenAI embeddings API (e.g. text-embedding-3-large)."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        from openai import OpenAI

        self.name = f"openai:{model}"
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def score(self, query: str, docs: list[str]) -> list[float]:
        resp = self.client.embeddings.create(model=self.model, input=[query, *docs])
        vecs = np.asarray([d.embedding for d in resp.data], dtype=float)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return [float(x) for x in vecs[1:] @ vecs[0]]


def make_scorer(spec: str, settings: Any | None = None) -> Scorer:
    """Build a scorer from 'bm25', 'st:<model>', 'ce:<model>' or 'openai:<model>'."""
    if spec == "bm25":
        return BM25Scorer()
    kind, _, model = spec.partition(":")
    if kind == "st":
        return SentenceTransformerScorer(model)
    if kind == "ce":
        return CrossEncoderScorer(model)
    if kind == "openai":
        key = getattr(settings, "openai_api_key", None) if settings else None
        return OpenAIEmbeddingScorer(model or "text-embedding-3-small", api_key=key)
    raise ValueError(f"Unknown scorer spec: {spec}")
