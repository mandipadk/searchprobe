"""Data models for cross-encoder validation."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class CrossEncoderScore:
    """Cross-encoder relevance score for a (query, document) pair."""

    query: str
    document_title: str
    document_url: str
    original_rank: int  # Position from the search provider
    cross_encoder_score: float  # Raw cross-encoder score
    reranked_position: int = 0  # Position after reranking

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "document_title": self.document_title,
            "document_url": self.document_url,
            "original_rank": self.original_rank,
            "cross_encoder_score": self.cross_encoder_score,
            "reranked_position": self.reranked_position,
        }


@dataclass
class ValidationResult:
    """Validation result for a single query against a provider."""

    query_id: str
    query_text: str
    provider: str
    category: str
    cross_encoder_model: str
    scores: list[CrossEncoderScore]
    original_ndcg: float  # NDCG of the original ranking
    reranked_ndcg: float  # NDCG after cross-encoder reranking
    ndcg_improvement: float  # reranked - original
    kendall_tau: float  # Rank correlation original vs reranked
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "provider": self.provider,
            "category": self.category,
            "cross_encoder_model": self.cross_encoder_model,
            "original_ndcg": self.original_ndcg,
            "reranked_ndcg": self.reranked_ndcg,
            "ndcg_improvement": self.ndcg_improvement,
            "kendall_tau": self.kendall_tau,
            "scores": [s.to_dict() for s in self.scores],
        }


@dataclass
class EmbeddingGapAnalysis:
    """Aggregate analysis of the embedding gap across categories."""

    category: str
    provider: str
    mean_ndcg_improvement: float
    median_ndcg_improvement: float
    mean_kendall_tau: float
    n_queries: int
    gap_severity: str  # "minimal", "moderate", "significant", "severe"
    validation_results: list[ValidationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "provider": self.provider,
            "mean_ndcg_improvement": self.mean_ndcg_improvement,
            "median_ndcg_improvement": self.median_ndcg_improvement,
            "mean_kendall_tau": self.mean_kendall_tau,
            "n_queries": self.n_queries,
            "gap_severity": self.gap_severity,
        }
