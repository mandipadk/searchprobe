"""Cross-encoder validation for re-scoring search results."""

import math
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from searchprobe.validation.models import CrossEncoderScore, ValidationResult


DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-12-v2"


class CrossEncoderValidator:
    """Validates search results using a cross-encoder model.

    Cross-encoders jointly encode (query, document) pairs, producing much more
    accurate relevance scores than bi-encoders. By re-scoring search results
    with a cross-encoder, we quantify the "embedding gap" — how much relevance
    is lost by using bi-encoder approximation.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        device: str | None = None,
    ) -> None:
        """Initialize the cross-encoder validator.

        Args:
            model_name: HuggingFace cross-encoder model name
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    def _load_model(self) -> Any:
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            kwargs: dict[str, Any] = {}
            if self.device:
                kwargs["device"] = self.device
            self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def score_results(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> list[CrossEncoderScore]:
        """Score search results using the cross-encoder.

        Args:
            query: The search query
            results: List of result dicts with 'title', 'url', 'snippet', optional 'content'

        Returns:
            List of CrossEncoderScore objects with cross-encoder scores
        """
        model = self._load_model()

        # Build (query, document) pairs for cross-encoder
        pairs = []
        for result in results:
            doc_text = result.get("content") or result.get("snippet", "")
            # Truncate to avoid memory issues
            if len(doc_text) > 512:
                doc_text = doc_text[:512]
            pairs.append((query, doc_text))

        # Score all pairs
        scores = model.predict(pairs)
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)

        # Build CrossEncoderScore objects
        ce_scores = []
        for i, (result, score) in enumerate(zip(results, scores)):
            ce_scores.append(
                CrossEncoderScore(
                    query=query,
                    document_title=result.get("title", ""),
                    document_url=result.get("url", ""),
                    original_rank=i,
                    cross_encoder_score=float(score),
                )
            )

        # Compute reranked positions
        sorted_by_score = sorted(
            range(len(ce_scores)),
            key=lambda i: ce_scores[i].cross_encoder_score,
            reverse=True,
        )
        for new_rank, orig_idx in enumerate(sorted_by_score):
            ce_scores[orig_idx].reranked_position = new_rank

        return ce_scores

    def validate_search_results(
        self,
        query_id: str,
        query_text: str,
        category: str,
        provider: str,
        results: list[dict[str, Any]],
    ) -> ValidationResult:
        """Validate search results and compute embedding gap metrics.

        Args:
            query_id: Query identifier
            query_text: The search query
            category: Adversarial category
            provider: Search provider name
            results: List of search result dicts

        Returns:
            ValidationResult with NDCG and rank correlation metrics
        """
        if not results:
            return ValidationResult(
                query_id=query_id,
                query_text=query_text,
                provider=provider,
                category=category,
                cross_encoder_model=self.model_name,
                scores=[],
                original_ndcg=0.0,
                reranked_ndcg=0.0,
                ndcg_improvement=0.0,
                kendall_tau=0.0,
            )

        scores = self.score_results(query_text, results)

        # Compute NDCG for original ranking
        original_relevance = [s.cross_encoder_score for s in scores]
        original_ndcg = _compute_ndcg(original_relevance)

        # Compute NDCG for reranked (optimal) ranking
        sorted_relevance = sorted(original_relevance, reverse=True)
        reranked_ndcg = _compute_ndcg(sorted_relevance)

        # Compute Kendall's tau rank correlation
        original_ranks = list(range(len(scores)))
        reranked_ranks = [s.reranked_position for s in scores]
        if len(scores) >= 2:
            tau, _ = scipy_stats.kendalltau(original_ranks, reranked_ranks)
            kendall_tau = float(tau) if not math.isnan(tau) else 0.0
        else:
            kendall_tau = 1.0

        ndcg_improvement = reranked_ndcg - original_ndcg

        return ValidationResult(
            query_id=query_id,
            query_text=query_text,
            provider=provider,
            category=category,
            cross_encoder_model=self.model_name,
            scores=scores,
            original_ndcg=original_ndcg,
            reranked_ndcg=reranked_ndcg,
            ndcg_improvement=ndcg_improvement,
            kendall_tau=kendall_tau,
        )


def _compute_ndcg(relevance_scores: list[float], k: int | None = None) -> float:
    """Compute Normalized Discounted Cumulative Gain.

    Args:
        relevance_scores: Relevance scores in current ranking order
        k: Number of results to consider (default: all)

    Returns:
        NDCG score in [0, 1]
    """
    if not relevance_scores:
        return 0.0

    if k is not None:
        relevance_scores = relevance_scores[:k]

    # DCG
    dcg = 0.0
    for i, rel in enumerate(relevance_scores):
        dcg += rel / math.log2(i + 2)  # +2 because positions are 1-indexed

    # Ideal DCG (scores sorted descending)
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_scores):
        idcg += rel / math.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg
