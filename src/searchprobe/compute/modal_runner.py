"""Thin wrapper for calling SearchProbe Modal functions from Python.

Usage:
    from searchprobe.compute.modal_runner import ModalRunner

    runner = ModalRunner()
    report = runner.geometry(models=["all-MiniLM-L6-v2"])
    validation = runner.validate(query="test query", results=[...])
    embeddings = runner.encode(texts=["hello", "world"])
"""

from __future__ import annotations

from typing import Any


class ModalRunner:
    """Python wrapper for SearchProbe Modal GPU functions.

    Provides a clean API that mirrors the local analyzer interfaces but
    runs computation remotely on Modal GPUs.

    Requires the ``modal`` package to be installed::

        pip install "searchprobe[modal]"
    """

    def __init__(self) -> None:
        try:
            import modal  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'modal' package is required for remote GPU computation. "
                "Install it with: pip install 'searchprobe[modal]'"
            ) from None
        self._app_ref: Any = None

    def _get_functions(self) -> Any:
        """Lazy-import the Modal app and return function references."""
        if self._app_ref is None:
            import modal
            self._app_ref = modal.App.lookup("searchprobe")
        return self._app_ref

    def geometry(
        self,
        models: list[str] | None = None,
        categories: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run embedding geometry analysis on Modal GPU.

        Args:
            models: Sentence-transformer model names.
                Defaults to ["all-MiniLM-L6-v2", "all-mpnet-base-v2"].
            categories: Categories to analyze. Defaults to all 13.

        Returns:
            Dict with vulnerability_matrix, profiles, and per-model details.
        """
        import modal
        analyze_fn = modal.Function.from_name("searchprobe", "analyze_geometry")
        return analyze_fn.remote(models=models, categories=categories)

    def validate(
        self,
        query: str,
        results: list[dict[str, str]],
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    ) -> dict[str, Any]:
        """Validate search results using cross-encoder on Modal GPU.

        Args:
            query: The search query.
            results: List of result dicts with title, url, snippet, optional content.
            model_name: Cross-encoder model name.

        Returns:
            Dict with scores, original_ndcg, reranked_ndcg, ndcg_improvement, kendall_tau.
        """
        import modal
        validate_fn = modal.Function.from_name("searchprobe", "validate_cross_encoder")
        return validate_fn.remote(
            query_text=query, results=results, model_name=model_name,
        )

    def batch_validate(
        self,
        items: list[dict[str, Any]],
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    ) -> list[dict[str, Any]]:
        """Batch cross-encoder validation on Modal GPU.

        Args:
            items: List of dicts with query_text, results, category, provider.
            model_name: Cross-encoder model name.

        Returns:
            List of validation result dicts.
        """
        import modal
        batch_fn = modal.Function.from_name("searchprobe", "batch_validate")
        return batch_fn.remote(items=items, model_name=model_name)

    def encode(
        self,
        texts: list[str],
        model_name: str = "all-MiniLM-L6-v2",
    ) -> list[list[float]]:
        """Generate embeddings on Modal GPU.

        Args:
            texts: Texts to encode.
            model_name: Sentence-transformer model name.

        Returns:
            List of embedding vectors (nested lists of floats).
        """
        import modal
        encode_fn = modal.Function.from_name("searchprobe", "encode_texts")
        return encode_fn.remote(texts=texts, model_name=model_name)

    def compute_fitness(
        self,
        queries: list[str],
        search_results: list[list[dict[str, str]]],
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    ) -> list[dict[str, Any]]:
        """Compute adversarial fitness using cross-encoder on Modal GPU.

        Args:
            queries: List of adversarial query strings.
            search_results: List of result lists (one per query).
            model_name: Cross-encoder model name.

        Returns:
            List of dicts with fitness, ndcg_improvement, original_ndcg, reranked_ndcg.
        """
        import modal
        fitness_fn = modal.Function.from_name("searchprobe", "compute_fitness_batch")
        return fitness_fn.remote(
            queries=queries, search_results=search_results, model_name=model_name,
        )
