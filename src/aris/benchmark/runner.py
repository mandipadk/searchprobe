"""Comparative benchmark runner: Aris vs other search providers."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

from aris.benchmark.adapter import ArisSearchProvider

logger = logging.getLogger(__name__)

# Adversarial test queries organized by failure mode category
BENCHMARK_QUERIES: dict[str, list[str]] = {
    "negation": [
        "Python web frameworks NOT Django NOT Flask",
        "Electric vehicles without Tesla",
        "Programming languages that are NOT object-oriented",
    ],
    "numeric_precision": [
        "Companies with more than 500 employees founded after 2010",
        "GitHub repositories with 1000+ stars in Rust",
        "Countries with population between 10 million and 50 million",
    ],
    "temporal_constraint": [
        "AI research papers published after 2023",
        "Tech companies founded in the last 5 years",
        "JavaScript frameworks released since 2022",
    ],
    "entity_disambiguation": [
        "Michael Jordan machine learning research papers",
        "Mercury programming language documentation",
        "Apple fruit nutrition facts not Apple Inc",
    ],
    "multi_constraint": [
        "Python ML libraries with 1000+ GitHub stars released after 2023, NOT PyTorch",
        "European companies with more than 1000 employees in renewable energy founded before 2010",
    ],
    "boolean_logic": [
        "Programming languages that support both functional and object-oriented paradigms but NOT Java",
        "Databases that are either document-based or graph-based, NOT relational",
    ],
}


async def run_benchmark(
    providers: list[str] | None = None,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Run benchmark comparing Aris against other providers.

    Args:
        providers: Provider names to compare (default: ["aris", "exa"]).
        categories: Categories to test (default: all).

    Returns:
        Benchmark results dict with per-provider, per-category scores.
    """
    from searchprobe.providers.models import SearchRequest

    categories = categories or list(BENCHMARK_QUERIES.keys())
    results: dict[str, dict[str, Any]] = {}

    # Initialize providers
    provider_instances = {}
    aris = ArisSearchProvider()
    provider_instances["aris"] = aris

    if providers:
        for name in providers:
            if name != "aris":
                try:
                    provider_instances[name] = _create_searchprobe_provider(name)
                except Exception as e:
                    logger.warning("Could not create provider %s: %s", name, e)

    for category in categories:
        queries = BENCHMARK_QUERIES.get(category, [])
        if not queries:
            continue

        category_results: dict[str, list[dict]] = {p: [] for p in provider_instances}

        for query in queries:
            request = SearchRequest(
                query=query, num_results=10, include_content=True
            )

            for pname, provider in provider_instances.items():
                try:
                    start = time.perf_counter()
                    response = await provider.search(request)
                    latency = (time.perf_counter() - start) * 1000

                    category_results[pname].append({
                        "query": query,
                        "num_results": len(response.results),
                        "latency_ms": latency,
                        "success": response.success,
                        "error": response.error,
                    })
                except Exception as e:
                    category_results[pname].append({
                        "query": query,
                        "num_results": 0,
                        "latency_ms": 0,
                        "success": False,
                        "error": str(e),
                    })

        results[category] = category_results

    # Clean up
    for provider in provider_instances.values():
        await provider.close()

    return results


def _create_searchprobe_provider(name: str):
    """Create a SearchProbe provider by name."""
    import os
    from searchprobe.providers.registry import create_provider
    return create_provider(name)
