"""Pipeline runner for orchestrating benchmark execution."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from searchprobe.config import Settings, get_settings
from searchprobe.core.exceptions import PipelineError
from searchprobe.manifest import build_manifest
from searchprobe.pipeline.cost_tracker import CostTracker
from searchprobe.pipeline.rate_limiter import RateLimiterPool
from searchprobe.providers.models import SearchRequest, SearchResponse
from searchprobe.providers.registry import ProviderRegistry
from searchprobe.providers.resilient import ResilientProvider
from searchprobe.storage import Database


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    providers: list[str]
    exa_modes: list[str] = field(default_factory=lambda: ["auto"])
    # Per-provider modes for Exa-backed providers; overrides exa_modes for listed ones
    # (a profile with exa:neural and aris+exa:auto must not run exa:auto too).
    modes_by_provider: dict[str, list[str]] = field(default_factory=dict)
    num_results: int = 10
    include_content: bool = True
    max_content_chars: int = 5000
    max_concurrent: int = 5
    budget_limit: float = 10.0


@dataclass
class QueryTask:
    """A single query to execute against a provider."""

    query_id: str
    query_text: str
    category: str
    provider_name: str
    search_mode: str | None = None


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    run_id: str
    started_at: datetime
    completed_at: datetime
    total_queries: int
    successful_queries: int
    failed_queries: int
    total_cost: float
    cost_by_provider: dict[str, float]
    results: list[SearchResponse]
    empty_queries: int = 0
    skipped_for_budget: int = 0
    status: str = "completed"  # completed | partial (budget exhausted)


class BenchmarkRunner:
    """Orchestrates benchmark execution across multiple providers.

    Features:
    - Concurrent execution with configurable parallelism
    - Per-provider rate limiting
    - Cost tracking and budget enforcement
    - Progress callbacks
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        settings: Settings | None = None,
        db: Database | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            config: Benchmark configuration
            settings: Application settings
            db: Database for storing results
        """
        self.config = config
        self.settings = settings or get_settings()
        self.db = db or Database(self.settings.database_path)

        # Initialize components
        self.rate_limiters = RateLimiterPool()
        self.cost_tracker = CostTracker(budget_limit=config.budget_limit)

        # Provider instances (wrapped with resilience)
        self._providers: dict[str, ResilientProvider] = {}

        # Progress callback
        self._progress_callback: Callable[[str, int, int], None] | None = None

    def set_progress_callback(self, callback: Callable[[str, int, int], None]) -> None:
        """Set a callback for progress updates.

        Args:
            callback: Function(status: str, completed: int, total: int)
        """
        self._progress_callback = callback

    def _get_provider(self, name: str) -> ResilientProvider:
        """Get or create a resilient, rate-limited provider instance."""
        if name not in self._providers:
            raw_provider = ProviderRegistry.get_provider(name, self.settings)
            self._providers[name] = ResilientProvider(
                raw_provider,
                before_attempt=lambda n=name: self.rate_limiters.acquire(n),
            )
        return self._providers[name]

    async def run(
        self,
        queries: list[dict[str, Any]],
        run_id: str | None = None,
        run_name: str | None = None,
    ) -> BenchmarkResult:
        """Execute the benchmark.

        Args:
            queries: List of query dicts with 'id', 'text', 'category'
            run_id: Optional pre-created run ID
            run_name: Name for the run

        Returns:
            BenchmarkResult with all responses
        """
        if not queries:
            raise ValueError("No queries to run")
        started_at = datetime.now(UTC)

        # Create run in database if not provided
        if run_id is None:
            run_id = self.db.create_run(
                query_set_id=queries[0].get("query_set_id", "unknown"),
                name=run_name,
                config={
                    "providers": self.config.providers,
                    "exa_modes": self.config.exa_modes,
                    "modes_by_provider": self.config.modes_by_provider,
                    "num_results": self.config.num_results,
                    "budget_limit": self.config.budget_limit,
                    "manifest": build_manifest(
                        self.settings,
                        n_queries=len(queries),
                        query_set_id=queries[0].get("query_set_id"),
                    ),
                },
            )

        # Build task list
        tasks = self._build_tasks(queries)
        total_tasks = len(tasks)

        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        results: list[SearchResponse] = []
        completed = 0
        skipped_for_budget = 0

        async def execute_task(task: QueryTask) -> SearchResponse | None:
            nonlocal completed, skipped_for_budget
            async with semaphore:
                request = self._build_request(task)
                try:
                    provider = self._get_provider(task.provider_name)
                except Exception as e:  # e.g. missing API key: fail this task, not the run
                    response = SearchResponse(
                        provider=task.provider_name,
                        search_mode=task.search_mode,
                        query=task.query_text,
                        error=f"provider unavailable: {type(e).__name__}: {e}",
                    )
                    self.db.add_search_result(run_id, task.query_id, response)
                    completed += 1
                    return response
                # Reserve the estimated cost *before* dispatch so concurrent tasks
                # cannot collectively overspend the budget.
                reservation = await self.cost_tracker.reserve(provider.estimate_cost(request))
                if reservation is None:
                    skipped_for_budget += 1
                    return None
                try:
                    response = await provider.search(request)
                except Exception as e:  # never lose a task silently
                    response = SearchResponse(
                        provider=task.provider_name,
                        search_mode=task.search_mode,
                        query=task.query_text,
                        error=f"{type(e).__name__}: {e}",
                    )
                self.cost_tracker.commit(
                    reservation,
                    provider=task.provider_name,
                    operation=f"search_{task.search_mode or 'default'}",
                    cost_usd=response.cost_usd,
                    metadata={"query_id": task.query_id},
                )
                # Store every response, including empty result sets: an empty
                # list is a retrieval failure, not a missing data point.
                self.db.add_search_result(run_id, task.query_id, response)

                completed += 1
                if self._progress_callback:
                    self._progress_callback(
                        f"Searching: {task.query_text[:50]}...", completed, total_tasks
                    )
                return response

        status = "failed"
        try:
            task_results = await asyncio.gather(*[execute_task(t) for t in tasks])
            results = [r for r in task_results if r is not None]
            status = "partial" if skipped_for_budget else "completed"
        finally:  # the run is closed and providers released even if a task raises
            self.db.complete_run(
                run_id,
                self.cost_tracker.get_total(),
                self.cost_tracker.get_total_by_provider(),
                status=status,
            )
            await self._cleanup_providers()
        completed_at = datetime.now(UTC)

        return BenchmarkResult(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            total_queries=len(tasks),
            successful_queries=len([r for r in results if r.success]),
            failed_queries=len([r for r in results if r.error]),
            empty_queries=len([r for r in results if not r.error and not r.results]),
            skipped_for_budget=skipped_for_budget,
            status=status,
            total_cost=self.cost_tracker.get_total(),
            cost_by_provider=self.cost_tracker.get_total_by_provider(),
            results=results,
        )

    def _build_tasks(self, queries: list[dict[str, Any]]) -> list[QueryTask]:
        """Build list of tasks to execute."""
        tasks = []

        for query in queries:
            for provider in self.config.providers:
                # For Exa (bare or wrapped, e.g. aris+exa), run each mode separately
                if provider == "exa" or provider.endswith("+exa"):
                    modes = self.config.modes_by_provider.get(provider, self.config.exa_modes)
                    for mode in modes:
                        tasks.append(
                            QueryTask(
                                query_id=query.get("id", "unknown"),
                                query_text=query["text"],
                                category=query.get("category", "unknown"),
                                provider_name=provider,
                                search_mode=mode,
                            )
                        )
                else:
                    tasks.append(
                        QueryTask(
                            query_id=query.get("id", "unknown"),
                            query_text=query["text"],
                            category=query.get("category", "unknown"),
                            provider_name=provider,
                            search_mode=None,
                        )
                    )

        return tasks

    async def _cleanup_providers(self) -> None:
        """Clean up all provider resources."""
        for provider in self._providers.values():
            try:
                await provider.close()
            except Exception:
                pass
        self._providers.clear()

    def _build_request(self, task: QueryTask) -> SearchRequest:
        """Build the provider request for a task."""
        return SearchRequest(
            query=task.query_text,
            num_results=self.config.num_results,
            include_content=self.config.include_content,
            max_content_chars=self.config.max_content_chars,
            search_mode=task.search_mode,
        )


async def run_benchmark(
    queries: list[dict[str, Any]],
    providers: list[str] | None = None,
    exa_modes: list[str] | None = None,
    num_results: int = 10,
    budget_limit: float = 10.0,
    run_name: str | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> BenchmarkResult:
    """Convenience function to run a benchmark.

    Args:
        queries: List of query dicts
        providers: Providers to test (default: all configured)
        exa_modes: Exa search modes (default: ["auto"])
        num_results: Results per query
        budget_limit: Maximum budget in USD
        run_name: Name for the run
        progress_callback: Progress update callback

    Returns:
        BenchmarkResult
    """
    settings = get_settings()

    if providers is None:
        providers = settings.configured_providers

    if not providers:
        raise PipelineError("No providers configured")

    if exa_modes is None:
        exa_modes = ["auto"]

    config = BenchmarkConfig(
        providers=providers,
        exa_modes=exa_modes,
        num_results=num_results,
        budget_limit=budget_limit,
    )

    runner = BenchmarkRunner(config, settings)
    if progress_callback:
        runner.set_progress_callback(progress_callback)

    return await runner.run(queries, run_name=run_name)
