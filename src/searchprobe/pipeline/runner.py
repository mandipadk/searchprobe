"""Pipeline runner for orchestrating benchmark execution."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from searchprobe.config import Settings, get_settings
from searchprobe.pipeline.cost_tracker import CostTracker
from searchprobe.pipeline.rate_limiter import RateLimiterPool
from searchprobe.providers.base import SearchProvider
from searchprobe.providers.models import SearchRequest, SearchResponse
from searchprobe.providers.registry import ProviderRegistry
from searchprobe.storage import Database


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    providers: list[str]
    exa_modes: list[str] = field(default_factory=lambda: ["auto"])
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

        # Provider instances
        self._providers: dict[str, SearchProvider] = {}

        # Progress callback
        self._progress_callback: Callable[[str, int, int], None] | None = None

    def set_progress_callback(
        self, callback: Callable[[str, int, int], None]
    ) -> None:
        """Set a callback for progress updates.

        Args:
            callback: Function(status: str, completed: int, total: int)
        """
        self._progress_callback = callback

    def _get_provider(self, name: str) -> SearchProvider:
        """Get or create a provider instance."""
        if name not in self._providers:
            self._providers[name] = ProviderRegistry.get_provider(name, self.settings)
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
        started_at = datetime.utcnow()

        # Create run in database if not provided
        if run_id is None:
            run_id = self.db.create_run(
                query_set_id=queries[0].get("query_set_id", "unknown"),
                name=run_name,
                config={
                    "providers": self.config.providers,
                    "exa_modes": self.config.exa_modes,
                    "num_results": self.config.num_results,
                    "budget_limit": self.config.budget_limit,
                },
            )

        # Build task list
        tasks = self._build_tasks(queries)
        total_tasks = len(tasks)

        # Execute tasks with concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        results: list[SearchResponse] = []
        completed = 0

        async def execute_task(task: QueryTask) -> SearchResponse | None:
            nonlocal completed

            # Check budget
            if self.cost_tracker.is_budget_exceeded():
                return None

            # Rate limit
            await self.rate_limiters.acquire(task.provider_name)

            # Execute search
            async with semaphore:
                response = await self._execute_search(task)

                # Track cost
                if response.cost_usd > 0:
                    self.cost_tracker.record(
                        provider=task.provider_name,
                        operation=f"search_{task.search_mode or 'default'}",
                        cost_usd=response.cost_usd,
                        metadata={"query_id": task.query_id},
                    )

                # Store result
                if response.success or response.error:
                    self.db.add_search_result(run_id, task.query_id, response)

                # Update progress
                completed += 1
                if self._progress_callback:
                    self._progress_callback(
                        f"Searching: {task.query_text[:50]}...",
                        completed,
                        total_tasks,
                    )

                return response

        # Run all tasks
        task_results = await asyncio.gather(
            *[execute_task(t) for t in tasks], return_exceptions=True
        )

        # Collect valid results
        for result in task_results:
            if isinstance(result, SearchResponse):
                results.append(result)
            elif isinstance(result, Exception):
                # Log error but continue
                pass

        # Complete run
        completed_at = datetime.utcnow()
        self.db.complete_run(
            run_id,
            self.cost_tracker.get_total(),
            self.cost_tracker.get_total_by_provider(),
        )

        # Clean up providers
        await self._cleanup_providers()

        # Calculate stats
        successful = len([r for r in results if r.success])
        failed = len([r for r in results if r.error])

        return BenchmarkResult(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            total_queries=len(tasks),
            successful_queries=successful,
            failed_queries=failed,
            total_cost=self.cost_tracker.get_total(),
            cost_by_provider=self.cost_tracker.get_total_by_provider(),
            results=results,
        )

    def _build_tasks(self, queries: list[dict[str, Any]]) -> list[QueryTask]:
        """Build list of tasks to execute."""
        tasks = []

        for query in queries:
            for provider in self.config.providers:
                # For Exa, run each mode as a separate task
                if provider == "exa":
                    for mode in self.config.exa_modes:
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

    async def _execute_search(self, task: QueryTask) -> SearchResponse:
        """Execute a single search task."""
        provider = self._get_provider(task.provider_name)

        request = SearchRequest(
            query=task.query_text,
            num_results=self.config.num_results,
            include_content=self.config.include_content,
            max_content_chars=self.config.max_content_chars,
            search_mode=task.search_mode,
        )

        return await provider.search(request)


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
        raise ValueError("No providers configured")

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
