"""Cost tracking for API usage."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class CostRecord:
    """Record of a single API cost."""

    provider: str
    operation: str  # e.g., "search", "content"
    cost_usd: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class CostTracker:
    """Tracks API costs per provider and operation."""

    # Approximate costs per operation (USD)
    COST_TABLE: dict[str, dict[str, float]] = {
        "exa": {
            "search_auto": 0.005,
            "search_neural": 0.005,
            "search_fast": 0.005,
            "search_deep": 0.015,
            "content": 0.001,  # per page
        },
        "tavily": {
            "search_basic": 0.005,
            "search_advanced": 0.01,
        },
        "brave": {
            "search_web": 0.005,
        },
        "serpapi": {
            "search_google": 0.01,
        },
    }

    def __init__(self, budget_limit: float | None = None) -> None:
        """Initialize the cost tracker.

        Args:
            budget_limit: Maximum budget in USD (None for no limit)
        """
        self.budget_limit = budget_limit
        self.records: list[CostRecord] = []
        self._total_by_provider: dict[str, float] = {}

    def record(
        self,
        provider: str,
        operation: str,
        cost_usd: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a cost.

        Args:
            provider: Provider name
            operation: Operation type (e.g., "search_auto")
            cost_usd: Cost in USD
            metadata: Additional metadata
        """
        record = CostRecord(
            provider=provider,
            operation=operation,
            cost_usd=cost_usd,
            metadata=metadata or {},
        )
        self.records.append(record)

        # Update running total
        if provider not in self._total_by_provider:
            self._total_by_provider[provider] = 0.0
        self._total_by_provider[provider] += cost_usd

    def get_total(self) -> float:
        """Get total cost across all providers."""
        return sum(self._total_by_provider.values())

    def get_total_by_provider(self) -> dict[str, float]:
        """Get cost breakdown by provider."""
        return self._total_by_provider.copy()

    def get_total_by_operation(self) -> dict[str, float]:
        """Get cost breakdown by operation."""
        by_operation: dict[str, float] = {}
        for record in self.records:
            key = f"{record.provider}:{record.operation}"
            if key not in by_operation:
                by_operation[key] = 0.0
            by_operation[key] += record.cost_usd
        return by_operation

    def is_budget_exceeded(self) -> bool:
        """Check if budget limit has been exceeded."""
        if self.budget_limit is None:
            return False
        return self.get_total() >= self.budget_limit

    def remaining_budget(self) -> float | None:
        """Get remaining budget, or None if no limit."""
        if self.budget_limit is None:
            return None
        return max(0.0, self.budget_limit - self.get_total())

    def estimate_operation_cost(self, provider: str, operation: str) -> float:
        """Estimate cost for an operation.

        Args:
            provider: Provider name
            operation: Operation type

        Returns:
            Estimated cost in USD
        """
        provider_costs = self.COST_TABLE.get(provider, {})
        return provider_costs.get(operation, 0.005)  # Default $0.005

    def estimate_remaining_queries(self, provider: str, operation: str) -> int | None:
        """Estimate how many more queries can be run within budget.

        Args:
            provider: Provider name
            operation: Operation type

        Returns:
            Estimated number of queries, or None if no budget limit
        """
        remaining = self.remaining_budget()
        if remaining is None:
            return None

        cost_per_query = self.estimate_operation_cost(provider, operation)
        if cost_per_query <= 0:
            return None

        return int(remaining / cost_per_query)

    def to_dict(self) -> dict[str, Any]:
        """Export tracker state as dictionary."""
        return {
            "total_cost": self.get_total(),
            "budget_limit": self.budget_limit,
            "remaining_budget": self.remaining_budget(),
            "by_provider": self.get_total_by_provider(),
            "by_operation": self.get_total_by_operation(),
            "record_count": len(self.records),
        }
