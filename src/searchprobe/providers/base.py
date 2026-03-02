"""Abstract base class and protocol for search providers."""

from abc import ABC, abstractmethod
from typing import ClassVar

from searchprobe.providers.models import SearchRequest, SearchResponse


class SearchProvider(ABC):
    """Abstract base class for all search providers.

    Each provider must implement the search method and provide
    metadata about the provider (name, supported modes, costs).
    """

    # Provider metadata - override in subclasses
    NAME: ClassVar[str] = "base"
    SUPPORTED_MODES: ClassVar[list[str]] = []
    COST_PER_QUERY: ClassVar[dict[str, float]] = {}  # mode -> cost in USD

    def __init__(self, api_key: str) -> None:
        """Initialize the provider with an API key."""
        self.api_key = api_key

    @property
    def name(self) -> str:
        """Return the provider name."""
        return self.NAME

    @abstractmethod
    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute a search and return normalized results.

        Args:
            request: Normalized search request

        Returns:
            Normalized search response with results
        """
        ...

    async def health_check(self) -> bool:
        """Check if the provider is accessible and API key is valid.

        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            # Try a simple search
            response = await self.search(
                SearchRequest(query="test", num_results=1, include_content=False)
            )
            return response.success
        except Exception:
            return False

    def get_cost(self, mode: str | None = None) -> float:
        """Get the cost per query for a specific mode.

        Args:
            mode: Search mode (uses default if None)

        Returns:
            Cost in USD per query
        """
        if mode is None:
            mode = self.SUPPORTED_MODES[0] if self.SUPPORTED_MODES else "default"
        return self.COST_PER_QUERY.get(mode, 0.005)  # Default $0.005 per query
