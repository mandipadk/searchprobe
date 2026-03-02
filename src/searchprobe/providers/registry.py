"""Provider registry and factory."""

from typing import Any

from searchprobe.config import Settings, get_settings
from searchprobe.providers.base import SearchProvider
from searchprobe.providers.brave import BraveProvider
from searchprobe.providers.exa import ExaProvider
from searchprobe.providers.serpapi import SerpAPIProvider
from searchprobe.providers.tavily import TavilyProvider


class ProviderRegistry:
    """Registry and factory for search providers.

    Handles provider instantiation and configuration.
    """

    # Map of provider names to classes
    PROVIDERS: dict[str, type[SearchProvider]] = {
        "exa": ExaProvider,
        "tavily": TavilyProvider,
        "brave": BraveProvider,
        "serpapi": SerpAPIProvider,
    }

    # Map of provider names to settings API key attribute names
    API_KEY_ATTRS: dict[str, str] = {
        "exa": "exa_api_key",
        "tavily": "tavily_api_key",
        "brave": "brave_api_key",
        "serpapi": "serpapi_api_key",
    }

    @classmethod
    def get_provider(
        cls, name: str, settings: Settings | None = None
    ) -> SearchProvider:
        """Get a configured provider instance by name.

        Args:
            name: Provider name (exa, tavily, brave, serpapi)
            settings: Settings instance (uses default if None)

        Returns:
            Configured provider instance

        Raises:
            ValueError: If provider not found or not configured
        """
        if name not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider: {name}. Available: {list(cls.PROVIDERS.keys())}")

        if settings is None:
            settings = get_settings()

        # Get API key from settings
        api_key_attr = cls.API_KEY_ATTRS.get(name)
        if not api_key_attr:
            raise ValueError(f"No API key attribute defined for provider: {name}")

        api_key = getattr(settings, api_key_attr, None)
        if not api_key:
            raise ValueError(
                f"Provider {name} not configured. Set SEARCHPROBE_{api_key_attr.upper()} in .env"
            )

        # Instantiate provider
        provider_class = cls.PROVIDERS[name]
        return provider_class(api_key=api_key)

    @classmethod
    def get_all_configured(
        cls, settings: Settings | None = None
    ) -> dict[str, SearchProvider]:
        """Get all providers that have API keys configured.

        Args:
            settings: Settings instance (uses default if None)

        Returns:
            Dict of provider name -> configured provider instance
        """
        if settings is None:
            settings = get_settings()

        providers = {}
        for name in cls.PROVIDERS:
            try:
                providers[name] = cls.get_provider(name, settings)
            except ValueError:
                # Provider not configured, skip
                pass

        return providers

    @classmethod
    def list_available(cls) -> list[str]:
        """List all available provider names."""
        return list(cls.PROVIDERS.keys())

    @classmethod
    def list_configured(cls, settings: Settings | None = None) -> list[str]:
        """List provider names that have API keys configured."""
        if settings is None:
            settings = get_settings()
        return settings.configured_providers

    @classmethod
    def get_provider_info(cls, name: str) -> dict[str, Any]:
        """Get metadata about a provider."""
        if name not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider: {name}")

        provider_class = cls.PROVIDERS[name]
        return {
            "name": name,
            "class": provider_class.__name__,
            "supported_modes": provider_class.SUPPORTED_MODES,
            "cost_per_query": provider_class.COST_PER_QUERY,
        }
