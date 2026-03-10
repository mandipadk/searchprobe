"""Source registry -- factory for creating and managing data sources."""

from __future__ import annotations

import logging

from aris.core.config import ArisConfig
from aris.sources.base import DataSource

logger = logging.getLogger(__name__)

# Registry of source name -> class
_SOURCE_CLASSES: dict[str, type[DataSource]] = {}


def register_source(cls: type[DataSource]) -> type[DataSource]:
    """Register a data source class."""
    _SOURCE_CLASSES[cls.NAME] = cls
    return cls


def _register_defaults() -> None:
    """Lazily register built-in sources."""
    if _SOURCE_CLASSES:
        return

    from aris.sources.brave import BraveSource
    from aris.sources.duckduckgo import DuckDuckGoSource
    from aris.sources.serpapi import SerpAPISource

    _SOURCE_CLASSES["duckduckgo"] = DuckDuckGoSource
    _SOURCE_CLASSES["brave"] = BraveSource
    _SOURCE_CLASSES["serpapi"] = SerpAPISource


class SourceRegistry:
    """Creates and manages data source instances."""

    def __init__(self, config: ArisConfig) -> None:
        _register_defaults()
        self._config = config
        self._sources: dict[str, DataSource] = {}

    def get(self, name: str) -> DataSource | None:
        """Get or create a data source by name."""
        if name in self._sources:
            return self._sources[name]

        source = self._create(name)
        if source:
            self._sources[name] = source
        return source

    def _create(self, name: str) -> DataSource | None:
        cls = _SOURCE_CLASSES.get(name)
        if cls is None:
            logger.warning("Unknown source: %s", name)
            return None

        if name == "brave":
            if not self._config.brave_api_key:
                logger.warning("Brave API key not configured, skipping")
                return None
            return cls(api_key=self._config.brave_api_key)
        elif name == "serpapi":
            if not self._config.serpapi_api_key:
                logger.warning("SerpAPI key not configured, skipping")
                return None
            return cls(api_key=self._config.serpapi_api_key)
        else:
            return cls()

    def get_available(self) -> list[DataSource]:
        """Get all available sources (those with valid config)."""
        available = []
        for name in _SOURCE_CLASSES:
            source = self.get(name)
            if source:
                available.append(source)
        return available

    async def close_all(self) -> None:
        """Close all active sources."""
        for source in self._sources.values():
            await source.close()
        self._sources.clear()
