"""Brave Search API data source."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from aris.core.exceptions import RateLimitError, SourceError
from aris.core.models import Document
from aris.sources.base import DataSource

logger = logging.getLogger(__name__)


class BraveSource(DataSource):
    """Brave Search API data source."""

    NAME = "brave"
    API_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str, max_retries: int = 3) -> None:
        super().__init__(max_retries=max_retries)
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": self._api_key,
                },
            )
        return self._client

    async def _fetch(self, query: str, num_results: int) -> list[Document]:
        client = await self._get_client()

        params: dict[str, Any] = {
            "q": query,
            "count": min(num_results, 20),
        }

        try:
            response = await client.get(self.API_ENDPOINT, params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(str(e), source=self.NAME)
            raise SourceError(str(e), source=self.NAME, status_code=e.response.status_code)

        data = response.json()
        web_results = data.get("web", {}).get("results", [])

        documents = []
        for r in web_results:
            documents.append(
                Document(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    snippet=r.get("description", ""),
                    content=r.get("description", ""),
                    source=self.NAME,
                )
            )
        return documents

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
