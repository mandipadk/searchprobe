"""Direct web crawling data source using httpx + trafilatura."""

from __future__ import annotations

import logging
from collections.abc import Callable
from urllib.parse import urljoin

import httpx

from aris.core.models import Document
from aris.sources.base import DataSource

logger = logging.getLogger(__name__)

MAX_REDIRECTS = 5


def is_public_http_url(url: str) -> bool:
    """Reject non-HTTP schemes and hosts that resolve to private/loopback/link-local IPs."""
    import ipaddress
    import socket
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parts.hostname, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
    return True


class WebSource(DataSource):
    """Fetch and extract content from URLs directly."""

    NAME = "web"

    def __init__(
        self, max_retries: int = 2, url_guard: Callable[[str], bool] | None = None
    ) -> None:
        """``url_guard``, when set, must accept every URL fetched, including each
        redirect target (redirects are then followed manually, not by httpx)."""
        super().__init__(max_retries=max_retries)
        self._client: httpx.AsyncClient | None = None
        self.url_guard = url_guard

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=self.url_guard is None,
                headers={"User-Agent": "Aris/0.1 (search engine)"},
            )
        return self._client

    async def _get(self, client: httpx.AsyncClient, url: str) -> httpx.Response:
        if self.url_guard is None:
            return await client.get(url)
        for _ in range(MAX_REDIRECTS + 1):
            if not self.url_guard(url):
                raise ValueError(f"URL refused by guard: {url}")
            response = await client.get(url)
            if not response.is_redirect:
                return response
            url = urljoin(url, response.headers.get("location", ""))
        raise ValueError(f"Too many redirects: {url}")

    async def fetch_url(self, url: str) -> Document | None:
        """Fetch and extract content from a single URL."""
        try:
            client = await self._get_client()
            response = await self._get(client, url)
            response.raise_for_status()
            html = response.text

            try:
                import trafilatura

                content = trafilatura.extract(html) or ""
                metadata = trafilatura.extract(html, output_format="json", include_links=False)
            except ImportError:
                content = html[:5000]
                metadata = None

            title = ""
            if metadata:
                import json

                try:
                    meta_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
                    title = meta_dict.get("title", "")
                except (json.JSONDecodeError, AttributeError):
                    pass

            if not title:
                # Try extracting title from HTML
                start = html.find("<title>")
                end = html.find("</title>")
                if start != -1 and end != -1:
                    title = html[start + 7 : end].strip()

            return Document(
                url=url,
                title=title,
                content=content,
                snippet=content[:300] if content else "",
                source=self.NAME,
            )
        except Exception as e:
            logger.debug("Failed to fetch %s: %s", url, e)
            return None

    async def _fetch(self, query: str, num_results: int) -> list[Document]:
        # WebSource is not a search engine -- it fetches known URLs.
        # Used by the ingestion pipeline, not for query-time retrieval.
        return []

    async def fetch_urls(self, urls: list[str]) -> list[Document]:
        """Fetch and extract content from multiple URLs."""
        import asyncio

        tasks = [self.fetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, Document)]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
