"""Direct web crawling data source using httpx + trafilatura."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from aris.core.models import Document
from aris.sources.base import DataSource

logger = logging.getLogger(__name__)

MAX_REDIRECTS = 5


def _system_resolve(host: str) -> list[str]:
    import socket

    return [info[4][0] for info in socket.getaddrinfo(host, None)]


def _is_public_ip(address: str) -> bool:
    import ipaddress

    ip = ipaddress.ip_address(address)
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)


def is_public_http_url(url: str, resolve: Callable[[str], list[str]] = _system_resolve) -> bool:
    """Reject non-HTTP schemes and hosts that resolve to private/loopback/link-local IPs."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    try:
        addresses = resolve(parts.hostname)
    except OSError:
        return False
    return bool(addresses) and all(_is_public_ip(a) for a in addresses)


class WebSource(DataSource):
    """Fetch and extract content from URLs directly."""

    NAME = "web"

    def __init__(
        self,
        max_retries: int = 2,
        public_only: bool = False,
        resolve: Callable[[str], list[str]] = _system_resolve,
    ) -> None:
        """With ``public_only``, every request (including each redirect hop) goes to a
        public IP: the host is resolved once, off the event loop, every address is
        checked, and the connection is pinned to the checked address, so a DNS answer
        that changes between the check and the connection cannot reach a private host."""
        super().__init__(max_retries=max_retries)
        self._client: httpx.AsyncClient | None = None
        self.public_only = public_only
        self.resolve = resolve

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=not self.public_only,
                headers={"User-Agent": "Aris/0.1 (search engine)"},
            )
        return self._client

    async def _pinned_get(self, client: httpx.AsyncClient, url: str) -> httpx.Response:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"Refusing non-HTTP URL: {url}")
        host = parts.hostname
        addresses = await asyncio.wait_for(asyncio.to_thread(self.resolve, host), timeout=10)
        if not addresses or not all(_is_public_ip(a) for a in addresses):
            raise ValueError(f"Refusing non-public host: {host}")
        ip = addresses[0]
        ip_host = f"[{ip}]" if ":" in ip else ip
        netloc = f"{ip_host}:{parts.port}" if parts.port else ip_host
        pinned = urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, ""))
        extensions = {"sni_hostname": host} if parts.scheme == "https" else {}
        return await client.get(pinned, headers={"Host": parts.netloc}, extensions=extensions)

    async def _get(self, client: httpx.AsyncClient, url: str) -> httpx.Response:
        if not self.public_only:
            return await client.get(url)
        for _ in range(MAX_REDIRECTS + 1):
            response = await self._pinned_get(client, url)
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
        tasks = [self.fetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, Document)]

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
