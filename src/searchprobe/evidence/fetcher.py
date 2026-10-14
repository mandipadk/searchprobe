"""Fetch documents once per URL and store extracted snapshots.

Every system's results go through the same fetcher and extractor, so the
judge sees comparable evidence regardless of how much text a provider's API
happens to return.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx

from searchprobe.evidence.extract import extract
from searchprobe.evidence.store import EvidenceStore, Snapshot, content_hash
from searchprobe.perturbation.stability import normalize_url

logger = logging.getLogger(__name__)

# Wikimedia-style UA: tool/version (project URL) library. Some sites (e.g.
# Wikipedia) reject generic or malformed agents.
USER_AGENT = f"SearchProbe/3.0 (https://github.com/mandipadk/searchprobe) httpx/{httpx.__version__}"
_HTML_TYPES = ("text/html", "application/xhtml+xml", "text/plain")


def _has_h2() -> bool:
    try:
        import h2  # noqa: F401
    except ImportError:
        return False
    return True


class EvidenceFetcher:
    """Concurrent, cached, polite document fetcher."""

    def __init__(
        self,
        store: EvidenceStore,
        client: httpx.AsyncClient | None = None,
        timeout: float = 20.0,
        max_bytes: int = 3_000_000,
        concurrency: int = 8,
        per_host: int = 2,
        max_age_days: float | None = 30.0,
        user_agent: str = USER_AGENT,
        offline: bool = False,
    ) -> None:
        self.store = store
        self._client = client
        self._owns_client = client is None
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.semaphore = asyncio.Semaphore(concurrency)
        self.per_host = per_host
        self._host_semaphores: dict[str, asyncio.Semaphore] = defaultdict(
            lambda: asyncio.Semaphore(self.per_host)
        )
        self.max_age_days = max_age_days
        self.user_agent = user_agent
        # Offline: serve stored snapshots only (any age); never touch the network.
        self.offline = offline

    async def __aenter__(self) -> EvidenceFetcher:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                http2=_has_h2(),  # some sites (e.g. Wikipedia) reject HTTP/1.1 clients
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent, "Accept": "text/html,*/*;q=0.5"},
            )
        return self._client

    async def fetch(self, url: str, refresh: bool = False) -> Snapshot:
        """Return a snapshot for ``url``, fetching only if no fresh one is stored."""
        key = normalize_url(url)
        if self.offline:
            cached = self.store.get_for_url(key, None)
            if cached is not None:
                return cached
            return Snapshot(
                content_hash=content_hash("", url),
                url=url,
                status="fetch_error",
                fetched_at=datetime.now(UTC),
                error="offline: no stored snapshot",
            )
        if not refresh:
            cached = self.store.get_for_url(key, self.max_age_days)
            if cached is not None:
                return cached
        host = urlsplit(url).netloc.lower()
        async with self.semaphore, self._host_semaphores[host]:
            snapshot = await self._download(url)
        self.store.put(snapshot, key)
        return snapshot

    async def fetch_many(self, urls: list[str], refresh: bool = False) -> dict[str, Snapshot]:
        """Fetch many URLs concurrently; result is keyed by the input URL."""
        unique = list(dict.fromkeys(urls))
        snaps = await asyncio.gather(*[self.fetch(u, refresh) for u in unique])
        return dict(zip(unique, snaps))

    async def _download(self, url: str) -> Snapshot:
        now = datetime.now(UTC)

        def failure(status: str, error: str, http_status: int | None = None) -> Snapshot:
            return Snapshot(
                content_hash=content_hash("", url),
                url=url,
                status=status,  # type: ignore[arg-type]
                http_status=http_status,
                fetched_at=now,
                error=error,
            )

        try:
            async with self._get_client().stream("GET", url) as response:
                if response.status_code >= 400:
                    return failure(
                        "http_error", f"HTTP {response.status_code}", response.status_code
                    )
                ctype = response.headers.get("content-type", "").lower()
                if ctype and not any(t in ctype for t in _HTML_TYPES):
                    return failure("unsupported", f"content-type {ctype}", response.status_code)
                chunks: list[bytes] = []
                size = 0
                async for chunk in response.aiter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size >= self.max_bytes:
                        break
                raw = b"".join(chunks)
                encoding = response.encoding or "utf-8"
                final_url = str(response.url)
                http_status = response.status_code
        except (httpx.HTTPError, OSError) as e:
            return failure("fetch_error", f"{type(e).__name__}: {e}")

        html = raw.decode(encoding, errors="replace")
        extracted = extract(html, url=final_url)
        if not extracted.text:
            return failure("empty", "no extractable text", http_status)
        return Snapshot(
            content_hash=content_hash(extracted.text),
            url=url,
            final_url=final_url,
            status="ok",
            http_status=http_status,
            title=extracted.title,
            text=extracted.text,
            language=extracted.language,
            language_confidence=extracted.language_confidence,
            published_date=extracted.published_date,
            extractor=extracted.extractor,
            fetched_at=now,
        )
