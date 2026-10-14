"""FastAPI route handlers."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from aris.api.models import (
    HealthResponse,
    IndexRequest,
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
)

router = APIRouter()

# These will be set by app.py on startup
_search_agent = None
_sources = None
_index_manager = None
_config = None


def set_dependencies(search_agent, sources, index_manager, config):
    global _search_agent, _sources, _index_manager, _config
    _search_agent = search_agent
    _sources = sources
    _index_manager = index_manager
    _config = config


@router.get("/health", response_model=HealthResponse)
async def health():
    from aris import __version__

    return HealthResponse(
        status="ok",
        version=__version__,
        sources_available=[s.name for s in (_sources or [])],
    )


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if _search_agent is None:
        raise HTTPException(status_code=503, detail="Search agent not initialized")

    try:
        response = await _search_agent.search(
            query=request.query,
            sources=_sources or [],
            num_results=request.num_results,
        )

        return SearchResponse(
            query=response.query,
            results=[
                SearchResultResponse(
                    title=r.title,
                    url=r.url,
                    snippet=r.snippet,
                    score=r.score,
                    confidence=r.confidence,
                    constraint_satisfaction=r.constraint_satisfaction,
                    source=r.source,
                )
                for r in response.results
            ],
            total_candidates=response.total_candidates,
            iterations=response.iterations,
            latency_ms=response.latency_ms,
            strategy_used=response.strategy_used,
            predicted_failure_modes=response.predicted_failure_modes,
            timestamp=response.timestamp,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _is_public_http_url(url: str) -> bool:
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


@router.post("/index")
async def index_urls(request: IndexRequest):
    import os

    # Fetching arbitrary URLs on behalf of unauthenticated callers is an SSRF risk:
    # the route is disabled unless explicitly enabled, and private hosts are refused.
    if os.environ.get("ARIS_ENABLE_INDEX_API") != "1":
        raise HTTPException(status_code=403, detail="Index API disabled (ARIS_ENABLE_INDEX_API=1)")
    if _index_manager is None:
        raise HTTPException(status_code=503, detail="Index manager not initialized")
    blocked = [u for u in request.urls if not _is_public_http_url(u)]
    if blocked:
        raise HTTPException(status_code=400, detail=f"Refusing non-public URLs: {blocked}")

    from aris.sources.web import WebSource

    web = WebSource()
    try:
        documents = await web.fetch_urls(request.urls)
        _index_manager.add_documents(documents)
        return {"indexed": len(documents), "urls": request.urls}
    finally:
        await web.close()
