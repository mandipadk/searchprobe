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


@router.post("/index")
async def index_urls(request: IndexRequest):
    import os

    # Fetching arbitrary URLs on behalf of unauthenticated callers is an SSRF risk:
    # the route is disabled unless explicitly enabled, and private hosts are refused.
    if os.environ.get("ARIS_ENABLE_INDEX_API") != "1":
        raise HTTPException(status_code=403, detail="Index API disabled (ARIS_ENABLE_INDEX_API=1)")
    if _index_manager is None:
        raise HTTPException(status_code=503, detail="Index manager not initialized")
    from aris.sources.web import WebSource, is_public_http_url

    blocked = [u for u in request.urls if not is_public_http_url(u)]
    if blocked:
        raise HTTPException(status_code=400, detail=f"Refusing non-public URLs: {blocked}")

    web = WebSource(url_guard=is_public_http_url)  # every redirect hop is checked too
    try:
        documents = await web.fetch_urls(request.urls)
        _index_manager.add_documents(documents)
        return {"indexed": len(documents), "urls": request.urls}
    finally:
        await web.close()
