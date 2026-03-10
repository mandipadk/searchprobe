"""FastAPI application for the Aris search API."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from aris import __version__


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and clean up resources."""
    from aris.agent.search_agent import SearchAgent
    from aris.api.routes import set_dependencies
    from aris.core.config import ArisConfig
    from aris.index.manager import IndexManager
    from aris.que.engine import QueryUnderstandingEngine
    from aris.ranking.engine import RankingEngine
    from aris.retrieval.engine import RetrievalEngine
    from aris.sources.registry import SourceRegistry
    from aris.verification.engine import ConstraintVerificationEngine

    config = ArisConfig()
    registry = SourceRegistry(config)
    sources = registry.get_available()
    index_manager = IndexManager(config)

    que = QueryUnderstandingEngine(config)
    retrieval = RetrievalEngine(
        config,
        dense_store=index_manager.dense,
        sparse_store=index_manager.sparse,
        structured_store=index_manager.structured,
    )
    verification = ConstraintVerificationEngine()
    ranking = RankingEngine(config)

    agent = SearchAgent(config, que, retrieval, verification, ranking)

    set_dependencies(agent, sources, index_manager, config)

    yield

    await registry.close_all()
    index_manager.close()


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    from aris.api.middleware import setup_middleware
    from aris.api.routes import router

    app = FastAPI(
        title="Aris Search API",
        description="Neural Search Engine Built on Adversarial Intelligence",
        version=__version__,
        lifespan=lifespan,
    )

    setup_middleware(app)
    app.include_router(router, prefix="/api/v1")

    return app
