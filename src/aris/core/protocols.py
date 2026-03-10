"""Pluggable component protocols for Aris.

Protocols allow swapping implementations without changing the pipeline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aris.core.models import Document, ScoredDocument


@runtime_checkable
class DataSource(Protocol):
    """Protocol for data sources that fetch documents from external services."""

    @property
    def name(self) -> str: ...

    async def search(self, query: str, num_results: int = 10) -> list[Document]: ...

    async def close(self) -> None: ...


@runtime_checkable
class Retriever(Protocol):
    """Protocol for retrieval strategies (dense, sparse, structured, etc.)."""

    async def retrieve(self, query: str, num_results: int = 100) -> list[ScoredDocument]: ...


@runtime_checkable
class Verifier(Protocol):
    """Protocol for constraint verifiers."""

    async def verify(self, document: Document, constraint: dict) -> tuple[bool, float]:
        """Verify a constraint against a document.

        Returns (satisfied, confidence) tuple.
        """
        ...


@runtime_checkable
class Ranker(Protocol):
    """Protocol for document rankers."""

    async def rank(self, query: str, documents: list[ScoredDocument]) -> list[ScoredDocument]: ...
