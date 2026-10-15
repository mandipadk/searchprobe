"""Test doubles shared across the suite (no network, no API keys)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from searchprobe.providers.models import SearchRequest, SearchResponse, SearchResult


class FakeMessages:
    """Mimics ``client.messages`` / ``client.beta.messages`` with async ``create``."""

    def __init__(
        self,
        responder: Callable[[dict[str, Any]], dict[str, Any] | str],
        delay: float = 0.0,
        stop_reason: str = "end_turn",
    ) -> None:
        self.responder = responder
        self.delay = delay
        self.stop_reason = stop_reason
        self.calls: list[dict[str, Any]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            payload = self.responder(kwargs)
        finally:
            self.in_flight -= 1
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason=self.stop_reason,
            stop_details=None,
            model=kwargs.get("model"),
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=0),
        )


class FakeAnthropic:
    """Async Anthropic client double. ``beta.messages`` and ``messages`` share state."""

    def __init__(self, responder: Callable[[dict[str, Any]], dict[str, Any] | str], **kw: Any):
        self.messages = FakeMessages(responder, **kw)
        self.beta = SimpleNamespace(messages=self.messages)


def user_prompt(kwargs: dict[str, Any]) -> str:
    """Extract the user prompt text from a ``create`` call."""
    return kwargs["messages"][0]["content"]


class FakeProvider:
    """Deterministic search provider: maps query text -> list of URLs."""

    name = "fake"
    NAME = "fake"

    def __init__(
        self,
        results: dict[str, list[str]] | Callable[[str], list[str]] | None = None,
        cost: float = 0.0,
        delay: float = 0.0,
        name: str = "fake",
        content: Callable[[str, str], str] | None = None,
    ) -> None:
        self.results = results or {}
        self.cost = cost
        self.delay = delay
        self.name = self.NAME = name
        self.content = content
        self.calls: list[SearchRequest] = []

    def _urls(self, query: str) -> list[str]:
        if callable(self.results):
            return self.results(query)
        return self.results.get(query, [])

    async def search(self, request: SearchRequest) -> SearchResponse:
        self.calls.append(request)
        if self.delay:
            await asyncio.sleep(self.delay)
        urls = self._urls(request.query)[: request.num_results]
        results = [
            SearchResult(
                title=f"Title {i}",
                url=u,
                snippet=f"snippet for {u}",
                content=self.content(request.query, u) if self.content else None,
                position=i,
            )
            for i, u in enumerate(urls)
        ]
        return SearchResponse(
            provider=self.name,
            search_mode=request.search_mode,
            query=request.query,
            results=results,
            cost_usd=self.cost,
        )

    def estimate_cost(self, request: SearchRequest) -> float:
        return self.cost

    async def close(self) -> None:
        pass
