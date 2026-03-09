"""Async SQLite database using aiosqlite.

Coexists with the synchronous Database class. Use AsyncDatabase in async
contexts (pipeline runner, analysis adapters) to avoid blocking the event loop.
Shares the same schema and database file.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite

logger = logging.getLogger(__name__)


class AsyncDatabase:
    """Async SQLite database with persistent connection and WAL mode."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open connection and enable WAL mode."""
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> AsyncDatabase:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        await self.close()

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() or use as context manager.")
        return self._conn

    # ------------------------------------------------------------------
    # Run operations
    # ------------------------------------------------------------------

    async def create_run(
        self,
        query_set_id: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Create a new benchmark run."""
        run_id = str(uuid4())
        await self.conn.execute(
            "INSERT INTO runs (id, name, query_set_id, started_at, config) VALUES (?, ?, ?, ?, ?)",
            (run_id, name or f"run-{run_id}", query_set_id, datetime.now(timezone.utc).isoformat(), json.dumps(config or {})),
        )
        await self.conn.commit()
        return run_id

    async def complete_run(
        self,
        run_id: str,
        total_cost: float,
        cost_breakdown: dict[str, float],
    ) -> None:
        """Mark a run as complete."""
        await self.conn.execute(
            "UPDATE runs SET completed_at = ?, cost_total = ?, cost_breakdown = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), total_cost, json.dumps(cost_breakdown), run_id),
        )
        await self.conn.commit()

    async def get_latest_run_id(self) -> str | None:
        """Get the most recent run ID."""
        cursor = await self.conn.execute(
            "SELECT id FROM runs ORDER BY started_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------
    # Search result operations
    # ------------------------------------------------------------------

    async def add_search_result(
        self,
        run_id: str,
        query_id: str,
        response: Any,
    ) -> str:
        """Store a search result."""
        result_id = str(uuid4())
        await self.conn.execute(
            """INSERT INTO search_results
               (id, run_id, query_id, provider, search_mode, results, latency_ms, cost_usd, error, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result_id,
                run_id,
                query_id,
                response.provider,
                response.search_mode,
                json.dumps([r.model_dump(mode="json") for r in response.results]),
                response.latency_ms,
                response.cost_usd,
                response.error,
                response.timestamp.isoformat(),
            ),
        )
        await self.conn.commit()
        return result_id

    async def get_search_results(
        self,
        run_id: str,
        provider: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get search results for a run."""
        query = "SELECT * FROM search_results WHERE run_id = ?"
        params: list[Any] = [run_id]
        if provider:
            query += " AND provider = ?"
            params.append(provider)

        cursor = await self.conn.execute(query, params)
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d.get("results"):
                d["results"] = json.loads(d["results"])
            results.append(d)
        return results

    # ------------------------------------------------------------------
    # Evaluation operations
    # ------------------------------------------------------------------

    async def add_evaluation(self, run_id: str, evaluation: dict[str, Any]) -> str:
        """Store an evaluation result."""
        eval_id = str(uuid4())
        await self.conn.execute(
            """INSERT INTO evaluations
               (id, run_id, query_id, provider, result_index, scores, weighted_score,
                reasoning, failure_mode, judge_model, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                eval_id,
                run_id,
                evaluation.get("query_id"),
                evaluation.get("provider"),
                evaluation.get("best_result_index"),
                json.dumps(evaluation.get("dimension_scores", {})),
                evaluation.get("weighted_score", 0.0),
                evaluation.get("overall_assessment", ""),
                json.dumps(evaluation.get("failure_modes", [])),
                evaluation.get("judge_model"),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await self.conn.commit()
        return eval_id

    async def get_evaluations(
        self,
        run_id: str,
        provider: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get evaluations for a run."""
        query = "SELECT * FROM evaluations WHERE run_id = ?"
        params: list[Any] = [run_id]
        if provider:
            query += " AND provider = ?"
            params.append(provider)

        cursor = await self.conn.execute(query, params)
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d.get("scores"):
                d["scores"] = json.loads(d["scores"])
            if d.get("failure_mode"):
                d["failure_mode"] = json.loads(d["failure_mode"])
            results.append(d)
        return results
