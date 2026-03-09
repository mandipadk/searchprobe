"""SQLite database for storing benchmark data."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator
from uuid import uuid4

from searchprobe.providers.models import SearchResponse


class Database:
    """SQLite database for SearchProbe benchmark data.

    Stores queries, search results, evaluations, and aggregate scores.
    Uses synchronous sqlite3 for simplicity (async not needed for benchmark workloads).
    """

    SCHEMA = """
    -- Query Sets (generation runs)
    CREATE TABLE IF NOT EXISTS query_sets (
        id TEXT PRIMARY KEY,
        name TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        config TEXT,  -- JSON
        total_queries INTEGER DEFAULT 0
    );

    -- Individual Queries
    CREATE TABLE IF NOT EXISTS queries (
        id TEXT PRIMARY KEY,
        query_set_id TEXT REFERENCES query_sets(id),
        text TEXT NOT NULL,
        category TEXT NOT NULL,
        difficulty TEXT,
        tier TEXT,  -- seed, template, llm
        ground_truth TEXT,  -- JSON
        metadata TEXT,  -- JSON
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Benchmark Runs
    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        name TEXT,
        query_set_id TEXT REFERENCES query_sets(id),
        started_at TIMESTAMP,
        completed_at TIMESTAMP,
        config TEXT,  -- JSON
        cost_total REAL DEFAULT 0,
        cost_breakdown TEXT  -- JSON
    );

    -- Search Results
    CREATE TABLE IF NOT EXISTS search_results (
        id TEXT PRIMARY KEY,
        run_id TEXT REFERENCES runs(id),
        query_id TEXT REFERENCES queries(id),
        provider TEXT NOT NULL,
        search_mode TEXT,
        results TEXT NOT NULL,  -- JSON list of SearchResult
        latency_ms REAL,
        cost_usd REAL,
        error TEXT,
        timestamp TIMESTAMP
    );

    -- Evaluations
    CREATE TABLE IF NOT EXISTS evaluations (
        id TEXT PRIMARY KEY,
        run_id TEXT REFERENCES runs(id),
        query_id TEXT REFERENCES queries(id),
        provider TEXT NOT NULL,
        result_index INTEGER,  -- which result in the list
        scores TEXT NOT NULL,  -- JSON {dimension: score}
        weighted_score REAL,
        reasoning TEXT,
        failure_mode TEXT,
        judge_model TEXT,
        timestamp TIMESTAMP
    );

    -- Aggregate Scores (materialized for fast queries)
    CREATE TABLE IF NOT EXISTS aggregate_scores (
        run_id TEXT REFERENCES runs(id),
        provider TEXT,
        category TEXT,
        dimension TEXT,
        mean_score REAL,
        std_score REAL,
        ci_lower REAL,
        ci_upper REAL,
        n_queries INTEGER,
        failure_rate REAL,
        PRIMARY KEY (run_id, provider, category, dimension)
    );

    -- Geometry Analysis Results
    CREATE TABLE IF NOT EXISTS geometry_results (
        id TEXT PRIMARY KEY,
        run_id TEXT REFERENCES runs(id),
        model_name TEXT NOT NULL,
        category TEXT NOT NULL,
        adversarial_similarity REAL,
        baseline_similarity REAL,
        collapse_ratio REAL,
        vulnerability_score REAL,
        intrinsic_dimensionality REAL,
        isotropy_score REAL,
        pair_details TEXT,  -- JSON
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Perturbation Results
    CREATE TABLE IF NOT EXISTS perturbation_results (
        id TEXT PRIMARY KEY,
        run_id TEXT REFERENCES runs(id),
        query_id TEXT REFERENCES queries(id),
        provider TEXT NOT NULL,
        operator TEXT NOT NULL,
        original_query TEXT,
        perturbed_query TEXT,
        jaccard_similarity REAL,
        rbo_score REAL,
        original_results TEXT,  -- JSON
        perturbed_results TEXT,  -- JSON
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Cross-Encoder Validation Results
    CREATE TABLE IF NOT EXISTS validation_results (
        id TEXT PRIMARY KEY,
        run_id TEXT REFERENCES runs(id),
        query_id TEXT REFERENCES queries(id),
        provider TEXT NOT NULL,
        cross_encoder_model TEXT,
        original_ndcg REAL,
        reranked_ndcg REAL,
        ndcg_improvement REAL,
        kendall_tau REAL,
        scores TEXT,  -- JSON
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Evolution Results
    CREATE TABLE IF NOT EXISTS evolution_results (
        id TEXT PRIMARY KEY,
        fitness_mode TEXT NOT NULL,
        provider TEXT,
        generations_completed INTEGER,
        total_evaluations INTEGER,
        total_cost REAL,
        best_individuals TEXT,  -- JSON
        fitness_history TEXT,   -- JSON
        config TEXT,            -- JSON
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_queries_category ON queries(category);
    CREATE INDEX IF NOT EXISTS idx_queries_query_set ON queries(query_set_id);
    CREATE INDEX IF NOT EXISTS idx_search_results_run ON search_results(run_id);
    CREATE INDEX IF NOT EXISTS idx_search_results_provider ON search_results(provider);
    CREATE INDEX IF NOT EXISTS idx_evaluations_run ON evaluations(run_id);
    CREATE INDEX IF NOT EXISTS idx_evaluations_query_provider ON evaluations(query_id, provider);
    CREATE INDEX IF NOT EXISTS idx_geometry_model_category ON geometry_results(model_name, category);
    CREATE INDEX IF NOT EXISTS idx_perturbation_run ON perturbation_results(run_id);
    CREATE INDEX IF NOT EXISTS idx_validation_run ON validation_results(run_id);
    CREATE INDEX IF NOT EXISTS idx_evolution_timestamp ON evolution_results(timestamp);
    """

    def __init__(self, db_path: str | Path = "searchprobe.db") -> None:
        """Initialize database connection."""
        self.db_path = Path(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(self.SCHEMA)
            # Migrate: add run_id to geometry_results if missing
            cols = [row[1] for row in conn.execute("PRAGMA table_info(geometry_results)").fetchall()]
            if "run_id" not in cols:
                conn.execute("ALTER TABLE geometry_results ADD COLUMN run_id TEXT REFERENCES runs(id)")
            conn.commit()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    # Query Set Operations
    def create_query_set(
        self, name: str | None = None, config: dict[str, Any] | None = None
    ) -> str:
        """Create a new query set and return its ID."""
        query_set_id = str(uuid4())
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO query_sets (id, name, config) VALUES (?, ?, ?)",
                (query_set_id, name, json.dumps(config or {})),
            )
            conn.commit()
        return query_set_id

    def get_latest_query_set_id(self) -> str | None:
        """Get the most recent query set ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT id FROM query_sets ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            return row["id"] if row else None

    # Query Operations
    def add_query(
        self,
        query_set_id: str,
        text: str,
        category: str,
        difficulty: str | None = None,
        tier: str | None = None,
        ground_truth: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a query to a query set."""
        query_id = str(uuid4())
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO queries
                   (id, query_set_id, text, category, difficulty, tier, ground_truth, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    query_id,
                    query_set_id,
                    text,
                    category,
                    difficulty,
                    tier,
                    json.dumps(ground_truth or {}),
                    json.dumps(metadata or {}),
                ),
            )
            # Update query count
            conn.execute(
                "UPDATE query_sets SET total_queries = total_queries + 1 WHERE id = ?",
                (query_set_id,),
            )
            conn.commit()
        return query_id

    def get_queries(
        self,
        query_set_id: str | None = None,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get queries, optionally filtered by query set and category."""
        query = "SELECT * FROM queries WHERE 1=1"
        params: list[Any] = []

        if query_set_id:
            query += " AND query_set_id = ?"
            params.append(query_set_id)
        if category:
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY created_at"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    # Run Operations
    def create_run(
        self,
        query_set_id: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Create a new benchmark run."""
        run_id = str(uuid4())
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO runs (id, name, query_set_id, started_at, config)
                   VALUES (?, ?, ?, ?, ?)""",
                (run_id, name, query_set_id, datetime.now(timezone.utc).isoformat(), json.dumps(config or {})),
            )
            conn.commit()
        return run_id

    def complete_run(
        self, run_id: str, cost_total: float, cost_breakdown: dict[str, float]
    ) -> None:
        """Mark a run as completed with cost information."""
        with self._get_connection() as conn:
            conn.execute(
                """UPDATE runs
                   SET completed_at = ?, cost_total = ?, cost_breakdown = ?
                   WHERE id = ?""",
                (datetime.now(timezone.utc).isoformat(), cost_total, json.dumps(cost_breakdown), run_id),
            )
            conn.commit()

    def get_latest_run_id(self) -> str | None:
        """Get the most recent run ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT id FROM runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            return row["id"] if row else None

    # Search Results Operations
    def add_search_result(self, run_id: str, query_id: str, response: SearchResponse) -> str:
        """Store a search response."""
        result_id = str(uuid4())
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO search_results
                   (id, run_id, query_id, provider, search_mode, results,
                    latency_ms, cost_usd, error, timestamp)
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
            conn.commit()
        return result_id

    def get_search_results(
        self, run_id: str, provider: str | None = None
    ) -> list[dict[str, Any]]:
        """Get search results for a run."""
        query = "SELECT * FROM search_results WHERE run_id = ?"
        params: list[Any] = [run_id]

        if provider:
            query += " AND provider = ?"
            params.append(provider)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                result = dict(row)
                result["results"] = json.loads(result["results"])
                results.append(result)
            return results

    # Evaluation Operations
    def get_search_results_for_evaluation(
        self,
        run_id: str,
        skip_evaluated: bool = True,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get search results that need evaluation.

        Args:
            run_id: Run ID to get results for
            skip_evaluated: Skip results that already have evaluations
            max_results: Maximum number of results to return

        Returns:
            List of result dicts with query info joined
        """
        query = """
            SELECT
                sr.id as result_id,
                sr.run_id,
                sr.query_id,
                sr.provider,
                sr.search_mode,
                sr.results,
                sr.latency_ms,
                sr.cost_usd,
                sr.error,
                q.text as query_text,
                q.category,
                q.ground_truth
            FROM search_results sr
            JOIN queries q ON sr.query_id = q.id
            WHERE sr.run_id = ?
        """
        params: list[Any] = [run_id]

        if skip_evaluated:
            query += """
                AND NOT EXISTS (
                    SELECT 1 FROM evaluations e
                    WHERE e.run_id = sr.run_id
                    AND e.query_id = sr.query_id
                    AND e.provider = sr.provider
                )
            """

        query += " ORDER BY sr.timestamp"

        if max_results:
            query += " LIMIT ?"
            params.append(max_results)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                result = dict(row)
                result["results"] = json.loads(result["results"])
                if result["ground_truth"]:
                    result["ground_truth"] = json.loads(result["ground_truth"])
                results.append(result)
            return results

    def add_evaluation(self, run_id: str, evaluation: dict[str, Any]) -> str:
        """Store an evaluation result.

        Args:
            run_id: Run ID this evaluation belongs to
            evaluation: Evaluation dict from EvaluationResult.to_dict()

        Returns:
            Evaluation ID
        """
        eval_id = str(uuid4())
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO evaluations
                   (id, run_id, query_id, provider, scores, weighted_score,
                    reasoning, failure_mode, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    eval_id,
                    run_id,
                    evaluation.get("query_id"),
                    evaluation.get("provider"),
                    json.dumps(evaluation.get("dimension_scores", {})),
                    evaluation.get("weighted_score", 0.0),
                    evaluation.get("overall_assessment", ""),
                    json.dumps(evaluation.get("failure_modes", [])),
                    evaluation.get("evaluated_at", datetime.now(timezone.utc).isoformat()),
                ),
            )
            conn.commit()
        return eval_id

    def get_evaluations(
        self,
        run_id: str,
        provider: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get evaluations for a run.

        Args:
            run_id: Run ID to get evaluations for
            provider: Optional provider filter
            category: Optional category filter

        Returns:
            List of evaluation dicts with query info
        """
        query = """
            SELECT
                e.*,
                q.text as query_text,
                q.category
            FROM evaluations e
            JOIN queries q ON e.query_id = q.id
            WHERE e.run_id = ?
        """
        params: list[Any] = [run_id]

        if provider:
            query += " AND e.provider = ?"
            params.append(provider)

        if category:
            query += " AND q.category = ?"
            params.append(category)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                result = dict(row)
                if result.get("scores"):
                    result["scores"] = json.loads(result["scores"])
                if result.get("failure_mode"):
                    result["failure_modes"] = json.loads(result["failure_mode"])
                results.append(result)
            return results

    # Geometry Operations
    def add_geometry_result(self, result: dict[str, Any]) -> str:
        """Store a geometry analysis result."""
        result_id = str(uuid4())
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO geometry_results
                   (id, run_id, model_name, category, adversarial_similarity,
                    baseline_similarity, collapse_ratio, vulnerability_score,
                    intrinsic_dimensionality, isotropy_score, pair_details)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result_id,
                    result.get("run_id"),
                    result["model_name"],
                    result["category"],
                    result.get("adversarial_similarity"),
                    result.get("baseline_similarity"),
                    result.get("collapse_ratio"),
                    result.get("vulnerability_score"),
                    result.get("intrinsic_dimensionality"),
                    result.get("isotropy_score"),
                    json.dumps(result.get("pair_details", {})),
                ),
            )
            conn.commit()
        return result_id

    def get_geometry_results(
        self,
        model_name: str | None = None,
        category: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get geometry analysis results."""
        query = "SELECT * FROM geometry_results WHERE 1=1"
        params: list[Any] = []
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        if category:
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY timestamp DESC"
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                if r.get("pair_details"):
                    r["pair_details"] = json.loads(r["pair_details"])
                results.append(r)
            return results

    # Perturbation Operations
    def add_perturbation_result(self, result: dict[str, Any]) -> str:
        """Store a perturbation analysis result."""
        result_id = str(uuid4())
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO perturbation_results
                   (id, run_id, query_id, provider, operator, original_query,
                    perturbed_query, jaccard_similarity, rbo_score,
                    original_results, perturbed_results)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result_id,
                    result.get("run_id"),
                    result.get("query_id"),
                    result["provider"],
                    result["operator"],
                    result.get("original_query"),
                    result.get("perturbed_query"),
                    result.get("jaccard_similarity"),
                    result.get("rbo_score"),
                    json.dumps(result.get("original_results", [])),
                    json.dumps(result.get("perturbed_results", [])),
                ),
            )
            conn.commit()
        return result_id

    def get_perturbation_results(
        self, run_id: str, provider: str | None = None
    ) -> list[dict[str, Any]]:
        """Get perturbation results for a run."""
        query = "SELECT * FROM perturbation_results WHERE run_id = ?"
        params: list[Any] = [run_id]
        if provider:
            query += " AND provider = ?"
            params.append(provider)
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                if r.get("original_results"):
                    r["original_results"] = json.loads(r["original_results"])
                if r.get("perturbed_results"):
                    r["perturbed_results"] = json.loads(r["perturbed_results"])
                results.append(r)
            return results

    # Validation Operations
    def add_validation_result(self, result: dict[str, Any]) -> str:
        """Store a cross-encoder validation result."""
        result_id = str(uuid4())
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO validation_results
                   (id, run_id, query_id, provider, cross_encoder_model,
                    original_ndcg, reranked_ndcg, ndcg_improvement, kendall_tau, scores)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result_id,
                    result.get("run_id"),
                    result.get("query_id"),
                    result["provider"],
                    result.get("cross_encoder_model"),
                    result.get("original_ndcg"),
                    result.get("reranked_ndcg"),
                    result.get("ndcg_improvement"),
                    result.get("kendall_tau"),
                    json.dumps(result.get("scores", [])),
                ),
            )
            conn.commit()
        return result_id

    def get_validation_results(
        self, run_id: str, provider: str | None = None
    ) -> list[dict[str, Any]]:
        """Get validation results for a run."""
        query = "SELECT * FROM validation_results WHERE run_id = ?"
        params: list[Any] = [run_id]
        if provider:
            query += " AND provider = ?"
            params.append(provider)
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                if r.get("scores"):
                    r["scores"] = json.loads(r["scores"])
                results.append(r)
            return results

    # Evolution Operations
    def add_evolution_result(self, result: dict[str, Any]) -> str:
        """Store an evolution optimization result."""
        result_id = str(uuid4())
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO evolution_results
                   (id, fitness_mode, provider, generations_completed,
                    total_evaluations, total_cost, best_individuals,
                    fitness_history, config)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result_id,
                    result["fitness_mode"],
                    result.get("provider"),
                    result.get("generations_completed"),
                    result.get("total_evaluations"),
                    result.get("total_cost"),
                    json.dumps(result.get("best_individuals", [])),
                    json.dumps(result.get("fitness_history", [])),
                    json.dumps(result.get("config", {})),
                ),
            )
            conn.commit()
        return result_id

    def get_evolution_results(self) -> list[dict[str, Any]]:
        """Get all evolution results, most recent first."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM evolution_results ORDER BY timestamp DESC"
            ).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                for field in ("best_individuals", "fitness_history", "config"):
                    if r.get(field):
                        r[field] = json.loads(r[field])
                results.append(r)
            return results

    # Join queries for category info
    def get_validation_results_with_category(
        self, run_id: str, provider: str | None = None
    ) -> list[dict[str, Any]]:
        """Get validation results joined with query category info."""
        query = """
            SELECT v.*, q.text as query_text, q.category
            FROM validation_results v
            JOIN queries q ON v.query_id = q.id
            WHERE v.run_id = ?
        """
        params: list[Any] = [run_id]
        if provider:
            query += " AND v.provider = ?"
            params.append(provider)
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                if r.get("scores"):
                    r["scores"] = json.loads(r["scores"])
                results.append(r)
            return results

    def get_perturbation_results_with_category(
        self, run_id: str, provider: str | None = None
    ) -> list[dict[str, Any]]:
        """Get perturbation results joined with query category info."""
        query = """
            SELECT p.*, q.text as query_text, q.category
            FROM perturbation_results p
            LEFT JOIN queries q ON p.query_id = q.id
            WHERE p.run_id = ?
        """
        params: list[Any] = [run_id]
        if provider:
            query += " AND p.provider = ?"
            params.append(provider)
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            results = []
            for row in rows:
                r = dict(row)
                if r.get("original_results"):
                    r["original_results"] = json.loads(r["original_results"])
                if r.get("perturbed_results"):
                    r["perturbed_results"] = json.loads(r["perturbed_results"])
                results.append(r)
            return results

    # Stats
    def get_run_stats(self, run_id: str) -> dict[str, Any]:
        """Get summary statistics for a run."""
        with self._get_connection() as conn:
            # Get run info
            run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if not run:
                return {}

            # Count results by provider
            provider_counts = conn.execute(
                """SELECT provider, COUNT(*) as count, SUM(cost_usd) as total_cost,
                          AVG(latency_ms) as avg_latency
                   FROM search_results WHERE run_id = ? GROUP BY provider""",
                (run_id,),
            ).fetchall()

            return {
                "run_id": run_id,
                "name": run["name"],
                "started_at": run["started_at"],
                "completed_at": run["completed_at"],
                "cost_total": run["cost_total"],
                "providers": [
                    {
                        "name": row["provider"],
                        "queries": row["count"],
                        "cost": row["total_cost"],
                        "avg_latency_ms": row["avg_latency"],
                    }
                    for row in provider_counts
                ],
            }
