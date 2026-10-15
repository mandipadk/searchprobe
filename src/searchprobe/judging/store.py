"""Cache of per-document judgments, keyed by (case, content, judge, prompt version)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from searchprobe.judging.models import ConstraintVerdict, DocJudgment


class JudgmentStore:
    """SQLite-backed judgment cache (shares the main database file)."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS doc_judgments (
        case_id TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        judge_id TEXT NOT NULL,
        prompt_version TEXT NOT NULL,
        url TEXT,
        relevance INTEGER NOT NULL,
        verdicts TEXT NOT NULL,
        evidence_source TEXT,
        rationale TEXT,
        cost_usd REAL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (case_id, content_hash, judge_id, prompt_version)
    );
    CREATE INDEX IF NOT EXISTS idx_doc_judgments_case ON doc_judgments(case_id);
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        with self._conn() as conn:
            conn.executescript(self.SCHEMA)
            conn.commit()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _row(row: sqlite3.Row) -> DocJudgment:
        return DocJudgment(
            case_id=row["case_id"],
            content_hash=row["content_hash"],
            url=row["url"] or "",
            judge_id=row["judge_id"],
            prompt_version=row["prompt_version"],
            relevance=row["relevance"],
            verdicts=[ConstraintVerdict(**v) for v in json.loads(row["verdicts"])],
            evidence_source=row["evidence_source"] or "page",
            rationale=row["rationale"] or "",
            cost_usd=row["cost_usd"] or 0.0,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def get(
        self, case_id: str, content_hash: str, judge_id: str, prompt_version: str
    ) -> DocJudgment | None:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT * FROM doc_judgments WHERE case_id = ? AND content_hash = ?
                   AND judge_id = ? AND prompt_version = ?""",
                (case_id, content_hash, judge_id, prompt_version),
            ).fetchone()
        return self._row(row) if row else None

    def put(self, j: DocJudgment) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO doc_judgments
                   (case_id, content_hash, judge_id, prompt_version, url, relevance, verdicts,
                    evidence_source, rationale, cost_usd, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    j.case_id,
                    j.content_hash,
                    j.judge_id,
                    j.prompt_version,
                    j.url,
                    j.relevance,
                    json.dumps([v.model_dump() for v in j.verdicts]),
                    j.evidence_source,
                    j.rationale,
                    j.cost_usd,
                    j.created_at.isoformat(),
                ),
            )
            conn.commit()

    def for_case(self, case_id: str, judge_id: str | None = None) -> list[DocJudgment]:
        query = "SELECT * FROM doc_judgments WHERE case_id = ?"
        params: list[str] = [case_id]
        if judge_id:
            query += " AND judge_id = ?"
            params.append(judge_id)
        with self._conn() as conn:
            return [self._row(r) for r in conn.execute(query, params).fetchall()]

    def all(self, judge_id: str | None = None) -> list[DocJudgment]:
        query = "SELECT * FROM doc_judgments"
        params: list[str] = []
        if judge_id:
            query += " WHERE judge_id = ?"
            params.append(judge_id)
        with self._conn() as conn:
            return [self._row(r) for r in conn.execute(query, params).fetchall()]
