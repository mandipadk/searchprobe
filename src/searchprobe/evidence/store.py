"""Content-addressed snapshot storage for fetched documents.

A snapshot is identified by the SHA-256 of its extracted text, so the same
document reached through different URLs (or providers) is judged once, and a
re-run of judging against stored snapshots is byte-for-byte reproducible.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

SnapshotStatus = Literal["ok", "http_error", "unsupported", "fetch_error", "empty"]


class Snapshot(BaseModel):
    """Extracted content of a document at a point in time."""

    content_hash: str
    url: str
    final_url: str | None = None
    status: SnapshotStatus = "ok"
    http_status: int | None = None
    title: str = ""
    text: str = ""
    language: str | None = None
    language_confidence: float = 0.0
    published_date: date | None = None
    extractor: str = ""
    fetched_at: datetime
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok" and bool(self.text)


def content_hash(text: str, url: str = "") -> str:
    """Hash of the extracted text; failed fetches hash their URL so they stay distinct."""
    payload = text if text else f"__nocontent__::{url}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EvidenceStore:
    """SQLite-backed snapshot store (shares the main database file)."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS snapshots (
        content_hash TEXT PRIMARY KEY,
        url TEXT NOT NULL,
        final_url TEXT,
        status TEXT NOT NULL,
        http_status INTEGER,
        title TEXT,
        text TEXT,
        language TEXT,
        language_confidence REAL,
        published_date TEXT,
        extractor TEXT,
        fetched_at TEXT NOT NULL,
        error TEXT
    );
    CREATE TABLE IF NOT EXISTS url_snapshots (
        url_norm TEXT PRIMARY KEY,
        content_hash TEXT NOT NULL REFERENCES snapshots(content_hash),
        fetched_at TEXT NOT NULL
    );
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
    def _row_to_snapshot(row: sqlite3.Row) -> Snapshot:
        d = dict(row)
        d["published_date"] = (
            date.fromisoformat(d["published_date"]) if d["published_date"] else None
        )
        d["fetched_at"] = datetime.fromisoformat(d["fetched_at"])
        d["title"] = d["title"] or ""
        d["text"] = d["text"] or ""
        d["language_confidence"] = d["language_confidence"] or 0.0
        d["extractor"] = d["extractor"] or ""
        return Snapshot(**d)

    def get(self, content_hash_: str) -> Snapshot | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM snapshots WHERE content_hash = ?", (content_hash_,)
            ).fetchone()
        return self._row_to_snapshot(row) if row else None

    def get_for_url(self, url_norm: str, max_age_days: float | None = None) -> Snapshot | None:
        """Latest snapshot for a normalized URL, optionally no older than ``max_age_days``."""
        with self._conn() as conn:
            row = conn.execute(
                """SELECT s.* FROM url_snapshots u JOIN snapshots s
                   ON s.content_hash = u.content_hash WHERE u.url_norm = ?""",
                (url_norm,),
            ).fetchone()
        if row is None:
            return None
        snap = self._row_to_snapshot(row)
        if max_age_days is not None:
            age = datetime.now(UTC) - snap.fetched_at.replace(tzinfo=snap.fetched_at.tzinfo or UTC)
            if age > timedelta(days=max_age_days):
                return None
        return snap

    def put(self, snapshot: Snapshot, url_norm: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO snapshots
                   (content_hash, url, final_url, status, http_status, title, text, language,
                    language_confidence, published_date, extractor, fetched_at, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    snapshot.content_hash,
                    snapshot.url,
                    snapshot.final_url,
                    snapshot.status,
                    snapshot.http_status,
                    snapshot.title,
                    snapshot.text,
                    snapshot.language,
                    snapshot.language_confidence,
                    snapshot.published_date.isoformat() if snapshot.published_date else None,
                    snapshot.extractor,
                    snapshot.fetched_at.isoformat(),
                    snapshot.error,
                ),
            )
            conn.execute(
                """INSERT INTO url_snapshots (url_norm, content_hash, fetched_at) VALUES (?, ?, ?)
                   ON CONFLICT(url_norm) DO UPDATE SET
                     content_hash = excluded.content_hash, fetched_at = excluded.fetched_at""",
                (url_norm, snapshot.content_hash, snapshot.fetched_at.isoformat()),
            )
            conn.commit()

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
