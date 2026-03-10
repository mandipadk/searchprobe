"""Structured metadata store using SQLite for numeric/temporal/entity queries."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from aris.core.models import Document
from aris.index.document import IndexedDocument

logger = logging.getLogger(__name__)


class StructuredStore:
    """SQLite-backed structured metadata store for constraint-based retrieval."""

    def __init__(self, db_path: str = ".aris/index/structured.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                url TEXT PRIMARY KEY,
                title TEXT,
                snippet TEXT,
                content TEXT,
                source TEXT,
                domain TEXT,
                published_date TEXT,
                word_count INTEGER DEFAULT 0,
                entities TEXT DEFAULT '[]',
                categories TEXT DEFAULT '[]',
                numeric_values TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_published_date ON documents(published_date);
            CREATE INDEX IF NOT EXISTS idx_domain ON documents(domain);
            CREATE INDEX IF NOT EXISTS idx_word_count ON documents(word_count);
        """)
        self._conn.commit()

    def add_documents(self, documents: list[Document | IndexedDocument]) -> None:
        """Add documents to the structured index."""
        for doc in documents:
            numeric_values = {}
            entities = []
            categories = []
            published_date = None
            word_count = 0

            if isinstance(doc, IndexedDocument):
                numeric_values = doc.numeric_values
                entities = doc.entities
                categories = doc.categories
                published_date = doc.published_date.isoformat() if doc.published_date else None
                word_count = doc.word_count
            else:
                content = doc.content or doc.snippet
                word_count = len(content.split()) if content else 0
                published_date = doc.published_date.isoformat() if doc.published_date else None

            self._conn.execute(
                """INSERT OR REPLACE INTO documents
                   (url, title, snippet, content, source, domain, published_date,
                    word_count, entities, categories, numeric_values)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    doc.url,
                    doc.title,
                    doc.snippet if hasattr(doc, 'snippet') else "",
                    doc.content[:5000] if doc.content else "",
                    doc.source if hasattr(doc, 'source') else "",
                    doc.domain if hasattr(doc, 'domain') else "",
                    published_date,
                    word_count,
                    json.dumps(entities),
                    json.dumps(categories),
                    json.dumps(numeric_values),
                ),
            )
        self._conn.commit()

    def query_by_date_range(
        self, start: str | None = None, end: str | None = None, limit: int = 100
    ) -> list[Document]:
        """Query documents by date range."""
        conditions = []
        params: list = []
        if start:
            conditions.append("published_date >= ?")
            params.append(start)
        if end:
            conditions.append("published_date <= ?")
            params.append(end)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._conn.execute(
            f"SELECT * FROM documents {where} ORDER BY published_date DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        return [self._row_to_document(r) for r in rows]

    def query_by_numeric(
        self, field: str, op: str, value: float, limit: int = 100
    ) -> list[Document]:
        """Query documents by numeric metadata field."""
        rows = self._conn.execute(
            "SELECT * FROM documents WHERE numeric_values != '{}' LIMIT ?",
            (limit * 5,),  # fetch extra since we filter in Python
        ).fetchall()

        results = []
        for row in rows:
            try:
                nums = json.loads(row["numeric_values"])
                if field in nums:
                    v = float(nums[field])
                    if self._compare(v, op, value):
                        results.append(self._row_to_document(row))
                        if len(results) >= limit:
                            break
            except (json.JSONDecodeError, ValueError):
                continue

        return results

    def _compare(self, a: float, op: str, b: float) -> bool:
        ops = {"eq": a == b, "gt": a > b, "gte": a >= b, "lt": a < b, "lte": a <= b}
        return ops.get(op, False)

    def _row_to_document(self, row: sqlite3.Row) -> Document:
        return Document(
            url=row["url"],
            title=row["title"] or "",
            snippet=row["snippet"] or "",
            content=row["content"] or "",
            source=row["source"] or "structured",
        )

    @property
    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
