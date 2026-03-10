"""Structured metadata retrieval for numeric/temporal/entity queries."""

from __future__ import annotations

import logging

from aris.core.models import ScoredDocument
from aris.index.structured_store import StructuredStore
from aris.que.models import ComparisonOp, Constraint, ConstraintType

logger = logging.getLogger(__name__)


class StructuredRetriever:
    """Retrieves documents using structured metadata queries."""

    def __init__(self, store: StructuredStore) -> None:
        self._store = store

    async def retrieve_by_constraints(
        self, constraints: list[Constraint], num_results: int = 100
    ) -> list[ScoredDocument]:
        """Retrieve documents matching structured constraints."""
        all_docs = []

        for constraint in constraints:
            if constraint.type == ConstraintType.TEMPORAL and constraint.operator:
                docs = self._retrieve_temporal(constraint, num_results)
                all_docs.extend(docs)
            elif constraint.type == ConstraintType.NUMERIC and constraint.operator:
                docs = self._retrieve_numeric(constraint, num_results)
                all_docs.extend(docs)

        # Deduplicate by URL
        seen = set()
        unique = []
        for doc in all_docs:
            if doc.document.url not in seen:
                seen.add(doc.document.url)
                unique.append(doc)

        return unique[:num_results]

    def _retrieve_temporal(self, constraint: Constraint, limit: int) -> list[ScoredDocument]:
        start = None
        end = None

        if constraint.operator in (ComparisonOp.GTE, ComparisonOp.GT):
            start = str(constraint.value)
        elif constraint.operator in (ComparisonOp.LTE, ComparisonOp.LT):
            end = str(constraint.value)
        elif constraint.operator == ComparisonOp.BETWEEN:
            start = str(constraint.value)
            end = str(constraint.value_upper) if constraint.value_upper else None
        elif constraint.operator == ComparisonOp.EQ:
            start = str(constraint.value)
            end = str(constraint.value)

        docs = self._store.query_by_date_range(start=start, end=end, limit=limit)
        return [
            ScoredDocument(document=doc, retrieval_score=0.8, final_score=0.8)
            for doc in docs
        ]

    def _retrieve_numeric(self, constraint: Constraint, limit: int) -> list[ScoredDocument]:
        if not constraint.field:
            return []

        op_map = {
            ComparisonOp.EQ: "eq",
            ComparisonOp.GT: "gt",
            ComparisonOp.GTE: "gte",
            ComparisonOp.LT: "lt",
            ComparisonOp.LTE: "lte",
        }
        op = op_map.get(constraint.operator, "gte")
        value = float(constraint.value) if constraint.value is not None else 0

        docs = self._store.query_by_numeric(constraint.field, op, value, limit=limit)
        return [
            ScoredDocument(document=doc, retrieval_score=0.9, final_score=0.9)
            for doc in docs
        ]
