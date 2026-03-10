"""Multi-constraint query decomposition.

Decomposes complex queries with multiple constraints into simpler sub-queries,
retrieves for each, and merges results.
"""

from __future__ import annotations

import logging

from aris.que.models import Constraint, StructuredQuery

logger = logging.getLogger(__name__)


def decompose_query(query: StructuredQuery) -> list[str]:
    """Decompose a complex query into sub-queries for independent retrieval.

    Each sub-query focuses on the core intent + one specific constraint,
    making it easier for embedding-based retrieval to handle.
    """
    core = query.core_intent or query.positive_query or query.original_query

    if not query.constraints or not query.use_decomposition:
        return [core]

    sub_queries = [core]  # Always include the core intent

    for constraint in query.constraints:
        if constraint.raw_text:
            sub_query = f"{core} {constraint.raw_text}"
        elif constraint.field and constraint.value is not None:
            sub_query = f"{core} {constraint.field} {constraint.value}"
        else:
            continue

        sub_queries.append(sub_query)

    logger.info("Decomposed into %d sub-queries", len(sub_queries))
    return sub_queries
