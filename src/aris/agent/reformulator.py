"""Query reformulation strategies for iterative refinement."""

from __future__ import annotations

import logging

from aris.que.models import StructuredQuery

logger = logging.getLogger(__name__)


class QueryReformulator:
    """Reformulates queries when initial results are insufficient."""

    def reformulate(
        self, query: StructuredQuery, reason: str, iteration: int
    ) -> StructuredQuery:
        """Create a reformulated query based on what failed.

        Strategies:
        1. Relax constraints (remove the most restrictive one)
        2. Broaden the search query
        3. Adjust retriever weights
        """
        reformulated = query.model_copy(deep=True)

        if iteration == 1:
            # Strategy 1: Broaden the query, keep constraints for verification
            reformulated.positive_query = reformulated.core_intent
            reformulated.use_decomposition = True
            # Increase external source weight
            reformulated.retriever_weights = {
                "dense": 0.25, "sparse": 0.25, "structured": 0.15, "hyde": 0.35,
            }
            logger.info("Reformulation %d: broadened query, enabled HyDE", iteration)

        elif iteration >= 2:
            # Strategy 2: Relax the most restrictive constraint
            if reformulated.constraints:
                # Remove the last constraint (most specific)
                removed = reformulated.constraints.pop()
                logger.info(
                    "Reformulation %d: relaxed constraint '%s'",
                    iteration, removed.raw_text,
                )

            # Also try without decomposition
            reformulated.use_decomposition = False
            reformulated.retriever_weights = {
                "dense": 0.4, "sparse": 0.3, "structured": 0.1, "hyde": 0.2,
            }

        return reformulated
