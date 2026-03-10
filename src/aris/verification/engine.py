"""Constraint verification engine -- orchestrates all verifiers."""

from __future__ import annotations

import logging

from aris.core.models import ScoredDocument
from aris.que.models import ConstraintType, StructuredQuery
from aris.verification.boolean import compute_constraint_score, verify_boolean_and
from aris.verification.domain import verify_domain
from aris.verification.entity import verify_entity
from aris.verification.models import VerificationResult
from aris.verification.negation import verify_negation
from aris.verification.numeric import verify_numeric
from aris.verification.temporal import verify_temporal

logger = logging.getLogger(__name__)


class ConstraintVerificationEngine:
    """Orchestrates constraint verification across all document candidates.

    For EACH candidate, verifies EACH constraint independently, then
    combines results using AND-logic (not averaging).
    """

    async def verify(
        self,
        query: StructuredQuery,
        candidates: list[ScoredDocument],
    ) -> list[ScoredDocument]:
        """Verify all constraints for all candidates.

        Updates each candidate's verification_score, constraint_results, and confidence.
        """
        if not query.constraints and not query.negations and not query.entities:
            return candidates

        for doc in candidates:
            results = self._verify_document(query, doc)

            # Update constraint results
            for r in results:
                doc.constraint_results[r.constraint_description or r.constraint_type] = r.satisfied

            # Compute overall verification score
            doc.verification_score = compute_constraint_score(results)

            # Update confidence
            if results:
                doc.confidence = sum(r.confidence for r in results) / len(results)

        logger.info(
            "Verified %d constraints across %d candidates",
            len(query.constraints) + len(query.negations) + len(query.entities),
            len(candidates),
        )
        return candidates

    def _verify_document(
        self, query: StructuredQuery, doc: ScoredDocument
    ) -> list[VerificationResult]:
        """Verify all constraints against a single document."""
        results: list[VerificationResult] = []

        # Verify negation constraints
        if query.negations:
            results.append(verify_negation(doc.document, query.negations))

        # Verify typed constraints
        for constraint in query.constraints:
            if constraint.type == ConstraintType.NUMERIC:
                results.append(verify_numeric(doc.document, constraint))
            elif constraint.type == ConstraintType.TEMPORAL:
                results.append(verify_temporal(doc.document, constraint))
            elif constraint.type == ConstraintType.DOMAIN:
                results.append(verify_domain(doc.document, constraint))

        # Verify entity constraints
        for entity in query.entities:
            results.append(verify_entity(doc.document, entity))

        return results
