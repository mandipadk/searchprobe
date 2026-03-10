"""Query Understanding Engine -- orchestrates parsing, constraint extraction, and strategy selection."""

from __future__ import annotations

import logging

from aris.core.config import ArisConfig
from aris.que.entities import extract_entities
from aris.que.failure_predictor import predict_failure_modes
from aris.que.models import StructuredQuery
from aris.que.negation import detect_negations, generate_positive_query
from aris.que.numeric import extract_numeric_constraints
from aris.que.parser import QueryParser
from aris.que.strategy import select_strategy
from aris.que.temporal import extract_temporal_constraints

logger = logging.getLogger(__name__)


class QueryUnderstandingEngine:
    """Orchestrates the full query understanding pipeline.

    Stages:
    1. Parse query into StructuredQuery (Claude-powered)
    2. Detect negations and generate positive query
    3. Extract numeric constraints
    4. Extract temporal constraints
    5. Extract entities with disambiguation context
    6. Predict failure modes
    7. Select search strategy
    """

    def __init__(self, config: ArisConfig) -> None:
        self._config = config
        self._parser = QueryParser(config)

    async def understand(self, query: str) -> StructuredQuery:
        """Run the full query understanding pipeline."""
        # Stage 1: Parse query with LLM
        structured = await self._parser.parse(query)

        # Stage 2: Enrich negation detection (rule-based catches what LLM might miss)
        rule_negations = detect_negations(query)
        for neg in rule_negations:
            if neg not in structured.negations:
                structured.negations.append(neg)
        if structured.negations:
            structured.positive_query = generate_positive_query(
                query, structured.negations
            )

        # Stage 3: Enrich numeric constraints
        rule_numeric = extract_numeric_constraints(query)
        existing_raw = {c.raw_text for c in structured.constraints}
        for c in rule_numeric:
            if c.raw_text not in existing_raw:
                structured.constraints.append(c)

        # Stage 4: Enrich temporal constraints
        rule_temporal = extract_temporal_constraints(query)
        for c in rule_temporal:
            if c.raw_text not in existing_raw:
                structured.constraints.append(c)

        # Stage 5: Enrich entity extraction
        rule_entities = extract_entities(query)
        existing_entity_texts = {e.text for e in structured.entities}
        for e in rule_entities:
            if e.text not in existing_entity_texts:
                structured.entities.append(e)

        # Stage 6: Predict failure modes
        structured.predicted_failure_modes = predict_failure_modes(structured)

        # Stage 7: Select search strategy
        structured = select_strategy(structured)

        logger.info(
            "QUE: intent=%r, negations=%s, constraints=%d, failure_modes=%s, strategies=%s",
            structured.core_intent,
            structured.negations,
            len(structured.constraints),
            structured.predicted_failure_modes,
            structured.verification_strategies,
        )

        return structured
