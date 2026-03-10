"""Claude-powered structured query parsing."""

from __future__ import annotations

import json
import logging

import anthropic

from aris.core.config import ArisConfig
from aris.que.models import (
    ComparisonOp,
    Constraint,
    ConstraintType,
    EntityReference,
    StructuredQuery,
)

logger = logging.getLogger(__name__)

PARSER_SYSTEM_PROMPT = """You are a query parser for a search engine. Analyze the user's search query and extract structured information.

Return a JSON object with these fields:
- core_intent: the main search intent without constraints, negations, or modifiers
- positive_query: the query rewritten for embedding search (remove negations like "NOT X", "without X", "no X"; keep the positive intent)
- negations: list of terms/concepts to EXCLUDE from results
- constraints: list of constraint objects, each with:
  - type: one of "negation", "numeric", "temporal", "entity", "domain", "boolean"
  - value: the constraint value (number, date string, etc.)
  - operator: one of "eq", "gt", "gte", "lt", "lte", "between", "approx" (for numeric/temporal)
  - field: what the constraint applies to (e.g. "employees", "stars", "date")
  - raw_text: the original text this was extracted from
  - unit: unit if applicable
  - value_upper: upper bound for "between" operator
  - negated_terms: for negation constraints, the terms to exclude
- entities: list of entity objects, each with:
  - text: the entity text
  - entity_type: type like "person", "company", "language", "technology"
  - disambiguation: context to help disambiguate (e.g. "Michael Jordan the basketball player, not the ML professor")

Be precise. Extract ALL constraints, even implicit ones. For temporal references like "after 2023", convert to a constraint with operator "gte" and value "2024-01-01"."""


class QueryParser:
    """Parses natural language queries into StructuredQuery using Claude."""

    def __init__(self, config: ArisConfig) -> None:
        self._config = config
        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    async def parse(self, query: str) -> StructuredQuery:
        """Parse a query into a StructuredQuery."""
        try:
            response = await self._client.messages.create(
                model=self._config.parser_model,
                max_tokens=1024,
                system=PARSER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": query}],
            )

            text = response.content[0].text
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text.strip())
            return self._build_structured_query(query, data)

        except Exception as e:
            logger.warning("LLM parsing failed, falling back to basic parsing: %s", e)
            return self._basic_parse(query)

    def _build_structured_query(self, original: str, data: dict) -> StructuredQuery:
        """Build StructuredQuery from parsed LLM output."""
        constraints = []
        for c in data.get("constraints", []):
            try:
                constraints.append(Constraint(
                    type=ConstraintType(c.get("type", "boolean")),
                    value=c.get("value"),
                    operator=ComparisonOp(c["operator"]) if c.get("operator") else None,
                    field=c.get("field", ""),
                    raw_text=c.get("raw_text", ""),
                    unit=c.get("unit", ""),
                    value_upper=c.get("value_upper"),
                    negated_terms=c.get("negated_terms", []),
                ))
            except (ValueError, KeyError) as e:
                logger.debug("Skipping malformed constraint: %s", e)

        entities = []
        for e in data.get("entities", []):
            entities.append(EntityReference(
                text=e.get("text", ""),
                entity_type=e.get("entity_type", ""),
                disambiguation=e.get("disambiguation", ""),
            ))

        return StructuredQuery(
            original_query=original,
            core_intent=data.get("core_intent", original),
            positive_query=data.get("positive_query", original),
            constraints=constraints,
            negations=data.get("negations", []),
            entities=entities,
        )

    def _basic_parse(self, query: str) -> StructuredQuery:
        """Fallback parser that doesn't need an LLM."""
        negation_markers = ["NOT ", "not ", " without ", " no ", " never "]
        negations = []
        positive_query = query

        for marker in negation_markers:
            if marker in query:
                parts = query.split(marker, 1)
                if len(parts) == 2:
                    negated = parts[1].split(",")[0].split(" and ")[0].strip()
                    negations.append(negated)
                    positive_query = positive_query.replace(marker + negated, " ")

        positive_query = " ".join(positive_query.split())  # normalize whitespace

        return StructuredQuery(
            original_query=query,
            core_intent=positive_query,
            positive_query=positive_query,
            negations=negations,
        )
