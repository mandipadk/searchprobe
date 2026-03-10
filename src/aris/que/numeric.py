"""Numeric constraint extraction from queries."""

from __future__ import annotations

import re
from typing import Any

from aris.que.models import ComparisonOp, Constraint, ConstraintType


# Patterns for numeric expressions
_NUMERIC_PATTERNS = [
    # "between X and Y"
    (r"between\s+([\d,.]+)\s+and\s+([\d,.]+)\s*(\w*)", ComparisonOp.BETWEEN),
    # "more than X", "greater than X", "over X", "above X", "X+"
    (r"(?:more|greater|over|above)\s+than\s+([\d,.]+)\s*(\w*)", ComparisonOp.GT),
    (r"([\d,.]+)\+\s*(\w*)", ComparisonOp.GTE),
    # "at least X", "minimum X", ">= X"
    (r"(?:at least|minimum|min)\s+([\d,.]+)\s*(\w*)", ComparisonOp.GTE),
    (r">=\s*([\d,.]+)\s*(\w*)", ComparisonOp.GTE),
    # "less than X", "under X", "below X", "fewer than X"
    (r"(?:less|fewer|under|below)\s+than\s+([\d,.]+)\s*(\w*)", ComparisonOp.LT),
    (r"<=\s*([\d,.]+)\s*(\w*)", ComparisonOp.LTE),
    # "exactly X", "= X"
    (r"exactly\s+([\d,.]+)\s*(\w*)", ComparisonOp.EQ),
    # "around X", "approximately X", "~X", "about X"
    (r"(?:around|approximately|about|~)\s*([\d,.]+)\s*(\w*)", ComparisonOp.APPROX),
]


def _parse_number(s: str) -> float:
    """Parse a number string, handling commas and common suffixes."""
    s = s.replace(",", "")
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    if s and s[-1].lower() in multipliers:
        return float(s[:-1]) * multipliers[s[-1].lower()]
    return float(s)


def extract_numeric_constraints(query: str) -> list[Constraint]:
    """Extract numeric constraints from a query string.

    Examples:
        "with 1000+ GitHub stars" -> Constraint(GTE, 1000, field="GitHub stars")
        "between 5 and 10 employees" -> Constraint(BETWEEN, 5, value_upper=10, field="employees")
    """
    constraints = []
    query_lower = query.lower()

    for pattern, op in _NUMERIC_PATTERNS:
        for match in re.finditer(pattern, query_lower):
            groups = match.groups()

            if op == ComparisonOp.BETWEEN:
                value = _parse_number(groups[0])
                value_upper = _parse_number(groups[1])
                unit = groups[2].strip() if len(groups) > 2 else ""
                constraints.append(Constraint(
                    type=ConstraintType.NUMERIC,
                    value=value,
                    value_upper=value_upper,
                    operator=op,
                    field=unit,
                    unit=unit,
                    raw_text=match.group(0),
                ))
            else:
                value = _parse_number(groups[0])
                unit = groups[1].strip() if len(groups) > 1 else ""
                constraints.append(Constraint(
                    type=ConstraintType.NUMERIC,
                    value=value,
                    operator=op,
                    field=unit,
                    unit=unit,
                    raw_text=match.group(0),
                ))

    return constraints
