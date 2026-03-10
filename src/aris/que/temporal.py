"""Temporal reference extraction from queries."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from aris.que.models import ComparisonOp, Constraint, ConstraintType


def _today() -> datetime:
    return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)


# Relative temporal patterns
_RELATIVE_PATTERNS = [
    (r"(?:in the )?last\s+(\d+)\s+days?", "days"),
    (r"(?:in the )?last\s+(\d+)\s+weeks?", "weeks"),
    (r"(?:in the )?last\s+(\d+)\s+months?", "months"),
    (r"(?:in the )?last\s+(\d+)\s+years?", "years"),
    (r"(?:in the )?past\s+(\d+)\s+days?", "days"),
    (r"(?:in the )?past\s+(\d+)\s+weeks?", "weeks"),
    (r"(?:in the )?past\s+(\d+)\s+months?", "months"),
    (r"(?:in the )?past\s+(\d+)\s+years?", "years"),
]

# Absolute year patterns
_YEAR_PATTERNS = [
    (r"(?:after|since|from)\s+((?:19|20)\d{2})", ComparisonOp.GTE),
    (r"(?:before|until|by)\s+((?:19|20)\d{2})", ComparisonOp.LT),
    (r"(?:in|during)\s+((?:19|20)\d{2})", ComparisonOp.EQ),
    (r"(?:released|published|founded|created)\s+(?:after|since)\s+((?:19|20)\d{2})", ComparisonOp.GTE),
    (r"(?:released|published|founded|created)\s+(?:before|until)\s+((?:19|20)\d{2})", ComparisonOp.LT),
    (r"(?:released|published|founded|created)\s+(?:in)\s+((?:19|20)\d{2})", ComparisonOp.EQ),
]

# Named temporal references
_NAMED_PATTERNS = [
    (r"\bthis\s+year\b", lambda: (_today().replace(month=1, day=1), None, ComparisonOp.GTE)),
    (r"\blast\s+year\b", lambda: (
        _today().replace(year=_today().year - 1, month=1, day=1),
        _today().replace(month=1, day=1),
        ComparisonOp.BETWEEN,
    )),
    (r"\brecent(?:ly)?\b", lambda: (_today() - timedelta(days=90), None, ComparisonOp.GTE)),
    (r"\btoday\b", lambda: (_today(), None, ComparisonOp.EQ)),
    (r"\byesterday\b", lambda: (_today() - timedelta(days=1), None, ComparisonOp.EQ)),
    (r"\bthis\s+week\b", lambda: (_today() - timedelta(days=_today().weekday()), None, ComparisonOp.GTE)),
    (r"\bthis\s+month\b", lambda: (_today().replace(day=1), None, ComparisonOp.GTE)),
]


def extract_temporal_constraints(query: str) -> list[Constraint]:
    """Extract temporal constraints from a query string."""
    constraints = []
    query_lower = query.lower()

    # Check relative patterns
    for pattern, unit in _RELATIVE_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            amount = int(match.group(1))
            if unit == "days":
                start = _today() - timedelta(days=amount)
            elif unit == "weeks":
                start = _today() - timedelta(weeks=amount)
            elif unit == "months":
                start = _today() - timedelta(days=amount * 30)
            else:  # years
                start = _today().replace(year=_today().year - amount)

            constraints.append(Constraint(
                type=ConstraintType.TEMPORAL,
                value=start.isoformat(),
                operator=ComparisonOp.GTE,
                field="date",
                raw_text=match.group(0),
            ))

    # Check absolute year patterns
    for pattern, op in _YEAR_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            year = int(match.group(1))
            if op == ComparisonOp.GTE:
                date_val = datetime(year, 1, 1)
            elif op == ComparisonOp.LT:
                date_val = datetime(year, 1, 1)
            else:  # EQ - year range
                date_val = datetime(year, 1, 1)
                constraints.append(Constraint(
                    type=ConstraintType.TEMPORAL,
                    value=date_val.isoformat(),
                    value_upper=datetime(year, 12, 31).isoformat(),
                    operator=ComparisonOp.BETWEEN,
                    field="date",
                    raw_text=match.group(0),
                ))
                continue

            constraints.append(Constraint(
                type=ConstraintType.TEMPORAL,
                value=date_val.isoformat(),
                operator=op,
                field="date",
                raw_text=match.group(0),
            ))

    # Check named patterns
    if not constraints:
        for pattern, resolver in _NAMED_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                start, end, op = resolver()
                constraint = Constraint(
                    type=ConstraintType.TEMPORAL,
                    value=start.isoformat(),
                    operator=op,
                    field="date",
                    raw_text=match.group(0),
                )
                if end is not None:
                    constraint.value_upper = end.isoformat()
                constraints.append(constraint)
                break

    return constraints
