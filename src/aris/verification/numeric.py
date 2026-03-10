"""Numeric constraint verifier.

Extracts numbers from result content and applies comparison operators.
This is the architectural solution to numeric blindness.
"""

from __future__ import annotations

import re

from aris.core.models import Document
from aris.que.models import ComparisonOp, Constraint
from aris.verification.models import ConstraintStatus, VerificationResult


def verify_numeric(document: Document, constraint: Constraint) -> VerificationResult:
    """Verify a numeric constraint against document content.

    Extracts numbers from the document and checks them against the constraint.
    """
    text = f"{document.title} {document.snippet} {document.content or ''}"
    target_value = float(constraint.value) if constraint.value is not None else None

    if target_value is None:
        return VerificationResult(
            constraint_type="numeric",
            constraint_description=constraint.raw_text,
            status=ConstraintStatus.UNKNOWN,
            confidence=0.5,
        )

    # Extract numbers from text, along with their context
    numbers = _extract_numbers_with_context(text)

    # If the constraint has a field/unit, try to match by context
    field = constraint.field.lower() if constraint.field else ""
    matched_numbers = []

    for value, context in numbers:
        if field and field in context.lower():
            matched_numbers.append(value)
        elif not field:
            matched_numbers.append(value)

    if not matched_numbers:
        return VerificationResult(
            constraint_type="numeric",
            constraint_description=constraint.raw_text,
            status=ConstraintStatus.UNKNOWN,
            confidence=0.3,
            evidence=f"No relevant numbers found for '{field or 'any'}'",
        )

    # Check if any extracted number satisfies the constraint
    for num in matched_numbers:
        if _check_comparison(num, constraint.operator, target_value, constraint.value_upper):
            return VerificationResult(
                constraint_type="numeric",
                constraint_description=constraint.raw_text,
                status=ConstraintStatus.SATISFIED,
                confidence=0.85,
                evidence=f"Found {num} which satisfies {constraint.operator.value if constraint.operator else '?'} {target_value}",
            )

    return VerificationResult(
        constraint_type="numeric",
        constraint_description=constraint.raw_text,
        status=ConstraintStatus.VIOLATED,
        confidence=0.8,
        evidence=f"Found numbers {matched_numbers[:5]} but none satisfy {constraint.operator.value if constraint.operator else '?'} {target_value}",
    )


def _extract_numbers_with_context(text: str) -> list[tuple[float, str]]:
    """Extract numbers from text with surrounding context."""
    results = []
    for match in re.finditer(r"([\d,]+(?:\.\d+)?)", text):
        try:
            value = float(match.group(1).replace(",", ""))
            # Get surrounding context (50 chars each side)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]
            results.append((value, context))
        except ValueError:
            continue
    return results


def _check_comparison(
    value: float,
    op: ComparisonOp | None,
    target: float,
    upper: float | None = None,
) -> bool:
    """Check if value satisfies the comparison."""
    if op is None:
        return True
    if op == ComparisonOp.EQ:
        return abs(value - target) < 0.01
    if op == ComparisonOp.GT:
        return value > target
    if op == ComparisonOp.GTE:
        return value >= target
    if op == ComparisonOp.LT:
        return value < target
    if op == ComparisonOp.LTE:
        return value <= target
    if op == ComparisonOp.BETWEEN:
        upper_val = float(upper) if upper is not None else float("inf")
        return target <= value <= upper_val
    if op == ComparisonOp.APPROX:
        margin = target * 0.15  # 15% margin
        return abs(value - target) <= margin
    return False
