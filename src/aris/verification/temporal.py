"""Temporal constraint verifier.

Extracts dates from result content and verifies range satisfaction.
"""

from __future__ import annotations

import re
from datetime import datetime

from aris.core.models import Document
from aris.que.models import ComparisonOp, Constraint
from aris.verification.models import ConstraintStatus, VerificationResult


def verify_temporal(document: Document, constraint: Constraint) -> VerificationResult:
    """Verify a temporal constraint against document content."""
    text = f"{document.title} {document.snippet} {document.content or ''}"

    # Parse the constraint date
    try:
        target_date = datetime.fromisoformat(str(constraint.value))
    except (ValueError, TypeError):
        return VerificationResult(
            constraint_type="temporal",
            constraint_description=constraint.raw_text,
            status=ConstraintStatus.UNKNOWN,
            confidence=0.3,
        )

    # Try document's published_date first
    if document.published_date:
        if _check_temporal(document.published_date, constraint.operator, target_date, constraint.value_upper):
            return VerificationResult(
                constraint_type="temporal",
                constraint_description=constraint.raw_text,
                status=ConstraintStatus.SATISFIED,
                confidence=0.95,
                evidence=f"Published date {document.published_date.isoformat()} satisfies constraint",
            )
        else:
            return VerificationResult(
                constraint_type="temporal",
                constraint_description=constraint.raw_text,
                status=ConstraintStatus.VIOLATED,
                confidence=0.9,
                evidence=f"Published date {document.published_date.isoformat()} does not satisfy constraint",
            )

    # Extract dates from content
    dates = _extract_dates(text)
    if not dates:
        return VerificationResult(
            constraint_type="temporal",
            constraint_description=constraint.raw_text,
            status=ConstraintStatus.UNKNOWN,
            confidence=0.3,
            evidence="No dates found in document content",
        )

    # Check if any extracted date satisfies the constraint
    for date in dates:
        if _check_temporal(date, constraint.operator, target_date, constraint.value_upper):
            return VerificationResult(
                constraint_type="temporal",
                constraint_description=constraint.raw_text,
                status=ConstraintStatus.SATISFIED,
                confidence=0.7,
                evidence=f"Found date {date.isoformat()} in content",
            )

    return VerificationResult(
        constraint_type="temporal",
        constraint_description=constraint.raw_text,
        status=ConstraintStatus.VIOLATED,
        confidence=0.6,
        evidence=f"Dates found ({len(dates)}) do not satisfy constraint",
    )


def _extract_dates(text: str) -> list[datetime]:
    """Extract dates from text."""
    dates = []

    # ISO format dates
    for match in re.finditer(r"\b(\d{4}-\d{2}-\d{2})\b", text):
        try:
            dates.append(datetime.fromisoformat(match.group(1)))
        except ValueError:
            pass

    # Common date formats
    patterns = [
        (r"\b(\w+)\s+(\d{1,2}),?\s+(\d{4})\b", "%B %d %Y"),  # January 15, 2024
        (r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", None),  # MM/DD/YYYY
    ]

    for pattern, fmt in patterns:
        for match in re.finditer(pattern, text):
            try:
                if fmt:
                    date_str = f"{match.group(1)} {match.group(2)} {match.group(3)}"
                    dates.append(datetime.strptime(date_str, fmt))
                else:
                    month, day, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        dates.append(datetime(year, month, day))
            except (ValueError, TypeError):
                pass

    # Year-only references
    for match in re.finditer(r"\b((?:19|20)\d{2})\b", text):
        try:
            year = int(match.group(1))
            if 1900 <= year <= 2100:
                dates.append(datetime(year, 6, 15))  # Mid-year approximation
        except ValueError:
            pass

    return dates


def _check_temporal(
    date: datetime,
    op: ComparisonOp | None,
    target: datetime,
    upper_str: str | float | None = None,
) -> bool:
    """Check if date satisfies the temporal comparison."""
    if op is None:
        return True
    if op == ComparisonOp.GTE:
        return date >= target
    if op == ComparisonOp.GT:
        return date > target
    if op == ComparisonOp.LTE:
        return date <= target
    if op == ComparisonOp.LT:
        return date < target
    if op == ComparisonOp.EQ:
        return target.year == date.year and target.month == date.month
    if op == ComparisonOp.BETWEEN and upper_str:
        try:
            upper = datetime.fromisoformat(str(upper_str))
            return target <= date <= upper
        except (ValueError, TypeError):
            pass
    return False
