"""Lenient number parsing for values that come from regexes or LLM output."""

from __future__ import annotations

import re

_SCALES = {
    "k": 1e3,
    "thousand": 1e3,
    "m": 1e6,
    "million": 1e6,
    "mn": 1e6,
    "b": 1e9,
    "bn": 1e9,
    "billion": 1e9,
    "t": 1e12,
    "trillion": 1e12,
}
# Accepts ".5" and "5." (a sentence-final period after a number) as well as "5.25".
_NUM = re.compile(r"^\s*[$€£]?\s*(-?(?:\d+(?:\.\d*)?|\.\d+))\s*([a-z]+)?\s*%?\s*$", re.IGNORECASE)


def to_float(value: object) -> float | None:
    """Parse '1,000', '10M', '$4.2 billion', 3 -> float; None if not a number."""
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    m = _NUM.match(str(value).replace(",", ""))
    if not m:
        return None
    number, suffix = float(m.group(1)), (m.group(2) or "").lower()
    if suffix and suffix not in _SCALES:
        return None
    return number * _SCALES.get(suffix, 1.0)
