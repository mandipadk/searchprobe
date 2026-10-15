"""Identity of a "system" under test.

A system is a provider plus an optional mode (and, later, an optional wrapper
such as ``aris``). Every comparison in SearchProbe is system vs. system, so
every grouping must use the same key. Never build ``f"{provider}:{mode}"``
by hand: missing modes become ``"None"``/``"default"`` inconsistently.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def system_key(provider: str | None, mode: str | None = None) -> str:
    """Return the canonical key for a provider/mode pair.

    >>> system_key("exa", "neural")
    'exa:neural'
    >>> system_key("brave", None)
    'brave'
    """
    provider = provider or "unknown"
    if mode and mode not in ("None", "default"):
        return f"{provider}:{mode}"
    return provider


def system_key_of(record: Mapping[str, Any] | Any) -> str:
    """Return the system key for a dict-like record or an object with attributes."""
    if isinstance(record, Mapping):
        return system_key(record.get("provider"), record.get("search_mode"))
    return system_key(getattr(record, "provider", None), getattr(record, "search_mode", None))


def split_system_key(key: str) -> tuple[str, str | None]:
    """Inverse of :func:`system_key`."""
    provider, _, mode = key.partition(":")
    return provider, (mode or None)
