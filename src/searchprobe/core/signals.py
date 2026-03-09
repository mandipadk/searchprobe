"""Synchronous signal / event system for cross-module communication.

Modules emit signals when they discover interesting findings (e.g. a vulnerable
category, a stability measurement). Other modules subscribe and react -- for
instance, the evolution adapter can focus on categories that geometry flagged
as vulnerable.

This is a simple synchronous observer pattern. No async, no threading.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of signals emitted across the framework."""

    # Analysis signals
    VULNERABILITY_DETECTED = "vulnerability_detected"
    STABILITY_MEASURED = "stability_measured"
    EMBEDDING_GAP_FOUND = "embedding_gap_found"
    EVOLUTION_COMPLETED = "evolution_completed"

    # Intelligence signals
    FAILURE_MODE_CLASSIFIED = "failure_mode_classified"
    CATEGORY_PROFILE_UPDATED = "category_profile_updated"
    GROUND_TRUTH_VALIDATED = "ground_truth_validated"

    # Pipeline signals
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    BUDGET_WARNING = "budget_warning"


@dataclass
class Signal:
    """A single signal emitted by a module."""

    type: SignalType
    source: str
    category: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SignalBus:
    """Simple synchronous pub/sub bus.

    Usage::

        bus = SignalBus()
        bus.subscribe(SignalType.VULNERABILITY_DETECTED, my_handler)
        bus.emit(Signal(
            type=SignalType.VULNERABILITY_DETECTED,
            source="geometry",
            category="negation",
            data={"vulnerability_score": 0.92},
        ))
    """

    def __init__(self) -> None:
        self._handlers: dict[SignalType, list[Callable[[Signal], None]]] = defaultdict(list)

    def subscribe(self, signal_type: SignalType, handler: Callable[[Signal], None]) -> None:
        """Register a handler for a signal type."""
        self._handlers[signal_type].append(handler)

    def emit(self, signal: Signal) -> None:
        """Emit a signal, calling all registered handlers synchronously."""
        for handler in self._handlers.get(signal.type, []):
            try:
                handler(signal)
            except Exception:
                logger.exception(
                    "Handler %r failed for signal %s", handler, signal.type.value
                )

    def clear(self) -> None:
        """Remove all handlers."""
        self._handlers.clear()

    @property
    def handler_count(self) -> int:
        """Total number of registered handlers across all signal types."""
        return sum(len(hs) for hs in self._handlers.values())
