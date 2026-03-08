"""Tests for the signal/event system."""

from searchprobe.core.signals import Signal, SignalBus, SignalType


def test_subscribe_and_emit():
    bus = SignalBus()
    received = []

    bus.subscribe(SignalType.VULNERABILITY_DETECTED, lambda s: received.append(s))

    signal = Signal(
        type=SignalType.VULNERABILITY_DETECTED,
        source="geometry",
        category="negation",
        data={"vulnerability_score": 0.92},
    )
    bus.emit(signal)

    assert len(received) == 1
    assert received[0].source == "geometry"
    assert received[0].data["vulnerability_score"] == 0.92


def test_multiple_handlers():
    bus = SignalBus()
    counts = {"a": 0, "b": 0}

    bus.subscribe(SignalType.STABILITY_MEASURED, lambda s: counts.__setitem__("a", counts["a"] + 1))
    bus.subscribe(SignalType.STABILITY_MEASURED, lambda s: counts.__setitem__("b", counts["b"] + 1))

    bus.emit(Signal(type=SignalType.STABILITY_MEASURED, source="test"))

    assert counts["a"] == 1
    assert counts["b"] == 1


def test_no_cross_signal_leakage():
    bus = SignalBus()
    received = []

    bus.subscribe(SignalType.VULNERABILITY_DETECTED, lambda s: received.append(s))

    bus.emit(Signal(type=SignalType.STABILITY_MEASURED, source="test"))

    assert len(received) == 0


def test_clear():
    bus = SignalBus()
    bus.subscribe(SignalType.VULNERABILITY_DETECTED, lambda s: None)
    assert bus.handler_count == 1

    bus.clear()
    assert bus.handler_count == 0


def test_signal_timestamp():
    signal = Signal(type=SignalType.STAGE_STARTED, source="session")
    assert signal.timestamp is not None


def test_handler_count():
    bus = SignalBus()
    assert bus.handler_count == 0

    bus.subscribe(SignalType.VULNERABILITY_DETECTED, lambda s: None)
    bus.subscribe(SignalType.STABILITY_MEASURED, lambda s: None)
    bus.subscribe(SignalType.STABILITY_MEASURED, lambda s: None)
    assert bus.handler_count == 3
