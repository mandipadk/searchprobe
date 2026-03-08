"""Tests for the circuit breaker state machine."""

import time

from searchprobe.providers.resilient import CircuitBreaker, CircuitState


def test_initial_state():
    cb = CircuitBreaker()
    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute() is True


def test_stays_closed_below_threshold():
    cb = CircuitBreaker(failure_threshold=5)
    for _ in range(4):
        cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute() is True


def test_opens_at_threshold():
    cb = CircuitBreaker(failure_threshold=3)
    for _ in range(3):
        cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.can_execute() is False


def test_success_resets():
    cb = CircuitBreaker(failure_threshold=5)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


def test_half_open_after_timeout():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.can_execute() is False

    time.sleep(0.06)

    assert cb.can_execute() is True
    assert cb.state == CircuitState.HALF_OPEN


def test_half_open_success_closes():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
    cb.record_failure()
    time.sleep(0.02)
    cb.can_execute()  # Transition to HALF_OPEN
    assert cb.state == CircuitState.HALF_OPEN

    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.failure_count == 0


def test_half_open_failure_reopens():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
    cb.record_failure()
    time.sleep(0.02)
    cb.can_execute()  # Transition to HALF_OPEN

    cb.record_failure()
    assert cb.state == CircuitState.OPEN
