"""Tests for the UCB1 operator bandit."""

from searchprobe.adversarial.bandit import OperatorBandit


def test_explores_all_first():
    """Bandit tries each operator at least once before using UCB1."""
    ops = ["a", "b", "c"]
    bandit = OperatorBandit(ops)

    selected = set()
    for _ in range(3):
        op = bandit.select()
        selected.add(op)
        bandit.update(op, 0.5)

    assert selected == {"a", "b", "c"}


def test_exploits_best():
    """After exploration, bandit favors the high-reward operator."""
    ops = ["good", "bad"]
    bandit = OperatorBandit(ops)

    # Explore phase
    bandit.update("good", 0.0)
    bandit.update("bad", 0.0)
    bandit.counts["good"] = 1
    bandit.counts["bad"] = 1
    bandit.total_count = 2

    # Give "good" much better rewards
    for _ in range(20):
        bandit.update("good", 1.0)
    for _ in range(20):
        bandit.update("bad", 0.0)

    # Now select should strongly favor "good"
    selections = [bandit.select() for _ in range(10)]
    assert selections.count("good") > selections.count("bad")


def test_get_stats():
    ops = ["x", "y"]
    bandit = OperatorBandit(ops)
    bandit.update("x", 0.5)
    bandit.update("x", 1.0)
    bandit.update("y", 0.2)

    stats = bandit.get_stats()
    assert stats["x"]["count"] == 2
    assert stats["x"]["mean_reward"] == 0.75
    assert stats["y"]["count"] == 1
    assert stats["y"]["mean_reward"] == 0.2


def test_empty_stats():
    bandit = OperatorBandit(["a"])
    stats = bandit.get_stats()
    assert stats["a"]["count"] == 0
