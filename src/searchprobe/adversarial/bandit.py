"""UCB1 multi-armed bandit for adaptive mutation operator selection.

Instead of selecting mutation operators uniformly at random, the bandit
learns which operators produce the best fitness improvements and balances
exploration (trying underused operators) vs exploitation (using the best ones).
"""

from __future__ import annotations

import math
from collections import defaultdict


class OperatorBandit:
    """UCB1 bandit for selecting mutation operators.

    Usage::

        bandit = OperatorBandit(["word_substitute", "negation_toggle", ...])
        operator = bandit.select()
        # ... apply operator, measure fitness improvement ...
        bandit.update(operator, fitness_improvement)
    """

    def __init__(self, operators: list[str]) -> None:
        self.operators = operators
        self.rewards: dict[str, list[float]] = defaultdict(list)
        self.counts: dict[str, int] = defaultdict(int)
        self.pending: dict[str, int] = defaultdict(int)  # selected, reward not yet known
        self.total_count = 0

    def select(self) -> str:
        """Select an operator using UCB1.

        Untried operators are selected first. Selections whose reward is still
        pending count as plays, so a batch of selections made within one
        generation spreads across operators instead of returning the same
        untried operator for every offspring.
        """
        plays = {op: self.counts[op] + self.pending[op] for op in self.operators}

        untried = [op for op in self.operators if plays[op] == 0]
        if untried:
            choice = untried[0]
        else:
            total = sum(plays.values())
            best_ucb = float("-inf")
            choice = self.operators[0]
            for op in self.operators:
                rewards = self.rewards[op]
                mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
                ucb = mean_reward + math.sqrt(2 * math.log(total) / plays[op])
                if ucb > best_ucb:
                    best_ucb, choice = ucb, op

        self.pending[choice] += 1
        return choice

    def update(self, operator: str, reward: float) -> None:
        """Record the reward (fitness improvement) for an operator.

        Args:
            operator: The operator that was used.
            reward: Fitness improvement (can be negative if fitness decreased).
        """
        self.rewards[operator].append(reward)
        self.counts[operator] += 1
        if self.pending[operator] > 0:
            self.pending[operator] -= 1
        self.total_count += 1

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for each operator."""
        stats = {}
        for op in self.operators:
            if self.counts[op] > 0:
                rewards = self.rewards[op]
                stats[op] = {
                    "count": self.counts[op],
                    "mean_reward": sum(rewards) / len(rewards),
                    "total_reward": sum(rewards),
                }
            else:
                stats[op] = {"count": 0, "mean_reward": 0.0, "total_reward": 0.0}
        return stats
