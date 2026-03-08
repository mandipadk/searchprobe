"""UCB1 multi-armed bandit for adaptive mutation operator selection.

Instead of selecting mutation operators uniformly at random, the bandit
learns which operators produce the best fitness improvements and balances
exploration (trying underused operators) vs exploitation (using the best ones).
"""

from __future__ import annotations

import math
import random
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
        self.total_count = 0

    def select(self) -> str:
        """Select an operator using UCB1.

        Returns the operator with the highest upper confidence bound.
        Operators that haven't been tried yet are selected first.
        """
        # Ensure every operator is tried at least once
        for op in self.operators:
            if self.counts[op] == 0:
                return op

        # UCB1: argmax(mean_reward + sqrt(2 * ln(total) / count))
        best_op = self.operators[0]
        best_ucb = float("-inf")

        for op in self.operators:
            mean_reward = sum(self.rewards[op]) / len(self.rewards[op])
            exploration = math.sqrt(2 * math.log(self.total_count) / self.counts[op])
            ucb = mean_reward + exploration

            if ucb > best_ucb:
                best_ucb = ucb
                best_op = op

        return best_op

    def update(self, operator: str, reward: float) -> None:
        """Record the reward (fitness improvement) for an operator.

        Args:
            operator: The operator that was used.
            reward: Fitness improvement (can be negative if fitness decreased).
        """
        self.rewards[operator].append(reward)
        self.counts[operator] += 1
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
