#!/usr/bin/env python3
"""Historical Memory mixin for strategy brains."""

from collections import deque
from typing import Deque, List

import random


class HistoricalMemoryMixin:
    """Mixin providing short-term and long-term trade memory."""

    def __init__(self, short_window: int = 50, long_window: int = 500, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._short_history: Deque[float] = deque(maxlen=short_window)
        self._long_history: Deque[float] = deque(maxlen=long_window)

    def record_trade_result(self, pnl_percent: float) -> None:
        """Record the result of a trade as percentage PnL."""
        self._short_history.append(pnl_percent)
        self._long_history.append(pnl_percent)

    def short_success_rate(self) -> float:
        """Return win rate over the short history."""
        if not self._short_history:
            return 0.0
        wins = sum(1 for p in self._short_history if p > 0)
        return wins / len(self._short_history)

    def long_success_rate(self) -> float:
        """Return win rate over the long history."""
        if not self._long_history:
            return 0.0
        wins = sum(1 for p in self._long_history if p > 0)
        return wins / len(self._long_history)

    def mutate_parameters(self) -> None:
        """Mutate strategy parameters slightly based on performance."""
        if not hasattr(self, "parameters"):
            return

        short_rate = self.short_success_rate()
        long_rate = self.long_success_rate()

        # If short-term performance lags long-term, explore new parameters
        if long_rate > 0 and short_rate < long_rate:
            for key, value in list(self.parameters.items()):
                if isinstance(value, (int, float)):
                    perturb = value * 0.1 * (random.random() - 0.5)
                    self.parameters[key] = type(value)(value + perturb)
