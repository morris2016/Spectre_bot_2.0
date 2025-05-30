#!/usr/bin/env python3
"""Reinforcement learning base agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


class BaseAgent(ABC):
    """Abstract base class for RL agents."""

    def __init__(self, state_dim: int, action_dim: int, config: dict | None = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}

    @abstractmethod
    def select_action(self, state: Any) -> int:
        """Select an action given a state."""

    @abstractmethod
    def store_transition(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        """Store experience in replay buffer."""

    @abstractmethod
    def update_model(self) -> float | None:
        """Update internal model and return loss."""

    def save_model(self, path: str) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required to save RL agents")
        torch.save(self, path)

    def load_model(self, path: str) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required to load RL agents")
        loaded = torch.load(path)
        self.__dict__.update(loaded.__dict__)


class RLAgent(ABC):
    """Abstract base class for reinforcement learning agents."""

    @abstractmethod
    def select_action(self, state: Any, test_mode: bool = False) -> Any:
        """Return action given state."""

    @abstractmethod
    def train_step(self) -> Any:
        """Perform one training step."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model parameters to path."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model parameters from path."""
