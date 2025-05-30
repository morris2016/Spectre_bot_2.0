from __future__ import annotations

from typing import Any, List

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore
    Categorical = None  # type: ignore
    TORCH_AVAILABLE = False

from .base_agent import RLAgent


if TORCH_AVAILABLE:
    class PolicyNetwork(nn.Module):
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
else:
    class PolicyNetwork:
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
            self.weights = np.zeros((state_dim, action_dim))

        def __call__(self, x: np.ndarray) -> np.ndarray:
            logits = x @ self.weights
            exp_logits = np.exp(logits - np.max(logits))
            return exp_logits / exp_logits.sum()


if TORCH_AVAILABLE:
    class ValueNetwork(nn.Module):
        def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
else:
    class ValueNetwork:
        def __init__(self, state_dim: int, hidden_dim: int = 128) -> None:
            self.weights = np.zeros((state_dim, 1))

        def __call__(self, x: np.ndarray) -> np.ndarray:
            return x @ self.weights


class PPOAgent(RLAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        update_epochs: int = 4,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        if TORCH_AVAILABLE:
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
            self.value = ValueNetwork(state_dim, hidden_dim).to(self.device)
            self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr)
        else:
            self.device = "cpu"
            self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
            self.value = ValueNetwork(state_dim, hidden_dim)
            self.optimizer = None

        self.states: List[Any] = []
        self.actions: List[int] = []
        self.log_probs: List[Any] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    def select_action(self, state: Any, test_mode: bool = False) -> int:
        if TORCH_AVAILABLE:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            probs = self.policy(state_t)
            dist = Categorical(probs)
            if test_mode:
                action = probs.argmax(dim=-1).item()
                self.log_probs.append(dist.log_prob(torch.tensor(action)))
            else:
                action = dist.sample().item()
                self.log_probs.append(dist.log_prob(torch.tensor(action)))
            self.states.append(state)
            self.actions.append(action)
            return action
        probs = self.policy(np.asarray(state))
        if test_mode:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(len(probs), p=probs))
        self.log_probs.append(np.log(probs[action]))
        self.states.append(np.asarray(state))
        self.actions.append(action)
        return action

    def store_reward(self, reward: float, done: bool) -> None:
        self.rewards.append(reward)
        self.dones.append(done)

    def _finish_path(self) -> None:
        if TORCH_AVAILABLE:
            states = torch.FloatTensor(self.states).to(self.device)
            actions = torch.LongTensor(self.actions).to(self.device)
            log_probs = torch.stack(self.log_probs).to(self.device)
            rewards = []
            ret = 0.0
            for r, d in zip(reversed(self.rewards), reversed(self.dones)):
                ret = r + self.gamma * ret * (1.0 - d)
                rewards.insert(0, ret)
            returns = torch.FloatTensor(rewards).to(self.device)
            advantages = returns - self.value(states).squeeze().detach()
            for _ in range(self.update_epochs):
                for idx in range(0, len(states), self.batch_size):
                    slice_idx = slice(idx, idx + self.batch_size)
                    s = states[slice_idx]
                    a = actions[slice_idx]
                    old_logp = log_probs[slice_idx].detach()
                    adv = advantages[slice_idx]
                    ret = returns[slice_idx]
                    probs = self.policy(s)
                    dist = Categorical(probs)
                    new_logp = dist.log_prob(a)
                    ratio = (new_logp - old_logp).exp()
                    clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    policy_loss = -(torch.min(ratio * adv, clipped_ratio * adv)).mean()
                    value_loss = torch.nn.functional.mse_loss(self.value(s).squeeze(), ret)
                    loss = policy_loss + 0.5 * value_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
        else:
            rewards = []
            ret = 0.0
            for r, d in zip(reversed(self.rewards), reversed(self.dones)):
                ret = r + self.gamma * ret * (1.0 - d)
                rewards.insert(0, ret)
            returns = np.array(rewards)
            states = np.array(self.states)
            actions = np.array(self.actions)
            baseline = self.value(states).flatten()
            advantages = returns - baseline
            for idx in range(len(states)):
                grad = advantages[idx] * states[idx][:, None]
                self.policy.weights[:, actions[idx]] += self.clip_eps * grad.flatten()
                self.value.weights[:, 0] += self.clip_eps * advantages[idx] * states[idx]
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()

    def train_step(self) -> None:
        if self.rewards:
            self._finish_path()

    def save(self, path: str) -> None:
        if TORCH_AVAILABLE:
            torch.save(
                {
                    "policy": self.policy.state_dict(),
                    "value": self.value.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                path,
            )
        else:
            with open(path, "wb") as f:
                np.savez(f, policy=self.policy.weights, value=self.value.weights)

    def load(self, path: str) -> None:
        if TORCH_AVAILABLE:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(checkpoint["policy"])
            self.value.load_state_dict(checkpoint["value"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            with open(path, "rb") as f:
                data = np.load(f)
                self.policy.weights = data["policy"]
                self.value.weights = data["value"]
