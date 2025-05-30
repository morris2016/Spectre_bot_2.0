#!/usr/bin/env python3
"""Simple DQN agent leveraging PyTorch."""

from __future__ import annotations

import random
from collections import deque, namedtuple
from typing import Any

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore
    F = None  # type: ignore
    TORCH_AVAILABLE = False

from .base_agent import RLAgent

# Constants
MAX_MEMORY_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
LEARNING_RATE = 0.0003
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 10000
ALPHA = 0.6  # Prioritized replay alpha
BETA_START = 0.4  # Prioritized replay beta
BETA_END = 1.0
BETA_DECAY = 10000
CLIP_GRAD = 1.0

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class PrioritizedReplayMemory:
    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(self, state: Any, action: int, reward: float, next_state: Any, done: bool, max_prio: float | None = None) -> None:
        if max_prio is None:
            max_prio = self.max_priority
        if len(self.memory) < self.capacity:
            self.memory.append(Experience(state, action, reward, next_state, done))
        else:
            self.memory[self.position] = Experience(state, action, reward, next_state, done)
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.memory) < self.capacity:
            prios = self.priorities[: len(self.memory)]
        else:
            prios = self.priorities
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()
        batch = [self.memory[idx] for idx in indices]
        return batch, indices, weights

    def update_priorities(self, indices, priorities) -> None:
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return len(self.memory)


if TORCH_AVAILABLE:
    class DQN(nn.Module):
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
            super().__init__()
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc4 = nn.Linear(hidden_dim // 2, action_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.fc4(x)
else:
    class DQN:
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
            self.weights = np.zeros((state_dim, action_dim))

        def __call__(self, x: np.ndarray) -> np.ndarray:
            return x @ self.weights


if TORCH_AVAILABLE:
    class DuelingDQN(nn.Module):
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
            super().__init__()
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.value_fc = nn.Linear(hidden_dim, hidden_dim // 2)
            self.value = nn.Linear(hidden_dim // 2, 1)
            self.advantage_fc = nn.Linear(hidden_dim, hidden_dim // 2)
            self.advantage = nn.Linear(hidden_dim // 2, action_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            value = F.relu(self.value_fc(x))
            value = self.value(value)
            advantage = F.relu(self.advantage_fc(x))
            advantage = self.advantage(advantage)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
else:
    class DuelingDQN(DQN):
        pass


if TORCH_AVAILABLE:
    class NoisyLinear(nn.Module):
        def __init__(self, in_features: int, out_features: int, std_init: float = 0.5) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.std_init = std_init

            self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
            self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
            self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
            self.register_buffer("bias_epsilon", torch.empty(out_features))
            self.reset_parameters()
            self.reset_noise()

        def reset_parameters(self) -> None:
            mu_range = 1 / np.sqrt(self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.weight_sigma.size(1)))
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.bias_sigma.size(0)))

        def reset_noise(self) -> None:
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
            self.bias_epsilon.copy_(epsilon_out)

        @staticmethod
        def _scale_noise(size: int) -> torch.Tensor:
            x = torch.randn(size)
            return x.sign().mul_(x.abs().sqrt_())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.training:
                weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
                bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                weight = self.weight_mu
                bias = self.bias_mu
            return F.linear(x, weight, bias)
else:
    class NoisyLinear:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def reset_noise(self) -> None:
            pass


if TORCH_AVAILABLE:
    class NoisyDQN(nn.Module):
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
            super().__init__()
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = NoisyLinear(hidden_dim, hidden_dim // 2)
            self.fc4 = NoisyLinear(hidden_dim // 2, action_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.fc4(x)

        def reset_noise(self) -> None:
            self.fc3.reset_noise()
            self.fc4.reset_noise()

    class NoisyDuelingDQN(nn.Module):
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
            super().__init__()
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.value_fc = NoisyLinear(hidden_dim, hidden_dim // 2)
            self.value = NoisyLinear(hidden_dim // 2, 1)
            self.advantage_fc = NoisyLinear(hidden_dim, hidden_dim // 2)
            self.advantage = NoisyLinear(hidden_dim // 2, action_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            value = F.relu(self.value_fc(x))
            value = self.value(value)
            advantage = F.relu(self.advantage_fc(x))
            advantage = self.advantage(advantage)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
else:
    class NoisyDQN(DQN):
        def reset_noise(self) -> None:
            pass

    class NoisyDuelingDQN(DuelingDQN):
        def reset_noise(self) -> None:
            pass

    def reset_noise(self) -> None:
        self.value_fc.reset_noise()
        self.value.reset_noise()
        self.advantage_fc.reset_noise()
        self.advantage.reset_noise()


class DQNAgent(RLAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = LEARNING_RATE,
        gamma: float = GAMMA,
        tau: float = TAU,
        epsilon_start: float = EPSILON_START,
        epsilon_end: float = EPSILON_END,
        epsilon_decay: float = EPSILON_DECAY,
        memory_size: int = MAX_MEMORY_SIZE,
        batch_size: int = BATCH_SIZE,
        prioritized_replay: bool = True,
        dueling_network: bool = True,
        double_dqn: bool = True,
        noisy_nets: bool = False,
        device: str | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.prioritized_replay = prioritized_replay
        self.dueling_network = dueling_network
        self.double_dqn = double_dqn
        self.noisy_nets = noisy_nets

        if TORCH_AVAILABLE:
            device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.device = torch.device(device_str)
            self._setup_networks()
            self._setup_memory(memory_size)
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        else:
            self.device = "cpu"
            self.memory = deque(maxlen=memory_size)
            self.weights = np.zeros((state_dim, action_dim))

        self.steps_done = 0

    def _setup_networks(self) -> None:
        if not TORCH_AVAILABLE:
            return
        if self.dueling_network:
            net_class = NoisyDuelingDQN if self.noisy_nets else DuelingDQN
        else:
            net_class = NoisyDQN if self.noisy_nets else DQN
        self.policy_net = net_class(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_net = net_class(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def _setup_memory(self, memory_size: int) -> None:
        if not TORCH_AVAILABLE:
            return
        if not TORCH_AVAILABLE:
            return
        if self.prioritized_replay:
            self.memory = PrioritizedReplayMemory(memory_size, alpha=ALPHA)
            self.beta = BETA_START
        else:
            self.memory = ReplayMemory(memory_size)

    def select_action(self, state: Any, test_mode: bool = False) -> int:
        if TORCH_AVAILABLE:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if not test_mode:
                self.steps_done += 1
                epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
                    -self.steps_done / self.epsilon_decay
                )
                explore = False if self.noisy_nets else random.random() < epsilon

                if explore:
                    return random.randint(0, self.action_dim - 1)
            with torch.no_grad():
                q_values = self.policy_net(state_t)
                return int(q_values.max(1)[1].item())
        q_values = np.dot(state, self.weights)
        return int(np.argmax(q_values))


    def store_transition(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        if TORCH_AVAILABLE:
            if self.prioritized_replay:
                self.memory.push(state, action, reward, next_state, done, max_prio=self.memory.max_priority)
            else:
                self.memory.push(state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))

    def update_model(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        if TORCH_AVAILABLE:
            batch = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            state_batch = torch.FloatTensor(states)
            action_batch = torch.LongTensor(actions).unsqueeze(1)
            reward_batch = torch.FloatTensor(rewards)
            next_state_batch = torch.FloatTensor(next_states)
            done_batch = torch.FloatTensor(dones)

            q_values = self.policy_net(state_batch).gather(1, action_batch)
            with torch.no_grad():
                if self.double_dqn:
                    next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
                    next_q = self.target_net(next_state_batch).gather(1, next_actions)
                else:
                    next_q = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            target_q = reward_batch.unsqueeze(1) + (1 - done_batch.unsqueeze(1)) * self.gamma * next_q

            loss = F.mse_loss(q_values, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), CLIP_GRAD)
            self.optimizer.step()
            return float(loss.item())
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            q_current = np.dot(state, self.weights)[action]
            q_next = np.max(np.dot(next_state, self.weights)) if not done else 0.0
            target = reward + self.gamma * q_next
            self.weights[:, action] += self.learning_rate * (target - q_current) * state
        return 0.0

    def train_step(self) -> None:
        """Execute a single training iteration."""
        loss = self.update_model()
        if TORCH_AVAILABLE and loss is not None:
            for t_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                t_param.data.copy_((1 - self.tau) * t_param.data + self.tau * param.data)



    def save(self, path: str) -> None:
        if TORCH_AVAILABLE:
            torch.save(
                {
                    "policy_net": self.policy_net.state_dict(),
                    "target_net": self.target_net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "steps_done": self.steps_done,
                    "config": {
                        "state_dim": self.state_dim,
                        "action_dim": self.action_dim,
                        "hidden_dim": self.hidden_dim,
                        "learning_rate": self.learning_rate,
                        "gamma": self.gamma,
                        "tau": self.tau,
                        "epsilon_start": self.epsilon_start,
                        "epsilon_end": self.epsilon_end,
                        "epsilon_decay": self.epsilon_decay,
                        "batch_size": self.batch_size,
                        "prioritized_replay": self.prioritized_replay,
                        "dueling_network": self.dueling_network,
                        "double_dqn": self.double_dqn,
                        "noisy_nets": self.noisy_nets,
                    },
                },
                path,
            )
        else:
            with open(path, "wb") as f:
                np.save(f, self.weights)

    def load(self, path: str) -> None:
        if TORCH_AVAILABLE:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.target_net.load_state_dict(checkpoint["target_net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.steps_done = checkpoint["steps_done"]
        else:
            with open(path, "rb") as f:
                self.weights = np.load(f)

