import numpy as np
import os

from ml_models.rl import DQNAgent, PPOAgent


def test_dqn_agent_basic(tmp_path):
    agent = DQNAgent(state_dim=4, action_dim=2, batch_size=1, memory_size=10)
    state = np.zeros(4)
    next_state = np.ones(4)
    action = agent.select_action(state, test_mode=True)
    assert 0 <= action < 2
    agent.store_transition(state, action, 1.0, next_state, False)
    agent.store_transition(next_state, 0, 0.5, state, True)
    loss = agent.update_model()
    assert loss is not None
    path = tmp_path / "dqn.pth"
    agent.save(str(path))
    assert path.exists()
    new_agent = DQNAgent(state_dim=4, action_dim=2)
    new_agent.load(str(path))


def test_ppo_agent_basic(tmp_path):
    agent = PPOAgent(state_dim=4, action_dim=2, batch_size=1)
    state = np.zeros(4)
    action = agent.select_action(state, test_mode=False)
    agent.store_reward(1.0, False)
    agent.store_reward(0.5, True)
    agent.train_step()
    path = tmp_path / "ppo.pth"
    agent.save(str(path))
    assert path.exists()
    new_agent = PPOAgent(state_dim=4, action_dim=2)
    new_agent.load(str(path))

