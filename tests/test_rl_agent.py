import numpy as np
from ml_models.rl.dqn_agent import DQNAgent


def test_dqn_agent_action():
    agent = DQNAgent(state_dim=4, action_dim=2)
    action = agent.select_action(np.zeros(4))
    assert action in [0, 1]
