import numpy as np
import pandas as pd

from intelligence.adaptive_learning.reinforcement import MarketEnvironment
from strategy_brains.reinforcement_brain import TradingEnvironment


def build_data(rows=20):
    index = pd.date_range("2022-01-01", periods=rows, freq="D")
    market_data = pd.DataFrame({
        "open": np.linspace(1, 2, rows),
        "high": np.linspace(1.1, 2.1, rows),
        "low": np.linspace(0.9, 1.9, rows),
        "close": np.linspace(1, 2, rows),
        "volume": np.ones(rows) * 1000,
    }, index=index)
    features = pd.DataFrame({
        "feat1": np.zeros(rows),
        "feat2": np.ones(rows),
    }, index=index)
    return market_data, features


def test_market_environment_step_signature():
    md, feats = build_data()
    env = MarketEnvironment(md, feats)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)

    result = env.step(0)
    assert len(result) == 5
    next_obs, reward, terminated, truncated, info = result
    assert isinstance(next_obs, np.ndarray)
    assert isinstance(info, dict)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_trading_environment_step_signature():
    md, feats = build_data()
    data = pd.concat([md, feats], axis=1)
    env = TradingEnvironment(data)
    obs, info = env.reset()
    assert obs.shape[0] == env.window_size
    result = env.step(1)
    assert len(result) == 5
    next_obs, reward, terminated, truncated, info = result
    assert next_obs.shape[0] == env.window_size
