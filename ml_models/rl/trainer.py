#!/usr/bin/env python3
"""Reinforcement Learning Trainer module."""

from typing import Any, Dict, Optional
import pandas as pd

from data_storage.market_data import MarketDataRepository
from feature_service.feature_extraction import FeatureExtractor
from intelligence.adaptive_learning.reinforcement import MarketEnvironment, DQNAgent
from common.async_utils import run_in_threadpool


class RLTradingAgent:
    """Helper class for training and running RL trading agents."""

    def __init__(
        self,
        feature_list: Optional[list] = None,
        market_repo: Optional[MarketDataRepository] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        agent_params: Optional[Dict[str, Any]] = None,
        timeframe: str = "1h",
    ) -> None:
        self.timeframe = timeframe
        self.market_repo = market_repo or MarketDataRepository()
        self.feature_extractor = feature_extractor or FeatureExtractor(feature_list or [])
        self.agent_params = agent_params or {}
        self.agent: Optional[DQNAgent] = None

    async def collect_data(
        self,
        asset: str,
        platform: str,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Collect OHLCV data from the repository."""
        return await self.market_repo.get_ohlcv_data(
            exchange=platform,
            symbol=asset,
            timeframe=self.timeframe,
            start_time=start,
            end_time=end,
        )

    async def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features using the configured feature extractor."""
        return await run_in_threadpool(self.feature_extractor.extract_features, data)

    async def train(
        self,
        asset: str,
        platform: str,
        start: Optional[Any] = None,
        end: Optional[Any] = None,
        episodes: int = 10,
        max_steps: int = 100,
    ) -> None:
        """Train a DQN agent on the specified asset data."""
        market_data = await self.collect_data(asset, platform, start, end)
        features = await self.extract_features(market_data)
        env = MarketEnvironment(market_data, features)

        self.agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            **self.agent_params,
        )

        for _ in range(episodes):
            state, _ = env.reset()
            for _ in range(max_steps):
                action = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.agent.store_transition(state, action, reward, next_state, done)
                self.agent.update_model()
                state = next_state
                if done:
                    break



