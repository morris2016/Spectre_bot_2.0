#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Reinforcement Learning Module

This module provides reinforcement learning capabilities for strategy optimization
and adaptive trading behavior. It leverages both traditional RL algorithms and
deep reinforcement learning techniques to continuously improve trading strategies
based on market feedback.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass
import threading
import queue
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.optim as optim  # type: ignore
    import torch.nn.functional as F  # type: ignore
    from torch.distributions import Categorical, Normal  # type: ignore
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency

    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore
    F = None  # type: ignore
    Categorical = Normal = None  # type: ignore
    TORCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch not available; reinforcement features are disabled")

from collections import deque, namedtuple
import random
try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
    GYM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    gym = None  # type: ignore
    spaces = None  # type: ignore
    GYM_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "gymnasium not available; using minimal environment implementation"
    )
# Force the use of the simplified environment during testing

GYM_AVAILABLE = False
import datetime
from concurrent.futures import ThreadPoolExecutor

# Internal imports
from common.utils import Timer, create_uuid, safe_divide
from common.logger import get_logger
from common.exceptions import (
    ModelTrainingError, InsufficientDataError, InvalidActionError,
    EnvironmentError, OptimizationError
)
from feature_service.features.technical import TechnicalFeatures
from feature_service.features.market_structure import MarketStructureFeatures
from data_storage.market_data import MarketDataRepository
try:
    from ml_models.rl import DQNAgent  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    DQNAgent = None  # type: ignore

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
UPDATE_TARGET_EVERY = 100
REWARD_SCALING = 0.01
CLIP_GRAD = 1.0

# Experience replay memory tuple
Experience = namedtuple('Experience', 
                        ['state', 'action', 'reward', 'next_state', 'done'])

# Logger setup
logger = get_logger(__name__)


if not GYM_AVAILABLE:
    class MarketEnvironment:
        """Minimal market environment used when gymnasium is unavailable."""

        def __init__(
            self,
            market_data: pd.DataFrame,
            features: pd.DataFrame,
            initial_balance: float = 1000.0,
            commission_rate: float = 0.001,
            slippage_model: str = "realistic",
            risk_factor: float = 0.01,
            max_position_size: float = 1.0,
            trading_frequency: str = "1m",
            reward_type: str = "sharpe",
            state_lookback: int = 60,
            include_market_features: bool = True,
            include_trade_history: bool = True,
            include_balance_history: bool = True,
            randomize_start: bool = True,
            market_impact_model: Optional[Callable] = None,
        ) -> None:
            self.market_data = market_data.reset_index(drop=True)
            self.features = features.reset_index(drop=True)
            self.initial_balance = initial_balance
            self.state_lookback = state_lookback
            self.current_idx = state_lookback
            self.initial_balance = initial_balance
            self.balance = initial_balance
            self.position = 0


            self._validate_data()

        def _validate_data(self) -> None:
            if self.market_data.empty or self.features.empty:
                raise ValueError("Market data or features are empty")
            if len(self.market_data) != len(self.features):
                raise ValueError("Market data and features length mismatch")
            required_cols = {"open", "high", "low", "close", "volume"}
            missing = required_cols - set(self.market_data.columns)
            if missing:
                raise ValueError(f"Missing market columns: {missing}")

        def _get_state(self) -> np.ndarray:
            start = self.current_idx - self.state_lookback
            df = pd.concat(
                [self.market_data.iloc[start:self.current_idx],
                 self.features.iloc[start:self.current_idx]],
                axis=1,
            )
            return df.values.flatten()

        def _validate_data(self) -> None:
            if self.market_data.empty or self.features.empty:
                raise ValueError("Market data or features cannot be empty")

            if len(self.market_data) != len(self.features):
                raise ValueError("Market data and features must have same length")

        def reset(self):
            self.current_idx = self.state_lookback
            self.balance = self.initial_balance
            return self._get_state(), {}

        def step(self, action, position_size_pct=None):
            self.current_idx += 1
            terminated = self.current_idx >= len(self.market_data)
            truncated = False
            reward = 0.0
            return self._get_state(), reward, terminated, truncated, {}

        def _validate_data(self):
            if self.market_data.empty or self.features.empty:
                raise ValueError("Market data or features cannot be empty")
            if len(self.market_data) != len(self.features):
                raise ValueError("Market data and features must have same length")


else:
    class MarketEnvironment:
        """Trading environment for reinforcement learning.

        This environment simulates market conditions and provides rewards based
        on trading actions and performance. It mimics a gymnasium-style interface
        with advanced market dynamics, transaction costs, and realistic
        constraints.
        """

        def __init__(
            self,
            market_data: pd.DataFrame,
            features: pd.DataFrame,
            initial_balance: float = 1000.0,
            commission_rate: float = 0.001,
            slippage_model: str = 'realistic',
            risk_factor: float = 0.01,
            max_position_size: float = 1.0,
            trading_frequency: str = '1m',
            reward_type: str = 'sharpe',
            state_lookback: int = 60,
            include_market_features: bool = True,
            include_trade_history: bool = True,
            include_balance_history: bool = True,
            randomize_start: bool = True,
            market_impact_model: Optional[Callable] = None,
        ) -> None:

            """Initialize the trading environment.

            Args:
                market_data: DataFrame containing OHLCV data
                features: DataFrame containing pre-calculated features
                initial_balance: Initial account balance
                commission_rate: Commission rate per trade
                slippage_model: Type of slippage model to use
            risk_factor: Maximum risk per trade as fraction of balance
            max_position_size: Maximum position size as multiple of balance
            trading_frequency: Frequency of trading decisions
            reward_type: Type of reward function to use
            state_lookback: Number of past periods to include in state
            include_market_features: Whether to include market features in state
            include_trade_history: Whether to include trade history in state
            include_balance_history: Whether to include balance history in state
            randomize_start: Whether to randomize the starting point
            market_impact_model: Optional function for market impact simulation
        """
            self.market_data = market_data
            self.features = features
            self.initial_balance = initial_balance
            self.commission_rate = commission_rate
            self.slippage_model = slippage_model
            self.risk_factor = risk_factor
            self.max_position_size = max_position_size
            self.trading_frequency = trading_frequency
            self.reward_type = reward_type
            self.state_lookback = state_lookback
            self.include_market_features = include_market_features
            self.include_trade_history = include_trade_history
            self.include_balance_history = include_balance_history
            self.randomize_start = randomize_start
            self.market_impact_model = market_impact_model
        
            # Validate input data
            self._validate_data()

            # Set up observation and action spaces
            self._setup_spaces()
        
            # Initialize environment state
            self.reset()
        
        def _validate_data(self):
            """Validate input data for consistency and completeness."""
            if self.market_data.empty or self.features.empty:
                raise InsufficientDataError("Market data or features DataFrame is empty")

            if len(self.market_data) != len(self.features):
                raise ValueError(
                    f"Market data length ({len(self.market_data)}) and features length "
                    f"({len(self.features)}) must match"
                )

            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in self.market_data.columns]
            if missing_cols:
                raise ValueError(f"Market data missing required columns: {missing_cols}")
        
            # Ensure indexes match and are datetime
            if not isinstance(self.market_data.index, pd.DatetimeIndex):
                raise ValueError("Market data index must be a DatetimeIndex")

            if not isinstance(self.features.index, pd.DatetimeIndex):
                raise ValueError("Features index must be a DatetimeIndex")

            # Ensure all index values in features exist in market_data
            if not self.features.index.isin(self.market_data.index).all():
                raise ValueError("Feature index contains values not in market data index")
            
    def _setup_spaces(self):
        """Define observation and action spaces."""
        # Determine state dimension based on features and configuration
        feature_dim = len(self.features.columns) if self.include_market_features else 0
        market_dim = 5  # OHLCV
        position_dim = 3  # position, entry_price, unrealized_pnl
        balance_history_dim = self.state_lookback if self.include_balance_history else 0
        trade_history_dim = self.state_lookback * 3 if self.include_trade_history else 0
        
        self.state_dim = (
            (feature_dim + market_dim) * self.state_lookback +
            position_dim + balance_history_dim + trade_history_dim
        )
        
        # Observation space: continuous state variables
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Action space: discrete actions for trading decisions
        # 0: Do nothing, 1: Buy, 2: Sell, 3: Close position
        self.action_space = spaces.Discrete(4)
        
        # Extended action space for position sizing
        self.position_size_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )
        
    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            tuple: (initial_state, info)
        """
        # Set initial position to the beginning or a random point if specified
        if self.randomize_start:
            lookback_buffer = self.state_lookback + 100  # Extra buffer for warm-up
            max_start = len(self.market_data) - lookback_buffer
            self.current_idx = random.randint(lookback_buffer, max_start)
        else:
            self.current_idx = self.state_lookback

            
            if len(self.market_data) != len(self.features):
                raise ValueError(
                    f"Market data length ({len(self.market_data)}) and features length "
                    f"({len(self.features)}) must match"
                )
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in self.market_data.columns]
            if missing_cols:
                raise ValueError(f"Market data missing required columns: {missing_cols}")
            
            # Ensure indexes match and are datetime
            if not isinstance(self.market_data.index, pd.DatetimeIndex):
                raise ValueError("Market data index must be a DatetimeIndex")
            
            if not isinstance(self.features.index, pd.DatetimeIndex):
                raise ValueError("Features index must be a DatetimeIndex")
                
            # Ensure all index values in features exist in market_data
            if not self.features.index.isin(self.market_data.index).all():
                raise ValueError("Feature index contains values not in market data index")
                
        def _setup_spaces(self):
            """Define observation and action spaces."""
            # Determine state dimension based on features and configuration
            feature_dim = len(self.features.columns) if self.include_market_features else 0
            market_dim = 5  # OHLCV
            position_dim = 3  # position, entry_price, unrealized_pnl
            balance_history_dim = self.state_lookback if self.include_balance_history else 0
            trade_history_dim = self.state_lookback * 3 if self.include_trade_history else 0
            
            self.state_dim = (
                (feature_dim + market_dim) * self.state_lookback +
                position_dim + balance_history_dim + trade_history_dim
            )
            
            # Observation space: continuous state variables
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
            )
            
            # Action space: discrete actions for trading decisions
            # 0: Do nothing, 1: Buy, 2: Sell, 3: Close position
            self.action_space = spaces.Discrete(4)
            
            # Extended action space for position sizing
            self.position_size_space = spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            )
            
        def reset(self):
            """
            Reset the environment to its initial state.
    
            Returns:
                tuple: (initial_state, info)
            """
            # Set initial position to the beginning or a random point if specified
            if self.randomize_start:
                lookback_buffer = self.state_lookback + 100  # Extra buffer for warm-up
                max_start = len(self.market_data) - lookback_buffer
                self.current_idx = random.randint(lookback_buffer, max_start)
            else:
                self.current_idx = self.state_lookback
                
            # Initialize account and position data
            self.balance = self.initial_balance
            self.position = 0  # 0: no position, 1: long, -1: short
            self.position_size = 0.0
            self.entry_price = 0.0
            self.position_start_time = None
            
            # History tracking
            self.balance_history = [self.balance] * self.state_lookback
            self.trade_history = []
            for _ in range(self.state_lookback):
                self.trade_history.append((0, 0, 0))  # (action, price, size)
                
            self.total_trades = 0
            self.profitable_trades = 0
            self.total_return = 0.0
            self.peak_balance = self.balance
            self.max_drawdown = 0.0
            self.cumulative_reward = 0.0
            
            # Get initial state
            return self._get_state(), {}
        
        def step(self, action, position_size_pct=None):
            """
            Take an action in the environment.
            
            Args:
                action: The action to take (0: hold, 1: buy, 2: sell, 3: close)
                position_size_pct: Position size as percentage of maximum (0.0-1.0)
                
            Returns:
                tuple: (next_state, reward, terminated, truncated, info)
            """
            # Validate action
            if action not in [0, 1, 2, 3]:
                raise InvalidActionError(f"Invalid action: {action}")
                
            # Default position size if not specified
            if position_size_pct is None:
                position_size_pct = 1.0
            else:
                position_size_pct = np.clip(position_size_pct, 0.0, 1.0)
                
            # Get current market data
            current_data = self.market_data.iloc[self.current_idx]
            current_price = current_data['close']
            
            # Track pre-action state
            prev_balance = self.balance
            prev_position = self.position
            
            # Process the action
            price, slippage = self._get_execution_price(action, current_data)
            info = self._execute_action(action, price, position_size_pct)
            
            # Move to next timestep
            self.current_idx += 1
            done = self.current_idx >= len(self.market_data) - 1
            
            # Update position P&L if we have an open position
            if self.position != 0:
                self._update_position_value(current_price)
                
            # Calculate reward
            reward = self._calculate_reward(prev_balance, prev_position, info)
            self.cumulative_reward += reward
            
            # Get new state
            next_state = self._get_state()
            
            # Update metrics
            self._update_metrics(prev_balance)
            
            # Prepare info dictionary
            info.update({
                'slippage': slippage,
                'balance': self.balance,
                'position': self.position,
                'position_size': self.position_size,
                'entry_price': self.entry_price,
                'current_price': current_price,
                'total_trades': self.total_trades,
                'profitable_trades': self.profitable_trades,
                'win_rate': safe_divide(self.profitable_trades, self.total_trades),
                'total_return': self.total_return,
                'max_drawdown': self.max_drawdown,
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'cumulative_reward': self.cumulative_reward
            })
            
            terminated = done
            truncated = False
            return next_state, reward, terminated, truncated, info
        
        def _get_state(self):
            """
            Construct the current state observation.
            
            Returns:
                numpy.ndarray: Current state vector
            """
            # Get market data history
            end_idx = self.current_idx
            start_idx = end_idx - self.state_lookback
            market_history = self.market_data.iloc[start_idx:end_idx]
            
            # Normalize market data relative to most recent close price
            reference_price = market_history['close'].iloc[-1]
            norm_ohlc = market_history[['open', 'high', 'low', 'close']] / reference_price - 1.0
            
            # Log-normalize volume
            norm_volume = np.log(market_history['volume'] / market_history['volume'].mean())
            
            # Combine normalized OHLCV
            market_states = pd.concat([norm_ohlc, norm_volume], axis=1).values.flatten()
            
            # Add feature history if enabled
            if self.include_market_features:
                feature_history = self.features.iloc[start_idx:end_idx]
                
                # Normalize features (simple z-score normalization)
                # This assumes features are already somewhat normalized or we should use a more
                # sophisticated normalization approach in production
                norm_features = (feature_history - feature_history.mean()) / (feature_history.std() + 1e-8)
                feature_states = norm_features.values.flatten()
                
                # Combine market data and features
                historic_states = np.concatenate([market_states, feature_states])
            else:
                historic_states = market_states
                
            # Add current position information
            position_vector = np.array([
                self.position,  # -1, 0, or 1
                self.entry_price / reference_price - 1.0 if self.position != 0 else 0,  # Normalized entry price
                (self.balance - self.initial_balance) / self.initial_balance  # Normalized P&L
            ])
            
            # Add balance history if enabled
            if self.include_balance_history:
                norm_balance_history = np.array(self.balance_history) / self.initial_balance - 1.0
                balance_states = norm_balance_history
            else:
                balance_states = np.array([])
                
            # Add trade history if enabled
            if self.include_trade_history:
                # Flatten trade history: action, price, size
                trade_states = np.array(self.trade_history).flatten()
                # Normalize price relative to reference
                for i in range(1, len(trade_states), 3):
                    if trade_states[i] > 0:  # Only normalize non-zero prices
                        trade_states[i] = trade_states[i] / reference_price - 1.0
            else:
                trade_states = np.array([])
                
            # Combine all state components
            state = np.concatenate([
                historic_states,
                position_vector,
                balance_states,
                trade_states
            ]).astype(np.float32)
            
            return state
        
        def _get_execution_price(self, action, current_data):
            """
            Calculate execution price including slippage.
            
            Args:
                action: Trading action
                current_data: Current market data row
                
            Returns:
                tuple: (execution_price, slippage_amount)
            """
            if action == 0 or action == 3:  # Do nothing or close - use close price
                base_price = current_data['close']
            elif action == 1:  # Buy - use higher price to simulate slippage
                base_price = current_data['close']
            elif action == 2:  # Sell - use lower price to simulate slippage
                base_price = current_data['close']
            else:
                base_price = current_data['close']
                
            # Apply slippage model
            if self.slippage_model == 'none':
                slippage = 0.0
            elif self.slippage_model == 'fixed':
                slippage = 0.0001 * base_price  # 1 pip fixed slippage
            elif self.slippage_model == 'realistic':
                # Dynamic slippage based on volatility and volume
                volatility = (current_data['high'] - current_data['low']) / current_data['close']
                volume_factor = 1.0  # Placeholder for volume-based slippage
                
                # Direction-dependent slippage
                if action == 1:  # Buy
                    slippage = base_price * volatility * 0.1 * volume_factor
                elif action == 2:  # Sell
                    slippage = -base_price * volatility * 0.1 * volume_factor
                else:
                    slippage = 0.0
            else:
                slippage = 0.0
                
            # Apply market impact if model provided and we're trading
            if self.market_impact_model is not None and action in [1, 2, 3]:
                impact = self.market_impact_model(
                    action, self.position_size, current_data, self.market_data, self.current_idx
                )
                slippage += impact
                
            # Calculate final execution price
            if action == 1:  # Buy
                exec_price = base_price + abs(slippage)
            elif action == 2:  # Sell
                exec_price = base_price - abs(slippage)
            else:
                exec_price = base_price
                
            return exec_price, slippage
        
        def _execute_action(self, action, price, position_size_pct):
            """
            Execute trading action and update environment state.
            
            Args:
                action: Trading action to execute
                price: Execution price including slippage
                position_size_pct: Position size percentage (0.0-1.0)
                
            Returns:
                dict: Transaction information
            """
            info = {
                'action': action,
                'price': price,
                'transaction_cost': 0.0,
                'trade_pnl': 0.0,
                'position_changed': False
            }
            
            # Calculate maximum position size based on risk factor and balance
            max_notional = self.balance * self.max_position_size
            target_notional = max_notional * position_size_pct
            
            # Handle different actions
            if action == 0:  # Do nothing
                pass
                
            elif action == 1:  # Buy
                # Close existing short position if any
                if self.position < 0:
                    close_size = abs(self.position_size)
                    close_cost = close_size * self.commission_rate * price
                    close_pnl = close_size * (self.entry_price - price)
                    
                    self.balance += close_pnl - close_cost
                    self.position = 0
                    self.position_size = 0
                    self.entry_price = 0
                    
                    info['transaction_cost'] += close_cost
                    info['trade_pnl'] += close_pnl
                    info['position_changed'] = True
                    
                    # Record trade
                    self.total_trades += 1
                    if close_pnl > 0:
                        self.profitable_trades += 1
                        
                # Open new long position if not already long
                if self.position <= 0:
                    # Calculate actual position size
                    size = target_notional / price
                    cost = size * self.commission_rate * price
                    
                    if cost + (size * price) <= self.balance:
                        self.position = 1
                        self.position_size = size
                        self.entry_price = price
                        self.balance -= cost
                        self.position_start_time = self.market_data.index[self.current_idx]
                        
                        info['transaction_cost'] += cost
                        info['position_changed'] = True
                        
                        # Update trade history
                        self.trade_history.append((1, price, size))
                        self.trade_history.pop(0)
                        
            elif action == 2:  # Sell
                # Close existing long position if any
                if self.position > 0:
                    close_size = self.position_size
                    close_cost = close_size * self.commission_rate * price
                    close_pnl = close_size * (price - self.entry_price)
                    
                    self.balance += close_pnl - close_cost
                    self.position = 0
                    self.position_size = 0
                    self.entry_price = 0
                    
                    info['transaction_cost'] += close_cost
                    info['trade_pnl'] += close_pnl
                    info['position_changed'] = True
                    
                    # Record trade
                    self.total_trades += 1
                    if close_pnl > 0:
                        self.profitable_trades += 1
                        
                # Open new short position if not already short
                if self.position >= 0:
                    # Calculate actual position size
                    size = target_notional / price
                    cost = size * self.commission_rate * price
                    
                    if cost <= self.balance:
                        self.position = -1
                        self.position_size = size
                        self.entry_price = price
                        self.balance -= cost
                        self.position_start_time = self.market_data.index[self.current_idx]
                        
                        info['transaction_cost'] += cost
                        info['position_changed'] = True
                        
                        # Update trade history
                        self.trade_history.append((2, price, size))
                        self.trade_history.pop(0)
                        
            elif action == 3:  # Close position
                if self.position != 0:
                    size = self.position_size
                    cost = size * self.commission_rate * price
                    
                    if self.position > 0:  # Close long
                        pnl = size * (price - self.entry_price)
                    else:  # Close short
                        pnl = size * (self.entry_price - price)
                        
                    self.balance += pnl - cost
                    self.position = 0
                    self.position_size = 0
                    self.entry_price = 0
                    
                    info['transaction_cost'] += cost
                    info['trade_pnl'] += pnl
                    info['position_changed'] = True
                    
                    # Record trade
                    self.total_trades += 1
                    if pnl > 0:
                        self.profitable_trades += 1
                        
                    # Update trade history
                    self.trade_history.append((3, price, size))
                    self.trade_history.pop(0)
                    
            # Update balance history
            self.balance_history.append(self.balance)
            self.balance_history.pop(0)
            
            return info
        
        def _update_position_value(self, current_price):
            """
            Update the unrealized P&L of the current position.
            
            Args:
                current_price: Current market price
            """
            if self.position == 0:
                return
                
            # Calculate unrealized P&L
            if self.position > 0:  # Long position
                unrealized_pnl = self.position_size * (current_price - self.entry_price)
            else:  # Short position
                unrealized_pnl = self.position_size * (self.entry_price - current_price)
                
            # Update unrealized P&L (doesn't affect balance until position is closed)
            self.unrealized_pnl = unrealized_pnl
        
        def _calculate_reward(self, prev_balance, prev_position, info):
            """
            Calculate reward based on the selected reward type.
            
            Args:
                prev_balance: Balance before action
                prev_position: Position before action
                info: Information from action execution
                
            Returns:
                float: Calculated reward
            """
            # Calculate basic P&L reward
            pnl_reward = (self.balance - prev_balance) * REWARD_SCALING
            
            if self.reward_type == 'pnl':
                # Simple profit/loss reward
                reward = pnl_reward
                
            elif self.reward_type == 'sharpe':
                # Sharpe-ratio based reward
                # We approximate this using recent balance changes
                recent_returns = np.diff(self.balance_history[-20:]) / self.balance_history[-21:-1]
                if len(recent_returns) > 0 and np.std(recent_returns) > 0:
                    sharpe = np.mean(recent_returns) / np.std(recent_returns) * np.sqrt(252)  # Annualized
                    reward = sharpe * 0.01  # Scale down the sharpe ratio
                else:
                    reward = 0
                    
                # Add small PnL component
                reward += pnl_reward
                
            elif self.reward_type == 'risk_adjusted':
                # Risk-adjusted return reward
                # This rewards higher returns with lower drawdowns
                pnl_ratio = (self.balance - prev_balance) / prev_balance if prev_balance > 0 else 0
                
                # Drawdown penalty component
                dd_penalty = -self.max_drawdown * 0.1 if self.max_drawdown > 0 else 0
                
                # Combine for risk-adjusted reward
                reward = pnl_ratio * REWARD_SCALING + dd_penalty
                
            elif self.reward_type == 'position_based':
                # Position-based reward that incentivizes holding good positions
                # and exiting bad ones
                current_data = self.market_data.iloc[self.current_idx]
                prev_data = self.market_data.iloc[self.current_idx - 1]
                price_change = (current_data['close'] - prev_data['close']) / prev_data['close']
                
                # Reward for being in a profitable position
                position_reward = 0
                if prev_position > 0 and price_change > 0:  # Correct long
                    position_reward = price_change * 10
                elif prev_position < 0 and price_change < 0:  # Correct short
                    position_reward = -price_change * 10
                elif prev_position > 0 and price_change < 0:  # Wrong long
                    position_reward = price_change * 5
                elif prev_position < 0 and price_change > 0:  # Wrong short
                    position_reward = -price_change * 5
                    
                # Combine with transaction reward
                reward = position_reward + pnl_reward
                
            else:  # Default to PnL reward
                reward = pnl_reward
                
            # Add transaction cost penalty
            transaction_cost_penalty = -info['transaction_cost'] * REWARD_SCALING * 2
            reward += transaction_cost_penalty
            
            # Add exploration penalty/reward to encourage exploration
            if info['position_changed']:
                exploration_bonus = 0.001  # Small bonus for exploring new positions
                reward += exploration_bonus
                
            return reward
        
        def _update_metrics(self, prev_balance):
            """
            Update tracking metrics after an action.
            
            Args:
                prev_balance: Balance before the action
            """
            # Update peak balance
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
                
            # Update drawdown
            current_drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Update total return
            self.total_return = (self.balance - self.initial_balance) / self.initial_balance
        
        def _calculate_sharpe_ratio(self):
            """
            Calculate Sharpe ratio based on balance history.
            
            Returns:
                float: Sharpe ratio
            """
            if len(self.balance_history) < 2:
                return 0
                
            returns = np.diff(self.balance_history) / self.balance_history[:-1]
            
            if len(returns) == 0 or np.std(returns) == 0:
                return 0
                
            # Annualized Sharpe ratio assuming daily data (adjust factor for different frequencies)
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            return sharpe
        
        def render(self, mode='human'):
            """
            Render the environment state.
            
            Args:
                mode: Rendering mode
            """
            current_data = self.market_data.iloc[self.current_idx]
            
            logger.info(f"\n==== Environment State at {current_data.name} ====")
            logger.info(f"Price: {current_data['close']:.4f}")
            logger.info(f"Balance: ${self.balance:.2f}")
            
            if self.position != 0:
                position_type = "LONG" if self.position > 0 else "SHORT"
                unrealized_pnl = self.position_size * (
                    (current_data['close'] - self.entry_price) if self.position > 0 
                    else (self.entry_price - current_data['close'])
                )
                logger.info(
                    f"Position: {position_type} {self.position_size:.4f} units at {self.entry_price:.4f}"
                )
                logger.info(f"Unrealized P&L: ${unrealized_pnl:.2f}")
                
            logger.info(f"Total Return: {self.total_return:.2%}")
            logger.info(f"Max Drawdown: {self.max_drawdown:.2%}")
            logger.info(
                f"Win Rate: {self.profitable_trades}/{self.total_trades} "
                f"({self.profitable_trades/self.total_trades:.2%} if self.total_trades > 0 else 'N/A')"
            )
            logger.info("=" * 40)
            
    
class LegacyDQNAgent:
    """
    Deep Q-Network agent for reinforcement learning-based trading.
    
    This agent implements a DQN with experience replay, target networks,
    and prioritized experience replay for stable training and efficient
    learning.
    """
    
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
        device: str = None
    ):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            tau: Target network update rate
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate of epsilon decay
            memory_size: Size of replay memory
            batch_size: Batch size for training
            prioritized_replay: Whether to use prioritized experience replay
            dueling_network: Whether to use dueling network architecture
            double_dqn: Whether to use double DQN algorithm
            noisy_nets: Whether to use noisy networks for exploration
            device: Computing device ('cpu' or 'cuda')
        """
        # Set parameters
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
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Set up networks
        self._setup_networks()
        
        # Set up memory
        self._setup_memory(memory_size)
        
        # Set up optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize step counter
        self.steps_done = 0
        
        # Setup logger
        self.logger = get_logger(__name__)
        
    def _setup_networks(self):
        """Set up neural networks."""
        # Determine network class based on configuration
        if self.dueling_network:
            if self.noisy_nets:
                net_class = NoisyDuelingDQN
            else:
                net_class = DuelingDQN
        else:
            if self.noisy_nets:
                net_class = NoisyDQN
            else:
                net_class = DQN
                
        # Create policy and target networks
        self.policy_net = net_class(
            self.state_dim, self.action_dim, self.hidden_dim
        ).to(self.device)
        
        self.target_net = net_class(
            self.state_dim, self.action_dim, self.hidden_dim
        ).to(self.device)
        
        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
    def _setup_memory(self, memory_size):
        """Set up replay memory."""
        if self.prioritized_replay:
            self.memory = PrioritizedReplayMemory(memory_size, alpha=ALPHA)
            self.beta = BETA_START
        else:
            self.memory = ReplayMemory(memory_size)
            
    def select_action(self, state, test_mode=False):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            test_mode: Whether in testing mode (no exploration)
            
        Returns:
            int: Selected action
        """
        # Convert state to tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Determine whether to explore or exploit
        if not test_mode:
            self.steps_done += 1
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      np.exp(-self.steps_done / self.epsilon_decay)
                      
            if self.noisy_nets:
                # Noisy nets handle exploration internally
                explore = False
            else:
                explore = random.random() < epsilon
                
            if explore:
                # Random action for exploration
                return random.randint(0, self.action_dim - 1)
        
        # Select greedy action
        with torch.no_grad():
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()
            
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        if self.prioritized_replay:
            # Store with maximum priority for new transitions
            self.memory.push(state, action, reward, next_state, done, 
                             max_prio=self.memory.max_priority)
        else:
            self.memory.push(state, action, reward, next_state, done)
            
    def update_model(self):
        """
        Update the model by sampling from replay memory.
        
        Returns:
            float: Loss value
        """
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample from memory
        if self.prioritized_replay:
            # Update beta for prioritized replay
            self.beta = min(1.0, BETA_END + (BETA_START - BETA_END) * \
                           np.exp(-self.steps_done / BETA_DECAY))
                           
            batch, indices, weights = self.memory.sample(self.batch_size, self.beta)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            batch = self.memory.sample(self.batch_size)
            indices = None
            weights = None
            
        # Unpack batch
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Compute current Q values
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: select actions using policy net
                next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
                # Evaluate actions using target net
                next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            else:
                # Regular DQN: use target net for both action selection and evaluation
                next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
                
        # Compute target Q values
        target_q_values = reward_batch.unsqueeze(1) + \
                          (1 - done_batch.unsqueeze(1)) * self.gamma * next_q_values
                          
        # Compute loss
        if self.prioritized_replay:
            # TD errors for updating priorities
            td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
            
            # Weighted MSE loss
            loss = (weights.unsqueeze(1) * F.mse_loss(q_values, target_q_values, reduction='none')).mean()
        else:
            loss = F.mse_loss(q_values, target_q_values)
            
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), CLIP_GRAD)
        self.optimizer.step()
        
        # Update target network
        with torch.no_grad():
            for target_param, policy_param in zip(self.target_net.parameters(), 
                                                self.policy_net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1 - self.tau) + policy_param.data * self.tau
                )
                
        # Update priorities in replay memory if using prioritized replay
        if self.prioritized_replay and indices is not None:
            self.memory.update_priorities(indices, td_errors)
            
        # Reset noisy layers if using noisy networks
        if self.noisy_nets:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()
            
        return loss.item()
        
    def save_model(self, path):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'tau': self.tau,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'prioritized_replay': self.prioritized_replay,
                'dueling_network': self.dueling_network,
                'double_dqn': self.double_dqn,
                'noisy_nets': self.noisy_nets,
            }
        }, path)
        self.logger.info(f"Model saved to {path}")
        
    def load_model(self, path):
        """
        Load model checkpoint.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Verify configuration matches
        config = checkpoint['config']
        for key, value in config.items():
            if hasattr(self, key) and getattr(self, key) != value:
                self.logger.warning(
                    f"Config mismatch: {key} loaded={value}, current={getattr(self, key)}"
                )
                
        # Load model parameters
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        
        self.logger.info(f"Model loaded from {path}")
        
        
# Neural network models for DQN
if TORCH_AVAILABLE:
    class DQN(nn.Module):
        """Basic DQN network."""

        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim

            # Network layers
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc4 = nn.Linear(hidden_dim // 2, action_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.fc4(x)
else:  # pragma: no cover - optional dependency
    class DQN:  # type: ignore
        """Placeholder when PyTorch is not available."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("PyTorch is required for DQN models")
        

if TORCH_AVAILABLE:
    class DuelingDQN(nn.Module):
        """Dueling DQN architecture."""

        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim

            # Feature layers
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)

            # Value stream
            self.value_fc = nn.Linear(hidden_dim, hidden_dim // 2)
            self.value = nn.Linear(hidden_dim // 2, 1)

            # Advantage stream
            self.advantage_fc = nn.Linear(hidden_dim, hidden_dim // 2)
            self.advantage = nn.Linear(hidden_dim // 2, action_dim)

        def forward(self, x):
            # Shared feature layers
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

            # Value stream
            value = F.relu(self.value_fc(x))
            value = self.value(value)

            # Advantage stream
            advantage = F.relu(self.advantage_fc(x))
            advantage = self.advantage(advantage)

            # Combine value and advantage
            return value + advantage - advantage.mean(dim=1, keepdim=True)
else:  # pragma: no cover - optional dependency
    class DuelingDQN:  # type: ignore
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("PyTorch is required for DuelingDQN models")
        

if TORCH_AVAILABLE:
    class NoisyLinear(nn.Module):
        """Noisy linear layer for exploration."""

        def __init__(self, in_features, out_features, std_init=0.5):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.std_init = std_init

            # Learnable parameters
            self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
            self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

            # Register buffer for noise
            self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
            self.register_buffer('bias_epsilon', torch.Tensor(out_features))

            # Initialize parameters
            self.reset_parameters()
            self.reset_noise()

        def reset_parameters(self):
            """Initialize the layer parameters."""
            mu_range = 1.0 / np.sqrt(self.in_features)
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

        def reset_noise(self):
            """Reset the noise."""
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
            self.bias_epsilon.copy_(epsilon_out)

        def _scale_noise(self, size):
            """Generate scaled noise."""
            x = torch.randn(size)
            return x.sign().mul(x.abs().sqrt())

        def forward(self, x):
            """Forward pass with noise."""
            if self.training:
                weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
                bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                weight = self.weight_mu
                bias = self.bias_mu

            return F.linear(x, weight, bias)
else:  # pragma: no cover - optional dependency
    class NoisyLinear:  # type: ignore
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("PyTorch is required for NoisyLinear layers")
        

if TORCH_AVAILABLE:
    class NoisyDQN(nn.Module):
        """DQN with noisy layers for exploration."""

        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim

            # Network layers
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = NoisyLinear(hidden_dim, hidden_dim // 2)
            self.fc4 = NoisyLinear(hidden_dim // 2, action_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return self.fc4(x)

        def reset_noise(self):
            """Reset noise in noisy layers."""
            self.fc3.reset_noise()
            self.fc4.reset_noise()
else:  # pragma: no cover - optional dependency
    class NoisyDQN:  # type: ignore
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("PyTorch is required for NoisyDQN models")
        

if TORCH_AVAILABLE:
    class NoisyDuelingDQN(nn.Module):
        """Dueling DQN with noisy layers."""

        def __init__(self, state_dim, action_dim, hidden_dim=128):
            super().__init__()
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim

            # Feature layers
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)

            # Value stream
            self.value_fc = NoisyLinear(hidden_dim, hidden_dim // 2)
            self.value = NoisyLinear(hidden_dim // 2, 1)

            # Advantage stream
            self.advantage_fc = NoisyLinear(hidden_dim, hidden_dim // 2)
            self.advantage = NoisyLinear(hidden_dim // 2, action_dim)

        def forward(self, x):
            # Shared feature layers
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))

            # Value stream
            value = F.relu(self.value_fc(x))
            value = self.value(value)

            # Advantage stream
            advantage = F.relu(self.advantage_fc(x))
            advantage = self.advantage(advantage)

            # Combine value and advantage
            return value + advantage - advantage.mean(dim=1, keepdim=True)

        def reset_noise(self):
            """Reset noise in all noisy layers."""
            self.value_fc.reset_noise()
            self.value.reset_noise()
            self.advantage_fc.reset_noise()
            self.advantage.reset_noise()
else:  # pragma: no cover - optional dependency
    class NoisyDuelingDQN:  # type: ignore
        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("PyTorch is required for NoisyDuelingDQN models")
        

# Experience replay memory implementations
class ReplayMemory:
    """Standard experience replay memory."""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append(Experience(state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """Sample a batch of experiences."""
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)
        

class PrioritizedReplayMemory:
    """Prioritized experience replay memory."""
    
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # How much to prioritize
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0  # Initial max priority
        
    def push(self, state, action, reward, next_state, done, max_prio=None):
        """Add a new experience to memory with maximum priority."""
        if max_prio is None:
            max_prio = self.max_priority
            
        if len(self.memory) < self.capacity:
            self.memory.append(Experience(state, action, reward, next_state, done))
        else:
            self.memory[self.position] = Experience(state, action, reward, next_state, done)
            
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences based on priorities.
        
        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent
            
        Returns:
            tuple: (batch, indices, weights)
        """
        if len(self.memory) < self.capacity:
            prios = self.priorities[:len(self.memory)]
        else:
            prios = self.priorities
            
        # Compute probabilities from priorities
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        
        # Compute importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights
        
        # Create batch
        batch = [self.memory[idx] for idx in indices]
        batch = Experience(*zip(*batch))
        
        return batch, indices, weights
        
    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on TD errors.
        
        Args:
            indices: Indices of experiences to update
            td_errors: TD errors for the experiences
        """
        for idx, error in zip(indices, td_errors):
            # Add small constant to ensure non-zero priority
            priority = error + 1e-5
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return len(self.memory)
        

class ReinforcementLearningService:
    """
    Service for training and deploying reinforcement learning models
    for the QuantumSpectre trading system.
    """
    
    def __init__(
        self,
        config=None,
        model_repository_path="./models/reinforcement/",
        market_data_repository=None,
        feature_service=None
    ):
        """
        Initialize the reinforcement learning service.
        
        Args:
            config: Configuration dictionary
            model_repository_path: Path to store model files
            market_data_repository: Market data repository instance
            feature_service: Feature service instance
        """
        self.config = config or {}
        self.model_repository_path = model_repository_path
        self.market_data_repository = market_data_repository
        self.feature_service = feature_service
        
        # Initialize models dictionary
        self.models = {}
        
        # Initialize logger
        self.logger = get_logger(__name__)
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_repository_path, exist_ok=True)
        
        # Initialize agent config
        self._init_default_config()
        
    def _init_default_config(self):
        """Initialize default configuration."""
        self.default_config = {
            'dqn': {
                'hidden_dim': 256,
                'learning_rate': 0.0003,
                'gamma': 0.99,
                'tau': 0.005,
                'epsilon_start': 1.0,
                'epsilon_end': 0.05,
                'epsilon_decay': 10000,
                'memory_size': 100000,
                'batch_size': 64,
                'prioritized_replay': True,
                'dueling_network': True,
                'double_dqn': True,
                'noisy_nets': True,
                'update_frequency': 4,
                'target_update_frequency': 1000,
                'validation_episodes': 10,
                'training_episodes': 1000,
                'max_timesteps': 10000,
                'reward_type': 'risk_adjusted',
                'state_lookback': 60
            },
            'env': {
                'initial_balance': 1000.0,
                'commission_rate': 0.001,
                'slippage_model': 'realistic',
                'risk_factor': 0.01,
                'max_position_size': 1.0,
                'trading_frequency': '1m',
                'randomize_start': True,
                'include_market_features': True,
                'include_trade_history': True,
                'include_balance_history': True,
            }
        }
        
    def get_model_path(self, asset, platform, model_type='dqn'):
        """
        Get the path for a specific model.
        
        Args:
            asset: Asset symbol
            platform: Trading platform
            model_type: Type of model
            
        Returns:
            str: Path to the model file
        """
        filename = f"{platform}_{asset}_{model_type}.pt"
        return os.path.join(self.model_repository_path, filename)
        
    def create_environment(self, asset, platform, start_time=None, end_time=None):
        """
        Create a trading environment for the specified asset and platform.
        
        Args:
            asset: Asset symbol
            platform: Trading platform
            start_time: Start time for data
            end_time: End time for data
            
        Returns:
            MarketEnvironment: The created environment
        """
        # Get market data
        if self.market_data_repository is None:
            raise ValueError("Market data repository must be provided")
            
        market_data = self.market_data_repository.get_ohlcv(
            asset=asset,
            platform=platform,
            start_time=start_time,
            end_time=end_time
        )
        
        # Get features
        if self.feature_service is None:
            raise ValueError("Feature service must be provided")
            
        features = self.feature_service.get_features(
            asset=asset,
            platform=platform,
            start_time=start_time,
            end_time=end_time
        )
        
        # Create environment with configuration
        env_config = self.config.get('env', self.default_config['env'])
        
        env = MarketEnvironment(
            market_data=market_data,
            features=features,
            initial_balance=env_config['initial_balance'],
            commission_rate=env_config['commission_rate'],
            slippage_model=env_config['slippage_model'],
            risk_factor=env_config['risk_factor'],
            max_position_size=env_config['max_position_size'],
            trading_frequency=env_config['trading_frequency'],
            reward_type=self.config.get('dqn', {}).get(
                'reward_type', self.default_config['dqn']['reward_type']
            ),
            state_lookback=self.config.get('dqn', {}).get(
                'state_lookback', self.default_config['dqn']['state_lookback']
            ),
            include_market_features=env_config['include_market_features'],
            include_trade_history=env_config['include_trade_history'],
            include_balance_history=env_config['include_balance_history'],
            randomize_start=env_config['randomize_start']
        )
        
        return env
        
    def create_agent(self, state_dim, action_dim, device=None):
        """
        Create a reinforcement learning agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            device: Computing device
            
        Returns:
            DQNAgent: The created agent
        """
        dqn_config = self.config.get('dqn', self.default_config['dqn'])
        
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=dqn_config['hidden_dim'],
            learning_rate=dqn_config['learning_rate'],
            gamma=dqn_config['gamma'],
            tau=dqn_config['tau'],
            epsilon_start=dqn_config['epsilon_start'],
            epsilon_end=dqn_config['epsilon_end'],
            epsilon_decay=dqn_config['epsilon_decay'],
            memory_size=dqn_config['memory_size'],
            batch_size=dqn_config['batch_size'],
            prioritized_replay=dqn_config['prioritized_replay'],
            dueling_network=dqn_config['dueling_network'],
            double_dqn=dqn_config['double_dqn'],
            noisy_nets=dqn_config['noisy_nets'],
            device=device
        )
        
        return agent
        
    def train_model(
        self,
        asset,
        platform,
        start_time=None,
        end_time=None,
        validation_start=None,
        validation_end=None,
        existing_model=None,
        device=None
    ):
        """
        Train a reinforcement learning model for the specified asset and platform.
        
        Args:
            asset: Asset symbol
            platform: Trading platform
            start_time: Start time for training data
            end_time: End time for training data
            validation_start: Start time for validation data
            validation_end: End time for validation data
            existing_model: Path to existing model to continue training
            device: Computing device
            
        Returns:
            tuple: (trained agent, training metrics)
        """
        self.logger.info(f"Training model for {platform}/{asset}")
        
        # Create training environment
        train_env = self.create_environment(
            asset=asset,
            platform=platform,
            start_time=start_time,
            end_time=end_time
        )
        
        # Initialize agent
        agent = self.create_agent(
            state_dim=train_env.observation_space.shape[0],
            action_dim=train_env.action_space.n,
            device=device
        )
        
        # Load existing model if specified
        if existing_model:
            try:
                agent.load_model(existing_model)
                self.logger.info(f"Loaded existing model from {existing_model}")
            except Exception as e:
                self.logger.error(f"Failed to load existing model: {e}")
                
        # Get configuration
        dqn_config = self.config.get('dqn', self.default_config['dqn'])
        
        # Initialize training metrics
        training_metrics = {
            'episode_rewards': [],
            'episode_returns': [],
            'episode_lengths': [],
            'losses': [],
            'win_rates': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'validation_results': []
        }
        
        # Create validation environment if validation period specified
        if validation_start and validation_end:
            val_env = self.create_environment(
                asset=asset,
                platform=platform,
                start_time=validation_start,
                end_time=validation_end
            )
        else:
            val_env = None
            
        # Training loop
        try:
            with Timer() as timer:
                for episode in range(dqn_config['training_episodes']):
                    state, _ = train_env.reset()
                    episode_reward = 0
                    losses = []
                    
                    for t in range(dqn_config['max_timesteps']):
                        # Select action
                        action = agent.select_action(state)
                        
                        # Take action in environment
                        next_state, reward, terminated, truncated, info = train_env.step(action)
                        done = terminated or truncated
                        
                        # Store transition in replay memory
                        agent.store_transition(state, action, reward, next_state, done)
                        
                        # Update state
                        state = next_state
                        episode_reward += reward
                        
                        # Update model
                        if t % dqn_config['update_frequency'] == 0:
                            loss = agent.update_model()
                            if loss is not None:
                                losses.append(loss)
                                
                        if done:
                            break
                            
                    # Collect episode metrics
                    episode_return = (train_env.balance - train_env.initial_balance) / train_env.initial_balance
                    training_metrics['episode_rewards'].append(episode_reward)
                    training_metrics['episode_returns'].append(episode_return)
                    training_metrics['episode_lengths'].append(t + 1)
                    training_metrics['losses'].append(np.mean(losses) if losses else 0)
                    training_metrics['win_rates'].append(
                        train_env.profitable_trades / train_env.total_trades 
                        if train_env.total_trades > 0 else 0
                    )
                    training_metrics['sharpe_ratios'].append(train_env._calculate_sharpe_ratio())
                    training_metrics['max_drawdowns'].append(train_env.max_drawdown)
                    
                    # Validation
                    if val_env and (episode + 1) % 10 == 0:
                        val_results = self.validate_model(agent, val_env, dqn_config['validation_episodes'])
                        training_metrics['validation_results'].append(val_results)
                        
                        # Log validation results
                        self.logger.info(
                            f"Episode {episode+1}/{dqn_config['training_episodes']} - "
                            f"Return: {episode_return:.4f}, "
                            f"Reward: {episode_reward:.4f}, "
                            f"Win Rate: {training_metrics['win_rates'][-1]:.4f}, "
                            f"Val Return: {val_results['mean_return']:.4f}, "
                            f"Val Win Rate: {val_results['win_rate']:.4f}"
                        )
                    else:
                        # Log training progress
                        self.logger.info(
                            f"Episode {episode+1}/{dqn_config['training_episodes']} - "
                            f"Return: {episode_return:.4f}, "
                            f"Reward: {episode_reward:.4f}, "
                            f"Win Rate: {training_metrics['win_rates'][-1]:.4f}, "
                            f"Loss: {np.mean(losses) if losses else 0:.6f}"
                        )
                        
                # Save the trained model
                model_path = self.get_model_path(asset, platform)
                agent.save_model(model_path)
                
                # Add final training time to metrics
                training_metrics['training_time'] = timer.elapsed
                
                self.logger.info(
                    f"Training completed in {timer.elapsed:.2f} seconds. "
                    f"Model saved to {model_path}"
                )
                
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise ModelTrainingError(f"Failed to train model for {platform}/{asset}: {e}")
            
        # Store the trained agent in memory
        model_key = f"{platform}_{asset}"
        self.models[model_key] = agent
        
        return agent, training_metrics
        
    def validate_model(self, agent, env, episodes=10):
        """
        Validate a trained agent on a validation environment.
        
        Args:
            agent: Trained agent
            env: Validation environment
            episodes: Number of validation episodes
            
        Returns:
            dict: Validation metrics
        """
        returns = []
        rewards = []
        win_rates = []
        sharpe_ratios = []
        drawdowns = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            # Run episode without exploration
            done = False
            while not done:
                action = agent.select_action(state, test_mode=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state
                episode_reward += reward
                
            # Collect metrics
            episode_return = (env.balance - env.initial_balance) / env.initial_balance
            returns.append(episode_return)
            rewards.append(episode_reward)
            win_rates.append(env.profitable_trades / env.total_trades if env.total_trades > 0 else 0)
            sharpe_ratios.append(env._calculate_sharpe_ratio())
            drawdowns.append(env.max_drawdown)
            
        # Calculate summary statistics
        results = {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_reward': np.mean(rewards),
            'win_rate': np.mean(win_rates),
            'mean_sharpe': np.mean(sharpe_ratios),
            'mean_drawdown': np.mean(drawdowns),
            'positive_return_rate': np.mean(np.array(returns) > 0)
        }
        
        return results
        
    def get_model(self, asset, platform, load_if_missing=True):
        """
        Get a trained model for the specified asset and platform.
        
        Args:
            asset: Asset symbol
            platform: Trading platform
            load_if_missing: Whether to load the model if not in memory
            
        Returns:
            DQNAgent: The trained agent
        """
        model_key = f"{platform}_{asset}"
        
        # Check if model is already loaded
        if model_key in self.models:
            return self.models[model_key]
            
        # Try to load model if requested
        if load_if_missing:
            model_path = self.get_model_path(asset, platform)
            
            if os.path.exists(model_path):
                # Create a temporary environment to get state and action dimensions
                temp_env = self.create_environment(
                    asset=asset,
                    platform=platform,
                    start_time=datetime.datetime.now() - datetime.timedelta(days=30),
                    end_time=datetime.datetime.now()
                )
                
                # Create agent with appropriate dimensions
                agent = self.create_agent(
                    state_dim=temp_env.observation_space.shape[0],
                    action_dim=temp_env.action_space.n
                )
                
                # Load model weights
                try:
                    agent.load_model(model_path)
                    self.models[model_key] = agent
                    return agent
                except Exception as e:
                    self.logger.error(f"Failed to load model for {platform}/{asset}: {e}")
                    raise
            else:
                self.logger.warning(f"No saved model found for {platform}/{asset} at {model_path}")
                return None
        else:
            self.logger.warning(f"Model for {platform}/{asset} not loaded and load_if_missing=False")
            return None
            
    def get_trading_action(self, asset, platform, state, test_mode=True):
        """
        Get a trading action from a trained model.
        
        Args:
            asset: Asset symbol
            platform: Trading platform
            state: Current environment state
            test_mode: Whether to use test mode (no exploration)
            
        Returns:
            int: Selected action
        """
        # Get the model
        model = self.get_model(asset, platform)
        
        if model is None:
            raise ValueError(f"No model available for {platform}/{asset}")
            
        # Get action from model
        action = model.select_action(state, test_mode=test_mode)
        
        return action
        
    def update_model_online(self, asset, platform, state, action, reward, next_state, done):
        """
        Update a model with a new observation during online learning.
        
        Args:
            asset: Asset symbol
            platform: Trading platform
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            
        Returns:
            float: Loss value (if model was updated)
        """
        # Get the model
        model = self.get_model(asset, platform)
        
        if model is None:
            raise ValueError(f"No model available for {platform}/{asset}")
            
        # Store the transition
        model.store_transition(state, action, reward, next_state, done)
        
        # Update the model (periodically)
        loss = None
        if model.steps_done % self.config.get('dqn', {}).get(
            'update_frequency', self.default_config['dqn']['update_frequency']
        ) == 0:
            loss = model.update_model()
            
        # Save the model periodically
        if model.steps_done % 1000 == 0:
            model_path = self.get_model_path(asset, platform)
            model.save_model(model_path)
            
        return loss
        
    def evaluate_model(self, asset, platform, start_time=None, end_time=None):
        """
        Evaluate a trained model on historical data.
        
        Args:
            asset: Asset symbol
            platform: Trading platform
            start_time: Start time for evaluation
            end_time: End time for evaluation
            
        Returns:
            dict: Evaluation metrics
        """
        # Get the model
        model = self.get_model(asset, platform)
        
        if model is None:
            raise ValueError(f"No model available for {platform}/{asset}")
            
        # Create evaluation environment
        env = self.create_environment(
            asset=asset,
            platform=platform,
            start_time=start_time,
            end_time=end_time
        )
        
        # Run a single evaluation episode
        state, _ = env.reset()
        done = False
        
        # Track actions for analysis
        actions = []
        states = []
        rewards = []
        balances = []
        positions = []
        prices = []
        
        while not done:
            # Store state
            states.append(state)
            
            # Get action from model
            action = model.select_action(state, test_mode=True)
            actions.append(action)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            
            # Track metrics
            balances.append(env.balance)
            positions.append(env.position)
            prices.append(info['current_price'])
            
            # Update state
            state = next_state
            
        # Calculate evaluation metrics
        metrics = {
            'final_balance': env.balance,
            'initial_balance': env.initial_balance,
            'total_return': env.total_return,
            'total_trades': env.total_trades,
            'profitable_trades': env.profitable_trades,
            'win_rate': env.profitable_trades / env.total_trades if env.total_trades > 0 else 0,
            'max_drawdown': env.max_drawdown,
            'sharpe_ratio': env._calculate_sharpe_ratio(),
            'actions': actions,
            'rewards': rewards,
            'balances': balances,
            'positions': positions,
            'prices': prices,
            'timestamps': env.market_data.index[env.state_lookback:env.current_idx].tolist()
        }
        
        return metrics
        
    def get_model_summary(self, asset, platform):
        """
        Get a summary of model information.
        
        Args:
            asset: Asset symbol
            platform: Trading platform
            
        Returns:
            dict: Model information
        """
        model_path = self.get_model_path(asset, platform)
        
        if not os.path.exists(model_path):
            return {
                'asset': asset,
                'platform': platform,
                'exists': False,
                'message': 'Model not found'
            }
            
        # Load model metadata
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            config = checkpoint.get('config', {})
            steps_done = checkpoint.get('steps_done', 0)
            
            # Get model file info
            file_info = os.stat(model_path)
            modified_time = datetime.datetime.fromtimestamp(file_info.st_mtime)
            file_size = file_info.st_size / 1024  # size in KB
            
            return {
                'asset': asset,
                'platform': platform,
                'exists': True,
                'steps_trained': steps_done,
                'modified_time': modified_time.isoformat(),
                'file_size_kb': file_size,
                'hidden_dim': config.get('hidden_dim'),
                'dueling_network': config.get('dueling_network'),
                'double_dqn': config.get('double_dqn'),
                'noisy_nets': config.get('noisy_nets'),
                'prioritized_replay': config.get('prioritized_replay')
            }
        except Exception as e:
            return {
                'asset': asset,
                'platform': platform,
                'exists': True,
                'error': str(e),
                'message': 'Error loading model metadata'
            }
            
    def delete_model(self, asset, platform):
        """
        Delete a trained model.
        
        Args:
            asset: Asset symbol
            platform: Trading platform
            
        Returns:
            bool: Whether the model was deleted
        """
        model_key = f"{platform}_{asset}"
        model_path = self.get_model_path(asset, platform)
        
        # Remove from memory if loaded
        if model_key in self.models:
            del self.models[model_key]
            
        # Delete file if exists
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                self.logger.info(f"Deleted model for {platform}/{asset}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to delete model for {platform}/{asset}: {e}")
                return False
        else:
            self.logger.warning(f"No model found to delete for {platform}/{asset}")
            return False
            
    def get_optimization_metrics(self, asset, platform, start_time=None, end_time=None):
        """
        Get metrics for hyperparameter optimization.
        
        Args:
            asset: Asset symbol
            platform: Trading platform
            start_time: Start time for evaluation
            end_time: End time for evaluation
            
        Returns:
            dict: Optimization metrics
        """
        eval_metrics = self.evaluate_model(
            asset=asset,
            platform=platform,
            start_time=start_time,
            end_time=end_time
        )
        
        # Return subset of metrics relevant for optimization
        return {
            'total_return': eval_metrics['total_return'],
            'win_rate': eval_metrics['win_rate'],
            'sharpe_ratio': eval_metrics['sharpe_ratio'],
            'max_drawdown': eval_metrics['max_drawdown'],
            'num_trades': eval_metrics['total_trades']
        }
        
    def optimize_hyperparameters(
        self,
        asset,
        platform,
        train_start,
        train_end,
        val_start,
        val_end,
        n_trials=20,
        optimization_metric='sharpe_ratio'
    ):
        """
        Optimize hyperparameters using Bayesian optimization.
        
        Args:
            asset: Asset symbol
            platform: Trading platform
            train_start: Start time for training data
            train_end: End time for training data
            val_start: Start time for validation data
            val_end: End time for validation data
            n_trials: Number of optimization trials
            optimization_metric: Metric to optimize
            
        Returns:
            dict: Optimization results
        """
        # This would be implemented using a Bayesian optimization library like Optuna
        # For brevity, this implementation is omitted but would include defining parameter
        # search spaces and running multiple training trials with different configurations
        
        self.logger.info(
            f"Hyperparameter optimization for {platform}/{asset} would require "
            f"an external optimization library like Optuna. Not implemented in this example."
        )
        
        return {
            'message': 'Hyperparameter optimization not implemented in this example',
            'asset': asset,
            'platform': platform
        }

# Backward compatibility
ReinforcementLearner = ReinforcementLearningService

