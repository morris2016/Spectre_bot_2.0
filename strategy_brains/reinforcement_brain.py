#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Reinforcement Learning Based Trading Strategy

This module implements a specialized trading brain that uses reinforcement
learning algorithms to dynamically adapt to market conditions and optimize
trading decisions through experience.
"""

import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import logging
import traceback
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, Flatten, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    tf = None
    Sequential = Model = load_model = None
    Dense = LSTM = GRU = Conv1D = Flatten = Input = Concatenate = None
    Adam = None
    TF_AVAILABLE = False
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
import random
from collections import deque

# Internal imports
from common.logger import get_logger
from common.utils import calculate_sharpe_ratio, calculate_max_drawdown
from common.constants import TIMEFRAMES, REWARD_FUNCTIONS
from common.exceptions import StrategyError, ModelLoadError
from feature_service.features.technical import TechnicalFeatures
from feature_service.features.volatility import VolatilityFeatures
from strategy_brains.base_brain import BaseBrain
try:
    from ml_models.models.deep_learning import create_deep_policy_network
    from ml_models.hardware.gpu import optimize_for_gpu, get_gpu_memory_usage
    from ml_models.rl import DQNAgent
except Exception:  # pragma: no cover - optional dependency
    create_deep_policy_network = None  # type: ignore
    optimize_for_gpu = lambda: None
    get_gpu_memory_usage = lambda: 0
    DQNAgent = None  # type: ignore

logger = get_logger("ReinforcementBrain")


class TradingEnvironment:
    """Simplified trading environment used for testing."""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 1000.0,
        max_position: float = 1.0,
        transaction_fee: float = 0.001,
        reward_function: str = "sharpe",
        window_size: int = 50,
        use_position_info: bool = True,
        action_type: str = "discrete",
    ) -> None:
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance

        self.window_size = window_size
        self.current_step = window_size
        self.balance = initial_balance
        self.position = 0.0

    def _get_observation(self) -> np.ndarray:
        start = self.current_step - self.window_size
        return self.data.iloc[start:self.current_step].values.astype(np.float32)

    def reset(self):
        self.current_step = self.window_size
        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        obs = self._get_observation()
        terminated = self.current_step >= len(self.data)
        truncated = False
        reward = 0.0
        return obs, reward, terminated, truncated, {}
    
    def _calculate_reward(self, action, prev_portfolio_value, current_portfolio_value):
        """
        Calculate reward based on selected reward function.
        
        Args:
            action: Action taken
            prev_portfolio_value: Portfolio value before action
            current_portfolio_value: Portfolio value after action
            
        Returns:
            reward: Calculated reward value
        """
        if self.reward_function == 'simple_return':
            # Simple return from previous step
            return (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
            
        elif self.reward_function == 'sharpe':
            # Approximate Sharpe ratio based on recent returns
            if len(self.portfolio_values) < 20:
                return 0
                
            # Calculate returns for last 20 steps
            returns = np.diff(self.portfolio_values[-20:]) / self.portfolio_values[-21:-1]
            
            if len(returns) > 0:
                sharpe = (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252/20)
                return sharpe
            return 0
            
        elif self.reward_function == 'risk_adjusted':
            # Risk-adjusted return that penalizes large drawdowns
            if len(self.portfolio_values) < 5:
                return 0
                
            # Calculate return
            ret = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
            
            # Calculate max drawdown over recent window
            recent_values = self.portfolio_values[-20:]
            max_drawdown = 0
            peak = recent_values[0]
            
            for value in recent_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Penalize return by drawdown factor
            return ret - (max_drawdown * 2)
            
        else:
            # Default to simple difference
            return current_portfolio_value - prev_portfolio_value
    

    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (buy/hold/sell or continuous value)
            
        Returns:
            observation: New state
            reward: Reward for the action
            terminated: Whether the episode has terminated
            truncated: Whether the episode was truncated
            info: Additional information
        """
        if self.current_step >= len(self.data) - 1:
            terminated = True
            return self._get_observation(), 0, terminated, False, {}
        
        # Get current price data
        current_price = self.data.iloc[self.current_step]['close']
        
        # Calculate portfolio value before trade
        prev_portfolio_value = self.balance + self.position_value
        
        # Process the action
        if self.action_type == 'discrete':
            self._process_discrete_action(action, current_price)
        else:
            self._process_continuous_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Get new price after action
        new_price = self.data.iloc[self.current_step]['close']
        
        # Update position value
        if self.position != 0:
            self.position_value = self.position * new_price
        
        # Calculate total portfolio value
        portfolio_value = self.balance + self.position_value
        self.portfolio_values.append(portfolio_value)
        
        # Calculate reward
        reward = self._calculate_reward(action, prev_portfolio_value, portfolio_value)
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        # Information dictionary
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'position': self.position,
            'position_value': self.position_value,
            'trades': len(self.trades),
            'timestamp': self.data.iloc[self.current_step]['timestamp']
        }
        
        terminated = done
        truncated = False
        return self._get_observation(), reward, terminated, truncated, info
    
    def _process_discrete_action(self, action, current_price):
        """
        Process a discrete action (sell/hold/buy).
        
        Args:
            action: Discrete action (0=sell, 1=hold, 2=buy)
            current_price: Current asset price
        """
        if action == 0:  # Sell
            if self.position > 0:
                # Calculate sale value after fees
                sale_value = self.position * current_price * (1 - self.transaction_fee)
                self.balance += sale_value
                
                # Record trade
                self.trades.append({
                    'type': 'sell',
                    'step': self.current_step,
                    'price': current_price,
                    'amount': self.position,
                    'value': sale_value,
                    'fee': self.position * current_price * self.transaction_fee
                })
                
                # Clear position
                self.position = 0
                self.position_value = 0
                self.entry_price = 0
        
        elif action == 2:  # Buy
            if self.position < self.max_position * self.balance / current_price:
                # Calculate how much to buy (target a position of max_position)
                target_position = self.max_position * self.balance / current_price
                amount_to_buy = min(target_position - self.position, 
                                   self.balance / (current_price * (1 + self.transaction_fee)))
                
                if amount_to_buy > 0:
                    # Calculate cost including fees
                    cost = amount_to_buy * current_price * (1 + self.transaction_fee)
                    self.balance -= cost
                    
                    # Update position
                    if self.position == 0:
                        self.entry_price = current_price
                    else:
                        # Calculate weighted average entry price
                        self.entry_price = (self.entry_price * self.position + 
                                          current_price * amount_to_buy) / (self.position + amount_to_buy)
                    
                    self.position += amount_to_buy
                    self.position_value = self.position * current_price
                    
                    # Record trade
                    self.trades.append({
                        'type': 'buy',
                        'step': self.current_step,
                        'price': current_price,
                        'amount': amount_to_buy,
                        'value': amount_to_buy * current_price,
                        'fee': amount_to_buy * current_price * self.transaction_fee
                    })
    
    def _process_continuous_action(self, action, current_price):
        """
        Process a continuous action from -1.0 to 1.0.
        
        Args:
            action: Continuous action value
            current_price: Current asset price
        """
        # Convert continuous action to target position (-1.0 = fully short, 1.0 = fully long)
        target_position = float(action[0]) * self.max_position * self.balance / current_price
        
        # Calculate position difference
        position_diff = target_position - self.position
        
        if position_diff > 0:  # Buy
            amount_to_buy = min(position_diff, 
                              self.balance / (current_price * (1 + self.transaction_fee)))
            
            if amount_to_buy > 0:
                # Calculate cost including fees
                cost = amount_to_buy * current_price * (1 + self.transaction_fee)
                self.balance -= cost
                
                # Update position
                if self.position == 0:
                    self.entry_price = current_price
                else:
                    # Calculate weighted average entry price
                    self.entry_price = (self.entry_price * self.position + 
                                      current_price * amount_to_buy) / (self.position + amount_to_buy)
                
                self.position += amount_to_buy
                self.position_value = self.position * current_price
                
                # Record trade
                self.trades.append({
                    'type': 'buy',
                    'step': self.current_step,
                    'price': current_price,
                    'amount': amount_to_buy,
                    'value': amount_to_buy * current_price,
                    'fee': amount_to_buy * current_price * self.transaction_fee
                })
                
        elif position_diff < 0:  # Sell
            amount_to_sell = min(abs(position_diff), self.position)
            
            if amount_to_sell > 0:
                # Calculate sale value after fees
                sale_value = amount_to_sell * current_price * (1 - self.transaction_fee)
                self.balance += sale_value
                
                # Record trade
                self.trades.append({
                    'type': 'sell',
                    'step': self.current_step,
                    'price': current_price,
                    'amount': amount_to_sell,
                    'value': sale_value,
                    'fee': amount_to_sell * current_price * self.transaction_fee
                })
                
                # Update position
                self.position -= amount_to_sell
                if self.position <= 1e-9:  # Account for floating point errors
                    self.position = 0
                    self.position_value = 0
                    self.entry_price = 0
                else:
                    self.position_value = self.position * current_price
    
    def render(self, mode='human'):
        """
        Render the environment (not implemented for trading environment).
        """
        pass



class LegacyDQNAgent:
    """
    Deep Q-Network Agent for reinforcement learning-based trading.
    """

    def __init__(self,
                state_size: Tuple[int, int],
                action_size: int,
                learning_rate: float = 0.001,
                gamma: float = 0.95,
                epsilon: float = 1.0,
                epsilon_min: float = 0.01,
                epsilon_decay: float = 0.995,
                batch_size: int = 32,
                memory_size: int = 10000):
        """
        Initialize the DQN Agent.

        Args:
            state_size: Shape of the state observation
            action_size: Number of possible actions
            learning_rate: Learning rate for the model
            gamma: Discount factor for future rewards
            epsilon: Exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            batch_size: Batch size for training
            memory_size: Size of the replay memory
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        # Build main model and target model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # Tracking variables
        self.target_update_counter = 0
        self.losses = []

    def _build_model(self):
        """
        Build the neural network model for the agent.

        Returns:
            model: Compiled Keras model
        """
        # Use GPU optimization if available
        optimize_for_gpu()

        # Get state shape
        timesteps, features = self.state_size

        # Input layer
        input_layer = Input(shape=(timesteps, features))

        # Feature extraction layers
        conv1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(input_layer)
        conv2 = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(conv1)

        # Sequential processing
        lstm1 = LSTM(128, return_sequences=True)(conv2)
        lstm2 = LSTM(64)(lstm1)

        # Action value prediction
        dense1 = Dense(64, activation='relu')(lstm2)
        output = Dense(self.action_size, activation='linear')(dense1)

        # Compile model
        model = Model(inputs=input_layer, outputs=output)
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.learning_rate)
        )

        return model

    def update_target_model(self):
        """
        Copy weights from main model to target model.
        """
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """
        Determine action based on current state.

        Args:
            state: Current state observation
            training: Whether the agent is in training mode

        Returns:
            action: Selected action
        """
        # Reshape state if needed
        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=0)

        # Exploration during training
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Exploitation - use model to predict best action
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size=None):
        """
        Train the model using experience replay.

        Args:
            batch_size: Size of the training batch

        Returns:
            loss: Training loss value
        """
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.memory) < batch_size:
            return 0

        # Sample random batch from memory
        minibatch = random.sample(self.memory, batch_size)

        # Extract batch data
        states = np.zeros((batch_size, *self.state_size))
        next_states = np.zeros((batch_size, *self.state_size))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            next_states[i] = next_state

        # Predict Q-values
        targets = self.model.predict(states, verbose=0)
        next_targets = self.target_model.predict(next_states, verbose=0)

        # Update targets for actions taken
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.gamma * np.max(next_targets[i])

        # Train the model
        history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        loss = history.history['loss'][0]
        self.losses.append(loss)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Periodically update target network
        self.target_update_counter += 1
        if self.target_update_counter >= 10:
            self.update_target_model()
            self.target_update_counter = 0

        return loss

    def load(self, name):
        """
        Load model weights from file.

        Args:
            name: Filename to load weights from
        """
        self.model.load_weights(name)
        self.update_target_model()

    def save(self, name):
        """
        Save model weights to file.

        Args:
            name: Filename to save weights to
        """
        self.model.save_weights(name)


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient Agent for continuous action reinforcement learning.
    """

    def __init__(self,
                state_size: Tuple[int, int],
                action_size: int,
                actor_learning_rate: float = 0.0001,
                critic_learning_rate: float = 0.001,
                gamma: float = 0.99,
                tau: float = 0.001,
                batch_size: int = 64,
                memory_size: int = 10000,
                noise_std: float = 0.1):
        """
        Initialize the DDPG Agent.

        Args:
            state_size: Shape of the state observation
            action_size: Dimension of the action space
            actor_learning_rate: Learning rate for the actor
            critic_learning_rate: Learning rate for the critic
            gamma: Discount factor for future rewards
            tau: Target network update factor
            batch_size: Batch size for training
            memory_size: Size of the replay memory
            noise_std: Standard deviation of exploration noise
        """
        # Ensure action_size is 1 for trading continuous control
        if action_size != 1:
            action_size = 1
            logger.warning(f"Action size set to 1 for DDPG trading agent")

        self.state_size = state_size
        self.action_size = action_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std
        self.memory = deque(maxlen=memory_size)

        # Optimize for GPU
        optimize_for_gpu()

        # Build actor and critic models
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.target_actor = self._build_actor()
        self.target_critic = self._build_critic()

        # Copy initial weights to target networks
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        # Tracking variables
        self.actor_losses = []
        self.critic_losses = []

    def _build_actor(self):
        """
        Build the actor network that selects actions.

        Returns:
            model: Compiled Keras model
        """
        # Get state shape
        timesteps, features = self.state_size

        # Input layer
        input_layer = Input(shape=(timesteps, features))

        # Feature extraction layers
        conv1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(input_layer)
        conv2 = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(conv1)

        # Sequential processing
        lstm1 = LSTM(128, return_sequences=True)(conv2)
        lstm2 = LSTM(64)(lstm1)

        # Action output (-1 to 1)
        dense1 = Dense(64, activation='relu')(lstm2)
        output = Dense(self.action_size, activation='tanh')(dense1)

        # Compile model
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(learning_rate=self.actor_learning_rate))

        return model

    def _build_critic(self):
        """
        Build the critic network that evaluates actions.

        Returns:
            model: Compiled Keras model
        """
        # Get state shape
        timesteps, features = self.state_size

        # State input
        state_input = Input(shape=(timesteps, features))

        # Action input
        action_input = Input(shape=(self.action_size,))

        # State processing
        conv1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(state_input)
        conv2 = Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(conv1)
        lstm1 = LSTM(128, return_sequences=True)(conv2)
        lstm2 = LSTM(64)(lstm1)

        # Combine state and action
        action_dense = Dense(64, activation='relu')(action_input)
        merged = Concatenate()([lstm2, action_dense])

        # Q-value output
        dense1 = Dense(64, activation='relu')(merged)
        output = Dense(1, activation='linear')(dense1)

        # Compile model
        model = Model(inputs=[state_input, action_input], outputs=output)
        model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=self.critic_learning_rate)
        )

        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """
        Determine action based on current state.

        Args:
            state: Current state observation
            training: Whether the agent is in training mode

        Returns:
            action: Selected action
        """
        # Reshape state if needed
        if len(state.shape) == 2:
            state = np.expand_dims(state, axis=0)

        # Get action from actor
        action = self.actor.predict(state, verbose=0)[0]

        # Add exploration noise during training
        if training:
            noise = np.random.normal(0, self.noise_std, size=self.action_size)
            action = np.clip(action + noise, -1.0, 1.0)

        return action

    def _update_target_network(self, target_network, source_network):
        """
        Soft update target network weights.

        Args:
            target_network: Target network to update
            source_network: Source network to get weights from
        """
        target_weights = target_network.get_weights()
        source_weights = source_network.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = self.tau * source_weights[i] + (1 - self.tau) * target_weights[i]

        target_network.set_weights(target_weights)

    def replay(self, batch_size=None):
        """
        Train the models using experience replay.

        Args:
            batch_size: Size of the training batch

        Returns:
            loss: Dictionary with critic and actor losses
        """
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.memory) < batch_size:
            return {'critic_loss': 0, 'actor_loss': 0}

        # Sample random batch from memory
        minibatch = random.sample(self.memory, batch_size)

        # Extract batch data
        states = np.zeros((batch_size, *self.state_size))
        actions = np.zeros((batch_size, self.action_size))
        rewards = np.zeros((batch_size, 1))
        next_states = np.zeros((batch_size, *self.state_size))
        dones = np.zeros((batch_size, 1))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            next_states[i] = next_state
            dones[i] = done

        # Get target actions and Q-values
        target_actions = self.target_actor.predict(next_states, verbose=0)
        target_q_values = self.target_critic.predict([next_states, target_actions], verbose=0)

        # Calculate critic targets
        critic_targets = rewards + self.gamma * target_q_values * (1 - dones)

        # Train critic
        critic_history = self.critic.fit(
            [states, actions],
            critic_targets,
            epochs=1,
            verbose=0,
            batch_size=batch_size
        )
        critic_loss = critic_history.history['loss'][0]
        self.critic_losses.append(critic_loss)

        # Train actor using critic gradients
        # This is done by defining a custom training function
        with tf.GradientTape() as tape:
            # Predict actions
            pred_actions = self.actor(states)
            # Get critic's evaluation
            actor_loss = -tf.reduce_mean(self.critic([states, pred_actions]))

        # Get actor gradients
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)

        # Apply gradients
        self.actor.optimizer.apply_gradients(
            zip(actor_gradients, self.actor.trainable_variables)
        )

        self.actor_losses.append(actor_loss.numpy())

        # Update target networks
        self._update_target_network(self.target_actor, self.actor)
        self._update_target_network(self.target_critic, self.critic)

        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss.numpy()
        }

    def load(self, actor_name, critic_name):
        """
        Load model weights from files.

        Args:
            actor_name: Filename to load actor weights from
            critic_name: Filename to load critic weights from
        """
        self.actor.load_weights(actor_name)
        self.critic.load_weights(critic_name)
        self.target_actor.load_weights(actor_name)
        self.target_critic.load_weights(critic_name)

    def save(self, actor_name, critic_name):
        """
        Save model weights to files.

        Args:
            actor_name: Filename to save actor weights to
            critic_name: Filename to save critic weights to
        """
        self.actor.save_weights(actor_name)
        self.critic.save_weights(critic_name)


class ReinforcementBrain(BaseBrain):
    """
    Trading strategy brain based on reinforcement learning that adapts
    to market conditions through experience.
    """

    def __init__(self,
                 name: str,
                 exchange: str,
                 symbol: str,
                 timeframe: str,
                 config: Dict[str, Any] = None):
        """
        Initialize the reinforcement learning brain.

        Args:
            name: Name of the brain
            exchange: Exchange to trade on
            symbol: Symbol to trade
            timeframe: Timeframe to use
            config: Configuration dictionary
        """
        super().__init__(name, exchange, symbol, timeframe, config)

        # Default configuration
        self.default_config = {
            'model_type': 'dqn',  # 'dqn' or 'ddpg'
            'action_type': 'discrete',  # 'discrete' or 'continuous'
            'reward_function': 'sharpe',  # 'simple_return', 'sharpe', 'risk_adjusted'
            'window_size': 50,
            'batch_size': 64,
            'learning_rate': 0.001,
            'gamma': 0.95,
            'epsilon': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'memory_size': 10000,
            'use_position_info': True,
            'enable_training': True,
            'training_frequency': 5,  # Train every n steps
            'max_position': 1.0,
            'transaction_fee': 0.001,
            'model_save_dir': './models/reinforcement',
            'technical_features': ['rsi', 'macd', 'bollinger_bands', 'atr'],
            'volatility_features': ['historical_volatility', 'parkinson'],
            'enable_gpu': True,
            'use_price_normalization': True,
            'replay_start_size': 1000
        }

        # Update with provided config
        self.config = {**self.default_config, **(config or {})}

        # Initialize feature generators
        self.technical_features = TechnicalFeatures()
        self.volatility_features = VolatilityFeatures()

        # Initialize state
        self.agent = None
        self.env = None
        self.current_state = None
        self.feature_data = None
        self.normalized_data = None
        self.step_counter = 0
        self.total_reward = 0
        self.is_trained = False
        self.training_steps = 0
        self.last_action = None
        self.last_signal = None
        self.current_episode = 0
        self.episode_rewards = []
        self.model_updated = False

        # Configure GPU if available
        if self.config['enable_gpu']:
            try:
                optimize_for_gpu()
                logger.info(f"GPU optimization enabled for {self.name}")
            except Exception as e:
                logger.warning(f"Failed to enable GPU: {str(e)}")

        # Create model directory if it doesn't exist
        os.makedirs(self.config['model_save_dir'], exist_ok=True)

        # Set model paths
        self.model_prefix = f"{self.exchange}_{self.symbol}_{self.timeframe}"
        if self.config['model_type'] == 'dqn':
            self.model_path = os.path.join(
                self.config['model_save_dir'],
                f"{self.model_prefix}_dqn.h5"
            )
        else:
            self.actor_path = os.path.join(
                self.config['model_save_dir'],
                f"{self.model_prefix}_ddpg_actor.h5"
            )
            self.critic_path = os.path.join(
                self.config['model_save_dir'],
                f"{self.model_prefix}_ddpg_critic.h5"
            )

        logger.info(f"Initialized {self.__class__.__name__} for {exchange}:{symbol}:{timeframe}")

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw OHLCV data to create features.

        Args:
            data: Raw OHLCV data

        Returns:
            processed_data: Data with additional features
        """
        # Clone the data to avoid modifying the original
        df = data.copy()

        # Add technical features
        for feature in self.config['technical_features']:
            try:
                feature_df = getattr(self.technical_features, feature)(df)
                # Merge only new columns to avoid duplicates
                for col in feature_df.columns:
                    if col not in df.columns:
                        df[col] = feature_df[col]
            except Exception as e:
                logger.error(f"Error calculating {feature}: {str(e)}")
                logger.error(traceback.format_exc())

        # Add volatility features
        for feature in self.config['volatility_features']:
            try:
                feature_df = getattr(self.volatility_features, feature)(df)
                # Merge only new columns to avoid duplicates
                for col in feature_df.columns:
                    if col not in df.columns:
                        df[col] = feature_df[col]
            except Exception as e:
                logger.error(f"Error calculating {feature}: {str(e)}")
                logger.error(traceback.format_exc())

        # Drop rows with NaN values
        df = df.dropna()

        # Normalize price-related features if enabled
        if self.config['use_price_normalization']:
            # Find columns that might be price-related
            price_cols = [col for col in df.columns if any(
                substring in col.lower() for substring in
                ['price', 'open', 'high', 'low', 'close', 'volume']
            )]

            # Normalize by dividing by the first value
            for col in price_cols:
                if col in df.columns and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    first_valid = df[col].iloc[0]
                    if first_valid != 0:
                        df[col] = df[col] / first_valid

        # Store the preprocessed data
        self.feature_data = df

        return df

    def initialize_environment(self, data: pd.DataFrame):
        """
        Initialize the trading environment with preprocessed data.

        Args:
            data: Preprocessed data with features
        """
        # Create the environment
        self.env = TradingEnvironment(
            data=data,
            initial_balance=1000.0,
            max_position=self.config['max_position'],
            transaction_fee=self.config['transaction_fee'],
            reward_function=self.config['reward_function'],
            window_size=self.config['window_size'],
            use_position_info=self.config['use_position_info'],
            action_type=self.config['action_type']
        )

        # Reset the environment to get initial state
        initial_state, _ = self.env.reset()
        self.current_state = initial_state

        # Determine state and action dimensions
        state_shape = initial_state.shape

        # Initialize the agent based on model type
        if self.config['model_type'] == 'dqn':
            flat_state = state_shape[0] * state_shape[1]
            self.agent = DQNAgent(
                state_dim=flat_state,
                action_dim=self.env.action_space.n,
                learning_rate=self.config['learning_rate'],
                gamma=self.config['gamma'],
                epsilon_start=self.config['epsilon'],
                epsilon_end=self.config['epsilon_min'],
                epsilon_decay=1000,
                batch_size=self.config['batch_size'],
                memory_size=self.config['memory_size']
            )
        else:
            # For continuous actions
            self.agent = DDPGAgent(
                state_size=state_shape,
                action_size=1,  # Single continuous action for trading
                actor_learning_rate=self.config['learning_rate'] * 0.1,
                critic_learning_rate=self.config['learning_rate'],
                gamma=self.config['gamma'],
                tau=0.001,
                batch_size=self.config['batch_size'],
                memory_size=self.config['memory_size'],
                noise_std=0.1
            )

        # Load existing model if available
        self._load_model()

    def _load_model(self):
        """
        Attempt to load saved model weights if they exist.
        """
        try:
            if self.config['model_type'] == 'dqn':
                if os.path.exists(self.model_path):
                    self.agent.load(self.model_path)
                    self.is_trained = True
                    logger.info(f"Loaded DQN model from {self.model_path}")
                    # Reduce epsilon for less exploration if model is already trained
                    self.agent.epsilon = max(self.agent.epsilon_min, self.agent.epsilon * 0.1)
            else:
                if os.path.exists(self.actor_path) and os.path.exists(self.critic_path):
                    self.agent.load(self.actor_path, self.critic_path)
                    self.is_trained = True
                    logger.info(f"Loaded DDPG models from {self.actor_path} and {self.critic_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            logger.error(traceback.format_exc())

    def _save_model(self):
        """
        Save the current model weights.
        """
        try:
            if self.config['model_type'] == 'dqn':
                self.agent.save(self.model_path)
                logger.info(f"Saved DQN model to {self.model_path}")
            else:
                self.agent.save(self.actor_path, self.critic_path)
                logger.info(f"Saved DDPG models to {self.actor_path} and {self.critic_path}")
            self.model_updated = True
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            logger.error(traceback.format_exc())

    def train(self, data: pd.DataFrame, episodes: int = 10) -> Dict[str, Any]:
        """
        Train the reinforcement learning agent on historical data.

        Args:
            data: Historical OHLCV data
            episodes: Number of training episodes

        Returns:
            metrics: Training performance metrics
        """
        # Preprocess data
        processed_data = self.preprocess_data(data)

        # Initialize environment if not already done
        if self.env is None or self.agent is None:
            self.initialize_environment(processed_data)

        # Training metrics
        metrics = {
            'episode_rewards': [],
            'final_portfolio_values': [],
            'losses': [],
            'win_rate': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }

        # Training loop
        for episode in range(episodes):
            self.current_episode = episode
            # Reset environment
            state, _ = self.env.reset()
            self.current_state = state

            done = False
            episode_reward = 0
            trades = []

            # Run one episode
            while not done:
                # Select action
                action = self.agent.select_action(state, test_mode=False)

                # Take action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Store in replay memory
                if self.config['model_type'] == 'dqn':
                    self.agent.store_transition(state, action, reward, next_state, done)
                else:
                    # Reshape action for memory if using DDPG
                    if not isinstance(action, np.ndarray):
                        action = np.array([action])
                    self.agent.remember(state, action, reward, next_state, done)

                # Update state
                state = next_state
                self.current_state = state
                episode_reward += reward

                # Train if we have enough samples
                if len(self.agent.memory) > self.config['replay_start_size']:
                    loss = self.agent.update_model()
                    if isinstance(loss, dict):
                        metrics['losses'].append(loss.get('critic_loss', 0))
                    else:
                        metrics['losses'].append(loss)
                    self.training_steps += 1

                # Record trade if action resulted in a trade
                if info.get('trades', 0) > len(trades):
                    trades.append(info)

            # Episode completed
            metrics['episode_rewards'].append(episode_reward)
            metrics['final_portfolio_values'].append(info['portfolio_value'])

            # Log episode results
            logger.info(f"Episode {episode+1}/{episodes}: Reward={episode_reward:.2f}, "
                       f"Final Value=${info['portfolio_value']:.2f}, "
                       f"Trades={len(trades)}")

            # Save model periodically
            if (episode + 1) % 5 == 0 or episode == episodes - 1:
                self._save_model()

        # Calculate performance metrics
        if len(metrics['final_portfolio_values']) > 0:
            # Filter trades that have profit data
            if hasattr(self.env, 'trades') and len(self.env.trades) > 0:
                profitable_trades = sum(1 for t in self.env.trades if
                                      (t['type'] == 'sell' and t['price'] > self.env.entry_price))
                total_trades = len(self.env.trades)
                metrics['win_rate'] = profitable_trades / total_trades if total_trades > 0 else 0

            # Calculate Sharpe ratio if we have enough portfolio values
            if len(self.env.portfolio_values) > 2:
                returns = np.diff(self.env.portfolio_values) / self.env.portfolio_values[:-1]
                metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns)
                metrics['max_drawdown'] = calculate_max_drawdown(self.env.portfolio_values)

        # Update state
        self.is_trained = True
        self.episode_rewards = metrics['episode_rewards']

        return metrics

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze market data using the reinforcement learning model.

        Args:
            data: OHLCV data to analyze

        Returns:
            analysis: DataFrame with analysis results including signals
        """
        # Preprocess data
        processed_data = self.preprocess_data(data)

        # Initialize environment if not already done
        if self.env is None or self.agent is None:
            self.initialize_environment(processed_data)

        # Create environment for analysis (no training)
        analysis_env = TradingEnvironment(
            data=processed_data,
            initial_balance=1000.0,
            max_position=self.config['max_position'],
            transaction_fee=self.config['transaction_fee'],
            reward_function=self.config['reward_function'],
            window_size=self.config['window_size'],
            use_position_info=self.config['use_position_info'],
            action_type=self.config['action_type']
        )

        # Reset environment
        state, _ = analysis_env.reset()

        # Analysis results
        actions = []
        rewards = []
        portfolio_values = []
        positions = []

        # Run through all data points
        done = False
        while not done:
            # Select action (without exploration)
            action = self.agent.select_action(state, test_mode=True)

            # Take action
            next_state, reward, terminated, truncated, info = analysis_env.step(action)
            done = terminated or truncated

            # Record results
            if self.config['action_type'] == 'discrete':
                # Convert discrete action to signal
                if action == 0:  # Sell
                    signal = -1
                elif action == 2:  # Buy
                    signal = 1
                else:  # Hold
                    signal = 0
            else:
                # Convert continuous action to signal
                signal = float(action[0])  # -1.0 to 1.0

            actions.append(signal)
            rewards.append(reward)
            portfolio_values.append(info['portfolio_value'])
            positions.append(info['position'])

            # Update state
            state = next_state

        # Create analysis DataFrame
        analysis = data.copy()

        # Add analysis results
        analysis_start_idx = self.config['window_size']
        if len(analysis) >= analysis_start_idx + len(actions):
            analysis.loc[analysis.index[analysis_start_idx:analysis_start_idx + len(actions)], 'signal'] = actions
            analysis.loc[analysis.index[analysis_start_idx:analysis_start_idx + len(actions)], 'reward'] = rewards
            analysis.loc[analysis.index[analysis_start_idx:analysis_start_idx + len(actions)], 'portfolio_value'] = portfolio_values
            analysis.loc[analysis.index[analysis_start_idx:analysis_start_idx + len(actions)], 'position'] = positions

        # Fill NaN values
        analysis['signal'] = analysis['signal'].fillna(0)
        analysis['reward'] = analysis['reward'].fillna(0)
        analysis['portfolio_value'] = analysis['portfolio_value'].fillna(1000.0)
        analysis['position'] = analysis['position'].fillna(0)

        return analysis

    def update(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Update the brain with new data and optionally train.

        Args:
            data: New OHLCV data

        Returns:
            update_info: Information about the update
        """
        # Preprocess new data
        processed_data = self.preprocess_data(data)

        # Initialize environment if not already done
        if self.env is None or self.agent is None:
            self.initialize_environment(processed_data)

        # Update state
        try:
            # Get current candle features
            current_features = processed_data.iloc[-1]

            # If training is enabled, perform online training
            if self.config['enable_training']:
                # If we have a current state and last action, we can update
                if self.current_state is not None and self.last_action is not None:
                    # Calculate reward (simplified for online updates)
                    if self.last_signal > 0:  # Long position
                        reward = (current_features['close'] - self.last_price) / self.last_price
                    elif self.last_signal < 0:  # Short position
                        reward = (self.last_price - current_features['close']) / self.last_price
                    else:  # No position
                        reward = 0

                    # Apply transaction fee penalty for trades
                    if self.last_signal != self.last_last_signal:
                        reward -= self.config['transaction_fee']

                    # Create next state (simplified)
                    next_state = self.current_state.copy()
                    next_state = np.roll(next_state, -1, axis=0)
                    next_state[-1] = current_features[self.feature_data.columns].values

                    # Store in replay memory
                    self.agent.store_transition(self.current_state, self.last_action, reward, next_state, False)

                    # Periodically train if we have enough samples
                    self.step_counter += 1
                    if (self.step_counter % self.config['training_frequency'] == 0 and
                        len(self.agent.memory) > self.config['replay_start_size']):
                        loss = self.agent.update_model()
                        self.training_steps += 1

                        # Save model periodically
                        if self.training_steps % 100 == 0:
                            self._save_model()

                    # Update current state
                    self.current_state = next_state
                    self.total_reward += reward

            # Generate prediction for new data
            if self.current_state is not None:
                action = self.agent.select_action(self.current_state, test_mode=True)

                # Convert action to signal
                if self.config['action_type'] == 'discrete':
                    if action == 0:  # Sell
                        signal = -1
                    elif action == 2:  # Buy
                        signal = 1
                    else:  # Hold
                        signal = 0
                else:
                    signal = float(action[0])  # -1.0 to 1.0

                # Store for next update
                self.last_last_signal = self.last_signal
                self.last_signal = signal
                self.last_action = action
                self.last_price = current_features['close']
            else:
                signal = 0

        except Exception as e:
            logger.error(f"Error updating reinforcement brain: {str(e)}")
            logger.error(traceback.format_exc())
            signal = 0

        # Return information about the update
        update_info = {
            'signal': signal,
            'confidence': 0.5,  # Reinforcement learning doesn't provide direct confidence
            'trained': self.is_trained,
            'training_steps': self.training_steps,
            'total_reward': self.total_reward,
            'model_updated': self.model_updated
        }

        # Reset model updated flag
        self.model_updated = False

        return update_info

    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a trading signal for the current market data.

        Args:
            data: Current OHLCV data

        Returns:
            signal_info: Trading signal information
        """
        # Update brain with new data
        update_info = self.update(data)

        # Enhanced signal information
        signal_info = {
            'signal': update_info['signal'],
            'confidence': update_info['confidence'],
            'timestamp': datetime.now().isoformat(),
            'source': self.name,
            'exchange': self.exchange,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'trained': self.is_trained,
            'strategy': 'reinforcement_learning',
            'action_type': self.config['model_type'],
            'metadata': {
                'training_steps': self.training_steps,
                'total_reward': self.total_reward,
                'model_updated': self.model_updated,
                'current_episode': self.current_episode
            }
        }

        # Add risk analysis
        try:
            # Calculate potential risk/reward
            last_close = data['close'].iloc[-1]

            # If we have volatility features, use them for risk estimation
            if 'atr' in data.columns:
                atr = data['atr'].iloc[-1]
                risk_info = {
                    'stop_loss': last_close - (atr * 2) if signal_info['signal'] > 0 else last_close + (atr * 2),
                    'take_profit': last_close + (atr * 3) if signal_info['signal'] > 0 else last_close - (atr * 3),
                    'risk_reward_ratio': 1.5  # 3/2
                }
            else:
                # Use a simple percentage-based approach if ATR not available
                risk_info = {
                    'stop_loss': last_close * 0.98 if signal_info['signal'] > 0 else last_close * 1.02,
                    'take_profit': last_close * 1.03 if signal_info['signal'] > 0 else last_close * 0.97,
                    'risk_reward_ratio': 1.5  # 3%/2%
                }

            signal_info['risk_analysis'] = risk_info

        except Exception as e:
            logger.error(f"Error calculating risk analysis: {str(e)}")

        return signal_info

    def save(self) -> bool:
        """
        Save the brain's state and model.

        Returns:
            success: Whether the save was successful
        """
        try:
            # Save model weights
            self._save_model()

            # Save brain state
            state_path = os.path.join(
                self.config['model_save_dir'],
                f"{self.model_prefix}_state.json"
            )

            state = {
                'is_trained': self.is_trained,
                'training_steps': self.training_steps,
                'total_reward': self.total_reward,
                'last_signal': self.last_signal if hasattr(self, 'last_signal') else None,
                'last_price': self.last_price if hasattr(self, 'last_price') else None,
                'current_episode': self.current_episode,
                'episode_rewards': self.episode_rewards,
                'config': self.config
            }

            with open(state_path, 'w') as f:
                json.dump(state, f)

            logger.info(f"Saved brain state to {state_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save brain state: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def load(self) -> bool:
        """
        Load the brain's state and model.

        Returns:
            success: Whether the load was successful
        """
        try:
            # Load brain state
            state_path = os.path.join(
                self.config['model_save_dir'],
                f"{self.model_prefix}_state.json"
            )

            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)

                self.is_trained = state.get('is_trained', False)
                self.training_steps = state.get('training_steps', 0)
                self.total_reward = state.get('total_reward', 0)
                self.last_signal = state.get('last_signal')
                self.last_price = state.get('last_price')
                self.current_episode = state.get('current_episode', 0)
                self.episode_rewards = state.get('episode_rewards', [])

                # Update config with saved values if any
                if 'config' in state:
                    for key, value in state['config'].items():
                        if key in self.config:
                            self.config[key] = value

                logger.info(f"Loaded brain state from {state_path}")

            # Load model weights
            self._load_model()

            return True

        except Exception as e:
            logger.error(f"Failed to load brain state: {str(e)}")
            logger.error(traceback.format_exc())
            return False


if __name__ == "__main__":
    # Test code
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    # Create sample OHLCV data
    n = 1000
    np.random.seed(42)
    dates = [datetime.now() - timedelta(minutes=i) for i in range(n, 0, -1)]

    # Create a trending market with some noise
    close = np.cumsum(np.random.normal(0, 1, n)) + 1000
    # Add a trend
    close = close + np.linspace(0, 50, n)
    # Add seasonality
    close = close + 20 * np.sin(np.linspace(0, 8 * np.pi, n))

    high = close + np.random.normal(0, 5, n)
    low = close - np.random.normal(0, 5, n)
    open_price = close - np.random.normal(0, 2, n)
    volume = np.random.normal(1000, 100, n) + 200 * np.sin(np.linspace(0, 6 * np.pi, n)) + 1000

    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    # Initialize and test the reinforcement brain
    config = {
        'model_type': 'dqn',
        'window_size': 30,
        'enable_training': True,
        'batch_size': 32,
        'model_save_dir': './test_models'
    }

    brain = ReinforcementBrain(
        name="TestReinforcementBrain",
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1h",
        config=config
    )

    # Train on historical data
    train_data = df.iloc[:-100]
    metrics = brain.train(train_data, episodes=5)
    print(f"Training metrics: {metrics}")

    # Analyze recent data
    test_data = df.iloc[-150:]
    analysis = brain.analyze(test_data)
    print(f"Analysis results: {analysis.tail()}")

    # Get current signal
    signal_info = brain.get_signal(df.iloc[-50:])
    print(f"Current signal: {signal_info}")

    # Save the brain
    brain.save()
