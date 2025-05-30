"""
Base Strategy for QuantumSpectre Intelligence System.

This module defines the core interfaces and base classes for all trading strategies.
"""

import abc
import time
import datetime
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

from common.logger import get_logger
from common.database import Database
from common.utils.metrics import calculate_sharpe_ratio, calculate_sortino_ratio
from common.utils.decorators import performance_monitor

logger = get_logger(__name__)

class Signal:
    """Trading signal class with confidence and metadata."""
    
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"
    
    def __init__(
        self, 
        action: str, 
        confidence: float, 
        source: str,
        asset: str,
        timestamp: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expiration: Optional[int] = None,
        id: Optional[str] = None
    ):
        """
        Initialize a trading signal.
        
        Args:
            action: Signal action (BUY, SELL, HOLD, CLOSE)
            confidence: Signal confidence [0.0-1.0]
            source: Signal source identifier
            asset: Asset symbol
            timestamp: Signal creation time (milliseconds)
            metadata: Additional signal metadata
            expiration: Signal expiration time (milliseconds)
            id: Unique signal identifier
        """
        self.action = action.upper()
        self.confidence = min(max(confidence, 0.0), 1.0)  # Clamp to [0,1]
        self.source = source
        self.asset = asset
        self.timestamp = timestamp or int(time.time() * 1000)
        self.metadata = metadata or {}
        self.expiration = expiration
        self.id = id or str(uuid.uuid4())
        
        # Validate action
        if self.action not in [self.BUY, self.SELL, self.HOLD, self.CLOSE]:
            raise ValueError(f"Invalid signal action: {self.action}")
    
    def is_expired(self) -> bool:
        """Check if signal has expired."""
        if not self.expiration:
            return False
        return int(time.time() * 1000) > self.expiration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            "id": self.id,
            "action": self.action,
            "confidence": self.confidence,
            "source": self.source,
            "asset": self.asset,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "expiration": self.expiration
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Signal':
        """Create signal from dictionary."""
        return cls(
            action=data["action"],
            confidence=data["confidence"],
            source=data["source"],
            asset=data["asset"],
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata"),
            expiration=data.get("expiration"),
            id=data.get("id")
        )
    
    def __repr__(self) -> str:
        """String representation of signal."""
        return f"Signal({self.action}, {self.confidence:.2f}, {self.source}, {self.asset})"


class StrategyPerformance:
    """Strategy performance tracking and metrics."""
    
    def __init__(self, strategy_id: str, asset: str):
        """
        Initialize strategy performance tracker.
        
        Args:
            strategy_id: Strategy identifier
            asset: Asset symbol
        """
        self.strategy_id = strategy_id
        self.asset = asset
        self.signals: List[Signal] = []
        self.trades: List[Dict[str, Any]] = []
        self.correct_signals = 0
        self.total_signals = 0
        self.pnl = 0.0
        self.win_streak = 0
        self.lose_streak = 0
        self.current_streak = 0
        self.max_drawdown = 0.0
        self.last_update = int(time.time() * 1000)
        self.returns: List[float] = []
        self.db = Database()
    
    def add_signal(self, signal: Signal) -> None:
        """
        Add a new signal to performance history.
        
        Args:
            signal: Trading signal
        """
        self.signals.append(signal)
        self.total_signals += 1
        self.last_update = int(time.time() * 1000)
        
        # Store in database
        self.db.insert(
            "strategy_signals",
            signal.to_dict()
        )
    
    def add_trade_result(self, trade: Dict[str, Any]) -> None:
        """
        Add a trade result to performance history.
        
        Args:
            trade: Trade result with PnL
        """
        self.trades.append(trade)
        self.pnl += trade.get("pnl", 0)
        self.returns.append(trade.get("pnl", 0) / trade.get("size", 1))
        
        # Update streaks
        if trade.get("pnl", 0) > 0:
            if self.current_streak >= 0:
                self.current_streak += 1
            else:
                self.current_streak = 1
            self.win_streak = max(self.win_streak, self.current_streak)
            self.correct_signals += 1
        elif trade.get("pnl", 0) < 0:
            if self.current_streak <= 0:
                self.current_streak -= 1
            else:
                self.current_streak = -1
            self.lose_streak = min(self.lose_streak, self.current_streak)
        
        # Update drawdown
        peak = max([0] + [sum(self.returns[:i+1]) for i in range(len(self.returns))])
        current = sum(self.returns)
        drawdown = peak - current
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Store in database
        self.db.insert(
            "strategy_trades",
            {
                "strategy_id": self.strategy_id,
                "asset": self.asset,
                "trade_id": trade.get("id", str(uuid.uuid4())),
                "timestamp": trade.get("timestamp", int(time.time() * 1000)),
                "pnl": trade.get("pnl", 0),
                "entry_price": trade.get("entry_price", 0),
                "exit_price": trade.get("exit_price", 0),
                "size": trade.get("size", 0),
                "direction": trade.get("direction", ""),
                "signal_id": trade.get("signal_id", ""),
                "metadata": trade.get("metadata", {})
            }
        )
    
    def get_win_rate(self) -> float:
        """Calculate win rate as percentage."""
        if self.total_signals == 0:
            return 0.0
        return (self.correct_signals / self.total_signals) * 100.0
    
    def get_expectancy(self) -> float:
        """Calculate mathematical expectancy."""
        if not self.trades:
            return 0.0
        
        wins = [t.get("pnl", 0) for t in self.trades if t.get("pnl", 0) > 0]
        losses = [abs(t.get("pnl", 0)) for t in self.trades if t.get("pnl", 0) < 0]
        
        if not wins:
            avg_win = 0.0
        else:
            avg_win = sum(wins) / len(wins)
            
        if not losses:
            avg_loss = 0.0
        else:
            avg_loss = sum(losses) / len(losses)
            
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        
        if avg_loss == 0:
            return 0.0
            
        return (win_rate * (avg_win / avg_loss)) - (1 - win_rate)
    
    def get_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        if len(self.returns) < 2:
            return 0.0
        return calculate_sharpe_ratio(self.returns)
    
    def get_sortino_ratio(self) -> float:
        """Calculate Sortino ratio."""
        if len(self.returns) < 2:
            return 0.0
        return calculate_sortino_ratio(self.returns)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics."""
        return {
            "strategy_id": self.strategy_id,
            "asset": self.asset,
            "win_rate": self.get_win_rate(),
            "total_signals": self.total_signals,
            "correct_signals": self.correct_signals,
            "pnl": self.pnl,
            "win_streak": self.win_streak,
            "lose_streak": abs(self.lose_streak),
            "current_streak": self.current_streak,
            "max_drawdown": self.max_drawdown,
            "expectancy": self.get_expectancy(),
            "sharpe_ratio": self.get_sharpe_ratio(),
            "sortino_ratio": self.get_sortino_ratio(),
            "last_update": self.last_update
        }
    
    def save_metrics(self) -> None:
        """Save performance metrics to database."""
        metrics = self.get_metrics()
        metrics["timestamp"] = int(time.time() * 1000)
        
        self.db.insert_or_update(
            "strategy_performance",
            {"strategy_id": self.strategy_id, "asset": self.asset},
            metrics
        )
    
    @classmethod
    def load(cls, strategy_id: str, asset: str) -> 'StrategyPerformance':
        """
        Load strategy performance from database.
        
        Args:
            strategy_id: Strategy identifier
            asset: Asset symbol
            
        Returns:
            StrategyPerformance instance
        """
        instance = cls(strategy_id, asset)
        db = Database()
        
        # Load performance metrics
        metrics = db.find_one(
            "strategy_performance",
            {"strategy_id": strategy_id, "asset": asset}
        )
        
        if metrics:
            instance.correct_signals = metrics.get("correct_signals", 0)
            instance.total_signals = metrics.get("total_signals", 0)
            instance.pnl = metrics.get("pnl", 0.0)
            instance.win_streak = metrics.get("win_streak", 0)
            instance.lose_streak = metrics.get("lose_streak", 0)
            instance.current_streak = metrics.get("current_streak", 0)
            instance.max_drawdown = metrics.get("max_drawdown", 0.0)
            instance.last_update = metrics.get("last_update", int(time.time() * 1000))
        
        # Load recent trades and calculate returns
        trades = db.find(
            "strategy_trades",
            {"strategy_id": strategy_id, "asset": asset},
            sort=[("timestamp", -1)],
            limit=1000
        )
        
        for trade in trades:
            instance.trades.append(trade)
            instance.returns.append(trade.get("pnl", 0) / trade.get("size", 1))
        
        # Load recent signals
        signals = db.find(
            "strategy_signals",
            {"source": strategy_id, "asset": asset},
            sort=[("timestamp", -1)],
            limit=1000
        )
        
        for signal_data in signals:
            instance.signals.append(Signal.from_dict(signal_data))
        
        return instance


class BaseStrategy(abc.ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, name: str, asset: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name
            asset: Asset symbol
            config: Strategy configuration
        """
        self.name = name
        self.asset = asset
        self.config = config or {}
        self.id = f"{name}_{asset}_{hash(str(self.config)) % 10000}"
        self.logger = get_logger(f"strategy.{self.id}")
        self.performance = StrategyPerformance.load(self.id, asset)
        self.enabled = True
        self.last_signal_time = 0
        self.signal_cooldown = self.config.get("signal_cooldown", 60000)  # 1 minute default
        
        self.logger.info(f"Initialized strategy {self.id} for {asset}")
    
    @abc.abstractmethod
    def evaluate(self, data: Dict[str, Any]) -> Signal:
        """
        Evaluate market data and generate trading signal.
        
        Args:
            data: Market data dictionary
            
        Returns:
            Signal instance
        """
        pass
    
    def evaluate_with_confidence(self, data: Dict[str, Any]) -> Signal:
        """
        Evaluate market data with confidence-based filtering.
        
        Args:
            data: Market data dictionary
            
        Returns:
            Signal instance
        """
        # Check cooldown
        current_time = int(time.time() * 1000)
        time_since_last = current_time - self.last_signal_time
        
        if time_since_last < self.signal_cooldown:
            self.logger.debug(f"In cooldown period ({time_since_last}ms < {self.signal_cooldown}ms)")
            return Signal(Signal.HOLD, 0.0, self.id, self.asset)
        
        # Evaluate with performance monitoring
        with performance_monitor(f"strategy_evaluation_{self.id}"):
            signal = self.evaluate(data)
        
        # Filter by confidence threshold
        min_confidence = self.config.get("min_confidence", 0.1)
        if signal.action != Signal.HOLD and signal.confidence < min_confidence:
            self.logger.debug(f"Signal confidence too low: {signal.confidence:.2f} < {min_confidence:.2f}")
            return Signal(Signal.HOLD, 0.0, self.id, self.asset)
        
        # Update timestamp and track performance
        if signal.action != Signal.HOLD:
            self.last_signal_time = current_time
            self.performance.add_signal(signal)
            self.performance.save_metrics()
            self.logger.info(f"Generated {signal.action} signal with {signal.confidence:.2f} confidence")
        
        return signal
    
    def get_performance(self) -> Dict[str, Any]:
        """Get strategy performance metrics."""
        return self.performance.get_metrics()
    
    def enable(self) -> None:
        """Enable the strategy."""
        self.enabled = True
        self.logger.info(f"Enabled strategy {self.id}")
    
    def disable(self) -> None:
        """Disable the strategy."""
        self.enabled = False
        self.logger.info(f"Disabled strategy {self.id}")
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update strategy configuration.
        
        Args:
            config: New configuration values
        """
        self.config.update(config)
        self.logger.info(f"Updated configuration for {self.id}: {config}")
    
    def add_trade_result(self, trade: Dict[str, Any]) -> None:
        """
        Add a trade result to performance history.
        
        Args:
            trade: Trade result with PnL
        """
        self.performance.add_trade_result(trade)
        self.performance.save_metrics()
    
    def __repr__(self) -> str:
        """String representation of strategy."""
        return f"{self.name}({self.asset})"


class StrategyManager:
    """Manager for registering, tracking, and evaluating strategies."""
    
    def __init__(self):
        """Initialize strategy manager."""
        self.strategies: Dict[str, BaseStrategy] = {}
        self.logger = get_logger("strategy_manager")
        self.db = Database()
    
    def register(self, strategy: BaseStrategy) -> None:
        """
        Register a strategy with the manager.
        
        Args:
            strategy: Strategy instance
        """
        self.strategies[strategy.id] = strategy
        self.logger.info(f"Registered strategy {strategy.id}")
    
    def unregister(self, strategy_id: str) -> None:
        """
        Unregister a strategy from the manager.
        
        Args:
            strategy_id: Strategy identifier
        """
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            self.logger.info(f"Unregistered strategy {strategy_id}")
    
    def get_strategy(self, strategy_id: str) -> Optional[BaseStrategy]:
        """
        Get a strategy by ID.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Strategy instance or None if not found
        """
        return self.strategies.get(strategy_id)
    
    def get_all_strategies(self) -> List[BaseStrategy]:
        """Get all registered strategies."""
        return list(self.strategies.values())
    
    def get_strategies_for_asset(self, asset: str) -> List[BaseStrategy]:
        """
        Get all strategies for a specific asset.
        
        Args:
            asset: Asset symbol
            
        Returns:
            List of strategy instances
        """
        return [s for s in self.strategies.values() if s.asset == asset and s.enabled]
    
    def evaluate_all(self, data: Dict[str, Any]) -> List[Signal]:
        """
        Evaluate all enabled strategies for the given data.
        
        Args:
            data: Market data dictionary
            
        Returns:
            List of signals from all strategies
        """
        asset = data.get("symbol", "")
        signals = []
        
        if not asset:
            self.logger.warning("No asset symbol in data, skipping evaluation")
            return signals
        
        strategies = self.get_strategies_for_asset(asset)
        
        for strategy in strategies:
            try:
                signal = strategy.evaluate_with_confidence(data)
                if signal.action != Signal.HOLD:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"Error evaluating strategy {strategy.id}: {str(e)}")
        
        return signals
    
    def get_performance_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all strategies."""
        metrics = {}
        
        for strategy_id, strategy in self.strategies.items():
            metrics[strategy_id] = strategy.get_performance()
        
        return metrics
    
    def add_trade_result(self, trade: Dict[str, Any]) -> None:
        """
        Add a trade result to the appropriate strategy.
        
        Args:
            trade: Trade result with strategy_id and PnL
        """
        strategy_id = trade.get("strategy_id")
        if not strategy_id:
            self.logger.warning("No strategy_id in trade result, skipping")
            return
        
        strategy = self.get_strategy(strategy_id)
        if strategy:
            strategy.add_trade_result(trade)
        else:
            self.logger.warning(f"Strategy {strategy_id} not found for trade result")
    
    def save_all_metrics(self) -> None:
        """Save metrics for all strategies."""
        for strategy in self.strategies.values():
            strategy.performance.save_metrics()
    
    def enable_strategy(self, strategy_id: str) -> bool:
        """
        Enable a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Success status
        """
        strategy = self.get_strategy(strategy_id)
        if strategy:
            strategy.enable()
            return True
        return False
    
    def disable_strategy(self, strategy_id: str) -> bool:
        """
        Disable a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Success status
        """
        strategy = self.get_strategy(strategy_id)
        if strategy:
            strategy.disable()
            return True
        return False
    
    def update_strategy_config(self, strategy_id: str, config: Dict[str, Any]) -> bool:
        """
        Update a strategy's configuration.
        
        Args:
            strategy_id: Strategy identifier
            config: New configuration values
            
        Returns:
            Success status
        """
        strategy = self.get_strategy(strategy_id)
        if strategy:
            strategy.update_config(config)
            return True
        return False


# Create a global strategy manager instance
strategy_manager = StrategyManager()
