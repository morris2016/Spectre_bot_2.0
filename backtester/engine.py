#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Backtester Engine

This module provides a sophisticated backtesting engine that simulates trading
with high precision, including realistic order execution, slippage modeling,
and comprehensive performance analysis.
"""

import asyncio
import datetime
import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from common.logger import get_logger
from common.constants import (
    ORDER_TYPE, POSITION_SIDE, TIME_FRAMES, 
    EXECUTION_MODES, SLIPPAGE_MODELS
)
from common.exceptions import (
    BacktestConfigError, InvalidTimeRangeError, DataInsufficientError,
    StrategyExecutionError
)
from common.db_client import DBClient
from common.redis_client import RedisClient
from common.metrics import calculate_trading_metrics

from data_storage.time_series import TimeSeriesDB
from data_storage.market_data import MarketDataRepository

from execution_engine.position_manager import PositionManager
from execution_engine.microstructure import MicrostructureAnalyzer
from execution_engine.capital_management import CapitalManager

from risk_manager.position_sizing import PositionSizer
from risk_manager.stop_loss import StopLossManager
from risk_manager.take_profit import TakeProfitManager
from risk_manager.exposure import ExposureManager

from backtester.data_provider import BacktestDataProvider

logger = get_logger(__name__)

class BacktestEngine:
    """
    Advanced backtesting engine that simulates trading strategies with high precision.
    
    Features:
    - Advanced order execution simulation with realistic fill modeling
    - Realistic slippage and market impact models
    - Multi-asset, multi-timeframe backtesting capabilities
    - Performance analysis across various market regimes
    - Monte Carlo simulation for robustness testing
    - Walk-forward optimization and testing
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        data_provider: Optional[BacktestDataProvider] = None,
        strategy_brain = None,
        brain_council = None,
        execution_engine = None,
        risk_manager = None,
        metrics_engine = None,
        db_client: Optional[DBClient] = None,
        redis_client: Optional[RedisClient] = None,
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            config: Configuration dictionary with backtest parameters
            data_provider: Data provider instance or None to create a new one
            strategy_brain: Strategy brain instance to test
            brain_council: Brain council instance for decision aggregation
            execution_engine: Execution engine for order simulation
            risk_manager: Risk management module
            metrics_engine: Performance metrics calculator
            db_client: Database client for persistent storage
            redis_client: Redis client for caching
        """
        self.config = self._validate_config(config)
        self.logger = get_logger(f"{__name__}.{self.config['backtest_id']}")
        
        # Core components
        self.db_client = db_client or DBClient()
        self.redis_client = redis_client or RedisClient()
        self.data_provider = data_provider or BacktestDataProvider(
            config=self.config,
            db_client=self.db_client,
            redis_client=self.redis_client
        )
        
        # Trading components
        self.strategy_brain = strategy_brain
        self.brain_council = brain_council
        self.execution_engine = execution_engine
        self.risk_manager = risk_manager
        self.metrics_engine = metrics_engine
        
        # Internal state
        self.is_running = False
        self.current_time = None
        self.start_time = None
        self.end_time = None
        self.initial_capital = self.config.get('initial_capital', 10000.0)
        self.current_capital = self.initial_capital
        self.positions = {}
        self.orders = {}
        self.trades = []
        self.equity_curve = []
        self.performance_metrics = {}
        
        # Execution models
        self.position_manager = PositionManager()
        self.microstructure_analyzer = MicrostructureAnalyzer()
        self.capital_manager = CapitalManager(
            initial_capital=self.initial_capital,
            risk_per_trade=self.config.get('risk_per_trade', 0.02)
        )
        
        # Risk models
        self.position_sizer = PositionSizer()
        self.stop_loss_manager = StopLossManager()
        self.take_profit_manager = TakeProfitManager()
        self.exposure_manager = ExposureManager()
        
        # Multiprocessing support
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 10)
        )
        
        self.logger.info(f"BacktestEngine initialized with ID: {self.config['backtest_id']}")
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the backtest configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            BacktestConfigError: If configuration is invalid
        """
        # Create a copy to avoid modifying the original
        validated = config.copy()
        
        # Required fields
        required_fields = [
            'start_date', 'end_date', 'assets', 'timeframes'
        ]
        
        for field in required_fields:
            if field not in validated:
                raise BacktestConfigError(f"Missing required config field: {field}")
        
        # Ensure dates are valid
        try:
            if isinstance(validated['start_date'], str):
                validated['start_date'] = pd.to_datetime(validated['start_date'])
            if isinstance(validated['end_date'], str):
                validated['end_date'] = pd.to_datetime(validated['end_date'])
        except Exception as e:
            raise BacktestConfigError(f"Invalid date format: {str(e)}")
            
        if validated['start_date'] >= validated['end_date']:
            raise InvalidTimeRangeError("Start date must be before end date")
        
        # Timeframes validation
        for tf in validated['timeframes']:
            if tf not in TIME_FRAMES:
                raise BacktestConfigError(f"Invalid timeframe: {tf}")
        
        # Execution and slippage models
        if 'execution_mode' not in validated:
            validated['execution_mode'] = EXECUTION_MODES.REALISTIC
        elif validated['execution_mode'] not in EXECUTION_MODES:
            raise BacktestConfigError(f"Invalid execution mode: {validated['execution_mode']}")
            
        if 'slippage_model' not in validated:
            validated['slippage_model'] = SLIPPAGE_MODELS.PROPORTIONAL
        elif validated['slippage_model'] not in SLIPPAGE_MODELS:
            raise BacktestConfigError(f"Invalid slippage model: {validated['slippage_model']}")
        
        # Add backtest_id if not present
        if 'backtest_id' not in validated:
            validated['backtest_id'] = str(uuid.uuid4())
            
        # Default values
        if 'initial_capital' not in validated:
            validated['initial_capital'] = 10000.0
            
        if 'commission_rate' not in validated:
            validated['commission_rate'] = 0.001  # 0.1% by default
            
        return validated
    
    async def load_data(self) -> bool:
        """
        Load historical data for the backtest.
        
        Returns:
            bool: True if data loaded successfully
        """
        self.logger.info("Loading historical data for backtest")
        try:
            await self.data_provider.load_data(
                assets=self.config['assets'],
                timeframes=self.config['timeframes'],
                start_date=self.config['start_date'],
                end_date=self.config['end_date'],
                include_volume=self.config.get('include_volume', True),
                include_orderbook=self.config.get('include_orderbook', False)
            )
            
            # Validate sufficient data
            for asset in self.config['assets']:
                for timeframe in self.config['timeframes']:
                    data = self.data_provider.get_ohlcv(asset, timeframe)
                    if data is None or len(data) < 10:  # Arbitrary minimum
                        raise DataInsufficientError(
                            f"Insufficient data for {asset} at {timeframe}"
                        )
            
            self.logger.info("Historical data loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise
    
    async def initialize_strategy(self) -> bool:
        """
        Initialize the strategy with historical data before starting the backtest.
        
        Returns:
            bool: True if initialization was successful
        """
        self.logger.info("Initializing strategy for backtest")
        
        try:
            # Initialize strategy if provided
            if self.strategy_brain:
                warmup_data = {}
                for asset in self.config['assets']:
                    warmup_data[asset] = {}
                    for timeframe in self.config['timeframes']:
                        warmup_data[asset][timeframe] = self.data_provider.get_ohlcv(
                            asset, timeframe, 
                            limit=self.config.get('warmup_bars', 200)
                        )
                
                await self.strategy_brain.initialize(warmup_data)
            
            # Initialize brain council if provided
            if self.brain_council:
                await self.brain_council.initialize(warmup_data)
                
            self.logger.info("Strategy initialization completed")
            return True
        except Exception as e:
            self.logger.error(f"Strategy initialization failed: {str(e)}")
            raise StrategyExecutionError(f"Failed to initialize strategy: {str(e)}")
    
    async def run(self) -> Dict[str, Any]:
        """
        Run the backtest.
        
        Returns:
            Dict: Performance metrics and backtest results
        """
        self.logger.info(f"Starting backtest run with ID: {self.config['backtest_id']}")
        
        if self.is_running:
            self.logger.warning("Backtest is already running")
            return False
            
        self.is_running = True
        start_time = time.time()
        
        try:
            # Load data if not already loaded
            if not self.data_provider.is_data_loaded():
                await self.load_data()
                
            # Initialize strategy
            await self.initialize_strategy()
            
            # Reset backtest state
            self._reset_backtest_state()
            
            # Get the time iterator
            time_iterator = self._create_time_iterator()
            
            # Main backtest loop
            for current_time in time_iterator:
                self.current_time = current_time
                
                # Skip if outside backtest range
                if current_time < self.config['start_date'] or current_time > self.config['end_date']:
                    continue
                
                # Update market data for current timestamp
                market_data = self._get_current_market_data(current_time)
                
                # Process pending orders
                await self._process_pending_orders(current_time, market_data)
                
                # Generate trading signals
                signals = await self._generate_signals(current_time, market_data)
                
                # Execute signals
                if signals:
                    await self._execute_signals(signals, market_data)
                
                # Update positions
                await self._update_positions(current_time, market_data)
                
                # Record equity point
                self._record_equity_point(current_time)
            
            # Finalize the backtest
            await self._finalize_backtest()
            
            # Calculate performance metrics
            self.performance_metrics = self._calculate_performance_metrics()
            
            # Save results
            await self._save_backtest_results()
            
            execution_time = time.time() - start_time
            self.logger.info(f"Backtest completed in {execution_time:.2f} seconds")
            
            return self.performance_metrics
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {str(e)}")
            self.is_running = False
            raise
        finally:
            self.is_running = False
    
    def _reset_backtest_state(self):
        """Reset the backtest state for a new run."""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.orders = {}
        self.trades = []
        self.equity_curve = []
        self.performance_metrics = {}
        self.current_time = None
        
    def _create_time_iterator(self):
        """Create an iterator for timestamps during the backtest."""
        # Determine the finest timeframe for the iteration
        timeframes = self.config['timeframes']
        base_timeframe = min(timeframes, key=lambda x: TIME_FRAMES[x])
        
        # Get data for the base timeframe
        all_timestamps = set()
        
        for asset in self.config['assets']:
            df = self.data_provider.get_ohlcv(asset, base_timeframe)
            if df is not None and not df.empty:
                all_timestamps.update(df.index.tolist())
        
        # Sort timestamps
        all_timestamps = sorted(all_timestamps)
        
        return all_timestamps
    
    def _get_current_market_data(self, timestamp):
        """Get market data for all assets at the current timestamp."""
        market_data = {}
        
        for asset in self.config['assets']:
            market_data[asset] = {}
            for timeframe in self.config['timeframes']:
                # Get data up to and including the timestamp
                df = self.data_provider.get_ohlcv_until(
                    asset, timeframe, timestamp
                )
                if df is not None and not df.empty:
                    market_data[asset][timeframe] = df
        
        return market_data
    
    async def _process_pending_orders(self, timestamp, market_data):
        """Process pending orders at the current timestamp."""
        orders_to_remove = []
        
        for order_id, order in self.orders.items():
            # Skip if order is expired
            if order.get('expiry') and timestamp > order['expiry']:
                order['status'] = 'EXPIRED'
                orders_to_remove.append(order_id)
                continue
                
            # Get current price
            asset = order['asset']
            current_price = self._get_current_price(asset, market_data)
            
            # Check if order can be executed
            can_execute = False
            
            if order['type'] == ORDER_TYPE.MARKET:
                can_execute = True
                execution_price = current_price
                
            elif order['type'] == ORDER_TYPE.LIMIT:
                if order['side'] == POSITION_SIDE.LONG and current_price <= order['price']:
                    can_execute = True
                    execution_price = order['price']
                elif order['side'] == POSITION_SIDE.SHORT and current_price >= order['price']:
                    can_execute = True
                    execution_price = order['price']
            
            elif order['type'] == ORDER_TYPE.STOP:
                if order['side'] == POSITION_SIDE.LONG and current_price >= order['price']:
                    can_execute = True
                    execution_price = self._apply_slippage(current_price, order)
                elif order['side'] == POSITION_SIDE.SHORT and current_price <= order['price']:
                    can_execute = True
                    execution_price = self._apply_slippage(current_price, order)
            
            # Execute order if conditions met
            if can_execute:
                await self._execute_order(order, execution_price, timestamp)
                orders_to_remove.append(order_id)
        
        # Remove processed orders
        for order_id in orders_to_remove:
            del self.orders[order_id]
    
    def _get_current_price(self, asset, market_data):
        """Get the current price for an asset from market data."""
        # Use the smallest timeframe for most accurate price
        timeframe = min(self.config['timeframes'], key=lambda x: TIME_FRAMES[x])
        
        if asset in market_data and timeframe in market_data[asset]:
            df = market_data[asset][timeframe]
            if not df.empty:
                # Get the last row
                last_row = df.iloc[-1]
                # Return the close price
                return last_row['close']
        
        # Fallback - check other timeframes
        for tf in self.config['timeframes']:
            if asset in market_data and tf in market_data[asset]:
                df = market_data[asset][tf]
                if not df.empty:
                    return df.iloc[-1]['close']
        
        raise ValueError(f"No price data available for {asset} at current time")
    
    def _apply_slippage(self, price, order):
        """Apply slippage model to the execution price."""
        slippage_model = self.config['slippage_model']
        slippage_factor = self.config.get('slippage_factor', 0.0005)  # Default 0.05%
        
        if slippage_model == SLIPPAGE_MODELS.NONE:
            return price
            
        elif slippage_model == SLIPPAGE_MODELS.FIXED:
            # Fixed pip value slippage
            direction = 1 if order['side'] == POSITION_SIDE.LONG else -1
            return price + (direction * slippage_factor)
            
        elif slippage_model == SLIPPAGE_MODELS.PROPORTIONAL:
            # Percentage-based slippage
            direction = 1 if order['side'] == POSITION_SIDE.LONG else -1
            return price * (1 + direction * slippage_factor)
            
        elif slippage_model == SLIPPAGE_MODELS.VARIABLE:
            # Variable slippage based on volatility and volume
            # For realistic simulation
            asset = order['asset']
            volatility = self._calculate_volatility(asset)
            volume = self._calculate_relative_volume(asset, order['quantity'])
            
            # Higher volatility or larger size relative to volume = more slippage
            variable_factor = slippage_factor * (1 + volatility) * (1 + volume)
            direction = 1 if order['side'] == POSITION_SIDE.LONG else -1
            
            return price * (1 + direction * variable_factor)
            
        return price  # Default fallback
    
    def _calculate_volatility(self, asset):
        """Calculate recent volatility for an asset."""
        # Simplified implementation - in reality would use more sophisticated methods
        try:
            # Get data for the smallest timeframe
            timeframe = min(self.config['timeframes'], key=lambda x: TIME_FRAMES[x])
            df = self.data_provider.get_ohlcv(
                asset, timeframe, 
                limit=20  # Use last 20 bars
            )
            
            if df is not None and not df.empty:
                # Calculate returns
                returns = df['close'].pct_change().dropna()
                # Annualized volatility
                return returns.std() * np.sqrt(252 * 24 / TIME_FRAMES[timeframe])
                
        except Exception as e:
            self.logger.warning(f"Error calculating volatility: {str(e)}")
            
        return 0.01  # Default fallback volatility
    
    def _calculate_relative_volume(self, asset, order_quantity):
        """Calculate the order size relative to recent volume."""
        try:
            # Get data for the smallest timeframe
            timeframe = min(self.config['timeframes'], key=lambda x: TIME_FRAMES[x])
            df = self.data_provider.get_ohlcv(
                asset, timeframe, 
                limit=5  # Use last 5 bars
            )
            
            if df is not None and not df.empty and 'volume' in df.columns:
                avg_volume = df['volume'].mean()
                if avg_volume > 0:
                    return order_quantity / avg_volume
                    
        except Exception as e:
            self.logger.warning(f"Error calculating relative volume: {str(e)}")
            
        return 0.01  # Default fallback
    
    async def _execute_order(self, order, execution_price, timestamp):
        """Execute an order at the given price and timestamp."""
        # Calculate commission
        commission_rate = self.config.get('commission_rate', 0.001)
        commission = order['quantity'] * execution_price * commission_rate
        
        # Create trade record
        trade = {
            'id': str(uuid.uuid4()),
            'order_id': order['id'],
            'asset': order['asset'],
            'side': order['side'],
            'entry_price': execution_price,
            'quantity': order['quantity'],
            'commission': commission,
            'entry_time': timestamp,
            'exit_price': None,
            'exit_time': None,
            'pnl': 0.0,
            'status': 'OPEN',
            'stop_loss': order.get('stop_loss'),
            'take_profit': order.get('take_profit')
        }
        
        # Update capital
        self.current_capital -= commission
        
        # Add or update position
        position_key = f"{order['asset']}_{order['side']}"
        
        if position_key in self.positions:
            # Update existing position
            position = self.positions[position_key]
            total_quantity = position['quantity'] + order['quantity']
            avg_price = (
                (position['entry_price'] * position['quantity']) + 
                (execution_price * order['quantity'])
            ) / total_quantity
            
            position['quantity'] = total_quantity
            position['entry_price'] = avg_price
            position['trades'].append(trade['id'])
            
            # Update stop loss and take profit if provided
            if order.get('stop_loss'):
                position['stop_loss'] = order['stop_loss']
            if order.get('take_profit'):
                position['take_profit'] = order['take_profit']
                
        else:
            # Create new position
            self.positions[position_key] = {
                'id': str(uuid.uuid4()),
                'asset': order['asset'],
                'side': order['side'],
                'quantity': order['quantity'],
                'entry_price': execution_price,
                'entry_time': timestamp,
                'current_price': execution_price,
                'trades': [trade['id']],
                'status': 'OPEN',
                'stop_loss': order.get('stop_loss'),
                'take_profit': order.get('take_profit')
            }
        
        # Add trade to history
        self.trades.append(trade)
        
        self.logger.debug(
            f"Executed {order['side']} order for {order['quantity']} {order['asset']} "
            f"at {execution_price} (timestamp: {timestamp})"
        )
    
    async def _generate_signals(self, timestamp, market_data):
        """
        Generate trading signals from strategy brain or brain council.
        
        Returns a list of signal dictionaries.
        """
        signals = []
        
        try:
            # If brain council is available, use it
            if self.brain_council:
                brain_signals = await self.brain_council.generate_signals(
                    timestamp=timestamp,
                    market_data=market_data
                )
                if brain_signals:
                    signals.extend(brain_signals)
            
            # If strategy brain is available and no council, use it directly
            elif self.strategy_brain:
                strategy_signals = await self.strategy_brain.generate_signals(
                    timestamp=timestamp,
                    market_data=market_data
                )
                if strategy_signals:
                    signals.extend(strategy_signals)
                    
            # Process signals through risk management if available
            if signals and self.risk_manager:
                signals = await self.risk_manager.process_signals(
                    signals=signals,
                    timestamp=timestamp,
                    market_data=market_data,
                    current_capital=self.current_capital,
                    positions=self.positions
                )
        
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
        
        return signals
    
    async def _execute_signals(self, signals, market_data):
        """Execute trading signals by creating appropriate orders."""
        for signal in signals:
            try:
                # Skip invalid signals
                if not self._validate_signal(signal):
                    continue
                    
                asset = signal['asset']
                side = signal['side']
                
                # Get current price for the asset
                current_price = self._get_current_price(asset, market_data)
                
                # Calculate position size
                position_size = self._calculate_position_size(
                    signal, current_price, asset
                )
                
                if position_size <= 0:
                    self.logger.warning(f"Invalid position size calculated for {asset}")
                    continue
                
                # Create order
                order = {
                    'id': str(uuid.uuid4()),
                    'asset': asset,
                    'side': side,
                    'type': signal.get('order_type', ORDER_TYPE.MARKET),
                    'price': signal.get('price', current_price),
                    'quantity': position_size,
                    'timestamp': self.current_time,
                    'status': 'PENDING',
                    'expiry': signal.get('expiry'),
                    'stop_loss': signal.get('stop_loss'),
                    'take_profit': signal.get('take_profit'),
                    'metadata': signal.get('metadata', {})
                }
                
                # Add order to pending orders
                self.orders[order['id']] = order
                
                self.logger.debug(
                    f"Created {order['type']} {side} order for {position_size} {asset} "
                    f"at {order['price']}"
                )
                
            except Exception as e:
                self.logger.error(f"Error executing signal: {str(e)}")
    
    def _validate_signal(self, signal):
        """Validate a trading signal."""
        required_fields = ['asset', 'side', 'confidence']
        
        # Check required fields
        for field in required_fields:
            if field not in signal:
                self.logger.warning(f"Signal missing required field: {field}")
                return False
        
        # Validate asset
        if signal['asset'] not in self.config['assets']:
            self.logger.warning(f"Signal contains invalid asset: {signal['asset']}")
            return False
            
        # Validate side
        if signal['side'] not in [POSITION_SIDE.LONG, POSITION_SIDE.SHORT]:
            self.logger.warning(f"Signal contains invalid side: {signal['side']}")
            return False
            
        # Validate confidence level
        if not (0 <= signal['confidence'] <= 1):
            self.logger.warning(f"Signal contains invalid confidence: {signal['confidence']}")
            return False
            
        # Apply confidence threshold
        confidence_threshold = self.config.get('confidence_threshold', 0.0)
        if signal['confidence'] < confidence_threshold:
            self.logger.debug(
                f"Signal confidence {signal['confidence']} below threshold "
                f"{confidence_threshold}"
            )
            return False
            
        return True
    
    def _calculate_position_size(self, signal, current_price, asset):
        """
        Calculate position size based on risk parameters and capital.
        
        Uses either fixed size, percentage of capital, or position sizer.
        """
        # Get sizing method from config
        sizing_method = self.config.get('position_sizing', 'percent_risk')
        
        if sizing_method == 'fixed':
            # Fixed position size in base currency
            fixed_size = self.config.get('fixed_position_size', 100.0)
            return fixed_size / current_price
            
        elif sizing_method == 'percent_capital':
            # Percentage of total capital
            percent = self.config.get('capital_percent', 0.02)
            position_value = self.current_capital * percent
            return position_value / current_price
            
        elif sizing_method == 'percent_risk':
            # Risk a percentage of capital based on stop loss
            risk_percent = self.config.get('risk_percent', 0.01)
            
            # Calculate stop level
            stop_loss = signal.get('stop_loss')
            if not stop_loss:
                # Default to 2% risk without stop
                stop_distance = current_price * 0.02
            else:
                stop_distance = abs(current_price - stop_loss)
                
            if stop_distance <= 0:
                return 0
                
            # Calculate position size
            risk_amount = self.current_capital * risk_percent
            return risk_amount / stop_distance
            
        elif sizing_method == 'kelly':
            # Kelly criterion
            # Simplified implementation
            win_prob = signal.get('win_probability', 0.5)
            if signal.get('stop_loss') and signal.get('take_profit'):
                risk = abs(current_price - signal['stop_loss'])
                reward = abs(signal['take_profit'] - current_price)
                if risk <= 0:
                    return 0
                    
                win_ratio = reward / risk
                kelly_fraction = (win_prob * win_ratio - (1 - win_prob)) / win_ratio
                
                # Cap the kelly fraction and apply to capital
                kelly_fraction = max(0, min(kelly_fraction, 0.2))  # Cap at 20%
                position_value = self.current_capital * kelly_fraction
                return position_value / current_price
        
        # Default fallback - 1% of capital
        position_value = self.current_capital * 0.01
        return position_value / current_price
    
    async def _update_positions(self, timestamp, market_data):
        """Update all open positions with current prices and check for exits."""
        positions_to_close = []
        
        for position_key, position in self.positions.items():
            if position['status'] != 'OPEN':
                continue
                
            asset = position['asset']
                
            try:
                # Get current price
                current_price = self._get_current_price(asset, market_data)
                
                # Update position's current price
                position['current_price'] = current_price
                
                # Calculate current pnl
                direction = 1 if position['side'] == POSITION_SIDE.LONG else -1
                price_diff = direction * (current_price - position['entry_price'])
                position['unrealized_pnl'] = price_diff * position['quantity']
                
                # Check stop loss
                if position.get('stop_loss'):
                    if (position['side'] == POSITION_SIDE.LONG and 
                            current_price <= position['stop_loss']):
                        positions_to_close.append((position_key, position['stop_loss'], 'stop_loss'))
                        
                    elif (position['side'] == POSITION_SIDE.SHORT and 
                            current_price >= position['stop_loss']):
                        positions_to_close.append((position_key, position['stop_loss'], 'stop_loss'))
                
                # Check take profit
                if position.get('take_profit'):
                    if (position['side'] == POSITION_SIDE.LONG and 
                            current_price >= position['take_profit']):
                        positions_to_close.append((position_key, position['take_profit'], 'take_profit'))
                        
                    elif (position['side'] == POSITION_SIDE.SHORT and 
                            current_price <= position['take_profit']):
                        positions_to_close.append((position_key, position['take_profit'], 'take_profit'))
                
            except Exception as e:
                self.logger.error(f"Error updating position {position_key}: {str(e)}")
        
        # Close positions that hit stop loss or take profit
        for position_key, exit_price, exit_reason in positions_to_close:
            await self._close_position(position_key, exit_price, timestamp, exit_reason)
    
    async def _close_position(self, position_key, exit_price, timestamp, reason='manual'):
        """Close a position at the specified price and timestamp."""
        if position_key not in self.positions:
            return
            
        position = self.positions[position_key]
        
        # Apply slippage to exit price
        if reason == 'stop_loss':
            # More slippage on stop losses
            slippage_factor = self.config.get('stop_loss_slippage', 0.001)
        else:
            slippage_factor = self.config.get('slippage_factor', 0.0005)
            
        direction = 1 if position['side'] == POSITION_SIDE.LONG else -1
        adjusted_exit_price = exit_price * (1 - direction * slippage_factor)
        
        # Calculate pnl
        direction = 1 if position['side'] == POSITION_SIDE.LONG else -1
        price_diff = direction * (adjusted_exit_price - position['entry_price'])
        pnl = price_diff * position['quantity']
        
        # Calculate commission
        commission_rate = self.config.get('commission_rate', 0.001)
        commission = position['quantity'] * adjusted_exit_price * commission_rate
        
        # Update capital
        self.current_capital += pnl - commission
        
        # Update position status
        position['status'] = 'CLOSED'
        position['exit_price'] = adjusted_exit_price
        position['exit_time'] = timestamp
        position['realized_pnl'] = pnl
        position['exit_reason'] = reason
        
        # Update all trades associated with this position
        for trade_id in position['trades']:
            for trade in self.trades:
                if trade['id'] == trade_id:
                    trade['exit_price'] = adjusted_exit_price
                    trade['exit_time'] = timestamp
                    trade['pnl'] = pnl * (trade['quantity'] / position['quantity'])
                    trade['status'] = 'CLOSED'
                    break
        
        self.logger.debug(
            f"Closed {position['side']} position for {position['quantity']} "
            f"{position['asset']} at {adjusted_exit_price} with PnL {pnl} "
            f"(reason: {reason})"
        )
    
    def _record_equity_point(self, timestamp):
        """Record the current equity (capital + unrealized pnl) at this timestamp."""
        # Calculate unrealized pnl from open positions
        unrealized_pnl = sum(
            position.get('unrealized_pnl', 0) 
            for position in self.positions.values() 
            if position['status'] == 'OPEN'
        )
        
        equity = self.current_capital + unrealized_pnl
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'capital': self.current_capital,
            'unrealized_pnl': unrealized_pnl
        })
    
    async def _finalize_backtest(self):
        """Finalize the backtest by closing all open positions."""
        # Close all open positions at the last price
        for position_key, position in list(self.positions.items()):
            if position['status'] == 'OPEN':
                await self._close_position(
                    position_key, 
                    position['current_price'], 
                    self.current_time,
                    'backtest_end'
                )
        
        # Create the equity curve dataframe
        if self.equity_curve:
            self.equity_df = pd.DataFrame(self.equity_curve)
            self.equity_df.set_index('timestamp', inplace=True)
        else:
            # Create empty dataframe if no equity points
            self.equity_df = pd.DataFrame(
                columns=['timestamp', 'equity', 'capital', 'unrealized_pnl']
            )
            
        # Create the trades dataframe
        if self.trades:
            self.trades_df = pd.DataFrame(self.trades)
        else:
            self.trades_df = pd.DataFrame(
                columns=['id', 'asset', 'side', 'entry_price', 'exit_price', 
                         'quantity', 'entry_time', 'exit_time', 'pnl', 'status']
            )
    
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics for the backtest."""
        if not hasattr(self, 'trades_df') or self.trades_df.empty:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'final_capital': self.current_capital,
                'total_return': (self.current_capital / self.initial_capital) - 1,
            }
            
        # Basic metrics
        closed_trades = self.trades_df[self.trades_df['status'] == 'CLOSED']
        total_trades = len(closed_trades)
        winning_trades = len(closed_trades[closed_trades['pnl'] > 0])
        losing_trades = len(closed_trades[closed_trades['pnl'] <= 0])
        
        # Avoid division by zero
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit metrics
        gross_profit = closed_trades[closed_trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(closed_trades[closed_trades['pnl'] <= 0]['pnl'].sum())
        net_profit = gross_profit - gross_loss
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate returns and drawdown from equity curve
        if hasattr(self, 'equity_df') and not self.equity_df.empty:
            equity_series = self.equity_df['equity']
            
            # Daily returns
            returns = equity_series.pct_change().dropna()
            
            # Sharpe ratio (annualized, assuming daily data)
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            # Calculate drawdown
            rolling_max = equity_series.cummax()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())
            
            # Maximum consecutive losses
            profit_series = closed_trades['pnl'].values
            max_consecutive_losses = self._calculate_max_consecutive(profit_series, lambda x: x <= 0)
            
            # Calculate time in market
            time_in_market = self._calculate_time_in_market()
            
        else:
            # Fallbacks if no equity curve
            sharpe_ratio = 0
            max_drawdown = 0
            max_consecutive_losses = 0
            time_in_market = 0
            
        # Return metrics dictionary
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'net_profit': net_profit,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_consecutive_losses': max_consecutive_losses,
            'time_in_market': time_in_market,
            'final_capital': self.current_capital,
            'total_return': (self.current_capital / self.initial_capital) - 1,
            'annualized_return': self._calculate_annualized_return(),
            'avg_trade_duration': self._calculate_avg_trade_duration(),
        }
        
        # Add advanced metrics if available
        if hasattr(self, 'trades_df') and not self.trades_df.empty:
            metrics.update({
                'avg_win': closed_trades[closed_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0,
                'avg_loss': closed_trades[closed_trades['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0,
                'largest_win': closed_trades['pnl'].max(),
                'largest_loss': closed_trades['pnl'].min(),
                'win_loss_ratio': (closed_trades[closed_trades['pnl'] > 0]['pnl'].mean() / 
                                  abs(closed_trades[closed_trades['pnl'] <= 0]['pnl'].mean())) 
                                  if losing_trades > 0 and winning_trades > 0 else 0,
                'expectancy': (win_rate * (gross_profit / winning_trades) - 
                              (1 - win_rate) * (gross_loss / losing_trades)) 
                              if winning_trades > 0 and losing_trades > 0 else 0,
            })
            
            # Calculate metrics by asset
            asset_metrics = {}
            for asset in closed_trades['asset'].unique():
                asset_trades = closed_trades[closed_trades['asset'] == asset]
                asset_total = len(asset_trades)
                asset_wins = len(asset_trades[asset_trades['pnl'] > 0])
                
                asset_metrics[asset] = {
                    'total_trades': asset_total,
                    'win_rate': asset_wins / asset_total if asset_total > 0 else 0,
                    'net_pnl': asset_trades['pnl'].sum(),
                    'avg_pnl': asset_trades['pnl'].mean() if asset_total > 0 else 0,
                }
                
            metrics['asset_performance'] = asset_metrics
            
        return metrics
    
    def _calculate_max_consecutive(self, series, condition_func):
        """Calculate maximum consecutive occurrences meeting a condition."""
        max_consecutive = 0
        current_consecutive = 0
        
        for value in series:
            if condition_func(value):
                current_consecutive += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 0
                
        return max(max_consecutive, current_consecutive)
    
    def _calculate_time_in_market(self):
        """Calculate the percentage of time with open positions."""
        if not hasattr(self, 'equity_df') or self.equity_df.empty:
            return 0
            
        closed_trades = self.trades_df[self.trades_df['status'] == 'CLOSED']
        if closed_trades.empty:
            return 0
            
        # Calculate total duration in market
        total_time = 0
        
        for _, trade in closed_trades.iterrows():
            if pd.notna(trade['entry_time']) and pd.notna(trade['exit_time']):
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds()
                total_time += duration
                
        # Calculate total backtest duration
        if self.config['start_date'] and self.config['end_date']:
            backtest_duration = (self.config['end_date'] - self.config['start_date']).total_seconds()
            if backtest_duration > 0:
                return total_time / backtest_duration
                
        return 0
    
    def _calculate_annualized_return(self):
        """Calculate the annualized return for the backtest."""
        if not hasattr(self, 'equity_df') or self.equity_df.empty:
            return 0
            
        total_return = (self.current_capital / self.initial_capital) - 1
        
        # Calculate duration in years
        if self.config['start_date'] and self.config['end_date']:
            days = (self.config['end_date'] - self.config['start_date']).days
            years = days / 365.0
            
            if years > 0:
                return (1 + total_return) ** (1 / years) - 1
                
        return 0
    
    def _calculate_avg_trade_duration(self):
        """Calculate the average duration of trades in hours."""
        closed_trades = self.trades_df[self.trades_df['status'] == 'CLOSED']
        if closed_trades.empty:
            return 0
            
        durations = []
        
        for _, trade in closed_trades.iterrows():
            if pd.notna(trade['entry_time']) and pd.notna(trade['exit_time']):
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600  # Hours
                durations.append(duration)
                
        if durations:
            return sum(durations) / len(durations)
            
        return 0
    
    async def _save_backtest_results(self):
        """Save backtest results to database."""
        try:
            # Save only if DB client is available
            if self.db_client:
                # Prepare data
                results = {
                    'backtest_id': self.config['backtest_id'],
                    'config': self.config,
                    'performance_metrics': self.performance_metrics,
                    'equity_curve': self.equity_df.reset_index().to_dict('records') if hasattr(self, 'equity_df') else [],
                    'trades': self.trades_df.to_dict('records') if hasattr(self, 'trades_df') else [],
                    'final_capital': self.current_capital,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                # Save to database
                await self.db_client.save_backtest_results(results)
                
                self.logger.info(f"Saved backtest results with ID: {self.config['backtest_id']}")
        
        except Exception as e:
            self.logger.error(f"Failed to save backtest results: {str(e)}")
    
    def get_equity_curve(self):
        """Get the equity curve dataframe."""
        if hasattr(self, 'equity_df'):
            return self.equity_df
        return None
    
    def get_trades(self):
        """Get the trades dataframe."""
        if hasattr(self, 'trades_df'):
            return self.trades_df
        return None
    
    def get_performance_metrics(self):
        """Get the calculated performance metrics."""
        return self.performance_metrics
    
    async def plot_results(self, show=True, save_path=None):
        """Generate plots of backtest results."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            if not hasattr(self, 'equity_df') or self.equity_df.empty:
                self.logger.warning("No equity data available for plotting")
                return None
                
            # Create figure with subplots
            fig, axs = plt.subplots(3, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Plot equity curve
            axs[0].plot(self.equity_df.index, self.equity_df['equity'], label='Equity')
            axs[0].plot(self.equity_df.index, self.equity_df['capital'], label='Capital', linestyle='--')
            axs[0].set_title('Equity Curve')
            axs[0].set_ylabel('Equity')
            axs[0].legend()
            axs[0].grid(True)
            
            # Format x-axis dates
            axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()
            
            # Plot drawdown
            equity_series = self.equity_df['equity']
            rolling_max = equity_series.cummax()
            drawdown = (equity_series - rolling_max) / rolling_max * 100
            
            axs[1].fill_between(self.equity_df.index, drawdown, 0, color='red', alpha=0.3)
            axs[1].set_title('Drawdown')
            axs[1].set_ylabel('Drawdown %')
            axs[1].grid(True)
            
            # Plot trades
            if hasattr(self, 'trades_df') and not self.trades_df.empty:
                # Filter completed trades
                completed_trades = self.trades_df[
                    (self.trades_df['status'] == 'CLOSED') & 
                    pd.notna(self.trades_df['exit_time'])
                ]
                
                if not completed_trades.empty:
                    # Plot trade entry/exit points
                    for i, trade in completed_trades.iterrows():
                        color = 'green' if trade['pnl'] > 0 else 'red'
                        marker = '^' if trade['side'] == POSITION_SIDE.LONG else 'v'
                        
                        # Find equity at entry and exit
                        entry_equity = self.equity_df.loc[self.equity_df.index >= trade['entry_time']].iloc[0]['equity']
                        exit_equity = self.equity_df.loc[self.equity_df.index >= trade['exit_time']].iloc[0]['equity']
                        
                        # Plot entry point
                        axs[0].scatter(trade['entry_time'], entry_equity, color=color, marker=marker, s=50)
                        
                        # Plot exit point
                        axs[0].scatter(trade['exit_time'], exit_equity, color=color, marker='x', s=50)
            
            # Plot trade PnL
            if hasattr(self, 'trades_df') and not self.trades_df.empty:
                completed_trades = self.trades_df[self.trades_df['status'] == 'CLOSED']
                
                if not completed_trades.empty:
                    # Sort by exit time
                    completed_trades = completed_trades.sort_values('exit_time')
                    
                    # Create bars for trade PnL
                    colors = ['green' if pnl > 0 else 'red' for pnl in completed_trades['pnl']]
                    axs[2].bar(range(len(completed_trades)), completed_trades['pnl'], color=colors)
                    axs[2].set_title('Trade PnL')
                    axs[2].set_xlabel('Trade Number')
                    axs[2].set_ylabel('PnL')
                    axs[2].grid(True)
            
            # Add performance metrics as text
            metrics_text = '\n'.join([
                f"Total Return: {self.performance_metrics.get('total_return', 0):.2%}",
                f"Annualized Return: {self.performance_metrics.get('annualized_return', 0):.2%}",
                f"Sharpe Ratio: {self.performance_metrics.get('sharpe_ratio', 0):.2f}",
                f"Max Drawdown: {self.performance_metrics.get('max_drawdown', 0):.2%}",
                f"Win Rate: {self.performance_metrics.get('win_rate', 0):.2%}",
                f"Profit Factor: {self.performance_metrics.get('profit_factor', 0):.2f}",
                f"Total Trades: {self.performance_metrics.get('total_trades', 0)}"
            ])
            
            plt.figtext(0.01, 0.01, metrics_text, fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Saved backtest plot to {save_path}")
                
            if show:
                plt.show()
                
            return fig
                
        except Exception as e:
            self.logger.error(f"Error plotting backtest results: {str(e)}")
            return None
    
    async def monte_carlo_analysis(self, num_simulations=1000):
        """
        Perform Monte Carlo analysis on backtest trades to estimate performance bounds.
        
        Returns a dictionary with simulation results.
        """
        if not hasattr(self, 'trades_df') or self.trades_df.empty:
            return None
            
        try:
            import numpy as np
            
            closed_trades = self.trades_df[self.trades_df['status'] == 'CLOSED']
            if closed_trades.empty:
                return None
                
            # Extract PnL values
            pnl_values = closed_trades['pnl'].values
            
            # Perform Monte Carlo simulations
            simulation_results = []
            
            for _ in range(num_simulations):
                # Shuffle the order of trades
                np.random.shuffle(pnl_values)
                
                # Calculate equity curve
                equity = np.zeros(len(pnl_values) + 1)
                equity[0] = self.initial_capital
                
                for i, pnl in enumerate(pnl_values):
                    equity[i+1] = equity[i] + pnl
                
                # Calculate metrics
                total_return = equity[-1] / equity[0] - 1
                
                # Calculate drawdown
                peak = np.maximum.accumulate(equity)
                drawdown = (equity - peak) / peak
                max_drawdown = abs(drawdown.min())
                
                simulation_results.append({
                    'final_equity': equity[-1],
                    'total_return': total_return,
                    'max_drawdown': max_drawdown
                })
            
            # Convert to DataFrame
            results_df = pd.DataFrame(simulation_results)
            
            # Calculate percentiles
            percentiles = [5, 25, 50, 75, 95]
            return_percentiles = np.percentile(results_df['total_return'], percentiles)
            drawdown_percentiles = np.percentile(results_df['max_drawdown'], percentiles)
            
            # Return summarized results
            monte_carlo_results = {
                'num_simulations': num_simulations,
                'return_percentiles': dict(zip(percentiles, return_percentiles)),
                'drawdown_percentiles': dict(zip(percentiles, drawdown_percentiles)),
                'median_return': return_percentiles[2],  # 50th percentile
                'median_drawdown': drawdown_percentiles[2],  # 50th percentile
                'worst_case_return': return_percentiles[0],  # 5th percentile
                'worst_case_drawdown': drawdown_percentiles[4]  # 95th percentile
            }
            
            return monte_carlo_results
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo analysis: {str(e)}")
            return None
    
    async def perform_walk_forward_optimization(self, params_grid, num_folds=5):
        """
        Perform walk-forward optimization of strategy parameters.
        
        Args:
            params_grid: Dictionary of parameter names and lists of values to test
            num_folds: Number of time periods to test
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Generate all parameter combinations
            import itertools
            
            param_names = list(params_grid.keys())
            param_values = list(params_grid.values())
            param_combinations = list(itertools.product(*param_values))
            
            # Split backtest period into folds
            start_date = self.config['start_date']
            end_date = self.config['end_date']
            total_days = (end_date - start_date).days
            fold_size = total_days // num_folds
            
            fold_periods = []
            for i in range(num_folds):
                fold_start = start_date + pd.Timedelta(days=i*fold_size)
                fold_end = start_date + pd.Timedelta(days=(i+1)*fold_size)
                if i == num_folds - 1:
                    fold_end = end_date  # Ensure we include the end date
                fold_periods.append((fold_start, fold_end))
            
            # Track results for each combination
            optimization_results = []
            
            # Test each parameter combination
            for param_combo in param_combinations:
                combo_params = dict(zip(param_names, param_combo))
                
                # Run backtest for each fold
                fold_results = []
                
                for i, (fold_start, fold_end) in enumerate(fold_periods):
                    # Create config for this fold and parameter set
                    fold_config = self.config.copy()
                    fold_config.update(combo_params)
                    fold_config['start_date'] = fold_start
                    fold_config['end_date'] = fold_end
                    fold_config['backtest_id'] = f"{self.config['backtest_id']}_fold{i}_opt"
                    
                    # Create and run backtest engine
                    fold_engine = BacktestEngine(
                        config=fold_config,
                        strategy_brain=self.strategy_brain,
                        brain_council=self.brain_council,
                        db_client=self.db_client,
                        redis_client=self.redis_client
                    )
                    
                    # Run backtest
                    fold_metrics = await fold_engine.run()
                    
                    fold_results.append({
                        'fold': i,
                        'period': (fold_start, fold_end),
                        'metrics': fold_metrics
                    })
                
                # Calculate average performance across folds
                avg_return = np.mean([f['metrics'].get('total_return', 0) for f in fold_results])
                avg_sharpe = np.mean([f['metrics'].get('sharpe_ratio', 0) for f in fold_results])
                avg_drawdown = np.mean([f['metrics'].get('max_drawdown', 0) for f in fold_results])
                avg_win_rate = np.mean([f['metrics'].get('win_rate', 0) for f in fold_results])
                
                # Calculate consistency of results
                return_std = np.std([f['metrics'].get('total_return', 0) for f in fold_results])
                
                optimization_results.append({
                    'parameters': combo_params,
                    'avg_return': avg_return,
                    'avg_sharpe': avg_sharpe,
                    'avg_drawdown': avg_drawdown,
                    'avg_win_rate': avg_win_rate,
                    'return_std': return_std,
                    'fold_results': fold_results
                })
            
            # Sort by desired metric (e.g., risk-adjusted return)
            # Here we use Sharpe ratio
            sorted_results = sorted(
                optimization_results,
                key=lambda x: x['avg_sharpe'],
                reverse=True
            )
            
            return {
                'best_parameters': sorted_results[0]['parameters'] if sorted_results else None,
                'all_results': sorted_results
            }
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward optimization: {str(e)}")
            return None
