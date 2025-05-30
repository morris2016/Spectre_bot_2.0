

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Risk Manager - Recovery System

This module implements intelligent recovery strategies to handle drawdowns and
losses. It provides sophisticated mechanisms to recover from losing periods while
maintaining risk management discipline.
"""

from typing import Dict, List, Optional, Any, Type
from datetime import datetime
from common.db_client import DatabaseClient, get_db_client
from common.redis_client import RedisClient
from common.constants import RECOVERY_STRATEGIES, ACCOUNT_STATES
from common.logger import get_logger
from common.exceptions import RecoveryStrategyError

from risk_manager.position_sizing import PositionSizer
from risk_manager.exposure import ExposureManager


class BaseRecoveryManager:
    """Base class for recovery management."""

    registry: Dict[str, Type["BaseRecoveryManager"]] = {}

    def __init_subclass__(cls, name: Optional[str] = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        key = name or cls.__name__
        BaseRecoveryManager.registry[key] = cls

    async def enter_recovery_mode(self, *args, **kwargs):
        """Enter recovery mode."""
        raise NotImplementedError

    async def exit_recovery_mode(self, *args, **kwargs):
        """Exit recovery mode."""
        raise NotImplementedError


class RecoveryManager(BaseRecoveryManager):
    """
    The RecoveryManager implements strategies to recover from drawdowns and losing
    streaks while maintaining proper risk management discipline.
    """
    
    def __init__(self, 
                 db_client: DatabaseClient = None, 
                 redis_client: RedisClient = None,
                 position_sizer: PositionSizer = None,
                 exposure_manager: ExposureManager = None,
                 config: Dict[str, Any] = None):
        """
        Initialize the Recovery Manager.
        
        Args:
            db_client: Database client for historical data
            redis_client: Redis client for real-time data
            position_sizer: Position sizer instance for adjusting position sizes
            exposure_manager: Exposure manager for controlling overall exposure
            config: Configuration parameters for recovery strategies
        """
        self.logger = get_logger(self.__class__.__name__)
        self.db_client = db_client
        self._db_params = {}
        self.redis_client = redis_client or RedisClient()
        self.position_sizer = position_sizer or PositionSizer()
        self.exposure_manager = exposure_manager or ExposureManager()
        
        # Default configuration
        self._default_config = {
            'drawdown_threshold': 0.05,  # 5% drawdown triggers recovery mode
            'consecutive_losses_threshold': 3,  # 3 consecutive losses triggers recovery mode
            'recovery_factor': 0.5,  # Reduce position sizes by 50% during recovery
            'reset_threshold': 0.02,  # 2% profit from bottom to exit recovery mode
            'max_recovery_time': 7,  # Maximum days in recovery mode
            'strategy_rotation_enabled': True,  # Enable strategy rotation during recovery
            'correlation_threshold': 0.7,  # Correlation threshold for diversification
            'psychological_breaks': True,  # Enable psychological breaks after losses
            'recovery_strategies': RECOVERY_STRATEGIES['CONSERVATIVE'],  # Default strategies
            'adapt_to_volatility': True,  # Adapt recovery to market volatility
            'reset_max_drawdown': False  # Whether to reset max drawdown after recovery
        }
        
        # Apply custom configuration
        self.config = self._default_config.copy()
        if config:
            self.config.update(config)
            
        # Internal state
        self._in_recovery_mode = False
        self._recovery_start_time = None
        self._recovery_lowest_equity = float('inf')
        self._pre_recovery_equity = None
        self._consecutive_losses = 0
        self._current_recovery_strategy = None
        self._recovery_progress = 0.0
        self._active_recovery_steps = []
        self._past_recoveries = []

        self.logger.info("Recovery Manager initialized with config: %s", self.config)

    async def initialize(self, db_connector: Optional[DatabaseClient] = None) -> None:
        """Obtain a database client and create tables."""
        if db_connector is not None:
            self.db_client = db_connector
        if self.db_client is None:
            self.db_client = await get_db_client(**self._db_params)
        if getattr(self.db_client, "pool", None) is None:
            await self.db_client.initialize()
            await self.db_client.create_tables()
    
    async def analyze_account_state(self, account_data: Dict[str, Any]) -> str:
        """
        Analyze the current account state to determine if recovery mode should be activated.
        
        Args:
            account_data: Dictionary containing account data including equity, balance, trades
        
        Returns:
            str: The determined account state
        """
        current_equity = account_data.get('equity', 0)
        peak_equity = account_data.get('peak_equity', current_equity)
        trades = account_data.get('recent_trades', [])
        
        # Calculate drawdown
        drawdown = 0 if peak_equity == 0 else (peak_equity - current_equity) / peak_equity
        
        # Count consecutive losses
        self._consecutive_losses = 0
        for trade in reversed(trades):
            if trade.get('profit', 0) < 0:
                self._consecutive_losses += 1
            else:
                break
        
        # Determine account state
        if drawdown >= self.config['drawdown_threshold']:
            self.logger.warning(f"Account in drawdown: {drawdown:.2%}")
            return ACCOUNT_STATES['DRAWDOWN']
        
        elif self._consecutive_losses >= self.config['consecutive_losses_threshold']:
            self.logger.warning(f"Account has {self._consecutive_losses} consecutive losses")
            return ACCOUNT_STATES['CONSECUTIVE_LOSSES']
        
        elif self._in_recovery_mode:
            # Check if we should exit recovery mode
            if self._recovery_lowest_equity != float('inf'):
                profit_from_bottom = (current_equity - self._recovery_lowest_equity) / self._recovery_lowest_equity
                if profit_from_bottom >= self.config['reset_threshold']:
                    self.logger.info(f"Recovery condition met: {profit_from_bottom:.2%} profit from lowest point")
                    return ACCOUNT_STATES['RECOVERY_EXIT']
            
            # Check recovery time limit
            if self._recovery_start_time:
                recovery_duration = datetime.now() - self._recovery_start_time
                if recovery_duration.days >= self.config['max_recovery_time']:
                    self.logger.info(f"Recovery time limit reached: {recovery_duration.days} days")
                    return ACCOUNT_STATES['RECOVERY_TIMEOUT']
            
            return ACCOUNT_STATES['RECOVERY']
        
        return ACCOUNT_STATES['NORMAL']
    
    async def enter_recovery_mode(self, account_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enter recovery mode and apply appropriate recovery strategies.
        
        Args:
            account_data: Dictionary containing account data
            
        Returns:
            Dict[str, Any]: Recovery plan details
        """
        if self._in_recovery_mode:
            self.logger.info("Already in recovery mode")
            return {"status": "already_active", "strategy": self._current_recovery_strategy}
        
        self._in_recovery_mode = True
        self._recovery_start_time = datetime.now()
        self._recovery_lowest_equity = account_data.get('equity', float('inf'))
        self._pre_recovery_equity = account_data.get('equity', 0)
        
        # Select recovery strategy based on account state and market conditions
        account_state = await self.analyze_account_state(account_data)
        market_volatility = await self._get_market_volatility()
        self._current_recovery_strategy = await self._select_recovery_strategy(account_state, market_volatility)
        
        # Apply recovery strategy
        recovery_plan = await self._apply_recovery_strategy(self._current_recovery_strategy, account_data)
        
        self.logger.info(f"Entered recovery mode with strategy: {self._current_recovery_strategy}")
        self.logger.info(f"Recovery plan: {recovery_plan}")
        
        return {
            "status": "activated",
            "strategy": self._current_recovery_strategy,
            "plan": recovery_plan,
            "start_time": self._recovery_start_time,
            "start_equity": self._pre_recovery_equity
        }
    
    async def exit_recovery_mode(self) -> Dict[str, Any]:
        """
        Exit recovery mode and restore normal trading parameters.
        
        Returns:
            Dict[str, Any]: Recovery results
        """
        if not self._in_recovery_mode:
            return {"status": "not_active"}
        
        recovery_duration = datetime.now() - self._recovery_start_time
        recovery_equity_change = 0
        
        # Get current equity
        current_equity = await self._get_current_equity()
        if self._pre_recovery_equity and current_equity:
            recovery_equity_change = (current_equity - self._pre_recovery_equity) / self._pre_recovery_equity
        
        # Store recovery history
        recovery_record = {
            "start_time": self._recovery_start_time,
            "end_time": datetime.now(),
            "duration": recovery_duration,
            "strategy": self._current_recovery_strategy,
            "start_equity": self._pre_recovery_equity,
            "lowest_equity": self._recovery_lowest_equity,
            "end_equity": current_equity,
            "equity_change": recovery_equity_change,
            "steps_taken": self._active_recovery_steps
        }
        
        self._past_recoveries.append(recovery_record)
        
        # Reset internal state
        self._in_recovery_mode = False
        self._recovery_start_time = None
        self._recovery_lowest_equity = float('inf')
        self._current_recovery_strategy = None
        self._active_recovery_steps = []
        
        # Restore normal trading parameters
        await self._restore_normal_parameters()
        
        self.logger.info(f"Exited recovery mode. Duration: {recovery_duration}, Equity change: {recovery_equity_change:.2%}")
        
        return {
            "status": "deactivated",
            "duration": str(recovery_duration),
            "equity_change": recovery_equity_change,
            "strategy_used": recovery_record["strategy"]
        }
    
    async def update_recovery_state(self, account_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update recovery state with new account data and adjust strategy if needed.
        
        Args:
            account_data: Dictionary containing account data
            
        Returns:
            Dict[str, Any]: Updated recovery state
        """
        if not self._in_recovery_mode:
            return {"status": "not_active"}
        
        current_equity = account_data.get('equity', 0)
        
        # Update lowest equity point if applicable
        if current_equity < self._recovery_lowest_equity:
            self._recovery_lowest_equity = current_equity
        
        # Calculate recovery progress
        if self._recovery_lowest_equity != float('inf') and self._recovery_lowest_equity != 0:
            self._recovery_progress = (current_equity - self._recovery_lowest_equity) / self._recovery_lowest_equity
        
        # Check if recovery strategy needs adjustment
        account_state = await self.analyze_account_state(account_data)
        if account_state in [ACCOUNT_STATES['RECOVERY_EXIT'], ACCOUNT_STATES['RECOVERY_TIMEOUT']]:
            return await self.exit_recovery_mode()
        
        # Adjust recovery strategy if needed based on progress
        if self._recovery_progress < -0.05:  # If we've lost another 5% during recovery
            market_volatility = await self._get_market_volatility()
            new_strategy = await self._select_recovery_strategy(ACCOUNT_STATES['DRAWDOWN'], market_volatility, True)
            
            if new_strategy != self._current_recovery_strategy:
                self.logger.warning(f"Adjusting recovery strategy from {self._current_recovery_strategy} to {new_strategy} due to continued losses")
                self._current_recovery_strategy = new_strategy
                await self._apply_recovery_strategy(new_strategy, account_data)
        
        return {
            "status": "active",
            "strategy": self._current_recovery_strategy,
            "progress": self._recovery_progress,
            "days_active": (datetime.now() - self._recovery_start_time).days,
            "lowest_equity": self._recovery_lowest_equity,
            "current_equity": current_equity
        }
    
    async def adjust_trade_parameters(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust trade parameters according to recovery settings.
        
        Args:
            trade_params: Original trade parameters
            
        Returns:
            Dict[str, Any]: Adjusted trade parameters
        """
        if not self._in_recovery_mode:
            return trade_params
        
        adjusted_params = trade_params.copy()
        
        # Apply position size reduction
        if 'position_size' in adjusted_params:
            recovery_factor = self.config['recovery_factor']
            adjusted_params['position_size'] *= recovery_factor
            self.logger.info(f"Reduced position size by recovery factor: {recovery_factor}")
        
        # Apply tighter stop loss if applicable
        if 'stop_loss' in adjusted_params and 'entry_price' in adjusted_params:
            entry = adjusted_params['entry_price']
            original_stop = adjusted_params['stop_loss']
            direction = 1 if adjusted_params.get('direction', 'buy').lower() == 'buy' else -1
            risk_distance = abs(entry - original_stop)
            
            # Tighten stop loss by 30% during recovery
            tighter_distance = risk_distance * 0.7
            new_stop = entry - (tighter_distance * direction) if direction > 0 else entry + (tighter_distance * direction)
            
            adjusted_params['stop_loss'] = new_stop
            self.logger.info(f"Tightened stop loss from {original_stop} to {new_stop}")
        
        return adjusted_params
    
    def is_in_recovery_mode(self) -> bool:
        """
        Check if the system is currently in recovery mode.
        
        Returns:
            bool: True if in recovery mode, False otherwise
        """
        return self._in_recovery_mode
    
    def get_recovery_history(self) -> List[Dict[str, Any]]:
        """
        Get history of past recovery periods.
        
        Returns:
            List[Dict[str, Any]]: List of past recovery records
        """
        return self._past_recoveries
    
    def get_current_recovery_info(self) -> Dict[str, Any]:
        """
        Get information about the current recovery state.
        
        Returns:
            Dict[str, Any]: Current recovery information
        """
        if not self._in_recovery_mode:
            return {"status": "not_active"}
        
        return {
            "status": "active",
            "strategy": self._current_recovery_strategy,
            "start_time": self._recovery_start_time,
            "days_active": (datetime.now() - self._recovery_start_time).days if self._recovery_start_time else 0,
            "lowest_equity": self._recovery_lowest_equity,
            "progress": self._recovery_progress,
            "active_steps": self._active_recovery_steps
        }
    
    async def _apply_recovery_strategy(self, strategy: str, account_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the selected recovery strategy.
        
        Args:
            strategy: The recovery strategy to apply
            account_data: Current account data
            
        Returns:
            Dict[str, Any]: Details of the applied recovery plan
        """
        self._active_recovery_steps = []
        recovery_plan = {"strategy": strategy, "actions": []}
        
        try:
            # Apply strategy-specific actions
            if strategy == RECOVERY_STRATEGIES['CONSERVATIVE']:
                # Conservative approach: reduce position sizes, focus on high probability setups
                await self.position_sizer.set_risk_factor(self.config['recovery_factor'])
                await self.exposure_manager.set_max_exposure(0.3)  # Limit max exposure to 30%
                
                recovery_plan["actions"].append({"type": "position_size", "factor": self.config['recovery_factor']})
                recovery_plan["actions"].append({"type": "max_exposure", "value": 0.3})
                
                self._active_recovery_steps.append("reduced_position_size")
                self._active_recovery_steps.append("limited_exposure")
            
            elif strategy == RECOVERY_STRATEGIES['AGGRESSIVE']:
                # Aggressive approach: Look for high quality setups to recover quickly
                await self.position_sizer.set_risk_factor(self.config['recovery_factor'] * 1.2)
                await self.exposure_manager.set_max_exposure(0.5)  # Allow more exposure for recovery
                
                recovery_plan["actions"].append({"type": "position_size", "factor": self.config['recovery_factor'] * 1.2})
                recovery_plan["actions"].append({"type": "max_exposure", "value": 0.5})
                
                self._active_recovery_steps.append("focused_recovery_sizing")
                self._active_recovery_steps.append("high_quality_filter")
            
            elif strategy == RECOVERY_STRATEGIES['DEFENSIVE']:
                # Defensive approach: minimize further losses, very small positions
                await self.position_sizer.set_risk_factor(self.config['recovery_factor'] * 0.5)
                await self.exposure_manager.set_max_exposure(0.2)  # Very limited exposure
                
                recovery_plan["actions"].append({"type": "position_size", "factor": self.config['recovery_factor'] * 0.5})
                recovery_plan["actions"].append({"type": "max_exposure", "value": 0.2})
                
                self._active_recovery_steps.append("minimal_position_size")
                self._active_recovery_steps.append("strict_exposure_limits")
            
            # Common recovery actions
            if self.config['strategy_rotation_enabled']:
                # Add strategy rotation signal
                recovery_plan["actions"].append({"type": "strategy_rotation", "enabled": True})
                self._active_recovery_steps.append("strategy_rotation")
            
            if self.config['psychological_breaks']:
                # Signal to take a psychological break after losses
                recovery_plan["actions"].append({"type": "psychological_break", "duration": "4h"})
                self._active_recovery_steps.append("psychological_break")
                
        except Exception as e:
            self.logger.error(f"Error applying recovery strategy: {e}")
            raise RecoveryStrategyError(f"Failed to apply recovery strategy {strategy}: {str(e)}")
            
        return recovery_plan
    
    async def _restore_normal_parameters(self) -> None:
        """
        Restore normal trading parameters after recovery mode.
        """
        # Reset position sizer to normal risk levels
        await self.position_sizer.reset_risk_factor()
        
        # Reset exposure limits to normal levels
        await self.exposure_manager.reset_max_exposure()
        
        # Signal to disable any strategy rotation enforced during recovery
        if "strategy_rotation" in self._active_recovery_steps:
            self.logger.info("Disabling forced strategy rotation")
        
        # Signal that psychological breaks are no longer mandatory
        if "psychological_break" in self._active_recovery_steps:
            self.logger.info("Disabling mandatory psychological breaks")
            
        self.logger.info("Restored normal trading parameters after recovery")
    
    async def _select_recovery_strategy(self, 
                                        account_state: str, 
                                        market_volatility: float,
                                        is_adjustment: bool = False) -> str:
        """
        Select the most appropriate recovery strategy based on account state and market conditions.
        
        Args:
            account_state: Current account state
            market_volatility: Current market volatility level
            is_adjustment: Whether this is an adjustment to an existing recovery
            
        Returns:
            str: Selected recovery strategy
        """
        # Default to conservative
        strategy = RECOVERY_STRATEGIES['CONSERVATIVE']
        
        # If we're adjusting an existing strategy that isn't working, get more defensive
        if is_adjustment:
            if self._current_recovery_strategy == RECOVERY_STRATEGIES['CONSERVATIVE']:
                return RECOVERY_STRATEGIES['DEFENSIVE']
            elif self._current_recovery_strategy == RECOVERY_STRATEGIES['AGGRESSIVE']:
                return RECOVERY_STRATEGIES['CONSERVATIVE']
            else:
                return RECOVERY_STRATEGIES['DEFENSIVE']
        
        # Otherwise select based on conditions
        if account_state == ACCOUNT_STATES['DRAWDOWN']:
            # Severe drawdown needs more conservative approach
            if market_volatility > 1.5:  # High volatility
                strategy = RECOVERY_STRATEGIES['DEFENSIVE']
            else:
                strategy = RECOVERY_STRATEGIES['CONSERVATIVE']
                
        elif account_state == ACCOUNT_STATES['CONSECUTIVE_LOSSES']:
            # Consecutive losses with normal drawdown might allow aggressive recovery
            if market_volatility < 0.8:  # Low volatility
                strategy = RECOVERY_STRATEGIES['AGGRESSIVE']
            else:
                strategy = RECOVERY_STRATEGIES['CONSERVATIVE']
        
        # Check recent strategy success rates to possibly override
        recent_strategy_stats = await self._get_strategy_success_rates()
        if recent_strategy_stats:
            # If we have a standout successful strategy, adjust our approach
            best_strategy = max(recent_strategy_stats.items(), key=lambda x: x[1]['win_rate'])
            if best_strategy[1]['win_rate'] > 0.7:  # If a strategy has 70%+ win rate
                self.logger.info(f"Adjusting recovery to leverage high-performing strategy: {best_strategy[0]}")
                strategy = RECOVERY_STRATEGIES['AGGRESSIVE']  # Take more aggressive approach with good strategies
        
        return strategy
    
    async def _get_market_volatility(self) -> float:
        """
        Calculate current market volatility relative to historical norms.
        
        Returns:
            float: Volatility ratio (1.0 = normal, >1.0 = higher than normal, <1.0 = lower than normal)
        """
        try:
            # Get recent volatility data from database
            query = """
                SELECT 
                    asset_id, 
                    AVG(volatility) as avg_recent_vol 
                FROM market_volatility 
                WHERE timestamp > DATE_SUB(NOW(), INTERVAL 1 DAY)
                GROUP BY asset_id
            """
            recent_vol_data = await self.db_client.execute_query(query)
            
            # Get historical baseline volatility
            query = """
                SELECT 
                    asset_id, 
                    AVG(volatility) as avg_historical_vol 
                FROM market_volatility 
                WHERE timestamp > DATE_SUB(NOW(), INTERVAL 30 DAY)
                AND timestamp < DATE_SUB(NOW(), INTERVAL 1 DAY)
                GROUP BY asset_id
            """
            historical_vol_data = await self.db_client.execute_query(query)
            
            # Calculate average volatility ratio across assets
            if not recent_vol_data or not historical_vol_data:
                return 1.0
                
            # Convert to dictionaries for easier lookup
            recent_vol_dict = {row['asset_id']: row['avg_recent_vol'] for row in recent_vol_data}
            historical_vol_dict = {row['asset_id']: row['avg_historical_vol'] for row in historical_vol_data}
            
            # Calculate volatility ratios for each asset
            vol_ratios = []
            for asset_id in recent_vol_dict:
                if asset_id in historical_vol_dict and historical_vol_dict[asset_id] > 0:
                    vol_ratio = recent_vol_dict[asset_id] / historical_vol_dict[asset_id]
                    vol_ratios.append(vol_ratio)
            
            # Return average volatility ratio
            if vol_ratios:
                avg_ratio = sum(vol_ratios) / len(vol_ratios)
                self.logger.debug(f"Current market volatility ratio: {avg_ratio:.2f}")
                return avg_ratio
            
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating market volatility: {e}")
            return 1.0  # Default to normal volatility on error
    
    async def _get_strategy_success_rates(self) -> Dict[str, Dict[str, float]]:
        """
        Get recent success rates for different strategies.
        
        Returns:
            Dict[str, Dict[str, float]]: Dictionary of strategy stats including win rates
        """
        try:
            # Query recent strategy performance from database
            query = """
                SELECT 
                    strategy_id,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as winning_trades,
                    AVG(profit) as avg_profit,
                    AVG(CASE WHEN profit > 0 THEN profit ELSE 0 END) as avg_win,
                    AVG(CASE WHEN profit < 0 THEN profit ELSE 0 END) as avg_loss
                FROM trades
                WHERE timestamp > DATE_SUB(NOW(), INTERVAL 7 DAY)
                GROUP BY strategy_id
                HAVING total_trades >= 5
            """
            
            strategy_data = await self.db_client.execute_query(query)
            
            # Calculate success metrics for each strategy
            result = {}
            for row in strategy_data:
                strategy_id = row['strategy_id']
                total_trades = row['total_trades']
                winning_trades = row['winning_trades']
                
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                avg_profit = row['avg_profit'] or 0
                avg_win = row['avg_win'] or 0
                avg_loss = row['avg_loss'] or 0
                
                # Calculate profit factor
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                
                result[strategy_id] = {
                    'win_rate': win_rate,
                    'avg_profit': avg_profit,
                    'profit_factor': profit_factor,
                    'total_trades': total_trades
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting strategy success rates: {e}")
            return {}
    
    async def _get_current_equity(self) -> float:
        """
        Get current account equity.
        
        Returns:
            float: Current equity value
        """
        try:
            # Try to get from Redis first for real-time data
            equity = await self.redis_client.get('account:equity')
            if equity is not None:
                return float(equity)
                
            # Fall back to database
            query = "SELECT equity FROM account_snapshots ORDER BY timestamp DESC LIMIT 1"
            result = await self.db_client.execute_query(query)
            
            if result and len(result) > 0:
                return float(result[0]['equity'])
                
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting current equity: {e}")
            return 0.0


def get_recovery_manager(name: str, *args, **kwargs) -> BaseRecoveryManager:
    """Instantiate a registered recovery manager by name."""
    cls = BaseRecoveryManager.registry.get(name)
    if cls is None:
        raise ValueError(f"Unknown recovery manager: {name}")
    return cls(*args, **kwargs)


__all__ = ["BaseRecoveryManager", "get_recovery_manager"]
