

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Exposure Management Module

This module provides sophisticated exposure management for the trading system,
including monitoring and controlling risk across different assets, platforms,
and account sizes.
"""

from typing import Dict, List, Optional, Any, Type


class BaseExposureManager:
    """Base class for exposure management."""

    registry: Dict[str, Type["BaseExposureManager"]] = {}

    def __init_subclass__(cls, name: Optional[str] = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        key = name or cls.__name__
        BaseExposureManager.registry[key] = cls

    async def adjust_exposure(self, *args, **kwargs):
        raise NotImplementedError

import logging
import asyncio
import numpy as np
import pandas as pd
from decimal import Decimal


from common.constants import Exchange, ASSETS

from common.utils import calculate_correlation_matrix, calculate_volatility
from common.async_utils import run_in_threadpool
from data_storage.market_data import MarketDataRepository

logger = logging.getLogger(__name__)


class ExposureManager(BaseExposureManager):
    """
    Advanced exposure management system that monitors and controls trading risk
    across different assets, platforms, and market conditions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ExposureManager with configuration.
        
        Args:
            config: Configuration dictionary for exposure settings
        """
        self.config = config or {}
        
        # Maximum exposure settings
        self.max_total_exposure = self.config.get('max_total_exposure', 0.8)  # 80% of account
        self.max_single_asset_exposure = self.config.get('max_single_asset_exposure', 0.25)  # 25% of account
        self.max_correlated_exposure = self.config.get('max_correlated_exposure', 0.4)  # 40% of account
        self.max_platform_exposure = self.config.get('max_platform_exposure', {
            Exchange.BINANCE: 0.8,  # 80% of account
            Exchange.DERIV: 0.8,  # 80% of account
        })
        
        # Dynamic exposure adjustment
        self.volatility_based_adjustment = self.config.get('volatility_based_adjustment', True)
        self.performance_based_adjustment = self.config.get('performance_based_adjustment', True)
        
        # Correlation settings
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.correlation_lookback = self.config.get('correlation_lookback', 30)  # 30 days
        
        # Tracking metrics
        self.current_exposure = {
            'total': Decimal('0'),
            'assets': {},
            'platforms': {
                Exchange.BINANCE: Decimal('0'),
                Exchange.DERIV: Decimal('0')
            },
            'groups': {}  # For correlated assets
        }
        
        # Performance tracking
        self.performance_metrics = {
            'win_rate': 0.5,
            'profit_factor': 1.0,
            'drawdown': 0.0
        }
        
        # Correlation matrix cache
        self.correlation_matrix = {}
        self.correlation_last_updated = 0
        
        logger.info(f"ExposureManager initialized with config: {self.config}")
    
    async def update_exposure(
        self,
        positions: List[Dict[str, Any]],
        account_balance: Dict[str, Decimal]
    ) -> Dict[str, Any]:
        """
        Update the current exposure based on open positions and account balance.
        
        Args:
            positions: List of open positions with details
            account_balance: Dictionary with account balance per platform
            
        Returns:
            Updated exposure metrics
        """
        # Reset current exposure
        self.current_exposure = {
            'total': Decimal('0'),
            'assets': {},
            'platforms': {
                Exchange.BINANCE: Decimal('0'),
                Exchange.DERIV: Decimal('0')
            },
            'groups': {}
        }
        
        # Calculate total balance across platforms
        total_balance = sum(account_balance.values())
        
        # Update correlation matrix if needed
        await self._update_correlation_matrix([p['symbol'] for p in positions])
        
        # Group correlated assets
        correlated_groups = await self._identify_correlated_assets([p['symbol'] for p in positions])
        
        # Calculate exposure for each position
        for position in positions:
            symbol = position['symbol']
            platform = position['platform']
            position_value = position['position_value']
            
            # Update asset exposure
            if symbol not in self.current_exposure['assets']:
                self.current_exposure['assets'][symbol] = Decimal('0')
            self.current_exposure['assets'][symbol] += position_value
            
            # Update platform exposure
            self.current_exposure['platforms'][platform] += position_value
            
            # Update total exposure
            self.current_exposure['total'] += position_value
            
            # Update group exposure
            for group_id, group_symbols in correlated_groups.items():
                if symbol in group_symbols:
                    if group_id not in self.current_exposure['groups']:
                        self.current_exposure['groups'][group_id] = Decimal('0')
                    self.current_exposure['groups'][group_id] += position_value
        
        # Calculate exposure as percentage of account balance
        exposure_percentage = {
            'total': self.current_exposure['total'] / total_balance if total_balance else Decimal('0'),
            'assets': {s: v / total_balance for s, v in self.current_exposure['assets'].items()},
            'platforms': {p: v / total_balance for p, v in self.current_exposure['platforms'].items()},
            'groups': {g: v / total_balance for g, v in self.current_exposure['groups'].items()}
        }
        
        # Add exposure percentage to current exposure
        self.current_exposure['percentage'] = exposure_percentage
        
        logger.info(f"Updated exposure: Total {exposure_percentage['total']:.2%}, "
                   f"Platforms: {dict((k, float(v)) for k, v in exposure_percentage['platforms'].items())}")
        
        return self.current_exposure
    
    async def check_exposure_limits(
        self,
        symbol: str,
        platform: Exchange,
        potential_position_value: Decimal,
        account_balance: Dict[Exchange, Decimal]
    ) -> Dict[str, Any]:
        """
        Check if a new position would exceed exposure limits.
        
        Args:
            symbol: The symbol to trade
            platform: The trading platform
            potential_position_value: Value of the potential position
            account_balance: Dictionary with account balance per platform
            
        Returns:
            Dictionary with exposure check results and reasons
        """
        # Calculate total balance across platforms
        total_balance = sum(account_balance.values())
        
        # Apply any dynamic adjustments to limits
        adjusted_limits = await self._get_adjusted_limits()
        
        # Check total exposure limit
        new_total_exposure = self.current_exposure['total'] + potential_position_value
        new_total_exposure_pct = new_total_exposure / total_balance if total_balance else Decimal('1')
        
        if new_total_exposure_pct > adjusted_limits['max_total_exposure']:
            return {
                'allowed': False,
                'reason': (
                    f"Total exposure limit exceeded: {new_total_exposure_pct:.2%} "
                    f"> {adjusted_limits['max_total_exposure']:.2%}"
                ),
                'limit_type': 'total_exposure'
            }
        
        # Check platform exposure limit
        new_platform_exposure = self.current_exposure['platforms'][platform] + potential_position_value
        new_platform_exposure_pct = new_platform_exposure / total_balance if total_balance else Decimal('1')
        
        if new_platform_exposure_pct > adjusted_limits['max_platform_exposure'][platform]:
            return {
                'allowed': False,
                'reason': f"Platform exposure limit exceeded for {platform}: "
                         f"{new_platform_exposure_pct:.2%} > {adjusted_limits['max_platform_exposure'][platform]:.2%}",
                'limit_type': 'platform_exposure'
            }
        
        # Check single asset exposure limit
        current_asset_exposure = self.current_exposure['assets'].get(symbol, Decimal('0'))
        new_asset_exposure = current_asset_exposure + potential_position_value
        new_asset_exposure_pct = new_asset_exposure / total_balance if total_balance else Decimal('1')
        
        if new_asset_exposure_pct > adjusted_limits['max_single_asset_exposure']:
            return {
                'allowed': False,
                'reason': f"Single asset exposure limit exceeded for {symbol}: "
                         f"{new_asset_exposure_pct:.2%} > {adjusted_limits['max_single_asset_exposure']:.2%}",
                'limit_type': 'asset_exposure'
            }
        
        # Check correlated assets exposure limit
        correlated_assets = await self._get_correlated_assets(symbol)
        
        if correlated_assets:
            # Calculate current exposure to correlated assets
            correlated_exposure = Decimal('0')
            for corr_symbol in correlated_assets:
                correlated_exposure += self.current_exposure['assets'].get(corr_symbol, Decimal('0'))
            
            # Add potential new position
            new_correlated_exposure = correlated_exposure + potential_position_value
            new_correlated_exposure_pct = new_correlated_exposure / total_balance if total_balance else Decimal('1')
            
            if new_correlated_exposure_pct > adjusted_limits['max_correlated_exposure']:
                return {
                    'allowed': False,
                    'reason': f"Correlated assets exposure limit exceeded: "
                             f"{new_correlated_exposure_pct:.2%} > {adjusted_limits['max_correlated_exposure']:.2%}",
                    'limit_type': 'correlated_exposure',
                    'correlated_assets': correlated_assets
                }
        
        # All checks passed
        return {
            'allowed': True,
            'new_exposure': {
                'total': new_total_exposure_pct,
                'platform': new_platform_exposure_pct,
                'asset': new_asset_exposure_pct,
                'correlated': new_correlated_exposure_pct if 'new_correlated_exposure_pct' in locals() else None
            }
        }
    
    async def get_available_margin(
        self,
        account_balance: Dict[Exchange, Decimal],
        platform: Optional[Exchange] = None
    ) -> Dict[str, Any]:
        """
        Calculate available margin for new positions.
        
        Args:
            account_balance: Dictionary with account balance per platform
            platform: Optional platform to check specific platform margin
            
        Returns:
            Dictionary with available margin details
        """
        # Calculate total balance across platforms
        total_balance = sum(account_balance.values())
        
        # Apply any dynamic adjustments to limits
        adjusted_limits = await self._get_adjusted_limits()
        
        # Calculate available margin for total exposure
        max_total_exposure = adjusted_limits['max_total_exposure'] * total_balance
        available_total_margin = max_total_exposure - self.current_exposure['total']
        
        # Calculate available margin for each platform
        available_platform_margin = {}
        for plat, max_exposure in adjusted_limits['max_platform_exposure'].items():
            if plat in account_balance:
                plat_balance = account_balance[plat]
                max_plat_exposure = max_exposure * plat_balance
                current_plat_exposure = self.current_exposure['platforms'][plat]
                available_platform_margin[plat] = max_plat_exposure - current_plat_exposure
        
        result = {
            'total_balance': total_balance,
            'total_exposure': self.current_exposure['total'],
            'total_exposure_pct': self.current_exposure['total'] / total_balance if total_balance else Decimal('0'),
            'available_total_margin': available_total_margin,
            'available_platform_margin': available_platform_margin,
        }
        
        # If specific platform requested, add platform-specific details
        if platform and platform in available_platform_margin:
            result['platform'] = platform
            result['platform_balance'] = account_balance.get(platform, Decimal('0'))
            result['platform_exposure'] = self.current_exposure['platforms'][platform]
            result['platform_exposure_pct'] = (
                self.current_exposure['platforms'][platform] / account_balance.get(platform, Decimal('1'))
            )
            result['available_margin'] = available_platform_margin[platform]
        
        return result
    
    async def calculate_max_position_size(
        self,
        symbol: str,
        platform: Exchange,
        account_balance: Dict[Exchange, Decimal],
        price: Decimal,
        risk_factor: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate the maximum position size based on exposure limits.
        
        Args:
            symbol: The symbol to trade
            platform: The trading platform
            account_balance: Dictionary with account balance per platform
            price: Current price of the asset
            risk_factor: Optional risk factor to apply (0.0-1.0)
            
        Returns:
            Dictionary with maximum position size details
        """
        # Get available margin information
        margin_info = await self.get_available_margin(account_balance, platform)
        available_margin = margin_info.get('available_margin', Decimal('0'))
        
        # Apply any dynamic adjustments to limits
        adjusted_limits = await self._get_adjusted_limits()
        
        # Apply risk factor if provided
        if risk_factor is not None:
            risk_multiplier = Decimal(str(risk_factor))
        else:
            # Default risk multiplier based on performance metrics
            risk_multiplier = Decimal(str(self._calculate_risk_multiplier()))
        
        # Calculate raw maximum position value
        max_position_value = available_margin * risk_multiplier
        
        # Check single asset exposure limit
        total_balance = sum(account_balance.values())
        current_asset_exposure = self.current_exposure['assets'].get(symbol, Decimal('0'))
        max_asset_exposure = adjusted_limits['max_single_asset_exposure'] * total_balance
        available_asset_margin = max_asset_exposure - current_asset_exposure
        
        if available_asset_margin < max_position_value:
            max_position_value = available_asset_margin
        
        # Check correlated assets exposure limit
        correlated_assets = await self._get_correlated_assets(symbol)
        
        if correlated_assets:
            # Calculate current exposure to correlated assets
            correlated_exposure = Decimal('0')
            for corr_symbol in correlated_assets:
                correlated_exposure += self.current_exposure['assets'].get(corr_symbol, Decimal('0'))
            
            max_correlated_exposure = adjusted_limits['max_correlated_exposure'] * total_balance
            available_correlated_margin = max_correlated_exposure - correlated_exposure
            
            if available_correlated_margin < max_position_value:
                max_position_value = available_correlated_margin
        
        # Calculate actual position size
        if price > Decimal('0'):
            max_position_size = max_position_value / price
        else:
            max_position_size = Decimal('0')
        
        # Apply additional platform-specific requirements
        if platform == Exchange.BINANCE:
            min_notional = Decimal('10')
            lot_step = Decimal('0.001')
            if max_position_value < min_notional:
                max_position_value = Decimal('0')
                max_position_size = Decimal('0')
            else:
                max_position_size = (max_position_size // lot_step) * lot_step
                max_position_value = max_position_size * price
        elif platform == Exchange.DERIV:
            contract_step = Decimal('0.1')
            max_position_size = (max_position_size // contract_step) * contract_step
            max_position_value = max_position_size * price
        
        return {
            'symbol': symbol,
            'platform': platform,
            'max_position_value': max_position_value,
            'max_position_size': max_position_size,
            'price': price,
            'risk_multiplier': risk_multiplier,
            'correlated_assets': correlated_assets,
            'limits': {
                'total': adjusted_limits['max_total_exposure'],
                'platform': adjusted_limits['max_platform_exposure'][platform],
                'asset': adjusted_limits['max_single_asset_exposure'],
                'correlated': adjusted_limits['max_correlated_exposure']
            }
        }
    
    async def update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Update performance metrics used for dynamic exposure adjustment.
        
        Args:
            metrics: Dictionary with performance metrics
        """
        if 'win_rate' in metrics:
            self.performance_metrics['win_rate'] = metrics['win_rate']
        
        if 'profit_factor' in metrics:
            self.performance_metrics['profit_factor'] = metrics['profit_factor']
        
        if 'drawdown' in metrics:
            self.performance_metrics['drawdown'] = metrics['drawdown']
        
        logger.info(f"Updated performance metrics: {self.performance_metrics}")
    
    async def _update_correlation_matrix(self, symbols: List[str]) -> None:
        """
        Update the correlation matrix for the given symbols.
        
        Args:
            symbols: List of symbols to include in correlation matrix
        """
        # Check if we have all symbols in our current matrix
        missing_symbols = [s for s in symbols if s not in self.correlation_matrix]
        
        if missing_symbols or (asyncio.get_event_loop().time() - self.correlation_last_updated > 86400):  # Update daily
            # Get historical data for calculation
            market_data = MarketDataRepository()
            symbol_data = {}

            for symbol in symbols:
                try:
                    # Get daily candles for correlation
                    candles = await market_data.get_ohlcv_data(
                        asset_id=symbol,
                        timeframe='1d',
                        limit=self.correlation_lookback,
                    )

                    # Extract closing prices
                    if hasattr(candles, 'empty'):
                        df = candles
                        if not df.empty:
                            closes = df['close'].astype(float).values
                            symbol_data[symbol] = np.array(closes)
                    elif candles and len(candles) > 0:
                        closes = [float(c['close']) for c in candles]
                        symbol_data[symbol] = np.array(closes)
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
            
            # Calculate correlation matrix if we have data
            if len(symbol_data) > 1:
                matrix = await run_in_threadpool(
                    calculate_correlation_matrix,
                    symbol_data
                )
                
                # Update the correlation matrix
                self.correlation_matrix = matrix
                self.correlation_last_updated = asyncio.get_event_loop().time()
                
                logger.info(f"Updated correlation matrix for {len(symbol_data)} symbols")
    
    async def _identify_correlated_assets(self, symbols: List[str]) -> Dict[str, List[str]]:
        """
        Identify groups of correlated assets.
        
        Args:
            symbols: List of symbols to check
            
        Returns:
            Dictionary of correlated asset groups
        """
        correlated_groups = {}
        group_id = 0
        processed_symbols = set()
        
        for symbol in symbols:
            if symbol in processed_symbols:
                continue
            
            # Find correlated symbols
            correlated = await self._get_correlated_assets(symbol)
            
            if correlated:
                # Create a new group
                group_id += 1
                group_name = f"group_{group_id}"
                correlated_groups[group_name] = [symbol] + correlated
                
                # Mark all as processed
                processed_symbols.update([symbol] + correlated)
        
        return correlated_groups
    
    async def _get_correlated_assets(self, symbol: str) -> List[str]:
        """
        Get list of assets correlated with the given symbol.
        
        Args:
            symbol: Symbol to check correlations for
            
        Returns:
            List of correlated asset symbols
        """
        correlated = []
        
        if symbol in self.correlation_matrix:
            for other_symbol, correlation in self.correlation_matrix[symbol].items():
                if other_symbol != symbol and abs(correlation) >= self.correlation_threshold:
                    correlated.append(other_symbol)
        
        return correlated
    
    async def _get_adjusted_limits(self) -> Dict[str, Any]:
        """
        Get dynamically adjusted exposure limits based on current market conditions
        and performance metrics.
        
        Returns:
            Dictionary with adjusted exposure limits
        """
        adjusted_limits = {
            'max_total_exposure': self.max_total_exposure,
            'max_single_asset_exposure': self.max_single_asset_exposure,
            'max_correlated_exposure': self.max_correlated_exposure,
            'max_platform_exposure': self.max_platform_exposure.copy()
        }
        
        # Adjust based on performance if enabled
        if self.performance_based_adjustment:
            performance_multiplier = self._calculate_performance_multiplier()
            
            # Apply performance-based adjustments
            adjusted_limits['max_total_exposure'] *= performance_multiplier
            adjusted_limits['max_single_asset_exposure'] *= performance_multiplier
            adjusted_limits['max_correlated_exposure'] *= performance_multiplier
            
            for platform in adjusted_limits['max_platform_exposure']:
                adjusted_limits['max_platform_exposure'][platform] *= performance_multiplier
        
        # Adjust based on volatility if enabled
        if self.volatility_based_adjustment:
            volatility_multiplier = await self._calculate_volatility_multiplier()
            
            # Apply volatility-based adjustments
            volatility_impact = 0.2  # 20% impact from volatility
            volatility_factor = 1.0 + (volatility_multiplier - 1.0) * volatility_impact
            
            adjusted_limits['max_total_exposure'] *= volatility_factor
            adjusted_limits['max_single_asset_exposure'] *= volatility_factor
            adjusted_limits['max_correlated_exposure'] *= volatility_factor
            
            for platform in adjusted_limits['max_platform_exposure']:
                adjusted_limits['max_platform_exposure'][platform] *= volatility_factor
        
        # Ensure limits don't exceed 1.0 (100%)
        adjusted_limits['max_total_exposure'] = min(
            Decimal('1.0'),
            Decimal(str(adjusted_limits['max_total_exposure']))
        )
        adjusted_limits['max_single_asset_exposure'] = min(
            Decimal('1.0'),
            Decimal(str(adjusted_limits['max_single_asset_exposure']))
        )
        adjusted_limits['max_correlated_exposure'] = min(
            Decimal('1.0'),
            Decimal(str(adjusted_limits['max_correlated_exposure']))
        )
        
        for platform in adjusted_limits['max_platform_exposure']:
            adjusted_limits['max_platform_exposure'][platform] = min(
                Decimal('1.0'), 
                Decimal(str(adjusted_limits['max_platform_exposure'][platform]))
            )
        
        return adjusted_limits
    
    def _calculate_performance_multiplier(self) -> float:
        """
        Calculate a multiplier for exposure limits based on recent performance.
        
        Returns:
            Performance multiplier (0.5-1.2)
        """
        win_rate = self.performance_metrics['win_rate']
        profit_factor = self.performance_metrics['profit_factor']
        drawdown = self.performance_metrics['drawdown']
        
        # Base multiplier on win rate
        win_rate_factor = 1.0
        if win_rate >= 0.7:  # Very good win rate
            win_rate_factor = 1.2
        elif win_rate >= 0.6:  # Good win rate
            win_rate_factor = 1.1
        elif win_rate <= 0.4:  # Poor win rate
            win_rate_factor = 0.8
        elif win_rate <= 0.3:  # Very poor win rate
            win_rate_factor = 0.6
        
        # Adjust based on profit factor
        profit_factor_adjustment = 0.0
        if profit_factor >= 2.0:
            profit_factor_adjustment = 0.1
        elif profit_factor >= 1.5:
            profit_factor_adjustment = 0.05
        elif profit_factor <= 0.8:
            profit_factor_adjustment = -0.1
        elif profit_factor <= 0.5:
            profit_factor_adjustment = -0.2
        
        # Adjust based on drawdown
        drawdown_adjustment = 0.0
        if drawdown >= 0.2:  # Large drawdown
            drawdown_adjustment = -0.2
        elif drawdown >= 0.15:
            drawdown_adjustment = -0.15
        elif drawdown >= 0.1:
            drawdown_adjustment = -0.1
        elif drawdown <= 0.05:  # Small drawdown
            drawdown_adjustment = 0.05
        
        # Calculate final multiplier with limits
        multiplier = win_rate_factor + profit_factor_adjustment + drawdown_adjustment
        return max(0.5, min(1.2, multiplier))  # Cap between 0.5 and 1.2
    
    async def _calculate_volatility_multiplier(self) -> float:
        """
        Calculate a multiplier for exposure limits based on current market volatility.

        Returns:
            Volatility multiplier (0.7-1.2)
        """
        try:
            assets = self.config.get("volatility_assets") or ASSETS[:5]
            short_window = self.config.get("volatility_short_window", 20)
            long_window = self.config.get("volatility_long_window", 60)
            ratios = []

            market_data = MarketDataRepository()
            for asset in assets:
                candles = await market_data.get_ohlcv_data(
                    asset_id=asset,
                    timeframe="1h",
                    limit=long_window + 1,
                )

                closes: List[float]
                if isinstance(candles, pd.DataFrame):
                    if candles.empty:
                        continue
                    closes = candles["close"].astype(float).tolist()
                elif candles:
                    closes = [
                        float(c.close if hasattr(c, "close") else c["close"])
                        for c in candles
                    ]
                else:
                    continue

                series = pd.Series(closes)
                short_vol = calculate_volatility(series[-short_window:], window=short_window)
                long_vol = calculate_volatility(series, window=long_window)

                if long_vol == 0:
                    continue

                ratios.append(short_vol / long_vol)

            if not ratios:
                return 1.0

            avg_ratio = float(np.mean(ratios))

            if avg_ratio >= 1.0:
                multiplier = 1.0 - min(0.3, (avg_ratio - 1.0) * 0.3)
            else:
                multiplier = 1.0 + min(0.2, (1.0 - avg_ratio) * 0.2)

            return max(0.7, min(1.2, multiplier))
        except Exception as e:  # pragma: no cover - graceful fallback
            logger.error(f"Error calculating volatility multiplier: {e}")
            return 1.0
    
    def _calculate_risk_multiplier(self) -> float:
        """
        Calculate a risk multiplier for position sizing based on performance metrics.
        
        Returns:
            Risk multiplier (0.3-1.0)
        """
        win_rate = self.performance_metrics['win_rate']
        profit_factor = self.performance_metrics['profit_factor']
        
        # Base risk on performance metrics
        if win_rate >= 0.7 and profit_factor >= 2.0:
            return 1.0  # Excellent performance - use maximum allowed size
        elif win_rate >= 0.6 and profit_factor >= 1.5:
            return 0.8  # Very good performance
        elif win_rate >= 0.55 and profit_factor >= 1.3:
            return 0.7  # Good performance
        elif win_rate >= 0.5 and profit_factor >= 1.1:
            return 0.6  # Decent performance
        elif win_rate >= 0.45 and profit_factor >= 1.0:
            return 0.5  # Average performance
        elif win_rate >= 0.4:
            return 0.4  # Below average performance
        else:
            return 0.3  # Poor performance - use minimum size
    
    def get_exposure_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive exposure report.
        
        Returns:
            Dictionary with detailed exposure information
        """
        return {
            'current_exposure': self.current_exposure,
            'performance_metrics': self.performance_metrics,
            'limits': {
                'max_total_exposure': self.max_total_exposure,
                'max_single_asset_exposure': self.max_single_asset_exposure,
                'max_correlated_exposure': self.max_correlated_exposure,
                'max_platform_exposure': self.max_platform_exposure
            },
            'correlation_data': {
                'threshold': self.correlation_threshold,
                'last_updated': self.correlation_last_updated,
                'matrix_size': len(self.correlation_matrix)
            }
        }


def get_exposure_manager(name: str, *args, **kwargs) -> BaseExposureManager:
    """Instantiate a registered exposure manager by name."""
    cls = BaseExposureManager.registry.get(name)
    if cls is None:
        raise ValueError(f"Unknown exposure manager: {name}")
    return cls(*args, **kwargs)

__all__ = [
    "BaseExposureManager",
    "get_exposure_manager",
    "ExposureManager",
    "MarketDataRepository",
]
