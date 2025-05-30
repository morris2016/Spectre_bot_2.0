#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Market Structure Analysis Module

This module provides advanced market structure analysis capabilities for the 
QuantumSpectre Elite Trading System. It identifies swing highs/lows, support/resistance
levels, market phases, and trend strength across multiple timeframes.

This analysis forms the foundation for many trading strategies that rely on
structural market movements and plays a critical role in delivering
consistently strong performance for the system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum, auto
import logging
from dataclasses import dataclass
from scipy.signal import argrelextrema
from scipy.stats import linregress
try:
    import ta  # type: ignore
    TA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    ta = None  # type: ignore
    TA_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "ta library not available; some market structure features disabled"
    )


def identify_swing_points(df: pd.DataFrame, window: int = 5) -> List[int]:
    """Identify simple swing high/low points."""
    points: List[int] = []
    highs = df['high']
    lows = df['low']
    for i in range(window, len(df) - window):
        if highs.iloc[i] >= highs.iloc[i - window:i + window + 1].max():
            points.append(i)
        elif lows.iloc[i] <= lows.iloc[i - window:i + window + 1].min():
            points.append(i)
    return points

from common.logger import get_logger
from common.utils import parallelize_calculation, window_calculation
from feature_service.features.base_feature import BaseFeature


def identify_swing_points(df: pd.DataFrame, order: int = 5) -> List[int]:
    """Identify swing high and low indexes using local extrema."""
    highs = argrelextrema(df['high'].values, np.greater, order=order)[0]
    lows = argrelextrema(df['low'].values, np.less, order=order)[0]
    return sorted(list(highs) + list(lows))

logger = get_logger(__name__)
class MarketStructure(BaseFeature):
    """
    Market structure analysis feature that identifies swing points, trends, and market phases.
    """
    
    def __init__(self, window_size=20, order=5, min_strength=0.3):
        """
        Initialize the market structure analyzer.
        
        Args:
            window_size: Window size for swing point detection
            order: Order parameter for extrema detection
            min_strength: Minimum strength for valid swing points
        """
        self.window_size = window_size
        self.order = order
        self.min_strength = min_strength
        self.logger = get_logger(__name__)
        
    def calculate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate market structure features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dict with market structure features
        """
        if len(data) < self.window_size * 2:
            self.logger.warning(f"Insufficient data for market structure analysis: {len(data)} rows")
            return {}
            
        # Find swing points
        swing_points = find_swing_points(data, self.window_size, self.order)
        
        # Identify market structure
        structure = identify_market_structure(data, swing_points)
        
        # Calculate trend strength
        trend_strength = calculate_trend_strength(data)
        
        return {
            "swing_points": swing_points,
            "market_structure": structure,
            "trend_strength": trend_strength,
            "support_levels": structure.get("support_levels", []),
            "resistance_levels": structure.get("resistance_levels", [])
        }
        
    def get_feature_names(self) -> List[str]:
        """Get the names of features provided by this calculator."""
        return [
            "swing_points",
            "market_structure",
            "trend_strength",
            "support_levels",
            "resistance_levels"
        ]


def find_swing_points(data: pd.DataFrame, window_size=20, order=5) -> Dict[str, List[Dict]]:
    """
    Find swing high and swing low points in price data.
    
    Args:
        data: DataFrame with OHLCV data
        window_size: Window size for extrema detection
        order: Order parameter for extrema detection
        
    Returns:
        Dict with swing highs and swing lows
    """
    if len(data) < window_size * 2:
        return {"highs": [], "lows": []}
        
    # Get high and low prices
    highs = data['high'].values
    lows = data['low'].values
    
    # Find local maxima and minima
    high_idx = list(argrelextrema(highs, np.greater, order=order)[0])
    low_idx = list(argrelextrema(lows, np.less, order=order)[0])
    
    # Filter out weak swing points
    swing_highs = []
    for idx in high_idx:
        if idx < window_size or idx > len(data) - window_size:
            continue
            
        # Calculate strength based on surrounding bars
        left_strength = (highs[idx] - np.max(highs[idx-window_size:idx])) / highs[idx]
        right_strength = (highs[idx] - np.max(highs[idx+1:idx+window_size+1])) / highs[idx]
        strength = min(left_strength, right_strength)
        
        swing_highs.append({
            "index": idx,
            "price": highs[idx],
            "timestamp": data.index[idx],
            "strength": strength
        })
    
    swing_lows = []
    for idx in low_idx:
        if idx < window_size or idx > len(data) - window_size:
            continue
            
        # Calculate strength based on surrounding bars
        left_strength = (np.min(lows[idx-window_size:idx]) - lows[idx]) / lows[idx]
        right_strength = (np.min(lows[idx+1:idx+window_size+1]) - lows[idx]) / lows[idx]
        strength = min(left_strength, right_strength)
        
        swing_lows.append({
            "index": idx,
            "price": lows[idx],
            "timestamp": data.index[idx],
            "strength": strength
        })
    
    return {
        "highs": swing_highs,
        "lows": swing_lows
    }


def find_support_resistance_levels(data: pd.DataFrame, window: int = 20) -> Tuple[List[float], List[float]]:
    """Identify simple support and resistance levels from high and low prices."""
    supports = []
    resistances = []
    highs = data['high']
    lows = data['low']

    for i in range(window, len(data)):
        if highs.iloc[i] >= highs.iloc[i - window:i].max():
            resistances.append(highs.iloc[i])
        if lows.iloc[i] <= lows.iloc[i - window:i].min():
            supports.append(lows.iloc[i])

    return supports, resistances


def classify_market_structure(data: pd.DataFrame, short_period: int = 20, long_period: int = 50) -> str:
    """Classify market structure as bullish, bearish, or sideways."""
    if len(data) < long_period:
        return "neutral"
    ma_short = data['close'].rolling(window=short_period).mean().iloc[-1]
    ma_long = data['close'].rolling(window=long_period).mean().iloc[-1]
    if ma_short > ma_long:
        return "bullish"
    if ma_short < ma_long:
        return "bearish"
    return "sideways"


def identify_market_structure(data: pd.DataFrame, swing_points: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Identify market structure based on swing points.
    
    Args:
        data: DataFrame with OHLCV data
        swing_points: Dict with swing highs and swing lows
        
    Returns:
        Dict with market structure information
    """
    if not swing_points["highs"] or not swing_points["lows"]:
        return {
            "trend": "neutral",
            "phase": "consolidation",
            "support_levels": [],
            "resistance_levels": []
        }
    
    # Sort swing points by index
    highs = sorted(swing_points["highs"], key=lambda x: x["index"])
    lows = sorted(swing_points["lows"], key=lambda x: x["index"])
    
    # Determine trend based on recent swing points
    if len(highs) >= 2 and len(lows) >= 2:
        recent_highs = highs[-2:]
        recent_lows = lows[-2:]
        
        higher_highs = recent_highs[1]["price"] > recent_highs[0]["price"]
        higher_lows = recent_lows[1]["price"] > recent_lows[0]["price"]
        
        if higher_highs and higher_lows:
            trend = "uptrend"
        elif not higher_highs and not higher_lows:
            trend = "downtrend"
        else:
            trend = "neutral"
    else:
        trend = "neutral"
    
    # Determine market phase
    if trend == "uptrend":
        if data['close'].iloc[-1] > data['close'].iloc[-5:].mean():
            phase = "accumulation"
        else:
            phase = "distribution"
    elif trend == "downtrend":
        if data['close'].iloc[-1] < data['close'].iloc[-5:].mean():
            phase = "distribution"
        else:
            phase = "accumulation"
    else:
        phase = "consolidation"
    
    # Identify support and resistance levels
    support_levels = [low["price"] for low in lows[-3:]]
    resistance_levels = [high["price"] for high in highs[-3:]]
    
    return {
        "trend": trend,
        "phase": phase,
        "support_levels": support_levels,
        "resistance_levels": resistance_levels
    }


def calculate_trend_strength(data: pd.DataFrame, window=20) -> float:
    """
    Calculate the strength of the current trend.
    
    Args:
        data: DataFrame with OHLCV data
        window: Window size for trend strength calculation
        
    Returns:
        Float representing trend strength (0.0 to 1.0)
    """
    if len(data) < window:
        return 0.0
    
    # Use linear regression to determine trend strength
    y = data['close'].values[-window:]
    x = np.arange(len(y))
    
    slope, _, r_value, _, _ = linregress(x, y)
    
    # Normalize r-squared to get trend strength
    trend_strength = min(1.0, abs(r_value) ** 2)
    
    # Adjust sign based on slope
    if slope < 0:
        trend_strength = -trend_strength
    
    return trend_strength


class MarketStructureAnalyzer:
    """
    Utility class for analyzing market structure across multiple timeframes.
    """
    
    def __init__(self, timeframes=None):
        """
        Initialize the market structure analyzer.
        
        Args:
            timeframes: List of timeframes to analyze
        """
        self.timeframes = timeframes or ["1h", "4h", "1d"]
        self.analyzers = {tf: MarketStructure() for tf in self.timeframes}
        
    def analyze(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Analyze market structure across multiple timeframes.
        
        Args:
            data_dict: Dict mapping timeframes to DataFrames
            
        Returns:
            Dict with market structure analysis for each timeframe
        """
        results = {}
        
        for tf, analyzer in self.analyzers.items():
            if tf in data_dict and len(data_dict[tf]) > 0:
                results[tf] = analyzer.calculate(data_dict[tf])
            else:
                results[tf] = {}
                
        return results


class MarketPhase(Enum):
    """Market structure phases used for classification."""
    ACCUMULATION = auto()
    UPTREND = auto()
    DISTRIBUTION = auto()
    DOWNTREND = auto()
    RANGING = auto()
    CHOPPY = auto()
    BREAKOUT_UP = auto()
    BREAKOUT_DOWN = auto()
    REVERSAL_UP = auto()
    REVERSAL_DOWN = auto()


@dataclass
class SwingPoint:
    """Represents a swing high or low in market structure."""
    index: int
    timestamp: pd.Timestamp
    price: float
    type: str  # 'high' or 'low'
    strength: float
    confluence_level: float
    timeframe: str
    validated: bool = False


@dataclass
class SupportResistanceLevel:
    """Represents a support or resistance level."""
    price: float
    type: str  # 'support' or 'resistance'
    strength: float
    touches: int
    created_at: pd.Timestamp
    last_tested: pd.Timestamp
    timeframe: str
    sources: List[str]  # what identified this level (swing, fib, round number, etc.)
    broken: bool = False


class MarketStructureFeature(BaseFeature):
    """
    Advanced market structure analysis for determining support/resistance,
    market regimes, trend strength, and structural patterns.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize market structure feature calculator.
        
        Args:
            config: Configuration parameters for market structure detection
        """
        super().__init__(config)
        self.name = "market_structure"
        self.description = "Advanced market structure analytics"
        self.version = "1.0.0"
        
        # Configuration with defaults
        self.config = {
            'swing_lookback': 5,
            'swing_leniency': 0.5,
            'pivot_threshold': 0.3,
            'level_proximity_pct': 0.2,
            'level_merger_pct': 0.1,
            'level_strength_decay': 0.95,
            'level_history': 100,
            'min_touches': 2,
            'volume_weight': 0.3,
            'sr_timeframes': ['1h', '4h', '1d'],
            'market_phase_lookback': 20,
            'order_blocks_lookback': 10,
            'liquidity_levels_pct': 0.5,
            'smart_money_threshold': 0.7,
            'vwap_std_multiplier': 2.0,
            'round_number_influence': 0.5
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # Initialize containers for market structure elements
        self.swing_points = {}
        self.sr_levels = {}
        self.market_phases = {}
        self.order_blocks = {}
        self.liquidity_levels = {}
        self.last_calculated_index = -1
        
        logger.info(f"Initialized {self.name} feature calculator v{self.version}")

    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate market structure features from price data.
        
        Args:
            df: DataFrame with OHLCV data
            **kwargs: Additional parameters including timeframe
        
        Returns:
            DataFrame with added market structure features
        """
        timeframe = kwargs.get('timeframe', '1h')
        asset = kwargs.get('asset', 'unknown')
        logger.debug(f"Calculating market structure features for {asset} on {timeframe} timeframe")
        
        # Make a copy to avoid modifying the original dataframe
        result_df = df.copy()
        
        try:
            # Identify swing points
            swing_points = self._identify_swing_points(result_df, timeframe)
            
            # Detect support/resistance levels
            sr_levels = self._detect_support_resistance(result_df, swing_points, timeframe)
            
            # Determine market phase
            result_df['market_phase'] = self._determine_market_phase(result_df, swing_points, timeframe)
            
            # Calculate trend strength
            result_df['trend_strength'] = self._calculate_trend_strength(result_df)
            
            # Calculate structural support/resistance levels
            sr_levels_array = np.array([level.price for level in sr_levels])
            
            # Add closest support levels
            result_df['closest_support'] = self._find_closest_level(
                result_df['close'].values, sr_levels_array, 'support')
            
            # Add closest resistance levels
            result_df['closest_resistance'] = self._find_closest_level(
                result_df['close'].values, sr_levels_array, 'resistance')
            
            # Additional structural features
            result_df['sr_density'] = self._calculate_sr_density(result_df, sr_levels)
            result_df['order_blocks'] = self._identify_order_blocks(result_df, timeframe)
            result_df['liquidity_levels'] = self._identify_liquidity_levels(result_df)
            result_df['smart_money_levels'] = self._identify_smart_money_levels(result_df)
            
            # Store results for future reference
            self.swing_points[timeframe] = swing_points
            self.sr_levels[timeframe] = sr_levels
            self.market_phases[timeframe] = result_df['market_phase'].iloc[-1]
            self.last_calculated_index = len(result_df) - 1
            
            logger.debug(f"Completed market structure calculation for {asset} on {timeframe}")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating market structure: {str(e)}")
            # Return original DataFrame if calculation fails
            return df

    def _identify_swing_points(self, df: pd.DataFrame, timeframe: str) -> List[SwingPoint]:
        """
        Identify swing highs and lows in the price data.
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe of the data
            
        Returns:
            List of SwingPoint objects
        """
        logger.debug(f"Identifying swing points for {timeframe}")
        
        lookback = self.config['swing_lookback']
        
        # Find local maxima and minima
        try:
            # Use scipy's argrelextrema for efficient detection
            highs_idx = argrelextrema(df['high'].values, np.greater, 
                                      order=lookback)[0]
            lows_idx = argrelextrema(df['low'].values, np.less, 
                                     order=lookback)[0]
            
            swing_points = []
            
            # Process swing highs
            for idx in highs_idx:
                if idx < lookback or idx > len(df) - lookback:
                    continue
                    
                # Calculate strength based on surrounding price action
                window = df['high'].iloc[idx-lookback:idx+lookback+1]
                strength = (df['high'].iloc[idx] - window.mean()) / window.std()
                
                # Calculate confluence based on volume and nearby indicators
                volume_factor = df['volume'].iloc[idx] / df['volume'].iloc[idx-lookback:idx+lookback+1].mean()
                confluence = min(1.0, (strength * 0.7) + (volume_factor * 0.3))
                
                swing_points.append(SwingPoint(
                    index=idx,
                    timestamp=df.index[idx],
                    price=df['high'].iloc[idx],
                    type='high',
                    strength=strength,
                    confluence_level=confluence,
                    timeframe=timeframe,
                    validated=True if idx < len(df) - lookback else False
                ))
            
            # Process swing lows
            for idx in lows_idx:
                if idx < lookback or idx > len(df) - lookback:
                    continue
                    
                # Calculate strength based on surrounding price action
                window = df['low'].iloc[idx-lookback:idx+lookback+1]
                strength = (window.mean() - df['low'].iloc[idx]) / window.std()
                
                # Calculate confluence based on volume and nearby indicators
                volume_factor = df['volume'].iloc[idx] / df['volume'].iloc[idx-lookback:idx+lookback+1].mean()
                confluence = min(1.0, (strength * 0.7) + (volume_factor * 0.3))
                
                swing_points.append(SwingPoint(
                    index=idx,
                    timestamp=df.index[idx],
                    price=df['low'].iloc[idx],
                    type='low',
                    strength=strength,
                    confluence_level=confluence,
                    timeframe=timeframe,
                    validated=True if idx < len(df) - lookback else False
                ))
            
            # Sort by index
            swing_points.sort(key=lambda x: x.index)
            
            logger.debug(f"Identified {len(swing_points)} swing points")
            return swing_points
            
        except Exception as e:
            logger.error(f"Error in swing point identification: {str(e)}")
            return []

    def _detect_support_resistance(self, df: pd.DataFrame, 
                                  swing_points: List[SwingPoint], 
                                  timeframe: str) -> List[SupportResistanceLevel]:
        """
        Detect support and resistance levels based on swing points and price action.
        
        Args:
            df: DataFrame with OHLCV data
            swing_points: List of swing points
            timeframe: Timeframe of the data
            
        Returns:
            List of SupportResistanceLevel objects
        """
        logger.debug(f"Detecting support/resistance levels for {timeframe}")
        
        proximity_pct = self.config['level_proximity_pct']
        merger_pct = self.config['level_merger_pct']
        
        # Extract price range for percentage calculations
        price_range = df['high'].max() - df['low'].min()
        proximity_threshold = price_range * proximity_pct / 100
        merger_threshold = price_range * merger_pct / 100
        
        sr_levels = []
        
        try:
            # Begin with swing points as initial S/R levels
            for point in swing_points:
                # Skip unvalidated recent swing points
                if not point.validated:
                    continue
                    
                price = point.price
                level_type = 'resistance' if point.type == 'high' else 'support'
                
                # Check if we already have a similar level
                similar_level = None
                for level in sr_levels:
                    if abs(level.price - price) < merger_threshold:
                        similar_level = level
                        break
                
                if similar_level:
                    # Update existing level with new information
                    similar_level.touches += 1
                    similar_level.strength = min(1.0, similar_level.strength + 0.1)
                    similar_level.last_tested = max(similar_level.last_tested, point.timestamp)
                    if point.strength > 0.7 and point.type == level_type:
                        similar_level.strength = min(1.0, similar_level.strength + 0.2)
                    
                    # Recalculate price as weighted average
                    similar_level.price = ((similar_level.price * (similar_level.touches - 1)) + 
                                         price) / similar_level.touches
                else:
                    # Create new level
                    sr_levels.append(SupportResistanceLevel(
                        price=price,
                        type=level_type,
                        strength=point.strength * 0.5,  # Initial strength
                        touches=1,
                        created_at=point.timestamp,
                        last_tested=point.timestamp,
                        timeframe=timeframe,
                        sources=[point.type]
                    ))
            
            # Additional detection of potential S/R levels using price clusters
            price_clusters = self._detect_price_clusters(df)
            for cluster in price_clusters:
                price = cluster['price']
                cluster_type = cluster['type']
                
                # Skip if too close to existing levels
                if any(abs(level.price - price) < proximity_threshold for level in sr_levels):
                    continue
                
                sr_levels.append(SupportResistanceLevel(
                    price=price,
                    type=cluster_type,
                    strength=cluster['strength'],
                    touches=cluster['count'],
                    created_at=df.index[cluster['first_idx']],
                    last_tested=df.index[cluster['last_idx']],
                    timeframe=timeframe,
                    sources=['price_cluster']
                ))
            
            # Add round number levels if they show significance
            round_levels = self._detect_round_number_levels(df)
            for level in round_levels:
                price = level['price']
                
                # Skip if too close to existing levels
                if any(abs(sr.price - price) < proximity_threshold for sr in sr_levels):
                    continue
                
                sr_levels.append(SupportResistanceLevel(
                    price=price,
                    type=level['type'],
                    strength=level['strength'],
                    touches=level['touches'],
                    created_at=df.index[0],  # Assuming it existed since start
                    last_tested=df.index[level['last_test']],
                    timeframe=timeframe,
                    sources=['round_number']
                ))
            
            # Perform level cleanup and merging
            sr_levels = self._merge_close_levels(sr_levels, merger_threshold)
            
            # Sort by price
            sr_levels.sort(key=lambda x: x.price)
            
            logger.debug(f"Detected {len(sr_levels)} support/resistance levels")
            return sr_levels
            
        except Exception as e:
            logger.error(f"Error in support/resistance detection: {str(e)}")
            return []

    def _detect_price_clusters(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect price clusters where price has spent significant time.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of dictionaries with cluster information
        """
        # Calculate price bins for clustering
        bins = 100
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / bins
        
        # Count candles in each price bin
        price_bins = {}
        for i, row in df.iterrows():
            # Check both high and low areas
            high_bin = int(row['high'] / bin_size)
            low_bin = int(row['low'] / bin_size)
            
            # Add volume weight to each bin the candle passed through
            for bin_idx in range(low_bin, high_bin + 1):
                if bin_idx not in price_bins:
                    price_bins[bin_idx] = {
                        'count': 0, 
                        'volume': 0, 
                        'first_idx': len(df), 
                        'last_idx': 0,
                        'price': bin_idx * bin_size
                    }
                price_bins[bin_idx]['count'] += 1
                price_bins[bin_idx]['volume'] += row['volume'] / (high_bin - low_bin + 1)
                price_bins[bin_idx]['first_idx'] = min(price_bins[bin_idx]['first_idx'], 
                                                    df.index.get_loc(i))
                price_bins[bin_idx]['last_idx'] = max(price_bins[bin_idx]['last_idx'], 
                                                   df.index.get_loc(i))
        
        # Identify significant clusters
        significant_clusters = []
        mean_count = np.mean([b['count'] for b in price_bins.values()])
        std_count = np.std([b['count'] for b in price_bins.values()])
        
        for bin_idx, bin_data in price_bins.items():
            if bin_data['count'] > mean_count + std_count:
                # This is a significant cluster
                cluster_price = bin_idx * bin_size + (bin_size / 2)
                
                # Determine if it's support or resistance
                before_after_ratios = []
                idx = bin_data['first_idx']
                while idx < bin_data['last_idx']:
                    # Count candles above and below
                    above = sum(1 for i in range(max(0, idx-10), idx) 
                               if df.iloc[i]['close'] > cluster_price)
                    below = sum(1 for i in range(max(0, idx-10), idx) 
                               if df.iloc[i]['close'] < cluster_price)
                    
                    if above > 0 and below > 0:
                        ratio = above / below if below > 0 else float('inf')
                        before_after_ratios.append(ratio)
                    
                    idx += 10
                
                if before_after_ratios:
                    avg_ratio = np.mean(before_after_ratios)
                    cluster_type = 'resistance' if avg_ratio > 1.2 else 'support' if avg_ratio < 0.8 else 'both'
                else:
                    cluster_type = 'both'
                
                significant_clusters.append({
                    'price': cluster_price,
                    'type': cluster_type,
                    'strength': (bin_data['count'] - mean_count) / std_count / 3,  # Normalize to 0-1
                    'count': bin_data['count'],
                    'volume': bin_data['volume'],
                    'first_idx': bin_data['first_idx'],
                    'last_idx': bin_data['last_idx']
                })
        
        return significant_clusters

    def _detect_round_number_levels(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect psychologically important round number levels.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of dictionaries with round number level information
        """
        # Determine price magnitude for rounding
        price_range = df['high'].max() - df['low'].min()
        mid_price = (df['high'].max() + df['low'].min()) / 2
        
        # Define round number thresholds based on price magnitude
        if mid_price < 1:
            round_factors = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5]
        elif mid_price < 10:
            round_factors = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
        elif mid_price < 100:
            round_factors = [0.1, 0.5, 1, 2.5, 5, 10, 25, 50]
        elif mid_price < 1000:
            round_factors = [1, 5, 10, 25, 50, 100, 250, 500]
        else:
            round_factors = [10, 50, 100, 250, 500, 1000, 2500, 5000]
        
        # Calculate proximity threshold
        proximity_pct = self.config['level_proximity_pct']
        proximity_threshold = price_range * proximity_pct / 100
        
        # Find potential round number levels
        potential_levels = []
        for factor in round_factors:
            # Find round numbers within and slightly beyond the price range
            min_price = df['low'].min() - (price_range * 0.1)
            max_price = df['high'].max() + (price_range * 0.1)
            
            levels = np.arange(
                np.floor(min_price / factor) * factor,
                np.ceil(max_price / factor) * factor,
                factor
            )
            
            for level in levels:
                # Look for touches at this level
                touches = 0
                last_test = 0
                
                for i, row in df.iterrows():
                    if abs(row['high'] - level) <= proximity_threshold or \
                       abs(row['low'] - level) <= proximity_threshold:
                        touches += 1
                        last_test = df.index.get_loc(i)
                
                # Only include levels with significant touches
                if touches >= self.config['min_touches']:
                    # Determine if support or resistance
                    above_count = sum(1 for _, row in df.iterrows() 
                                     if row['close'] > level)
                    below_count = sum(1 for _, row in df.iterrows() 
                                     if row['close'] < level)
                    
                    if above_count > below_count * 1.5:
                        level_type = 'resistance'
                    elif below_count > above_count * 1.5:
                        level_type = 'support'
                    else:
                        level_type = 'both'
                    
                    # Calculate strength based on touches and factor significance
                    factor_significance = round_factors.index(factor) / len(round_factors)
                    strength = min(1.0, (touches / 10) * (0.5 + factor_significance * 0.5))
                    
                    potential_levels.append({
                        'price': level,
                        'type': level_type,
                        'strength': strength,
                        'touches': touches,
                        'last_test': last_test,
                        'factor': factor
                    })
        
        return potential_levels

    def _merge_close_levels(self, sr_levels: List[SupportResistanceLevel], 
                           threshold: float) -> List[SupportResistanceLevel]:
        """
        Merge levels that are very close to each other.
        
        Args:
            sr_levels: List of support/resistance levels
            threshold: Price proximity threshold for merging
            
        Returns:
            List of merged support/resistance levels
        """
        if not sr_levels:
            return []
            
        # Sort by price
        sr_levels.sort(key=lambda x: x.price)
        
        merged_levels = []
        current_group = [sr_levels[0]]
        
        for i in range(1, len(sr_levels)):
            current_level = sr_levels[i]
            last_level = current_group[-1]
            
            if abs(current_level.price - last_level.price) < threshold:
                # Merge into the group
                current_group.append(current_level)
            else:
                # Process the current group
                if len(current_group) == 1:
                    merged_levels.append(current_group[0])
                else:
                    # Create a new merged level
                    weights = [level.strength * level.touches for level in current_group]
                    total_weight = sum(weights)
                    
                    # Weighted average price
                    merged_price = sum(level.price * weight for level, weight in 
                                     zip(current_group, weights)) / total_weight
                    
                    # Determine type (support, resistance, or both)
                    support_weight = sum(weights[i] for i in range(len(current_group)) 
                                        if current_group[i].type == 'support')
                    resistance_weight = sum(weights[i] for i in range(len(current_group)) 
                                           if current_group[i].type == 'resistance')
                    
                    if support_weight > resistance_weight * 1.5:
                        merged_type = 'support'
                    elif resistance_weight > support_weight * 1.5:
                        merged_type = 'resistance'
                    else:
                        merged_type = 'both'
                    
                    # Calculate merged attributes
                    total_touches = sum(level.touches for level in current_group)
                    earliest_creation = min(level.created_at for level in current_group)
                    latest_test = max(level.last_tested for level in current_group)
                    merged_strength = min(1.0, sum(level.strength for level in current_group) / len(current_group) * 1.2)
                    
                    # Combine sources
                    all_sources = []
                    for level in current_group:
                        all_sources.extend(level.sources)
                    unique_sources = list(set(all_sources))
                    
                    merged_levels.append(SupportResistanceLevel(
                        price=merged_price,
                        type=merged_type,
                        strength=merged_strength,
                        touches=total_touches,
                        created_at=earliest_creation,
                        last_tested=latest_test,
                        timeframe=current_group[0].timeframe,
                        sources=unique_sources
                    ))
                
                # Start a new group
                current_group = [current_level]
        
        # Process the last group
        if len(current_group) == 1:
            merged_levels.append(current_group[0])
        else:
            # Merge the last group (repeating the above logic)
            weights = [level.strength * level.touches for level in current_group]
            total_weight = sum(weights)
            merged_price = sum(level.price * weight for level, weight in 
                             zip(current_group, weights)) / total_weight
            
            support_weight = sum(weights[i] for i in range(len(current_group)) 
                               if current_group[i].type == 'support')
            resistance_weight = sum(weights[i] for i in range(len(current_group)) 
                                  if current_group[i].type == 'resistance')
            
            if support_weight > resistance_weight * 1.5:
                merged_type = 'support'
            elif resistance_weight > support_weight * 1.5:
                merged_type = 'resistance'
            else:
                merged_type = 'both'
            
            total_touches = sum(level.touches for level in current_group)
            earliest_creation = min(level.created_at for level in current_group)
            latest_test = max(level.last_tested for level in current_group)
            merged_strength = min(1.0, sum(level.strength for level in current_group) / len(current_group) * 1.2)
            
            all_sources = []
            for level in current_group:
                all_sources.extend(level.sources)
            unique_sources = list(set(all_sources))
            
            merged_levels.append(SupportResistanceLevel(
                price=merged_price,
                type=merged_type,
                strength=merged_strength,
                touches=total_touches,
                created_at=earliest_creation,
                last_tested=latest_test,
                timeframe=current_group[0].timeframe,
                sources=unique_sources
            ))
        
        return merged_levels

    def _determine_market_phase(self, df: pd.DataFrame, 
                               swing_points: List[SwingPoint], 
                               timeframe: str) -> List[MarketPhase]:
        """
        Determine the market phase for each candle in the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            swing_points: List of swing points
            timeframe: Timeframe of the data
            
        Returns:
            List of MarketPhase enums
        """
        logger.debug(f"Determining market phase for {timeframe}")
        
        lookback = self.config['market_phase_lookback']
        phases = [MarketPhase.RANGING] * len(df)  # Default to ranging
        
        try:
            # Calculate some basic indicators for phase determination
            df_temp = df.copy()
            df_temp['sma20'] = ta.sma(df['close'], length=20)
            df_temp['sma50'] = ta.sma(df['close'], length=50)
            df_temp['sma200'] = ta.sma(df['close'], length=200)
            df_temp['atr'] = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14)
            
            # Calculate trend direction and strength
            df_temp['trend_direction'] = np.where(df_temp['sma20'] > df_temp['sma50'], 1, 
                                                np.where(df_temp['sma20'] < df_temp['sma50'], -1, 0))
            
            # Prepare to analyze market phase patterns
            for i in range(lookback, len(df)):
                # Extract recent data
                window = df_temp.iloc[i-lookback:i+1]
                recent_close = df_temp['close'].iloc[i]
                
                # Calculate recent high/low
                recent_high = window['high'].max()
                recent_low = window['low'].min()
                
                # Calculate volatility metrics
                volatility = window['atr'].iloc[-1] / recent_close
                price_range_pct = (recent_high - recent_low) / recent_close
                
                # Calculate trend metrics
                trend_direction = window['trend_direction'].iloc[-1]
                trend_consistency = window['trend_direction'].abs().sum() / lookback
                
                # Get recent swing points in this window
                recent_swings = [p for p in swing_points if i-lookback <= p.index <= i]
                
                # Classify the market phase
                if trend_direction > 0 and trend_consistency > 0.7:
                    # Strong uptrend
                    if len(recent_swings) >= 3 and all(s.type == 'high' for s in recent_swings[-2:]):
                        # Potential distribution if making higher highs but with weakening momentum
                        if recent_swings[-1].strength < recent_swings[-3].strength * 0.8:
                            phases[i] = MarketPhase.DISTRIBUTION
                        else:
                            phases[i] = MarketPhase.UPTREND
                    else:
                        phases[i] = MarketPhase.UPTREND
                
                elif trend_direction < 0 and trend_consistency > 0.7:
                    # Strong downtrend
                    if len(recent_swings) >= 3 and all(s.type == 'low' for s in recent_swings[-2:]):
                        # Potential accumulation if making lower lows but with weakening momentum
                        if recent_swings[-1].strength < recent_swings[-3].strength * 0.8:
                            phases[i] = MarketPhase.ACCUMULATION
                        else:
                            phases[i] = MarketPhase.DOWNTREND
                    else:
                        phases[i] = MarketPhase.DOWNTREND
                
                elif price_range_pct < 0.02 and volatility < 0.01:
                    # Very tight range - potential accumulation or distribution
                    if trend_direction > 0:
                        phases[i] = MarketPhase.DISTRIBUTION
                    elif trend_direction < 0:
                        phases[i] = MarketPhase.ACCUMULATION
                    else:
                        phases[i] = MarketPhase.RANGING
                
                elif price_range_pct > 0.05 and volatility > 0.03:
                    # High volatility - potential breakout or reversal
                    if recent_close > recent_high * 0.98 and trend_direction >= 0:
                        phases[i] = MarketPhase.BREAKOUT_UP
                    elif recent_close < recent_low * 1.02 and trend_direction <= 0:
                        phases[i] = MarketPhase.BREAKOUT_DOWN
                    elif window['close'].iloc[-3] < window['close'].iloc[-2] < window['close'].iloc[-1] and trend_direction < 0:
                        phases[i] = MarketPhase.REVERSAL_UP
                    elif window['close'].iloc[-3] > window['close'].iloc[-2] > window['close'].iloc[-1] and trend_direction > 0:
                        phases[i] = MarketPhase.REVERSAL_DOWN
                    else:
                        phases[i] = MarketPhase.CHOPPY
                
                else:
                    # Default ranging market
                    phases[i] = MarketPhase.RANGING
            
            logger.debug(f"Market phase determination complete")
            return phases
            
        except Exception as e:
            logger.error(f"Error in market phase determination: {str(e)}")
            return [MarketPhase.RANGING] * len(df)  # Return default if error

    def _calculate_trend_strength(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate the strength of the current trend.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Array of trend strength values between -1 and 1
        """
        try:
            # Use multiple indicators to gauge trend strength
            # 1. Moving average directions
            df_temp = df.copy()
            df_temp['sma20'] = ta.sma(df['close'], length=20)
            df_temp['sma50'] = ta.sma(df['close'], length=50)
            df_temp['sma100'] = ta.sma(df['close'], length=100)
            
            # 2. ADX for trend strength
            df_temp['adx'] = ta.adx(high=df['high'], low=df['low'], close=df['close'], length=14)['ADX_14']
            
            # 3. MACD for momentum
            macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
            df_temp['macd'] = macd_df.iloc[:, 0]
            df_temp['macd_signal'] = macd_df.iloc[:, 2]
            
            # Calculate trend strength components
            ma_alignment = np.zeros(len(df))
            for i in range(100, len(df)):
                # Moving average alignment (-1 to 1)
                ma_slopes = [
                    (df_temp['sma20'].iloc[i] - df_temp['sma20'].iloc[i-5]) / df_temp['sma20'].iloc[i-5],
                    (df_temp['sma50'].iloc[i] - df_temp['sma50'].iloc[i-5]) / df_temp['sma50'].iloc[i-5],
                    (df_temp['sma100'].iloc[i] - df_temp['sma100'].iloc[i-5]) / df_temp['sma100'].iloc[i-5]
                ]
                
                # Check if slopes are aligned
                slopes_positive = all(slope > 0 for slope in ma_slopes)
                slopes_negative = all(slope < 0 for slope in ma_slopes)
                
                if slopes_positive:
                    ma_component = min(1.0, sum(ma_slopes) * 100)  # Scale appropriately
                elif slopes_negative:
                    ma_component = max(-1.0, sum(ma_slopes) * 100)  # Scale appropriately
                else:
                    # Mixed slope directions
                    ma_component = sum(ma_slopes) * 50  # Reduced impact
                
                # Normalize ADX to 0-1 range (ADX is 0-100)
                adx_norm = min(1.0, df_temp['adx'].iloc[i] / 50)
                
                # MACD momentum direction
                if i > 0:
                    macd_direction = 1 if (df_temp['macd'].iloc[i] > df_temp['macd'].iloc[i-1]) else -1
                else:
                    macd_direction = 0
                    
                # Combine components with appropriate weights
                ma_alignment[i] = (
                    ma_component * 0.5 +  # MA alignment
                    (adx_norm * np.sign(ma_component)) * 0.3 +  # ADX weighted by direction
                    macd_direction * 0.2  # MACD direction
                )
                
                # Ensure values are in [-1, 1] range
                ma_alignment[i] = max(-1.0, min(1.0, ma_alignment[i]))
            
            return ma_alignment
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            return np.zeros(len(df))

    def _find_closest_level(self, prices: np.ndarray, 
                           levels: np.ndarray, 
                           level_type: str) -> np.ndarray:
        """
        Find the closest support or resistance level for each price.
        
        Args:
            prices: Array of price values
            levels: Array of level prices
            level_type: 'support' or 'resistance'
            
        Returns:
            Array of closest level prices
        """
        result = np.zeros_like(prices)
        
        if len(levels) == 0:
            return result
            
        for i, price in enumerate(prices):
            if level_type == 'support':
                # Find closest level below price
                levels_below = levels[levels < price]
                if len(levels_below) > 0:
                    result[i] = levels_below.max()
            else:  # resistance
                # Find closest level above price
                levels_above = levels[levels > price]
                if len(levels_above) > 0:
                    result[i] = levels_above.min()
        
        return result

    def _calculate_sr_density(self, df: pd.DataFrame, 
                             sr_levels: List[SupportResistanceLevel]) -> np.ndarray:
        """
        Calculate the density of support/resistance levels around the current price.
        
        Args:
            df: DataFrame with OHLCV data
            sr_levels: List of support/resistance levels
            
        Returns:
            Array of density values
        """
        if not sr_levels:
            return np.zeros(len(df))
            
        densities = np.zeros(len(df))
        price_range = df['high'].max() - df['low'].min()
        proximity_threshold = price_range * 0.1  # 10% of price range
        
        for i, row in df.iterrows():
            idx = df.index.get_loc(i)
            price = row['close']
            
            # Count levels within threshold and weight by proximity and strength
            density = 0
            for level in sr_levels:
                distance = abs(level.price - price)
                if distance < proximity_threshold:
                    # Inverse distance weighting
                    proximity_weight = 1 - (distance / proximity_threshold)
                    density += level.strength * proximity_weight
            
            densities[idx] = min(1.0, density / 3)  # Normalize to [0, 1]
        
        return densities

    def _identify_order_blocks(self, df: pd.DataFrame, timeframe: str) -> np.ndarray:
        """
        Identify potential order blocks based on price action.
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe of the data
            
        Returns:
            Array with order block identification (0=none, 1=bullish, -1=bearish)
        """
        lookback = self.config['order_blocks_lookback']
        result = np.zeros(len(df))
        
        try:
            for i in range(lookback, len(df) - 1):
                # Look for strong momentum candles
                current_candle = df.iloc[i]
                prev_candle = df.iloc[i-1]
                
                candle_range = current_candle['high'] - current_candle['low']
                avg_range = df['high'].iloc[i-lookback:i].mean() - df['low'].iloc[i-lookback:i].mean()
                
                # Identify bullish order blocks (high volume down candle followed by strong up move)
                if (current_candle['close'] < current_candle['open'] and  # Bearish candle
                    current_candle['volume'] > df['volume'].iloc[i-lookback:i].mean() * 1.5 and  # High volume
                    df.iloc[i+1]['close'] > df.iloc[i+1]['open'] and  # Next candle bullish
                    df.iloc[i+1]['close'] > current_candle['high']):  # Strong movement up
                    
                    result[i] = -1  # Bullish order block (previous bearish candle)
                
                # Identify bearish order blocks (high volume up candle followed by strong down move)
                elif (current_candle['close'] > current_candle['open'] and  # Bullish candle
                      current_candle['volume'] > df['volume'].iloc[i-lookback:i].mean() * 1.5 and  # High volume
                      df.iloc[i+1]['close'] < df.iloc[i+1]['open'] and  # Next candle bearish
                      df.iloc[i+1]['close'] < current_candle['low']):  # Strong movement down
                    
                    result[i] = 1  # Bearish order block (previous bullish candle)
            
            return result
            
        except Exception as e:
            logger.error(f"Error identifying order blocks: {str(e)}")
            return np.zeros(len(df))

    def _identify_liquidity_levels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Identify potential liquidity levels based on stop clusters.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Array with liquidity level scores (higher values indicate more liquidity)
        """
        pct = self.config['liquidity_levels_pct']
        result = np.zeros(len(df))
        
        try:
            # Find swing points for potential stop locations
            highs = argrelextrema(df['high'].values, np.greater, order=5)[0]
            lows = argrelextrema(df['low'].values, np.less, order=5)[0]
            
            # Calculate typical stop placement zones
            liquidity_zones = []
            
            # Above recent highs - common stop placements
            for idx in highs:
                if idx < len(df) - 1:  # Ensure not the last point
                    level = df['high'].iloc[idx] * (1 + pct/100)
                    strength = 0.5 + 0.5 * (df['volume'].iloc[idx] / df['volume'].mean())
                    liquidity_zones.append((level, strength, idx))
            
            # Below recent lows - common stop placements
            for idx in lows:
                if idx < len(df) - 1:  # Ensure not the last point
                    level = df['low'].iloc[idx] * (1 - pct/100)
                    strength = 0.5 + 0.5 * (df['volume'].iloc[idx] / df['volume'].mean())
                    liquidity_zones.append((level, strength, idx))
            
            # Score each candle based on proximity to liquidity zones
            for i in range(len(df)):
                price = df['close'].iloc[i]
                
                # Calculate liquidity score based on proximity to zones
                liquidity_score = 0
                for level, strength, zone_idx in liquidity_zones:
                    # Only consider zones that formed before this candle
                    if zone_idx < i:
                        distance = abs(price - level)
                        max_distance = df['high'].max() * 0.01  # 1% of max price
                        
                        if distance < max_distance:
                            # Inverse distance weighting
                            proximity = 1 - (distance / max_distance)
                            # Decay strength based on time passed
                            time_decay = max(0.2, min(1.0, (1 - (i - zone_idx) / 20)))
                            
                            liquidity_score += strength * proximity * time_decay
                
                result[i] = min(1.0, liquidity_score / 5)  # Normalize to [0, 1]
            
            return result
            
        except Exception as e:
            logger.error(f"Error identifying liquidity levels: {str(e)}")
            return np.zeros(len(df))

    def _identify_smart_money_levels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Identify potential smart money levels based on price action and volume.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Array with smart money level scores
        """
        threshold = self.config['smart_money_threshold']
        result = np.zeros(len(df))
        
        try:
            # Calculate VWAP and standard deviations
            df_temp = df.copy()
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            cumulative_tp_vol = (typical_price * df['volume']).cumsum()
            cumulative_vol = df['volume'].cumsum()
            
            # Avoid division by zero
            df_temp['vwap'] = np.where(cumulative_vol > 0, 
                                      cumulative_tp_vol / cumulative_vol, 
                                      df['close'])
            
            # Calculate standard deviation of price from VWAP
            df_temp['price_dev'] = np.abs(df['close'] - df_temp['vwap'])
            df_temp['dev_mean'] = df_temp['price_dev'].rolling(window=20).mean()
            df_temp['dev_std'] = df_temp['price_dev'].rolling(window=20).std()
            
            # Calculate VWAP bands
            std_multiplier = self.config['vwap_std_multiplier']
            df_temp['vwap_upper'] = df_temp['vwap'] + df_temp['dev_std'] * std_multiplier
            df_temp['vwap_lower'] = df_temp['vwap'] - df_temp['dev_std'] * std_multiplier
            
            # Identify potential smart money action
            for i in range(20, len(df)):
                current_price = df['close'].iloc[i]
                
                # Test if price is near VWAP bands with exceptional volume
                near_upper = current_price > df_temp['vwap_upper'].iloc[i] * 0.98
                near_lower = current_price < df_temp['vwap_lower'].iloc[i] * 1.02
                high_volume = df['volume'].iloc[i] > df['volume'].iloc[i-20:i].mean() * 1.5
                
                if (near_upper or near_lower) and high_volume:
                    # Calculate strength based on volume and price action
                    vol_ratio = min(3.0, df['volume'].iloc[i] / df['volume'].iloc[i-20:i].mean())
                    price_action = abs(df['close'].iloc[i] - df['open'].iloc[i]) / \
                                  (df['high'].iloc[i] - df['low'].iloc[i])
                    
                    # Higher score if candle closes strong in expected direction
                    if near_upper and df['close'].iloc[i] < df['open'].iloc[i]:
                        direction_alignment = 1.2  # Bearish at resistance
                    elif near_lower and df['close'].iloc[i] > df['open'].iloc[i]:
                        direction_alignment = 1.2  # Bullish at support
                    else:
                        direction_alignment = 0.8
                    
                    smart_money_score = min(1.0, (vol_ratio * 0.5 + price_action * 0.5) * direction_alignment)
                    
                    # Only register significant scores
                    if smart_money_score > threshold:
                        result[i] = smart_money_score
            
            return result
            
        except Exception as e:
            logger.error(f"Error identifying smart money levels: {str(e)}")
            return np.zeros(len(df))

    def get_recent_structure(self, timeframe: str) -> Dict[str, Any]:
        """
        Get the most recent market structure information for a specific timeframe.
        
        Args:
            timeframe: The timeframe to retrieve information for
            
        Returns:
            Dictionary with recent structure information
        """
        if timeframe not in self.market_phases:
            return {
                'phase': MarketPhase.RANGING,
                'swing_points': [],
                'sr_levels': [],
                'time': None
            }
        
        return {
            'phase': self.market_phases[timeframe],
            'swing_points': self.swing_points.get(timeframe, [])[-5:],
            'sr_levels': self.sr_levels.get(timeframe, []),
            'time': pd.Timestamp.now()
        }

    def is_near_key_level(self, price: float, timeframe: str, 
                         threshold_pct: float = 0.5) -> Tuple[bool, str, float]:
        """
        Check if a price is near a key support/resistance level.
        
        Args:
            price: The price to check
            timeframe: The timeframe to use for levels
            threshold_pct: Proximity threshold as percentage of price
            
        Returns:
            Tuple of (is_near_level, level_type, distance_percentage)
        """
        if timeframe not in self.sr_levels or not self.sr_levels[timeframe]:
            return False, "", 0.0
        
        threshold = price * threshold_pct / 100
        
        # Check all levels
        closest_level = None
        closest_distance = float('inf')
        closest_type = ""
        
        for level in self.sr_levels[timeframe]:
            distance = abs(level.price - price)
            if distance < threshold and distance < closest_distance:
                closest_distance = distance
                closest_level = level
                closest_type = level.type
        
        if closest_level:
            return True, closest_type, (closest_distance / price) * 100
        else:
            return False, "", 0.0

# Add MarketStructureFeatures class (plural form) as an alias for MarketStructureFeature
MarketStructureFeatures = MarketStructureFeature


def identify_swing_points(
    df: pd.DataFrame, timeframe: str, lookback: int = 20
) -> List[SwingPoint]:
    """Convenience wrapper to identify swing points."""
    feature = MarketStructureFeature({'swing_lookback': lookback})
    return feature._identify_swing_points(df, timeframe)


def find_support_resistance_levels(
    df: pd.DataFrame, window_size: int = 20, order: int = 5
) -> Tuple[List[float], List[float]]:
    """Return support and resistance levels for the given data."""
    swings = find_swing_points(df, window_size, order)
    structure = identify_market_structure(df, swings)
    return structure.get('support_levels', []), structure.get('resistance_levels', [])


def classify_market_structure(df: pd.DataFrame, window_size: int = 20, order: int = 5) -> str:
    """Return a simple market structure classification (trend)."""
    swings = find_swing_points(df, window_size, order)
    structure = identify_market_structure(df, swings)
    return structure.get('trend', 'neutral')


def detect_consolidation(
    df: pd.DataFrame,
    min_periods: int = 5,
    max_periods: int = 50,
    threshold: float = 0.1,
) -> pd.Series:
    """Return a boolean Series indicating consolidation periods."""
    if df.empty or min_periods <= 0:
        return pd.Series(False, index=df.index)

    range_pct = (
        df['high'].rolling(min_periods).max() - df['low'].rolling(min_periods).min()
    ) / df['close']
    is_consolidating = range_pct < threshold

    if max_periods > min_periods:
        for w in range(min_periods + 1, max_periods + 1):
            range_pct = (
                df['high'].rolling(w).max() - df['low'].rolling(w).min()
            ) / df['close']
            is_consolidating &= range_pct < threshold

    return is_consolidating.fillna(False)


__all__ = [
    'MarketStructureFeature',
    'MarketStructureFeatures',
    'SwingPoint',
    'identify_swing_points',
    'find_support_resistance_levels',
    'classify_market_structure',
    'detect_consolidation',
]
