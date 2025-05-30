#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Support and Resistance Pattern Recognition Module

This module provides advanced algorithms for detecting, analyzing, and 
predicting support and resistance levels across multiple timeframes with
exceptional accuracy.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import scipy.signal as signal
from sklearn.cluster import DBSCAN, MeanShift
from scipy.stats import linregress
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

# Internal imports
from common.utils import memoize, threaded_calculation
from common.logger import get_logger
from common.constants import (
    TIMEFRAMES, SUPPORT_RESISTANCE_METHODS, 
    ZONE_CONFIDENCE_LEVELS, ZONE_TYPES
)
from feature_service.features.market_structure import MarketStructure
from feature_service.features.volume import VolumeProfiler
from data_storage.time_series import TimeSeriesStorage, TimeSeriesStore

# Import intelligence module to avoid name error
import intelligence

# Initialize logger
logger = get_logger("SupportResistance")


@dataclass
class ZoneLevel:
    """Data class for support/resistance zone levels with metadata."""
    price: float
    zone_type: str  # 'support' or 'resistance'
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    touches: int  # Number of times price has interacted with this level
    created_at: datetime
    last_tested: datetime
    volume_profile: Dict[str, float]  # Volume profile at this level
    timeframe: str  # Timeframe this level was detected on
    method: str  # Method used to detect this level
    source_points: List[Tuple[datetime, float]]  # Original points used to identify this zone
    fractals: List[Tuple[datetime, float]]  # Fractal points within this zone
    historical_respect: float  # Historical probability of respecting this level
    age_factor: float  # Decay factor based on age
    parent_zone: Optional[int] = None  # Reference to parent zone in higher timeframe
    zone_id: Optional[int] = None  # Unique identifier for this zone
    context: Dict[str, Any] = None  # Additional contextual information


class AdvancedSupportResistance:
    """
    Advanced support and resistance detection using multiple methods with
    multi-timeframe analysis and probabilistic forecasting.
    """
    
    def __init__(
        self, 
        time_series_store: TimeSeriesStore,
        market_structure: MarketStructure,
        volume_profiler: VolumeProfiler,
        use_gpu: bool = True,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the support and resistance detector.
        
        Args:
            time_series_store: Time series database access
            market_structure: Market structure analysis component
            volume_profiler: Volume profiling component
            use_gpu: Whether to use GPU acceleration for computations
            config: Configuration parameters
        """
        self.ts_store = time_series_store
        self.market_structure = market_structure
        self.volume_profiler = volume_profiler
        self.use_gpu = use_gpu
        
        # Default configuration
        self.config = {
            'peak_prominence_factor': 0.3,
            'zone_merge_threshold': 0.0015,  # 0.15% of price
            'zone_strength_recency_factor': 0.8,
            'zone_strength_touch_factor': 0.6,
            'zone_strength_volume_factor': 0.7,
            'fractal_lookback': 5,
            'min_zone_touches': 2,
            'strength_decay_factor': 0.95,
            'price_action_confirmation_lookback': 150,
            'cluster_eps': 0.002,  # For DBSCAN clustering
            'cluster_min_samples': 3,
            'historical_test_periods': 10,
            'multi_timeframe_weights': {
                '1m': 0.5,
                '5m': 0.6,
                '15m': 0.7,
                '1h': 0.8,
                '4h': 0.9,
                '1d': 1.0,
            },
            'validation_methods': ['price_action', 'volume', 'time'],
            'zone_confidence_threshold': 0.65,
            'active_zone_lookback': 100,  # Candles to look back for active zones
            'extension_projection_periods': 50,  # Periods to project extensions forward
            'moving_average_periods': [50, 200],  # MAs to consider for S/R
        }
        
        # Override with user configuration if provided
        if config:
            self.config.update(config)
        
        # Initialize zone storage
        self.zones = {tf: [] for tf in TIMEFRAMES}
        self.active_zones = {tf: [] for tf in TIMEFRAMES}
        
        # Initialize execution pools
        self.thread_executor = ThreadPoolExecutor(max_workers=12)
        
        logger.info("Advanced Support Resistance detector initialized")
    
    def detect_all(
        self, 
        symbol: str, 
        timeframes: List[str] = None,
        methods: List[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        force_recalculation: bool = False
    ) -> Dict[str, List[ZoneLevel]]:
        """
        Detect support and resistance levels using all available methods.
        
        Args:
            symbol: Trading symbol to analyze
            timeframes: List of timeframes to analyze
            methods: List of detection methods to use
            start_time: Start time for historical data
            end_time: End time for historical data
            force_recalculation: Force recalculation even if cached results exist
            
        Returns:
            Dictionary of support/resistance zones by timeframe
        """
        if timeframes is None:
            timeframes = list(self.config['multi_timeframe_weights'].keys())
        
        if methods is None:
            methods = SUPPORT_RESISTANCE_METHODS
        
        logger.info(f"Detecting support/resistance for {symbol} across {timeframes} timeframes")
        
        results = {}
        all_zones = []
        
        # Process each timeframe
        for tf in timeframes:
            try:
                # Fetch required data
                ohlcv_data = self.ts_store.get_ohlcv_data(
                    symbol=symbol,
                    timeframe=tf,
                    start_time=start_time,
                    end_time=end_time,
                    limit=2000  # Large enough for accurate detection
                )
                
                if ohlcv_data.empty:
                    logger.warning(f"No data available for {symbol} on {tf} timeframe")
                    continue
                
                # Apply all detection methods in parallel
                method_futures = {}
                for method in methods:
                    method_futures[method] = self.thread_executor.submit(
                        self._apply_detection_method,
                        method=method,
                        ohlcv_data=ohlcv_data,
                        timeframe=tf,
                        symbol=symbol
                    )
                
                # Collect zones from all methods
                tf_zones = []
                for method, future in method_futures.items():
                    try:
                        method_zones = future.result()
                        tf_zones.extend(method_zones)
                        logger.debug(f"Detected {len(method_zones)} zones using {method} on {tf}")
                    except Exception as e:
                        logger.error(f"Error detecting zones with {method}: {str(e)}")
                
                # Merge overlapping zones
                merged_zones = self._merge_overlapping_zones(tf_zones)
                
                # Calculate zone strength and confidence
                analyzed_zones = self._analyze_zone_quality(merged_zones, ohlcv_data, tf)
                
                # Store and return results
                self.zones[tf] = analyzed_zones
                results[tf] = analyzed_zones
                all_zones.extend(analyzed_zones)
                
                logger.info(f"Detected {len(analyzed_zones)} S/R zones on {tf} timeframe for {symbol}")
                
            except Exception as e:
                logger.error(f"Error detecting S/R for {symbol} on {tf}: {str(e)}")
        
        # Perform multi-timeframe zone correlation
        self._correlate_multi_timeframe_zones(all_zones)
        
        return results
    
    def _apply_detection_method(
        self, 
        method: str, 
        ohlcv_data: pd.DataFrame,
        timeframe: str,
        symbol: str
    ) -> List[ZoneLevel]:
        """Apply a specific detection method to identify support/resistance zones."""
        method_map = {
            'swing_high_low': self._detect_swing_points,
            'volume_profile': self._detect_volume_zones,
            'fibonacci': self._detect_fibonacci_levels,
            'pivot_points': self._detect_pivot_points,
            'price_clusters': self._detect_price_clusters,
            'fractals': self._detect_fractal_levels,
            'moving_averages': self._detect_moving_average_levels,
            'order_blocks': self._detect_order_blocks,
            'vwap_bands': self._detect_vwap_zones,
            'liquidity_pools': self._detect_liquidity_pools,
            'market_structure': self._detect_structure_levels,
            'psychological': self._detect_psychological_levels,
        }
        
        if method not in method_map:
            logger.warning(f"Unknown detection method: {method}")
            return []
        
        try:
            return method_map[method](ohlcv_data, timeframe, symbol)
        except Exception as e:
            logger.error(f"Error in {method} detection: {str(e)}")
            return []
    
    def _detect_swing_points(
        self, 
        ohlcv_data: pd.DataFrame,
        timeframe: str,
        symbol: str
    ) -> List[ZoneLevel]:
        """
        Detect support and resistance based on swing high and low points.
        Uses adaptive prominence thresholds based on volatility.
        """
        logger.debug("Detecting swing points")
        zones = []
        
        # Calculate adaptive prominence threshold based on volatility
        volatility = ohlcv_data['high'].rolling(20).max() - ohlcv_data['low'].rolling(20).min()
        mean_volatility = volatility.mean()
        prominence = mean_volatility * self.config['peak_prominence_factor']
        
        # Detect peaks in high prices (resistance)
        peak_indices, _ = signal.find_peaks(
            ohlcv_data['high'].values, 
            prominence=prominence,
            distance=self.config['fractal_lookback']
        )
        
        # Detect valleys in low prices (support)
        valley_indices, _ = signal.find_peaks(
            -ohlcv_data['low'].values,
            prominence=prominence,
            distance=self.config['fractal_lookback']
        )
        
        # Create resistance zones from peaks
        for idx in peak_indices:
            if idx >= len(ohlcv_data):
                continue
                
            price = ohlcv_data.iloc[idx]['high']
            timestamp = ohlcv_data.index[idx]
            
            # Create source points (surrounding candles)
            source_points = []
            for i in range(max(0, idx-2), min(len(ohlcv_data), idx+3)):
                source_points.append((ohlcv_data.index[i], ohlcv_data.iloc[i]['high']))
            
            zone = ZoneLevel(
                price=price,
                zone_type='resistance',
                strength=0.0,  # Will be calculated later
                confidence=0.0,  # Will be calculated later
                touches=1,  # Initial identification
                created_at=timestamp,
                last_tested=timestamp,
                volume_profile={},  # Will be populated later
                timeframe=timeframe,
                method='swing_high_low',
                source_points=source_points,
                fractals=[],
                historical_respect=0.0,
                age_factor=1.0,
                context={'initial_detection': 'peak'}
            )
            zones.append(zone)
        
        # Create support zones from valleys
        for idx in valley_indices:
            if idx >= len(ohlcv_data):
                continue
                
            price = ohlcv_data.iloc[idx]['low']
            timestamp = ohlcv_data.index[idx]
            
            # Create source points (surrounding candles)
            source_points = []
            for i in range(max(0, idx-2), min(len(ohlcv_data), idx+3)):
                source_points.append((ohlcv_data.index[i], ohlcv_data.iloc[i]['low']))
            
            zone = ZoneLevel(
                price=price,
                zone_type='support',
                strength=0.0,  # Will be calculated later
                confidence=0.0,  # Will be calculated later
                touches=1,  # Initial identification
                created_at=timestamp,
                last_tested=timestamp,
                volume_profile={},  # Will be populated later
                timeframe=timeframe,
                method='swing_high_low',
                source_points=source_points,
                fractals=[],
                historical_respect=0.0,
                age_factor=1.0,
                context={'initial_detection': 'valley'}
            )
            zones.append(zone)
        
        logger.debug(f"Detected {len(zones)} swing point zones")
        return zones
    
    def _detect_volume_zones(
        self, 
        ohlcv_data: pd.DataFrame,
        timeframe: str,
        symbol: str
    ) -> List[ZoneLevel]:
        """
        Detect support and resistance based on volume profile analysis.
        Identifies high volume nodes and low volume nodes.
        """
        logger.debug("Detecting volume-based zones")
        zones = []
        
        # Get volume profile data
        volume_nodes = self.volume_profiler.calculate_volume_profile(
            ohlcv_data,
            num_bins=50,
            percentile_threshold=0.85
        )
        
        if not volume_nodes:
            return zones
        
        # Process high volume nodes (potential S/R)
        for node_data in volume_nodes['high_volume_nodes']:
            price = node_data['price']
            volume = node_data['volume']
            pct_of_total = node_data['percent_of_total']
            
            # Determine if it's more likely to be support or resistance based on price action
            recent_data = ohlcv_data.iloc[-100:]
            if price > recent_data['close'].iloc[-1]:
                zone_type = 'resistance'
            else:
                zone_type = 'support'
            
            # Create zone with volume profile information
            zone = ZoneLevel(
                price=price,
                zone_type=zone_type,
                strength=pct_of_total,  # Initial strength based on volume concentration
                confidence=0.0,  # Will be calculated later
                touches=0,  # Will be calculated during analysis
                created_at=ohlcv_data.index[0],  # Use range start time
                last_tested=ohlcv_data.index[-1],  # Use range end time
                volume_profile={
                    'node_volume': volume,
                    'percent_of_total': pct_of_total,
                    'deltaVolume': node_data.get('delta_volume', 0)
                },
                timeframe=timeframe,
                method='volume_profile',
                source_points=[],  # Will be populated during analysis
                fractals=[],
                historical_respect=0.0,
                age_factor=1.0,
                context={'node_type': 'high_volume'}
            )
            zones.append(zone)
        
        # Process low volume nodes (potential breakout areas)
        for node_data in volume_nodes['low_volume_nodes']:
            price = node_data['price']
            volume = node_data['volume']
            pct_of_total = node_data['percent_of_total']
            
            # Low volume nodes between high volume areas are potential breakout zones
            # Mark them as both potential support and resistance
            zone = ZoneLevel(
                price=price,
                zone_type='breakout_zone',  # Special type for low volume nodes
                strength=1.0 - pct_of_total,  # Higher strength for lower volume (paradoxically)
                confidence=0.0,  # Will be calculated later
                touches=0,  # Will be calculated during analysis
                created_at=ohlcv_data.index[0],
                last_tested=ohlcv_data.index[-1],
                volume_profile={
                    'node_volume': volume,
                    'percent_of_total': pct_of_total,
                    'deltaVolume': node_data.get('delta_volume', 0)
                },
                timeframe=timeframe,
                method='volume_profile',
                source_points=[],
                fractals=[],
                historical_respect=0.0,
                age_factor=1.0,
                context={'node_type': 'low_volume'}
            )
            zones.append(zone)
        
        logger.debug(f"Detected {len(zones)} volume-based zones")
        return zones
    
    def _detect_fibonacci_levels(
        self, 
        ohlcv_data: pd.DataFrame,
        timeframe: str,
        symbol: str
    ) -> List[ZoneLevel]:
        """
        Detect support and resistance based on Fibonacci retracement and extension levels
        from significant market moves.
        """
        logger.debug("Detecting Fibonacci levels")
        zones = []
        
        # Get significant market structures to use as anchor points
        structure_points = self.market_structure.get_significant_points(ohlcv_data)
        if not structure_points or len(structure_points) < 2:
            return zones
            
        # Fibonacci ratios for retracements and extensions
        fib_ratios = {
            'retracement': [0.236, 0.382, 0.5, 0.618, 0.786],
            'extension': [1.272, 1.414, 1.618, 2.0, 2.618]
        }
        
        # Find significant swing highs and lows
        significant_swings = []
        for i in range(len(structure_points) - 1):
            curr_point = structure_points[i]
            next_point = structure_points[i + 1]
            
            # Calculate swing size
            swing_size = abs(curr_point['price'] - next_point['price'])
            avg_price = (curr_point['price'] + next_point['price']) / 2
            relative_size = swing_size / avg_price
            
            # Consider it significant if relative size is substantial
            if relative_size > 0.02:  # 2% move
                significant_swings.append((curr_point, next_point))
        
        # Calculate Fibonacci levels for each significant swing
        for swing_start, swing_end in significant_swings:
            start_price = swing_start['price']
            end_price = swing_end['price']
            swing_size = end_price - start_price
            start_time = swing_start['time']
            end_time = swing_end['time']
            
            # Direction of the swing
            is_upswing = swing_size > 0
            
            # Calculate retracement levels
            for ratio in fib_ratios['retracement']:
                level_price = end_price - (swing_size * ratio)
                
                # Determine if this is likely support or resistance
                zone_type = 'support' if is_upswing else 'resistance'
                
                # Create source points from the swing
                source_points = [(start_time, start_price), (end_time, end_price)]
                
                zone = ZoneLevel(
                    price=level_price,
                    zone_type=zone_type,
                    strength=0.0,  # Will be calculated later
                    confidence=0.0,  # Will be calculated later
                    touches=0,  # Will be calculated during analysis
                    created_at=end_time,  # Fib levels are valid after the swing completes
                    last_tested=end_time,
                    volume_profile={},
                    timeframe=timeframe,
                    method='fibonacci',
                    source_points=source_points,
                    fractals=[],
                    historical_respect=0.0,
                    age_factor=1.0,
                    context={
                        'fib_type': 'retracement',
                        'ratio': ratio,
                        'swing_start': start_price,
                        'swing_end': end_price,
                        'is_upswing': is_upswing
                    }
                )
                zones.append(zone)
            
            # Calculate extension levels
            for ratio in fib_ratios['extension']:
                level_price = end_price + (swing_size * ratio)
                
                # Determine if this is likely support or resistance
                # Extensions of upswings are resistance, extensions of downswings are support
                zone_type = 'resistance' if is_upswing else 'support'
                
                # Create source points from the swing
                source_points = [(start_time, start_price), (end_time, end_price)]
                
                zone = ZoneLevel(
                    price=level_price,
                    zone_type=zone_type,
                    strength=0.0,  # Will be calculated later
                    confidence=0.0,  # Will be calculated later
                    touches=0,  # Will be calculated during analysis
                    created_at=end_time,
                    last_tested=end_time,
                    volume_profile={},
                    timeframe=timeframe,
                    method='fibonacci',
                    source_points=source_points,
                    fractals=[],
                    historical_respect=0.0,
                    age_factor=1.0,
                    context={
                        'fib_type': 'extension',
                        'ratio': ratio,
                        'swing_start': start_price,
                        'swing_end': end_price,
                        'is_upswing': is_upswing
                    }
                )
                zones.append(zone)
        
        logger.debug(f"Detected {len(zones)} Fibonacci levels")
        return zones
    
    def _detect_pivot_points(
        self, 
        ohlcv_data: pd.DataFrame,
        timeframe: str,
        symbol: str
    ) -> List[ZoneLevel]:
        """
        Detect support and resistance based on traditional and advanced pivot point calculations.
        """
        logger.debug("Detecting pivot points")
        zones = []
        
        # Function to calculate pivot points based on a reference period
        def calculate_period_pivots(period_data):
            if period_data.empty:
                return []
                
            # Get OHLC values
            high = period_data['high'].max()
            low = period_data['low'].min()
            close = period_data['close'].iloc[-1]
            
            # Calculate pivot point (PP)
            pp = (high + low + close) / 3
            
            # Calculate support levels
            s1 = (2 * pp) - high
            s2 = pp - (high - low)
            s3 = low - 2 * (high - pp)
            
            # Calculate resistance levels
            r1 = (2 * pp) - low
            r2 = pp + (high - low)
            r3 = high + 2 * (pp - low)
            
            period_start = period_data.index[0]
            period_end = period_data.index[-1]
            
            # Create pivot zones
            pivot_zones = []
            
            # Add pivot point
            pivot_zones.append(ZoneLevel(
                price=pp,
                zone_type='pivot',  # Can act as both support and resistance
                strength=0.8,  # Pivot points generally have good strength
                confidence=0.0,  # Will be calculated later
                touches=0,
                created_at=period_end,
                last_tested=period_end,
                volume_profile={},
                timeframe=timeframe,
                method='pivot_points',
                source_points=[(period_start, period_data['high'].iloc[0]), 
                              (period_end, period_data['close'].iloc[-1])],
                fractals=[],
                historical_respect=0.0,
                age_factor=1.0,
                context={'pivot_type': 'pp', 'period_start': period_start, 'period_end': period_end}
            ))
            
            # Add support levels
            for i, s in enumerate([s1, s2, s3], 1):
                pivot_zones.append(ZoneLevel(
                    price=s,
                    zone_type='support',
                    strength=0.9 - (i * 0.1),  # S1 stronger than S2, S2 stronger than S3
                    confidence=0.0,
                    touches=0,
                    created_at=period_end,
                    last_tested=period_end,
                    volume_profile={},
                    timeframe=timeframe,
                    method='pivot_points',
                    source_points=[(period_start, period_data['low'].iloc[0]), 
                                  (period_end, period_data['close'].iloc[-1])],
                    fractals=[],
                    historical_respect=0.0,
                    age_factor=1.0,
                    context={'pivot_type': f's{i}', 'period_start': period_start, 'period_end': period_end}
                ))
            
            # Add resistance levels
            for i, r in enumerate([r1, r2, r3], 1):
                pivot_zones.append(ZoneLevel(
                    price=r,
                    zone_type='resistance',
                    strength=0.9 - (i * 0.1),  # R1 stronger than R2, R2 stronger than R3
                    confidence=0.0,
                    touches=0,
                    created_at=period_end,
                    last_tested=period_end,
                    volume_profile={},
                    timeframe=timeframe,
                    method='pivot_points',
                    source_points=[(period_start, period_data['high'].iloc[0]), 
                                  (period_end, period_data['close'].iloc[-1])],
                    fractals=[],
                    historical_respect=0.0,
                    age_factor=1.0,
                    context={'pivot_type': f'r{i}', 'period_start': period_start, 'period_end': period_end}
                ))
            
            return pivot_zones
        
        # Determine periods based on timeframe
        if timeframe in ['1m', '5m', '15m', '30m', '1h']:
            # For intraday timeframes, calculate daily pivots
            daily_data = ohlcv_data.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Calculate pivots for the most recent days
            for i in range(min(5, len(daily_data))):
                day_data = daily_data.iloc[-(i+1):-(i) if i < len(daily_data)-1 else None]
                day_pivots = calculate_period_pivots(day_data)
                zones.extend(day_pivots)
                
        elif timeframe in ['4h', '6h', '8h', '12h', '1d']:
            # For daily timeframes, calculate weekly pivots
            weekly_data = ohlcv_data.resample('W').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Calculate pivots for the most recent weeks
            for i in range(min(3, len(weekly_data))):
                week_data = weekly_data.iloc[-(i+1):-(i) if i < len(weekly_data)-1 else None]
                week_pivots = calculate_period_pivots(week_data)
                zones.extend(week_pivots)
                
        elif timeframe in ['3d', '1w']:
            # For weekly timeframes, calculate monthly pivots
            monthly_data = ohlcv_data.resample('M').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Calculate pivots for the most recent months
            for i in range(min(2, len(monthly_data))):
                month_data = monthly_data.iloc[-(i+1):-(i) if i < len(monthly_data)-1 else None]
                month_pivots = calculate_period_pivots(month_data)
                zones.extend(month_pivots)
        
        logger.debug(f"Detected {len(zones)} pivot point levels")
        return zones
    
    def _detect_price_clusters(
        self, 
        ohlcv_data: pd.DataFrame,
        timeframe: str,
        symbol: str
    ) -> List[ZoneLevel]:
        """
        Detect support and resistance based on clustering of price action.
        Uses density-based clustering to identify areas of price concentration.
        """
        logger.debug("Detecting price clusters")
        zones = []
        
        # Extract price points for clustering
        price_points = np.column_stack([
            np.arange(len(ohlcv_data)),  # Use index as x-coordinate
            ohlcv_data['high'].values,   # High prices
            ohlcv_data['low'].values     # Low prices
        ])
        
        # Normalize price scale for effective clustering
        min_price = min(ohlcv_data['low'].min(), ohlcv_data['high'].min())
        max_price = max(ohlcv_data['low'].max(), ohlcv_data['high'].max())
        price_range = max_price - min_price
        
        # Skip clustering if price range is too small
        if price_range < 1e-5:
            return zones
        
        # Normalize prices to 0-1 range
        normalized_points = price_points.copy()
        normalized_points[:, 1] = (price_points[:, 1] - min_price) / price_range
        normalized_points[:, 2] = (price_points[:, 2] - min_price) / price_range
        
        # Use more weight on price dimension than time dimension
        weights = np.array([0.1, 1.0, 1.0])
        weighted_points = normalized_points * weights
        
        # Perform clustering on high prices
        high_clusters = self._cluster_price_points(weighted_points[:, [0, 1]])
        
        # Perform clustering on low prices
        low_clusters = self._cluster_price_points(weighted_points[:, [0, 2]])
        
        # Process high price clusters (potential resistance)
        for cluster_idx, cluster in enumerate(high_clusters):
            if not cluster or len(cluster) < self.config['cluster_min_samples']:
                continue
                
            # Get original indices
            cluster_indices = [int(weighted_points[i, 0] / weights[0]) for i in cluster]
            
            # Calculate cluster center (mean price)
            cluster_prices = [ohlcv_data.iloc[idx]['high'] for idx in cluster_indices]
            center_price = np.mean(cluster_prices)
            
            # Create source points
            source_points = [(ohlcv_data.index[idx], ohlcv_data.iloc[idx]['high']) 
                            for idx in cluster_indices]
            
            # Create resistance zone
            zone = ZoneLevel(
                price=center_price,
                zone_type='resistance',
                strength=0.0,  # Will be calculated later
                confidence=0.0,  # Will be calculated later
                touches=len(cluster),  # Number of points in cluster
                created_at=ohlcv_data.index[min(cluster_indices)],
                last_tested=ohlcv_data.index[max(cluster_indices)],
                volume_profile={},
                timeframe=timeframe,
                method='price_clusters',
                source_points=source_points,
                fractals=[],
                historical_respect=0.0,
                age_factor=1.0,
                context={'cluster_size': len(cluster), 'price_std': np.std(cluster_prices)}
            )
            zones.append(zone)
        
        # Process low price clusters (potential support)
        for cluster_idx, cluster in enumerate(low_clusters):
            if not cluster or len(cluster) < self.config['cluster_min_samples']:
                continue
                
            # Get original indices
            cluster_indices = [int(weighted_points[i, 0] / weights[0]) for i in cluster]
            
            # Calculate cluster center (mean price)
            cluster_prices = [ohlcv_data.iloc[idx]['low'] for idx in cluster_indices]
            center_price = np.mean(cluster_prices)
            
            # Create source points
            source_points = [(ohlcv_data.index[idx], ohlcv_data.iloc[idx]['low']) 
                            for idx in cluster_indices]
            
            # Create support zone
            zone = ZoneLevel(
                price=center_price,
                zone_type='support',
                strength=0.0,  # Will be calculated later
                confidence=0.0,  # Will be calculated later
                touches=len(cluster),  # Number of points in cluster
                created_at=ohlcv_data.index[min(cluster_indices)],
                last_tested=ohlcv_data.index[max(cluster_indices)],
                volume_profile={},
                timeframe=timeframe,
                method='price_clusters',
                source_points=source_points,
                fractals=[],
                historical_respect=0.0,
                age_factor=1.0,
                context={'cluster_size': len(cluster), 'price_std': np.std(cluster_prices)}
            )
            zones.append(zone)
        
        logger.debug(f"Detected {len(zones)} price cluster zones")
        return zones
    
    def _cluster_price_points(self, points: np.ndarray) -> List[List[int]]:
        """
        Perform density-based clustering on price points.
        
        Args:
            points: Array of points to cluster
            
        Returns:
            List of clusters, where each cluster is a list of point indices
        """
        if len(points) < self.config['cluster_min_samples']:
            return []
            
        try:
            # Use DBSCAN for clustering
            dbscan = DBSCAN(
                eps=self.config['cluster_eps'],
                min_samples=self.config['cluster_min_samples'],
                metric='euclidean'
            )
            
            # Fit and get labels
            cluster_labels = dbscan.fit_predict(points)
            
            # Group points by cluster label
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label != -1:  # Skip noise points
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(i)
            
            return list(clusters.values())
            
        except Exception as e:
            logger.error(f"Error in clustering: {str(e)}")
            return []
    
    def _detect_fractal_levels(
        self, 
        ohlcv_data: pd.DataFrame,
        timeframe: str,
        symbol: str
    ) -> List[ZoneLevel]:
        """
        Detect support and resistance based on fractal patterns.
        Uses Bill Williams' fractal indicator enhanced with confirmation rules.
        """
        logger.debug("Detecting fractal levels")
        zones = []
        
        lookback = self.config['fractal_lookback']
        if len(ohlcv_data) < 2 * lookback + 1:
            return zones
        
        # Lists to collect fractal points
        bull_fractals = []  # Bottom fractals (potential support)
        bear_fractals = []  # Top fractals (potential resistance)
        
        # Detect fractal patterns
        for i in range(lookback, len(ohlcv_data) - lookback):
            # Bull fractal (support)
            is_bull_fractal = True
            for j in range(1, lookback + 1):
                if ohlcv_data.iloc[i]['low'] >= ohlcv_data.iloc[i-j]['low'] or \
                   ohlcv_data.iloc[i]['low'] >= ohlcv_data.iloc[i+j]['low']:
                    is_bull_fractal = False
                    break
            
            if is_bull_fractal:
                bull_fractals.append(i)
            
            # Bear fractal (resistance)
            is_bear_fractal = True
            for j in range(1, lookback + 1):
                if ohlcv_data.iloc[i]['high'] <= ohlcv_data.iloc[i-j]['high'] or \
                   ohlcv_data.iloc[i]['high'] <= ohlcv_data.iloc[i+j]['high']:
                    is_bear_fractal = False
                    break
            
            if is_bear_fractal:
                bear_fractals.append(i)
        
        # Create resistance zones from bear fractals
        for idx in bear_fractals:
            price = ohlcv_data.iloc[idx]['high']
            timestamp = ohlcv_data.index[idx]
            
            # Create source points (surrounding candles)
            source_points = []
            for i in range(max(0, idx-lookback), min(len(ohlcv_data), idx+lookback+1)):
                source_points.append((ohlcv_data.index[i], ohlcv_data.iloc[i]['high']))
            
            # Check for confirmation - was this fractal tested and respected?
            confirmed = False
            confirmed_touches = 1
            last_tested = timestamp
            
            # Look ahead to see if price approached this level again
            for i in range(idx + lookback + 1, min(len(ohlcv_data), idx + 100)):
                curr_high = ohlcv_data.iloc[i]['high']
                # If price approaches within 0.1% of the fractal high
                if abs(curr_high - price) / price < 0.001:
                    confirmed_touches += 1
                    last_tested = ohlcv_data.index[i]
                    # If price was rejected (didn't close above)
                    if ohlcv_data.iloc[i]['close'] < price:
                        confirmed = True
            
            zone = ZoneLevel(
                price=price,
                zone_type='resistance',
                strength=0.0,  # Will be calculated later
                confidence=0.0,  # Will be calculated later
                touches=confirmed_touches,
                created_at=timestamp,
                last_tested=last_tested,
                volume_profile={},
                timeframe=timeframe,
                method='fractals',
                source_points=source_points,
                fractals=[(timestamp, price)],
                historical_respect=0.0,
                age_factor=1.0,
                context={'confirmed': confirmed, 'fractal_type': 'bear'}
            )
            zones.append(zone)
        
        # Create support zones from bull fractals
        for idx in bull_fractals:
            price = ohlcv_data.iloc[idx]['low']
            timestamp = ohlcv_data.index[idx]
            
            # Create source points (surrounding candles)
            source_points = []
            for i in range(max(0, idx-lookback), min(len(ohlcv_data), idx+lookback+1)):
                source_points.append((ohlcv_data.index[i], ohlcv_data.iloc[i]['low']))
            
            # Check for confirmation - was this fractal tested and respected?
            confirmed = False
            confirmed_touches = 1
            last_tested = timestamp
            
            # Look ahead to see if price approached this level again
            for i in range(idx + lookback + 1, min(len(ohlcv_data), idx + 100)):
                curr_low = ohlcv_data.iloc[i]['low']
                # If price approaches within 0.1% of the fractal low
                if abs(curr_low - price) / price < 0.001:
                    confirmed_touches += 1
                    last_tested = ohlcv_data.index[i]
                    # If price was supported (didn't close below)
                    if ohlcv_data.iloc[i]['close'] > price:
                        confirmed = True
            
            zone = ZoneLevel(
                price=price,
                zone_type='support',
                strength=0.0,  # Will be calculated later
                confidence=0.0,  # Will be calculated later
                touches=confirmed_touches,
                created_at=timestamp,
                last_tested=last_tested,
                volume_profile={},
                timeframe=timeframe,
                method='fractals',
                source_points=source_points,
                fractals=[(timestamp, price)],
                historical_respect=0.0,
                age_factor=1.0,
                context={'confirmed': confirmed, 'fractal_type': 'bull'}
            )
            zones.append(zone)
        
        logger.debug(f"Detected {len(zones)} fractal levels")
        return zones
    
    def _detect_moving_average_levels(
        self, 
        ohlcv_data: pd.DataFrame,
        timeframe: str,
        symbol: str
    ) -> List[ZoneLevel]:
        """
        Detect support and resistance based on key moving averages.
        Uses several popular MAs and their interactions as potential S/R levels.
        """
        logger.debug("Detecting moving average levels")
        zones = []
        
        # Calculate common moving averages
        ma_periods = self.config['moving_average_periods']
        
        for period in ma_periods:
            if len(ohlcv_data) < period:
                continue
                
            # Calculate Simple Moving Average
            ohlcv_data[f'sma_{period}'] = ohlcv_data['close'].rolling(window=period).mean()
            
            # Calculate Exponential Moving Average
            ohlcv_data[f'ema_{period}'] = ohlcv_data['close'].ewm(span=period, adjust=False).mean()
        
        # Skip if not enough data after calculations
        if ohlcv_data.iloc[-1].isnull().any():
            return zones
        
        # Get current price and most recent MAs
        current_price = ohlcv_data['close'].iloc[-1]
        latest_data = ohlcv_data.iloc[-1]
        
        # Create zones from current MA values
        for period in ma_periods:
            sma_value = latest_data[f'sma_{period}']
            ema_value = latest_data[f'ema_{period}']
            
            # Determine if MA is likely support or resistance based on current price
            sma_type = 'resistance' if sma_value > current_price else 'support'
            ema_type = 'resistance' if ema_value > current_price else 'support'
            
            # Calculate how often price has respected this MA
            sma_touches = 0
            sma_respect_count = 0
            ema_touches = 0
            ema_respect_count = 0
            
            # Look back to analyze MA interactions with price
            lookback_period = min(500, len(ohlcv_data) - period)
            for i in range(len(ohlcv_data) - lookback_period, len(ohlcv_data)):
                # Check SMA interactions
                curr_close = ohlcv_data.iloc[i]['close']
                prev_close = ohlcv_data.iloc[i-1]['close'] if i > 0 else curr_close
                curr_sma = ohlcv_data.iloc[i][f'sma_{period}']
                
                # Price crossed SMA
                if (prev_close < curr_sma and curr_close > curr_sma) or \
                   (prev_close > curr_sma and curr_close < curr_sma):
                    sma_touches += 1
                
                # Price approached SMA (within 0.2%) but didn't cross
                elif abs(curr_close - curr_sma) / curr_sma < 0.002 and \
                     ((prev_close < curr_sma and curr_close < curr_sma) or \
                      (prev_close > curr_sma and curr_close > curr_sma)):
                    sma_touches += 1
                    sma_respect_count += 1
                
                # Check EMA interactions similarly
                curr_ema = ohlcv_data.iloc[i][f'ema_{period}']
                
                # Price crossed EMA
                if (prev_close < curr_ema and curr_close > curr_ema) or \
                   (prev_close > curr_ema and curr_close < curr_ema):
                    ema_touches += 1
                
                # Price approached EMA but didn't cross
                elif abs(curr_close - curr_ema) / curr_ema < 0.002 and \
                     ((prev_close < curr_ema and curr_close < curr_ema) or \
                      (prev_close > curr_ema and curr_close > curr_ema)):
                    ema_touches += 1
                    ema_respect_count += 1
            
            # Calculate historical respect ratios
            sma_respect_ratio = sma_respect_count / max(1, sma_touches)
            ema_respect_ratio = ema_respect_count / max(1, ema_touches)
            
            # Create SMA zone
            zone_sma = ZoneLevel(
                price=sma_value,
                zone_type=sma_type,
                strength=0.0,  # Will be calculated later
                confidence=0.0,  # Will be calculated later
                touches=sma_touches,
                created_at=ohlcv_data.index[-period],
                last_tested=ohlcv_data.index[-1],
                volume_profile={},
                timeframe=timeframe,
                method='moving_averages',
                source_points=[],
                fractals=[],
                historical_respect=sma_respect_ratio,
                age_factor=1.0,
                context={'ma_type': 'sma', 'period': period}
            )
            zones.append(zone_sma)
            
            # Create EMA zone
            zone_ema = ZoneLevel(
                price=ema_value,
                zone_type=ema_type,
                strength=0.0,  # Will be calculated later
                confidence=0.0,  # Will be calculated later
                touches=ema_touches,
                created_at=ohlcv_data.index[-period],
                last_tested=ohlcv_data.index[-1],
                volume_profile={},
                timeframe=timeframe,
                method='moving_averages',
                source_points=[],
                fractals=[],
                historical_respect=ema_respect_ratio,
                age_factor=1.0,
                context={'ma_type': 'ema', 'period': period}
            )
            zones.append(zone_ema)
        
        logger.debug(f"Detected {len(zones)} moving average levels")
        return zones
    
    def _detect_order_blocks(
        self, 
        ohlcv_data: pd.DataFrame,
        timeframe: str,
        symbol: str
    ) -> List[ZoneLevel]:
        """
        Detect support and resistance based on order block theory.
        Identifies imbalance candles preceding strong moves as potential order blocks.
        """
        logger.debug("Detecting order blocks")
        zones = []
        
        if len(ohlcv_data) < 20:
            return zones
        
        # Function to identify strong moves
        def is_strong_move(candle_idx, direction='up', threshold_pct=0.5):
            if candle_idx >= len(ohlcv_data) - 1:
                return False
                
            curr_candle = ohlcv_data.iloc[candle_idx]
            next_candle = ohlcv_data.iloc[candle_idx + 1]
            
            if direction == 'up':
                body_size = next_candle['close'] - next_candle['open']
                candle_range = next_candle['high'] - next_candle['low']
                is_bullish = next_candle['close'] > next_candle['open']
                move_size_pct = (next_candle['high'] - curr_candle['high']) / curr_candle['high'] * 100
                return is_bullish and move_size_pct > threshold_pct and body_size > 0.6 * candle_range
            else:
                body_size = next_candle['open'] - next_candle['close']
                candle_range = next_candle['high'] - next_candle['low']
                is_bearish = next_candle['close'] < next_candle['open']
                move_size_pct = (curr_candle['low'] - next_candle['low']) / curr_candle['low'] * 100
                return is_bearish and move_size_pct > threshold_pct and body_size > 0.6 * candle_range
        
        # Scan for potential order blocks
        for i in range(1, len(ohlcv_data) - 3):
            # Check for bullish order block (potential support zone)
            if is_strong_move(i, direction='up'):
                # The candle before the strong move is the order block
                ob_candle = ohlcv_data.iloc[i]
                ob_price_low = ob_candle['low']
                ob_price_high = ob_candle['high']
                ob_price_mid = (ob_price_low + ob_price_high) / 2
                ob_timestamp = ohlcv_data.index[i]
                
                # Check if this order block has been respected later
                respected = False
                last_tested = ob_timestamp
                test_count = 0
                
                # Look for later price interactions with this zone
                for j in range(i + 2, min(len(ohlcv_data), i + 100)):
                    test_candle = ohlcv_data.iloc[j]
                    
                    # Price came down to the order block zone
                    if test_candle['low'] <= ob_price_high and test_candle['low'] >= ob_price_low:
                        test_count += 1
                        last_tested = ohlcv_data.index[j]
                        
                        # Check if price respected the zone (bounced up)
                        if j < len(ohlcv_data) - 1 and ohlcv_data.iloc[j+1]['close'] > test_candle['close']:
                            respected = True
                
                # Create source points
                source_points = [(ob_timestamp, ob_price_low), (ob_timestamp, ob_price_high)]
                
                zone = ZoneLevel(
                    price=ob_price_mid,  # Use middle of the order block
                    zone_type='support',
                    strength=0.0,  # Will be calculated later
                    confidence=0.0,  # Will be calculated later
                    touches=test_count,
                    created_at=ob_timestamp,
                    last_tested=last_tested,
                    volume_profile={'candle_volume': ob_candle['volume']},
                    timeframe=timeframe,
                    method='order_blocks',
                    source_points=source_points,
                    fractals=[],
                    historical_respect=1.0 if respected else 0.0,
                    age_factor=1.0,
                    context={
                        'respected': respected,
                        'ob_type': 'bullish',
                        'ob_low': ob_price_low,
                        'ob_high': ob_price_high,
                        'next_candle_move_pct': (ohlcv_data.iloc[i+1]['high'] - ob_candle['high']) / ob_candle['high'] * 100
                    }
                )
                zones.append(zone)
            
            # Check for bearish order block (potential resistance zone)
            if is_strong_move(i, direction='down'):
                # The candle before the strong move is the order block
                ob_candle = ohlcv_data.iloc[i]
                ob_price_low = ob_candle['low']
                ob_price_high = ob_candle['high']
                ob_price_mid = (ob_price_low + ob_price_high) / 2
                ob_timestamp = ohlcv_data.index[i]
                
                # Check if this order block has been respected later
                respected = False
                last_tested = ob_timestamp
                test_count = 0
                
                # Look for later price interactions with this zone
                for j in range(i + 2, min(len(ohlcv_data), i + 100)):
                    test_candle = ohlcv_data.iloc[j]
                    
                    # Price came up to the order block zone
                    if test_candle['high'] >= ob_price_low and test_candle['high'] <= ob_price_high:
                        test_count += 1
                        last_tested = ohlcv_data.index[j]
                        
                        # Check if price respected the zone (dropped down)
                        if j < len(ohlcv_data) - 1 and ohlcv_data.iloc[j+1]['close'] < test_candle['close']:
                            respected = True
                
                # Create source points
                source_points = [(ob_timestamp, ob_price_low), (ob_timestamp, ob_price_high)]
                
                zone = ZoneLevel(
                    price=ob_price_mid,  # Use middle of the order block
                    zone_type='resistance',
                    strength=0.0,  # Will be calculated later
                    confidence=0.0,  # Will be calculated later
                    touches=test_count,
                    created_at=ob_timestamp,
                    last_tested=last_tested,
                    volume_profile={'candle_volume': ob_candle['volume']},
                    timeframe=timeframe,
                    method='order_blocks',
                    source_points=source_points,
                    fractals=[],
                    historical_respect=1.0 if respected else 0.0,
                    age_factor=1.0,
                    context={
                        'respected': respected,
                        'ob_type': 'bearish',
                        'ob_low': ob_price_low,
                        'ob_high': ob_price_high,
                        'next_candle_move_pct': (ob_candle['low'] - ohlcv_data.iloc[i+1]['low']) / ob_candle['low'] * 100
                    }
                )
                zones.append(zone)
        
        logger.debug(f"Detected {len(zones)} order block levels")
        return zones
    
    def _detect_vwap_zones(
        self, 
        ohlcv_data: pd.DataFrame,
        timeframe: str,
        symbol: str
    ) -> List[ZoneLevel]:
        """
        Detect support and resistance based on VWAP and its bands.
        Creates S/R levels based on standard deviation bands of VWAP.
        """
        logger.debug("Detecting VWAP zones")
        zones = []
        
        if len(ohlcv_data) < 20:
            return zones
        
        # Calculate VWAP
        typical_price = (ohlcv_data['high'] + ohlcv_data['low'] + ohlcv_data['close']) / 3
        vprice = typical_price * ohlcv_data['volume']
        cumulative_vprice = vprice.cumsum()
        cumulative_volume = ohlcv_data['volume'].cumsum()
        vwap = cumulative_vprice / cumulative_volume
        
        # Calculate VWAP standard deviation
        sq_diff = (typical_price - vwap) ** 2
        cumulative_sq_diff = (sq_diff * ohlcv_data['volume']).cumsum()
        vwap_std = np.sqrt(cumulative_sq_diff / cumulative_volume)
        
        # Calculate VWAP bands
        upper_band1 = vwap + 1 * vwap_std
        upper_band2 = vwap + 2 * vwap_std
        lower_band1 = vwap - 1 * vwap_std
        lower_band2 = vwap - 2 * vwap_std
        
        # Get latest values
        latest_vwap = vwap.iloc[-1]
        latest_upper1 = upper_band1.iloc[-1]
        latest_upper2 = upper_band2.iloc[-1]
        latest_lower1 = lower_band1.iloc[-1]
        latest_lower2 = lower_band2.iloc[-1]
        latest_timestamp = ohlcv_data.index[-1]
        
        # Create VWAP level (can be both support and resistance)
        vwap_zone = ZoneLevel(
            price=latest_vwap,
            zone_type='pivot',  # Can act as both
            strength=0.0,  # Will be calculated later
            confidence=0.0,  # Will be calculated later
            touches=0,  # Will be calculated
            created_at=ohlcv_data.index[0],
            last_tested=latest_timestamp,
            volume_profile={},
            timeframe=timeframe,
            method='vwap_bands',
            source_points=[],
            fractals=[],
            historical_respect=0.0,
            age_factor=1.0,
            context={'band_type': 'vwap_base'}
        )
        zones.append(vwap_zone)
        
        # Create upper band levels (resistance)
        upper1_zone = ZoneLevel(
            price=latest_upper1,
            zone_type='resistance',
            strength=0.0,
            confidence=0.0,
            touches=0,
            created_at=ohlcv_data.index[0],
            last_tested=latest_timestamp,
            volume_profile={},
            timeframe=timeframe,
            method='vwap_bands',
            source_points=[],
            fractals=[],
            historical_respect=0.0,
            age_factor=1.0,
            context={'band_type': 'upper_band1'}
        )
        zones.append(upper1_zone)
        
        upper2_zone = ZoneLevel(
            price=latest_upper2,
            zone_type='resistance',
            strength=0.0,
            confidence=0.0,
            touches=0,
            created_at=ohlcv_data.index[0],
            last_tested=latest_timestamp,
            volume_profile={},
            timeframe=timeframe,
            method='vwap_bands',
            source_points=[],
            fractals=[],
            historical_respect=0.0,
            age_factor=1.0,
            context={'band_type': 'upper_band2'}
        )
        zones.append(upper2_zone)
        
        # Create lower band levels (support)
        lower1_zone = ZoneLevel(
            price=latest_lower1,
            zone_type='support',
            strength=0.0,
            confidence=0.0,
            touches=0,
            created_at=ohlcv_data.index[0],
            last_tested=latest_timestamp,
            volume_profile={},
            timeframe=timeframe,
            method='vwap_bands',
            source_points=[],
            fractals=[],
            historical_respect=0.0,
            age_factor=1.0,
            context={'band_type': 'lower_band1'}
        )
        zones.append(lower1_zone)
        
        lower2_zone = ZoneLevel(
            price=latest_lower2,
            zone_type='support',
            strength=0.0,
            confidence=0.0,
            touches=0,
            created_at=ohlcv_data.index[0],
            last_tested=latest_timestamp,
            volume_profile={},
            timeframe=timeframe,
            method='vwap_bands',
            source_points=[],
            fractals=[],
            historical_respect=0.0,
            age_factor=1.0,
            context={'band_type': 'lower_band2'}
        )
        zones.append(lower2_zone)
        
        # Calculate how often price has interacted with VWAP and its bands
        for i, zone in enumerate(zones):
            band_type = zone.context['band_type']
            band_price_series = None
            
            if band_type == 'vwap_base':
                band_price_series = vwap
            elif band_type == 'upper_band1':
                band_price_series = upper_band1
            elif band_type == 'upper_band2':
                band_price_series = upper_band2
            elif band_type == 'lower_band1':
                band_price_series = lower_band1
            elif band_type == 'lower_band2':
                band_price_series = lower_band2
            
            if band_price_series is not None:
                touch_count = 0
                respect_count = 0
                
                # Look for interactions in the last 50% of data
                start_idx = len(ohlcv_data) // 2
                for j in range(start_idx, len(ohlcv_data) - 1):
                    curr_price = ohlcv_data.iloc[j]['close']
                    next_price = ohlcv_data.iloc[j+1]['close']
                    band_price = band_price_series.iloc[j]
                    
                    # Price crossed the band
                    if (curr_price < band_price and next_price > band_price) or \
                       (curr_price > band_price and next_price < band_price):
                        touch_count += 1
                    
                    # Price approached the band but bounced
                    elif abs(curr_price - band_price) / band_price < 0.002 and \
                         ((curr_price < band_price and next_price < band_price and next_price < curr_price) or \
                          (curr_price > band_price and next_price > band_price and next_price > curr_price)):
                        touch_count += 1
                        respect_count += 1
                
                # Update zone with touch counts
                zones[i].touches = touch_count
                if touch_count > 0:
                    zones[i].historical_respect = respect_count / touch_count
        
        logger.debug(f"Detected {len(zones)} VWAP band levels")
        return zones
    
    def _detect_liquidity_pools(
        self, 
        ohlcv_data: pd.DataFrame,
        timeframe: str,
        symbol: str
    ) -> List[ZoneLevel]:
        """
        Detect support and resistance based on liquidity pools.
        Identifies areas where stop losses are likely to be clustered.
        """
        logger.debug("Detecting liquidity pools")
        zones = []
        
        if len(ohlcv_data) < 50:
            return zones
        
        # Find significant swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(5, len(ohlcv_data) - 5):
            # Check for swing high
            is_swing_high = True
            for j in range(1, 5):
                if ohlcv_data.iloc[i]['high'] <= ohlcv_data.iloc[i-j]['high'] or \
                   ohlcv_data.iloc[i]['high'] <= ohlcv_data.iloc[i+j]['high']:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append(i)
            
            # Check for swing low
            is_swing_low = True
            for j in range(1, 5):
                if ohlcv_data.iloc[i]['low'] >= ohlcv_data.iloc[i-j]['low'] or \
                   ohlcv_data.iloc[i]['low'] >= ohlcv_data.iloc[i+j]['low']:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append(i)
        
        # Find liquidity above swing highs (where stop losses might be placed)
        for idx in swing_highs:
            swing_high = ohlcv_data.iloc[idx]['high']
            timestamp = ohlcv_data.index[idx]
            
            # Liquidity level is slightly above the swing high
            liquidity_level = swing_high * 1.002  # 0.2% above
            
            # Create source points
            source_points = [(timestamp, swing_high)]
            
            # Check if this liquidity has been hunted
            liquidity_hunted = False
            hunt_timestamp = None
            
            # Look for price spikes that may have hunted this liquidity
            for i in range(idx + 5, min(len(ohlcv_data), idx + 100)):
                # Price spiked above the liquidity level
                if ohlcv_data.iloc[i]['high'] > liquidity_level:
                    # But then dropped back below
                    if ohlcv_data.iloc[i]['close'] < swing_high:
                        liquidity_hunted = True
                        hunt_timestamp = ohlcv_data.index[i]
                        break
            
            zone = ZoneLevel(
                price=liquidity_level,
                zone_type='liquidity_resistance',
                strength=0.0,  # Will be calculated later
                confidence=0.0,  # Will be calculated later
                touches=0,
                created_at=timestamp,
                last_tested=hunt_timestamp if hunt_timestamp else timestamp,
                volume_profile={},
                timeframe=timeframe,
                method='liquidity_pools',
                source_points=source_points,
                fractals=[],
                historical_respect=0.0,
                age_factor=1.0,
                context={
                    'pool_type': 'high_side',
                    'hunted': liquidity_hunted,
                    'base_level': swing_high
                }
            )
            zones.append(zone)
        
        # Find liquidity below swing lows (where stop losses might be placed)
        for idx in swing_lows:
            swing_low = ohlcv_data.iloc[idx]['low']
            timestamp = ohlcv_data.index[idx]
            
            # Liquidity level is slightly below the swing low
            liquidity_level = swing_low * 0.998  # 0.2% below
            
            # Create source points
            source_points = [(timestamp, swing_low)]
            
            # Check if this liquidity has been hunted
            liquidity_hunted = False
            hunt_timestamp = None
            
            # Look for price drops that may have hunted this liquidity
            for i in range(idx + 5, min(len(ohlcv_data), idx + 100)):
                # Price dropped below the liquidity level
                if ohlcv_data.iloc[i]['low'] < liquidity_level:
                    # But then recovered back above
                    if ohlcv_data.iloc[i]['close'] > swing_low:
                        liquidity_hunted = True
                        hunt_timestamp = ohlcv_data.index[i]
                        break
            
            zone = ZoneLevel(
                price=liquidity_level,
                zone_type='liquidity_support',
                strength=0.0,  # Will be calculated later
                confidence=0.0,  # Will be calculated later
                touches=0,
                created_at=timestamp,
                last_tested=hunt_timestamp if hunt_timestamp else timestamp,
                volume_profile={},
                timeframe=timeframe,
                method='liquidity_pools',
                source_points=source_points,
                fractals=[],
                historical_respect=0.0,
                age_factor=1.0,
                context={
                    'pool_type': 'low_side',
                    'hunted': liquidity_hunted,
                    'base_level': swing_low
                }
            )
            zones.append(zone)
        
        logger.debug(f"Detected {len(zones)} liquidity pool levels")
        return zones
    
    def _detect_structure_levels(
        self, 
        ohlcv_data: pd.DataFrame,
        timeframe: str,
        symbol: str
    ) -> List[ZoneLevel]:
        """
        Detect support and resistance based on market structure.
        Uses higher timeframe market structure for potential zones.
        """
        logger.debug("Detecting structure levels")
        zones = []
        
        # Get market structure points
        structure_points = self.market_structure.get_structure_points(ohlcv_data)
        
        if not structure_points:
            return zones
        
        # Process each structure point
        for point in structure_points:
            point_type = point['type']
            price = point['price']
            timestamp = point['time']
            
            # Decide zone type based on structure point type
            if point_type in ['higher_high', 'lower_high']:
                zone_type = 'resistance'
            elif point_type in ['higher_low', 'lower_low']:
                zone_type = 'support'
            else:
                continue  # Skip if not a clear structure point
            
            # Check how often this structure level has been tested
            test_count = 0
            last_tested = timestamp
            respect_count = 0
            
            # Look for later price interactions with this level
            for i in range(len(ohlcv_data)):
                if ohlcv_data.index[i] <= timestamp:
                    continue
                
                curr_candle = ohlcv_data.iloc[i]
                
                # For resistance, check if price approached from below
                if zone_type == 'resistance' and abs(curr_candle['high'] - price) / price < 0.003:
                    test_count += 1
                    last_tested = ohlcv_data.index[i]
                    
                    # If price was rejected (didn't close above)
                    if curr_candle['close'] < price:
                        respect_count += 1
                
                # For support, check if price approached from above
                elif zone_type == 'support' and abs(curr_candle['low'] - price) / price < 0.003:
                    test_count += 1
                    last_tested = ohlcv_data.index[i]
                    
                    # If price was supported (didn't close below)
                    if curr_candle['close'] > price:
                        respect_count += 1
            
            # Calculate historical respect ratio
            historical_respect = respect_count / max(1, test_count)
            
            zone = ZoneLevel(
                price=price,
                zone_type=zone_type,
                strength=0.0,  # Will be calculated later
                confidence=0.0,  # Will be calculated later
                touches=test_count,
                created_at=timestamp,
                last_tested=last_tested,
                volume_profile={},
                timeframe=timeframe,
                method='market_structure',
                source_points=[(timestamp, price)],
                fractals=[],
                historical_respect=historical_respect,
                age_factor=1.0,
                context={'structure_type': point_type, 'structure_strength': point.get('strength', 1.0)}
            )
            zones.append(zone)
        
        logger.debug(f"Detected {len(zones)} structure levels")
        return zones
    
    def _detect_psychological_levels(
        self, 
        ohlcv_data: pd.DataFrame,
        timeframe: str,
        symbol: str
    ) -> List[ZoneLevel]:
        """
        Detect support and resistance based on psychological levels.
        Identifies round numbers and other psychological price points.
        """
        logger.debug("Detecting psychological levels")
        zones = []
        
        if len(ohlcv_data) < 20:
            return zones
        
        # Get price range
        price_min = ohlcv_data['low'].min()
        price_max = ohlcv_data['high'].max()
        current_price = ohlcv_data['close'].iloc[-1]
        
        # Function to find the magnitude of a price
        def find_magnitude(price):
            magnitude = 1
            while price >= 10:
                price /= 10
                magnitude *= 10
            return magnitude
        
        # Find appropriate magnitude for psychological levels
        magnitude = find_magnitude(current_price)
        
        # Generate psychological levels
        psych_levels = []
        
        # Round numbers (1.0000, 2.0000, etc.)
        level = (int(price_min / magnitude) - 1) * magnitude
        while level <= price_max * 1.1:  # Extend slightly beyond max
            psych_levels.append(level)
            level += magnitude
        
        # Half-round numbers (0.5000, 1.5000, etc.)
        level = (int(price_min / magnitude) - 1) * magnitude + (magnitude / 2)
        while level <= price_max * 1.1:
            psych_levels.append(level)
            level += magnitude
        
        # Quarter-round numbers for major markets
        if current_price > 100:  # Only for higher-priced assets
            level = (int(price_min / magnitude) - 1) * magnitude + (magnitude / 4)
            while level <= price_max * 1.1:
                psych_levels.append(level)
                level += magnitude / 2  # Add 0.25, then 0.75, etc.
        
        # Process each psychological level
        for level_price in psych_levels:
            # Skip if level is too far from current price
            if level_price < price_min * 0.9 or level_price > price_max * 1.1:
                continue
            
            # Determine if level is resistance or support
            if level_price > current_price:
                zone_type = 'resistance'
            else:
                zone_type = 'support'
            
            # Check how often this level has been tested
            test_count = 0
            respect_count = 0
            last_tested = ohlcv_data.index[0]
            
            # Look for price interactions with this level
            for i in range(len(ohlcv_data) - 1):
                curr_candle = ohlcv_data.iloc[i]
                next_candle = ohlcv_data.iloc[i + 1]
                
                # Price approaches or crosses the level
                level_touched = False
                
                # For resistance, check if price approached from below
                if zone_type == 'resistance' and curr_candle['high'] >= level_price * 0.997 and curr_candle['high'] <= level_price * 1.003:
                    level_touched = True
                    
                    # If price was rejected (next candle shows downward movement)
                    if next_candle['close'] < curr_candle['close']:
                        respect_count += 1
                
                # For support, check if price approached from above
                elif zone_type == 'support' and curr_candle['low'] <= level_price * 1.003 and curr_candle['low'] >= level_price * 0.997:
                    level_touched = True
                    
                    # If price was supported (next candle shows upward movement)
                    if next_candle['close'] > curr_candle['close']:
                        respect_count += 1
                
                if level_touched:
                    test_count += 1
                    last_tested = ohlcv_data.index[i]
            
            # Calculate historical respect ratio
            historical_respect = respect_count / max(1, test_count)
            
            # Determine level significance
            if level_price % magnitude == 0:
                level_type = 'round'
                strength_factor = 0.9
            elif level_price % (magnitude / 2) == 0:
                level_type = 'half_round'
                strength_factor = 0.7
            else:
                level_type = 'quarter_round'
                strength_factor = 0.5
            
            zone = ZoneLevel(
                price=level_price,
                zone_type=zone_type,
                strength=0.0,  # Will be calculated later
                confidence=0.0,  # Will be calculated later
                touches=test_count,
                created_at=ohlcv_data.index[0],  # Psychological levels exist from the beginning
                last_tested=last_tested,
                volume_profile={},
                timeframe=timeframe,
                method='psychological',
                source_points=[],
                fractals=[],
                historical_respect=historical_respect,
                age_factor=1.0,
                context={'level_type': level_type, 'base_magnitude': magnitude, 'strength_factor': strength_factor}
            )
            zones.append(zone)
        
        logger.debug(f"Detected {len(zones)} psychological levels")
        return zones
    
    def _merge_overlapping_zones(self, zones: List[ZoneLevel]) -> List[ZoneLevel]:
        """
        Merge zones that are very close to each other to avoid redundancy.
        
        Args:
            zones: List of initially detected zones
            
        Returns:
            List of merged zones
        """
        if not zones:
            return []
            
        logger.debug(f"Merging overlapping zones from initial {len(zones)} zones")
        
        # Sort zones by price
        sorted_zones = sorted(zones, key=lambda x: x.price)
        
        merged_zones = []
        current_zone = sorted_zones[0]
        
        for next_zone in sorted_zones[1:]:
            # Check if zones are close enough to merge
            price_gap = abs(next_zone.price - current_zone.price) / current_zone.price
            
            if (price_gap < self.config['zone_merge_threshold'] and 
                next_zone.zone_type == current_zone.zone_type):
                # Merge zones
                merged_price = (current_zone.price + next_zone.price) / 2
                merged_touches = current_zone.touches + next_zone.touches
                merged_source_points = current_zone.source_points + next_zone.source_points
                merged_fractals = current_zone.fractals + next_zone.fractals
                
                # Take the earliest creation date
                merged_created_at = min(current_zone.created_at, next_zone.created_at)
                
                # Take the latest test date
                merged_last_tested = max(current_zone.last_tested, next_zone.last_tested)
                
                # Merge context information
                merged_context = current_zone.context.copy() if current_zone.context else {}
                if next_zone.context:
                    if merged_context:
                        merged_context.update({f"merged_{k}": v for k, v in next_zone.context.items()})
                    else:
                        merged_context = next_zone.context.copy()
                
                # Create merged zone
                current_zone = ZoneLevel(
                    price=merged_price,
                    zone_type=current_zone.zone_type,
                    strength=0.0,  # Will be calculated later
                    confidence=0.0,  # Will be calculated later
                    touches=merged_touches,
                    created_at=merged_created_at,
                    last_tested=merged_last_tested,
                    volume_profile=current_zone.volume_profile,  # Use primary zone's volume profile
                    timeframe=current_zone.timeframe,
                    method=f"{current_zone.method}+{next_zone.method}",
                    source_points=merged_source_points,
                    fractals=merged_fractals,
                    historical_respect=max(current_zone.historical_respect, next_zone.historical_respect),
                    age_factor=1.0,
                    context=merged_context
                )
            else:
                # Zones are not mergeable, add current to results and move to next
                merged_zones.append(current_zone)
                current_zone = next_zone
        
        # Add the last zone
        merged_zones.append(current_zone)
        
        logger.debug(f"Merged into {len(merged_zones)} zones")
        return merged_zones
    
    def _analyze_zone_quality(
        self, 
        zones: List[ZoneLevel],
        ohlcv_data: pd.DataFrame,
        timeframe: str
    ) -> List[ZoneLevel]:
        """
        Analyze zone quality by calculating strength and confidence metrics.
        
        Args:
            zones: List of zones to analyze
            ohlcv_data: OHLCV data for analysis
            timeframe: Current timeframe
            
        Returns:
            List of zones with calculated quality metrics
        """
        if not zones:
            return []
            
        logger.debug(f"Analyzing quality of {len(zones)} zones")
        
        current_time = ohlcv_data.index[-1]
        current_price = ohlcv_data['close'].iloc[-1]
        
        for i, zone in enumerate(zones):
            # 1. Calculate zone strength based on multiple factors
            
            # Factor 1: Recency (newer zones get higher strength)
            time_factor = self._calculate_time_factor(zone.created_at, current_time)
            
            # Factor 2: Number of touches (more touches = higher strength)
            touch_factor = min(1.0, zone.touches / 10) * self.config['zone_strength_touch_factor']
            
            # Factor 3: Historical respect ratio
            respect_factor = zone.historical_respect
            
            # Factor 4: Volume profile (higher volume = higher strength)
            volume_factor = 0.5  # Default
            if 'node_volume' in zone.volume_profile:
                volume_pct = zone.volume_profile.get('percent_of_total', 0.1)
                volume_factor = min(1.0, volume_pct * 10) * self.config['zone_strength_volume_factor']
            
            # Factor 5: Method strength (some methods are inherently stronger)
            method_strength = {
                'swing_high_low': 0.85,
                'price_clusters': 0.80,
                'order_blocks': 0.90,
                'market_structure': 0.85,
                'volume_profile': 0.75,
                'fractals': 0.70,
                'fibonacci': 0.65,
                'pivot_points': 0.70,
                'moving_averages': 0.60,
                'vwap_bands': 0.60,
                'liquidity_pools': 0.75,
                'psychological': 0.55
            }
            
            # Handle merged methods
            if '+' in zone.method:
                methods = zone.method.split('+')
                method_factor = max([method_strength.get(m, 0.5) for m in methods])
            else:
                method_factor = method_strength.get(zone.method, 0.5)
            
            # Factor 6: Age decay
            age_factor = zone.age_factor
            
            # Calculate final strength (weighted combination of factors)
            strength = (
                time_factor * 0.15 +
                touch_factor * 0.25 +
                respect_factor * 0.25 +
                volume_factor * 0.15 +
                method_factor * 0.10 +
                age_factor * 0.10
            )
            
            # 2. Calculate zone confidence
            
            # Factor 1: Distance from current price (closer = higher confidence)
            price_distance = abs(zone.price - current_price) / current_price
            distance_factor = max(0, 1 - (price_distance * 20))  # 5% distance -> 0 confidence
            
            # Factor 2: Strength is a factor in confidence
            strength_factor = strength
            
            # Factor 3: Timeframe weight (higher timeframes = higher confidence)
            tf_weight = self.config['multi_timeframe_weights'].get(timeframe, 0.6)
            
            # Factor 4: Number of source points
            source_points_factor = min(1.0, len(zone.source_points) / 10)
            
            # Calculate final confidence
            confidence = (
                distance_factor * 0.35 +
                strength_factor * 0.35 +
                tf_weight * 0.20 +
                source_points_factor * 0.10
            )
            
            # Update zone with calculated metrics
            zones[i].strength = strength
            zones[i].confidence = confidence
            
            # Extra: Calculate volume profile if not present
            if not zone.volume_profile and len(ohlcv_data) > 0:
                # Find volume at price level
                price_level = zone.price
                price_tolerance = price_level * 0.002  # 0.2% tolerance
                
                # Filter data points within tolerance of the price level
                level_data = ohlcv_data[
                    ((ohlcv_data['high'] >= price_level - price_tolerance) &
                     (ohlcv_data['low'] <= price_level + price_tolerance))
                ]
                
                if not level_data.empty:
                    total_volume = ohlcv_data['volume'].sum()
                    level_volume = level_data['volume'].sum()
                    
                    zones[i].volume_profile = {
                        'level_volume': level_volume,
                        'percent_of_total': level_volume / total_volume if total_volume > 0 else 0
                    }
        
        # Sort zones by strength for better management
        sorted_zones = sorted(zones, key=lambda x: x.strength, reverse=True)
        
        # Return only zones with sufficient confidence
        filtered_zones = [
            zone for zone in sorted_zones 
            if zone.confidence >= self.config['zone_confidence_threshold']
        ]
        
        logger.debug(f"Analysis complete, returning {len(filtered_zones)} high-quality zones")
        return filtered_zones
    
    def _calculate_time_factor(self, created_time: datetime, current_time: datetime) -> float:
        """Calculate a time decay factor based on zone age."""
        time_diff = (current_time - created_time).total_seconds()
        
        # Convert to hours
        hours_diff = time_diff / 3600
        
        # Decay factor (newer zones are stronger)
        # After about 1 week, factor drops to around 0.5
        decay_factor = np.exp(-hours_diff / (24 * 7))
        
        return decay_factor
    
    def _correlate_multi_timeframe_zones(self, all_zones: List[ZoneLevel]) -> None:
        """
        Correlate zones across multiple timeframes.
        Identifies parent-child relationships between zones on different timeframes.
        
        Args:
            all_zones: List of zones from all timeframes
        """
        if not all_zones:
            return
            
        logger.debug(f"Correlating zones across timeframes for {len(all_zones)} zones")
        
        # Sort zones by timeframe (higher timeframes first)
        timeframe_order = {tf: i for i, tf in enumerate(reversed(list(self.config['multi_timeframe_weights'].keys())))}
        sorted_zones = sorted(all_zones, key=lambda x: timeframe_order.get(x.timeframe, 99))
        
        # Group zones by timeframe
        timeframe_zones = {}
        for zone in sorted_zones:
            if zone.timeframe not in timeframe_zones:
                timeframe_zones[zone.timeframe] = []
            timeframe_zones[zone.timeframe].append(zone)
        
        # Assign zone IDs
        zone_id = 0
        for zone in sorted_zones:
            zone.zone_id = zone_id
            zone_id += 1
        
        # Find parent-child relationships
        for tf, zones in timeframe_zones.items():
            # Skip highest timeframe (it has no parents)
            if tf == list(timeframe_order.keys())[0]:
                continue
                
            # Find potential parent timeframe
            parent_tfs = [t for t in timeframe_order.keys() if timeframe_order[t] < timeframe_order[tf]]
            if not parent_tfs:
                continue
                
            parent_tf = parent_tfs[0]
            if parent_tf not in timeframe_zones:
                continue
                
            parent_zones = timeframe_zones[parent_tf]
            
            # For each zone in current timeframe, find potential parent in higher timeframe
            for zone in zones:
                best_parent = None
                best_distance = float('inf')
                
                for parent in parent_zones:
                    # Skip if zone types don't match (support/resistance)
                    if zone.zone_type != parent.zone_type and not (
                        zone.zone_type == 'pivot' or parent.zone_type == 'pivot'
                    ):
                        continue
                        
                    # Calculate price distance
                    price_distance = abs(zone.price - parent.price) / parent.price
                    
                    # Consider it a match if distance is small enough
                    if price_distance < 0.01 and price_distance < best_distance:
                        best_distance = price_distance
                        best_parent = parent
                
                if best_parent:
                    zone.parent_zone = best_parent.zone_id
    
    def get_active_zones(
        self,
        symbol: str,
        timeframe: str,
        price: Optional[float] = None,
        num_zones: int = 5,
        zone_types: Optional[List[str]] = None
    ) -> List[ZoneLevel]:
        """
        Get the most relevant active zones for the current price.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to analyze
            price: Current price (if None, will be fetched)
            num_zones: Number of zones to return
            zone_types: Types of zones to include
            
        Returns:
            List of active zones
        """
        if zone_types is None:
            zone_types = ['support', 'resistance', 'pivot']
        
        # Ensure zones are detected
        if not self.zones.get(timeframe, []):
            logger.warning(f"No zones detected for {symbol} on {timeframe}")
            return []
        
        # Get current price if not provided
        if price is None:
            try:
                recent_data = self.ts_store.get_ohlcv_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=1
                )
                if recent_data.empty:
                    logger.error(f"Could not get current price for {symbol}")
                    return []
                price = recent_data['close'].iloc[0]
            except Exception as e:
                logger.error(f"Error getting current price: {str(e)}")
                return []
        
        # Filter zones by type
        filtered_zones = [z for z in self.zones[timeframe] if z.zone_type in zone_types]
        
        # Sort zones by distance from current price
        sorted_zones = sorted(
            filtered_zones,
            key=lambda z: (abs(z.price - price) / price, -z.strength)
        )
        
        # Return top N zones
        active_zones = sorted_zones[:num_zones]
        self.active_zones[timeframe] = active_zones
        
        return active_zones
    
    def get_next_zones(
        self,
        symbol: str,
        timeframe: str,
        price: float,
        direction: str,
        num_zones: int = 3
    ) -> List[ZoneLevel]:
        """
        Get the next zones in a given direction from current price.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to analyze
            price: Current price
            direction: 'up' or 'down'
            num_zones: Number of zones to return
            
        Returns:
            List of next zones in the specified direction
        """
        if not self.zones.get(timeframe, []):
            logger.warning(f"No zones detected for {symbol} on {timeframe}")
            return []
        
        if direction == 'up':
            # Get resistance zones above current price
            candidate_zones = [
                z for z in self.zones[timeframe] 
                if (z.zone_type == 'resistance' or z.zone_type == 'pivot') and z.price > price
            ]
            
            # Sort by price (nearest first)
            sorted_zones = sorted(candidate_zones, key=lambda z: z.price)
        else:
            # Get support zones below current price
            candidate_zones = [
                z for z in self.zones[timeframe] 
                if (z.zone_type == 'support' or z.zone_type == 'pivot') and z.price < price
            ]
            
            # Sort by price (nearest first)
            sorted_zones = sorted(candidate_zones, key=lambda z: -z.price)
        
        # Return top N zones
        return sorted_zones[:num_zones]
    
    def get_zone_projections(
        self,
        symbol: str,
        timeframe: str,
        start_price: float,
        direction: str,
        target_confidence: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Get zone projections for potential targets and stops.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to analyze
            start_price: Starting price for projection
            direction: 'long' or 'short'
            target_confidence: Minimum confidence for target zones
            
        Returns:
            List of projected zones with metadata
        """
        if not self.zones.get(timeframe, []):
            logger.warning(f"No zones detected for {symbol} on {timeframe}")
            return []
        
        projections = []
        
        if direction == 'long':
            # For long positions, resistance zones are targets, support zones are stops
            target_zones = [
                z for z in self.zones[timeframe]
                if (z.zone_type == 'resistance' or z.zone_type == 'pivot') and 
                z.price > start_price and
                z.confidence >= target_confidence
            ]
            
            stop_zones = [
                z for z in self.zones[timeframe]
                if (z.zone_type == 'support' or z.zone_type == 'pivot') and 
                z.price < start_price
            ]
            
            # Sort targets by price (nearest first)
            target_zones = sorted(target_zones, key=lambda z: z.price)
            
            # Sort stops by price (nearest to entry first) and strength (strongest first)
            stop_zones = sorted(stop_zones, key=lambda z: (-z.price, -z.strength))
            
        else:
            # For short positions, support zones are targets, resistance zones are stops
            target_zones = [
                z for z in self.zones[timeframe]
                if (z.zone_type == 'support' or z.zone_type == 'pivot') and 
                z.price < start_price and
                z.confidence >= target_confidence
            ]
            
            stop_zones = [
                z for z in self.zones[timeframe]
                if (z.zone_type == 'resistance' or z.zone_type == 'pivot') and 
                z.price > start_price
            ]
            
            # Sort targets by price (nearest first)
            target_zones = sorted(target_zones, key=lambda z: -z.price)
            
            # Sort stops by price (nearest to entry first) and strength (strongest first)
            stop_zones = sorted(stop_zones, key=lambda z: (z.price, -z.strength))
        
        # Take top 3 target zones
        for zone in target_zones[:3]:
            # Calculate risk-reward ratio
            rr_ratio = abs(zone.price - start_price) / abs(stop_zones[0].price - start_price) if stop_zones else 0
            
            projections.append({
                'type': 'target',
                'price': zone.price,
                'confidence': zone.confidence,
                'strength': zone.strength,
                'method': zone.method,
                'timeframe': zone.timeframe,
                'risk_reward': rr_ratio,
                'projected_touches': 0,  # Will be calculated if needed
                'zone_id': zone.zone_id
            })
        
        # Take top stop zone
        if stop_zones:
            projections.append({
                'type': 'stop',
                'price': stop_zones[0].price,
                'confidence': stop_zones[0].confidence,
                'strength': stop_zones[0].strength,
                'method': stop_zones[0].method,
                'timeframe': stop_zones[0].timeframe,
                'risk_reward': 1.0,  # Base risk
                'projected_touches': 0,  # Will be calculated if needed
                'zone_id': stop_zones[0].zone_id
            })
        
        return projections
    
    def find_confluence_zones(
        self,
        symbol: str,
        timeframes: List[str] = None,
        price_tolerance: float = 0.005
    ) -> List[Dict[str, Any]]:
        """
        Find zones that appear across multiple timeframes (confluence zones).
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes to analyze
            price_tolerance: Percentage tolerance for price matching
            
        Returns:
            List of confluence zones with metadata
        """
        if timeframes is None:
            timeframes = list(self.config['multi_timeframe_weights'].keys())
        
        # Ensure we have zones for all timeframes
        missing_tfs = []
        for tf in timeframes:
            if not self.zones.get(tf, []):
                missing_tfs.append(tf)
        
        if missing_tfs:
            logger.warning(f"Missing zones for timeframes: {missing_tfs}")
            # Remove missing timeframes
            timeframes = [tf for tf in timeframes if tf not in missing_tfs]
        
        if not timeframes:
            return []
        
        # Collect all zones from all timeframes
        all_zones = []
        for tf in timeframes:
            all_zones.extend(self.zones[tf])
        
        # Group zones by price (within tolerance)
        price_groups = {}
        
        for zone in all_zones:
            found_group = False
            
            for price in price_groups:
                if abs(zone.price - price) / price <= price_tolerance:
                    price_groups[price].append(zone)
                    found_group = True
                    break
            
            if not found_group:
                price_groups[zone.price] = [zone]
        
        # Find groups with zones from multiple timeframes
        confluence_zones = []
        
        for price, zones in price_groups.items():
            # Get unique timeframes in this group
            group_tfs = set(zone.timeframe for zone in zones)
            
            # Consider it confluence if it appears in at least 2 timeframes
            if len(group_tfs) >= 2:
                # Calculate average price, strength, and confidence
                avg_price = sum(z.price for z in zones) / len(zones)
                avg_strength = sum(z.strength for z in zones) / len(zones)
                avg_confidence = sum(z.confidence for z in zones) / len(zones)
                
                # Get zone type (majority vote)
                type_counts = {}
                for z in zones:
                    type_counts[z.zone_type] = type_counts.get(z.zone_type, 0) + 1
                zone_type = max(type_counts.items(), key=lambda x: x[1])[0]
                
                # Calculate a confluence score
                timeframe_weights = {tf: self.config['multi_timeframe_weights'].get(tf, 0.5) for tf in group_tfs}
                weighted_sum = sum(timeframe_weights.values())
                confluence_score = weighted_sum * (len(group_tfs) / len(timeframes))
                
                # Create confluence zone
                confluence_zone = {
                    'price': avg_price,
                    'type': zone_type,
                    'strength': avg_strength,
                    'confidence': avg_confidence,
                    'confluence_score': confluence_score,
                    'timeframes': list(group_tfs),
                    'methods': list(set(z.method for z in zones)),
                    'zone_ids': [z.zone_id for z in zones]
                }
                
                confluence_zones.append(confluence_zone)
        
        # Sort by confluence score
        sorted_zones = sorted(confluence_zones, key=lambda x: x['confluence_score'], reverse=True)
        
        return sorted_zones
    
    def get_zone_statistics(
        self, 
        symbol: str,
        timeframe: str,
        zone_type: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> Dict[str, Any]:
        """
        Get statistics about detected zones.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe to analyze
            zone_type: Optional filter by zone type
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary of zone statistics
        """
        if not self.zones.get(timeframe, []):
            return {
                'total_zones': 0,
                'filtered_zones': 0,
                'by_type': {},
                'by_method': {},
                'avg_strength': 0,
                'avg_confidence': 0,
                'avg_touches': 0
            }
        
        # Filter zones
        zones = self.zones[timeframe]
        if zone_type:
            zones = [z for z in zones if z.zone_type == zone_type]
        
        zones = [z for z in zones if z.confidence >= min_confidence]
        
        if not zones:
            return {
                'total_zones': len(self.zones[timeframe]),
                'filtered_zones': 0,
                'by_type': {},
                'by_method': {},
                'avg_strength': 0,
                'avg_confidence': 0,
                'avg_touches': 0
            }
        
        # Calculate statistics
        by_type = {}
        for z in zones:
            by_type[z.zone_type] = by_type.get(z.zone_type, 0) + 1
        
        by_method = {}
        for z in zones:
            methods = z.method.split('+')
            for m in methods:
                by_method[m] = by_method.get(m, 0) + 1
        
        avg_strength = sum(z.strength for z in zones) / len(zones)
        avg_confidence = sum(z.confidence for z in zones) / len(zones)
        avg_touches = sum(z.touches for z in zones) / len(zones)
        
        return {
            'total_zones': len(self.zones[timeframe]),
            'filtered_zones': len(zones),
            'by_type': by_type,
            'by_method': by_method,
            'avg_strength': avg_strength,
            'avg_confidence': avg_confidence,
            'avg_touches': avg_touches
        }
    
    def reset(self):
        """Reset all detected zones."""
        self.zones = {tf: [] for tf in TIMEFRAMES}
        self.active_zones = {tf: [] for tf in TIMEFRAMES}
        logger.info("Support/resistance zones reset")


def identify_support_resistance_zones(
    data: pd.DataFrame,
    zone_width_pct: float = 0.01,
    min_touches: int = 2,
    swing_strength: float = 0.5,
) -> Dict[str, List[float]]:
    """Lightweight support/resistance detection using pivot points."""
    supports: List[float] = []
    resistances: List[float] = []
    highs = data['high']
    lows = data['low']

    for i in range(1, len(data) - 1):
        if lows.iloc[i] <= lows.iloc[i - 1] and lows.iloc[i] <= lows.iloc[i + 1]:
            supports.append(lows.iloc[i])
        if highs.iloc[i] >= highs.iloc[i - 1] and highs.iloc[i] >= highs.iloc[i + 1]:
            resistances.append(highs.iloc[i])

    return {
        'support': sorted(set(supports)),
        'resistance': sorted(set(resistances)),
    }

