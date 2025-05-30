

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Pattern Recognition Features Module

This module provides sophisticated pattern recognition capabilities for the QuantumSpectre Elite
Trading System. It includes advanced detection of various chart patterns, harmonic patterns,
candlestick patterns, and custom pattern recognition algorithms designed to identify
high-probability trading setups with strong success rates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from scipy import stats
from scipy.signal import argrelextrema, find_peaks
import logging
try:
    import ta  # type: ignore
    TA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    ta = None  # type: ignore
    TA_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "ta library not available; pattern features will be limited"
    )
from common.ta_candles import cdl_pattern
from dataclasses import dataclass
from datetime import datetime, timedelta

from common.logger import get_logger
from common.utils import timing_decorator, parallelize
from feature_service.features.market_structure import MarketStructureAnalyzer

logger = get_logger(__name__)

@dataclass
class PatternConfiguration:
    """Configuration parameters for pattern detection"""
    min_pattern_bars: int = 3  # Minimum number of bars required for a pattern
    max_pattern_bars: int = 50  # Maximum number of bars to consider for pattern detection
    similarity_threshold: float = 0.85  # Threshold for pattern similarity matching
    peak_prominence: float = 0.5  # Prominence parameter for peak detection
    min_pattern_height: float = 0.2  # Minimum height as percentage of price
    min_completion_level: float = 0.618  # Minimum Fibonacci level for pattern completion
    harmonic_tolerance: float = 0.05  # Tolerance for harmonic ratio matching
    pivot_lookback: int = 20  # Bars to look back for pivot points
    fibonacci_levels: List[float] = None  # Fibonacci retracement levels
    
    def __post_init__(self):
        if self.fibonacci_levels is None:
            self.fibonacci_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]


class PatternFeatures:
    """
    Advanced pattern recognition features for the QuantumSpectre Elite Trading System.
    This class provides sophisticated pattern detection and analysis capabilities,
    identifying high-probability trading opportunities across multiple timeframes.
    """
    
    def __init__(self, config: Optional[PatternConfiguration] = None):
        """
        Initialize the pattern recognition feature generator.
        
        Args:
            config: Optional configuration parameters for pattern detection
        """
        self.config = config if config else PatternConfiguration()
        self.market_structure = MarketStructureAnalyzer()
        self.pattern_memory = {}  # Cache for storing previously identified patterns
        self.pattern_success_rates = {}  # Track success rates of patterns
        logger.info("Initializing PatternRecognitionFeatures with configuration: %s", self.config)
        
    @timing_decorator
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all pattern recognition features for the given data.
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with added pattern recognition features
        """
        logger.debug("Calculating pattern features for %d rows of data", len(data))
        
        # Create a copy to avoid modifying the input
        result = data.copy()
        
        # Ensure we have enough data for pattern detection
        if len(result) < self.config.max_pattern_bars:
            logger.warning("Insufficient data for pattern detection: %d rows", len(result))
            return result
        
        # Calculate common patterns using TA-Lib
        self._add_talib_patterns(result)
        
        # Calculate harmonic patterns
        self._add_harmonic_patterns(result)
        
        # Advanced chart patterns
        self._add_advanced_chart_patterns(result)
        
        # Custom pattern DNA sequencing
        self._add_pattern_dna_features(result)
        
        # Calculate multi-timeframe pattern convergence
        self._add_pattern_convergence_features(result)
        
        # Pattern completion percentage and target projections
        self._add_pattern_completion_features(result)
        
        # Pattern similarity to historical high-success patterns
        self._add_historical_pattern_similarity(result)
        
        # Add pattern success probability estimates
        self._add_pattern_success_probability(result)
        
        logger.debug("Completed pattern recognition feature calculation")
        return result
    
    def _add_talib_patterns(self, data: pd.DataFrame) -> None:
        """Add candlestick pattern features using ta_candles utilities."""
        logger.debug("Adding candlestick pattern recognition features")

        pattern_names = [
            'abandoned_baby', 'doji', 'engulfing', 'hammer', 'hanging_man',
            'harami', 'morning_star', 'shooting_star', 'marubozu', 'spinning_top',
            'three_white_soldiers', 'three_black_crows', 'evening_star',
            'dark_cloud_cover', 'breakaway', 'dragonfly_doji', 'gravestone_doji'
        ]

        for name in pattern_names:
            try:
                series = cdl_pattern(data, name=name)
                data[f'pattern_CDL_{name.upper()}'] = series
            except Exception as e:
                logger.error(f"Error calculating {name}: {str(e)}")
                data[f'pattern_CDL_{name.upper()}'] = 0
        
        # Normalize pattern signals to [-1, 0, 1] range
        for col in data.columns:
            if col.startswith('pattern_CDL_'):
                data[col] = data[col].apply(lambda x: np.sign(x))
        
        # Add pattern count features
        data['pattern_bullish_count'] = sum((data[col] > 0).astype(int) for col in data.columns if col.startswith('pattern_CDL_'))
        data['pattern_bearish_count'] = sum((data[col] < 0).astype(int) for col in data.columns if col.startswith('pattern_CDL_'))
        data['pattern_strength'] = data['pattern_bullish_count'] - data['pattern_bearish_count']
        
        logger.debug("Completed adding TA-Lib pattern features")
    
    def _add_harmonic_patterns(self, data: pd.DataFrame) -> None:
        """
        Add harmonic pattern features (Gartley, Butterfly, Bat, etc.).
        
        Args:
            data: DataFrame to add features to (modified in-place)
        """
        logger.debug("Adding harmonic pattern recognition features")
        
        # Find potential pivot points (alternating highs and lows)
        highs = argrelextrema(data['high'].values, np.greater, order=self.config.pivot_lookback)[0]
        lows = argrelextrema(data['low'].values, np.less, order=self.config.pivot_lookback)[0]
        
        # Initialize harmonic pattern columns
        harmonic_patterns = ['GARTLEY', 'BUTTERFLY', 'BAT', 'CRAB', 'SHARK', 'CYPHER', '5O']
        for pattern in harmonic_patterns:
            data[f'harmonic_{pattern}_bullish'] = 0
            data[f'harmonic_{pattern}_bearish'] = 0
        
        # Harmonic ratio targets with tolerance
        harmonic_ratios = {
            'GARTLEY': {'XA': 1.0, 'AB': 0.618, 'BC': 0.382, 'CD': 1.272},
            'BUTTERFLY': {'XA': 1.0, 'AB': 0.786, 'BC': 0.382, 'CD': 1.618},
            'BAT': {'XA': 1.0, 'AB': 0.382, 'BC': 0.382, 'CD': 1.618},
            'CRAB': {'XA': 1.0, 'AB': 0.382, 'BC': 0.618, 'CD': 2.618},
            'SHARK': {'XA': 1.0, 'AB': 0.5, 'BC': 1.618, 'CD': 0.886},
            'CYPHER': {'XA': 1.0, 'AB': 0.382, 'BC': 1.272, 'CD': 0.786},
            '5O': {'XA': 1.0, 'AB': 0.5, 'BC': 0.5, 'CD': 1.27}
        }
        
        # Function to check if ratios match a pattern with tolerance
        def match_pattern(points, pattern, is_bullish):
            # Extract price values at the pivot points
            if is_bullish:
                x, a, b, c, d = [data['low'].iloc[pt] if i % 2 == 0 else data['high'].iloc[pt] 
                              for i, pt in enumerate(points)]
            else:
                x, a, b, c, d = [data['high'].iloc[pt] if i % 2 == 0 else data['low'].iloc[pt] 
                              for i, pt in enumerate(points)]
            
            # Calculate legs and their ratios
            xa = abs(a - x)
            ab = abs(b - a)
            bc = abs(c - b)
            cd = abs(d - c)
            
            # Skip if any leg is too small (noise)
            if min(xa, ab, bc, cd) / data['close'].iloc[points[-1]] < 0.001:
                return False
            
            # Calculate actual ratios
            ab_xa = ab / xa if xa != 0 else float('inf')
            bc_ab = bc / ab if ab != 0 else float('inf')
            cd_bc = cd / bc if bc != 0 else float('inf')
            
            # Get target ratios for this pattern
            target_ratios = harmonic_ratios[pattern]
            tolerance = self.config.harmonic_tolerance
            
            # Check if ratios match within tolerance
            match_ab = abs(ab_xa - target_ratios['AB']) <= tolerance
            match_bc = abs(bc_ab - target_ratios['BC']) <= tolerance
            match_cd = abs(cd_bc - target_ratios['CD']) <= tolerance
            
            return match_ab and match_bc and match_cd
        
        # Find 5-point harmonic patterns
        max_lookback = min(100, len(data) - 1)  # Limit pattern search to last 100 bars
        
        # For each potential D point (the most recent pivot)
        for d_idx in sorted(np.concatenate([highs, lows]))[-20:]:  # Only check recent potential D points
            if d_idx >= len(data) - 1:  # Skip if D is the latest bar (not confirmed yet)
                continue
                
            # Look for potential C points before D
            c_candidates = [idx for idx in np.concatenate([highs, lows]) if idx < d_idx and idx > d_idx - max_lookback]
            
            for c_idx in c_candidates:
                # Look for potential B points before C
                b_candidates = [idx for idx in np.concatenate([highs, lows]) if idx < c_idx and idx > c_idx - max_lookback]
                
                for b_idx in b_candidates:
                    # Look for potential A points before B
                    a_candidates = [idx for idx in np.concatenate([highs, lows]) if idx < b_idx and idx > b_idx - max_lookback]
                    
                    for a_idx in a_candidates:
                        # Look for potential X points before A
                        x_candidates = [idx for idx in np.concatenate([highs, lows]) if idx < a_idx and idx > a_idx - max_lookback]
                        
                        for x_idx in x_candidates:
                            # Check pattern points for alternating high/low
                            points = [x_idx, a_idx, b_idx, c_idx, d_idx]
                            
                            # Check for bullish patterns (X and B are lows, A and C are highs)
                            is_bullish_alignment = (
                                x_idx in lows and a_idx in highs and 
                                b_idx in lows and c_idx in highs and 
                                d_idx in lows
                            )
                            
                            # Check for bearish patterns (X and B are highs, A and C are lows)
                            is_bearish_alignment = (
                                x_idx in highs and a_idx in lows and 
                                b_idx in highs and c_idx in lows and 
                                d_idx in highs
                            )
                            
                            # Skip if pivot alignment doesn't match a potential harmonic pattern
                            if not is_bullish_alignment and not is_bearish_alignment:
                                continue
                            
                            # Check each harmonic pattern
                            for pattern in harmonic_patterns:
                                # For bullish patterns
                                if is_bullish_alignment and match_pattern(points, pattern, True):
                                    data.loc[d_idx:, f'harmonic_{pattern}_bullish'] = 1
                                    
                                # For bearish patterns
                                if is_bearish_alignment and match_pattern(points, pattern, False):
                                    data.loc[d_idx:, f'harmonic_{pattern}_bearish'] = 1
        
        # Calculate overall harmonic pattern presence
        data['harmonic_bullish_count'] = sum((data[f'harmonic_{pat}_bullish'] > 0).astype(int) 
                                            for pat in harmonic_patterns)
        data['harmonic_bearish_count'] = sum((data[f'harmonic_{pat}_bearish'] > 0).astype(int) 
                                           for pat in harmonic_patterns)
        data['harmonic_pattern_strength'] = data['harmonic_bullish_count'] - data['harmonic_bearish_count']
        
        logger.debug("Completed adding harmonic pattern features")
    
    def _add_advanced_chart_patterns(self, data: pd.DataFrame) -> None:
        """
        Add advanced chart pattern features (head and shoulders, double top/bottom, etc.).
        
        Args:
            data: DataFrame to add features to (modified in-place)
        """
        logger.debug("Adding advanced chart pattern recognition features")
        
        # Find local extrema for pattern detection
        highs = argrelextrema(data['high'].values, np.greater, order=self.config.pivot_lookback)[0]
        lows = argrelextrema(data['low'].values, np.less, order=self.config.pivot_lookback)[0]
        
        # Initialize pattern columns
        chart_patterns = [
            'HEAD_AND_SHOULDERS', 'INVERSE_HEAD_AND_SHOULDERS',
            'DOUBLE_TOP', 'DOUBLE_BOTTOM',
            'TRIPLE_TOP', 'TRIPLE_BOTTOM',
            'ASCENDING_TRIANGLE', 'DESCENDING_TRIANGLE',
            'SYMMETRICAL_TRIANGLE', 'RECTANGLE',
            'CUP_AND_HANDLE', 'INVERSE_CUP_AND_HANDLE',
            'ROUNDING_BOTTOM', 'ROUNDING_TOP',
            'FALLING_WEDGE', 'RISING_WEDGE'
        ]
        
        for pattern in chart_patterns:
            data[f'chart_{pattern}'] = 0
        
        # Head and Shoulders pattern detection
        self._detect_head_and_shoulders(data, highs, lows)
        
        # Double/Triple Top/Bottom detection
        self._detect_double_triple_patterns(data, highs, lows)
        
        # Triangle pattern detection
        self._detect_triangle_patterns(data, highs, lows)
        
        # Cup and Handle pattern detection
        self._detect_cup_and_handle(data)
        
        # Rounding patterns
        self._detect_rounding_patterns(data)
        
        # Wedge patterns
        self._detect_wedge_patterns(data, highs, lows)
        
        # Calculate chart pattern strength metrics
        bullish_patterns = ['INVERSE_HEAD_AND_SHOULDERS', 'DOUBLE_BOTTOM', 'TRIPLE_BOTTOM', 
                           'ASCENDING_TRIANGLE', 'CUP_AND_HANDLE', 'ROUNDING_BOTTOM', 'FALLING_WEDGE']
        
        bearish_patterns = ['HEAD_AND_SHOULDERS', 'DOUBLE_TOP', 'TRIPLE_TOP', 
                           'DESCENDING_TRIANGLE', 'INVERSE_CUP_AND_HANDLE', 'ROUNDING_TOP', 'RISING_WEDGE']
        
        data['chart_bullish_count'] = sum((data[f'chart_{pat}'] > 0).astype(int) for pat in bullish_patterns)
        data['chart_bearish_count'] = sum((data[f'chart_{pat}'] > 0).astype(int) for pat in bearish_patterns)
        data['chart_pattern_strength'] = data['chart_bullish_count'] - data['chart_bearish_count']
        
        logger.debug("Completed adding advanced chart pattern features")
    
    def _detect_head_and_shoulders(self, data: pd.DataFrame, highs: np.ndarray, lows: np.ndarray) -> None:
        """
        Detect Head and Shoulders and Inverse Head and Shoulders patterns.
        
        Args:
            data: DataFrame to add features to (modified in-place)
            highs: Indices of local high points
            lows: Indices of local low points
        """
        min_bars = self.config.min_pattern_bars
        
        # Regular Head and Shoulders (bearish)
        for i in range(len(highs) - 2):
            # Need at least 5 pivot points (3 peaks and 2 troughs) with sufficient bars between
            if i + 2 >= len(highs) or highs[i+2] - highs[i] < min_bars:
                continue
                
            # Find shoulder peaks and head
            left_shoulder_idx, head_idx, right_shoulder_idx = highs[i], highs[i+1], highs[i+2]
            
            # Find neckline troughs between shoulders and head
            left_trough = None
            right_trough = None
            
            for low_idx in lows:
                if left_shoulder_idx < low_idx < head_idx:
                    left_trough = low_idx
                elif head_idx < low_idx < right_shoulder_idx:
                    right_trough = low_idx
            
            if left_trough is None or right_trough is None:
                continue
                
            # Get price values
            left_shoulder = data['high'].iloc[left_shoulder_idx]
            head = data['high'].iloc[head_idx]
            right_shoulder = data['high'].iloc[right_shoulder_idx]
            left_trough_val = data['low'].iloc[left_trough]
            right_trough_val = data['low'].iloc[right_trough]
            
            # Check pattern criteria
            if (head > left_shoulder and head > right_shoulder and
                abs(left_shoulder - right_shoulder) / left_shoulder < 0.1 and  # Shoulders roughly equal
                abs(left_trough_val - right_trough_val) / left_trough_val < 0.1):  # Neckline roughly horizontal
                
                # Mark the pattern from first shoulder to last bar
                data.loc[left_shoulder_idx:, 'chart_HEAD_AND_SHOULDERS'] = 1
        
        # Inverse Head and Shoulders (bullish)
        for i in range(len(lows) - 2):
            # Need at least 5 pivot points with sufficient bars between
            if i + 2 >= len(lows) or lows[i+2] - lows[i] < min_bars:
                continue
                
            # Find shoulder troughs and head
            left_shoulder_idx, head_idx, right_shoulder_idx = lows[i], lows[i+1], lows[i+2]
            
            # Find neckline peaks between shoulders and head
            left_peak = None
            right_peak = None
            
            for high_idx in highs:
                if left_shoulder_idx < high_idx < head_idx:
                    left_peak = high_idx
                elif head_idx < high_idx < right_shoulder_idx:
                    right_peak = high_idx
            
            if left_peak is None or right_peak is None:
                continue
                
            # Get price values
            left_shoulder = data['low'].iloc[left_shoulder_idx]
            head = data['low'].iloc[head_idx]
            right_shoulder = data['low'].iloc[right_shoulder_idx]
            left_peak_val = data['high'].iloc[left_peak]
            right_peak_val = data['high'].iloc[right_peak]
            
            # Check pattern criteria
            if (head < left_shoulder and head < right_shoulder and
                abs(left_shoulder - right_shoulder) / left_shoulder < 0.1 and  # Shoulders roughly equal
                abs(left_peak_val - right_peak_val) / left_peak_val < 0.1):  # Neckline roughly horizontal
                
                # Mark the pattern from first shoulder to last bar
                data.loc[left_shoulder_idx:, 'chart_INVERSE_HEAD_AND_SHOULDERS'] = 1
    
    def _detect_double_triple_patterns(self, data: pd.DataFrame, highs: np.ndarray, lows: np.ndarray) -> None:
        """
        Detect Double/Triple Top/Bottom patterns.
        
        Args:
            data: DataFrame to add features to (modified in-place)
            highs: Indices of local high points
            lows: Indices of local low points
        """
        # Double Top detection (bearish)
        for i in range(len(highs) - 1):
            if i + 1 >= len(highs):
                continue
                
            # Get two peaks
            peak1_idx, peak2_idx = highs[i], highs[i+1]
            
            # Find trough between peaks
            trough_idx = None
            for low_idx in lows:
                if peak1_idx < low_idx < peak2_idx:
                    trough_idx = low_idx
                    break
            
            if trough_idx is None:
                continue
                
            # Get price values
            peak1 = data['high'].iloc[peak1_idx]
            peak2 = data['high'].iloc[peak2_idx]
            trough = data['low'].iloc[trough_idx]
            
            # Check pattern criteria
            peak_diff_pct = abs(peak1 - peak2) / peak1
            if peak_diff_pct < 0.05 and peak2_idx - peak1_idx > self.config.min_pattern_bars:
                # Confirm pattern only after price breaks below the trough
                confirmation_idx = next((i for i in range(peak2_idx + 1, len(data)) 
                                        if data['low'].iloc[i] < trough), None)
                if confirmation_idx:
                    data.loc[peak1_idx:confirmation_idx, 'chart_DOUBLE_TOP'] = 1
        
        # Double Bottom detection (bullish)
        for i in range(len(lows) - 1):
            if i + 1 >= len(lows):
                continue
                
            # Get two troughs
            trough1_idx, trough2_idx = lows[i], lows[i+1]
            
            # Find peak between troughs
            peak_idx = None
            for high_idx in highs:
                if trough1_idx < high_idx < trough2_idx:
                    peak_idx = high_idx
                    break
            
            if peak_idx is None:
                continue
                
            # Get price values
            trough1 = data['low'].iloc[trough1_idx]
            trough2 = data['low'].iloc[trough2_idx]
            peak = data['high'].iloc[peak_idx]
            
            # Check pattern criteria
            trough_diff_pct = abs(trough1 - trough2) / trough1
            if trough_diff_pct < 0.05 and trough2_idx - trough1_idx > self.config.min_pattern_bars:
                # Confirm pattern only after price breaks above the peak
                confirmation_idx = next((i for i in range(trough2_idx + 1, len(data)) 
                                        if data['high'].iloc[i] > peak), None)
                if confirmation_idx:
                    data.loc[trough1_idx:confirmation_idx, 'chart_DOUBLE_BOTTOM'] = 1
        
        # Triple Top detection (bearish)
        for i in range(len(highs) - 2):
            if i + 2 >= len(highs):
                continue
                
            # Get three peaks
            peak1_idx, peak2_idx, peak3_idx = highs[i], highs[i+1], highs[i+2]
            
            # Find troughs between peaks
            trough1_idx = None
            trough2_idx = None
            
            for low_idx in lows:
                if peak1_idx < low_idx < peak2_idx:
                    trough1_idx = low_idx
                elif peak2_idx < low_idx < peak3_idx:
                    trough2_idx = low_idx
            
            if trough1_idx is None or trough2_idx is None:
                continue
                
            # Get price values
            peak1 = data['high'].iloc[peak1_idx]
            peak2 = data['high'].iloc[peak2_idx]
            peak3 = data['high'].iloc[peak3_idx]
            trough1 = data['low'].iloc[trough1_idx]
            trough2 = data['low'].iloc[trough2_idx]
            
            # Check pattern criteria
            peak1_2_diff = abs(peak1 - peak2) / peak1
            peak1_3_diff = abs(peak1 - peak3) / peak1
            peak2_3_diff = abs(peak2 - peak3) / peak2
            
            if (peak1_2_diff < 0.05 and peak1_3_diff < 0.05 and peak2_3_diff < 0.05 and
                peak3_idx - peak1_idx > self.config.min_pattern_bars):
                
                # Using the lowest trough for confirmation
                lowest_trough = min(trough1, trough2)
                
                # Confirm pattern only after price breaks below the lowest trough
                confirmation_idx = next((i for i in range(peak3_idx + 1, len(data)) 
                                        if data['low'].iloc[i] < lowest_trough), None)
                if confirmation_idx:
                    data.loc[peak1_idx:confirmation_idx, 'chart_TRIPLE_TOP'] = 1
        
        # Triple Bottom detection (bullish)
        for i in range(len(lows) - 2):
            if i + 2 >= len(lows):
                continue
                
            # Get three troughs
            trough1_idx, trough2_idx, trough3_idx = lows[i], lows[i+1], lows[i+2]
            
            # Find peaks between troughs
            peak1_idx = None
            peak2_idx = None
            
            for high_idx in highs:
                if trough1_idx < high_idx < trough2_idx:
                    peak1_idx = high_idx
                elif trough2_idx < high_idx < trough3_idx:
                    peak2_idx = high_idx
            
            if peak1_idx is None or peak2_idx is None:
                continue
                
            # Get price values
            trough1 = data['low'].iloc[trough1_idx]
            trough2 = data['low'].iloc[trough2_idx]
            trough3 = data['low'].iloc[trough3_idx]
            peak1 = data['high'].iloc[peak1_idx]
            peak2 = data['high'].iloc[peak2_idx]
            
            # Check pattern criteria
            trough1_2_diff = abs(trough1 - trough2) / trough1
            trough1_3_diff = abs(trough1 - trough3) / trough1
            trough2_3_diff = abs(trough2 - trough3) / trough2
            
            if (trough1_2_diff < 0.05 and trough1_3_diff < 0.05 and trough2_3_diff < 0.05 and
                trough3_idx - trough1_idx > self.config.min_pattern_bars):
                
                # Using the highest peak for confirmation
                highest_peak = max(peak1, peak2)
                
                # Confirm pattern only after price breaks above the highest peak
                confirmation_idx = next((i for i in range(trough3_idx + 1, len(data)) 
                                        if data['high'].iloc[i] > highest_peak), None)
                if confirmation_idx:
                    data.loc[trough1_idx:confirmation_idx, 'chart_TRIPLE_BOTTOM'] = 1
    
    def _detect_triangle_patterns(self, data: pd.DataFrame, highs: np.ndarray, lows: np.ndarray) -> None:
        """
        Detect triangle patterns (ascending, descending, symmetrical).
        
        Args:
            data: DataFrame to add features to (modified in-place)
            highs: Indices of local high points
            lows: Indices of local low points
        """
        # Need at least 3 points to form a triangle
        if len(highs) < 3 or len(lows) < 3:
            return
        
        max_pattern_bars = min(self.config.max_pattern_bars, len(data) // 4)
        
        # For potential triangles, look at recent price action
        recent_highs = sorted([h for h in highs if h > len(data) - max_pattern_bars])
        recent_lows = sorted([l for l in lows if l > len(data) - max_pattern_bars])
        
        if len(recent_highs) < 3 or len(recent_lows) < 3:
            return
        
        # Function to check if points form a line with expected slope
        def check_trend(points, prices, expected_dir):
            # Need at least 3 points to confirm a trend
            if len(points) < 3:
                return False
                
            # Calculate linear regression
            y = np.array([prices.iloc[p] for p in points])
            x = np.array(points).reshape(-1, 1)
            
            # Skip if points are too close together
            if np.max(x) - np.min(x) < self.config.min_pattern_bars:
                return False
            
            model = stats.linregress(x.flatten(), y)
            slope = model.slope
            r_squared = model.rvalue ** 2
            
            # Check if slope direction matches expected and fit is decent
            slope_dir = np.sign(slope)
            return (slope_dir == expected_dir and r_squared > 0.7)
        
        # Ascending Triangle: Flat top, rising bottom
        high_points = recent_highs[-3:]  # Last 3 highs
        low_points = recent_lows[-3:]    # Last 3 lows
        
        high_prices = data['high']
        low_prices = data['low']
        
        # Check for horizontal resistance (flat top)
        high_is_flat = check_trend(high_points, high_prices, 0) or abs(np.std([high_prices.iloc[p] for p in high_points]) / np.mean([high_prices.iloc[p] for p in high_points])) < 0.01
        
        # Check for rising support (upward sloping bottom)
        low_is_rising = check_trend(low_points, low_prices, 1)
        
        if high_is_flat and low_is_rising:
            start_idx = min(high_points[0], low_points[0])
            data.loc[start_idx:, 'chart_ASCENDING_TRIANGLE'] = 1
        
        # Descending Triangle: Flat bottom, falling top
        # Check for horizontal support (flat bottom)
        low_is_flat = check_trend(low_points, low_prices, 0) or abs(np.std([low_prices.iloc[p] for p in low_points]) / np.mean([low_prices.iloc[p] for p in low_points])) < 0.01
        
        # Check for falling resistance (downward sloping top)
        high_is_falling = check_trend(high_points, high_prices, -1)
        
        if low_is_flat and high_is_falling:
            start_idx = min(high_points[0], low_points[0])
            data.loc[start_idx:, 'chart_DESCENDING_TRIANGLE'] = 1
        
        # Symmetrical Triangle: Falling tops, rising bottoms
        if high_is_falling and low_is_rising:
            start_idx = min(high_points[0], low_points[0])
            data.loc[start_idx:, 'chart_SYMMETRICAL_TRIANGLE'] = 1
        
        # Rectangle pattern: Both tops and bottoms are flat
        if high_is_flat and low_is_flat:
            start_idx = min(high_points[0], low_points[0])
            data.loc[start_idx:, 'chart_RECTANGLE'] = 1
    
    def _detect_cup_and_handle(self, data: pd.DataFrame) -> None:
        """
        Detect Cup and Handle and Inverse Cup and Handle patterns.
        
        Args:
            data: DataFrame to add features to (modified in-place)
        """
        min_pattern_length = self.config.min_pattern_bars
        max_pattern_length = self.config.max_pattern_bars
        
        # Cup and Handle (bullish)
        for i in range(len(data) - max_pattern_length):
            # Define potential cup region
            cup_start = i
            cup_end = min(i + max_pattern_length, len(data) - 1)
            
            if cup_end - cup_start < min_pattern_length:
                continue
            
            # Get price series for the potential cup
            cup_data = data.iloc[cup_start:cup_end+1]
            
            # Find left rim, bottom, and right rim of cup
            left_rim_idx = cup_start
            right_rim_idx = cup_end
            bottom_idx = cup_data['low'].idxmin()
            
            # Make sure bottom is between rims
            if not (left_rim_idx < bottom_idx < right_rim_idx):
                continue
            
            # Get price values
            left_rim = data['high'].iloc[left_rim_idx]
            right_rim = data['high'].iloc[right_rim_idx]
            bottom = data['low'].iloc[bottom_idx]
            
            # Check cup criteria
            rim_diff_pct = abs(left_rim - right_rim) / left_rim
            cup_depth = (left_rim - bottom) / left_rim
            
            # Cup should have roughly equal rims and significant depth
            if rim_diff_pct > 0.1 or cup_depth < 0.1:
                continue
            
            # Check for handle formation after cup
            handle_start = right_rim_idx
            handle_end = min(handle_start + max_pattern_length // 2, len(data) - 1)
            
            if handle_end - handle_start < min_pattern_length // 2:
                continue
            
            handle_data = data.iloc[handle_start:handle_end+1]
            handle_low = handle_data['low'].min()
            
            # Handle should be a smaller, shallower pullback
            handle_depth = (right_rim - handle_low) / right_rim
            
            if handle_depth > cup_depth or handle_depth < 0.05:
                continue
            
            # Mark the pattern
            data.loc[left_rim_idx:handle_end, 'chart_CUP_AND_HANDLE'] = 1
        
        # Inverse Cup and Handle (bearish)
        for i in range(len(data) - max_pattern_length):
            # Define potential inverse cup region
            cup_start = i
            cup_end = min(i + max_pattern_length, len(data) - 1)
            
            if cup_end - cup_start < min_pattern_length:
                continue
            
            # Get price series for the potential inverse cup
            cup_data = data.iloc[cup_start:cup_end+1]
            
            # Find left rim, top, and right rim of inverse cup
            left_rim_idx = cup_start
            right_rim_idx = cup_end
            top_idx = cup_data['high'].idxmax()
            
            # Make sure top is between rims
            if not (left_rim_idx < top_idx < right_rim_idx):
                continue
            
            # Get price values
            left_rim = data['low'].iloc[left_rim_idx]
            right_rim = data['low'].iloc[right_rim_idx]
            top = data['high'].iloc[top_idx]
            
            # Check cup criteria
            rim_diff_pct = abs(left_rim - right_rim) / left_rim
            cup_height = (top - left_rim) / left_rim
            
            # Inverse cup should have roughly equal rims and significant height
            if rim_diff_pct > 0.1 or cup_height < 0.1:
                continue
            
            # Check for handle formation after inverse cup
            handle_start = right_rim_idx
            handle_end = min(handle_start + max_pattern_length // 2, len(data) - 1)
            
            if handle_end - handle_start < min_pattern_length // 2:
                continue
            
            handle_data = data.iloc[handle_start:handle_end+1]
            handle_high = handle_data['high'].max()
            
            # Handle should be a smaller, shallower rally
            handle_height = (handle_high - right_rim) / right_rim
            
            if handle_height > cup_height or handle_height < 0.05:
                continue
            
            # Mark the pattern
            data.loc[left_rim_idx:handle_end, 'chart_INVERSE_CUP_AND_HANDLE'] = 1
    
    def _detect_rounding_patterns(self, data: pd.DataFrame) -> None:
        """
        Detect Rounding Bottom and Rounding Top patterns.
        
        Args:
            data: DataFrame to add features to (modified in-place)
        """
        min_pattern_length = self.config.min_pattern_bars
        max_pattern_length = self.config.max_pattern_bars
        
        # Function to check for rounding shape using polynomial fit
        def is_rounding(prices, expected_shape):
            # Normalize the x-axis and prices for better fitting
            x = np.linspace(0, 1, len(prices))
            y = (prices - prices.min()) / (prices.max() - prices.min())
            
            # Fit a 2nd degree polynomial (parabola)
            coeffs = np.polyfit(x, y, 2)
            
            # For rounding bottom: a > 0 (upward parabola)
            # For rounding top: a < 0 (downward parabola)
            a = coeffs[0]
            
            # Check if parabola shape matches expected shape and fit is good
            fitted_y = np.polyval(coeffs, x)
            r_squared = 1 - np.sum((y - fitted_y) ** 2) / np.sum((y - y.mean()) ** 2)
            
            if expected_shape == 'bottom':
                return a > 0 and r_squared > 0.7
            else:  # top
                return a < 0 and r_squared > 0.7
        
        # Scan for patterns
        for i in range(len(data) - min_pattern_length):
            for length in range(min_pattern_length, min(max_pattern_length, len(data) - i)):
                pattern_slice = slice(i, i + length)
                
                # Check for Rounding Bottom
                if is_rounding(data.iloc[pattern_slice]['low'], 'bottom'):
                    data.loc[pattern_slice, 'chart_ROUNDING_BOTTOM'] = 1
                
                # Check for Rounding Top
                if is_rounding(data.iloc[pattern_slice]['high'], 'top'):
                    data.loc[pattern_slice, 'chart_ROUNDING_TOP'] = 1
    
    def _detect_wedge_patterns(self, data: pd.DataFrame, highs: np.ndarray, lows: np.ndarray) -> None:
        """
        Detect Rising and Falling Wedge patterns.
        
        Args:
            data: DataFrame to add features to (modified in-place)
            highs: Indices of local high points
            lows: Indices of local low points
        """
        # Need at least 3 points for each line
        if len(highs) < 3 or len(lows) < 3:
            return
        
        max_pattern_bars = min(self.config.max_pattern_bars, len(data) // 4)
        
        # Focus on recent price action
        recent_highs = sorted([h for h in highs if h > len(data) - max_pattern_bars])
        recent_lows = sorted([l for l in lows if l > len(data) - max_pattern_bars])
        
        if len(recent_highs) < 3 or len(recent_lows) < 3:
            return
        
        # Calculate linear regression for highs and lows
        high_prices = np.array([data['high'].iloc[h] for h in recent_highs])
        low_prices = np.array([data['low'].iloc[l] for l in recent_lows])
        high_indices = np.array(recent_highs).reshape(-1, 1)
        low_indices = np.array(recent_lows).reshape(-1, 1)
        
        high_model = stats.linregress(high_indices.flatten(), high_prices)
        low_model = stats.linregress(low_indices.flatten(), low_prices)
        
        high_slope = high_model.slope
        low_slope = low_model.slope
        
        # Check for wedge patterns
        
        # Rising Wedge (bearish): both lines slope up, but lows rise faster than highs
        if high_slope > 0 and low_slope > 0 and low_slope > high_slope:
            start_idx = min(recent_highs[0], recent_lows[0])
            data.loc[start_idx:, 'chart_RISING_WEDGE'] = 1
        
        # Falling Wedge (bullish): both lines slope down, but highs fall faster than lows
        if high_slope < 0 and low_slope < 0 and high_slope < low_slope:
            start_idx = min(recent_highs[0], recent_lows[0])
            data.loc[start_idx:, 'chart_FALLING_WEDGE'] = 1
    
    def _add_pattern_dna_features(self, data: pd.DataFrame) -> None:
        """
        Add pattern DNA sequencing features that identify unique price action signatures.
        
        Args:
            data: DataFrame to add features to (modified in-place)
        """
        logger.debug("Adding pattern DNA sequencing features")
        
        # Calculate price movement sequences (DNA)
        # 1: strong up, 2: weak up, 0: flat, -1: weak down, -2: strong down
        volatility = data['high'].rolling(10).max() - data['low'].rolling(10).min()
        avg_volatility = volatility.mean()
        
        # Calculate daily changes
        close_change = data['close'].diff()
        
        # Normalize by recent volatility
        normalized_change = close_change / avg_volatility
        
        # Create DNA codes
        dna_codes = pd.Series(index=data.index, dtype='int')
        
        dna_codes[normalized_change > 0.5] = 2  # Strong up
        dna_codes[(normalized_change > 0) & (normalized_change <= 0.5)] = 1  # Weak up
        dna_codes[(normalized_change >= -0.1) & (normalized_change <= 0.1)] = 0  # Flat
        dna_codes[(normalized_change < 0) & (normalized_change >= -0.5)] = -1  # Weak down
        dna_codes[normalized_change < -0.5] = -2  # Strong down
        
        # Create DNA sequences (last 5 days)
        for i in range(5, len(data)):
            dna_seq = ''.join([str(int(dna_codes.iloc[j])) for j in range(i-5, i)])
            data.loc[data.index[i], 'pattern_dna_seq'] = dna_seq
        
        # Add pattern DNA success history if available
        if self.pattern_success_rates:
            data['pattern_dna_success_rate'] = np.nan
            
            for i in range(5, len(data)):
                dna_seq = data.loc[data.index[i], 'pattern_dna_seq']
                if dna_seq in self.pattern_success_rates:
                    data.loc[data.index[i], 'pattern_dna_success_rate'] = self.pattern_success_rates[dna_seq]
        
        logger.debug("Completed adding pattern DNA features")
    
    def _add_pattern_convergence_features(self, data: pd.DataFrame) -> None:
        """
        Add multi-timeframe pattern convergence features.
        
        Args:
            data: DataFrame to add features to (modified in-place)
        """
        # This is a simplified implementation - in production, this would use
        # actual multi-timeframe data with resampling
        
        logger.debug("Adding pattern convergence features")
        
        # Create dummy multi-timeframe pattern columns for demonstration
        # In production, this would use actual multi-timeframe pattern detection
        
        # Simulate higher timeframe signals by smoothing current signals
        data['mt_bullish_pattern'] = (data['pattern_bullish_count'] > 0).rolling(5).mean() > 0.6
        data['mt_bearish_pattern'] = (data['pattern_bearish_count'] > 0).rolling(5).mean() > 0.6
        
        # Calculate convergence (patterns aligned across timeframes)
        has_current_bullish = data['pattern_bullish_count'] > 0
        has_current_bearish = data['pattern_bearish_count'] > 0
        
        data['pattern_bullish_convergence'] = (has_current_bullish & data['mt_bullish_pattern']).astype(int)
        data['pattern_bearish_convergence'] = (has_current_bearish & data['mt_bearish_pattern']).astype(int)
        
        # Clean up temporary columns
        data.drop(['mt_bullish_pattern', 'mt_bearish_pattern'], axis=1, inplace=True)
        
        logger.debug("Completed adding pattern convergence features")
    
    def _add_pattern_completion_features(self, data: pd.DataFrame) -> None:
        """
        Add pattern completion percentage and target projection features.
        
        Args:
            data: DataFrame to add features to (modified in-place)
        """
        logger.debug("Adding pattern completion features")
        
        # Patterns that typically have measurable completion targets
        target_patterns = ['harmonic_GARTLEY_bullish', 'harmonic_GARTLEY_bearish',
                          'harmonic_BUTTERFLY_bullish', 'harmonic_BUTTERFLY_bearish',
                          'harmonic_BAT_bullish', 'harmonic_BAT_bearish',
                          'chart_HEAD_AND_SHOULDERS', 'chart_INVERSE_HEAD_AND_SHOULDERS']
        
        # Initialize completion features
        data['pattern_completion_pct'] = 0.0
        data['pattern_target_distance'] = 0.0
        
        # Process each pattern
        for pattern in target_patterns:
            if pattern not in data.columns:
                continue
                
            # For each active pattern
            pattern_starts = data.index[data[pattern].diff() == 1]
            
            for start_idx in pattern_starts:
                start_pos = data.index.get_loc(start_idx)
                
                # Find where pattern ends (signal goes back to 0)
                # For this simplified example, assume pattern stays active until overridden
                pattern_range = data.iloc[start_pos:][pattern] > 0
                if not pattern_range.any():
                    continue
                    
                pattern_length = pattern_range.sum()
                
                # Skip patterns that are too short
                if pattern_length < self.config.min_pattern_bars:
                    continue
                    
                # Calculate theoretical pattern target
                # This is a simple implementation - actual targets would depend on pattern specifics
                
                if 'harmonic_' in pattern:
                    # For harmonic patterns, project based on XA distance
                    start_price = data.iloc[start_pos]['close']
                    
                    # Simplified target calculation
                    target_move = 0.0
                    
                    if 'bullish' in pattern:
                        # Bullish harmonic targets are typically upward projections
                        target_move = data.iloc[start_pos:start_pos+pattern_length]['high'].max() - start_price
                    else:
                        # Bearish harmonic targets are typically downward projections
                        target_move = start_price - data.iloc[start_pos:start_pos+pattern_length]['low'].min()
                        
                    # Apply pattern-specific multipliers (simplified)
                    if 'GARTLEY' in pattern:
                        target_price = start_price + (target_move * 1.272) if 'bullish' in pattern else start_price - (target_move * 1.272)
                    elif 'BUTTERFLY' in pattern:
                        target_price = start_price + (target_move * 1.618) if 'bullish' in pattern else start_price - (target_move * 1.618)
                    elif 'BAT' in pattern:
                        target_price = start_price + (target_move * 1.618) if 'bullish' in pattern else start_price - (target_move * 1.618)
                    else:
                        # Default target
                        target_price = start_price + (target_move * 1.0) if 'bullish' in pattern else start_price - (target_move * 1.0)
                        
                elif 'HEAD_AND_SHOULDERS' in pattern:
                    # For H&S patterns, target is typically the height of head to neckline
                    if 'INVERSE' in pattern:
                        # Bullish Inverse H&S
                        neckline = data.iloc[start_pos:start_pos+pattern_length]['high'].max()
                        head = data.iloc[start_pos:start_pos+pattern_length]['low'].min()
                        height = neckline - head
                        target_price = neckline + height
                    else:
                        # Bearish H&S
                        neckline = data.iloc[start_pos:start_pos+pattern_length]['low'].min()
                        head = data.iloc[start_pos:start_pos+pattern_length]['high'].max()
                        height = head - neckline
                        target_price = neckline - height
                else:
                    # Default generic target calculation
                    pattern_high = data.iloc[start_pos:start_pos+pattern_length]['high'].max()
                    pattern_low = data.iloc[start_pos:start_pos+pattern_length]['low'].min()
                    pattern_height = pattern_high - pattern_low
                    
                    if any(bullish in pattern for bullish in ['INVERSE', 'BOTTOM', 'bullish']):
                        target_price = pattern_high + pattern_height
                    else:
                        target_price = pattern_low - pattern_height
                
                # Calculate completion percentage and distance to target
                end_pos = min(start_pos + pattern_length, len(data) - 1)
                
                for i in range(start_pos, end_pos + 1):
                    current_price = data.iloc[i]['close']
                    start_to_target = abs(target_price - data.iloc[start_pos]['close'])
                    
                    # Avoid division by zero
                    if start_to_target > 0:
                        current_to_target = abs(target_price - current_price)
                        completion = 1.0 - (current_to_target / start_to_target)
                        data.loc[data.index[i], 'pattern_completion_pct'] = max(
                            data.loc[data.index[i], 'pattern_completion_pct'],
                            min(1.0, completion)  # Cap at 100%
                        )
                    
                    # Distance to target as percentage of current price
                    target_distance = abs(target_price - current_price) / current_price
                    data.loc[data.index[i], 'pattern_target_distance'] = target_distance
        
        logger.debug("Completed adding pattern completion features")
    
    def _add_historical_pattern_similarity(self, data: pd.DataFrame) -> None:
        """
        Add pattern similarity to historical high-success patterns.
        
        Args:
            data: DataFrame to add features to (modified in-place)
        """
        logger.debug("Adding historical pattern similarity features")
        
        # In a real implementation, this would compare current patterns to a database
        # of historical patterns with known success rates
        
        # Simplified implementation using recent pattern history in the current data
        
        # Initialize similarity feature
        data['pattern_historical_similarity'] = 0.0
        
        # Use a rolling window to find pattern similarities
        window_size = 10  # Size of pattern to compare
        
        # Skip if we don't have enough data
        if len(data) < window_size * 2:
            return
            
        # Function to calculate pattern similarity
        def pattern_similarity(window1, window2):
            # Normalize both windows
            norm1 = (window1 - window1.mean()) / (window1.std() if window1.std() > 0 else 1)
            norm2 = (window2 - window2.mean()) / (window2.std() if window2.std() > 0 else 1)
            
            # Calculate correlation coefficient
            corr = np.corrcoef(norm1, norm2)[0, 1]
            return max(0, corr)  # Only consider positive correlations
        
        # For each position in the data
        for i in range(window_size, len(data)):
            current_window = data['close'].iloc[i - window_size:i].values
            
            # Compare to historical windows
            max_similarity = 0.0
            
            # Look back at historical patterns (simple approach)
            for j in range(window_size, i - window_size):
                hist_window = data['close'].iloc[j - window_size:j].values
                
                # Calculate similarity
                similarity = pattern_similarity(current_window, hist_window)
                
                # Update if this is the most similar pattern
                if similarity > max_similarity and similarity >= self.config.similarity_threshold:
                    max_similarity = similarity
            
            # Store the highest similarity
            data.loc[data.index[i], 'pattern_historical_similarity'] = max_similarity
        
        logger.debug("Completed adding historical pattern similarity features")
    
    def _add_pattern_success_probability(self, data: pd.DataFrame) -> None:
        """
        Add pattern success probability estimates based on historical performance.
        
        Args:
            data: DataFrame to add features to (modified in-place)
        """
        logger.debug("Adding pattern success probability features")
        
        # Initialize success probability feature
        data['pattern_success_probability'] = 0.0
        
        # In a real implementation, this would use actual historical pattern success rates
        # from a database of past patterns and outcomes
        
        # Example success rates (these would come from actual historical analysis)
        example_success_rates = {
            'harmonic_GARTLEY_bullish': 0.78,
            'harmonic_BUTTERFLY_bullish': 0.82,
            'harmonic_BAT_bullish': 0.75,
            'harmonic_CRAB_bullish': 0.85,
            'harmonic_GARTLEY_bearish': 0.76,
            'harmonic_BUTTERFLY_bearish': 0.80,
            'harmonic_BAT_bearish': 0.73,
            'harmonic_CRAB_bearish': 0.82,
            'chart_HEAD_AND_SHOULDERS': 0.65,
            'chart_INVERSE_HEAD_AND_SHOULDERS': 0.67,
            'chart_DOUBLE_TOP': 0.70,
            'chart_DOUBLE_BOTTOM': 0.72,
            'chart_ASCENDING_TRIANGLE': 0.76,
            'chart_DESCENDING_TRIANGLE': 0.74,
            'chart_CUP_AND_HANDLE': 0.79,
            'chart_FALLING_WEDGE': 0.77,
            'chart_RISING_WEDGE': 0.71
        }
        
        # Extract pattern columns
        pattern_cols = [col for col in data.columns if col.startswith('harmonic_') or col.startswith('chart_')]
        
        # For each row, calculate the weighted success probability
        for i in range(len(data)):
            active_patterns = [col for col in pattern_cols if data.iloc[i][col] > 0]
            
            if not active_patterns:
                continue
                
            # Calculate weighted probability
            total_weight = 0
            weighted_prob = 0
            
            for pattern in active_patterns:
                if pattern in example_success_rates:
                    # Get base success rate
                    success_rate = example_success_rates[pattern]
                    
                    # Adjust based on pattern strength if available
                    strength = 1.0  # Default strength
                    
                    if 'pattern_completion_pct' in data.columns:
                        completion = data.iloc[i]['pattern_completion_pct']
                        # Patterns closer to completion have higher probability
                        strength = 0.5 + (0.5 * completion)
                    
                    weight = 1.0  # Default equal weighting
                    weighted_prob += success_rate * weight * strength
                    total_weight += weight
            
            # Calculate final probability if we have weights
            if total_weight > 0:
                final_prob = weighted_prob / total_weight
                
                # Apply historical similarity bonus if available
                if 'pattern_historical_similarity' in data.columns:
                    similarity = data.iloc[i]['pattern_historical_similarity']
                    if similarity > 0:
                        # Boost probability based on similarity to successful historical patterns
                        final_prob = final_prob * (1.0 + (similarity * 0.2))
                
                # Cap at 100%
                data.loc[data.index[i], 'pattern_success_probability'] = min(1.0, final_prob)
        
        logger.debug("Completed adding pattern success probability features")
    
    def update_pattern_success_rates(self, pattern_success_data: Dict[str, float]) -> None:
        """
        Update the internal database of pattern success rates based on observed outcomes.
        
        Args:
            pattern_success_data: Dictionary mapping pattern identifiers to success rates
        """
        logger.info("Updating pattern success rates with %d patterns", len(pattern_success_data))
        self.pattern_success_rates.update(pattern_success_data)


def detect_patterns(data: pd.DataFrame, config: Optional[PatternConfiguration] = None) -> pd.DataFrame:
    """Convenience wrapper returning pattern features for the given data."""
    pf = PatternFeatures(config)
    return pf.calculate_features(data)


def detect_rectangle_pattern(data: pd.DataFrame, lookback: int = 20) -> List[Dict[str, Any]]:
    """Detect simple rectangle consolidation patterns."""
    if len(data) < lookback:
        return []

    highs = data['high'].rolling(lookback).max()
    lows = data['low'].rolling(lookback).min()
    range_pct = (highs - lows) / data['close']

    patterns: List[Dict[str, Any]] = []
    in_pattern = False
    start = None

    for idx, flag in range_pct.lt(0.02).items():
        if flag and not in_pattern:
            in_pattern = True
            start = idx
        elif not flag and in_pattern:
            patterns.append({
                'start': start,
                'end': idx,
                'high': float(highs.loc[idx]),
                'low': float(lows.loc[idx]),
            })
            in_pattern = False

    if in_pattern:
        patterns.append({
            'start': start,
            'end': data.index[-1],
            'high': float(highs.iloc[-1]),
            'low': float(lows.iloc[-1]),
        })

    return patterns


def detect_triangle_pattern(data: pd.DataFrame, lookback: int = 20) -> List[Dict[str, Any]]:
    """Detect basic triangle patterns based on converging highs and lows."""
    if len(data) < lookback:
        return []

    highs = data['high']
    lows = data['low']
    slope_high = highs.diff().rolling(lookback).mean()
    slope_low = lows.diff().rolling(lookback).mean()

    patterns: List[Dict[str, Any]] = []
    in_pattern = False
    start = None

    for i in range(lookback, len(data)):
        decreasing_high = slope_high.iloc[i] < 0
        increasing_low = slope_low.iloc[i] > 0
        if decreasing_high and increasing_low:
            if not in_pattern:
                start = data.index[i - lookback + 1]
                in_pattern = True
        else:
            if in_pattern:
                patterns.append({'start': start, 'end': data.index[i]})
                in_pattern = False

    if in_pattern:
        patterns.append({'start': start, 'end': data.index[-1]})

    return patterns

