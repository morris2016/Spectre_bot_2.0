#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Harmonic Pattern Recognition Module

This module implements advanced harmonic pattern recognition algorithms for the
QuantumSpectre Elite Trading System. It identifies a comprehensive set of harmonic
patterns including Gartley, Bat, Butterfly, Crab, Shark, Cypher, and others
with high precision and configurable tolerances.

Features:
- Multi-timeframe pattern detection
- Dynamic Fibonacci ratio tolerance adjustments
- Pattern quality scoring and filtering
- Pattern completion projection
- Historical success rate tracking
- Pattern mutation detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Set, Any
from dataclasses import dataclass
import math
from enum import Enum
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import argrelextrema

# Internal imports
from common.logger import get_logger
from common.utils import TimeFrame, get_current_time, calculate_distance_percentage
from common.exceptions import PatternRecognitionError
from feature_service.features.market_structure import MarketStructureAnalyzer
from data_storage.models.market_data import PatternOccurrence, PatternMetrics

logger = get_logger(__name__)

class HarmonicPatternType(Enum):
    """Enumeration of supported harmonic pattern types."""
    GARTLEY = "Gartley"
    BAT = "Bat" 
    BUTTERFLY = "Butterfly"
    CRAB = "Crab"
    SHARK = "Shark"
    CYPHER = "Cypher"
    THREE_DRIVES = "Three Drives"
    ABCD = "ABCD"
    FIVE_ZERO = "5-0"
    DEEP_CRAB = "Deep Crab"
    ALT_BUTTERFLY = "Alt Butterfly"
    ALT_BAT = "Alt Bat"
    WHITE_SWAN = "White Swan"
    BLACK_SWAN = "Black Swan"

class HarmonicDirection(Enum):
    """Pattern direction enumeration."""
    BULLISH = "bullish"
    BEARISH = "bearish"

@dataclass
class FibonacciRatio:
    """Fibonacci ratio with tolerance range."""
    ideal: float
    min_tolerance: float
    max_tolerance: float
    
    def is_valid(self, value: float) -> bool:
        """Check if a value falls within the ratio's tolerance range."""
        return self.min_tolerance <= value <= self.max_tolerance
    
    def deviation(self, value: float) -> float:
        """Calculate percentage deviation from ideal value."""
        if value < self.min_tolerance:
            return (value - self.min_tolerance) / self.ideal * 100
        elif value > self.max_tolerance:
            return (value - self.max_tolerance) / self.ideal * 100
        return (value - self.ideal) / self.ideal * 100

@dataclass
class HarmonicPattern:
    """Harmonic pattern configuration with Fibonacci ratios."""
    pattern_type: HarmonicPatternType
    xab_ratio: FibonacciRatio
    abc_ratio: FibonacciRatio
    bcd_ratio: FibonacciRatio
    xad_ratio: FibonacciRatio
    description: str = ""
    success_rate: Dict[str, float] = None
    
    def __post_init__(self):
        if self.success_rate is None:
            self.success_rate = {"overall": 0.0, "bullish": 0.0, "bearish": 0.0}

@dataclass
class PatternPoint:
    """A point in a harmonic pattern."""
    index: int
    price: float
    timestamp: int
    
@dataclass
class PatternInstance:
    """Representation of a detected harmonic pattern."""
    pattern_type: HarmonicPatternType
    direction: HarmonicDirection
    points: Dict[str, PatternPoint]
    quality_score: float
    confidence: float
    completion_timestamp: int
    potential_reversal_zone: Tuple[float, float]
    expected_move_magnitude: float
    timeframe: TimeFrame
    asset: str
    pattern_id: str = ""

class HarmonicPatternRecognizer:
    """
    Advanced harmonic pattern recognition engine.
    
    This class identifies harmonic patterns in price data across multiple
    timeframes with configurable tolerances and quality filtering.
    """
    
    def __init__(self, 
                 tolerance_multiplier: float = 1.0,
                 min_quality_score: float = 0.7,
                 min_leg_size_percent: float = 1.0,
                 max_age_bars: int = 100,
                 pattern_history_size: int = 1000,
                 enable_auto_adaptation: bool = True,
                 enable_parallel_processing: bool = True):
        """
        Initialize the harmonic pattern recognizer.
        
        Args:
            tolerance_multiplier: Multiplier for Fibonacci ratio tolerances
            min_quality_score: Minimum quality score for pattern acceptance
            min_leg_size_percent: Minimum size of pattern legs as percentage of price
            max_age_bars: Maximum age of a pattern in bars
            pattern_history_size: Number of patterns to keep in history
            enable_auto_adaptation: Enable automatic adaptation based on success rates
            enable_parallel_processing: Enable parallel processing of timeframes
        """
        self.tolerance_multiplier = tolerance_multiplier
        self.min_quality_score = min_quality_score
        self.min_leg_size_percent = min_leg_size_percent
        self.max_age_bars = max_age_bars
        self.pattern_history_size = pattern_history_size
        self.enable_auto_adaptation = enable_auto_adaptation
        self.enable_parallel_processing = enable_parallel_processing
        
        # Initialize pattern configurations
        self.patterns = self._initialize_patterns()
        
        # Pattern history for tracking and learning
        self.pattern_history: List[PatternInstance] = []
        
        # Market structure analyzer for swing point detection
        self.market_structure = MarketStructureAnalyzer()
        
        # Performance metrics
        self.metrics = {
            "pattern_counts": {pattern.name: 0 for pattern in HarmonicPatternType},
            "success_rates": {pattern.name: 0.0 for pattern in HarmonicPatternType},
            "processing_times": []
        }
        
        logger.info(f"Harmonic pattern recognizer initialized with tolerance={tolerance_multiplier}, "
                   f"min_quality={min_quality_score}, min_leg_size={min_leg_size_percent}%")
    
    def _initialize_patterns(self) -> Dict[HarmonicPatternType, HarmonicPattern]:
        """Initialize the configuration for all supported harmonic patterns."""
        patterns = {}
        
        # Multiplier for tolerance adjustments
        tm = self.tolerance_multiplier
        
        # Gartley Pattern (XABCD)
        patterns[HarmonicPatternType.GARTLEY] = HarmonicPattern(
            pattern_type=HarmonicPatternType.GARTLEY,
            xab_ratio=FibonacciRatio(0.618, 0.618 - 0.05 * tm, 0.618 + 0.05 * tm),
            abc_ratio=FibonacciRatio(0.382, 0.382 - 0.05 * tm, 0.886 + 0.05 * tm),
            bcd_ratio=FibonacciRatio(1.272, 1.272 - 0.10 * tm, 1.618 + 0.10 * tm),
            xad_ratio=FibonacciRatio(0.786, 0.786 - 0.05 * tm, 0.786 + 0.05 * tm),
            description="The Gartley pattern is characterized by a 0.618 XAB retracement and a 0.786 XAD retracement."
        )
        
        # Bat Pattern (XABCD)
        patterns[HarmonicPatternType.BAT] = HarmonicPattern(
            pattern_type=HarmonicPatternType.BAT,
            xab_ratio=FibonacciRatio(0.50, 0.382 - 0.05 * tm, 0.50 + 0.05 * tm),
            abc_ratio=FibonacciRatio(0.382, 0.382 - 0.05 * tm, 0.886 + 0.05 * tm),
            bcd_ratio=FibonacciRatio(1.618, 1.618 - 0.10 * tm, 2.618 + 0.10 * tm),
            xad_ratio=FibonacciRatio(0.886, 0.886 - 0.05 * tm, 0.886 + 0.05 * tm),
            description="The Bat pattern is characterized by a 0.50 XAB retracement and a 0.886 XAD retracement."
        )
        
        # Butterfly Pattern (XABCD)
        patterns[HarmonicPatternType.BUTTERFLY] = HarmonicPattern(
            pattern_type=HarmonicPatternType.BUTTERFLY,
            xab_ratio=FibonacciRatio(0.786, 0.786 - 0.05 * tm, 0.786 + 0.05 * tm),
            abc_ratio=FibonacciRatio(0.382, 0.382 - 0.05 * tm, 0.886 + 0.05 * tm),
            bcd_ratio=FibonacciRatio(1.618, 1.618 - 0.10 * tm, 2.618 + 0.10 * tm),
            xad_ratio=FibonacciRatio(1.27, 1.27 - 0.10 * tm, 1.618 + 0.10 * tm),
            description="The Butterfly pattern is characterized by a 0.786 XAB retracement and a 1.27-1.618 XAD extension."
        )
        
        # Crab Pattern (XABCD)
        patterns[HarmonicPatternType.CRAB] = HarmonicPattern(
            pattern_type=HarmonicPatternType.CRAB,
            xab_ratio=FibonacciRatio(0.382, 0.382 - 0.05 * tm, 0.618 + 0.05 * tm),
            abc_ratio=FibonacciRatio(0.382, 0.382 - 0.05 * tm, 0.886 + 0.05 * tm),
            bcd_ratio=FibonacciRatio(2.618, 2.618 - 0.20 * tm, 3.618 + 0.20 * tm),
            xad_ratio=FibonacciRatio(1.618, 1.618 - 0.10 * tm, 1.618 + 0.10 * tm),
            description="The Crab pattern is characterized by a 0.382-0.618 XAB retracement and a 1.618 XAD extension."
        )
        
        # Deep Crab Pattern (XABCD)
        patterns[HarmonicPatternType.DEEP_CRAB] = HarmonicPattern(
            pattern_type=HarmonicPatternType.DEEP_CRAB,
            xab_ratio=FibonacciRatio(0.886, 0.886 - 0.05 * tm, 0.886 + 0.05 * tm),
            abc_ratio=FibonacciRatio(0.382, 0.382 - 0.05 * tm, 0.886 + 0.05 * tm),
            bcd_ratio=FibonacciRatio(2.618, 2.618 - 0.20 * tm, 3.618 + 0.20 * tm),
            xad_ratio=FibonacciRatio(1.618, 1.618 - 0.10 * tm, 1.618 + 0.10 * tm),
            description="The Deep Crab pattern is a variation of the Crab with a deeper 0.886 XAB retracement."
        )
        
        # Shark Pattern (XABCD)
        patterns[HarmonicPatternType.SHARK] = HarmonicPattern(
            pattern_type=HarmonicPatternType.SHARK,
            xab_ratio=FibonacciRatio(0.50, 0.50 - 0.05 * tm, 0.618 + 0.05 * tm),
            abc_ratio=FibonacciRatio(1.13, 1.13 - 0.10 * tm, 1.618 + 0.10 * tm),
            bcd_ratio=FibonacciRatio(1.618, 1.618 - 0.10 * tm, 2.24 + 0.10 * tm),
            xad_ratio=FibonacciRatio(0.886, 0.886 - 0.05 * tm, 1.13 + 0.05 * tm),
            description="The Shark pattern is characterized by a 0.5-0.618 XAB retracement and a 0.886-1.13 XAD retracement/extension."
        )
        
        # Cypher Pattern (XABCD)
        patterns[HarmonicPatternType.CYPHER] = HarmonicPattern(
            pattern_type=HarmonicPatternType.CYPHER,
            xab_ratio=FibonacciRatio(0.382, 0.382 - 0.05 * tm, 0.618 + 0.05 * tm),
            abc_ratio=FibonacciRatio(1.13, 1.13 - 0.10 * tm, 1.414 + 0.10 * tm),
            bcd_ratio=FibonacciRatio(1.272, 1.272 - 0.10 * tm, 2.0 + 0.10 * tm),
            xad_ratio=FibonacciRatio(0.786, 0.786 - 0.05 * tm, 0.786 + 0.05 * tm),
            description="The Cypher pattern is characterized by a 0.382-0.618 XAB retracement and a 0.786 XAD retracement."
        )
        
        # Three Drives Pattern
        patterns[HarmonicPatternType.THREE_DRIVES] = HarmonicPattern(
            pattern_type=HarmonicPatternType.THREE_DRIVES,
            xab_ratio=FibonacciRatio(0.618, 0.618 - 0.05 * tm, 0.618 + 0.05 * tm),  # Drive 1 retracement
            abc_ratio=FibonacciRatio(1.272, 1.272 - 0.10 * tm, 1.272 + 0.10 * tm),  # Drive 2
            bcd_ratio=FibonacciRatio(0.786, 0.786 - 0.05 * tm, 0.786 + 0.05 * tm),  # Drive 2 retracement
            xad_ratio=FibonacciRatio(1.618, 1.618 - 0.10 * tm, 1.618 + 0.10 * tm),  # Drive 3
            description="The Three Drives pattern consists of three consecutive drives with Fibonacci relationships."
        )
        
        # ABCD Pattern
        patterns[HarmonicPatternType.ABCD] = HarmonicPattern(
            pattern_type=HarmonicPatternType.ABCD,
            xab_ratio=FibonacciRatio(0.0, 0.0, 0.0),  # Not used for ABCD
            abc_ratio=FibonacciRatio(0.618, 0.618 - 0.05 * tm, 0.618 + 0.05 * tm),  # AB=CD ratio
            bcd_ratio=FibonacciRatio(1.27, 1.27 - 0.10 * tm, 1.618 + 0.10 * tm),  # BC projection
            xad_ratio=FibonacciRatio(0.0, 0.0, 0.0),  # Not used for ABCD
            description="The ABCD pattern uses equal leg measurements where AB=CD within Fibonacci relationships."
        )
        
        # 5-0 Pattern
        patterns[HarmonicPatternType.FIVE_ZERO] = HarmonicPattern(
            pattern_type=HarmonicPatternType.FIVE_ZERO,
            xab_ratio=FibonacciRatio(1.13, 1.13 - 0.10 * tm, 1.618 + 0.10 * tm),
            abc_ratio=FibonacciRatio(1.618, 1.618 - 0.10 * tm, 2.24 + 0.10 * tm),
            bcd_ratio=FibonacciRatio(0.50, 0.50 - 0.05 * tm, 0.50 + 0.05 * tm),
            xad_ratio=FibonacciRatio(0.0, 0.0, 0.0),  # Calculated differently
            description="The 5-0 pattern is characterized by an initial extension followed by a 0.50 retracement to the origin."
        )
        
        # Alt Butterfly Pattern
        patterns[HarmonicPatternType.ALT_BUTTERFLY] = HarmonicPattern(
            pattern_type=HarmonicPatternType.ALT_BUTTERFLY,
            xab_ratio=FibonacciRatio(0.382, 0.382 - 0.05 * tm, 0.50 + 0.05 * tm),
            abc_ratio=FibonacciRatio(1.13, 1.13 - 0.10 * tm, 1.618 + 0.10 * tm),
            bcd_ratio=FibonacciRatio(1.618, 1.618 - 0.10 * tm, 2.24 + 0.10 * tm),
            xad_ratio=FibonacciRatio(1.27, 1.27 - 0.10 * tm, 1.41 + 0.10 * tm),
            description="The Alt Butterfly pattern is a variation with a shallower B point retracement."
        )
        
        # Alt Bat Pattern
        patterns[HarmonicPatternType.ALT_BAT] = HarmonicPattern(
            pattern_type=HarmonicPatternType.ALT_BAT,
            xab_ratio=FibonacciRatio(0.382, 0.382 - 0.05 * tm, 0.382 + 0.05 * tm),
            abc_ratio=FibonacciRatio(0.382, 0.382 - 0.05 * tm, 0.886 + 0.05 * tm),
            bcd_ratio=FibonacciRatio(2.0, 2.0 - 0.15 * tm, 2.618 + 0.15 * tm),
            xad_ratio=FibonacciRatio(1.13, 1.13 - 0.10 * tm, 1.13 + 0.10 * tm),
            description="The Alt Bat pattern is a variation with a deeper D point extension."
        )
        
        # Swan Patterns
        patterns[HarmonicPatternType.WHITE_SWAN] = HarmonicPattern(
            pattern_type=HarmonicPatternType.WHITE_SWAN,
            xab_ratio=FibonacciRatio(0.236, 0.236 - 0.05 * tm, 0.236 + 0.05 * tm),
            abc_ratio=FibonacciRatio(1.618, 1.618 - 0.10 * tm, 1.618 + 0.10 * tm),
            bcd_ratio=FibonacciRatio(1.618, 1.618 - 0.10 * tm, 1.618 + 0.10 * tm),
            xad_ratio=FibonacciRatio(1.618, 1.618 - 0.10 * tm, 1.618 + 0.10 * tm),
            description="The White Swan (bullish) pattern is characterized by shallow retracements and strong extensions."
        )
        
        patterns[HarmonicPatternType.BLACK_SWAN] = HarmonicPattern(
            pattern_type=HarmonicPatternType.BLACK_SWAN,
            xab_ratio=FibonacciRatio(0.236, 0.236 - 0.05 * tm, 0.236 + 0.05 * tm),
            abc_ratio=FibonacciRatio(1.618, 1.618 - 0.10 * tm, 1.618 + 0.10 * tm),
            bcd_ratio=FibonacciRatio(1.618, 1.618 - 0.10 * tm, 1.618 + 0.10 * tm),
            xad_ratio=FibonacciRatio(1.618, 1.618 - 0.10 * tm, 1.618 + 0.10 * tm),
            description="The Black Swan (bearish) pattern is characterized by shallow retracements and strong extensions."
        )
        
        return patterns

    def recognize_patterns(self, 
                          df: pd.DataFrame, 
                          asset: str, 
                          timeframe: TimeFrame, 
                          pattern_types: Optional[List[HarmonicPatternType]] = None) -> List[PatternInstance]:
        """
        Identify harmonic patterns in the provided price data.
        
        Args:
            df: DataFrame with OHLCV data
            asset: Asset name/symbol
            timeframe: Timeframe of the provided data
            pattern_types: Optional list of pattern types to search for (default: all)
            
        Returns:
            List of identified pattern instances
        """
        start_time = time.time()
        
        try:
            if pattern_types is None:
                pattern_types = list(HarmonicPatternType)
            
            # Validate data
            required_columns = ['open', 'high', 'low', 'close', 'timestamp']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                raise PatternRecognitionError(f"Missing required columns in dataframe: {missing}")
            
            if len(df) < 100:
                logger.warning(f"Dataframe contains less than 100 bars ({len(df)}). Pattern recognition may be unreliable.")
            
            # Find potential pattern pivots
            pivot_indices = self._find_potential_pivots(df)
            
            # Identify patterns
            patterns = []
            for pattern_type in pattern_types:
                bullish_patterns = self._identify_patterns(
                    df, pivot_indices, pattern_type, HarmonicDirection.BULLISH, asset, timeframe
                )
                bearish_patterns = self._identify_patterns(
                    df, pivot_indices, pattern_type, HarmonicDirection.BEARISH, asset, timeframe
                )
                patterns.extend(bullish_patterns)
                patterns.extend(bearish_patterns)
            
            # Filter out low-quality patterns
            filtered_patterns = [p for p in patterns if p.quality_score >= self.min_quality_score]
            
            # Sort by quality and completion time
            filtered_patterns.sort(key=lambda p: (p.quality_score, p.completion_timestamp), reverse=True)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics["processing_times"].append(processing_time)
            for pattern in filtered_patterns:
                self.metrics["pattern_counts"][pattern.pattern_type.name] += 1
            
            # Add to pattern history
            self.pattern_history.extend(filtered_patterns)
            if len(self.pattern_history) > self.pattern_history_size:
                self.pattern_history = self.pattern_history[-self.pattern_history_size:]
                
            logger.debug(f"Recognized {len(filtered_patterns)} patterns out of {len(patterns)} potential matches in {processing_time:.3f}s")
            
            return filtered_patterns
            
        except Exception as e:
            logger.error(f"Error in harmonic pattern recognition: {str(e)}", exc_info=True)
            raise PatternRecognitionError(f"Pattern recognition failed: {str(e)}")
    
    def _find_potential_pivots(self, df: pd.DataFrame) -> List[int]:
        """
        Find potential pivot points in price data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of indices of potential pivot points
        """
        # Use argrelextrema for fast pivot point detection
        order = max(5, len(df) // 50)  # Adaptive window size
        
        highs = df['high'].values
        lows = df['low'].values
        
        local_max_indices = argrelextrema(highs, np.greater_equal, order=order)[0]
        local_min_indices = argrelextrema(lows, np.less_equal, order=order)[0]
        
        # Combine and sort pivot points
        pivot_indices = np.unique(np.concatenate([local_max_indices, local_min_indices]))
        pivot_indices = pivot_indices.tolist()
        
        # Filter out weak pivots
        filtered_pivots = []
        for i in range(len(pivot_indices)):
            idx = pivot_indices[i]
            
            # Skip pivots at the edges
            if idx < 2 or idx > len(df) - 3:
                continue
                
            # Check if it's a swing high
            if idx in local_max_indices:
                # Verify it's a significant swing high
                if self._is_significant_pivot(df, idx, is_high=True):
                    filtered_pivots.append(idx)
            
            # Check if it's a swing low
            if idx in local_min_indices:
                # Verify it's a significant swing low
                if self._is_significant_pivot(df, idx, is_high=False):
                    filtered_pivots.append(idx)
        
        return sorted(filtered_pivots)
    
    def _is_significant_pivot(self, df: pd.DataFrame, idx: int, is_high: bool) -> bool:
        """
        Determine if a pivot point is significant.
        
        Args:
            df: DataFrame with OHLCV data
            idx: Index of potential pivot point
            is_high: True if checking a high, False if checking a low
            
        Returns:
            True if the pivot is significant, False otherwise
        """
        # Check surrounding bars to confirm pivot significance
        value = df['high'].iloc[idx] if is_high else df['low'].iloc[idx]
        
        # Calculate average price for reference
        avg_price = df['close'].iloc[max(0, idx-10):min(len(df), idx+10)].mean()
        
        # Calculate minimum required price movement
        min_movement = avg_price * (self.min_leg_size_percent / 100)
        
        # For a high, check if it's higher than surrounding bars
        if is_high:
            left_bars = df['high'].iloc[max(0, idx-5):idx].values
            right_bars = df['high'].iloc[idx+1:min(len(df), idx+6)].values
            
            if len(left_bars) == 0 or len(right_bars) == 0:
                return False
                
            if (value > np.max(left_bars) and value > np.max(right_bars) and 
                value - max(np.min(left_bars), np.min(right_bars)) > min_movement):
                return True
        # For a low, check if it's lower than surrounding bars
        else:
            left_bars = df['low'].iloc[max(0, idx-5):idx].values
            right_bars = df['low'].iloc[idx+1:min(len(df), idx+6)].values
            
            if len(left_bars) == 0 or len(right_bars) == 0:
                return False
                
            if (value < np.min(left_bars) and value < np.min(right_bars) and 
                min(np.max(left_bars), np.max(right_bars)) - value > min_movement):
                return True
                
        return False
    
    def _identify_patterns(self, 
                          df: pd.DataFrame, 
                          pivot_indices: List[int],
                          pattern_type: HarmonicPatternType,
                          direction: HarmonicDirection,
                          asset: str,
                          timeframe: TimeFrame) -> List[PatternInstance]:
        """
        Identify specific pattern types in the provided pivot points.
        
        Args:
            df: DataFrame with OHLCV data
            pivot_indices: List of indices of potential pivot points
            pattern_type: Type of pattern to identify
            direction: Pattern direction (bullish/bearish)
            asset: Asset name/symbol
            timeframe: Timeframe of the provided data
            
        Returns:
            List of identified pattern instances
        """
        patterns = []
        pattern_config = self.patterns[pattern_type]
        
        # Get the last valid index
        last_idx = len(df) - 1
        
        # We need at least 5 pivots for a complete pattern (X, A, B, C, D)
        if len(pivot_indices) < 5:
            return patterns
            
        # Define price accessor based on direction
        def get_price(idx, is_pivot):
            if is_pivot:
                return df['high'].iloc[idx] if direction == HarmonicDirection.BULLISH else df['low'].iloc[idx]
            else:
                return df['low'].iloc[idx] if direction == HarmonicDirection.BULLISH else df['high'].iloc[idx]
                
        # Iterate through potential combinations of pivot points
        for i in range(len(pivot_indices) - 4):
            x_idx = pivot_indices[i]
            
            # Skip patterns that are too old
            if last_idx - x_idx > self.max_age_bars:
                continue
                
            x_price = get_price(x_idx, direction == HarmonicDirection.BEARISH)
            
            # Try each combination of remaining pivots
            for j in range(i + 1, len(pivot_indices) - 3):
                a_idx = pivot_indices[j]
                a_price = get_price(a_idx, direction == HarmonicDirection.BULLISH)
                
                # Verify X->A direction
                if (direction == HarmonicDirection.BULLISH and a_price >= x_price) or \
                   (direction == HarmonicDirection.BEARISH and a_price <= x_price):
                    continue
                    
                for k in range(j + 1, len(pivot_indices) - 2):
                    b_idx = pivot_indices[k]
                    b_price = get_price(b_idx, direction == HarmonicDirection.BEARISH)
                    
                    # Verify A->B direction
                    if (direction == HarmonicDirection.BULLISH and b_price <= a_price) or \
                       (direction == HarmonicDirection.BEARISH and b_price >= a_price):
                        continue
                        
                    # Calculate XAB ratio
                    xa_range = abs(x_price - a_price)
                    ab_range = abs(a_price - b_price)
                    xab_ratio = ab_range / xa_range if xa_range > 0 else 0
                    
                    # Validate XAB ratio
                    if not pattern_config.xab_ratio.is_valid(xab_ratio):
                        continue
                        
                    for l in range(k + 1, len(pivot_indices) - 1):
                        c_idx = pivot_indices[l]
                        c_price = get_price(c_idx, direction == HarmonicDirection.BULLISH)
                        
                        # Verify B->C direction
                        if (direction == HarmonicDirection.BULLISH and c_price >= b_price) or \
                           (direction == HarmonicDirection.BEARISH and c_price <= b_price):
                            continue
                            
                        # Calculate ABC ratio
                        bc_range = abs(b_price - c_price)
                        abc_ratio = bc_range / ab_range if ab_range > 0 else 0
                        
                        # Validate ABC ratio
                        if not pattern_config.abc_ratio.is_valid(abc_ratio):
                            continue
                            
                        for m in range(l + 1, len(pivot_indices)):
                            d_idx = pivot_indices[m]
                            d_price = get_price(d_idx, direction == HarmonicDirection.BEARISH)
                            
                            # Verify C->D direction
                            if (direction == HarmonicDirection.BULLISH and d_price <= c_price) or \
                               (direction == HarmonicDirection.BEARISH and d_price >= c_price):
                                continue
                                
                            # Calculate BCD ratio
                            cd_range = abs(c_price - d_price)
                            bcd_ratio = cd_range / bc_range if bc_range > 0 else 0
                            
                            # Validate BCD ratio
                            if not pattern_config.bcd_ratio.is_valid(bcd_ratio):
                                continue
                                
                            # Calculate XAD ratio
                            xad_ratio = abs(x_price - d_price) / xa_range if xa_range > 0 else 0
                            
                            # Validate XAD ratio if applicable
                            if pattern_type != HarmonicPatternType.FIVE_ZERO and not pattern_config.xad_ratio.is_valid(xad_ratio):
                                continue
                                
                            # For 5-0 pattern, validate the specific requirement that D returns to X
                            if pattern_type == HarmonicPatternType.FIVE_ZERO:
                                # D should be near the level of X
                                if abs(d_price - x_price) / x_price > 0.03:  # 3% tolerance
                                    continue
                            
                            # Calculate quality score
                            quality_score = self._calculate_quality_score(
                                pattern_config,
                                xab_ratio, abc_ratio, bcd_ratio, xad_ratio,
                                xa_range, ab_range, bc_range, cd_range,
                                df['close'].iloc[-1]  # Current close price
                            )
                            
                            # Calculate confidence based on historical success
                            confidence = self._calculate_confidence(pattern_type, direction)
                            
                            # Create pattern instance
                            pattern = PatternInstance(
                                pattern_type=pattern_type,
                                direction=direction,
                                points={
                                    'X': PatternPoint(x_idx, x_price, int(df['timestamp'].iloc[x_idx])),
                                    'A': PatternPoint(a_idx, a_price, int(df['timestamp'].iloc[a_idx])),
                                    'B': PatternPoint(b_idx, b_price, int(df['timestamp'].iloc[b_idx])),
                                    'C': PatternPoint(c_idx, c_price, int(df['timestamp'].iloc[c_idx])),
                                    'D': PatternPoint(d_idx, d_price, int(df['timestamp'].iloc[d_idx]))
                                },
                                quality_score=quality_score,
                                confidence=confidence,
                                completion_timestamp=int(df['timestamp'].iloc[d_idx]),
                                potential_reversal_zone=self._calculate_prz(
                                    x_price, a_price, b_price, c_price, d_price, direction
                                ),
                                expected_move_magnitude=self._calculate_expected_move(
                                    pattern_type, direction, x_price, a_price, b_price, c_price, d_price
                                ),
                                timeframe=timeframe,
                                asset=asset,
                                pattern_id=f"{pattern_type.value}_{direction.value}_{int(df['timestamp'].iloc[d_idx])}"
                            )
                            
                            patterns.append(pattern)
                                
        return patterns
    
    def _calculate_quality_score(self, 
                               pattern_config: HarmonicPattern,
                               xab_ratio: float, 
                               abc_ratio: float, 
                               bcd_ratio: float, 
                               xad_ratio: float,
                               xa_range: float,
                               ab_range: float,
                               bc_range: float,
                               cd_range: float,
                               current_price: float) -> float:
        """
        Calculate a quality score for a pattern based on its ratios and structure.
        
        Args:
            pattern_config: Pattern configuration with ideal ratios
            xab_ratio: Actual XAB ratio found
            abc_ratio: Actual ABC ratio found
            bcd_ratio: Actual BCD ratio found
            xad_ratio: Actual XAD ratio found
            xa_range: Size of XA leg
            ab_range: Size of AB leg
            bc_range: Size of BC leg
            cd_range: Size of CD leg
            current_price: Current market price
            
        Returns:
            Quality score between 0 and 1
        """
        # Calculate ratio deviations
        xab_deviation = abs(pattern_config.xab_ratio.deviation(xab_ratio))
        abc_deviation = abs(pattern_config.abc_ratio.deviation(abc_ratio))
        bcd_deviation = abs(pattern_config.bcd_ratio.deviation(bcd_ratio))
        
        # For 5-0 pattern, we don't use XAD
        if pattern_config.pattern_type == HarmonicPatternType.FIVE_ZERO:
            xad_deviation = 0
        else:
            xad_deviation = abs(pattern_config.xad_ratio.deviation(xad_ratio))
        
        # Calculate the average deviation percentage (lower is better)
        avg_deviation = (xab_deviation + abc_deviation + bcd_deviation + xad_deviation) / 4
        
        # Penalize very small patterns
        avg_leg_size = (xa_range + ab_range + bc_range + cd_range) / 4
        size_factor = min(1.0, avg_leg_size / (current_price * self.min_leg_size_percent / 100))
        
        # Bonus for symmetrical patterns
        symmetry_score = 1.0 - min(1.0, (
            abs(xa_range - cd_range) / max(xa_range, cd_range) * 0.5 +
            abs(ab_range - bc_range) / max(ab_range, bc_range) * 0.5
        ))
        
        # Convert deviations to a quality score (0-1 range)
        ratio_score = max(0, 1.0 - min(1.0, avg_deviation / 10))  # 10% deviation max
        
        # Calculate final score with weightings
        final_score = (ratio_score * 0.6) + (size_factor * 0.2) + (symmetry_score * 0.2)
        
        # Bonus for patterns with a strong historical success rate
        if pattern_config.success_rate and pattern_config.success_rate["overall"] > 0.6:
            final_score *= 1.1
            final_score = min(1.0, final_score)  # Cap at 1.0
            
        return round(final_score, 3)
    
    def _calculate_confidence(self, pattern_type: HarmonicPatternType, direction: HarmonicDirection) -> float:
        """
        Calculate confidence score based on historical success rates.
        
        Args:
            pattern_type: Type of harmonic pattern
            direction: Pattern direction (bullish/bearish)
            
        Returns:
            Confidence score between 0 and 1
        """
        pattern_config = self.patterns[pattern_type]
        
        # If no historical data, use default
        if not pattern_config.success_rate:
            return 0.5  # Neutral confidence
            
        # Use direction-specific success rate if available
        if direction.value in pattern_config.success_rate and pattern_config.success_rate[direction.value] > 0:
            return pattern_config.success_rate[direction.value]
            
        # Fall back to overall success rate
        return pattern_config.success_rate["overall"]
    
    def _calculate_prz(self, 
                     x_price: float, 
                     a_price: float, 
                     b_price: float, 
                     c_price: float, 
                     d_price: float, 
                     direction: HarmonicDirection) -> Tuple[float, float]:
        """
        Calculate the Potential Reversal Zone (PRZ) for a pattern.
        
        Args:
            x_price, a_price, b_price, c_price, d_price: Pattern point prices
            direction: Pattern direction
            
        Returns:
            Tuple of (lower PRZ, upper PRZ)
        """
        # The D point is the center of the PRZ
        center = d_price
        
        # Calculate the PRZ range as a percentage of the XA leg
        xa_range = abs(x_price - a_price)
        prz_width = xa_range * 0.0382  # 3.82% of XA range
        
        if direction == HarmonicDirection.BULLISH:
            return (center - prz_width, center + prz_width * 0.5)
        else:
            return (center - prz_width * 0.5, center + prz_width)
    
    def _calculate_expected_move(self, 
                               pattern_type: HarmonicPatternType,
                               direction: HarmonicDirection,
                               x_price: float, 
                               a_price: float, 
                               b_price: float, 
                               c_price: float, 
                               d_price: float) -> float:
        """
        Calculate the expected magnitude of the move after pattern completion.
        
        Args:
            pattern_type: Type of harmonic pattern
            direction: Pattern direction
            x_price, a_price, b_price, c_price, d_price: Pattern point prices
            
        Returns:
            Expected price movement as absolute value
        """
        # Calculate XA range as reference
        xa_range = abs(x_price - a_price)
        
        # Different pattern types have different expected moves
        if pattern_type in [HarmonicPatternType.BUTTERFLY, HarmonicPatternType.CRAB, HarmonicPatternType.DEEP_CRAB]:
            # Deep patterns typically have stronger reversals
            expected_move = xa_range * 0.618
        elif pattern_type in [HarmonicPatternType.GARTLEY, HarmonicPatternType.BAT]:
            # More moderate patterns
            expected_move = xa_range * 0.382
        elif pattern_type == HarmonicPatternType.SHARK:
            expected_move = xa_range * 0.5
        elif pattern_type == HarmonicPatternType.CYPHER:
            expected_move = xa_range * 0.382
        else:
            # Default for other patterns
            expected_move = xa_range * 0.382
            
        return expected_move
    
    def update_pattern_success(self, pattern_id: str, success: bool, profit_pct: float) -> None:
        """
        Update the success record for a specific pattern.
        
        Args:
            pattern_id: Unique identifier of the pattern
            success: Whether the pattern resulted in a successful trade
            profit_pct: Profit/loss percentage of the trade
        """
        # Find the pattern in history
        pattern = next((p for p in self.pattern_history if p.pattern_id == pattern_id), None)
        if not pattern:
            logger.warning(f"Cannot update success for unknown pattern: {pattern_id}")
            return
            
        # Store the pattern outcome in the database
        try:
            pattern_occurrence = PatternOccurrence(
                pattern_id=pattern.pattern_id,
                pattern_type=pattern.pattern_type.value,
                direction=pattern.direction.value,
                asset=pattern.asset,
                timeframe=pattern.timeframe.value,
                completion_timestamp=pattern.completion_timestamp,
                quality_score=pattern.quality_score,
                success=success,
                profit_percentage=profit_pct
            )
            pattern_occurrence.save()
            
            # Update the success metrics for this pattern type
            pattern_metrics = PatternMetrics.get_or_create(
                pattern_type=pattern.pattern_type.value,
                direction=pattern.direction.value,
                asset=pattern.asset,
                timeframe=pattern.timeframe.value
            )
            
            # Update the metrics
            if success:
                pattern_metrics.success_count += 1
            else:
                pattern_metrics.failure_count += 1
                
            pattern_metrics.total_profit += profit_pct
            pattern_metrics.avg_profit = pattern_metrics.total_profit / (pattern_metrics.success_count + pattern_metrics.failure_count)
            pattern_metrics.success_rate = pattern_metrics.success_count / (pattern_metrics.success_count + pattern_metrics.failure_count)
            pattern_metrics.save()
            
            # Update in-memory success rates if auto-adaptation is enabled
            if self.enable_auto_adaptation:
                self._adapt_pattern_config(pattern.pattern_type, pattern.direction, success)
                
            logger.info(f"Updated pattern success: {pattern.pattern_type.value} {pattern.direction.value} - " + 
                       f"Success: {success}, Profit: {profit_pct:.2f}%")
            
        except Exception as e:
            logger.error(f"Error updating pattern success: {str(e)}", exc_info=True)
    
    def _adapt_pattern_config(self, 
                            pattern_type: HarmonicPatternType, 
                            direction: HarmonicDirection, 
                            success: bool) -> None:
        """
        Adapt pattern configuration based on success/failure.
        
        Args:
            pattern_type: Type of harmonic pattern
            direction: Pattern direction
            success: Whether the pattern was successful
        """
        pattern_config = self.patterns[pattern_type]
        
        # Initialize success rates if not present
        if pattern_config.success_rate is None:
            pattern_config.success_rate = {"overall": 0.5, "bullish": 0.5, "bearish": 0.5}
            
        # Update direction-specific success rate
        dir_key = direction.value
        if dir_key not in pattern_config.success_rate:
            pattern_config.success_rate[dir_key] = 0.5
            
        # Update with exponential moving average (more weight to recent results)
        alpha = 0.05  # Learning rate
        current_rate = pattern_config.success_rate[dir_key]
        new_data = 1.0 if success else 0.0
        pattern_config.success_rate[dir_key] = current_rate * (1 - alpha) + new_data * alpha
        
        # Update overall success rate
        overall = (pattern_config.success_rate["bullish"] + pattern_config.success_rate["bearish"]) / 2
        pattern_config.success_rate["overall"] = overall
        
        # If pattern has been consistently successful, reduce tolerance to improve precision
        if success and pattern_config.success_rate[dir_key] > 0.75:
            self._tighten_pattern_tolerances(pattern_type, 0.95)
        # If pattern has been consistently unsuccessful, increase tolerance to find more opportunities
        elif not success and pattern_config.success_rate[dir_key] < 0.4:
            self._tighten_pattern_tolerances(pattern_type, 1.05)
    
    def _tighten_pattern_tolerances(self, pattern_type: HarmonicPatternType, factor: float) -> None:
        """
        Tighten or loosen pattern tolerances based on performance.
        
        Args:
            pattern_type: Type of harmonic pattern
            factor: Factor to adjust tolerances by (< 1 tightens, > 1 loosens)
        """
        pattern = self.patterns[pattern_type]
        
        # Adjust XAB ratio tolerance
        ideal = pattern.xab_ratio.ideal
        min_diff = ideal - pattern.xab_ratio.min_tolerance
        max_diff = pattern.xab_ratio.max_tolerance - ideal
        pattern.xab_ratio.min_tolerance = ideal - min_diff * factor
        pattern.xab_ratio.max_tolerance = ideal + max_diff * factor
        
        # Adjust ABC ratio tolerance
        ideal = pattern.abc_ratio.ideal
        min_diff = ideal - pattern.abc_ratio.min_tolerance
        max_diff = pattern.abc_ratio.max_tolerance - ideal
        pattern.abc_ratio.min_tolerance = ideal - min_diff * factor
        pattern.abc_ratio.max_tolerance = ideal + max_diff * factor
        
        # Adjust BCD ratio tolerance
        ideal = pattern.bcd_ratio.ideal
        min_diff = ideal - pattern.bcd_ratio.min_tolerance
        max_diff = pattern.bcd_ratio.max_tolerance - ideal
        pattern.bcd_ratio.min_tolerance = ideal - min_diff * factor
        pattern.bcd_ratio.max_tolerance = ideal + max_diff * factor
        
        # Adjust XAD ratio tolerance if applicable
        if pattern.pattern_type != HarmonicPatternType.FIVE_ZERO:
            ideal = pattern.xad_ratio.ideal
            min_diff = ideal - pattern.xad_ratio.min_tolerance
            max_diff = pattern.xad_ratio.max_tolerance - ideal
            pattern.xad_ratio.min_tolerance = ideal - min_diff * factor
            pattern.xad_ratio.max_tolerance = ideal + max_diff * factor
    
    def get_pattern_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics for all pattern types.
        
        Returns:
            Dictionary of pattern metrics
        """
        metrics = {}
        
        for pattern_type in HarmonicPatternType:
            pattern_config = self.patterns[pattern_type]
            type_name = pattern_type.value
            
            metrics[type_name] = {
                "count": self.metrics["pattern_counts"].get(pattern_type.name, 0),
                "success_rate": pattern_config.success_rate if pattern_config.success_rate else {"overall": 0.0}
            }
            
            # Add database metrics if available
            try:
                db_metrics = PatternMetrics.get_aggregated(pattern_type=type_name)
                if db_metrics:
                    metrics[type_name].update({
                        "db_success_rate": db_metrics.get("success_rate", 0.0),
                        "avg_profit": db_metrics.get("avg_profit", 0.0),
                        "total_occurrences": db_metrics.get("total_count", 0)
                    })
            except Exception as e:
                logger.error(f"Error retrieving database metrics for {type_name}: {str(e)}")
        
        return metrics
    
    def export_pattern_definitions(self) -> Dict[str, Dict[str, Any]]:
        """
        Export the current pattern definitions with their ratios.
        
        Returns:
            Dictionary of pattern definitions
        """
        definitions = {}
        
        for pattern_type, pattern in self.patterns.items():
            definitions[pattern_type.value] = {
                "description": pattern.description,
                "xab_ratio": {
                    "ideal": pattern.xab_ratio.ideal,
                    "min": pattern.xab_ratio.min_tolerance,
                    "max": pattern.xab_ratio.max_tolerance
                },
                "abc_ratio": {
                    "ideal": pattern.abc_ratio.ideal,
                    "min": pattern.abc_ratio.min_tolerance,
                    "max": pattern.abc_ratio.max_tolerance
                },
                "bcd_ratio": {
                    "ideal": pattern.bcd_ratio.ideal,
                    "min": pattern.bcd_ratio.min_tolerance,
                    "max": pattern.bcd_ratio.max_tolerance
                },
                "xad_ratio": {
                    "ideal": pattern.xad_ratio.ideal,
                    "min": pattern.xad_ratio.min_tolerance,
                    "max": pattern.xad_ratio.max_tolerance
                },
                "success_rate": pattern.success_rate
            }
        
        return definitions
    
    def import_pattern_definitions(self, definitions: Dict[str, Dict[str, Any]]) -> None:
        """
        Import pattern definitions to update the current configurations.
        
        Args:
            definitions: Dictionary of pattern definitions
        """
        for pattern_name, pattern_def in definitions.items():
            try:
                # Find the corresponding pattern type
                pattern_type = next((pt for pt in HarmonicPatternType if pt.value == pattern_name), None)
                if not pattern_type:
                    logger.warning(f"Unknown pattern type in import: {pattern_name}")
                    continue
                
                # Update pattern configuration
                pattern = self.patterns[pattern_type]
                
                if "xab_ratio" in pattern_def:
                    ratio = pattern_def["xab_ratio"]
                    pattern.xab_ratio.ideal = ratio["ideal"]
                    pattern.xab_ratio.min_tolerance = ratio["min"]
                    pattern.xab_ratio.max_tolerance = ratio["max"]
                
                if "abc_ratio" in pattern_def:
                    ratio = pattern_def["abc_ratio"]
                    pattern.abc_ratio.ideal = ratio["ideal"]
                    pattern.abc_ratio.min_tolerance = ratio["min"]
                    pattern.abc_ratio.max_tolerance = ratio["max"]
                
                if "bcd_ratio" in pattern_def:
                    ratio = pattern_def["bcd_ratio"]
                    pattern.bcd_ratio.ideal = ratio["ideal"]
                    pattern.bcd_ratio.min_tolerance = ratio["min"]
                    pattern.bcd_ratio.max_tolerance = ratio["max"]
                
                if "xad_ratio" in pattern_def:
                    ratio = pattern_def["xad_ratio"]
                    pattern.xad_ratio.ideal = ratio["ideal"]
                    pattern.xad_ratio.min_tolerance = ratio["min"]
                    pattern.xad_ratio.max_tolerance = ratio["max"]
                
                if "success_rate" in pattern_def:
                    pattern.success_rate = pattern_def["success_rate"]
                
                if "description" in pattern_def:
                    pattern.description = pattern_def["description"]
                    
                logger.info(f"Updated pattern definition for {pattern_name}")
                
            except Exception as e:
                logger.error(f"Error importing pattern definition for {pattern_name}: {str(e)}", exc_info=True)
    
    def recognize_patterns_multi_timeframe(self,
                                          dataframes: Dict[TimeFrame, pd.DataFrame],
                                          asset: str) -> Dict[TimeFrame, List[PatternInstance]]:
        """
        Identify patterns across multiple timeframes simultaneously.
        
        Args:
            dataframes: Dictionary of DataFrames for different timeframes
            asset: Asset name/symbol
            
        Returns:
            Dictionary of identified patterns per timeframe
        """
        results = {}
        
        if self.enable_parallel_processing:
            # Process timeframes in parallel
            with ThreadPoolExecutor() as executor:
                futures = {}
                for timeframe, df in dataframes.items():
                    futures[timeframe] = executor.submit(
                        self.recognize_patterns, df, asset, timeframe
                    )
                
                # Collect results
                for timeframe, future in futures.items():
                    try:
                        results[timeframe] = future.result()
                    except Exception as e:
                        logger.error(f"Error processing timeframe {timeframe}: {str(e)}", exc_info=True)
                        results[timeframe] = []
        else:
            # Process timeframes sequentially
            for timeframe, df in dataframes.items():
                try:
                    results[timeframe] = self.recognize_patterns(df, asset, timeframe)
                except Exception as e:
                    logger.error(f"Error processing timeframe {timeframe}: {str(e)}", exc_info=True)
                    results[timeframe] = []
        
        return results
    
    def reset_metrics(self) -> None:
        """Reset all in-memory metrics."""
        self.metrics = {
            "pattern_counts": {pattern.name: 0 for pattern in HarmonicPatternType},
            "success_rates": {pattern.name: 0.0 for pattern in HarmonicPatternType},
            "processing_times": []
        }
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the pattern recognizer.
        
        Returns:
            Dictionary of performance statistics
        """
        processing_times = self.metrics["processing_times"]
        
        if not processing_times:
            return {
                "avg_processing_time": 0,
                "max_processing_time": 0,
                "min_processing_time": 0,
                "total_patterns_detected": 0
            }
            
        total_patterns = sum(self.metrics["pattern_counts"].values())
        
        return {
            "avg_processing_time": sum(processing_times) / len(processing_times),
            "max_processing_time": max(processing_times),
            "min_processing_time": min(processing_times),
            "total_patterns_detected": total_patterns
        }
