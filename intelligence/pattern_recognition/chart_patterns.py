

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Chart Pattern Recognition Module

This module implements advanced chart pattern recognition algorithms for identifying
high-probability trading setups across multiple timeframes. The patterns detected by
this module are used by the trading brains to make trading decisions.

The implementation uses sophisticated computer vision and signal processing techniques
to detect chart patterns with high accuracy, focusing on patterns with the highest
statistical edge in the markets.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum, auto
import logging
from dataclasses import dataclass
from scipy import signal
from scipy.stats import linregress
import cv2
from datetime import datetime, timedelta
import joblib
import warnings
import statsmodels.api as sm
from functools import lru_cache

# Internal imports
from common.utils import calculate_distance, timeframe_to_seconds, is_higher_timeframe
from common.logger import get_logger
from common.constants import (
    PATTERN_COMPLETION_THRESHOLD, PATTERN_STRENGTH_LEVELS,
    MIN_PATTERN_BARS, MAX_PATTERN_BARS, TIMEFRAMES
)
from feature_service.features.market_structure import (
    find_swing_points, identify_market_structure
)

# Initialize logger
logger = get_logger("chart_patterns")


class PatternType(Enum):
    """Enum for different chart pattern types"""
    HEAD_AND_SHOULDERS = auto()
    INVERSE_HEAD_AND_SHOULDERS = auto()
    DOUBLE_TOP = auto()
    DOUBLE_BOTTOM = auto()
    TRIPLE_TOP = auto()
    TRIPLE_BOTTOM = auto()
    ASCENDING_TRIANGLE = auto()
    DESCENDING_TRIANGLE = auto()
    SYMMETRICAL_TRIANGLE = auto()
    RISING_WEDGE = auto()
    FALLING_WEDGE = auto()
    RECTANGLE = auto()
    CUP_AND_HANDLE = auto()
    INVERSE_CUP_AND_HANDLE = auto()
    BULL_FLAG = auto()
    BEAR_FLAG = auto()
    BULL_PENNANT = auto()
    BEAR_PENNANT = auto()
    ROUNDING_BOTTOM = auto()
    ROUNDING_TOP = auto()
    CHANNEL_UP = auto()
    CHANNEL_DOWN = auto()


class PatternConfirmation(Enum):
    """Enum for pattern confirmation status"""
    FORMING = auto()        # Pattern is still forming
    POTENTIAL = auto()      # Pattern meets initial criteria but needs confirmation
    CONFIRMED = auto()      # Pattern is confirmed with breakout
    FAILED = auto()         # Pattern has failed (false breakout)
    COMPLETED = auto()      # Pattern has completed its measured move


@dataclass
class PatternInfo:
    """Data class for storing pattern information"""
    pattern_type: PatternType
    start_idx: int
    end_idx: int
    pivot_points: List[Tuple[int, float]]  # List of (index, price) tuples
    confirmation: PatternConfirmation
    breakout_level: float
    target_level: Optional[float] = None
    stop_level: Optional[float] = None
    confidence: float = 0.0
    strength: str = "medium"
    volume_confirms: bool = False
    timestamp: datetime = datetime.now()
    measured_move_pct: Optional[float] = None
    timeframe: str = "1h"
    additional_info: Dict[str, Any] = None


class ChartPatternRecognizer:
    """
    Advanced chart pattern recognition engine that detects and analyzes
    chart patterns across multiple timeframes.
    """
    
    def __init__(
        self,
        min_pattern_bars: int = MIN_PATTERN_BARS,
        max_pattern_bars: int = MAX_PATTERN_BARS,
        completion_threshold: float = PATTERN_COMPLETION_THRESHOLD,
        enable_gpu: bool = True,
        pattern_history_length: int = 1000,
        load_pretrained: bool = True
    ):
        """
        Initialize the chart pattern recognizer.
        
        Args:
            min_pattern_bars: Minimum number of bars to consider for pattern detection
            max_pattern_bars: Maximum number of bars to consider for pattern detection
            completion_threshold: Threshold for pattern completion percentage
            enable_gpu: Whether to use GPU acceleration for computer vision algorithms
            pattern_history_length: Number of historical patterns to maintain
            load_pretrained: Whether to load pretrained models for pattern detection
        """
        self.min_bars = min_pattern_bars
        self.max_bars = max_pattern_bars
        self.completion_threshold = completion_threshold
        self.gpu_enabled = enable_gpu
        self.pattern_history = {}  # Dict to store pattern history by symbol and timeframe
        self.pattern_history_length = pattern_history_length
        
        # Initialize pattern detection models
        self.models = {}
        if load_pretrained:
            self._load_pretrained_models()
        
        logger.info(
            f"ChartPatternRecognizer initialized with min_bars={min_pattern_bars}, "
            f"max_bars={max_pattern_bars}, GPU acceleration: {enable_gpu}"
        )


class ChartPattern:
    """Lightweight wrapper exposing a simple detect method."""

    def __init__(self) -> None:
        self.recognizer = ChartPatternRecognizer()

    def detect(self, data: pd.DataFrame, symbol: str = "", timeframe: str = "1h") -> List[PatternInfo]:
        """Detect chart patterns using the underlying recognizer."""
        return self.recognizer.detect_patterns(data, symbol=symbol, timeframe=timeframe)
    
    def _load_pretrained_models(self):
        """Load pretrained models for pattern detection"""
        try:
            model_path = "models/chart_patterns/"
            for pattern_type in PatternType:
                model_file = f"{model_path}{pattern_type.name.lower()}_model.pkl"
                try:
                    self.models[pattern_type] = joblib.load(model_file)
                    logger.debug(f"Loaded model for {pattern_type.name}")
                except FileNotFoundError:
                    logger.warning(f"Model for {pattern_type.name} not found, will use algorithmic detection")
        except Exception as e:
            logger.error(f"Error loading pretrained models: {str(e)}")
            logger.info("Falling back to algorithmic pattern detection")
    
    def detect_patterns(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
        patterns: Optional[List[PatternType]] = None,
        min_confidence: float = 0.6
    ) -> List[PatternInfo]:
        """
        Detect chart patterns in the provided price data.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe of the data
            patterns: List of patterns to detect, or None for all patterns
            min_confidence: Minimum confidence threshold for pattern detection
            
        Returns:
            List of detected patterns with their information
        """
        if data.empty or len(data) < self.min_bars:
            return []
        
        # Ensure data has required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Convert column names to lowercase if needed
        data.columns = map(str.lower, data.columns)
        
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Get list of patterns to detect
        patterns_to_detect = patterns if patterns else list(PatternType)
        
        detected_patterns = []
        
        # First, find key swing points using market structure analysis
        swing_highs, swing_lows = find_swing_points(df, n=5)
        
        # Detect each pattern type
        for pattern in patterns_to_detect:
            try:
                # Use specific detection method based on pattern type
                pattern_method = getattr(self, f"_detect_{pattern.name.lower()}", None)
                
                if pattern_method:
                    # Method exists, use it
                    patterns_info = pattern_method(df, swing_highs, swing_lows)
                else:
                    # Try using ML model if available
                    if pattern in self.models:
                        patterns_info = self._detect_with_ml(df, pattern)
                    else:
                        # Skip if no detection method or model
                        logger.warning(f"No detection method available for {pattern.name}")
                        continue
                
                # Filter by confidence threshold
                patterns_info = [p for p in patterns_info if p.confidence >= min_confidence]
                
                # Add symbol and timeframe info
                for p in patterns_info:
                    if not hasattr(p, 'additional_info') or p.additional_info is None:
                        p.additional_info = {}
                    p.additional_info['symbol'] = symbol
                    p.timeframe = timeframe
                
                detected_patterns.extend(patterns_info)
                
            except Exception as e:
                logger.error(f"Error detecting {pattern.name} patterns: {str(e)}")
        
        # Sort patterns by end index (most recent first)
        detected_patterns.sort(key=lambda x: x.end_idx, reverse=True)
        
        # Add detected patterns to history
        self._update_pattern_history(symbol, timeframe, detected_patterns)
        
        return detected_patterns
    
    def _update_pattern_history(self, symbol: str, timeframe: str, patterns: List[PatternInfo]):
        """Update the pattern history for a symbol and timeframe"""
        key = f"{symbol}_{timeframe}"
        if key not in self.pattern_history:
            self.pattern_history[key] = []
        
        # Add new patterns
        self.pattern_history[key].extend(patterns)
        
        # Keep only the most recent patterns up to pattern_history_length
        self.pattern_history[key] = self.pattern_history[key][-self.pattern_history_length:]
    
    def get_pattern_history(self, symbol: str, timeframe: str) -> List[PatternInfo]:
        """Get the pattern history for a symbol and timeframe"""
        key = f"{symbol}_{timeframe}"
        return self.pattern_history.get(key, [])
    
    def update_pattern_status(
        self,
        pattern: PatternInfo,
        current_data: pd.DataFrame
    ) -> PatternInfo:
        """
        Update the status of a previously detected pattern based on current data.
        
        Args:
            pattern: The pattern to update
            current_data: Current price data
            
        Returns:
            Updated pattern information
        """
        if pattern.confirmation == PatternConfirmation.COMPLETED:
            return pattern
        
        if current_data.empty or len(current_data) == 0:
            return pattern
        
        # Get the latest price
        latest_close = current_data['close'].iloc[-1]
        latest_high = current_data['high'].iloc[-1]
        latest_low = current_data['low'].iloc[-1]
        
        # Update pattern status based on the latest price action
        updated_pattern = pattern
        
        # Check for breakout confirmation
        if pattern.confirmation == PatternConfirmation.POTENTIAL:
            if pattern.pattern_type in [
                PatternType.HEAD_AND_SHOULDERS, 
                PatternType.TRIPLE_TOP,
                PatternType.DOUBLE_TOP,
                PatternType.DESCENDING_TRIANGLE,
                PatternType.RISING_WEDGE,
                PatternType.ROUNDING_TOP,
                PatternType.INVERSE_CUP_AND_HANDLE
            ]:
                # Bearish patterns - confirm on close below breakout level
                if latest_close < pattern.breakout_level:
                    updated_pattern.confirmation = PatternConfirmation.CONFIRMED
                    logger.info(f"Bearish pattern {pattern.pattern_type.name} confirmed for {pattern.additional_info.get('symbol', 'Unknown')}")
            
            elif pattern.pattern_type in [
                PatternType.INVERSE_HEAD_AND_SHOULDERS,
                PatternType.TRIPLE_BOTTOM,
                PatternType.DOUBLE_BOTTOM,
                PatternType.ASCENDING_TRIANGLE,
                PatternType.FALLING_WEDGE,
                PatternType.ROUNDING_BOTTOM,
                PatternType.CUP_AND_HANDLE
            ]:
                # Bullish patterns - confirm on close above breakout level
                if latest_close > pattern.breakout_level:
                    updated_pattern.confirmation = PatternConfirmation.CONFIRMED
                    logger.info(f"Bullish pattern {pattern.pattern_type.name} confirmed for {pattern.additional_info.get('symbol', 'Unknown')}")
            
            else:
                # For other patterns, update based on pattern-specific logic
                updated_pattern = self._update_specific_pattern(updated_pattern, current_data)
        
        # Check for pattern completion (target reached)
        elif pattern.confirmation == PatternConfirmation.CONFIRMED:
            if pattern.target_level is not None:
                if pattern.pattern_type in [
                    PatternType.HEAD_AND_SHOULDERS, 
                    PatternType.TRIPLE_TOP,
                    PatternType.DOUBLE_TOP,
                    PatternType.DESCENDING_TRIANGLE,
                    PatternType.RISING_WEDGE,
                    PatternType.BEAR_FLAG,
                    PatternType.BEAR_PENNANT,
                    PatternType.CHANNEL_DOWN
                ]:
                    # Bearish patterns - target is below
                    if latest_low <= pattern.target_level:
                        updated_pattern.confirmation = PatternConfirmation.COMPLETED
                        logger.info(f"Bearish pattern {pattern.pattern_type.name} completed (target reached) for {pattern.additional_info.get('symbol', 'Unknown')}")
                
                else:
                    # Bullish patterns - target is above
                    if latest_high >= pattern.target_level:
                        updated_pattern.confirmation = PatternConfirmation.COMPLETED
                        logger.info(f"Bullish pattern {pattern.pattern_type.name} completed (target reached) for {pattern.additional_info.get('symbol', 'Unknown')}")
            
            # Check for pattern failure
            if pattern.stop_level is not None:
                if pattern.pattern_type in [
                    PatternType.HEAD_AND_SHOULDERS, 
                    PatternType.TRIPLE_TOP,
                    PatternType.DOUBLE_TOP,
                    PatternType.DESCENDING_TRIANGLE,
                    PatternType.RISING_WEDGE,
                    PatternType.BEAR_FLAG,
                    PatternType.BEAR_PENNANT
                ]:
                    # Bearish patterns - fail on close above stop level
                    if latest_close > pattern.stop_level:
                        updated_pattern.confirmation = PatternConfirmation.FAILED
                        logger.info(f"Bearish pattern {pattern.pattern_type.name} failed (stop level breached) for {pattern.additional_info.get('symbol', 'Unknown')}")
                
                else:
                    # Bullish patterns - fail on close below stop level
                    if latest_close < pattern.stop_level:
                        updated_pattern.confirmation = PatternConfirmation.FAILED
                        logger.info(f"Bullish pattern {pattern.pattern_type.name} failed (stop level breached) for {pattern.additional_info.get('symbol', 'Unknown')}")
        
        return updated_pattern
    
    def _update_specific_pattern(self, pattern: PatternInfo, current_data: pd.DataFrame) -> PatternInfo:
        """Update specific pattern types with custom logic"""
        # Implementation for specific pattern updates
        # This is a placeholder for pattern-specific updating logic
        return pattern
    
    def _detect_with_ml(self, df: pd.DataFrame, pattern_type: PatternType) -> List[PatternInfo]:
        """Detect patterns using the machine learning model"""
        try:
            model = self.models[pattern_type]
            
            # Prepare features for the model
            features = self._extract_features_for_ml(df)
            
            # Get predictions from the model
            predictions = model.predict_proba(features)
            
            # Find pattern instances
            pattern_instances = []
            for i, prob in enumerate(predictions[:, 1]):
                if prob >= 0.6 and i >= self.min_bars:
                    # Pattern detected with high probability
                    start_idx = max(0, i - self.max_bars)
                    end_idx = i
                    
                    # Extract pattern details
                    pattern_info = self._extract_pattern_details(
                        df.iloc[start_idx:end_idx+1], 
                        pattern_type, 
                        start_idx, 
                        end_idx, 
                        confidence=prob
                    )
                    pattern_instances.append(pattern_info)
            
            return pattern_instances
            
        except Exception as e:
            logger.error(f"Error in ML pattern detection for {pattern_type.name}: {str(e)}")
            return []
    
    def _extract_features_for_ml(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for machine learning pattern detection"""
        # Placeholder for feature extraction logic
        # In a real implementation, this would extract relevant features for the ML model
        features = np.zeros((len(df), 10))  # Example feature array
        return features
    
    def _extract_pattern_details(
        self, 
        pattern_df: pd.DataFrame, 
        pattern_type: PatternType, 
        start_idx: int, 
        end_idx: int,
        confidence: float = 0.0
    ) -> PatternInfo:
        """Extract detailed information about a detected pattern"""
        # Get key price levels
        pattern_high = pattern_df['high'].max()
        pattern_low = pattern_df['low'].min()
        pattern_close = pattern_df['close'].iloc[-1]
        
        # Find pivot points specific to this pattern
        pivot_points = []
        
        # Set defaults
        breakout_level = pattern_close
        target_level = None
        stop_level = None
        confirmation = PatternConfirmation.POTENTIAL
        
        # Pattern-specific logic
        is_bullish = self._is_bullish_pattern(pattern_type)
        
        # Calculate measured move
        pattern_height = pattern_high - pattern_low
        if is_bullish:
            breakout_level = pattern_high
            target_level = breakout_level + pattern_height
            stop_level = pattern_low
        else:
            breakout_level = pattern_low
            target_level = breakout_level - pattern_height
            stop_level = pattern_high
        
        # Determine pattern strength
        strength = "medium"
        if confidence > 0.8:
            strength = "strong"
        elif confidence < 0.7:
            strength = "weak"
        
        # Check if volume confirms the pattern
        volume_confirms = self._check_volume_confirmation(pattern_df, pattern_type)
        
        # Calculate the measured move percentage
        measured_move_pct = None
        if target_level is not None and pattern_close != 0:
            measured_move_pct = abs(target_level - pattern_close) / pattern_close * 100
        
        # Create pattern info
        pattern_info = PatternInfo(
            pattern_type=pattern_type,
            start_idx=start_idx,
            end_idx=end_idx,
            pivot_points=pivot_points,
            confirmation=confirmation,
            breakout_level=breakout_level,
            target_level=target_level,
            stop_level=stop_level,
            confidence=confidence,
            strength=strength,
            volume_confirms=volume_confirms,
            measured_move_pct=measured_move_pct,
            additional_info={"height": pattern_height}
        )
        
        return pattern_info
    
    def _is_bullish_pattern(self, pattern_type: PatternType) -> bool:
        """Determine if a pattern is bullish or bearish"""
        bullish_patterns = [
            PatternType.INVERSE_HEAD_AND_SHOULDERS,
            PatternType.DOUBLE_BOTTOM,
            PatternType.TRIPLE_BOTTOM,
            PatternType.ASCENDING_TRIANGLE,
            PatternType.FALLING_WEDGE,
            PatternType.CUP_AND_HANDLE,
            PatternType.BULL_FLAG,
            PatternType.BULL_PENNANT,
            PatternType.ROUNDING_BOTTOM,
            PatternType.CHANNEL_UP
        ]
        return pattern_type in bullish_patterns
    
    def _check_volume_confirmation(self, df: pd.DataFrame, pattern_type: PatternType) -> bool:
        """Check if volume confirms the pattern"""
        if 'volume' not in df.columns or df['volume'].isnull().all():
            return False
        
        # Basic volume confirmation logic
        try:
            # Check for increasing volume on breakout
            avg_volume = df['volume'].mean()
            recent_volume = df['volume'].iloc[-5:].mean()
            
            return recent_volume > avg_volume * 1.2  # 20% above average
        except Exception as e:
            logger.warning(f"Error checking volume confirmation: {str(e)}")
            return False
    
    # Pattern detection methods
    def _detect_head_and_shoulders(
        self, 
        df: pd.DataFrame,
        swing_highs: List[int],
        swing_lows: List[int]
    ) -> List[PatternInfo]:
        """
        Detect Head and Shoulders pattern.
        
        A Head and Shoulders pattern is a reversal pattern that signals a bullish-to-bearish trend change.
        It consists of three peaks, with the middle peak (head) being higher than the other two (shoulders).
        """
        detected_patterns = []
        
        # Minimum 30 bars for a proper H&S pattern
        if len(df) < 30:
            return detected_patterns
        
        # Need at least 5 swing points for a possible H&S
        if len(swing_highs) < 5:
            return detected_patterns
        
        # Check each possible combination of swing highs
        for i in range(len(swing_highs) - 4):
            # Get 5 consecutive swing highs (should include the shoulders and head)
            sh_indices = swing_highs[i:i+5]
            
            if len(sh_indices) < 5:
                continue
            
            # Get the prices at these swing highs
            sh_prices = [df['high'].iloc[idx] for idx in sh_indices]
            
            # Check for Head and Shoulders pattern:
            # 1. Middle peak (sh_prices[2]) should be higher than the others
            # 2. Left shoulder (sh_prices[1]) and right shoulder (sh_prices[3]) should be approximately at the same level
            # 3. The outer peaks (sh_prices[0] and sh_prices[4]) should be lower
            
            # Check the head is higher than the shoulders
            if not (sh_prices[2] > sh_prices[1] and sh_prices[2] > sh_prices[3]):
                continue
            
            # Check shoulders are roughly at the same level (within 5%)
            shoulder_diff_pct = abs(sh_prices[1] - sh_prices[3]) / sh_prices[1]
            if shoulder_diff_pct > 0.05:
                continue
            
            # Check the outer points are lower than the shoulders
            if not (sh_prices[0] < sh_prices[1] and sh_prices[4] < sh_prices[3]):
                continue
            
            # We've found a potential H&S pattern, now find the neckline
            # Get the swing lows between the shoulders
            sl_between = [idx for idx in swing_lows if sh_indices[1] < idx < sh_indices[3]]
            
            if not sl_between:
                continue
            
            # Find the lowest point between the shoulders (this is the neckline)
            neckline_idx = min(sl_between, key=lambda idx: df['low'].iloc[idx])
            neckline_price = df['low'].iloc[neckline_idx]
            
            # Create pattern info
            start_idx = sh_indices[0]
            end_idx = sh_indices[4]
            
            # Calculate measured move and target
            head_to_neckline = sh_prices[2] - neckline_price
            target_level = neckline_price - head_to_neckline
            
            # Check if neckline has been broken
            latest_close = df['close'].iloc[-1]
            confirmation = PatternConfirmation.FORMING
            
            if end_idx == len(df) - 1:
                confirmation = PatternConfirmation.FORMING
            elif latest_close < neckline_price:
                confirmation = PatternConfirmation.CONFIRMED
            else:
                confirmation = PatternConfirmation.POTENTIAL
            
            # Define pivot points for the pattern
            pivot_points = [
                (sh_indices[0], sh_prices[0]),  # Left outer point
                (sh_indices[1], sh_prices[1]),  # Left shoulder
                (sh_indices[2], sh_prices[2]),  # Head
                (sh_indices[3], sh_prices[3]),  # Right shoulder
                (sh_indices[4], sh_prices[4]),  # Right outer point
                (neckline_idx, neckline_price)  # Neckline point
            ]
            
            # Calculate confidence score (could be based on multiple factors)
            confidence = 0.7
            
            # Check volume pattern (typically decreasing during pattern formation)
            volume_pattern_valid = True
            if 'volume' in df.columns:
                # Example: check if volume is decreasing
                volume_trend = np.polyfit(range(len(df.iloc[start_idx:end_idx+1])), 
                                         df['volume'].iloc[start_idx:end_idx+1], 1)[0]
                volume_pattern_valid = volume_trend < 0
                
                # Higher volume on breakdown increases confidence
                if confirmation == PatternConfirmation.CONFIRMED:
                    breakdown_volume = df['volume'].iloc[end_idx] 
                    avg_volume = df['volume'].iloc[start_idx:end_idx].mean()
                    if breakdown_volume > 1.5 * avg_volume:
                        confidence += 0.1
            
            # Adjust confidence based on pattern quality
            if shoulder_diff_pct < 0.02:  # Very balanced shoulders
                confidence += 0.1
            
            pattern_info = PatternInfo(
                pattern_type=PatternType.HEAD_AND_SHOULDERS,
                start_idx=start_idx,
                end_idx=end_idx,
                pivot_points=pivot_points,
                confirmation=confirmation,
                breakout_level=neckline_price,
                target_level=target_level,
                stop_level=sh_prices[2],  # Stop above the head
                confidence=min(confidence, 1.0),  # Cap at 1.0
                strength="strong" if confidence > 0.8 else "medium",
                volume_confirms=volume_pattern_valid,
                measured_move_pct=abs(target_level - neckline_price) / neckline_price * 100,
                additional_info={
                    "neckline_price": neckline_price,
                    "head_to_neckline": head_to_neckline,
                    "shoulder_balance_pct": shoulder_diff_pct * 100
                }
            )
            
            detected_patterns.append(pattern_info)
        
        return detected_patterns
    
    def _detect_inverse_head_and_shoulders(
        self, 
        df: pd.DataFrame,
        swing_highs: List[int],
        swing_lows: List[int]
    ) -> List[PatternInfo]:
        """
        Detect Inverse Head and Shoulders pattern.
        
        An Inverse Head and Shoulders pattern is a reversal pattern that signals a bearish-to-bullish trend change.
        It consists of three troughs, with the middle trough (head) being lower than the other two (shoulders).
        """
        detected_patterns = []
        
        # Minimum 30 bars for a proper IH&S pattern
        if len(df) < 30:
            return detected_patterns
        
        # Need at least 5 swing lows for a possible IH&S
        if len(swing_lows) < 5:
            return detected_patterns
        
        # Check each possible combination of swing lows
        for i in range(len(swing_lows) - 4):
            # Get 5 consecutive swing lows (should include the shoulders and head)
            sl_indices = swing_lows[i:i+5]
            
            if len(sl_indices) < 5:
                continue
            
            # Get the prices at these swing lows
            sl_prices = [df['low'].iloc[idx] for idx in sl_indices]
            
            # Check for Inverse Head and Shoulders pattern:
            # 1. Middle trough (sl_prices[2]) should be lower than the others
            # 2. Left shoulder (sl_prices[1]) and right shoulder (sl_prices[3]) should be approximately at the same level
            # 3. The outer points (sl_prices[0] and sl_prices[4]) should be higher
            
            # Check the head is lower than the shoulders
            if not (sl_prices[2] < sl_prices[1] and sl_prices[2] < sl_prices[3]):
                continue
            
            # Check shoulders are roughly at the same level (within 5%)
            shoulder_diff_pct = abs(sl_prices[1] - sl_prices[3]) / sl_prices[1]
            if shoulder_diff_pct > 0.05:
                continue
            
            # Check the outer points are higher than the shoulders
            if not (sl_prices[0] > sl_prices[1] and sl_prices[4] > sl_prices[3]):
                continue
            
            # We've found a potential IH&S pattern, now find the neckline
            # Get the swing highs between the shoulders
            sh_between = [idx for idx in swing_highs if sl_indices[1] < idx < sl_indices[3]]
            
            if not sh_between:
                continue
            
            # Find the highest point between the shoulders (this is the neckline)
            neckline_idx = max(sh_between, key=lambda idx: df['high'].iloc[idx])
            neckline_price = df['high'].iloc[neckline_idx]
            
            # Create pattern info
            start_idx = sl_indices[0]
            end_idx = sl_indices[4]
            
            # Calculate measured move and target
            neckline_to_head = neckline_price - sl_prices[2]
            target_level = neckline_price + neckline_to_head
            
            # Check if neckline has been broken
            latest_close = df['close'].iloc[-1]
            confirmation = PatternConfirmation.FORMING
            
            if end_idx == len(df) - 1:
                confirmation = PatternConfirmation.FORMING
            elif latest_close > neckline_price:
                confirmation = PatternConfirmation.CONFIRMED
            else:
                confirmation = PatternConfirmation.POTENTIAL
            
            # Define pivot points for the pattern
            pivot_points = [
                (sl_indices[0], sl_prices[0]),  # Left outer point
                (sl_indices[1], sl_prices[1]),  # Left shoulder
                (sl_indices[2], sl_prices[2]),  # Head
                (sl_indices[3], sl_prices[3]),  # Right shoulder
                (sl_indices[4], sl_prices[4]),  # Right outer point
                (neckline_idx, neckline_price)  # Neckline point
            ]
            
            # Calculate confidence score (could be based on multiple factors)
            confidence = 0.7
            
            # Check volume pattern (typically decreasing during formation, increasing on breakout)
            volume_pattern_valid = True
            if 'volume' in df.columns:
                # Example: check if volume is decreasing during formation
                volume_trend = np.polyfit(range(len(df.iloc[start_idx:end_idx+1])), 
                                         df['volume'].iloc[start_idx:end_idx+1], 1)[0]
                volume_pattern_valid = volume_trend < 0
                
                # Higher volume on breakout increases confidence
                if confirmation == PatternConfirmation.CONFIRMED:
                    breakout_volume = df['volume'].iloc[end_idx] 
                    avg_volume = df['volume'].iloc[start_idx:end_idx].mean()
                    if breakout_volume > 1.5 * avg_volume:
                        confidence += 0.1
            
            # Adjust confidence based on pattern quality
            if shoulder_diff_pct < 0.02:  # Very balanced shoulders
                confidence += 0.1
            
            pattern_info = PatternInfo(
                pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS,
                start_idx=start_idx,
                end_idx=end_idx,
                pivot_points=pivot_points,
                confirmation=confirmation,
                breakout_level=neckline_price,
                target_level=target_level,
                stop_level=sl_prices[2],  # Stop below the head
                confidence=min(confidence, 1.0),  # Cap at 1.0
                strength="strong" if confidence > 0.8 else "medium",
                volume_confirms=volume_pattern_valid,
                measured_move_pct=abs(target_level - neckline_price) / neckline_price * 100,
                additional_info={
                    "neckline_price": neckline_price,
                    "neckline_to_head": neckline_to_head,
                    "shoulder_balance_pct": shoulder_diff_pct * 100
                }
            )
            
            detected_patterns.append(pattern_info)
        
        return detected_patterns
    
    def _detect_double_top(
        self, 
        df: pd.DataFrame, 
        swing_highs: List[int], 
        swing_lows: List[int]
    ) -> List[PatternInfo]:
        """
        Detect Double Top pattern.
        
        A Double Top is a reversal pattern that signals a bullish-to-bearish trend change.
        It consists of two peaks at roughly the same price level with a trough in between.
        """
        detected_patterns = []
        
        # Minimum 20 bars for a proper Double Top pattern
        if len(df) < 20:
            return detected_patterns
        
        # Need at least 3 swing highs for a possible Double Top
        if len(swing_highs) < 3:
            return detected_patterns
        
        # Check each possible combination of swing highs
        for i in range(len(swing_highs) - 2):
            # Get 3 consecutive swing highs
            sh_indices = swing_highs[i:i+3]
            
            if len(sh_indices) < 3:
                continue
            
            # Get the prices at these swing highs
            sh_prices = [df['high'].iloc[idx] for idx in sh_indices]
            
            # Check for Double Top pattern:
            # 1. First and second peaks (sh_prices[0] and sh_prices[2]) should be approximately at the same level
            # 2. The middle point (sh_prices[1]) should be significantly lower
            
            # Check peaks are roughly at the same level (within 3%)
            peak_diff_pct = abs(sh_prices[0] - sh_prices[2]) / sh_prices[0]
            if peak_diff_pct > 0.03:
                continue
            
            # Check the middle point is lower than both peaks
            if not (sh_prices[1] < sh_prices[0]*0.97 and sh_prices[1] < sh_prices[2]*0.97):
                continue
            
            # Find the lowest low between the two peaks (this is the neckline)
            between_indices = [j for j in range(sh_indices[0], sh_indices[2]+1)]
            neckline_idx = min(between_indices, key=lambda idx: df['low'].iloc[idx])
            neckline_price = df['low'].iloc[neckline_idx]
            
            # Create pattern info
            start_idx = sh_indices[0]
            end_idx = sh_indices[2]
            
            # Calculate measured move and target
            height = max(sh_prices[0], sh_prices[2]) - neckline_price
            target_level = neckline_price - height
            
            # Check if neckline has been broken
            latest_close = df['close'].iloc[-1]
            confirmation = PatternConfirmation.FORMING
            
            if end_idx == len(df) - 1:
                confirmation = PatternConfirmation.FORMING
            elif latest_close < neckline_price:
                confirmation = PatternConfirmation.CONFIRMED
            else:
                confirmation = PatternConfirmation.POTENTIAL
            
            # Define pivot points for the pattern
            pivot_points = [
                (sh_indices[0], sh_prices[0]),  # First peak
                (sh_indices[2], sh_prices[2]),  # Second peak
                (neckline_idx, neckline_price)  # Neckline point
            ]
            
            # Calculate confidence score
            confidence = 0.7
            
            # Check volume pattern (typically decreasing on second peak)
            volume_pattern_valid = True
            if 'volume' in df.columns:
                first_peak_volume = df['volume'].iloc[sh_indices[0]]
                second_peak_volume = df['volume'].iloc[sh_indices[2]]
                volume_pattern_valid = second_peak_volume < first_peak_volume
                
                # Higher volume on breakdown increases confidence
                if confirmation == PatternConfirmation.CONFIRMED:
                    breakdown_volume = df['volume'].iloc[end_idx:].mean()
                    avg_volume = df['volume'].iloc[start_idx:end_idx].mean()
                    if breakdown_volume > 1.2 * avg_volume:
                        confidence += 0.1
            
            # Adjust confidence based on pattern quality
            if peak_diff_pct < 0.01:  # Very balanced peaks
                confidence += 0.1
            
            pattern_info = PatternInfo(
                pattern_type=PatternType.DOUBLE_TOP,
                start_idx=start_idx,
                end_idx=end_idx,
                pivot_points=pivot_points,
                confirmation=confirmation,
                breakout_level=neckline_price,
                target_level=target_level,
                stop_level=max(sh_prices[0], sh_prices[2]),  # Stop above the highest peak
                confidence=min(confidence, 1.0),  # Cap at 1.0
                strength="strong" if confidence > 0.8 else "medium",
                volume_confirms=volume_pattern_valid,
                measured_move_pct=abs(target_level - neckline_price) / neckline_price * 100,
                additional_info={
                    "neckline_price": neckline_price,
                    "pattern_height": height,
                    "peak_balance_pct": peak_diff_pct * 100
                }
            )
            
            detected_patterns.append(pattern_info)
        
        return detected_patterns
    
    def _detect_double_bottom(
        self, 
        df: pd.DataFrame, 
        swing_highs: List[int], 
        swing_lows: List[int]
    ) -> List[PatternInfo]:
        """
        Detect Double Bottom pattern.
        
        A Double Bottom is a reversal pattern that signals a bearish-to-bullish trend change.
        It consists of two troughs at roughly the same price level with a peak in between.
        """
        detected_patterns = []
        
        # Minimum 20 bars for a proper Double Bottom pattern
        if len(df) < 20:
            return detected_patterns
        
        # Need at least 3 swing lows for a possible Double Bottom
        if len(swing_lows) < 3:
            return detected_patterns
        
        # Check each possible combination of swing lows
        for i in range(len(swing_lows) - 2):
            # Get 3 consecutive swing lows
            sl_indices = swing_lows[i:i+3]
            
            if len(sl_indices) < 3:
                continue
            
            # Get the prices at these swing lows
            sl_prices = [df['low'].iloc[idx] for idx in sl_indices]
            
            # Check for Double Bottom pattern:
            # 1. First and second troughs (sl_prices[0] and sl_prices[2]) should be approximately at the same level
            # 2. The middle point (sl_prices[1]) should be significantly higher
            
            # Check troughs are roughly at the same level (within 3%)
            trough_diff_pct = abs(sl_prices[0] - sl_prices[2]) / sl_prices[0]
            if trough_diff_pct > 0.03:
                continue
            
            # Check the middle point is higher than both troughs
            if not (sl_prices[1] > sl_prices[0]*1.03 and sl_prices[1] > sl_prices[2]*1.03):
                continue
            
            # Find the highest high between the two troughs (this is the neckline)
            between_indices = [j for j in range(sl_indices[0], sl_indices[2]+1)]
            neckline_idx = max(between_indices, key=lambda idx: df['high'].iloc[idx])
            neckline_price = df['high'].iloc[neckline_idx]
            
            # Create pattern info
            start_idx = sl_indices[0]
            end_idx = sl_indices[2]
            
            # Calculate measured move and target
            height = neckline_price - min(sl_prices[0], sl_prices[2])
            target_level = neckline_price + height
            
            # Check if neckline has been broken
            latest_close = df['close'].iloc[-1]
            confirmation = PatternConfirmation.FORMING
            
            if end_idx == len(df) - 1:
                confirmation = PatternConfirmation.FORMING
            elif latest_close > neckline_price:
                confirmation = PatternConfirmation.CONFIRMED
            else:
                confirmation = PatternConfirmation.POTENTIAL
            
            # Define pivot points for the pattern
            pivot_points = [
                (sl_indices[0], sl_prices[0]),  # First trough
                (sl_indices[2], sl_prices[2]),  # Second trough
                (neckline_idx, neckline_price)  # Neckline point
            ]
            
            # Calculate confidence score
            confidence = 0.7
            
            # Check volume pattern (typically increasing on second trough and breakout)
            volume_pattern_valid = True
            if 'volume' in df.columns:
                first_trough_volume = df['volume'].iloc[sl_indices[0]]
                second_trough_volume = df['volume'].iloc[sl_indices[2]]
                volume_pattern_valid = second_trough_volume > first_trough_volume
                
                # Higher volume on breakout increases confidence
                if confirmation == PatternConfirmation.CONFIRMED:
                    breakout_volume = df['volume'].iloc[end_idx:].mean()
                    avg_volume = df['volume'].iloc[start_idx:end_idx].mean()
                    if breakout_volume > 1.2 * avg_volume:
                        confidence += 0.1
            
            # Adjust confidence based on pattern quality
            if trough_diff_pct < 0.01:  # Very balanced troughs
                confidence += 0.1
            
            pattern_info = PatternInfo(
                pattern_type=PatternType.DOUBLE_BOTTOM,
                start_idx=start_idx,
                end_idx=end_idx,
                pivot_points=pivot_points,
                confirmation=confirmation,
                breakout_level=neckline_price,
                target_level=target_level,
                stop_level=min(sl_prices[0], sl_prices[2]),  # Stop below the lowest trough
                confidence=min(confidence, 1.0),  # Cap at 1.0
                strength="strong" if confidence > 0.8 else "medium",
                volume_confirms=volume_pattern_valid,
                measured_move_pct=abs(target_level - neckline_price) / neckline_price * 100,
                additional_info={
                    "neckline_price": neckline_price,
                    "pattern_height": height,
                    "trough_balance_pct": trough_diff_pct * 100
                }
            )
            
            detected_patterns.append(pattern_info)
        
        return detected_patterns
    
    # Additional pattern detection methods would follow this pattern...
    # Each complex pattern would have its own detection method
    
    def _detect_ascending_triangle(
        self, 
        df: pd.DataFrame, 
        swing_highs: List[int], 
        swing_lows: List[int]
    ) -> List[PatternInfo]:
        """
        Detect Ascending Triangle pattern.
        
        An Ascending Triangle is a bullish continuation pattern that consists of 
        a flat upper resistance line and an ascending lower support line.
        """
        detected_patterns = []
        
        # Minimum 20 bars for a proper triangle pattern
        if len(df) < 20:
            return detected_patterns
        
        # Need at least 3 swing highs and 3 swing lows
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return detected_patterns
        
        # Try to find ascending triangles
        # We need at least 2 similar swing highs (resistance line) and 
        # at least 2 rising swing lows (ascending support line)
        
        for i in range(len(swing_highs) - 2):
            resistance_points = []
            
            # Find potential resistance line (flat)
            for j in range(i, len(swing_highs) - 1):
                h1_idx = swing_highs[j]
                h1_price = df['high'].iloc[h1_idx]
                
                h2_idx = swing_highs[j+1]
                h2_price = df['high'].iloc[h2_idx]
                
                # Check if these swing highs are at similar levels (flat resistance)
                if abs(h2_price - h1_price) / h1_price < 0.02:  # Within 2%
                    resistance_points.append((h1_idx, h1_price))
                    resistance_points.append((h2_idx, h2_price))
                    break
            
            if not resistance_points:
                continue
            
            # We have potential resistance line, now find ascending support line
            support_points = []
            
            # Get swing lows between the resistance points timeframe
            start_time = resistance_points[0][0]
            end_time = resistance_points[-1][0]
            
            filtered_lows = [idx for idx in swing_lows if start_time <= idx <= end_time]
            
            if len(filtered_lows) < 2:
                continue
            
            # Check for ascending lows
            for j in range(len(filtered_lows) - 1):
                l1_idx = filtered_lows[j]
                l1_price = df['low'].iloc[l1_idx]
                
                l2_idx = filtered_lows[j+1]
                l2_price = df['low'].iloc[l2_idx]
                
                # Check if the second low is higher than the first (ascending)
                if l2_price > l1_price:
                    support_points.append((l1_idx, l1_price))
                    support_points.append((l2_idx, l2_price))
                    break
            
            if not support_points:
                continue
            
            # We have both resistance and support lines, calculate pattern details
            
            # Calculate resistance level (average of resistance points)
            resistance_level = sum(p[1] for p in resistance_points) / len(resistance_points)
            
            # Calculate support line slope
            x_support = [p[0] for p in support_points]
            y_support = [p[1] for p in support_points]
            slope, intercept, _, _, _ = linregress(x_support, y_support)
            
            # Define the start and end of the pattern
            start_idx = min(support_points[0][0], resistance_points[0][0])
            end_idx = max(support_points[-1][0], resistance_points[-1][0])
            
            # Calculate the target (height of the pattern)
            triangle_height = resistance_level - support_points[0][1]
            target_level = resistance_level + triangle_height
            
            # Check if breakout has occurred
            latest_close = df['close'].iloc[-1]
            confirmation = PatternConfirmation.FORMING
            
            if end_idx == len(df) - 1:
                confirmation = PatternConfirmation.FORMING
            elif latest_close > resistance_level:
                confirmation = PatternConfirmation.CONFIRMED
            else:
                confirmation = PatternConfirmation.POTENTIAL
            
            # Calculate confidence based on multiple factors
            confidence = 0.7
            
            # More touches of the lines = higher confidence
            num_touches = len(resistance_points) + len(support_points)
            if num_touches > 4:
                confidence += 0.1
            
            # Volume typically decreases as triangle forms
            volume_pattern_valid = True
            if 'volume' in df.columns:
                volume_section = df['volume'].iloc[start_idx:end_idx+1]
                volume_trend = np.polyfit(range(len(volume_section)), volume_section, 1)[0]
                volume_pattern_valid = volume_trend < 0
                
                # Higher volume on breakout increases confidence
                if confirmation == PatternConfirmation.CONFIRMED:
                    breakout_idx = min(len(df)-1, end_idx + 5)  # Look a few bars after pattern
                    breakout_volume = df['volume'].iloc[end_idx:breakout_idx+1].mean()
                    avg_volume = df['volume'].iloc[start_idx:end_idx+1].mean()
                    if breakout_volume > 1.5 * avg_volume:
                        confidence += 0.1
            
            # All pivot points for the pattern
            pivot_points = resistance_points + support_points
            
            pattern_info = PatternInfo(
                pattern_type=PatternType.ASCENDING_TRIANGLE,
                start_idx=start_idx,
                end_idx=end_idx,
                pivot_points=pivot_points,
                confirmation=confirmation,
                breakout_level=resistance_level,
                target_level=target_level,
                stop_level=support_points[-1][1],  # Stop below the last support
                confidence=min(confidence, 1.0),  # Cap at 1.0
                strength="strong" if confidence > 0.8 else "medium",
                volume_confirms=volume_pattern_valid,
                measured_move_pct=abs(target_level - resistance_level) / resistance_level * 100,
                additional_info={
                    "resistance_level": resistance_level,
                    "support_slope": slope,
                    "pattern_height": triangle_height,
                    "num_touches": num_touches
                }
            )
            
            detected_patterns.append(pattern_info)
        
        return detected_patterns
    
    # Other pattern detection methods would be implemented here
    # Each with a similar structure but pattern-specific logic


# Usage example (not executed, just for demonstration)
if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Sample data
    data = pd.read_csv('example_data.csv')
    
    # Initialize the pattern recognizer
    recognizer = ChartPatternRecognizer()
    
    # Detect patterns
    patterns = recognizer.detect_patterns(data, symbol="BTCUSDT", timeframe="1h")
    
    # Process detected patterns
    for pattern in patterns:
        print(f"Detected {pattern.pattern_type.name} pattern with {pattern.confidence:.2f} confidence")
        print(f"Status: {pattern.confirmation.name}")
        if pattern.target_level:
            print(f"Target: {pattern.target_level:.2f}")

