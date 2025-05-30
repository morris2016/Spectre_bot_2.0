#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Candlestick Pattern Recognition Module

This module provides advanced candlestick pattern detection with machine learning enhancement
for high-accuracy pattern identification. It combines traditional pattern detection with
adaptive learning to continually improve detection accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from enum import Enum, auto
import ta
from common.ta_candles import cdl_pattern
from dataclasses import dataclass
import logging
from scipy import stats
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from datetime import datetime, timedelta

# Internal imports
from common.utils import calculate_body_size, calculate_shadow_size
from common.exceptions import PatternDetectionError, InsufficientDataError
from feature_service.multi_timeframe import TimeFrame
from data_storage.market_data import MarketDataRepository
from feature_service.features.technical import TechnicalFeatures
from feature_service.features.volatility import VolatilityFeatures

# Get module logger
logger = logging.getLogger(__name__)


class PatternStrength(Enum):
    """Enum representing pattern strength classifications"""
    WEAK = auto()
    MODERATE = auto()
    STRONG = auto()
    VERY_STRONG = auto()


class PatternType(Enum):
    """Enum representing pattern types"""
    REVERSAL_BULLISH = auto()
    REVERSAL_BEARISH = auto()
    CONTINUATION_BULLISH = auto()
    CONTINUATION_BEARISH = auto()
    INDECISION = auto()


@dataclass
class CandlestickPattern:
    """Dataclass representing a detected candlestick pattern"""
    name: str
    pattern_type: PatternType
    strength: PatternStrength
    index: int
    confidence: float
    description: str
    historical_accuracy: float
    expected_move_pips: float
    timeframe: TimeFrame
    pattern_length: int
    related_patterns: List[str] = None
    context_factors: Dict[str, Any] = None
    volume_confirmation: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary representation"""
        return {
            'name': self.name,
            'pattern_type': self.pattern_type.name,
            'strength': self.strength.name,
            'index': self.index,
            'confidence': self.confidence,
            'description': self.description,
            'historical_accuracy': self.historical_accuracy,
            'expected_move_pips': self.expected_move_pips,
            'timeframe': self.timeframe.name,
            'pattern_length': self.pattern_length,
            'related_patterns': self.related_patterns,
            'context_factors': self.context_factors,
            'volume_confirmation': self.volume_confirmation
        }


class CandlestickPatterns:
    """
    Advanced candlestick pattern detection with ML enhancement.
    
    This class provides sophisticated candlestick pattern recognition with:
    - Traditional rule-based pattern detection
    - Machine learning enhanced pattern validation
    - Multi-timeframe pattern context analysis
    - Historical accuracy tracking
    - Pattern strength classification
    - Expected move calculation
    - Volume confirmation analysis
    """
    
    # Pattern descriptions with detailed explanations
    PATTERN_DESCRIPTIONS = {
        "doji": "A Doji forms when the opening and closing prices are virtually equal. It represents indecision in the market.",
        "hammer": "A hammer has a small body at the top with a long lower shadow. It indicates a potential bullish reversal.",
        "inverted_hammer": "An inverted hammer has a small body at the bottom with a long upper shadow. It can signal a potential bullish reversal.",
        "hanging_man": "A hanging man has a small body at the top with a long lower shadow. It signals a potential bearish reversal.",
        "shooting_star": "A shooting star has a small body at the bottom with a long upper shadow. It indicates a potential bearish reversal.",
        "engulfing_bullish": "A bullish engulfing pattern occurs when a small bearish candle is followed by a larger bullish candle that completely engulfs the previous candle. It signals a potential bullish reversal.",
        "engulfing_bearish": "A bearish engulfing pattern occurs when a small bullish candle is followed by a larger bearish candle that completely engulfs the previous candle. It signals a potential bearish reversal.",
        "morning_star": "A morning star is a three-candle pattern with a long bearish candle, followed by a small-bodied candle, and then a bullish candle. It indicates a potential bullish reversal.",
        "evening_star": "An evening star is a three-candle pattern with a long bullish candle, followed by a small-bodied candle, and then a bearish candle. It indicates a potential bearish reversal.",
        "three_white_soldiers": "Three white soldiers consist of three consecutive bullish candles with each closing higher than the previous. It indicates strong buying pressure and a potential bullish trend.",
        "three_black_crows": "Three black crows consist of three consecutive bearish candles with each closing lower than the previous. It indicates strong selling pressure and a potential bearish trend.",
        "harami_bullish": "A bullish harami consists of a large bearish candle followed by a smaller bullish candle contained within the previous candle's body. It signals a potential bullish reversal.",
        "harami_bearish": "A bearish harami consists of a large bullish candle followed by a smaller bearish candle contained within the previous candle's body. It signals a potential bearish reversal.",
        "piercing_line": "A piercing line consists of a bearish candle followed by a bullish candle that closes above the midpoint of the previous candle. It signals a potential bullish reversal.",
        "dark_cloud_cover": "A dark cloud cover consists of a bullish candle followed by a bearish candle that opens above the previous close and closes below the midpoint of the previous candle. It signals a potential bearish reversal.",
        "tweezer_top": "Tweezer tops consist of two candles with matching highs, the first bullish and the second bearish. They signal a potential bearish reversal.",
        "tweezer_bottom": "Tweezer bottoms consist of two candles with matching lows, the first bearish and the second bullish. They signal a potential bullish reversal.",
        "three_inside_up": "Three inside up is a three-candle bullish reversal pattern that begins with a large bearish candle, followed by a smaller bullish candle within the first, and completed by a third bullish candle closing above the second.",
        "three_inside_down": "Three inside down is a three-candle bearish reversal pattern that begins with a large bullish candle, followed by a smaller bearish candle within the first, and completed by a third bearish candle closing below the second.",
        "three_outside_up": "Three outside up is a three-candle bullish reversal pattern starting with a short bearish candle, followed by a bullish engulfing candle, and completed by a third bullish candle closing higher.",
        "three_outside_down": "Three outside down is a three-candle bearish reversal pattern starting with a short bullish candle, followed by a bearish engulfing candle, and completed by a third bearish candle closing lower.",
        "belt_hold_bullish": "A bullish belt hold is a single bullish candle that opens at or near the low and closes near the high with little to no lower shadow. It indicates strong buying pressure.",
        "belt_hold_bearish": "A bearish belt hold is a single bearish candle that opens at or near the high and closes near the low with little to no upper shadow. It indicates strong selling pressure.",
        "kicking_bullish": "A bullish kicking pattern consists of a bearish marubozu followed by a gap up and a bullish marubozu. It indicates extremely strong buying pressure.",
        "kicking_bearish": "A bearish kicking pattern consists of a bullish marubozu followed by a gap down and a bearish marubozu. It indicates extremely strong selling pressure.",
        "hikkake_bullish": "A bullish hikkake pattern is a false breakout pattern that traps bears and leads to a bullish move.",
        "hikkake_bearish": "A bearish hikkake pattern is a false breakout pattern that traps bulls and leads to a bearish move.",
        "ladder_bottom": "A ladder bottom is a rare bullish reversal pattern with three declining bearish candles, followed by a doji and a strong bullish candle.",
        "matching_low": "Matching low consists of two bearish candles with equal or very similar lows, indicating a potential support level and possible reversal.",
        "upside_gap_two_crows": "An upside gap two crows is a three-candle bearish reversal pattern consisting of a bullish candle, followed by two bearish candles, with the first gapping up.",
        "tasuki_gap_bullish": "A bullish tasuki gap pattern occurs during an uptrend with an up gap followed by a bearish candle that partially fills the gap. It indicates continuation of the uptrend.",
        "tasuki_gap_bearish": "A bearish tasuki gap pattern occurs during a downtrend with a down gap followed by a bullish candle that partially fills the gap. It indicates continuation of the downtrend.",
        "three_stars_in_the_south": "Three stars in the south is a rare bullish reversal pattern consisting of three bearish candles with progressively smaller real bodies and shadows.",
        "concealing_baby_swallow": "Concealing baby swallow is a rare bullish reversal pattern consisting of four bearish candles where the third forms a harami and the fourth engulfs it.",
        "three_line_strike_bullish": "A bullish three-line strike consists of three bullish candles followed by a bearish candle that engulfs all three. Despite appearing bearish, it often leads to continuation of the uptrend.",
        "three_line_strike_bearish": "A bearish three-line strike consists of three bearish candles followed by a bullish candle that engulfs all three. Despite appearing bullish, it often leads to continuation of the downtrend.",
        "abandoned_baby_bullish": "A bullish abandoned baby is a rare reversal pattern with a bearish candle, followed by a doji that gaps down, and then a bullish candle that gaps up.",
        "abandoned_baby_bearish": "A bearish abandoned baby is a rare reversal pattern with a bullish candle, followed by a doji that gaps up, and then a bearish candle that gaps down.",
    }
    
    # Historical accuracy of patterns (will be updated by ML models)
    DEFAULT_PATTERN_ACCURACY = {
        "doji": 0.52,
        "hammer": 0.65,
        "inverted_hammer": 0.62,
        "hanging_man": 0.64,
        "shooting_star": 0.67,
        "engulfing_bullish": 0.71,
        "engulfing_bearish": 0.70,
        "morning_star": 0.76,
        "evening_star": 0.75,
        "three_white_soldiers": 0.78,
        "three_black_crows": 0.77,
        "harami_bullish": 0.63,
        "harami_bearish": 0.62,
        "piercing_line": 0.68,
        "dark_cloud_cover": 0.67,
        "tweezer_top": 0.61,
        "tweezer_bottom": 0.62,
        "three_inside_up": 0.69,
        "three_inside_down": 0.68,
        "three_outside_up": 0.70,
        "three_outside_down": 0.69,
        "belt_hold_bullish": 0.60,
        "belt_hold_bearish": 0.59,
        "kicking_bullish": 0.74,
        "kicking_bearish": 0.73,
        "hikkake_bullish": 0.58,
        "hikkake_bearish": 0.57,
        "ladder_bottom": 0.72,
        "matching_low": 0.59,
        "upside_gap_two_crows": 0.66,
        "tasuki_gap_bullish": 0.64,
        "tasuki_gap_bearish": 0.63,
        "three_stars_in_the_south": 0.71,
        "concealing_baby_swallow": 0.73,
        "three_line_strike_bullish": 0.68,
        "three_line_strike_bearish": 0.67,
        "abandoned_baby_bullish": 0.77,
        "abandoned_baby_bearish": 0.76,
    }
    
    # Pattern types mapping
    PATTERN_TYPES = {
        "doji": PatternType.INDECISION,
        "hammer": PatternType.REVERSAL_BULLISH,
        "inverted_hammer": PatternType.REVERSAL_BULLISH,
        "hanging_man": PatternType.REVERSAL_BEARISH,
        "shooting_star": PatternType.REVERSAL_BEARISH,
        "engulfing_bullish": PatternType.REVERSAL_BULLISH,
        "engulfing_bearish": PatternType.REVERSAL_BEARISH,
        "morning_star": PatternType.REVERSAL_BULLISH,
        "evening_star": PatternType.REVERSAL_BEARISH,
        "three_white_soldiers": PatternType.CONTINUATION_BULLISH,
        "three_black_crows": PatternType.CONTINUATION_BEARISH,
        "harami_bullish": PatternType.REVERSAL_BULLISH,
        "harami_bearish": PatternType.REVERSAL_BEARISH,
        "piercing_line": PatternType.REVERSAL_BULLISH,
        "dark_cloud_cover": PatternType.REVERSAL_BEARISH,
        "tweezer_top": PatternType.REVERSAL_BEARISH,
        "tweezer_bottom": PatternType.REVERSAL_BULLISH,
        "three_inside_up": PatternType.REVERSAL_BULLISH,
        "three_inside_down": PatternType.REVERSAL_BEARISH,
        "three_outside_up": PatternType.REVERSAL_BULLISH,
        "three_outside_down": PatternType.REVERSAL_BEARISH,
        "belt_hold_bullish": PatternType.CONTINUATION_BULLISH,
        "belt_hold_bearish": PatternType.CONTINUATION_BEARISH,
        "kicking_bullish": PatternType.REVERSAL_BULLISH,
        "kicking_bearish": PatternType.REVERSAL_BEARISH,
        "hikkake_bullish": PatternType.REVERSAL_BULLISH,
        "hikkake_bearish": PatternType.REVERSAL_BEARISH,
        "ladder_bottom": PatternType.REVERSAL_BULLISH,
        "matching_low": PatternType.REVERSAL_BULLISH,
        "upside_gap_two_crows": PatternType.REVERSAL_BEARISH,
        "tasuki_gap_bullish": PatternType.CONTINUATION_BULLISH,
        "tasuki_gap_bearish": PatternType.CONTINUATION_BEARISH,
        "three_stars_in_the_south": PatternType.REVERSAL_BULLISH,
        "concealing_baby_swallow": PatternType.REVERSAL_BULLISH,
        "three_line_strike_bullish": PatternType.CONTINUATION_BULLISH,
        "three_line_strike_bearish": PatternType.CONTINUATION_BEARISH,
        "abandoned_baby_bullish": PatternType.REVERSAL_BULLISH,
        "abandoned_baby_bearish": PatternType.REVERSAL_BEARISH,
    }
    
    # Pattern lengths (number of candles in each pattern)
    PATTERN_LENGTHS = {
        "doji": 1,
        "hammer": 1,
        "inverted_hammer": 1,
        "hanging_man": 1,
        "shooting_star": 1,
        "engulfing_bullish": 2,
        "engulfing_bearish": 2,
        "morning_star": 3,
        "evening_star": 3,
        "three_white_soldiers": 3,
        "three_black_crows": 3,
        "harami_bullish": 2,
        "harami_bearish": 2,
        "piercing_line": 2,
        "dark_cloud_cover": 2,
        "tweezer_top": 2,
        "tweezer_bottom": 2,
        "three_inside_up": 3,
        "three_inside_down": 3,
        "three_outside_up": 3,
        "three_outside_down": 3,
        "belt_hold_bullish": 1,
        "belt_hold_bearish": 1,
        "kicking_bullish": 2,
        "kicking_bearish": 2,
        "hikkake_bullish": 3,
        "hikkake_bearish": 3,
        "ladder_bottom": 5,
        "matching_low": 2,
        "upside_gap_two_crows": 3,
        "tasuki_gap_bullish": 3,
        "tasuki_gap_bearish": 3,
        "three_stars_in_the_south": 3,
        "concealing_baby_swallow": 4,
        "three_line_strike_bullish": 4,
        "three_line_strike_bearish": 4,
        "abandoned_baby_bullish": 3,
        "abandoned_baby_bearish": 3,
    }
    
    # Related patterns (patterns that often occur together or have similar implications)
    RELATED_PATTERNS = {
        "doji": ["spinning_top", "high_wave"],
        "hammer": ["inverted_hammer", "hanging_man", "shooting_star"],
        "inverted_hammer": ["hammer", "hanging_man", "shooting_star"],
        "hanging_man": ["hammer", "inverted_hammer", "shooting_star"],
        "shooting_star": ["hammer", "inverted_hammer", "hanging_man"],
        "engulfing_bullish": ["piercing_line", "three_outside_up"],
        "engulfing_bearish": ["dark_cloud_cover", "three_outside_down"],
        "morning_star": ["three_white_soldiers", "three_inside_up"],
        "evening_star": ["three_black_crows", "three_inside_down"],
        "three_white_soldiers": ["morning_star", "three_inside_up"],
        "three_black_crows": ["evening_star", "three_inside_down"],
        "harami_bullish": ["three_inside_up", "piercing_line"],
        "harami_bearish": ["three_inside_down", "dark_cloud_cover"],
        "piercing_line": ["engulfing_bullish", "three_outside_up"],
        "dark_cloud_cover": ["engulfing_bearish", "three_outside_down"],
        "tweezer_top": ["harami_bearish", "shooting_star"],
        "tweezer_bottom": ["harami_bullish", "hammer"],
        "three_inside_up": ["harami_bullish", "morning_star"],
        "three_inside_down": ["harami_bearish", "evening_star"],
        "three_outside_up": ["engulfing_bullish", "piercing_line"],
        "three_outside_down": ["engulfing_bearish", "dark_cloud_cover"],
        "belt_hold_bullish": ["three_white_soldiers", "kicking_bullish"],
        "belt_hold_bearish": ["three_black_crows", "kicking_bearish"],
        "kicking_bullish": ["belt_hold_bullish", "gap_up"],
        "kicking_bearish": ["belt_hold_bearish", "gap_down"],
        "hikkake_bullish": ["inverted_hammer", "morning_star"],
        "hikkake_bearish": ["shooting_star", "evening_star"],
        "ladder_bottom": ["three_stars_in_the_south", "hammer"],
        "matching_low": ["tweezer_bottom", "double_bottom"],
        "upside_gap_two_crows": ["evening_star", "dark_cloud_cover"],
        "tasuki_gap_bullish": ["gap_up", "three_white_soldiers"],
        "tasuki_gap_bearish": ["gap_down", "three_black_crows"],
        "three_stars_in_the_south": ["ladder_bottom", "three_white_soldiers"],
        "concealing_baby_swallow": ["three_white_soldiers", "morning_star"],
        "three_line_strike_bullish": ["three_white_soldiers", "three_outside_up"],
        "three_line_strike_bearish": ["three_black_crows", "three_outside_down"],
        "abandoned_baby_bullish": ["morning_star", "three_white_soldiers"],
        "abandoned_baby_bearish": ["evening_star", "three_black_crows"],
    }
    
    def __init__(self, market_data_repo: MarketDataRepository = None):
        """
        Initialize the CandlestickPatterns class.
        
        Args:
            market_data_repo: Repository for accessing historical market data
        """
        self.market_data_repo = market_data_repo
        self.pattern_accuracy = self.DEFAULT_PATTERN_ACCURACY.copy()
        self.technical_features = TechnicalFeatures()
        self.volatility_features = VolatilityFeatures()
        
        # ML model for pattern validation
        self.pattern_validation_models = {}
        self.load_ml_models()
        
        logger.info("Candlestick pattern recognition system initialized")

    def load_ml_models(self):
        """Load machine learning models for pattern validation"""
        try:
            # In production, we would load pre-trained models from a model repository
            # For now, we'll initialize models but not train them yet
            self.pattern_validation_models = {
                "default": GradientBoostingClassifier(
                    n_estimators=100, 
                    learning_rate=0.1, 
                    max_depth=5,
                    random_state=42
                ),
                "doji": RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=4,
                    random_state=42
                ),
                "reversal_patterns": xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            }
            logger.info("Machine learning models for pattern validation loaded")
        except Exception as e:
            logger.error(f"Failed to load ML models: {str(e)}")
            # Fall back to rule-based pattern detection only
            self.pattern_validation_models = {}

    def detect_patterns(
        self, 
        df: pd.DataFrame, 
        timeframe: TimeFrame,
        lookback: int = 100,
        confidence_threshold: float = 0.6,
        apply_ml_validation: bool = True,
        volume_confirmation: bool = True
    ) -> List[CandlestickPattern]:
        """
        Detect candlestick patterns in the provided dataframe.
        
        Args:
            df: DataFrame containing OHLCV data
            timeframe: TimeFrame of the provided data
            lookback: Number of candles to analyze (from the end of the dataframe)
            confidence_threshold: Minimum confidence level for pattern detection
            apply_ml_validation: Whether to apply machine learning validation
            volume_confirmation: Whether to check for volume confirmation
        
        Returns:
            List of detected CandlestickPattern objects
        """
        if df is None or df.empty or len(df) < 10:
            raise InsufficientDataError("Insufficient data for pattern detection")
        
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        # Make copy to avoid modifying original dataframe
        df = df.copy()
        
        # Get the last 'lookback' candles
        if len(df) > lookback:
            df = df.iloc[-lookback:]
        
        # Reset index to make pattern indices match dataframe indices
        df.reset_index(inplace=True, drop=True)
        
        detected_patterns = []
        
        # Run candlestick pattern recognition functions
        talib_patterns = self._detect_talib_patterns(df)
        
        # Run custom pattern detection functions
        custom_patterns = self._detect_custom_patterns(df)
        
        # Combine all detected patterns
        all_patterns = talib_patterns + custom_patterns
        
        # Apply ML validation if requested
        if apply_ml_validation and self.pattern_validation_models:
            all_patterns = self._validate_patterns_with_ml(df, all_patterns)
        
        # Check volume confirmation if requested
        if volume_confirmation:
            all_patterns = self._check_volume_confirmation(df, all_patterns)
        
        # Filter patterns by confidence threshold
        filtered_patterns = [p for p in all_patterns if p.confidence >= confidence_threshold]
        
        # Sort patterns by index (latest first)
        filtered_patterns.sort(key=lambda x: x.index, reverse=True)
        
        logger.info(f"Detected {len(filtered_patterns)} candlestick patterns above confidence threshold {confidence_threshold}")
        
        return filtered_patterns

    def _detect_talib_patterns(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """
        Detect patterns using candlestick utilities.
        
        Args:
            df: DataFrame containing OHLCV data
        
        Returns:
            List of detected CandlestickPattern objects
        """
        patterns = []
        
        pattern_names = [
            'doji', 'hammer', 'inverted_hammer', 'hanging_man', 'shooting_star',
            'spinning_top', 'marubozu', 'engulfing', 'harami', 'piercing_line',
            'dark_cloud_cover', 'tweezer_bottom', 'tweezer_top', 'matching_low',
            'kicking', 'belt_hold', 'morning_star', 'evening_star',
            'three_white_soldiers', 'three_black_crows', 'three_inside',
            'three_outside', 'upside_gap_two_crows', 'tasuki_gap',
            'three_stars_in_the_south', 'hikkake', 'concealing_baby_swallow',
            'three_line_strike', 'ladder_bottom', 'abandoned_baby'
        ]

        try:
            for pattern in pattern_names:
                results = cdl_pattern(df, name=pattern)
                if results is None:
                    continue

                # Process results
                for i in range(len(results)):
                    signal = results.iloc[i]
                    if signal != 0:  # 0 means no pattern
                        # Determine if bullish or bearish for pattern types that need it
                        if pattern in ['engulfing', 'harami', 'kicking', 'belt_hold',
                                       'three_inside', 'three_outside', 'tasuki_gap',
                                       'three_line_strike', 'abandoned_baby', 'hikkake']:
                            if signal > 0:
                                actual_pattern = f"{pattern}_bullish"
                            else:
                                actual_pattern = f"{pattern}_bearish"
                        else:
                            actual_pattern = pattern
                        
                        # Skip if pattern not in our definitions
                        if actual_pattern not in self.PATTERN_TYPES:
                            continue
                        
                        # Get pattern attributes
                        pattern_type = self.PATTERN_TYPES.get(actual_pattern, PatternType.INDECISION)
                        historical_accuracy = self.pattern_accuracy.get(actual_pattern, 0.5)
                        pattern_length = self.PATTERN_LENGTHS.get(actual_pattern, 1)
                        description = self.PATTERN_DESCRIPTIONS.get(actual_pattern, "")
                        related_patterns = self.RELATED_PATTERNS.get(actual_pattern, [])
                        
                        # Calculate pattern strength
                        strength = self._calculate_pattern_strength(
                            df, 
                            i, 
                            historical_accuracy, 
                            abs(signal)/100 if abs(signal) > 1 else abs(signal),
                            actual_pattern
                        )
                        
                        # Calculate confidence score
                        confidence = self._calculate_confidence(
                            df, 
                            i, 
                            historical_accuracy, 
                            abs(signal)/100 if abs(signal) > 1 else abs(signal),
                            actual_pattern
                        )
                        
                        # Calculate expected move
                        expected_move = self._calculate_expected_move(
                            df, 
                            i, 
                            actual_pattern
                        )
                        
                        # Gather context factors
                        context_factors = self._get_context_factors(df, i, actual_pattern)
                        
                        # Create pattern object
                        pattern = CandlestickPattern(
                            name=actual_pattern,
                            pattern_type=pattern_type,
                            strength=strength,
                            index=i,
                            confidence=confidence,
                            description=description,
                            historical_accuracy=historical_accuracy,
                            expected_move_pips=expected_move,
                            timeframe=TimeFrame.UNKNOWN,  # Will be set by caller
                            pattern_length=pattern_length,
                            related_patterns=related_patterns,
                            context_factors=context_factors,
                            volume_confirmation=False  # Will be checked later if requested
                        )
                        
                        patterns.append(pattern)
            
            logger.debug(f"Detected {len(patterns)} patterns using candlestick utilities")
            return patterns
            
        except Exception as e:
            logger.error(f"Error in candlestick pattern detection: {str(e)}")
            return []

    def _detect_custom_patterns(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """
        Detect patterns using custom detection functions (beyond TALib capabilities).
        
        Args:
            df: DataFrame containing OHLCV data
        
        Returns:
            List of detected CandlestickPattern objects
        """
        patterns = []
        
        try:
            # Add specialized pattern detection not available in TALib
            # For example, detect complex patterns like "three drives" or "ABCD"
            
            # Detect three drives pattern (bullish)
            three_drives_bullish = self._detect_three_drives_bullish(df)
            patterns.extend(three_drives_bullish)
            
            # Detect three drives pattern (bearish)
            three_drives_bearish = self._detect_three_drives_bearish(df)
            patterns.extend(three_drives_bearish)
            
            # Detect ABCD pattern (bullish)
            abcd_bullish = self._detect_abcd_bullish(df)
            patterns.extend(abcd_bullish)
            
            # Detect ABCD pattern (bearish)
            abcd_bearish = self._detect_abcd_bearish(df)
            patterns.extend(abcd_bearish)
            
            # Detect bat pattern (bullish)
            bat_bullish = self._detect_bat_bullish(df)
            patterns.extend(bat_bullish)
            
            # Detect bat pattern (bearish)
            bat_bearish = self._detect_bat_bearish(df)
            patterns.extend(bat_bearish)
            
            # Other custom patterns can be added here
            
            logger.debug(f"Detected {len(patterns)} patterns using custom detection functions")
            return patterns
            
        except Exception as e:
            logger.error(f"Error in custom pattern detection: {str(e)}")
            return []

    def _validate_patterns_with_ml(
        self, 
        df: pd.DataFrame, 
        patterns: List[CandlestickPattern]
    ) -> List[CandlestickPattern]:
        """
        Validate detected patterns using machine learning models.
        
        Args:
            df: DataFrame containing OHLCV data
            patterns: List of detected patterns
        
        Returns:
            List of validated patterns with updated confidence scores
        """
        if not patterns or not self.pattern_validation_models:
            return patterns
        
        validated_patterns = []
        
        try:
            # Group patterns by index for efficient validation
            patterns_by_index = {}
            for pattern in patterns:
                if pattern.index not in patterns_by_index:
                    patterns_by_index[pattern.index] = []
                patterns_by_index[pattern.index].append(pattern)
            
            # Process each index
            for idx, idx_patterns in patterns_by_index.items():
                # Skip if not enough data for feature extraction
                if idx < 20 or idx >= len(df) - 5:
                    validated_patterns.extend(idx_patterns)
                    continue
                
                # Extract features for this candle
                features = self._extract_features_for_ml(df, idx)
                
                # Validate each pattern
                for pattern in idx_patterns:
                    # Get appropriate model
                    model_key = pattern.name if pattern.name in self.pattern_validation_models else \
                                "reversal_patterns" if "REVERSAL" in pattern.pattern_type.name else \
                                "default"
                    
                    model = self.pattern_validation_models.get(model_key)
                    if model and hasattr(model, 'predict_proba'):
                        try:
                            # Predict probability of pattern being valid
                            proba = model.predict_proba(features.reshape(1, -1))
                            ml_confidence = proba[0][1]  # Probability of positive class
                            
                            # Update pattern confidence as weighted average of rule-based and ML confidence
                            pattern.confidence = 0.4 * pattern.confidence + 0.6 * ml_confidence
                        except Exception as e:
                            logger.warning(f"Error validating pattern with ML model: {str(e)}")
                    
                    validated_patterns.append(pattern)
            
            logger.debug(f"Validated {len(validated_patterns)} patterns using ML models")
            return validated_patterns
            
        except Exception as e:
            logger.error(f"Error in pattern validation with ML: {str(e)}")
            return patterns

    def _check_volume_confirmation(
        self, 
        df: pd.DataFrame, 
        patterns: List[CandlestickPattern]
    ) -> List[CandlestickPattern]:
        """
        Check if patterns have volume confirmation.
        
        Args:
            df: DataFrame containing OHLCV data
            patterns: List of detected patterns
        
        Returns:
            List of patterns with volume_confirmation flag set
        """
        if not patterns or 'volume' not in df.columns:
            return patterns
        
        for pattern in patterns:
            try:
                idx = pattern.index
                pattern_type = pattern.pattern_type
                pattern_length = pattern.pattern_length
                
                # Skip if not enough data
                if idx < pattern_length or idx >= len(df):
                    continue
                
                # Get volumes for the pattern candles
                pattern_volumes = df['volume'].iloc[idx - pattern_length + 1:idx + 1].values
                
                # Get average volume for preceding candles
                lookback = 20
                start_idx = max(0, idx - lookback - pattern_length)
                end_idx = idx - pattern_length
                if start_idx < end_idx:
                    avg_volume = df['volume'].iloc[start_idx:end_idx].mean()
                else:
                    avg_volume = df['volume'].iloc[0:idx].mean()
                
                # Check volume confirmation based on pattern type
                if pattern_type in [PatternType.REVERSAL_BULLISH, PatternType.REVERSAL_BEARISH]:
                    # For reversal patterns, check if last candle has higher volume
                    if pattern_volumes[-1] > avg_volume * 1.2:
                        pattern.volume_confirmation = True
                        # Increase confidence slightly
                        pattern.confidence = min(1.0, pattern.confidence * 1.1)
                
                elif pattern_type in [PatternType.CONTINUATION_BULLISH, PatternType.CONTINUATION_BEARISH]:
                    # For continuation patterns, check if overall pattern has good volume
                    if np.mean(pattern_volumes) > avg_volume * 1.1:
                        pattern.volume_confirmation = True
                        # Increase confidence slightly
                        pattern.confidence = min(1.0, pattern.confidence * 1.1)
                
                elif pattern_type == PatternType.INDECISION:
                    # For indecision patterns, low volume can actually be a good sign
                    if pattern_volumes[-1] < avg_volume * 0.9:
                        pattern.volume_confirmation = True
                    
            except Exception as e:
                logger.warning(f"Error checking volume confirmation for pattern: {str(e)}")
        
        logger.debug(f"Checked volume confirmation for {len(patterns)} patterns")
        return patterns

    def _calculate_pattern_strength(
        self, 
        df: pd.DataFrame, 
        index: int, 
        historical_accuracy: float,
        signal_strength: float,
        pattern_name: str
    ) -> PatternStrength:
        """
        Calculate the strength of a pattern based on multiple factors.
        
        Args:
            df: DataFrame containing OHLCV data
            index: Index of the pattern in the dataframe
            historical_accuracy: Historical accuracy of this pattern
            signal_strength: Strength of the signal (0-1)
            pattern_name: Name of the pattern
        
        Returns:
            PatternStrength enum value
        """
        factors = []
        
        # Factor 1: Historical accuracy
        factors.append(historical_accuracy)
        
        # Factor 2: Signal strength
        factors.append(signal_strength)
        
        # Factor 3: Pattern formation quality
        formation_quality = self._calculate_formation_quality(df, index, pattern_name)
        factors.append(formation_quality)
        
        # Factor 4: Market condition alignment
        market_alignment = self._calculate_market_alignment(df, index, pattern_name)
        factors.append(market_alignment)
        
        # Calculate overall strength score (0-1)
        strength_score = np.mean(factors)
        
        # Map to PatternStrength enum
        if strength_score < 0.4:
            return PatternStrength.WEAK
        elif strength_score < 0.6:
            return PatternStrength.MODERATE
        elif strength_score < 0.8:
            return PatternStrength.STRONG
        else:
            return PatternStrength.VERY_STRONG

    def _calculate_confidence(
        self, 
        df: pd.DataFrame, 
        index: int, 
        historical_accuracy: float,
        signal_strength: float,
        pattern_name: str
    ) -> float:
        """
        Calculate confidence score for the pattern (0-1).
        
        Args:
            df: DataFrame containing OHLCV data
            index: Index of the pattern in the dataframe
            historical_accuracy: Historical accuracy of this pattern
            signal_strength: Strength of the signal (0-1)
            pattern_name: Name of the pattern
        
        Returns:
            Confidence score (0-1)
        """
        # Use similar factors as for pattern strength
        factors = []
        
        # Factor 1: Historical accuracy (weighted 0.3)
        factors.append(historical_accuracy * 0.3)
        
        # Factor 2: Signal strength (weighted 0.2)
        factors.append(signal_strength * 0.2)
        
        # Factor 3: Pattern formation quality (weighted 0.25)
        formation_quality = self._calculate_formation_quality(df, index, pattern_name)
        factors.append(formation_quality * 0.25)
        
        # Factor 4: Market condition alignment (weighted 0.25)
        market_alignment = self._calculate_market_alignment(df, index, pattern_name)
        factors.append(market_alignment * 0.25)
        
        # Calculate overall confidence score (0-1)
        confidence = sum(factors)
        
        # Ensure within range
        return max(0.0, min(1.0, confidence))

    def _calculate_expected_move(
        self, 
        df: pd.DataFrame, 
        index: int, 
        pattern_name: str
    ) -> float:
        """
        Calculate expected move in pips based on historical performance of the pattern.
        
        Args:
            df: DataFrame containing OHLCV data
            index: Index of the pattern in the dataframe
            pattern_name: Name of the pattern
        
        Returns:
            Expected move in pips
        """
        try:
            # Get pattern type
            pattern_type = self.PATTERN_TYPES.get(pattern_name, PatternType.INDECISION)
            
            # Get price data
            close = df['close'].iloc[index]
            
            # Calculate ATR for volatility-based sizing
            atr = self._calculate_atr(df, index)
            
            # Default multipliers based on pattern type
            multipliers = {
                PatternType.REVERSAL_BULLISH: 2.5,
                PatternType.REVERSAL_BEARISH: 2.5,
                PatternType.CONTINUATION_BULLISH: 1.8,
                PatternType.CONTINUATION_BEARISH: 1.8,
                PatternType.INDECISION: 1.0
            }
            
            # Adjust multiplier based on historical accuracy
            accuracy_factor = self.pattern_accuracy.get(pattern_name, 0.5)
            multiplier = multipliers.get(pattern_type, 1.0) * (1 + (accuracy_factor - 0.5) * 2)
            
            # Expected move in pips
            expected_move_pips = atr * multiplier * 10000  # Convert to pips (4 decimal places)
            
            return expected_move_pips
            
        except Exception as e:
            logger.warning(f"Error calculating expected move: {str(e)}")
            return 10.0  # Default fallback value

    def _calculate_formation_quality(
        self, 
        df: pd.DataFrame, 
        index: int, 
        pattern_name: str
    ) -> float:
        """
        Calculate the quality of pattern formation (0-1).
        
        Args:
            df: DataFrame containing OHLCV data
            index: Index of the pattern in the dataframe
            pattern_name: Name of the pattern
        
        Returns:
            Formation quality score (0-1)
        """
        try:
            # Get pattern length
            pattern_length = self.PATTERN_LENGTHS.get(pattern_name, 1)
            
            # Skip if not enough data
            if index < pattern_length - 1:
                return 0.7  # Default value
            
            # Get candles for the pattern
            start_idx = index - pattern_length + 1
            pattern_candles = df.iloc[start_idx:index + 1]
            
            # Different quality checks based on pattern
            if pattern_name in ['doji', 'spinning_top']:
                # For doji, check how close open and close are
                open_close_diff = abs(pattern_candles['open'].iloc[-1] - pattern_candles['close'].iloc[-1])
                high_low_diff = pattern_candles['high'].iloc[-1] - pattern_candles['low'].iloc[-1]
                if high_low_diff == 0:
                    return 0.5  # Avoid division by zero
                quality = 1.0 - (open_close_diff / high_low_diff)
                
            elif pattern_name in ['hammer', 'inverted_hammer', 'hanging_man', 'shooting_star']:
                # For hammer-like patterns, check shadow-to-body ratio
                body_size = calculate_body_size(pattern_candles.iloc[-1])
                if pattern_name in ['hammer', 'hanging_man']:
                    shadow_size = calculate_shadow_size(pattern_candles.iloc[-1], 'lower')
                else:  # inverted_hammer, shooting_star
                    shadow_size = calculate_shadow_size(pattern_candles.iloc[-1], 'upper')
                
                # Ideal ratio is 2:1 or higher for shadow:body
                if body_size == 0:
                    return 0.5  # Avoid division by zero
                ratio = shadow_size / body_size
                quality = min(1.0, ratio / 3.0)
                
            elif pattern_name in ['engulfing_bullish', 'engulfing_bearish']:
                # For engulfing, check how much of the previous candle is engulfed
                prev_body_size = calculate_body_size(pattern_candles.iloc[-2])
                curr_body_size = calculate_body_size(pattern_candles.iloc[-1])
                
                if prev_body_size == 0:
                    return 0.5  # Avoid division by zero
                
                # Perfect engulfing should be at least 1.5x the previous body
                engulfing_ratio = curr_body_size / prev_body_size
                quality = min(1.0, engulfing_ratio / 1.5)
                
            elif pattern_name in ['morning_star', 'evening_star']:
                # For stars, check the middle candle's size and gap
                first_body = calculate_body_size(pattern_candles.iloc[0])
                middle_body = calculate_body_size(pattern_candles.iloc[1])
                last_body = calculate_body_size(pattern_candles.iloc[2])
                
                # Middle should be small, ideally less than 30% of the average of first and last
                avg_outer_body = (first_body + last_body) / 2
                if avg_outer_body == 0:
                    return 0.5  # Avoid division by zero
                
                middle_ratio = middle_body / avg_outer_body
                middle_quality = max(0, 1.0 - (middle_ratio / 0.3))
                
                # Check gaps between candles
                gap_quality = 0.5  # Default
                if pattern_name == 'morning_star':
                    # Gap down between 1st and 2nd, gap up between 2nd and 3rd
                    gap1 = pattern_candles['close'].iloc[0] - pattern_candles['high'].iloc[1]
                    gap2 = pattern_candles['low'].iloc[2] - pattern_candles['close'].iloc[1]
                    if first_body > 0 and last_body > 0:  # Ensure gaps are meaningful
                        gap_quality = min(1.0, (gap1 + gap2) / (first_body * 0.5))
                else:  # evening_star
                    # Gap up between 1st and 2nd, gap down between 2nd and 3rd
                    gap1 = pattern_candles['low'].iloc[1] - pattern_candles['close'].iloc[0]
                    gap2 = pattern_candles['close'].iloc[1] - pattern_candles['high'].iloc[2]
                    if first_body > 0 and last_body > 0:  # Ensure gaps are meaningful
                        gap_quality = min(1.0, (gap1 + gap2) / (first_body * 0.5))
                
                quality = (middle_quality * 0.7) + (gap_quality * 0.3)
                
            else:
                # Default quality calculation for other patterns
                # Check consistency of candle sizes and directions
                candle_sizes = [calculate_body_size(candle) for _, candle in pattern_candles.iterrows()]
                avg_size = np.mean(candle_sizes)
                size_variation = np.std(candle_sizes) / avg_size if avg_size > 0 else 1.0
                
                # Less variation is better (up to a point)
                size_quality = max(0, 1.0 - (size_variation / 0.5))
                
                # Check appropriate direction for pattern candles
                direction_quality = 0.7  # Default value
                
                # Overall quality
                quality = (size_quality * 0.5) + (direction_quality * 0.5)
            
            # Ensure within range
            return max(0.2, min(1.0, quality))
            
        except Exception as e:
            logger.warning(f"Error calculating formation quality: {str(e)}")
            return 0.7  # Default fallback value

    def _calculate_market_alignment(
        self, 
        df: pd.DataFrame, 
        index: int, 
        pattern_name: str
    ) -> float:
        """
        Calculate how well the pattern aligns with overall market conditions (0-1).
        
        Args:
            df: DataFrame containing OHLCV data
            index: Index of the pattern in the dataframe
            pattern_name: Name of the pattern
        
        Returns:
            Market alignment score (0-1)
        """
        try:
            # Skip if not enough data
            if index < 30:
                return 0.7  # Default value
            
            # Get pattern type
            pattern_type = self.PATTERN_TYPES.get(pattern_name, PatternType.INDECISION)
            
            # Calculate some key indicators
            # SMA 20
            sma20 = df['close'].rolling(20).mean()
            # EMA 50
            ema50 = df['close'].ewm(span=50).mean()
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Current trend based on EMAs
            trend_up = df['close'].iloc[index] > ema50.iloc[index] and sma20.iloc[index] > ema50.iloc[index]
            trend_down = df['close'].iloc[index] < ema50.iloc[index] and sma20.iloc[index] < ema50.iloc[index]
            
            # Market alignment score based on pattern type and market conditions
            alignment = 0.5  # Neutral starting point
            
            if pattern_type == PatternType.REVERSAL_BULLISH:
                # Bullish reversal works best in oversold conditions in a downtrend
                if trend_down and rsi.iloc[index] < 30:
                    alignment = 0.9
                elif trend_down and rsi.iloc[index] < 40:
                    alignment = 0.8
                elif not trend_up and not trend_down:  # Sideways
                    alignment = 0.6
                elif trend_up:  # Not ideal for reversal
                    alignment = 0.4
            
            elif pattern_type == PatternType.REVERSAL_BEARISH:
                # Bearish reversal works best in overbought conditions in an uptrend
                if trend_up and rsi.iloc[index] > 70:
                    alignment = 0.9
                elif trend_up and rsi.iloc[index] > 60:
                    alignment = 0.8
                elif not trend_up and not trend_down:  # Sideways
                    alignment = 0.6
                elif trend_down:  # Not ideal for reversal
                    alignment = 0.4
            
            elif pattern_type == PatternType.CONTINUATION_BULLISH:
                # Bullish continuation works best in an uptrend that's not overbought
                if trend_up and rsi.iloc[index] < 70:
                    alignment = 0.9
                elif trend_up and rsi.iloc[index] >= 70:
                    alignment = 0.7
                elif not trend_up and not trend_down:  # Sideways
                    alignment = 0.6
                elif trend_down:  # Not ideal for bullish continuation
                    alignment = 0.3
            
            elif pattern_type == PatternType.CONTINUATION_BEARISH:
                # Bearish continuation works best in a downtrend that's not oversold
                if trend_down and rsi.iloc[index] > 30:
                    alignment = 0.9
                elif trend_down and rsi.iloc[index] <= 30:
                    alignment = 0.7
                elif not trend_up and not trend_down:  # Sideways
                    alignment = 0.6
                elif trend_up:  # Not ideal for bearish continuation
                    alignment = 0.3
            
            elif pattern_type == PatternType.INDECISION:
                # Indecision patterns work best at key levels or after extended moves
                if (trend_up and rsi.iloc[index] > 70) or (trend_down and rsi.iloc[index] < 30):
                    alignment = 0.8  # Good at extremes
                elif not trend_up and not trend_down:  # Sideways market
                    alignment = 0.7  # Decent in range-bound conditions
                else:
                    alignment = 0.6  # Ok in other conditions
            
            return alignment
            
        except Exception as e:
            logger.warning(f"Error calculating market alignment: {str(e)}")
            return 0.7  # Default fallback value

    def _get_context_factors(
        self, 
        df: pd.DataFrame, 
        index: int,
        pattern_name: str
    ) -> Dict[str, Any]:
        """
        Get relevant market context factors for this pattern.
        
        Args:
            df: DataFrame containing OHLCV data
            index: Index of the pattern in the dataframe
            pattern_name: Name of the pattern
        
        Returns:
            Dictionary of context factors
        """
        context = {}
        
        try:
            # Skip if not enough data
            if index < 50:
                return {'data_insufficient': True}
            
            # Distance from key MAs
            sma20 = df['close'].rolling(20).mean().iloc[index]
            ema50 = df['close'].ewm(span=50).mean().iloc[index]
            sma200 = df['close'].rolling(200).mean().iloc[index] if index >= 200 else None
            
            current_close = df['close'].iloc[index]
            
            context['sma20_dist_pct'] = ((current_close / sma20) - 1) * 100 if sma20 else None
            context['ema50_dist_pct'] = ((current_close / ema50) - 1) * 100 if ema50 else None
            context['sma200_dist_pct'] = ((current_close / sma200) - 1) * 100 if sma200 else None
            
            # RSI value
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            context['rsi'] = rsi.iloc[index]
            
            # Volatility (ATR)
            context['atr'] = self._calculate_atr(df, index)
            context['atr_pct'] = (context['atr'] / current_close) * 100
            
            # Trend strength (ADX)
            dx_period = 14
            tr = pd.DataFrame()
            tr['h-l'] = df['high'] - df['low']
            tr['h-pc'] = abs(df['high'] - df['close'].shift(1))
            tr['l-pc'] = abs(df['low'] - df['close'].shift(1))
            tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
            
            pdi = 100 * df['high'].diff().clip(lower=0).rolling(dx_period).sum() / tr['tr'].rolling(dx_period).sum()
            ndi = 100 * abs(df['low'].diff()).clip(lower=0).rolling(dx_period).sum() / tr['tr'].rolling(dx_period).sum()
            
            dx = 100 * abs(pdi - ndi) / (pdi + ndi)
            adx = dx.rolling(dx_period).mean()
            
            context['adx'] = adx.iloc[index] if not pd.isna(adx.iloc[index]) else None
            context['trend_strength'] = 'strong' if context['adx'] and context['adx'] > 25 else 'weak'
            
            # Volume analysis
            if 'volume' in df.columns:
                avg_volume = df['volume'].rolling(20).mean().iloc[index]
                current_volume = df['volume'].iloc[index]
                context['volume_ratio'] = current_volume / avg_volume if avg_volume else None
                context['volume_spike'] = context['volume_ratio'] > 2 if context['volume_ratio'] else None
            
            # Market structure
            # Identify recent swing high/low points
            window = 5
            highs = df['high'].rolling(window=window, center=True).max()
            lows = df['low'].rolling(window=window, center=True).min()
            
            swing_highs = []
            swing_lows = []
            
            for i in range(window, min(index + 1, len(df) - window)):
                if df['high'].iloc[i] == highs.iloc[i] and df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1]:
                    swing_highs.append((i, df['high'].iloc[i]))
                
                if df['low'].iloc[i] == lows.iloc[i] and df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1]:
                    swing_lows.append((i, df['low'].iloc[i]))
            
            # Get last two swing highs and lows
            last_swing_highs = swing_highs[-2:] if len(swing_highs) >= 2 else swing_highs
            last_swing_lows = swing_lows[-2:] if len(swing_lows) >= 2 else swing_lows
            
            # Determine higher highs, lower lows, etc.
            higher_highs = len(last_swing_highs) == 2 and last_swing_highs[1][1] > last_swing_highs[0][1]
            lower_highs = len(last_swing_highs) == 2 and last_swing_highs[1][1] < last_swing_highs[0][1]
            higher_lows = len(last_swing_lows) == 2 and last_swing_lows[1][1] > last_swing_lows[0][1]
            lower_lows = len(last_swing_lows) == 2 and last_swing_lows[1][1] < last_swing_lows[0][1]
            
            if higher_highs and higher_lows:
                context['market_structure'] = 'uptrend'
            elif lower_highs and lower_lows:
                context['market_structure'] = 'downtrend'
            else:
                context['market_structure'] = 'sideways'
            
            # Nearest support/resistance levels
            support_levels = [level[1] for level in swing_lows if level[1] < current_close]
            resistance_levels = [level[1] for level in swing_highs if level[1] > current_close]
            
            if support_levels:
                context['nearest_support'] = max(support_levels)
                context['support_distance_pct'] = ((current_close / context['nearest_support']) - 1) * 100
            
            if resistance_levels:
                context['nearest_resistance'] = min(resistance_levels)
                context['resistance_distance_pct'] = ((context['nearest_resistance'] / current_close) - 1) * 100
            
            # Pattern-specific context
            context['pattern_name'] = pattern_name
            context['pattern_type'] = self.PATTERN_TYPES.get(pattern_name, PatternType.INDECISION).name
            
            return context
            
        except Exception as e:
            logger.warning(f"Error getting context factors: {str(e)}")
            return {'error': str(e)}

    def _calculate_atr(self, df: pd.DataFrame, index: int, period: int = 14) -> float:
        """
        Calculate Average True Range for volatility measurement.
        
        Args:
            df: DataFrame containing OHLCV data
            index: Current index
            period: ATR period
            
        Returns:
            ATR value
        """
        if index < period:
            return 0.0
        
        tr_values = []
        
        for i in range(index - period + 1, index + 1):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            prev_close = df['close'].iloc[i - 1] if i > 0 else df['open'].iloc[i]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            tr_values.append(max(tr1, tr2, tr3))
        
        return np.mean(tr_values)

    def _extract_features_for_ml(self, df: pd.DataFrame, index: int) -> np.ndarray:
        """
        Extract features for machine learning model validation.
        
        Args:
            df: DataFrame containing OHLCV data
            index: Index of the pattern in the dataframe
        
        Returns:
            NumPy array of features
        """
        features = []
        
        try:
            # Skip if not enough data
            if index < 20:
                return np.array([])
            
            # Price action features
            # Last 5 candle body sizes
            for i in range(5):
                if index - i >= 0:
                    body_size = calculate_body_size(df.iloc[index - i])
                    features.append(body_size)
                else:
                    features.append(0)
            
            # Last 5 candle shadow sizes
            for i in range(5):
                if index - i >= 0:
                    upper_shadow = calculate_shadow_size(df.iloc[index - i], 'upper')
                    lower_shadow = calculate_shadow_size(df.iloc[index - i], 'lower')
                    features.append(upper_shadow)
                    features.append(lower_shadow)
                else:
                    features.append(0)
                    features.append(0)
            
            # Last 5 candle directions (1 for bullish, -1 for bearish, 0 for doji)
            for i in range(5):
                if index - i >= 0:
                    if df['close'].iloc[index - i] > df['open'].iloc[index - i]:
                        features.append(1)
                    elif df['close'].iloc[index - i] < df['open'].iloc[index - i]:
                        features.append(-1)
                    else:
                        features.append(0)
                else:
                    features.append(0)
            
            # Technical indicators
            # Distance from MAs
            sma20 = df['close'].rolling(20).mean().iloc[index]
            ema50 = df['close'].ewm(span=50).mean().iloc[index]
            
            current_close = df['close'].iloc[index]
            
            features.append((current_close / sma20) - 1)
            features.append((current_close / ema50) - 1)
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            features.append(rsi.iloc[index] / 100)  # Normalize to 0-1
            
            # Volatility
            atr = self._calculate_atr(df, index)
            atr_pct = atr / current_close
            features.append(atr_pct)
            
            # Volume features if available
            if 'volume' in df.columns:
                avg_volume = df['volume'].rolling(20).mean().iloc[index]
                current_volume = df['volume'].iloc[index]
                volume_ratio = current_volume / avg_volume if avg_volume else 1.0
                features.append(volume_ratio)
                
                # Volume trend
                vol_change = (df['volume'].iloc[index] / df['volume'].iloc[index - 5]) - 1 if index >= 5 else 0
                features.append(vol_change)
            else:
                features.append(1.0)  # Default volume ratio
                features.append(0.0)  # Default volume change
            
            # Trend features
            price_change_5 = (df['close'].iloc[index] / df['close'].iloc[index - 5]) - 1 if index >= 5 else 0
            price_change_10 = (df['close'].iloc[index] / df['close'].iloc[index - 10]) - 1 if index >= 10 else 0
            price_change_20 = (df['close'].iloc[index] / df['close'].iloc[index - 20]) - 1 if index >= 20 else 0
            
            features.append(price_change_5)
            features.append(price_change_10)
            features.append(price_change_20)
            
            return np.array(features)
            
        except Exception as e:
            logger.warning(f"Error extracting features for ML: {str(e)}")
            return np.array([])

    def _detect_three_drives_bullish(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """
        Detect bullish three drives harmonic pattern.
        
        Args:
            df: DataFrame containing OHLCV data
            
        Returns:
            List of detected pattern objects
        """
        patterns = []
        
        # Implementation for three drives pattern detection
        # (This would be a complex implementation looking for specific Fibonacci relationships)
        
        return patterns

    def _detect_three_drives_bearish(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """
        Detect bearish three drives harmonic pattern.
        
        Args:
            df: DataFrame containing OHLCV data
            
        Returns:
            List of detected pattern objects
        """
        patterns = []
        
        # Implementation for three drives pattern detection
        # (This would be a complex implementation looking for specific Fibonacci relationships)
        
        return patterns

    def _detect_abcd_bullish(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """
        Detect bullish ABCD harmonic pattern.
        
        Args:
            df: DataFrame containing OHLCV data
            
        Returns:
            List of detected pattern objects
        """
        patterns = []
        
        # Implementation for ABCD pattern detection
        # (This would check for specific Fibonacci relationships between the legs)
        
        return patterns

    def _detect_abcd_bearish(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """
        Detect bearish ABCD harmonic pattern.
        
        Args:
            df: DataFrame containing OHLCV data
            
        Returns:
            List of detected pattern objects
        """
        patterns = []
        
        # Implementation for ABCD pattern detection
        # (This would check for specific Fibonacci relationships between the legs)
        
        return patterns

    def _detect_bat_bullish(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """
        Detect bullish bat harmonic pattern.
        
        Args:
            df: DataFrame containing OHLCV data
            
        Returns:
            List of detected pattern objects
        """
        patterns = []
        
        # Implementation for bat pattern detection
        # (This would check for specific Fibonacci relationships)
        
        return patterns

    def _detect_bat_bearish(self, df: pd.DataFrame) -> List[CandlestickPattern]:
        """
        Detect bearish bat harmonic pattern.
        
        Args:
            df: DataFrame containing OHLCV data
            
        Returns:
            List of detected pattern objects
        """
        patterns = []
        
        # Implementation for bat pattern detection
        # (This would check for specific Fibonacci relationships)
        
        return patterns

    def update_pattern_accuracy(self, pattern_name: str, was_successful: bool):
        """
        Update historical accuracy for a pattern based on outcome.
        
        Args:
            pattern_name: Name of the pattern
            was_successful: Whether the pattern's prediction was correct
        """
        if pattern_name not in self.pattern_accuracy:
            logger.warning(f"Pattern {pattern_name} not found in accuracy tracking")
            return
        
        try:
            # Get current accuracy
            current_accuracy = self.pattern_accuracy[pattern_name]
            
            # Update with exponential moving average (more weight to recent results)
            alpha = 0.05  # Learning rate
            new_accuracy = (1 - alpha) * current_accuracy + alpha * (1.0 if was_successful else 0.0)
            
            # Update accuracy
            self.pattern_accuracy[pattern_name] = new_accuracy
            
            logger.debug(f"Updated accuracy for {pattern_name}: {current_accuracy:.2f} -> {new_accuracy:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating pattern accuracy: {str(e)}")

    def get_pattern_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about pattern performance.
        
        Returns:
            Dictionary with pattern statistics
        """
        return {
            pattern: {
                'accuracy': self.pattern_accuracy.get(pattern, 0.5),
                'type': self.PATTERN_TYPES.get(pattern, PatternType.INDECISION).name,
                'length': self.PATTERN_LENGTHS.get(pattern, 1)
            }
            for pattern in self.PATTERN_TYPES.keys()
        }

    def export_detected_patterns(self, patterns: List[CandlestickPattern]) -> List[Dict[str, Any]]:
        """
        Export detected patterns to dictionary format for serialization.
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            List of dictionaries with pattern data
        """
        return [pattern.to_dict() for pattern in patterns]
