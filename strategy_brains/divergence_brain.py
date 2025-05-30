#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Divergence-Based Trading Strategy Brain

This module implements an advanced divergence-based trading strategy that
identifies divergences between price action and various technical indicators
to generate high-probability trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import asyncio
from datetime import datetime, timedelta

from common.utils import find_local_extrema, detect_slope, calculate_trend_strength
from feature_service.features.technical import get_indicator_data, calculate_macd, calculate_rsi
from data_storage.market_data import MarketDataRepository
from ml_models.models.classification import ClassificationModel
from strategy_brains.base_brain import StrategyBrain


class DivergenceBrain(StrategyBrain):
    """
    Advanced divergence trading strategy that identifies and exploits divergences
    between price action and technical indicators.
    
    This brain specializes in:
    1. Regular and hidden divergences
    2. Multi-indicator divergence analysis
    3. Divergence strength qualification
    4. Timeframe confluence
    5. Algorithmic divergence identification
    """
    
    STRATEGY_TYPE = "divergence"
    DEFAULT_CONFIG = {
        "lookback_periods": 60,
        "peak_distance": 5,  # Minimum distance between peaks
        "min_divergence_duration": 2,  # Minimum bars between divergent points
        "max_divergence_duration": 30,  # Maximum bars to look for divergence
        "divergence_confirmation_threshold": 3,  # Number of confirming factors needed
        "indicators": [
            {"name": "rsi", "params": {"timeperiod": 14}, "threshold": 0.15},
            {"name": "macd", "params": {"fast": 12, "slow": 26, "signal": 9}, "threshold": 0.1},
            {"name": "cci", "params": {"timeperiod": 20}, "threshold": 0.2},
            {"name": "stoch", "params": {"k": 14, "d": 3, "smooth_k": 3}, "threshold": 0.1},
            {"name": "adx", "params": {"timeperiod": 14}, "threshold": 0.1},
            {"name": "mfi", "params": {"timeperiod": 14}, "threshold": 0.15},
            {"name": "obv", "params": {}, "threshold": 0.2}
        ],
        "overbought_thresholds": {
            "rsi": 70,
            "cci": 100,
            "stoch": 80,
            "mfi": 80
        },
        "oversold_thresholds": {
            "rsi": 30,
            "cci": -100,
            "stoch": 20,
            "mfi": 20
        },
        "confidence_thresholds": {
            "multi_indicator": 0.8,  # Multiple indicators showing divergence
            "multi_timeframe": 0.9,  # Confirmed across timeframes
            "strength": 0.7,  # Strong divergence
            "volume_confirm": 0.6,  # Volume confirms
            "extrema": 0.75  # At significant extrema
        },
        "divergence_types": [
            "regular_bullish", "regular_bearish",
            "hidden_bullish", "hidden_bearish"
        ],
        "use_ml_validation": True,  # Use ML to validate divergence signals
        "ml_validation_threshold": 0.7,  # Minimum ML confidence
        "max_signal_age": 5,  # Maximum age of signal in bars
        "volume_confirmation_threshold": 1.5,  # Volume vs average
        "overbought_oversold_boost": 0.15,  # Extra confidence when in extreme zone
        "trend_context": True,  # Consider trend context
        "trend_window": 100,  # Window for trend context
        "divergence_quality_threshold": 0.6  # Minimum quality score
    }
    
    def __init__(self, 
                 config: Dict[str, Any] = None, 
                 asset_id: str = None,
                 platform: str = None):
        """
        Initialize the divergence brain with configuration.
        
        Args:
            config: Configuration dictionary
            asset_id: The primary asset ID this brain is responsible for
            platform: The trading platform (e.g., "binance", "deriv")
        """
        super().__init__(config, asset_id, platform)
        self.logger = logging.getLogger(f"divergence_brain_{asset_id}_{platform}")
        
        # Initialize repositories and models
        self.market_data_repo = MarketDataRepository()
        
        # Initialize ML model for divergence validation if enabled
        self.ml_model = ClassificationModel() if self.config["use_ml_validation"] else None
        
        # Tracking of identified divergences
        self.active_divergences = {}  # timeframe -> list of divergences
        self.historical_divergences = {}  # For performance tracking
        
        # Cache for data and indicator values
        self.data_cache = {}
        self.indicator_cache = {}
        
        # ML model for divergence validation (initialized on demand)
        self.validation_models = {}  # indicator -> model
        
        # Initialize
        asyncio.create_task(self._load_models())
    
    async def _load_models(self):
        """Loads and initializes machine learning models for divergence validation."""
        if not self.config["use_ml_validation"]:
            return
            
        self.logger.info(f"Loading divergence validation models for {self.asset_id}")
        
        # Load or create models for each indicator
        for indicator_config in self.config["indicators"]:
            indicator_name = indicator_config["name"]
            
            try:
                # Try to load existing model
                model_id = f"divergence_{indicator_name}_{self.asset_id}"
                model = await self.ml_model.load_model(model_id)
                
                if model is None:
                    # Create new model if not found
                    model = self.ml_model.create_model(
                        model_type="gradient_boosting",
                        params={
                            "n_estimators": 100,
                            "max_depth": 5,
                            "learning_rate": 0.1
                        }
                    )
                    
                    # Initial training with default parameters
                    # This will be improved with online learning
                    X_default, y_default = self._generate_default_training_data(indicator_name)
                    if X_default is not None and len(X_default) > 0:
                        model.fit(X_default, y_default)
                
                self.validation_models[indicator_name] = model
                self.logger.info(f"Loaded divergence model for {indicator_name}")
                
            except Exception as e:
                self.logger.error(f"Error loading model for {indicator_name}: {str(e)}")
    
    def _generate_default_training_data(self, indicator_name: str):
        """
        Generates default training data for a new divergence model.
        This provides a baseline until we have real trading results to learn from.
        
        Args:
            indicator_name: The indicator to generate data for
            
        Returns:
            X_train, y_train arrays for model training
        """
        # This is a simplified version - in production we would use labeled historical data
        # Generate simplified patterns for common divergence scenarios
        X = []
        y = []
        
        # Bullish regular divergence patterns
        for i in range(50):
            # Price making lower lows
            price_change = -np.random.uniform(0.01, 0.05)
            
            # Indicator making higher lows (divergence)
            indicator_change = np.random.uniform(0.01, 0.1)
            
            # Features: [price_change, indicator_change, price_slope, indicator_slope, ...]
            features = [
                price_change,
                indicator_change,
                -1,  # Price downward slope
                1,   # Indicator upward slope
                np.random.uniform(0.3, 0.8),  # Divergence strength
                np.random.uniform(5, 20),     # Bars between extrema
                np.random.uniform(0.5, 1.5)   # Volume ratio
            ]
            
            X.append(features)
            y.append(1)  # Successful divergence
            
        # Bearish regular divergence patterns
        for i in range(50):
            # Price making higher highs
            price_change = np.random.uniform(0.01, 0.05)
            
            # Indicator making lower highs (divergence)
            indicator_change = -np.random.uniform(0.01, 0.1)
            
            # Features
            features = [
                price_change,
                indicator_change,
                1,   # Price upward slope
                -1,  # Indicator downward slope
                np.random.uniform(0.3, 0.8),  # Divergence strength
                np.random.uniform(5, 20),     # Bars between extrema
                np.random.uniform(0.5, 1.5)   # Volume ratio
            ]
            
            X.append(features)
            y.append(1)  # Successful divergence
            
        # Hidden divergence patterns
        for i in range(50):
            # Price making higher lows (in uptrend)
            price_change = np.random.uniform(0.01, 0.03)
            
            # Indicator making lower lows (divergence)
            indicator_change = -np.random.uniform(0.01, 0.1)
            
            features = [
                price_change,
                indicator_change,
                1,   # Price upward slope
                -1,  # Indicator downward slope
                np.random.uniform(0.3, 0.8),  # Divergence strength
                np.random.uniform(5, 20),     # Bars between extrema
                np.random.uniform(0.5, 1.5)   # Volume ratio
            ]
            
            X.append(features)
            y.append(1)  # Successful divergence
            
        # Failed divergence patterns
        for i in range(100):
            # Random price and indicator changes with less clear divergence
            price_change = np.random.uniform(-0.03, 0.03)
            indicator_change = np.random.uniform(-0.05, 0.05)
            
            # Weaker or unclear patterns
            features = [
                price_change,
                indicator_change,
                np.random.choice([-1, 0, 1]),  # Random slope
                np.random.choice([-1, 0, 1]),  # Random slope
                np.random.uniform(0.1, 0.3),   # Weak divergence
                np.random.uniform(2, 5),       # Too close extrema
                np.random.uniform(0.2, 0.5)    # Low volume
            ]
            
            X.append(features)
            y.append(0)  # Failed divergence
            
        return np.array(X), np.array(y)
    
    async def generate_signals(self) -> List[Dict[str, Any]]:
        """
        Generates trading signals based on divergence analysis.
        
        Returns:
            List of trading signal dictionaries
        """
        signals = []
        
        # Generate divergence signals for each timeframe
        for timeframe in self.supported_timeframes:
            # Find divergences
            divergences = await self._find_divergences(timeframe)
            
            # Store active divergences
            self.active_divergences[timeframe] = divergences
            
            # Process each divergence
            for div in divergences:
                # Check if divergence is fresh (not too old)
                if div["age"] <= self.config["max_signal_age"]:
                    # Create signal
                    signal = {
                        'signal_type': 'divergence',
                        'sub_type': div["type"],
                        'direction': div["direction"],
                        'confidence': div["confidence"],
                        'timeframe': timeframe,
                        'strategy': self.STRATEGY_TYPE,
                        'asset_id': self.asset_id,
                        'platform': self.platform,
                        'timestamp': datetime.utcnow().isoformat(),
                        'metadata': {
                            'indicator': div["indicator"],
                            'divergence_strength': div["strength"],
                            'quality_score': div["quality_score"],
                            'confirming_indicators': div["confirming_indicators"],
                            'bars_since_identified': div["age"],
                            'price_extrema': div["price_extrema"],
                            'indicator_extrema': div["indicator_extrema"]
                        }
                    }
                    
                    signals.append(signal)
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        return signals
    
    async def _find_divergences(self, timeframe: str) -> List[Dict[str, Any]]:
        """
        Finds divergences between price action and indicators.
        
        Args:
            timeframe: The timeframe to analyze
            
        Returns:
            List of divergence dictionaries
        """
        # Get market data
        data = await self._get_data(timeframe)
        if data is None or data.empty:
            return []
            
        divergences = []
        
        # Process each indicator
        for indicator_config in self.config["indicators"]:
            indicator_name = indicator_config["name"]
            indicator_params = indicator_config["params"]
            divergence_threshold = indicator_config["threshold"]
            
            # Get indicator data
            indicator_data = await self._get_indicator(indicator_name, indicator_params, data, timeframe)
            if indicator_data is None:
                continue
                
            # Find regular bullish divergences (price lower lows, indicator higher lows)
            if "regular_bullish" in self.config["divergence_types"]:
                regular_bullish = self._find_regular_bullish_divergences(
                    data, indicator_data, indicator_name, divergence_threshold
                )
                divergences.extend(regular_bullish)
                
            # Find regular bearish divergences (price higher highs, indicator lower highs)
            if "regular_bearish" in self.config["divergence_types"]:
                regular_bearish = self._find_regular_bearish_divergences(
                    data, indicator_data, indicator_name, divergence_threshold
                )
                divergences.extend(regular_bearish)
                
            # Find hidden bullish divergences (price higher lows, indicator lower lows)
            if "hidden_bullish" in self.config["divergence_types"]:
                hidden_bullish = self._find_hidden_bullish_divergences(
                    data, indicator_data, indicator_name, divergence_threshold
                )
                divergences.extend(hidden_bullish)
                
            # Find hidden bearish divergences (price lower highs, indicator higher highs)
            if "hidden_bearish" in self.config["divergence_types"]:
                hidden_bearish = self._find_hidden_bearish_divergences(
                    data, indicator_data, indicator_name, divergence_threshold
                )
                divergences.extend(hidden_bearish)
        
        # Qualify divergences
        qualified_divergences = []
        for div in divergences:
            quality_score = self._calculate_divergence_quality(div, data, timeframe)
            
            if quality_score >= self.config["divergence_quality_threshold"]:
                div["quality_score"] = quality_score
                qualified_divergences.append(div)
                
        # Cross-validate with other indicators
        for div in qualified_divergences:
            div["confirming_indicators"] = self._find_confirming_indicators(div, data, timeframe)
            
            # Boost confidence if multiple indicators confirm
            if len(div["confirming_indicators"]) >= self.config["divergence_confirmation_threshold"]:
                div["confidence"] = min(0.95, div["confidence"] + 0.1)
                
        # Check for overbought/oversold conditions to boost confidence
        for div in qualified_divergences:
            div["confidence"] = self._apply_overbought_oversold_boost(div, data)
                
        # Validate with ML if enabled
        if self.config["use_ml_validation"]:
            for div in qualified_divergences:
                ml_validation = await self._validate_with_ml(div, data)
                div["ml_validation"] = ml_validation
                
                # Adjust confidence based on ML
                if ml_validation >= self.config["ml_validation_threshold"]:
                    div["confidence"] = min(0.95, div["confidence"] + 0.05)
                else:
                    div["confidence"] = max(0.3, div["confidence"] - 0.1)
        
        # Store divergences for performance tracking
        for div in qualified_divergences:
            div_key = f"{div['type']}_{div['indicator']}_{div['price_extrema'][0]}"
            if div_key not in self.historical_divergences:
                self.historical_divergences[div_key] = {
                    "divergence": div,
                    "entry_price": data['close'].iloc[-1],
                    "entry_time": datetime.utcnow(),
                    "resolved": False,
                    "profitable": None,
                    "exit_price": None,
                    "exit_time": None
                }
                
        # Sort by confidence
        qualified_divergences.sort(key=lambda x: x['confidence'], reverse=True)
        
        return qualified_divergences
    
    def _find_regular_bullish_divergences(self, data: pd.DataFrame, indicator_data: pd.Series, 
                                          indicator_name: str, threshold: float) -> List[Dict[str, Any]]:
        """
        Finds regular bullish divergences (price making lower lows, indicator making higher lows).
        
        Args:
            data: OHLCV data
            indicator_data: Indicator values
            indicator_name: Name of the indicator
            threshold: Minimum divergence threshold
            
        Returns:
            List of identified divergences
        """
        divergences = []
        
        # Find local minima in price and indicator
        price_minima = find_local_extrema(data['low'].values, window=self.config["peak_distance"], extrema_type='min')
        indicator_minima = find_local_extrema(indicator_data.values, window=self.config["peak_distance"], extrema_type='min')
        
        if len(price_minima) < 2 or len(indicator_minima) < 2:
            return divergences
        
        # Look for divergence patterns
        lookback = min(len(price_minima), len(indicator_minima), 5)  # Look at most 5 extrema back
        
        for i in range(min(lookback, len(price_minima) - 1)):
            # Get price extrema
            current_price_idx = price_minima[i]
            previous_price_idx = price_minima[i + 1]
            
            current_price = data['low'].iloc[current_price_idx]
            previous_price = data['low'].iloc[previous_price_idx]
            
            # Check if price made lower lows
            if current_price >= previous_price:
                continue
                
            # Look for corresponding indicator extrema
            for j in range(min(lookback, len(indicator_minima) - 1)):
                current_ind_idx = indicator_minima[j]
                previous_ind_idx = indicator_minima[j + 1]
                
                # Make sure the indicator extrema are aligned with price extrema (within a few bars)
                if abs(current_ind_idx - current_price_idx) > 3 or abs(previous_ind_idx - previous_price_idx) > 3:
                    continue
                
                current_ind = indicator_data.iloc[current_ind_idx]
                previous_ind = indicator_data.iloc[previous_ind_idx]
                
                # Check if indicator made higher lows (divergence)
                if current_ind <= previous_ind:
                    continue
                    
                # Calculate divergence strength
                price_change = (current_price / previous_price) - 1
                indicator_change = (current_ind / previous_ind) - 1
                
                # Need opposite directions with sufficient strength
                if price_change >= 0 or indicator_change <= 0 or abs(price_change) < threshold:
                    continue
                
                # Calculate distance between extrema
                bars_between = previous_price_idx - current_price_idx
                
                # Ensure minimum and maximum divergence duration
                if bars_between < self.config["min_divergence_duration"] or bars_between > self.config["max_divergence_duration"]:
                    continue
                
                # Calculate bars since most recent extrema (divergence age)
                bars_since = len(data) - 1 - current_price_idx
                
                # Calculate confidence based on divergence strength
                # Stronger divergence = higher confidence
                confidence = min(0.85, 0.5 + abs(indicator_change - price_change) * 2)
                
                # Create divergence object
                divergence = {
                    "type": "regular_bullish",
                    "direction": "buy",
                    "indicator": indicator_name,
                    "price_extrema": (current_price_idx, previous_price_idx),
                    "indicator_extrema": (current_ind_idx, previous_ind_idx),
                    "price_values": (current_price, previous_price),
                    "indicator_values": (current_ind, previous_ind),
                    "strength": abs(indicator_change - price_change),
                    "bars_between": bars_between,
                    "age": bars_since,
                    "confidence": confidence,
                    "confirming_indicators": []
                }
                
                divergences.append(divergence)
        
        return divergences
    
    def _find_regular_bearish_divergences(self, data: pd.DataFrame, indicator_data: pd.Series, 
                                          indicator_name: str, threshold: float) -> List[Dict[str, Any]]:
        """
        Finds regular bearish divergences (price making higher highs, indicator making lower highs).
        
        Args:
            data: OHLCV data
            indicator_data: Indicator values
            indicator_name: Name of the indicator
            threshold: Minimum divergence threshold
            
        Returns:
            List of identified divergences
        """
        divergences = []
        
        # Find local maxima in price and indicator
        price_maxima = find_local_extrema(data['high'].values, window=self.config["peak_distance"], extrema_type='max')
        indicator_maxima = find_local_extrema(indicator_data.values, window=self.config["peak_distance"], extrema_type='max')
        
        if len(price_maxima) < 2 or len(indicator_maxima) < 2:
            return divergences
        
        # Look for divergence patterns
        lookback = min(len(price_maxima), len(indicator_maxima), 5)
        
        for i in range(min(lookback, len(price_maxima) - 1)):
            # Get price extrema
            current_price_idx = price_maxima[i]
            previous_price_idx = price_maxima[i + 1]
            
            current_price = data['high'].iloc[current_price_idx]
            previous_price = data['high'].iloc[previous_price_idx]
            
            # Check if price made higher highs
            if current_price <= previous_price:
                continue
                
            # Look for corresponding indicator extrema
            for j in range(min(lookback, len(indicator_maxima) - 1)):
                current_ind_idx = indicator_maxima[j]
                previous_ind_idx = indicator_maxima[j + 1]
                
                # Make sure the indicator extrema are aligned with price extrema (within a few bars)
                if abs(current_ind_idx - current_price_idx) > 3 or abs(previous_ind_idx - previous_price_idx) > 3:
                    continue
                
                current_ind = indicator_data.iloc[current_ind_idx]
                previous_ind = indicator_data.iloc[previous_ind_idx]
                
                # Check if indicator made lower highs (divergence)
                if current_ind >= previous_ind:
                    continue
                    
                # Calculate divergence strength
                price_change = (current_price / previous_price) - 1
                indicator_change = (current_ind / previous_ind) - 1
                
                # Need opposite directions with sufficient strength
                if price_change <= 0 or indicator_change >= 0 or abs(price_change) < threshold:
                    continue
                
                # Calculate distance between extrema
                bars_between = previous_price_idx - current_price_idx
                
                # Ensure minimum and maximum divergence duration
                if bars_between < self.config["min_divergence_duration"] or bars_between > self.config["max_divergence_duration"]:
                    continue
                
                # Calculate bars since most recent extrema (divergence age)
                bars_since = len(data) - 1 - current_price_idx
                
                # Calculate confidence based on divergence strength
                confidence = min(0.85, 0.5 + abs(indicator_change - price_change) * 2)
                
                # Create divergence object
                divergence = {
                    "type": "regular_bearish",
                    "direction": "sell",
                    "indicator": indicator_name,
                    "price_extrema": (current_price_idx, previous_price_idx),
                    "indicator_extrema": (current_ind_idx, previous_ind_idx),
                    "price_values": (current_price, previous_price),
                    "indicator_values": (current_ind, previous_ind),
                    "strength": abs(indicator_change - price_change),
                    "bars_between": bars_between,
                    "age": bars_since,
                    "confidence": confidence,
                    "confirming_indicators": []
                }
                
                divergences.append(divergence)
        
        return divergences
    
    def _find_hidden_bullish_divergences(self, data: pd.DataFrame, indicator_data: pd.Series, 
                                         indicator_name: str, threshold: float) -> List[Dict[str, Any]]:
        """
        Finds hidden bullish divergences (price making higher lows, indicator making lower lows).
        
        Args:
            data: OHLCV data
            indicator_data: Indicator values
            indicator_name: Name of the indicator
            threshold: Minimum divergence threshold
            
        Returns:
            List of identified divergences
        """
        divergences = []
        
        # Check if we're in an uptrend (required for hidden bullish divergence)
        if self.config["trend_context"]:
            trend = calculate_trend_strength(data, window=self.config["trend_window"])
            if trend <= 0:  # Not in uptrend
                return divergences
        
        # Find local minima in price and indicator
        price_minima = find_local_extrema(data['low'].values, window=self.config["peak_distance"], extrema_type='min')
        indicator_minima = find_local_extrema(indicator_data.values, window=self.config["peak_distance"], extrema_type='min')
        
        if len(price_minima) < 2 or len(indicator_minima) < 2:
            return divergences
        
        # Look for divergence patterns
        lookback = min(len(price_minima), len(indicator_minima), 5)
        
        for i in range(min(lookback, len(price_minima) - 1)):
            # Get price extrema
            current_price_idx = price_minima[i]
            previous_price_idx = price_minima[i + 1]
            
            current_price = data['low'].iloc[current_price_idx]
            previous_price = data['low'].iloc[previous_price_idx]
            
            # Check if price made higher lows (in uptrend)
            if current_price <= previous_price:
                continue
                
            # Look for corresponding indicator extrema
            for j in range(min(lookback, len(indicator_minima) - 1)):
                current_ind_idx = indicator_minima[j]
                previous_ind_idx = indicator_minima[j + 1]
                
                # Make sure the indicator extrema are aligned with price extrema (within a few bars)
                if abs(current_ind_idx - current_price_idx) > 3 or abs(previous_ind_idx - previous_price_idx) > 3:
                    continue
                
                current_ind = indicator_data.iloc[current_ind_idx]
                previous_ind = indicator_data.iloc[previous_ind_idx]
                
                # Check if indicator made lower lows (divergence)
                if current_ind >= previous_ind:
                    continue
                    
                # Calculate divergence strength
                price_change = (current_price / previous_price) - 1
                indicator_change = (current_ind / previous_ind) - 1
                
                # Need opposite directions with sufficient strength
                if price_change <= 0 or indicator_change >= 0 or abs(price_change) < threshold:
                    continue
                
                # Calculate distance between extrema
                bars_between = previous_price_idx - current_price_idx
                
                # Ensure minimum and maximum divergence duration
                if bars_between < self.config["min_divergence_duration"] or bars_between > self.config["max_divergence_duration"]:
                    continue
                
                # Calculate bars since most recent extrema (divergence age)
                bars_since = len(data) - 1 - current_price_idx
                
                # Hidden divergences may be slightly less reliable than regular ones
                confidence = min(0.8, 0.45 + abs(indicator_change - price_change) * 2)
                
                # Create divergence object
                divergence = {
                    "type": "hidden_bullish",
                    "direction": "buy",
                    "indicator": indicator_name,
                    "price_extrema": (current_price_idx, previous_price_idx),
                    "indicator_extrema": (current_ind_idx, previous_ind_idx),
                    "price_values": (current_price, previous_price),
                    "indicator_values": (current_ind, previous_ind),
                    "strength": abs(indicator_change - price_change),
                    "bars_between": bars_between,
                    "age": bars_since,
                    "confidence": confidence,
                    "confirming_indicators": []
                }
                
                divergences.append(divergence)
        
        return divergences
    
    def _find_hidden_bearish_divergences(self, data: pd.DataFrame, indicator_data: pd.Series, 
                                         indicator_name: str, threshold: float) -> List[Dict[str, Any]]:
        """
        Finds hidden bearish divergences (price making lower highs, indicator making higher highs).
        
        Args:
            data: OHLCV data
            indicator_data: Indicator values
            indicator_name: Name of the indicator
            threshold: Minimum divergence threshold
            
        Returns:
            List of identified divergences
        """
        divergences = []
        
        # Check if we're in a downtrend (required for hidden bearish divergence)
        if self.config["trend_context"]:
            trend = calculate_trend_strength(data, window=self.config["trend_window"])
            if trend >= 0:  # Not in downtrend
                return divergences
        
        # Find local maxima in price and indicator
        price_maxima = find_local_extrema(data['high'].values, window=self.config["peak_distance"], extrema_type='max')
        indicator_maxima = find_local_extrema(indicator_data.values, window=self.config["peak_distance"], extrema_type='max')
        
        if len(price_maxima) < 2 or len(indicator_maxima) < 2:
            return divergences
        
        # Look for divergence patterns
        lookback = min(len(price_maxima), len(indicator_maxima), 5)
        
        for i in range(min(lookback, len(price_maxima) - 1)):
            # Get price extrema
            current_price_idx = price_maxima[i]
            previous_price_idx = price_maxima[i + 1]
            
            current_price = data['high'].iloc[current_price_idx]
            previous_price = data['high'].iloc[previous_price_idx]
            
            # Check if price made lower highs (in downtrend)
            if current_price >= previous_price:
                continue
                
            # Look for corresponding indicator extrema
            for j in range(min(lookback, len(indicator_maxima) - 1)):
                current_ind_idx = indicator_maxima[j]
                previous_ind_idx = indicator_maxima[j + 1]
                
                # Make sure the indicator extrema are aligned with price extrema (within a few bars)
                if abs(current_ind_idx - current_price_idx) > 3 or abs(previous_ind_idx - previous_price_idx) > 3:
                    continue
                
                current_ind = indicator_data.iloc[current_ind_idx]
                previous_ind = indicator_data.iloc[previous_ind_idx]
                
                # Check if indicator made higher highs (divergence)
                if current_ind <= previous_ind:
                    continue
                    
                # Calculate divergence strength
                price_change = (current_price / previous_price) - 1
                indicator_change = (current_ind / previous_ind) - 1
                
                # Need opposite directions with sufficient strength
                if price_change >= 0 or indicator_change <= 0 or abs(price_change) < threshold:
                    continue
                
                # Calculate distance between extrema
                bars_between = previous_price_idx - current_price_idx
                
                # Ensure minimum and maximum divergence duration
                if bars_between < self.config["min_divergence_duration"] or bars_between > self.config["max_divergence_duration"]:
                    continue
                
                # Calculate bars since most recent extrema (divergence age)
                bars_since = len(data) - 1 - current_price_idx
                
                # Hidden divergences may be slightly less reliable than regular ones
                confidence = min(0.8, 0.45 + abs(indicator_change - price_change) * 2)
                
                # Create divergence object
                divergence = {
                    "type": "hidden_bearish",
                    "direction": "sell",
                    "indicator": indicator_name,
                    "price_extrema": (current_price_idx, previous_price_idx),
                    "indicator_extrema": (current_ind_idx, previous_ind_idx),
                    "price_values": (current_price, previous_price),
                    "indicator_values": (current_ind, previous_ind),
                    "strength": abs(indicator_change - price_change),
                    "bars_between": bars_between,
                    "age": bars_since,
                    "confidence": confidence,
                    "confirming_indicators": []
                }
                
                divergences.append(divergence)
        
        return divergences
    
    def _calculate_divergence_quality(self, divergence: Dict[str, Any], data: pd.DataFrame, timeframe: str) -> float:
        """
        Calculates a quality score for a divergence.
        
        Args:
            divergence: The divergence to evaluate
            data: OHLCV data
            timeframe: The timeframe being analyzed
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        quality_factors = []
        
        # Factor 1: Divergence strength
        strength_score = min(1.0, divergence["strength"] * 5)
        quality_factors.append(strength_score)
        
        # Factor 2: Age of divergence (fresher is better)
        max_age = self.config["max_signal_age"]
        age_score = 1.0 - (divergence["age"] / max_age) if max_age > 0 else 0
        quality_factors.append(age_score)
        
        # Factor 3: Bars between extrema (more distance is generally better)
        optimal_bars = 15  # Arbitrary optimal distance
        bar_score = min(1.0, divergence["bars_between"] / optimal_bars) if divergence["bars_between"] < optimal_bars else \
                    min(1.0, optimal_bars / divergence["bars_between"])
        quality_factors.append(bar_score)
        
        # Factor 4: Volume confirmation
        volume_score = self._check_volume_confirmation(divergence, data)
        quality_factors.append(volume_score)
        
        # Factor 5: Price action confirmation
        price_score = self._check_price_confirmation(divergence, data)
        quality_factors.append(price_score)
        
        # Factor 6: Trend context
        if self.config["trend_context"]:
            trend_score = self._check_trend_context(divergence, data)
            quality_factors.append(trend_score)
        
        # Calculate weighted average
        weights = [0.3, 0.15, 0.1, 0.2, 0.15, 0.1]
        if not self.config["trend_context"]:
            weights = weights[:5]
            weights = [w / sum(weights) for w in weights]
            
        quality_score = sum(f * w for f, w in zip(quality_factors, weights))
        
        return quality_score
    
    def _check_volume_confirmation(self, divergence: Dict[str, Any], data: pd.DataFrame) -> float:
        """
        Checks if volume confirms the divergence.
        
        Args:
            divergence: The divergence to evaluate
            data: OHLCV data
            
        Returns:
            Volume confirmation score (0.0 to 1.0)
        """
        if 'volume' not in data.columns:
            return 0.5  # Neutral if no volume data
            
        # Get extrema indices
        current_idx = divergence["price_extrema"][0]
        
        # Check volume at extrema vs average
        recent_volume = data['volume'].iloc[current_idx]
        avg_volume = data['volume'].iloc[max(0, current_idx-10):current_idx+1].mean()
        
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Higher volume at extrema is better
        if volume_ratio >= self.config["volume_confirmation_threshold"]:
            return min(1.0, volume_ratio / 2)
        else:
            return max(0.3, volume_ratio / 2)
    
    def _check_price_confirmation(self, divergence: Dict[str, Any], data: pd.DataFrame) -> float:
        """
        Checks if price action confirms the divergence.
        
        Args:
            divergence: The divergence to evaluate
            data: OHLCV data
            
        Returns:
            Price confirmation score (0.0 to 1.0)
        """
        # Get extrema indices
        current_idx = divergence["price_extrema"][0]
        
        # Check if price has started to move in the expected direction after divergence
        if current_idx >= len(data) - 1:
            return 0.5  # No confirmation yet
            
        expected_direction = divergence["direction"]
        
        # Get price movement since extrema
        if expected_direction == "buy":
            # For bullish divergence, check if price has started moving up
            price_change = data['close'].iloc[-1] / data['low'].iloc[current_idx] - 1
            if price_change > 0:
                return min(1.0, price_change * 20 + 0.5)
            else:
                return max(0.2, 0.5 + price_change * 10)
        else:
            # For bearish divergence, check if price has started moving down
            price_change = data['close'].iloc[-1] / data['high'].iloc[current_idx] - 1
            if price_change < 0:
                return min(1.0, abs(price_change) * 20 + 0.5)
            else:
                return max(0.2, 0.5 - price_change * 10)
    
    def _check_trend_context(self, divergence: Dict[str, Any], data: pd.DataFrame) -> float:
        """
        Checks if the overall trend supports the divergence.
        
        Args:
            divergence: The divergence to evaluate
            data: OHLCV data
            
        Returns:
            Trend context score (0.0 to 1.0)
        """
        divergence_type = divergence["type"]
        
        # Calculate trend strength
        trend = calculate_trend_strength(data, window=self.config["trend_window"])
        
        # Different divergence types work better in different trends
        if divergence_type == "regular_bullish":
            # Regular bullish works better in downtrends or at bottoms
            if trend < -0.3:
                return 0.8  # Strong downtrend
            elif trend < 0:
                return 0.7  # Mild downtrend
            else:
                return 0.4  # Not ideal context
                
        elif divergence_type == "regular_bearish":
            # Regular bearish works better in uptrends or at tops
            if trend > 0.3:
                return 0.8  # Strong uptrend
            elif trend > 0:
                return 0.7  # Mild uptrend
            else:
                return 0.4  # Not ideal context
                
        elif divergence_type == "hidden_bullish":
            # Hidden bullish works better in established uptrends
            if trend > 0.3:
                return 0.9  # Strong uptrend
            elif trend > 0:
                return 0.7  # Mild uptrend
            else:
                return 0.3  # Not ideal context
                
        elif divergence_type == "hidden_bearish":
            # Hidden bearish works better in established downtrends
            if trend < -0.3:
                return 0.9  # Strong downtrend
            elif trend < 0:
                return 0.7  # Mild downtrend
            else:
                return 0.3  # Not ideal context
        
        return 0.5  # Default
    
    def _find_confirming_indicators(self, divergence: Dict[str, Any], data: pd.DataFrame, timeframe: str) -> List[str]:
        """
        Finds other indicators that confirm the divergence.
        
        Args:
            divergence: The divergence to check for confirmation
            data: OHLCV data
            timeframe: The timeframe being analyzed
            
        Returns:
            List of confirming indicator names
        """
        confirming_indicators = []
        primary_indicator = divergence["indicator"]
        
        for indicator_config in self.config["indicators"]:
            indicator_name = indicator_config["name"]
            
            # Skip the primary indicator
            if indicator_name == primary_indicator:
                continue
                
            indicator_params = indicator_config["params"]
            
            # Get indicator data
            indicator_data = self._get_cached_indicator(indicator_name, indicator_params, timeframe)
            if indicator_data is None:
                continue
                
            # Check if this indicator also shows divergence at similar locations
            if self._check_indicator_confirms(divergence, indicator_data, indicator_name):
                confirming_indicators.append(indicator_name)
        
        return confirming_indicators
    
    def _check_indicator_confirms(self, divergence: Dict[str, Any], indicator_data: pd.Series, indicator_name: str) -> bool:
        """
        Checks if another indicator confirms the primary divergence.
        
        Args:
            divergence: The primary divergence
            indicator_data: Data for the confirming indicator
            indicator_name: Name of the confirming indicator
            
        Returns:
            True if indicator confirms, False otherwise
        """
        divergence_type = divergence["type"]
        price_extrema = divergence["price_extrema"]
        
        # Get indices
        current_price_idx, previous_price_idx = price_extrema
        
        # Check for similar pattern in this indicator
        if divergence_type == "regular_bullish":
            # Need higher lows in indicator
            if current_price_idx >= len(indicator_data) or previous_price_idx >= len(indicator_data):
                return False
                
            current_ind = indicator_data.iloc[current_price_idx]
            previous_ind = indicator_data.iloc[previous_price_idx]
            
            return current_ind > previous_ind
            
        elif divergence_type == "regular_bearish":
            # Need lower highs in indicator
            if current_price_idx >= len(indicator_data) or previous_price_idx >= len(indicator_data):
                return False
                
            current_ind = indicator_data.iloc[current_price_idx]
            previous_ind = indicator_data.iloc[previous_price_idx]
            
            return current_ind < previous_ind
            
        elif divergence_type == "hidden_bullish":
            # Need lower lows in indicator
            if current_price_idx >= len(indicator_data) or previous_price_idx >= len(indicator_data):
                return False
                
            current_ind = indicator_data.iloc[current_price_idx]
            previous_ind = indicator_data.iloc[previous_price_idx]
            
            return current_ind < previous_ind
            
        elif divergence_type == "hidden_bearish":
            # Need higher highs in indicator
            if current_price_idx >= len(indicator_data) or previous_price_idx >= len(indicator_data):
                return False
                
            current_ind = indicator_data.iloc[current_price_idx]
            previous_ind = indicator_data.iloc[previous_price_idx]
            
            return current_ind > previous_ind
        
        return False
    
    def _apply_overbought_oversold_boost(self, divergence: Dict[str, Any], data: pd.DataFrame) -> float:
        """
        Applies a confidence boost if the divergence occurs in overbought/oversold conditions.
        
        Args:
            divergence: The divergence to evaluate
            data: OHLCV data
            
        Returns:
            Updated confidence value
        """
        indicator_name = divergence["indicator"]
        current_value = divergence["indicator_values"][0]
        direction = divergence["direction"]
        confidence = divergence["confidence"]
        
        # Check if the indicator has overbought/oversold thresholds
        if indicator_name not in self.config["overbought_thresholds"] or indicator_name not in self.config["oversold_thresholds"]:
            return confidence
            
        overbought = self.config["overbought_thresholds"][indicator_name]
        oversold = self.config["oversold_thresholds"][indicator_name]
        
        # Boost confidence for divergences in extreme zones
        if direction == "buy" and current_value <= oversold:
            # Bullish divergence in oversold zone
            confidence = min(0.95, confidence + self.config["overbought_oversold_boost"])
        elif direction == "sell" and current_value >= overbought:
            # Bearish divergence in overbought zone
            confidence = min(0.95, confidence + self.config["overbought_oversold_boost"])
            
        return confidence
    
    async def _validate_with_ml(self, divergence: Dict[str, Any], data: pd.DataFrame) -> float:
        """
        Validates a divergence using machine learning models.
        
        Args:
            divergence: The divergence to validate
            data: OHLCV data
            
        Returns:
            ML validation score (0.0 to 1.0)
        """
        if not self.config["use_ml_validation"]:
            return 0.5  # Neutral
            
        indicator_name = divergence["indicator"]
        
        # Check if we have a model for this indicator
        if indicator_name not in self.validation_models:
            return 0.5  # Neutral if no model
            
        model = self.validation_models[indicator_name]
        
        # Prepare features for prediction
        features = self._prepare_divergence_features(divergence, data)
        
        try:
            # Get probability of successful divergence
            prediction = model.predict_proba([features])[0][1]
            return prediction
        except Exception as e:
            self.logger.error(f"Error validating divergence with ML: {str(e)}")
            return 0.5  # Neutral on error
    
    def _prepare_divergence_features(self, divergence: Dict[str, Any], data: pd.DataFrame) -> List[float]:
        """
        Prepares features for ML validation of a divergence.
        
        Args:
            divergence: The divergence to prepare features for
            data: OHLCV data
            
        Returns:
            Feature vector for ML prediction
        """
        # Extract divergence information
        current_price_idx, previous_price_idx = divergence["price_extrema"]
        price_values = divergence["price_values"]
        indicator_values = divergence["indicator_values"]
        
        # Price change
        price_change = (price_values[0] / price_values[1]) - 1
        
        # Indicator change
        indicator_change = (indicator_values[0] / indicator_values[1]) - 1
        
        # Price slope
        price_slope = detect_slope(data['close'].iloc[current_price_idx-5:current_price_idx+5].values)
        
        # Indicator slope
        indicator_data = self._get_cached_indicator(
            divergence["indicator"], 
            self._get_indicator_params(divergence["indicator"]),
            self.default_timeframe
        )
        
        if indicator_data is not None and current_price_idx+5 < len(indicator_data):
            indicator_slope = detect_slope(indicator_data.iloc[current_price_idx-5:current_price_idx+5].values)
        else:
            indicator_slope = 0
        
        # Divergence strength
        strength = divergence["strength"]
        
        # Bars between
        bars_between = divergence["bars_between"]
        
        # Volume ratio
        if 'volume' in data.columns:
            recent_volume = data['volume'].iloc[current_price_idx]
            avg_volume = data['volume'].iloc[max(0, current_price_idx-10):current_price_idx+1].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
        else:
            volume_ratio = 1.0
        
        # Combine features
        features = [
            price_change,
            indicator_change,
            price_slope,
            indicator_slope,
            strength,
            bars_between,
            volume_ratio
        ]
        
        return features
    
    async def _get_data(self, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Gets market data for the asset.
        
        Args:
            timeframe: Timeframe to get data for
            
        Returns:
            DataFrame with OHLCV data or None if not available
        """
        # Check cache first
        cache_key = f"data_{timeframe}"
        if cache_key in self.data_cache:
            cache_time, data = self.data_cache[cache_key]
            if (datetime.utcnow() - cache_time).seconds < 300:  # Cache for 5 minutes
                return data
        
        try:
            data = await self.market_data_repo.get_ohlcv_data(
                asset_id=self.asset_id,
                platform=self.platform,
                timeframe=timeframe,
                limit=self.config["lookback_periods"]
            )
            
            if not data.empty:
                self.data_cache[cache_key] = (datetime.utcnow(), data)
                return data
            else:
                self.logger.warning(f"No data available for {self.asset_id} on {timeframe}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            return None
    
    async def _get_indicator(self, indicator_name: str, params: Dict[str, Any], data: pd.DataFrame, timeframe: str) -> Optional[pd.Series]:
        """
        Gets indicator data.
        
        Args:
            indicator_name: Name of the indicator
            params: Parameters for the indicator
            data: OHLCV data
            timeframe: Timeframe for caching
            
        Returns:
            Series with indicator values or None if error
        """
        # Check cache first
        cache_key = f"{indicator_name}_{timeframe}_{str(params)}"
        if cache_key in self.indicator_cache:
            cache_time, indicator_data = self.indicator_cache[cache_key]
            if (datetime.utcnow() - cache_time).seconds < 300:  # Cache for 5 minutes
                return indicator_data
        
        try:
            indicator_data = get_indicator_data(data, indicator_name, params)
            
            if indicator_data is not None:
                self.indicator_cache[cache_key] = (datetime.utcnow(), indicator_data)
                return indicator_data
            else:
                self.logger.warning(f"Failed to calculate {indicator_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error calculating {indicator_name}: {str(e)}")
            return None
    
    def _get_cached_indicator(self, indicator_name: str, params: Dict[str, Any], timeframe: str) -> Optional[pd.Series]:
        """
        Gets indicator data from cache only (no calculation).
        
        Args:
            indicator_name: Name of the indicator
            params: Parameters for the indicator
            timeframe: Timeframe to look up
            
        Returns:
            Series with indicator values or None if not in cache
        """
        cache_key = f"{indicator_name}_{timeframe}_{str(params)}"
        if cache_key in self.indicator_cache:
            cache_time, indicator_data = self.indicator_cache[cache_key]
            if (datetime.utcnow() - cache_time).seconds < 300:  # Cache for 5 minutes
                return indicator_data
        
        return None
    
    def _get_indicator_params(self, indicator_name: str) -> Dict[str, Any]:
        """
        Gets parameters for an indicator from config.
        
        Args:
            indicator_name: Name of the indicator
            
        Returns:
            Parameter dictionary
        """
        for indicator_config in self.config["indicators"]:
            if indicator_config["name"] == indicator_name:
                return indicator_config["params"]
        
        return {}
    
    async def update_historical_performance(self, price_data: pd.DataFrame):
        """
        Updates the performance tracking of historical divergences.
        
        Args:
            price_data: Recent price data to evaluate divergence outcomes
        """
        current_price = price_data['close'].iloc[-1]
        
        # Check each unresolved divergence
        resolved_keys = []
        
        for div_key, record in self.historical_divergences.items():
            if record["resolved"]:
                continue
                
            divergence = record["divergence"]
            entry_price = record["entry_price"]
            entry_time = record["entry_time"]
            
            # Check if enough time has passed to evaluate
            days_since_entry = (datetime.utcnow() - entry_time).days
            if days_since_entry < 1:
                continue
                
            # Calculate price change
            price_change = (current_price / entry_price) - 1
            
            # Determine if profitable based on direction
            if divergence["direction"] == "buy":
                profitable = price_change > 0
            else:  # sell
                profitable = price_change < 0
                
            # Record outcome
            record["resolved"] = True
            record["profitable"] = profitable
            record["exit_price"] = current_price
            record["exit_time"] = datetime.utcnow()
            
            # For training data collection
            resolved_keys.append(div_key)
            
            # Log result
            result_str = "profitable" if profitable else "unprofitable"
            self.logger.info(
                f"Divergence {div_key} resolved: {result_str} with {abs(price_change)*100:.2f}% change"
            )
        
        # Use resolved divergences to improve ML model
        if self.config["use_ml_validation"] and resolved_keys:
            await self._update_ml_models(resolved_keys)
    
    async def _update_ml_models(self, resolved_keys: List[str]):
        """
        Updates ML models with new training data from resolved divergences.
        
        Args:
            resolved_keys: Keys of resolved divergences to use for training
        """
        # Group training data by indicator
        training_data = {}
        
        for div_key in resolved_keys:
            if div_key not in self.historical_divergences:
                continue
                
            record = self.historical_divergences[div_key]
            divergence = record["divergence"]
            indicator_name = divergence["indicator"]
            
            if indicator_name not in training_data:
                training_data[indicator_name] = {'X': [], 'y': []}
                
            # Get data at time of divergence
            timeframe = self.default_timeframe
            data_key = f"data_{timeframe}"
            
            if data_key in self.data_cache:
                data = self.data_cache[data_key][1]
                
                # Prepare features
                features = self._prepare_divergence_features(divergence, data)
                
                # Add to training data
                training_data[indicator_name]['X'].append(features)
                training_data[indicator_name]['y'].append(1 if record["profitable"] else 0)
        
        # Update each model with new data
        for indicator_name, data in training_data.items():
            if len(data['X']) < 5:  # Need reasonable amount of new data
                continue
                
            if indicator_name not in self.validation_models:
                continue
                
            model = self.validation_models[indicator_name]
            
            try:
                # Convert to numpy arrays
                X = np.array(data['X'])
                y = np.array(data['y'])
                
                # Update model (partial_fit for online learning)
                model.partial_fit(X, y)
                
                # Save updated model
                model_id = f"divergence_{indicator_name}_{self.asset_id}"
                await self.ml_model.save_model(model, model_id)
                
                self.logger.info(f"Updated ML model for {indicator_name} with {len(X)} new samples")
                
            except Exception as e:
                self.logger.error(f"Error updating ML model for {indicator_name}: {str(e)}")
    
    async def update_state(self, state_data: Dict[str, Any]):
        """
        Updates the brain's state with new information from other system components.
        
        Args:
            state_data: Dictionary containing state updates
        """
        # Check for configuration updates
        if 'config' in state_data:
            new_config = state_data['config']
            self.config.update(new_config)
            self.logger.info(f"Updated configuration with {len(new_config)} parameters")
        
        # Update performance if price data provided
        if 'price_data' in state_data:
            await self.update_historical_performance(state_data['price_data'])
    
    def get_status(self) -> Dict[str, Any]:
        """
        Gets the current status of the divergence brain.
        
        Returns:
            Dictionary with status information
        """
        # Count active divergences
        active_count = sum(len(divs) for divs in self.active_divergences.values()) if self.active_divergences else 0
        
        # Calculate success rate
        success_count = 0
        total_resolved = 0
        
        for record in self.historical_divergences.values():
            if record["resolved"]:
                total_resolved += 1
                if record["profitable"]:
                    success_count += 1
        
        success_rate = success_count / total_resolved if total_resolved > 0 else 0
        
        return {
            'brain_type': self.STRATEGY_TYPE,
            'asset_id': self.asset_id,
            'platform': self.platform,
            'active_divergences': active_count,
            'historical_divergences': len(self.historical_divergences),
            'resolved_divergences': total_resolved,
            'success_rate': success_rate,
            'ml_models_count': len(self.validation_models) if self.config["use_ml_validation"] else 0
        }
    
    def reset_state(self):
        """Resets the brain state for a fresh start."""
        # Clear caches
        self.data_cache = {}
        self.indicator_cache = {}
        
        # Reset divergence tracking
        self.active_divergences = {}
        
        # Keep historical data for learning
        
        self.logger.info(f"Reset state for {self.asset_id} divergence brain")
