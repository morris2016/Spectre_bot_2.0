#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Regime Council Module

This module implements the Regime Council, which specializes in detecting and adapting
to different market regimes (trending, ranging, volatile, etc.) and coordinating
strategy brains appropriate for the current market conditions.
"""

import os
import time
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
import logging

from common.constants import TIMEFRAMES, MARKET_REGIMES
from common.logger import get_logger
from common.utils import generate_id
from common.async_utils import gather_with_concurrency
from common.exceptions import RegimeCouncilError

from data_storage.models.market_data import MarketRegimeData
from feature_service.features.volatility import VolatilityFeatures
from feature_service.features.market_structure import MarketStructureFeatures

from brain_council.base_council import BaseCouncil
from brain_council.signal_generator import TradeSignal, SignalStrength, SignalConfidence


class RegimeCouncil(BaseCouncil):
    """
    Regime Council specializes in detecting and adapting to different market regimes.
    
    This council tracks market regimes (trending, ranging, volatile, etc.) and coordinates
    strategy brains appropriate for the current market conditions, ensuring optimal
    strategy selection based on prevailing market characteristics.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Regime Council.
        
        Args:
            config: Configuration parameters for this council
        """
        super().__init__("regime_council", config)
        self.logger = get_logger("regime_council")
        
        # Current regimes by asset and timeframe
        self.current_regimes = {}
        
        # Regime transition metrics
        self.regime_transitions = {}
        
        # Regime-specific strategy effectiveness
        self.strategy_regime_effectiveness = {}
        
        # Regime detection models
        self.regime_models = {}
        
        # Recent feature data for regime detection
        self.feature_data = {}
        
        # Regime change alerts
        self.regime_alerts = []
        
        # Initialize volatility features calculator
        self.volatility_features = VolatilityFeatures()
        
        # Initialize market structure features calculator
        self.market_structure_features = MarketStructureFeatures()
        
        # Council operation flags
        self.initialized = False
        self.running = False
        
        # Initialize the council
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the Regime Council."""
        try:
            self.logger.info("Initializing Regime Council")
            
            # Define recognized market regimes
            # Note: These could be loaded from config
            self.recognized_regimes = MARKET_REGIMES
            
            # Initialize regime models for each timeframe
            self._initialize_regime_models()
            
            # Initialize strategy effectiveness data
            self._initialize_strategy_effectiveness()
            
            self.initialized = True
            self.logger.info("Regime Council initialized")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize Regime Council: {str(e)}")
            raise RegimeCouncilError(f"Initialization error: {str(e)}")
    
    def _initialize_regime_models(self) -> None:
        """Initialize regime detection models for each timeframe."""
        try:
            # In a production system, this would load trained models
            # for regime classification
            
            # For each supported timeframe, initialize a model
            for timeframe in TIMEFRAMES:
                self.regime_models[timeframe] = {
                    "model": self._create_regime_model(timeframe),
                    "last_update": None,
                    "confidence": 0.0
                }
            
            self.logger.info(f"Initialized regime models for {len(TIMEFRAMES)} timeframes")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize regime models: {str(e)}")
            raise RegimeCouncilError(f"Regime model initialization error: {str(e)}")
    
    def _create_regime_model(self, timeframe: str) -> Any:
        """
        Create a regime detection model for the given timeframe.
        
        Args:
            timeframe: Market timeframe for this model
            
        Returns:
            A regime detection model
        """
        # In a production system, this would create or load an ML model
        # For now, we'll create a simplified rule-based model
        
        model = {
            "timeframe": timeframe,
            "feature_weights": {
                "volatility": 0.3,
                "trend_strength": 0.3,
                "support_resistance": 0.2,
                "volume_profile": 0.2
            },
            "regime_thresholds": {
                "trending_up": {
                    "trend_strength": 0.7,
                    "volatility": 0.5,
                    "directional_movement": 0.7
                },
                "trending_down": {
                    "trend_strength": 0.7,
                    "volatility": 0.5,
                    "directional_movement": -0.7
                },
                "ranging": {
                    "range_adherence": 0.7,
                    "mean_reversion": 0.7,
                    "volatility": 0.4
                },
                "volatile": {
                    "volatility": 0.8,
                    "volatility_expansion": 0.7
                },
                "breakout": {
                    "range_breakout": 0.8,
                    "volume_surge": 0.7,
                    "volatility_expansion": 0.6
                }
            }
        }
        
        return model
    
    def _initialize_strategy_effectiveness(self) -> None:
        """Initialize data on strategy effectiveness by regime."""
        try:
            # In a production system, this would load data on how different
            # strategies perform in different regimes, based on historical analysis
            
            # For now, we'll use predefined effectiveness values
            self.strategy_regime_effectiveness = {
                "momentum": {
                    "trending_up": 0.90,
                    "trending_down": 0.85,
                    "ranging": 0.30,
                    "volatile": 0.50,
                    "breakout": 0.75
                },
                "mean_reversion": {
                    "trending_up": 0.35,
                    "trending_down": 0.35,
                    "ranging": 0.90,
                    "volatile": 0.60,
                    "breakout": 0.30
                },
                "breakout": {
                    "trending_up": 0.40,
                    "trending_down": 0.40,
                    "ranging": 0.50,
                    "volatile": 0.70,
                    "breakout": 0.95
                },
                "pattern": {
                    "trending_up": 0.75,
                    "trending_down": 0.75,
                    "ranging": 0.80,
                    "volatile": 0.50,
                    "breakout": 0.85
                },
                "volatility": {
                    "trending_up": 0.60,
                    "trending_down": 0.60,
                    "ranging": 0.40,
                    "volatile": 0.90,
                    "breakout": 0.75
                },
                "trend": {
                    "trending_up": 0.95,
                    "trending_down": 0.95,
                    "ranging": 0.20,
                    "volatile": 0.50,
                    "breakout": 0.70
                },
                "order_flow": {
                    "trending_up": 0.80,
                    "trending_down": 0.80,
                    "ranging": 0.75,
                    "volatile": 0.85,
                    "breakout": 0.90
                },
                "sentiment": {
                    "trending_up": 0.70,
                    "trending_down": 0.70,
                    "ranging": 0.50,
                    "volatile": 0.65,
                    "breakout": 0.80
                },
                "market_structure": {
                    "trending_up": 0.85,
                    "trending_down": 0.85,
                    "ranging": 0.75,
                    "volatile": 0.60,
                    "breakout": 0.90
                },
                "ml": {
                    "trending_up": 0.80,
                    "trending_down": 0.80,
                    "ranging": 0.75,
                    "volatile": 0.80,
                    "breakout": 0.80
                },
                "reinforcement": {
                    "trending_up": 0.85,
                    "trending_down": 0.85,
                    "ranging": 0.85,
                    "volatile": 0.85,
                    "breakout": 0.85
                },
                "ensemble": {
                    "trending_up": 0.90,
                    "trending_down": 0.90,
                    "ranging": 0.90,
                    "volatile": 0.90,
                    "breakout": 0.90
                }
            }
            
            self.logger.info("Initialized strategy effectiveness data")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize strategy effectiveness: {str(e)}")
            raise RegimeCouncilError(f"Strategy effectiveness initialization error: {str(e)}")
    
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process new market data through the regime council.
        
        Args:
            data: Market data including features needed for regime detection
            
        Returns:
            Dictionary with current regime information and strategy recommendations
        """
        if not self.initialized:
            self.logger.warning("Regime Council not initialized, initializing now")
            self._initialize()
        
        try:
            # Extract key information
            asset_id = data.get('asset_id')
            platform = data.get('platform')
            timeframe = data.get('timeframe')
            timestamp = data.get('timestamp', datetime.now())
            
            if not all([asset_id, platform, timeframe]):
                raise ValueError("Missing required data fields: asset_id, platform, or timeframe")
            
            # Store feature data for regime detection
            self._update_feature_data(asset_id, platform, timeframe, data)
            
            # Detect current regime
            regime_data = await self._detect_regime(asset_id, platform, timeframe)
            
            # Check for regime changes
            await self._check_regime_change(asset_id, platform, timeframe, regime_data)
            
            # Get strategy recommendations for this regime
            strategy_weights = self._get_strategy_weights(regime_data['regime'])
            
            # Generate response
            result = {
                'asset_id': asset_id,
                'platform': platform,
                'timeframe': timeframe,
                'timestamp': timestamp,
                'regime': regime_data['regime'],
                'regime_data': regime_data,
                'strategy_weights': strategy_weights,
                'regime_alerts': self._get_relevant_alerts(asset_id, platform, timeframe)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise RegimeCouncilError(f"Data processing error: {str(e)}")
    
    def _update_feature_data(self, 
                            asset_id: str, 
                            platform: str, 
                            timeframe: str, 
                            data: Dict[str, Any]) -> None:
        """
        Update stored feature data for regime detection.
        
        Args:
            asset_id: Asset identifier
            platform: Trading platform
            timeframe: Market timeframe
            data: New market data
        """
        # Create asset key
        asset_key = f"{platform}:{asset_id}"
        
        # Initialize if needed
        if asset_key not in self.feature_data:
            self.feature_data[asset_key] = {}
        
        if timeframe not in self.feature_data[asset_key]:
            self.feature_data[asset_key][timeframe] = []
        
        # Extract features relevant for regime detection
        regime_features = self._extract_regime_features(data)
        
        # Add timestamp
        regime_features['timestamp'] = data.get('timestamp', datetime.now())
        
        # Add to feature data history
        self.feature_data[asset_key][timeframe].append(regime_features)
        
        # Limit history size
        max_history = 100  # Store last 100 data points
        if len(self.feature_data[asset_key][timeframe]) > max_history:
            self.feature_data[asset_key][timeframe] = self.feature_data[asset_key][timeframe][-max_history:]
    
    def _extract_regime_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract features relevant for regime detection.
        
        Args:
            data: Market data
            
        Returns:
            Dictionary of features for regime detection
        """
        features = {}
        
        # Extract or calculate basic features
        if 'close' in data and 'open' in data:
            features['price_change'] = data['close'] - data['open']
            features['price_change_pct'] = (data['close'] / data['open']) - 1 if data['open'] > 0 else 0
        
        # Extract existing features if available
        feature_fields = [
            'volatility', 'trend_strength', 'range_adherence', 'directional_movement',
            'mean_reversion', 'volatility_expansion', 'range_breakout', 'volume_surge',
            'support_test', 'resistance_test', 'support_break', 'resistance_break'
        ]
        
        for field in feature_fields:
            if field in data:
                features[field] = data[field]
        
        # Generate additional volatility features if needed
        if 'volatility' not in features and 'ohlc_data' in data:
            try:
                vol_features = self.volatility_features.calculate(data['ohlc_data'])
                features.update(vol_features)
            except Exception as e:
                self.logger.warning(f"Error calculating volatility features: {str(e)}")
        
        # Generate additional market structure features if needed
        if ('support_test' not in features or 'resistance_test' not in features) and 'ohlc_data' in data:
            try:
                structure_features = self.market_structure_features.calculate(data['ohlc_data'])
                features.update(structure_features)
            except Exception as e:
                self.logger.warning(f"Error calculating market structure features: {str(e)}")
        
        return features
    
    async def _detect_regime(self, 
                           asset_id: str, 
                           platform: str, 
                           timeframe: str) -> Dict[str, Any]:
        """
        Detect the current market regime.
        
        Args:
            asset_id: Asset identifier
            platform: Trading platform
            timeframe: Market timeframe
            
        Returns:
            Dictionary with regime information
        """
        asset_key = f"{platform}:{asset_id}"
        
        # Ensure we have feature data
        if asset_key not in self.feature_data or timeframe not in self.feature_data[asset_key]:
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'sub_regimes': {},
                'transition_probability': {},
                'regime_duration': 0,
                'detected_at': datetime.now()
            }
        
        # Get recent feature data
        features = self.feature_data[asset_key][timeframe]
        
        # In a production system, this would use ML models for regime classification
        # For now, we'll use a rule-based approach
        regime_scores = self._calculate_regime_scores(features, timeframe)
        
        # Determine the highest scoring regime
        if regime_scores:
            top_regime = max(regime_scores.items(), key=lambda x: x[1]['score'])
            regime_name = top_regime[0]
            regime_data = top_regime[1]
            confidence = regime_data['score']
            
            # Calculate sub-regime probabilities (normalized)
            total_score = sum(r['score'] for r in regime_scores.values())
            sub_regimes = {name: {'probability': r['score'] / total_score if total_score > 0 else 0,
                                 'factors': r['factors']}
                         for name, r in regime_scores.items()}
            
            # Store current regime
            current_key = f"{asset_key}:{timeframe}"
            prev_regime = self.current_regimes.get(current_key, {}).get('regime')
            
            # Calculate regime duration
            regime_duration = 1  # Default to 1 time period
            if prev_regime == regime_name and current_key in self.current_regimes:
                regime_duration = self.current_regimes[current_key].get('regime_duration', 0) + 1
            
            # Calculate transition probabilities
            transition_probability = self._calculate_transition_probabilities(
                regime_name, asset_id, platform, timeframe)
            
            # Update current regime data
            self.current_regimes[current_key] = {
                'regime': regime_name,
                'confidence': confidence,
                'sub_regimes': sub_regimes,
                'regime_duration': regime_duration,
                'detected_at': datetime.now(),
                'transition_probability': transition_probability
            }
            
            # Store regime data for historical analysis
            await self._store_regime_data(asset_id, platform, timeframe, regime_name, confidence)
            
            return self.current_regimes[current_key]
        
        else:
            # No clear regime detected
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'sub_regimes': {},
                'transition_probability': {},
                'regime_duration': 0,
                'detected_at': datetime.now()
            }
    
    def _calculate_regime_scores(self, 
                               features: List[Dict[str, Any]], 
                               timeframe: str) -> Dict[str, Dict[str, Any]]:
        """
        Calculate scores for each possible market regime.
        
        Args:
            features: List of feature dictionaries for regime detection
            timeframe: Market timeframe
            
        Returns:
            Dictionary of regime scores with contributing factors
        """
        if not features:
            return {}
        
        # Get the most recent feature data
        current_features = features[-1]
        
        # Get model for this timeframe
        model = self.regime_models.get(timeframe, {}).get('model')
        if not model:
            return {}
        
        # Calculate scores for each regime
        regime_scores = {}
        
        # Check trending up regime
        trending_up_score, trending_up_factors = self._check_trending_up_regime(current_features, features)
        regime_scores['trending_up'] = {
            'score': trending_up_score,
            'factors': trending_up_factors
        }
        
        # Check trending down regime
        trending_down_score, trending_down_factors = self._check_trending_down_regime(current_features, features)
        regime_scores['trending_down'] = {
            'score': trending_down_score,
            'factors': trending_down_factors
        }
        
        # Check ranging regime
        ranging_score, ranging_factors = self._check_ranging_regime(current_features, features)
        regime_scores['ranging'] = {
            'score': ranging_score,
            'factors': ranging_factors
        }
        
        # Check volatile regime
        volatile_score, volatile_factors = self._check_volatile_regime(current_features, features)
        regime_scores['volatile'] = {
            'score': volatile_score,
            'factors': volatile_factors
        }
        
        # Check breakout regime
        breakout_score, breakout_factors = self._check_breakout_regime(current_features, features)
        regime_scores['breakout'] = {
            'score': breakout_score,
            'factors': breakout_factors
        }
        
        return regime_scores
    
    def _check_trending_up_regime(self, 
                                 current_features: Dict[str, Any],
                                 feature_history: List[Dict[str, Any]]
                                 ) -> Tuple[float, List[str]]:
        """
        Check for trending up regime characteristics.
        
        Args:
            current_features: Most recent feature data
            feature_history: Historical feature data
            
        Returns:
            Tuple of (score, list of contributing factors)
        """
        score = 0.0
        factors = []
        
        # Check trend strength
        trend_strength = current_features.get('trend_strength', 0)
        if trend_strength > 0.7:
            score += 0.3
            factors.append(f"Strong positive trend strength: {trend_strength:.2f}")
        elif trend_strength > 0.5:
            score += 0.2
            factors.append(f"Moderate positive trend strength: {trend_strength:.2f}")
        
        # Check directional movement
        directional_movement = current_features.get('directional_movement', 0)
        if directional_movement > 0.7:
            score += 0.3
            factors.append(f"Strong positive directional movement: {directional_movement:.2f}")
        elif directional_movement > 0.5:
            score += 0.2
            factors.append(f"Moderate positive directional movement: {directional_movement:.2f}")
        
        # Check consecutive higher highs and higher lows
        if len(feature_history) >= 3:
            price_increases = 0
            for i in range(len(feature_history) - 1, 0, -1):
                if 'price_change' in feature_history[i] and feature_history[i]['price_change'] > 0:
                    price_increases += 1
                else:
                    break
            
            if price_increases >= 3:
                score += 0.2
                factors.append(f"Consecutive price increases: {price_increases}")
        
        # Check support tests
        support_test = current_features.get('support_test', 0)
        if support_test > 0.7:
            score += 0.1
            factors.append(f"Recent support test: {support_test:.2f}")
        
        # Check volatility - moderate volatility better for trends
        volatility = current_features.get('volatility', 0)
        if 0.3 <= volatility <= 0.7:
            score += 0.1
            factors.append(f"Appropriate trend volatility: {volatility:.2f}")
        
        return score, factors
    
    def _check_trending_down_regime(self, 
                                   current_features: Dict[str, Any],
                                   feature_history: List[Dict[str, Any]]
                                   ) -> Tuple[float, List[str]]:
        """
        Check for trending down regime characteristics.
        
        Args:
            current_features: Most recent feature data
            feature_history: Historical feature data
            
        Returns:
            Tuple of (score, list of contributing factors)
        """
        score = 0.0
        factors = []
        
        # Check trend strength
        trend_strength = current_features.get('trend_strength', 0)
        if trend_strength > 0.7:
            score += 0.3
            factors.append(f"Strong trend strength: {trend_strength:.2f}")
        elif trend_strength > 0.5:
            score += 0.2
            factors.append(f"Moderate trend strength: {trend_strength:.2f}")
        
        # Check directional movement - negative for downtrends
        directional_movement = current_features.get('directional_movement', 0)
        if directional_movement < -0.7:
            score += 0.3
            factors.append(f"Strong negative directional movement: {directional_movement:.2f}")
        elif directional_movement < -0.5:
            score += 0.2
            factors.append(f"Moderate negative directional movement: {directional_movement:.2f}")
        
        # Check consecutive lower highs and lower lows
        if len(feature_history) >= 3:
            price_decreases = 0
            for i in range(len(feature_history) - 1, 0, -1):
                if 'price_change' in feature_history[i] and feature_history[i]['price_change'] < 0:
                    price_decreases += 1
                else:
                    break
            
            if price_decreases >= 3:
                score += 0.2
                factors.append(f"Consecutive price decreases: {price_decreases}")
        
        # Check resistance tests
        resistance_test = current_features.get('resistance_test', 0)
        if resistance_test > 0.7:
            score += 0.1
            factors.append(f"Recent resistance test: {resistance_test:.2f}")
        
        # Check volatility - moderate volatility better for trends
        volatility = current_features.get('volatility', 0)
        if 0.3 <= volatility <= 0.7:
            score += 0.1
            factors.append(f"Appropriate trend volatility: {volatility:.2f}")
        
        return score, factors
    
    def _check_ranging_regime(self, 
                             current_features: Dict[str, Any],
                             feature_history: List[Dict[str, Any]]
                             ) -> Tuple[float, List[str]]:
        """
        Check for ranging regime characteristics.
        
        Args:
            current_features: Most recent feature data
            feature_history: Historical feature data
            
        Returns:
            Tuple of (score, list of contributing factors)
        """
        score = 0.0
        factors = []
        
        # Check range adherence
        range_adherence = current_features.get('range_adherence', 0)
        if range_adherence > 0.8:
            score += 0.3
            factors.append(f"Strong range adherence: {range_adherence:.2f}")
        elif range_adherence > 0.6:
            score += 0.2
            factors.append(f"Moderate range adherence: {range_adherence:.2f}")
        
        # Check mean reversion tendency
        mean_reversion = current_features.get('mean_reversion', 0)
        if mean_reversion > 0.8:
            score += 0.3
            factors.append(f"Strong mean reversion: {mean_reversion:.2f}")
        elif mean_reversion > 0.6:
            score += 0.2
            factors.append(f"Moderate mean reversion: {mean_reversion:.2f}")
        
        # Check low trend strength
        trend_strength = current_features.get('trend_strength', 1)
        if trend_strength < 0.3:
            score += 0.2
            factors.append(f"Low trend strength: {trend_strength:.2f}")
        elif trend_strength < 0.5:
            score += 0.1
            factors.append(f"Moderate-low trend strength: {trend_strength:.2f}")
        
        # Check support and resistance tests
        support_test = current_features.get('support_test', 0)
        resistance_test = current_features.get('resistance_test', 0)
        if support_test > 0.6 and resistance_test > 0.6:
            score += 0.2
            factors.append(f"Both support ({support_test:.2f}) and resistance ({resistance_test:.2f}) tests")
        elif support_test > 0.7 or resistance_test > 0.7:
            score += 0.1
            if support_test > resistance_test:
                factors.append(f"Support test: {support_test:.2f}")
            else:
                factors.append(f"Resistance test: {resistance_test:.2f}")
        
        # Check for alternating price direction
        if len(feature_history) >= 4:
            alternating = True
            for i in range(len(feature_history) - 1, 1, -1):
                if 'price_change' in feature_history[i] and 'price_change' in feature_history[i-1]:
                    if (feature_history[i]['price_change'] > 0 and feature_history[i-1]['price_change'] > 0) or \
                       (feature_history[i]['price_change'] < 0 and feature_history[i-1]['price_change'] < 0):
                        alternating = False
                        break
            
            if alternating:
                score += 0.1
                factors.append("Alternating price direction")
        
        return score, factors
    
    def _check_volatile_regime(self, 
                              current_features: Dict[str, Any],
                              feature_history: List[Dict[str, Any]]
                              ) -> Tuple[float, List[str]]:
        """
        Check for volatile regime characteristics.
        
        Args:
            current_features: Most recent feature data
            feature_history: Historical feature data
            
        Returns:
            Tuple of (score, list of contributing factors)
        """
        score = 0.0
        factors = []
        
        # Check high volatility
        volatility = current_features.get('volatility', 0)
        if volatility > 0.9:
            score += 0.4
            factors.append(f"Extreme volatility: {volatility:.2f}")
        elif volatility > 0.7:
            score += 0.3
            factors.append(f"High volatility: {volatility:.2f}")
        elif volatility > 0.5:
            score += 0.1
            factors.append(f"Elevated volatility: {volatility:.2f}")
        
        # Check volatility expansion
        volatility_expansion = current_features.get('volatility_expansion', 0)
        if volatility_expansion > 0.8:
            score += 0.3
            factors.append(f"Strong volatility expansion: {volatility_expansion:.2f}")
        elif volatility_expansion > 0.6:
            score += 0.2
            factors.append(f"Moderate volatility expansion: {volatility_expansion:.2f}")
        
        # Check large price swings
        if len(feature_history) >= 3:
            recent_changes = [abs(f.get('price_change_pct', 0)) for f in feature_history[-3:] 
                             if 'price_change_pct' in f]
            if recent_changes and max(recent_changes) > 0.03:  # >3% moves
                score += 0.2
                factors.append(f"Large price swings: {max(recent_changes)*100:.1f}%")
        
        # Check volume surge
        volume_surge = current_features.get('volume_surge', 0)
        if volume_surge > 0.8:
            score += 0.1
            factors.append(f"Volume surge: {volume_surge:.2f}")
        
        return score, factors
    
    def _check_breakout_regime(self, 
                              current_features: Dict[str, Any],
                              feature_history: List[Dict[str, Any]]
                              ) -> Tuple[float, List[str]]:
        """
        Check for breakout regime characteristics.
        
        Args:
            current_features: Most recent feature data
            feature_history: Historical feature data
            
        Returns:
            Tuple of (score, list of contributing factors)
        """
        score = 0.0
        factors = []
        
        # Check range breakout
        range_breakout = current_features.get('range_breakout', 0)
        if range_breakout > 0.9:
            score += 0.4
            factors.append(f"Strong range breakout: {range_breakout:.2f}")
        elif range_breakout > 0.7:
            score += 0.3
            factors.append(f"Moderate range breakout: {range_breakout:.2f}")
        
        # Check support or resistance break
        support_break = current_features.get('support_break', 0)
        resistance_break = current_features.get('resistance_break', 0)
        if resistance_break > 0.8:
            score += 0.3
            factors.append(f"Resistance break: {resistance_break:.2f}")
        elif support_break > 0.8:
            score += 0.3
            factors.append(f"Support break: {support_break:.2f}")
        
        # Check volume surge
        volume_surge = current_features.get('volume_surge', 0)
        if volume_surge > 0.8:
            score += 0.2
            factors.append(f"Volume surge: {volume_surge:.2f}")
        elif volume_surge > 0.6:
            score += 0.1
            factors.append(f"Elevated volume: {volume_surge:.2f}")
        
        # Check volatility expansion
        volatility_expansion = current_features.get('volatility_expansion', 0)
        if volatility_expansion > 0.7:
            score += 0.1
            factors.append(f"Volatility expansion: {volatility_expansion:.2f}")
        
        # Check directional movement - either direction for breakouts
        directional_movement = abs(current_features.get('directional_movement', 0))
        if directional_movement > 0.8:
            score += 0.1
            factors.append(f"Strong directional movement: {directional_movement:.2f}")
        
        return score, factors
    
    async def _store_regime_data(self, 
                               asset_id: str, 
                               platform: str, 
                               timeframe: str, 
                               regime: str, 
                               confidence: float) -> None:
        """
        Store regime data for historical analysis.
        
        Args:
            asset_id: Asset identifier
            platform: Trading platform
            timeframe: Market timeframe
            regime: Detected regime
            confidence: Confidence score for the regime
        """
        try:
            # Create new regime data record
            regime_data = MarketRegimeData(
                asset_id=asset_id,
                platform=platform,
                timeframe=timeframe,
                regime=regime,
                confidence=confidence,
                detected_at=datetime.now()
            )
            
            # Save to database (async operation)
            await regime_data.save()
            
        except Exception as e:
            self.logger.warning(f"Failed to store regime data: {str(e)}")
    
    def _calculate_transition_probabilities(self, 
                                          current_regime: str, 
                                          asset_id: str, 
                                          platform: str, 
                                          timeframe: str) -> Dict[str, float]:
        """
        Calculate probabilities of transitioning to other regimes.
        
        Args:
            current_regime: Current market regime
            asset_id: Asset identifier
            platform: Trading platform
            timeframe: Market timeframe
            
        Returns:
            Dictionary of regimes with transition probabilities
        """
        # In a production system, this would be based on historical analysis
        # of regime transitions for this specific asset
        
        # For now, we'll use simplified probabilities
        key = f"{platform}:{asset_id}:{timeframe}"
        
        # Different transition models by current regime
        if current_regime == "trending_up":
            return {
                "trending_up": 0.70,      # Continue trending up
                "ranging": 0.20,          # Consolidate into range
                "volatile": 0.05,         # Become volatile
                "breakout": 0.02,         # New breakout
                "trending_down": 0.03     # Reverse to downtrend
            }
        
        elif current_regime == "trending_down":
            return {
                "trending_down": 0.70,    # Continue trending down
                "ranging": 0.20,          # Consolidate into range
                "volatile": 0.05,         # Become volatile
                "breakout": 0.02,         # New breakout
                "trending_up": 0.03       # Reverse to uptrend
            }
        
        elif current_regime == "ranging":
            return {
                "ranging": 0.65,          # Continue ranging
                "breakout": 0.15,         # Break out of range
                "trending_up": 0.10,      # Begin uptrend
                "trending_down": 0.08,    # Begin downtrend
                "volatile": 0.02          # Become volatile
            }
        
        elif current_regime == "volatile":
            return {
                "volatile": 0.50,         # Continue volatility
                "ranging": 0.25,          # Settle into range
                "trending_up": 0.10,      # Begin uptrend
                "trending_down": 0.10,    # Begin downtrend
                "breakout": 0.05          # New breakout
            }
        
        elif current_regime == "breakout":
            return {
                "trending_up": 0.40,      # Continue into uptrend
                "trending_down": 0.30,    # Continue into downtrend
                "ranging": 0.15,          # Return to ranging
                "volatile": 0.10,         # Become volatile
                "breakout": 0.05          # Another breakout
            }
        
        else:
            # Unknown or undefined regime
            return {regime: 0.2 for regime in self.recognized_regimes}
    
    async def _check_regime_change(self, 
                                 asset_id: str, 
                                 platform: str, 
                                 timeframe: str, 
                                 regime_data: Dict[str, Any]) -> None:
        """
        Check if there's been a regime change and generate alerts.
        
        Args:
            asset_id: Asset identifier
            platform: Trading platform
            timeframe: Market timeframe
            regime_data: Current regime data
        """
        current_key = f"{platform}:{asset_id}:{timeframe}"
        current_regime = regime_data.get('regime')
        
        # Initialize transition tracking if needed
        if current_key not in self.regime_transitions:
            self.regime_transitions[current_key] = {
                'previous_regime': current_regime,
                'last_change': datetime.now(),
                'changes': []
            }
            return  # No previous data to compare
        
        prev_regime = self.regime_transitions[current_key]['previous_regime']
        
        # Check for regime change
        if prev_regime != current_regime and prev_regime != 'unknown' and current_regime != 'unknown':
            # Record transition
            self.regime_transitions[current_key]['changes'].append({
                'from': prev_regime,
                'to': current_regime,
                'timestamp': datetime.now(),
                'confidence': regime_data.get('confidence', 0)
            })
            
            # Update last change time
            self.regime_transitions[current_key]['last_change'] = datetime.now()
            
            # Limit history
            max_changes = 20
            if len(self.regime_transitions[current_key]['changes']) > max_changes:
                self.regime_transitions[current_key]['changes'] = \
                    self.regime_transitions[current_key]['changes'][-max_changes:]
            
            # Generate alert
            self._add_regime_alert(asset_id, platform, timeframe, prev_regime, current_regime)
        
        # Update previous regime
        self.regime_transitions[current_key]['previous_regime'] = current_regime
    
    def _add_regime_alert(self, 
                         asset_id: str, 
                         platform: str, 
                         timeframe: str, 
                         prev_regime: str, 
                         current_regime: str) -> None:
        """
        Add a regime change alert.
        
        Args:
            asset_id: Asset identifier
            platform: Trading platform
            timeframe: Market timeframe
            prev_regime: Previous regime
            current_regime: New regime
        """
        alert = {
            'asset_id': asset_id,
            'platform': platform,
            'timeframe': timeframe,
            'from_regime': prev_regime,
            'to_regime': current_regime,
            'timestamp': datetime.now(),
            'message': f"Market regime change: {prev_regime} -> {current_regime}",
            'suggested_actions': self._get_regime_change_actions(prev_regime, current_regime)
        }
        
        # Add to alerts
        self.regime_alerts.append(alert)
        
        # Limit alerts history
        max_alerts = 100
        if len(self.regime_alerts) > max_alerts:
            self.regime_alerts = self.regime_alerts[-max_alerts:]
        
        self.logger.info(f"Regime change alert: {platform}:{asset_id}:{timeframe} - {prev_regime} -> {current_regime}")
    
    def _get_regime_change_actions(self, 
                                  prev_regime: str, 
                                  current_regime: str) -> List[str]:
        """
        Generate suggested actions for a regime change.
        
        Args:
            prev_regime: Previous regime
            current_regime: New regime
            
        Returns:
            List of suggested actions
        """
        # Different suggestions based on the type of transition
        if prev_regime == "ranging" and current_regime == "breakout":
            return [
                "Consider breakout trading strategies",
                "Prepare for increased volatility",
                "Monitor for trend confirmation",
                "Consider increasing position sizes on confirmation"
            ]
        
        elif prev_regime == "ranging" and current_regime in ["trending_up", "trending_down"]:
            return [
                "Shift to trend-following strategies",
                "Consider trailing stops for trend capture",
                "Look for pullback entry opportunities",
                "Monitor momentum indicators for trend strength"
            ]
        
        elif prev_regime in ["trending_up", "trending_down"] and current_regime == "ranging":
            return [
                "Shift to range-trading strategies",
                "Consider mean-reversion approaches",
                "Tighten stop losses due to reduced directional movement",
                "Look for support/resistance levels to trade"
            ]
        
        elif prev_regime in ["trending_up", "trending_down", "ranging"] and current_regime == "volatile":
            return [
                "Reduce position sizes due to increased risk",
                "Widen stop losses to accommodate volatility",
                "Consider options strategies to capitalize on volatility",
                "Prepare for potential breakouts or trend reversals"
            ]
        
        elif prev_regime == "volatile" and current_regime in ["trending_up", "trending_down"]:
            return [
                "Transition to trend-following strategies",
                "Look for high probability entry after volatility subsides",
                "Monitor volume for confirmation of trend",
                "Consider scaling into positions as trend confirms"
            ]
        
        elif current_regime == "trending_up":
            return [
                "Focus on momentum and trend strategies",
                "Look for pullback entry opportunities",
                "Consider trailing stops to capture trend",
                "Monitor higher timeframes for trend alignment"
            ]
        
        elif current_regime == "trending_down":
            return [
                "Focus on short positions or trend-following strategies",
                "Look for bounces as potential short entries",
                "Consider trailing stops to capture downward trend",
                "Monitor support levels for potential trend exhaustion"
            ]
        
        # Default suggestions
        return [
            "Adapt strategies to the new market regime",
            "Review position sizing and risk parameters",
            "Monitor for further regime confirmation",
            "Consider strategies that perform well in this regime"
        ]
    
    def _get_relevant_alerts(self, 
                            asset_id: str, 
                            platform: str, 
                            timeframe: str
                            ) -> List[Dict[str, Any]]:
        """
        Get relevant regime alerts for a specific asset and timeframe.
        
        Args:
            asset_id: Asset identifier
            platform: Trading platform
            timeframe: Market timeframe
            
        Returns:
            List of relevant alerts
        """
        # Filter alerts for the specific asset and timeframe
        relevant_alerts = [
            alert for alert in self.regime_alerts
            if alert['asset_id'] == asset_id and
               alert['platform'] == platform and
               alert['timeframe'] == timeframe
        ]
        
        # Sort by timestamp (most recent first)
        return sorted(relevant_alerts, key=lambda x: x['timestamp'], reverse=True)
    
    def _get_strategy_weights(self, regime: str) -> Dict[str, float]:
        """
        Get recommended strategy weights for the current regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Dictionary of strategy weights
        """
        # If unknown regime, use balanced weights
        if regime == 'unknown':
            strategies = list(self.strategy_regime_effectiveness.keys())
            weight = 1.0 / len(strategies)
            return {strategy: weight for strategy in strategies}
        
        # Get effectiveness scores for each strategy in this regime
        weights = {}
        total_score = 0
        
        for strategy, regime_scores in self.strategy_regime_effectiveness.items():
            if regime in regime_scores:
                score = regime_scores[regime]
                weights[strategy] = score
                total_score += score
        
        # Normalize weights
        if total_score > 0:
            return {strategy: score/total_score for strategy, score in weights.items()}
        else:
            # Fallback to equal weights
            return {strategy: 1.0/len(weights) for strategy in weights.keys()}
    
    async def get_regime_history(self, 
                               asset_id: str, 
                               platform: str, 
                               timeframe: str = None, 
                               limit: int = 20
                               ) -> List[Dict[str, Any]]:
        """
        Get historical regime data for an asset.
        
        Args:
            asset_id: Asset identifier
            platform: Trading platform
            timeframe: Optional timeframe filter
            limit: Maximum number of records to return
            
        Returns:
            List of historical regime data
        """
        try:
            # Build query parameters
            query = {
                'asset_id': asset_id,
                'platform': platform
            }
            
            if timeframe:
                query['timeframe'] = timeframe
            
            # Retrieve from database
            history = await MarketRegimeData.get_history(**query, limit=limit)
            
            # Format results
            result = []
            for item in history:
                result.append({
                    'asset_id': item.asset_id,
                    'platform': item.platform,
                    'timeframe': item.timeframe,
                    'regime': item.regime,
                    'confidence': item.confidence,
                    'detected_at': item.detected_at
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve regime history: {str(e)}")
            return []
    
    async def get_regime_transition_stats(self, 
                                        asset_id: str, 
                                        platform: str, 
                                        timeframe: str = None
                                        ) -> Dict[str, Any]:
        """
        Get regime transition statistics for an asset.
        
        Args:
            asset_id: Asset identifier
            platform: Trading platform
            timeframe: Optional timeframe filter
            
        Returns:
            Dictionary with regime transition statistics
        """
        try:
            # Get regime history
            history = await self.get_regime_history(asset_id, platform, timeframe, limit=100)
            
            if not history:
                return {
                    'asset_id': asset_id,
                    'platform': platform,
                    'timeframe': timeframe,
                    'transitions': {},
                    'regime_durations': {},
                    'current_regime': 'unknown'
                }
            
            # Calculate transitions
            transitions = {}
            durations = {regime: [] for regime in self.recognized_regimes}
            
            current_regime = None
            regime_start = None
            
            for i, entry in enumerate(reversed(history)):  # Process from oldest to newest
                regime = entry['regime']
                
                # Track durations
                if current_regime != regime:
                    # Record duration of previous regime
                    if current_regime and regime_start:
                        duration = (entry['detected_at'] - regime_start).total_seconds() / 3600  # Hours
                        durations[current_regime].append(duration)
                    
                    # Record transition
                    if current_regime:
                        transition_key = f"{current_regime}->{regime}"
                        transitions[transition_key] = transitions.get(transition_key, 0) + 1
                    
                    # Update tracking
                    current_regime = regime
                    regime_start = entry['detected_at']
            
            # Calculate average durations
            avg_durations = {}
            for regime, duration_list in durations.items():
                if duration_list:
                    avg_durations[regime] = sum(duration_list) / len(duration_list)
                else:
                    avg_durations[regime] = 0
            
            return {
                'asset_id': asset_id,
                'platform': platform,
                'timeframe': timeframe,
                'transitions': transitions,
                'regime_durations': avg_durations,
                'current_regime': history[-1]['regime'] if history else 'unknown'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate regime transition stats: {str(e)}")
            return {
                'asset_id': asset_id,
                'platform': platform,
                'timeframe': timeframe,
                'transitions': {},
                'regime_durations': {},
                'current_regime': 'unknown',
                'error': str(e)
            }
    
    async def get_regime_performance(self, 
                                   asset_id: str, 
                                   platform: str
                                   ) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for different regimes.
        
        Args:
            asset_id: Asset identifier
            platform: Trading platform
            
        Returns:
            Dictionary with performance metrics by regime
        """
        # In a production system, this would analyze historical trade performance
        # in different market regimes to identify which regimes are most profitable
        
        # For demonstration purposes, return simulated data
        return {
            'trending_up': {
                'win_rate': 0.72,
                'avg_return': 0.031,
                'max_drawdown': -0.015,
                'sharpe_ratio': 1.8
            },
            'trending_down': {
                'win_rate': 0.65,
                'avg_return': 0.025,
                'max_drawdown': -0.022,
                'sharpe_ratio': 1.5
            },
            'ranging': {
                'win_rate': 0.68,
                'avg_return': 0.018,
                'max_drawdown': -0.012,
                'sharpe_ratio': 1.6
            },
            'volatile': {
                'win_rate': 0.55,
                'avg_return': 0.035,
                'max_drawdown': -0.042,
                'sharpe_ratio': 1.2
            },
            'breakout': {
                'win_rate': 0.65,
                'avg_return': 0.045,
                'max_drawdown': -0.028,
                'sharpe_ratio': 1.7
            }
        }
    
    async def get_current_regimes(self, 
                                platform: str = None, 
                                timeframe: str = None
                                ) -> List[Dict[str, Any]]:
        """
        Get current regimes for all tracked assets.
        
        Args:
            platform: Optional platform filter
            timeframe: Optional timeframe filter
            
        Returns:
            List of current regime data for assets
        """
        results = []
        
        for key, regime_data in self.current_regimes.items():
            parts = key.split(':')
            if len(parts) == 3:
                asset_platform, asset_id, asset_timeframe = parts
                
                # Apply filters if specified
                if platform and asset_platform != platform:
                    continue
                
                if timeframe and asset_timeframe != timeframe:
                    continue
                
                # Add to results
                results.append({
                    'asset_id': asset_id,
                    'platform': asset_platform,
                    'timeframe': asset_timeframe,
                    'regime': regime_data.get('regime', 'unknown'),
                    'confidence': regime_data.get('confidence', 0),
                    'detected_at': regime_data.get('detected_at', datetime.now()),
                    'regime_duration': regime_data.get('regime_duration', 0),
                    'sub_regimes': {k: v['probability'] for k, v in regime_data.get('sub_regimes', {}).items()}
                })
        
        return results
    
    async def stop(self) -> None:
        """Stop the regime council operations."""
        try:
            self.running = False
            self.logger.info("Stopped Regime Council")
        
        except Exception as e:
            self.logger.error(f"Error stopping Regime Council: {str(e)}")
            raise RegimeCouncilError(f"Stop error: {str(e)}")
