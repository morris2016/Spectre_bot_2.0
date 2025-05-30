#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Ensemble Brain - Combines multiple strategy approaches

This brain combines the signals from multiple strategy brains to create
a more robust and reliable trading strategy with higher win rates. It implements
various ensemble techniques including voting, weighting, stacking, and adaptive
combination based on performance history.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import asyncio
from datetime import datetime, timedelta
import json
import time

from strategy_brains.base_brain import BaseBrain
from common.constants import (
    SIGNAL_TYPES, TIMEFRAMES, RISK_LEVELS, 
    ENSEMBLE_METHODS, STRATEGY_PERFORMANCE_WINDOW,
    DEFAULT_ENSEMBLE_CONFIG
)
from common.utils import (
    calculate_weighted_signal, normalize_weights,
    calculate_success_rate, generate_ensemble_id
)
from common.exceptions import (
    EnsembleConfigError, 
    StrategyNotFoundError, 
    InsufficientDataError
)
from common.metrics import MetricsCollector

logger = logging.getLogger(__name__)

class EnsembleBrain(BaseBrain):
    """
    Ensemble Brain that combines multiple strategy approaches

    This brain synthesizes signals from multiple strategy brains using various
    ensemble techniques to create a more robust trading strategy with higher
    success rates. It adapts over time by learning which strategies perform
    best in different market conditions.
    """

    def __init__(
        self, 
        brain_id: str,
        config: Dict[str, Any],
        platform: str,
        symbol: str, 
        timeframe: str,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize the Ensemble Brain

        Args:
            brain_id: Unique identifier for this brain
            config: Configuration parameters for the brain
            platform: Trading platform (Binance, Deriv)
            symbol: Trading instrument symbol
            timeframe: Timeframe for analysis
            metrics_collector: Optional metrics collector for performance tracking
        """
        super().__init__(
            brain_id=brain_id,
            brain_type="ensemble",
            config=config,
            platform=platform,
            symbol=symbol,
            timeframe=timeframe,
            metrics_collector=metrics_collector
        )
        
        # Ensemble-specific configuration
        self.ensemble_method = config.get("ensemble_method", ENSEMBLE_METHODS.WEIGHTED)
        self.member_brains = {}
        self.member_weights = {}
        self.performance_history = {}
        self.weight_adjustment_interval = config.get("weight_adjustment_interval", 24)
        self.last_weight_adjustment = datetime.now()
        self.min_members_for_signal = config.get("min_members_for_signal", 3)
        self.confidence_threshold = config.get("confidence_threshold", 0.65)
        self.diversification_factor = config.get("diversification_factor", 0.2)
        self.use_stacking = config.get("use_stacking", True)
        self.stacking_model = None
        self.meta_learner_enabled = config.get("meta_learner_enabled", True)
        self.meta_features = []
        
        # Initialize performance metrics
        self.metrics = {
            "overall_success_rate": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "total_signals": 0,
            "best_performing_brain": None,
            "worst_performing_brain": None,
            "strategy_correlations": {},
        }
        
        # Load the member brains
        self._load_member_brains()
        
        # Initialize the stacking model if enabled
        if self.use_stacking:
            self._initialize_stacking_model()
        
        logger.info(f"Initialized Ensemble Brain {brain_id} for {platform} {symbol} {timeframe}")

    def _load_member_brains(self) -> None:
        """
        Load and initialize all member brains from configuration
        """
        if "member_brains" not in self.config:
            raise EnsembleConfigError("No member brains specified in configuration")

        from strategy_brains import get_brain_by_type
        
        members = self.config["member_brains"]
        if len(members) < 2:
            raise EnsembleConfigError("Ensemble brain requires at least 2 member brains")
        
        # Initialize each member brain
        for member in members:
            brain_type = member["type"]
            brain_config = member["config"]
            
            # Add platform, symbol and timeframe to config if not present
            brain_config.setdefault("platform", self.platform)
            brain_config.setdefault("symbol", self.symbol)
            brain_config.setdefault("timeframe", self.timeframe)
            
            # Generate a unique ID for this member brain
            member_id = member.get("id", f"{brain_type}_{generate_ensemble_id()}")
            
            try:
                # Instantiate the brain
                brain_class = get_brain_by_type(brain_type)
                brain_instance = brain_class(
                    brain_id=member_id,
                    config=brain_config,
                    platform=self.platform,
                    symbol=self.symbol,
                    timeframe=self.timeframe,
                    metrics_collector=self.metrics_collector
                )
                
                # Add to member brains dictionary
                self.member_brains[member_id] = brain_instance
                
                # Initialize weights evenly across all brains
                self.member_weights[member_id] = member.get("initial_weight", 1.0)
                
                # Initialize performance history
                self.performance_history[member_id] = {
                    "success_rate": 0.5,  # Start with neutral assumption
                    "signals": [],
                    "wins": 0,
                    "losses": 0,
                    "last_signal": None,
                    "market_conditions": {}
                }
                
                logger.info(f"Added member brain {member_id} of type {brain_type} to ensemble")
                
            except Exception as e:
                logger.error(f"Failed to initialize member brain {brain_type}: {str(e)}")
                raise
        
        # Normalize weights to ensure they sum to 1.0
        self._normalize_weights()
        
        logger.info(f"Loaded {len(self.member_brains)} member brains for ensemble")

    def _normalize_weights(self) -> None:
        """
        Normalize brain weights to ensure they sum to 1.0
        """
        total_weight = sum(self.member_weights.values())
        if total_weight > 0:
            for brain_id in self.member_weights:
                self.member_weights[brain_id] /= total_weight
        else:
            # If all weights are zero (unlikely), set equal weights
            equal_weight = 1.0 / len(self.member_weights)
            for brain_id in self.member_weights:
                self.member_weights[brain_id] = equal_weight

    def _initialize_stacking_model(self) -> None:
        """
        Initialize the stacking model for meta-learning
        """
        from ml_models.models.ensemble import StackingEnsemble
        
        try:
            self.stacking_model = StackingEnsemble(
                model_id=f"ensemble_stacker_{self.brain_id}",
                config={
                    "base_models": len(self.member_brains),
                    "meta_learner": "random_forest",
                    "cv_folds": 3,
                    "use_probabilities": True,
                }
            )
            logger.info(f"Initialized stacking model for ensemble brain {self.brain_id}")
        except Exception as e:
            logger.error(f"Failed to initialize stacking model: {str(e)}")
            self.use_stacking = False

    async def update(self, market_data: pd.DataFrame) -> None:
        """
        Update the ensemble brain and all member brains with new market data
        
        Args:
            market_data: DataFrame containing market data
        """
        if market_data is None or len(market_data) < self.min_data_points:
            raise InsufficientDataError(
                f"Insufficient data points for Ensemble Brain: {len(market_data) if market_data is not None else 0} "
                f"(minimum required: {self.min_data_points})"
            )
            
        # Update state and metadata
        self.last_update_time = datetime.now()
        self.last_processed_candle = market_data.iloc[-1]
        
        # Update all member brains asynchronously
        update_tasks = []
        for brain_id, brain in self.member_brains.items():
            update_tasks.append(self._update_member_brain(brain, market_data))
        
        # Wait for all updates to complete
        await asyncio.gather(*update_tasks)
        
        # Check if it's time to adjust weights
        if self._should_adjust_weights():
            self._adjust_weights_based_on_performance()
            
        # Update stacking model if enabled
        if self.use_stacking and len(self.meta_features) > 10:
            self._update_stacking_model()
            
        # Calculate current market conditions
        market_conditions = self._analyze_market_conditions(market_data)
        self.state["market_conditions"] = market_conditions
        
        # Calculate strategy correlations
        self._update_strategy_correlations()
        
        logger.debug(f"Updated Ensemble Brain {self.brain_id} with {len(market_data)} data points")

    async def _update_member_brain(self, brain: BaseBrain, market_data: pd.DataFrame) -> None:
        """
        Update an individual member brain with new market data
        
        Args:
            brain: The member brain to update
            market_data: DataFrame containing market data
        """
        try:
            await brain.update(market_data)
        except Exception as e:
            logger.error(f"Error updating member brain {brain.brain_id}: {str(e)}")
            # Flag this brain as having update issues
            brain.state["update_error"] = str(e)

    def _analyze_market_conditions(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze current market conditions
        
        Args:
            market_data: DataFrame containing market data
            
        Returns:
            Dictionary of market condition indicators
        """
        try:
            # Extract last 100 candles or available data
            recent_data = market_data.tail(min(100, len(market_data)))
            
            # Calculate volatility
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252 * 24 / self.timeframe_multiplier)
            
            # Determine trend strength using ADX
            from feature_service.features.technical import ADXIndicator
            adx = ADXIndicator().calculate(recent_data)
            adx_value = adx.iloc[-1] if not adx.empty else 0
            
            # Determine if market is ranging or trending
            is_trending = adx_value > 25
            
            # Determine volume profile
            avg_volume = recent_data['volume'].mean()
            current_volume = recent_data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Determine market regime using regime detection
            from feature_service.features.market_structure import MarketRegimeDetector
            regime_detector = MarketRegimeDetector()
            regime = regime_detector.detect_regime(recent_data)
            
            return {
                "volatility": float(volatility),
                "adx": float(adx_value),
                "is_trending": bool(is_trending),
                "volume_ratio": float(volume_ratio),
                "regime": regime,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {str(e)}")
            return {
                "volatility": 0.0,
                "adx": 0.0,
                "is_trending": False,
                "volume_ratio": 1.0,
                "regime": "unknown",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def _should_adjust_weights(self) -> bool:
        """
        Determine if it's time to adjust brain weights
        
        Returns:
            True if weights should be adjusted, False otherwise
        """
        # Check if enough time has passed since last adjustment
        hours_since_adjustment = (datetime.now() - self.last_weight_adjustment).total_seconds() / 3600
        
        return hours_since_adjustment >= self.weight_adjustment_interval

    def _adjust_weights_based_on_performance(self) -> None:
        """
        Adjust member brain weights based on performance history
        """
        try:
            # Get current market conditions
            market_conditions = self.state.get("market_conditions", {})
            regime = market_conditions.get("regime", "unknown")
            
            # Calculate success rates for each brain
            for brain_id, history in self.performance_history.items():
                signals = history["signals"]
                
                # Only adjust weights if we have enough performance data
                if len(signals) >= 5:
                    # Filter signals by regime if possible
                    if regime != "unknown" and regime in history["market_conditions"]:
                        regime_signals = [s for s in signals if s.get("regime") == regime]
                        if len(regime_signals) >= 3:
                            signals = regime_signals
                    
                    # Calculate success rate
                    wins = sum(1 for s in signals if s.get("result") == "win")
                    total = len(signals)
                    success_rate = wins / total if total > 0 else 0.5
                    
                    # Update performance history
                    history["success_rate"] = success_rate
                    
                    # Apply weight based on success rate with smoothing
                    # We don't want to completely eliminate a strategy
                    min_weight = 0.05  # Minimum weight to ensure some diversity
                    self.member_weights[brain_id] = max(
                        min_weight,
                        success_rate ** 2  # Square to amplify differences
                    )
            
            # Apply diversification factor - adjust weights to maintain some diversity
            if self.diversification_factor > 0:
                # Get min and max weights
                min_weight = min(self.member_weights.values())
                max_weight = max(self.member_weights.values())
                
                # Only apply if there's a significant difference
                if max_weight > 2 * min_weight:
                    for brain_id in self.member_weights:
                        # Pull weights slightly toward the average
                        self.member_weights[brain_id] = (
                            (1 - self.diversification_factor) * self.member_weights[brain_id] +
                            self.diversification_factor * (1.0 / len(self.member_weights))
                        )
            
            # Normalize weights
            self._normalize_weights()
            
            # Update metrics for best and worst brains
            if self.member_weights:
                best_brain = max(self.member_weights.items(), key=lambda x: x[1])[0]
                worst_brain = min(self.member_weights.items(), key=lambda x: x[1])[0]
                
                self.metrics["best_performing_brain"] = best_brain
                self.metrics["worst_performing_brain"] = worst_brain
            
            self.last_weight_adjustment = datetime.now()
            
            # Log the new weights
            weight_info = {brain_id: round(weight, 3) for brain_id, weight in self.member_weights.items()}
            logger.info(f"Adjusted ensemble weights: {json.dumps(weight_info)}")
            
            # Track metrics
            if self.metrics_collector:
                self.metrics_collector.track_metric(
                    f"ensemble_weights.{self.brain_id}",
                    weight_info,
                    tags={
                        "platform": self.platform,
                        "symbol": self.symbol,
                        "timeframe": self.timeframe,
                    }
                )
                
        except Exception as e:
            logger.error(f"Error adjusting weights for ensemble brain {self.brain_id}: {str(e)}")

    def _update_strategy_correlations(self) -> None:
        """
        Update correlation metrics between different strategies
        """
        # We need at least 10 signals from each brain to calculate correlations
        min_signals = 10
        
        # Extract signals and results
        brain_results = {}
        for brain_id, history in self.performance_history.items():
            signals = history["signals"]
            if len(signals) >= min_signals:
                # Convert win/loss to 1/0 for correlation calculation
                results = [1 if s.get("result") == "win" else 0 for s in signals[-min_signals:]]
                brain_results[brain_id] = results
        
        # Calculate correlations if we have enough data
        if len(brain_results) >= 2:
            correlations = {}
            brain_ids = list(brain_results.keys())
            
            for i in range(len(brain_ids)):
                for j in range(i+1, len(brain_ids)):
                    brain1, brain2 = brain_ids[i], brain_ids[j]
                    results1, results2 = brain_results[brain1], brain_results[brain2]
                    
                    # Calculate correlation (handle case where all results are the same)
                    try:
                        corr = np.corrcoef(results1, results2)[0, 1]
                        correlations[f"{brain1}_{brain2}"] = float(corr)
                    except:
                        correlations[f"{brain1}_{brain2}"] = 0.0
            
            self.metrics["strategy_correlations"] = correlations

    def _update_stacking_model(self) -> None:
        """
        Update the stacking model with performance data
        """
        if not self.use_stacking or self.stacking_model is None:
            return
            
        try:
            # Prepare training data
            X = []
            y = []
            
            # For each historical signal with meta features
            for meta_feature in self.meta_features:
                X.append(meta_feature["features"])
                y.append(1 if meta_feature["result"] == "win" else 0)
            
            # Only train if we have enough data
            if len(X) >= 20:
                X = np.array(X)
                y = np.array(y)
                
                # Train the model
                self.stacking_model.train(X, y)
                logger.info(f"Updated stacking model with {len(X)} examples")
                
                # Trim meta features to prevent excessive memory usage
                if len(self.meta_features) > 1000:
                    self.meta_features = self.meta_features[-1000:]
        
        except Exception as e:
            logger.error(f"Error updating stacking model: {str(e)}")

    async def generate_signal(self) -> Dict[str, Any]:
        """
        Generate a trading signal by combining signals from member brains
        
        Returns:
            Dict containing the signal details
        """
        signal_start_time = time.time()
        
        try:
            # Get signals from all member brains asynchronously
            brain_signals = {}
            signal_tasks = []
            
            for brain_id, brain in self.member_brains.items():
                task = self._get_brain_signal(brain_id, brain)
                signal_tasks.append(task)
            
            # Wait for all signals to be generated
            brain_signal_results = await asyncio.gather(*signal_tasks)
            
            # Process the results
            for brain_id, signal in brain_signal_results:
                if signal and "signal_type" in signal:
                    brain_signals[brain_id] = signal
            
            # Check if we have enough signals to proceed
            if len(brain_signals) < self.min_members_for_signal:
                logger.info(f"Not enough member brain signals: {len(brain_signals)}/{self.min_members_for_signal}")
                return self._create_no_signal_response()
            
            # Combine signals based on ensemble method
            if self.ensemble_method == ENSEMBLE_METHODS.VOTING:
                final_signal = self._combine_signals_voting(brain_signals)
            elif self.ensemble_method == ENSEMBLE_METHODS.WEIGHTED:
                final_signal = self._combine_signals_weighted(brain_signals)
            elif self.ensemble_method == ENSEMBLE_METHODS.STACKING and self.use_stacking:
                final_signal = self._combine_signals_stacking(brain_signals)
            else:
                # Default to weighted method
                final_signal = self._combine_signals_weighted(brain_signals)
            
            # Add ensemble meta-information
            final_signal["ensemble_info"] = {
                "method": self.ensemble_method,
                "member_signals": len(brain_signals),
                "total_members": len(self.member_brains),
                "top_brain": self.metrics.get("best_performing_brain"),
                "weights": {k: round(v, 2) for k, v in self.member_weights.items()},
                "brain_signals": {k: v["signal_type"] for k, v in brain_signals.items()},
                "market_conditions": self.state.get("market_conditions", {})
            }
            
            # Track performance metrics for this signal
            if self.metrics_collector:
                self.metrics_collector.track_metric(
                    f"ensemble_signal.{self.brain_id}",
                    {
                        "confidence": final_signal.get("confidence", 0),
                        "signal_type": final_signal.get("signal_type", "NONE"),
                        "generation_time": time.time() - signal_start_time
                    },
                    tags={
                        "platform": self.platform,
                        "symbol": self.symbol,
                        "timeframe": self.timeframe,
                    }
                )
            
            logger.info(
                f"Generated {final_signal.get('signal_type')} signal with "
                f"{final_signal.get('confidence'):.2f} confidence using "
                f"{self.ensemble_method} method"
            )
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Error generating ensemble signal: {str(e)}")
            return self._create_no_signal_response()

    async def _get_brain_signal(self, brain_id: str, brain: BaseBrain) -> Tuple[str, Dict[str, Any]]:
        """
        Get signal from a member brain
        
        Args:
            brain_id: Identifier for the brain
            brain: The brain instance
            
        Returns:
            Tuple of (brain_id, signal_dict)
        """
        try:
            signal = await brain.generate_signal()
            return (brain_id, signal)
        except Exception as e:
            logger.error(f"Error getting signal from brain {brain_id}: {str(e)}")
            return (brain_id, None)

    def _combine_signals_voting(self, brain_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine signals using majority voting
        
        Args:
            brain_signals: Dictionary of brain_id -> signal mappings
            
        Returns:
            Combined signal dictionary
        """
        # Count votes for each signal type
        signal_votes = {
            SIGNAL_TYPES.BUY: 0,
            SIGNAL_TYPES.SELL: 0,
            SIGNAL_TYPES.NONE: 0
        }
        
        # Count weighted votes
        for brain_id, signal in brain_signals.items():
            signal_type = signal.get("signal_type", SIGNAL_TYPES.NONE)
            # Use equal weights for pure voting
            signal_votes[signal_type] += 1
        
        # Determine winning signal
        max_votes = max(signal_votes.values())
        winning_signals = [s for s, v in signal_votes.items() if v == max_votes]
        
        # Handle tie - prioritize NONE for safety
        if len(winning_signals) > 1:
            if SIGNAL_TYPES.NONE in winning_signals:
                winning_signal = SIGNAL_TYPES.NONE
            else:
                # Tie between BUY and SELL, use weighted method as tiebreaker
                weighted_signal = self._combine_signals_weighted(brain_signals)
                winning_signal = weighted_signal["signal_type"]
        else:
            winning_signal = winning_signals[0]
        
        # Calculate confidence based on vote proportion
        total_votes = sum(signal_votes.values())
        confidence = signal_votes[winning_signal] / total_votes if total_votes > 0 else 0.0
        
        # Build final signal
        return {
            "brain_id": self.brain_id,
            "brain_type": self.brain_type,
            "signal_type": winning_signal,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "platform": self.platform,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "entry_price": self._get_consensus_value(brain_signals, "entry_price"),
            "stop_loss": self._get_consensus_value(brain_signals, "stop_loss"),
            "take_profit": self._get_consensus_value(brain_signals, "take_profit"),
            "risk_level": self._get_consensus_value(brain_signals, "risk_level", RISK_LEVELS.MEDIUM),
            "expiration": self._get_farthest_expiration(brain_signals),
            "voting_results": signal_votes
        }

    def _combine_signals_weighted(self, brain_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine signals using weighted voting
        
        Args:
            brain_signals: Dictionary of brain_id -> signal mappings
            
        Returns:
            Combined signal dictionary
        """
        # Initialize vote counts with weights
        weighted_votes = {
            SIGNAL_TYPES.BUY: 0.0,
            SIGNAL_TYPES.SELL: 0.0,
            SIGNAL_TYPES.NONE: 0.0
        }
        
        confidence_sum = 0.0
        
        # Calculate weighted votes
        for brain_id, signal in brain_signals.items():
            if brain_id not in self.member_weights:
                continue
                
            signal_type = signal.get("signal_type", SIGNAL_TYPES.NONE)
            confidence = signal.get("confidence", 0.5)
            
            # Weight by both brain weight and signal confidence
            vote_weight = self.member_weights[brain_id] * confidence
            weighted_votes[signal_type] += vote_weight
            confidence_sum += vote_weight
        
        # Normalize votes
        if confidence_sum > 0:
            for signal_type in weighted_votes:
                weighted_votes[signal_type] /= confidence_sum
        
        # Determine winning signal - must exceed confidence threshold
        max_weight = max(weighted_votes.values())
        if max_weight < self.confidence_threshold:
            winning_signal = SIGNAL_TYPES.NONE
            confidence = max_weight
        else:
            winning_signals = [s for s, v in weighted_votes.items() if v == max_weight]
            winning_signal = winning_signals[0]
            if winning_signal == SIGNAL_TYPES.NONE:
                confidence = 1.0 - max(weighted_votes[SIGNAL_TYPES.BUY], weighted_votes[SIGNAL_TYPES.SELL])
            else:
                confidence = max_weight
        
        # Build final signal
        return {
            "brain_id": self.brain_id,
            "brain_type": self.brain_type,
            "signal_type": winning_signal,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "platform": self.platform,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "entry_price": self._get_weighted_consensus(brain_signals, "entry_price"),
            "stop_loss": self._get_weighted_consensus(brain_signals, "stop_loss"),
            "take_profit": self._get_weighted_consensus(brain_signals, "take_profit"),
            "risk_level": self._get_weighted_consensus(brain_signals, "risk_level", RISK_LEVELS.MEDIUM),
            "expiration": self._get_farthest_expiration(brain_signals),
            "weighted_votes": {k: round(v, 3) for k, v in weighted_votes.items()}
        }

    def _combine_signals_stacking(self, brain_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine signals using stacking ensemble method
        
        Args:
            brain_signals: Dictionary of brain_id -> signal mappings
            
        Returns:
            Combined signal dictionary
        """
        if not self.use_stacking or self.stacking_model is None:
            # Fall back to weighted method if stacking is not available
            return self._combine_signals_weighted(brain_signals)
        
        try:
            # Create feature vector from brain signals
            # Convert each brain's signal to a feature
            features = []
            
            # For each brain in our ensemble
            for brain_id in self.member_brains:
                if brain_id in brain_signals:
                    signal = brain_signals[brain_id]
                    signal_type = signal.get("signal_type", SIGNAL_TYPES.NONE)
                    confidence = signal.get("confidence", 0.5)
                    
                    # Encode signal type: BUY=1.0, NONE=0.0, SELL=-1.0
                    if signal_type == SIGNAL_TYPES.BUY:
                        signal_value = 1.0
                    elif signal_type == SIGNAL_TYPES.SELL:
                        signal_value = -1.0
                    else:
                        signal_value = 0.0
                    
                    # Add signal value and confidence to features
                    features.extend([signal_value, confidence])
                else:
                    # Missing brain - add neutral values
                    features.extend([0.0, 0.5])
            
            # Add market condition features
            market_conditions = self.state.get("market_conditions", {})
            features.append(market_conditions.get("volatility", 0.0))
            features.append(market_conditions.get("adx", 0.0))
            features.append(1.0 if market_conditions.get("is_trending", False) else 0.0)
            features.append(market_conditions.get("volume_ratio", 1.0))
            
            # Predict using stacking model
            features_array = np.array([features])
            prediction, confidence = self.stacking_model.predict_with_confidence(features_array)
            
            # Convert prediction to signal type
            if prediction == 1 and confidence >= self.confidence_threshold:
                signal_type = SIGNAL_TYPES.BUY
            elif prediction == 0 and confidence >= self.confidence_threshold:
                signal_type = SIGNAL_TYPES.SELL
            else:
                signal_type = SIGNAL_TYPES.NONE
                # Adjust confidence for no signal
                confidence = max(confidence, 1.0 - confidence)
            
            # Save features for future training
            self.meta_features.append({
                "features": features,
                "result": None,  # Will be updated when we know the outcome
                "signal_type": signal_type,
                "timestamp": datetime.now().isoformat(),
                "market_conditions": market_conditions
            })
            
            # Build final signal
            return {
                "brain_id": self.brain_id,
                "brain_type": self.brain_type,
                "signal_type": signal_type,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "platform": self.platform,
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "entry_price": self._get_weighted_consensus(brain_signals, "entry_price"),
                "stop_loss": self._get_weighted_consensus(brain_signals, "stop_loss"),
                "take_profit": self._get_weighted_consensus(brain_signals, "take_profit"),
                "risk_level": self._get_weighted_consensus(brain_signals, "risk_level", RISK_LEVELS.MEDIUM),
                "expiration": self._get_farthest_expiration(brain_signals),
                "meta_method": "stacking",
                "stacking_features": len(features)
            }
            
        except Exception as e:
            logger.error(f"Error in stacking ensemble: {str(e)}")
            # Fall back to weighted method
            return self._combine_signals_weighted(brain_signals)

    def _get_consensus_value(
        self, 
        brain_signals: Dict[str, Dict[str, Any]], 
        field: str, 
        default_value: Any = None
    ) -> Any:
        """
        Get consensus value from brain signals
        
        Args:
            brain_signals: Dictionary of brain_id -> signal mappings
            field: Field to extract
            default_value: Default value if none found
            
        Returns:
            Consensus value
        """
        values = [signal.get(field) for signal in brain_signals.values() if field in signal]
        
        if not values:
            return default_value
            
        # If the values are numeric, take the median
        try:
            return float(np.median([float(v) for v in values if v is not None]))
        except:
            # For non-numeric values, take most common
            from collections import Counter
            counter = Counter(values)
            return counter.most_common(1)[0][0]

    def _get_weighted_consensus(
        self, 
        brain_signals: Dict[str, Dict[str, Any]], 
        field: str, 
        default_value: Any = None
    ) -> Any:
        """
        Get weighted consensus value from brain signals
        
        Args:
            brain_signals: Dictionary of brain_id -> signal mappings
            field: Field to extract
            default_value: Default value if none found
            
        Returns:
            Weighted consensus value
        """
        values = []
        weights = []
        
        for brain_id, signal in brain_signals.items():
            if field in signal and brain_id in self.member_weights:
                values.append(signal[field])
                weights.append(self.member_weights[brain_id])
        
        if not values:
            return default_value
            
        # If the values are numeric, take the weighted average
        try:
            numeric_values = [float(v) for v in values if v is not None]
            valid_weights = [w for i, w in enumerate(weights) if values[i] is not None]
            
            if not numeric_values:
                return default_value
                
            return float(np.average(numeric_values, weights=valid_weights))
        except:
            # For non-numeric values, take weighted most common
            from collections import Counter
            weighted_counter = {}
            
            for i, value in enumerate(values):
                if value not in weighted_counter:
                    weighted_counter[value] = 0
                weighted_counter[value] += weights[i]
                
            if not weighted_counter:
                return default_value
                
            return max(weighted_counter.items(), key=lambda x: x[1])[0]

    def _get_farthest_expiration(self, brain_signals: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """
        Get the farthest expiration time from brain signals
        
        Args:
            brain_signals: Dictionary of brain_id -> signal mappings
            
        Returns:
            ISO-formatted datetime string of the farthest expiration
        """
        expirations = []
        
        for signal in brain_signals.values():
            expiration = signal.get("expiration")
            if expiration:
                try:
                    expirations.append(datetime.fromisoformat(expiration))
                except:
                    pass
        
        if not expirations:
            # Default to 24 hours from now
            return (datetime.now() + timedelta(hours=24)).isoformat()
            
        return max(expirations).isoformat()

    def _create_no_signal_response(self) -> Dict[str, Any]:
        """
        Create a response for when no signal is generated
        
        Returns:
            Signal dictionary with NONE signal type
        """
        return {
            "brain_id": self.brain_id,
            "brain_type": self.brain_type,
            "signal_type": SIGNAL_TYPES.NONE,
            "confidence": 1.0,
            "timestamp": datetime.now().isoformat(),
            "platform": self.platform,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "reason": "insufficient_brain_signals"
        }

    async def update_signal_result(
        self, 
        signal_id: str, 
        result: str, 
        profit_pips: Optional[float] = None
    ) -> None:
        """
        Update the result of a signal and propagate to member brains
        
        Args:
            signal_id: Unique identifier for the signal
            result: "win" or "loss"
            profit_pips: Optional profit/loss in pips
        """
        # Update overall metrics
        if result == "win":
            self.metrics["win_count"] += 1
        else:
            self.metrics["loss_count"] += 1
            
        self.metrics["total_signals"] += 1
        
        win_count = self.metrics["win_count"]
        total = self.metrics["total_signals"]
        
        self.metrics["overall_success_rate"] = win_count / total if total > 0 else 0
        
        # Find the meta feature entry for this signal and update it
        for feature in self.meta_features:
            if feature.get("signal_id") == signal_id:
                feature["result"] = result
                feature["profit_pips"] = profit_pips
                break
        
        # Propagate result to member brains
        update_tasks = []
        
        for brain_id, brain in self.member_brains.items():
            # Each brain might have contributed to this ensemble signal
            history = self.performance_history.get(brain_id, {})
            
            # Add to this brain's performance history
            if "signals" in history:
                # Get the market conditions at the time of the signal
                market_conditions = self.state.get("market_conditions", {})
                regime = market_conditions.get("regime", "unknown")
                
                # Add to signal history
                history["signals"].append({
                    "signal_id": signal_id,
                    "result": result,
                    "profit_pips": profit_pips,
                    "timestamp": datetime.now().isoformat(),
                    "regime": regime
                })
                
                # Update win/loss counters
                if result == "win":
                    history["wins"] += 1
                else:
                    history["losses"] += 1
                
                # Keep only recent signals
                max_signals = 100
                if len(history["signals"]) > max_signals:
                    history["signals"] = history["signals"][-max_signals:]
                
                # Update regime-specific performance
                if regime != "unknown":
                    if regime not in history["market_conditions"]:
                        history["market_conditions"][regime] = {"wins": 0, "losses": 0, "total": 0}
                    
                    regime_stats = history["market_conditions"][regime]
                    regime_stats["total"] += 1
                    if result == "win":
                        regime_stats["wins"] += 1
                    else:
                        regime_stats["losses"] += 1
            
            # Propagate to the actual brain
            update_tasks.append(
                brain.update_signal_result(signal_id, result, profit_pips)
            )
        
        # Execute all update tasks
        await asyncio.gather(*update_tasks)
        
        # Update weights based on new performance data
        self._adjust_weights_based_on_performance()
        
        logger.info(
            f"Updated signal result for ensemble {self.brain_id}: {result} "
            f"with {profit_pips if profit_pips is not None else 'unknown'} pips"
        )

    async def save_state(self) -> Dict[str, Any]:
        """
        Save the state of the ensemble brain and all member brains
        
        Returns:
            Dictionary containing state data
        """
        ensemble_state = {
            "brain_id": self.brain_id,
            "brain_type": self.brain_type,
            "platform": self.platform,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "ensemble_method": self.ensemble_method,
            "member_weights": self.member_weights,
            "performance_history": self.performance_history,
            "metrics": self.metrics,
            "last_weight_adjustment": self.last_weight_adjustment.isoformat(),
            "meta_features_count": len(self.meta_features),
            "use_stacking": self.use_stacking,
            "last_update_time": self.last_update_time.isoformat() if self.last_update_time else None,
            "state": self.state,
            "version": "1.0.0"
        }
        
        # Save member brain states
        member_states = {}
        for brain_id, brain in self.member_brains.items():
            member_states[brain_id] = await brain.save_state()
        
        ensemble_state["member_states"] = member_states
        
        return ensemble_state

    async def load_state(self, state_data: Dict[str, Any]) -> None:
        """
        Load the state of the ensemble brain and all member brains
        
        Args:
            state_data: Dictionary containing state data
        """
        try:
            # Check if the state is compatible
            if state_data.get("brain_id") != self.brain_id:
                logger.warning(f"State brain ID {state_data.get('brain_id')} does not match {self.brain_id}")
                return
                
            if state_data.get("brain_type") != self.brain_type:
                logger.warning(f"State brain type {state_data.get('brain_type')} does not match {self.brain_type}")
                return
            
            # Load the main ensemble state
            self.ensemble_method = state_data.get("ensemble_method", self.ensemble_method)
            self.member_weights = state_data.get("member_weights", self.member_weights)
            self.performance_history = state_data.get("performance_history", self.performance_history)
            self.metrics = state_data.get("metrics", self.metrics)
            
            # Parse datetime
            if "last_weight_adjustment" in state_data:
                try:
                    self.last_weight_adjustment = datetime.fromisoformat(state_data["last_weight_adjustment"])
                except:
                    self.last_weight_adjustment = datetime.now()
            
            if "last_update_time" in state_data and state_data["last_update_time"]:
                try:
                    self.last_update_time = datetime.fromisoformat(state_data["last_update_time"])
                except:
                    self.last_update_time = datetime.now()
            
            self.use_stacking = state_data.get("use_stacking", self.use_stacking)
            self.state = state_data.get("state", self.state)
            
            # Load member brain states
            member_states = state_data.get("member_states", {})
            for brain_id, brain_state in member_states.items():
                if brain_id in self.member_brains:
                    await self.member_brains[brain_id].load_state(brain_state)
                else:
                    logger.warning(f"Member brain {brain_id} in state not found in current ensemble")
            
            logger.info(f"Loaded state for ensemble brain {self.brain_id}")
            
        except Exception as e:
            logger.error(f"Error loading state for ensemble brain {self.brain_id}: {str(e)}")
            raise
