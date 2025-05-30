#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Brain Council - Signal Generator

This module is responsible for generating final trading signals based on the 
recommendations from different councils. It applies sophisticated filtering,
confidence scoring, and risk assessment to produce high-precision signals.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

from common.logger import get_logger
from common.utils import (
    calculate_risk_reward, calculate_confidence_score, 
    normalize_probability, validate_signal
)
from common.constants import (
    SIGNAL_TYPES, TIMEFRAMES, MIN_CONFIDENCE_THRESHOLD,
    ACTION_BUY, ACTION_SELL, ACTION_HOLD, ACTION_CLOSE
)
from common.exceptions import (
    SignalGenerationError, InvalidSignalError, NoConsensusError
)
from data_storage.models.market_data import SignalRecord
from feature_service.features.market_structure import MarketStructureAnalyzer
from feature_service.features.volatility import VolatilityAnalyzer
from brain_council.voting_system import VotingResult
from brain_council.weighting_system import WeightingSystem
from brain_council.performance_tracker import PerformanceTracker


class SignalGenerator:
    """
    Signal Generator for the Brain Council that produces final trading signals
    based on the aggregated recommendations from different councils.
    
    This component applies sophisticated filtering, confidence scoring, and
    risk assessment to produce high-precision signals with an emphasis on
    consistent win rates.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Signal Generator.
        
        Args:
            config: Configuration for the signal generator including thresholds,
                   filters, and risk parameters.
        """
        self.logger = get_logger("brain_council.signal_generator")
        self.config = config
        self.weighting_system = WeightingSystem(config.get("weighting", {}))
        self.performance_tracker = PerformanceTracker(config.get("performance", {}))
        self.market_structure = MarketStructureAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()
        
        # Signal filtering thresholds
        self.min_confidence = config.get("min_confidence", MIN_CONFIDENCE_THRESHOLD)
        self.min_council_agreement = config.get("min_council_agreement", 0.6)
        self.min_risk_reward = config.get("min_risk_reward", 1.5)
        
        # Signal history for pattern recognition and validation
        self.signal_history = defaultdict(list)
        self.max_history_size = config.get("max_history_size", 1000)
        
        # Counters for statistics
        self.signals_generated = 0
        self.signals_filtered = 0

        self.logger.info("Signal Generator initialized with min confidence: %.2f",
                        self.min_confidence)

    async def initialize(self) -> None:
        """Await initialization of dependent components."""
        if self.performance_tracker.initialization_task is not None:
            await self.performance_tracker.initialization_task
    
    async def generate_signal(self, voting_results: Dict[str, VotingResult], 
                       market_data: Dict[str, Any], 
                       asset_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a trading signal based on voting results and market data.
        
        Args:
            voting_results: Voting results from different councils
            market_data: Current market data including price, volume, etc.
            asset_config: Asset-specific configuration
            
        Returns:
            A signal dictionary or None if no valid signal could be generated
        """
        try:
            self.logger.debug("Generating signal for %s", asset_config.get("symbol", "unknown"))
            
            # Apply council weights based on past performance
            weighted_votes = self.weighting_system.apply_weights(voting_results, asset_config["symbol"])
            
            # Calculate consensus and confidence
            consensus_action, confidence, vote_distribution = self._calculate_consensus(weighted_votes)
            
            # If no consensus or confidence too low, return None
            if not consensus_action or confidence < self.min_confidence:
                self.signals_filtered += 1
                self.logger.debug("No consensus or low confidence (%.2f). No signal generated.", 
                                confidence if confidence else 0)
                return None
            
            # Analyze market structure for optimal entry/exit points
            entry_price, stop_loss, take_profit = self._calculate_entry_exit_points(
                consensus_action, market_data, asset_config
            )
            
            # Calculate risk-reward ratio
            risk_reward = calculate_risk_reward(
                consensus_action, entry_price, stop_loss, take_profit
            )
            
            # Filter based on risk-reward
            if risk_reward < self.min_risk_reward:
                self.signals_filtered += 1
                self.logger.debug("Risk-reward ratio too low (%.2f). No signal generated.", risk_reward)
                return None
            
            # Create signal
            signal = {
                "symbol": asset_config["symbol"],
                "action": consensus_action,
                "confidence": confidence,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward": risk_reward,
                "timeframe": asset_config.get("timeframe", "1h"),
                "timestamp": datetime.utcnow().isoformat(),
                "vote_distribution": vote_distribution,
                "reasoning": self._generate_signal_reasoning(weighted_votes, consensus_action),
                "expected_duration": self._estimate_trade_duration(market_data, consensus_action),
                "market_context": self._extract_market_context(market_data),
                "volatility_context": self._analyze_volatility_context(market_data)
            }
            
            # Validate signal
            if not validate_signal(signal):
                raise InvalidSignalError(f"Generated signal failed validation: {signal}")
            
            # Record signal for history
            self._record_signal(signal)
            
            # Update statistics
            self.signals_generated += 1
            
            # Log signal generation
            self.logger.info(
                "Signal generated: %s %s at %.4f, SL: %.4f, TP: %.4f, Confidence: %.2f, R:R: %.2f",
                signal["symbol"], signal["action"], signal["entry_price"], 
                signal["stop_loss"], signal["take_profit"], signal["confidence"],
                signal["risk_reward"]
            )
            
            return signal
            
        except Exception as e:
            self.logger.error("Error generating signal: %s", str(e), exc_info=True)
            raise SignalGenerationError(f"Failed to generate signal: {str(e)}")
    
    def _calculate_consensus(self, weighted_votes: Dict[str, VotingResult]) -> Tuple[str, float, Dict[str, float]]:
        """
        Calculate the consensus action and confidence based on weighted votes.
        
        Args:
            weighted_votes: Weighted voting results from different councils
            
        Returns:
            Tuple of (consensus_action, confidence, vote_distribution)
        """
        # Initialize vote counters
        votes = {
            ACTION_BUY: 0.0,
            ACTION_SELL: 0.0,
            ACTION_HOLD: 0.0,
            ACTION_CLOSE: 0.0
        }
        
        # Combine votes from all councils
        total_weight = 0.0
        reasoning_data = {}
        
        for council_name, vote_result in weighted_votes.items():
            weight = vote_result.weight
            total_weight += weight
            
            # Add weighted votes
            for action, vote_value in vote_result.votes.items():
                votes[action] += vote_value * weight
            
            # Store reasoning data
            reasoning_data[council_name] = {
                "action": vote_result.recommended_action,
                "confidence": vote_result.confidence,
                "reasoning": vote_result.reasoning
            }
        
        # Normalize votes
        if total_weight > 0:
            for action in votes:
                votes[action] /= total_weight
        
        # Find action with highest vote
        max_action = max(votes, key=votes.get)
        max_vote = votes[max_action]
        
        # Check if we have a clear consensus
        if max_vote < self.min_council_agreement:
            return None, 0.0, votes
        
        # Calculate confidence
        confidence = calculate_confidence_score(votes, reasoning_data)
        
        return max_action, confidence, votes
    
    def _calculate_entry_exit_points(
        self, action: str, market_data: Dict[str, Any], 
        asset_config: Dict[str, Any]
    ) -> Tuple[float, float, float]:
        """
        Calculate optimal entry point, stop loss, and take profit levels.
        
        Args:
            action: The trade action (buy, sell)
            market_data: Current market data
            asset_config: Asset-specific configuration
            
        Returns:
            Tuple of (entry_price, stop_loss, take_profit)
        """
        # Get current price and recent price data
        current_price = market_data["last_price"]
        ohlcv_data = market_data.get("ohlcv", None)
        
        # Default values
        entry_price = current_price
        
        # Use market structure to identify support/resistance levels
        if ohlcv_data is not None:
            price_data = pd.DataFrame(ohlcv_data)
            levels = self.market_structure.identify_key_levels(price_data)
            volatility = self.volatility_analyzer.calculate_atr(price_data, 14)
        else:
            # Fallback to basic calculation if no OHLCV data
            volatility = current_price * 0.005  # Approximate 0.5% volatility
            levels = {
                "support": [current_price * 0.99, current_price * 0.98],
                "resistance": [current_price * 1.01, current_price * 1.02]
            }
        
        # Calculate stop loss and take profit based on action
        if action == ACTION_BUY:
            # For buy, stop loss below support, take profit at resistance
            stop_loss = self._find_optimal_stop_loss(
                current_price, levels["support"], volatility, is_buy=True
            )
            take_profit = self._find_optimal_take_profit(
                current_price, levels["resistance"], volatility, is_buy=True
            )
        elif action == ACTION_SELL:
            # For sell, stop loss above resistance, take profit at support
            stop_loss = self._find_optimal_stop_loss(
                current_price, levels["resistance"], volatility, is_buy=False
            )
            take_profit = self._find_optimal_take_profit(
                current_price, levels["support"], volatility, is_buy=False
            )
        else:
            # For hold/close, use default values
            stop_loss = current_price * 0.97 if action == ACTION_BUY else current_price * 1.03
            take_profit = current_price * 1.05 if action == ACTION_BUY else current_price * 0.95
        
        # Apply asset-specific adjustments
        stop_loss_multiplier = asset_config.get("stop_loss_multiplier", 1.0)
        take_profit_multiplier = asset_config.get("take_profit_multiplier", 1.0)
        
        if action == ACTION_BUY:
            stop_loss = current_price - ((current_price - stop_loss) * stop_loss_multiplier)
            take_profit = current_price + ((take_profit - current_price) * take_profit_multiplier)
        else:
            stop_loss = current_price + ((stop_loss - current_price) * stop_loss_multiplier)
            take_profit = current_price - ((current_price - take_profit) * take_profit_multiplier)
        
        return entry_price, stop_loss, take_profit
    
    def _find_optimal_stop_loss(
        self, current_price: float, levels: List[float], 
        volatility: float, is_buy: bool
    ) -> float:
        """
        Find the optimal stop loss level based on support/resistance levels.
        
        Args:
            current_price: Current market price
            levels: List of support or resistance levels
            volatility: Market volatility measure (e.g., ATR)
            is_buy: True if this is a buy signal, False for sell
            
        Returns:
            Optimal stop loss price
        """
        # Filter levels that are valid for stop loss
        if is_buy:
            # For buy, stop loss should be below current price
            valid_levels = [level for level in levels if level < current_price]
        else:
            # For sell, stop loss should be above current price
            valid_levels = [level for level in levels if level > current_price]
        
        if not valid_levels:
            # If no valid levels, use volatility-based stop loss
            if is_buy:
                return current_price - (volatility * 1.5)
            else:
                return current_price + (volatility * 1.5)
        
        # For buy orders, use the highest support below current price
        # For sell orders, use the lowest resistance above current price
        if is_buy:
            return max(valid_levels)
        else:
            return min(valid_levels)
    
    def _find_optimal_take_profit(
        self, current_price: float, levels: List[float], 
        volatility: float, is_buy: bool
    ) -> float:
        """
        Find the optimal take profit level based on support/resistance levels.
        
        Args:
            current_price: Current market price
            levels: List of support or resistance levels
            volatility: Market volatility measure (e.g., ATR)
            is_buy: True if this is a buy signal, False for sell
            
        Returns:
            Optimal take profit price
        """
        # Filter levels that are valid for take profit
        if is_buy:
            # For buy, take profit should be above current price
            valid_levels = [level for level in levels if level > current_price]
        else:
            # For sell, take profit should be below current price
            valid_levels = [level for level in levels if level < current_price]
        
        if not valid_levels:
            # If no valid levels, use volatility-based take profit
            if is_buy:
                return current_price + (volatility * 2.5)
            else:
                return current_price - (volatility * 2.5)
        
        # For buy orders, use the lowest resistance above current price
        # For sell orders, use the highest support below current price
        if is_buy:
            return min(valid_levels)
        else:
            return max(valid_levels)
    
    def _generate_signal_reasoning(
        self, weighted_votes: Dict[str, VotingResult], consensus_action: str
    ) -> str:
        """
        Generate reasoning text explaining why this signal was generated.
        
        Args:
            weighted_votes: Weighted voting results
            consensus_action: The consensus action
            
        Returns:
            Reasoning text
        """
        # Extract reasoning from councils that agreed with consensus
        supporting_councils = []
        
        for council_name, vote_result in weighted_votes.items():
            if vote_result.recommended_action == consensus_action:
                confidence = vote_result.confidence
                reason = vote_result.reasoning
                supporting_councils.append(f"{council_name} ({confidence:.2f}): {reason}")
        
        # Format the final reasoning text
        action_text = {
            ACTION_BUY: "buy",
            ACTION_SELL: "sell",
            ACTION_HOLD: "hold",
            ACTION_CLOSE: "close position"
        }.get(consensus_action, consensus_action)
        
        reasoning = f"Recommendation to {action_text} based on: "
        reasoning += " | ".join(supporting_councils)
        
        return reasoning
    
    def _estimate_trade_duration(
        self, market_data: Dict[str, Any], action: str
    ) -> Dict[str, Any]:
        """
        Estimate the expected duration of the trade.
        
        Args:
            market_data: Current market data
            action: The trade action
            
        Returns:
            Dictionary with expected duration information
        """
        # Extract trends and volatility
        timeframe = market_data.get("timeframe", "1h")
        volatility = market_data.get("volatility", {})
        trends = market_data.get("trends", {})
        
        # Default duration estimates by timeframe
        default_durations = {
            "1m": timedelta(minutes=5),
            "5m": timedelta(minutes=25),
            "15m": timedelta(hours=1, minutes=15),
            "30m": timedelta(hours=2, minutes=30),
            "1h": timedelta(hours=5),
            "4h": timedelta(hours=20),
            "1d": timedelta(days=5)
        }
        
        # Base duration on timeframe
        base_duration = default_durations.get(timeframe, timedelta(hours=5))
        
        # Adjust based on volatility and trend strength
        volatility_multiplier = 1.0
        if volatility:
            volatility_level = volatility.get("level", "medium")
            if volatility_level == "high":
                volatility_multiplier = 0.7  # High volatility = shorter duration
            elif volatility_level == "low":
                volatility_multiplier = 1.3  # Low volatility = longer duration
        
        trend_multiplier = 1.0
        if trends:
            trend_strength = trends.get("strength", 0.5)
            trend_direction = trends.get("direction", 0)
            
            # Strong trend in same direction = longer duration
            if (action == ACTION_BUY and trend_direction > 0) or \
               (action == ACTION_SELL and trend_direction < 0):
                trend_multiplier = 1.0 + (trend_strength * 0.5)
            # Strong trend in opposite direction = shorter duration
            else:
                trend_multiplier = 1.0 - (trend_strength * 0.3)
        
        # Calculate adjusted duration
        adjusted_duration = base_duration * volatility_multiplier * trend_multiplier
        
        # Convert to hours and minutes
        total_seconds = adjusted_duration.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        
        return {
            "hours": hours,
            "minutes": minutes,
            "timeframe": timeframe,
            "factors": {
                "volatility": volatility_multiplier,
                "trend": trend_multiplier
            }
        }
    
    def _extract_market_context(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant market context information for the signal.
        
        Args:
            market_data: Current market data
            
        Returns:
            Dictionary with market context information
        """
        return {
            "market_regime": market_data.get("market_regime", "unknown"),
            "trend_info": market_data.get("trends", {}),
            "liquidity": market_data.get("liquidity", {}),
            "correlation_status": market_data.get("correlations", {}),
            "sentiment": market_data.get("sentiment", {})
        }
    
    def _analyze_volatility_context(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze volatility context for the signal.
        
        Args:
            market_data: Current market data
            
        Returns:
            Dictionary with volatility context information
        """
        # Extract basic volatility data
        vol_data = market_data.get("volatility", {})
        
        # If we have OHLCV data, calculate additional metrics
        ohlcv_data = market_data.get("ohlcv", None)
        additional_metrics = {}
        
        if ohlcv_data is not None:
            price_data = pd.DataFrame(ohlcv_data)
            
            # Calculate various volatility metrics
            atr = self.volatility_analyzer.calculate_atr(price_data, 14)
            historical_vol = self.volatility_analyzer.calculate_historical_volatility(price_data, 20)
            vol_trend = self.volatility_analyzer.analyze_volatility_trend(price_data, [7, 14, 30])
            
            additional_metrics = {
                "atr": float(atr),
                "historical_volatility": float(historical_vol),
                "volatility_trend": vol_trend
            }
        
        # Combine with existing volatility data
        return {**vol_data, **additional_metrics}
    
    def _record_signal(self, signal: Dict[str, Any]) -> None:
        """
        Record a generated signal to the history.
        
        Args:
            signal: The generated signal
        """
        symbol = signal["symbol"]
        timeframe = signal["timeframe"]
        key = f"{symbol}_{timeframe}"
        
        # Add to in-memory history
        self.signal_history[key].append(signal)
        
        # Trim history if needed
        if len(self.signal_history[key]) > self.max_history_size:
            self.signal_history[key] = self.signal_history[key][-self.max_history_size:]
        
        # Store in database
        try:
            SignalRecord.create(
                symbol=symbol,
                timeframe=timeframe,
                action=signal["action"],
                confidence=signal["confidence"],
                entry_price=signal["entry_price"],
                stop_loss=signal["stop_loss"],
                take_profit=signal["take_profit"],
                risk_reward=signal["risk_reward"],
                reasoning=signal["reasoning"],
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            self.logger.error("Failed to store signal in database: %s", str(e), exc_info=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the signal generator performance.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_signals_generated": self.signals_generated,
            "signals_filtered": self.signals_filtered,
            "percentage_filtered": (self.signals_filtered / max(1, self.signals_generated + self.signals_filtered)) * 100,
            "signal_history_count": {k: len(v) for k, v in self.signal_history.items()},
            "weighting_stats": self.weighting_system.get_statistics(),
            "performance_stats": self.performance_tracker.get_statistics()
        }
    
    def reset_statistics(self) -> None:
        """Reset the signal generator statistics."""
        self.signals_generated = 0
        self.signals_filtered = 0
        self.weighting_system.reset_statistics()
