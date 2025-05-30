#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Master Council Module

This module implements the MasterCouncil class, which serves as the final decision-making
authority coordinating signals from all specialized councils to generate high-confidence
trading decisions.
"""

import asyncio
import time
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from concurrent.futures import ThreadPoolExecutor

from common.logger import get_logger
from common.exceptions import CouncilError, DecisionError
from common.utils import sigmoid
from common.constants import (
    VOTE_THRESHOLDS, COUNCIL_WEIGHTS, ASSET_CLASSES, 
    SIGNAL_TYPES, TIMEFRAMES, CONFIDENCE_LEVELS
)
from brain_council.base_council import BaseCouncil
from brain_council.timeframe_council import TimeframeCouncil
from brain_council.asset_council import AssetCouncil
from brain_council.regime_council import RegimeCouncil
from brain_council.voting_system import VotingSystem
from feature_service.features.market_structure import MarketStructure
from data_storage.models.strategy_data import StrategyPerformance

class MasterCouncil(BaseCouncil):
    """
    Master Council coordinates all specialized councils and makes final trading decisions.
    It dynamically weights contributions from various councils based on their historical 
    performance, market conditions, and confidence levels to generate high-probability
    trading signals with consistently strong success rates.
    """
    
    def __init__(self, config: Dict[str, Any], debug_mode: bool = False):
        """
        Initialize the Master Council with all specialized councils.
        
        Args:
            config: Configuration dictionary for the council
            debug_mode: Whether to enable extended debug logging
        """
        super().__init__(config, "master_council")
        self.logger = get_logger("brain_council.master_council")
        self.debug_mode = debug_mode
        self.performance_history = {}
        self.council_weights = {}
        self.current_market_regime = None
        self.active_signals = {}
        self.voting_system = VotingSystem(config)
        
        # Initialize specialized councils
        self.timeframe_councils = {}
        self.asset_councils = {}
        self.regime_council = RegimeCouncil(config)
        
        # Initialize councils for each timeframe
        for timeframe in config.get("timeframes", TIMEFRAMES):
            self.timeframe_councils[timeframe] = TimeframeCouncil(
                config, 
                timeframe=timeframe
            )
        
        # Initialize councils for each asset class
        for asset_class in config.get("asset_classes", ASSET_CLASSES):
            self.asset_councils[asset_class] = AssetCouncil(
                config, 
                asset_class=asset_class
            )
        
        # Performance tracking
        self.recent_decisions = []
        self.performance_metrics = {}
        self.decision_history = {}
        
        self.logger.info(f"Master Council initialized with {len(self.timeframe_councils)} "
                      f"timeframe councils and {len(self.asset_councils)} asset councils")
        
        # Dynamic weighting parameters
        self.weight_adjustment_rate = config.get("weight_adjustment_rate", 0.05)
        self.performance_memory_length = config.get("performance_memory_length", 100)
        self.min_council_weight = config.get("min_council_weight", 0.1)
        
        # Load initial weights from performance data if available
        self._load_initial_weights()
    
    def _load_initial_weights(self) -> None:
        """
        Load initial council weights from historical performance data.
        """
        try:
            # Get performance data from database
            performance_data = StrategyPerformance.get_all_council_performance()
            
            # Process performance data to initialize weights
            for council_type, performance in performance_data.items():
                if council_type in self.council_weights:
                    # Calculate weighted win rate based on recency
                    weights = np.exp(np.linspace(0, 1, len(performance['win_history'])))
                    weights = weights / np.sum(weights)
                    
                    weighted_win_rate = np.sum(
                        np.array(performance['win_history']) * weights
                    )
                    
                    # Set initial weight based on weighted win rate
                    self.council_weights[council_type] = max(
                        weighted_win_rate, 
                        self.min_council_weight
                    )
            
            self.logger.info(f"Loaded initial council weights: {self.council_weights}")
            
        except Exception as e:
            self.logger.warning(f"Could not load initial weights: {str(e)}. "
                            f"Using default weights.")
            # Use default weights from constants
            self.council_weights = COUNCIL_WEIGHTS.copy()
    
    async def update_market_regime(self) -> None:
        """
        Update the current market regime assessment.
        """
        try:
            self.current_market_regime = await self.regime_council.analyze_market_regime()
            self.logger.info(f"Updated market regime: {self.current_market_regime}")
        except Exception as e:
            self.logger.error(f"Failed to update market regime: {str(e)}")
            # Keep the previous regime if update fails
    
    async def analyze_asset(self, asset: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyze a specific asset in a specific timeframe using appropriate councils.
        
        Args:
            asset: Asset identifier
            timeframe: Timeframe for analysis
            
        Returns:
            Dictionary containing the analysis results
        """
        results = {}
        
        # Get asset class for this asset
        asset_class = self._determine_asset_class(asset)
        
        # Get timeframe council analysis
        tf_result = await self.timeframe_councils[timeframe].analyze(asset)
        
        # Get asset council analysis
        asset_result = await self.asset_councils[asset_class].analyze(asset, timeframe)
        
        # Get regime-specific analysis
        if self.current_market_regime:
            regime_result = await self.regime_council.analyze(
                asset, 
                timeframe, 
                self.current_market_regime
            )
        else:
            # Update market regime if it's not available
            await self.update_market_regime()
            regime_result = await self.regime_council.analyze(
                asset, 
                timeframe, 
                self.current_market_regime
            )
        
        # Combine all analyses
        results = {
            "timeframe_analysis": tf_result,
            "asset_analysis": asset_result,
            "regime_analysis": regime_result,
            "timestamp": time.time()
        }
        
        return results
    
    def _determine_asset_class(self, asset: str) -> str:
        """
        Determine the asset class for a given asset.
        
        Args:
            asset: Asset identifier
            
        Returns:
            Asset class string
        """
        # This could be expanded with more sophisticated asset classification
        if '_' in asset:
            parts = asset.split('_')
            if parts[0].lower() in ('btc', 'eth', 'xrp'):
                return 'crypto'
        
        # Default classifications based on common patterns
        if asset.endswith('USD') or asset.endswith('USDT'):
            return 'crypto'
        if 'JPY' in asset or 'EUR' in asset or 'GBP' in asset:
            return 'forex'
        if asset in ('GOLD', 'SILVER', 'OIL'):
            return 'commodities'
        if '.' in asset:
            return 'stocks'
            
        # Default fallback
        return 'other'
    
    async def generate_signal(self, asset: str, timeframe: str) -> Dict[str, Any]:
        """
        Generate a trading signal for a specific asset and timeframe.
        
        Args:
            asset: Asset identifier
            timeframe: Timeframe for signal generation
            
        Returns:
            Signal dictionary with direction, confidence, and supporting data
        """
        try:
            self.logger.info(f"Generating signal for {asset} on {timeframe} timeframe")
            
            # Ensure we have the current market regime
            if not self.current_market_regime:
                await self.update_market_regime()
            
            # Get analysis from all councils
            analysis = await self.analyze_asset(asset, timeframe)
            
            # Get votes from each council
            timeframe_vote = analysis["timeframe_analysis"].get("vote", {})
            asset_vote = analysis["asset_analysis"].get("vote", {})
            regime_vote = analysis["regime_analysis"].get("vote", {})
            
            # Prepare votes for the voting system
            votes = {
                "timeframe_council": timeframe_vote,
                "asset_council": asset_vote,
                "regime_council": regime_vote
            }
            
            # Get current weights for each council
            weights = {
                "timeframe_council": self.council_weights.get(
                    f"timeframe_{timeframe}", 
                    COUNCIL_WEIGHTS.get("timeframe_council", 1.0)
                ),
                "asset_council": self.council_weights.get(
                    f"asset_{self._determine_asset_class(asset)}", 
                    COUNCIL_WEIGHTS.get("asset_council", 1.0)
                ),
                "regime_council": self.council_weights.get(
                    f"regime_{self.current_market_regime}", 
                    COUNCIL_WEIGHTS.get("regime_council", 1.0)
                )
            }
            
            # Generate final signal using voting system
            signal = await self.voting_system.generate_decision(votes, weights)
            
            # Enrich signal with additional metadata
            signal.update({
                "asset": asset,
                "timeframe": timeframe,
                "timestamp": time.time(),
                "market_regime": self.current_market_regime,
                "council_weights": weights,
                "analysis_summary": {
                    "timeframe": analysis["timeframe_analysis"].get("summary", {}),
                    "asset": analysis["asset_analysis"].get("summary", {}),
                    "regime": analysis["regime_analysis"].get("summary", {})
                }
            })
            
            # Store signal in active signals
            self._register_active_signal(signal)
            
            self.logger.info(
                f"Generated signal for {asset} on {timeframe}: "
                f"{signal.get('direction')} with "
                f"{signal.get('confidence'):.2f} confidence"
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {asset} on {timeframe}: {str(e)}")
            raise DecisionError(f"Failed to generate signal: {str(e)}")
    
    def _register_active_signal(self, signal: Dict[str, Any]) -> None:
        """
        Register a new active signal for tracking.
        
        Args:
            signal: Signal dictionary to register
        """
        asset = signal.get("asset")
        timeframe = signal.get("timeframe")
        
        if asset not in self.active_signals:
            self.active_signals[asset] = {}
        
        # Store the signal
        self.active_signals[asset][timeframe] = signal
        
        # Add to recent decisions for performance tracking
        self.recent_decisions.append({
            "signal": signal,
            "timestamp": time.time(),
            "result": None,  # To be filled when outcome is known
            "pnl": None      # To be filled when trade is closed
        })
        
        # Trim recent decisions to maintain performance_memory_length
        if len(self.recent_decisions) > self.performance_memory_length:
            self.recent_decisions.pop(0)
    
    async def update_signal_performance(
        self, 
        asset: str, 
        timeframe: str, 
        result: bool, 
        pnl: float
    ) -> None:
        """
        Update the performance record for a previously generated signal.
        
        Args:
            asset: Asset identifier
            timeframe: Timeframe of the signal
            result: Whether the signal was successful (True/False)
            pnl: Profit/loss from the trade
        """
        # Find the signal in recent decisions
        for decision in reversed(self.recent_decisions):
            signal = decision.get("signal", {})
            if (signal.get("asset") == asset and 
                signal.get("timeframe") == timeframe and
                decision.get("result") is None):  # Not yet evaluated
                
                # Update the decision record
                decision["result"] = result
                decision["pnl"] = pnl
                decision["evaluation_time"] = time.time()
                
                # Update council weights based on this result
                await self._update_council_weights(signal, result, pnl)
                
                self.logger.info(
                    f"Updated signal performance for {asset} on {timeframe}: "
                    f"{'Success' if result else 'Failure'} with PnL: {pnl}"
                )
                break
    
    async def _update_council_weights(
        self, 
        signal: Dict[str, Any], 
        result: bool, 
        pnl: float
    ) -> None:
        """
        Update the weights of councils based on signal performance.
        
        Args:
            signal: The signal that was generated
            result: Whether the signal was successful
            pnl: Profit/loss from the trade
        """
        # Get the councils that contributed to this signal
        timeframe = signal.get("timeframe")
        asset_class = self._determine_asset_class(signal.get("asset"))
        regime = signal.get("market_regime")
        
        # Update weights based on performance
        adjustment = self.weight_adjustment_rate
        if not result:
            adjustment = -adjustment  # Decrease weight for incorrect predictions
        
        # Adjust weight based on PnL magnitude (higher PnL = more weight increase)
        if result and pnl > 0:
            pnl_factor = min(abs(pnl) / 100.0, 1.0)  # Normalize to max of 1
            adjustment *= (1 + pnl_factor)
        
        # Update timeframe council weight
        tf_key = f"timeframe_{timeframe}"
        if tf_key not in self.council_weights:
            self.council_weights[tf_key] = COUNCIL_WEIGHTS.get("timeframe_council", 1.0)
        self.council_weights[tf_key] = max(
            self.council_weights[tf_key] + adjustment,
            self.min_council_weight
        )
        
        # Update asset council weight
        asset_key = f"asset_{asset_class}"
        if asset_key not in self.council_weights:
            self.council_weights[asset_key] = COUNCIL_WEIGHTS.get("asset_council", 1.0)
        self.council_weights[asset_key] = max(
            self.council_weights[asset_key] + adjustment,
            self.min_council_weight
        )
        
        # Update regime council weight
        if regime:
            regime_key = f"regime_{regime}"
            if regime_key not in self.council_weights:
                self.council_weights[regime_key] = COUNCIL_WEIGHTS.get("regime_council", 1.0)
            self.council_weights[regime_key] = max(
                self.council_weights[regime_key] + adjustment,
                self.min_council_weight
            )
        
        # Save updated weights to database for persistence
        try:
            await self._save_council_weights()
        except Exception as e:
            self.logger.error(f"Failed to save council weights: {str(e)}")
    
    async def _save_council_weights(self) -> None:
        """
        Save the current council weights to the database.
        """
        if not hasattr(self, "_db"):
            from common.db_client import get_db_client

            self._db = await get_db_client()
            await self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS council_weights (
                    timestamp REAL,
                    weights TEXT
                )
                """
            )
            await self._db.commit()

        await self._db.execute(
            "INSERT INTO council_weights (timestamp, weights) VALUES (?, ?)",
            (time.time(), json.dumps(self.council_weights)),
        )
        await self._db.commit()
    
    async def get_council_performance(self) -> Dict[str, Any]:
        """
        Get the performance metrics for all councils.
        
        Returns:
            Dictionary with performance metrics
        """
        performance = {}
        
        # Calculate performance metrics from recent decisions
        for council_type, weight in self.council_weights.items():
            relevant_decisions = []
            
            # Filter decisions relevant to this council
            for decision in self.recent_decisions:
                signal = decision.get("signal", {})
                result = decision.get("result")
                
                if result is not None:  # Only count evaluated decisions
                    if council_type.startswith("timeframe_") and signal.get("timeframe") in council_type:
                        relevant_decisions.append(decision)
                    elif council_type.startswith("asset_") and self._determine_asset_class(signal.get("asset")) in council_type:
                        relevant_decisions.append(decision)
                    elif council_type.startswith("regime_") and signal.get("market_regime") in council_type:
                        relevant_decisions.append(decision)
            
            # Calculate win rate and other metrics
            if relevant_decisions:
                wins = sum(1 for d in relevant_decisions if d.get("result", False))
                total = len(relevant_decisions)
                win_rate = wins / total if total > 0 else 0
                
                # Calculate average PnL
                pnl_values = [d.get("pnl", 0) for d in relevant_decisions if d.get("pnl") is not None]
                avg_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else 0
                
                performance[council_type] = {
                    "win_rate": win_rate,
                    "sample_size": total,
                    "avg_pnl": avg_pnl,
                    "current_weight": weight
                }
        
        return performance
    
    async def multi_timeframe_consensus(self, asset: str, timeframes: List[str]) -> Dict[str, Any]:
        """
        Generate a consensus signal across multiple timeframes.
        
        Args:
            asset: Asset identifier
            timeframes: List of timeframes to analyze
            
        Returns:
            Consensus signal with combined confidence
        """
        signals = {}
        
        # Generate signals for each timeframe
        for tf in timeframes:
            signals[tf] = await self.generate_signal(asset, tf)
        
        # Calculate consensus
        buy_votes = 0
        sell_votes = 0
        hold_votes = 0
        total_confidence = 0
        
        for tf, signal in signals.items():
            direction = signal.get("direction", "hold")
            confidence = signal.get("confidence", 0)
            
            if direction == "buy":
                buy_votes += 1
                total_confidence += confidence
            elif direction == "sell":
                sell_votes += 1
                total_confidence += confidence
            else:
                hold_votes += 1
        
        # Determine consensus direction
        if buy_votes > sell_votes and buy_votes > hold_votes:
            consensus_direction = "buy"
            confirming_votes = buy_votes
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            consensus_direction = "sell"
            confirming_votes = sell_votes
        else:
            consensus_direction = "hold"
            confirming_votes = hold_votes
        
        # Calculate consensus confidence
        if confirming_votes > 0:
            avg_confidence = total_confidence / len(timeframes)
            consensus_confidence = avg_confidence * (confirming_votes / len(timeframes))
        else:
            consensus_confidence = 0
        
        # Create consensus signal
        consensus_signal = {
            "asset": asset,
            "direction": consensus_direction,
            "confidence": consensus_confidence,
            "timeframe": "multi",
            "constituent_signals": signals,
            "timestamp": time.time(),
            "timeframe_agreement": confirming_votes / len(timeframes)
        }
        
        return consensus_signal
    
    async def process_trade_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a trade request from the execution engine.
        
        Args:
            request: Trade request dictionary
            
        Returns:
            Trade decision with execution parameters
        """
        asset = request.get("asset")
        timeframe = request.get("timeframe", "multi")
        
        # For multi-timeframe analysis
        if timeframe == "multi":
            timeframes = request.get("timeframes", TIMEFRAMES)
            signal = await self.multi_timeframe_consensus(asset, timeframes)
        else:
            signal = await self.generate_signal(asset, timeframe)
        
        # Enrich signal with execution parameters
        risk_params = await self._calculate_risk_parameters(asset, signal)
        
        trade_decision = {
            **signal,
            "risk_params": risk_params,
            "request_id": request.get("request_id"),
            "timestamp": time.time()
        }
        
        self.logger.info(f"Processed trade request for {asset}: {signal.get('direction')} "
                      f"with {signal.get('confidence'):.2f} confidence")
        
        return trade_decision
    
    async def _calculate_risk_parameters(self, asset: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate appropriate risk parameters based on the signal and market conditions.
        
        Args:
            asset: Asset identifier
            signal: Signal dictionary
            
        Returns:
            Risk parameters for trade execution
        """
        # Base risk on signal confidence and market regime
        confidence = signal.get("confidence", 0)
        regime = self.current_market_regime
        
        # Default parameters
        params = {
            "position_size_factor": 0.02,  # Base position size as % of capital
            "stop_loss_pips": 20,
            "take_profit_pips": 40,
            "trailing_stop": False,
            "entry_type": "market",
            "allow_partial_fill": True
        }
        
        # Adjust based on confidence
        if confidence > 0.8:
            params["position_size_factor"] = 0.03
            params["take_profit_pips"] = 50
            params["trailing_stop"] = True
        elif confidence < 0.6:
            params["position_size_factor"] = 0.01
            params["take_profit_pips"] = 30
        
        # Adjust based on market regime
        if regime == "high_volatility":
            params["stop_loss_pips"] = int(params["stop_loss_pips"] * 1.5)
            params["take_profit_pips"] = int(params["take_profit_pips"] * 1.5)
            params["position_size_factor"] *= 0.8  # Reduce size in volatile markets
        elif regime == "trending":
            params["trailing_stop"] = True
            params["take_profit_pips"] = int(params["take_profit_pips"] * 1.2)
        elif regime == "ranging":
            params["stop_loss_pips"] = int(params["stop_loss_pips"] * 0.8)
            params["take_profit_pips"] = int(params["take_profit_pips"] * 0.8)
        
        # Add market context information
        params["market_context"] = {
            "regime": regime,
            "volatility_level": signal.get("analysis_summary", {}).get("regime", {}).get("volatility_level"),
            "trend_strength": signal.get("analysis_summary", {}).get("regime", {}).get("trend_strength")
        }
        
        return params
    
    async def process_bulk_analysis(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a bulk analysis request for multiple assets and timeframes.
        
        Args:
            request: Analysis request dictionary
            
        Returns:
            Dictionary with analysis results for all requested assets/timeframes
        """
        assets = request.get("assets", [])
        timeframes = request.get("timeframes", [])
        results = {}
        
        # Process each asset
        for asset in assets:
            results[asset] = {}
            
            # Process each timeframe
            for tf in timeframes:
                signal = await self.generate_signal(asset, tf)
                results[asset][tf] = signal
            
            # If multiple timeframes, also generate a consensus
            if len(timeframes) > 1:
                consensus = await self.multi_timeframe_consensus(asset, timeframes)
                results[asset]["consensus"] = consensus
        
        return {
            "analysis_results": results,
            "request_id": request.get("request_id"),
            "timestamp": time.time()
        }
    
    async def get_active_signals(self, asset: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current active signals, optionally filtered by asset.
        
        Args:
            asset: Optional asset to filter by
            
        Returns:
            Dictionary of active signals
        """
        if asset:
            return {asset: self.active_signals.get(asset, {})}
        return self.active_signals
    
    async def run(self) -> None:
        """
        Main run loop for the Master Council.
        """
        self.logger.info("Master Council run loop starting")
        try:
            while True:
                # Update market regime periodically
                await self.update_market_regime()
                
                # Process any pending operations
                await self.process_pending_operations()
                
                # Sleep to prevent CPU overuse
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            self.logger.info("Master Council run loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in Master Council run loop: {str(e)}")
            raise
    
    async def process_pending_operations(self) -> None:
        """
        Process any pending operations in the council.
        """
        try:
            now = time.time()
            max_age = self.config.get("max_signal_age", 3600)

            assets_to_remove: List[str] = []
            for asset, tf_map in list(self.active_signals.items()):
                for tf, signal in list(tf_map.items()):
                    if now - signal.get("timestamp", now) > max_age:
                        del tf_map[tf]
                if not tf_map:
                    assets_to_remove.append(asset)

            for asset in assets_to_remove:
                del self.active_signals[asset]
        except Exception as e:
            self.logger.error(f"Error processing pending operations: {str(e)}")
