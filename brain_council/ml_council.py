#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
ML Model Council

This module implements a specialized council for ML models to better integrate
machine learning predictions into the trading decision process.
"""

import time
import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set

from common.logger import get_logger
from common.metrics import MetricsCollector
from common.exceptions import (
    ConfigurationError, ServiceStartupError, ServiceShutdownError,
    StrategyError, SignalGenerationError, ModelNotFoundError
)
from common.constants import SIGNAL_TYPES, POSITION_DIRECTION, MARKET_REGIMES
from .voting_system import VotingSystem, VotingResult

class MLCouncil:
    """
    Specialized council for integrating ML model predictions.
    
    This council aggregates predictions from various ML models and generates
    consolidated signals that can be used by asset councils or the master council.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 parent_council=None,
                 redis_client=None, 
                 db_client=None):
        """
        Initialize the ML Council.
        
        Args:
            config: Configuration dictionary
            parent_council: Reference to the parent council
            redis_client: Redis client for communication
            db_client: Database client
        """
        self.config = config
        self.parent_council = parent_council
        self.redis_client = redis_client
        self.db_client = db_client
        self.logger = get_logger("MLCouncil")
        self.metrics = MetricsCollector("ml_council")
        
        # Initialize voting system
        voting_config = config.get("ml_council", {}).get("voting_system", {})
        self.voting_system = VotingSystem({"voting_system": voting_config})
        
        # Track registered ML models
        self.registered_models = {}
        
        # Track model performance
        self.model_performance = {}
        
        # Asset-specific ML models
        self.asset_models = {}
        
        # Model type weights
        self.model_type_weights = config.get("ml_council", {}).get("model_type_weights", {
            "classification": 1.0,
            "regression": 0.9,
            "time_series": 1.1,
            "deep_learning": 1.2,
            "reinforcement_learning": 1.3,
            "ensemble": 1.4
        })
        
        # Default weight for unknown model types
        self.default_model_weight = 0.8
        
        self.logger.info("ML Council initialized")
    
    async def register_model(self, model_name: str, model_type: str, asset_ids: List[str] = None):
        """
        Register an ML model with the council.
        
        Args:
            model_name: Name of the ML model
            model_type: Type of the ML model
            asset_ids: List of asset IDs this model specializes in (None for all assets)
        """
        self.registered_models[model_name] = {
            "type": model_type,
            "registered_at": time.time(),
            "asset_ids": asset_ids,
            "last_prediction": None,
            "prediction_count": 0,
            "success_count": 0
        }
        
        # Register with asset-specific tracking
        if asset_ids:
            for asset_id in asset_ids:
                if asset_id not in self.asset_models:
                    self.asset_models[asset_id] = {}
                self.asset_models[asset_id][model_name] = {
                    "type": model_type,
                    "registered_at": time.time()
                }
        
        self.logger.info(f"Registered ML model: {model_name} ({model_type})")
        
        # Initialize performance metrics
        self.model_performance[model_name] = {
            "accuracy": 0.5,  # Default starting accuracy
            "precision": 0.5,
            "recall": 0.5,
            "f1_score": 0.5,
            "profit_factor": 1.0,
            "sharpe_ratio": 0.0,
            "last_updated": time.time()
        }
    
    async def process_predictions(self, 
                                 predictions: List[Dict[str, Any]], 
                                 asset_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process ML model predictions and generate consolidated signals.
        
        Args:
            predictions: List of predictions from various ML models
            asset_id: Optional asset ID to filter predictions
            
        Returns:
            Dictionary of consolidated signals by asset and timeframe
        """
        if not predictions:
            return {}
            
        self.logger.debug(f"Processing {len(predictions)} ML predictions")
        
        # Group predictions by asset and timeframe
        grouped_predictions = self._group_predictions(predictions, asset_id)
        
        # Process each group and generate signals
        consolidated_signals = {}
        
        for group_key, group_predictions in grouped_predictions.items():
            parts = group_key.split(":")
            if len(parts) != 2:
                continue
                
            current_asset_id, timeframe = parts
            
            # Generate signal for this group
            signal = await self._generate_signal_from_predictions(
                group_predictions, current_asset_id, timeframe
            )
            
            if signal:
                if current_asset_id not in consolidated_signals:
                    consolidated_signals[current_asset_id] = {}
                    
                consolidated_signals[current_asset_id][timeframe] = signal
        
        return consolidated_signals
    
    def _group_predictions(self, 
                          predictions: List[Dict[str, Any]], 
                          asset_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group predictions by asset and timeframe.
        
        Args:
            predictions: List of predictions
            asset_id: Optional asset ID to filter predictions
            
        Returns:
            Dictionary mapping group keys to prediction lists
        """
        result = {}
        
        for prediction in predictions:
            # Skip if asset_id filter is provided and doesn't match
            pred_asset_id = prediction.get("asset_id")
            if asset_id and pred_asset_id != asset_id:
                continue
                
            # Skip invalid predictions
            if not pred_asset_id or "timeframe" not in prediction:
                continue
                
            timeframe = prediction.get("timeframe")
            group_key = f"{pred_asset_id}:{timeframe}"
            
            if group_key not in result:
                result[group_key] = []
                
            result[group_key].append(prediction)
        
        return result
    
    async def _generate_signal_from_predictions(self, 
                                              predictions: List[Dict[str, Any]],
                                              asset_id: str,
                                              timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Generate a consolidated signal from a group of predictions.
        
        Args:
            predictions: List of predictions for the same asset and timeframe
            asset_id: Asset ID
            timeframe: Timeframe
            
        Returns:
            Signal dictionary or None if no signal could be generated
        """
        if not predictions:
            return None
            
        # Prepare votes for voting system
        votes = {}
        weights = {}
        
        for prediction in predictions:
            model_name = prediction.get("model_name")
            if not model_name:
                continue
                
            # Get prediction direction and confidence
            direction = prediction.get("direction", "hold")
            confidence = prediction.get("confidence", 0.5)
            
            # Create vote
            votes[model_name] = {
                "direction": direction,
                "confidence": confidence,
                "timestamp": prediction.get("timestamp", time.time()),
                "signals": {},
                "metadata": prediction.get("metadata", {})
            }
            
            # Determine weight based on model type and performance
            model_info = self.registered_models.get(model_name, {})
            model_type = model_info.get("type", "unknown")
            
            # Base weight from model type
            base_weight = self.model_type_weights.get(model_type, self.default_model_weight)
            
            # Adjust weight based on historical performance
            performance = self.model_performance.get(model_name, {})
            accuracy = performance.get("accuracy", 0.5)
            
            # Performance factor (1.0 for average performance, up to 1.5 for excellent)
            performance_factor = 0.5 + accuracy
            
            # Final weight
            weights[model_name] = base_weight * performance_factor
        
        # Generate decision using voting system
        if votes:
            # Add contextual information
            context = {
                "asset_id": asset_id,
                "timeframe": timeframe,
                "prediction_count": len(predictions),
                "timestamp": time.time()
            }
            
            # Get decision from voting system
            decision = await self.voting_system.generate_decision(votes, weights, context)
            
            # Convert decision to signal format
            signal = {
                "source": "ml_council",
                "asset_id": asset_id,
                "timeframe": timeframe,
                "direction": decision.get("direction", "hold"),
                "confidence": decision.get("confidence", 0.0),
                "timestamp": time.time(),
                "metadata": {
                    "model_count": len(votes),
                    "voting_method": self.voting_system.primary_method,
                    "agreement_level": decision.get("agreement_level", 0.0)
                }
            }
            
            return signal
        
        return None
    
    async def update_model_performance(self, model_name: str, metrics: Dict[str, float]):
        """
        Update performance metrics for an ML model.
        
        Args:
            model_name: Name of the ML model
            metrics: Performance metrics dictionary
        """
        if model_name in self.model_performance:
            self.model_performance[model_name].update(metrics)
            self.model_performance[model_name]["last_updated"] = time.time()
            self.logger.debug(f"Updated performance metrics for model {model_name}")
    
    async def get_best_models_for_asset(self, asset_id: str, limit: int = 3) -> List[str]:
        """
        Get the best performing ML models for a specific asset.
        
        Args:
            asset_id: Asset ID
            limit: Maximum number of models to return
            
        Returns:
            List of model names
        """
        if asset_id not in self.asset_models:
            return []
            
        # Get models for this asset
        asset_model_names = list(self.asset_models[asset_id].keys())
        
        # Sort by performance
        sorted_models = sorted(
            asset_model_names,
            key=lambda m: self.model_performance.get(m, {}).get("accuracy", 0),
            reverse=True
        )
        
        return sorted_models[:limit]
    
    async def request_predictions(self, asset_id: str, timeframe: str) -> List[Dict[str, Any]]:
        """
        Request predictions from ML models for a specific asset and timeframe.
        
        Args:
            asset_id: Asset ID
            timeframe: Timeframe
            
        Returns:
            List of predictions
        """
        if not self.redis_client:
            self.logger.warning("Redis client not available for ML prediction requests")
            return []
            
        # Get best models for this asset
        best_models = await self.get_best_models_for_asset(asset_id)
        
        if not best_models:
            # If no asset-specific models, get general models
            best_models = [
                model_name for model_name, info in self.registered_models.items()
                if not info.get("asset_ids") or asset_id in info.get("asset_ids", [])
            ]
        
        if not best_models:
            return []
            
        # Create prediction request
        request_id = f"ml_pred_{asset_id}_{timeframe}_{int(time.time())}"
        response_channel = f"ml_prediction_response.{request_id}"
        
        request = {
            "id": request_id,
            "response_channel": response_channel,
            "asset_id": asset_id,
            "timeframe": timeframe,
            "models": best_models,
            "timestamp": time.time()
        }
        
        # Send request
        try:
            await self.redis_client.publish("ml_prediction_request", request)
            
            # Wait for response with timeout
            timeout = 10  # seconds
            response = await self.redis_client.subscribe_and_wait(
                response_channel, timeout=timeout
            )
            
            if response and "predictions" in response:
                return response["predictions"]
                
        except Exception as e:
            self.logger.error(f"Error requesting ML predictions: {str(e)}")
            
        return []