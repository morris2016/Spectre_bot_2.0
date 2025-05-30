#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Brain Council Manager

This module implements the Council Manager that coordinates the hierarchy of
specialized councils (asset councils, ML council, regime councils, etc.)
"""

import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set

from common.logger import get_logger
from common.metrics import MetricsCollector
from common.exceptions import (
    ConfigurationError, ServiceStartupError, ServiceShutdownError,
    StrategyError, SignalGenerationError
)
from common.constants import SIGNAL_TYPES, POSITION_DIRECTION, MARKET_REGIMES
from .voting_system import VotingSystem, VotingResult
from .asset_council import AssetCouncil
from .ml_council import MLCouncil

class CouncilManager:
    """
    Manager for coordinating the hierarchy of specialized councils.
    
    This manager creates and coordinates asset-specific councils, the ML council,
    and other specialized councils, facilitating communication between them.
    """
    
    def __init__(self, config: Dict[str, Any], redis_client=None, db_client=None):
        """
        Initialize the Council Manager.
        
        Args:
            config: Configuration dictionary
            redis_client: Redis client for communication
            db_client: Database client
        """
        self.config = config
        self.redis_client = redis_client
        self.db_client = db_client
        self.logger = get_logger("CouncilManager")
        self.metrics = MetricsCollector("council_manager")
        
        # Initialize asset councils
        self.asset_councils = {}
        
        # Initialize ML council
        self.ml_council = MLCouncil(config, parent_council=None, 
                                   redis_client=redis_client, db_client=db_client)
        
        # Initialize master voting system
        voting_config = config.get("council_manager", {}).get("voting_system", {})
        self.voting_system = VotingSystem({"voting_system": voting_config})
        
        # Track active assets
        self.active_assets = set()
        
        # Track market regimes by asset
        self.asset_regimes = {}
        
        # Council weights for master decisions
        self.council_weights = config.get("council_manager", {}).get("council_weights", {
            "asset_council": 1.0,
            "ml_council": 0.8,
            "regime_council": 0.7,
            "timeframe_council": 0.6
        })
        
        self.logger.info("Council Manager initialized")
    
    async def initialize(self):
        """Initialize the council manager and its sub-councils."""
        
        self.logger.info("Initializing Council Manager")
        
        # Get list of assets to monitor
        assets_config = self.config.get("council_manager", {}).get("assets", [])
        
        # If no specific assets configured, use default list
        if not assets_config:
            assets_config = self.config.get("data_feeds", {}).get("binance", {}).get("symbols", [])
        
        # Create asset councils for each asset
        for asset_id in assets_config:
            await self.create_asset_council(asset_id)
            
        self.logger.info(f"Initialized {len(self.asset_councils)} asset councils")
    
    async def create_asset_council(self, asset_id: str) -> AssetCouncil:
        """
        Create an asset-specific council.
        
        Args:
            asset_id: Asset identifier
            
        Returns:
            Newly created AssetCouncil instance
        """
        if asset_id in self.asset_councils:
            return self.asset_councils[asset_id]
            
        # Create new asset council
        council = AssetCouncil(
            asset_id=asset_id,
            config=self.config,
            parent_council=self,
            redis_client=self.redis_client,
            db_client=self.db_client
        )
        
        self.asset_councils[asset_id] = council
        self.active_assets.add(asset_id)
        
        # Initialize market regime for this asset
        self.asset_regimes[asset_id] = MARKET_REGIMES["RANGING"]
        
        self.logger.info(f"Created asset council for {asset_id}")
        return council
    
    async def process_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Process signals and generate decisions using the council hierarchy.
        
        Args:
            signals: List of signals from various sources
            
        Returns:
            Dictionary mapping asset IDs to decision dictionaries
        """
        if not signals:
            return {}
            
        self.logger.debug(f"Processing {len(signals)} signals")
        
        # Group signals by asset
        asset_signals = self._group_by_asset(signals)
        
        # Process ML model predictions separately
        ml_signals = [s for s in signals if s.get("source", "").startswith("ml_")]
        if ml_signals:
            ml_decisions = await self.ml_council.process_predictions(ml_signals)
            
            # Merge ML decisions into asset signals
            for asset_id, timeframe_decisions in ml_decisions.items():
                if asset_id not in asset_signals:
                    asset_signals[asset_id] = []
                    
                for timeframe, decision in timeframe_decisions.items():
                    asset_signals[asset_id].append(decision)
        
        # Process each asset's signals through its council
        asset_decisions = {}
        
        for asset_id, asset_signal_list in asset_signals.items():
            # Ensure we have a council for this asset
            if asset_id not in self.asset_councils:
                await self.create_asset_council(asset_id)
                
            council = self.asset_councils[asset_id]
            
            # Update council with current market regime
            if asset_id in self.asset_regimes:
                await council.update_regime(self.asset_regimes[asset_id])
                
            # Process signals through asset council
            decision = await council.process_signals(asset_signal_list)
            
            if decision:
                asset_decisions[asset_id] = decision
        
        return asset_decisions
    
    def _group_by_asset(self, signals: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group signals by asset ID.
        
        Args:
            signals: List of signals
            
        Returns:
            Dictionary mapping asset IDs to signal lists
        """
        result = {}
        
        for signal in signals:
            asset_id = signal.get("asset_id", signal.get("symbol"))
            if not asset_id:
                continue
                
            if asset_id not in result:
                result[asset_id] = []
                
            result[asset_id].append(signal)
        
        return result
    
    async def update_market_regime(self, asset_id: str, regime: str):
        """
        Update the market regime for a specific asset.
        
        Args:
            asset_id: Asset identifier
            regime: Market regime
        """
        if asset_id not in self.asset_regimes or self.asset_regimes[asset_id] != regime:
            self.logger.info(f"Market regime changed for {asset_id}: {self.asset_regimes.get(asset_id, 'unknown')} -> {regime}")
            self.asset_regimes[asset_id] = regime
            
            # Update asset council if it exists
            if asset_id in self.asset_councils:
                await self.asset_councils[asset_id].update_regime(regime)
    
    async def register_ml_model(self, model_name: str, model_type: str, asset_ids: List[str] = None):
        """
        Register an ML model with the council system.
        
        Args:
            model_name: Name of the ML model
            model_type: Type of the ML model
            asset_ids: List of asset IDs this model specializes in (None for all assets)
        """
        # Register with ML council
        await self.ml_council.register_model(model_name, model_type, asset_ids)
        
        # Register with specific asset councils if applicable
        if asset_ids:
            for asset_id in asset_ids:
                if asset_id in self.asset_councils:
                    await self.asset_councils[asset_id].register_ml_model(model_name, model_type)
                    
        self.logger.info(f"Registered ML model {model_name} ({model_type}) with council system")
    
    async def register_brain(self, brain_name: str, asset_ids: List[str] = None):
        """
        Register a strategy brain with the council system.
        
        Args:
            brain_name: Name of the strategy brain
            asset_ids: List of asset IDs this brain specializes in (None for all assets)
        """
        # Register with specific asset councils
        if asset_ids:
            for asset_id in asset_ids:
                if asset_id in self.asset_councils:
                    await self.asset_councils[asset_id].register_brain(brain_name)
        else:
            # Register with all asset councils if no specific assets
            for asset_id, council in self.asset_councils.items():
                await council.register_brain(brain_name)
                
        self.logger.info(f"Registered brain {brain_name} with council system")
    
    async def get_asset_performance(self, asset_id: str) -> Dict[str, float]:
        """
        Get performance metrics for a specific asset.
        
        Args:
            asset_id: Asset identifier
            
        Returns:
            Performance metrics dictionary
        """
        if asset_id in self.asset_councils:
            return self.asset_councils[asset_id].performance_metrics
            
        return {}
    
    async def get_ml_model_performance(self, model_name: str) -> Dict[str, float]:
        """
        Get performance metrics for a specific ML model.
        
        Args:
            model_name: Model name
            
        Returns:
            Performance metrics dictionary
        """
        return self.ml_council.model_performance.get(model_name, {})
    
    async def get_best_models_for_asset(self, asset_id: str, limit: int = 3) -> List[str]:
        """
        Get the best performing ML models for a specific asset.
        
        Args:
            asset_id: Asset identifier
            limit: Maximum number of models to return
            
        Returns:
            List of model names
        """
        return await self.ml_council.get_best_models_for_asset(asset_id, limit)