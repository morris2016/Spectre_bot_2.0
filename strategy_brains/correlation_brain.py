#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Correlation-Based Trading Strategy Brain

This module implements a sophisticated correlation-based trading strategy that
identifies and exploits relationships between different assets, timeframes, and
market indicators to generate high-probability trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import asyncio
from datetime import datetime, timedelta

from common.utils import calculate_distance_correlation, calculate_dynamic_correlation
from common.async_utils import gather_with_concurrency
from feature_service.features.technical import calculate_z_score
from feature_service.transformers.normalizers import normalize_to_range
from data_storage.market_data import MarketDataRepository
from ml_models.models.regression import RegressionModel
from ml_models.models.time_series import TimeSeriesModel

from strategy_brains.base_brain import StrategyBrain


class CorrelationBrain(StrategyBrain):
    """
    Advanced correlation-based trading strategy that identifies and exploits
    relationships between assets, timeframes, and indicators for generating
    high-probability trading signals.
    
    This brain specializes in:
    1. Multi-asset correlation analysis
    2. Lead-lag relationship identification
    3. Correlation divergence detection
    4. Correlation regime shifting
    5. Inter-market analysis
    """
    
    STRATEGY_TYPE = "correlation"
    DEFAULT_CONFIG = {
        "correlation_window": 30,
        "min_correlation_strength": 0.7,
        "lookback_periods": [5, 10, 20, 30, 60],
        "z_score_threshold": 2.0,
        "divergence_threshold": 0.3,
        "regime_shift_threshold": 0.4,
        "lead_lag_max_periods": 10,
        "correlation_types": ["pearson", "spearman", "kendall", "distance"],
        "asset_universe": None,  # Will be populated dynamically
        "reference_assets": [],  # Key assets to compare against
        "use_ml_correlation": True,
        "ml_recalibration_interval": 7,  # Days
        "correlation_confidence_threshold": 0.75,
        "minimum_observation_periods": 20,
        "pair_selection_method": "strength",  # "strength", "stability", "profit"
        "max_active_correlations": 5,
        "use_cross_sectional": True,
        "use_time_series": True,
        "signal_confirmation_threshold": 2  # Number of methods confirming
    }
    
    def __init__(self, 
                 config: Dict[str, Any] = None, 
                 asset_id: str = None,
                 platform: str = None):
        """
        Initialize the correlation brain with configuration.
        
        Args:
            config: Configuration dictionary
            asset_id: The primary asset ID this brain is responsible for
            platform: The trading platform (e.g., "binance", "deriv")
        """
        super().__init__(config, asset_id, platform)
        self.logger = logging.getLogger(f"correlation_brain_{asset_id}_{platform}")
        
        # Initialize repositories and models
        self.market_data_repo = MarketDataRepository()
        self.regression_model = RegressionModel()
        self.time_series_model = TimeSeriesModel()
        
        # Correlation state tracking
        self.correlation_matrices = {}  # Timeframe -> correlation matrix
        self.lead_lag_relationships = {}  # (asset1, asset2) -> lead/lag periods
        self.correlation_history = {}  # (asset1, asset2) -> historical correlation
        self.active_correlation_pairs = []  # Currently tracked pairs
        self.correlation_regimes = {}  # Correlation regime state tracking
        
        # Last recalibration timestamps
        self.last_full_recalibration = datetime.utcnow() - timedelta(days=30)  # Force initial
        self.last_incremental_update = datetime.utcnow() - timedelta(hours=24)  # Force initial
        
        # Model states
        self.ml_models = {}  # (asset1, asset2) -> ML correlation model
        self.ml_model_performances = {}  # Track how well models are performing
        
        # Asset and feature caches to avoid redundant fetching
        self.asset_data_cache = {}
        self.feature_cache = {}
        
        # Initialization tasks
        self._initialize_correlation_matrices()
        
    async def _initialize_correlation_matrices(self):
        """Initializes the correlation matrices for all timeframes."""
        self.logger.info(f"Initializing correlation matrices for {self.asset_id} on {self.platform}")
        
        # Get asset universe if not specified
        if not self.config["asset_universe"]:
            self.config["asset_universe"] = await self._get_asset_universe()
            
        # Add our primary asset to reference assets if not already there
        if self.asset_id not in self.config["reference_assets"]:
            self.config["reference_assets"].append(self.asset_id)
        
        # Initialize empty matrices for each timeframe
        for timeframe in self.supported_timeframes:
            self.correlation_matrices[timeframe] = pd.DataFrame(
                index=self.config["asset_universe"],
                columns=self.config["asset_universe"]
            )
            
        # Schedule initial recalibration
        asyncio.create_task(self.recalibrate_correlations())
        
    async def _get_asset_universe(self) -> List[str]:
        """
        Dynamically determines the asset universe to track correlations with.
        
        Returns:
            List of asset IDs to include in correlation analysis
        """
        try:
            # Get top assets by volume and volatility
            top_assets = await self.market_data_repo.get_top_assets_by_metrics(
                platform=self.platform,
                metrics=["volume", "volatility"],
                limit=20
            )
            
            # Get assets in the same sector/category
            related_assets = await self.market_data_repo.get_related_assets(
                asset_id=self.asset_id,
                platform=self.platform,
                limit=10
            )
            
            # Combine and remove duplicates
            combined_universe = list(set(top_assets + related_assets))
            
            # Always include the primary asset
            if self.asset_id not in combined_universe:
                combined_universe.append(self.asset_id)
                
            self.logger.info(f"Determined asset universe with {len(combined_universe)} assets")
            return combined_universe
            
        except Exception as e:
            self.logger.error(f"Error determining asset universe: {str(e)}")
            # Fallback to a minimal universe
            return [self.asset_id]
            
    async def recalibrate_correlations(self, force_full: bool = False):
        """
        Performs a comprehensive recalibration of all correlation matrices.
        
        Args:
            force_full: If True, forces a full recalibration regardless of timing
        """
        now = datetime.utcnow()
        ml_recalibration_due = (now - self.last_full_recalibration).days >= self.config["ml_recalibration_interval"]
        
        if force_full or ml_recalibration_due:
            self.logger.info(f"Starting full correlation recalibration for {self.asset_id}")
            await self._perform_full_recalibration()
            self.last_full_recalibration = now
        else:
            self.logger.info(f"Starting incremental correlation update for {self.asset_id}")
            await self._perform_incremental_update()
            self.last_incremental_update = now
    
    async def _perform_full_recalibration(self):
        """Performs a full recalibration of all correlation matrices and models."""
        tasks = []
        
        # Calculate correlation matrices for each timeframe
        for timeframe in self.supported_timeframes:
            tasks.append(self._calculate_correlation_matrix(timeframe))
            
        # Calculate lead-lag relationships
        tasks.append(self._calculate_lead_lag_relationships())
        
        # Retrain ML correlation models if enabled
        if self.config["use_ml_correlation"]:
            tasks.append(self._train_ml_correlation_models())
            
        # Run all recalibration tasks concurrently
        await gather_with_concurrency(5, *tasks)
        
        # Identify the most promising correlation pairs
        await self._select_active_correlation_pairs()
        
        self.logger.info(f"Full correlation recalibration completed for {self.asset_id}")
    
    async def _perform_incremental_update(self):
        """Updates existing correlation data with the most recent market movements."""
        for timeframe in self.supported_timeframes:
            # Only update active correlation pairs to save resources
            for asset1, asset2 in self.active_correlation_pairs:
                await self._update_correlation_pair(asset1, asset2, timeframe)
        
        self.logger.info(f"Incremental correlation update completed for {self.asset_id}")
    
    async def _calculate_correlation_matrix(self, timeframe: str):
        """
        Calculates the full correlation matrix for a specific timeframe.
        
        Args:
            timeframe: The timeframe to calculate for (e.g., "1h", "4h", "1d")
        """
        assets = self.config["asset_universe"]
        window = self.config["correlation_window"]
        
        # Get price data for all assets in the universe
        price_data = {}
        for asset in assets:
            try:
                data = await self.market_data_repo.get_ohlcv_data(
                    asset_id=asset,
                    platform=self.platform,
                    timeframe=timeframe,
                    limit=window * 2  # Extra data for stability
                )
                if not data.empty:
                    price_data[asset] = data['close']
            except Exception as e:
                self.logger.warning(f"Failed to get data for {asset}: {str(e)}")
        
        # Create DataFrame from collected price data
        df = pd.DataFrame(price_data)
        
        # Calculate correlation matrices for different methods
        correlation_results = {}
        for method in self.config["correlation_types"]:
            if method == "distance":
                # Distance correlation handles non-linear relationships
                correlation_results[method] = calculate_distance_correlation(df)
            elif method == "dynamic":
                # Dynamic correlation adjusts for changing market regimes
                correlation_results[method] = calculate_dynamic_correlation(df)
            else:
                # Standard correlation methods
                correlation_results[method] = df.corr(method=method)
        
        # Combine correlation matrices with weights
        weights = {"pearson": 0.3, "spearman": 0.3, "kendall": 0.2, "distance": 0.2}
        if "dynamic" in correlation_results:
            weights = {k: v * 0.8 for k, v in weights.items()}
            weights["dynamic"] = 0.2
            
        combined_matrix = pd.DataFrame(0, index=assets, columns=assets)
        for method, matrix in correlation_results.items():
            if method in weights:
                combined_matrix += matrix * weights[method]
        
        # Store the result
        self.correlation_matrices[timeframe] = combined_matrix
        
        # Update correlation history
        for asset1 in assets:
            for asset2 in assets:
                if asset1 != asset2:
                    pair_key = (asset1, asset2) if asset1 < asset2 else (asset2, asset1)
                    if pair_key not in self.correlation_history:
                        self.correlation_history[pair_key] = {}
                    
                    if timeframe not in self.correlation_history[pair_key]:
                        self.correlation_history[pair_key][timeframe] = []
                        
                    self.correlation_history[pair_key][timeframe].append(
                        (datetime.utcnow(), combined_matrix.loc[asset1, asset2])
                    )
                    
                    # Prune history to prevent memory bloat
                    if len(self.correlation_history[pair_key][timeframe]) > 100:
                        self.correlation_history[pair_key][timeframe] = \
                            self.correlation_history[pair_key][timeframe][-100:]
    
    async def _calculate_lead_lag_relationships(self):
        """Identifies lead-lag relationships between assets."""
        assets = self.config["asset_universe"]
        max_lag = self.config["lead_lag_max_periods"]
        primary_timeframe = self.default_timeframe
        
        # Get extended price data for lead-lag analysis
        price_data = {}
        for asset in assets:
            try:
                data = await self.market_data_repo.get_ohlcv_data(
                    asset_id=asset,
                    platform=self.platform,
                    timeframe=primary_timeframe,
                    limit=self.config["correlation_window"] * 3  # Extra data for lag analysis
                )
                if not data.empty:
                    price_data[asset] = data['close']
            except Exception as e:
                self.logger.warning(f"Failed to get data for {asset}: {str(e)}")
        
        if not price_data:
            self.logger.error("No price data available for lead-lag analysis")
            return
            
        df = pd.DataFrame(price_data)
        
        # Calculate lead-lag for each asset pair
        for asset1 in assets:
            if asset1 not in df.columns:
                continue
                
            for asset2 in assets:
                if asset2 not in df.columns or asset1 == asset2:
                    continue
                    
                # Calculate cross-correlation for different lags
                series1 = df[asset1].pct_change().dropna()
                series2 = df[asset2].pct_change().dropna()
                
                if len(series1) < max_lag * 2 or len(series2) < max_lag * 2:
                    continue
                    
                corr_values = {}
                for lag in range(-max_lag, max_lag + 1):
                    if lag < 0:
                        # asset1 lags behind asset2
                        s1 = series1.iloc[-lag:]
                        s2 = series2.iloc[:len(s1)]
                    elif lag > 0:
                        # asset2 lags behind asset1
                        s2 = series2.iloc[lag:]
                        s1 = series1.iloc[:len(s2)]
                    else:
                        # No lag
                        s1 = series1
                        s2 = series2.iloc[:len(s1)]
                        
                    if len(s1) > 10 and len(s2) > 10:  # Ensure enough data
                        corr_values[lag] = np.abs(np.corrcoef(s1, s2)[0, 1])
                
                if corr_values:
                    # Find lag with highest correlation
                    best_lag = max(corr_values, key=corr_values.get)
                    pair_key = (asset1, asset2) if asset1 < asset2 else (asset2, asset1)
                    
                    # Only store if correlation is meaningful
                    if corr_values[best_lag] > self.config["min_correlation_strength"]:
                        if best_lag != 0:  # Only store if there's actual lead-lag
                            self.lead_lag_relationships[pair_key] = (best_lag, corr_values[best_lag])
                            
                            # Log significant lead-lag discoveries
                            leader = asset1 if best_lag > 0 else asset2
                            follower = asset2 if best_lag > 0 else asset1
                            lag_periods = abs(best_lag)
                            self.logger.info(
                                f"Lead-lag relationship: {leader} leads {follower} "
                                f"by {lag_periods} periods with corr={corr_values[best_lag]:.2f}"
                            )
    
    async def _train_ml_correlation_models(self):
        """Trains machine learning models to predict correlations between assets."""
        if not self.config["use_ml_correlation"]:
            return
            
        for asset1, asset2 in self.active_correlation_pairs:
            pair_key = (asset1, asset2) if asset1 < asset2 else (asset2, asset1)
            
            # Check if we need to train for this pair
            if pair_key in self.ml_models and not self._should_retrain_model(pair_key):
                continue
                
            self.logger.info(f"Training ML correlation model for {asset1} - {asset2}")
            
            # Prepare training data
            X, y = await self._prepare_correlation_training_data(asset1, asset2)
            
            if X is not None and len(X) > self.config["minimum_observation_periods"]:
                try:
                    # Train regression model to predict correlation
                    model = self.regression_model.create_model(
                        model_type="gradient_boosting",
                        params={
                            "n_estimators": 100,
                            "max_depth": 5,
                            "learning_rate": 0.1
                        }
                    )
                    
                    # Split into train/test
                    train_size = int(len(X) * 0.8)
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]
                    
                    # Train the model
                    model.fit(X_train, y_train)
                    
                    # Evaluate performance
                    test_score = model.score(X_test, y_test)
                    self.ml_model_performances[pair_key] = test_score
                    
                    # Store the model if performance is acceptable
                    if test_score > 0.6:  # Minimum R² of 0.6
                        self.ml_models[pair_key] = model
                        self.logger.info(
                            f"ML correlation model for {asset1} - {asset2} trained successfully. "
                            f"Test R²: {test_score:.2f}"
                        )
                    else:
                        self.logger.warning(
                            f"ML correlation model for {asset1} - {asset2} has poor performance. "
                            f"Test R²: {test_score:.2f}"
                        )
                        
                except Exception as e:
                    self.logger.error(f"Failed to train ML correlation model for {asset1} - {asset2}: {str(e)}")
    
    async def _prepare_correlation_training_data(self, asset1: str, asset2: str):
        """
        Prepares training data for correlation prediction models.
        
        Args:
            asset1: First asset in the pair
            asset2: Second asset in the pair
            
        Returns:
            Tuple of (X, y) where X is feature matrix and y is correlation values
        """
        try:
            # Get features that might predict correlation changes
            timeframe = self.default_timeframe
            
            # Get market data for both assets
            data1 = await self.market_data_repo.get_ohlcv_data(
                asset_id=asset1,
                platform=self.platform,
                timeframe=timeframe,
                limit=200  # Larger dataset for training
            )
            
            data2 = await self.market_data_repo.get_ohlcv_data(
                asset_id=asset2,
                platform=self.platform,
                timeframe=timeframe,
                limit=200
            )
            
            if data1.empty or data2.empty:
                return None, None
                
            # Calculate features that might predict correlation
            features = []
            correlations = []
            
            # Use rolling windows to create training samples
            window = self.config["correlation_window"]
            
            for i in range(len(data1) - window - 1):
                # Window for feature calculation
                d1 = data1.iloc[i:i+window]
                d2 = data2.iloc[i:i+window]
                
                # Next window for correlation (target)
                next_d1 = data1.iloc[i+1:i+window+1]
                next_d2 = data2.iloc[i+1:i+window+1]
                
                # Calculate features
                feature_vector = []
                
                # Volatility features
                feature_vector.append(d1['close'].pct_change().std())
                feature_vector.append(d2['close'].pct_change().std())
                feature_vector.append(d1['close'].pct_change().std() / d2['close'].pct_change().std())
                
                # Trend features
                feature_vector.append((d1['close'].iloc[-1] / d1['close'].iloc[0]) - 1)
                feature_vector.append((d2['close'].iloc[-1] / d2['close'].iloc[0]) - 1)
                
                # Relative trend
                feature_vector.append(
                    ((d1['close'].iloc[-1] / d1['close'].iloc[0]) - 1) -
                    ((d2['close'].iloc[-1] / d2['close'].iloc[0]) - 1)
                )
                
                # Volume features
                if 'volume' in d1.columns and 'volume' in d2.columns:
                    feature_vector.append(d1['volume'].mean())
                    feature_vector.append(d2['volume'].mean())
                    feature_vector.append(d1['volume'].mean() / d2['volume'].mean())
                
                # Current correlation as a feature
                current_corr = d1['close'].corr(d2['close'])
                feature_vector.append(current_corr)
                
                # Correlation stability (last few windows)
                if i >= 3:
                    corr_stability = []
                    for j in range(3):
                        prev_d1 = data1.iloc[i-j-1:i-j+window-1]
                        prev_d2 = data2.iloc[i-j-1:i-j+window-1]
                        prev_corr = prev_d1['close'].corr(prev_d2['close'])
                        corr_stability.append(prev_corr)
                    
                    feature_vector.append(np.std(corr_stability))
                else:
                    feature_vector.append(0)  # Placeholder when not enough history
                
                # Target: next window correlation
                target_corr = next_d1['close'].corr(next_d2['close'])
                
                # Add to training data
                features.append(feature_vector)
                correlations.append(target_corr)
            
            return np.array(features), np.array(correlations)
            
        except Exception as e:
            self.logger.error(f"Error preparing correlation training data: {str(e)}")
            return None, None
    
    def _should_retrain_model(self, pair_key: Tuple[str, str]) -> bool:
        """
        Determines if a correlation model should be retrained based on performance.
        
        Args:
            pair_key: The asset pair key (asset1, asset2)
            
        Returns:
            True if model should be retrained, False otherwise
        """
        # Always retrain if no performance record
        if pair_key not in self.ml_model_performances:
            return True
            
        # Retrain if performance is poor
        if self.ml_model_performances[pair_key] < 0.5:
            return True
            
        # Retrain based on recalibration interval
        days_since_recal = (datetime.utcnow() - self.last_full_recalibration).days
        if days_since_recal >= self.config["ml_recalibration_interval"]:
            return True
            
        return False
    
    async def _update_correlation_pair(self, asset1: str, asset2: str, timeframe: str):
        """
        Updates the correlation data for a specific asset pair.
        
        Args:
            asset1: First asset in the pair
            asset2: Second asset in the pair
            timeframe: Timeframe to update
        """
        try:
            # Get latest data
            data1 = await self.market_data_repo.get_ohlcv_data(
                asset_id=asset1,
                platform=self.platform,
                timeframe=timeframe,
                limit=self.config["correlation_window"]
            )
            
            data2 = await self.market_data_repo.get_ohlcv_data(
                asset_id=asset2,
                platform=self.platform,
                timeframe=timeframe,
                limit=self.config["correlation_window"]
            )
            
            if data1.empty or data2.empty:
                return
                
            # Calculate current correlation
            current_corr = data1['close'].corr(data2['close'])
            
            # Update correlation matrix
            if (asset1 in self.correlation_matrices[timeframe].index and 
                asset2 in self.correlation_matrices[timeframe].columns):
                self.correlation_matrices[timeframe].loc[asset1, asset2] = current_corr
                self.correlation_matrices[timeframe].loc[asset2, asset1] = current_corr
            
            # Update correlation history
            pair_key = (asset1, asset2) if asset1 < asset2 else (asset2, asset1)
            if pair_key not in self.correlation_history:
                self.correlation_history[pair_key] = {}
            
            if timeframe not in self.correlation_history[pair_key]:
                self.correlation_history[pair_key][timeframe] = []
                
            self.correlation_history[pair_key][timeframe].append(
                (datetime.utcnow(), current_corr)
            )
            
            # Prune history
            if len(self.correlation_history[pair_key][timeframe]) > 100:
                self.correlation_history[pair_key][timeframe] = \
                    self.correlation_history[pair_key][timeframe][-100:]
                    
            # Check for correlation regime shifts
            if len(self.correlation_history[pair_key][timeframe]) >= 5:
                recent_corrs = [c[1] for c in self.correlation_history[pair_key][timeframe][-5:]]
                older_corrs = [c[1] for c in self.correlation_history[pair_key][timeframe][:-5]]
                
                if older_corrs:  # Make sure we have older data
                    recent_avg = np.mean(recent_corrs)
                    older_avg = np.mean(older_corrs)
                    
                    if abs(recent_avg - older_avg) > self.config["regime_shift_threshold"]:
                        self.correlation_regimes[pair_key] = {
                            'timestamp': datetime.utcnow(),
                            'old_regime': older_avg,
                            'new_regime': recent_avg,
                            'shift_magnitude': abs(recent_avg - older_avg)
                        }
                        
                        self.logger.info(
                            f"Correlation regime shift detected for {asset1}-{asset2}: "
                            f"{older_avg:.2f} → {recent_avg:.2f}"
                        )
            
        except Exception as e:
            self.logger.error(f"Error updating correlation for {asset1}-{asset2}: {str(e)}")
    
    async def _select_active_correlation_pairs(self):
        """Selects the most promising correlation pairs to actively track."""
        assets = self.config["asset_universe"]
        primary_timeframe = self.default_timeframe
        max_pairs = self.config["max_active_correlations"]
        
        # Always include pairs with the primary asset
        primary_pairs = []
        for asset in assets:
            if asset != self.asset_id:
                pair = (self.asset_id, asset) if self.asset_id < asset else (asset, self.asset_id)
                primary_pairs.append((pair, self._get_pair_score(pair, primary_timeframe)))
        
        # Sort by score
        primary_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Take top pairs involving primary asset
        selected_pairs = [p[0] for p in primary_pairs[:max(3, max_pairs // 2)]]
        
        # Find other promising pairs if we still have slots
        if len(selected_pairs) < max_pairs:
            other_pairs = []
            for i, asset1 in enumerate(assets):
                for asset2 in assets[i+1:]:
                    if asset1 != self.asset_id and asset2 != self.asset_id:
                        pair = (asset1, asset2)
                        other_pairs.append((pair, self._get_pair_score(pair, primary_timeframe)))
            
            # Sort by score
            other_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Add top pairs until we reach max
            for pair, score in other_pairs:
                if len(selected_pairs) >= max_pairs:
                    break
                selected_pairs.append(pair)
        
        # Update active pairs
        self.active_correlation_pairs = selected_pairs
        self.logger.info(f"Selected {len(selected_pairs)} active correlation pairs")
    
    def _get_pair_score(self, pair: Tuple[str, str], timeframe: str) -> float:
        """
        Calculates a score for a correlation pair based on strength, stability, and usefulness.
        
        Args:
            pair: The asset pair (asset1, asset2)
            timeframe: Timeframe to use for scoring
            
        Returns:
            Score value (higher is better)
        """
        asset1, asset2 = pair
        
        # Check if we have the correlation matrix for this timeframe
        if timeframe not in self.correlation_matrices:
            return 0
            
        matrix = self.correlation_matrices[timeframe]
        
        # Check if both assets are in the matrix
        if asset1 not in matrix.index or asset2 not in matrix.columns:
            return 0
            
        # Get current correlation
        current_corr = matrix.loc[asset1, asset2]
        corr_strength = abs(current_corr)
        
        # Check if we have correlation history
        history_score = 0
        if pair in self.correlation_history and timeframe in self.correlation_history[pair]:
            history = self.correlation_history[pair][timeframe]
            if len(history) >= 5:
                # Calculate correlation stability (inverse of standard deviation)
                recent_corrs = [c[1] for c in history[-5:]]
                stability = 1.0 / (np.std(recent_corrs) + 0.01)  # Avoid division by zero
                history_score = stability * 0.5
        
        # Check for lead-lag relationship
        lead_lag_score = 0
        if pair in self.lead_lag_relationships:
            lag, lag_corr = self.lead_lag_relationships[pair]
            lead_lag_score = abs(lag_corr) * 0.3
        
        # Check for correlation regime shifts (higher score if recently shifted)
        regime_score = 0
        if pair in self.correlation_regimes:
            regime_data = self.correlation_regimes[pair]
            days_since_shift = (datetime.utcnow() - regime_data['timestamp']).days
            if days_since_shift < 5:  # Recent shift
                regime_score = regime_data['shift_magnitude'] * 0.2
        
        # Combine scores
        total_score = corr_strength + history_score + lead_lag_score + regime_score
        
        return total_score
    
    async def generate_signals(self) -> List[Dict[str, Any]]:
        """
        Generates trading signals based on correlation analysis.
        
        Returns:
            List of trading signal dictionaries
        """
        signals = []
        
        # Ensure we have fresh data
        if (datetime.utcnow() - self.last_incremental_update).seconds > 3600:
            await self._perform_incremental_update()
        
        # Check for regime-based signals
        regime_signals = await self._check_correlation_regime_signals()
        signals.extend(regime_signals)
        
        # Check for divergence-based signals
        divergence_signals = await self._check_correlation_divergence_signals()
        signals.extend(divergence_signals)
        
        # Check for lead-lag-based signals
        lead_lag_signals = await self._check_lead_lag_signals()
        signals.extend(lead_lag_signals)
        
        # Check for ml-predicted correlation signals
        if self.config["use_ml_correlation"]:
            ml_signals = await self._check_ml_correlation_signals()
            signals.extend(ml_signals)
        
        # Add standard strategy metadata
        for signal in signals:
            signal['strategy'] = self.STRATEGY_TYPE
            signal['asset_id'] = self.asset_id
            signal['platform'] = self.platform
            signal['timestamp'] = datetime.utcnow().isoformat()
        
        return signals
    
    async def _check_correlation_regime_signals(self) -> List[Dict[str, Any]]:
        """
        Checks for trading signals based on correlation regime shifts.
        
        Returns:
            List of trading signal dictionaries
        """
        signals = []
        
        # Only look for regime signals for active pairs involving primary asset
        for asset1, asset2 in self.active_correlation_pairs:
            # Only consider pairs involving the primary asset
            if asset1 != self.asset_id and asset2 != self.asset_id:
                continue
                
            pair_key = (asset1, asset2) if asset1 < asset2 else (asset2, asset1)
            
            # Check if there's a recent regime shift
            if pair_key in self.correlation_regimes:
                regime_data = self.correlation_regimes[pair_key]
                days_since_shift = (datetime.utcnow() - regime_data['timestamp']).days
                
                # Only consider recent shifts (within 3 days)
                if days_since_shift <= 3:
                    # Get direction of shift and magnitude
                    old_regime = regime_data['old_regime']
                    new_regime = regime_data['new_regime']
                    magnitude = regime_data['shift_magnitude']
                    
                    if magnitude > self.config["regime_shift_threshold"]:
                        non_primary = asset2 if asset1 == self.asset_id else asset1
                        
                        # Get prices to check current relationship
                        primary_data = await self._get_asset_data(self.asset_id)
                        other_data = await self._get_asset_data(non_primary)
                        
                        if primary_data is None or other_data is None:
                            continue
                            
                        # Check if prices are moving together or opposite recently
                        primary_returns = primary_data['close'].pct_change().dropna()
                        other_returns = other_data['close'].pct_change().dropna()
                        
                        if len(primary_returns) < 5 or len(other_returns) < 5:
                            continue
                            
                        recent_primary = primary_returns.iloc[-5:]
                        recent_other = other_returns.iloc[-5:]
                        
                        # Determine if we should trade the primary asset based on the other
                        signal_direction = None
                        confidence = 0.0
                        
                        # If correlation is becoming more positive
                        if new_regime > old_regime:
                            # Assets should move together
                            if new_regime > 0.7:  # Strong positive correlation
                                # If other asset is trending, primary should follow
                                other_trend = other_data['close'].iloc[-1] / other_data['close'].iloc[-5] - 1
                                
                                if abs(other_trend) > 0.01:  # Meaningful trend
                                    signal_direction = "buy" if other_trend > 0 else "sell"
                                    confidence = min(0.8, new_regime * (1 + abs(other_trend) * 10))
                        
                        # If correlation is becoming more negative
                        elif new_regime < old_regime:
                            # Assets should move opposite
                            if new_regime < -0.7:  # Strong negative correlation
                                # If other asset is trending, primary should do opposite
                                other_trend = other_data['close'].iloc[-1] / other_data['close'].iloc[-5] - 1
                                
                                if abs(other_trend) > 0.01:  # Meaningful trend
                                    signal_direction = "sell" if other_trend > 0 else "buy"
                                    confidence = min(0.8, abs(new_regime) * (1 + abs(other_trend) * 10))
                        
                        if signal_direction and confidence > self.config["correlation_confidence_threshold"]:
                            signals.append({
                                'signal_type': 'correlation_regime',
                                'direction': signal_direction,
                                'confidence': confidence,
                                'timeframe': self.default_timeframe,
                                'metadata': {
                                    'correlated_asset': non_primary,
                                    'old_correlation': old_regime,
                                    'new_correlation': new_regime,
                                    'days_since_shift': days_since_shift
                                }
                            })
        
        return signals
    
    async def _check_correlation_divergence_signals(self) -> List[Dict[str, Any]]:
        """
        Checks for trading signals based on correlation divergences.
        
        Returns:
            List of trading signal dictionaries
        """
        signals = []
        
        # Look for divergences in correlation patterns
        for asset1, asset2 in self.active_correlation_pairs:
            # Only consider pairs involving the primary asset
            if asset1 != self.asset_id and asset2 != self.asset_id:
                continue
                
            non_primary = asset2 if asset1 == self.asset_id else asset1
            pair_key = (asset1, asset2) if asset1 < asset2 else (asset2, asset1)
            
            # Need historical correlation data
            if pair_key not in self.correlation_history:
                continue
                
            # Check different timeframes for divergences
            divergence_signals = []
            
            for timeframe in self.supported_timeframes:
                if timeframe not in self.correlation_history[pair_key]:
                    continue
                    
                history = self.correlation_history[pair_key][timeframe]
                if len(history) < 10:
                    continue
                
                # Get correlation values
                dates = [h[0] for h in history[-10:]]
                values = [h[1] for h in history[-10:]]
                
                # Calculate Z-score of recent correlation
                z_score = calculate_z_score(values, window=5)
                
                # Check if correlation is abnormally high or low
                if abs(z_score) > self.config["z_score_threshold"]:
                    # Get price data to check for potential reversion
                    primary_data = await self._get_asset_data(self.asset_id, timeframe)
                    other_data = await self._get_asset_data(non_primary, timeframe)
                    
                    if primary_data is None or other_data is None:
                        continue
                        
                    # Check how assets have been moving
                    primary_returns = primary_data['close'].pct_change().dropna()
                    other_returns = other_data['close'].pct_change().dropna()
                    
                    if len(primary_returns) < 5 or len(other_returns) < 5:
                        continue
                        
                    # Recent returns
                    recent_primary_return = primary_data['close'].iloc[-1] / primary_data['close'].iloc[-5] - 1
                    recent_other_return = other_data['close'].iloc[-1] / other_data['close'].iloc[-5] - 1
                    
                    signal_direction = None
                    confidence = 0.0
                    
                    # Determine signal based on correlation and return patterns
                    current_corr = values[-1]
                    
                    # Case 1: Assets have been moving together but correlation is abnormally high
                    if current_corr > 0.7 and z_score > self.config["z_score_threshold"]:
                        # Strongly correlated - might revert to mean
                        if np.sign(recent_primary_return) == np.sign(recent_other_return):
                            # If both moving in same direction, but correlation is stretched
                            # Expect reversion - go against primary trend
                            signal_direction = "sell" if recent_primary_return > 0 else "buy"
                            confidence = min(0.7, 0.5 + abs(z_score) / 10)
                    
                    # Case 2: Assets have been moving opposite but correlation is abnormally negative
                    elif current_corr < -0.7 and z_score < -self.config["z_score_threshold"]:
                        # Strongly negatively correlated - might revert to mean
                        if np.sign(recent_primary_return) != np.sign(recent_other_return):
                            # If moving in opposite directions, but correlation is stretched
                            # Expect reversion - go against primary trend
                            signal_direction = "sell" if recent_primary_return > 0 else "buy"
                            confidence = min(0.7, 0.5 + abs(z_score) / 10)
                    
                    # Case 3: Correlation breakdown - has been correlated but diverging
                    elif abs(current_corr) > 0.5 and abs(z_score) > self.config["z_score_threshold"]:
                        historical_corr = np.mean(values[:-3])
                        
                        if abs(current_corr - historical_corr) > self.config["divergence_threshold"]:
                            # Correlation breakdown - follow the trend of more strongly moving asset
                            if abs(recent_other_return) > abs(recent_primary_return) * 1.5:
                                expected_primary = recent_other_return * np.sign(historical_corr)
                                if np.sign(expected_primary) != np.sign(recent_primary_return):
                                    # Primary should follow other based on historical correlation
                                    signal_direction = "buy" if expected_primary > 0 else "sell"
                                    confidence = min(0.65, 0.4 + abs(z_score) / 10)
                    
                    if signal_direction and confidence > self.config["correlation_confidence_threshold"]:
                        divergence_signals.append({
                            'signal_type': 'correlation_divergence',
                            'direction': signal_direction,
                            'confidence': confidence,
                            'timeframe': timeframe,
                            'metadata': {
                                'correlated_asset': non_primary,
                                'current_correlation': current_corr,
                                'correlation_z_score': z_score,
                                'primary_recent_return': recent_primary_return,
                                'other_recent_return': recent_other_return
                            }
                        })
            
            # Take highest confidence signal from any timeframe
            if divergence_signals:
                best_signal = max(divergence_signals, key=lambda x: x['confidence'])
                signals.append(best_signal)
        
        return signals
    
    async def _check_lead_lag_signals(self) -> List[Dict[str, Any]]:
        """
        Checks for trading signals based on lead-lag relationships.
        
        Returns:
            List of trading signal dictionaries
        """
        signals = []
        
        # Check for lead-lag signals
        for asset1, asset2 in self.active_correlation_pairs:
            # Only consider pairs involving the primary asset
            if asset1 != self.asset_id and asset2 != self.asset_id:
                continue
                
            non_primary = asset2 if asset1 == self.asset_id else asset1
            pair_key = (asset1, asset2) if asset1 < asset2 else (asset2, asset1)
            
            # Check if there's a lead-lag relationship
            if pair_key in self.lead_lag_relationships:
                lag, lag_corr = self.lead_lag_relationships[pair_key]
                
                # Only consider strong relationships
                if abs(lag_corr) > self.config["min_correlation_strength"]:
                    # Determine if primary asset leads or lags
                    primary_leads = (self.asset_id == asset1 and lag > 0) or (self.asset_id == asset2 and lag < 0)
                    
                    if primary_leads:
                        # Primary asset leads - not useful for predicting primary
                        continue
                    
                    # Non-primary asset leads, get its data
                    primary_data = await self._get_asset_data(self.asset_id)
                    other_data = await self._get_asset_data(non_primary)
                    
                    if primary_data is None or other_data is None:
                        continue
                    
                    # Calculate returns
                    other_returns = other_data['close'].pct_change().dropna()
                    
                    if len(other_returns) < abs(lag) + 5:
                        continue
                    
                    # Get the leading pattern from the non-primary asset
                    recent_other_return = other_data['close'].iloc[-1] / other_data['close'].iloc[-5] - 1
                    
                    signal_direction = None
                    confidence = 0.0
                    
                    # If strong trend in leading asset
                    if abs(recent_other_return) > 0.01:
                        # If positive correlation, follow the leading asset's direction
                        if lag_corr > 0:
                            signal_direction = "buy" if recent_other_return > 0 else "sell"
                        # If negative correlation, do opposite of leading asset
                        else:
                            signal_direction = "sell" if recent_other_return > 0 else "buy"
                        
                        # Confidence based on correlation strength and trend magnitude
                        confidence = min(0.85, abs(lag_corr) * (1 + min(1, abs(recent_other_return) * 10)))
                    
                    if signal_direction and confidence > self.config["correlation_confidence_threshold"]:
                        signals.append({
                            'signal_type': 'lead_lag',
                            'direction': signal_direction,
                            'confidence': confidence,
                            'timeframe': self.default_timeframe,
                            'metadata': {
                                'leading_asset': non_primary,
                                'lag_periods': abs(lag),
                                'lag_correlation': lag_corr,
                                'leading_asset_return': recent_other_return
                            }
                        })
        
        return signals
    
    async def _check_ml_correlation_signals(self) -> List[Dict[str, Any]]:
        """
        Checks for trading signals based on machine learning correlation predictions.
        
        Returns:
            List of trading signal dictionaries
        """
        if not self.config["use_ml_correlation"]:
            return []
            
        signals = []
        
        # Check ML-based correlation predictions
        for asset1, asset2 in self.active_correlation_pairs:
            # Only consider pairs involving the primary asset
            if asset1 != self.asset_id and asset2 != self.asset_id:
                continue
                
            non_primary = asset2 if asset1 == self.asset_id else asset1
            pair_key = (asset1, asset2) if asset1 < asset2 else (asset2, asset1)
            
            # Check if we have an ML model for this pair
            if pair_key not in self.ml_models:
                continue
                
            # Get model and performance
            model = self.ml_models[pair_key]
            model_score = self.ml_model_performances.get(pair_key, 0)
            
            # Only use models with decent performance
            if model_score < 0.6:
                continue
                
            # Prepare features for prediction
            X = await self._prepare_prediction_features(self.asset_id, non_primary)
            
            if X is None:
                continue
                
            # Predict future correlation
            try:
                predicted_corr = model.predict([X])[0]
                
                # Get current correlation
                current_corr = None
                for tf in self.correlation_matrices:
                    if (self.asset_id in self.correlation_matrices[tf].index and 
                        non_primary in self.correlation_matrices[tf].columns):
                        current_corr = self.correlation_matrices[tf].loc[self.asset_id, non_primary]
                        break
                
                if current_corr is None:
                    continue
                
                # Check if big correlation change predicted
                corr_change = predicted_corr - current_corr
                
                if abs(corr_change) > self.config["divergence_threshold"]:
                    # Get price data
                    primary_data = await self._get_asset_data(self.asset_id)
                    other_data = await self._get_asset_data(non_primary)
                    
                    if primary_data is None or other_data is None:
                        continue
                    
                    # Recent returns
                    recent_primary_return = primary_data['close'].iloc[-1] / primary_data['close'].iloc[-5] - 1
                    recent_other_return = other_data['close'].iloc[-1] / other_data['close'].iloc[-5] - 1
                    
                    signal_direction = None
                    confidence = 0.0
                    
                    # Case 1: Correlation predicted to increase
                    if corr_change > 0:
                        # Assets will move more together
                        if abs(recent_other_return) > abs(recent_primary_return):
                            # Other asset moving more strongly
                            signal_direction = "buy" if recent_other_return > 0 else "sell"
                            confidence = min(0.75, 0.5 + abs(corr_change) + model_score * 0.2)
                    
                    # Case 2: Correlation predicted to decrease
                    else:
                        # Assets will move more independently or oppositely
                        if predicted_corr < -0.3 and recent_other_return != 0:
                            # Predicted negative correlation, do opposite
                            signal_direction = "sell" if recent_other_return > 0 else "buy"
                            confidence = min(0.7, 0.5 + abs(corr_change) + model_score * 0.1)
                    
                    if signal_direction and confidence > self.config["correlation_confidence_threshold"]:
                        signals.append({
                            'signal_type': 'ml_correlation',
                            'direction': signal_direction,
                            'confidence': confidence,
                            'timeframe': self.default_timeframe,
                            'metadata': {
                                'correlated_asset': non_primary,
                                'current_correlation': current_corr,
                                'predicted_correlation': predicted_corr,
                                'correlation_change': corr_change,
                                'model_r2_score': model_score,
                                'primary_recent_return': recent_primary_return,
                                'other_recent_return': recent_other_return
                            }
                        })
            
            except Exception as e:
                self.logger.error(f"Error predicting correlation with ML model: {str(e)}")
        
        return signals
    
    async def _prepare_prediction_features(self, asset1: str, asset2: str):
        """
        Prepares features for correlation prediction.
        
        Args:
            asset1: First asset in the pair
            asset2: Second asset in the pair
            
        Returns:
            Feature vector for prediction
        """
        try:
            # Get market data
            data1 = await self._get_asset_data(asset1)
            data2 = await self._get_asset_data(asset2)
            
            if data1 is None or data2 is None:
                return None
                
            window = self.config["correlation_window"]
            
            if len(data1) < window or len(data2) < window:
                return None
                
            # Calculate features (same as in training)
            feature_vector = []
            
            # Volatility features
            feature_vector.append(data1['close'].pct_change().std())
            feature_vector.append(data2['close'].pct_change().std())
            feature_vector.append(data1['close'].pct_change().std() / data2['close'].pct_change().std())
            
            # Trend features
            feature_vector.append((data1['close'].iloc[-1] / data1['close'].iloc[0]) - 1)
            feature_vector.append((data2['close'].iloc[-1] / data2['close'].iloc[0]) - 1)
            
            # Relative trend
            feature_vector.append(
                ((data1['close'].iloc[-1] / data1['close'].iloc[0]) - 1) -
                ((data2['close'].iloc[-1] / data2['close'].iloc[0]) - 1)
            )
            
            # Volume features
            if 'volume' in data1.columns and 'volume' in data2.columns:
                feature_vector.append(data1['volume'].mean())
                feature_vector.append(data2['volume'].mean())
                feature_vector.append(data1['volume'].mean() / data2['volume'].mean())
            else:
                # Add placeholders if volume not available
                feature_vector.extend([0, 0, 1])
            
            # Current correlation
            current_corr = data1['close'].corr(data2['close'])
            feature_vector.append(current_corr)
            
            # Correlation stability
            pair_key = (asset1, asset2) if asset1 < asset2 else (asset2, asset1)
            corr_stability = 0
            
            if pair_key in self.correlation_history:
                for tf in self.correlation_history[pair_key]:
                    history = self.correlation_history[pair_key][tf]
                    if len(history) >= 3:
                        recent_corrs = [c[1] for c in history[-3:]]
                        corr_stability = np.std(recent_corrs)
                        break
            
            feature_vector.append(corr_stability)
            
            return feature_vector
            
        except Exception as e:
            self.logger.error(f"Error preparing prediction features: {str(e)}")
            return None
    
    async def _get_asset_data(self, asset_id: str, timeframe: str = None) -> Optional[pd.DataFrame]:
        """
        Gets market data for an asset with caching to avoid redundant fetches.
        
        Args:
            asset_id: Asset to get data for
            timeframe: Timeframe to get data for, defaults to default_timeframe
            
        Returns:
            Pandas DataFrame with OHLCV data or None if not available
        """
        if timeframe is None:
            timeframe = self.default_timeframe
            
        # Check cache first
        cache_key = f"{asset_id}_{timeframe}"
        if cache_key in self.asset_data_cache:
            # Check if cache is still fresh (less than 5 minutes old)
            cache_time, data = self.asset_data_cache[cache_key]
            if (datetime.utcnow() - cache_time).seconds < 300:
                return data
        
        # Fetch new data
        try:
            data = await self.market_data_repo.get_ohlcv_data(
                asset_id=asset_id,
                platform=self.platform,
                timeframe=timeframe,
                limit=self.config["correlation_window"] * 2  # Extra data for more context
            )
            
            if not data.empty:
                # Update cache
                self.asset_data_cache[cache_key] = (datetime.utcnow(), data)
                return data
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting data for {asset_id}: {str(e)}")
            return None
    
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
        
        # Check for forced recalibration
        if state_data.get('recalibrate', False):
            self.logger.info("Forced recalibration requested")
            await self.recalibrate_correlations(force_full=True)
        
        # Check for asset universe updates
        if 'asset_universe' in state_data:
            self.config["asset_universe"] = state_data['asset_universe']
            self.logger.info(f"Updated asset universe with {len(self.config['asset_universe'])} assets")
            await self._initialize_correlation_matrices()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Gets the current status of the correlation brain.
        
        Returns:
            Dictionary with status information
        """
        return {
            'brain_type': self.STRATEGY_TYPE,
            'asset_id': self.asset_id,
            'platform': self.platform,
            'active_correlation_pairs': self.active_correlation_pairs,
            'last_recalibration': self.last_full_recalibration.isoformat(),
            'last_update': self.last_incremental_update.isoformat(),
            'tracked_assets': len(self.config["asset_universe"]),
            'ml_models_count': len(self.ml_models),
            'regime_shifts_detected': len(self.correlation_regimes)
        }
    
    def reset_state(self):
        """Resets the brain state for a fresh start."""
        # Correlation state tracking
        self.correlation_matrices = {}
        self.lead_lag_relationships = {}
        self.correlation_history = {}
        self.active_correlation_pairs = []
        self.correlation_regimes = {}
        
        # Caches
        self.asset_data_cache = {}
        self.feature_cache = {}
        
        # Reset timestamps to force recalibration
        self.last_full_recalibration = datetime.utcnow() - timedelta(days=30)
        self.last_incremental_update = datetime.utcnow() - timedelta(hours=24)
        
        # Re-initialize
        asyncio.create_task(self._initialize_correlation_matrices())
        
        self.logger.info(f"Reset state for {self.asset_id} correlation brain")
