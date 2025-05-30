#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Market Regime Feed

This module implements a sophisticated system for identifying market regimes and regime changes
to enable optimized strategy selection and adaptation. It uses multiple advanced analytical approaches
to classify current market conditions and predict transitions between different regimes.
"""

import os
import numpy as np
import pandas as pd
import datetime
import logging
import asyncio
import json
import warnings
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass
from enum import Enum, auto
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from hmmlearn import hmm

from data_feeds.base_feed import BaseFeed
from common.logger import get_logger
from common.utils import rolling_window, exponential_smoothing, periodic_reset
from common.constants import (
    REGIME_SCAN_INTERVAL,
    REGIME_LOOKBACK_PERIODS,
    MARKETS_OF_INTEREST,
    ASSETS_OF_INTEREST,
    MAX_REGIMES
)
from common.exceptions import (
    DataFeedConnectionError,
    DataParsingError,
    RegimeDetectionError
)


class MarketRegimeType(Enum):
    """Enum defining different market regime types."""
    BULL_TRENDING = auto()
    BEAR_TRENDING = auto()
    SIDEWAYS = auto()
    HIGH_VOLATILITY = auto()
    LOW_VOLATILITY = auto()
    BULL_VOLATILE = auto()
    BEAR_VOLATILE = auto()
    BULL_EXHAUSTION = auto()
    BEAR_EXHAUSTION = auto()
    BREAKOUT = auto()
    BREAKDOWN = auto()
    ROTATION = auto()
    LIQUIDITY_CRISIS = auto()
    ACCUMULATION = auto()
    DISTRIBUTION = auto()
    EUPHORIA = auto()
    CAPITULATION = auto()
    RECOVERY = auto()
    UNKNOWN = auto()


@dataclass
class MarketRegime:
    """Class representing a market regime with its properties."""
    regime_type: MarketRegimeType
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime]
    confidence: float
    features: Dict[str, float]
    description: str
    source_models: List[str]
    asset: str
    timeframe: str
    transition_probability: Dict[MarketRegimeType, float] = None
    expected_duration: Optional[float] = None
    historical_performance: Dict[str, float] = None


class RegimeFeed(BaseFeed):
    """
    Market Regime Feed for the QuantumSpectre Elite Trading System.
    
    Identifies and tracks market regimes using multiple analytical methods to
    enable optimized strategy selection and adaptation. Combines statistical,
    machine learning, and domain-specific approaches for robust regime detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Market Regime Feed with configuration settings.
        
        Args:
            config: Configuration dictionary for the Regime Feed
        """
        super().__init__(config)
        self.logger = get_logger("RegimeFeed")
        self.name = "regime_feed"
        self.description = "Market Regime Detection Feed"
        
        # Configure detection parameters
        self.assets = config.get('assets', ASSETS_OF_INTEREST)
        self.markets = config.get('markets', MARKETS_OF_INTEREST)
        self.scan_interval = config.get('scan_interval', REGIME_SCAN_INTERVAL)
        self.lookback_periods = config.get('lookback_periods', REGIME_LOOKBACK_PERIODS)
        self.max_regimes = config.get('max_regimes', MAX_REGIMES)
        
        # Configure timeframes for analysis
        self.timeframes = config.get('timeframes', ['1m', '5m', '15m', '1h', '4h', '1d', '1w'])
        
        # Regime models for different detection methods
        self.models = {}
        
        # Active regimes by asset and timeframe
        self.current_regimes = {}
        
        # Historical regimes for reference
        self.historical_regimes = {}
        
        # Transition matrices for regime prediction
        self.transition_matrices = {}
        
        # Feature importance for interpretation
        self.feature_importance = {}
        
        # Feature store for regime detection
        self.feature_store = {}
        
        # Initialize the models
        self._initialize_models()
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
        
        # Last scan time tracking
        self.last_scans = {}
        
        # Performance metrics for each regime
        self.regime_performance = {}
        
        self.logger.info(f"Initialized {self.description}")
    
    async def connect(self) -> bool:
        """
        Establish required data connections for regime detection.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            # No direct connection needed as we rely on other feeds' data
            # We just need to ensure we have access to the feature store
            
            # Check if feature store is accessible
            if not hasattr(self, 'feature_store'):
                self.feature_store = {}
            
            self.connected = True
            self.logger.info("Successfully connected regime feed")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to connect regime feed: {str(e)}")
            self.connected = False
            return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from any resources.
        
        Returns:
            bool: True if disconnection is successful, False otherwise
        """
        try:
            # Save any cached data if needed
            self._save_regime_history()
            
            self.connected = False
            self.logger.info("Disconnected regime feed")
            return True
        
        except Exception as e:
            self.logger.error(f"Error during regime feed disconnection: {str(e)}")
            return False
    
    def _initialize_models(self) -> None:
        """Initialize models for regime detection."""
        for asset in self.assets:
            self.models[asset] = {}
            self.current_regimes[asset] = {}
            self.historical_regimes[asset] = {}
            self.transition_matrices[asset] = {}
            self.regime_performance[asset] = {}
            
            for timeframe in self.timeframes:
                self.models[asset][timeframe] = {
                    'hmm': self._create_hmm_model(),
                    'kmeans': self._create_kmeans_model(),
                    'gmm': self._create_gmm_model(),
                    'statistical': self._create_statistical_model(),
                    'domain': self._create_domain_model()
                }
                
                self.current_regimes[asset][timeframe] = None
                self.historical_regimes[asset][timeframe] = []
                self.transition_matrices[asset][timeframe] = self._initialize_transition_matrix()
                self.regime_performance[asset][timeframe] = {}
    
    def _create_hmm_model(self) -> hmm.GaussianHMM:
        """
        Create Hidden Markov Model for regime detection.
        
        Returns:
            hmm.GaussianHMM: Configured HMM model
        """
        # Create HMM with random initialization
        model = hmm.GaussianHMM(
            n_components=8,  # Number of regime states
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        return model
    
    def _create_kmeans_model(self) -> KMeans:
        """
        Create K-means clustering model for regime detection.
        
        Returns:
            KMeans: Configured K-means model
        """
        model = KMeans(
            n_clusters=8,  # Number of regime clusters
            random_state=42,
            n_init=10
        )
        return model
    
    def _create_gmm_model(self) -> GaussianMixture:
        """
        Create Gaussian Mixture Model for regime detection.
        
        Returns:
            GaussianMixture: Configured GMM model
        """
        model = GaussianMixture(
            n_components=8,  # Number of regime components
            covariance_type='full',
            random_state=42,
            n_init=10
        )
        return model
    
    def _create_statistical_model(self) -> Dict[str, Any]:
        """
        Create statistical model configuration for regime detection.
        
        Returns:
            Dict[str, Any]: Statistical model parameters
        """
        return {
            'lookback_short': 20,
            'lookback_medium': 50,
            'lookback_long': 100,
            'volatility_threshold_low': 0.5,
            'volatility_threshold_high': 2.0,
            'trend_threshold': 0.6,
            'last_results': None
        }
    
    def _create_domain_model(self) -> Dict[str, Any]:
        """
        Create domain-specific model for regime detection.
        
        Returns:
            Dict[str, Any]: Domain model parameters
        """
        return {
            'support_resistance_zones': [],
            'volume_profile_zones': [],
            'key_price_levels': [],
            'market_structure_points': [],
            'accumulation_threshold': 0.7,
            'distribution_threshold': 0.7,
            'exhaustion_threshold': 0.8,
            'last_results': None
        }
    
    def _initialize_transition_matrix(self) -> Dict[MarketRegimeType, Dict[MarketRegimeType, float]]:
        """
        Initialize transition probability matrix with equal probabilities.
        
        Returns:
            Dict: Nested dictionary for transition probabilities
        """
        regime_types = list(MarketRegimeType)
        n_regimes = len(regime_types)
        initial_prob = 1.0 / n_regimes
        
        matrix = {}
        for from_regime in regime_types:
            matrix[from_regime] = {to_regime: initial_prob for to_regime in regime_types}
        
        return matrix
    
    @periodic_reset(hours=24)
    def _save_regime_history(self) -> None:
        """Periodically save regime history for reference and recovery."""
        try:
            # In a production system, this would save to disk or database
            # For now, we just log it
            self.logger.info(f"Saving regime history for {len(self.historical_regimes)} assets")
        except Exception as e:
            self.logger.error(f"Error saving regime history: {str(e)}")
    
    async def fetch_data(self) -> List[Dict[str, Any]]:
        """
        Identify market regimes across configured assets and timeframes.
        
        Returns:
            List[Dict[str, Any]]: List of current market regime information
        """
        if not self.connected:
            await self.connect()
        
        regimes = []
        
        # Process each asset across all timeframes
        for asset in self.assets:
            for timeframe in self.timeframes:
                # Check if we need to update this asset/timeframe yet
                last_scan_key = f"{asset}_{timeframe}"
                last_scan = self.last_scans.get(last_scan_key, 0)
                
                if time.time() - last_scan < self.scan_interval:
                    # Use existing regime if we're not due for a refresh
                    if self.current_regimes[asset][timeframe]:
                        regimes.append(self._regime_to_dict(
                            self.current_regimes[asset][timeframe],
                            asset,
                            timeframe
                        ))
                    continue
                
                # Get feature data for this asset/timeframe
                features = await self._get_features(asset, timeframe)
                
                if not features or len(features) < self.lookback_periods[timeframe]:
                    self.logger.warning(f"Insufficient data for regime detection: {asset} {timeframe}")
                    continue
                
                # Analyze regimes
                try:
                    regime = await self._detect_regime(asset, timeframe, features)
                    
                    if regime:
                        # Store current regime
                        async with self.lock:
                            self.current_regimes[asset][timeframe] = regime
                            
                            # Add to historical if it's different from previous
                            if not self.historical_regimes[asset][timeframe] or \
                               self.historical_regimes[asset][timeframe][-1].regime_type != regime.regime_type:
                                # Close previous regime if it exists
                                if self.historical_regimes[asset][timeframe]:
                                    self.historical_regimes[asset][timeframe][-1].end_time = regime.start_time
                                
                                self.historical_regimes[asset][timeframe].append(regime)
                                
                                # Keep history size manageable
                                if len(self.historical_regimes[asset][timeframe]) > self.max_regimes:
                                    self.historical_regimes[asset][timeframe] = self.historical_regimes[asset][timeframe][-self.max_regimes:]
                        
                        # Update last scan time
                        self.last_scans[last_scan_key] = time.time()
                        
                        # Add to results
                        regimes.append(self._regime_to_dict(regime, asset, timeframe))
                
                except Exception as e:
                    self.logger.error(f"Error detecting regime for {asset} {timeframe}: {str(e)}")
        
        return regimes
    
    async def _get_features(self, asset: str, timeframe: str) -> pd.DataFrame:
        """
        Get feature data for regime detection.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe for analysis
        
        Returns:
            pd.DataFrame: Feature data for regime detection
        """
        # In a real system, this would get data from a feature service or database
        # For demonstration, we'll simulate features
        
        try:
            # Check if we have cached features
            feature_key = f"{asset}_{timeframe}"
            if feature_key in self.feature_store:
                return self.feature_store[feature_key]
            
            # In real implementation, we would fetch from feature service
            # For demonstration, create simulated features
            periods = self.lookback_periods[timeframe]
            
            # Create feature dataframe with simulated data
            dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq=timeframe)
            
            # Create base price series with some noise and trends
            np.random.seed(42)  # For reproducible results
            base = 100
            trend = np.linspace(0, 20, periods)
            noise = np.random.normal(0, 1, periods)
            cycle = 10 * np.sin(np.linspace(0, 4 * np.pi, periods))
            
            # Create different price series based on asset
            asset_hash = hash(asset) % 100
            asset_trend = trend * (1 + asset_hash / 200)
            asset_cycle = cycle * (1 + asset_hash / 100)
            
            # Generate prices
            prices = base + asset_trend + noise + asset_cycle
            
            # Create a DataFrame with OHLCV data
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': prices * (1 + np.random.uniform(0, 0.02, periods)),
                'low': prices * (1 - np.random.uniform(0, 0.02, periods)),
                'close': prices * (1 + np.random.normal(0, 0.005, periods)),
                'volume': np.random.uniform(1000, 5000, periods) * (1 + 0.5 * np.sin(np.linspace(0, 8 * np.pi, periods)))
            })
            
            # Calculate features
            df['returns'] = df['close'].pct_change().fillna(0)
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
            
            # Volatility features
            df['volatility'] = df['returns'].rolling(window=20).std().fillna(0)
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            
            # Trend features
            df['ma_20'] = df['close'].rolling(window=20).mean().fillna(method='bfill')
            df['ma_50'] = df['close'].rolling(window=50).mean().fillna(method='bfill')
            df['ma_20_50_ratio'] = df['ma_20'] / df['ma_50']
            
            # Volume features
            df['volume_ma'] = df['volume'].rolling(window=20).mean().fillna(method='bfill')
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Momentum features
            df['roc_5'] = df['close'].pct_change(periods=5).fillna(0)
            df['roc_20'] = df['close'].pct_change(periods=20).fillna(0)
            
            # Mean reversion features
            df['zscore_20'] = (df['close'] - df['ma_20']) / df['close'].rolling(window=20).std().fillna(0)
            
            # Set the timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Store in feature cache
            self.feature_store[feature_key] = df
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error getting features for {asset} {timeframe}: {str(e)}")
            return pd.DataFrame()
    
    async def _detect_regime(
        self, 
        asset: str, 
        timeframe: str, 
        features: pd.DataFrame
    ) -> Optional[MarketRegime]:
        """
        Detect the current market regime using multiple models.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe for analysis
            features: Feature data for regime detection
        
        Returns:
            Optional[MarketRegime]: Detected market regime or None if detection fails
        """
        try:
            # Use multiple detection methods
            regimes = {}
            confidence_weights = {}
            
            # 1. Hidden Markov Model
            hmm_regime, hmm_confidence = self._detect_with_hmm(asset, timeframe, features)
            if hmm_regime:
                regimes[hmm_regime] = regimes.get(hmm_regime, 0) + hmm_confidence
                confidence_weights['hmm'] = hmm_confidence
            
            # 2. K-means Clustering
            kmeans_regime, kmeans_confidence = self._detect_with_kmeans(asset, timeframe, features)
            if kmeans_regime:
                regimes[kmeans_regime] = regimes.get(kmeans_regime, 0) + kmeans_confidence
                confidence_weights['kmeans'] = kmeans_confidence
            
            # 3. Gaussian Mixture Model
            gmm_regime, gmm_confidence = self._detect_with_gmm(asset, timeframe, features)
            if gmm_regime:
                regimes[gmm_regime] = regimes.get(gmm_regime, 0) + gmm_confidence
                confidence_weights['gmm'] = gmm_confidence
            
            # 4. Statistical Analysis
            stat_regime, stat_confidence = self._detect_with_statistical(asset, timeframe, features)
            if stat_regime:
                regimes[stat_regime] = regimes.get(stat_regime, 0) + stat_confidence
                confidence_weights['statistical'] = stat_confidence
            
            # 5. Domain-Specific Rules
            domain_regime, domain_confidence = self._detect_with_domain(asset, timeframe, features)
            if domain_regime:
                regimes[domain_regime] = regimes.get(domain_regime, 0) + domain_confidence
                confidence_weights['domain'] = domain_confidence
            
            # If no regimes detected by any method, return None
            if not regimes:
                return None
            
            # Find regime with highest weighted confidence
            detected_regime = max(regimes.items(), key=lambda x: x[1])[0]
            total_confidence = sum(confidence_weights.values())
            weighted_confidence = regimes[detected_regime] / total_confidence if total_confidence > 0 else 0.5
            
            # Create regime object
            now = datetime.datetime.now()
            
            # Extract key features for the regime
            key_features = self._extract_regime_features(features, detected_regime)
            
            # Generate description
            description = self._generate_regime_description(
                detected_regime,
                asset,
                timeframe,
                key_features
            )
            
            # Get source models that contributed to this detection
            source_models = [model for model, conf in confidence_weights.items() if conf > 0]
            
            # Calculate transition probabilities from historical data
            transition_probs = self._calculate_transition_probabilities(
                asset,
                timeframe,
                detected_regime
            )
            
            # Calculate expected duration based on historical regimes
            expected_duration = self._calculate_expected_duration(
                asset,
                timeframe,
                detected_regime
            )
            
            # Get historical performance statistics for this regime
            historical_performance = self._get_regime_performance(
                asset,
                timeframe,
                detected_regime
            )
            
            # Create and return the regime object
            regime = MarketRegime(
                regime_type=detected_regime,
                start_time=now,
                end_time=None,  # Will be set when regime changes
                confidence=weighted_confidence,
                features=key_features,
                description=description,
                source_models=source_models,
                asset=asset,
                timeframe=timeframe,
                transition_probability=transition_probs,
                expected_duration=expected_duration,
                historical_performance=historical_performance
            )
            
            return regime
        
        except Exception as e:
            self.logger.error(f"Error in regime detection: {str(e)}")
            raise RegimeDetectionError(f"Failed to detect regime: {str(e)}")
    
    def _detect_with_hmm(
        self, 
        asset: str, 
        timeframe: str, 
        features: pd.DataFrame
    ) -> Tuple[Optional[MarketRegimeType], float]:
        """
        Detect regime using Hidden Markov Model.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe for analysis
            features: Feature data for regime detection
        
        Returns:
            Tuple[Optional[MarketRegimeType], float]: Detected regime and confidence
        """
        try:
            # Prepare feature data for HMM
            hmm_features = features[['returns', 'volatility', 'high_low_range', 'volume_ratio']].copy()
            
            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(hmm_features.values)
            
            # Train or update the model
            model = self.models[asset][timeframe]['hmm']
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(scaled_data)
            
            # Predict hidden states
            hidden_states = model.predict(scaled_data)
            
            # The most recent state is our regime
            current_state = hidden_states[-1]
            
            # Map states to regime types
            # This would typically be done based on analyzing the properties of each state
            # Here we use a simple mapping for demonstration
            state_regime_map = self._map_hmm_states_to_regimes(model, scaled_data, hidden_states)
            
            if current_state in state_regime_map:
                regime_type = state_regime_map[current_state]
                
                # Calculate confidence based on emission probability
                log_probs = model.score_samples(scaled_data)
                confidence = np.exp(log_probs[-1]) / np.exp(log_probs).max()
                confidence = min(max(confidence, 0.3), 0.9)  # Bound confidence
                
                return regime_type, confidence
            
            return None, 0.0
        
        except Exception as e:
            self.logger.error(f"Error in HMM regime detection: {str(e)}")
            return None, 0.0
    
    def _map_hmm_states_to_regimes(
        self, 
        model: hmm.GaussianHMM, 
        data: np.ndarray, 
        states: np.ndarray
    ) -> Dict[int, MarketRegimeType]:
        """
        Map HMM states to market regime types based on their properties.
        
        Args:
            model: Trained HMM model
            data: Scaled feature data
            states: Predicted hidden states
        
        Returns:
            Dict[int, MarketRegimeType]: Mapping from state indices to regime types
        """
        # Get means for each state
        state_means = model.means_
        
        # Extract means for relevant features
        # 0: returns, 1: volatility, 2: high_low_range, 3: volume_ratio
        returns_means = state_means[:, 0]
        volatility_means = state_means[:, 1]
        range_means = state_means[:, 2]
        volume_means = state_means[:, 3]
        
        # Calculate state frequencies
        state_counts = np.bincount(states, minlength=model.n_components)
        state_freqs = state_counts / state_counts.sum()
        
        # Create mapping from states to regimes
        state_regime_map = {}
        
        for i in range(model.n_components):
            ret = returns_means[i]
            vol = volatility_means[i]
            rng = range_means[i]
            vol_ratio = volume_means[i]
            
            # Determine regime type based on feature statistics
            if ret > 0.5 and vol > 0.5:
                regime = MarketRegimeType.BULL_VOLATILE
            elif ret > 0.5 and vol <= 0.5:
                regime = MarketRegimeType.BULL_TRENDING
            elif ret < -0.5 and vol > 0.5:
                regime = MarketRegimeType.BEAR_VOLATILE
            elif ret < -0.5 and vol <= 0.5:
                regime = MarketRegimeType.BEAR_TRENDING
            elif abs(ret) <= 0.5 and vol <= 0.3:
                regime = MarketRegimeType.SIDEWAYS
            elif abs(ret) <= 0.3 and vol > 0.7:
                regime = MarketRegimeType.HIGH_VOLATILITY
            elif vol_ratio > 1.5 and ret > 0:
                regime = MarketRegimeType.BREAKOUT
            elif vol_ratio > 1.5 and ret < 0:
                regime = MarketRegimeType.BREAKDOWN
            elif ret > 1.0 and vol_ratio > 2.0:
                regime = MarketRegimeType.EUPHORIA
            elif ret < -1.0 and vol_ratio > 2.0:
                regime = MarketRegimeType.CAPITULATION
            else:
                regime = MarketRegimeType.UNKNOWN
            
            state_regime_map[i] = regime
        
        return state_regime_map
    
    def _detect_with_kmeans(
        self, 
        asset: str, 
        timeframe: str, 
        features: pd.DataFrame
    ) -> Tuple[Optional[MarketRegimeType], float]:
        """
        Detect regime using K-means clustering.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe for analysis
            features: Feature data for regime detection
        
        Returns:
            Tuple[Optional[MarketRegimeType], float]: Detected regime and confidence
        """
        try:
            # Prepare feature data for K-means
            kmeans_features = features[['returns', 'volatility', 'ma_20_50_ratio', 'volume_ratio', 'roc_5', 'zscore_20']].copy()
            
            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(kmeans_features.values)
            
            # Train or update the model
            model = self.models[asset][timeframe]['kmeans']
            model.fit(scaled_data)
            
            # Predict clusters
            clusters = model.predict(scaled_data)
            current_cluster = clusters[-1]
            
            # Calculate distance to cluster center for confidence
            centers = model.cluster_centers_
            current_point = scaled_data[-1].reshape(1, -1)
            distances = np.linalg.norm(centers - current_point, axis=1)
            closest_dist = distances[current_cluster]
            
            # Normalize to a confidence value (closer = more confident)
            max_dist = np.max(distances)
            confidence = 1.0 - (closest_dist / max_dist if max_dist > 0 else 0)
            confidence = min(max(confidence, 0.3), 0.9)  # Bound confidence
            
            # Map clusters to regime types
            cluster_regime_map = self._map_clusters_to_regimes(model, scaled_data, clusters)
            
            if current_cluster in cluster_regime_map:
                regime_type = cluster_regime_map[current_cluster]
                return regime_type, confidence
            
            return None, 0.0
        
        except Exception as e:
            self.logger.error(f"Error in K-means regime detection: {str(e)}")
            return None, 0.0
    
    def _map_clusters_to_regimes(
        self, 
        model: KMeans, 
        data: np.ndarray, 
        clusters: np.ndarray
    ) -> Dict[int, MarketRegimeType]:
        """
        Map K-means clusters to market regime types based on their properties.
        
        Args:
            model: Trained K-means model
            data: Scaled feature data
            clusters: Predicted cluster assignments
        
        Returns:
            Dict[int, MarketRegimeType]: Mapping from cluster indices to regime types
        """
        # Get cluster centers
        centers = model.cluster_centers_
        
        # Extract center values for relevant features
        # 0: returns, 1: volatility, 2: ma_20_50_ratio, 3: volume_ratio, 4: roc_5, 5: zscore_20
        returns_centers = centers[:, 0]
        volatility_centers = centers[:, 1]
        ma_ratio_centers = centers[:, 2]
        volume_centers = centers[:, 3]
        roc_centers = centers[:, 4]
        zscore_centers = centers[:, 5]
        
        # Create mapping from clusters to regimes
        cluster_regime_map = {}
        
        for i in range(model.n_clusters):
            ret = returns_centers[i]
            vol = volatility_centers[i]
            ma_ratio = ma_ratio_centers[i]
            vol_ratio = volume_centers[i]
            roc = roc_centers[i]
            zscore = zscore_centers[i]
            
            # Determine regime type based on center statistics
            if ma_ratio > 1.02 and ret > 0.3:
                regime = MarketRegimeType.BULL_TRENDING
            elif ma_ratio < 0.98 and ret < -0.3:
                regime = MarketRegimeType.BEAR_TRENDING
            elif abs(ma_ratio - 1.0) < 0.02 and vol < 0.5:
                regime = MarketRegimeType.SIDEWAYS
            elif vol > 1.5:
                regime = MarketRegimeType.HIGH_VOLATILITY
            elif vol < 0.5:
                regime = MarketRegimeType.LOW_VOLATILITY
            elif ma_ratio > 1.05 and vol > 1.0:
                regime = MarketRegimeType.BULL_VOLATILE
            elif ma_ratio < 0.95 and vol > 1.0:
                regime = MarketRegimeType.BEAR_VOLATILE
            elif zscore > 2.0 and roc < 0:
                regime = MarketRegimeType.BULL_EXHAUSTION
            elif zscore < -2.0 and roc > 0:
                regime = MarketRegimeType.BEAR_EXHAUSTION
            elif vol_ratio > 2.0 and ma_ratio > 1.0:
                regime = MarketRegimeType.BREAKOUT
            elif vol_ratio > 2.0 and ma_ratio < 1.0:
                regime = MarketRegimeType.BREAKDOWN
            else:
                regime = MarketRegimeType.UNKNOWN
            
            cluster_regime_map[i] = regime
        
        return cluster_regime_map
    
    def _detect_with_gmm(
        self, 
        asset: str, 
        timeframe: str, 
        features: pd.DataFrame
    ) -> Tuple[Optional[MarketRegimeType], float]:
        """
        Detect regime using Gaussian Mixture Model.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe for analysis
            features: Feature data for regime detection
        
        Returns:
            Tuple[Optional[MarketRegimeType], float]: Detected regime and confidence
        """
        try:
            # Prepare feature data for GMM
            gmm_features = features[['returns', 'volatility', 'high_low_range', 'ma_20_50_ratio', 'volume_ratio']].copy()
            
            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(gmm_features.values)
            
            # Reduce dimensionality for better GMM performance
            pca = PCA(n_components=3)
            pca_data = pca.fit_transform(scaled_data)
            
            # Train or update the model
            model = self.models[asset][timeframe]['gmm']
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(pca_data)
            
            # Predict mixture components
            components = model.predict(pca_data)
            current_component = components[-1]
            
            # Calculate probability for confidence
            probs = model.predict_proba(pca_data)
            current_prob = probs[-1, current_component]
            
            # Use probability as confidence
            confidence = min(max(current_prob, 0.3), 0.9)  # Bound confidence
            
            # Map components to regime types
            component_regime_map = self._map_gmm_components_to_regimes(model, pca_data, components, pca)
            
            if current_component in component_regime_map:
                regime_type = component_regime_map[current_component]
                return regime_type, confidence
            
            return None, 0.0
        
        except Exception as e:
            self.logger.error(f"Error in GMM regime detection: {str(e)}")
            return None, 0.0
    
    def _map_gmm_components_to_regimes(
        self, 
        model: GaussianMixture, 
        data: np.ndarray, 
        components: np.ndarray,
        pca: PCA
    ) -> Dict[int, MarketRegimeType]:
        """
        Map GMM components to market regime types based on their properties.
        
        Args:
            model: Trained GMM model
            data: PCA-transformed feature data
            components: Predicted component assignments
            pca: Fitted PCA transformer
        
        Returns:
            Dict[int, MarketRegimeType]: Mapping from component indices to regime types
        """
        # Get component means in original space
        pca_means = model.means_
        orig_means = pca.inverse_transform(pca_means)
        
        # Extract means for relevant features
        # 0: returns, 1: volatility, 2: high_low_range, 3: ma_20_50_ratio, 4: volume_ratio
        returns_means = orig_means[:, 0]
        volatility_means = orig_means[:, 1]
        range_means = orig_means[:, 2]
        ma_ratio_means = orig_means[:, 3]
        volume_means = orig_means[:, 4]
        
        # Calculate component frequencies
        component_counts = np.bincount(components, minlength=model.n_components)
        component_freqs = component_counts / component_counts.sum()
        
        # Create mapping from components to regimes
        component_regime_map = {}
        
        for i in range(model.n_components):
            ret = returns_means[i]
            vol = volatility_means[i]
            rng = range_means[i]
            ma_ratio = ma_ratio_means[i]
            vol_ratio = volume_means[i]
            freq = component_freqs[i]
            
            # Determine regime type based on feature statistics
            if ret > 0.5 and ma_ratio > 1.05:
                regime = MarketRegimeType.BULL_TRENDING
            elif ret < -0.5 and ma_ratio < 0.95:
                regime = MarketRegimeType.BEAR_TRENDING
            elif abs(ret) < 0.3 and abs(ma_ratio - 1.0) < 0.03 and vol < 0.7:
                regime = MarketRegimeType.SIDEWAYS
            elif vol > 1.5 and rng > 1.5:
                regime = MarketRegimeType.HIGH_VOLATILITY
            elif vol < 0.5 and rng < 0.5:
                regime = MarketRegimeType.LOW_VOLATILITY
            elif ret > 0.3 and vol > 1.2:
                regime = MarketRegimeType.BULL_VOLATILE
            elif ret < -0.3 and vol > 1.2:
                regime = MarketRegimeType.BEAR_VOLATILE
            elif ret > 0.7 and vol_ratio > 1.5 and ma_ratio > 1.1:
                regime = MarketRegimeType.EUPHORIA
            elif ret < -0.7 and vol_ratio > 1.5 and ma_ratio < 0.9:
                regime = MarketRegimeType.CAPITULATION
            elif freq < 0.05 and vol_ratio > 2.0:
                # Rare state with high volume
                if ret > 0:
                    regime = MarketRegimeType.BREAKOUT
                else:
                    regime = MarketRegimeType.BREAKDOWN
            elif freq > 0.3:
                # Most common state is likely a base/accumulation state
                regime = MarketRegimeType.ACCUMULATION
            else:
                regime = MarketRegimeType.UNKNOWN
            
            component_regime_map[i] = regime
        
        return component_regime_map
    
    def _detect_with_statistical(
        self, 
        asset: str, 
        timeframe: str, 
        features: pd.DataFrame
    ) -> Tuple[Optional[MarketRegimeType], float]:
        """
        Detect regime using statistical methods.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe for analysis
            features: Feature data for regime detection
        
        Returns:
            Tuple[Optional[MarketRegimeType], float]: Detected regime and confidence
        """
        try:
            model_params = self.models[asset][timeframe]['statistical']
            
            # Calculate statistical indicators
            lookback_short = model_params['lookback_short']
            lookback_medium = model_params['lookback_medium']
            lookback_long = model_params['lookback_long']
            
            # Get recent data
            recent_data = features.tail(lookback_long)
            
            # Calculate moving averages
            ma_short = recent_data['close'].tail(lookback_short).mean()
            ma_medium = recent_data['close'].tail(lookback_medium).mean()
            ma_long = recent_data['close'].tail(lookback_long).mean()
            
            # Calculate volatility
            recent_volatility = recent_data['returns'].tail(lookback_short).std()
            medium_volatility = recent_data['returns'].tail(lookback_medium).std()
            
            # Calculate trend strength
            returns = recent_data['returns'].tail(lookback_short)
            positive_days = (returns > 0).sum()
            negative_days = (returns < 0).sum()
            trend_strength = abs(positive_days - negative_days) / (positive_days + negative_days) if (positive_days + negative_days) > 0 else 0
            
            # Determine if trending or mean-reverting
            # Perform Augmented Dickey-Fuller test for stationarity
            adf_result = adfuller(recent_data['close'].tail(lookback_medium).values)
            adf_pvalue = adf_result[1]
            
            # Mean-reverting if stationary (p < 0.05)
            is_mean_reverting = adf_pvalue < 0.05
            
            # Calculate volume profile
            recent_volume = recent_data['volume'].tail(lookback_short).mean()
            medium_volume = recent_data['volume'].tail(lookback_medium).mean()
            volume_ratio = recent_volume / medium_volume if medium_volume > 0 else 1.0
            
            # Trend direction
            uptrend = ma_short > ma_medium and ma_medium > ma_long
            downtrend = ma_short < ma_medium and ma_medium < ma_long
            
            # Combine signals to determine regime
            if uptrend and trend_strength > model_params['trend_threshold'] and not is_mean_reverting:
                if recent_volatility > model_params['volatility_threshold_high']:
                    regime = MarketRegimeType.BULL_VOLATILE
                else:
                    regime = MarketRegimeType.BULL_TRENDING
            elif downtrend and trend_strength > model_params['trend_threshold'] and not is_mean_reverting:
                if recent_volatility > model_params['volatility_threshold_high']:
                    regime = MarketRegimeType.BEAR_VOLATILE
                else:
                    regime = MarketRegimeType.BEAR_TRENDING
            elif is_mean_reverting and trend_strength < model_params['trend_threshold']:
                regime = MarketRegimeType.SIDEWAYS
            elif recent_volatility > model_params['volatility_threshold_high'] * 1.5:
                regime = MarketRegimeType.HIGH_VOLATILITY
            elif recent_volatility < model_params['volatility_threshold_low']:
                regime = MarketRegimeType.LOW_VOLATILITY
            elif volume_ratio > 2.0 and uptrend and ma_short > ma_medium * 1.05:
                regime = MarketRegimeType.BREAKOUT
            elif volume_ratio > 2.0 and downtrend and ma_short < ma_medium * 0.95:
                regime = MarketRegimeType.BREAKDOWN
            elif uptrend and recent_volatility > medium_volatility * 1.5 and volume_ratio > 1.5:
                regime = MarketRegimeType.BULL_EXHAUSTION
            elif downtrend and recent_volatility > medium_volatility * 1.5 and volume_ratio > 1.5:
                regime = MarketRegimeType.BEAR_EXHAUSTION
            else:
                regime = MarketRegimeType.UNKNOWN
            
            # Calculate confidence based on strength of signals
            confidence_factors = [
                min(1.0, trend_strength / 0.8) * 0.5,  # 50% from trend strength
                min(1.0, volume_ratio / 3.0) * 0.3,    # 30% from volume confirmation
                min(1.0, abs(ma_short / ma_medium - 1.0) * 20) * 0.2  # 20% from MA separation
            ]
            
            confidence = sum(confidence_factors)
            confidence = min(max(confidence, 0.3), 0.9)  # Bound confidence
            
            # Store results for later reference
            model_params['last_results'] = {
                'ma_short': ma_short,
                'ma_medium': ma_medium,
                'ma_long': ma_long,
                'recent_volatility': recent_volatility,
                'medium_volatility': medium_volatility,
                'trend_strength': trend_strength,
                'is_mean_reverting': is_mean_reverting,
                'volume_ratio': volume_ratio,
                'uptrend': uptrend,
                'downtrend': downtrend
            }
            
            return regime, confidence
        
        except Exception as e:
            self.logger.error(f"Error in statistical regime detection: {str(e)}")
            return None, 0.0
    
    def _detect_with_domain(
        self, 
        asset: str, 
        timeframe: str, 
        features: pd.DataFrame
    ) -> Tuple[Optional[MarketRegimeType], float]:
        """
        Detect regime using domain-specific rules and market structure analysis.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe for analysis
            features: Feature data for regime detection
        
        Returns:
            Tuple[Optional[MarketRegimeType], float]: Detected regime and confidence
        """
        try:
            model_params = self.models[asset][timeframe]['domain']
            
            # Get recent data
            recent_data = features.tail(50)
            
            # Identify market structure
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            closes = recent_data['close'].values
            volumes = recent_data['volume'].values
            
            # Detect swing points
            swing_highs = self._detect_swing_highs(highs, 5)
            swing_lows = self._detect_swing_lows(lows, 5)
            
            # Identify higher highs, higher lows, lower highs, lower lows
            higher_highs = self._check_higher_highs(swing_highs)
            higher_lows = self._check_higher_lows(swing_lows)
            lower_highs = self._check_lower_highs(swing_highs)
            lower_lows = self._check_lower_lows(swing_lows)
            
            # Determine market structure
            uptrend_structure = higher_highs and higher_lows
            downtrend_structure = lower_highs and lower_lows
            
            # Detect volume patterns
            rising_volume = volumes[-10:].mean() > volumes[-20:-10].mean()
            falling_volume = volumes[-10:].mean() < volumes[-20:-10].mean()
            volume_climax = volumes[-5:].max() > volumes[-30:].max() * 1.5
            
            # Detect price patterns
            # Narrow range periods
            ranges = highs - lows
            avg_range = np.mean(ranges[-20:-5])
            recent_ranges = ranges[-5:]
            narrow_range = np.mean(recent_ranges) < avg_range * 0.7
            
            # Wide range periods
            wide_range = np.mean(recent_ranges) > avg_range * 1.3
            
            # Detect accumulation/distribution
            price_change = (closes[-1] - closes[-10]) / closes[-10]
            volume_change = (volumes[-10:].sum() - volumes[-20:-10].sum()) / volumes[-20:-10].sum()
            
            accumulation = abs(price_change) < 0.03 and volume_change > 0.2
            distribution = abs(price_change) < 0.03 and volume_change < -0.2
            
            # Detect exhaustion moves
            last_day_range = (highs[-1] - lows[-1]) / closes[-1]
            avg_day_range = np.mean((highs[-6:-1] - lows[-6:-1]) / closes[-6:-1])
            last_day_volume = volumes[-1]
            avg_volume = np.mean(volumes[-6:-1])
            
            exhaustion_up = closes[-1] > closes[-2] * 1.03 and last_day_range > avg_day_range * 1.5 and last_day_volume > avg_volume * 2
            exhaustion_down = closes[-1] < closes[-2] * 0.97 and last_day_range > avg_day_range * 1.5 and last_day_volume > avg_volume * 2
            
            # Detect rotation
            sector_rotation = False  # This would require cross-asset analysis
            
            # Combine signals to determine regime
            if uptrend_structure and rising_volume:
                if volume_climax and exhaustion_up:
                    regime = MarketRegimeType.BULL_EXHAUSTION
                elif narrow_range:
                    regime = MarketRegimeType.BULL_VOLATILE
                else:
                    regime = MarketRegimeType.BULL_TRENDING
            elif downtrend_structure and rising_volume:
                if volume_climax and exhaustion_down:
                    regime = MarketRegimeType.BEAR_EXHAUSTION
                elif narrow_range:
                    regime = MarketRegimeType.BEAR_VOLATILE
                else:
                    regime = MarketRegimeType.BEAR_TRENDING
            elif accumulation and volume_change > model_params['accumulation_threshold']:
                regime = MarketRegimeType.ACCUMULATION
            elif distribution and abs(volume_change) > model_params['distribution_threshold']:
                regime = MarketRegimeType.DISTRIBUTION
            elif narrow_range and falling_volume:
                regime = MarketRegimeType.SIDEWAYS
            elif wide_range and volume_climax:
                if closes[-1] > closes[-2]:
                    regime = MarketRegimeType.BREAKOUT
                else:
                    regime = MarketRegimeType.BREAKDOWN
            elif sector_rotation:
                regime = MarketRegimeType.ROTATION
            elif volume_climax and exhaustion_up and price_change > 0.1:
                regime = MarketRegimeType.EUPHORIA
            elif volume_climax and exhaustion_down and price_change < -0.1:
                regime = MarketRegimeType.CAPITULATION
            else:
                regime = MarketRegimeType.UNKNOWN
            
            # Calculate confidence based on strength of signals
            confidence_factors = []
            
            if uptrend_structure or downtrend_structure:
                confidence_factors.append(0.7)  # Strong structure signal
            
            if volume_climax:
                confidence_factors.append(0.6)  # Strong volume signal
            
            if narrow_range or wide_range:
                confidence_factors.append(0.5)  # Clear range pattern
            
            if accumulation or distribution:
                confidence_factors.append(0.6)  # Clear accumulation/distribution
            
            if exhaustion_up or exhaustion_down:
                confidence_factors.append(0.7)  # Clear exhaustion
            
            # If we have no confidence factors, use a default medium confidence
            if not confidence_factors:
                confidence = 0.5
            else:
                confidence = sum(confidence_factors) / len(confidence_factors)
            
            confidence = min(max(confidence, 0.3), 0.9)  # Bound confidence
            
            # Store market structure for later reference
            model_params['market_structure_points'] = {
                'swing_highs': swing_highs,
                'swing_lows': swing_lows,
                'uptrend_structure': uptrend_structure,
                'downtrend_structure': downtrend_structure
            }
            
            # Store results for later reference
            model_params['last_results'] = {
                'uptrend_structure': uptrend_structure,
                'downtrend_structure': downtrend_structure,
                'rising_volume': rising_volume,
                'falling_volume': falling_volume,
                'volume_climax': volume_climax,
                'narrow_range': narrow_range,
                'wide_range': wide_range,
                'accumulation': accumulation,
                'distribution': distribution,
                'exhaustion_up': exhaustion_up,
                'exhaustion_down': exhaustion_down
            }
            
            return regime, confidence
        
        except Exception as e:
            self.logger.error(f"Error in domain-specific regime detection: {str(e)}")
            return None, 0.0
    
    def _detect_swing_highs(self, prices: np.ndarray, window: int = 5) -> List[Tuple[int, float]]:
        """
        Detect swing high points in price series.
        
        Args:
            prices: Array of price values
            window: Window size for swing point detection
        
        Returns:
            List[Tuple[int, float]]: List of (index, price) tuples for swing highs
        """
        swing_highs = []
        
        for i in range(window, len(prices) - window):
            left_higher = all(prices[i] > prices[i-j] for j in range(1, window+1))
            right_higher = all(prices[i] > prices[i+j] for j in range(1, window+1))
            
            if left_higher and right_higher:
                swing_highs.append((i, prices[i]))
        
        return swing_highs
    
    def _detect_swing_lows(self, prices: np.ndarray, window: int = 5) -> List[Tuple[int, float]]:
        """
        Detect swing low points in price series.
        
        Args:
            prices: Array of price values
            window: Window size for swing point detection
        
        Returns:
            List[Tuple[int, float]]: List of (index, price) tuples for swing lows
        """
        swing_lows = []
        
        for i in range(window, len(prices) - window):
            left_lower = all(prices[i] < prices[i-j] for j in range(1, window+1))
            right_lower = all(prices[i] < prices[i+j] for j in range(1, window+1))
            
            if left_lower and right_lower:
                swing_lows.append((i, prices[i]))
        
        return swing_lows
    
    def _check_higher_highs(self, swing_highs: List[Tuple[int, float]]) -> bool:
        """
        Check if recent swing highs are making higher highs.
        
        Args:
            swing_highs: List of swing high points
        
        Returns:
            bool: True if making higher highs, False otherwise
        """
        if len(swing_highs) < 2:
            return False
        
        # Get last few swing highs sorted by index
        recent_swing_highs = sorted(swing_highs[-3:])
        
        # Check if they're increasing in price
        for i in range(1, len(recent_swing_highs)):
            if recent_swing_highs[i][1] <= recent_swing_highs[i-1][1]:
                return False
        
        return True
    
    def _check_higher_lows(self, swing_lows: List[Tuple[int, float]]) -> bool:
        """
        Check if recent swing lows are making higher lows.
        
        Args:
            swing_lows: List of swing low points
        
        Returns:
            bool: True if making higher lows, False otherwise
        """
        if len(swing_lows) < 2:
            return False
        
        # Get last few swing lows sorted by index
        recent_swing_lows = sorted(swing_lows[-3:])
        
        # Check if they're increasing in price
        for i in range(1, len(recent_swing_lows)):
            if recent_swing_lows[i][1] <= recent_swing_lows[i-1][1]:
                return False
        
        return True
    
    def _check_lower_highs(self, swing_highs: List[Tuple[int, float]]) -> bool:
        """
        Check if recent swing highs are making lower highs.
        
        Args:
            swing_highs: List of swing high points
        
        Returns:
            bool: True if making lower highs, False otherwise
        """
        if len(swing_highs) < 2:
            return False
        
        # Get last few swing highs sorted by index
        recent_swing_highs = sorted(swing_highs[-3:])
        
        # Check if they're decreasing in price
        for i in range(1, len(recent_swing_highs)):
            if recent_swing_highs[i][1] >= recent_swing_highs[i-1][1]:
                return False
        
        return True
    
    def _check_lower_lows(self, swing_lows: List[Tuple[int, float]]) -> bool:
        """
        Check if recent swing lows are making lower lows.
        
        Args:
            swing_lows: List of swing low points
        
        Returns:
            bool: True if making lower lows, False otherwise
        """
        if len(swing_lows) < 2:
            return False
        
        # Get last few swing lows sorted by index
        recent_swing_lows = sorted(swing_lows[-3:])
        
        # Check if they're decreasing in price
        for i in range(1, len(recent_swing_lows)):
            if recent_swing_lows[i][1] >= recent_swing_lows[i-1][1]:
                return False
        
        return True
    
    def _extract_regime_features(
        self, 
        features: pd.DataFrame, 
        regime_type: MarketRegimeType
    ) -> Dict[str, float]:
        """
        Extract key features that define the detected regime.
        
        Args:
            features: Feature data
            regime_type: Detected regime type
        
        Returns:
            Dict[str, float]: Key features with values
        """
        # Get the most recent values for key features
        recent = features.iloc[-10:]
        
        # Calculate key metrics for this regime
        key_features = {
            'avg_return': recent['returns'].mean(),
            'volatility': recent['volatility'].iloc[-1],
            'range_ratio': recent['high_low_range'].mean(),
            'ma_ratio': recent['ma_20_50_ratio'].iloc[-1] if 'ma_20_50_ratio' in recent else 1.0,
            'volume_ratio': recent['volume_ratio'].mean() if 'volume_ratio' in recent else 1.0,
            'roc_5': recent['roc_5'].iloc[-1] if 'roc_5' in recent else 0.0,
            'zscore': recent['zscore_20'].iloc[-1] if 'zscore_20' in recent else 0.0
        }
        
        return key_features
    
    def _generate_regime_description(
        self, 
        regime_type: MarketRegimeType, 
        asset: str, 
        timeframe: str,
        features: Dict[str, float]
    ) -> str:
        """
        Generate a human-readable description of the detected regime.
        
        Args:
            regime_type: Detected regime type
            asset: Asset symbol
            timeframe: Timeframe for analysis
            features: Key features of the regime
        
        Returns:
            str: Human-readable description
        """
        # Get basic regime description
        base_descriptions = {
            MarketRegimeType.BULL_TRENDING: "Strong bullish trend with consistent upward movement",
            MarketRegimeType.BEAR_TRENDING: "Strong bearish trend with consistent downward movement",
            MarketRegimeType.SIDEWAYS: "Sideways consolidation with limited directional movement",
            MarketRegimeType.HIGH_VOLATILITY: "High volatility environment with large price swings",
            MarketRegimeType.LOW_VOLATILITY: "Low volatility environment with compressed price action",
            MarketRegimeType.BULL_VOLATILE: "Bullish trend with high volatility and sharp moves",
            MarketRegimeType.BEAR_VOLATILE: "Bearish trend with high volatility and sharp moves",
            MarketRegimeType.BULL_EXHAUSTION: "Potential exhaustion of bullish trend with signs of reversal",
            MarketRegimeType.BEAR_EXHAUSTION: "Potential exhaustion of bearish trend with signs of reversal",
            MarketRegimeType.BREAKOUT: "Price breakout with strong momentum and increased participation",
            MarketRegimeType.BREAKDOWN: "Price breakdown with strong momentum and increased participation",
            MarketRegimeType.ROTATION: "Market rotation between sectors or asset classes",
            MarketRegimeType.LIQUIDITY_CRISIS: "Liquidity crisis with extreme volatility and correlated moves",
            MarketRegimeType.ACCUMULATION: "Accumulation phase with sideways price and increasing volume",
            MarketRegimeType.DISTRIBUTION: "Distribution phase with sideways price and increasing volume",
            MarketRegimeType.EUPHORIA: "Market euphoria with extreme bullish sentiment and overextension",
            MarketRegimeType.CAPITULATION: "Market capitulation with extreme bearish sentiment and selling",
            MarketRegimeType.RECOVERY: "Recovery phase after a significant decline",
            MarketRegimeType.UNKNOWN: "Undefined market regime with mixed signals"
        }
        
        base_desc = base_descriptions.get(regime_type, "Unknown market regime")
        
        # Add feature-specific details
        details = []
        
        if 'avg_return' in features:
            ret = features['avg_return']
            if abs(ret) > 0.01:
                details.append(f"average returns of {ret:.2%} per period")
        
        if 'volatility' in features:
            vol = features['volatility']
            if vol > 0.02:
                details.append(f"high volatility ({vol:.2%})")
            elif vol < 0.005:
                details.append(f"low volatility ({vol:.2%})")
        
        if 'volume_ratio' in features:
            vol_ratio = features['volume_ratio']
            if vol_ratio > 1.5:
                details.append(f"above-average volume ({vol_ratio:.1f}x)")
            elif vol_ratio < 0.7:
                details.append(f"below-average volume ({vol_ratio:.1f}x)")
        
        # Create time-specific context
        time_context = f"on the {timeframe} timeframe"
        
        # Put it all together
        if details:
            detail_str = " with " + ", ".join(details)
            description = f"{base_desc}{detail_str} for {asset} {time_context}"
        else:
            description = f"{base_desc} for {asset} {time_context}"
        
        return description
    
    def _calculate_transition_probabilities(
        self, 
        asset: str, 
        timeframe: str, 
        current_regime: MarketRegimeType
    ) -> Dict[MarketRegimeType, float]:
        """
        Calculate transition probabilities from current regime to other regimes.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe for analysis
            current_regime: Current market regime
        
        Returns:
            Dict[MarketRegimeType, float]: Transition probabilities to each regime
        """
        # Get historical regimes for this asset/timeframe
        history = self.historical_regimes.get(asset, {}).get(timeframe, [])
        
        # If we don't have enough history, use default transition matrix
        if len(history) < 5:
            return self.transition_matrices[asset][timeframe].get(current_regime, {})
        
        # Count transitions from current regime to other regimes
        transitions = {}
        total_transitions = 0
        
        for i in range(len(history) - 1):
            from_regime = history[i].regime_type
            to_regime = history[i+1].regime_type
            
            if from_regime == current_regime:
                transitions[to_regime] = transitions.get(to_regime, 0) + 1
                total_transitions += 1
        
        # Calculate probabilities
        probabilities = {}
        all_regimes = list(MarketRegimeType)
        
        for regime in all_regimes:
            if total_transitions > 0:
                probabilities[regime] = transitions.get(regime, 0) / total_transitions
            else:
                # If no transitions observed, use default probability
                probabilities[regime] = 0.1 if regime == current_regime else 0.05
        
        # Ensure probabilities sum to 1
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            for regime in probabilities:
                probabilities[regime] /= total_prob
        
        return probabilities
    
    def _calculate_expected_duration(
        self, 
        asset: str, 
        timeframe: str, 
        regime_type: MarketRegimeType
    ) -> Optional[float]:
        """
        Calculate expected duration for the current regime based on historical data.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe for analysis
            regime_type: Current regime type
        
        Returns:
            Optional[float]: Expected duration in periods, None if unknown
        """
        # Get historical regimes for this asset/timeframe
        history = self.historical_regimes.get(asset, {}).get(timeframe, [])
        
        # If we don't have enough history, return None
        if len(history) < 3:
            return None
        
        # Calculate durations of past regimes of the same type
        durations = []
        
        for regime in history:
            if regime.regime_type == regime_type and regime.end_time:
                duration = (regime.end_time - regime.start_time).total_seconds()
                
                # Convert to periods based on timeframe
                if timeframe.endswith('m'):
                    minutes = int(timeframe[:-1])
                    periods = duration / (minutes * 60)
                elif timeframe.endswith('h'):
                    hours = int(timeframe[:-1])
                    periods = duration / (hours * 3600)
                elif timeframe.endswith('d'):
                    periods = duration / (24 * 3600)
                elif timeframe.endswith('w'):
                    periods = duration / (7 * 24 * 3600)
                else:
                    periods = duration / 3600  # Default to hours
                
                durations.append(periods)
        
        # Calculate average duration if we have data
        if durations:
            return sum(durations) / len(durations)
        
        return None
    
    def _get_regime_performance(
        self, 
        asset: str, 
        timeframe: str, 
        regime_type: MarketRegimeType
    ) -> Dict[str, float]:
        """
        Get historical performance statistics for the specified regime.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe for analysis
            regime_type: Regime type
        
        Returns:
            Dict[str, float]: Performance statistics
        """
        # Check if we have cached performance data
        if asset in self.regime_performance and \
           timeframe in self.regime_performance[asset] and \
           regime_type in self.regime_performance[asset][timeframe]:
            return self.regime_performance[asset][timeframe][regime_type]
        
        # Default performance metrics
        performance = {
            'win_rate': 0.0,
            'avg_return': 0.0,
            'max_return': 0.0,
            'min_return': 0.0,
            'sharpe_ratio': 0.0,
            'volatility': 0.0,
            'max_drawdown': 0.0,
            'samples': 0
        }
        
        # In a real system, this would calculate these statistics
        # from historical data where this regime was active
        # For demonstration, we provide simulated statistics
        
        # Different regimes have different performance characteristics
        if regime_type == MarketRegimeType.BULL_TRENDING:
            performance.update({
                'win_rate': 0.78,
                'avg_return': 0.025,
                'max_return': 0.15,
                'min_return': -0.05,
                'sharpe_ratio': 2.1,
                'volatility': 0.012,
                'max_drawdown': 0.07,
                'samples': 42
            })
        elif regime_type == MarketRegimeType.BEAR_TRENDING:
            performance.update({
                'win_rate': 0.25,
                'avg_return': -0.02,
                'max_return': 0.08,
                'min_return': -0.18,
                'sharpe_ratio': -1.8,
                'volatility': 0.025,
                'max_drawdown': 0.22,
                'samples': 38
            })
        elif regime_type == MarketRegimeType.SIDEWAYS:
            performance.update({
                'win_rate': 0.52,
                'avg_return': 0.003,
                'max_return': 0.05,
                'min_return': -0.05,
                'sharpe_ratio': 0.4,
                'volatility': 0.008,
                'max_drawdown': 0.04,
                'samples': 65
            })
        elif regime_type == MarketRegimeType.HIGH_VOLATILITY:
            performance.update({
                'win_rate': 0.45,
                'avg_return': -0.005,
                'max_return': 0.25,
                'min_return': -0.28,
                'sharpe_ratio': -0.2,
                'volatility': 0.035,
                'max_drawdown': 0.18,
                'samples': 30
            })
        elif regime_type == MarketRegimeType.BREAKOUT:
            performance.update({
                'win_rate': 0.82,
                'avg_return': 0.045,
                'max_return': 0.22,
                'min_return': -0.08,
                'sharpe_ratio': 2.8,
                'volatility': 0.018,
                'max_drawdown': 0.05,
                'samples': 22
            })
        else:
            # For other regimes, use moderate default values
            performance.update({
                'win_rate': 0.55,
                'avg_return': 0.01,
                'max_return': 0.12,
                'min_return': -0.10,
                'sharpe_ratio': 0.8,
                'volatility': 0.015,
                'max_drawdown': 0.08,
                'samples': 25
            })
        
        # Cache the performance data
        if asset not in self.regime_performance:
            self.regime_performance[asset] = {}
        
        if timeframe not in self.regime_performance[asset]:
            self.regime_performance[asset][timeframe] = {}
        
        self.regime_performance[asset][timeframe][regime_type] = performance
        
        return performance
    
    def _regime_to_dict(
        self, 
        regime: MarketRegime, 
        asset: str, 
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Convert regime object to dictionary for API response.
        
        Args:
            regime: MarketRegime object
            asset: Asset symbol
            timeframe: Timeframe for analysis
        
        Returns:
            Dict[str, Any]: Regime information as dictionary
        """
        regime_dict = {
            'regime_type': regime.regime_type.name,
            'start_time': regime.start_time.isoformat() if regime.start_time else None,
            'end_time': regime.end_time.isoformat() if regime.end_time else None,
            'confidence': regime.confidence,
            'description': regime.description,
            'asset': asset,
            'timeframe': timeframe,
            'features': regime.features,
            'source_models': regime.source_models
        }
        
        # Add transition probabilities if available
        if regime.transition_probability:
            regime_dict['transition_probability'] = {
                k.name: v for k, v in regime.transition_probability.items()
            }
        
        # Add expected duration if available
        if regime.expected_duration is not None:
            regime_dict['expected_duration'] = regime.expected_duration
        
        # Add historical performance if available
        if regime.historical_performance:
            regime_dict['historical_performance'] = regime.historical_performance
        
        return regime_dict
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the feed.
        
        Returns:
            Dict[str, Any]: Status information
        """
        # Count the number of assets and timeframes with active regimes
        active_regime_count = sum(
            1 for asset in self.current_regimes
            for timeframe in self.current_regimes[asset]
            if self.current_regimes[asset][timeframe] is not None
        )
        
        # Count the number of historical regimes stored
        historical_regime_count = sum(
            len(self.historical_regimes.get(asset, {}).get(timeframe, []))
            for asset in self.assets
            for timeframe in self.timeframes
        )
        
        return {
            'name': self.name,
            'description': self.description,
            'connected': self.connected,
            'last_refresh': self.last_refresh.isoformat() if self.last_refresh else None,
            'assets_monitored': len(self.assets),
            'timeframes_monitored': len(self.timeframes),
            'active_regimes': active_regime_count,
            'historical_regimes': historical_regime_count
        }
