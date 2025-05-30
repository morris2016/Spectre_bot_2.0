

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Risk Manager - Correlation Risk

This module analyzes and manages correlation risk across multiple assets and trades.
It prevents over-exposure to correlated assets and ensures proper diversification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Type
from datetime import datetime
from sklearn.cluster import DBSCAN

from common.logger import get_logger
from common.db_client import DatabaseClient, get_db_client
from common.redis_client import RedisClient
from common.constants import CORRELATION_LOOKBACK_PERIODS
from common.exceptions import CorrelationCalculationError


class BaseCorrelationRiskManager:
    """Base class for correlation risk managers."""

    registry: Dict[str, Type["BaseCorrelationRiskManager"]] = {}

    def __init_subclass__(cls, name: Optional[str] = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        key = name or cls.__name__
        BaseCorrelationRiskManager.registry[key] = cls

    async def check_correlation_risk(self, *args, **kwargs):
        raise NotImplementedError

    async def adjust_position_size(self, *args, **kwargs):
        raise NotImplementedError

class CorrelationRiskManager(BaseCorrelationRiskManager):
    """
    The CorrelationRiskManager monitors and manages correlation risk across assets
    to prevent over-exposure to correlated markets and ensure proper diversification.
    """
    
    def __init__(self, 
                 db_client: DatabaseClient = None, 
                 redis_client: RedisClient = None,
                 config: Dict[str, Any] = None):
        """
        Initialize the Correlation Risk Manager.
        
        Args:
            db_client: Database client for historical data
            redis_client: Redis client for real-time data
            config: Configuration parameters for correlation management
        """
        self.logger = get_logger(self.__class__.__name__)
        self.db_client = db_client
        self._db_params = {}
        self.redis_client = redis_client or RedisClient()
        
        # Default configuration
        self._default_config = {
            'correlation_threshold': 0.7,  # Correlation threshold for considering assets correlated
            'max_clustered_exposure': 0.5,  # Maximum exposure to a correlated cluster
            'correlation_lookback': CORRELATION_LOOKBACK_PERIODS['MEDIUM'],  # Lookback period
            'enable_dynamic_correlation': True,  # Enable dynamic correlation analysis
            'min_data_points': 30,  # Minimum data points for correlation calculation
            'correlation_cache_ttl': 3600,  # Cache correlation matrix for 1 hour
            'correlation_update_frequency': 300,  # Update correlation every 5 minutes
            'enable_diversification_boost': True,  # Enable diversification incentives
            'negative_correlation_incentive': 0.2,  # 20% size boost for negatively correlated assets
            'regime_adaptive_correlation': True,  # Adapt correlation analysis to market regimes
            'cluster_detection_method': 'dbscan',  # Method for detecting correlated clusters
            'max_correlation_variance': 0.3  # Maximum variance in correlation before recalculation
        }
        
        # Apply custom configuration
        self.config = self._default_config.copy()
        if config:
            self.config.update(config)
            
        # Internal state
        self._correlation_matrix = None
        self._last_correlation_update = None
        self._asset_clusters = []
        self._cluster_exposures = {}
        self._correlation_cache = {}

        self.logger.info("Correlation Risk Manager initialized with config: %s", self.config)

    async def initialize(self, db_connector: Optional[DatabaseClient] = None) -> None:
        """Obtain a database client and create tables."""
        if db_connector is not None:
            self.db_client = db_connector
        if self.db_client is None:
            self.db_client = await get_db_client(**self._db_params)
        if getattr(self.db_client, "pool", None) is None:
            await self.db_client.initialize()
            await self.db_client.create_tables()
    
    async def calculate_correlation_matrix(self, 
                                           assets: List[str], 
                                           timeframe: str = '1h', 
                                           force_recalculate: bool = False) -> np.ndarray:
        """
        Calculate correlation matrix for the given assets.
        
        Args:
            assets: List of asset identifiers
            timeframe: Data timeframe for correlation calculation
            force_recalculate: Force recalculation even if cached
            
        Returns:
            np.ndarray: Correlation matrix
        """
        # Check if we have a cached matrix and it's still valid
        cache_key = f"correlation:{'-'.join(sorted(assets))}:{timeframe}"
        
        if not force_recalculate and self._last_correlation_update:
            time_since_update = (datetime.now() - self._last_correlation_update).total_seconds()
            if time_since_update < self.config['correlation_update_frequency']:
                if self._correlation_matrix is not None:
                    return self._correlation_matrix
        
        try:
            # Get historical data for correlation calculation
            lookback_days = {
                CORRELATION_LOOKBACK_PERIODS['SHORT']: 7,
                CORRELATION_LOOKBACK_PERIODS['MEDIUM']: 30,
                CORRELATION_LOOKBACK_PERIODS['LONG']: 90
            }.get(self.config['correlation_lookback'], 30)
            
            data_frames = {}
            
            # Check cache first
            cached_data = await self.redis_client.get(cache_key)
            if cached_data and not force_recalculate:
                self.logger.debug(f"Using cached correlation matrix for {cache_key}")
                self._correlation_matrix = np.array(cached_data)
                self._last_correlation_update = datetime.now()
                return self._correlation_matrix
            
            # Get price data for each asset
            for asset in assets:
                query = f"""
                    SELECT timestamp, close_price 
                    FROM market_data 
                    WHERE asset_id = '{asset}' 
                    AND timeframe = '{timeframe}'
                    AND timestamp > DATE_SUB(NOW(), INTERVAL {lookback_days} DAY)
                    ORDER BY timestamp ASC
                """
                
                asset_data = await self.db_client.execute_query(query)
                if not asset_data or len(asset_data) < self.config['min_data_points']:
                    self.logger.warning(f"Insufficient data for asset {asset} in timeframe {timeframe}")
                    continue
                
                # Convert to DataFrame and set timestamp as index
                df = pd.DataFrame(asset_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                data_frames[asset] = df['close_price']
            
            if not data_frames:
                raise CorrelationCalculationError("No valid data frames for correlation calculation")
            
            # Create combined DataFrame and calculate returns
            combined_df = pd.concat(data_frames, axis=1)
            combined_df.columns = list(data_frames.keys())
            
            # Calculate percentage returns
            returns_df = combined_df.pct_change().dropna()
            
            # Handle insufficient data
            if len(returns_df) < self.config['min_data_points']:
                self.logger.warning(f"Insufficient return data points: {len(returns_df)} < {self.config['min_data_points']}")
                # If we have old data, continue using it rather than failing
                if self._correlation_matrix is not None:
                    return self._correlation_matrix
                else:
                    # Create identity matrix (no correlation) as fallback
                    self._correlation_matrix = np.identity(len(assets))
                    return self._correlation_matrix
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr().values
            
            # Cache the correlation matrix
            await self.redis_client.set(cache_key, correlation_matrix.tolist(), self.config['correlation_cache_ttl'])
            
            self._correlation_matrix = correlation_matrix
            self._last_correlation_update = datetime.now()
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
            if self._correlation_matrix is not None:
                return self._correlation_matrix
            
            # Return identity matrix (no correlation) as fallback
            return np.identity(len(assets))
    
    async def detect_correlated_clusters(self, assets: List[str], timeframe: str = '1h') -> List[List[str]]:
        """
        Detect clusters of correlated assets.
        
        Args:
            assets: List of asset identifiers
            timeframe: Data timeframe for correlation calculation
            
        Returns:
            List[List[str]]: Clusters of correlated assets
        """
        try:
            # Calculate correlation matrix if needed
            correlation_matrix = await self.calculate_correlation_matrix(assets, timeframe)
            
            # Convert correlation to distance matrix (1 - |correlation|)
            # This makes highly correlated assets (both positive and negative) close to each other
            distance_matrix = 1 - np.abs(correlation_matrix)
            np.fill_diagonal(distance_matrix, 0)  # Zero distance to self
            
            clusters = []
            
            if self.config['cluster_detection_method'] == 'dbscan':
                # Use DBSCAN for clustering
                eps = 1 - self.config['correlation_threshold']  # Convert threshold to distance
                min_samples = 2  # Minimum 2 assets to form a cluster
                
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
                labels = dbscan.fit_predict(distance_matrix)
                
                # Group assets by cluster
                cluster_dict = {}
                for i, label in enumerate(labels):
                    if label != -1:  # -1 is noise in DBSCAN
                        if label not in cluster_dict:
                            cluster_dict[label] = []
                        cluster_dict[label].append(assets[i])
                
                clusters = list(cluster_dict.values())
                
            else:
                # Simple threshold-based clustering
                for i in range(len(assets)):
                    for j in range(i+1, len(assets)):
                        correlation = abs(correlation_matrix[i, j])
                        
                        if correlation >= self.config['correlation_threshold']:
                            # Find if these assets are already in clusters
                            asset_i = assets[i]
                            asset_j = assets[j]
                            
                            # Find clusters containing these assets
                            i_cluster = None
                            j_cluster = None
                            
                            for idx, cluster in enumerate(clusters):
                                if asset_i in cluster:
                                    i_cluster = idx
                                if asset_j in cluster:
                                    j_cluster = idx
                            
                            # Three cases: both in clusters, one in cluster, none in clusters
                            if i_cluster is not None and j_cluster is not None:
                                if i_cluster != j_cluster:
                                    # Merge clusters
                                    clusters[i_cluster].extend(clusters[j_cluster])
                                    clusters.pop(j_cluster)
                            elif i_cluster is not None:
                                # Add asset_j to asset_i's cluster
                                clusters[i_cluster].append(asset_j)
                            elif j_cluster is not None:
                                # Add asset_i to asset_j's cluster
                                clusters[j_cluster].append(asset_i)
                            else:
                                # Create new cluster
                                clusters.append([asset_i, asset_j])
            
            # Ensure clusters contain unique assets
            for i in range(len(clusters)):
                clusters[i] = list(set(clusters[i]))
            
            # Store clusters for later use
            self._asset_clusters = clusters
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error detecting correlated clusters: {e}")
            return []
    
    async def calculate_cluster_exposures(self, positions: List[Dict[str, Any]]) -> Dict[int, float]:
        """
        Calculate the current exposure to each correlated cluster.
        
        Args:
            positions: List of current positions with asset_id and exposure
            
        Returns:
            Dict[int, float]: Exposure to each cluster
        """
        try:
            if not positions:
                return {}
                
            # Extract assets from positions
            assets = [p['asset_id'] for p in positions]
            
            # Ensure we have clusters
            if not self._asset_clusters:
                await self.detect_correlated_clusters(assets)
            
            # Calculate exposure per cluster
            cluster_exposures = {}
            
            for i, cluster in enumerate(self._asset_clusters):
                total_exposure = 0.0
                
                for position in positions:
                    if position['asset_id'] in cluster:
                        total_exposure += position.get('exposure', 0.0)
                
                cluster_exposures[i] = total_exposure
            
            # Store for later use
            self._cluster_exposures = cluster_exposures
            
            return cluster_exposures
            
        except Exception as e:
            self.logger.error(f"Error calculating cluster exposures: {e}")
            return {}
    
    async def check_correlation_risk(self, 
                                     new_trade: Dict[str, Any], 
                                     current_positions: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a new trade would introduce excessive correlation risk.
        
        Args:
            new_trade: New trade to check
            current_positions: List of current positions
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (trade_allowed, risk_details)
        """
        try:
            if not current_positions:
                return True, {"reason": "No existing positions, correlation risk is zero"}
                
            asset_id = new_trade.get('asset_id')
            if not asset_id:
                return False, {"reason": "Missing asset_id in new trade"}
                
            exposure = new_trade.get('exposure', 0.0)
            if exposure <= 0:
                return True, {"reason": "Zero or negative exposure has no correlation risk"}
                
            # Get all assets including the new one
            assets = [p['asset_id'] for p in current_positions]
            if asset_id not in assets:
                assets.append(asset_id)
                
            # Calculate correlation matrix
            correlation_matrix = await self.calculate_correlation_matrix(assets)
            
            # Convert assets to indices for matrix lookup
            asset_indices = {asset: i for i, asset in enumerate(assets)}
            
            # Check for excessive correlation
            excessive_correlations = []
            for position in current_positions:
                position_asset = position.get('asset_id')
                if position_asset == asset_id:
                    continue  # Skip self correlation
                    
                if position_asset in asset_indices and asset_id in asset_indices:
                    i = asset_indices[position_asset]
                    j = asset_indices[asset_id]
                    
                    # Get correlation if indices are valid
                    if 0 <= i < correlation_matrix.shape[0] and 0 <= j < correlation_matrix.shape[1]:
                        correlation = correlation_matrix[i, j]
                        
                        if abs(correlation) >= self.config['correlation_threshold']:
                            excessive_correlations.append({
                                "asset": position_asset,
                                "correlation": correlation,
                                "exposure": position.get('exposure', 0.0)
                            })
            
            # Detect correlated clusters and calculate exposures
            await self.detect_correlated_clusters(assets)
            
            # Create position list including the new trade for cluster exposure calculation
            test_positions = current_positions.copy()
            test_positions.append({
                "asset_id": asset_id,
                "exposure": exposure
            })
            
            cluster_exposures = await self.calculate_cluster_exposures(test_positions)
            
            # Check if any cluster exceeds maximum allowed exposure
            excessive_clusters = []
            for cluster_id, cluster_exposure in cluster_exposures.items():
                if cluster_exposure > self.config['max_clustered_exposure']:
                    excessive_clusters.append({
                        "cluster_id": cluster_id,
                        "exposure": cluster_exposure,
                        "max_allowed": self.config['max_clustered_exposure'],
                        "assets": self._asset_clusters[cluster_id]
                    })
            
            # Determine if trade is allowed
            trade_allowed = len(excessive_clusters) == 0
            
            # Check for diversification benefit (negative correlation)
            diversification_assets = []
            if self.config['enable_diversification_boost']:
                for position in current_positions:
                    position_asset = position.get('asset_id')
                    if position_asset in asset_indices and asset_id in asset_indices:
                        i = asset_indices[position_asset]
                        j = asset_indices[asset_id]
                        
                        if 0 <= i < correlation_matrix.shape[0] and 0 <= j < correlation_matrix.shape[1]:
                            correlation = correlation_matrix[i, j]
                            
                            if correlation <= -0.5:  # Significant negative correlation
                                diversification_assets.append({
                                    "asset": position_asset,
                                    "correlation": correlation
                                })
            
            return trade_allowed, {
                "excessive_correlations": excessive_correlations,
                "excessive_clusters": excessive_clusters,
                "diversification_assets": diversification_assets,
                "trade_allowed": trade_allowed
            }
            
        except Exception as e:
            self.logger.error(f"Error checking correlation risk: {e}")
            # Default to allowing the trade if error occurs
            return True, {"reason": f"Error during correlation check: {str(e)}"}
    
    async def adjust_position_size(self, 
                                  trade: Dict[str, Any], 
                                  current_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Adjust position size based on correlation with existing positions.
        
        Args:
            trade: Trade to adjust
            current_positions: List of current positions
            
        Returns:
            Dict[str, Any]: Adjusted trade
        """
        try:
            if not current_positions:
                return trade
                
            # Check correlation risk
            is_allowed, risk_details = await self.check_correlation_risk(trade, current_positions)
            
            adjusted_trade = trade.copy()
            
            if not is_allowed:
                # Reduce position size based on excessive correlation
                excessive_clusters = risk_details.get('excessive_clusters', [])
                
                if excessive_clusters:
                    # Calculate how much we need to reduce size
                    max_reduction = 0.0
                    
                    for cluster in excessive_clusters:
                        cluster_exposure = cluster.get('exposure', 0.0)
                        max_allowed = cluster.get('max_allowed', self.config['max_clustered_exposure'])
                        
                        reduction_factor = max_allowed / cluster_exposure if cluster_exposure > 0 else 1.0
                        max_reduction = max(max_reduction, 1.0 - reduction_factor)
                    
                    # Apply reduction to position size and/or exposure
                    if 'position_size' in adjusted_trade:
                        adjusted_trade['position_size'] *= (1.0 - max_reduction)
                    
                    if 'exposure' in adjusted_trade:
                        adjusted_trade['exposure'] *= (1.0 - max_reduction)
                        
                    self.logger.info(f"Reduced position size due to correlation risk by factor: {max_reduction:.2f}")
                    
                    # Add reason for adjustment
                    if 'adjustments' not in adjusted_trade:
                        adjusted_trade['adjustments'] = []
                        
                    adjusted_trade['adjustments'].append({
                        "type": "correlation_risk",
                        "reduction_factor": 1.0 - max_reduction,
                        "excessive_clusters": len(excessive_clusters)
                    })
            
            # Check for diversification boost
            if self.config['enable_diversification_boost']:
                diversification_assets = risk_details.get('diversification_assets', [])
                
                if diversification_assets:
                    # Apply boost for negative correlation (diversification benefit)
                    boost_factor = 1.0 + self.config['negative_correlation_incentive']
                    
                    if 'position_size' in adjusted_trade:
                        adjusted_trade['position_size'] *= boost_factor
                    
                    if 'exposure' in adjusted_trade:
                        adjusted_trade['exposure'] *= boost_factor
                        
                    self.logger.info(f"Boosted position size due to diversification benefit by factor: {boost_factor:.2f}")
                    
                    # Add reason for adjustment
                    if 'adjustments' not in adjusted_trade:
                        adjusted_trade['adjustments'] = []
                        
                    adjusted_trade['adjustments'].append({
                        "type": "diversification_boost",
                        "boost_factor": boost_factor,
                        "diversifying_assets": len(diversification_assets)
                    })
            
            return adjusted_trade
            
        except Exception as e:
            self.logger.error(f"Error adjusting position size for correlation: {e}")
            return trade  # Return original trade if error occurs
    
    async def get_correlation(self, asset1: str, asset2: str, timeframe: str = '1h') -> float:
        """
        Get correlation between two assets.
        
        Args:
            asset1: First asset identifier
            asset2: Second asset identifier
            timeframe: Data timeframe for correlation calculation
            
        Returns:
            float: Correlation coefficient
        """
        try:
            # Sort assets for consistent caching
            assets = sorted([asset1, asset2])
            
            # Check cache
            cache_key = f"correlation:{assets[0]}:{assets[1]}:{timeframe}"
            cached_corr = self._correlation_cache.get(cache_key)
            
            if cached_corr is not None:
                return cached_corr
            
            # Calculate correlation matrix
            correlation_matrix = await self.calculate_correlation_matrix(assets, timeframe)
            
            # Extract correlation value
            correlation = correlation_matrix[0, 1]
            
            # Cache result
            self._correlation_cache[cache_key] = correlation
            
            return correlation
            
        except Exception as e:
            self.logger.error(f"Error getting correlation between {asset1} and {asset2}: {e}")
            return 0.0  # Default to no correlation on error
    
    async def get_correlated_assets(self, asset: str, threshold: float = None, timeframe: str = '1h') -> List[Dict[str, Any]]:
        """
        Get list of assets correlated with the given asset above the threshold.
        
        Args:
            asset: Asset identifier
            threshold: Correlation threshold (defaults to config value)
            timeframe: Data timeframe for correlation calculation
            
        Returns:
            List[Dict[str, Any]]: List of correlated assets with correlation values
        """
        try:
            if threshold is None:
                threshold = self.config['correlation_threshold']
                
            # Get list of active assets from database
            query = """
                SELECT DISTINCT asset_id FROM market_data 
                WHERE timestamp > DATE_SUB(NOW(), INTERVAL 1 DAY)
            """
            
            result = await self.db_client.execute_query(query)
            all_assets = [row['asset_id'] for row in result if row['asset_id'] != asset]
            
            if not all_assets:
                return []
                
            # Add the target asset to the beginning
            assets = [asset] + all_assets
            
            # Calculate correlation matrix
            correlation_matrix = await self.calculate_correlation_matrix(assets, timeframe)
            
            # First row/column corresponds to the target asset
            correlations = correlation_matrix[0, 1:]
            
            # Find correlated assets
            correlated_assets = []
            for i, corr in enumerate(correlations):
                if abs(corr) >= threshold:
                    correlated_assets.append({
                        "asset_id": all_assets[i],
                        "correlation": corr
                    })
            
            # Sort by absolute correlation (descending)
            correlated_assets.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            return correlated_assets
            
        except Exception as e:
            self.logger.error(f"Error getting correlated assets for {asset}: {e}")
            return []
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get information about current correlation clusters.
        
        Returns:
            Dict[str, Any]: Cluster information
        """
        result = {
            "num_clusters": len(self._asset_clusters),
            "clusters": [],
            "cluster_exposures": self._cluster_exposures
        }
        
        for i, cluster in enumerate(self._asset_clusters):
            cluster_info = {
                "id": i,
                "assets": cluster,
                "size": len(cluster),
                "exposure": self._cluster_exposures.get(i, 0.0)
            }
            result["clusters"].append(cluster_info)

        return result


def get_correlation_risk_manager(name: str, *args, **kwargs) -> BaseCorrelationRiskManager:
    """Instantiate a registered correlation risk manager by name."""
    cls = BaseCorrelationRiskManager.registry.get(name)
    if cls is None:
        raise ValueError(f"Unknown correlation risk manager: {name}")
    return cls(*args, **kwargs)


__all__ = ["BaseCorrelationRiskManager", "get_correlation_risk_manager"]
