

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Volume Feature Module

This module provides sophisticated volume analysis features that extract deep insights
from trading volume data across multiple timeframes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy.stats import norm, linregress
from scipy.signal import find_peaks, savgol_filter

from common.constants import VOLUME_PROFILE_BINS, VOLUME_ZONE_SIGNIFICANCE
from common.utils import rolling_window, exponential_decay_weights
from feature_service.features.base_feature import BaseFeature
from data_storage.time_series import TimeSeriesStorage

logger = logging.getLogger(__name__)


@dataclass
class VolumeProfileResult:
    """Data class for storing volume profile analysis results."""
    value_area_high: float
    value_area_low: float
    point_of_control: float
    volume_by_price: Dict[float, float]
    high_volume_nodes: List[float]
    low_volume_nodes: List[float]
    distribution_type: str  # 'normal', 'bimodal', 'skewed_high', 'skewed_low'


@dataclass
class BuyingSellingPressureResult:
    """Data class for buying/selling pressure analysis results."""
    buying_pressure: float  # 0-100 scale
    selling_pressure: float  # 0-100 scale
    net_pressure: float  # -100 to 100 scale
    pressure_trend: str  # 'increasing', 'decreasing', 'stable'
    exhaustion_signals: List[Dict[str, Any]]
    climax_points: List[Dict[str, Any]]


@dataclass
class VolumeBreakdownResult:
    """Data class for volume breakdown analysis."""
    buying_volume: float
    selling_volume: float
    neutral_volume: float
    relative_buying_volume: float  # 0-1 scale
    relative_selling_volume: float  # 0-1 scale
    buying_volume_trend: List[float]
    selling_volume_trend: List[float]


class VolumeFeatures(BaseFeature):
    """
    Volume analysis feature extraction class providing sophisticated volume-based
    insights for the QuantumSpectre Elite Trading System.
    """

    def __init__(self, ts_storage: TimeSeriesStorage):
        """
        Initialize the VolumeFeatures class.
        
        Args:
            ts_storage: TimeSeriesStorage instance for accessing historical data
        """
        super().__init__(ts_storage)
        self.volume_cache = {}  # Cache for volume calculations
        logger.info("VolumeFeatures initialized")
    
    def calculate_vwap(self, symbol: str, timeframe: str, 
                       lookback_periods: int = 100,
                       anchor_type: str = 'session',  # 'session', 'day', 'week', 'month'
                       include_extended_hours: bool = False) -> np.ndarray:
        """
        Calculate Volume Weighted Average Price (VWAP) for the given symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data
            lookback_periods: Number of periods to include
            anchor_type: Type of anchoring for VWAP calculation
            include_extended_hours: Whether to include extended hours data
            
        Returns:
            numpy array of VWAP values
        """
        # Create cache key
        cache_key = f"{symbol}_{timeframe}_{lookback_periods}_{anchor_type}_{include_extended_hours}_vwap"
        if cache_key in self.volume_cache:
            return self.volume_cache[cache_key]
            
        # Get OHLCV data
        data = self.ts_storage.get_ohlcv(symbol, timeframe, lookback_periods)
        
        if data is None or len(data) == 0:
            logger.warning(f"No data available for VWAP calculation: {symbol} {timeframe}")
            return np.array([])
        
        # Filter for session type if needed
        if anchor_type != 'session' and not include_extended_hours:
            # Implement session filtering logic based on anchor_type
            # This would filter out data points outside regular trading hours
            pass
            
        # Calculate typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate VWAP
        cumulative_tp_vol = (typical_price * data['volume']).cumsum()
        cumulative_vol = data['volume'].cumsum()
        
        # Avoid division by zero
        cumulative_vol = np.where(cumulative_vol == 0, 1, cumulative_vol)
        
        vwap = cumulative_tp_vol / cumulative_vol
        
        # Store in cache
        self.volume_cache[cache_key] = vwap
        
        return vwap
    
    def calculate_volume_profile(self, symbol: str, timeframe: str,
                                periods: int = 100,
                                value_area_volume_percent: float = 0.68,  # ~1 std dev
                                adaptive_bins: bool = True) -> VolumeProfileResult:
        """
        Calculate volume profile (volume by price) for market profile analysis.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data
            periods: Number of periods to include
            value_area_volume_percent: Percentage of volume to include in value area
            adaptive_bins: Whether to adaptively determine bin count based on data
            
        Returns:
            VolumeProfileResult object with volume profile analysis
        """
        # Create cache key
        cache_key = f"{symbol}_{timeframe}_{periods}_{value_area_volume_percent}_{adaptive_bins}_vp"
        if cache_key in self.volume_cache:
            return self.volume_cache[cache_key]
        
        # Get OHLCV data
        data = self.ts_storage.get_ohlcv(symbol, timeframe, periods)
        
        if data is None or len(data) == 0:
            logger.warning(f"No data available for volume profile: {symbol} {timeframe}")
            return None
        
        # Determine price range
        price_min = data['low'].min()
        price_max = data['high'].max()
        price_range = price_max - price_min
        
        # Determine number of bins
        n_bins = VOLUME_PROFILE_BINS
        if adaptive_bins:
            # Adaptive bin sizing based on volatility and range
            volatility = data['high'].std() / data['close'].mean()
            n_bins = max(20, min(200, int(n_bins * (1 + 5 * volatility))))
        
        # Create price bins
        bins = np.linspace(price_min, price_max, n_bins)
        bin_width = price_range / (n_bins - 1)
        
        # Initialize volume by price dictionary
        volume_by_price = {float(price): 0.0 for price in bins}
        
        # Calculate volume at each price level
        for i in range(len(data)):
            bar_low = data['low'].iloc[i]
            bar_high = data['high'].iloc[i]
            bar_vol = data['volume'].iloc[i]
            
            # Distribute volume across price points within the bar
            for price in bins:
                if bar_low <= price <= bar_high:
                    # Weight by proximity to close price
                    close_proximity = 1 - abs(price - data['close'].iloc[i]) / (bar_high - bar_low + 0.0001)
                    proximity_weight = 0.5 + (0.5 * close_proximity)  # 0.5-1.0 range
                    
                    # Distribute volume proportionally
                    price_coverage = bin_width / (bar_high - bar_low + bin_width)
                    allocated_volume = bar_vol * price_coverage * proximity_weight
                    volume_by_price[float(price)] += allocated_volume
        
        # Total volume
        total_volume = sum(volume_by_price.values())
        
        # Find point of control (price with highest volume)
        point_of_control = max(volume_by_price, key=volume_by_price.get)
        
        # Calculate value area (70% of volume centered at POC)
        sorted_prices = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)
        running_vol = 0
        value_area_prices = []
        
        for price, vol in sorted_prices:
            running_vol += vol
            value_area_prices.append(price)
            if running_vol >= total_volume * value_area_volume_percent:
                break
        
        value_area_high = max(value_area_prices)
        value_area_low = min(value_area_prices)
        
        # Identify high and low volume nodes
        mean_vol = total_volume / len(bins)
        std_vol = np.std(list(volume_by_price.values()))
        
        high_volume_threshold = mean_vol + (std_vol * 1.0)
        low_volume_threshold = mean_vol - (std_vol * 0.8)
        
        high_volume_nodes = [price for price, vol in volume_by_price.items() 
                            if vol > high_volume_threshold]
        low_volume_nodes = [price for price, vol in volume_by_price.items() 
                           if vol < low_volume_threshold and vol > 0]
        
        # Determine distribution type
        vol_array = np.array(list(volume_by_price.values()))
        price_array = np.array(list(volume_by_price.keys()))
        
        # Perform normality test and skew analysis
        distribution_type = self._analyze_distribution(vol_array, price_array, point_of_control)
        
        # Create result object
        result = VolumeProfileResult(
            value_area_high=value_area_high,
            value_area_low=value_area_low,
            point_of_control=point_of_control,
            volume_by_price=volume_by_price,
            high_volume_nodes=high_volume_nodes,
            low_volume_nodes=low_volume_nodes,
            distribution_type=distribution_type
        )
        
        # Store in cache
        self.volume_cache[cache_key] = result
        
        return result
    
    def _analyze_distribution(self, volumes: np.ndarray, prices: np.ndarray, 
                             poc: float) -> str:
        """
        Analyze the distribution of volume to determine its shape.
        
        Args:
            volumes: Array of volumes
            prices: Array of prices
            poc: Point of control price
            
        Returns:
            String describing the distribution type
        """
        # Normalize volumes
        if volumes.sum() > 0:
            norm_volumes = volumes / volumes.sum()
        else:
            return "insufficient_data"
        
        # Calculate skewness
        mean_price = np.average(prices, weights=norm_volumes)
        variance = np.average((prices - mean_price)**2, weights=norm_volumes)
        std_dev = np.sqrt(variance)
        
        if std_dev == 0:
            return "single_point"
            
        skewness = np.average(((prices - mean_price) / std_dev)**3, weights=norm_volumes)
        
        # Smooth the volume curve
        smoothed = savgol_filter(norm_volumes, min(11, len(norm_volumes) - (len(norm_volumes) % 2 + 1)), 3)
        
        # Find peaks for bimodality testing
        peaks, _ = find_peaks(smoothed, height=np.max(smoothed) * 0.4, distance=len(smoothed) // 10)
        
        if len(peaks) >= 2:
            # Check if peaks are significant
            peak_heights = smoothed[peaks]
            peak_significance = peak_heights / np.max(peak_heights)
            
            if np.min(peak_significance) > 0.5 and np.max(prices[peaks]) - np.min(prices[peaks]) > std_dev:
                return "bimodal"
        
        # Check for skewness
        if skewness > 0.5:
            return "skewed_high"
        elif skewness < -0.5:
            return "skewed_low"
        else:
            return "normal"
    
    def calculate_volume_zones(self, symbol: str, timeframe: str,
                              periods: int = 100,
                              zone_significance: float = VOLUME_ZONE_SIGNIFICANCE,
                              min_zone_separation: float = 0.01) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify significant volume-based support and resistance zones.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data
            periods: Number of periods to include
            zone_significance: Minimum significance for a zone (0-1)
            min_zone_separation: Minimum separation between zones as % of price
            
        Returns:
            Dict with 'support' and 'resistance' zones, each a list of dicts
        """
        # Get volume profile
        profile = self.calculate_volume_profile(symbol, timeframe, periods)
        
        if profile is None:
            return {'support': [], 'resistance': []}
        
        # Get high volume nodes as potential zones
        volume_by_price = profile.volume_by_price
        high_volume_nodes = profile.high_volume_nodes
        
        # Get price data for current levels
        recent_data = self.ts_storage.get_ohlcv(symbol, timeframe, 5)
        if recent_data is None or len(recent_data) == 0:
            current_price = None
        else:
            current_price = recent_data['close'].iloc[-1]
        
        # Total volume in high nodes
        total_high_volume = sum(volume_by_price[price] for price in high_volume_nodes)
        
        # Initialize zones
        support_zones = []
        resistance_zones = []
        
        # Cluster nearby high volume nodes
        sorted_nodes = sorted(high_volume_nodes)
        
        if not sorted_nodes:
            return {'support': [], 'resistance': []}
        
        current_zone = [sorted_nodes[0]]
        current_zone_volume = volume_by_price[sorted_nodes[0]]
        zones = []
        
        for i in range(1, len(sorted_nodes)):
            current_price_node = sorted_nodes[i]
            prev_price_node = sorted_nodes[i-1]
            
            # Check if nodes are close enough to cluster
            if (current_price_node - prev_price_node) / prev_price_node < min_zone_separation:
                # Add to current zone
                current_zone.append(current_price_node)
                current_zone_volume += volume_by_price[current_price_node]
            else:
                # Finalize current zone and start new one
                if current_zone:
                    zone_avg_price = sum(current_zone) / len(current_zone)
                    zone_significance_score = current_zone_volume / total_high_volume
                    
                    if zone_significance_score >= zone_significance:
                        zones.append({
                            'price': zone_avg_price,
                            'volume': current_zone_volume,
                            'significance': zone_significance_score,
                            'price_range': [min(current_zone), max(current_zone)]
                        })
                
                current_zone = [current_price_node]
                current_zone_volume = volume_by_price[current_price_node]
        
        # Add the last zone if it exists
        if current_zone:
            zone_avg_price = sum(current_zone) / len(current_zone)
            zone_significance_score = current_zone_volume / total_high_volume
            
            if zone_significance_score >= zone_significance:
                zones.append({
                    'price': zone_avg_price,
                    'volume': current_zone_volume,
                    'significance': zone_significance_score,
                    'price_range': [min(current_zone), max(current_zone)]
                })
        
        # Classify zones as support or resistance based on current price
        if current_price is not None:
            for zone in zones:
                # Define zone strength based on volume significance
                strength = zone['significance'] * 100  # Convert to percentage
                
                # Add tested status based on recent price action
                recent_lows = recent_data['low'].values
                recent_highs = recent_data['high'].values
                
                zone_near_tests = sum(1 for low in recent_lows if abs(low - zone['price']) / zone['price'] < 0.005)
                zone_near_tests += sum(1 for high in recent_highs if abs(high - zone['price']) / zone['price'] < 0.005)
                
                zone['recently_tested'] = zone_near_tests > 0
                zone['test_count'] = zone_near_tests
                zone['strength'] = strength
                
                # Classify as support or resistance
                if zone['price'] < current_price:
                    support_zones.append(zone)
                else:
                    resistance_zones.append(zone)
        
        # Sort by significance
        support_zones.sort(key=lambda x: x['significance'], reverse=True)
        resistance_zones.sort(key=lambda x: x['significance'], reverse=True)
        
        return {
            'support': support_zones,
            'resistance': resistance_zones
        }
    
    def calculate_buying_selling_pressure(self, symbol: str, timeframe: str,
                                         periods: int = 100) -> BuyingSellingPressureResult:
        """
        Analyze buying and selling pressure to identify imbalances.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data
            periods: Number of periods to include
            
        Returns:
            BuyingSellingPressureResult object with pressure analysis
        """
        # Create cache key
        cache_key = f"{symbol}_{timeframe}_{periods}_pressure"
        if cache_key in self.volume_cache:
            return self.volume_cache[cache_key]
        
        # Get OHLCV data
        data = self.ts_storage.get_ohlcv(symbol, timeframe, periods)
        
        if data is None or len(data) == 0:
            logger.warning(f"No data available for buying/selling pressure: {symbol} {timeframe}")
            return None
        
        # Calculate buying/selling volume for each bar
        buying_volume = np.zeros(len(data))
        selling_volume = np.zeros(len(data))
        neutral_volume = np.zeros(len(data))
        
        for i in range(len(data)):
            open_price = data['open'].iloc[i]
            close_price = data['close'].iloc[i]
            high_price = data['high'].iloc[i]
            low_price = data['low'].iloc[i]
            volume = data['volume'].iloc[i]
            
            price_range = high_price - low_price
            if price_range == 0:
                # Flat candle, all volume is neutral
                neutral_volume[i] = volume
                continue
            
            # Calculate close position within the range (0-1)
            close_position = (close_price - low_price) / price_range
            
            # Calculate weighted buying/selling volume based on close position
            if close_price > open_price:  # Bullish candle
                # Buying pressure increases as close approaches high
                buying_weight = 0.5 + (0.5 * close_position)
                selling_weight = 1 - buying_weight
                
                buying_volume[i] = volume * buying_weight
                selling_volume[i] = volume * selling_weight
            
            elif close_price < open_price:  # Bearish candle
                # Selling pressure increases as close approaches low
                selling_weight = 0.5 + (0.5 * (1 - close_position))
                buying_weight = 1 - selling_weight
                
                selling_volume[i] = volume * selling_weight
                buying_volume[i] = volume * buying_weight
            
            else:  # Doji
                # Equal buying and selling with some neutral
                buying_volume[i] = volume * 0.4
                selling_volume[i] = volume * 0.4
                neutral_volume[i] = volume * 0.2
        
        # Calculate cumulative values for trend analysis
        cum_buying = np.cumsum(buying_volume)
        cum_selling = np.cumsum(selling_volume)
        
        # Calculate net pressure (recent periods weighted more heavily)
        weights = exponential_decay_weights(periods, decay_factor=0.95)
        
        recent_buying = np.sum(buying_volume * weights)
        recent_selling = np.sum(selling_volume * weights)
        total_recent_volume = recent_buying + recent_selling
        
        if total_recent_volume > 0:
            buying_pressure = (recent_buying / total_recent_volume) * 100
            selling_pressure = (recent_selling / total_recent_volume) * 100
            net_pressure = buying_pressure - selling_pressure
        else:
            buying_pressure = 50
            selling_pressure = 50
            net_pressure = 0
        
        # Determine pressure trend
        if len(buying_volume) >= 10:
            recent_net = buying_volume[-10:] - selling_volume[-10:]
            trend_slope, _, _, _, _ = linregress(range(10), recent_net)
            
            if trend_slope > 0.5:
                pressure_trend = "increasing"
            elif trend_slope < -0.5:
                pressure_trend = "decreasing"
            else:
                pressure_trend = "stable"
        else:
            pressure_trend = "insufficient_data"
        
        # Identify exhaustion signals
        exhaustion_signals = []
        
        # Look for volume climax points
        for i in range(5, len(data)):
            current_volume = data['volume'].iloc[i]
            avg_volume = data['volume'].iloc[i-5:i].mean()
            
            # Volume climax detection (3x average volume)
            if current_volume > avg_volume * 3:
                # Check if it's a buying or selling climax
                if buying_volume[i] > selling_volume[i] * 1.5:
                    climax_type = "buying_climax"
                elif selling_volume[i] > buying_volume[i] * 1.5:
                    climax_type = "selling_climax"
                else:
                    climax_type = "volume_climax"
                
                exhaustion_signals.append({
                    'type': climax_type,
                    'timestamp': data.index[i],
                    'volume_ratio': current_volume / avg_volume,
                    'price': data['close'].iloc[i]
                })
        
        # Find climax points (highest relative buying/selling pressure)
        buy_pressure_ratio = buying_volume / (selling_volume + 1e-10)
        sell_pressure_ratio = selling_volume / (buying_volume + 1e-10)
        
        # Smooth the ratios
        if len(buy_pressure_ratio) >= 5:
            buy_pressure_ratio_smooth = pd.Series(buy_pressure_ratio).rolling(5).mean().values
            sell_pressure_ratio_smooth = pd.Series(sell_pressure_ratio).rolling(5).mean().values
        else:
            buy_pressure_ratio_smooth = buy_pressure_ratio
            sell_pressure_ratio_smooth = sell_pressure_ratio
        
        # Find peaks in pressure ratios
        buy_peaks, _ = find_peaks(buy_pressure_ratio_smooth, height=2.0, distance=5)
        sell_peaks, _ = find_peaks(sell_pressure_ratio_smooth, height=2.0, distance=5)
        
        climax_points = []
        
        for peak in buy_peaks:
            if peak < len(data):
                climax_points.append({
                    'type': 'buying_pressure_peak',
                    'timestamp': data.index[peak],
                    'ratio': buy_pressure_ratio_smooth[peak],
                    'price': data['close'].iloc[peak]
                })
        
        for peak in sell_peaks:
            if peak < len(data):
                climax_points.append({
                    'type': 'selling_pressure_peak',
                    'timestamp': data.index[peak],
                    'ratio': sell_pressure_ratio_smooth[peak],
                    'price': data['close'].iloc[peak]
                })
        
        # Create result
        result = BuyingSellingPressureResult(
            buying_pressure=buying_pressure,
            selling_pressure=selling_pressure,
            net_pressure=net_pressure,
            pressure_trend=pressure_trend,
            exhaustion_signals=exhaustion_signals,
            climax_points=climax_points
        )
        
        # Store in cache
        self.volume_cache[cache_key] = result
        
        return result
    
    def calculate_volume_breakdown(self, symbol: str, timeframe: str,
                                  periods: int = 100) -> VolumeBreakdownResult:
        """
        Provide detailed breakdown of volume by type (buying, selling, neutral).
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data
            periods: Number of periods to include
            
        Returns:
            VolumeBreakdownResult object with volume breakdown
        """
        # Create cache key
        cache_key = f"{symbol}_{timeframe}_{periods}_breakdown"
        if cache_key in self.volume_cache:
            return self.volume_cache[cache_key]
        
        # Get OHLCV data
        data = self.ts_storage.get_ohlcv(symbol, timeframe, periods)
        
        if data is None or len(data) == 0:
            logger.warning(f"No data available for volume breakdown: {symbol} {timeframe}")
            return None
        
        # Calculate buying/selling/neutral volume
        buying_volume = 0.0
        selling_volume = 0.0
        neutral_volume = 0.0
        
        buying_volume_trend = []
        selling_volume_trend = []
        
        window_size = min(10, len(data) // 5)  # Window for trend calculation
        
        for i in range(len(data)):
            open_price = data['open'].iloc[i]
            close_price = data['close'].iloc[i]
            high_price = data['high'].iloc[i]
            low_price = data['low'].iloc[i]
            volume = data['volume'].iloc[i]
            
            # Calculate volume allocation
            if close_price > open_price:  # Bullish candle
                # Calculate how bullish the candle is
                body_size = close_price - open_price
                range_size = high_price - low_price
                
                if range_size > 0:
                    bullishness = body_size / range_size
                else:
                    bullishness = 0.5
                
                # Allocate volume based on bullishness
                bar_buying = volume * (0.6 + 0.4 * bullishness)
                bar_selling = volume * (0.4 - 0.4 * bullishness)
                bar_neutral = volume * 0.0  # No neutral volume in this case
                
            elif close_price < open_price:  # Bearish candle
                # Calculate how bearish the candle is
                body_size = open_price - close_price
                range_size = high_price - low_price
                
                if range_size > 0:
                    bearishness = body_size / range_size
                else:
                    bearishness = 0.5
                
                # Allocate volume based on bearishness
                bar_buying = volume * (0.4 - 0.4 * bearishness)
                bar_selling = volume * (0.6 + 0.4 * bearishness)
                bar_neutral = volume * 0.0  # No neutral volume in this case
                
            else:  # Doji
                # Equal buying and selling with some neutral
                bar_buying = volume * 0.4
                bar_selling = volume * 0.4
                bar_neutral = volume * 0.2
            
            # Add to totals
            buying_volume += bar_buying
            selling_volume += bar_selling
            neutral_volume += bar_neutral
            
            # Calculate rolling trends
            if i >= window_size:
                window_buying = 0
                window_selling = 0
                
                for j in range(i - window_size + 1, i + 1):
                    o = data['open'].iloc[j]
                    c = data['close'].iloc[j]
                    v = data['volume'].iloc[j]
                    
                    if c > o:
                        window_buying += v
                    elif c < o:
                        window_selling += v
                    else:
                        window_buying += v * 0.5
                        window_selling += v * 0.5
                
                buying_volume_trend.append(window_buying)
                selling_volume_trend.append(window_selling)
        
        # Calculate relative volumes
        total_volume = buying_volume + selling_volume + neutral_volume
        
        if total_volume > 0:
            relative_buying = buying_volume / total_volume
            relative_selling = selling_volume / total_volume
        else:
            relative_buying = 0.5
            relative_selling = 0.5
        
        # Create result
        result = VolumeBreakdownResult(
            buying_volume=buying_volume,
            selling_volume=selling_volume,
            neutral_volume=neutral_volume,
            relative_buying_volume=relative_buying,
            relative_selling_volume=relative_selling,
            buying_volume_trend=buying_volume_trend,
            selling_volume_trend=selling_volume_trend
        )
        
        # Store in cache
        self.volume_cache[cache_key] = result
        
        return result
    
    def detect_volume_anomalies(self, symbol: str, timeframe: str,
                               periods: int = 100,
                               sensitivity: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect volume anomalies that may indicate significant market events.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data
            periods: Number of periods to include
            sensitivity: Detection sensitivity (higher = more sensitive)
            
        Returns:
            List of anomaly events with timestamp, type and metrics
        """
        # Create cache key
        cache_key = f"{symbol}_{timeframe}_{periods}_{sensitivity}_anomalies"
        if cache_key in self.volume_cache:
            return self.volume_cache[cache_key]
        
        # Get OHLCV data
        data = self.ts_storage.get_ohlcv(symbol, timeframe, periods)
        
        if data is None or len(data) == 0:
            logger.warning(f"No data available for volume anomalies: {symbol} {timeframe}")
            return []
        
        # Calculate volume metrics
        volumes = data['volume'].values
        log_volumes = np.log1p(volumes)  # Log transform to handle skewed distribution
        
        # Calculate rolling statistics
        window_size = min(20, len(data) // 5)
        
        if len(data) <= window_size:
            return []
        
        rolling_mean = pd.Series(log_volumes).rolling(window_size).mean().values
        rolling_std = pd.Series(log_volumes).rolling(window_size).std().values
        
        # Identify anomalies
        anomalies = []
        
        for i in range(window_size, len(data)):
            # Skip if missing data
            if np.isnan(rolling_mean[i]) or np.isnan(rolling_std[i]):
                continue
            
            current_volume = log_volumes[i]
            volume_mean = rolling_mean[i]
            volume_std = rolling_std[i]
            
            # Z-score
            if volume_std > 0:
                z_score = (current_volume - volume_mean) / volume_std
            else:
                z_score = 0
            
            # Check for anomalies
            if z_score > sensitivity:
                # High volume anomaly
                candle_pattern = self._classify_candle_pattern(
                    data['open'].iloc[i],
                    data['high'].iloc[i],
                    data['low'].iloc[i],
                    data['close'].iloc[i],
                    data['close'].iloc[i-1] if i > 0 else data['open'].iloc[i]
                )
                
                anomalies.append({
                    'timestamp': data.index[i],
                    'type': 'high_volume',
                    'z_score': z_score,
                    'volume': volumes[i],
                    'volume_ratio': np.exp(current_volume) / np.exp(volume_mean),
                    'price': data['close'].iloc[i],
                    'candle_pattern': candle_pattern
                })
            elif z_score < -sensitivity and i > 0:
                # Low volume anomaly (potentially significant)
                price_change = abs(data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
                
                # Low volume with price movement is interesting
                if price_change > 0.005:  # 0.5% move
                    anomalies.append({
                        'timestamp': data.index[i],
                        'type': 'low_volume_with_movement',
                        'z_score': z_score,
                        'volume': volumes[i],
                        'volume_ratio': np.exp(current_volume) / np.exp(volume_mean),
                        'price': data['close'].iloc[i],
                        'price_change_percent': price_change * 100
                    })
        
        # Detect volume divergence (price making new highs/lows with declining volume)
        if len(data) >= 10:
            for i in range(10, len(data)):
                # Check for new local high with declining volume
                if data['high'].iloc[i] > max(data['high'].iloc[i-10:i]):
                    recent_volume_trend = volumes[i] / np.mean(volumes[i-5:i])
                    
                    if recent_volume_trend < 0.8:  # Volume declining
                        anomalies.append({
                            'timestamp': data.index[i],
                            'type': 'bullish_volume_divergence',
                            'price': data['high'].iloc[i],
                            'volume_trend_ratio': recent_volume_trend
                        })
                
                # Check for new local low with declining volume
                if data['low'].iloc[i] < min(data['low'].iloc[i-10:i]):
                    recent_volume_trend = volumes[i] / np.mean(volumes[i-5:i])
                    
                    if recent_volume_trend < 0.8:  # Volume declining
                        anomalies.append({
                            'timestamp': data.index[i],
                            'type': 'bearish_volume_divergence',
                            'price': data['low'].iloc[i],
                            'volume_trend_ratio': recent_volume_trend
                        })
        
        # Sort anomalies by timestamp
        anomalies.sort(key=lambda x: x['timestamp'])
        
        # Store in cache
        self.volume_cache[cache_key] = anomalies
        
        return anomalies
    
    def _classify_candle_pattern(self, open_price: float, high: float, low: float, 
                                close: float, prev_close: float) -> str:
        """
        Classify the candle pattern for context in volume analysis.
        
        Args:
            open_price: Opening price
            high: High price
            low: Low price
            close: Close price
            prev_close: Previous close price
            
        Returns:
            String describing the candle pattern
        """
        body_size = abs(close - open_price)
        range_size = high - low
        
        # Avoid division by zero
        if range_size == 0:
            return "doji"
        
        body_ratio = body_size / range_size
        
        # Price direction
        if close > open_price:
            direction = "bullish"
        elif close < open_price:
            direction = "bearish"
        else:
            direction = "neutral"
        
        # Previous direction
        prev_direction = "up" if close > prev_close else "down"
        
        # Classify pattern
        if body_ratio < 0.2:
            if high - max(open_price, close) > 2 * body_size and min(open_price, close) - low > 2 * body_size:
                return f"{direction}_high_wave_doji"
            elif high - max(open_price, close) > 2 * body_size:
                return f"{direction}_shooting_star" if prev_direction == "up" else f"{direction}_inverted_hammer"
            elif min(open_price, close) - low > 2 * body_size:
                return f"{direction}_hammer" if prev_direction == "down" else f"{direction}_hanging_man"
            else:
                return f"{direction}_doji"
        elif body_ratio > 0.8:
            return f"strong_{direction}"
        else:
            return direction
    
    def get_features(self, symbol: str, timeframe: str, 
                    include_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate and return requested volume features.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data
            include_features: List of features to include, or None for all
            
        Returns:
            Dict with feature name keys and calculated values
        """
        # Default to all features if not specified
        if include_features is None:
            include_features = [
                'vwap', 'volume_profile', 'volume_zones',
                'buying_selling_pressure', 'volume_breakdown', 'volume_anomalies'
            ]
        
        features = {}
        
        # Calculate requested features
        for feature in include_features:
            if feature == 'vwap':
                features['vwap'] = self.calculate_vwap(symbol, timeframe)
            elif feature == 'volume_profile':
                features['volume_profile'] = self.calculate_volume_profile(symbol, timeframe)
            elif feature == 'volume_zones':
                features['volume_zones'] = self.calculate_volume_zones(symbol, timeframe)
            elif feature == 'buying_selling_pressure':
                features['buying_selling_pressure'] = self.calculate_buying_selling_pressure(symbol, timeframe)
            elif feature == 'volume_breakdown':
                features['volume_breakdown'] = self.calculate_volume_breakdown(symbol, timeframe)
            elif feature == 'volume_anomalies':
                features['volume_anomalies'] = self.detect_volume_anomalies(symbol, timeframe)
        
        return features
    
    def clear_cache(self):
        """Clear the volume calculation cache."""
        self.volume_cache.clear()
        logger.info("Volume features cache cleared")

class VolumeProfile:
    """
    Class for volume profile analysis and visualization.
    Provides detailed analysis of volume distribution across price levels.
    """
    
    def __init__(self, price_data, volume_data, num_bins=50, value_area_pct=0.68):
        """
        Initialize the volume profile analyzer.
        
        Args:
            price_data: Series or array of price data
            volume_data: Series or array of volume data
            num_bins: Number of price bins to use
            value_area_pct: Percentage of volume to include in value area (default: 68%)
        """
        self.price_data = np.array(price_data)
        self.volume_data = np.array(volume_data)
        self.num_bins = num_bins
        self.value_area_pct = value_area_pct
        
        # Calculate profile
        self.profile = self._calculate_profile()
        
    def _calculate_profile(self):
        """Calculate the volume profile"""
        if len(self.price_data) == 0 or len(self.volume_data) == 0:
            return None
            
        # Create price bins
        price_min = np.min(self.price_data)
        price_max = np.max(self.price_data)
        bins = np.linspace(price_min, price_max, self.num_bins)
        
        # Initialize volume by price
        volume_by_price = {float(price): 0.0 for price in bins}
        
        # Distribute volume across price levels
        for i in range(len(self.price_data)):
            price = self.price_data[i]
            volume = self.volume_data[i]
            
            # Find closest bin
            closest_bin = bins[np.abs(bins - price).argmin()]
            volume_by_price[float(closest_bin)] += volume
            
        # Find point of control (price with highest volume)
        poc = max(volume_by_price, key=volume_by_price.get)
        
        # Calculate value area
        total_volume = sum(volume_by_price.values())
        sorted_prices = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)
        
        running_vol = 0
        value_area_prices = []
        
        for price, vol in sorted_prices:
            running_vol += vol
            value_area_prices.append(price)
            if running_vol >= total_volume * self.value_area_pct:
                break
                
        value_area_high = max(value_area_prices)
        value_area_low = min(value_area_prices)
        
        return {
            "volume_by_price": volume_by_price,
            "point_of_control": poc,
            "value_area_high": value_area_high,
            "value_area_low": value_area_low,
            "total_volume": total_volume
        }
        
    def get_profile_data(self):
        """Get the calculated profile data"""
        return self.profile
        
    def get_value_area(self):
        """Get the value area (where 68% of volume is traded)"""
        if not self.profile:
            return None
            
        return {
            "high": self.profile["value_area_high"],
            "low": self.profile["value_area_low"]
        }
        
    def get_point_of_control(self):
        """Get the point of control (price with highest volume)"""
        if not self.profile:
            return None
            
class VolumeProfiler:
    """
    Volume profiler class that provides advanced volume profile analysis for support and resistance detection.
    This class is used by the support_resistance module to identify key volume-based levels.
    """
    
    def __init__(self, price_data=None, volume_data=None, num_bins=50, value_area_pct=0.68):
        """
        Initialize the volume profiler.
        
        Args:
            price_data: Price data (high, low, close)
            volume_data: Volume data
            num_bins: Number of bins for the volume profile
            value_area_pct: Percentage of volume to include in value area
        """
        self.price_data = price_data
        self.volume_data = volume_data
        self.num_bins = num_bins
        self.value_area_pct = value_area_pct
        self.profile = None
        
        if price_data is not None and volume_data is not None:
            self.calculate_profile()
    
    def set_data(self, price_data, volume_data):
        """
        Set price and volume data.
        
        Args:
            price_data: Price data (high, low, close)
            volume_data: Volume data
        """
        self.price_data = price_data
        self.volume_data = volume_data
        self.calculate_profile()
    
    def calculate_profile(self):
        """
        Calculate the volume profile.
        """
        if self.price_data is None or self.volume_data is None:
            return None
            
        # Create a VolumeProfile instance
        self.profile = VolumeProfile(self.price_data, self.volume_data, self.num_bins, self.value_area_pct)
        self.profile._calculate_profile()
        
        return self.profile
    
    def get_value_area(self):
        """
        Get the value area (range containing specified percentage of volume).
        
        Returns:
            Tuple of (value_area_low, value_area_high)
        """
        if self.profile is None:
            return None
            
        return self.profile.get_value_area()
    
    def get_point_of_control(self):
        """
        Get the point of control (price level with highest volume).
        
        Returns:
            Point of control price
        """
        if self.profile is None:
            return None
            
        return self.profile.get_point_of_control()
    
    def get_high_volume_nodes(self):
        """
        Get high volume nodes (price levels with significantly high volume).
        
        Returns:
            List of high volume price levels
        """
        if self.profile is None:
            return []
            
        # Extract high volume nodes from profile
        volume_by_price = self.profile.volume_by_price
        if not volume_by_price:
            return []
            
        # Calculate threshold for high volume
        mean_vol = np.mean(list(volume_by_price.values()))
        std_vol = np.std(list(volume_by_price.values()))
        high_volume_threshold = mean_vol + std_vol
        
        # Find high volume nodes
        high_volume_nodes = [price for price, vol in volume_by_price.items() 
                           if vol > high_volume_threshold]
        
        return sorted(high_volume_nodes)
    
    def get_low_volume_nodes(self):
        """
        Get low volume nodes (price levels with significantly low volume).
        
        Returns:
            List of low volume price levels
        """
        if self.profile is None:
            return []
            
        # Extract low volume nodes from profile
        volume_by_price = self.profile.volume_by_price
        if not volume_by_price:
            return []
            
        # Calculate threshold for low volume
        mean_vol = np.mean(list(volume_by_price.values()))
        std_vol = np.std(list(volume_by_price.values()))
        low_volume_threshold = mean_vol - (0.8 * std_vol)
        
        # Find low volume nodes
        low_volume_nodes = [price for price, vol in volume_by_price.items() 
                          if vol < low_volume_threshold and vol > 0]
        
        return sorted(low_volume_nodes)
    
    def get_volume_profile_zones(self):
        """
        Get support and resistance zones based on volume profile.
        
        Returns:
            Dict with 'support' and 'resistance' zones
        """
        if self.profile is None:
            return {'support': [], 'resistance': []}
            
        # Get high volume nodes
        high_volume_nodes = self.get_high_volume_nodes()
        
        # Get current price (last price in the data)
        if isinstance(self.price_data, dict):
            current_price = self.price_data.get('close', [])[-1] if self.price_data.get('close') else None
        elif isinstance(self.price_data, pd.DataFrame):
            current_price = self.price_data['close'].iloc[-1] if 'close' in self.price_data.columns else None
        else:
            current_price = None
            
        # Separate into support and resistance
        support_zones = []
        resistance_zones = []
        
        if current_price is not None:
            for price in high_volume_nodes:
                if price < current_price:
                    support_zones.append({
                        'price': price,
                        'strength': 0.8,  # Default strength
                        'type': 'volume'
                    })
                else:
                    resistance_zones.append({
                        'price': price,
                        'strength': 0.8,  # Default strength
                        'type': 'volume'
                    })
        
        return {
            'support': support_zones,
            'resistance': resistance_zones
        }
        return self.profile["point_of_control"]

class DepthOfMarketAnalysis:
    """
    Analyze market depth and order book data to identify liquidity zones,
    imbalances, and potential price movements based on order flow.
    """
    
    def __init__(self, order_book_data=None):
        """
        Initialize with optional order book data.
        
        Args:
            order_book_data: Optional order book data to analyze
        """
        self.order_book = order_book_data
        self.results = {}
        
    def set_order_book(self, order_book_data):
        """Set the order book data for analysis."""
        self.order_book = order_book_data
        self.results = {}
        
    def analyze_liquidity(self, price_range=0.05) -> Dict[str, Any]:
        """
        Analyze liquidity distribution in the order book.
        
        Args:
            price_range: Price range to analyze as percentage of current price
            
        Returns:
            Dictionary with liquidity analysis results
        """
        if self.order_book is None:
            return {"error": "No order book data available"}
            
        if not isinstance(self.order_book, dict) or 'bids' not in self.order_book or 'asks' not in self.order_book:
            return {"error": "Invalid order book format"}
            
        bids = self.order_book['bids']  # [[price, volume], ...]
        asks = self.order_book['asks']  # [[price, volume], ...]
        
        if not bids or not asks:
            return {"error": "Empty order book"}
            
        # Current mid price
        mid_price = (bids[0][0] + asks[0][0]) / 2
        
        # Define analysis range
        min_price = mid_price * (1 - price_range)
        max_price = mid_price * (1 + price_range)
        
        # Filter orders within range
        bids_in_range = [bid for bid in bids if bid[0] >= min_price]
        asks_in_range = [ask for ask in asks if ask[0] <= max_price]
        
        # Calculate total liquidity
        bid_liquidity = sum(bid[1] for bid in bids_in_range)
        ask_liquidity = sum(ask[1] for ask in asks_in_range)
        total_liquidity = bid_liquidity + ask_liquidity
        
        # Calculate liquidity imbalance
        if total_liquidity > 0:
            bid_ratio = bid_liquidity / total_liquidity
            ask_ratio = ask_liquidity / total_liquidity
            imbalance = bid_ratio - ask_ratio  # -1 to 1 scale
        else:
            bid_ratio = 0.5
            ask_ratio = 0.5
            imbalance = 0
            
        # Find liquidity clusters
        bid_clusters = self._find_liquidity_clusters(bids_in_range)
        ask_clusters = self._find_liquidity_clusters(asks_in_range)
        
        # Identify significant liquidity walls
        bid_walls = [cluster for cluster in bid_clusters if cluster['volume'] > bid_liquidity * 0.2]
        ask_walls = [cluster for cluster in ask_clusters if cluster['volume'] > ask_liquidity * 0.2]
        
        # Identify liquidity gaps
        bid_gaps = self._find_liquidity_gaps(bids_in_range)
        ask_gaps = self._find_liquidity_gaps(asks_in_range)
        
        results = {
            'mid_price': mid_price,
            'bid_liquidity': bid_liquidity,
            'ask_liquidity': ask_liquidity,
            'bid_ratio': bid_ratio,
            'ask_ratio': ask_ratio,
            'imbalance': imbalance,
            'bid_clusters': bid_clusters,
            'ask_clusters': ask_clusters,
            'bid_walls': bid_walls,
            'ask_walls': ask_walls,
            'bid_gaps': bid_gaps,
            'ask_gaps': ask_gaps
        }
        
        self.results['liquidity'] = results
        return results
        
    def _find_liquidity_clusters(self, orders, threshold=0.1):
        """Find clusters of liquidity in the order book."""
        if not orders:
            return []
            
        clusters = []
        current_cluster = {
            'start_price': orders[0][0],
            'end_price': orders[0][0],
            'volume': orders[0][1],
            'orders': 1
        }
        
        for i in range(1, len(orders)):
            current_price = orders[i][0]
            prev_price = orders[i-1][0]
            
            # If prices are close, extend current cluster
            if abs(current_price - prev_price) / prev_price < threshold:
                current_cluster['end_price'] = current_price
                current_cluster['volume'] += orders[i][1]
                current_cluster['orders'] += 1
            else:
                # Finalize current cluster and start a new one
                current_cluster['avg_price'] = (current_cluster['start_price'] + current_cluster['end_price']) / 2
                clusters.append(current_cluster)
                
                current_cluster = {
                    'start_price': current_price,
                    'end_price': current_price,
                    'volume': orders[i][1],
                    'orders': 1
                }
        
        # Add the last cluster
        if current_cluster:
            current_cluster['avg_price'] = (current_cluster['start_price'] + current_cluster['end_price']) / 2
            clusters.append(current_cluster)
            
        return clusters
        
    def _find_liquidity_gaps(self, orders, threshold=0.005):
        """Find gaps in liquidity in the order book."""
        if len(orders) < 2:
            return []
            
        gaps = []
        
        for i in range(1, len(orders)):
            current_price = orders[i][0]
            prev_price = orders[i-1][0]
            
            # Calculate gap size
            gap_size = abs(current_price - prev_price) / prev_price
            
            # If gap is significant
            if gap_size > threshold:
                gaps.append({
                    'start_price': prev_price,
                    'end_price': current_price,
                    'gap_size': gap_size,
                    'gap_percent': gap_size * 100
                })
                
        return gaps
        
    def detect_spoofing(self) -> List[Dict[str, Any]]:
        """
        Detect potential spoofing patterns in the order book.
        
        Returns:
            List of potential spoofing incidents
        """
        if not self.order_book or 'updates' not in self.order_book:
            return []
            
        # This requires order book update history
        updates = self.order_book['updates']
        
        spoofing_incidents = []
        large_orders = {}
        
        # Track large orders that appear and disappear
        for update in updates:
            timestamp = update['timestamp']
            action = update['action']  # 'add', 'modify', 'remove'
            order_id = update.get('order_id')
            price = update.get('price')
            volume = update.get('volume', 0)
            
            if action == 'add' and volume > self.order_book.get('avg_order_size', 0) * 5:
                # Track large order
                large_orders[order_id] = {
                    'timestamp': timestamp,
                    'price': price,
                    'volume': volume
                }
            
            elif action == 'remove' and order_id in large_orders:
                # Order was removed
                added_time = large_orders[order_id]['timestamp']
                time_alive = timestamp - added_time
                
                # If large order was short-lived (potential spoofing)
                if time_alive < 5000:  # 5 seconds
                    spoofing_incidents.append({
                        'order_id': order_id,
                        'price': large_orders[order_id]['price'],
                        'volume': large_orders[order_id]['volume'],
                        'time_alive_ms': time_alive,
                        'added_at': added_time,
                        'removed_at': timestamp
                    })
                
                # Remove from tracking
                del large_orders[order_id]
                
        return spoofing_incidents

class VolumeAnalysis:
    """
    Advanced volume analysis class for detecting volume patterns, anomalies,
    and providing insights into market behavior based on volume data.
    """
    
    def __init__(self, ohlcv_data: pd.DataFrame = None):
        """
        Initialize the volume analysis with optional OHLCV data.
        
        Args:
            ohlcv_data: Optional OHLCV DataFrame to analyze
        """
        self.data = ohlcv_data
        self.results = {}
        
    def set_data(self, ohlcv_data: pd.DataFrame):
        """Set the OHLCV data for analysis."""
        self.data = ohlcv_data
        self.results = {}  # Reset results when data changes
        
    def analyze_volume_patterns(self) -> Dict[str, Any]:
        """
        Analyze volume patterns in the data.
        
        Returns:
            Dictionary of detected volume patterns
        """
        if self.data is None or len(self.data) < 20:
            return {"error": "Insufficient data for volume pattern analysis"}
            
        patterns = {}
        
        # Calculate volume moving average
        volume_ma = self.data['volume'].rolling(window=20).mean()
        
        # Detect volume spikes (2x average)
        volume_ratio = self.data['volume'] / volume_ma
        spikes = self.data[volume_ratio > 2.0]
        
        if not spikes.empty:
            patterns['volume_spikes'] = [{
                'timestamp': idx.strftime('%Y-%m-%d %H:%M:%S'),
                'price': row['close'],
                'volume': row['volume'],
                'volume_ratio': volume_ratio[idx]
            } for idx, row in spikes.iterrows()]
            
        # Detect volume climax (3 consecutive bars of increasing volume)
        climax_points = []
        for i in range(2, len(self.data)):
            if (self.data['volume'].iloc[i] > self.data['volume'].iloc[i-1] >
                self.data['volume'].iloc[i-2] and
                self.data['volume'].iloc[i] > volume_ma.iloc[i] * 1.5):
                
                climax_points.append({
                    'timestamp': self.data.index[i].strftime('%Y-%m-%d %H:%M:%S'),
                    'price': self.data['close'].iloc[i],
                    'volume': self.data['volume'].iloc[i],
                    'volume_ratio': volume_ratio.iloc[i]
                })
                
        if climax_points:
            patterns['volume_climax'] = climax_points
            
        # Detect volume divergence
        divergence_points = []
        for i in range(20, len(self.data)):
            # Price making new high but volume declining
            if (self.data['close'].iloc[i] > self.data['close'].iloc[i-20:i].max() and
                self.data['volume'].iloc[i] < self.data['volume'].iloc[i-5:i].mean()):
                
                divergence_points.append({
                    'timestamp': self.data.index[i].strftime('%Y-%m-%d %H:%M:%S'),
                    'price': self.data['close'].iloc[i],
                    'volume': self.data['volume'].iloc[i],
                    'type': 'bearish_divergence'
                })
                
            # Price making new low but volume declining
            if (self.data['close'].iloc[i] < self.data['close'].iloc[i-20:i].min() and
                self.data['volume'].iloc[i] < self.data['volume'].iloc[i-5:i].mean()):
                
                divergence_points.append({
                    'timestamp': self.data.index[i].strftime('%Y-%m-%d %H:%M:%S'),
                    'price': self.data['close'].iloc[i],
                    'volume': self.data['volume'].iloc[i],
                    'type': 'bullish_divergence'
                })
                
        if divergence_points:
            patterns['volume_divergence'] = divergence_points
            
        self.results['patterns'] = patterns
        return patterns
        
    def get_volume_profile(self, num_bins: int = 50) -> Dict[str, Any]:
        """
        Calculate volume profile (volume by price).
        
        Args:
            num_bins: Number of price bins to use
            
        Returns:
            Dictionary with volume profile data
        """
        if self.data is None or len(self.data) < 5:
            return {"error": "Insufficient data for volume profile"}
            
        # Create price bins
        price_min = self.data['low'].min()
        price_max = self.data['high'].max()
        price_bins = np.linspace(price_min, price_max, num_bins)
        
        # Initialize volume by price
        volume_by_price = {float(price): 0.0 for price in price_bins}
        
        # Distribute volume across price levels
        for i in range(len(self.data)):
            low = self.data['low'].iloc[i]
            high = self.data['high'].iloc[i]
            close = self.data['close'].iloc[i]
            volume = self.data['volume'].iloc[i]
            
            # Find price bins that fall within this candle's range
            for price in price_bins:
                if low <= price <= high:
                    # Weight by proximity to close
                    weight = 1.0 - abs(price - close) / (high - low + 0.0001)
                    volume_by_price[float(price)] += volume * weight
        
        # Find point of control (price with highest volume)
        poc = max(volume_by_price, key=volume_by_price.get)
        
        # Calculate value area (70% of volume)
        total_volume = sum(volume_by_price.values())
        sorted_prices = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)
        
        running_vol = 0
        value_area_prices = []
        
        for price, vol in sorted_prices:
            running_vol += vol
            value_area_prices.append(price)
            if running_vol >= total_volume * 0.7:
                break
                
        value_area_high = max(value_area_prices)
        value_area_low = min(value_area_prices)
        
        profile = {
            'volume_by_price': volume_by_price,
            'point_of_control': poc,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'total_volume': total_volume
        }
        
        self.results['profile'] = profile
        return profile
        
    def get_buying_selling_pressure(self) -> Dict[str, Any]:
        """
        Calculate buying and selling pressure based on candle analysis.
        
        Returns:
            Dictionary with buying/selling pressure metrics
        """
        if self.data is None or len(self.data) < 5:
            return {"error": "Insufficient data for pressure analysis"}
            
        buying_volume = 0
        selling_volume = 0
        neutral_volume = 0
        
        for i in range(len(self.data)):
            open_price = self.data['open'].iloc[i]
            close_price = self.data['close'].iloc[i]
            volume = self.data['volume'].iloc[i]
            
            if close_price > open_price:  # Bullish candle
                buying_volume += volume * 0.7
                selling_volume += volume * 0.3
            elif close_price < open_price:  # Bearish candle
                buying_volume += volume * 0.3
                selling_volume += volume * 0.7
            else:  # Neutral candle
                buying_volume += volume * 0.5
                selling_volume += volume * 0.5
                neutral_volume += volume * 0.0  # No neutral allocation in this model
        
        total_volume = buying_volume + selling_volume + neutral_volume
        
        if total_volume > 0:
            buying_pressure = (buying_volume / total_volume) * 100
            selling_pressure = (selling_volume / total_volume) * 100
        else:
            buying_pressure = 50
            selling_pressure = 50
            
        net_pressure = buying_pressure - selling_pressure
        
        pressure = {
            'buying_volume': buying_volume,
            'selling_volume': selling_volume,
            'neutral_volume': neutral_volume,
            'buying_pressure': buying_pressure,
            'selling_pressure': selling_pressure,
            'net_pressure': net_pressure
        }
        
        self.results['pressure'] = pressure
        return pressure

class VolumeFeature(BaseFeature):
    """Volume-based feature calculations"""
    def __init__(self, name="volume", description="Volume analysis features"):
        super().__init__(name, description)
        
    def compute(self, data):
        # Implementation
        pass

class VolumeAnalyzer:
    """
    Advanced volume analysis class for detecting volume-based patterns and anomalies.
    
    This class provides methods for analyzing trading volume to identify:
    - Volume anomalies and spikes
    - Volume-based support/resistance zones
    - Buying/selling pressure
    - Volume profile analysis
    - Volume trend analysis
    - Liquidity analysis
    """
    
    def __init__(self, ohlcv_data: pd.DataFrame = None):
        """
        Initialize the VolumeAnalyzer.
        
        Args:
            ohlcv_data: Optional DataFrame with OHLCV data
        """
        self.data = ohlcv_data
        self.logger = logging.getLogger(__name__)
        
    def set_data(self, ohlcv_data: pd.DataFrame):
        """
        Set the OHLCV data for analysis.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
        """
        self.data = ohlcv_data
        
    def detect_volume_anomalies(self, lookback_period: int = 20, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """
        Detect volume anomalies based on statistical analysis.
        
        Args:
            lookback_period: Period for calculating baseline volume
            threshold: Standard deviation threshold for anomaly detection
            
        Returns:
            List of detected volume anomalies with details
        """
        if self.data is None or len(self.data) < lookback_period + 1:
            return []
            
        # Calculate rolling mean and standard deviation
        rolling_mean = self.data['volume'].rolling(window=lookback_period).mean()
        rolling_std = self.data['volume'].rolling(window=lookback_period).std()
        
        # Identify anomalies
        anomalies = []
        
        for i in range(lookback_period, len(self.data)):
            volume = self.data['volume'].iloc[i]
            mean = rolling_mean.iloc[i]
            std = rolling_std.iloc[i]
            
            if std == 0:
                continue
                
            z_score = (volume - mean) / std
            
            if abs(z_score) > threshold:
                anomalies.append({
                    'index': i,
                    'timestamp': self.data.index[i] if hasattr(self.data, 'index') else i,
                    'volume': volume,
                    'baseline_volume': mean,
                    'z_score': z_score,
                    'price': self.data['close'].iloc[i],
                    'price_change': self.data['close'].iloc[i] / self.data['close'].iloc[i-1] - 1,
                    'is_bullish': self.data['close'].iloc[i] > self.data['open'].iloc[i]
                })
                
        return anomalies


class VolumeProfileAnalyzer:
    """
    Advanced volume profile analysis tool that provides detailed insights into
    price-volume relationships and market structure.
    """
    
    def __init__(self, ohlcv_data: pd.DataFrame = None, num_bins: int = 50, value_area_pct: float = 0.68):
        """
        Initialize the VolumeProfileAnalyzer.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            num_bins: Number of price bins for volume profile
            value_area_pct: Percentage of volume to include in value area (default: 68%)
        """
        self.ohlcv_data = ohlcv_data
        self.num_bins = num_bins
        self.value_area_pct = value_area_pct
        self.profile_data = None
        
    def set_data(self, ohlcv_data: pd.DataFrame):
        """Set OHLCV data for analysis."""
        self.ohlcv_data = ohlcv_data
        self.profile_data = None
        
    def calculate_profile(self, time_period: str = None) -> Dict[str, Any]:
        """
        Calculate volume profile for the given data.
        
        Args:
            time_period: Optional time period filter ('session', 'day', 'week', 'month')
            
        Returns:
            Dictionary with volume profile data
        """
        if self.ohlcv_data is None or len(self.ohlcv_data) == 0:
            return None
            
        # Filter data by time period if specified
        data = self.ohlcv_data
        if time_period:
            # Implement time period filtering logic here
            pass
            
        # Extract price and volume data
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values
        
        # Determine price range
        price_min = low.min()
        price_max = high.max()
        price_range = price_max - price_min
        
        # Create price bins
        bins = np.linspace(price_min, price_max, self.num_bins)
        bin_width = price_range / (self.num_bins - 1)
        
        # Initialize volume by price
        volume_by_price = {float(price): 0.0 for price in bins}
        
        # Calculate volume at each price level
        for i in range(len(data)):
            bar_low = low[i]
            bar_high = high[i]
            bar_close = close[i]
            bar_vol = volume[i]
            
            # Distribute volume across price points within the bar
            for price in bins:
                if bar_low <= price <= bar_high:
                    # Weight by proximity to close price
                    close_proximity = 1 - abs(price - bar_close) / (bar_high - bar_low + 0.0001)
                    proximity_weight = 0.5 + (0.5 * close_proximity)  # 0.5-1.0 range
                    
                    # Distribute volume proportionally
                    price_coverage = bin_width / (bar_high - bar_low + bin_width)
                    allocated_volume = bar_vol * price_coverage * proximity_weight
                    volume_by_price[float(price)] += allocated_volume
        
        # Find point of control (price with highest volume)
        point_of_control = max(volume_by_price, key=volume_by_price.get)
        
        # Calculate value area
        sorted_prices = sorted(volume_by_price.items(), key=lambda x: x[1], reverse=True)
        total_volume = sum(volume_by_price.values())
        running_vol = 0
        value_area_prices = []
        
        for price, vol in sorted_prices:
            running_vol += vol
            value_area_prices.append(price)
            if running_vol >= total_volume * self.value_area_pct:
                break
        
        value_area_high = max(value_area_prices)
        value_area_low = min(value_area_prices)
        
        # Store results
        self.profile_data = {
            'volume_by_price': volume_by_price,
            'point_of_control': point_of_control,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'total_volume': total_volume
        }
        
        return self.profile_data
    
    def identify_support_resistance(self, significance_threshold: float = 0.1) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify support and resistance levels based on volume profile.
        
        Args:
            significance_threshold: Minimum volume significance for a level
            
        Returns:
            Dictionary with support and resistance levels
        """
        if self.profile_data is None:
            self.calculate_profile()
            
        if self.profile_data is None:
            return {'support': [], 'resistance': []}
            
        volume_by_price = self.profile_data['volume_by_price']
        total_volume = self.profile_data['total_volume']
        
        # Get current price
        if self.ohlcv_data is None or len(self.ohlcv_data) == 0:
            current_price = None
        else:
            current_price = self.ohlcv_data['close'].iloc[-1]
            
        # Find significant volume nodes
        significant_nodes = []
        
        for price, volume in volume_by_price.items():
            significance = volume / total_volume
            if significance >= significance_threshold:
                significant_nodes.append({
                    'price': price,
                    'volume': volume,
                    'significance': significance
                })
                
        # Classify as support or resistance
        support = []
        resistance = []
        
        if current_price is not None:
            for node in significant_nodes:
                if node['price'] < current_price:
                    support.append(node)
                else:
                    resistance.append(node)
                    
            # Sort by significance
            support.sort(key=lambda x: x['significance'], reverse=True)
            resistance.sort(key=lambda x: x['significance'], reverse=True)
            
        return {
            'support': support,
            'resistance': resistance
        }
    
    def get_volume_distribution_type(self) -> str:
        """
        Analyze the volume distribution to determine its shape.
        
        Returns:
            String describing the distribution type
        """
        if self.profile_data is None:
            self.calculate_profile()
            
        if self.profile_data is None:
            return "unknown"
            
        volume_by_price = self.profile_data['volume_by_price']
        
        # Convert to arrays for analysis
        prices = np.array(list(volume_by_price.keys()))
        volumes = np.array(list(volume_by_price.values()))
        
        # Normalize volumes
        if volumes.sum() > 0:
            norm_volumes = volumes / volumes.sum()
        else:
            return "insufficient_data"
        
        # Calculate skewness
        mean_price = np.average(prices, weights=norm_volumes)
        variance = np.average((prices - mean_price)**2, weights=norm_volumes)
        std_dev = np.sqrt(variance)
        
        if std_dev == 0:
            return "single_point"
            
        skewness = np.average(((prices - mean_price) / std_dev)**3, weights=norm_volumes)
        
        # Smooth the volume curve
        smoothed = savgol_filter(norm_volumes, min(11, len(norm_volumes) - (len(norm_volumes) % 2 + 1)), 3)
        
        # Find peaks for bimodality testing
        peaks, _ = find_peaks(smoothed, height=np.max(smoothed) * 0.4, distance=len(smoothed) // 10)
        
        if len(peaks) >= 2:
            # Check if peaks are significant
            peak_heights = smoothed[peaks]
            peak_significance = peak_heights / np.max(peak_heights)
            
            if np.min(peak_significance) > 0.5 and np.max(prices[peaks]) - np.min(prices[peaks]) > std_dev:
                return "bimodal"
        
        # Check for skewness
        if skewness > 0.5:
            return "skewed_high"
        elif skewness < -0.5:
            return "skewed_low"
        else:
            return "normal"


def analyze_volume_profile(prices: Union[pd.Series, List[float]],
                           volumes: Union[pd.Series, List[float]],
                           periods: int = 20) -> Dict[str, pd.Series]:
    """Simple volume profile analysis used by strategy brains."""
    df = pd.DataFrame({"price": prices, "volume": volumes})
    volume_sma = df["volume"].rolling(periods).mean()
    relative_volume = df["volume"] / (volume_sma + 1e-9)
    return {"volume_sma": volume_sma, "relative_volume": relative_volume}


def detect_volume_climax(volumes: Union[pd.Series, List[float]],
                          prices: Union[pd.Series, List[float]],
                          lookback: int = 20) -> pd.Series:
    """Detect simple volume climax events as volume spikes."""
    volumes_series = pd.Series(volumes)
    avg_volume = volumes_series.rolling(lookback).mean()
    climax = volumes_series > avg_volume * 3
    return climax


def calculate_volume_profile(prices: Union[pd.Series, List[float]],
                             volumes: Union[pd.Series, List[float]],
                             periods: int = 100,
                             num_bins: int = 50) -> Dict[str, Any]:
    """Calculate a basic volume profile for the given price and volume data."""
    prices_arr = np.array(prices)[-periods:]
    volumes_arr = np.array(volumes)[-periods:]
    profiler = VolumeProfile(prices_arr, volumes_arr, num_bins=num_bins)
    return profiler.get_profile_data()


def relative_volume_profile(data: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Return volume divided by its moving average."""
    volume_ma = data['volume'].rolling(lookback).mean()
    return data['volume'] / volume_ma


def volume_breakout_confirmation(data: pd.DataFrame, threshold: float = 1.5) -> pd.Series:
    """Detect volume breakout based on relative volume profile."""
    rvp = relative_volume_profile(data)
    return rvp > threshold
