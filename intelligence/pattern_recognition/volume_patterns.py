#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Volume Pattern Recognition Module

This module provides advanced volume pattern recognition capabilities, identifying
significant volume patterns that can precede or confirm price movements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from datetime import datetime

from common.logger import get_logger
from common.exceptions import (
    PatternDetectionError, InsufficientDataError
)
from feature_service.features.volume import (
    VolumeProfile, VolumeAnalysis, DepthOfMarketAnalysis
)

logger = get_logger(__name__)

@dataclass
class VolumePattern:
    """Represents a detected volume pattern"""
    pattern_type: str
    start_idx: int
    end_idx: int
    confidence: float
    price_target: Optional[float] = None
    expected_direction: Optional[str] = None
    significance: float = 0.0
    notes: Optional[str] = None
    additional_data: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary for serialization"""
        return {
            'pattern_type': self.pattern_type,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'confidence': self.confidence,
            'price_target': self.price_target,
            'expected_direction': self.expected_direction,
            'significance': self.significance,
            'notes': self.notes,
            'additional_data': self.additional_data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VolumePattern':
        """Create pattern instance from dictionary"""
        return cls(
            pattern_type=data['pattern_type'],
            start_idx=data['start_idx'],
            end_idx=data['end_idx'],
            confidence=data['confidence'],
            price_target=data.get('price_target'),
            expected_direction=data.get('expected_direction'),
            significance=data.get('significance', 0.0),
            notes=data.get('notes'),
            additional_data=data.get('additional_data', {})
        )


class VolumePatternDetector:
    """Class for identifying patterns based on volume analysis"""

    def __init__(self, 
                 sensitivity: float = 0.5,
                 lookback_periods: int = 100,
                 min_confidence: float = 0.6,
                 volume_weight: float = 1.0,
                 price_correlation_threshold: float = 0.3):
        """
        Initialize the volume pattern detector.
        
        Args:
            sensitivity: Sensitivity factor for pattern detection (0.0-1.0)
            lookback_periods: Number of periods to look back for pattern formation
            min_confidence: Minimum confidence threshold for pattern reporting
            volume_weight: Weight factor for volume significance
            price_correlation_threshold: Minimum correlation between price and volume
        """
        self.sensitivity = max(0.01, min(0.99, sensitivity))
        self.lookback_periods = lookback_periods
        self.min_confidence = min_confidence
        self.volume_weight = volume_weight
        self.price_correlation_threshold = price_correlation_threshold
        self.volume_analyzer = VolumeAnalysis()
        self.dom_analyzer = DepthOfMarketAnalysis()
        self._patterns_cache = {}
        logger.info(f"Initialized VolumePatternDetector with sensitivity={sensitivity}, "
                   f"lookback_periods={lookback_periods}")

    def find_patterns(self, 
                     df: pd.DataFrame, 
                     additional_data: Optional[Dict[str, Any]] = None) -> List[VolumePattern]:
        """
        Detect volume patterns in the provided data.
        
        Args:
            df: DataFrame with OHLCV data
            additional_data: Optional dictionary with additional data like depth of market
            
        Returns:
            List of detected volume patterns
        """
        if len(df) < self.lookback_periods:
            logger.warning(f"Insufficient data for volume pattern detection. "
                          f"Required: {self.lookback_periods}, Provided: {len(df)}")
            return []
            
        # Get cache key based on last timestamp and data shape
        cache_key = f"{df.index[-1].strftime('%Y%m%d%H%M%S')}_{len(df)}"
        if cache_key in self._patterns_cache:
            logger.debug(f"Returning cached volume patterns for {cache_key}")
            return self._patterns_cache[cache_key]
        
        try:
            # Identify all types of volume patterns
            patterns = []
            patterns.extend(self._detect_volume_climax(df))
            patterns.extend(self._detect_volume_divergence(df))
            patterns.extend(self._detect_volume_trend_confirmation(df))
            patterns.extend(self._detect_buying_selling_pressure(df, additional_data))
            patterns.extend(self._detect_abnormal_volume_spikes(df))
            patterns.extend(self._detect_volume_exhaustion(df))
            patterns.extend(self._detect_volume_contraction(df))
            
            # Filter by minimum confidence
            patterns = [p for p in patterns if p.confidence >= self.min_confidence]
            
            # Cache the results
            self._patterns_cache[cache_key] = patterns
            
            # Keep cache size manageable
            if len(self._patterns_cache) > 100:
                # Remove oldest entries
                oldest_keys = sorted(self._patterns_cache.keys())[:50]
                for key in oldest_keys:
                    del self._patterns_cache[key]
                    
            return patterns
            
        except Exception as e:
            logger.error(f"Error in volume pattern detection: {str(e)}", exc_info=True)
            raise PatternDetectionError(f"Volume pattern detection failed: {str(e)}")

    def _detect_volume_climax(self, df: pd.DataFrame) -> List[VolumePattern]:
        """
        Detect volume climax patterns (capitulation, exhaustion).
        
        Volume climax occurs when there's an extreme spike in volume, often at
        the end of a trend, signaling potential reversal.
        """
        patterns = []
        volume = df['volume'].values
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Calculate rolling window statistics
        vol_mean = np.convolve(volume, np.ones(20)/20, mode='valid')
        vol_std = np.array([np.std(volume[max(0, i-20):i]) for i in range(len(volume))])
        
        # Add padding to match original length
        padding = len(volume) - len(vol_mean)
        vol_mean = np.append(np.array([np.nan] * padding), vol_mean)
        
        # Look for volume spikes at turning points
        for i in range(30, len(df)-1):
            # Check for volume spike (3+ standard deviations)
            if volume[i] > vol_mean[i] + self.sensitivity * 3 * vol_std[i]:
                # Check for price exhaustion
                price_change_before = (close[i] - close[i-5]) / close[i-5]
                price_change_after = (close[i+1] - close[i]) / close[i]
                
                # Selling climax: high volume + price reversal from down to up
                if price_change_before < -0.01 and price_change_after > 0.0025:
                    confidence = min(1.0, (volume[i] / vol_mean[i]) * self.sensitivity)
                    patterns.append(VolumePattern(
                        pattern_type='selling_climax',
                        start_idx=i-5,
                        end_idx=i,
                        confidence=confidence,
                        expected_direction='up',
                        significance=min(1.0, volume[i] / (vol_mean[i] * 2)),
                        notes=f"Selling climax with {volume[i]/vol_mean[i]:.2f}x normal volume"
                    ))
                
                # Buying climax: high volume + price reversal from up to down
                elif price_change_before > 0.01 and price_change_after < -0.0025:
                    confidence = min(1.0, (volume[i] / vol_mean[i]) * self.sensitivity)
                    patterns.append(VolumePattern(
                        pattern_type='buying_climax',
                        start_idx=i-5,
                        end_idx=i,
                        confidence=confidence,
                        expected_direction='down',
                        significance=min(1.0, volume[i] / (vol_mean[i] * 2)),
                        notes=f"Buying climax with {volume[i]/vol_mean[i]:.2f}x normal volume"
                    ))
                    
        return patterns

    def _detect_volume_divergence(self, df: pd.DataFrame) -> List[VolumePattern]:
        """
        Detect volume divergence patterns.
        
        Volume divergence occurs when price makes a new high/low but volume doesn't
        confirm, indicating potential weakness in the trend.
        """
        patterns = []
        volume = df['volume'].values
        close = df['close'].values
        
        # Calculate rolling window statistics
        vol_sma_20 = np.convolve(volume, np.ones(20)/20, mode='valid')
        padding = len(volume) - len(vol_sma_20)
        vol_sma_20 = np.append(np.array([np.nan] * padding), vol_sma_20)
        
        # Look for price making new highs/lows with decreasing volume
        for i in range(20, len(df)-10):
            # Check for new price high in last 20 bars
            if close[i] >= np.max(close[i-20:i]):
                # Check if volume is decreasing
                vol_trend = np.polyfit(np.arange(10), volume[i-9:i+1], 1)[0]
                if vol_trend < 0:
                    # Bearish volume divergence: new price high with decreasing volume
                    confidence = min(1.0, abs(vol_trend) / np.mean(volume[i-9:i+1]) * 20 * self.sensitivity)
                    if confidence >= self.min_confidence:
                        patterns.append(VolumePattern(
                            pattern_type='bearish_volume_divergence',
                            start_idx=i-9,
                            end_idx=i,
                            confidence=confidence,
                            expected_direction='down',
                            significance=min(1.0, abs(vol_trend) / np.mean(volume[i-9:i+1]) * 15),
                            notes=f"Price making new highs with decreasing volume"
                        ))
            
            # Check for new price low in last 20 bars
            if close[i] <= np.min(close[i-20:i]):
                # Check if volume is decreasing
                vol_trend = np.polyfit(np.arange(10), volume[i-9:i+1], 1)[0]
                if vol_trend < 0:
                    # Bullish volume divergence: new price low with decreasing volume
                    confidence = min(1.0, abs(vol_trend) / np.mean(volume[i-9:i+1]) * 20 * self.sensitivity)
                    if confidence >= self.min_confidence:
                        patterns.append(VolumePattern(
                            pattern_type='bullish_volume_divergence',
                            start_idx=i-9,
                            end_idx=i,
                            confidence=confidence,
                            expected_direction='up',
                            significance=min(1.0, abs(vol_trend) / np.mean(volume[i-9:i+1]) * 15),
                            notes=f"Price making new lows with decreasing volume"
                        ))
        
        return patterns

    def _detect_volume_trend_confirmation(self, df: pd.DataFrame) -> List[VolumePattern]:
        """
        Detect volume trend confirmation patterns.
        
        Volume should increase in the direction of the trend. This pattern identifies
        strong volume confirming the current price trend.
        """
        patterns = []
        volume = df['volume'].values
        close = df['close'].values
        
        # Calculate price and volume trends
        for i in range(20, len(df)-1):
            # Calculate 10-day price and volume trends
            price_trend = np.polyfit(np.arange(10), close[i-9:i+1], 1)[0]
            vol_trend = np.polyfit(np.arange(10), volume[i-9:i+1], 1)[0]
            
            # Calculate average volume
            avg_volume = np.mean(volume[i-19:i+1])
            
            # Bullish confirmation: rising prices with rising volume
            if price_trend > 0 and vol_trend > 0:
                # Calculate how strong the confirmation is
                price_change_pct = (close[i] - close[i-9]) / close[i-9]
                vol_change_pct = (np.mean(volume[i-4:i+1]) - np.mean(volume[i-9:i-4])) / np.mean(volume[i-9:i-4])
                
                # Strong trend needs both significant price and volume increases
                if price_change_pct > 0.01 and vol_change_pct > 0.15:
                    confidence = min(1.0, (price_change_pct * 50 + vol_change_pct * 2) * self.sensitivity)
                    if confidence >= self.min_confidence:
                        patterns.append(VolumePattern(
                            pattern_type='bullish_volume_confirmation',
                            start_idx=i-9,
                            end_idx=i,
                            confidence=confidence,
                            expected_direction='up',
                            significance=min(1.0, vol_change_pct * 2),
                            notes=f"Uptrend confirmed with {vol_change_pct*100:.1f}% volume increase"
                        ))
            
            # Bearish confirmation: falling prices with rising volume
            elif price_trend < 0 and vol_trend > 0:
                # Calculate how strong the confirmation is
                price_change_pct = (close[i] - close[i-9]) / close[i-9]
                vol_change_pct = (np.mean(volume[i-4:i+1]) - np.mean(volume[i-9:i-4])) / np.mean(volume[i-9:i-4])
                
                # Strong trend needs both significant price decrease and volume increase
                if price_change_pct < -0.01 and vol_change_pct > 0.15:
                    confidence = min(1.0, (abs(price_change_pct) * 50 + vol_change_pct * 2) * self.sensitivity)
                    if confidence >= self.min_confidence:
                        patterns.append(VolumePattern(
                            pattern_type='bearish_volume_confirmation',
                            start_idx=i-9,
                            end_idx=i,
                            confidence=confidence,
                            expected_direction='down',
                            significance=min(1.0, vol_change_pct * 2),
                            notes=f"Downtrend confirmed with {vol_change_pct*100:.1f}% volume increase"
                        ))
        
        return patterns

    def _detect_buying_selling_pressure(self, 
                                       df: pd.DataFrame,
                                       additional_data: Optional[Dict[str, Any]] = None) -> List[VolumePattern]:
        """
        Detect buying and selling pressure using volume and price action.
        
        This method looks at where price closes within its range and the corresponding
        volume to determine buying or selling pressure.
        """
        patterns = []
        
        if len(df) < 20:
            return patterns
            
        # Extract data
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values
        
        # Check for depth of market data
        dom_data = None
        if additional_data and 'depth_of_market' in additional_data:
            dom_data = additional_data['depth_of_market']
        
        for i in range(20, len(df)-1):
            # Calculate where price closed in its range (0 = at low, 1 = at high)
            range_position = (closes[i] - lows[i]) / (highs[i] - lows[i]) if (highs[i] - lows[i]) > 0 else 0.5
            
            # Calculate relative volume
            rel_volume = volumes[i] / np.mean(volumes[i-20:i])
            
            # Strong buying pressure: high volume + close near high
            if rel_volume > 1.5 and range_position > 0.8:
                # Check depth of market data if available
                dom_confirmation = 1.0
                if dom_data is not None and i < len(dom_data):
                    # Use DOM data to analyze buy/sell imbalance
                    dom_confirmation = self.dom_analyzer.calculate_buy_pressure(dom_data[i])
                
                confidence = min(1.0, rel_volume * range_position * dom_confirmation * self.sensitivity)
                if confidence >= self.min_confidence:
                    patterns.append(VolumePattern(
                        pattern_type='strong_buying_pressure',
                        start_idx=i,
                        end_idx=i,
                        confidence=confidence,
                        expected_direction='up',
                        significance=min(1.0, rel_volume * 0.4),
                        notes=f"Strong buying with {rel_volume:.1f}x volume, closing at {range_position*100:.0f}% of range"
                    ))
            
            # Strong selling pressure: high volume + close near low
            elif rel_volume > 1.5 and range_position < 0.2:
                # Check depth of market data if available
                dom_confirmation = 1.0
                if dom_data is not None and i < len(dom_data):
                    # Use DOM data to analyze buy/sell imbalance
                    dom_confirmation = self.dom_analyzer.calculate_sell_pressure(dom_data[i])
                
                confidence = min(1.0, rel_volume * (1 - range_position) * dom_confirmation * self.sensitivity)
                if confidence >= self.min_confidence:
                    patterns.append(VolumePattern(
                        pattern_type='strong_selling_pressure',
                        start_idx=i,
                        end_idx=i,
                        confidence=confidence,
                        expected_direction='down',
                        significance=min(1.0, rel_volume * 0.4),
                        notes=f"Strong selling with {rel_volume:.1f}x volume, closing at {range_position*100:.0f}% of range"
                    ))
        
        return patterns

    def _detect_abnormal_volume_spikes(self, df: pd.DataFrame) -> List[VolumePattern]:
        """
        Detect abnormal volume spikes that may indicate significant events.
        
        Extremely high volume spikes often indicate important market events and
        potential turning points.
        """
        patterns = []
        volumes = df['volume'].values
        
        if len(volumes) < 20:
            return patterns
            
        # Calculate rolling statistics
        for i in range(20, len(volumes)):
            # Get baseline volume
            baseline_vol = np.median(volumes[i-20:i])
            vol_std = np.std(volumes[i-20:i])
            
            # Check for abnormal spike
            if volumes[i] > baseline_vol + 3 * vol_std:
                # Calculate z-score
                z_score = (volumes[i] - baseline_vol) / vol_std
                
                # More extreme spikes get higher confidence
                confidence = min(1.0, (z_score - 2) / 8 * self.sensitivity)
                
                if confidence >= self.min_confidence:
                    # Determine if bullish or bearish
                    if df['close'].iloc[i] > df['open'].iloc[i]:
                        pattern_type = 'bullish_volume_spike'
                        direction = 'up'
                    else:
                        pattern_type = 'bearish_volume_spike'
                        direction = 'down'
                        
                    patterns.append(VolumePattern(
                        pattern_type=pattern_type,
                        start_idx=i,
                        end_idx=i,
                        confidence=confidence,
                        expected_direction=direction,
                        significance=min(1.0, z_score / 15),
                        notes=f"Abnormal volume spike ({z_score:.1f} std dev above normal)"
                    ))
        
        return patterns

    def _detect_volume_exhaustion(self, df: pd.DataFrame) -> List[VolumePattern]:
        """
        Detect volume exhaustion patterns.
        
        Volume exhaustion occurs when volume steadily decreases during a trend,
        potentially indicating weakening momentum and upcoming reversal.
        """
        patterns = []
        
        if len(df) < 20:
            return patterns
            
        close = df['close'].values
        volume = df['volume'].values
        
        for i in range(20, len(df)-1):
            # Check for a clear price trend
            price_change = (close[i] - close[i-10]) / close[i-10]
            
            # If we have a significant trend, check for volume exhaustion
            if abs(price_change) > 0.02:
                # Check if volume is declining
                recent_vols = volume[i-9:i+1]
                vol_trend = np.polyfit(np.arange(len(recent_vols)), recent_vols, 1)[0]
                
                # Significant volume decay
                if vol_trend < 0 and abs(vol_trend) > np.mean(recent_vols) * 0.03:
                    # Calculate how severe the exhaustion is
                    start_vol = np.mean(volume[i-9:i-6])
                    end_vol = np.mean(volume[i-2:i+1])
                    vol_decay = (start_vol - end_vol) / start_vol
                    
                    if vol_decay > 0.25:  # At least 25% volume decrease
                        confidence = min(1.0, vol_decay * self.sensitivity * 1.5)
                        
                        if confidence >= self.min_confidence:
                            # Bullish exhaustion (downtrend losing steam)
                            if price_change < 0:
                                patterns.append(VolumePattern(
                                    pattern_type='bullish_volume_exhaustion',
                                    start_idx=i-9,
                                    end_idx=i,
                                    confidence=confidence,
                                    expected_direction='up',
                                    significance=min(1.0, vol_decay * 1.2),
                                    notes=f"Selling exhaustion with {vol_decay*100:.0f}% volume decrease"
                                ))
                            # Bearish exhaustion (uptrend losing steam)
                            else:
                                patterns.append(VolumePattern(
                                    pattern_type='bearish_volume_exhaustion',
                                    start_idx=i-9,
                                    end_idx=i,
                                    confidence=confidence,
                                    expected_direction='down',
                                    significance=min(1.0, vol_decay * 1.2),
                                    notes=f"Buying exhaustion with {vol_decay*100:.0f}% volume decrease"
                                ))
        
        return patterns

    def _detect_volume_contraction(self, df: pd.DataFrame) -> List[VolumePattern]:
        """
        Detect volume contraction patterns.
        
        Volume contraction occurs when volume significantly decreases, often
        preceding explosive price moves.
        """
        patterns = []
        
        if len(df) < 30:
            return patterns
            
        volume = df['volume'].values
        close = df['close'].values
        
        for i in range(30, len(df)-1):
            # Calculate recent volume stats
            recent_vol_avg = np.mean(volume[i-5:i+1])
            longer_vol_avg = np.mean(volume[i-30:i-5])
            
            # Look for significant volume contraction
            if recent_vol_avg < longer_vol_avg * 0.6:
                # Calculate contraction ratio
                contraction_ratio = longer_vol_avg / recent_vol_avg
                
                # Price volatility contraction (narrow range)
                recent_ranges = [df['high'].iloc[j] - df['low'].iloc[j] for j in range(i-5, i+1)]
                longer_ranges = [df['high'].iloc[j] - df['low'].iloc[j] for j in range(i-30, i-5)]
                
                range_contraction = np.mean(longer_ranges) / np.mean(recent_ranges) if np.mean(recent_ranges) > 0 else 1.0
                
                # Strong pattern: both volume AND price range contracting
                if range_contraction > 1.2 and contraction_ratio > 1.5:
                    confidence = min(1.0, (contraction_ratio - 1) * 0.5 * self.sensitivity)
                    
                    if confidence >= self.min_confidence:
                        patterns.append(VolumePattern(
                            pattern_type='volume_and_range_contraction',
                            start_idx=i-10,
                            end_idx=i,
                            confidence=confidence,
                            # Direction not predictable from contraction alone
                            expected_direction=None,
                            significance=min(1.0, (contraction_ratio - 1) * 0.5),
                            notes=f"Volume contracted {contraction_ratio:.1f}x with narrowing price range"
                        ))
        
        return patterns

    def reset_cache(self):
        """Clear the pattern cache"""
        self._patterns_cache = {}
        logger.debug("Volume pattern detector cache reset")

    def adjust_sensitivity(self, new_sensitivity: float):
        """Adjust detector sensitivity"""
        self.sensitivity = max(0.01, min(0.99, new_sensitivity))
        self.reset_cache()
        logger.info(f"Volume pattern detector sensitivity adjusted to {self.sensitivity}")

    def adjust_lookback(self, new_lookback: int):
        """Adjust lookback period"""
        self.lookback_periods = max(10, new_lookback)
        self.reset_cache()
        logger.info(f"Volume pattern detector lookback adjusted to {self.lookback_periods}")

    def analyze_volume_at_price_levels(self, 
                                      df: pd.DataFrame, 
                                      volume_profile: VolumeProfile) -> Dict[str, Any]:
        """
        Analyze volume distribution across price levels.
        
        Args:
            df: DataFrame with OHLCV data
            volume_profile: Volume profile object with volume-at-price data
            
        Returns:
            Dictionary with volume analysis at key price levels
        """
        analysis = {}
        
        try:
            # Find high volume nodes (price levels with high volume)
            high_vol_nodes = volume_profile.get_high_volume_nodes(threshold=0.8)
            analysis['high_volume_nodes'] = high_vol_nodes
            
            # Find volume gaps (price levels with low volume)
            volume_gaps = volume_profile.get_volume_gaps(threshold=0.2)
            analysis['volume_gaps'] = volume_gaps
            
            # Calculate where current price is in relation to volume profile
            if not df.empty:
                current_price = df['close'].iloc[-1]
                profile_position = volume_profile.get_price_position(current_price)
                analysis['profile_position'] = profile_position
            
            # Add value area analysis
            value_area = volume_profile.get_value_area(value_area_pct=0.7)
            analysis['value_area'] = value_area
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in volume-at-price analysis: {str(e)}", exc_info=True)
            return {'error': str(e)}


# Factory function to create preconfigured detector
def create_volume_pattern_detector(
        market_type: str = 'crypto',
        timeframe: str = '1h',
        volatility: str = 'medium') -> VolumePatternDetector:
    """
    Create a volume pattern detector with settings optimized for the market conditions.
    
    Args:
        market_type: Type of market ('crypto', 'forex', 'stock')
        timeframe: Timeframe of the data ('1m', '5m', '15m', '1h', '4h', '1d')
        volatility: Expected market volatility ('low', 'medium', 'high')
    
    Returns:
        Configured VolumePatternDetector instance
    """
    # Base sensitivity for different market types
    sensitivity_map = {
        'crypto': 0.7,
        'forex': 0.5,
        'stock': 0.6
    }
    
    # Adjust for timeframe (higher timeframes need lower sensitivity)
    timeframe_factor = {
        '1m': 1.3,
        '5m': 1.2,
        '15m': 1.1,
        '1h': 1.0,
        '4h': 0.9,
        '1d': 0.8
    }
    
    # Adjust for volatility
    volatility_factor = {
        'low': 1.2,
        'medium': 1.0,
        'high': 0.8
    }
    
    # Calculate optimal settings
    base_sensitivity = sensitivity_map.get(market_type, 0.6)
    tf_factor = timeframe_factor.get(timeframe, 1.0)
    vol_factor = volatility_factor.get(volatility, 1.0)
    
    final_sensitivity = base_sensitivity * tf_factor * vol_factor
    
    # Adjust lookback periods based on timeframe
    lookback_map = {
        '1m': 200,
        '5m': 150,
        '15m': 120,
        '1h': 100,
        '4h': 80,
        '1d': 60
    }
    lookback = lookback_map.get(timeframe, 100)
    
    # Create and return the detector
    detector = VolumePatternDetector(
        sensitivity=final_sensitivity,
        lookback_periods=lookback,
        min_confidence=0.65 if market_type == 'crypto' else 0.6
    )
    
    logger.info(f"Created volume pattern detector for {market_type} market on {timeframe} "
               f"timeframe with sensitivity {final_sensitivity:.2f}")
    
    return detector
