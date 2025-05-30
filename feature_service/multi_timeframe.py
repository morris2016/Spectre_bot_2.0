#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Multi-Timeframe Analysis Module

This module provides advanced multi-timeframe analysis capabilities, allowing
pattern recognition across different timeframes for higher-confidence signals
and deeper market structure insights. It handles the complexities of data
synchronization, alignment, and correlation between timeframes.

Key features:
- Synchronized data handling across multiple timeframes
- Hierarchical pattern recognition and confirmation
- Timeframe correlation analysis
- Divergence detection across timeframes
- Multi-timeframe market structure identification
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from enum import Enum

from common.logger import get_logger
from common.utils import timeframe_to_seconds, validate_timeframe, get_higher_timeframes
from common.exceptions import DataAlignmentError, InvalidTimeframeError
from data_storage.time_series import TimeSeriesStorage
from feature_service.feature_extraction import FeatureExtractor

class TimeFrame(Enum):
    """
    Enum for standard timeframes.
    This is an alias for the Timeframe enum in common.constants for backward compatibility.
    """
    M1 = "1m"     # 1 minute
    M5 = "5m"     # 5 minutes
    M15 = "15m"   # 15 minutes
    M30 = "30m"   # 30 minutes
    H1 = "1h"     # 1 hour
    H4 = "4h"     # 4 hours
    D1 = "1d"     # 1 day
    W1 = "1w"     # 1 week
    MN1 = "1M"    # 1 month

class MultiTimeframeAnalyzer:
    """
    Advanced multi-timeframe analysis system for pattern recognition, 
    confirmation, and market structure identification across timeframes.
    
    This class provides tools to analyze market data across multiple timeframes
    to identify patterns with higher confidence and gain deeper insights into
    market structure and directional bias.
    """
    
    def __init__(
        self, 
        time_series_store: TimeSeriesStorage,
        feature_extractor: FeatureExtractor,
        primary_timeframe: str = '1m',
        additional_timeframes: List[str] = None,
        sync_method: str = 'forward_fill',
        max_workers: int = 4
    ):
        """
        Initialize the multi-timeframe analyzer with specified timeframes.
        
        Args:
            time_series_store: Database interface for retrieving time series data
            feature_extractor: Component for extracting features from market data
            primary_timeframe: Base timeframe for analysis
            additional_timeframes: List of additional timeframes to analyze
            sync_method: Method for synchronizing data across timeframes
                         ('forward_fill', 'backward_fill', 'nearest')
            max_workers: Maximum number of threads for parallel processing
        """
        self.logger = get_logger('multi_timeframe')
        self.time_series_store = time_series_store
        self.feature_extractor = feature_extractor
        
        # Validate and set timeframes
        self.primary_timeframe = validate_timeframe(primary_timeframe)
        self.primary_tf_seconds = timeframe_to_seconds(self.primary_timeframe)
        
        if additional_timeframes is None:
            # Automatically select higher timeframes if none specified
            self.additional_timeframes = get_higher_timeframes(primary_timeframe, count=3)
        else:
            self.additional_timeframes = [validate_timeframe(tf) for tf in additional_timeframes]
        
        # Include primary in all timeframes for convenience
        self.all_timeframes = [self.primary_timeframe] + self.additional_timeframes
        self.logger.info(f"Initialized with timeframes: {self.all_timeframes}")
        
        # Map each timeframe to its seconds equivalent for calculations
        self.timeframe_seconds = {
            tf: timeframe_to_seconds(tf) for tf in self.all_timeframes
        }
        
        # Sort timeframes from shortest to longest
        self.sorted_timeframes = sorted(
            self.all_timeframes, 
            key=lambda x: self.timeframe_seconds[x]
        )
        
        self.sync_method = sync_method
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Cache for optimization
        self._cached_data = {}
        self._cache_expiry = {}
        self.cache_duration = 300  # seconds
        
    async def get_multi_timeframe_data(
        self, 
        symbol: str, 
        start_time: datetime, 
        end_time: datetime = None,
        include_current_candle: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Retrieve and align data for all configured timeframes.
        
        Args:
            symbol: The trading symbol to retrieve data for
            start_time: Start time for data retrieval
            end_time: End time for data retrieval (defaults to now)
            include_current_candle: Whether to include the current forming candle
            
        Returns:
            Dictionary mapping timeframe to its DataFrame of OHLCV data
        """
        if end_time is None:
            end_time = datetime.utcnow()
            
        # Extend the start time to ensure we have enough data for larger timeframes
        max_tf_seconds = max(self.timeframe_seconds.values())
        extended_start = start_time - timedelta(seconds=max_tf_seconds * 10)
        
        # Fetch data for all timeframes concurrently
        tasks = []
        loop = asyncio.get_event_loop()
        
        for tf in self.all_timeframes:
            cache_key = f"{symbol}_{tf}_{extended_start.timestamp()}_{end_time.timestamp()}"
            
            # Check if we have cached data that's still valid
            if (cache_key in self._cached_data and 
                cache_key in self._cache_expiry and 
                datetime.utcnow().timestamp() < self._cache_expiry[cache_key]):
                
                self.logger.debug(f"Using cached data for {symbol} {tf}")
                continue
                
            # Create task for fetching this timeframe's data
            task = loop.run_in_executor(
                self.executor,
                self._get_timeframe_data,
                symbol, tf, extended_start, end_time, include_current_candle
            )
            tasks.append((tf, task))
        
        # Process results and build the data dictionary
        result = {}
        
        # First add any cached data
        for tf in self.all_timeframes:
            cache_key = f"{symbol}_{tf}_{extended_start.timestamp()}_{end_time.timestamp()}"
            if cache_key in self._cached_data:
                result[tf] = self._cached_data[cache_key]
        
        # Then add newly fetched data
        for tf, task in tasks:
            try:
                df = await task
                
                if df is not None and not df.empty:
                    # Cache the result
                    cache_key = f"{symbol}_{tf}_{extended_start.timestamp()}_{end_time.timestamp()}"
                    self._cached_data[cache_key] = df
                    self._cache_expiry[cache_key] = datetime.utcnow().timestamp() + self.cache_duration
                    
                    result[tf] = df
                else:
                    self.logger.warning(f"No data returned for {symbol} on {tf} timeframe")
            except Exception as e:
                self.logger.error(f"Error fetching {tf} data for {symbol}: {str(e)}")
                
        if not result:
            raise DataAlignmentError(f"Failed to retrieve data for all timeframes for {symbol}")
            
        return result
    
    def _get_timeframe_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_time: datetime, 
        end_time: datetime,
        include_current: bool
    ) -> pd.DataFrame:
        """
        Retrieve data for a specific timeframe.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe to retrieve
            start_time: Start time
            end_time: End time
            include_current: Whether to include current candle
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            df = self.time_series_store.get_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                include_current_candle=include_current
            )
            
            if df is not None and not df.empty:
                self.logger.debug(f"Retrieved {len(df)} rows for {symbol} {timeframe}")
            return df
        except Exception as e:
            self.logger.error(f"Error in _get_timeframe_data for {symbol} {timeframe}: {str(e)}")
            raise
    
    async def analyze_multi_timeframe(
        self,
        symbol: str,
        lookback_periods: int = 100,
        feature_sets: List[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive multi-timeframe analysis.
        
        Args:
            symbol: The trading symbol to analyze
            lookback_periods: Number of periods to look back on primary timeframe
            feature_sets: List of feature sets to calculate (None = all)
            
        Returns:
            Dictionary with comprehensive multi-timeframe analysis results
        """
        try:
            # Calculate how far back we need to go
            start_time = datetime.utcnow() - timedelta(
                seconds=self.primary_tf_seconds * lookback_periods * 2
            )
            
            # Get aligned data for all timeframes
            multi_tf_data = await self.get_multi_timeframe_data(
                symbol=symbol,
                start_time=start_time
            )
            
            if not multi_tf_data or len(multi_tf_data) != len(self.all_timeframes):
                self.logger.warning(f"Incomplete data for multi-timeframe analysis: {len(multi_tf_data)}/{len(self.all_timeframes)} timeframes available")
            
            # Process each timeframe to extract features
            tf_features = {}
            tasks = []
            loop = asyncio.get_event_loop()
            
            for tf, data in multi_tf_data.items():
                if data is None or data.empty:
                    self.logger.warning(f"No data for {symbol} on {tf} timeframe")
                    continue
                    
                # Trim to the requested number of candles
                if len(data) > lookback_periods:
                    data = data.iloc[-lookback_periods:]
                
                # Create task for feature extraction
                task = loop.run_in_executor(
                    self.executor,
                    self._extract_timeframe_features,
                    data, tf, feature_sets
                )
                tasks.append((tf, task))
            
            # Wait for all feature extraction to complete
            for tf, task in tasks:
                try:
                    features = await task
                    tf_features[tf] = features
                except Exception as e:
                    self.logger.error(f"Error extracting features for {symbol} {tf}: {str(e)}")
            
            # Perform cross-timeframe analysis
            result = self._cross_timeframe_analysis(symbol, tf_features, multi_tf_data)
            
            # Add multi-timeframe trend analysis
            result['trend_analysis'] = self._analyze_trend_across_timeframes(tf_features)
            
            # Detect divergences across timeframes
            result['divergences'] = self._detect_cross_timeframe_divergences(tf_features, multi_tf_data)
            
            # Identify support/resistance levels across timeframes
            result['support_resistance'] = self._identify_multi_tf_support_resistance(tf_features)
            
            # Calculate timeframe alignment score
            result['alignment_score'] = self._calculate_timeframe_alignment(tf_features)
            
            # Add trading bias based on multi-timeframe analysis
            result['trading_bias'] = self._determine_trading_bias(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in analyze_multi_timeframe for {symbol}: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _extract_timeframe_features(
        self, 
        data: pd.DataFrame, 
        timeframe: str,
        feature_sets: List[str]
    ) -> Dict[str, Any]:
        """
        Extract features for a specific timeframe.
        
        Args:
            data: DataFrame with OHLCV data
            timeframe: The timeframe being analyzed
            feature_sets: List of feature sets to calculate
            
        Returns:
            Dictionary containing extracted features
        """
        try:
            # Clone data to avoid modification warnings
            df = data.copy()
            
            # Use the feature extractor to calculate features
            features = self.feature_extractor.extract_features(
                df, feature_sets=feature_sets
            )
            
            # Add timeframe-specific metadata
            features['timeframe'] = timeframe
            features['candle_seconds'] = self.timeframe_seconds[timeframe]
            features['data_points'] = len(df)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features for {timeframe}: {str(e)}")
            raise
    
    def _cross_timeframe_analysis(
        self,
        symbol: str,
        tf_features: Dict[str, Dict[str, Any]],
        tf_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Analyze relationships and patterns across multiple timeframes.
        
        Args:
            symbol: The trading symbol
            tf_features: Features calculated for each timeframe
            tf_data: Raw data for each timeframe
            
        Returns:
            Cross-timeframe analysis results
        """
        result = {
            'symbol': symbol,
            'analysis_time': datetime.utcnow(),
            'timeframes_analyzed': list(tf_features.keys()),
            'higher_tf_bias': {},
            'confirmation_signals': {},
            'pattern_confluence': [],
            'structural_levels': [],
        }
        
        try:
            # Determine directional bias from higher timeframes
            for tf in self.sorted_timeframes[1:]:  # Skip the lowest timeframe
                if tf in tf_features:
                    features = tf_features[tf]
                    
                    # Extract trend information
                    if 'trend' in features:
                        trend_info = features['trend']
                        result['higher_tf_bias'][tf] = {
                            'trend': trend_info.get('direction', 'neutral'),
                            'strength': trend_info.get('strength', 0),
                            'duration': trend_info.get('duration', 0),
                            'momentum': trend_info.get('momentum', 'neutral')
                        }
            
            # Look for confirmation across timeframes
            for pattern_type in ['support_resistance', 'candlestick_patterns', 'chart_patterns']:
                confirmed_patterns = self._find_confirmed_patterns(tf_features, pattern_type)
                if confirmed_patterns:
                    result['confirmation_signals'][pattern_type] = confirmed_patterns
            
            # Find pattern confluence points (where multiple patterns align)
            result['pattern_confluence'] = self._identify_pattern_confluence(tf_features, tf_data)
            
            # Identify important structural levels visible across multiple timeframes
            result['structural_levels'] = self._find_structural_levels(tf_features)
            
            # Calculate overall multi-timeframe bias score (-100 to +100)
            result['multi_tf_bias_score'] = self._calculate_bias_score(tf_features, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in cross-timeframe analysis for {symbol}: {str(e)}")
            raise
    
    def _find_confirmed_patterns(
        self,
        tf_features: Dict[str, Dict[str, Any]],
        pattern_type: str
    ) -> List[Dict[str, Any]]:
        """
        Find patterns that are confirmed across multiple timeframes.
        
        Args:
            tf_features: Features for each timeframe
            pattern_type: Type of pattern to check for confirmation
            
        Returns:
            List of confirmed patterns with their details
        """
        confirmed = []
        
        # Skip if we don't have at least 2 timeframes to compare
        if len(tf_features) < 2:
            return confirmed
        
        # For each lower timeframe pattern, look for confirmation in higher timeframes
        for i, tf_lower in enumerate(self.sorted_timeframes[:-1]):
            if tf_lower not in tf_features:
                continue
                
            lower_features = tf_features[tf_lower]
            
            # Skip if pattern type not in features
            if pattern_type not in lower_features:
                continue
                
            lower_patterns = lower_features[pattern_type]
            
            # Check each higher timeframe for confirmation
            for tf_higher in self.sorted_timeframes[i+1:]:
                if tf_higher not in tf_features:
                    continue
                    
                higher_features = tf_features[tf_higher]
                
                # Skip if pattern type not in higher timeframe features
                if pattern_type not in higher_features:
                    continue
                    
                higher_patterns = higher_features[pattern_type]
                
                # Compare patterns and find confirmations
                for lower_pattern in lower_patterns:
                    for higher_pattern in higher_patterns:
                        if self._patterns_confirm_each_other(lower_pattern, higher_pattern):
                            confirmed.append({
                                'pattern': lower_pattern['name'],
                                'lower_timeframe': tf_lower,
                                'higher_timeframe': tf_higher,
                                'direction': lower_pattern.get('direction', 'unknown'),
                                'strength': (lower_pattern.get('strength', 0) + 
                                            higher_pattern.get('strength', 0)) / 2,
                                'price_level': lower_pattern.get('price_level'),
                                'confirmation_quality': self._calculate_confirmation_quality(
                                    lower_pattern, higher_pattern
                                )
                            })
        
        return confirmed
    
    def _patterns_confirm_each_other(
        self,
        pattern1: Dict[str, Any],
        pattern2: Dict[str, Any]
    ) -> bool:
        """
        Determine if two patterns confirm each other.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            True if patterns confirm each other, False otherwise
        """
        # Check if patterns have the same name
        same_name = pattern1.get('name') == pattern2.get('name')
        
        # Check if patterns have the same direction
        same_direction = pattern1.get('direction') == pattern2.get('direction')
        
        # Check if price levels are close
        price_level1 = pattern1.get('price_level')
        price_level2 = pattern2.get('price_level')
        
        price_levels_close = False
        if price_level1 is not None and price_level2 is not None:
            # Allow for some deviation in price levels
            deviation = min(price_level1, price_level2) * 0.01  # 1% tolerance
            price_levels_close = abs(price_level1 - price_level2) <= deviation
        
        # If it's a support/resistance level, check for proximity
        if 'type' in pattern1 and pattern1['type'] in ['support', 'resistance']:
            # Support/resistance patterns confirm if they're of the same type
            # and their price levels are close
            return (pattern1['type'] == pattern2.get('type') and 
                   price_levels_close)
        
        # For other patterns, they confirm if they have same name, direction
        # and close price levels (if applicable)
        return same_name and same_direction and (price_levels_close if price_level1 and price_level2 else True)
    
    def _calculate_confirmation_quality(
        self,
        pattern1: Dict[str, Any],
        pattern2: Dict[str, Any]
    ) -> float:
        """
        Calculate a quality score for the confirmation between two patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        # Start with base score
        score = 0.5
        
        # Adjust based on pattern strengths
        p1_strength = pattern1.get('strength', 0)
        p2_strength = pattern2.get('strength', 0)
        avg_strength = (p1_strength + p2_strength) / 2
        
        score += avg_strength * 0.3  # Up to 0.3 bonus for strong patterns
        
        # Adjust based on price level proximity if available
        p1_level = pattern1.get('price_level')
        p2_level = pattern2.get('price_level')
        
        if p1_level is not None and p2_level is not None:
            # Calculate normalized distance between price levels
            normalized_dist = abs(p1_level - p2_level) / ((p1_level + p2_level) / 2)
            
            # Closer price levels increase quality
            if normalized_dist <= 0.005:  # Within 0.5%
                score += 0.2
            elif normalized_dist <= 0.01:  # Within 1%
                score += 0.1
            elif normalized_dist <= 0.02:  # Within 2%
                score += 0.05
        
        # Cap the score at 1.0
        return min(1.0, score)
    
    def _identify_pattern_confluence(
        self,
        tf_features: Dict[str, Dict[str, Any]],
        tf_data: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """
        Identify points where multiple patterns from different timeframes align.
        
        Args:
            tf_features: Features for each timeframe
            tf_data: Raw data for each timeframe
            
        Returns:
            List of confluence points with details
        """
        confluence_points = []
        
        # Skip if we don't have enough timeframes
        if len(tf_features) < 2:
            return confluence_points
            
        # Collect all price-specific patterns
        all_price_patterns = []
        
        pattern_types = [
            'support_resistance', 
            'chart_patterns', 
            'candlestick_patterns',
            'harmonic_patterns'
        ]
        
        # Collect patterns from all timeframes
        for tf, features in tf_features.items():
            for pattern_type in pattern_types:
                if pattern_type in features:
                    for pattern in features[pattern_type]:
                        # Only include patterns with a specific price level
                        if 'price_level' in pattern:
                            # Add the pattern with its timeframe
                            pattern_copy = pattern.copy()
                            pattern_copy['timeframe'] = tf
                            all_price_patterns.append(pattern_copy)
        
        # Group patterns by price proximity
        grouped_patterns = self._group_by_price_proximity(all_price_patterns)
        
        # Create confluence points from groups with enough patterns
        for price_group in grouped_patterns:
            if len(price_group) >= 2:  # At least 2 patterns required for confluence
                # Calculate average price level
                avg_price = sum(p.get('price_level', 0) for p in price_group) / len(price_group)
                
                # Get unique timeframes in this confluence
                timeframes = list(set(p.get('timeframe') for p in price_group))
                
                # Only include if patterns come from at least 2 different timeframes
                if len(timeframes) >= 2:
                    # Calculate confluence strength based on number and quality of patterns
                    strength = min(1.0, 0.3 + (len(price_group) * 0.15) + 
                                 (len(timeframes) * 0.1))
                    
                    # Determine direction based on majority of patterns
                    directions = [p.get('direction', 'neutral') for p in price_group 
                                 if 'direction' in p]
                    
                    if directions:
                        bullish = directions.count('bullish')
                        bearish = directions.count('bearish')
                        
                        if bullish > bearish:
                            direction = 'bullish'
                        elif bearish > bullish:
                            direction = 'bearish'
                        else:
                            direction = 'neutral'
                    else:
                        direction = 'neutral'
                    
                    # Create confluence point entry
                    confluence = {
                        'price_level': avg_price,
                        'timeframes': timeframes,
                        'pattern_count': len(price_group),
                        'patterns': [p.get('name', 'Unknown') for p in price_group],
                        'strength': strength,
                        'direction': direction
                    }
                    
                    confluence_points.append(confluence)
        
        # Sort by strength (descending)
        confluence_points.sort(key=lambda x: x['strength'], reverse=True)
        
        return confluence_points
    
    def _group_by_price_proximity(
        self,
        patterns: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """
        Group patterns that have price levels close to each other.
        
        Args:
            patterns: List of patterns with price levels
            
        Returns:
            List of pattern groups, where each group contains related patterns
        """
        if not patterns:
            return []
            
        # Sort patterns by price level
        sorted_patterns = sorted(patterns, key=lambda x: x.get('price_level', 0))
        
        groups = []
        current_group = [sorted_patterns[0]]
        
        for i in range(1, len(sorted_patterns)):
            current = sorted_patterns[i]
            previous = current_group[-1]
            
            current_price = current.get('price_level', 0)
            previous_price = previous.get('price_level', 0)
            
            # Define proximity as 0.5% of the price level
            proximity = previous_price * 0.005
            
            if abs(current_price - previous_price) <= proximity:
                # Price is close enough, add to current group
                current_group.append(current)
            else:
                # Price too different, start new group
                groups.append(current_group)
                current_group = [current]
                
        # Add the last group
        if current_group:
            groups.append(current_group)
            
        return groups
    
    def _find_structural_levels(
        self,
        tf_features: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify important structural price levels visible across multiple timeframes.
        
        Args:
            tf_features: Features for each timeframe
            
        Returns:
            List of structural levels with details
        """
        # Collect support & resistance levels from all timeframes
        structural_levels = []
        
        for tf, features in tf_features.items():
            if 'support_resistance' in features:
                for level in features['support_resistance']:
                    # Add timeframe information
                    level_copy = level.copy()
                    level_copy['timeframe'] = tf
                    structural_levels.append(level_copy)
        
        # Group levels by price proximity
        grouped_levels = self._group_by_price_proximity(structural_levels)
        
        # Create multi-timeframe structural levels
        multi_tf_levels = []
        
        for level_group in grouped_levels:
            if not level_group:
                continue
                
            # Only include groups with levels from multiple timeframes
            timeframes = list(set(level.get('timeframe') for level in level_group))
            
            if len(timeframes) >= 2:  # At least 2 timeframes required
                # Calculate average price
                avg_price = sum(level.get('price_level', 0) for level in level_group) / len(level_group)
                
                # Determine if this is support or resistance based on majority
                types = [level.get('type') for level in level_group if 'type' in level]
                support_count = types.count('support')
                resistance_count = types.count('resistance')
                
                level_type = 'support' if support_count >= resistance_count else 'resistance'
                
                # Calculate strength based on occurrence across timeframes
                # Higher timeframes contribute more to strength
                strength = 0
                for level in level_group:
                    tf = level.get('timeframe')
                    tf_seconds = self.timeframe_seconds.get(tf, 0)
                    
                    # Higher timeframes get more weight
                    tf_weight = np.log10(tf_seconds) / 4 if tf_seconds else 0.1
                    strength += tf_weight * level.get('strength', 0.5)
                
                # Normalize strength to 0-1 range
                strength = min(1.0, strength / len(timeframes))
                
                # Create the structural level
                multi_tf_level = {
                    'price_level': avg_price,
                    'type': level_type,
                    'timeframes': timeframes,
                    'strength': strength,
                    'touch_count': sum(level.get('touch_count', 0) for level in level_group),
                    'most_recent_test': max((level.get('most_recent_test', 
                                             datetime(1970, 1, 1)) for level in level_group), 
                                           default=None)
                }
                
                multi_tf_levels.append(multi_tf_level)
        
        # Sort by strength (descending)
        multi_tf_levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return multi_tf_levels
    
    def _analyze_trend_across_timeframes(
        self,
        tf_features: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze trend characteristics across multiple timeframes.
        
        Args:
            tf_features: Features for each timeframe
            
        Returns:
            Dictionary with trend analysis across timeframes
        """
        trend_analysis = {
            'trend_by_timeframe': {},
            'trend_bias': 'neutral',
            'trend_strength_overall': 0,
            'trend_alignment': False,
            'trend_transition': False,
            'dominant_timeframe': None
        }
        
        # Collect trend data from each timeframe
        trend_directions = {}
        trend_strengths = {}
        
        for tf, features in tf_features.items():
            if 'trend' in features:
                trend_info = features['trend']
                direction = trend_info.get('direction', 'neutral')
                strength = trend_info.get('strength', 0)
                
                trend_directions[tf] = direction
                trend_strengths[tf] = strength
                
                trend_analysis['trend_by_timeframe'][tf] = {
                    'direction': direction,
                    'strength': strength,
                    'duration': trend_info.get('duration', 0),
                    'momentum': trend_info.get('momentum', 'neutral')
                }
        
        # Check for trend alignment across timeframes
        directions = list(trend_directions.values())
        if directions and all(d == directions[0] for d in directions) and directions[0] != 'neutral':
            trend_analysis['trend_alignment'] = True
            trend_analysis['trend_bias'] = directions[0]
        
        # Check for trend transitions (lower timeframes changing before higher ones)
        if len(self.sorted_timeframes) >= 2:
            for i in range(len(self.sorted_timeframes) - 1):
                tf_lower = self.sorted_timeframes[i]
                tf_higher = self.sorted_timeframes[i+1]
                
                if (tf_lower in trend_directions and tf_higher in trend_directions and
                    trend_directions[tf_lower] != trend_directions[tf_higher] and
                    trend_directions[tf_lower] != 'neutral'):
                    
                    trend_analysis['trend_transition'] = True
                    trend_analysis['transition_details'] = {
                        'lower_timeframe': tf_lower,
                        'higher_timeframe': tf_higher,
                        'lower_direction': trend_directions[tf_lower],
                        'higher_direction': trend_directions[tf_higher]
                    }
                    break
        
        # Calculate overall trend strength, weighted by timeframe
        total_weight = 0
        weighted_strength = 0
        
        for tf, strength in trend_strengths.items():
            # Higher timeframes get more weight
            tf_seconds = self.timeframe_seconds.get(tf, 0)
            weight = np.log10(tf_seconds) if tf_seconds else 1
            
            total_weight += weight
            weighted_strength += strength * weight
        
        if total_weight > 0:
            trend_analysis['trend_strength_overall'] = weighted_strength / total_weight
        
        # Determine dominant timeframe (the one with highest strength)
        if trend_strengths:
            trend_analysis['dominant_timeframe'] = max(
                trend_strengths.keys(), key=lambda tf: trend_strengths[tf]
            )
        
        # Determine overall bias based on weighted average of directions
        if trend_directions:
            bullish_weight = 0
            bearish_weight = 0
            
            for tf, direction in trend_directions.items():
                # Weight by timeframe and strength
                tf_seconds = self.timeframe_seconds.get(tf, 0)
                strength = trend_strengths.get(tf, 0.5)
                weight = np.log10(tf_seconds) * strength if tf_seconds else strength
                
                if direction == 'bullish':
                    bullish_weight += weight
                elif direction == 'bearish':
                    bearish_weight += weight
            
            if bullish_weight > bearish_weight * 1.2:  # 20% threshold for clear bias
                trend_analysis['trend_bias'] = 'bullish'
            elif bearish_weight > bullish_weight * 1.2:
                trend_analysis['trend_bias'] = 'bearish'
            else:
                trend_analysis['trend_bias'] = 'neutral'
        
        return trend_analysis
    
    def _detect_cross_timeframe_divergences(
        self,
        tf_features: Dict[str, Dict[str, Any]],
        tf_data: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """
        Detect divergences across timeframes, such as price making new high while
        momentum on higher timeframe is decreasing.
        
        Args:
            tf_features: Features for each timeframe
            tf_data: Raw data for each timeframe
            
        Returns:
            List of divergences with details
        """
        divergences = []
        
        if len(tf_features) < 2:
            return divergences
            
        # Check each pair of adjacent timeframes
        for i in range(len(self.sorted_timeframes) - 1):
            tf_lower = self.sorted_timeframes[i]
            tf_higher = self.sorted_timeframes[i+1]
            
            if tf_lower not in tf_features or tf_higher not in tf_features:
                continue
                
            lower_features = tf_features[tf_lower]
            higher_features = tf_features[tf_higher]
            
            # Skip if required data not available
            if ('momentum' not in lower_features or 
                'momentum' not in higher_features or 
                tf_lower not in tf_data or 
                tf_higher not in tf_data):
                continue
            
            lower_momentum = lower_features['momentum']
            higher_momentum = higher_features['momentum']
            
            lower_data = tf_data[tf_lower].iloc[-10:]  # Last 10 candles
            higher_data = tf_data[tf_higher].iloc[-5:]  # Last 5 candles
            
            # Check for regular bullish divergence (price making lower low, momentum making higher low)
            if (lower_data['close'].iloc[-1] < lower_data['close'].min() and 
                lower_momentum.get('value', 0) > lower_momentum.get('previous_value', 0) and
                higher_momentum.get('direction', 'neutral') != 'bearish'):
                
                divergences.append({
                    'type': 'regular_bullish',
                    'lower_timeframe': tf_lower,
                    'higher_timeframe': tf_higher,
                    'strength': 0.7,
                    'description': f"Price making lower low on {tf_lower} while momentum increasing, with {tf_higher} momentum not bearish",
                    'timestamp': datetime.utcnow()
                })
            
            # Check for regular bearish divergence (price making higher high, momentum making lower high)
            if (lower_data['close'].iloc[-1] > lower_data['close'].max() and 
                lower_momentum.get('value', 0) < lower_momentum.get('previous_value', 0) and
                higher_momentum.get('direction', 'neutral') != 'bullish'):
                
                divergences.append({
                    'type': 'regular_bearish',
                    'lower_timeframe': tf_lower,
                    'higher_timeframe': tf_higher,
                    'strength': 0.7,
                    'description': f"Price making higher high on {tf_lower} while momentum decreasing, with {tf_higher} momentum not bullish",
                    'timestamp': datetime.utcnow()
                })
            
            # Check for timeframe momentum divergence
            if (lower_momentum.get('direction', 'neutral') != 'neutral' and
                higher_momentum.get('direction', 'neutral') != 'neutral' and
                lower_momentum.get('direction') != higher_momentum.get('direction')):
                
                divergences.append({
                    'type': 'timeframe_momentum',
                    'lower_timeframe': tf_lower,
                    'higher_timeframe': tf_higher,
                    'lower_direction': lower_momentum.get('direction'),
                    'higher_direction': higher_momentum.get('direction'),
                    'strength': 0.8,
                    'description': f"Momentum direction differs between {tf_lower} ({lower_momentum.get('direction')}) and {tf_higher} ({higher_momentum.get('direction')})",
                    'timestamp': datetime.utcnow()
                })
                
        return divergences
    
    def _identify_multi_tf_support_resistance(
        self,
        tf_features: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify support and resistance levels confirmed across multiple timeframes.
        
        Args:
            tf_features: Features for each timeframe
            
        Returns:
            Dictionary with multi-timeframe support and resistance levels
        """
        # Reuse structural levels function which already does this
        all_levels = self._find_structural_levels(tf_features)
        
        # Separate into support and resistance
        supports = [level for level in all_levels if level.get('type') == 'support']
        resistances = [level for level in all_levels if level.get('type') == 'resistance']
        
        # Sort by strength
        supports.sort(key=lambda x: x.get('strength', 0), reverse=True)
        resistances.sort(key=lambda x: x.get('strength', 0), reverse=True)
        
        return {
            'support': supports,
            'resistance': resistances
        }
    
    def _calculate_timeframe_alignment(
        self,
        tf_features: Dict[str, Dict[str, Any]]
    ) -> float:
        """
        Calculate a score indicating how well aligned the signals are across timeframes.
        
        Args:
            tf_features: Features for each timeframe
            
        Returns:
            Alignment score (0.0 to 1.0)
        """
        if len(tf_features) < 2:
            return 0.0
            
        # Collect trend and momentum directions
        trend_directions = {}
        momentum_directions = {}
        
        for tf, features in tf_features.items():
            if 'trend' in features:
                trend_directions[tf] = features['trend'].get('direction', 'neutral')
                
            if 'momentum' in features:
                momentum_directions[tf] = features['momentum'].get('direction', 'neutral')
        
        # Calculate trend alignment score
        trend_alignment = 0.0
        if trend_directions:
            # Count each direction
            bullish = sum(1 for d in trend_directions.values() if d == 'bullish')
            bearish = sum(1 for d in trend_directions.values() if d == 'bearish')
            neutral = sum(1 for d in trend_directions.values() if d == 'neutral')
            
            total = len(trend_directions)
            
            # Calculate alignment as the proportion of the dominant direction
            dominant = max(bullish, bearish, neutral)
            trend_alignment = dominant / total if total > 0 else 0.0
        
        # Calculate momentum alignment score
        momentum_alignment = 0.0
        if momentum_directions:
            # Count each direction
            bullish = sum(1 for d in momentum_directions.values() if d == 'bullish')
            bearish = sum(1 for d in momentum_directions.values() if d == 'bearish')
            neutral = sum(1 for d in momentum_directions.values() if d == 'neutral')
            
            total = len(momentum_directions)
            
            # Calculate alignment as the proportion of the dominant direction
            dominant = max(bullish, bearish, neutral)
            momentum_alignment = dominant / total if total > 0 else 0.0
        
        # Combine scores (trend weighted more heavily)
        combined_alignment = (trend_alignment * 0.7) + (momentum_alignment * 0.3)
        
        return combined_alignment
    
    def _calculate_bias_score(
        self,
        tf_features: Dict[str, Dict[str, Any]],
        analysis_result: Dict[str, Any]
    ) -> int:
        """
        Calculate an overall directional bias score from -100 (bearish) to +100 (bullish).
        
        Args:
            tf_features: Features for each timeframe
            analysis_result: Previous multi-timeframe analysis results
            
        Returns:
            Bias score (-100 to +100)
        """
        score = 0
        
        # Factor 1: Trend across timeframes (weighted by timeframe)
        for tf, features in tf_features.items():
            if 'trend' in features:
                trend = features['trend']
                direction = trend.get('direction', 'neutral')
                strength = trend.get('strength', 0.5)
                
                # Weight by timeframe (higher timeframes matter more)
                tf_seconds = self.timeframe_seconds.get(tf, 60)
                weight = min(5, np.log10(tf_seconds)) / 5  # Normalize to 0-1
                
                # Add to score
                if direction == 'bullish':
                    score += 25 * strength * weight
                elif direction == 'bearish':
                    score -= 25 * strength * weight
        
        # Factor 2: Momentum across timeframes
        for tf, features in tf_features.items():
            if 'momentum' in features:
                momentum = features['momentum']
                direction = momentum.get('direction', 'neutral')
                value = momentum.get('value', 0)
                
                # Weight by timeframe
                tf_seconds = self.timeframe_seconds.get(tf, 60)
                weight = min(3, np.log10(tf_seconds)) / 5  # Less weight than trend
                
                # Add to score
                if direction == 'bullish':
                    score += 15 * abs(value) * weight
                elif direction == 'bearish':
                    score -= 15 * abs(value) * weight
        
        # Factor 3: Pattern confluence
        pattern_confluence = analysis_result.get('pattern_confluence', [])
        for confluence in pattern_confluence:
            direction = confluence.get('direction', 'neutral')
            strength = confluence.get('strength', 0.5)
            
            if direction == 'bullish':
                score += 10 * strength
            elif direction == 'bearish':
                score -= 10 * strength
        
        # Factor 4: Structural levels
        current_price = self._get_current_price(tf_features)
        if current_price:
            sr_levels = analysis_result.get('structural_levels', [])
            
            for level in sr_levels:
                level_price = level.get('price_level', 0)
                level_type = level.get('type', '')
                strength = level.get('strength', 0.5)
                
                # Distance from current price (normalized)
                distance = abs(current_price - level_price) / current_price
                
                # Closer levels have more impact
                proximity_factor = max(0, 1 - (distance * 100))
                
                # Support below price is bullish, resistance above is bearish
                if level_type == 'support' and level_price < current_price:
                    score += 5 * strength * proximity_factor
                elif level_type == 'resistance' and level_price > current_price:
                    score -= 5 * strength * proximity_factor
        
        # Factor 5: Timeframe alignment
        alignment = analysis_result.get('alignment_score', 0)
        trend_bias = analysis_result.get('trend_analysis', {}).get('trend_bias', 'neutral')
        
        if trend_bias == 'bullish':
            score += 20 * alignment
        elif trend_bias == 'bearish':
            score -= 20 * alignment
        
        # Ensure score is within -100 to +100 range
        return max(-100, min(100, score))
    
    def _get_current_price(self, tf_features: Dict[str, Dict[str, Any]]) -> Optional[float]:
        """
        Extract current price from the shortest timeframe available.
        
        Args:
            tf_features: Features for each timeframe
            
        Returns:
            Current price or None if not available
        """
        # Try to get from shortest timeframe first
        for tf in self.sorted_timeframes:
            if tf in tf_features and 'current_price' in tf_features[tf]:
                return tf_features[tf]['current_price']
        
        return None
    
    def _determine_trading_bias(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine overall trading bias based on multi-timeframe analysis.
        
        Args:
            analysis_result: Multi-timeframe analysis results
            
        Returns:
            Trading bias information
        """
        bias_score = analysis_result.get('multi_tf_bias_score', 0)
        
        # Determine direction based on score
        if bias_score >= 30:
            direction = 'bullish'
        elif bias_score <= -30:
            direction = 'bearish'
        else:
            direction = 'neutral'
        
        # Determine strength based on absolute value of score
        abs_score = abs(bias_score)
        if abs_score >= 70:
            strength = 'strong'
        elif abs_score >= 40:
            strength = 'moderate'
        else:
            strength = 'weak'
        
        # Determine confidence based on alignment and other factors
        alignment = analysis_result.get('alignment_score', 0)
        pattern_count = len(analysis_result.get('pattern_confluence', []))
        
        confidence = min(1.0, (alignment * 0.6) + (pattern_count * 0.1) + (abs_score / 200))
        
        return {
            'direction': direction,
            'strength': strength,
            'confidence': confidence,
            'score': bias_score,
            'recommended_exposure': self._calculate_recommended_exposure(
                direction, abs_score, confidence
            )
        }
    
    def _calculate_recommended_exposure(
        self,
        direction: str,
        score_magnitude: float,
        confidence: float
    ) -> float:
        """
        Calculate recommended capital exposure based on trading bias.
        
        Args:
            direction: Trading direction (bullish, bearish, neutral)
            score_magnitude: Absolute value of bias score
            confidence: Confidence in the trading bias
            
        Returns:
            Recommended exposure (0.0 to 1.0)
        """
        if direction == 'neutral':
            return 0.0
            
        # Base exposure on score magnitude
        base_exposure = score_magnitude / 100
        
        # Adjust by confidence
        adjusted_exposure = base_exposure * confidence
        
        # Cap at 0.8 for safety
        return min(0.8, adjusted_exposure)

