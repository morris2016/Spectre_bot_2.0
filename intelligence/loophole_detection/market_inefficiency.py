#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Market Inefficiency Detection Module

This module detects various forms of market inefficiencies and exploitable anomalies
across different timeframes and market conditions. The system identifies opportunities
where markets are improperly priced or where there are exploitable patterns in market
behavior.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import asyncio
import logging
from datetime import datetime, timedelta
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# Internal imports
from common.logger import get_logger
from common.utils import create_async_task, TimeFrame, calculate_z_score
from common.constants import (
    INEFFICIENCY_DETECTION_THRESHOLDS,
    MIN_INEFFICIENCY_SCORE,
    MAX_INEFFICIENCY_AGE,
    ANOMALY_CONTAMINATION_FACTOR
)
from data_storage.market_data import MarketDataRepository
from feature_service.features.technical import TechnicalFeatures
from feature_service.features.volume import VolumeFeatures
from feature_service.features.market_structure import MarketStructureFeatures


@dataclass
class MarketInefficiency:
    """Represents a detected market inefficiency."""
    
    type: str  # Type of inefficiency detected
    asset: str  # Asset where inefficiency was detected
    timeframe: TimeFrame  # Timeframe of detection
    timestamp: datetime  # When the inefficiency was detected
    score: float  # Confidence score (0-100)
    direction: int  # Expected price direction (1=up, -1=down, 0=neutral)
    expected_duration: timedelta  # Expected duration of the inefficiency
    details: Dict[str, Any]  # Additional details specific to this inefficiency
    source_data: pd.DataFrame  # Reference to data that triggered detection
    exploitation_strategy: str  # Recommended strategy to exploit this inefficiency
    risk_assessment: Dict[str, Any]  # Risk assessment for this opportunity
    
    @property
    def age(self) -> timedelta:
        """Calculate the age of this inefficiency detection."""
        return datetime.now() - self.timestamp
    
    @property
    def is_valid(self) -> bool:
        """Check if this inefficiency is still considered valid."""
        return (
            self.age < self.expected_duration and 
            self.age < MAX_INEFFICIENCY_AGE and
            self.score >= MIN_INEFFICIENCY_SCORE
        )
    
    @property
    def adjusted_score(self) -> float:
        """Calculate score adjusted for age as inefficiencies decay over time."""
        if self.expected_duration.total_seconds() == 0:
            return 0
        
        age_factor = max(0, 1 - (self.age.total_seconds() / self.expected_duration.total_seconds()))
        return self.score * age_factor
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation for storage/transmission."""
        return {
            "type": self.type,
            "asset": self.asset,
            "timeframe": self.timeframe.name,
            "timestamp": self.timestamp.isoformat(),
            "score": self.score,
            "direction": self.direction,
            "expected_duration_seconds": self.expected_duration.total_seconds(),
            "details": self.details,
            "exploitation_strategy": self.exploitation_strategy,
            "risk_assessment": self.risk_assessment,
            "adjusted_score": self.adjusted_score,
            "is_valid": self.is_valid
        }


class MarketInefficiencyDetector:
    """
    Advanced detector for market inefficiencies and exploitable anomalies.
    
    This class implements various algorithms for detecting inefficiencies across
    different market conditions, timeframes, and assets. It integrates data from
    multiple sources to identify high-probability trading opportunities resulting
    from market mispricing, trader behavior patterns, or other exploitable anomalies.
    """
    
    def __init__(
        self,
        market_data_repo: MarketDataRepository,
        technical_features: TechnicalFeatures,
        volume_features: VolumeFeatures,
        market_structure_features: MarketStructureFeatures
    ):
        """
        Initialize the market inefficiency detector with required dependencies.
        
        Args:
            market_data_repo: Repository for accessing market data
            technical_features: Technical indicator service
            volume_features: Volume analysis service
            market_structure_features: Market structure analysis service
        """
        self.logger = get_logger("MarketInefficiencyDetector")
        self.market_data = market_data_repo
        self.technical = technical_features
        self.volume = volume_features
        self.market_structure = market_structure_features
        
        # Store detected inefficiencies for tracking and analysis
        self.active_inefficiencies: List[MarketInefficiency] = []
        
        # Analysis models for different types of anomaly detection
        self._isolation_forest = None
        self._dbscan = None
        
        # Lock for thread safety during updates
        self._lock = asyncio.Lock()
        
        # Metrics for tracking detector performance
        self.metrics = {
            "total_detections": 0,
            "true_positives": 0,
            "false_positives": 0,
            "detection_accuracy": 0.0,
            "avg_profit_factor": 0.0,
            "detection_by_type": {},
            "detection_by_asset": {}
        }
        
        self.logger.info("Market Inefficiency Detector initialized")
    
    async def initialize_models(self, historical_data: Dict[str, pd.DataFrame]):
        """
        Initialize anomaly detection models with historical data.
        
        Args:
            historical_data: Dictionary of historical data frames by asset
        """
        self.logger.info("Initializing anomaly detection models...")
        
        # Prepare concatenated features for training
        all_features = []
        
        for asset, data in historical_data.items():
            if len(data) < 100:
                self.logger.warning(f"Insufficient data for {asset}, skipping in model training")
                continue
                
            # Extract features for anomaly detection
            features = self._extract_anomaly_features(data)
            if features is not None:
                all_features.append(features)
        
        if not all_features:
            self.logger.error("No valid feature data available for model training")
            return
            
        # Combine all features for training
        combined_features = pd.concat(all_features, axis=0)
        combined_features = combined_features.dropna()
        
        if len(combined_features) < 100:
            self.logger.error("Insufficient clean data for model training")
            return
            
        # Initialize isolation forest for outlier detection
        self._isolation_forest = IsolationForest(
            n_estimators=200,
            max_samples='auto',
            contamination=ANOMALY_CONTAMINATION_FACTOR,
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        self._isolation_forest.fit(combined_features)
        
        # Initialize DBSCAN for cluster-based anomaly detection
        self._dbscan = DBSCAN(
            eps=0.5,
            min_samples=5,
            metric='euclidean',
            n_jobs=-1
        )
        
        self.logger.info("Anomaly detection models successfully initialized")
    
    def _extract_anomaly_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Extract features for anomaly detection from market data.
        
        Args:
            data: Market data dataframe with OHLCV data
            
        Returns:
            DataFrame with extracted features or None if extraction fails
        """
        try:
            # Extract relevant columns and ensure required data is present
            if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                self.logger.warning("Missing required OHLCV columns in data")
                return None
            
            # Calculate basic features
            features = pd.DataFrame(index=data.index)
            
            # Price action features
            features['returns'] = data['close'].pct_change()
            features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            features['high_low_range'] = (data['high'] - data['low']) / data['low']
            features['body_size'] = abs(data['close'] - data['open']) / data['open']
            features['upper_wick'] = (data['high'] - data[['open', 'close']].max(axis=1)) / data['open']
            features['lower_wick'] = (data[['open', 'close']].min(axis=1) - data['low']) / data['open']
            
            # Volume features
            features['volume_change'] = data['volume'].pct_change()
            features['rel_volume'] = data['volume'] / data['volume'].rolling(20).mean()
            
            # Volatility features
            features['volatility'] = data['returns'].rolling(20).std()
            features['volatility_change'] = features['volatility'].pct_change()
            
            # Add momentum and trend features
            features['rsi'] = self._calculate_rsi(data['close'])
            features['macd_hist'] = self._calculate_macd(data['close'])[2]
            features['distance_from_ma'] = (data['close'] / data['close'].rolling(50).mean()) - 1
            
            # Normalize all features to avoid scale issues
            for col in features.columns:
                if features[col].dtype != 'object':
                    # Handle outliers by winsorizing
                    lower, upper = np.nanpercentile(features[col], [1, 99])
                    features[col] = features[col].clip(lower, upper)
                    
                    # Z-score normalization
                    features[col] = (features[col] - features[col].mean()) / features[col].std(ddof=0)
            
            # Drop any remaining NaN values from the beginning of the data
            features = features.dropna()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting anomaly features: {str(e)}", exc_info=True)
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(
        self, 
        prices: pd.Series, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    async def detect_inefficiencies(
        self, 
        asset: str, 
        timeframe: TimeFrame,
        lookback_periods: int = 500
    ) -> List[MarketInefficiency]:
        """
        Detect market inefficiencies for the specified asset and timeframe.
        
        Args:
            asset: The asset to analyze
            timeframe: The timeframe to analyze
            lookback_periods: Number of periods to look back for analysis
            
        Returns:
            List of detected market inefficiencies
        """
        self.logger.info(f"Detecting inefficiencies for {asset} on {timeframe.name} timeframe")
        
        # Get market data
        data = await self.market_data.get_candles(
            asset=asset,
            timeframe=timeframe,
            limit=lookback_periods
        )
        
        if data is None or len(data) < 100:
            self.logger.warning(f"Insufficient data for {asset} on {timeframe.name}")
            return []
            
        # Run all detection methods in parallel
        async with self._lock:
            detection_tasks = [
                create_async_task(self._detect_pricing_inefficiencies(asset, timeframe, data)),
                create_async_task(self._detect_volatility_inefficiencies(asset, timeframe, data)),
                create_async_task(self._detect_liquidity_inefficiencies(asset, timeframe, data)),
                create_async_task(self._detect_order_flow_inefficiencies(asset, timeframe, data)),
                create_async_task(self._detect_pattern_inefficiencies(asset, timeframe, data)),
                create_async_task(self._detect_statistical_arbitrage(asset, timeframe, data)),
                create_async_task(self._detect_market_microstructure_anomalies(asset, timeframe, data)),
                create_async_task(self._detect_behavior_based_inefficiencies(asset, timeframe, data)),
            ]
            
            # Gather all results
            results = await asyncio.gather(*detection_tasks, return_exceptions=True)
            
            # Process results, filtering out any exceptions
            inefficiencies = []
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Error in inefficiency detection: {str(result)}", exc_info=True)
                else:
                    inefficiencies.extend(result)
            
            # Update active inefficiencies
            self._update_active_inefficiencies(inefficiencies)
            
            # Track metrics
            self.metrics["total_detections"] += len(inefficiencies)
            for ineff in inefficiencies:
                self.metrics["detection_by_type"][ineff.type] = self.metrics["detection_by_type"].get(ineff.type, 0) + 1
                self.metrics["detection_by_asset"][ineff.asset] = self.metrics["detection_by_asset"].get(ineff.asset, 0) + 1
            
            return inefficiencies
    
    def _update_active_inefficiencies(self, new_inefficiencies: List[MarketInefficiency]):
        """Update the list of active inefficiencies with new detections."""
        # Remove expired inefficiencies
        self.active_inefficiencies = [
            ineff for ineff in self.active_inefficiencies if ineff.is_valid
        ]
        
        # Add new inefficiencies
        self.active_inefficiencies.extend(new_inefficiencies)
        
        # Sort by adjusted score (highest first)
        self.active_inefficiencies.sort(key=lambda x: x.adjusted_score, reverse=True)
    
    async def _detect_pricing_inefficiencies(
        self, 
        asset: str, 
        timeframe: TimeFrame,
        data: pd.DataFrame
    ) -> List[MarketInefficiency]:
        """
        Detect pricing inefficiencies like gaps, extended moves, and mean reversion opportunities.
        
        Args:
            asset: The asset being analyzed
            timeframe: The timeframe being analyzed
            data: Market data DataFrame
            
        Returns:
            List of detected pricing inefficiencies
        """
        inefficiencies = []
        
        try:
            # Calculate required metrics
            data['returns'] = data['close'].pct_change()
            data['ma_20'] = data['close'].rolling(window=20).mean()
            data['ma_50'] = data['close'].rolling(window=50).mean()
            data['ma_200'] = data['close'].rolling(window=200).mean()
            data['std_20'] = data['returns'].rolling(window=20).std()
            data['upper_band'] = data['ma_20'] + (data['std_20'] * 2)
            data['lower_band'] = data['ma_20'] - (data['std_20'] * 2)
            data['z_score'] = (data['close'] - data['ma_20']) / data['std_20']
            data['rsi'] = self._calculate_rsi(data['close'])
            
            # 1. Detect price gaps
            data['gap_up'] = (data['low'] > data['high'].shift(1)) & (data['open'] > data['close'].shift(1))
            data['gap_down'] = (data['high'] < data['low'].shift(1)) & (data['open'] < data['close'].shift(1))
            data['gap_size'] = np.zeros(len(data))
            data.loc[data['gap_up'], 'gap_size'] = (data['low'] - data['high'].shift(1)) / data['high'].shift(1)
            data.loc[data['gap_down'], 'gap_size'] = (data['high'] - data['low'].shift(1)) / data['low'].shift(1)
            
            # Find significant gaps in recent data
            recent_data = data.iloc[-30:]
            significant_gaps = recent_data[
                (abs(recent_data['gap_size']) > INEFFICIENCY_DETECTION_THRESHOLDS['min_gap_size'])
            ]
            
            for idx, row in significant_gaps.iterrows():
                # Calculate gap fill probability based on historical gaps
                gap_direction = 1 if row['gap_up'] else -1
                gap_fill_probability = self._calculate_gap_fill_probability(data, row['gap_size'], gap_direction)
                
                # Only include if probability is above threshold
                if gap_fill_probability > INEFFICIENCY_DETECTION_THRESHOLDS['min_gap_fill_probability']:
                    inefficiencies.append(MarketInefficiency(
                        type="price_gap",
                        asset=asset,
                        timeframe=timeframe,
                        timestamp=datetime.now(),
                        score=min(100, gap_fill_probability * 100),
                        direction=-gap_direction,  # Opposite of gap direction for mean reversion
                        expected_duration=timedelta(hours=24),  # Typically gaps fill within a day
                        details={
                            "gap_size": row['gap_size'],
                            "gap_direction": "up" if gap_direction == 1 else "down",
                            "gap_fill_probability": gap_fill_probability,
                            "gap_date": idx.isoformat()
                        },
                        source_data=data,
                        exploitation_strategy="gap_fill_reversion",
                        risk_assessment={
                            "stop_loss_distance": abs(row['gap_size']) * 1.5,
                            "profit_target_distance": abs(row['gap_size']),
                            "risk_reward_ratio": 1 / 1.5
                        }
                    ))
            
            # 2. Detect extended moves (overbought/oversold conditions)
            extreme_conditions = data.iloc[-10:][
                (abs(data['z_score']) > INEFFICIENCY_DETECTION_THRESHOLDS['extreme_z_score']) |
                (data['rsi'] > INEFFICIENCY_DETECTION_THRESHOLDS['extreme_rsi_high']) |
                (data['rsi'] < INEFFICIENCY_DETECTION_THRESHOLDS['extreme_rsi_low'])
            ]
            
            for idx, row in extreme_conditions.iterrows():
                # Determine direction (mean reversion)
                direction = -1 if row['z_score'] > 0 or row['rsi'] > 70 else 1
                
                # Calculate confidence score based on extremity
                confidence = (
                    min(100, (abs(row['z_score']) / INEFFICIENCY_DETECTION_THRESHOLDS['extreme_z_score']) * 80) if abs(row['z_score']) > INEFFICIENCY_DETECTION_THRESHOLDS['extreme_z_score']
                    else min(100, (abs(70 - row['rsi']) / 30) * 80)
                )
                
                # Calculate reversion probability
                reversion_prob = self._calculate_mean_reversion_probability(data, row['z_score'], row['rsi'])
                
                if reversion_prob > INEFFICIENCY_DETECTION_THRESHOLDS['min_reversion_probability']:
                    inefficiencies.append(MarketInefficiency(
                        type="extended_move",
                        asset=asset,
                        timeframe=timeframe,
                        timestamp=datetime.now(),
                        score=min(100, reversion_prob * 100),
                        direction=direction,
                        expected_duration=timedelta(hours=int(timeframe.value * 5)),  # Typically 5 candles
                        details={
                            "z_score": row['z_score'],
                            "rsi": row['rsi'],
                            "reversion_probability": reversion_prob,
                            "condition": "overbought" if direction == -1 else "oversold",
                        },
                        source_data=data,
                        exploitation_strategy="mean_reversion",
                        risk_assessment={
                            "stop_loss_distance": abs(row['std_20']) * 2,
                            "profit_target_distance": abs(row['z_score']) * 0.5 * row['std_20'],
                            "risk_reward_ratio": (abs(row['z_score']) * 0.5) / 2
                        }
                    ))
            
            # 3. Detect trend-following momentum opportunities
            data['trend_direction'] = np.where(
                (data['ma_20'] > data['ma_50']) & (data['ma_50'] > data['ma_200']),
                1,  # Uptrend
                np.where(
                    (data['ma_20'] < data['ma_50']) & (data['ma_50'] < data['ma_200']),
                    -1,  # Downtrend
                    0  # No clear trend
                )
            )
            
            # Identify pullbacks in strong trends
            data['pullback'] = (
                ((data['trend_direction'] == 1) & (data['close'] < data['ma_20']) & (data['close'] > data['ma_50'])) |
                ((data['trend_direction'] == -1) & (data['close'] > data['ma_20']) & (data['close'] < data['ma_50']))
            )
            
            recent_pullbacks = data.iloc[-5:][data['pullback']]
            for idx, row in recent_pullbacks.iterrows():
                trend_strength = self._calculate_trend_strength(data)
                if trend_strength > INEFFICIENCY_DETECTION_THRESHOLDS['min_trend_strength']:
                    inefficiencies.append(MarketInefficiency(
                        type="trend_pullback",
                        asset=asset,
                        timeframe=timeframe,
                        timestamp=datetime.now(),
                        score=min(100, trend_strength * 100),
                        direction=row['trend_direction'],
                        expected_duration=timedelta(hours=int(timeframe.value * 10)),  # Typically 10 candles
                        details={
                            "trend_direction": "up" if row['trend_direction'] == 1 else "down",
                            "trend_strength": trend_strength,
                            "ma_20": row['ma_20'],
                            "ma_50": row['ma_50'],
                            "ma_200": row['ma_200']
                        },
                        source_data=data,
                        exploitation_strategy="trend_following",
                        risk_assessment={
                            "stop_loss_distance": abs(row['close'] - row['ma_50']) * 1.2,
                            "profit_target_distance": abs(row['close'] - row['ma_20']) * 3,
                            "risk_reward_ratio": 3 / 1.2
                        }
                    ))
                    
        except Exception as e:
            self.logger.error(f"Error detecting pricing inefficiencies: {str(e)}", exc_info=True)
            
        return inefficiencies
    
    def _calculate_gap_fill_probability(
        self, 
        data: pd.DataFrame, 
        gap_size: float, 
        gap_direction: int
    ) -> float:
        """
        Calculate the probability of a gap filling based on historical data.
        
        Args:
            data: Historical price data
            gap_size: Size of the gap as percentage
            gap_direction: Direction of the gap (1=up, -1=down)
            
        Returns:
            Probability of gap filling (0.0-1.0)
        """
        try:
            # Find similar historical gaps
            data['historical_gap_up'] = (data['low'] > data['high'].shift(1)) & (data['open'] > data['close'].shift(1))
            data['historical_gap_down'] = (data['high'] < data['low'].shift(1)) & (data['open'] < data['close'].shift(1))
            data['historical_gap_size'] = np.zeros(len(data))
            data.loc[data['historical_gap_up'], 'historical_gap_size'] = (data['low'] - data['high'].shift(1)) / data['high'].shift(1)
            data.loc[data['historical_gap_down'], 'historical_gap_size'] = (data['high'] - data['low'].shift(1)) / data['low'].shift(1)
            
            # Filter for gaps in the same direction and similar size
            if gap_direction == 1:
                similar_gaps = data[
                    data['historical_gap_up'] & 
                    (data['historical_gap_size'] >= gap_size * 0.7) & 
                    (data['historical_gap_size'] <= gap_size * 1.3)
                ]
            else:
                similar_gaps = data[
                    data['historical_gap_down'] & 
                    (data['historical_gap_size'] >= gap_size * 0.7) & 
                    (data['historical_gap_size'] <= gap_size * 1.3)
                ]
            
            if len(similar_gaps) < 5:
                # Not enough historical examples
                return 0.6  # Default probability
            
            # Calculate how many gaps were filled within 5 candles
            filled_count = 0
            for i in similar_gaps.index:
                # Find the next 5 candles after the gap
                next_candles_idx = data.index.get_indexer([i])[0] + np.arange(1, 6)
                next_candles_idx = next_candles_idx[next_candles_idx < len(data)]
                
                if len(next_candles_idx) == 0:
                    continue
                    
                next_candles = data.iloc[next_candles_idx]
                
                # Check if gap was filled
                if gap_direction == 1:
                    # For gap up, need low price to go below previous high
                    gap_filled = (next_candles['low'] <= data.loc[i, 'high'].shift(1)).any()
                else:
                    # For gap down, need high price to go above previous low
                    gap_filled = (next_candles['high'] >= data.loc[i, 'low'].shift(1)).any()
                    
                if gap_filled:
                    filled_count += 1
            
            # Calculate probability
            return filled_count / len(similar_gaps)
            
        except Exception as e:
            self.logger.warning(f"Error calculating gap fill probability: {str(e)}")
            return 0.6  # Default probability
    
    def _calculate_mean_reversion_probability(
        self, 
        data: pd.DataFrame, 
        z_score: float, 
        rsi: float
    ) -> float:
        """
        Calculate the probability of mean reversion based on historical data.
        
        Args:
            data: Historical price data
            z_score: Current z-score value
            rsi: Current RSI value
            
        Returns:
            Probability of mean reversion (0.0-1.0)
        """
        try:
            # Find historical instances of similar extreme conditions
            extreme_z = (abs(data['z_score']) >= abs(z_score) * 0.8) & (abs(data['z_score']) <= abs(z_score) * 1.2)
            
            if z_score > 0:
                extreme_rsi = (data['rsi'] >= rsi * 0.9) & (data['rsi'] <= rsi * 1.1) & (data['rsi'] > 70)
            else:
                extreme_rsi = (data['rsi'] >= rsi * 0.9) & (data['rsi'] <= rsi * 1.1) & (data['rsi'] < 30)
                
            # Combine conditions
            similar_conditions = data[extreme_z | extreme_rsi]
            
            if len(similar_conditions) < 5:
                # Not enough historical examples
                return 0.7  # Default probability
            
            # Calculate how many times price reverted to mean within 5 candles
            reverted_count = 0
            for i in similar_conditions.index:
                # Find the next 5 candles after the extreme condition
                next_candles_idx = data.index.get_indexer([i])[0] + np.arange(1, 6)
                next_candles_idx = next_candles_idx[next_candles_idx < len(data)]
                
                if len(next_candles_idx) == 0:
                    continue
                    
                next_candles = data.iloc[next_candles_idx]
                
                # Check if price reverted towards mean
                if z_score > 0 or rsi > 70:
                    # For overbought conditions, check if price decreased
                    reversion = (next_candles['close'] <= data.loc[i, 'ma_20']).any()
                else:
                    # For oversold conditions, check if price increased
                    reversion = (next_candles['close'] >= data.loc[i, 'ma_20']).any()
                    
                if reversion:
                    reverted_count += 1
            
            # Calculate probability
            return reverted_count / len(similar_conditions)
            
        except Exception as e:
            self.logger.warning(f"Error calculating mean reversion probability: {str(e)}")
            return 0.7  # Default probability
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate the strength of the current trend.
        
        Args:
            data: Historical price data
            
        Returns:
            Trend strength (0.0-1.0)
        """
        try:
            # Use recent data for trend strength calculation
            recent_data = data.iloc[-50:]
            
            # Calculate directional movement
            if 'trend_direction' not in recent_data.columns:
                return 0.5
                
            # Count consistent trend periods
            trend_consistency = recent_data['trend_direction'].value_counts()
            dominant_trend = trend_consistency.idxmax() if len(trend_consistency) > 0 else 0
            
            if dominant_trend == 0:
                return 0.5  # No clear trend
                
            # Calculate trend strength metrics
            trend_length = len(recent_data[recent_data['trend_direction'] == dominant_trend])
            trend_percentage = trend_length / len(recent_data)
            
            # Calculate slope of MA20
            ma20_slope = recent_data['ma_20'].diff().mean() * 100
            
            # Normalize and combine metrics
            normalized_trend_percentage = min(1.0, trend_percentage / 0.8)
            normalized_ma20_slope = min(1.0, abs(ma20_slope) / 0.5)
            
            # Combine metrics with weights
            trend_strength = (normalized_trend_percentage * 0.7) + (normalized_ma20_slope * 0.3)
            
            return min(1.0, max(0.0, trend_strength))
            
        except Exception as e:
            self.logger.warning(f"Error calculating trend strength: {str(e)}")
            return 0.5  # Default strength
    
    async def _detect_volatility_inefficiencies(
        self, 
        asset: str, 
        timeframe: TimeFrame,
        data: pd.DataFrame
    ) -> List[MarketInefficiency]:
        """
        Detect volatility-based inefficiencies such as volatility squeezes and expansions.
        
        Args:
            asset: The asset being analyzed
            timeframe: The timeframe being analyzed
            data: Market data DataFrame
            
        Returns:
            List of detected volatility inefficiencies
        """
        inefficiencies = []
        
        try:
            # Calculate volatility metrics
            data['returns'] = data['close'].pct_change()
            data['atr'] = self._calculate_atr(data, window=14)
            data['atr_percent'] = data['atr'] / data['close']
            data['volatility_20'] = data['returns'].rolling(window=20).std()
            data['volatility_20_sma'] = data['volatility_20'].rolling(window=20).mean()
            data['volatility_ratio'] = data['volatility_20'] / data['volatility_20_sma']
            
            # Calculate Bollinger Bands
            data['bb_middle'] = data['close'].rolling(window=20).mean()
            data['bb_std'] = data['close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * 2)
            data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * 2)
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            data['bb_width_sma'] = data['bb_width'].rolling(window=50).mean()
            data['bb_squeeze'] = data['bb_width'] / data['bb_width_sma']
            
            # 1. Detect Volatility Squeezes (compression before expansion)
            # Look for narrowing Bollinger Bands
            data['volatility_squeeze'] = (
                (data['bb_squeeze'] < INEFFICIENCY_DETECTION_THRESHOLDS['squeeze_threshold']) &
                (data['bb_squeeze'].shift(1) < INEFFICIENCY_DETECTION_THRESHOLDS['squeeze_threshold']) &
                (data['bb_squeeze'].shift(2) < INEFFICIENCY_DETECTION_THRESHOLDS['squeeze_threshold'])
            )
            
            # Check recent data for volatility squeezes
            recent_data = data.iloc[-10:]
            squeezes = recent_data[recent_data['volatility_squeeze']]
            
            for idx, row in squeezes.iterrows():
                # Calculate success probability for volatility breakouts
                breakout_prob = self._calculate_volatility_breakout_probability(data)
                
                if breakout_prob > INEFFICIENCY_DETECTION_THRESHOLDS['min_breakout_probability']:
                    # Determine potential breakout direction based on recent momentum
                    momentum = self._calculate_short_term_momentum(data)
                    direction = np.sign(momentum) if abs(momentum) > 0.3 else 0
                    
                    inefficiencies.append(MarketInefficiency(
                        type="volatility_squeeze",
                        asset=asset,
                        timeframe=timeframe,
                        timestamp=datetime.now(),
                        score=min(100, breakout_prob * 100),
                        direction=direction,  # Direction based on momentum
                        expected_duration=timedelta(hours=int(timeframe.value * 5)),  # Typically 5 candles
                        details={
                            "bb_squeeze": row['bb_squeeze'],
                            "bb_width": row['bb_width'],
                            "volatility_20": row['volatility_20'],
                            "breakout_probability": breakout_prob,
                            "momentum": momentum
                        },
                        source_data=data,
                        exploitation_strategy="volatility_breakout",
                        risk_assessment={
                            "stop_loss_distance": row['atr'] * 1.5,
                            "profit_target_distance": row['atr'] * 3,
                            "risk_reward_ratio": 3 / 1.5
                        }
                    ))
            
            # 2. Detect Volatility Regime Changes
            data['volatility_regime_change'] = (
                (data['volatility_ratio'] > INEFFICIENCY_DETECTION_THRESHOLDS['volatility_regime_change']) &
                (data['volatility_ratio'].shift(1) <= INEFFICIENCY_DETECTION_THRESHOLDS['volatility_regime_change'])
            )
            
            recent_regime_changes = data.iloc[-5:][data['volatility_regime_change']]
            
            for idx, row in recent_regime_changes.iterrows():
                # Calculate success probability for volatility regime trades
                regime_trade_prob = self._calculate_volatility_regime_probability(data)
                
                if regime_trade_prob > INEFFICIENCY_DETECTION_THRESHOLDS['min_regime_probability']:
                    inefficiencies.append(MarketInefficiency(
                        type="volatility_regime_change",
                        asset=asset,
                        timeframe=timeframe,
                        timestamp=datetime.now(),
                        score=min(100, regime_trade_prob * 100),
                        direction=0,  # Neutral direction, strategy depends on regime
                        expected_duration=timedelta(hours=int(timeframe.value * 10)),  # Typically 10 candles
                        details={
                            "volatility_ratio": row['volatility_ratio'],
                            "current_volatility": row['volatility_20'],
                            "average_volatility": row['volatility_20_sma'],
                            "regime_change_probability": regime_trade_prob
                        },
                        source_data=data,
                        exploitation_strategy="volatility_regime_adaptation",
                        risk_assessment={
                            "stop_loss_distance": row['atr'] * 2,
                            "profit_target_distance": row['atr'] * 3,
                            "risk_reward_ratio": 3 / 2
                        }
                    ))
            
        except Exception as e:
            self.logger.error(f"Error detecting volatility inefficiencies: {str(e)}", exc_info=True)
            
        return inefficiencies
    
    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    def _calculate_volatility_breakout_probability(self, data: pd.DataFrame) -> float:
        """
        Calculate probability of successful volatility breakout trade.
        
        Args:
            data: Historical price data
            
        Returns:
            Probability of successful breakout (0.0-1.0)
        """
        try:
            # Find historical volatility squeezes
            if 'volatility_squeeze' not in data.columns:
                return 0.65  # Default probability
                
            historical_squeezes = data[data['volatility_squeeze']].index
            
            if len(historical_squeezes) < 5:
                return 0.65  # Not enough historical examples
                
            # Calculate success rate of historical breakouts
            success_count = 0
            for idx in historical_squeezes:
                position = data.index.get_loc(idx)
                
                # Look at the next 10 candles after squeeze
                if position + 10 >= len(data):
                    continue
                    
                next_candles = data.iloc[position+1:position+11]
                
                # Calculate price range before squeeze
                pre_squeeze = data.iloc[position-5:position+1]
                price_range = pre_squeeze['high'].max() - pre_squeeze['low'].min()
                
                # Check if price broke out by at least 1.5x the prior range
                max_move = max(
                    next_candles['high'].max() - data.iloc[position]['close'],
                    data.iloc[position]['close'] - next_candles['low'].min()
                )
                
                if max_move > price_range * 1.5:
                    success_count += 1
            
            return success_count / len(historical_squeezes)
            
        except Exception as e:
            self.logger.warning(f"Error calculating volatility breakout probability: {str(e)}")
            return 0.65  # Default probability
    
    def _calculate_short_term_momentum(self, data: pd.DataFrame) -> float:
        """
        Calculate short-term price momentum to predict breakout direction.
        
        Args:
            data: Historical price data
            
        Returns:
            Momentum indicator (-1.0 to 1.0)
        """
        try:
            # Use recent price action for momentum calculation
            recent_data = data.iloc[-20:]
            
            # Calculate momentum indicators
            if 'returns' not in recent_data.columns:
                recent_data['returns'] = recent_data['close'].pct_change()
                
            # 1. Price vs short-term moving averages
            recent_data['ma5'] = recent_data['close'].rolling(window=5).mean()
            recent_data['ma10'] = recent_data['close'].rolling(window=10).mean()
            ma_relation = 1 if recent_data['close'].iloc[-1] > recent_data['ma5'].iloc[-1] else -1
            ma_relation *= 1 if recent_data['ma5'].iloc[-1] > recent_data['ma10'].iloc[-1] else -1
            
            # 2. Recent returns direction
            recent_returns = recent_data['returns'].dropna().iloc[-5:].sum()
            returns_direction = np.sign(recent_returns)
            
            # 3. Volume-weighted momentum
            if 'volume' in recent_data.columns:
                vol_returns = recent_data['returns'].dropna().iloc[-5:] * recent_data['volume'].iloc[-5:].values
                vol_weighted_direction = np.sign(vol_returns.sum())
            else:
                vol_weighted_direction = 0
            
            # Combine signals with weights
            momentum = (
                (ma_relation * 0.4) + 
                (returns_direction * 0.4) + 
                (vol_weighted_direction * 0.2)
            )
            
            return max(-1.0, min(1.0, momentum))
            
        except Exception as e:
            self.logger.warning(f"Error calculating short-term momentum: {str(e)}")
            return 0.0  # Neutral momentum
    
    def _calculate_volatility_regime_probability(self, data: pd.DataFrame) -> float:
        """
        Calculate probability of successful volatility regime change trade.
        
        Args:
            data: Historical price data
            
        Returns:
            Probability of successful regime trade (0.0-1.0)
        """
        try:
            # Find historical volatility regime changes
            if 'volatility_regime_change' not in data.columns:
                return 0.6  # Default probability
                
            historical_changes = data[data['volatility_regime_change']].index
            
            if len(historical_changes) < 5:
                return 0.6  # Not enough historical examples
                
            # Calculate success rate of historical regime trades
            # Success defined as profitable range trading during high volatility
            success_count = 0
            for idx in historical_changes:
                position = data.index.get_loc(idx)
                
                # Look at the next 10 candles after regime change
                if position + 10 >= len(data):
                    continue
                    
                next_candles = data.iloc[position+1:position+11]
                
                # Calculate price range in the period
                price_range = next_candles['high'].max() - next_candles['low'].min()
                avg_atr = data.iloc[position]['atr'] * 10  # Expected range for 10 periods
                
                # If actual range > 1.5x expected range, consider successful
                if price_range > avg_atr * 1.5:
                    success_count += 1
            
            return success_count / len(historical_changes)
            
        except Exception as e:
            self.logger.warning(f"Error calculating volatility regime probability: {str(e)}")
            return 0.6  # Default probability
    
    async def _detect_liquidity_inefficiencies(
        self, 
        asset: str, 
        timeframe: TimeFrame,
        data: pd.DataFrame
    ) -> List[MarketInefficiency]:
        """
        Detect liquidity-based inefficiencies such as stop clusters and liquidity voids.
        
        Args:
            asset: The asset being analyzed
            timeframe: The timeframe being analyzed
            data: Market data DataFrame
            
        Returns:
            List of detected liquidity inefficiencies
        """
        inefficiencies = []
        
        try:
            # Need additional order book data for more accurate liquidity analysis
            # This is a simplified approach based only on price/volume data
            
            # Calculate swing highs and lows
            data['swing_high'] = (
                (data['high'] > data['high'].shift(1)) &
                (data['high'] > data['high'].shift(2)) &
                (data['high'] > data['high'].shift(-1)) &
                (data['high'] > data['high'].shift(-2))
            )
            
            data['swing_low'] = (
                (data['low'] < data['low'].shift(1)) &
                (data['low'] < data['low'].shift(2)) &
                (data['low'] < data['low'].shift(-1)) &
                (data['low'] < data['low'].shift(-2))
            )
            
            # 1. Detect stop loss clusters
            # Common stop locations are just beyond recent swing points
            recent_data = data.iloc[-50:]
            swing_highs = recent_data[recent_data['swing_high']]['high']
            swing_lows = recent_data[recent_data['swing_low']]['low']
            
            # Find clusters of swing points (potential liquidity pools)
            high_clusters = self._find_price_clusters(swing_highs)
            low_clusters = self._find_price_clusters(swing_lows)
            
            # Check current price relative to clusters
            current_price = data['close'].iloc[-1]
            
            # Check high clusters for short stop hunting
            for cluster_price, strength in high_clusters:
                if current_price < cluster_price < current_price * 1.02:
                    # Close to a high cluster, potential liquidity raid opportunity
                    exploitation_prob = self._calculate_liquidity_raid_probability(data, cluster_price, True)
                    
                    if exploitation_prob > INEFFICIENCY_DETECTION_THRESHOLDS['min_liquidity_probability']:
                        inefficiencies.append(MarketInefficiency(
                            type="stop_cluster_high",
                            asset=asset,
                            timeframe=timeframe,
                            timestamp=datetime.now(),
                            score=min(100, exploitation_prob * 100),
                            direction=1,  # Upward (stop hunting above cluster)
                            expected_duration=timedelta(hours=int(timeframe.value * 3)),  # Typically quick
                            details={
                                "cluster_price": cluster_price,
                                "cluster_strength": strength,
                                "distance_percent": (cluster_price / current_price - 1) * 100,
                                "exploitation_probability": exploitation_prob
                            },
                            source_data=data,
                            exploitation_strategy="liquidity_raid",
                            risk_assessment={
                                "stop_loss_distance": abs(cluster_price - current_price) * 1.5,
                                "profit_target_distance": abs(cluster_price - current_price) * 2,
                                "risk_reward_ratio": 2 / 1.5
                            }
                        ))
            
            # Check low clusters for long stop hunting
            for cluster_price, strength in low_clusters:
                if current_price > cluster_price > current_price * 0.98:
                    # Close to a low cluster, potential liquidity raid opportunity
                    exploitation_prob = self._calculate_liquidity_raid_probability(data, cluster_price, False)
                    
                    if exploitation_prob > INEFFICIENCY_DETECTION_THRESHOLDS['min_liquidity_probability']:
                        inefficiencies.append(MarketInefficiency(
                            type="stop_cluster_low",
                            asset=asset,
                            timeframe=timeframe,
                            timestamp=datetime.now(),
                            score=min(100, exploitation_prob * 100),
                            direction=-1,  # Downward (stop hunting below cluster)
                            expected_duration=timedelta(hours=int(timeframe.value * 3)),  # Typically quick
                            details={
                                "cluster_price": cluster_price,
                                "cluster_strength": strength,
                                "distance_percent": (1 - cluster_price / current_price) * 100,
                                "exploitation_probability": exploitation_prob
                            },
                            source_data=data,
                            exploitation_strategy="liquidity_raid",
                            risk_assessment={
                                "stop_loss_distance": abs(cluster_price - current_price) * 1.5,
                                "profit_target_distance": abs(cluster_price - current_price) * 2,
                                "risk_reward_ratio": 2 / 1.5
                            }
                        ))
            
            # 2. Detect liquidity voids (areas of thin trading)
            # Calculate volume profile
            data['price_rounded'] = (data['close'] / (data['close'].iloc[-1] * 0.001)).round() * (data['close'].iloc[-1] * 0.001)
            volume_profile = data.groupby('price_rounded')['volume'].sum()
            
            # Find low liquidity zones near current price
            low_liquidity_zones = self._find_liquidity_voids(volume_profile, current_price)
            
            for zone_price, zone_width in low_liquidity_zones:
                # For low liquidity zones, determine if price is likely to move quickly through them
                void_exploitation_prob = self._calculate_liquidity_void_probability(data, zone_price, zone_width)
                
                if void_exploitation_prob > INEFFICIENCY_DETECTION_THRESHOLDS['min_void_probability']:
                    # Determine direction (toward the void)
                    direction = 1 if zone_price > current_price else -1
                    
                    inefficiencies.append(MarketInefficiency(
                        type="liquidity_void",
                        asset=asset,
                        timeframe=timeframe,
                        timestamp=datetime.now(),
                        score=min(100, void_exploitation_prob * 100),
                        direction=direction,
                        expected_duration=timedelta(hours=int(timeframe.value * 5)),
                        details={
                            "void_price": zone_price,
                            "void_width": zone_width,
                            "distance_percent": abs(zone_price / current_price - 1) * 100,
                            "exploitation_probability": void_exploitation_prob
                        },
                        source_data=data,
                        exploitation_strategy="liquidity_void_exploitation",
                        risk_assessment={
                            "stop_loss_distance": zone_width * 0.5,
                            "profit_target_distance": zone_width * 2,
                            "risk_reward_ratio": 2 / 0.5
                        }
                    ))
            
        except Exception as e:
            self.logger.error(f"Error detecting liquidity inefficiencies: {str(e)}", exc_info=True)
            
        return inefficiencies
    
    def _find_price_clusters(self, prices: pd.Series, proximity_pct: float = 0.005) -> List[Tuple[float, int]]:
        """
        Find clusters of prices that are close to each other.
        
        Args:
            prices: Series of price points to cluster
            proximity_pct: How close prices must be to form a cluster (as % of price)
            
        Returns:
            List of (cluster_price, strength) tuples
        """
        if len(prices) == 0:
            return []
            
        # Convert Series to sorted list
        price_list = sorted(prices.tolist())
        
        clusters = []
        current_cluster = [price_list[0]]
        
        # Group prices into clusters
        for price in price_list[1:]:
            if price <= current_cluster[-1] * (1 + proximity_pct):
                current_cluster.append(price)
            else:
                # Save the completed cluster
                if len(current_cluster) > 1:
                    cluster_price = sum(current_cluster) / len(current_cluster)
                    clusters.append((cluster_price, len(current_cluster)))
                
                # Start a new cluster
                current_cluster = [price]
        
        # Don't forget the last cluster
        if len(current_cluster) > 1:
            cluster_price = sum(current_cluster) / len(current_cluster)
            clusters.append((cluster_price, len(current_cluster)))
            
        # Sort by strength (descending)
        return sorted(clusters, key=lambda x: x[1], reverse=True)
    
    def _calculate_liquidity_raid_probability(
        self, 
        data: pd.DataFrame, 
        cluster_price: float, 
        is_high_cluster: bool
    ) -> float:
        """
        Calculate probability of successful liquidity raid trade.
        
        Args:
            data: Historical price data
            cluster_price: Price level of the stop cluster
            is_high_cluster: Whether this is a high or low cluster
            
        Returns:
            Probability of successful liquidity raid (0.0-1.0)
        """
        try:
            # Find historical swing points
            if not ('swing_high' in data.columns and 'swing_low' in data.columns):
                return 0.6  # Default probability
            
            # Identify historical similar setups
            current_price = data['close'].iloc[-1]
            price_diff_pct = abs(cluster_price / current_price - 1)
            
            # Look for similar historical scenarios
            similar_setups = 0
            successful_raids = 0
            
            for i in range(50, len(data) - 10):
                ref_price = data['close'].iloc[i]
                
                # For high clusters
                if is_high_cluster:
                    # Find nearby swing highs
                    nearby_highs = data.iloc[i-30:i][data.iloc[i-30:i]['swing_high']]['high']
                    
                    for high in nearby_highs:
                        high_diff_pct = high / ref_price - 1
                        
                        # If similar setup (high cluster at similar distance)
                        if 0.8 * price_diff_pct <= high_diff_pct <= 1.2 * price_diff_pct:
                            similar_setups += 1
                            
                            # Check if price reached the cluster and then reversed
                            next_candles = data.iloc[i+1:i+11]
                            if (next_candles['high'] >= high * 0.998).any() and (next_candles['close'].iloc[-1] < high):
                                successful_raids += 1
                else:
                    # Find nearby swing lows
                    nearby_lows = data.iloc[i-30:i][data.iloc[i-30:i]['swing_low']]['low']
                    
                    for low in nearby_lows:
                        low_diff_pct = 1 - low / ref_price
                        
                        # If similar setup (low cluster at similar distance)
                        if 0.8 * price_diff_pct <= low_diff_pct <= 1.2 * price_diff_pct:
                            similar_setups += 1
                            
                            # Check if price reached the cluster and then reversed
                            next_candles = data.iloc[i+1:i+11]
                            if (next_candles['low'] <= low * 1.002).any() and (next_candles['close'].iloc[-1] > low):
                                successful_raids += 1
            
            # Calculate success probability
            if similar_setups >= 5:
                return successful_raids / similar_setups
            else:
                return 0.6  # Default if not enough historical examples
                
        except Exception as e:
            self.logger.warning(f"Error calculating liquidity raid probability: {str(e)}")
            return 0.6  # Default probability
    
    def _find_liquidity_voids(
        self, 
        volume_profile: pd.Series, 
        current_price: float, 
        max_distance_pct: float = 0.05
    ) -> List[Tuple[float, float]]:
        """
        Find liquidity voids (price areas with significantly lower volume).
        
        Args:
            volume_profile: Volume by price level
            current_price: Current market price
            max_distance_pct: Maximum distance to look for voids as % of price
            
        Returns:
            List of (void_price, void_width) tuples
        """
        try:
            # Filter to relevant price range near current price
            min_price = current_price * (1 - max_distance_pct)
            max_price = current_price * (1 + max_distance_pct)
            
            nearby_profile = volume_profile[(volume_profile.index >= min_price) & 
                                           (volume_profile.index <= max_price)]
            
            if len(nearby_profile) < 5:
                return []  # Not enough price points to find voids
                
            # Calculate average volume in this range
            avg_volume = nearby_profile.mean()
            
            # Find low volume areas (less than 30% of average)
            low_vol_prices = nearby_profile[nearby_profile < avg_volume * 0.3].index.tolist()
            
            if not low_vol_prices:
                return []
                
            # Group into continuous voids
            voids = []
            current_void = [low_vol_prices[0]]
            
            for price in low_vol_prices[1:]:
                # Check if prices are close enough to be in same void
                if price <= current_void[-1] * 1.005:
                    current_void.append(price)
                else:
                    # Save completed void
                    if len(current_void) >= 3:  # Require at least 3 price points
                        void_center = sum(current_void) / len(current_void)
                        void_width = max(current_void) - min(current_void)
                        voids.append((void_center, void_width))
                    
                    # Start a new void
                    current_void = [price]
            
            # Don't forget the last void
            if len(current_void) >= 3:
                void_center = sum(current_void) / len(current_void)
                void_width = max(current_void) - min(current_void)
                voids.append((void_center, void_width))
                
            return voids
            
        except Exception as e:
            self.logger.warning(f"Error finding liquidity voids: {str(e)}")
            return []
    
    def _calculate_liquidity_void_probability(
        self, 
        data: pd.DataFrame, 
        void_price: float, 
        void_width: float
    ) -> float:
        """
        Calculate probability of price moving quickly through a liquidity void.
        
        Args:
            data: Historical price data
            void_price: Center price of the liquidity void
            void_width: Width of the liquidity void
            
        Returns:
            Probability of successful void exploitation (0.0-1.0)
        """
        try:
            # Using basic price action to estimate void exploitation probability
            current_price = data['close'].iloc[-1]
            
            # Calculate distance metrics
            distance_pct = abs(void_price / current_price - 1)
            
            # Decrease probability as distance increases
            distance_factor = max(0, 1 - (distance_pct / 0.05))
            
            # Width factor - wider voids are more significant
            average_daily_range = data['high'].iloc[-20:].max() - data['low'].iloc[-20:].min()
            width_factor = min(1, void_width / (average_daily_range * 0.1))
            
            # Momentum factor - price tends to accelerate with trend
            momentum = self._calculate_short_term_momentum(data)
            momentum_alignment = 0.5
            if (void_price > current_price and momentum > 0) or (void_price < current_price and momentum < 0):
                # Momentum aligned with void direction
                momentum_alignment = 0.5 + (abs(momentum) * 0.5)
            
            # Combine factors
            probability = (distance_factor * 0.4) + (width_factor * 0.3) + (momentum_alignment * 0.3)
            
            return max(0, min(1, probability))
            
        except Exception as e:
            self.logger.warning(f"Error calculating liquidity void probability: {str(e)}")
            return 0.5  # Default probability
    
    async def _detect_order_flow_inefficiencies(
        self, 
        asset: str, 
        timeframe: TimeFrame,
        data: pd.DataFrame
    ) -> List[MarketInefficiency]:
        """
        Detect order flow-based inefficiencies from market microstructure.
        
        Args:
            asset: The asset being analyzed
            timeframe: The timeframe being analyzed
            data: Market data DataFrame
            
        Returns:
            List of detected order flow inefficiencies
        """
        # This requires order book data which we don't have in the standard OHLCV data
        # For a complete implementation, integrate with order book snapshots
        # This is a placeholder implementation with basic approximations
        return []
    
    async def _detect_pattern_inefficiencies(
        self, 
        asset: str, 
        timeframe: TimeFrame,
        data: pd.DataFrame
    ) -> List[MarketInefficiency]:
        """
        Detect pattern-based inefficiencies from chart patterns, harmonics, etc.
        
        Args:
            asset: The asset being analyzed
            timeframe: The timeframe being analyzed
            data: Market data DataFrame
            
        Returns:
            List of detected pattern inefficiencies
        """
        # This would typically integrate with the pattern recognition module
        # For a complete implementation, connect with pattern recognition service
        # This is a placeholder implementation with basic approximations
        return []
    
    async def _detect_statistical_arbitrage(
        self, 
        asset: str, 
        timeframe: TimeFrame,
        data: pd.DataFrame
    ) -> List[MarketInefficiency]:
        """
        Detect statistical arbitrage opportunities from cointegration and pair trading.
        
        Args:
            asset: The asset being analyzed
            timeframe: The timeframe being analyzed
            data: Market data DataFrame
            
        Returns:
            List of detected statistical arbitrage opportunities
        """
        # This requires data from multiple correlated assets
        # For a complete implementation, integrate with cross-asset analysis
        # This is a placeholder implementation with basic approximations
        return []
    
    async def _detect_market_microstructure_anomalies(
        self, 
        asset: str, 
        timeframe: TimeFrame,
        data: pd.DataFrame
    ) -> List[MarketInefficiency]:
        """
        Detect anomalies in market microstructure like order book imbalances.
        
        Args:
            asset: The asset being analyzed
            timeframe: The timeframe being analyzed
            data: Market data DataFrame
            
        Returns:
            List of detected market microstructure anomalies
        """
        # This requires tick-level and order book data
        # For a complete implementation, integrate with market microstructure analysis
        # This is a placeholder implementation with basic approximations
        return []
    
    async def _detect_behavior_based_inefficiencies(
        self, 
        asset: str, 
        timeframe: TimeFrame,
        data: pd.DataFrame
    ) -> List[MarketInefficiency]:
        """
        Detect inefficiencies based on trader behavior and psychology.
        
        Args:
            asset: The asset being analyzed
            timeframe: The timeframe being analyzed
            data: Market data DataFrame
            
        Returns:
            List of detected behavior-based inefficiencies
        """
        # This requires sentiment data and trader positioning
        # For a complete implementation, integrate with sentiment analysis
        # This is a placeholder implementation with basic approximations
        return []
    
    async def get_inefficiency_performance(self, lookback_days: int = 30) -> Dict:
        """
        Get performance metrics for detected inefficiencies.
        
        Args:
            lookback_days: Number of days to look back for performance analysis
            
        Returns:
            Dictionary of performance metrics
        """
        # Calculate performance metrics for detected inefficiencies
        # This would typically be updated by a feedback loop that tracks
        # the outcome of inefficiency-based trading signals
        return self.metrics
    
    async def clear_stale_inefficiencies(self):
        """Remove stale and invalid inefficiencies from tracking."""
        async with self._lock:
            self.active_inefficiencies = [
                ineff for ineff in self.active_inefficiencies if ineff.is_valid
            ]
