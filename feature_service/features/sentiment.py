

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Sentiment Feature Module

This module provides sophisticated sentiment analysis features that extract
insights from news, social media, and market sentiment indicators to predict
market movements and trader behavior.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import re
import datetime
from collections import defaultdict
import math
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

from common.constants import (
    SENTIMENT_DECAY_FACTOR, SENTIMENT_SOURCE_WEIGHTS,
    SENTIMENT_IMPACT_WINDOW, SENTIMENT_ENTITIES
)
from common.utils import exponential_decay_weights, sigmoid, calculate_rolling_correlation
from feature_service.features.base_feature import BaseFeature
from data_storage.time_series import TimeSeriesStorage

logger = logging.getLogger(__name__)


@dataclass
class SentimentAnalysisResult:
    """Data class for storing sentiment analysis results."""
    compound_score: float  # Overall sentiment score (-1 to 1)
    positive_score: float  # Positive component (0 to 1)
    negative_score: float  # Negative component (0 to 1)
    neutral_score: float  # Neutral component (0 to 1)
    sources: Dict[str, float]  # Sentiment by source
    entities: Dict[str, float]  # Sentiment by entity mentioned
    confidence: float  # Model confidence (0 to 1)
    market_impact: Dict[str, float]  # Estimated impact by timeframe


@dataclass
class SentimentTrendResult:
    """Data class for sentiment trend analysis."""
    trend_direction: str  # 'improving', 'deteriorating', 'stable'
    trend_strength: float  # 0 to 1 scale
    change_rate: float  # Rate of sentiment change
    reversals: List[Dict[str, Any]]  # List of sentiment reversal points
    sentiment_series: Dict[str, List[float]]  # Time series of sentiment


@dataclass
class MarketMoodResult:
    """Data class for overall market mood analysis."""
    fear_greed_index: float  # 0-100 scale (0=extreme fear, 100=extreme greed)
    market_regime: str  # 'risk_on', 'risk_off', 'neutral'
    institutional_sentiment: float  # -1 to 1 scale
    retail_sentiment: float  # -1 to 1 scale
    volatility_sentiment: float  # -1 to 1 scale
    sector_sentiment: Dict[str, float]  # Sentiment by market sector
    divergence: Dict[str, float]  # Sentiment divergence metrics


@dataclass
class NewsSentimentResult:
    """Data class for news-specific sentiment analysis."""
    current_sentiment: float  # Current news sentiment score (-1 to 1)
    headline_impact: List[Dict[str, Any]]  # Top impactful headlines
    recent_events: List[Dict[str, Any]]  # Recent significant events
    topic_sentiment: Dict[str, float]  # Sentiment by news topic
    source_reliability: Dict[str, float]  # Reliability score by source


@dataclass
class SocialMediaSentimentResult:
    """Data class for social media sentiment analysis."""
    current_sentiment: float  # Current social sentiment score (-1 to 1)
    platform_breakdown: Dict[str, float]  # Sentiment by platform
    influencer_impact: List[Dict[str, Any]]  # High-impact influencer content
    trending_topics: List[Dict[str, Any]]  # Sentiment of trending topics
    retail_consensus: str  # 'bullish', 'bearish', 'mixed', 'neutral'


@dataclass
class SentimentDivergenceResult:
    """Data class for sentiment divergence analysis."""
    price_sentiment_divergence: float  # -1 to 1 scale
    news_social_divergence: float  # -1 to 1 scale
    retail_institutional_divergence: float  # -1 to 1 scale
    divergence_signals: List[Dict[str, Any]]  # Actionable divergence signals


class SentimentFeatures(BaseFeature):
    """
    Sentiment analysis feature extraction class providing sophisticated sentiment-based
    insights for the QuantumSpectre Elite Trading System.
    """

    def __init__(self, ts_storage: TimeSeriesStorage):
        """
        Initialize the SentimentFeatures class.
        
        Args:
            ts_storage: TimeSeriesStorage instance for accessing historical data
        """
        super().__init__()
        self.ts_storage = ts_storage
        self.sentiment_cache = {}  # Cache for sentiment calculations
        self.sources_reliability = {  # Initial source reliability scores
            'financial_news': 0.85,
            'general_news': 0.65,
            'twitter': 0.60,
            'reddit': 0.55,
            'stocktwits': 0.65,
            'trading_forums': 0.70,
            'analyst_ratings': 0.75,
            'earnings_calls': 0.80,
            'sec_filings': 0.90,
            'company_news': 0.75,
            'blogs': 0.60,
            'telegram': 0.50,
            'discord': 0.45
        } 
        logger.info("SentimentFeatures initialized")

    async def calculate(self, symbol: str, lookback_hours: int = 24,
                         include_sources: Optional[List[str]] | None = None) -> Dict[str, float]:
        """Asynchronously compute sentiment metrics for a symbol."""
        result = self.analyze_sentiment(symbol, lookback_hours, include_sources)
        return {
            "compound": result.compound_score,
            "positive": result.positive_score,
            "negative": result.negative_score,
            "neutral": result.neutral_score,
            "confidence": result.confidence,
        }

    
    def analyze_sentiment(self, symbol: str, 
                         lookback_hours: int = 24,
                         include_sources: Optional[List[str]] = None) -> SentimentAnalysisResult:
        """
        Perform comprehensive sentiment analysis for the given symbol.
        
        Args:
            symbol: Trading symbol
            lookback_hours: Hours to look back for sentiment data
            include_sources: List of sources to include, or None for all
            
        Returns:
            SentimentAnalysisResult object with sentiment analysis
        """
        # Create cache key
        cache_key = f"{symbol}_{lookback_hours}_sentiment"
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        # Default to all sources if not specified
        if include_sources is None:
            include_sources = list(SENTIMENT_SOURCE_WEIGHTS.keys())
        
        # Get sentiment data from storage
        sentiment_data = self._get_sentiment_data(symbol, lookback_hours, include_sources)
        
        if not sentiment_data:
            logger.warning(f"No sentiment data available for {symbol}")
            return SentimentAnalysisResult(
                compound_score=0.0,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=1.0,
                sources={},
                entities={},
                confidence=0.0,
                market_impact={}
            )
        
        # Process sentiment by source
        source_sentiments = {}
        total_weight = 0
        compound_weighted_sum = 0
        positive_weighted_sum = 0
        negative_weighted_sum = 0
        neutral_weighted_sum = 0
        
        # Process each source
        for source in include_sources:
            if source in sentiment_data:
                entries = sentiment_data[source]
                
                if not entries:
                    continue
                
                # Calculate time-weighted sentiment for this source
                time_weighted_compound = 0
                time_weighted_positive = 0
                time_weighted_negative = 0
                time_weighted_neutral = 0
                total_time_weight = 0
                
                # Current time for decay calculation
                now = datetime.datetime.now()
                
                # Process all entries for this source
                for entry in entries:
                    # Calculate time decay weight (newer items weighted higher)
                    hours_ago = (now - entry['timestamp']).total_seconds() / 3600
                    
                    if hours_ago > lookback_hours:
                        continue
                    
                    time_weight = math.exp(-SENTIMENT_DECAY_FACTOR * hours_ago / lookback_hours)
                    
                    # Include reliability score in weight
                    reliability = entry.get('reliability', 0.7)
                    entry_weight = time_weight * reliability
                    
                    # Accumulate weighted sentiment
                    time_weighted_compound += entry['compound'] * entry_weight
                    time_weighted_positive += entry['positive'] * entry_weight
                    time_weighted_negative += entry['negative'] * entry_weight
                    time_weighted_neutral += entry['neutral'] * entry_weight
                    total_time_weight += entry_weight
                
                # Avoid division by zero
                if total_time_weight > 0:
                    source_compound = time_weighted_compound / total_time_weight
                    source_positive = time_weighted_positive / total_time_weight
                    source_negative = time_weighted_negative / total_time_weight
                    source_neutral = time_weighted_neutral / total_time_weight
                else:
                    source_compound = 0
                    source_positive = 0
                    source_negative = 0
                    source_neutral = 1
                
                # Store source sentiment
                source_sentiments[source] = source_compound
                
                # Apply source weight from constants
                source_weight = SENTIMENT_SOURCE_WEIGHTS.get(source, 1.0)
                
                # Update weighted sums
                compound_weighted_sum += source_compound * source_weight
                positive_weighted_sum += source_positive * source_weight
                negative_weighted_sum += source_negative * source_weight
                neutral_weighted_sum += source_neutral * source_weight
                total_weight += source_weight
        
        # Calculate final scores
        if total_weight > 0:
            compound_score = compound_weighted_sum / total_weight
            positive_score = positive_weighted_sum / total_weight
            negative_score = negative_weighted_sum / total_weight
            neutral_score = neutral_weighted_sum / total_weight
        else:
            compound_score = 0
            positive_score = 0
            negative_score = 0
            neutral_score = 1
        
        # Extract entity-specific sentiment
        entity_sentiments = self._extract_entity_sentiments(sentiment_data, include_sources)
        
        # Estimate confidence based on data quantity and quality
        confidence = self._calculate_sentiment_confidence(sentiment_data, include_sources)
        
        # Estimate market impact by timeframe
        market_impact = self._estimate_market_impact(compound_score, confidence, symbol)
        
        # Create result object
        result = SentimentAnalysisResult(
            compound_score=compound_score,
            positive_score=positive_score,
            negative_score=negative_score,
            neutral_score=neutral_score,
            sources=source_sentiments,
            entities=entity_sentiments,
            confidence=confidence,
            market_impact=market_impact
        )
        
        # Store in cache
        self.sentiment_cache[cache_key] = result
        
        return result
    
    def _get_sentiment_data(self, symbol: str, lookback_hours: int,
                           sources: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve sentiment data from storage for the given parameters.
        
        Args:
            symbol: Trading symbol
            lookback_hours: Hours to look back
            sources: List of sources to include
            
        Returns:
            Dict with source keys and lists of sentiment entries
        """
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(hours=lookback_hours)

        sentiment_data = {}

        for source in sources:
            try:
                entries = self.ts_storage.get_sentiment(
                    symbol=symbol,
                    timeframe="1h",
                    source=source,
                    start_time=start_time,
                    end_time=end_time,
                    limit=None,
                )

                if not entries:
                    entries = self._simulate_sentiment_data(symbol, source, start_time, end_time)

                sentiment_data[source] = entries
            except Exception as e:
                logger.error(f"Error retrieving sentiment data for {symbol} from {source}: {str(e)}")
        
        return sentiment_data
    
    def _simulate_sentiment_data(self, symbol: str, source: str, 
                               start_time: datetime.datetime,
                               end_time: datetime.datetime) -> List[Dict[str, Any]]:
        """
        Temporary function to simulate sentiment data retrieval.
        In production, this would be replaced with actual database queries.
        
        Args:
            symbol: Trading symbol
            source: Data source
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of sentiment data entries
        """
        # In production, this would query the actual sentiment database
        # For now, we'll generate synthetic data based on price movement
        
        # Get price data for the symbol
        try:
            # Try to get hourly data for the lookback period
            hourly_data = self.ts_storage.get_ohlcv(
                symbol, '1h', 
                int((end_time - start_time).total_seconds() / 3600) + 1
            )
            
            if hourly_data is None or len(hourly_data) == 0:
                # Fall back to daily data
                daily_data = self.ts_storage.get_ohlcv(symbol, '1d', 10)
                if daily_data is None or len(daily_data) == 0:
                    # No data available
                    return []
                
                # Use daily data
                price_data = daily_data
            else:
                # Use hourly data
                price_data = hourly_data
            
            # Generate synthetic sentiment based on price movement
            sentiments = []
            
            # Determine how many sentiment entries to generate
            # More frequent for social media, less for news
            if source in ['twitter', 'reddit', 'stocktwits', 'telegram', 'discord']:
                entries_per_day = 24  # Hourly
            elif source in ['financial_news', 'general_news', 'blogs']:
                entries_per_day = 8  # Every 3 hours
            else:
                entries_per_day = 4  # Every 6 hours
            
            # Calculate time delta between entries
            hours_total = (end_time - start_time).total_seconds() / 3600
            hours_per_entry = 24 / entries_per_day
            
            # Generate timestamps
            timestamps = []
            current_time = start_time
            while current_time <= end_time:
                timestamps.append(current_time)
                current_time += datetime.timedelta(hours=hours_per_entry)
            
            # Generate sentiment entries
            for timestamp in timestamps:
                # Find closest price data point
                price_index = None
                for i, idx in enumerate(price_data.index):
                    if isinstance(idx, pd.Timestamp):
                        data_time = idx.to_pydatetime()
                    else:
                        # If not timestamp, use current time
                        data_time = datetime.datetime.now()
                    
                    if data_time >= timestamp:
                        price_index = i
                        break
                
                if price_index is None:
                    # Use the last available data point
                    price_index = len(price_data) - 1
                
                if price_index < 0 or price_index >= len(price_data):
                    # Skip if no valid price data
                    continue
                
                # Use price movement to generate sentiment
                # Previous close if available
                prev_close = price_data['close'].iloc[price_index-1] if price_index > 0 else None
                current_close = price_data['close'].iloc[price_index]
                
                if prev_close is not None:
                    # Calculate return
                    price_return = (current_close - prev_close) / prev_close
                    
                    # Base sentiment on return
                    # Add noise to make it realistic
                    noise = np.random.normal(0, 0.2)
                    
                    # Different sources have different biases
                    source_bias = {
                        'twitter': 0.1,  # Slightly positive bias
                        'reddit': 0.0,
                        'stocktwits': 0.1,
                        'financial_news': -0.05,  # Slightly negative bias
                        'general_news': -0.1,
                        'blogs': 0.0,
                        'analyst_ratings': 0.05,
                        'trading_forums': -0.05,
                        'earnings_calls': 0.0,
                        'sec_filings': 0.0,
                        'company_news': 0.05,
                        'telegram': 0.15,
                        'discord': 0.1
                    }
                    
                    # Apply bias
                    bias = source_bias.get(source, 0.0)
                    
                    # Calculate sentiment scores
                    # Scale returns to sentiment range (-1 to 1)
                    return_sentiment = np.clip(price_return * 10, -0.8, 0.8)
                    compound = return_sentiment + noise + bias
                    compound = np.clip(compound, -1.0, 1.0)
                    
                    # Derive component scores
                    if compound > 0:
                        positive = 0.5 + compound / 2
                        negative = 0.5 - compound / 2
                        neutral = 0.2
                    else:
                        positive = 0.5 + compound / 2
                        negative = 0.5 - compound / 2
                        neutral = 0.2
                    
                    # Normalize to ensure sum is 1.0
                    total = positive + negative + neutral
                    positive /= total
                    negative /= total
                    neutral /= total
                else:
                    # No previous close, use neutral sentiment with noise
                    noise = np.random.normal(0, 0.3)
                    compound = noise
                    positive = 0.5 + noise / 2
                    negative = 0.5 - noise / 2
                    neutral = 0.2
                    
                    # Normalize
                    total = positive + negative + neutral
                    positive /= total
                    negative /= total
                    neutral /= total
                
                # Generate simulated entities mentioned
                entities = []
                if np.random.random() < 0.7:  # 70% chance to mention the symbol
                    entities.append(symbol)
                
                # Maybe mention market entities
                for entity in SENTIMENT_ENTITIES:
                    if np.random.random() < 0.2:  # 20% chance per entity
                        entities.append(entity)
                
                # Create sentiment entry
                entry = {
                    'timestamp': timestamp,
                    'compound': float(compound),
                    'positive': float(positive),
                    'negative': float(negative),
                    'neutral': float(neutral),
                    'text': self._generate_synthetic_text(symbol, compound),
                    'entities': entities,
                    'reliability': self.sources_reliability.get(source, 0.7),
                    'source': source
                }
                
                sentiments.append(entry)
            
            return sentiments
            
        except Exception as e:
            logger.error(f"Error generating simulated sentiment data: {str(e)}")
            return []
    
    def _generate_synthetic_text(self, symbol: str, sentiment: float) -> str:
        """
        Generate synthetic text for simulated sentiment data.
        
        Args:
            symbol: Trading symbol
            sentiment: Sentiment score
            
        Returns:
            Synthetic text content
        """
        # Very positive
        if sentiment > 0.6:
            templates = [
                f"Extremely bullish on ${symbol}! Price target raised significantly.",
                f"${symbol} showing incredible strength, breaking out with volume!",
                f"Major buy signal for ${symbol} - technical and fundamental alignment.",
                f"${symbol} crushing earnings expectations! Strong guidance too.",
                f"Accumulating ${symbol} at these levels. This is going much higher!"
            ]
        # Positive
        elif sentiment > 0.2:
            templates = [
                f"${symbol} looking good with steady uptrend developing.",
                f"Positive outlook for ${symbol} based on recent developments.",
                f"${symbol} appears undervalued at current levels, good entry point.",
                f"Healthy consolidation for ${symbol} before next leg up.",
                f"Adding to ${symbol} position gradually, seeing strength building."
            ]
        # Neutral
        elif sentiment > -0.2:
            templates = [
                f"${symbol} trading sideways in current range, waiting for breakout.",
                f"Monitoring ${symbol} for clearer signals before taking position.",
                f"Mixed signals on ${symbol} - some bullish, some bearish indicators.",
                f"${symbol} at key level, could go either way from here.",
                f"No strong opinion on ${symbol} at the moment, needs more data."
            ]
        # Negative
        elif sentiment > -0.6:
            templates = [
                f"${symbol} showing weakness, considering reducing position.",
                f"Bearish pattern forming on ${symbol}, use caution here.",
                f"${symbol} facing headwinds with current market conditions.",
                f"Taking profits on ${symbol}, risk/reward no longer favorable.",
                f"Concerns about ${symbol}'s ability to maintain growth."
            ]
        # Very negative
        else:
            templates = [
                f"${symbol} breaking down hard! Major support levels failing.",
                f"Avoiding ${symbol} completely, too many red flags appearing.",
                f"${symbol} earnings disaster! Guidance slashed dramatically.",
                f"Strong sell signal triggered for ${symbol}, significant downside ahead.",
                f"Bearish reversal confirmed for ${symbol}, cutting losses now."
            ]
        
        # Return random template
        return np.random.choice(templates)
    
    def _extract_entity_sentiments(self, sentiment_data: Dict[str, List[Dict[str, Any]]],
                                  include_sources: List[str]) -> Dict[str, float]:
        """
        Extract sentiment scores for specific entities mentioned in the data.
        
        Args:
            sentiment_data: Dictionary of sentiment data by source
            include_sources: List of sources to include
            
        Returns:
            Dictionary of sentiment scores by entity
        """
        entity_mentions = defaultdict(list)
        entity_weights = defaultdict(float)
        
        # Process all sources
        for source in include_sources:
            if source not in sentiment_data:
                continue
            
            # Source weight for weighting mentions
            source_weight = SENTIMENT_SOURCE_WEIGHTS.get(source, 1.0)
            
            # Process entries
            for entry in sentiment_data[source]:
                # Extract entities mentioned
                entities = entry.get('entities', [])
                
                # Entry age for time decay
                now = datetime.datetime.now()
                hours_ago = (now - entry['timestamp']).total_seconds() / 3600
                time_weight = math.exp(-SENTIMENT_DECAY_FACTOR * hours_ago / 48)  # 48-hour reference
                
                # Reliability weight
                reliability = entry.get('reliability', 0.7)
                
                # Total entry weight
                entry_weight = source_weight * time_weight * reliability
                
                # Add sentiment for each entity
                for entity in entities:
                    entity_mentions[entity].append(entry['compound'] * entry_weight)
                    entity_weights[entity] += entry_weight
        
        # Calculate average sentiment for each entity
        entity_sentiments = {}
        for entity, mentions in entity_mentions.items():
            if entity_weights[entity] > 0:
                entity_sentiments[entity] = sum(mentions) / entity_weights[entity]
        
        return entity_sentiments
    
    def _calculate_sentiment_confidence(self, sentiment_data: Dict[str, List[Dict[str, Any]]],
                                       include_sources: List[str]) -> float:
        """
        Calculate confidence score for sentiment analysis based on data quantity and quality.
        
        Args:
            sentiment_data: Dictionary of sentiment data by source
            include_sources: List of sources to include
            
        Returns:
            Confidence score from 0-1
        """
        # Base factors for confidence
        data_quantity_score = 0.0
        data_recency_score = 0.0
        data_consistency_score = 0.0
        source_diversity_score = 0.0
        
        # Count total entries and calculate average age
        total_entries = 0
        total_age_hours = 0
        all_sentiments = []
        sources_with_data = 0
        
        for source in include_sources:
            if source in sentiment_data and sentiment_data[source]:
                sources_with_data += 1
                entries = sentiment_data[source]
                total_entries += len(entries)
                
                # Calculate average age
                now = datetime.datetime.now()
                for entry in entries:
                    hours_ago = (now - entry['timestamp']).total_seconds() / 3600
                    total_age_hours += hours_ago
                    all_sentiments.append(entry['compound'])
        
        # Data quantity score (sigmoid function to cap effect of very large numbers)
        data_quantity_score = sigmoid(total_entries / 20) * 0.9  # Scale to 0-0.9
        
        # Data recency score
        if total_entries > 0:
            avg_age_hours = total_age_hours / total_entries
            data_recency_score = math.exp(-avg_age_hours / 24) * 0.9  # Scale to 0-0.9
        
        # Data consistency score (inverse of standard deviation)
        if len(all_sentiments) > 1:
            sentiment_std = np.std(all_sentiments)
            data_consistency_score = math.exp(-sentiment_std * 2) * 0.9  # Scale to 0-0.9
        
        # Source diversity score
        if len(include_sources) > 0:
            source_diversity_score = (sources_with_data / len(include_sources)) * 0.9  # Scale to 0-0.9
        
        # Weight factors
        weights = {
            'quantity': 0.3,
            'recency': 0.3,
            'consistency': 0.2,
            'diversity': 0.2
        }
        
        # Calculate final confidence
        confidence = (
            weights['quantity'] * data_quantity_score +
            weights['recency'] * data_recency_score +
            weights['consistency'] * data_consistency_score +
            weights['diversity'] * source_diversity_score
        )
        
        # Ensure confidence is between 0 and 1
        confidence = max(0.1, min(1.0, confidence))
        
        return confidence
    
    def _estimate_market_impact(self, sentiment_score: float, confidence: float,
                              symbol: str) -> Dict[str, float]:
        """
        Estimate the potential market impact of the sentiment by timeframe.
        
        Args:
            sentiment_score: Overall sentiment score
            confidence: Confidence score for the sentiment
            symbol: Trading symbol
            
        Returns:
            Dict with timeframe keys and impact values (-1 to 1 scale)
        """
        # Base timeframes
        timeframes = ['5m', '15m', '1h', '4h', '1d']
        
        # Impact decays with time (short-term impact is stronger)
        decay_factors = {
            '5m': 1.0,
            '15m': 0.9,
            '1h': 0.7,
            '4h': 0.5,
            '1d': 0.3
        }
        
        # Get historical correlation between sentiment and price movement
        # In production, this would use actual historical correlations from database
        # For now, use realistic synthetic values
        correlations = {
            '5m': 0.3,  # Weakest correlation in shortest timeframe (noise)
            '15m': 0.4,
            '1h': 0.5,
            '4h': 0.6,
            '1d': 0.7   # Strongest correlation in daily timeframe
        }
        
        # Calculate impact for each timeframe
        impact = {}
        
        for timeframe in timeframes:
            # Impact formula: sentiment * confidence * correlation * decay
            timeframe_impact = (
                sentiment_score *
                confidence *
                correlations[timeframe] *
                decay_factors[timeframe]
            )
            
            # Scale to ensure -1 to 1 range
            timeframe_impact = max(-1.0, min(1.0, timeframe_impact))
            
            impact[timeframe] = timeframe_impact
        
        return impact
    
    def analyze_sentiment_trend(self, symbol: str, 
                               days: int = 7,
                               smoothing: bool = True) -> SentimentTrendResult:
        """
        Analyze the trend of sentiment over time.
        
        Args:
            symbol: Trading symbol
            days: Number of days to analyze
            smoothing: Whether to apply smoothing to the sentiment series
            
        Returns:
            SentimentTrendResult object with trend analysis
        """
        # Create cache key
        cache_key = f"{symbol}_{days}_sentiment_trend"
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        # Calculate time periods (daily)
        end_time = datetime.datetime.now()
        
        # Get sentiment for each day
        sentiment_by_day = {}
        confidence_by_day = {}
        
        for day in range(days):
            day_start = end_time - datetime.timedelta(days=day+1)
            day_end = end_time - datetime.timedelta(days=day)
            
            # Get sentiment for this day
            sentiment_result = self._get_daily_sentiment(symbol, day_start, day_end)
            
            if sentiment_result:
                date_key = day_end.strftime("%Y-%m-%d")
                sentiment_by_day[date_key] = sentiment_result['compound']
                confidence_by_day[date_key] = sentiment_result['confidence']
        
        # Check if we have enough data
        if len(sentiment_by_day) < 3:
            logger.warning(f"Insufficient data for sentiment trend analysis: {symbol}")
            return SentimentTrendResult(
                trend_direction="insufficient_data",
                trend_strength=0.0,
                change_rate=0.0,
                reversals=[],
                sentiment_series={'dates': [], 'values': [], 'confidence': []}
            )
        
        # Sort dates
        dates = sorted(sentiment_by_day.keys())
        sentiment_values = [sentiment_by_day[date] for date in dates]
        confidence_values = [confidence_by_day[date] for date in dates]
        
        # Apply smoothing if requested
        if smoothing and len(sentiment_values) >= 5:
            # Use simple moving average
            window_size = min(5, len(sentiment_values))
            smooth_values = []
            for i in range(len(sentiment_values)):
                if i < window_size - 1:
                    # Not enough data points for full window
                    smooth_values.append(sentiment_values[i])
                else:
                    # Calculate average of window
                    window_avg = sum(sentiment_values[i-(window_size-1):i+1]) / window_size
                    smooth_values.append(window_avg)
            
            # Replace with smoothed values
            sentiment_values = smooth_values
        
        # Calculate trend
        if len(sentiment_values) >= 2:
            # Use linear regression to calculate trend
            x = np.arange(len(sentiment_values))
            slope, _, r_value, _, _ = np.polyfit(x, sentiment_values, 1, full=True)[0:5]
            
            # Determine trend direction
            if slope > 0.05:
                trend_direction = "improving"
            elif slope < -0.05:
                trend_direction = "deteriorating"
            else:
                trend_direction = "stable"
            
            # Calculate trend strength (r-squared)
            trend_strength = r_value ** 2
            
            # Calculate change rate (normalized to per-day)
            if len(sentiment_values) > 1:
                first_value = sentiment_values[0]
                last_value = sentiment_values[-1]
                total_change = last_value - first_value
                change_rate = total_change / len(sentiment_values)
            else:
                change_rate = 0.0
        else:
            trend_direction = "insufficient_data"
            trend_strength = 0.0
            change_rate = 0.0
        
        # Identify sentiment reversals
        reversals = []
        
        if len(sentiment_values) >= 3:
            for i in range(1, len(sentiment_values) - 1):
                # Check for local minimum (negative to positive shift)
                if (sentiment_values[i-1] > sentiment_values[i] and 
                    sentiment_values[i] < sentiment_values[i+1]):
                    reversals.append({
                        'type': 'positive_reversal',
                        'date': dates[i],
                        'value': sentiment_values[i],
                        'confidence': confidence_values[i]
                    })
                
                # Check for local maximum (positive to negative shift)
                if (sentiment_values[i-1] < sentiment_values[i] and 
                    sentiment_values[i] > sentiment_values[i+1]):
                    reversals.append({
                        'type': 'negative_reversal',
                        'date': dates[i],
                        'value': sentiment_values[i],
                        'confidence': confidence_values[i]
                    })
        
        # Create result
        result = SentimentTrendResult(
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            change_rate=change_rate,
            reversals=reversals,
            sentiment_series={
                'dates': dates,
                'values': sentiment_values,
                'confidence': confidence_values
            }
        )
        
        # Store in cache
        self.sentiment_cache[cache_key] = result
        
        return result
    
    def _get_daily_sentiment(self, symbol: str, 
                           start_time: datetime.datetime,
                           end_time: datetime.datetime) -> Dict[str, float]:
        """
        Calculate sentiment for a specific day.
        
        Args:
            symbol: Trading symbol
            start_time: Start of day
            end_time: End of day
            
        Returns:
            Dict with sentiment values
        """
        # Calculate hours difference
        hours_diff = (end_time - start_time).total_seconds() / 3600
        
        # Get all sentiment sources
        include_sources = list(SENTIMENT_SOURCE_WEIGHTS.keys())
        
        # Get sentiment data
        sentiment_data = self._get_sentiment_data(symbol, hours_diff, include_sources)
        
        if not sentiment_data or all(len(entries) == 0 for entries in sentiment_data.values()):
            return None
        
        # Process sentiment by source
        source_sentiments = {}
        total_weight = 0
        compound_weighted_sum = 0
        total_entries = 0
        
        # Process each source
        for source in include_sources:
            if source in sentiment_data:
                entries = sentiment_data[source]
                
                if not entries:
                    continue
                
                # Filter entries to this day's time range
                day_entries = [e for e in entries 
                              if start_time <= e['timestamp'] < end_time]
                
                if not day_entries:
                    continue
                
                total_entries += len(day_entries)
                
                # Calculate average sentiment for this source
                source_compound = sum(e['compound'] for e in day_entries) / len(day_entries)
                
                # Store source sentiment
                source_sentiments[source] = source_compound
                
                # Apply source weight from constants
                source_weight = SENTIMENT_SOURCE_WEIGHTS.get(source, 1.0)
                
                # Update weighted sums
                compound_weighted_sum += source_compound * source_weight
                total_weight += source_weight
        
        # Calculate final score
        if total_weight > 0:
            compound_score = compound_weighted_sum / total_weight
        else:
            compound_score = 0
        
        # Calculate confidence
        confidence = sigmoid(total_entries / 10) * 0.9
        
        return {
            'compound': compound_score,
            'confidence': confidence
        }
    
    def analyze_market_mood(self, include_sectors: bool = True) -> MarketMoodResult:
        """
        Analyze overall market mood and sentiment across different market segments.
        
        Args:
            include_sectors: Whether to include sector-specific sentiment
            
        Returns:
            MarketMoodResult object with market mood analysis
        """
        # Create cache key
        cache_key = f"market_mood_{include_sectors}"
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        # Calculate fear & greed index
        # This would normally use actual market data
        # For this implementation, we'll calculate synthetically
        
        # Get major indices for sentiment analysis
        indices = ['SPY', 'QQQ', 'IWM', 'VIX']  # S&P 500, Nasdaq, Russell 2000, Volatility
        
        # Get sentiment for each index
        index_sentiment = {}
        for index in indices:
            sentiment = self.analyze_sentiment(index, lookback_hours=48)
            index_sentiment[index] = sentiment.compound_score
        
        # Get recent price action for major indices
        price_momentum = {}
        for index in indices:
            try:
                # Get recent daily data
                daily_data = self.ts_storage.get_ohlcv(index, '1d', 10)
                
                if daily_data is not None and len(daily_data) >= 2:
                    # Calculate 5-day return
                    latest_close = daily_data['close'].iloc[-1]
                    five_day_ago = daily_data['close'].iloc[-min(5, len(daily_data))]
                    
                    price_momentum[index] = (latest_close - five_day_ago) / five_day_ago
                else:
                    price_momentum[index] = 0.0
            except Exception as e:
                logger.error(f"Error calculating price momentum for {index}: {str(e)}")
                price_momentum[index] = 0.0
        
        # Calculate fear & greed components
        try:
            # Price momentum component (SPY & QQQ)
            momentum_component = 50 + (
                (price_momentum.get('SPY', 0) * 200) +
                (price_momentum.get('QQQ', 0) * 200)
            ) / 2
            
            # Volatility component (VIX)
            # Higher VIX = more fear
            vix_component = 50
            if 'VIX' in price_momentum:
                vix_change = price_momentum['VIX']
                if vix_change > 0:
                    # VIX increasing = more fear
                    vix_component = max(0, 50 - (vix_change * 200))
                else:
                    # VIX decreasing = more greed
                    vix_component = min(100, 50 - (vix_change * 200))
            
            # Sentiment component
            sentiment_component = 50
            if index_sentiment:
                avg_sentiment = sum(index_sentiment.values()) / len(index_sentiment)
                sentiment_component = 50 + (avg_sentiment * 50)
            
            # Calculate combined fear & greed index
            fear_greed_index = (momentum_component * 0.4 +
                              vix_component * 0.3 +
                              sentiment_component * 0.3)
            
            # Ensure in range 0-100
            fear_greed_index = max(0, min(100, fear_greed_index))
            
        except Exception as e:
            logger.error(f"Error calculating fear & greed index: {str(e)}")
            fear_greed_index = 50  # Neutral default
        
        # Determine market regime
        if fear_greed_index >= 70:
            market_regime = "risk_on"
        elif fear_greed_index <= 30:
            market_regime = "risk_off"
        else:
            market_regime = "neutral"
        
        # Estimate institutional sentiment
        institutional_sentiment = 0.0
        
        # Use price action in liquid instruments as proxy for institutional sentiment
        try:
            # SPY, large caps, bonds
            instruments = ['SPY', 'TLT', 'HYG']
            weights = [0.6, 0.2, 0.2]  # Weighting for each instrument
            
            inst_score = 0.0
            total_weight = 0.0
            
            for i, instrument in enumerate(instruments):
                daily_data = self.ts_storage.get_ohlcv(instrument, '1d', 10)
                
                if daily_data is not None and len(daily_data) >= 5:
                    # Use volume-weighted price movement
                    recent_data = daily_data.iloc[-5:]
                    
                    # Calculate volume-weighted return
                    returns = recent_data['close'].pct_change().fillna(0)
                    volumes = recent_data['volume'] / recent_data['volume'].mean()
                    weighted_returns = returns * volumes
                    
                    # Sum weighted returns
                    total_return = weighted_returns.sum()
                    
                    # Scale to sentiment range (-1 to 1)
                    inst_component = np.clip(total_return * 20, -1, 1)
                    
                    # Apply instrument weight
                    inst_score += inst_component * weights[i]
                    total_weight += weights[i]
            
            if total_weight > 0:
                institutional_sentiment = inst_score / total_weight
        
        except Exception as e:
            logger.error(f"Error calculating institutional sentiment: {str(e)}")
            institutional_sentiment = 0.0  # Neutral default
        
        # Estimate retail sentiment
        retail_sentiment = 0.0
        
        # Use social media sentiment as proxy for retail
        try:
            retail_sources = ['twitter', 'reddit', 'stocktwits']
            retail_symbols = ['SPY', 'AAPL', 'TSLA', 'AMC', 'GME']  # Popular retail symbols
            
            retail_scores = []
            
            for symbol in retail_symbols:
                # Get sentiment limited to retail sources
                sentiment = self.analyze_sentiment(
                    symbol, lookback_hours=24, include_sources=retail_sources
                )
                
                if sentiment:
                    retail_scores.append(sentiment.compound_score)
            
            if retail_scores:
                retail_sentiment = sum(retail_scores) / len(retail_scores)
        
        except Exception as e:
            logger.error(f"Error calculating retail sentiment: {str(e)}")
            retail_sentiment = 0.0  # Neutral default
        
        # Calculate volatility sentiment
        volatility_sentiment = 0.0
        
        try:
            # Use VIX and its trend
            vix_data = self.ts_storage.get_ohlcv('VIX', '1d', 10)
            
            if vix_data is not None and len(vix_data) >= 5:
                # Current VIX level
                current_vix = vix_data['close'].iloc[-1]
                
                # VIX average over last 20 days
                vix_avg_20d = vix_data['close'].iloc[-min(10, len(vix_data)):].mean()
                
                # VIX 5-day trend
                vix_5d_ago = vix_data['close'].iloc[-min(5, len(vix_data))]
                vix_trend = (current_vix - vix_5d_ago) / vix_5d_ago
                
                # Convert to sentiment score
                # High VIX = negative sentiment, Rising VIX = negative sentiment
                vix_level_sentiment = -1 * (current_vix - 20) / 10  # 20 is neutral
                vix_trend_sentiment = -1 * vix_trend * 5
                
                # Combine with weights
                volatility_sentiment = (vix_level_sentiment * 0.7 + 
                                      vix_trend_sentiment * 0.3)
                
                # Ensure in range -1 to 1
                volatility_sentiment = np.clip(volatility_sentiment, -1, 1)
        
        except Exception as e:
            logger.error(f"Error calculating volatility sentiment: {str(e)}")
            volatility_sentiment = 0.0  # Neutral default
        
        # Calculate sector sentiment if requested
        sector_sentiment = {}
        
        if include_sectors:
            # Map of sector ETFs
            sectors = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Energy': 'XLE',
                'Consumer_Discretionary': 'XLY',
                'Consumer_Staples': 'XLP',
                'Industrials': 'XLI',
                'Materials': 'XLB',
                'Utilities': 'XLU',
                'Real_Estate': 'XLRE',
                'Communication_Services': 'XLC'
            }
            
            for sector_name, etf in sectors.items():
                try:
                    # Get sentiment for sector ETF
                    sentiment = self.analyze_sentiment(etf, lookback_hours=24)
                    sector_sentiment[sector_name] = sentiment.compound_score
                except Exception as e:
                    logger.error(f"Error analyzing sentiment for sector {sector_name}: {str(e)}")
                    sector_sentiment[sector_name] = 0.0
        
        # Calculate sentiment divergence metrics
        divergence = {}
        
        # Institutional vs retail divergence
        divergence['institutional_retail'] = institutional_sentiment - retail_sentiment
        
        # SPY sentiment vs price divergence
        try:
            spy_sentiment = index_sentiment.get('SPY', 0)
            spy_price_move = price_momentum.get('SPY', 0)
            
            # Convert price move to comparable scale
            spy_price_sentiment = np.clip(spy_price_move * 20, -1, 1)
            
            divergence['spy_sentiment_price'] = spy_sentiment - spy_price_sentiment
        except Exception:
            divergence['spy_sentiment_price'] = 0.0
        
        # Create result
        result = MarketMoodResult(
            fear_greed_index=fear_greed_index,
            market_regime=market_regime,
            institutional_sentiment=institutional_sentiment,
            retail_sentiment=retail_sentiment,
            volatility_sentiment=volatility_sentiment,
            sector_sentiment=sector_sentiment,
            divergence=divergence
        )
        
        # Store in cache
        self.sentiment_cache[cache_key] = result
        
        return result
    
    def analyze_news_sentiment(self, symbol: str, 
                              hours: int = 24) -> NewsSentimentResult:
        """
        Analyze sentiment specifically from news sources.
        
        Args:
            symbol: Trading symbol
            hours: Hours to look back
            
        Returns:
            NewsSentimentResult object with news sentiment analysis
        """
        # Create cache key
        cache_key = f"{symbol}_{hours}_news"
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        # Define news sources
        news_sources = [
            'financial_news',
            'general_news',
            'company_news',
            'blogs',
            'analyst_ratings',
            'earnings_calls',
            'sec_filings'
        ]
        
        # Get sentiment data
        sentiment_data = self._get_sentiment_data(symbol, hours, news_sources)
        
        if not sentiment_data or all(len(entries) == 0 for source, entries in sentiment_data.items()):
            logger.warning(f"No news sentiment data available for {symbol}")
            return NewsSentimentResult(
                current_sentiment=0.0,
                headline_impact=[],
                recent_events=[],
                topic_sentiment={},
                source_reliability={}
            )
        
        # Calculate current overall news sentiment
        total_sentiment = 0.0
        total_weight = 0.0
        
        # Process sources
        for source in news_sources:
            if source in sentiment_data and sentiment_data[source]:
                entries = sentiment_data[source]
                
                # Calculate time-weighted average for this source
                source_total = 0.0
                source_weight = 0.0
                
                now = datetime.datetime.now()
                
                for entry in entries:
                    # Calculate time weight
                    hours_ago = (now - entry['timestamp']).total_seconds() / 3600
                    
                    if hours_ago > hours:
                        continue
                    
                    time_weight = math.exp(-SENTIMENT_DECAY_FACTOR * hours_ago / hours)
                    source_total += entry['compound'] * time_weight
                    source_weight += time_weight
                
                # Source average
                if source_weight > 0:
                    source_avg = source_total / source_weight
                    
                    # Apply source importance weight
                    source_importance = SENTIMENT_SOURCE_WEIGHTS.get(source, 1.0)
                    total_sentiment += source_avg * source_importance
                    total_weight += source_importance
        
        # Final sentiment score
        current_sentiment = 0.0
        if total_weight > 0:
            current_sentiment = total_sentiment / total_weight
        
        # Identify high-impact headlines
        headline_impact = []
        
        # Flatten all entries across sources
        all_entries = []
        for source, entries in sentiment_data.items():
            for entry in entries:
                entry['source_type'] = source
                all_entries.append(entry)
        
        # Sort by absolute sentiment to find most impactful
        all_entries.sort(key=lambda x: abs(x['compound']), reverse=True)
        
        # Take top headlines
        for entry in all_entries[:10]:
            headline_impact.append({
                'text': entry['text'],
                'sentiment': entry['compound'],
                'timestamp': entry['timestamp'],
                'source': entry['source_type'],
                'impact': abs(entry['compound'])
            })
        
        # Identify recent significant events
        recent_events = []
        
        # Find clusters of similar sentiment at similar times
        if all_entries:
            # Group by day for now (could use more sophisticated clustering)
            day_groups = defaultdict(list)
            
            for entry in all_entries:
                day_key = entry['timestamp'].strftime('%Y-%m-%d')
                day_groups[day_key].append(entry)
            
            # For each day, check if there's a significant event
            for day, day_entries in day_groups.items():
                # If multiple sources reporting similar sentiment on same day
                if len(day_entries) >= 3:
                    # Calculate average sentiment
                    day_sentiment = sum(e['compound'] for e in day_entries) / len(day_entries)
                    
                    # If sentiment is strong enough to be significant
                    if abs(day_sentiment) > 0.4:
                        # Get most representative text
                        # Find entry with sentiment closest to average
                        closest_entry = min(day_entries, 
                                          key=lambda x: abs(x['compound'] - day_sentiment))
                        
                        recent_events.append({
                            'date': day,
                            'sentiment': day_sentiment,
                            'text': closest_entry['text'],
                            'sources': len(set(e['source_type'] for e in day_entries)),
                            'entries': len(day_entries)
                        })
        
        # Sort events by date (most recent first)
        recent_events.sort(key=lambda x: x['date'], reverse=True)
        
        # Calculate sentiment by topic
        topic_sentiment = self._extract_topic_sentiment(all_entries)
        
        # Calculate source reliability
        source_reliability = {}
        
        for source in news_sources:
            if source in sentiment_data and sentiment_data[source]:
                # Use predefined reliability scores
                source_reliability[source] = self.sources_reliability.get(source, 0.7)
        
        # Create result
        result = NewsSentimentResult(
            current_sentiment=current_sentiment,
            headline_impact=headline_impact,
            recent_events=recent_events,
            topic_sentiment=topic_sentiment,
            source_reliability=source_reliability
        )
        
        # Store in cache
        self.sentiment_cache[cache_key] = result
        
        return result
    
    def _extract_topic_sentiment(self, entries: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Extract sentiment scores by topic from text content.
        
        Args:
            entries: List of sentiment entries with text content
            
        Returns:
            Dict with topic keys and sentiment values
        """
        # Define common topics and their keywords
        topics = {
            'earnings': ['earnings', 'revenue', 'profit', 'eps', 'quarter', 'guidance', 'forecast'],
            'products': ['product', 'launch', 'release', 'announced', 'new', 'innovation'],
            'management': ['ceo', 'executive', 'management', 'leadership', 'board', 'director'],
            'regulation': ['regulation', 'compliance', 'legal', 'lawsuit', 'sec', 'regulatory'],
            'competition': ['competitor', 'market share', 'industry', 'rival'],
            'economy': ['economy', 'economic', 'inflation', 'recession', 'gdp', 'federal reserve', 'fed'],
            'partnerships': ['partnership', 'agreement', 'deal', 'collaboration', 'alliance'],
            'technology': ['technology', 'tech', 'innovation', 'software', 'hardware']
        }
        
        # Initialize topic sentiment counts and sums
        topic_counts = defaultdict(int)
        topic_sentiments = defaultdict(float)
        
        # Process each entry
        for entry in entries:
            text = entry['text'].lower()
            sentiment = entry['compound']
            
            # Check each topic
            for topic, keywords in topics.items():
                # Check if any keyword is in the text
                if any(keyword.lower() in text for keyword in keywords):
                    topic_sentiments[topic] += sentiment
                    topic_counts[topic] += 1
        
        # Calculate average sentiment per topic
        topic_avg_sentiment = {}
        for topic, count in topic_counts.items():
            if count > 0:
                topic_avg_sentiment[topic] = topic_sentiments[topic] / count
        
        return topic_avg_sentiment
    
    def analyze_social_sentiment(self, symbol: str,
                               hours: int = 24) -> SocialMediaSentimentResult:
        """
        Analyze sentiment specifically from social media sources.
        
        Args:
            symbol: Trading symbol
            hours: Hours to look back
            
        Returns:
            SocialMediaSentimentResult object with social media sentiment analysis
        """
        # Create cache key
        cache_key = f"{symbol}_{hours}_social"
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        # Define social media sources
        social_sources = [
            'twitter',
            'reddit',
            'stocktwits',
            'telegram',
            'discord',
            'trading_forums'
        ]
        
        # Get sentiment data
        sentiment_data = self._get_sentiment_data(symbol, hours, social_sources)
        
        if not sentiment_data or all(len(entries) == 0 for source, entries in sentiment_data.items()):
            logger.warning(f"No social sentiment data available for {symbol}")
            return SocialMediaSentimentResult(
                current_sentiment=0.0,
                platform_breakdown={},
                influencer_impact=[],
                trending_topics=[],
                retail_consensus="neutral"
            )
        
        # Calculate current overall social sentiment
        total_sentiment = 0.0
        total_weight = 0.0
        
        # Platform breakdown
        platform_breakdown = {}
        
        # Process sources
        for source in social_sources:
            if source in sentiment_data and sentiment_data[source]:
                entries = sentiment_data[source]
                
                # Calculate time-weighted average for this source
                source_total = 0.0
                source_weight = 0.0
                
                now = datetime.datetime.now()
                
                for entry in entries:
                    # Calculate time weight
                    hours_ago = (now - entry['timestamp']).total_seconds() / 3600
                    
                    if hours_ago > hours:
                        continue
                    
                    time_weight = math.exp(-SENTIMENT_DECAY_FACTOR * hours_ago / hours)
                    source_total += entry['compound'] * time_weight
                    source_weight += time_weight
                
                # Source average
                if source_weight > 0:
                    source_avg = source_total / source_weight
                    platform_breakdown[source] = source_avg
                    
                    # Apply source importance weight
                    source_importance = SENTIMENT_SOURCE_WEIGHTS.get(source, 1.0)
                    total_sentiment += source_avg * source_importance
                    total_weight += source_importance
        
        # Final sentiment score
        current_sentiment = 0.0
        if total_weight > 0:
            current_sentiment = total_sentiment / total_weight
        
        # Identify high-impact influencer content
        influencer_impact = []
        
        # Flatten all entries across sources
        all_entries = []
        for source, entries in sentiment_data.items():
            for entry in entries:
                entry['source_type'] = source
                all_entries.append(entry)
        
        # Sort by absolute sentiment to find most impactful
        all_entries.sort(key=lambda x: abs(x['compound']), reverse=True)
        
        # Take top influencer posts
        for entry in all_entries[:10]:
            influencer_impact.append({
                'text': entry['text'],
                'sentiment': entry['compound'],
                'timestamp': entry['timestamp'],
                'platform': entry['source_type'],
                'impact': abs(entry['compound'])
            })
        
        # Identify trending topics
        trending_topics = self._extract_trending_topics(all_entries)
        
        # Determine retail consensus
        if current_sentiment > 0.3:
            retail_consensus = "bullish"
        elif current_sentiment < -0.3:
            retail_consensus = "bearish"
        elif abs(current_sentiment) <= 0.1:
            retail_consensus = "neutral"
        else:
            retail_consensus = "mixed"
        
        # Create result
        result = SocialMediaSentimentResult(
            current_sentiment=current_sentiment,
            platform_breakdown=platform_breakdown,
            influencer_impact=influencer_impact,
            trending_topics=trending_topics,
            retail_consensus=retail_consensus
        )
        
        # Store in cache
        self.sentiment_cache[cache_key] = result
        
        return result
    
    def _extract_trending_topics(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract trending topics from social media entries.
        
        Args:
            entries: List of sentiment entries
            
        Returns:
            List of trending topics with sentiment
        """
        # This would normally use NLP techniques for topic extraction
        # For this implementation, we'll use a simple keyword-based approach
        
        # Count keyword occurrences
        keyword_counts = defaultdict(int)
        keyword_sentiment = defaultdict(list)
        
        # Extract keywords from text
        for entry in entries:
            text = entry['text'].lower()
            sentiment = entry['compound']
            
            # Extract cashtags like $AAPL
            cashtags = re.findall(r'\$([A-Za-z]+)', text)
            for cashtag in cashtags:
                keyword_counts[cashtag] += 1
                keyword_sentiment[cashtag].append(sentiment)
            
            # Extract common trading terms
            trading_terms = [
                'bullish', 'bearish', 'long', 'short', 'call', 'put',
                'support', 'resistance', 'breakout', 'breakdown',
                'squeeze', 'oversold', 'overbought', 'buy', 'sell'
            ]
            
            for term in trading_terms:
                if term in text:
                    keyword_counts[term] += 1
                    keyword_sentiment[term].append(sentiment)
            
            # Extract common action phrases
            action_phrases = [
                'going up', 'going down', 'to the moon', 'buying more',
                'selling now', 'taking profits', 'cutting losses',
                'holding strong', 'adding position', 'all in'
            ]
            
            for phrase in action_phrases:
                if phrase in text:
                    keyword_counts[phrase] += 1
                    keyword_sentiment[phrase].append(sentiment)
        
        # Create trending topics list
        trending = []
        
        for keyword, count in keyword_counts.items():
            if count >= 3:  # Only include if mentioned multiple times
                # Calculate average sentiment
                avg_sentiment = sum(keyword_sentiment[keyword]) / len(keyword_sentiment[keyword])
                
                trending.append({
                    'topic': keyword,
                    'mentions': count,
                    'sentiment': avg_sentiment
                })
        
        # Sort by mention count
        trending.sort(key=lambda x: x['mentions'], reverse=True)
        
        # Return top trending topics
        return trending[:10]
    
    def analyze_sentiment_divergence(self, symbol: str) -> SentimentDivergenceResult:
        """
        Analyze divergences between sentiment and price, or between different sentiment sources.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            SentimentDivergenceResult object with divergence analysis
        """
        # Create cache key
        cache_key = f"{symbol}_divergence"
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        # Get price data
        price_data = self.ts_storage.get_ohlcv(symbol, '1d', 20)
        
        if price_data is None or len(price_data) < 5:
            logger.warning(f"Insufficient price data for divergence analysis: {symbol}")
            return SentimentDivergenceResult(
                price_sentiment_divergence=0.0,
                news_social_divergence=0.0,
                retail_institutional_divergence=0.0,
                divergence_signals=[]
            )
        
        # Get various sentiment data
        overall_sentiment = self.analyze_sentiment(symbol, lookback_hours=48)
        news_sentiment = self.analyze_news_sentiment(symbol, hours=48)
        social_sentiment = self.analyze_social_sentiment(symbol, hours=48)
        
        # Get sentiment trend
        sentiment_trend = self.analyze_sentiment_trend(symbol, days=7)
        
        # Calculate price trend
        price_returns = price_data['close'].pct_change().fillna(0)
        cumulative_return = (1 + price_returns).cumprod() - 1
        price_trend = cumulative_return.iloc[-1]
        
        # Normalize to -1 to 1 scale
        price_trend_normalized = np.clip(price_trend * 5, -1, 1)
        
        # Calculate price-sentiment divergence
        price_sentiment_divergence = 0.0
        
        if overall_sentiment and sentiment_trend.trend_direction != "insufficient_data":
            # Current sentiment vs price trend
            sentiment_score = overall_sentiment.compound_score
            
            # Calculate divergence
            price_sentiment_divergence = sentiment_score - price_trend_normalized
        
        # Calculate news-social divergence
        news_social_divergence = 0.0
        
        if news_sentiment and social_sentiment:
            news_score = news_sentiment.current_sentiment
            social_score = social_sentiment.current_sentiment
            
            news_social_divergence = news_score - social_score
        
        # Calculate retail-institutional divergence
        # Use social media as retail proxy and news/analyst as institutional proxy
        retail_institutional_divergence = 0.0
        
        if social_sentiment and news_sentiment:
            retail_score = social_sentiment.current_sentiment
            
            # Weight news sources to represent institutional view
            institutional_sources = {
                'financial_news': 0.3,
                'analyst_ratings': 0.4,
                'earnings_calls': 0.2,
                'sec_filings': 0.1
            }
            
            # Calculate institutional score
            inst_score = 0.0
            total_weight = 0.0
            
            for source, weight in institutional_sources.items():
                if source in news_sentiment.source_reliability:
                    # Use source sentiment from news sentiment result
                    platform_scores = {}
                    for src, entries in sentiment_data.items():
                        if src == source and entries:
                            avg_score = sum(entry['compound'] for entry in entries) / len(entries)
                            platform_scores[src] = avg_score
                    
                    if source in platform_scores:
                        inst_score += platform_scores[source] * weight
                        total_weight += weight
            
            if total_weight > 0:
                inst_score = inst_score / total_weight
                retail_institutional_divergence = retail_score - inst_score
        
        # Identify divergence signals
        divergence_signals = []
        
        # Price-sentiment divergence signal
        if abs(price_sentiment_divergence) > 0.5:
            signal_type = "bullish" if price_sentiment_divergence > 0 else "bearish"
            
            divergence_signals.append({
                'type': f"{signal_type}_price_sentiment_divergence",
                'strength': abs(price_sentiment_divergence),
                'description': (
                    f"Sentiment is {signal_type} while price action is not reflecting this. "
                    f"This suggests potential {signal_type} reversal."
                )
            })
        
        # News-social divergence signal
        if abs(news_social_divergence) > 0.5:
            if news_social_divergence > 0:
                signal_type = "institutional_bullish_retail_bearish"
                description = (
                    "News sources are more bullish than social media. "
                    "This suggests institutional optimism not yet reflected in retail sentiment."
                )
            else:
                signal_type = "retail_bullish_institutional_bearish"
                description = (
                    "Social media is more bullish than news sources. "
                    "This suggests retail optimism not supported by institutional sentiment."
                )
            
            divergence_signals.append({
                'type': signal_type,
                'strength': abs(news_social_divergence),
                'description': description
            })
        
        # Check for sentiment reversals
        if sentiment_trend and sentiment_trend.reversals:
            for reversal in sentiment_trend.reversals:
                if reversal['type'] == 'positive_reversal':
                    divergence_signals.append({
                        'type': 'sentiment_positive_reversal',
                        'date': reversal['date'],
                        'strength': abs(reversal['value']) + 0.2,
                        'description': "Sentiment has reversed from negative to positive."
                    })
                elif reversal['type'] == 'negative_reversal':
                    divergence_signals.append({
                        'type': 'sentiment_negative_reversal',
                        'date': reversal['date'],
                        'strength': abs(reversal['value']) + 0.2,
                        'description': "Sentiment has reversed from positive to negative."
                    })
        
        # Create result
        result = SentimentDivergenceResult(
            price_sentiment_divergence=price_sentiment_divergence,
            news_social_divergence=news_social_divergence,
            retail_institutional_divergence=retail_institutional_divergence,
            divergence_signals=divergence_signals
        )
        
        # Store in cache
        self.sentiment_cache[cache_key] = result
        
        return result
    
    def get_features(self, symbol: str, 
                    include_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate and return requested sentiment features.
        
        Args:
            symbol: Trading symbol
            include_features: List of features to include, or None for all
            
        Returns:
            Dict with feature name keys and calculated values
        """
        # Default to all features if not specified
        if include_features is None:
            include_features = [
                'sentiment', 'trend', 'news', 'social', 'divergence'
            ]
        
        features = {}
        
        # Calculate requested features
        for feature in include_features:
            if feature == 'sentiment':
                features['sentiment'] = self.analyze_sentiment(symbol)
            elif feature == 'trend':
                features['trend'] = self.analyze_sentiment_trend(symbol)
            elif feature == 'news':
                features['news'] = self.analyze_news_sentiment(symbol)
            elif feature == 'social':
                features['social'] = self.analyze_social_sentiment(symbol)
            elif feature == 'divergence':
                features['divergence'] = self.analyze_sentiment_divergence(symbol)
            elif feature == 'market_mood':
                features['market_mood'] = self.analyze_market_mood()
        
        return features
    
    def clear_cache(self):
        """Clear the sentiment calculation cache."""
        self.sentiment_cache.clear()
        logger.info("Sentiment features cache cleared")

