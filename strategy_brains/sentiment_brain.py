
#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Sentiment Brain Strategy Implementation

This module implements a trading strategy based on market sentiment analysis,
integrating news, social media, and other sentiment indicators to identify
trading opportunities driven by market psychology and crowd behavior.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import asyncio
from dataclasses import dataclass
import json

from common.utils import sigmoid, detect_market_condition, weighted_average
from common.logger import get_logger
from common.exceptions import StrategyError, SentimentAnalysisError

from strategy_brains.base_brain import BaseBrain, Signal, SignalStrength, SignalType
from data_feeds.news_feed import NewsFeed
from data_feeds.social_feed import SocialFeed
from feature_service.features.sentiment import SentimentFeatures

logger = get_logger(__name__)


@dataclass
class SentimentSource:
    """Data class for tracking sentiment sources and their reliability."""
    name: str
    weight: float  # How much to weight this source (0.0-1.0)
    lookback_period: timedelta  # How far back to look for sentiment
    reliability: float  # How reliable this source has been historically (0.0-1.0)
    update_frequency: timedelta  # How often this source updates


@dataclass
class SentimentData:
    """Data class for storing sentiment analysis results."""
    score: float  # -1.0 to 1.0 (bearish to bullish)
    confidence: float  # 0.0 to 1.0
    source: str
    timestamp: datetime
    topics: List[str]
    relevance: float  # How relevant to the asset (0.0-1.0)
    impact_duration: timedelta  # How long this sentiment might affect the market


class SentimentBrain(BaseBrain):
    """
    Advanced sentiment-based trading strategy utilizing multiple data sources to
    assess market sentiment and generate trading signals based on sentiment shifts,
    anomalies, and trend confirmation.
    """
    
    NAME = "sentiment_brain"
    VERSION = "2.0.0"
    DESCRIPTION = "Advanced sentiment analysis trading strategy"
    SUPPORTED_TIMEFRAMES = ["15m", "30m", "1h", "4h", "1d"]
    SUPPORTED_ASSETS = ["*"]  # Supports all assets
    MIN_HISTORY_CANDLES = 100
    
    def __init__(self, 
                 config: Dict[str, Any],
                 asset_id: str,
                 exchange_id: str,
                 timeframe: str = "1h"):
        """
        Initialize the SentimentBrain strategy.
        
        Args:
            config: Strategy configuration parameters
            asset_id: Identifier for the trading asset
            exchange_id: Identifier for the exchange
            timeframe: Primary trading timeframe
        """
        super().__init__(config, asset_id, exchange_id, timeframe)
        
        # Initialize sentiment data sources
        self.news_feed = NewsFeed(self.config.get("news_feed", {}))
        self.social_feed = SocialFeed(self.config.get("social_feed", {}))
        self.sentiment_features = SentimentFeatures()
        
        # Configure strategy parameters
        self.sentiment_threshold = self.config.get("sentiment_threshold", 0.3)  # Min sentiment to generate signal
        self.confidence_threshold = self.config.get("confidence_threshold", 0.65)  # Min confidence required
        self.news_weight = self.config.get("news_weight", 0.4)
        self.social_weight = self.config.get("social_weight", 0.3)
        self.market_data_weight = self.config.get("market_data_weight", 0.3)
        self.sentiment_lookback = self.config.get("sentiment_lookback_hours", 24)
        
        # Configure sentiment sources and their weights
        self.sentiment_sources = [
            SentimentSource(
                name="financial_news", 
                weight=0.25, 
                lookback_period=timedelta(hours=12),
                reliability=0.75,
                update_frequency=timedelta(minutes=15)
            ),
            SentimentSource(
                name="twitter", 
                weight=0.15, 
                lookback_period=timedelta(hours=6),
                reliability=0.65,
                update_frequency=timedelta(minutes=5)
            ),
            SentimentSource(
                name="reddit", 
                weight=0.10, 
                lookback_period=timedelta(hours=12),
                reliability=0.60,
                update_frequency=timedelta(minutes=10)
            ),
            SentimentSource(
                name="trading_forums", 
                weight=0.10, 
                lookback_period=timedelta(hours=24),
                reliability=0.70,
                update_frequency=timedelta(hours=1)
            ),
            SentimentSource(
                name="analyst_ratings", 
                weight=0.20, 
                lookback_period=timedelta(days=7),
                reliability=0.80,
                update_frequency=timedelta(hours=4)
            ),
            SentimentSource(
                name="market_data", 
                weight=0.20, 
                lookback_period=timedelta(hours=4),
                reliability=0.85,
                update_frequency=timedelta(minutes=15)
            ),
        ]
        
        # Normalize weights
        total_weight = sum(source.weight for source in self.sentiment_sources)
        for source in self.sentiment_sources:
            source.weight = source.weight / total_weight
        
        # Initialize sentiment caches
        self.sentiment_cache = {}
        self.historical_sentiment = []
        self.sentiment_change_threshold = self.config.get("sentiment_change_threshold", 0.15)
        
        # Topic relevance scores (how relevant different news/social topics are to this asset)
        self.topic_relevance = self._initialize_topic_relevance()
        
        # Initialize sentiment shift detection
        self.sentiment_baseline = 0.0
        self.sentiment_volatility = 0.0
        
        logger.info(f"SentimentBrain initialized for {asset_id} on {exchange_id}, timeframe: {timeframe}")

    async def analyze(self, data: Dict[str, pd.DataFrame]) -> Signal:
        """
        Analyze market data and sentiment sources to generate trading signals.
        
        Args:
            data: Dictionary of DataFrames containing OHLCV data for multiple timeframes
            
        Returns:
            Signal object containing the trading signal details
        """
        try:
            primary_data = data.get(self.timeframe)
            if primary_data is None or len(primary_data) < self.MIN_HISTORY_CANDLES:
                return Signal(
                    brain_id=self.id,
                    signal_type=SignalType.NEUTRAL,
                    strength=SignalStrength.WEAK,
                    metadata={"error": "Insufficient data for sentiment analysis"}
                )
            
            # Detect market condition to adapt analysis
            market_condition = detect_market_condition(primary_data)
            logger.debug(f"Market condition detected: {market_condition}")
            
            # Gather sentiment data from various sources
            sentiment_data = await self._gather_sentiment_data()
            
            # Calculate overall sentiment score and confidence
            sentiment_score, confidence, sentiment_metadata = self._calculate_sentiment(sentiment_data, primary_data)
            
            # Detect significant sentiment shifts
            sentiment_shift = self._detect_sentiment_shift(sentiment_score)
            
            # Determine signal based on sentiment analysis
            signal_type, signal_strength = self._determine_signal(
                sentiment_score, 
                confidence, 
                sentiment_shift,
                market_condition
            )
            
            # Create signal with detailed metadata
            signal = Signal(
                brain_id=self.id,
                signal_type=signal_type,
                strength=signal_strength,
                metadata={
                    "sentiment_score": sentiment_score,
                    "confidence": confidence,
                    "sentiment_shift": sentiment_shift,
                    "market_condition": market_condition,
                    "sources_count": len(sentiment_data),
                    "dominant_source": sentiment_metadata.get("dominant_source"),
                    "top_topics": sentiment_metadata.get("top_topics", []),
                    "most_relevant_news": sentiment_metadata.get("most_relevant_news", None)
                }
            )
            
            # Update historical sentiment tracking
            self._update_sentiment_history(sentiment_score, confidence)
            
            # Log sentiment analysis details
            logger.info(
                f"SentimentBrain signal: {signal_type} with strength {signal_strength}. "
                f"Sentiment score: {sentiment_score:.2f}, Confidence: {confidence:.2f}"
            )
            
            return signal
            
        except SentimentAnalysisError as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return Signal(
                brain_id=self.id,
                signal_type=SignalType.NEUTRAL,
                strength=SignalStrength.WEAK,
                metadata={"error": str(e)}
            )
        except Exception as e:
            logger.exception(f"Unexpected error in SentimentBrain analysis: {str(e)}")
            return Signal(
                brain_id=self.id,
                signal_type=SignalType.NEUTRAL,
                strength=SignalStrength.WEAK,
                metadata={"error": f"Unexpected error: {str(e)}"}
            )

    async def _gather_sentiment_data(self) -> List[SentimentData]:
        """
        Gather sentiment data from various sources.
        
        Returns:
            List of SentimentData objects from different sources
        """
        sentiment_data = []
        cutoff_time = datetime.now() - timedelta(hours=self.sentiment_lookback)
        
        # Check if we have cached data that's still fresh
        current_time = datetime.now()
        cache_expired = False
        
        for source in self.sentiment_sources:
            cache_key = f"{source.name}_{self.asset_id}"
            
            if cache_key in self.sentiment_cache:
                cached_data = self.sentiment_cache[cache_key]
                if current_time - cached_data["timestamp"] < source.update_frequency:
                    # Use cached data
                    sentiment_data.extend(cached_data["data"])
                    continue
            
            cache_expired = True
        
        # If all cache is fresh, return the cached data
        if not cache_expired and sentiment_data:
            return sentiment_data
            
        # Gather new data from news feed
        try:
            news_items = await self.news_feed.get_news_for_asset(
                self.asset_id, 
                limit=50, 
                since=cutoff_time
            )
            
            for news in news_items:
                # Filter by relevance
                relevance = self._calculate_news_relevance(news)
                if relevance < 0.3:  # Skip low relevance news
                    continue
                    
                topics = news.get("topics", [])
                
                sentiment_data.append(SentimentData(
                    score=news["sentiment_score"],
                    confidence=news["sentiment_confidence"],
                    source=f"news_{news['source']}",
                    timestamp=news["timestamp"],
                    topics=topics,
                    relevance=relevance,
                    impact_duration=self._estimate_news_impact_duration(news, topics)
                ))
                
            # Cache this data
            self.sentiment_cache["financial_news_" + self.asset_id] = {
                "timestamp": current_time,
                "data": [sd for sd in sentiment_data if "news_" in sd.source]
            }
                
        except Exception as e:
            logger.error(f"Error gathering news sentiment: {str(e)}")
        
        # Gather new data from social feed
        try:
            social_items = await self.social_feed.get_social_sentiment(
                self.asset_id, 
                platforms=["twitter", "reddit", "trading_forums"],
                limit=100, 
                since=cutoff_time
            )
            
            for item in social_items:
                # Filter by relevance
                relevance = self._calculate_social_relevance(item)
                if relevance < 0.2:  # Social has lower relevance threshold
                    continue
                    
                topics = item.get("topics", [])
                platform = item.get("platform", "unknown")
                
                sentiment_data.append(SentimentData(
                    score=item["sentiment_score"],
                    confidence=item["sentiment_confidence"],
                    source=f"social_{platform}",
                    timestamp=item["timestamp"],
                    topics=topics,
                    relevance=relevance,
                    impact_duration=self._estimate_social_impact_duration(item, platform)
                ))
                
            # Cache by platform
            for platform in ["twitter", "reddit", "trading_forums"]:
                platform_data = [sd for sd in sentiment_data if f"social_{platform}" == sd.source]
                self.sentiment_cache[f"{platform}_{self.asset_id}"] = {
                    "timestamp": current_time,
                    "data": platform_data
                }
                
        except Exception as e:
            logger.error(f"Error gathering social sentiment: {str(e)}")
            
        # Get analyst ratings if available
        try:
            ratings = await self.news_feed.get_analyst_ratings(
                self.asset_id,
                limit=20,
                since=cutoff_time
            )
            
            for rating in ratings:
                sentiment_score = self._convert_analyst_rating_to_sentiment(rating)
                
                sentiment_data.append(SentimentData(
                    score=sentiment_score,
                    confidence=0.8,  # Analyst ratings typically have high confidence
                    source="analyst_ratings",
                    timestamp=rating["timestamp"],
                    topics=["analyst_rating", rating.get("firm", "unknown")],
                    relevance=0.9,  # Analyst ratings are highly relevant
                    impact_duration=timedelta(days=3)  # Ratings typically impact for several days
                ))
                
            # Cache analyst ratings
            self.sentiment_cache[f"analyst_ratings_{self.asset_id}"] = {
                "timestamp": current_time,
                "data": [sd for sd in sentiment_data if sd.source == "analyst_ratings"]
            }
                
        except Exception as e:
            logger.error(f"Error gathering analyst ratings: {str(e)}")
        
        logger.info(f"Gathered {len(sentiment_data)} sentiment data items from various sources")
        return sentiment_data

    def _calculate_sentiment(self, 
                           sentiment_data: List[SentimentData], 
                           market_data: pd.DataFrame) -> Tuple[float, float, Dict[str, Any]]:
        """
        Calculate overall sentiment score and confidence from multiple sources.
        
        Args:
            sentiment_data: List of sentiment data points from different sources
            market_data: Market price data for technical sentiment indicators
            
        Returns:
            Tuple of (sentiment_score, confidence, metadata)
        """
        if not sentiment_data:
            # Use market data based sentiment only
            market_sentiment = self._calculate_market_data_sentiment(market_data)
            return market_sentiment, 0.6, {"dominant_source": "market_data"}
        
        # Group by source
        source_sentiments = {}
        for data in sentiment_data:
            if data.source not in source_sentiments:
                source_sentiments[data.source] = []
            source_sentiments[data.source].append(data)
        
        # Calculate sentiment by source
        source_scores = {}
        source_confidences = {}
        
        for source_name, source_data in source_sentiments.items():
            # Weight by recency, relevance and confidence
            weights = []
            scores = []
            confidences = []
            
            for data in source_data:
                # Age factor - more recent data gets higher weight
                age_hours = (datetime.now() - data.timestamp).total_seconds() / 3600
                age_factor = np.exp(-0.05 * age_hours)  # Exponential decay with time
                
                # Combined weight
                weight = data.relevance * data.confidence * age_factor
                
                weights.append(weight)
                scores.append(data.score)
                confidences.append(data.confidence)
            
            # Calculate weighted average for this source
            if weights:
                source_scores[source_name] = np.average(scores, weights=weights)
                source_confidences[source_name] = np.average(confidences, weights=weights)
        
        # Add market data sentiment
        market_sentiment, market_confidence = self._calculate_market_data_sentiment(market_data)
        source_scores["market_data"] = market_sentiment
        source_confidences["market_data"] = market_confidence
        
        # Prepare weighted sentiment calculation across sources
        final_weights = []
        final_scores = []
        final_confidences = []
        
        for source in self.sentiment_sources:
            if source.name in source_scores:
                # Adjust weight by source reliability
                adjusted_weight = source.weight * source.reliability
                
                final_weights.append(adjusted_weight)
                final_scores.append(source_scores[source.name])
                final_confidences.append(source_confidences.get(source.name, 0.5))
        
        # Calculate final sentiment score and confidence
        if not final_weights:
            return 0.0, 0.0, {}
            
        # Normalize weights
        final_weights = np.array(final_weights) / sum(final_weights)
        
        # Calculate weighted sentiment score
        sentiment_score = np.average(final_scores, weights=final_weights)
        
        # Calculate confidence as weighted average of source confidences,
        # but also consider agreement between sources
        source_confidence = np.average(final_confidences, weights=final_weights)
        
        # Measure agreement between sources (lower standard deviation = higher agreement)
        if len(final_scores) > 1:
            agreement_factor = 1.0 - min(np.std(final_scores), 1.0)
        else:
            agreement_factor = 0.5
        
        # Combine source confidence with agreement factor
        confidence = 0.7 * source_confidence + 0.3 * agreement_factor
        
        # Prepare metadata about this sentiment calculation
        metadata = {
            "dominant_source": max(source_scores.items(), key=lambda x: abs(x[1]))[0],
            "source_scores": source_scores,
            "top_topics": self._extract_top_topics(sentiment_data),
            "most_relevant_news": self._extract_most_relevant_news(sentiment_data)
        }
        
        return sentiment_score, confidence, metadata

    def _calculate_market_data_sentiment(self, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate sentiment score from market data indicators.
        
        Args:
            data: Market OHLCV data
            
        Returns:
            Tuple of (sentiment_score, confidence)
        """
        # Use pre-calculated sentiment features if available
        if 'sentiment_score' in data.columns and 'sentiment_confidence' in data.columns:
            return data['sentiment_score'].iloc[-1], data['sentiment_confidence'].iloc[-1]
        
        # Calculate using sentiment features
        try:
            # Generate sentiment features
            features = self.sentiment_features.calculate_sentiment_features(data)
            
            # Extract sentiment indicators
            rsi = features.get('rsi_14', 50)
            macd_hist = features.get('macd_histogram', 0)
            bb_position = features.get('bbands_position', 0.5)
            
            # More indicators
            volume_trend = features.get('volume_trend', 0)
            price_trend = features.get('price_trend', 0)
            trend_strength = features.get('trend_strength', 0.5)
            
            # Convert RSI to -1 to 1 scale
            rsi_sentiment = (rsi - 50) / 50
            
            # Normalize MACD histogram
            macd_max = np.max(np.abs(data['macd_histogram'].values[-20:])) if 'macd_histogram' in data.columns else 1
            macd_sentiment = macd_hist / (macd_max if macd_max > 0 else 1)
            
            # BB position already on 0-1 scale, convert to -1 to 1
            bb_sentiment = (bb_position - 0.5) * 2
            
            # Combine indicators with different weights
            sentiment_score = (
                0.3 * rsi_sentiment +
                0.3 * macd_sentiment +
                0.2 * bb_sentiment +
                0.1 * volume_trend +
                0.1 * price_trend
            )
            
            # Clip to -1 to 1 range
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            # Confidence based on trend strength
            confidence = 0.5 + (0.5 * trend_strength)
            
            return sentiment_score, confidence
            
        except Exception as e:
            logger.error(f"Error calculating market data sentiment: {str(e)}")
            return 0.0, 0.5  # Neutral sentiment with moderate confidence

    def _calculate_news_relevance(self, news_item: Dict[str, Any]) -> float:
        """
        Calculate how relevant a news item is to the current asset.
        
        Args:
            news_item: Dictionary containing news details
            
        Returns:
            Relevance score from 0.0 to 1.0
        """
        relevance = 0.0
        
        # Check if asset is directly mentioned
        if self.asset_id.lower() in news_item.get("title", "").lower() or \
           self.asset_id.lower() in news_item.get("summary", "").lower():
            relevance += 0.6
        
        # Check topics
        for topic in news_item.get("topics", []):
            if topic in self.topic_relevance:
                relevance += self.topic_relevance[topic]
        
        # Check source credibility
        source = news_item.get("source", "unknown")
        if source in ["bloomberg", "reuters", "wsj", "financial_times"]:
            relevance += 0.1
        
        # Cap at 1.0
        return min(1.0, relevance)

    def _calculate_social_relevance(self, social_item: Dict[str, Any]) -> float:
        """
        Calculate how relevant a social media item is to the current asset.
        
        Args:
            social_item: Dictionary containing social media post details
            
        Returns:
            Relevance score from 0.0 to 1.0
        """
        relevance = 0.0
        
        # Check if asset is directly mentioned
        if self.asset_id.lower() in social_item.get("text", "").lower():
            relevance += 0.4
        
        # Check topics
        for topic in social_item.get("topics", []):
            if topic in self.topic_relevance:
                relevance += self.topic_relevance[topic] * 0.7  # Social topics lower weight than news
        
        # Check user credibility
        if social_item.get("user_credibility", 0) > 0.7:
            relevance += 0.1
            
        # Check engagement - higher engagement = higher relevance
        engagement = social_item.get("engagement", 0)
        if engagement > 100:
            relevance += 0.05 * min(engagement / 1000, 1.0)
        
        # Cap at 1.0
        return min(1.0, relevance)

    def _estimate_news_impact_duration(self, news_item: Dict[str, Any], topics: List[str]) -> timedelta:
        """
        Estimate how long a news item will impact market sentiment.
        
        Args:
            news_item: Dictionary containing news details
            topics: List of topics associated with the news
            
        Returns:
            Estimated impact duration as timedelta
        """
        # Base duration
        hours = 12
        
        # Adjust based on news type
        if "earnings" in topics:
            hours = 48  # Earnings news has longer impact
        elif "analyst_rating" in topics:
            hours = 36  # Analyst ratings have medium-long impact
        elif "regulation" in topics or "legal" in topics:
            hours = 72  # Regulatory news has long impact
        elif "rumor" in topics:
            hours = 6   # Rumors have short impact
        
        # Adjust based on source
        source = news_item.get("source", "unknown")
        if source in ["bloomberg", "reuters", "wsj", "financial_times"]:
            hours *= 1.2  # Credible sources have longer impact
        
        return timedelta(hours=hours)

    def _estimate_social_impact_duration(self, social_item: Dict[str, Any], platform: str) -> timedelta:
        """
        Estimate how long a social media item will impact market sentiment.
        
        Args:
            social_item: Dictionary containing social media post details
            platform: Social media platform
            
        Returns:
            Estimated impact duration as timedelta
        """
        # Base duration - social media has shorter impact than news
        hours = 3
        
        # Adjust based on platform
        if platform == "twitter":
            hours = 4
        elif platform == "reddit":
            hours = 6  # Reddit discussions tend to last longer
        elif platform == "trading_forums":
            hours = 12  # Trading forum posts have longer relevance
            
        # Adjust based on engagement
        engagement = social_item.get("engagement", 0)
        if engagement > 1000:
            hours *= 1.5  # High engagement extends impact
        elif engagement > 10000:
            hours *= 2.0  # Viral content has much longer impact
            
        # Adjust based on user credibility
        user_cred = social_item.get("user_credibility", 0.5)
        if user_cred > 0.8:
            hours *= 1.3  # Credible users have more lasting impact
            
        return timedelta(hours=hours)

    def _convert_analyst_rating_to_sentiment(self, rating: Dict[str, Any]) -> float:
        """
        Convert analyst rating to sentiment score.
        
        Args:
            rating: Dictionary containing analyst rating details
            
        Returns:
            Sentiment score from -1.0 to 1.0
        """
        rating_type = rating.get("rating_type", "").lower()
        
        # Direct ratings
        if rating_type in ["buy", "strong buy", "overweight"]:
            return 0.8
        elif rating_type in ["outperform", "accumulate"]:
            return 0.6
        elif rating_type in ["hold", "neutral", "market perform"]:
            return 0.0
        elif rating_type in ["underperform", "reduce"]:
            return -0.6
        elif rating_type in ["sell", "strong sell", "underweight"]:
            return -0.8
            
        # Price target change
        if "previous_target" in rating and "new_target" in rating:
            prev = float(rating["previous_target"])
            new = float(rating["new_target"])
            
            if prev > 0:
                change_pct = (new - prev) / prev
                # Convert to sentiment scale
                return min(1.0, max(-1.0, change_pct * 5.0))  # Scale to make meaningful changes near +/-1
                
        return 0.0  # Neutral if we can't determine

    def _detect_sentiment_shift(self, current_sentiment: float) -> float:
        """
        Detect significant shifts in sentiment compared to baseline.
        
        Args:
            current_sentiment: The current sentiment score
            
        Returns:
            Sentiment shift magnitude (positive = increasing bullishness, negative = increasing bearishness)
        """
        if not self.historical_sentiment:
            return 0.0
            
        # Calculate sentiment baseline (moving average)
        recent_sentiments = [item["score"] for item in self.historical_sentiment[-10:]]
        self.sentiment_baseline = np.mean(recent_sentiments)
        
        # Calculate recent sentiment volatility
        if len(recent_sentiments) > 1:
            self.sentiment_volatility = np.std(recent_sentiments)
        
        # Calculate sentiment shift
        sentiment_shift = current_sentiment - self.sentiment_baseline
        
        # Normalize by volatility if possible
        if self.sentiment_volatility > 0:
            normalized_shift = sentiment_shift / self.sentiment_volatility
        else:
            normalized_shift = sentiment_shift * 2.0  # Apply fixed scaling
            
        return normalized_shift

    def _determine_signal(self, 
                        sentiment_score: float, 
                        confidence: float,
                        sentiment_shift: float,
                        market_condition: str) -> Tuple[SignalType, SignalStrength]:
        """
        Determine trading signal based on sentiment analysis.
        
        Args:
            sentiment_score: Overall sentiment score (-1.0 to 1.0)
            confidence: Confidence in the sentiment score (0.0 to 1.0)
            sentiment_shift: Magnitude of recent sentiment shift
            market_condition: Current market condition
            
        Returns:
            Tuple of (SignalType, SignalStrength)
        """
        # Default to neutral signal
        signal_type = SignalType.NEUTRAL
        signal_strength = SignalStrength.WEAK
        
        # Adjust thresholds based on market condition
        sentiment_threshold = self.sentiment_threshold
        confidence_threshold = self.confidence_threshold
        
        if market_condition == "ranging":
            # In ranging markets, require stronger signals
            sentiment_threshold *= 1.2
            confidence_threshold *= 1.1
        elif market_condition == "trending":
            # In trending markets, can be slightly more aggressive
            sentiment_threshold *= 0.9
            confidence_threshold *= 0.95
        elif market_condition == "volatile":
            # In volatile markets, require more confidence
            confidence_threshold *= 1.2
            
        # Consider both absolute sentiment and sentiment shift
        effective_sentiment = sentiment_score
        
        # If there's a significant shift, increase its importance
        if abs(sentiment_shift) >= self.sentiment_change_threshold:
            effective_sentiment = 0.7 * sentiment_score + 0.3 * (sentiment_shift * 0.5)
            
        # Determine signal direction
        if effective_sentiment >= sentiment_threshold and confidence >= confidence_threshold:
            signal_type = SignalType.BUY
        elif effective_sentiment <= -sentiment_threshold and confidence >= confidence_threshold:
            signal_type = SignalType.SELL
        
        # Determine signal strength
        if signal_type != SignalType.NEUTRAL:
            # Base strength on sentiment magnitude and confidence
            strength_value = abs(effective_sentiment) * confidence
            
            # Adjust for sentiment shift - strengthen signal if shift aligns with sentiment
            if (effective_sentiment > 0 and sentiment_shift > 0) or \
               (effective_sentiment < 0 and sentiment_shift < 0):
                strength_value += abs(sentiment_shift) * 0.2
                
            # Adjust for market condition alignment
            if (signal_type == SignalType.BUY and market_condition == "uptrend") or \
               (signal_type == SignalType.SELL and market_condition == "downtrend"):
                strength_value *= 1.1
                
            # Convert to enum
            if strength_value >= 0.8:
                signal_strength = SignalStrength.VERY_STRONG
            elif strength_value >= 0.65:
                signal_strength = SignalStrength.STRONG
            elif strength_value >= 0.5:
                signal_strength = SignalStrength.MODERATE
            else:
                signal_strength = SignalStrength.WEAK
        
        return signal_type, signal_strength

    def _update_sentiment_history(self, sentiment_score: float, confidence: float):
        """
        Update historical sentiment tracking.
        
        Args:
            sentiment_score: The current sentiment score
            confidence: Confidence in the sentiment score
        """
        self.historical_sentiment.append({
            "score": sentiment_score,
            "confidence": confidence,
            "timestamp": datetime.now()
        })
        
        # Keep only the most recent history
        max_history = 100
        if len(self.historical_sentiment) > max_history:
            self.historical_sentiment = self.historical_sentiment[-max_history:]

    def _extract_top_topics(self, sentiment_data: List[SentimentData], limit: int = 5) -> List[str]:
        """
        Extract the most frequently mentioned topics from sentiment data.
        
        Args:
            sentiment_data: List of sentiment data points
            limit: Maximum number of topics to return
            
        Returns:
            List of top topics
        """
        topic_counts = {}
        
        for data in sentiment_data:
            for topic in data.topics:
                if topic not in topic_counts:
                    topic_counts[topic] = 0
                topic_counts[topic] += 1
        
        # Sort by count and return top topics
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_topics[:limit]]

    def _extract_most_relevant_news(self, sentiment_data: List[SentimentData]) -> Optional[Dict[str, Any]]:
        """
        Extract the most relevant news item from sentiment data.
        
        Args:
            sentiment_data: List of sentiment data points
            
        Returns:
            Most relevant news item or None
        """
        news_items = [data for data in sentiment_data if data.source.startswith("news_")]
        
        if not news_items:
            return None
            
        # Sort by relevance * confidence
        news_items.sort(key=lambda x: x.relevance * x.confidence, reverse=True)
        
        top_news = news_items[0]
        
        # In a real implementation, we would return the actual news details
        # For now, return basic information
        return {
            "sentiment": top_news.score,
            "confidence": top_news.confidence,
            "source": top_news.source,
            "timestamp": top_news.timestamp.isoformat(),
            "topics": top_news.topics
        }

    def _initialize_topic_relevance(self) -> Dict[str, float]:
        """
        Initialize relevance scores for different news/social topics.
        
        Returns:
            Dictionary mapping topics to relevance scores
        """
        # This would normally be customized per asset
        # For now, provide reasonable defaults
        return {
            "earnings": 0.9,
            "revenue": 0.85,
            "profit": 0.8,
            "growth": 0.7,
            "analyst_rating": 0.8,
            "price_target": 0.75,
            "partnership": 0.6,
            "acquisition": 0.85,
            "regulation": 0.7,
            "legal": 0.65,
            "management_change": 0.6,
            "product_launch": 0.55,
            "competitive_analysis": 0.5,
            "industry_trend": 0.4,
            "macro_economic": 0.3,
            "technical_analysis": 0.6,
            "rumor": 0.4,
            "insider_trading": 0.7,
            "short_interest": 0.65,
            "options_activity": 0.6,
            "market_sentiment": 0.5
        }

    async def adapt(self, performance_metrics: Dict[str, Any]):
        """
        Adapt the strategy based on recent performance.
        This is called periodically by the system to allow the strategy to adapt.
        
        Args:
            performance_metrics: Recent performance metrics for this strategy
        """
        try:
            # Extract relevant metrics
            win_rate = performance_metrics.get("win_rate", 0.5)
            profit_factor = performance_metrics.get("profit_factor", 1.0)
            avg_profit = performance_metrics.get("avg_profit", 0.0)
            avg_loss = performance_metrics.get("avg_loss", 0.0)
            
            # Adapt strategy parameters based on performance
            if win_rate < 0.4 or profit_factor < 0.8:
                # Poor performance - increase thresholds to be more selective
                self.sentiment_threshold = min(self.sentiment_threshold + 0.05, 0.6)
                self.confidence_threshold = min(self.confidence_threshold + 0.05, 0.8)
                logger.info(
                    f"Adapting: Increased sentiment threshold to {self.sentiment_threshold:.2f}, "
                    f"confidence threshold to {self.confidence_threshold:.2f}"
                )
            
            # If losing too much on losses, increase change threshold to react faster
            if avg_loss < -2.0 * avg_profit:
                self.sentiment_change_threshold = max(self.sentiment_change_threshold - 0.02, 0.08)
                logger.info(f"Adapting: Decreased sentiment change threshold to {self.sentiment_change_threshold:.2f}")
            
            # If performing well, can relax thresholds slightly
            if win_rate > 0.6 and profit_factor > 1.5:
                self.sentiment_threshold = max(self.sentiment_threshold - 0.03, 0.2)
                logger.info(f"Adapting: Decreased sentiment threshold to {self.sentiment_threshold:.2f}")
                
            # Adjust source weights based on performance correlation
            self._adapt_source_weights(performance_metrics)
                
            # Log adaptation
            logger.info(
                f"Strategy adapted based on performance metrics: "
                f"win_rate={win_rate:.2f}, profit_factor={profit_factor:.2f}, "
                f"avg_profit={avg_profit:.2f}, avg_loss={avg_loss:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error in strategy adaptation: {str(e)}")

    def _adapt_source_weights(self, performance_metrics: Dict[str, Any]):
        """
        Adapt source weights based on their correlation with successful trades.
        
        Args:
            performance_metrics: Recent performance metrics
        """
        if "source_performance" not in performance_metrics:
            return
            
        source_performance = performance_metrics["source_performance"]
        
        # Update source reliability based on performance
        for source in self.sentiment_sources:
            if source.name in source_performance:
                perf = source_performance[source.name]
                
                # Update reliability score
                current_reliability = source.reliability
                target_reliability = perf.get("win_rate", 0.5)
                
                # Smooth adaptation
                source.reliability = 0.8 * current_reliability + 0.2 * target_reliability
                
                logger.debug(
                    f"Adapted source {source.name} reliability: "
                    f"{current_reliability:.2f} -> {source.reliability:.2f}"
                )
        
        # Normalize weights
        total_weight = sum(source.weight * source.reliability for source in self.sentiment_sources)
        for source in self.sentiment_sources:
            source.weight = (source.weight * source.reliability) / total_weight
