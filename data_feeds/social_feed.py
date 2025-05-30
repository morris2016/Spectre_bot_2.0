#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Social Media Feed Module

This module handles social media data gathering, processing, and sentiment analysis
for the QuantumSpectre Elite Trading System. It monitors multiple social platforms
including Twitter, Reddit, Telegram channels, Discord servers, and specialized forums
to extract market sentiment and detect emerging trends that could affect asset prices.
"""

import os
import re
import time
import json
import asyncio
import datetime
from typing import Dict, List, Set, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import threading
import queue
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import traceback

# Data gathering and processing
import httpx
import websockets
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# NLP and sentiment analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import transformers
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import spacy

# Internal imports
from config import Config
from common.logger import get_logger
from common.utils import (
    retry_with_backoff_decorator, rate_limit, SafeDict, hash_content,
    retry_with_backoff_decorator,
    rate_limit,
    SafeDict,
    hash_content,
    safe_nltk_download,
)
from common.constants import (
    SOCIAL_PLATFORMS, SOCIAL_API_KEYS, SOCIAL_QUERY_PARAMS,
    SOCIAL_UPDATE_INTERVALS, NLP_MODELS, ASSET_KEYWORDS
)
from common.metrics import MetricsCollector
from common.exceptions import (
    DataSourceError, RateLimitError, AuthenticationError,
    ParsingError, ModelLoadError
)
from common.db_client import DatabaseClient
from common.redis_client import RedisClient
from common.async_utils import gather_with_concurrency, timed_cache

from data_feeds.base_feed import BaseDataFeed

# Initialize logger
logger = get_logger(__name__)

# NLP model configurations
FINANCIAL_BERT_MODEL = "ProsusAI/finbert"
CRYPTO_SENTIMENT_MODEL = "ElKulako/cryptobert"
MARKET_SENTIMENT_MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

@dataclass
class SocialMediaPost:
    """Data class for standardized social media content"""
    platform: str
    content_id: str
    author: str
    content: str
    timestamp: datetime.datetime
    likes: int = 0
    shares: int = 0
    comments: int = 0
    views: int = 0
    url: str = ""
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    related_assets: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    is_verified_author: bool = False
    author_followers: int = 0
    author_influence_score: float = 0.0
    extracted_symbols: List[str] = field(default_factory=list)
    language: str = "en"
    
    def __post_init__(self):
        """Post initialization processing"""
        # Ensure timestamp is datetime object
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
        
        # Generate content hash for deduplication
        self.content_hash = hash_content(f"{self.platform}_{self.content}_{self.author}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        result = {k: v for k, v in self.__dict__.items()}
        # Convert datetime to ISO format string for serialization
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SocialMediaPost':
        """Create instance from dictionary"""
        return cls(**data)


class SocialMediaFeed(BaseDataFeed):
    """
    Social Media Feed for gathering and analyzing social data relevant to trading.
    
    This class monitors various social media platforms to extract market sentiment,
    discover emerging trends, and detect potential market-moving information.
    It implements advanced filtering, deduplication, and credibility scoring.
    """
    
    def __init__(self, config: Config, db_client: DatabaseClient, redis_client: RedisClient):
        """
        Initialize the Social Media Feed.
        
        Args:
            config: Application configuration
            db_client: Database client for persistent storage
            redis_client: Redis client for real-time data and caching
        """
        super().__init__(name="social_media_feed", config=config)
        self.db_client = db_client
        self.redis_client = redis_client
        self.metrics = MetricsCollector(namespace="social_feed")
        
        # Configure platforms to monitor
        self.platforms = config.get("social_feed.platforms", SOCIAL_PLATFORMS)
        self.api_keys = config.get("social_feed.api_keys", SOCIAL_API_KEYS)
        self.update_intervals = config.get("social_feed.update_intervals", SOCIAL_UPDATE_INTERVALS)
        
        # Initialize NLP and sentiment analysis tools
        self.nlp_models = {}
        self.initialize_nlp_models()
        
        # Asset-related keyword dictionaries for relevance filtering
        self.asset_keywords = config.get("social_feed.asset_keywords", ASSET_KEYWORDS)
        self.expanded_keywords = self._expand_keywords()
        
        # Platform-specific clients
        self.platform_clients = {}
        self.initialize_platform_clients()
        
        # Data processing queues and workers
        self.raw_data_queue = asyncio.Queue()
        self.processed_data_queue = asyncio.Queue()
        self.worker_threads = []
        
        # Statistics and tracking
        self.stats = {
            "posts_processed": 0,
            "relevant_posts": 0,
            "high_impact_posts": 0,
            "platform_stats": {p: {"fetched": 0, "relevant": 0} for p in self.platforms},
            "asset_mentions": {},
            "influencer_activity": {}
        }
        
        # Set up real-time processing pipeline
        self.processing_pipeline = [
            self._preprocess_content,
            self._extract_entities,
            self._analyze_sentiment,
            self._score_relevance,
            self._score_credibility,
            self._enrich_with_metadata
        ]
        
        logger.info(f"Social Media Feed initialized with {len(self.platforms)} platforms")
    
    async def start(self):
        """Start the social media feed data collection and processing"""
        logger.info("Starting Social Media Feed service")
        
        # Initialize worker tasks
        tasks = []
        
        # Start platform-specific data collectors
        for platform in self.platforms:
            platform_method = getattr(self, f"_collect_{platform}", None)
            if platform_method:
                interval = self.update_intervals.get(platform, 60)
                tasks.append(self._periodic_collector(platform_method, interval))
                logger.info(f"Started collector for {platform} with interval {interval}s")
            else:
                logger.warning(f"No collector method found for platform: {platform}")
        
        # Start data processing workers
        tasks.append(self._data_processor())
        
        # Start output publisher
        tasks.append(self._publish_processed_data())
        
        # Run all tasks concurrently
        self.running = True
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop the social media feed service"""
        logger.info("Stopping Social Media Feed service")
        self.running = False
        
        # Close platform-specific clients
        for client in self.platform_clients.values():
            if hasattr(client, "close"):
                await client.close()
        
        # Cleanup NLP models
        for model in self.nlp_models.values():
            del model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Social Media Feed service stopped")
    
    def initialize_nlp_models(self):
        """Initialize NLP models for sentiment analysis and entity recognition"""
        try:
            # Ensure required NLTK resources are available without downloading
            safe_nltk_download('tokenizers/punkt')
            safe_nltk_download('vader_lexicon')
            # Download required NLTK resources if not already present
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                safe_nltk_download('punkt')
            
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                safe_nltk_download('vader_lexicon')
            
            # Initialize VADER sentiment analyzer (fast but less accurate)
            self.nlp_models['vader'] = SentimentIntensityAnalyzer()
            
            # Load device-appropriate settings for transformers
            device = 0 if torch.cuda.is_available() else -1
            device_str = "GPU" if device == 0 else "CPU"
            logger.info(f"Loading NLP models on {device_str}")
            
            # Transformers pipeline for financial sentiment - fine-tuned for financial text
            self.nlp_models['financial'] = pipeline(
                "sentiment-analysis",
                model=FINANCIAL_BERT_MODEL,
                tokenizer=FINANCIAL_BERT_MODEL,
                device=device
            )
            
            # Transformers pipeline for crypto sentiment - specialized for crypto discussions
            self.nlp_models['crypto'] = pipeline(
                "sentiment-analysis",
                model=CRYPTO_SENTIMENT_MODEL,
                tokenizer=CRYPTO_SENTIMENT_MODEL,
                device=device
            )
            
            # Load spaCy model for entity recognition and linguistic analysis
            self.nlp_models['spacy'] = spacy.load("en_core_web_sm")
            
            logger.info("Successfully loaded all NLP models")
            
        except Exception as e:
            logger.error(f"Error initializing NLP models: {str(e)}")
            logger.error(traceback.format_exc())
            raise ModelLoadError(f"Failed to initialize NLP models: {str(e)}")
    
    def initialize_platform_clients(self):
        """Initialize API clients for each social media platform"""
        # Create HTTP client with proper settings
        timeout = httpx.Timeout(30.0, connect=10.0)
        self.http_client = httpx.AsyncClient(timeout=timeout)
        
        # Initialize selenium driver for platforms requiring browser automation
        self.selenium_initialized = False
        
        # Platform-specific initializations will be done on first use
        logger.info(f"Initialized base HTTP client for social media APIs")
    
    async def initialize_selenium(self):
        """Initialize Selenium for platforms requiring browser automation"""
        if self.selenium_initialized:
            return
        
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
            
            # Initialize the Chrome WebDriver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.selenium_initialized = True
            logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Selenium: {str(e)}")
            logger.error(traceback.format_exc())
            raise DataSourceError(f"Failed to initialize Selenium for web scraping: {str(e)}")
    
    def _expand_keywords(self) -> Dict[str, Set[str]]:
        """
        Expand asset keywords with variations, common misspellings, and related terms
        to improve detection of relevant content.
        """
        expanded = {}
        for asset, keywords in self.asset_keywords.items():
            # Create base set with original keywords
            expanded_set = set(keywords)
            
            # Add lowercase and uppercase variations
            for kw in keywords:
                expanded_set.add(kw.lower())
                expanded_set.add(kw.upper())
                expanded_set.add(kw.capitalize())
                
                # Add variations without spaces
                if ' ' in kw:
                    expanded_set.add(kw.replace(' ', ''))
                
                # Add common symbol variations
                if asset.startswith('BTC'):
                    expanded_set.add('bitcoin')
                    expanded_set.add('btc')
                    expanded_set.add('₿')
                elif asset.startswith('ETH'):
                    expanded_set.add('ethereum')
                    expanded_set.add('eth')
                    expanded_set.add('Ξ')
                
                # Add variations with underscores and dashes
                if ' ' in kw:
                    expanded_set.add(kw.replace(' ', '_'))
                    expanded_set.add(kw.replace(' ', '-'))
            
            expanded[asset] = expanded_set
        
        return expanded
    
    async def _periodic_collector(self, collector_method: Callable, interval: int):
        """
        Run a collector method periodically at specified interval.
        
        Args:
            collector_method: The collector method to run
            interval: Collection interval in seconds
        """
        while self.running:
            try:
                start_time = time.time()
                await collector_method()
                
                # Calculate sleep time, ensuring at least 1 second between runs
                elapsed = time.time() - start_time
                sleep_time = max(1, interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in collector {collector_method.__name__}: {str(e)}")
                logger.error(traceback.format_exc())
                # Add exponential backoff for repeated errors
                await asyncio.sleep(min(300, interval * 2))
    
    async def _data_processor(self):
        """Process raw social media data from the queue"""
        while self.running:
            try:
                # Get raw data from queue with timeout
                raw_data = await asyncio.wait_for(self.raw_data_queue.get(), timeout=1.0)
                
                # Skip if None (can happen when queue is empty and timeout)
                if raw_data is None:
                    continue
                
                # Process the raw data through the pipeline
                processed_data = raw_data
                for process_step in self.processing_pipeline:
                    processed_data = await process_step(processed_data)
                    # If processing step returned None, skip further processing
                    if processed_data is None:
                        break
                
                # If we have processed data, put it in the output queue
                if processed_data is not None:
                    await self.processed_data_queue.put(processed_data)
                    
                    # Update statistics
                    self.stats["posts_processed"] += 1
                    platform = processed_data.platform
                    self.stats["platform_stats"][platform]["relevant"] += 1
                    
                    # Track asset mentions
                    for asset in processed_data.related_assets:
                        if asset not in self.stats["asset_mentions"]:
                            self.stats["asset_mentions"][asset] = 0
                        self.stats["asset_mentions"][asset] += 1
                
                # Mark task as done
                self.raw_data_queue.task_done()
            
            except asyncio.TimeoutError:
                # This is expected when the queue is empty
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data processor: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(1)  # Prevent tight loop if there's an error
    
    async def _publish_processed_data(self):
        """Publish processed social media data to Redis and database"""
        while self.running:
            try:
                # Get processed data with timeout
                processed_data = await asyncio.wait_for(self.processed_data_queue.get(), timeout=1.0)
                
                if processed_data is None:
                    continue
                
                # Convert to dictionary for storage
                data_dict = processed_data.to_dict()
                
                # Publish to Redis for real-time consumers
                channel = f"social_feed:{processed_data.platform}"
                await self.redis_client.publish(channel, json.dumps(data_dict))
                
                # For high-impact posts, publish to a special channel
                if processed_data.relevance_score > 0.7 and processed_data.author_influence_score > 0.6:
                    await self.redis_client.publish(
                        "social_feed:high_impact", 
                        json.dumps(data_dict)
                    )
                    self.stats["high_impact_posts"] += 1
                
                # Store in database for historical analysis
                collection = f"social_media_{processed_data.platform}"
                await self.db_client.insert_one(collection, data_dict)
                
                # Update Redis sorted sets for trending analysis
                timestamp = int(processed_data.timestamp.timestamp())
                
                # Add to platform-specific trending set
                await self.redis_client.zadd(
                    f"trending:social:{processed_data.platform}", 
                    {processed_data.content_hash: timestamp}
                )
                
                # Add to asset-specific trending sets
                for asset in processed_data.related_assets:
                    score = processed_data.relevance_score * processed_data.author_influence_score
                    await self.redis_client.zadd(
                        f"trending:social:asset:{asset}", 
                        {processed_data.content_hash: score}
                    )
                
                # Mark task as done
                self.processed_data_queue.task_done()
                
            except asyncio.TimeoutError:
                # This is expected when the queue is empty
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in data publisher: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(1)  # Prevent tight loop if there's an error
    
    # Data Processing Pipeline Steps
    
    async def _preprocess_content(self, post: SocialMediaPost) -> Optional[SocialMediaPost]:
        """
        Preprocess social media content to clean and normalize text.
        
        Args:
            post: The social media post to process
            
        Returns:
            Processed post or None if post should be filtered out
        """
        # Skip empty content
        if not post.content or len(post.content.strip()) < 5:
            return None
        
        # Normalize whitespace
        post.content = re.sub(r'\s+', ' ', post.content).strip()
        
        # Extract hashtags
        hashtags = re.findall(r'#(\w+)', post.content)
        post.hashtags = [tag.lower() for tag in hashtags]
        
        # Extract mentions
        mentions = re.findall(r'@(\w+)', post.content)
        post.mentions = [mention.lower() for mention in mentions]
        
        # Extract cashtags and symbols
        cashtags = re.findall(r'\$(\w+)', post.content)
        post.extracted_symbols = [symbol.upper() for symbol in cashtags]
        
        # Clean content for sentiment analysis (remove URLs)
        cleaned_content = re.sub(r'https?://\S+', '', post.content)
        cleaned_content = re.sub(r'[^\w\s\.,!?]', '', cleaned_content)
        post.cleaned_content = cleaned_content.strip()
        
        # Check for language to ensure it's supported by our models
        try:
            # Use spaCy for language detection
            doc = self.nlp_models['spacy'](post.cleaned_content[:100])
            post.language = doc.lang_
            
            # Skip non-English content for now
            if post.language != 'en':
                logger.debug(f"Skipping non-English content: {post.language}")
                return None
                
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            # Default to English if detection fails
            post.language = 'en'
        
        return post
    
    async def _extract_entities(self, post: SocialMediaPost) -> SocialMediaPost:
        """
        Extract entities and key information from the post content.
        
        Args:
            post: The social media post
            
        Returns:
            Post with extracted entities
        """
        # Use spaCy for entity extraction
        try:
            doc = self.nlp_models['spacy'](post.cleaned_content)
            
            # Extract named entities
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            post.entities = entities
            
            # Look for asset-related keywords
            for asset, keywords in self.expanded_keywords.items():
                # Check content for keyword matches
                content_lower = post.content.lower()
                if any(kw.lower() in content_lower for kw in keywords):
                    if asset not in post.related_assets:
                        post.related_assets.append(asset)
                
                # Check hashtags for keyword matches
                for hashtag in post.hashtags:
                    if hashtag.lower() in keywords:
                        if asset not in post.related_assets:
                            post.related_assets.append(asset)
                
                # Check extracted symbols
                for symbol in post.extracted_symbols:
                    # Direct symbol match
                    if symbol == asset or symbol in keywords:
                        if asset not in post.related_assets:
                            post.related_assets.append(asset)
            
            # Filter out if no related assets found
            if not post.related_assets:
                logger.debug(f"No related assets found in post: {post.content_id}")
                return None
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {str(e)}")
        
        return post
    
    async def _analyze_sentiment(self, post: SocialMediaPost) -> SocialMediaPost:
        """
        Analyze sentiment of the post content using multiple models.
        
        Args:
            post: The social media post
            
        Returns:
            Post with sentiment analysis results
        """
        if not post or not post.cleaned_content:
            return post
        
        try:
            # Use VADER for initial quick sentiment analysis
            sentiment = self.nlp_models['vader'].polarity_scores(post.cleaned_content)
            post.vader_sentiment = sentiment
            post.sentiment_score = sentiment['compound']
            
            # Determine sentiment label from compound score
            if sentiment['compound'] >= 0.05:
                post.sentiment_label = "positive"
            elif sentiment['compound'] <= -0.05:
                post.sentiment_label = "negative"
            else:
                post.sentiment_label = "neutral"
            
            # For significant sentiment or posts mentioning crypto, use specialized model
            if abs(sentiment['compound']) > 0.2 or any(asset.startswith('BTC') or asset.startswith('ETH') for asset in post.related_assets):
                # Limit content length for transformer models
                content = post.cleaned_content[:512]
                
                try:
                    # Select appropriate model based on asset type
                    if any(asset.startswith('BTC') or asset.startswith('ETH') for asset in post.related_assets):
                        model_key = 'crypto'
                    else:
                        model_key = 'financial'
                    
                    # Get sentiment from appropriate transformer model
                    specialized_sentiment = self.nlp_models[model_key](content)
                    
                    # Extract and normalize the sentiment score
                    if specialized_sentiment and len(specialized_sentiment) > 0:
                        label = specialized_sentiment[0]['label']
                        score = specialized_sentiment[0]['score']
                        
                        # Map different model output formats to standardized scores
                        if label == 'positive' or label == 'LABEL_2' or label == 'Positive':
                            post.specialized_sentiment_score = score
                            post.specialized_sentiment_label = 'positive'
                        elif label == 'negative' or label == 'LABEL_0' or label == 'Negative':
                            post.specialized_sentiment_score = -score
                            post.specialized_sentiment_label = 'negative'
                        else:
                            post.specialized_sentiment_score = 0
                            post.specialized_sentiment_label = 'neutral'
                        
                        # Average the VADER and specialized model scores
                        post.sentiment_score = (post.sentiment_score + post.specialized_sentiment_score) / 2
                        
                        # Use specialized model label if confidence is high
                        if abs(post.specialized_sentiment_score) > 0.7:
                            post.sentiment_label = post.specialized_sentiment_label
                
                except Exception as e:
                    logger.warning(f"Specialized sentiment analysis failed: {str(e)}")
                    # Fall back to VADER results if specialized model fails
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {str(e)}")
        
        return post
    
    async def _score_relevance(self, post: SocialMediaPost) -> Optional[SocialMediaPost]:
        """
        Calculate relevance score for the post based on content relevance to trading.
        
        Args:
            post: The social media post
            
        Returns:
            Post with relevance score or None if not relevant
        """
        if not post:
            return None
            
        # Initialize base score
        relevance_score = 0.0
        
        # Increase score based on number of related assets mentioned
        relevance_score += min(0.3, len(post.related_assets) * 0.1)
        
        # Trading-specific keywords increase relevance
        trading_keywords = [
            'buy', 'sell', 'long', 'short', 'trade', 'position', 'entry', 'exit',
            'support', 'resistance', 'breakout', 'breakdown', 'bullish', 'bearish',
            'trend', 'oversold', 'overbought', 'signal', 'indicator', 'pattern',
            'accumulation', 'distribution', 'volatility', 'momentum', 'reversal'
        ]
        
        content_lower = post.content.lower()
        keyword_matches = sum(1 for kw in trading_keywords if kw in content_lower)
        relevance_score += min(0.3, keyword_matches * 0.03)
        
        # Increase score for strong sentiment
        sentiment_strength = abs(post.sentiment_score)
        relevance_score += min(0.2, sentiment_strength * 0.2)
        
        # Increase score based on asset-keyword match density
        total_words = len(content_lower.split())
        if total_words > 0:
            # Count all matched keywords in content
            keyword_count = 0
            for asset in post.related_assets:
                for keyword in self.expanded_keywords[asset]:
                    if keyword.lower() in content_lower:
                        keyword_count += 1
            
            keyword_density = keyword_count / total_words
            relevance_score += min(0.2, keyword_density * 2)
        
        # Reduce score for very short or very long content
        content_length = len(post.content)
        if content_length < 20:
            relevance_score *= 0.5
        elif content_length > 1000:
            relevance_score *= 0.8
        
        # Set the final relevance score
        post.relevance_score = min(1.0, relevance_score)
        
        # Filter out posts with low relevance
        if post.relevance_score < 0.3:
            logger.debug(f"Filtering out low relevance post: {post.content_id}, score: {post.relevance_score}")
            return None
            
        return post
    
    async def _score_credibility(self, post: SocialMediaPost) -> SocialMediaPost:
        """
        Score the credibility and influence of the post author.
        
        Args:
            post: The social media post
            
        Returns:
            Post with author credibility score
        """
        if not post:
            return post
            
        # Base influence score starts from verification status
        influence_score = 0.3 if post.is_verified_author else 0.1
        
        # Account for author's follower count (log scale)
        if post.author_followers > 0:
            follower_score = min(0.4, np.log10(post.author_followers) / 7)
            influence_score += follower_score
        
        # Account for engagement metrics
        engagement = post.likes + post.shares * 2 + post.comments * 1.5
        engagement_score = min(0.3, np.log10(max(10, engagement)) / 5)
        influence_score += engagement_score
        
        # Check if author is in our known influencers list
        influencer_key = f"{post.platform}:{post.author}"
        influencer_data = await self.redis_client.hgetall(f"social:influencers:{influencer_key}")
        
        if influencer_data:
            # Existing influencer - use historical data to adjust score
            historical_accuracy = float(influencer_data.get('accuracy', 0.5))
            historical_impact = float(influencer_data.get('impact', 0.5))
            
            # Blend historical metrics with current calculation
            influence_score = (influence_score * 0.7) + (historical_accuracy * 0.15) + (historical_impact * 0.15)
            
            # Update influencer tracking
            if post.author not in self.stats["influencer_activity"]:
                self.stats["influencer_activity"][post.author] = 0
            self.stats["influencer_activity"][post.author] += 1
        
        # Set the final author influence score
        post.author_influence_score = min(1.0, influence_score)
        
        return post
    
    async def _enrich_with_metadata(self, post: SocialMediaPost) -> SocialMediaPost:
        """
        Enrich the post with additional metadata and derived fields.
        
        Args:
            post: The social media post
            
        Returns:
            Post with additional metadata
        """
        if not post:
            return post
            
        # Calculate a combined impact score
        post.impact_score = post.relevance_score * 0.6 + post.author_influence_score * 0.4
        
        # Add geo-location if available (platform specific)
        if hasattr(post, 'geo') and post.geo:
            # Placeholder for geo-enrichment
            pass
        
        # Add timestamp-based features
        dt = post.timestamp
        post.hour_of_day = dt.hour
        post.day_of_week = dt.weekday()
        post.is_market_hours = 9 <= dt.hour < 16 and 0 <= dt.weekday() < 5  # Simplified market hours check
        
        # Add counters for monitoring frequency
        for asset in post.related_assets:
            # Update Redis counters for real-time monitoring
            await self.redis_client.incr(f"social:asset:mentions:{asset}")
            await self.redis_client.incr(f"social:asset:mentions:{asset}:{post.sentiment_label}")
            
            # Add to time-series for trend analysis
            current_time = int(time.time())
            await self.redis_client.zadd(
                f"social:asset:timeseries:{asset}", 
                {f"{post.platform}:{post.content_id}": current_time}
            )
        
        return post
    
    # Platform-specific collectors
    
    async def _collect_twitter(self):
        """Collect data from Twitter/X Platform"""
        logger.debug("Collecting data from Twitter")
        platform = "twitter"
        
        try:
            # Check if API keys are available
            if platform not in self.api_keys or not self.api_keys[platform].get('bearer_token'):
                logger.warning(f"No API keys configured for {platform}")
                return
            
            bearer_token = self.api_keys[platform]['bearer_token']
            
            # Set up headers for Twitter API v2
            headers = {
                "Authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json"
            }
            
            # Get search queries for each asset
            queries = []
            for asset, keywords in self.asset_keywords.items():
                # Create Twitter search query string
                asset_query = ' OR '.join([f'"{kw}"' for kw in keywords if ' ' in kw] + 
                                       [kw for kw in keywords if ' ' not in kw])
                
                # Add cashtag for financial assets
                if any(asset.startswith(prefix) for prefix in ['BTC', 'ETH', 'USD', 'EUR', 'JPY']):
                    asset_query += f' OR ${asset}'
                
                queries.append((asset, asset_query))
            
            # Process each search query
            for asset, query in queries:
                # Construct search API URL
                url = "https://api.twitter.com/2/tweets/search/recent"
                
                # Parameters for the search request
                params = {
                    "query": query,
                    "max_results": 25,
                    "tweet.fields": "created_at,public_metrics,entities,author_id,lang",
                    "user.fields": "verified,public_metrics,description",
                    "expansions": "author_id"
                }
                
                # Get tweets from the last search
                last_id = await self.redis_client.get(f"social:twitter:last_id:{asset}")
                if last_id:
                    params["since_id"] = last_id
                
                # Make the API request
                async with self.http_client.stream("GET", url, headers=headers, params=params) as response:
                    if response.status_code != 200:
                        logger.error(f"Twitter API error: {response.status_code} - {await response.text()}")
                        continue
                    
                    # Process the response
                    response_data = await response.json()
                    
                    # Check if we have tweets and users
                    if 'data' not in response_data or 'includes' not in response_data or 'users' not in response_data['includes']:
                        logger.debug(f"No tweets found for query: {query}")
                        continue
                    
                    tweets = response_data['data']
                    users = {user['id']: user for user in response_data['includes']['users']}
                    
                    # Get newest tweet ID for next search
                    if tweets:
                        newest_id = max(tweets, key=lambda x: x['id'])['id']
                        await self.redis_client.set(f"social:twitter:last_id:{asset}", newest_id)
                    
                    # Process each tweet
                    for tweet in tweets:
                        # Skip retweets for now
                        if 'RT @' in tweet.get('text', ''):
                            continue
                        
                        # Get author data
                        author_id = tweet.get('author_id')
                        author = users.get(author_id, {})
                        
                        # Create standardized post object
                        post = SocialMediaPost(
                            platform="twitter",
                            content_id=tweet['id'],
                            author=author.get('username', ''),
                            content=tweet.get('text', ''),
                            timestamp=datetime.datetime.fromisoformat(tweet['created_at'].replace('Z', '+00:00')),
                            likes=tweet.get('public_metrics', {}).get('like_count', 0),
                            shares=tweet.get('public_metrics', {}).get('retweet_count', 0),
                            comments=tweet.get('public_metrics', {}).get('reply_count', 0),
                            is_verified_author=author.get('verified', False),
                            author_followers=author.get('public_metrics', {}).get('followers_count', 0),
                            language=tweet.get('lang', 'en'),
                            related_assets=[asset]  # Pre-fill with the asset we searched for
                        )
                        
                        # Add to processing queue
                        await self.raw_data_queue.put(post)
                    
                    # Update stats
                    self.stats["platform_stats"]["twitter"]["fetched"] += len(tweets)
                    
                # Sleep briefly between queries to respect rate limits
                await asyncio.sleep(2)
                
        except Exception as e:
            logger.error(f"Error collecting Twitter data: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _collect_reddit(self):
        """Collect data from Reddit"""
        logger.debug("Collecting data from Reddit")
        platform = "reddit"
        
        try:
            # Check if API credentials are available
            if platform not in self.api_keys or not all(k in self.api_keys[platform] for k in ['client_id', 'client_secret']):
                logger.warning(f"No API keys configured for {platform}")
                return
            
            # Reddit API authentication
            client_id = self.api_keys[platform]['client_id']
            client_secret = self.api_keys[platform]['client_secret']
            user_agent = self.api_keys[platform].get('user_agent', 'QuantumSpectre/1.0')
            
            # Get OAuth token
            auth_url = "https://www.reddit.com/api/v1/access_token"
            auth_data = {
                "grant_type": "client_credentials",
                "username": self.api_keys[platform].get('username', ''),
                "password": self.api_keys[platform].get('password', '')
            }
            auth_headers = {
                "User-Agent": user_agent
            }
            
            # Get token from cache or refresh
            token = await self.redis_client.get("social:reddit:token")
            if not token:
                # Get new token
                async with self.http_client.stream("POST", auth_url, auth=(client_id, client_secret), data=auth_data, headers=auth_headers) as response:
                    if response.status_code != 200:
                        logger.error(f"Reddit API auth error: {response.status_code} - {await response.text()}")
                        return
                    
                    auth_response = await response.json()
                    token = auth_response.get('access_token')
                    if not token:
                        logger.error("Failed to get Reddit API token")
                        return
                    
                    # Cache token with expiration
                    expires_in = auth_response.get('expires_in', 3600)
                    await self.redis_client.set("social:reddit:token", token, ex=expires_in-60)
            
            # Set up headers for API requests
            headers = {
                "Authorization": f"Bearer {token}",
                "User-Agent": user_agent
            }
            
            # Subreddits to monitor for each asset
            subreddits = {
                "BTC/USD": ["Bitcoin", "CryptoCurrency", "CryptoMarkets"],
                "ETH/USD": ["ethereum", "CryptoCurrency", "CryptoMarkets"],
                "EUR/USD": ["Forex", "trading", "economy"],
                # Add more asset-specific subreddits
            }
            
            # Process each asset and its subreddits
            for asset, asset_subs in subreddits.items():
                for subreddit in asset_subs:
                    # Construct URLs for hot, new, and rising posts
                    for section in ["hot", "new", "rising"]:
                        url = f"https://oauth.reddit.com/r/{subreddit}/{section}.json"
                        params = {"limit": 25}
                        
                        # Make the API request
                        async with self.http_client.stream("GET", url, headers=headers, params=params) as response:
                            if response.status_code != 200:
                                logger.error(f"Reddit API error for /r/{subreddit}: {response.status_code} - {await response.text()}")
                                continue
                            
                            # Process the response
                            response_data = await response.json()
                            
                            if 'data' not in response_data or 'children' not in response_data['data']:
                                logger.debug(f"No posts found in /r/{subreddit}/{section}")
                                continue
                            
                            posts = response_data['data']['children']
                            
                            # Process each post
                            for post_data in posts:
                                if 'data' not in post_data:
                                    continue
                                
                                post = post_data['data']
                                
                                # Skip posts we've seen before
                                post_id = post.get('id')
                                if await self.redis_client.exists(f"social:reddit:seen:{post_id}"):
                                    continue
                                
                                # Mark as seen
                                await self.redis_client.set(f"social:reddit:seen:{post_id}", 1, ex=86400)  # 24 hour expiry
                                
                                # Create timestamp from created_utc
                                created_utc = post.get('created_utc', 0)
                                timestamp = datetime.datetime.fromtimestamp(created_utc, tz=datetime.timezone.utc)
                                
                                # Create standardized post object
                                standardized_post = SocialMediaPost(
                                    platform="reddit",
                                    content_id=post_id,
                                    author=post.get('author', 'deleted'),
                                    content=post.get('title', '') + '\n' + post.get('selftext', ''),
                                    timestamp=timestamp,
                                    likes=post.get('ups', 0),
                                    comments=post.get('num_comments', 0),
                                    shares=0,  # Reddit doesn't have direct shares
                                    url=post.get('permalink', ''),
                                    is_verified_author=post.get('distinguished') is not None,  # Mod or admin posts
                                    related_assets=[asset]  # Pre-fill with the asset for this subreddit
                                )
                                
                                # Add to processing queue
                                await self.raw_data_queue.put(standardized_post)
                            
                            # Update stats
                            self.stats["platform_stats"]["reddit"]["fetched"] += len(posts)
                    
                    # Sleep briefly between requests to respect rate limits
                    await asyncio.sleep(2)
        
        except Exception as e:
            logger.error(f"Error collecting Reddit data: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _collect_telegram(self):
        """Collect data from Telegram channels"""
        logger.debug("Collecting data from Telegram")
        platform = "telegram"
        
        # For Telegram, we'll need to use web scraping as there's no public API for channel content
        try:
            # Initialize Selenium for web scraping if not already done
            if not self.selenium_initialized:
                await self.initialize_selenium()
            
            # Trading/crypto Telegram channels to monitor
            channels = {
                "BTC/USD": ["bitcoin", "cryptosignals", "binance_announcements"],
                "ETH/USD": ["ethereum", "cryptosignals", "ethtradepro"],
                # Add more asset-specific channels
            }
            
            # Process each asset and its channels
            for asset, asset_channels in channels.items():
                for channel in asset_channels:
                    # Construct URL for Telegram web
                    url = f"https://t.me/s/{channel}"
                    
                    # Load the page
                    self.driver.get(url)
                    await asyncio.sleep(5)  # Wait for page to load
                    
                    # Extract messages
                    message_elements = self.driver.find_elements_by_class_name("tgme_widget_message")
                    
                    # Process messages in reverse order (oldest first)
                    for element in reversed(message_elements[-30:]):  # Limit to last 30 messages
                        try:
                            # Extract message ID
                            message_url = element.get_attribute("data-post") or ""
                            if not message_url:
                                continue
                                
                            message_id = message_url.split("/")[-1]
                            
                            # Skip messages we've seen before
                            if await self.redis_client.exists(f"social:telegram:seen:{message_id}"):
                                continue
                            
                            # Mark as seen
                            await self.redis_client.set(f"social:telegram:seen:{message_id}", 1, ex=86400)  # 24 hour expiry
                            
                            # Extract content
                            content_element = element.find_element_by_class_name("tgme_widget_message_text")
                            content = content_element.text if content_element else ""
                            
                            # Skip empty messages
                            if not content:
                                continue
                            
                            # Extract timestamp
                            time_element = element.find_element_by_class_name("tgme_widget_message_date")
                            time_str = time_element.get_attribute("datetime") if time_element else ""
                            timestamp = datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) if time_str else datetime.datetime.now(datetime.timezone.utc)
                            
                            # Extract views
                            views_element = element.find_element_by_class_name("tgme_widget_message_views")
                            views_text = views_element.text if views_element else "0"
                            views = int(''.join(filter(str.isdigit, views_text))) if views_text else 0
                            
                            # Create standardized post object
                            standardized_post = SocialMediaPost(
                                platform="telegram",
                                content_id=message_id,
                                author=channel,  # Channel name as author
                                content=content,
                                timestamp=timestamp,
                                views=views,
                                likes=0,  # Telegram web doesn't show reactions
                                comments=0,  # Comments not visible in web view
                                shares=0,
                                url=f"https://t.me/{channel}/{message_id}",
                                related_assets=[asset]  # Pre-fill with the asset for this channel
                            )
                            
                            # Add to processing queue
                            await self.raw_data_queue.put(standardized_post)
                            
                        except Exception as e:
                            logger.warning(f"Error processing Telegram message: {str(e)}")
                    
                    # Update stats
                    self.stats["platform_stats"]["telegram"]["fetched"] += len(message_elements)
                    
                    # Sleep between channels
                    await asyncio.sleep(5)
        
        except Exception as e:
            logger.error(f"Error collecting Telegram data: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _collect_discord(self):
        """Collect data from Discord servers"""
        logger.debug("Collecting data from Discord")
        platform = "discord"
        
        # Discord requires a bot token with proper permissions
        # This is a simplified implementation and would need a dedicated Discord bot in production
        try:
            # Check if API token is available
            if platform not in self.api_keys or 'bot_token' not in self.api_keys[platform]:
                logger.warning(f"No API keys configured for {platform}")
                return
            
            token = self.api_keys[platform]['bot_token']
            
            # Discord API base URL
            base_url = "https://discord.com/api/v10"
            
            # Headers for Discord API
            headers = {
                "Authorization": f"Bot {token}",
                "Content-Type": "application/json"
            }
            
            # Channels to monitor for each asset
            # Format: {asset: [(server_id, channel_id), ...]}
            channels = {
                "BTC/USD": [
                    ("123456789012345678", "123456789012345678"),  # Example server and channel IDs
                    ("234567890123456789", "234567890123456789")
                ],
                "ETH/USD": [
                    ("123456789012345678", "123456789012345679"),
                    ("345678901234567890", "345678901234567890")
                ],
                # Add more asset-specific channels
            }
            
            # Process each asset and its channels
            for asset, asset_channels in channels.items():
                for server_id, channel_id in asset_channels:
                    # Get channel messages
                    url = f"{base_url}/channels/{channel_id}/messages"
                    params = {"limit": 50}
                    
                    # Get messages after the last seen message ID
                    last_id = await self.redis_client.get(f"social:discord:last_id:{channel_id}")
                    if last_id:
                        params["after"] = last_id
                    
                    # Make the API request
                    async with self.http_client.stream("GET", url, headers=headers, params=params) as response:
                        if response.status_code != 200:
                            logger.error(f"Discord API error: {response.status_code} - {await response.text()}")
                            continue
                        
                        # Process the response
                        messages = await response.json()
                        
                        # Check if we have messages
                        if not messages:
                            logger.debug(f"No messages found in Discord channel {channel_id}")
                            continue
                        
                        # Update last seen message ID
                        newest_id = messages[0]['id']
                        await self.redis_client.set(f"social:discord:last_id:{channel_id}", newest_id)
                        
                        # Process each message (in reverse order, oldest first)
                        for message in reversed(messages):
                            # Skip bot messages
                            if message.get('author', {}).get('bot', False):
                                continue
                            
                            # Skip messages with no content
                            content = message.get('content', '')
                            if not content:
                                continue
                            
                            # Create timestamp from Discord timestamp
                            timestamp_str = message.get('timestamp', '')
                            timestamp = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')) if timestamp_str else datetime.datetime.now(datetime.timezone.utc)
                            
                            # Create standardized post object
                            standardized_post = SocialMediaPost(
                                platform="discord",
                                content_id=message['id'],
                                author=message.get('author', {}).get('username', 'unknown'),
                                content=content,
                                timestamp=timestamp,
                                likes=len(message.get('reactions', [])),  # Count reactions as likes
                                comments=0,  # No direct comment count in Discord
                                shares=0,  # No sharing in Discord
                                related_assets=[asset]  # Pre-fill with the asset for this channel
                            )
                            
                            # Add author verification status if available
                            standardized_post.is_verified_author = False  # Default
                            
                            # Add to processing queue
                            await self.raw_data_queue.put(standardized_post)
                        
                        # Update stats
                        self.stats["platform_stats"]["discord"]["fetched"] += len(messages)
                    
                    # Sleep briefly between requests to respect rate limits
                    await asyncio.sleep(2)
                    
        except Exception as e:
            logger.error(f"Error collecting Discord data: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _collect_stocktwits(self):
        """Collect data from StockTwits"""
        logger.debug("Collecting data from StockTwits")
        platform = "stocktwits"
        
        try:
            # StockTwits API base URL
            base_url = "https://api.stocktwits.com/api/2"
            
            # Assets to monitor (with StockTwits symbol format)
            symbols = {
                "BTC/USD": "BTC.X",
                "ETH/USD": "ETH.X",
                "EUR/USD": "EURUSD",
                "JPY/USD": "USDJPY",
                # Add more assets
            }
            
            # Get data for each symbol
            for asset, symbol in symbols.items():
                # Get symbol streams
                url = f"{base_url}/streams/symbol/{symbol}.json"
                
                # Make the API request
                async with self.http_client.stream("GET", url) as response:
                    if response.status_code != 200:
                        logger.error(f"StockTwits API error: {response.status_code} - {await response.text()}")
                        continue
                    
                    # Process the response
                    response_data = await response.json()
                    
                    if 'messages' not in response_data:
                        logger.debug(f"No messages found for symbol {symbol}")
                        continue
                    
                    messages = response_data['messages']
                    
                    # Process each message
                    for message in messages:
                        # Skip messages we've seen before
                        message_id = str(message.get('id', 0))
                        if await self.redis_client.exists(f"social:stocktwits:seen:{message_id}"):
                            continue
                        
                        # Mark as seen
                        await self.redis_client.set(f"social:stocktwits:seen:{message_id}", 1, ex=86400)  # 24 hour expiry
                        
                        # Get user data
                        user = message.get('user', {})
                        
                        # Create timestamp
                        created_at = message.get('created_at', '')
                        timestamp = datetime.datetime.fromisoformat(created_at.replace('Z', '+00:00')) if created_at else datetime.datetime.now(datetime.timezone.utc)
                        
                        # Create standardized post object
                        standardized_post = SocialMediaPost(
                            platform="stocktwits",
                            content_id=message_id,
                            author=user.get('username', 'unknown'),
                            content=message.get('body', ''),
                            timestamp=timestamp,
                            likes=message.get('likes', {}).get('total', 0),
                            comments=message.get('conversation', {}).get('total', 0),
                            shares=message.get('reshares', {}).get('total', 0) if 'reshares' in message else 0,
                            is_verified_author=user.get('official', False),
                            author_followers=user.get('followers', 0),
                            related_assets=[asset],  # Pre-fill with the asset for this symbol
                            extracted_symbols=[s.get('symbol', '') for s in message.get('symbols', [])]
                        )
                        
                        # Add to processing queue
                        await self.raw_data_queue.put(standardized_post)
                        
                    # Update stats
                    self.stats["platform_stats"]["stocktwits"]["fetched"] += len(messages)
                    
                # Sleep briefly between symbols to respect rate limits
                await asyncio.sleep(2)
                
        except Exception as e:
            logger.error(f"Error collecting StockTwits data: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _collect_tradingview(self):
        """Collect data from TradingView Ideas and forums"""
        logger.debug("Collecting data from TradingView")
        platform = "tradingview"
        
        try:
            # For TradingView, we'll need to use web scraping
            if not self.selenium_initialized:
                await self.initialize_selenium()
            
            # Assets to monitor (with TradingView symbol format)
            symbols = {
                "BTC/USD": "BTCUSD",
                "ETH/USD": "ETHUSD",
                "EUR/USD": "EURUSD",
                "JPY/USD": "USDJPY",
                # Add more assets
            }
            
            # Process each symbol
            for asset, symbol in symbols.items():
                # Construct URL for TradingView ideas
                url = f"https://www.tradingview.com/symbols/{symbol}/ideas/"
                
                # Load the page
                self.driver.get(url)
                await asyncio.sleep(5)  # Wait for page to load
                
                # Extract idea cards
                idea_elements = self.driver.find_elements_by_class_name("tv-widget-idea")
                
                # Process each idea
                for element in idea_elements[:20]:  # Limit to first 20 ideas
                    try:
                        # Extract idea ID
                        idea_link = element.find_element_by_css_selector(".tv-widget-idea__title a")
                        idea_url = idea_link.get_attribute("href") or ""
                        idea_id = idea_url.split("/")[-2] if idea_url else ""
                        
                        # Skip if no ID found
                        if not idea_id:
                            continue
                        
                        # Skip ideas we've seen before
                        if await self.redis_client.exists(f"social:tradingview:seen:{idea_id}"):
                            continue
                        
                        # Mark as seen
                        await self.redis_client.set(f"social:tradingview:seen:{idea_id}", 1, ex=86400*7)  # 7 day expiry
                        
                        # Extract content
                        title = idea_link.text
                        
                        # Try to extract description
                        description_element = element.find_element_by_class_name("tv-widget-idea__description-row")
                        description = description_element.text if description_element else ""
                        
                        content = f"{title}\n{description}"
                        
                        # Extract author
                        author_element = element.find_element_by_css_selector(".tv-widget-idea__author-username")
                        author = author_element.text if author_element else "unknown"
                        
                        # Extract likes and comments
                        likes_element = element.find_element_by_css_selector(".tv-social-stats__item--likes .tv-social-stats__count")
                        likes = int(likes_element.text) if likes_element else 0
                        
                        comments_element = element.find_element_by_css_selector(".tv-social-stats__item--comments .tv-social-stats__count")
                        comments = int(comments_element.text) if comments_element else 0
                        
                        # Use current time as timestamp (TradingView doesn't show exact timestamps on ideas list)
                        timestamp = datetime.datetime.now(datetime.timezone.utc)
                        
                        # Create standardized post object
                        standardized_post = SocialMediaPost(
                            platform="tradingview",
                            content_id=idea_id,
                            author=author,
                            content=content,
                            timestamp=timestamp,
                            likes=likes,
                            comments=comments,
                            shares=0,  # TradingView doesn't have direct shares
                            url=idea_url,
                            related_assets=[asset],  # Pre-fill with the asset for this symbol
                            extracted_symbols=[symbol]
                        )
                        
                        # Add to processing queue
                        await self.raw_data_queue.put(standardized_post)
                        
                    except Exception as e:
                        logger.warning(f"Error processing TradingView idea: {str(e)}")
                
                # Update stats
                self.stats["platform_stats"]["tradingview"]["fetched"] += len(idea_elements)
                
                # Sleep between symbols
                await asyncio.sleep(5)
                
        except Exception as e:
            logger.error(f"Error collecting TradingView data: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get current statistics for the social media feed"""
        return self.stats
    
    async def reset_stats(self):
        """Reset statistics counters"""
        self.stats = {
            "posts_processed": 0,
            "relevant_posts": 0,
            "high_impact_posts": 0,
            "platform_stats": {p: {"fetched": 0, "relevant": 0} for p in self.platforms},
            "asset_mentions": {},
            "influencer_activity": {}
        }
        logger.info("Social Media Feed statistics reset")
