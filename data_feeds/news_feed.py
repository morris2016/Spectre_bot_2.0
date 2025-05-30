#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
News Feed Module

This module provides advanced news gathering, parsing, and sentiment analysis capabilities
for the QuantumSpectre Elite Trading System. It integrates with multiple news sources,
performs real-time sentiment analysis, and identifies market-moving events to provide
a competitive edge in trading decisions.
"""

import os
import re
import json
import time
import asyncio
import aiohttp
import hashlib
import datetime
import threading
from typing import Dict, List, Set, Any, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import feedparser
import trafilatura
from newspaper import Article, Source
from dateutil import parser as date_parser

# Internal imports
from common.logger import get_logger
from common.utils import (
    rate_limited, async_retry_with_backoff, cache_with_ttl,
    safe_execute, parse_datetime, calculate_checksum,
    safe_nltk_download,
)
from common.constants import NEWS_SOURCES, ASSET_KEYWORDS, MARKET_IMPACT_PHRASES
from common.exceptions import NewsFeedError, NewsParsingError, NewsSourceUnavailableError
from data_feeds.base_feed import BaseFeed
from common.utils import safe_nltk_download

# Configure NLTK and transformers without network calls
# Configure NLTK and transformers
safe_nltk_download('vader_lexicon')
safe_nltk_download('punkt')
safe_nltk_download('stopwords')

class NewsFeed(BaseFeed):
    """
    Advanced news gathering and analysis feed for the QuantumSpectre Elite Trading System.
    
    This class provides comprehensive news intelligence from multiple sources including:
    - Major financial news websites
    - Press releases
    - Economic calendars
    - Central bank communications
    - Regulatory filings
    - Social media monitoring for market sentiment
    
    It includes sophisticated NLP capabilities for:
    - Sentiment analysis specific to financial markets
    - Market impact prediction
    - Asset-specific news filtering
    - Event categorization
    - Breaking news detection
    """
    
    def __init__(self, config: Dict[str, Any], cache_client=None, db_client=None):
        """
        Initialize the NewsFeed with configuration and dependencies.
        
        Args:
            config: Configuration dictionary for the news feed
            cache_client: Redis or similar cache client for caching news data
            db_client: Database client for persistent storage of news data
        """
        super().__init__(name="news_feed", config=config, cache_client=cache_client, db_client=db_client)
        
        self.logger = get_logger("news_feed")
        self.logger.info("Initializing NewsFeed with advanced NLP capabilities")
        
        # Sources configuration
        self.sources = self.config.get("sources", NEWS_SOURCES)
        self.source_credentials = self.config.get("source_credentials", {})
        self.premium_apis = self.config.get("premium_apis", {})
        
        # NLP configuration
        self.use_advanced_nlp = self.config.get("use_advanced_nlp", True)
        self.nlp_batch_size = self.config.get("nlp_batch_size", 16)
        self.sentiment_threshold = self.config.get("sentiment_threshold", 0.2)
        
        # Asset mapping
        self.asset_keywords = self.config.get("asset_keywords", ASSET_KEYWORDS)
        self.market_impact_phrases = self.config.get("market_impact_phrases", MARKET_IMPACT_PHRASES)
        
        # Cache configuration
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # 1 hour default
        self.news_history_lookback = self.config.get("news_history_lookback", 86400 * 7)  # 7 days
        
        # Initialize NLP components
        self._init_nlp_components()
        
        # Initialize crawlers and scrapers
        self._init_crawlers()
        
        # Threading and async setup
        self.executor = ThreadPoolExecutor(max_workers=self.config.get("max_workers", 10))
        self.article_queue = asyncio.Queue(maxsize=1000)
        self.processed_article_queue = asyncio.Queue(maxsize=1000)
        self.seen_urls = set()
        self.article_checksums = set()
        
        # Initialize internal storage
        self.news_cache = {}
        self.breaking_news = []
        self.recent_sentiment = {}
        self.event_calendar = {}
        
        # Feed state
        self.is_running = False
        self.tasks = []
        
        self.logger.info("NewsFeed initialization complete")
    
    def _init_nlp_components(self):
        """Initialize NLP components for sentiment analysis and news categorization"""
        self.logger.info("Initializing NLP components")
        
        # Basic sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Advanced NLP components if enabled
        if self.use_advanced_nlp:
            try:
                # Financial sentiment analysis model
                self.logger.info("Loading advanced financial sentiment model")
                model_name = "yiyanghkust/finbert-tone"
                self.financial_sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.financial_sentiment_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.financial_sentiment = pipeline(
                    "text-classification", 
                    model=self.financial_sentiment_model, 
                    tokenizer=self.financial_sentiment_tokenizer
                )
                
                # News categorization model
                self.logger.info("Loading news categorization model")
                self.news_categorizer = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
                
                # Named entity recognition for asset identification
                self.logger.info("Loading NER model for asset identification")
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dslim/bert-base-NER"
                )
                
                self.logger.info("Advanced NLP components loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load advanced NLP components: {e}")
                self.logger.info("Falling back to basic sentiment analysis")
                self.use_advanced_nlp = False
    
    def _init_crawlers(self):
        """Initialize web crawlers for different news sources"""
        self.logger.info("Initializing news crawlers and scrapers")
        
        # Create crawlers for different source types
        self.crawlers = {
            'rss': self._crawl_rss_feed,
            'api': self._crawl_api_source,
            'website': self._crawl_website,
            'twitter': self._crawl_twitter,
            'reddit': self._crawl_reddit,
            'sec_filings': self._crawl_sec_filings,
            'economic_calendar': self._crawl_economic_calendar,
        }
        
        # Custom headers for web requests to avoid detection
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Specialized scrapers for specific sites
        self.specialized_scrapers = {
            'bloomberg': self._scrape_bloomberg,
            'reuters': self._scrape_reuters,
            'wsj': self._scrape_wsj,
            'ft': self._scrape_ft,
            'cnbc': self._scrape_cnbc,
            'investing': self._scrape_investing,
            'marketwatch': self._scrape_marketwatch,
            'seekingalpha': self._scrape_seeking_alpha,
        }
        
        self.logger.info("News crawlers and scrapers initialized")
    
    async def start(self):
        """Start the news feed processing"""
        if self.is_running:
            self.logger.warning("NewsFeed is already running")
            return
        
        self.logger.info("Starting NewsFeed service")
        self.is_running = True
        
        # Start worker tasks
        self.tasks = [
            asyncio.create_task(self._article_processor()),
            asyncio.create_task(self._sentiment_analyzer()),
        ]
        
        # Start source crawlers
        for source in self.sources:
            source_type = source.get("type", "rss")
            crawler = self.crawlers.get(source_type)
            if crawler:
                self.tasks.append(asyncio.create_task(crawler(source)))
            else:
                self.logger.warning(f"Unknown source type: {source_type}")
        
        # Start periodic tasks
        self.tasks.append(asyncio.create_task(self._periodic_cleanup()))
        self.tasks.append(asyncio.create_task(self._update_economic_calendar()))
        
        self.logger.info(f"NewsFeed started with {len(self.tasks)} tasks")
    
    async def stop(self):
        """Stop the news feed processing"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping NewsFeed service")
        self.is_running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks = []
        
        # Close the thread pool
        self.executor.shutdown(wait=True)
        
        self.logger.info("NewsFeed service stopped")
    
    async def get_data(self, asset: str = None, start_time: int = None, end_time: int = None) -> List[Dict[str, Any]]:
        """
        Get news data for a specific asset within a time range.
        
        Args:
            asset: Asset symbol or name (optional)
            start_time: Start timestamp in milliseconds (optional)
            end_time: End timestamp in milliseconds (optional)
            
        Returns:
            List of news articles with sentiment and impact scores
        """
        self.logger.debug(f"Getting news data for asset: {asset}, time range: {start_time} to {end_time}")
        
        # Use current time if end_time not provided
        if end_time is None:
            end_time = int(time.time() * 1000)
        
        # Use lookback period if start_time not provided
        if start_time is None:
            start_time = end_time - (self.news_history_lookback * 1000)
        
        # Query database for historical news
        if self.db_client:
            query = {
                "timestamp": {"$gte": start_time, "$lte": end_time}
            }
            
            if asset:
                # Get related keywords for the asset
                keywords = self.asset_keywords.get(asset, [asset])
                query["$or"] = [
                    {"assets": {"$in": [asset]}},
                    {"keywords": {"$in": keywords}}
                ]
                
            try:
                news_data = await self.db_client.find(
                    "news_articles", 
                    query,
                    sort=[("timestamp", -1)]
                )
                return news_data
            except Exception as e:
                self.logger.error(f"Error querying database for news: {e}")
        
        # Fall back to in-memory cache if database query fails
        result = []
        for article_id, article in self.news_cache.items():
            if article['timestamp'] >= start_time and article['timestamp'] <= end_time:
                if not asset or asset in article.get('assets', []) or any(kw in article.get('keywords', []) for kw in self.asset_keywords.get(asset, [asset])):
                    result.append(article)
        
        # Sort by timestamp, newest first
        result.sort(key=lambda x: x['timestamp'], reverse=True)
        return result
    
    async def get_sentiment(self, asset: str, timeframe: str = '1d') -> Dict[str, Any]:
        """
        Get aggregated sentiment data for a specific asset over a timeframe.
        
        Args:
            asset: Asset symbol or name
            timeframe: Timeframe for sentiment aggregation ('1h', '4h', '1d', '1w')
            
        Returns:
            Dictionary with sentiment metrics and scores
        """
        self.logger.debug(f"Getting sentiment for asset: {asset}, timeframe: {timeframe}")
        
        # Convert timeframe to milliseconds
        timeframe_ms = {
            '1h': 3600 * 1000,
            '4h': 14400 * 1000,
            '1d': 86400 * 1000,
            '1w': 604800 * 1000,
        }.get(timeframe, 86400 * 1000)  # Default to 1 day
        
        end_time = int(time.time() * 1000)
        start_time = end_time - timeframe_ms
        
        # Get news data for the specified period
        news_data = await self.get_data(asset, start_time, end_time)
        
        if not news_data:
            return {
                'asset': asset,
                'timeframe': timeframe,
                'sentiment_score': 0,
                'sentiment_label': 'neutral',
                'article_count': 0,
                'impact_score': 0,
                'breaking_news': False,
                'sentiment_trend': 'stable',
            }
        
        # Calculate aggregated sentiment
        sentiment_values = [article.get('sentiment', {}).get('score', 0) for article in news_data]
        impact_values = [article.get('impact_score', 0) for article in news_data]
        
        avg_sentiment = sum(sentiment_values) / len(sentiment_values)
        avg_impact = sum(impact_values) / len(impact_values)
        
        # Determine sentiment label
        sentiment_label = 'neutral'
        if avg_sentiment > 0.2:
            sentiment_label = 'positive'
        elif avg_sentiment < -0.2:
            sentiment_label = 'negative'
        
        # Check for breaking news
        has_breaking = any(article.get('breaking', False) for article in news_data)
        
        # Calculate sentiment trend by comparing to previous period
        previous_start = start_time - timeframe_ms
        previous_news = await self.get_data(asset, previous_start, start_time)
        
        sentiment_trend = 'stable'
        if previous_news:
            previous_sentiment = sum(article.get('sentiment', {}).get('score', 0) for article in previous_news) / len(previous_news)
            if avg_sentiment > previous_sentiment + 0.2:
                sentiment_trend = 'improving'
            elif avg_sentiment < previous_sentiment - 0.2:
                sentiment_trend = 'deteriorating'
        
        return {
            'asset': asset,
            'timeframe': timeframe,
            'sentiment_score': round(avg_sentiment, 3),
            'sentiment_label': sentiment_label,
            'article_count': len(news_data),
            'impact_score': round(avg_impact, 3),
            'breaking_news': has_breaking,
            'sentiment_trend': sentiment_trend,
            'latest_headlines': [article.get('title', '') for article in news_data[:5]],
            'top_keywords': self._extract_top_keywords(news_data),
        }
    
    async def get_economic_events(self, start_time: int = None, end_time: int = None) -> List[Dict[str, Any]]:
        """
        Get economic events within a time range.
        
        Args:
            start_time: Start timestamp in milliseconds (optional)
            end_time: End timestamp in milliseconds (optional)
            
        Returns:
            List of economic events with impact scores and affected assets
        """
        self.logger.debug(f"Getting economic events for time range: {start_time} to {end_time}")
        
        # Use current time if end_time not provided
        if end_time is None:
            end_time = int(time.time() * 1000)
        
        # Use next 7 days if start_time not provided
        if start_time is None:
            start_time = int(time.time() * 1000)
            
        # Filter events within the time range
        events = []
        for event_id, event in self.event_calendar.items():
            if event['timestamp'] >= start_time and event['timestamp'] <= end_time:
                events.append(event)
        
        # Sort by timestamp
        events.sort(key=lambda x: x['timestamp'])
        return events
    
    def get_breaking_news(self) -> List[Dict[str, Any]]:
        """Get recent breaking news articles"""
        # Return copy of the breaking news list to avoid modification
        return self.breaking_news.copy()
    
    async def _article_processor(self):
        """Process articles from the queue"""
        self.logger.info("Starting article processor")
        
        while self.is_running:
            try:
                # Get article from queue
                article = await self.article_queue.get()
                
                # Process article
                processed = await self._process_article(article)
                if processed:
                    # Add to processed queue for sentiment analysis
                    await self.processed_article_queue.put(processed)
                    
                    # Store in cache
                    self.news_cache[processed['id']] = processed
                    
                    # Store in database if available
                    if self.db_client:
                        try:
                            await self.db_client.insert_one("news_articles", processed)
                        except Exception as e:
                            self.logger.error(f"Error storing article in database: {e}")
                
                # Mark task as done
                self.article_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in article processor: {e}")
                
        self.logger.info("Article processor stopped")
    
    async def _sentiment_analyzer(self):
        """Analyze sentiment of processed articles"""
        self.logger.info("Starting sentiment analyzer")
        
        while self.is_running:
            try:
                # Get article from queue
                article = await self.processed_article_queue.get()
                
                # Analyze sentiment
                await self._analyze_article_sentiment(article)
                
                # Check if breaking news
                is_breaking = self._is_breaking_news(article)
                if is_breaking:
                    article['breaking'] = True
                    self.breaking_news.append(article)
                    
                    # Trim breaking news list
                    if len(self.breaking_news) > 50:
                        self.breaking_news = self.breaking_news[-50:]
                
                # Update recent sentiment for assets
                for asset in article.get('assets', []):
                    if asset not in self.recent_sentiment:
                        self.recent_sentiment[asset] = []
                    
                    self.recent_sentiment[asset].append({
                        'timestamp': article['timestamp'],
                        'sentiment': article.get('sentiment', {}).get('score', 0),
                        'impact': article.get('impact_score', 0)
                    })
                    
                    # Keep only recent sentiment
                    current_time = int(time.time() * 1000)
                    cutoff = current_time - (24 * 3600 * 1000)  # 24 hours
                    self.recent_sentiment[asset] = [
                        s for s in self.recent_sentiment[asset] if s['timestamp'] >= cutoff
                    ]
                
                # Mark task as done
                self.processed_article_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in sentiment analyzer: {e}")
                
        self.logger.info("Sentiment analyzer stopped")
    
    async def _periodic_cleanup(self):
        """Periodically clean up cached data"""
        self.logger.info("Starting periodic cleanup task")
        
        while self.is_running:
            try:
                # Sleep for cleanup interval
                await asyncio.sleep(3600)  # 1 hour
                
                # Clean up news cache
                current_time = int(time.time() * 1000)
                cutoff = current_time - (self.news_history_lookback * 1000)
                
                # Remove old articles from cache
                old_articles = [
                    article_id for article_id, article in self.news_cache.items()
                    if article['timestamp'] < cutoff
                ]
                
                for article_id in old_articles:
                    del self.news_cache[article_id]
                
                # Clean up seen URLs
                self.seen_urls = set()
                self.article_checksums = set()
                
                self.logger.info(f"Cleaned up {len(old_articles)} old articles")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")
        
        self.logger.info("Periodic cleanup task stopped")
    
    async def _update_economic_calendar(self):
        """Periodically update the economic calendar"""
        self.logger.info("Starting economic calendar update task")
        
        while self.is_running:
            try:
                # Update economic calendar
                await self._fetch_economic_calendar()
                
                # Sleep for update interval (4 hours)
                await asyncio.sleep(14400)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error updating economic calendar: {e}")
                # Sleep after error
                await asyncio.sleep(300)  # 5 minutes
                
        self.logger.info("Economic calendar update task stopped")
    
    async def _fetch_economic_calendar(self):
        """Fetch economic calendar data from various sources"""
        self.logger.info("Fetching economic calendar data")
        
        # Sources for economic calendar
        sources = [
            {
                'name': 'investing_com',
                'url': 'https://www.investing.com/economic-calendar/',
                'function': self._parse_investing_calendar
            },
            {
                'name': 'forexfactory',
                'url': 'https://www.forexfactory.com/calendar',
                'function': self._parse_forexfactory_calendar
            },
            {
                'name': 'tradingeconomics',
                'url': 'https://tradingeconomics.com/calendar',
                'function': self._parse_tradingeconomics_calendar
            }
        ]
        
        # Premium API sources if available
        if 'economic_calendar' in self.premium_apis:
            sources.append({
                'name': 'premium_api',
                'url': self.premium_apis['economic_calendar']['url'],
                'function': self._parse_premium_calendar_api,
                'api_key': self.premium_apis['economic_calendar'].get('api_key')
            })
        
        # Fetch and parse from each source
        async with aiohttp.ClientSession() as session:
            for source in sources:
                try:
                    self.logger.debug(f"Fetching calendar from {source['name']}")
                    
                    if source.get('api_key'):
                        # API request with key
                        headers = {
                            'Authorization': f"Bearer {source['api_key']}",
                            'Content-Type': 'application/json'
                        }
                        
                        async with session.get(source['url'], headers=headers) as response:
                            if response.status == 200:
                                data = await response.json()
                                events = await source['function'](data)
                                self._update_event_calendar(events, source['name'])
                            else:
                                self.logger.warning(f"Failed to fetch calendar from {source['name']}: {response.status}")
                    else:
                        # Regular web scraping
                        async with session.get(source['url'], headers=self.headers) as response:
                            if response.status == 200:
                                html = await response.text()
                                events = await source['function'](html)
                                self._update_event_calendar(events, source['name'])
                            else:
                                self.logger.warning(f"Failed to fetch calendar from {source['name']}: {response.status}")
                
                except Exception as e:
                    self.logger.error(f"Error fetching calendar from {source['name']}: {e}")
        
        self.logger.info(f"Economic calendar updated with {len(self.event_calendar)} events")
    
    def _update_event_calendar(self, events: List[Dict[str, Any]], source: str):
        """Update the event calendar with new events"""
        for event in events:
            # Generate unique ID for the event
            event_id = hashlib.md5(f"{event['title']}_{event['timestamp']}".encode()).hexdigest()
            
            # Add source information
            event['source'] = source
            
            # Update existing event or add new one
            if event_id in self.event_calendar:
                self.event_calendar[event_id].update(event)
            else:
                self.event_calendar[event_id] = event
    
    async def _process_article(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a news article to extract content, metadata, and relevance.
        
        Args:
            article: Raw article data with URL and basic metadata
            
        Returns:
            Processed article with extracted content and metadata, or None if invalid
        """
        url = article.get('url')
        if not url or url in self.seen_urls:
            return None
        
        self.seen_urls.add(url)
        
        try:
            # Extract full text if not already present
            if 'content' not in article or not article['content']:
                content = await self._extract_article_content(article)
                if not content:
                    return None
                article['content'] = content
            
            # Generate checksum to detect duplicates
            content_checksum = calculate_checksum(article['content'])
            if content_checksum in self.article_checksums:
                return None
            
            self.article_checksums.add(content_checksum)
            
            # Ensure we have basic metadata
            if 'title' not in article or not article['title']:
                article['title'] = self._extract_title(article['content'])
            
            if 'timestamp' not in article or not article['timestamp']:
                article['timestamp'] = article.get('published', int(time.time() * 1000))
            
            # Generate unique ID
            article_id = hashlib.md5(f"{url}_{article['timestamp']}".encode()).hexdigest()
            article['id'] = article_id
            
            # Extract relevant assets mentioned in the article
            article['assets'] = self._extract_assets(article)
            
            # Extract keywords
            article['keywords'] = self._extract_keywords(article)
            
            # Calculate initial relevance score
            article['relevance_score'] = self._calculate_relevance(article)
            
            # Add source domain
            article['domain'] = urlparse(url).netloc
            
            # Calculate initial impact score based on source reputation and content
            article['impact_score'] = self._calculate_impact_score(article)
            
            return article
            
        except Exception as e:
            self.logger.error(f"Error processing article {url}: {e}")
            return None
    
    async def _extract_article_content(self, article: Dict[str, Any]) -> Optional[str]:
        """Extract full text content from an article URL"""
        url = article.get('url')
        if not url:
            return None
        
        # Check if specialized scraper exists for this domain
        domain = urlparse(url).netloc
        domain_key = next((k for k in self.specialized_scrapers.keys() if k in domain), None)
        
        if domain_key:
            # Use specialized scraper
            scraper = self.specialized_scrapers[domain_key]
            return await scraper(url)
        
        # Use general extraction methods
        try:
            # Method 1: newspaper3k
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(self.executor, self._extract_with_newspaper, url)
            if content:
                return content
            
            # Method 2: trafilatura
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        content = await loop.run_in_executor(
                            self.executor, 
                            lambda h: trafilatura.extract(h), 
                            html
                        )
                        if content:
                            return content
                        
                        # Method 3: BeautifulSoup fallback
                        content = await loop.run_in_executor(
                            self.executor,
                            self._extract_with_soup,
                            html
                        )
                        return content
                    
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {e}")
        
        return None
    
    def _extract_with_newspaper(self, url: str) -> Optional[str]:
        """Extract article content using newspaper3k"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except:
            return None
    
    def _extract_with_soup(self, html: str) -> str:
        """Extract article content using BeautifulSoup as fallback"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        
        # Find main content area
        main_content = soup.find('article') or soup.find('main') or soup.find(id=re.compile('content|article|main', re.I))
        
        if main_content:
            # Extract paragraphs from main content
            paragraphs = main_content.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs])
            return content
        
        # Fallback to all paragraphs
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 100])
        return content
    
    def _extract_title(self, content: str) -> str:
        """Extract a title from content if none is provided"""
        # Take first non-empty line as title
        lines = content.split('\n')
        for line in lines:
            clean_line = line.strip()
            if clean_line and len(clean_line) < 200:
                return clean_line
        
        # Or take first 100 characters
        return content[:100].replace('\n', ' ').strip() + '...'
    
    def _extract_assets(self, article: Dict[str, Any]) -> List[str]:
        """Extract mentioned assets from article content"""
        content = article.get('content', '')
        title = article.get('title', '')
        combined = f"{title} {content}"
        
        assets = []
        
        # Check for explicit asset mentions across all asset keywords
        for asset, keywords in self.asset_keywords.items():
            for keyword in keywords:
                pattern = rf'\b{re.escape(keyword)}\b'
                if re.search(pattern, combined, re.IGNORECASE):
                    assets.append(asset)
                    break
        
        # Advanced NER for asset extraction if enabled
        if self.use_advanced_nlp:
            try:
                # Use NER to find organization and currency mentions
                entities = self.ner_pipeline(combined[:512])  # Process first part of text
                
                for entity in entities:
                    if entity['entity'].startswith('B-ORG') or entity['entity'].startswith('I-ORG'):
                        # Check if organization is a known asset
                        org_name = entity['word'].strip('#').strip('@')
                        for asset, keywords in self.asset_keywords.items():
                            if any(org_name.lower() in kw.lower() for kw in keywords):
                                assets.append(asset)
            except Exception as e:
                self.logger.error(f"Error in NER-based asset extraction: {e}")
        
        # Remove duplicates and return
        return list(set(assets))
    
    def _extract_keywords(self, article: Dict[str, Any]) -> List[str]:
        """Extract relevant keywords from article content"""
        content = article.get('content', '')
        title = article.get('title', '')
        
        # Basic keyword extraction from title
        title_words = set(w.lower() for w in re.findall(r'\b[A-Za-z][A-Za-z-]{2,}\b', title))
        
        # Extract keywords from first few paragraphs of content
        paragraphs = content.split('\n')[:3]
        short_content = ' '.join(paragraphs)
        
        content_words = set(w.lower() for w in re.findall(r'\b[A-Za-z][A-Za-z-]{5,}\b', short_content))
        
        # Combine and filter common words
        common_words = {
            'press', 'release', 'announces', 'today', 'reported', 'according', 
            'business', 'company', 'market', 'price', 'share', 'stock', 'trade',
            'million', 'billion', 'percent', 'reuters', 'bloomberg', 'financial',
            'investor', 'analyst', 'report', 'statement', 'quarter', 'year'
        }
        
        keywords = (title_words | content_words) - common_words
        
        # Add market impact phrases if present
        for phrase in self.market_impact_phrases:
            if phrase.lower() in content.lower() or phrase.lower() in title.lower():
                keywords.add(phrase.lower())
        
        return list(keywords)
    
    def _calculate_relevance(self, article: Dict[str, Any]) -> float:
        """Calculate relevance score for an article"""
        score = 0.0
        
        # Higher score for articles mentioning specific assets
        assets = article.get('assets', [])
        score += min(len(assets) * 0.2, 0.6)
        
        # Higher score for recent articles
        age_hours = (int(time.time() * 1000) - article.get('timestamp', 0)) / (3600 * 1000)
        recency_score = max(0, 1 - (age_hours / 24))  # Highest for < 24 hours
        score += recency_score * 0.2
        
        # Higher score for articles with market impact phrases
        content = article.get('content', '').lower()
        impact_phrases_count = sum(1 for phrase in self.market_impact_phrases if phrase.lower() in content)
        score += min(impact_phrases_count * 0.05, 0.2)
        
        # Source reputation factor
        domain = urlparse(article.get('url', '')).netloc
        reputation_score = {
            'bloomberg.com': 0.95,
            'reuters.com': 0.95,
            'wsj.com': 0.9,
            'ft.com': 0.9,
            'cnbc.com': 0.85,
            'marketwatch.com': 0.85,
            'investing.com': 0.8,
            'seekingalpha.com': 0.75,
            'fool.com': 0.7,
            'yahoo.com': 0.7,
            'finance.yahoo.com': 0.8,
        }.get(domain, 0.5)
        
        score += reputation_score * 0.2
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def _calculate_impact_score(self, article: Dict[str, Any]) -> float:
        """Calculate potential market impact score"""
        # Start with base score from relevance
        impact = article.get('relevance_score', 0) * 0.5
        
        # Source credibility factor
        domain = urlparse(article.get('url', '')).netloc
        credibility = {
            'bloomberg.com': 0.95,
            'reuters.com': 0.95,
            'wsj.com': 0.9,
            'ft.com': 0.9,
            'cnbc.com': 0.85,
            'federalreserve.gov': 1.0,
            'ecb.europa.eu': 1.0,
            'sec.gov': 1.0,
            'whitehouse.gov': 0.95,
            'investing.com': 0.7,
        }.get(domain, 0.5)
        
        impact += credibility * 0.2
        
        # Check for high-impact phrases
        content = article.get('content', '').lower()
        title = article.get('title', '').lower()
        
        high_impact_phrases = [
            'announces acquisition', 'merger agreement', 'bankruptcy', 'unexpected profit',
            'profit warning', 'exceeds expectations', 'fails to meet', 'sec investigation',
            'fraud allegations', 'major announcement', 'surprise decision', 'rate hike',
            'rate cut', 'federal reserve', 'central bank', 'surprise announcement',
            'major discovery', 'breakthrough', 'recall', 'lawsuit', 'settlement',
            'ceo resigns', 'regulatory approval', 'denied approval', 'major contract',
            'missed earnings', 'beat earnings', 'guidance raised', 'guidance lowered',
            'stock split', 'dividend cut', 'dividend increase', 'buyback', 'major layoffs'
        ]
        
        # Higher weight for phrases in title
        title_impact = sum(0.15 for phrase in high_impact_phrases if phrase in title)
        content_impact = sum(0.05 for phrase in high_impact_phrases if phrase in content)
        
        impact += min(title_impact, 0.3) + min(content_impact, 0.2)
        
        # Consider if article mentions multiple assets (cross-market impact)
        assets = article.get('assets', [])
        if len(assets) > 3:
            impact += 0.1
        
        return min(impact, 1.0)
    
    async def _analyze_article_sentiment(self, article: Dict[str, Any]):
        """Analyze sentiment of an article and update the article object"""
        try:
            title = article.get('title', '')
            content = article.get('content', '')
            
            # Start with simple VADER sentiment for baseline
            title_sentiment = self.vader_analyzer.polarity_scores(title)
            
            # Use the first few paragraphs for content sentiment (faster, still accurate)
            paragraphs = content.split('\n')[:5]
            short_content = ' '.join(paragraphs)
            content_sentiment = self.vader_analyzer.polarity_scores(short_content)
            
            # Combine title and content sentiment, with title weighted more heavily
            basic_score = title_sentiment['compound'] * 0.6 + content_sentiment['compound'] * 0.4
            
            sentiment_result = {
                'score': basic_score,
                'magnitude': abs(basic_score),
                'label': 'neutral',
            }
            
            # Label based on score
            if basic_score > 0.2:
                sentiment_result['label'] = 'positive'
            elif basic_score < -0.2:
                sentiment_result['label'] = 'negative'
            
            # Use advanced financial sentiment model if enabled
            if self.use_advanced_nlp:
                try:
                    # Process title and key paragraphs for financial sentiment
                    fin_sentiment = self.financial_sentiment(short_content)
                    
                    # Map finbert output to normalized score
                    label_map = {
                        'positive': 0.7,
                        'neutral': 0.0,
                        'negative': -0.7
                    }
                    
                    fin_score = label_map.get(fin_sentiment[0]['label'], 0.0)
                    
                    # Blend basic and financial sentiment (weighted toward financial)
                    sentiment_result['score'] = basic_score * 0.3 + fin_score * 0.7
                    sentiment_result['financial_label'] = fin_sentiment[0]['label']
                    sentiment_result['confidence'] = fin_sentiment[0]['score']
                    
                    # Update sentiment label
                    if sentiment_result['score'] > 0.2:
                        sentiment_result['label'] = 'positive'
                    elif sentiment_result['score'] < -0.2:
                        sentiment_result['label'] = 'negative'
                    else:
                        sentiment_result['label'] = 'neutral'
                
                except Exception as e:
                    self.logger.error(f"Error in advanced sentiment analysis: {e}")
            
            article['sentiment'] = sentiment_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            # Set default neutral sentiment
            article['sentiment'] = {
                'score': 0,
                'magnitude': 0,
                'label': 'neutral'
            }
    
    def _is_breaking_news(self, article: Dict[str, Any]) -> bool:
        """Determine if an article represents breaking news"""
        # Check if article is recent (< 1 hour old)
        age_ms = int(time.time() * 1000) - article.get('timestamp', 0)
        if age_ms > 3600 * 1000:  # Older than 1 hour
            return False
        
        # Check for breaking news indicators in title
        title = article.get('title', '').lower()
        breaking_indicators = [
            'breaking', 'alert', 'just in', 'urgent', 'flash', 'update', 
            'developing', 'exclusive'
        ]
        
        if any(indicator in title for indicator in breaking_indicators):
            return True
        
        # Check for significant market impact score
        if article.get('impact_score', 0) > 0.8:
            return True
        
        # Check for extreme sentiment
        sentiment = article.get('sentiment', {}).get('score', 0)
        if abs(sentiment) > 0.7:
            return True
        
        # Check for price action keywords with significant assets
        price_action_phrases = [
            'plunges', 'soars', 'crashes', 'surges', 'tumbles', 'rallies',
            'collapses', 'skyrockets', 'tanks', 'jumps', 'dives', 'spikes'
        ]
        
        if any(phrase in title for phrase in price_action_phrases) and len(article.get('assets', [])) > 0:
            return True
        
        return False
    
    def _extract_top_keywords(self, articles: List[Dict[str, Any]], count: int = 10) -> List[str]:
        """Extract top keywords from a list of articles"""
        keyword_counter = {}
        
        for article in articles:
            for keyword in article.get('keywords', []):
                keyword_counter[keyword] = keyword_counter.get(keyword, 0) + 1
        
        # Get top keywords by frequency
        top_keywords = sorted(keyword_counter.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in top_keywords[:count]]
    
    # RSS feed crawler
    @async_retry_with_backoff(max_retries=3, backoff_factor=2)
    async def _crawl_rss_feed(self, source: Dict[str, Any]):
        """Crawl an RSS feed for news articles"""
        url = source.get('url')
        if not url:
            return
        
        self.logger.debug(f"Crawling RSS feed: {url}")
        
        while self.is_running:
            try:
                # Parse RSS feed
                loop = asyncio.get_event_loop()
                feed = await loop.run_in_executor(self.executor, feedparser.parse, url)
                
                for entry in feed.entries:
                    # Extract data
                    article_url = entry.get('link')
                    if not article_url or article_url in self.seen_urls:
                        continue
                    
                    # Parse published date
                    published = entry.get('published')
                    if published:
                        try:
                            dt = date_parser.parse(published)
                            timestamp = int(dt.timestamp() * 1000)
                        except:
                            timestamp = int(time.time() * 1000)
                    else:
                        timestamp = int(time.time() * 1000)
                    
                    # Create article object
                    article = {
                        'url': article_url,
                        'title': entry.get('title'),
                        'summary': entry.get('summary'),
                        'published': timestamp,
                        'timestamp': timestamp,
                        'source': source.get('name', 'rss'),
                        'source_type': 'rss'
                    }
                    
                    # Add to queue
                    await self.article_queue.put(article)
                
                # Sleep before next update (adjust based on source)
                update_interval = source.get('update_interval', 300)  # 5 minutes default
                await asyncio.sleep(update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error crawling RSS feed {url}: {e}")
                await asyncio.sleep(60)  # Sleep after error
    
    # API source crawler
    @async_retry_with_backoff(max_retries=3, backoff_factor=2)
    async def _crawl_api_source(self, source: Dict[str, Any]):
        """Crawl a news API for articles"""
        url = source.get('url')
        api_key = source.get('api_key')
        
        if not url:
            return
            
        self.logger.debug(f"Crawling API source: {url}")
        
        while self.is_running:
            try:
                # Prepare request
                headers = {}
                params = {}
                
                if api_key:
                    # Add API key based on auth method
                    auth_method = source.get('auth_method', 'header')
                    if auth_method == 'header':
                        headers['Authorization'] = f"Bearer {api_key}"
                    elif auth_method == 'param':
                        api_key_param = source.get('api_key_param', 'apiKey')
                        params[api_key_param] = api_key
                
                # Add additional parameters
                params.update(source.get('params', {}))
                
                # Make request
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Process articles based on API format
                            articles_path = source.get('articles_path', 'articles')
                            articles = self._extract_from_json_path(data, articles_path)
                            
                            if articles:
                                # Map fields according to source configuration
                                field_mapping = source.get('field_mapping', {})
                                
                                for item in articles:
                                    article = {}
                                    
                                    # Map fields
                                    for target_field, source_field in field_mapping.items():
                                        article[target_field] = self._extract_from_json_path(item, source_field)
                                    
                                    # Ensure we have a URL
                                    if 'url' not in article or not article['url']:
                                        continue
                                        
                                    if article['url'] in self.seen_urls:
                                        continue
                                    
                                    # Parse published date if available
                                    if 'published' in article and article['published']:
                                        try:
                                            if isinstance(article['published'], str):
                                                dt = date_parser.parse(article['published'])
                                                article['timestamp'] = int(dt.timestamp() * 1000)
                                            elif isinstance(article['published'], (int, float)):
                                                article['timestamp'] = int(article['published'])
                                        except:
                                            article['timestamp'] = int(time.time() * 1000)
                                    else:
                                        article['timestamp'] = int(time.time() * 1000)
                                    
                                    # Add source information
                                    article['source'] = source.get('name', 'api')
                                    article['source_type'] = 'api'
                                    
                                    # Add to queue
                                    await self.article_queue.put(article)
                        else:
                            self.logger.warning(f"API request failed: {response.status}")
                
                # Sleep before next update
                update_interval = source.get('update_interval', 300)  # 5 minutes default
                await asyncio.sleep(update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error crawling API source {url}: {e}")
                await asyncio.sleep(60)  # Sleep after error
    
    def _extract_from_json_path(self, data: Any, path: str) -> Any:
        """Extract value from a nested JSON object using path notation"""
        if not path:
            return data
            
        components = path.split('.')
        result = data
        
        for component in components:
            if isinstance(result, dict) and component in result:
                result = result[component]
            else:
                return None
                
        return result
    
    # Website crawler
    @async_retry_with_backoff(max_retries=3, backoff_factor=2)
    async def _crawl_website(self, source: Dict[str, Any]):
        """Crawl a website for news articles"""
        url = source.get('url')
        if not url:
            return
            
        self.logger.debug(f"Crawling website: {url}")
        
        while self.is_running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=self.headers) as response:
                        if response.status == 200:
                            html = await response.text()
                            
                            # Parse with BeautifulSoup
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # Extract articles based on selector
                            article_selector = source.get('article_selector', 'article')
                            link_selector = source.get('link_selector', 'a')
                            
                            articles = soup.select(article_selector)
                            if not articles and article_selector != 'article':
                                # Fallback to generic article tag
                                articles = soup.find_all('article')
                            
                            for article_elem in articles:
                                # Extract link
                                link_elem = article_elem.select_one(link_selector)
                                if not link_elem:
                                    continue
                                    
                                link_url = link_elem.get('href')
                                if not link_url:
                                    continue
                                
                                # Make sure we have absolute URL
                                if not link_url.startswith('http'):
                                    link_url = self._make_absolute_url(url, link_url)
                                
                                if link_url in self.seen_urls:
                                    continue
                                
                                # Extract title
                                title_selector = source.get('title_selector', 'h2, h3')
                                title_elem = article_elem.select_one(title_selector)
                                title = title_elem.get_text().strip() if title_elem else link_elem.get_text().strip()
                                
                                # Create article object
                                article = {
                                    'url': link_url,
                                    'title': title,
                                    'timestamp': int(time.time() * 1000),
                                    'source': source.get('name', 'website'),
                                    'source_type': 'website'
                                }
                                
                                # Add to queue
                                await self.article_queue.put(article)
                
                # Sleep before next update
                update_interval = source.get('update_interval', 1800)  # 30 minutes default
                await asyncio.sleep(update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error crawling website {url}: {e}")
                await asyncio.sleep(60)  # Sleep after error
    
    def _make_absolute_url(self, base_url: str, relative_url: str) -> str:
        """Convert relative URLs to absolute URLs"""
        parsed_base = urlparse(base_url)
        
        if relative_url.startswith('//'):
            # Protocol-relative URL
            return f"{parsed_base.scheme}:{relative_url}"
        elif relative_url.startswith('/'):
            # Root-relative URL
            return f"{parsed_base.scheme}://{parsed_base.netloc}{relative_url}"
        else:
            # Path-relative URL
            base_path = parsed_base.path
            if not base_path.endswith('/'):
                base_path = '/'.join(base_path.split('/')[:-1]) + '/'
            
            return f"{parsed_base.scheme}://{parsed_base.netloc}{base_path}{relative_url}"
    
    # Twitter crawler
    @async_retry_with_backoff(max_retries=3, backoff_factor=2)
    async def _crawl_twitter(self, source: Dict[str, Any]):
        """Crawl Twitter for news from financial accounts"""
        if 'twitter' not in self.premium_apis:
            self.logger.warning("Twitter API credentials not configured")
            return
            
        api_key = self.premium_apis['twitter'].get('api_key')
        api_secret = self.premium_apis['twitter'].get('api_secret')
        
        if not api_key or not api_secret:
            self.logger.warning("Twitter API credentials incomplete")
            return
            
        accounts = source.get('accounts', [])
        if not accounts:
            self.logger.warning("No Twitter accounts specified")
            return
            
        self.logger.debug(f"Crawling Twitter for {len(accounts)} accounts")
        
        while self.is_running:
            try:
                # Twitter API v2 endpoint
                url = "https://api.twitter.com/2/tweets/search/recent"
                
                # Prepare query with accounts
                query = ' OR '.join([f"from:{account}" for account in accounts])
                
                # Add keywords if available
                keywords = source.get('keywords', [])
                if keywords:
                    keyword_query = ' OR '.join(keywords)
                    query = f"({query}) ({keyword_query})"
                
                # Encode query
                encoded_query = query.replace(' ', '%20')
                
                # Prepare request
                headers = {
                    'Authorization': f"Bearer {api_key}"
                }
                
                params = {
                    'query': encoded_query,
                    'max_results': 100,
                    'tweet.fields': 'created_at,entities,public_metrics',
                    'expansions': 'author_id',
                    'user.fields': 'name,username,verified'
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            tweets = data.get('data', [])
                            users = {user['id']: user for user in data.get('includes', {}).get('users', [])}
                            
                            for tweet in tweets:
                                # Create article object
                                tweet_id = tweet['id']
                                tweet_url = f"https://twitter.com/user/status/{tweet_id}"
                                
                                if tweet_url in self.seen_urls:
                                    continue
                                    
                                # Extract timestamp
                                created_at = tweet.get('created_at')
                                if created_at:
                                    timestamp = int(date_parser.parse(created_at).timestamp() * 1000)
                                else:
                                    timestamp = int(time.time() * 1000)
                                
                                # Get author info
                                author_id = tweet.get('author_id')
                                author = users.get(author_id, {})
                                
                                article = {
                                    'url': tweet_url,
                                    'title': f"{author.get('name', 'Twitter User')}: {tweet['text'][:100]}{'...' if len(tweet['text']) > 100 else ''}",
                                    'content': tweet['text'],
                                    'timestamp': timestamp,
                                    'published': timestamp,
                                    'source': 'twitter',
                                    'source_type': 'social',
                                    'metrics': tweet.get('public_metrics', {}),
                                    'author': {
                                        'name': author.get('name'),
                                        'username': author.get('username'),
                                        'verified': author.get('verified', False)
                                    }
                                }
                                
                                # Add to queue
                                await self.article_queue.put(article)
                        else:
                            self.logger.warning(f"Twitter API request failed: {response.status}")
                
                # Sleep before next update
                update_interval = source.get('update_interval', 300)  # 5 minutes default
                await asyncio.sleep(update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error crawling Twitter: {e}")
                await asyncio.sleep(60)  # Sleep after error
    
    # Reddit crawler
    @async_retry_with_backoff(max_retries=3, backoff_factor=2)
    async def _crawl_reddit(self, source: Dict[str, Any]):
        """Crawl Reddit for news from financial subreddits"""
        subreddits = source.get('subreddits', [])
        if not subreddits:
            self.logger.warning("No Reddit subreddits specified")
            return
            
        self.logger.debug(f"Crawling Reddit for {len(subreddits)} subreddits")
        
        while self.is_running:
            try:
                for subreddit in subreddits:
                    # Reddit JSON API endpoint
                    url = f"https://www.reddit.com/r/{subreddit}/hot.json"
                    
                    # Prepare request
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, headers=headers) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                posts = data.get('data', {}).get('children', [])
                                
                                for post in posts:
                                    post_data = post.get('data', {})
                                    
                                    # Skip stickied posts, deleted posts, and non-link posts
                                    if post_data.get('stickied') or post_data.get('removed') or post_data.get('deleted'):
                                        continue
                                    
                                    # Create article object
                                    post_id = post_data.get('id')
                                    post_url = f"https://www.reddit.com{post_data.get('permalink')}"
                                    
                                    if post_url in self.seen_urls:
                                        continue
                                    
                                    # Extract timestamp
                                    created_utc = post_data.get('created_utc')
                                    if created_utc:
                                        timestamp = int(created_utc * 1000)
                                    else:
                                        timestamp = int(time.time() * 1000)
                                    
                                    article = {
                                        'url': post_url,
                                        'title': post_data.get('title'),
                                        'content': post_data.get('selftext'),
                                        'timestamp': timestamp,
                                        'published': timestamp,
                                        'source': 'reddit',
                                        'source_type': 'social',
                                        'metrics': {
                                            'score': post_data.get('score'),
                                            'comments': post_data.get('num_comments'),
                                            'awards': post_data.get('total_awards_received')
                                        },
                                        'author': post_data.get('author'),
                                        'subreddit': post_data.get('subreddit')
                                    }
                                    
                                    # Add external URL if available
                                    if post_data.get('url') and not post_data.get('is_self'):
                                        article['external_url'] = post_data.get('url')
                                    
                                    # Add to queue
                                    await self.article_queue.put(article)
                            else:
                                self.logger.warning(f"Reddit API request failed: {response.status}")
                    
                # Sleep before next update
                update_interval = source.get('update_interval', 900)  # 15 minutes default
                await asyncio.sleep(update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error crawling Reddit: {e}")
                await asyncio.sleep(60)  # Sleep after error
    
    # SEC filings crawler
    @async_retry_with_backoff(max_retries=3, backoff_factor=2)
    async def _crawl_sec_filings(self, source: Dict[str, Any]):
        """Crawl SEC filings for corporate disclosures"""
        self.logger.debug("Crawling SEC filings")
        
        while self.is_running:
            try:
                # SEC RSS feed URL
                url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK=&type=&company=&dateb=&owner=include&start=0&count=100&output=atom"
                
                # Parse RSS feed
                loop = asyncio.get_event_loop()
                feed = await loop.run_in_executor(self.executor, feedparser.parse, url)
                
                for entry in feed.entries:
                    # Extract data
                    article_url = entry.get('link')
                    if not article_url or article_url in self.seen_urls:
                        continue
                    
                    # Parse published date
                    published = entry.get('published')
                    if published:
                        try:
                            dt = date_parser.parse(published)
                            timestamp = int(dt.timestamp() * 1000)
                        except:
                            timestamp = int(time.time() * 1000)
                    else:
                        timestamp = int(time.time() * 1000)
                    
                    # Extract CIK and form type
                    title = entry.get('title', '')
                    form_type = ''
                    company = ''
                    
                    # Parse title format "form_type - company (CIK)"
                    if ' - ' in title:
                        form_type, company_part = title.split(' - ', 1)
                        form_type = form_type.strip()
                        
                        # Extract company name without CIK
                        if '(' in company_part:
                            company = company_part.split('(')[0].strip()
                    
                    # Create article object
                    article = {
                        'url': article_url,
                        'title': title,
                        'summary': entry.get('summary'),
                        'timestamp': timestamp,
                        'published': timestamp,
                        'source': 'sec',
                        'source_type': 'sec_filing',
                        'form_type': form_type,
                        'company': company
                    }
                    
                    # Add to queue
                    await self.article_queue.put(article)
                
                # Sleep before next update
                update_interval = source.get('update_interval', 3600)  # 1 hour default
                await asyncio.sleep(update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error crawling SEC filings: {e}")
                await asyncio.sleep(60)  # Sleep after error
    
    # Economic calendar crawler
    @async_retry_with_backoff(max_retries=3, backoff_factor=2)
    async def _crawl_economic_calendar(self, source: Dict[str, Any]):
        """Crawl economic calendar for events"""
        # This is a simpler wrapper to call the fetch_economic_calendar method
        self.logger.debug("Crawling economic calendar")
        
        while self.is_running:
            try:
                # Fetch calendar
                await self._fetch_economic_calendar()
                
                # Sleep before next update
                update_interval = source.get('update_interval', 14400)  # 4 hours default
                await asyncio.sleep(update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error crawling economic calendar: {e}")
                await asyncio.sleep(300)  # 5 minutes
    
    # Economic calendar parsers
    async def _parse_investing_calendar(self, html: str) -> List[Dict[str, Any]]:
        """Parse Investing.com economic calendar HTML"""
        events = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            event_rows = soup.select('#economicCalendarData .js-event-item')
            
            for row in event_rows:
                try:
                    # Extract time
                    time_cell = row.select_one('.time')
                    time_str = time_cell.text.strip() if time_cell else ''
                    
                    # Extract date from data attributes
                    event_timestamp = row.get('data-timestamp')
                    if event_timestamp:
                        timestamp = int(event_timestamp) * 1000
                    else:
                        # Fallback to parsing date string
                        date_str = row.get('data-event-datetime', '')
                        try:
                            dt = date_parser.parse(date_str)
                            timestamp = int(dt.timestamp() * 1000)
                        except:
                            # Use current time as fallback
                            timestamp = int(time.time() * 1000)
                    
                    # Extract country
                    country_cell = row.select_one('.flagCur')
                    country = country_cell['title'] if country_cell and 'title' in country_cell.attrs else ''
                    
                    # Extract event name
                    event_cell = row.select_one('.event')
                    event_name = event_cell.text.strip() if event_cell else ''
                    
                    # Extract importance
                    importance_element = row.select_one('.sentiment')
                    importance = 0
                    if importance_element:
                        bull_icons = importance_element.select('.grayFullBullishIcon')
                        importance = len(bull_icons) if bull_icons else 0
                    
                    # Extract actual, forecast, previous values
                    actual_cell = row.select_one('.act')
                    forecast_cell = row.select_one('.fore')
                    previous_cell = row.select_one('.prev')
                    
                    actual = actual_cell.text.strip() if actual_cell else ''
                    forecast = forecast_cell.text.strip() if forecast_cell else ''
                    previous = previous_cell.text.strip() if previous_cell else ''
                    
                    # Create event object
                    event = {
                        'title': event_name,
                        'country': country,
                        'timestamp': timestamp,
                        'time': time_str,
                        'importance': importance,
                        'actual': actual,
                        'forecast': forecast,
                        'previous': previous,
                        'impact_assets': self._map_country_to_assets(country),
                        'event_type': self._categorize_economic_event(event_name)
                    }
                    
                    events.append(event)
                    
                except Exception as e:
                    self.logger.error(f"Error parsing event row: {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Error parsing Investing.com calendar: {e}")
        
        return events
    
    async def _parse_forexfactory_calendar(self, html: str) -> List[Dict[str, Any]]:
        """Parse ForexFactory economic calendar HTML"""
        events = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            event_rows = soup.select('.calendar_row')
            
            current_date = None
            
            for row in event_rows:
                try:
                    # Check if this is a date row
                    date_cell = row.select_one('.date')
                    if date_cell and date_cell.text.strip():
                        date_str = date_cell.text.strip()
                        try:
                            # Parse date (format varies)
                            current_date = date_parser.parse(date_str)
                        except:
                            pass
                        continue
                    
                    if not current_date:
                        continue
                    
                    # Extract time
                    time_cell = row.select_one('.time')
                    time_str = time_cell.text.strip() if time_cell else ''
                    
                    # Combine date and time
                    if time_str and ':' in time_str:
                        try:
                            hour, minute = time_str.split(':')
                            event_datetime = current_date.replace(
                                hour=int(hour),
                                minute=int(minute)
                            )
                            timestamp = int(event_datetime.timestamp() * 1000)
                        except:
                            timestamp = int(current_date.timestamp() * 1000)
                    else:
                        timestamp = int(current_date.timestamp() * 1000)
                    
                    # Extract country
                    country_cell = row.select_one('.country')
                    country = country_cell.text.strip() if country_cell else ''
                    
                    # Extract event name
                    event_cell = row.select_one('.event')
                    event_name = event_cell.text.strip() if event_cell else ''
                    
                    # Extract importance
                    importance_cell = row.select_one('.impact')
                    importance = 0
                    if importance_cell:
                        if 'high' in importance_cell.get('class', []):
                            importance = 3
                        elif 'medium' in importance_cell.get('class', []):
                            importance = 2
                        elif 'low' in importance_cell.get('class', []):
                            importance = 1
                    
                    # Extract actual, forecast, previous values
                    actual_cell = row.select_one('.actual')
                    forecast_cell = row.select_one('.forecast')
                    previous_cell = row.select_one('.previous')
                    
                    actual = actual_cell.text.strip() if actual_cell else ''
                    forecast = forecast_cell.text.strip() if forecast_cell else ''
                    previous = previous_cell.text.strip() if previous_cell else ''
                    
                    # Create event object
                    event = {
                        'title': event_name,
                        'country': country,
                        'timestamp': timestamp,
                        'time': time_str,
                        'importance': importance,
                        'actual': actual,
                        'forecast': forecast,
                        'previous': previous,
                        'impact_assets': self._map_country_to_assets(country),
                        'event_type': self._categorize_economic_event(event_name)
                    }
                    
                    events.append(event)
                    
                except Exception as e:
                    self.logger.error(f"Error parsing ForexFactory event row: {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Error parsing ForexFactory calendar: {e}")
        
        return events
    
    async def _parse_tradingeconomics_calendar(self, html: str) -> List[Dict[str, Any]]:
        """Parse TradingEconomics economic calendar HTML"""
        events = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            event_rows = soup.select('#calendar tbody tr')
            
            for row in event_rows:
                try:
                    # Extract time and date
                    time_cell = row.select_one('.calendar-date')
                    date_str = time_cell.text.strip() if time_cell else ''
                    
                    if date_str:
                        try:
                            dt = date_parser.parse(date_str)
                            timestamp = int(dt.timestamp() * 1000)
                        except:
                            timestamp = int(time.time() * 1000)
                    else:
                        timestamp = int(time.time() * 1000)
                    
                    # Extract country
                    country_cell = row.select_one('.calendar-country')
                    country = country_cell.text.strip() if country_cell else ''
                    
                    # Extract event name
                    event_cell = row.select_one('.calendar-event')
                    event_name = event_cell.text.strip() if event_cell else ''
                    
                    # Extract importance based on the star symbol
                    importance = 0
                    try:
                        # Check for star icons
                        star_icons = row.select('.fas.fa-star')
                        importance = len(star_icons) if star_icons else 0
                    except:
                        pass
                    
                    # Extract actual, forecast, previous values
                    actual_cell = row.select_one('.calendar-actual')
                    forecast_cell = row.select_one('.calendar-forecast')
                    previous_cell = row.select_one('.calendar-previous')
                    
                    actual = actual_cell.text.strip() if actual_cell else ''
                    forecast = forecast_cell.text.strip() if forecast_cell else ''
                    previous = previous_cell.text.strip() if previous_cell else ''
                    
                    # Create event object
                    event = {
                        'title': event_name,
                        'country': country,
                        'timestamp': timestamp,
                        'time': date_str,
                        'importance': importance,
                        'actual': actual,
                        'forecast': forecast,
                        'previous': previous,
                        'impact_assets': self._map_country_to_assets(country),
                        'event_type': self._categorize_economic_event(event_name)
                    }
                    
                    events.append(event)
                    
                except Exception as e:
                    self.logger.error(f"Error parsing TradingEconomics event row: {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Error parsing TradingEconomics calendar: {e}")
        
        return events
    
    async def _parse_premium_calendar_api(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse premium economic calendar API response"""
        events = []
        
        try:
            # Extract events based on API response structure
            api_events = data.get('events', [])
            
            for api_event in api_events:
                try:
                    # Extract timestamp
                    date_str = api_event.get('date', '')
                    time_str = api_event.get('time', '')
                    
                    if date_str:
                        try:
                            if time_str:
                                dt = date_parser.parse(f"{date_str} {time_str}")
                            else:
                                dt = date_parser.parse(date_str)
                            timestamp = int(dt.timestamp() * 1000)
                        except:
                            timestamp = int(time.time() * 1000)
                    else:
                        timestamp = int(time.time() * 1000)
                    
                    # Extract basic data
                    country = api_event.get('country', '')
                    event_name = api_event.get('event', '')
                    importance = api_event.get('importance', 0)
                    
                    # Map string importance to numeric if needed
                    if isinstance(importance, str):
                        importance_map = {
                            'high': 3,
                            'medium': 2,
                            'low': 1
                        }
                        importance = importance_map.get(importance.lower(), 0)
                    
                    # Create event object
                    event = {
                        'title': event_name,
                        'country': country,
                        'timestamp': timestamp,
                        'time': api_event.get('time', ''),
                        'importance': importance,
                        'actual': api_event.get('actual', ''),
                        'forecast': api_event.get('forecast', ''),
                        'previous': api_event.get('previous', ''),
                        'impact_assets': self._map_country_to_assets(country),
                        'event_type': self._categorize_economic_event(event_name)
                    }
                    
                    events.append(event)
                    
                except Exception as e:
                    self.logger.error(f"Error parsing premium API event: {e}")
                    continue
            
        except Exception as e:
            self.logger.error(f"Error parsing premium calendar API: {e}")
        
        return events
    
    def _map_country_to_assets(self, country: str) -> List[str]:
        """Map country names to related financial assets"""
        country_asset_map = {
            'United States': ['USD', 'US30', 'SPX500', 'NAS100', 'FAANG'],
            'US': ['USD', 'US30', 'SPX500', 'NAS100', 'FAANG'],
            'USA': ['USD', 'US30', 'SPX500', 'NAS100', 'FAANG'],
            'Euro Zone': ['EUR', 'STOXX50', 'DAX30', 'CAC40'],
            'Eurozone': ['EUR', 'STOXX50', 'DAX30', 'CAC40'],
            'Europe': ['EUR', 'STOXX50', 'DAX30', 'CAC40'],
            'Japan': ['JPY', 'JPN225'],
            'United Kingdom': ['GBP', 'UK100'],
            'UK': ['GBP', 'UK100'],
            'China': ['CNH', 'CHI50', 'USDCNH'],
            'Australia': ['AUD', 'AUS200'],
            'Canada': ['CAD', 'USDCAD'],
            'New Zealand': ['NZD', 'NZDUSD'],
            'Switzerland': ['CHF', 'USDCHF'],
            'Mexico': ['MXN', 'USDMXN'],
            'India': ['INR', 'USDINR', 'BSE'],
            'Brazil': ['BRL', 'USDBRL', 'BOV'],
            'Russia': ['RUB', 'USDRUB'],
            'South Africa': ['ZAR', 'USDZAR'],
            'Global': ['GOLD', 'SILVER', 'OIL', 'BTC', 'ETH']
        }
        
        # Look for country match
        for key, assets in country_asset_map.items():
            if country.lower() in key.lower():
                return assets
        
        # Default global assets
        return ['GOLD', 'SILVER', 'OIL', 'BTC', 'ETH']
    
    def _categorize_economic_event(self, event_name: str) -> str:
        """Categorize economic events into types"""
        event_name_lower = event_name.lower()
        
        # Interest rate related
        if any(term in event_name_lower for term in ['interest rate', 'rate decision', 'fomc', 'boe', 'ecb', 'boj', 'rba', 'fed']):
            return 'interest_rate'
            
        # Inflation related
        if any(term in event_name_lower for term in ['cpi', 'inflation', 'price index', 'ppi']):
            return 'inflation'
            
        # Employment related
        if any(term in event_name_lower for term in ['employment', 'unemployment', 'nonfarm', 'payroll', 'jobless', 'jobs']):
            return 'employment'
            
        # GDP related
        if any(term in event_name_lower for term in ['gdp', 'gross domestic']):
            return 'gdp'
            
        # Manufacturing related
        if any(term in event_name_lower for term in ['pmi', 'manufacturing', 'industrial', 'production', 'ism']):
            return 'manufacturing'
            
        # Retail/Consumer related
        if any(term in event_name_lower for term in ['retail', 'sales', 'consumer', 'spending', 'confidence']):
            return 'consumer'
            
        # Housing related
        if any(term in event_name_lower for term in ['housing', 'home', 'building', 'construction', 'mortgage']):
            return 'housing'
            
        # Trade related
        if any(term in event_name_lower for term in ['trade', 'export', 'import', 'balance']):
            return 'trade'
            
        # Central bank communication
        if any(term in event_name_lower for term in ['minutes', 'speech', 'testimony', 'statement']):
            return 'central_bank'
            
        # Default
        return 'other'
    
    # Specialized scrapers for specific financial news sites
    @async_retry_with_backoff(max_retries=3, backoff_factor=2)
    async def _scrape_bloomberg(self, url: str) -> Optional[str]:
        """Specialized scraper for Bloomberg"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find article content
                        article_body = soup.select_one('.body-content')
                        if not article_body:
                            article_body = soup.select_one('.body-copy')
                        
                        if article_body:
                            paragraphs = article_body.find_all('p')
                            content = '\n'.join([p.get_text().strip() for p in paragraphs])
                            return content
                            
                        # Fallback to generic extraction
                        return await self._extract_article_content({'url': url})
        except Exception as e:
            self.logger.error(f"Error scraping Bloomberg article: {e}")
            
        return None
    
    @async_retry_with_backoff(max_retries=3, backoff_factor=2)
    async def _scrape_reuters(self, url: str) -> Optional[str]:
        """Specialized scraper for Reuters"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find article content
                        article_body = soup.select_one('.ArticleBody__body')
                        if article_body:
                            paragraphs = article_body.find_all('p')
                            content = '\n'.join([p.get_text().strip() for p in paragraphs])
                            return content
                            
                        # Fallback to generic extraction
                        return await self._extract_article_content({'url': url})
        except Exception as e:
            self.logger.error(f"Error scraping Reuters article: {e}")
            
        return None
    
    @async_retry_with_backoff(max_retries=3, backoff_factor=2)
    async def _scrape_wsj(self, url: str) -> Optional[str]:
        """Specialized scraper for Wall Street Journal"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find article content
                        article_body = soup.select_one('.article-content')
                        if article_body:
                            paragraphs = article_body.find_all('p')
                            content = '\n'.join([p.get_text().strip() for p in paragraphs])
                            return content
                            
                        # Fallback to generic extraction
                        return await self._extract_article_content({'url': url})
        except Exception as e:
            self.logger.error(f"Error scraping WSJ article: {e}")
            
        return None
    
    @async_retry_with_backoff(max_retries=3, backoff_factor=2)
    async def _scrape_ft(self, url: str) -> Optional[str]:
        """Specialized scraper for Financial Times"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find article content
                        article_body = soup.select_one('.article__content')
                        if article_body:
                            paragraphs = article_body.find_all('p')
                            content = '\n'.join([p.get_text().strip() for p in paragraphs])
                            return content
                            
                        # Fallback to generic extraction
                        return await self._extract_article_content({'url': url})
        except Exception as e:
            self.logger.error(f"Error scraping FT article: {e}")
            
        return None
    
    @async_retry_with_backoff(max_retries=3, backoff_factor=2)
    async def _scrape_cnbc(self, url: str) -> Optional[str]:
        """Specialized scraper for CNBC"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find article content
                        article_body = soup.select_one('.ArticleBody-wrapper')
                        if article_body:
                            paragraphs = article_body.find_all('p')
                            content = '\n'.join([p.get_text().strip() for p in paragraphs])
                            return content
                            
                        # Fallback to generic extraction
                        return await self._extract_article_content({'url': url})
        except Exception as e:
            self.logger.error(f"Error scraping CNBC article: {e}")
            
        return None
    
    @async_retry_with_backoff(max_retries=3, backoff_factor=2)
    async def _scrape_investing(self, url: str) -> Optional[str]:
        """Specialized scraper for Investing.com"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find article content
                        article_body = soup.select_one('.articlePage')
                        if article_body:
                            paragraphs = article_body.find_all('p')
                            content = '\n'.join([p.get_text().strip() for p in paragraphs])
                            return content
                            
                        # Fallback to generic extraction
                        return await self._extract_article_content({'url': url})
        except Exception as e:
            self.logger.error(f"Error scraping Investing.com article: {e}")
            
        return None
    
    @async_retry_with_backoff(max_retries=3, backoff_factor=2)
    async def _scrape_marketwatch(self, url: str) -> Optional[str]:
        """Specialized scraper for MarketWatch"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find article content
                        article_body = soup.select_one('.article__body')
                        if article_body:
                            paragraphs = article_body.find_all('p')
                            content = '\n'.join([p.get_text().strip() for p in paragraphs])
                            return content
                            
                        # Fallback to generic extraction
                        return await self._extract_article_content({'url': url})
        except Exception as e:
            self.logger.error(f"Error scraping MarketWatch article: {e}")
            
        return None
    
    @async_retry_with_backoff(max_retries=3, backoff_factor=2)
    async def _scrape_seeking_alpha(self, url: str) -> Optional[str]:
        """Specialized scraper for Seeking Alpha"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find article content
                        article_body = soup.select_one('.sa-art')
                        if article_body:
                            paragraphs = article_body.find_all('p')
                            content = '\n'.join([p.get_text().strip() for p in paragraphs])
                            return content
                            
                        # Fallback to generic extraction
                        return await self._extract_article_content({'url': url})
        except Exception as e:
            self.logger.error(f"Error scraping Seeking Alpha article: {e}")
            
        return None

# Module initialization
def create_news_feed(config: Dict[str, Any], cache_client=None, db_client=None) -> NewsFeed:
    """
    Factory function to create a NewsFeed instance.
    
    Args:
        config: Configuration dictionary
        cache_client: Redis or similar cache client
        db_client: Database client
        
    Returns:
        Configured NewsFeed instance
    """
    return NewsFeed(config, cache_client, db_client)
