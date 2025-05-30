#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
News Data Processor

This module implements the News Data Processor, which processes
news articles and calculates sentiment and relevance.
"""

import time
import asyncio
import re
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from data_ingest.processor import DataProcessor
from common.exceptions import DataValidationError, DataProcessorError
from common.async_utils import run_in_threadpool


class NewsDataProcessor(DataProcessor):
    """Processes news data and calculates sentiment and relevance."""
    
    def __init__(self, config, logger=None):
        """Initialize the news data processor."""
        super().__init__(config, logger=logger)
        
        # Load configuration
        self.sentiment_analysis = config.get("sentiment_analysis", True)
        self.relevance_threshold = config.get("relevance_threshold", 0.5)
        self.use_nlp = config.get("use_nlp", True)
        
        # Initialize NLP if enabled
        self.nlp = None
        if self.use_nlp:
            self._init_nlp()
        
        # Load keyword dictionaries
        self.load_keywords()
    
    def initialize_validation_rules(self):
        """Initialize validation rules for news data."""
        self.validation_rules = {
            "type": {"required": True, "type": str},
            "source": {"required": True, "type": str},
            "timestamp": {"required": True, "type": (int, float)},
            "title": {"required": True, "type": str},
            "content": {"required": False, "type": str},  # Content might be empty for some sources
            "url": {"required": False, "type": str},
        }
    
    def load_keywords(self):
        """Load keyword dictionaries for sentiment and relevance analysis."""
        # Positive and negative financial terms
        self.positive_terms = {
            "bullish", "rally", "surge", "gain", "growth", "profit", "boost", "soar",
            "jump", "advancing", "upgrade", "outperform", "overweight", "buy", "strong buy",
            "beat", "exceeded expectations", "record high", "success", "breakthrough", "innovation",
            "partnership", "launch", "expanding", "opportunity", "positive", "impressive",
            "robust", "recovery", "upside", "upward", "promising", "optimistic"
        }
        
        self.negative_terms = {
            "bearish", "crash", "tumble", "plunge", "drop", "decline", "fall", "loss",
            "downgrade", "underperform", "underweight", "sell", "strong sell", "miss",
            "missed expectations", "record low", "failure", "bankruptcy", "lawsuit", "investigation",
            "fine", "penalty", "recall", "layoff", "downsize", "restructure", "concern", "risk",
            "warning", "negative", "disappointing", "weak", "recession", "downturn", "pessimistic"
        }
        
        # Cryptocurrency terms
        self.crypto_terms = {
            "bitcoin", "btc", "ethereum", "eth", "blockchain", "cryptocurrency", "crypto",
            "token", "altcoin", "binance", "exchange", "wallet", "mining", "halving",
            "defi", "nft", "stablecoin", "tether", "usdt", "smart contract", "transaction",
            "decentralized", "peer-to-peer", "p2p", "scalability", "fork", "consensus",
            "proof of work", "pow", "proof of stake", "pos", "hash", "private key", "public key"
        }
        
        # Market terms
        self.market_terms = {
            "market", "stock", "equity", "share", "index", "etf", "fund", "bond", "treasury",
            "yield", "interest rate", "fed", "federal reserve", "central bank", "inflation",
            "deflation", "recession", "gdp", "economy", "economic", "stimulus", "fiscal",
            "monetary policy", "quantitative easing", "qe", "rate hike", "rate cut",
            "bull market", "bear market", "volatility", "resistance", "support", "trend"
        }
    
    def _init_nlp(self):
        """Initialize NLP libraries for advanced text processing."""
        try:
            # Import NLP libraries only if enabled
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            # Ensure required NLTK data is available without network downloads
            from common.utils import safe_nltk_download
            vader_ok = safe_nltk_download('vader_lexicon')
            punkt_ok = safe_nltk_download('tokenizers/punkt')
            stopwords_ok = safe_nltk_download('stopwords')

            if vader_ok and punkt_ok and stopwords_ok:
                # Initialize sentiment analyzer only if all resources exist
                self.nlp = {
                    'sentiment': SentimentIntensityAnalyzer()
                }
                self.logger.info("NLP initialized successfully")
            else:
                self.logger.warning(
                    "Required NLTK data missing; disabling advanced NLP features"
                )
                self.use_nlp = False
                self.nlp = None
                return
            
        except ImportError as e:
            self.logger.warning(f"NLP libraries not available: {str(e)}. Falling back to basic analysis.")
            self.use_nlp = False
        except Exception as e:
            self.logger.error(f"Error initializing NLP: {str(e)}. Falling back to basic analysis.")
            self.use_nlp = False
    
    async def _process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process news data."""
        try:
            # Extract news data
            title = data.get("title", "")
            content = data.get("content", "")
            source = data.get("source", "")
            url = data.get("url", "")
            
            # Calculate sentiment
            if self.sentiment_analysis:
                sentiment = await self._calculate_sentiment(title, content)
            else:
                sentiment = {"score": 0, "label": "neutral"}
            
            # Calculate relevance
            relevance = await self._calculate_relevance(title, content)
            
            # Create keywords and summary
            keywords = await self._extract_keywords(title, content)
            summary = await self._generate_summary(title, content)
            
            # Create entities (companies, assets mentioned)
            entities = await self._extract_entities(title, content)
            
            # Create processed data
            processed_data = {
                "type": "processed_news",
                "source": source,
                "timestamp": data.get("timestamp"),
                "title": title,
                "url": url,
                "sentiment": sentiment,
                "relevance": relevance,
                "keywords": keywords,
                "summary": summary,
                "entities": entities,
                "source_data": data
            }
            
            self.metrics.increment("news.processed")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing news data: {str(e)}")
            self.metrics.increment("news.error")
            raise DataProcessorError(f"Failed to process news data: {str(e)}")
    
    async def _calculate_sentiment(self, title: str, content: str) -> Dict[str, Any]:
        """Calculate sentiment score for the news article."""
        try:
            # Combine title and content, with title weighted more heavily
            text = title + " " + (content or "")
            
            if self.use_nlp and self.nlp:
                # Use NLTK's VADER for sentiment analysis
                return await run_in_threadpool(self._nlp_sentiment_analysis, text)
            else:
                # Fallback to basic sentiment analysis
                return await self._basic_sentiment_analysis(text)
                
        except Exception as e:
            self.logger.error(f"Error calculating sentiment: {str(e)}")
            return {"score": 0, "label": "neutral", "error": str(e)}
    
    def _nlp_sentiment_analysis(self, text):
        """Perform sentiment analysis using NLTK."""
        # Get sentiment scores
        sentiment_scores = self.nlp['sentiment'].polarity_scores(text)
        
        # Determine sentiment label
        score = sentiment_scores['compound']
        if score >= 0.05:
            label = "positive"
        elif score <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "score": score,
            "label": label,
            "positive": sentiment_scores['pos'],
            "negative": sentiment_scores['neg'],
            "neutral": sentiment_scores['neu'],
        }
    
    async def _basic_sentiment_analysis(self, text):
        """Perform basic sentiment analysis using keyword matching."""
        text = text.lower()
        
        # Count positive and negative terms
        positive_count = sum(1 for term in self.positive_terms if term in text)
        negative_count = sum(1 for term in self.negative_terms if term in text)
        
        # Calculate score (-1 to 1)
        total = positive_count + negative_count
        if total == 0:
            score = 0
        else:
            score = (positive_count - negative_count) / total
        
        # Determine sentiment label
        if score >= 0.05:
            label = "positive"
        elif score <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "score": score,
            "label": label,
            "positive_terms": positive_count,
            "negative_terms": negative_count,
        }
    
    async def _calculate_relevance(self, title: str, content: str) -> Dict[str, Any]:
        """Calculate relevance score for the news article."""
        try:
            # Combine title and content, with title weighted more heavily
            text = (title.lower() + " " + (content or "").lower())
            
            # Count crypto and market terms
            crypto_count = sum(1 for term in self.crypto_terms if term in text)
            market_count = sum(1 for term in self.market_terms if term in text)
            
            # Calculate relevance score (0 to 1)
            total_terms = len(text.split())
            if total_terms == 0:
                relevance_score = 0
            else:
                relevance_score = min(1.0, (crypto_count + market_count) / (total_terms * 0.05))
            
            # Determine if the article is relevant based on threshold
            is_relevant = relevance_score >= self.relevance_threshold
            
            return {
                "score": relevance_score,
                "is_relevant": is_relevant,
                "crypto_terms": crypto_count,
                "market_terms": market_count,
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating relevance: {str(e)}")
            return {"score": 0, "is_relevant": False, "error": str(e)}
    
    async def _extract_keywords(self, title: str, content: str) -> List[str]:
        """Extract important keywords from the article."""
        try:
            # Combine title and content
            text = (title + " " + (content or "")).lower()
            
            # Extract all crypto and market terms
            keywords = []
            for term in self.crypto_terms:
                if term in text:
                    keywords.append(term)
            
            for term in self.market_terms:
                if term in text:
                    keywords.append(term)
            
            # Remove duplicates and sort
            keywords = sorted(set(keywords))
            
            return keywords
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    async def _generate_summary(self, title: str, content: str) -> str:
        """Generate a summary of the article."""
        if not content:
            return title
        
        try:
            # Simple summary: first 2-3 sentences
            sentences = re.split(r'(?<=[.!?])\s+', content)
            summary_length = min(3, len(sentences))
            summary = ' '.join(sentences[:summary_length])
            
            # Truncate if too long
            if len(summary) > 500:
                summary = summary[:497] + "..."
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return title
    
    async def _extract_entities(self, title: str, content: str) -> Dict[str, List[str]]:
        """Extract entities (companies, assets) mentioned in the article."""
        try:
            # Combine title and content
            text = (title + " " + (content or "")).lower()
            
            # Define common assets and companies
            assets = {
                "BTC": ["bitcoin", "btc"],
                "ETH": ["ethereum", "eth"],
                "BNB": ["binance coin", "bnb"],
                "XRP": ["ripple", "xrp"],
                "ADA": ["cardano", "ada"],
                "SOL": ["solana", "sol"],
                "DOGE": ["dogecoin", "doge"]
            }
            
            companies = {
                "Binance": ["binance"],
                "Coinbase": ["coinbase"],
                "Tesla": ["tesla", "tsla"],
                "Apple": ["apple", "aapl"],
                "Google": ["google", "alphabet", "googl"],
                "Amazon": ["amazon", "amzn"],
                "Microsoft": ["microsoft", "msft"]
            }
            
            # Find mentioned assets and companies
            mentioned_assets = []
            for asset, terms in assets.items():
                if any(term in text for term in terms):
                    mentioned_assets.append(asset)
            
            mentioned_companies = []
            for company, terms in companies.items():
                if any(term in text for term in terms):
                    mentioned_companies.append(company)
            
            return {
                "assets": mentioned_assets,
                "companies": mentioned_companies
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            return {"assets": [], "companies": []}
