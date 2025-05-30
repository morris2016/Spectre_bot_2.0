#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Dark Web Intelligence Feed

This module implements a sophisticated system for gathering trading-relevant
intelligence from dark web sources **only** through publicly accessible
channels. It does not purchase data or attempt to bypass authentication
mechanisms.

**Legal Notice**
----------------
Operating on the dark web carries significant legal risk. Ensure you fully
understand and comply with the laws in your jurisdiction before enabling this
feed. The maintainers provide this functionality for educational and research
purposes and assume **no** responsibility for misuse. The feed is disabled by
default and must be explicitly opted into via configuration.

This module focuses on identifying market manipulation patterns, upcoming
events, and sentiment that may affect trading decisions while adhering to legal
and ethical boundaries.
"""

import os
import re
import time
import json
import random
import asyncio
import logging
import hashlib
import datetime
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import aiohttp
import socks
import stem
import stem.control
from stem import Signal
from stem.connection import authenticate_none, authenticate_password
from bs4 import BeautifulSoup

from data_feeds.base_feed import BaseFeed
from common.logger import get_logger
from common.utils import rate_limited, obfuscate_sensitive_data
from common.security import hash_content, decrypt_credentials
from common.exceptions import (
    DataFeedConnectionError, 
    DataParsingError, 
    CredentialError, 
    FeedRateLimitError,
    SecurityViolationError
)
from common.constants import (
    DARK_WEB_FEED_CONFIG,
    DARK_WEB_SITES,
    DARK_WEB_FORUMS,
    DARK_WEB_MARKETS,
    DARK_WEB_RELEVANCE_KEYWORDS,
    MAX_RETRIES,
    RETRY_DELAY,
    USER_AGENTS,
    DARK_WEB_SCAN_INTERVAL,
    MARKETS_OF_INTEREST,
    ASSETS_OF_INTEREST,
    SCAN_THREAD_COUNT
)


class DarkWebFeed(BaseFeed):
    """
    Dark Web Intelligence Feed for the QuantumSpectre Elite Trading System.
    
    Gathers trading-relevant intelligence from dark web sources in a legal and
    ethical manner, focusing on public forums, marketplaces, and information
    exchanges that may contain market-moving information.
    
    This feed does NOT engage in any illegal activity, purchase of data,
    or access to private/protected information. It only scans publicly 
    available information through legal means.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Dark Web Feed with configuration settings.
        
        Args:
            config: Configuration dictionary for the Dark Web Feed
        """
        super().__init__(config)
        self.logger = get_logger("DarkWebFeed")
        self.name = "dark_web_feed"
        self.description = "Dark Web Intelligence Feed"
        
        # Configure TOR connection
        self.tor_proxy = config.get('tor_proxy', 'socks5://127.0.0.1:9050')
        self.tor_control_port = config.get('tor_control_port', 9051)
        self.tor_control_password = decrypt_credentials(
            config.get('tor_control_password', '')
        )
        
        # Parse proxy URL
        parsed = urlparse(self.tor_proxy)
        self.proxy_scheme = parsed.scheme
        self.proxy_host = parsed.hostname
        self.proxy_port = parsed.port
        
        # Configure scanning parameters
        self.scan_interval = config.get('scan_interval', DARK_WEB_SCAN_INTERVAL)
        self.sites_to_scan = config.get('sites_to_scan', DARK_WEB_SITES)
        self.forums_to_scan = config.get('forums_to_scan', DARK_WEB_FORUMS)
        self.markets_to_scan = config.get('markets_to_scan', DARK_WEB_MARKETS)
        self.relevance_keywords = config.get('relevance_keywords', DARK_WEB_RELEVANCE_KEYWORDS)
        self.thread_count = config.get('thread_count', SCAN_THREAD_COUNT)
        self.markets_of_interest = config.get('markets_of_interest', MARKETS_OF_INTEREST)
        self.assets_of_interest = config.get('assets_of_interest', ASSETS_OF_INTEREST)
        
        # Create regex patterns from keywords for efficient matching
        self.relevance_patterns = [
            re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            for keyword in self.relevance_keywords
        ]
        
        # Additional security measures
        self.content_hash_cache = set()  # For deduplication
        self.visited_urls = set()  # To avoid revisiting
        self.forbidden_terms = self._load_forbidden_terms()
        
        # Session and proxy management
        self.session = None
        self.user_agents = USER_AGENTS
        self._initialize_session()
        
        # Intelligence cache
        self.intelligence_cache = {}
        self.last_scan_time = {}
        
        # Metrics tracking
        self.intelligence_found_count = 0
        self.sites_scanned_count = 0
        self.connection_errors = 0
        
        self.logger.info(f"Initialized {self.description}")
    
    async def connect(self) -> bool:
        """
        Establish connection to the TOR network.
        
        Returns:
            bool: True if connection is successful, False otherwise
        """
        try:
            self._initialize_session()
            await self._test_connection()
            self.connected = True
            self.logger.info("Successfully connected to TOR network")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to TOR network: {str(e)}")
            self.connected = False
            return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from the TOR network.
        
        Returns:
            bool: True if disconnection is successful, False otherwise
        """
        try:
            if self.session:
                await self.session.close()
                self.session = None
            
            self.connected = False
            self.logger.info("Disconnected from TOR network")
            return True
        except Exception as e:
            self.logger.error(f"Error during disconnection: {str(e)}")
            return False
    
    async def refresh_connection(self) -> bool:
        """
        Refresh the TOR connection to get a new identity.
        
        Returns:
            bool: True if refresh is successful, False otherwise
        """
        try:
            # Request new identity from TOR control port
            with stem.control.Controller.from_port(
                port=self.tor_control_port
            ) as controller:
                if self.tor_control_password:
                    controller.authenticate(password=self.tor_control_password)
                else:
                    controller.authenticate_none()
                
                controller.signal(Signal.NEWNYM)
                self.logger.debug("TOR identity refreshed")
            
            # Wait a moment for the change to take effect
            await asyncio.sleep(5)
            
            # Close and reinitialize the session
            if self.session:
                await self.session.close()
            
            self._initialize_session()
            await self._test_connection()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to refresh TOR connection: {str(e)}")
            return False
    
    def _initialize_session(self) -> None:
        """Initialize the aiohttp session with TOR proxy."""
        connector = aiohttp.TCPConnector(
            ssl=False,
            proxy_type=socks.SOCKS5 if self.proxy_scheme == 'socks5' else socks.SOCKS4,
            proxy_host=self.proxy_host,
            proxy_port=self.proxy_port
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=60),
            headers=self._get_random_headers()
        )
    
    def _get_random_headers(self) -> Dict[str, str]:
        """
        Generate random headers to avoid tracking.
        
        Returns:
            Dict[str, str]: Dictionary of HTTP headers
        """
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache'
        }
    
    async def _test_connection(self) -> None:
        """
        Test the TOR connection by accessing a .onion test site.
        
        Raises:
            DataFeedConnectionError: If connection test fails
        """
        test_url = "http://zqktlwiuavvvqqt4ybvgvi7tyo4hjl5xgfuvpdf6otjiycgwqbym2qad.onion"  # Example .onion URL (Tor66)
        
        try:
            async with self.session.get(test_url, timeout=30) as response:
                if response.status != 200:
                    raise DataFeedConnectionError(f"TOR connection test failed with status {response.status}")
                
                await response.text()
                self.logger.debug("TOR connection test successful")
        except Exception as e:
            raise DataFeedConnectionError(f"TOR connection test failed: {str(e)}")
    
    async def fetch_data(self) -> List[Dict[str, Any]]:
        """
        Fetch intelligence data from configured dark web sources.
        
        Returns:
            List[Dict[str, Any]]: List of intelligence data points
        """
        if not self.connected:
            await self.connect()
        
        results = []
        
        # Create tasks for scanning different source types
        tasks = [
            self._scan_sites(self.sites_to_scan),
            self._scan_forums(self.forums_to_scan),
            self._scan_markets(self.markets_to_scan)
        ]
        
        # Wait for all tasks to complete
        scan_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results, filtering out exceptions
        for result_set in scan_results:
            if isinstance(result_set, Exception):
                self.logger.error(f"Error during scanning: {str(result_set)}")
                continue
            
            results.extend(result_set)
        
        # Process and normalize the results
        processed_results = self._process_intelligence(results)
        
        # Update metrics
        self.intelligence_found_count += len(processed_results)
        
        return processed_results
    
    @rate_limited(5, 60)  # 5 requests per minute max
    async def _scan_sites(self, sites: List[str]) -> List[Dict[str, Any]]:
        """
        Scan dark web sites for intelligence.
        
        Args:
            sites: List of .onion site URLs to scan
        
        Returns:
            List[Dict[str, Any]]: Intelligence data from sites
        """
        results = []
        
        for site in sites:
            if site in self.visited_urls and (time.time() - self.last_scan_time.get(site, 0)) < self.scan_interval:
                continue
            
            self.sites_scanned_count += 1
            self.last_scan_time[site] = time.time()
            self.visited_urls.add(site)
            
            try:
                # Get site content
                content = await self._fetch_url(site)
                
                # Safety check
                if self._contains_forbidden_content(content):
                    self.logger.warning(f"Found forbidden content on {site}, skipping")
                    continue
                
                # Process content for relevant information
                intel = self._extract_intelligence_from_html(content, site)
                
                if intel:
                    results.extend(intel)
            
            except Exception as e:
                self.logger.error(f"Error scanning site {site}: {str(e)}")
                self.connection_errors += 1
                
                # Refresh connection if we're having issues
                if self.connection_errors % 5 == 0:
                    await self.refresh_connection()
        
        return results
    
    async def _scan_forums(self, forums: List[str]) -> List[Dict[str, Any]]:
        """
        Scan dark web forums for intelligence.
        
        Args:
            forums: List of forum URLs to scan
        
        Returns:
            List[Dict[str, Any]]: Intelligence data from forums
        """
        results = []
        
        for forum in forums:
            if forum in self.visited_urls and (time.time() - self.last_scan_time.get(forum, 0)) < self.scan_interval:
                continue
            
            self.sites_scanned_count += 1
            self.last_scan_time[forum] = time.time()
            self.visited_urls.add(forum)
            
            try:
                # First, get the forum index to find relevant threads
                content = await self._fetch_url(forum)
                
                # Safety check
                if self._contains_forbidden_content(content):
                    self.logger.warning(f"Found forbidden content on {forum}, skipping")
                    continue
                
                # Extract thread links that may be relevant
                soup = BeautifulSoup(content, 'html.parser')
                relevant_threads = []
                
                for link in soup.find_all('a'):
                    href = link.get('href')
                    text = link.get_text()
                    
                    if not href or not text:
                        continue
                    
                    # Check if thread title contains any of our keywords of interest
                    if any(pattern.search(text) for pattern in self.relevance_patterns):
                        # Construct full URL if it's a relative link
                        if not href.startswith('http'):
                            if href.startswith('/'):
                                base_url = forum
                                if base_url.endswith('/'):
                                    base_url = base_url[:-1]
                                href = base_url + href
                            else:
                                href = forum + ('/' if not forum.endswith('/') else '') + href
                        
                        relevant_threads.append((href, text))
                
                # Now scan each relevant thread
                for thread_url, title in relevant_threads:
                    if thread_url in self.visited_urls:
                        continue
                    
                    self.visited_urls.add(thread_url)
                    
                    try:
                        thread_content = await self._fetch_url(thread_url)
                        
                        # Safety check
                        if self._contains_forbidden_content(thread_content):
                            self.logger.warning(f"Found forbidden content in thread {thread_url}, skipping")
                            continue
                        
                        # Extract intelligence from thread
                        intel = self._extract_intelligence_from_html(
                            thread_content, 
                            thread_url,
                            context={"forum": forum, "thread_title": title}
                        )
                        
                        if intel:
                            results.extend(intel)
                    
                    except Exception as e:
                        self.logger.error(f"Error scanning thread {thread_url}: {str(e)}")
            
            except Exception as e:
                self.logger.error(f"Error scanning forum {forum}: {str(e)}")
                self.connection_errors += 1
                
                # Refresh connection if we're having issues
                if self.connection_errors % 5 == 0:
                    await self.refresh_connection()
        
        return results
    
    async def _scan_markets(self, markets: List[str]) -> List[Dict[str, Any]]:
        """
        Scan dark web markets for intelligence.
        
        Args:
            markets: List of market URLs to scan
        
        Returns:
            List[Dict[str, Any]]: Intelligence data from markets
        """
        results = []
        
        for market in markets:
            if market in self.visited_urls and (time.time() - self.last_scan_time.get(market, 0)) < self.scan_interval:
                continue
            
            self.sites_scanned_count += 1
            self.last_scan_time[market] = time.time()
            self.visited_urls.add(market)
            
            try:
                # Get market index page
                content = await self._fetch_url(market)
                
                # Safety check
                if self._contains_forbidden_content(content):
                    self.logger.warning(f"Found forbidden content on {market}, skipping")
                    continue
                
                # We're only interested in legal intelligence, so focus on market trends and discussions
                # rather than actual listings which may involve illegal goods
                soup = BeautifulSoup(content, 'html.parser')
                
                # Look for forums, discussions, or announcements sections
                discussion_links = []
                
                for link in soup.find_all('a'):
                    href = link.get('href')
                    text = link.get_text().lower()
                    
                    if not href or not text:
                        continue
                    
                    # Look for discussion/forum/announcement links
                    if any(keyword in text for keyword in ['forum', 'discussion', 'announcement', 'news']):
                        if not href.startswith('http'):
                            if href.startswith('/'):
                                base_url = market
                                if base_url.endswith('/'):
                                    base_url = base_url[:-1]
                                href = base_url + href
                            else:
                                href = market + ('/' if not market.endswith('/') else '') + href
                        
                        discussion_links.append(href)
                
                # Scan discussion areas
                for link in discussion_links:
                    if link in self.visited_urls:
                        continue
                    
                    self.visited_urls.add(link)
                    
                    try:
                        discussion_content = await self._fetch_url(link)
                        
                        # Safety check
                        if self._contains_forbidden_content(discussion_content):
                            self.logger.warning(f"Found forbidden content in {link}, skipping")
                            continue
                        
                        # Extract intelligence from discussions
                        intel = self._extract_intelligence_from_html(
                            discussion_content, 
                            link,
                            context={"market": market}
                        )
                        
                        if intel:
                            results.extend(intel)
                    
                    except Exception as e:
                        self.logger.error(f"Error scanning discussion {link}: {str(e)}")
            
            except Exception as e:
                self.logger.error(f"Error scanning market {market}: {str(e)}")
                self.connection_errors += 1
                
                # Refresh connection if we're having issues
                if self.connection_errors % 5 == 0:
                    await self.refresh_connection()
        
        return results
    
    async def _fetch_url(self, url: str) -> str:
        """
        Fetch content from a URL with retries and error handling.
        
        Args:
            url: URL to fetch
        
        Returns:
            str: Content of the URL
        
        Raises:
            DataFeedConnectionError: If fetch fails after retries
        """
        retries = 0
        last_error = None
        
        while retries < MAX_RETRIES:
            try:
                # Update session headers for each request to avoid tracking
                self.session._default_headers = self._get_random_headers()
                
                async with self.session.get(url, timeout=30) as response:
                    if response.status != 200:
                        raise DataFeedConnectionError(f"HTTP error {response.status} for URL {url}")
                    
                    return await response.text()
            
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                retries += 1
                
                self.logger.warning(f"Error fetching {url}, retry {retries}/{MAX_RETRIES}: {str(e)}")
                
                # Exponential backoff for retries
                await asyncio.sleep(RETRY_DELAY * (2 ** (retries - 1)))
                
                # Refresh TOR connection on failure
                if retries % 2 == 0:
                    await self.refresh_connection()
            
            except Exception as e:
                self.logger.error(f"Unexpected error fetching {url}: {str(e)}")
                raise DataFeedConnectionError(f"Failed to fetch {url}: {str(e)}")
        
        # If we get here, all retries failed
        raise DataFeedConnectionError(f"Failed to fetch {url} after {MAX_RETRIES} retries: {str(last_error)}")
    
    def _extract_intelligence_from_html(
        self, 
        html_content: str, 
        source_url: str,
        context: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract trading-relevant intelligence from HTML content.
        
        Args:
            html_content: HTML content to process
            source_url: URL source of the content
            context: Additional context information
        
        Returns:
            List[Dict[str, Any]]: Extracted intelligence items
        """
        if not html_content:
            return []
        
        # Calculate content hash for deduplication
        content_hash = hash_content(html_content)
        if content_hash in self.content_hash_cache:
            return []
        
        self.content_hash_cache.add(content_hash)
        
        intelligence_items = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text content
            text = soup.get_text(separator=' ', strip=True)
            
            # Check for relevant content
            found_patterns = set()
            for i, pattern in enumerate(self.relevance_patterns):
                matches = pattern.findall(text)
                if matches:
                    found_patterns.add(self.relevance_keywords[i])
            
            if not found_patterns:
                return []
            
            # Look for specific sections with relevant information
            # This depends on the structure of the sites we're scanning
            
            # Process paragraphs for intelligence
            intelligence_texts = []
            
            # Look for paragraphs near our keywords
            for paragraph in soup.find_all(['p', 'div']):
                p_text = paragraph.get_text(strip=True)
                
                # Skip short or empty paragraphs
                if len(p_text) < 20:
                    continue
                
                # Check if paragraph contains any of our keywords
                if any(pattern.search(p_text) for pattern in self.relevance_patterns):
                    # Clean and normalize text
                    p_text = ' '.join(p_text.split())
                    
                    # Skip duplicate paragraphs
                    if p_text in intelligence_texts:
                        continue
                    
                    intelligence_texts.append(p_text)
            
            # Extract dates from the content for timestamp information
            publication_date = self._extract_dates(soup, html_content)
            
            # Create intelligence items from extracted text
            for i, text_item in enumerate(intelligence_texts):
                # Determine relevance for specific assets
                relevant_assets = []
                for asset in self.assets_of_interest:
                    if re.search(r'\b' + re.escape(asset) + r'\b', text_item, re.IGNORECASE):
                        relevant_assets.append(asset)
                
                # Only include items that mention assets we care about
                if not relevant_assets:
                    continue
                
                # Determine relevance for specific markets
                relevant_markets = []
                for market in self.markets_of_interest:
                    if re.search(r'\b' + re.escape(market) + r'\b', text_item, re.IGNORECASE):
                        relevant_markets.append(market)
                
                # Create intelligence item
                intel_item = {
                    'id': hashlib.sha256(f"{source_url}:{i}:{text_item}".encode()).hexdigest(),
                    'source': 'dark_web',
                    'source_url': obfuscate_sensitive_data(source_url),  # Obfuscate for logs
                    'timestamp': datetime.datetime.now().isoformat(),
                    'publication_date': publication_date,
                    'content': text_item,
                    'keywords': list(found_patterns),
                    'relevant_assets': relevant_assets,
                    'relevant_markets': relevant_markets,
                    'confidence': self._calculate_confidence(text_item, found_patterns, relevant_assets),
                    'intelligence_type': self._determine_intelligence_type(text_item)
                }
                
                # Add context if provided
                if context:
                    intel_item['context'] = context
                
                intelligence_items.append(intel_item)
        
        except Exception as e:
            self.logger.error(f"Error extracting intelligence from {source_url}: {str(e)}")
            return []
        
        return intelligence_items
    
    def _extract_dates(self, soup: BeautifulSoup, html_content: str) -> Optional[str]:
        """
        Extract publication dates from content.
        
        Args:
            soup: BeautifulSoup object of the content
            html_content: Raw HTML content
        
        Returns:
            Optional[str]: ISO format date string if found, None otherwise
        """
        # Look for common date patterns in HTML
        date_patterns = [
            # Common date formats
            r'\d{4}-\d{2}-\d{2}',                    # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',                    # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',                    # DD-MM-YYYY
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}'  # Month DD, YYYY
        ]
        
        # First check meta tags for date information
        for meta in soup.find_all('meta'):
            if meta.get('name') and meta.get('content'):
                name = meta.get('name').lower()
                if 'date' in name or 'published' in name or 'time' in name:
                    content = meta.get('content')
                    try:
                        # Try to parse and convert to ISO format
                        dt = datetime.datetime.fromisoformat(content)
                        return dt.isoformat()
                    except (ValueError, TypeError):
                        # Not a valid ISO date, continue with other methods
                        pass
        
        # Look for date patterns in the HTML content
        for pattern in date_patterns:
            match = re.search(pattern, html_content)
            if match:
                date_str = match.group(0)
                try:
                    # Try different date formats based on the pattern
                    if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                        dt = datetime.datetime.strptime(date_str, '%Y-%m-%d')
                    elif re.match(r'\d{2}/\d{2}/\d{4}', date_str):
                        dt = datetime.datetime.strptime(date_str, '%m/%d/%Y')
                    elif re.match(r'\d{2}-\d{2}-\d{4}', date_str):
                        dt = datetime.datetime.strptime(date_str, '%d-%m-%Y')
                    elif re.match(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}', date_str):
                        # Handle various month formats
                        dt = datetime.datetime.strptime(date_str, '%b %d, %Y')
                    else:
                        continue
                    
                    return dt.isoformat()
                except ValueError:
                    # If we can't parse the date, just continue
                    continue
        
        # If we couldn't find a date, return None
        return None
    
    def _calculate_confidence(
        self, 
        text: str, 
        found_patterns: Set[str],
        relevant_assets: List[str]
    ) -> float:
        """
        Calculate confidence score for intelligence relevance.
        
        Args:
            text: Intelligence text
            found_patterns: Set of found keyword patterns
            relevant_assets: List of relevant assets mentioned
        
        Returns:
            float: Confidence score between 0 and 1
        """
        # Base confidence based on number of keywords found
        base_confidence = min(len(found_patterns) / 5.0, 0.5)
        
        # Increase confidence based on specificity to assets we care about
        asset_confidence = min(len(relevant_assets) / 3.0, 0.3)
        
        # Higher confidence for longer, more detailed text (up to a point)
        length_confidence = min(len(text) / 1000.0, 0.2)
        
        # Confidence boost for specific trading-related terms
        trading_terms = ['price', 'market', 'trader', 'volume', 'buy', 'sell', 'strategy', 'pump', 'dump']
        trading_term_count = sum(1 for term in trading_terms if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE))
        trading_confidence = min(trading_term_count / len(trading_terms), 0.2)
        
        # Calculate final confidence score
        confidence = base_confidence + asset_confidence + length_confidence + trading_confidence
        
        # Cap at 1.0 for normalization
        return min(confidence, 1.0)
    
    def _determine_intelligence_type(self, text: str) -> str:
        """
        Determine the type of intelligence based on content analysis.
        
        Args:
            text: Intelligence text
        
        Returns:
            str: Intelligence type classification
        """
        # Define patterns for different intelligence types
        patterns = {
            'market_manipulation': [
                r'\b(pump|dump|manipulate|scheme|scam|fraud)\b',
                r'\b(coordinate|group|together|collective)\b.*\b(buy|sell)\b',
                r'\b(artificial|fake)\b.*\b(volume|price|market)\b'
            ],
            'upcoming_event': [
                r'\b(upcoming|planned|scheduled|imminent)\b',
                r'\b(release|announcement|launch|fork|update|upgrade)\b',
                r'\b(tomorrow|next week|next month|soon)\b'
            ],
            'insider_information': [
                r'\b(insider|internal|confidential|leak|private)\b',
                r'\b(know|source|told|informed|heard)\b.*\b(before|early|advance)\b',
                r'\b(ahead of|before)\b.*\b(public|announcement|release)\b'
            ],
            'security_vulnerability': [
                r'\b(vulnerability|exploit|hack|breach|flaw|attack)\b',
                r'\b(broken|compromised|insecure|unsafe)\b',
                r'\b(security|protection|defense)\b.*\b(weak|issue|problem)\b'
            ],
            'sentiment_shift': [
                r'\b(sentiment|feeling|confidence|trust|belief)\b',
                r'\b(shift|change|turn|swing|pivot)\b',
                r'\b(bearish|bullish|optimistic|pessimistic|fear|greed)\b'
            ]
        }
        
        # Count matches for each type
        type_scores = {}
        for intel_type, type_patterns in patterns.items():
            score = 0
            for pattern in type_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                score += len(matches)
            
            type_scores[intel_type] = score
        
        # Find the type with the highest score
        if max(type_scores.values(), default=0) == 0:
            return 'general_information'
        
        return max(type_scores.items(), key=lambda x: x[1])[0]
    
    def _process_intelligence(self, raw_intelligence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process and normalize raw intelligence data.
        
        Args:
            raw_intelligence: List of raw intelligence items
        
        Returns:
            List[Dict[str, Any]]: Processed intelligence items
        """
        processed = []
        
        # Deduplicate based on content similarity
        seen_hashes = set()
        
        for item in raw_intelligence:
            # Generate a content hash for deduplication
            content_hash = hashlib.md5(item.get('content', '').encode()).hexdigest()
            
            if content_hash in seen_hashes:
                continue
            
            seen_hashes.add(content_hash)
            
            # Normalize timestamps
            try:
                if 'timestamp' not in item:
                    item['timestamp'] = datetime.datetime.now().isoformat()
                elif not isinstance(item['timestamp'], str):
                    item['timestamp'] = item['timestamp'].isoformat()
            except (AttributeError, TypeError):
                item['timestamp'] = datetime.datetime.now().isoformat()
            
            # Add processing metadata
            item['processed_at'] = datetime.datetime.now().isoformat()
            item['feed_name'] = self.name
            
            processed.append(item)
        
        return processed
    
    def _contains_forbidden_content(self, content: str) -> bool:
        """
        Check if content contains forbidden terms or illegal material references.
        
        Args:
            content: Content to check
        
        Returns:
            bool: True if forbidden content is found, False otherwise
        """
        if not content:
            return False
        
        # Check against forbidden terms
        for term in self.forbidden_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', content, re.IGNORECASE):
                return True
        
        return False
    
    def _load_forbidden_terms(self) -> List[str]:
        """
        Load list of forbidden terms that indicate illegal content.
        
        Returns:
            List[str]: List of forbidden terms
        """
        # This is a basic set of terms indicating potentially illegal content
        # In a production system, this would be loaded from a comprehensive database
        return [
            # This is a sanitized list for demonstration purposes only
            "child abuse",
            "child pornography",
            "assassination",
            "murder for hire",
            "hitman",
            "terrorist attack",
            "human trafficking",
            "illegal firearms",
            "illegal weapons"
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the feed.
        
        Returns:
            Dict[str, Any]: Status information
        """
        return {
            'name': self.name,
            'description': self.description,
            'connected': self.connected,
            'last_refresh': self.last_refresh.isoformat() if self.last_refresh else None,
            'intelligence_found_count': self.intelligence_found_count,
            'sites_scanned_count': self.sites_scanned_count,
            'connection_errors': self.connection_errors
        }
