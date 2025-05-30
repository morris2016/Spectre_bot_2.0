#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Deriv Feed - Advanced Integration with the Deriv Platform

This module provides a sophisticated integration with the Deriv trading platform,
including real-time data streaming, order book analysis, market structure detection,
and advanced pattern recognition specifically optimized for Deriv's unique characteristics.

Key features:
- High-performance WebSocket connection with automatic reconnection
- Real-time market data processing with minimal latency
- Order book analysis for liquidity detection
- Market microstructure analysis for identifying exchange patterns
- Platform-specific behavior modeling for exploitation of inefficiencies
- Multi-asset support with specialized optimizations per instrument
"""

import os
import json
import time
import hmac
import hashlib
import uuid
import asyncio
import websockets
import aiohttp
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
from dataclasses import dataclass, field
import logging
from collections import defaultdict, deque

# Internal imports
from common.logger import get_logger
from common.utils import (
    exponential_backoff, rate_limit, time_execution, 
    calculate_checksum, ensure_future, 
    AtomicCounter, JsonEncoder, create_task_name
)
from common.constants import (
    MAX_RECONNECT_ATTEMPTS, INITIAL_RECONNECT_DELAY,
    MAX_RECONNECT_DELAY, DERIV_ENDPOINTS, 
    DERIV_ASSET_CLASSES, DERIV_MARKETS,
    DEFAULT_SUBSCRIPTION_TIMEOUT, DEFAULT_PING_INTERVAL,
    MARKET_ORDER_BOOK_DEPTH, DERIV_PRICE_REFRESH_RATE
)
from common.exceptions import (
    FeedConnectionError, FeedAuthenticationError, 
    FeedSubscriptionError, FeedRateLimitError,
    FeedDataError, FeedDisconnectedError
)
from common.async_utils import AsyncBatcher, AsyncRateLimiter, AsyncRetrier
from common.metrics import (
    record_latency, increment_counter, record_value,
    record_success, record_failure
)
from common.redis_client import RedisClient

from data_feeds.base_feed import BaseDataFeed, FeedOptions, DataProcessor

from data_ingest.processor import normalize_instrument_id

# Set up logger
logger = get_logger("deriv_feed")


@dataclass
class DerivCredentials:
    """Credentials for authenticating with the Deriv API."""
    app_id: str
    api_token: Optional[str] = None
    account_id: Optional[str] = None
    
    def validate(self) -> bool:
        """Validate that the credentials contain the minimum required fields."""
        return bool(self.app_id)


@dataclass
class DerivFeedOptions(FeedOptions):
    """Configuration options specific to the Deriv feed."""
    endpoints: Dict[str, str] = field(default_factory=lambda: DERIV_ENDPOINTS.copy())
    ping_interval: float = DEFAULT_PING_INTERVAL
    subscription_timeout: float = DEFAULT_SUBSCRIPTION_TIMEOUT
    max_subscriptions_per_connection: int = 25
    price_refresh_rate: float = DERIV_PRICE_REFRESH_RATE
    order_book_depth: int = MARKET_ORDER_BOOK_DEPTH
    rate_limit_max_requests: int = 30
    rate_limit_period: float = 1.0
    use_enhanced_ticks: bool = True
    track_market_structure: bool = True
    model_platform_behavior: bool = True
    detect_platform_patterns: bool = True
    analyze_liquidity_profile: bool = True
    cache_historical_data: bool = True
    detect_synthetic_price_movements: bool = True
    analyze_contract_availability: bool = True

    def __post_init__(self):
        # Additional validation for Deriv-specific options
        if self.ping_interval < 5:
            logger.warning("Ping interval is too low, setting to minimum of 5 seconds")
            self.ping_interval = 5
        if self.subscription_timeout < 10:
            logger.warning("Subscription timeout is too low, setting to minimum of 10 seconds")
            self.subscription_timeout = 10


class DerivPlatformAnalyzer:
    """
    Analyzes Deriv platform behavior to detect patterns, inefficiencies, and exploitable characteristics.
    This intelligence helps the trading system achieve superior win rates by understanding
    how the platform operates under various conditions.
    """
    
    def __init__(self, options: DerivFeedOptions):
        self.options = options
        self.logger = get_logger("deriv_platform_analyzer")
        
        # Pattern detection storage
        self.price_movement_patterns = defaultdict(list)
        self.tick_timing_patterns = defaultdict(list)
        self.spread_change_patterns = defaultdict(list)
        
        # Platform behavior models
        self.liquidity_profiles = {}
        self.synthetic_price_models = {}
        self.contract_availability_patterns = {}
        self.server_response_profiles = defaultdict(deque)
        
        # Anomaly detection
        self.price_anomalies = defaultdict(list)
        self.spread_anomalies = defaultdict(list)
        
        # Statistical tracking
        self.tick_intervals = defaultdict(lambda: deque(maxlen=1000))
        self.price_jumps = defaultdict(lambda: deque(maxlen=1000))
        self.spread_changes = defaultdict(lambda: deque(maxlen=1000))
        
        # Redis client for pattern storage
        self.redis = RedisClient()
    
    async def initialize(self):
        """Initialize the platform analyzer with historical data if available."""
        try:
            # Load stored patterns from Redis if they exist
            patterns = await self.redis.get("deriv:platform_patterns")
            if patterns:
                patterns = json.loads(patterns)
                self.price_movement_patterns = defaultdict(list, patterns.get("price_movement", {}))
                self.tick_timing_patterns = defaultdict(list, patterns.get("tick_timing", {}))
                self.spread_change_patterns = defaultdict(list, patterns.get("spread_change", {}))
                self.logger.info("Loaded platform behavior patterns from cache")
        except Exception as e:
            self.logger.warning(f"Failed to load platform patterns: {str(e)}")
    
    async def analyze_tick(self, symbol: str, tick_data: Dict[str, Any]):
        """
        Analyze an individual price tick to detect patterns and platform behavior.
        
        Args:
            symbol: The trading instrument symbol
            tick_data: The tick data including price, time, etc.
        """
        if not self.options.model_platform_behavior:
            return
        
        try:
            # Record tick timing
            now = time.time()
            timestamp = tick_data.get("epoch", now)
            
            # Add to interval tracking
            self.tick_intervals[symbol].append((now, timestamp))
            if len(self.tick_intervals[symbol]) > 1:
                # Calculate server-to-client latency
                prev_time = self.tick_intervals[symbol][-2][0]
                interval = now - prev_time
                self.server_response_profiles[symbol].append(interval)
            
            # Analyze price movement
            if "quote" in tick_data and len(self.price_jumps[symbol]) > 0:
                current_price = float(tick_data["quote"])
                last_price = self.price_jumps[symbol][-1]
                price_change = abs(current_price - last_price)
                price_change_pct = price_change / last_price if last_price else 0
                
                # Detect unusual price movements
                if len(self.price_jumps[symbol]) > 10:
                    mean_change = np.mean([abs(self.price_jumps[symbol][i] - self.price_jumps[symbol][i-1]) 
                                         for i in range(1, len(self.price_jumps[symbol]))])
                    std_change = np.std([abs(self.price_jumps[symbol][i] - self.price_jumps[symbol][i-1]) 
                                       for i in range(1, len(self.price_jumps[symbol]))])
                    
                    # If price change is more than 3 standard deviations from mean, flag as anomaly
                    if price_change > mean_change + 3 * std_change:
                        self.price_anomalies[symbol].append({
                            "timestamp": now,
                            "server_time": timestamp,
                            "price": current_price,
                            "change": price_change,
                            "change_pct": price_change_pct,
                            "z_score": (price_change - mean_change) / std_change if std_change else 0
                        })
                        self.logger.debug(f"Detected price anomaly for {symbol}: {price_change_pct:.4f}% change")
                
                # Store current price
                self.price_jumps[symbol].append(current_price)
            
            # Analyze synthetic price movements (patterns unique to derived/synthetic assets)
            if self.options.detect_synthetic_price_movements and "is_synthetic" in tick_data:
                if tick_data.get("is_synthetic", False):
                    # Synthetic asset prices often follow deterministic patterns
                    # Store patterns for later exploitation
                    if symbol not in self.synthetic_price_models:
                        self.synthetic_price_models[symbol] = {
                            "samples": deque(maxlen=1000),
                            "patterns": [],
                            "last_analysis": 0
                        }
                    
                    self.synthetic_price_models[symbol]["samples"].append({
                        "time": timestamp,
                        "price": float(tick_data.get("quote", 0)),
                        "direction": 1 if len(self.synthetic_price_models[symbol]["samples"]) == 0 
                                    else (1 if float(tick_data.get("quote", 0)) > 
                                         self.synthetic_price_models[symbol]["samples"][-1]["price"] else -1)
                    })
                    
                    # Periodically analyze for patterns
                    if now - self.synthetic_price_models[symbol]["last_analysis"] > 300:  # Every 5 minutes
                        await self._analyze_synthetic_patterns(symbol)
                        self.synthetic_price_models[symbol]["last_analysis"] = now
            
            # Update platform behavior models periodically
            if (symbol in self.server_response_profiles and 
                len(self.server_response_profiles[symbol]) >= 100):
                await self._update_platform_behavior_model(symbol)
        
        except Exception as e:
            self.logger.error(f"Error analyzing tick data: {str(e)}")
    
    async def analyze_order_book(self, symbol: str, order_book: Dict[str, Any]):
        """
        Analyze order book data to understand liquidity profiles and market structure.
        
        Args:
            symbol: The trading instrument symbol
            order_book: The order book data including bids and asks
        """
        if not self.options.analyze_liquidity_profile:
            return
            
        try:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            
            if not bids or not asks:
                return
                
            # Calculate basic liquidity metrics
            bid_liquidity = sum(float(bid[1]) for bid in bids)
            ask_liquidity = sum(float(ask[1]) for ask in asks)
            
            spread = float(asks[0][0]) - float(bids[0][0]) if bids and asks else 0
            spread_pct = spread / float(bids[0][0]) if bids else 0
            
            # Update liquidity profile
            if symbol not in self.liquidity_profiles:
                self.liquidity_profiles[symbol] = {
                    "spread_history": deque(maxlen=1000),
                    "bid_liquidity_history": deque(maxlen=1000),
                    "ask_liquidity_history": deque(maxlen=1000),
                    "imbalance_history": deque(maxlen=1000),
                    "last_update": time.time()
                }
            
            profile = self.liquidity_profiles[symbol]
            profile["spread_history"].append(spread)
            profile["bid_liquidity_history"].append(bid_liquidity)
            profile["ask_liquidity_history"].append(ask_liquidity)
            profile["imbalance_history"].append((bid_liquidity - ask_liquidity) / (bid_liquidity + ask_liquidity) 
                                               if (bid_liquidity + ask_liquidity) > 0 else 0)
            profile["last_update"] = time.time()
            
            # Detect spread anomalies
            if len(profile["spread_history"]) > 10:
                mean_spread = np.mean(list(profile["spread_history"])[:-1])
                std_spread = np.std(list(profile["spread_history"])[:-1])
                
                if spread > mean_spread + 3 * std_spread:
                    self.spread_anomalies[symbol].append({
                        "timestamp": time.time(),
                        "spread": spread,
                        "spread_pct": spread_pct,
                        "z_score": (spread - mean_spread) / std_spread if std_spread else 0
                    })
                    self.logger.debug(f"Detected spread anomaly for {symbol}: {spread_pct:.4f}% spread")
        
        except Exception as e:
            self.logger.error(f"Error analyzing order book: {str(e)}")
    
    async def analyze_contract_availability(self, symbol: str, contracts_data: Dict[str, Any]):
        """
        Analyze available contracts to detect patterns in contract offerings.
        
        Args:
            symbol: The trading instrument symbol
            contracts_data: The available contracts data
        """
        if not self.options.analyze_contract_availability:
            return
            
        try:
            available_contracts = contracts_data.get("available", [])
            
            # Initialize if this is the first time seeing this symbol
            if symbol not in self.contract_availability_patterns:
                self.contract_availability_patterns[symbol] = {
                    "history": deque(maxlen=100),
                    "patterns": {},
                    "last_analysis": 0
                }
            
            # Record current availability with timestamp
            self.contract_availability_patterns[symbol]["history"].append({
                "timestamp": time.time(),
                "contracts": available_contracts
            })
            
            # Periodically analyze for patterns in contract availability
            now = time.time()
            if now - self.contract_availability_patterns[symbol]["last_analysis"] > 3600:  # Every hour
                await self._analyze_contract_patterns(symbol)
                self.contract_availability_patterns[symbol]["last_analysis"] = now
        
        except Exception as e:
            self.logger.error(f"Error analyzing contract availability: {str(e)}")
    
    async def _analyze_synthetic_patterns(self, symbol: str):
        """Analyze patterns in synthetic asset price movements."""
        try:
            if symbol not in self.synthetic_price_models:
                return
                
            samples = list(self.synthetic_price_models[symbol]["samples"])
            if len(samples) < 50:
                return
                
            # Look for repeating directional patterns
            # This is a simplified pattern detection - in production we would use
            # more sophisticated time series analysis and machine learning
            
            directions = [s["direction"] for s in samples]
            
            # Look for patterns of length 3-10 moves
            for pattern_length in range(3, min(11, len(directions) // 3)):
                patterns = {}
                
                # Extract all subsequences of the given length
                for i in range(len(directions) - pattern_length):
                    pattern = tuple(directions[i:i+pattern_length])
                    
                    if pattern not in patterns:
                        patterns[pattern] = []
                    
                    # If there's enough room for a "next" direction, record it
                    if i + pattern_length < len(directions):
                        patterns[pattern].append(directions[i+pattern_length])
                
                # Find patterns that have predictive value
                for pattern, next_dirs in patterns.items():
                    if len(next_dirs) >= 5:  # Need enough samples
                        # Calculate probability of each direction following this pattern
                        up_prob = next_dirs.count(1) / len(next_dirs)
                        down_prob = next_dirs.count(-1) / len(next_dirs)
                        
                        # If there's a strong bias in one direction, record this pattern
                        if up_prob >= 0.7 or down_prob >= 0.7:
                            self.synthetic_price_models[symbol]["patterns"].append({
                                "pattern": pattern,
                                "up_probability": up_prob,
                                "down_probability": down_prob,
                                "samples": len(next_dirs)
                            })
                            
                            self.logger.info(
                                f"Detected predictive pattern for {symbol}: "
                                f"Pattern: {pattern}, Up: {up_prob:.2f}, Down: {down_prob:.2f}, "
                                f"Samples: {len(next_dirs)}"
                            )
            
            # Keep only the top 20 most predictive patterns
            self.synthetic_price_models[symbol]["patterns"] = sorted(
                self.synthetic_price_models[symbol]["patterns"],
                key=lambda x: max(x["up_probability"], x["down_probability"]),
                reverse=True
            )[:20]
            
            # Store the patterns in Redis for persistence
            await self.redis.set(
                f"deriv:synthetic_patterns:{symbol}", 
                json.dumps(self.synthetic_price_models[symbol]["patterns"])
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing synthetic patterns: {str(e)}")
    
    async def _update_platform_behavior_model(self, symbol: str):
        """Update platform behavior models based on collected data."""
        try:
            # Analyze server response timing
            if symbol in self.server_response_profiles and len(self.server_response_profiles[symbol]) > 10:
                intervals = list(self.server_response_profiles[symbol])
                
                # Clean outliers (values > 3 std devs from mean)
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                clean_intervals = [i for i in intervals if abs(i - mean_interval) <= 3 * std_interval]
                
                if clean_intervals:
                    # Calculate typical interval range
                    mean_clean = np.mean(clean_intervals)
                    std_clean = np.std(clean_intervals)
                    
                    # Store in Redis for model use
                    await self.redis.set(
                        f"deriv:response_profile:{symbol}",
                        json.dumps({
                            "mean_interval": mean_clean,
                            "std_interval": std_clean,
                            "min_interval": min(clean_intervals),
                            "max_interval": max(clean_intervals),
                            "updated_at": time.time()
                        })
                    )
                    
                    self.logger.debug(
                        f"Updated response profile for {symbol}: "
                        f"Mean: {mean_clean:.4f}s, Std: {std_clean:.4f}s"
                    )
                    
                    # Clear the deque to start fresh
                    self.server_response_profiles[symbol].clear()
            
            # Store any detected anomalies
            if symbol in self.price_anomalies and self.price_anomalies[symbol]:
                await self.redis.set(
                    f"deriv:price_anomalies:{symbol}",
                    json.dumps(self.price_anomalies[symbol][-100:])  # Keep last 100 anomalies
                )
                
            if symbol in self.spread_anomalies and self.spread_anomalies[symbol]:
                await self.redis.set(
                    f"deriv:spread_anomalies:{symbol}",
                    json.dumps(self.spread_anomalies[symbol][-100:])  # Keep last 100 anomalies
                )
        
        except Exception as e:
            self.logger.error(f"Error updating platform behavior model: {str(e)}")
    
    async def _analyze_contract_patterns(self, symbol: str):
        """Analyze patterns in contract availability."""
        try:
            if symbol not in self.contract_availability_patterns:
                return
                
            history = list(self.contract_availability_patterns[symbol]["history"])
            if len(history) < 10:
                return
                
            # Analyze time-of-day patterns in contract availability
            contract_types = set()
            for entry in history:
                for contract in entry.get("contracts", []):
                    contract_types.add(contract.get("contract_type", ""))
            
            # For each contract type, check time-of-day availability patterns
            patterns = {}
            for contract_type in contract_types:
                if not contract_type:
                    continue
                    
                # Map availability to hour of day
                hour_availability = [0] * 24
                hour_samples = [0] * 24
                
                for entry in history:
                    dt = datetime.fromtimestamp(entry["timestamp"])
                    hour = dt.hour
                    
                    # Check if this contract type is available
                    is_available = any(
                        c.get("contract_type", "") == contract_type 
                        for c in entry.get("contracts", [])
                    )
                    
                    hour_availability[hour] += 1 if is_available else 0
                    hour_samples[hour] += 1
                
                # Calculate availability percentage by hour
                hourly_pattern = [
                    hour_availability[h] / hour_samples[h] if hour_samples[h] > 0 else 0
                    for h in range(24)
                ]
                
                patterns[contract_type] = hourly_pattern
            
            # Store the patterns
            self.contract_availability_patterns[symbol]["patterns"] = patterns
            
            # Save to Redis
            await self.redis.set(
                f"deriv:contract_patterns:{symbol}",
                json.dumps(patterns)
            )
            
            self.logger.info(f"Updated contract availability patterns for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error analyzing contract patterns: {str(e)}")
    
    def get_platform_insights(self, symbol: str) -> Dict[str, Any]:
        """
        Get insights about platform behavior for a specific symbol.
        
        Args:
            symbol: The trading instrument symbol
            
        Returns:
            Dict containing insights about platform behavior
        """
        insights = {
            "has_synthetic_patterns": False,
            "synthetic_patterns": [],
            "has_liquidity_profile": False,
            "liquidity_profile": {},
            "has_contract_patterns": False,
            "contract_patterns": {},
            "anomalies": {
                "price": [],
                "spread": []
            }
        }
        
        # Add synthetic patterns if available
        if (symbol in self.synthetic_price_models and 
            self.synthetic_price_models[symbol]["patterns"]):
            insights["has_synthetic_patterns"] = True
            insights["synthetic_patterns"] = self.synthetic_price_models[symbol]["patterns"]
        
        # Add liquidity profile if available
        if symbol in self.liquidity_profiles:
            profile = self.liquidity_profiles[symbol]
            if profile["spread_history"]:
                insights["has_liquidity_profile"] = True
                insights["liquidity_profile"] = {
                    "avg_spread": np.mean(list(profile["spread_history"])),
                    "avg_bid_liquidity": np.mean(list(profile["bid_liquidity_history"])),
                    "avg_ask_liquidity": np.mean(list(profile["ask_liquidity_history"])),
                    "avg_imbalance": np.mean(list(profile["imbalance_history"])),
                    "updated_at": profile["last_update"]
                }
        
        # Add contract patterns if available
        if (symbol in self.contract_availability_patterns and 
            self.contract_availability_patterns[symbol]["patterns"]):
            insights["has_contract_patterns"] = True
            insights["contract_patterns"] = self.contract_availability_patterns[symbol]["patterns"]
        
        # Add recent anomalies
        if symbol in self.price_anomalies:
            insights["anomalies"]["price"] = self.price_anomalies[symbol][-5:]  # Last 5 anomalies
            
        if symbol in self.spread_anomalies:
            insights["anomalies"]["spread"] = self.spread_anomalies[symbol][-5:]  # Last 5 anomalies
        
        return insights


class DerivContractHandler:
    """
    Handles contract specification, validation, and optimal parameter selection
    for Deriv trading instruments. This helps maximize win rates by selecting
    the most favorable contract parameters.
    """
    
    def __init__(self):
        self.logger = get_logger("deriv_contract_handler")
        self.contracts_cache = {}
        self.contract_performance = {}
        self.redis = RedisClient()
    
    async def initialize(self):
        """Initialize the contract handler with cached data if available."""
        try:
            # Load contract performance data from Redis
            performance_data = await self.redis.get("deriv:contract_performance")
            if performance_data:
                self.contract_performance = json.loads(performance_data)
                self.logger.info(f"Loaded performance data for {len(self.contract_performance)} contract configurations")
        except Exception as e:
            self.logger.warning(f"Failed to load contract performance data: {str(e)}")
    
    async def update_contracts(self, symbol: str, contracts_data: Dict[str, Any]):
        """
        Update the contracts cache with the latest contract specifications.
        
        Args:
            symbol: The trading instrument symbol
            contracts_data: The contracts data from the API
        """
        try:
            if not contracts_data:
                return
                
            self.contracts_cache[symbol] = {
                "timestamp": time.time(),
                "data": contracts_data
            }
            
            # Store in Redis for persistence
            await self.redis.set(
                f"deriv:contracts:{symbol}",
                json.dumps({
                    "timestamp": time.time(),
                    "data": contracts_data
                }),
                ex=86400  # Expire after 24 hours
            )
            
            self.logger.debug(f"Updated contracts cache for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error updating contracts cache: {str(e)}")
    
    def get_contracts(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the cached contract specifications for a symbol.
        
        Args:
            symbol: The trading instrument symbol
            
        Returns:
            Dict containing contract specifications or None if not available
        """
        if symbol in self.contracts_cache:
            cache_entry = self.contracts_cache[symbol]
            # If cache is less than 1 hour old, return it
            if time.time() - cache_entry["timestamp"] < 3600:
                return cache_entry["data"]
        
        return None
    
    async def find_optimal_contract(self, symbol: str, contract_type: str, 
                                  duration_unit: str, platform_insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find the optimal contract configuration based on historical performance
        and current market conditions.
        
        Args:
            symbol: The trading instrument symbol
            contract_type: The type of contract (e.g., 'CALL', 'PUT', 'DIGITOVER', etc.)
            duration_unit: The duration unit (e.g., 'm', 'h', 'd')
            platform_insights: Insights about platform behavior
            
        Returns:
            Dict containing optimal contract parameters
        """
        try:
            # Get contract specifications
            contracts = self.get_contracts(symbol)
            if not contracts:
                self.logger.warning(f"No contract specifications available for {symbol}")
                return {}
            
            # Filter available contracts by type and duration unit
            available_contracts = []
            for contract in contracts.get("available", []):
                if contract.get("contract_type") == contract_type:
                    durations = contract.get("trading_period", {}).get("available", [])
                    for duration in durations:
                        if duration.get("unit") == duration_unit:
                            available_contracts.append({
                                "contract_type": contract_type,
                                "duration_unit": duration_unit,
                                "min_duration": duration.get("min", 0),
                                "max_duration": duration.get("max", 0)
                            })
            
            if not available_contracts:
                self.logger.warning(f"No matching contracts for {symbol}, type {contract_type}, unit {duration_unit}")
                return {}
            
            # Now find the optimal duration based on historical performance
            optimal_contract = None
            max_score = -float('inf')
            
            for contract in available_contracts:
                # Use the middle of the duration range as a starting point
                min_duration = contract["min_duration"]
                max_duration = contract["max_duration"]
                
                # Try several durations within the allowed range
                for duration in range(min_duration, max_duration + 1, max(1, (max_duration - min_duration) // 5)):
                    contract_key = f"{symbol}:{contract_type}:{duration_unit}:{duration}"
                    
                    # Get historical performance data for this configuration
                    performance = self.contract_performance.get(contract_key, {
                        "wins": 0,
                        "losses": 0,
                        "total": 0
                    })
                    
                    # Calculate win rate (with Bayesian smoothing for small sample sizes)
                    alpha = 1  # Prior "wins"
                    beta = 1   # Prior "losses"
                    wins = performance.get("wins", 0)
                    total = performance.get("total", 0)
                    
                    if total == 0:
                        win_rate = 0.5  # No data, assume 50% win rate
                    else:
                        # Bayesian smoothed win rate
                        win_rate = (wins + alpha) / (total + alpha + beta)
                    
                    # Calculate confidence based on sample size
                    confidence = min(1.0, total / 50.0)  # Scale up to 1.0 as we get more samples
                    
                    # Incorporate platform insights if available
                    insight_bonus = 0
                    
                    if platform_insights.get("has_synthetic_patterns") and "is_synthetic" in contracts:
                        # Add bonus for synthetic patterns if this is a synthetic asset
                        insight_bonus += 0.1
                        
                    if platform_insights.get("has_contract_patterns"):
                        contract_patterns = platform_insights.get("contract_patterns", {})
                        if contract_type in contract_patterns:
                            # Get current hour
                            current_hour = datetime.now().hour
                            hourly_pattern = contract_patterns[contract_type]
                            
                            # If availability is high during current hour, add bonus
                            if current_hour < len(hourly_pattern) and hourly_pattern[current_hour] > 0.7:
                                insight_bonus += 0.15
                    
                    # Calculate overall score for this contract configuration
                    score = (win_rate * 0.7) + (confidence * 0.2) + insight_bonus
                    
                    if score > max_score:
                        max_score = score
                        optimal_contract = {
                            "contract_type": contract_type,
                            "duration_unit": duration_unit,
                            "duration": duration,
                            "win_rate": win_rate,
                            "confidence": confidence,
                            "samples": total,
                            "score": score
                        }
            
            if optimal_contract:
                self.logger.info(
                    f"Found optimal contract for {symbol}: "
                    f"Type: {optimal_contract['contract_type']}, "
                    f"Duration: {optimal_contract['duration']} {optimal_contract['duration_unit']}, "
                    f"Win Rate: {optimal_contract['win_rate']:.2f}, "
                    f"Confidence: {optimal_contract['confidence']:.2f}"
                )
                return optimal_contract
            else:
                self.logger.warning(f"Could not determine optimal contract for {symbol}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error finding optimal contract: {str(e)}")
            return {}
    
    async def record_contract_result(self, symbol: str, contract_type: str, 
                                   duration_unit: str, duration: int, 
                                   result: bool):
        """
        Record the result of a contract to improve future recommendations.
        
        Args:
            symbol: The trading instrument symbol
            contract_type: The type of contract
            duration_unit: The duration unit
            duration: The duration value
            result: True for win, False for loss
        """
        try:
            contract_key = f"{symbol}:{contract_type}:{duration_unit}:{duration}"
            
            if contract_key not in self.contract_performance:
                self.contract_performance[contract_key] = {
                    "wins": 0,
                    "losses": 0,
                    "total": 0
                }
            
            if result:
                self.contract_performance[contract_key]["wins"] += 1
            else:
                self.contract_performance[contract_key]["losses"] += 1
                
            self.contract_performance[contract_key]["total"] += 1
            
            # Store in Redis for persistence
            await self.redis.set(
                "deriv:contract_performance",
                json.dumps(self.contract_performance)
            )
            
            self.logger.debug(
                f"Recorded contract result for {contract_key}: "
                f"{'Win' if result else 'Loss'}"
            )
            
        except Exception as e:
            self.logger.error(f"Error recording contract result: {str(e)}")


class DerivFeed(BaseDataFeed):
    """
    Advanced Deriv platform data feed with sophisticated pattern recognition, platform behavior 
    modeling, and market microstructure analysis to exploit inefficiencies and achieve superior 
    trading results.
    """
    
    def __init__(self, credentials: DerivCredentials, options: Optional[DerivFeedOptions] = None):
        """
        Initialize the Deriv feed with the provided credentials and options.
        
        Args:
            credentials: API credentials for authentication
            options: Optional configuration options
        """
        self.credentials = credentials
        self.options = options or DerivFeedOptions()
        
        super().__init__("deriv", self.options)

        
        # Platform and market state trackers
        self.connections = {}
        self.subscriptions = {}
        self.instruments = {}
        self.active_symbols = {}
        self.request_id_counter = AtomicCounter()
        self.request_callbacks = {}
        self.last_ping_time = {}
        self.last_pong_time = {}
        self.connection_status = {}
        self.market_open_status = {}
        
        # Component initialization
        self.platform_analyzer = DerivPlatformAnalyzer(self.options)
        self.contract_handler = DerivContractHandler()
        
        # Locks for thread safety
        self.connection_locks = {}
        self.subscription_locks = {}
        self.tasks: List[asyncio.Task] = []
        
        # Rate limiters
        self.rest_rate_limiter = AsyncRateLimiter(
            rate_limit=self.options.rate_limit_max_requests,
            time_period=self.options.rate_limit_period
        )
        
        # Monitoring
        self.connection_attempts = defaultdict(int)
        self.connection_failures = defaultdict(int)
        self.subscription_attempts = defaultdict(int)
        self.subscription_failures = defaultdict(int)
        self.message_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.last_tick_time = {}
        self.tick_latencies = defaultdict(lambda: deque(maxlen=100))
    
    async def initialize(self):
        """Initialize the feed with necessary setup before data streaming."""
        self.logger.info("Initializing Deriv feed")
        
        # Validate credentials
        if not self.credentials.validate():
            raise FeedAuthenticationError("Invalid Deriv credentials provided")
        
        # Initialize components
        await self.platform_analyzer.initialize()
        await self.contract_handler.initialize()
        
        # Register with the metrics system
        metrics_tags = {"feed": "deriv"}
        self.metrics.register_gauge("feed.connection_status", metrics_tags)
        self.metrics.register_counter("feed.messages_received", metrics_tags)
        self.metrics.register_counter("feed.errors", metrics_tags)
        self.metrics.register_histogram("feed.tick_latency", metrics_tags)
        self.metrics.register_gauge("feed.subscriptions", metrics_tags)
    
    async def start(self):
        """Start the feed data streaming and processing."""
        self.logger.info("Starting Deriv feed")
        await self.initialize()
        
        # Fetch supported instruments
        await self.fetch_active_symbols()
        
        # Log capabilities based on configuration
        self.logger.info(f"Platform behavior modeling: {self.options.model_platform_behavior}")
        self.logger.info(f"Pattern detection: {self.options.detect_platform_patterns}")
        self.logger.info(f"Liquidity analysis: {self.options.analyze_liquidity_profile}")
        
        # Start background tasks
        self.create_task(self.monitor_connections(), name="deriv_connection_monitor")
        self.create_task(self.metrics_reporter(), name="deriv_metrics_reporter")
        
        self.logger.info("Deriv feed started successfully")
    
    async def stop(self):
        """Stop the feed and clean up resources."""
        self.logger.info("Stopping Deriv feed")
        
        # Close all connections
        for endpoint in list(self.connections.keys()):
            await self.close_connection(endpoint)
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.logger.info("Deriv feed stopped")
    
    async def subscribe(self, instrument_id: str):
        """
        Subscribe to updates for the specified instrument.
        
        Args:
            instrument_id: The normalized instrument identifier
        """
        try:
            # Get the appropriate symbol for this instrument
            symbol = self.get_symbol_for_instrument(instrument_id)
            if not symbol:
                raise FeedSubscriptionError(f"Unknown instrument: {instrument_id}")
            
            # Lock to prevent concurrent subscription attempts for the same instrument
            if instrument_id not in self.subscription_locks:
                self.subscription_locks[instrument_id] = asyncio.Lock()
                
            async with self.subscription_locks[instrument_id]:
                if self.is_subscribed(instrument_id):
                    self.logger.debug(f"Already subscribed to {instrument_id}")
                    return True
                    
                # Increment metrics
                self.subscription_attempts[instrument_id] += 1
                
                # Connect if needed
                endpoint = self.options.endpoints["websocket"]
                if not await self.ensure_connection(endpoint):
                    raise FeedConnectionError(f"Failed to connect to Deriv API: {endpoint}")
                
                # Subscribe to tick stream
                self.logger.info(f"Subscribing to ticks for {symbol}")
                
                tick_sub_req = {
                    "ticks": symbol,
                    "subscribe": 1
                }
                
                if self.options.use_enhanced_ticks:
                    tick_sub_req["style"] = "ticks"
                
                tick_sub_id = await self.send_request(endpoint, tick_sub_req)
                
                # Wait for subscription confirmation
                response = await self.wait_for_response(tick_sub_id, timeout=self.options.subscription_timeout)
                
                if not response or "error" in response:
                    error_msg = response.get("error", {}).get("message", "Unknown error") if response else "Timeout"
                    raise FeedSubscriptionError(f"Failed to subscribe to ticks for {symbol}: {error_msg}")
                
                # Subscribe to order book if supported and configured
                orderbook_sub_id = None
                if self.options.order_book_depth > 0:
                    try:
                        orderbook_sub_req = {
                            "ticks": symbol,
                            "subscribe": 1,
                            "style": "orderBook",
                            "depth": self.options.order_book_depth
                        }
                        
                        orderbook_sub_id = await self.send_request(endpoint, orderbook_sub_req)
                        # We don't wait for confirmation as not all symbols support order book
                    except Exception as e:
                        self.logger.warning(f"Order book subscription failed for {symbol}: {str(e)}")
                
                # Get contract specifications
                try:
                    contracts_req = {
                        "contracts_for": symbol,
                        "currency": "USD",  # Use default currency
                        "landing_company": "svg"  # Default landing company
                    }
                    
                    contracts_req_id = await self.send_request(endpoint, contracts_req)
                    contracts_resp = await self.wait_for_response(contracts_req_id, timeout=self.options.subscription_timeout)
                    
                    if contracts_resp and "error" not in contracts_resp:
                        await self.contract_handler.update_contracts(symbol, contracts_resp.get("contracts_for"))
                        
                        # Analyze contract availability if enabled
                        if self.options.analyze_contract_availability:
                            await self.platform_analyzer.analyze_contract_availability(
                                symbol, contracts_resp.get("contracts_for", {})
                            )
                except Exception as e:
                    self.logger.warning(f"Failed to fetch contract specifications for {symbol}: {str(e)}")
                
                # Store subscription details
                self.subscriptions[instrument_id] = {
                    "symbol": symbol,
                    "endpoint": endpoint,
                    "tick_sub_id": tick_sub_id,
                    "orderbook_sub_id": orderbook_sub_id,
                    "last_tick": None,
                    "subscribed_at": time.time()
                }
                
                self.metrics.set_gauge("feed.subscriptions", len(self.subscriptions), {"feed": "deriv"})
                
                self.logger.info(f"Successfully subscribed to {instrument_id} (symbol: {symbol})")
                return True
                
        except Exception as e:
            self.subscription_failures[instrument_id] += 1
            self.logger.error(f"Error subscribing to {instrument_id}: {str(e)}")
            record_failure("deriv_subscription", {"instrument": instrument_id})
            return False
    
    async def unsubscribe(self, instrument_id: str):
        """
        Unsubscribe from updates for the specified instrument.
        
        Args:
            instrument_id: The normalized instrument identifier
        """
        try:
            if instrument_id not in self.subscriptions:
                self.logger.debug(f"Not subscribed to {instrument_id}")
                return True
            
            # Lock to prevent concurrent operations
            async with self.subscription_locks.get(instrument_id, asyncio.Lock()):
                subscription = self.subscriptions.get(instrument_id)
                if not subscription:
                    return True
                    
                symbol = subscription["symbol"]
                endpoint = subscription["endpoint"]
                
                # Unsubscribe from tick stream
                if "tick_sub_id" in subscription:
                    try:
                        unsubscribe_req = {
                            "forget": subscription["tick_sub_id"]
                        }
                        
                        forget_req_id = await self.send_request(endpoint, unsubscribe_req)
                        response = await self.wait_for_response(forget_req_id, timeout=10)
                        
                        if not response or "error" in response:
                            self.logger.warning(
                                f"Error unsubscribing from ticks for {symbol}: "
                                f"{response.get('error', {}).get('message', 'Unknown error') if response else 'Timeout'}"
                            )
                    except Exception as e:
                        self.logger.warning(f"Error sending unsubscribe request for {symbol} ticks: {str(e)}")
                
                # Unsubscribe from order book if subscribed
                if "orderbook_sub_id" in subscription and subscription["orderbook_sub_id"]:
                    try:
                        unsubscribe_ob_req = {
                            "forget": subscription["orderbook_sub_id"]
                        }
                        
                        forget_ob_req_id = await self.send_request(endpoint, unsubscribe_ob_req)
                        # No need to wait for response
                    except Exception as e:
                        self.logger.warning(f"Error sending unsubscribe request for {symbol} order book: {str(e)}")
                
                # Remove subscription
                del self.subscriptions[instrument_id]
                
                self.metrics.set_gauge("feed.subscriptions", len(self.subscriptions), {"feed": "deriv"})
                
                self.logger.info(f"Unsubscribed from {instrument_id} (symbol: {symbol})")
                
                # If this was the last subscription for this connection, consider closing it
                if not any(sub["endpoint"] == endpoint for sub in self.subscriptions.values()):
                    self.logger.debug(f"No more subscriptions for endpoint {endpoint}, closing connection")
                    await self.close_connection(endpoint)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error unsubscribing from {instrument_id}: {str(e)}")
            return False
    
    def is_subscribed(self, instrument_id: str) -> bool:
        """
        Check if we're subscribed to the specified instrument.
        
        Args:
            instrument_id: The normalized instrument identifier
            
        Returns:
            True if subscribed, False otherwise
        """
        return instrument_id in self.subscriptions
    
    async def get_snapshot(self, instrument_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest market data snapshot for the specified instrument.
        
        Args:
            instrument_id: The normalized instrument identifier
            
        Returns:
            Dict containing the latest market data or None if not available
        """
        if not self.is_subscribed(instrument_id):
            await self.subscribe(instrument_id)
            
        if instrument_id in self.subscriptions:
            # Return the last tick data if available
            last_tick = self.subscriptions[instrument_id].get("last_tick")
            if last_tick:
                return {
                    "instrument_id": instrument_id,
                    "timestamp": last_tick.get("epoch", time.time()) * 1000,  # Convert to milliseconds
                    "bid": last_tick.get("bid", last_tick.get("quote")),
                    "ask": last_tick.get("ask", last_tick.get("quote")),
                    "last": last_tick.get("quote"),
                    "volume": last_tick.get("volume", 0),
                    "symbol": self.subscriptions[instrument_id]["symbol"],
                    "source": "deriv"
                }
        
        return None
    
    async def get_order_book(self, instrument_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest order book data for the specified instrument.
        
        Args:
            instrument_id: The normalized instrument identifier
            
        Returns:
            Dict containing order book data or None if not available
        """
        if not self.is_subscribed(instrument_id):
            await self.subscribe(instrument_id)
            
        if instrument_id in self.subscriptions:
            order_book = self.subscriptions[instrument_id].get("order_book")
            if order_book:
                return {
                    "instrument_id": instrument_id,
                    "timestamp": order_book.get("timestamp", time.time() * 1000),
                    "bids": order_book.get("bids", []),
                    "asks": order_book.get("asks", []),
                    "symbol": self.subscriptions[instrument_id]["symbol"],
                    "source": "deriv"
                }
        
        return None
    
    def get_supported_instruments(self) -> List[Dict[str, Any]]:
        """
        Get the list of instruments supported by this feed.
        
        Returns:
            List of instrument definitions
        """
        return list(self.instruments.values())
    
    def get_platform_insights(self, instrument_id: str) -> Dict[str, Any]:
        """
        Get insights about platform behavior for a specific instrument.
        
        Args:
            instrument_id: The normalized instrument identifier
            
        Returns:
            Dict containing insights about platform behavior
        """
        if instrument_id in self.subscriptions:
            symbol = self.subscriptions[instrument_id]["symbol"]
            return self.platform_analyzer.get_platform_insights(symbol)
        
        return {}
    
    async def find_optimal_contract(self, instrument_id: str, contract_type: str, 
                                  duration_unit: str) -> Dict[str, Any]:
        """
        Find the optimal contract configuration for trading.
        
        Args:
            instrument_id: The normalized instrument identifier
            contract_type: The type of contract (e.g., 'CALL', 'PUT')
            duration_unit: The duration unit (e.g., 'm', 'h', 'd')
            
        Returns:
            Dict containing optimal contract parameters
        """
        if instrument_id not in self.subscriptions:
            await self.subscribe(instrument_id)
            
        if instrument_id in self.subscriptions:
            symbol = self.subscriptions[instrument_id]["symbol"]
            insights = self.platform_analyzer.get_platform_insights(symbol)
            return await self.contract_handler.find_optimal_contract(
                symbol, contract_type, duration_unit, insights
            )
        
        return {}
    
    async def record_contract_result(self, instrument_id: str, contract_type: str, 
                                   duration_unit: str, duration: int, 
                                   result: bool):
        """
        Record the result of a contract to improve future recommendations.
        
        Args:
            instrument_id: The normalized instrument identifier
            contract_type: The type of contract
            duration_unit: The duration unit
            duration: The duration value
            result: True for win, False for loss
        """
        if instrument_id in self.subscriptions:
            symbol = self.subscriptions[instrument_id]["symbol"]
            await self.contract_handler.record_contract_result(
                symbol, contract_type, duration_unit, duration, result
            )
    
    # WebSocket connection management
    async def ensure_connection(self, endpoint: str) -> bool:
        """
        Ensure an active connection to the specified endpoint.
        
        Args:
            endpoint: The WebSocket endpoint URL
            
        Returns:
            True if connected, False otherwise
        """
        # Create lock if it doesn't exist
        if endpoint not in self.connection_locks:
            self.connection_locks[endpoint] = asyncio.Lock()
        
        async with self.connection_locks[endpoint]:
            if self.is_connected(endpoint):
                return True
                
            return await self.connect(endpoint)
    
    async def connect(self, endpoint: str) -> bool:
        """
        Connect to the specified WebSocket endpoint.
        
        Args:
            endpoint: The WebSocket endpoint URL
            
        Returns:
            True if connected successfully, False otherwise
        """
        self.logger.info(f"Connecting to Deriv API: {endpoint}")
        self.connection_attempts[endpoint] += 1
        
        try:
            # Create new WebSocket connection
            self.logger.debug(f"Opening WebSocket connection to {endpoint}")
            connection = await websockets.connect(
                endpoint,
                ssl=self.options.use_ssl,
                max_size=None,  # No message size limit
                close_timeout=10,
                compression=None,
                ping_interval=None,  # We'll handle pings manually
                ping_timeout=None
            )
            
            # Store connection
            self.connections[endpoint] = connection
            self.connection_status[endpoint] = True
            self.metrics.set_gauge("feed.connection_status", 1, {"feed": "deriv", "endpoint": endpoint})
            
            # Initialize connection state
            self.last_ping_time[endpoint] = time.time()
            self.last_pong_time[endpoint] = time.time()
            
            # Start message handler for this connection
            self.create_task(
                self.message_handler(endpoint, connection), 
                name=f"deriv_message_handler_{hash(endpoint)}"
            )
            
            # Start ping task to keep the connection alive
            self.create_task(
                self.ping_handler(endpoint), 
                name=f"deriv_ping_handler_{hash(endpoint)}"
            )
            
            # Authorize if we have an API token
            if self.credentials.api_token:
                authorized = await self.authorize(endpoint)
                if not authorized:
                    self.logger.warning(f"Failed to authorize with Deriv API using provided token")
                    # Continue anyway, many operations work without authorization
            
            self.logger.info(f"Successfully connected to Deriv API: {endpoint}")
            record_success("deriv_connection", {"endpoint": endpoint})
            return True
            
        except Exception as e:
            self.connection_failures[endpoint] += 1
            self.logger.error(f"Failed to connect to Deriv API {endpoint}: {str(e)}")
            self.connection_status[endpoint] = False
            self.metrics.set_gauge("feed.connection_status", 0, {"feed": "deriv", "endpoint": endpoint})
            record_failure("deriv_connection", {"endpoint": endpoint, "error": str(e)})
            return False
    
    async def close_connection(self, endpoint: str):
        """
        Close the connection to the specified endpoint.
        
        Args:
            endpoint: The WebSocket endpoint URL
        """
        async with self.connection_locks.get(endpoint, asyncio.Lock()):
            connection = self.connections.get(endpoint)
            if not connection:
                return
                
            self.logger.info(f"Closing connection to {endpoint}")
            
            try:
                await connection.close()
                self.logger.debug(f"WebSocket connection to {endpoint} closed")
            except Exception as e:
                self.logger.warning(f"Error closing WebSocket connection to {endpoint}: {str(e)}")
            
            # Clean up connection state
            if endpoint in self.connections:
                del self.connections[endpoint]
            
            self.connection_status[endpoint] = False
            self.metrics.set_gauge("feed.connection_status", 0, {"feed": "deriv", "endpoint": endpoint})
    
    def is_connected(self, endpoint: str) -> bool:
        """
        Check if we're connected to the specified endpoint.
        
        Args:
            endpoint: The WebSocket endpoint URL
            
        Returns:
            True if connected, False otherwise
        """
        connection = self.connections.get(endpoint)
        return connection is not None and not connection.closed
    
    async def authorize(self, endpoint: str) -> bool:
        """
        Send authorization request to the Deriv API.
        
        Args:
            endpoint: The WebSocket endpoint URL
            
        Returns:
            True if authorized successfully, False otherwise
        """
        if not self.credentials.api_token:
            return False
            
        try:
            auth_req = {
                "authorize": self.credentials.api_token
            }
            
            auth_req_id = await self.send_request(endpoint, auth_req)
            auth_resp = await self.wait_for_response(auth_req_id, timeout=30)
            
            if not auth_resp or "error" in auth_resp:
                error_msg = auth_resp.get("error", {}).get("message", "Unknown error") if auth_resp else "Timeout"
                self.logger.error(f"Authorization failed: {error_msg}")
                return False
                
            self.logger.info("Successfully authorized with Deriv API")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during authorization: {str(e)}")
            return False
    
    async def send_request(self, endpoint: str, request: Dict[str, Any]) -> str:
        """
        Send a request to the Deriv API.
        
        Args:
            endpoint: The WebSocket endpoint URL
            request: The request payload
            
        Returns:
            Request ID for tracking the response
        """
        if not self.is_connected(endpoint):
            if not await self.ensure_connection(endpoint):
                raise FeedConnectionError(f"Not connected to {endpoint}")
        
        connection = self.connections[endpoint]
        
        # Generate request ID if not provided
        req_id = str(self.request_id_counter.increment())
        request["req_id"] = req_id
        
        # Add app_id if not already present
        if "app_id" not in request:
            request["app_id"] = self.credentials.app_id
        
        # Apply rate limiting
        await self.rest_rate_limiter.acquire()
        
        try:
            # Send the request
            request_json = json.dumps(request)
            self.logger.debug(f"Sending request to {endpoint}: {request_json}")
            await connection.send(request_json)
            
            return req_id
            
        except Exception as e:
            self.error_counts[endpoint] += 1
            self.logger.error(f"Error sending request to {endpoint}: {str(e)}")
            raise FeedConnectionError(f"Failed to send request: {str(e)}")
    
    async def wait_for_response(self, req_id: str, timeout: float = 30) -> Optional[Dict[str, Any]]:
        """
        Wait for a response to a specific request.
        
        Args:
            req_id: The request ID to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            The response payload or None if timed out
        """
        response_future = asyncio.Future()
        self.request_callbacks[req_id] = response_future
        
        try:
            return await asyncio.wait_for(response_future, timeout=timeout)
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout waiting for response to request {req_id}")
            return None
        finally:
            # Clean up the callback
            if req_id in self.request_callbacks:
                del self.request_callbacks[req_id]
    
    async def message_handler(self, endpoint: str, connection):
        """
        Handle incoming WebSocket messages for a specific connection.
        
        Args:
            endpoint: The WebSocket endpoint URL
            connection: The WebSocket connection object
        """
        self.logger.debug(f"Starting message handler for {endpoint}")
        
        try:
            async for message in connection:
                try:
                    # Parse message
                    data = json.loads(message)
                    self.message_counts[endpoint] += 1
                    self.metrics.increment_counter("feed.messages_received", {"feed": "deriv", "endpoint": endpoint})
                    
                    # Handle pong message
                    if "ping" in data:
                        self.last_pong_time[endpoint] = time.time()
                        await connection.send(json.dumps({"pong": data["ping"]}))
                        continue
                    
                    # Handle incoming message
                    await self.process_message(endpoint, data)
                    
                except json.JSONDecodeError:
                    self.logger.warning(f"Received invalid JSON from {endpoint}: {message[:100]}...")
                    self.error_counts[endpoint] += 1
                    self.metrics.increment_counter("feed.errors", {"feed": "deriv", "endpoint": endpoint, "type": "json_parse"})
                except Exception as e:
                    self.logger.error(f"Error processing message from {endpoint}: {str(e)}")
                    self.error_counts[endpoint] += 1
                    self.metrics.increment_counter("feed.errors", {"feed": "deriv", "endpoint": endpoint, "type": "processing"})
        
        except websockets.exceptions.ConnectionClosed as e:
            self.logger.warning(f"WebSocket connection to {endpoint} closed: {str(e)}")
            self.connection_status[endpoint] = False
            self.metrics.set_gauge("feed.connection_status", 0, {"feed": "deriv", "endpoint": endpoint})
            
            # Remove the connection
            if endpoint in self.connections:
                del self.connections[endpoint]
            
            # Attempt to reconnect for active subscriptions
            active_subs = [inst_id for inst_id, sub in self.subscriptions.items() if sub["endpoint"] == endpoint]
            if active_subs:
                self.logger.info(f"Attempting to reconnect for {len(active_subs)} active subscriptions")
                await self.reconnect_subscriptions(active_subs)
        
        except Exception as e:
            self.logger.error(f"Unexpected error in message handler for {endpoint}: {str(e)}")
            self.error_counts[endpoint] += 1
            self.metrics.increment_counter("feed.errors", {"feed": "deriv", "endpoint": endpoint, "type": "handler"})
    
    async def ping_handler(self, endpoint: str):
        """
        Periodically send ping messages to keep the connection alive.
        
        Args:
            endpoint: The WebSocket endpoint URL
        """
        self.logger.debug(f"Starting ping handler for {endpoint}")
        
        while endpoint in self.connections and self.is_connected(endpoint):
            try:
                now = time.time()
                
                # Send ping message every ping_interval seconds
                if now - self.last_ping_time.get(endpoint, 0) >= self.options.ping_interval:
                    if endpoint in self.connections and self.is_connected(endpoint):
                        connection = self.connections[endpoint]
                        ping_msg = json.dumps({"ping": 1})
                        await connection.send(ping_msg)
                        self.last_ping_time[endpoint] = now
                
                # Check if we've received a pong response within 2*ping_interval
                if now - self.last_pong_time.get(endpoint, 0) > 2 * self.options.ping_interval:
                    self.logger.warning(f"No pong response received from {endpoint}, connection may be stale")
                    
                    # Force reconnect
                    await self.close_connection(endpoint)
                    
                    # Exit this ping handler
                    break
                
                # Sleep for a short time before checking again
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in ping handler for {endpoint}: {str(e)}")
                await asyncio.sleep(5)  # Sleep longer on error
        
        self.logger.debug(f"Ping handler for {endpoint} exited")
    
    async def process_message(self, endpoint: str, data: Dict[str, Any]):
        """
        Process an incoming message from the Deriv API.
        
        Args:
            endpoint: The WebSocket endpoint URL
            data: The message payload
        """
        # Check for request ID response
        req_id = data.get("req_id")
        if req_id and req_id in self.request_callbacks:
            future = self.request_callbacks[req_id]
            if not future.done():
                future.set_result(data)
            return
        
        # Handle tick data
        if "tick" in data:
            await self.process_tick(data)
            return
        
        # Handle order book data
        if "orderBook" in data:
            await self.process_order_book(data)
            return
        
        # Handle errors
        if "error" in data:
            self.logger.warning(f"Received error from Deriv API: {data['error'].get('message', 'Unknown error')}")
            self.error_counts[endpoint] += 1
            self.metrics.increment_counter("feed.errors", {"feed": "deriv", "endpoint": endpoint, "type": "api_error"})
            return
        
        # Handle other message types (ignoring informational messages)
        if not any(key in data for key in ["echo_req", "msg_type"]):
            self.logger.debug(f"Received unknown message type: {data}")
    
    async def process_tick(self, data: Dict[str, Any]):
        """
        Process a tick message from the Deriv API.
        
        Args:
            data: The tick message payload
        """
        tick_data = data.get("tick", {})
        symbol = tick_data.get("symbol")
        
        if not symbol:
            self.logger.warning(f"Received tick data without symbol: {tick_data}")
            return
        
        # Find matching instrument
        instrument_id = None
        for inst_id, sub in self.subscriptions.items():
            if sub.get("symbol") == symbol:
                instrument_id = inst_id
                break
        
        if not instrument_id:
            self.logger.debug(f"Received tick for unknown subscription: {symbol}")
            return
        
        # Calculate latency
        receive_time = time.time()
        server_time = tick_data.get("epoch", receive_time)
        latency = receive_time - server_time
        
        # Store in latency tracking
        self.tick_latencies[symbol].append(latency)
        self.metrics.record_histogram("feed.tick_latency", latency, {"feed": "deriv", "symbol": symbol})
        
        # Normalize and enhance the tick data
        normalized_tick = self.normalize_tick_data(tick_data)
        
        # Store the tick
        self.subscriptions[instrument_id]["last_tick"] = normalized_tick
        self.last_tick_time[instrument_id] = receive_time
        
        # Analyze the tick for platform behavior patterns
        await self.platform_analyzer.analyze_tick(symbol, normalized_tick)
        
        # Process the tick through data processor
        await self.process_data(instrument_id, normalized_tick)
    
    async def process_order_book(self, data: Dict[str, Any]):
        """
        Process an order book message from the Deriv API.
        
        Args:
            data: The order book message payload
        """
        orderbook_data = data.get("orderBook", {})
        symbol = orderbook_data.get("symbol")
        
        if not symbol:
            self.logger.warning(f"Received order book data without symbol: {orderbook_data}")
            return
        
        # Find matching instrument
        instrument_id = None
        for inst_id, sub in self.subscriptions.items():
            if sub.get("symbol") == symbol:
                instrument_id = inst_id
                break
        
        if not instrument_id:
            self.logger.debug(f"Received order book for unknown subscription: {symbol}")
            return
        
        # Store order book
        self.subscriptions[instrument_id]["order_book"] = {
            "timestamp": time.time() * 1000,
            "bids": orderbook_data.get("bids", []),
            "asks": orderbook_data.get("asks", [])
        }
        
        # Analyze order book for liquidity profiles
        await self.platform_analyzer.analyze_order_book(symbol, orderbook_data)
    
    async def fetch_active_symbols(self):
        """Fetch the list of active symbols from the Deriv API."""
        self.logger.info("Fetching active symbols from Deriv API")
        
        endpoint = self.options.endpoints["websocket"]
        
        try:
            # Connect if needed
            if not await self.ensure_connection(endpoint):
                raise FeedConnectionError(f"Failed to connect to Deriv API: {endpoint}")
            
            # Request active symbols
            symbols_req = {
                "active_symbols": "brief",
                "product_type": "basic"
            }
            
            req_id = await self.send_request(endpoint, symbols_req)
            response = await self.wait_for_response(req_id, timeout=30)
            
            if not response or "error" in response:
                error_msg = response.get("error", {}).get("message", "Unknown error") if response else "Timeout"
                raise FeedDataError(f"Failed to fetch active symbols: {error_msg}")
            
            # Process symbols
            symbols = response.get("active_symbols", [])
            self.logger.info(f"Received {len(symbols)} active symbols from Deriv API")
            
            # Store symbols
            self.active_symbols = {}
            for symbol_data in symbols:
                symbol = symbol_data.get("symbol")
                if symbol:
                    self.active_symbols[symbol] = symbol_data
                    
                    # Create instrument entry
                    instrument_id = normalize_instrument_id("deriv", symbol)
                    self.instruments[instrument_id] = {
                        "instrument_id": instrument_id,
                        "symbol": symbol,
                        "exchange": "deriv",
                        "name": symbol_data.get("display_name", symbol),
                        "type": symbol_data.get("market", ""),
                        "currency": symbol_data.get("currency", "USD"),
                        "is_active": symbol_data.get("exchange_is_open", True),
                        "pip_size": 10 ** (-1 * symbol_data.get("pip", 0)),
                        "market": symbol_data.get("market_display_name", ""),
                        "submarket": symbol_data.get("submarket_display_name", ""),
                        "exchange_name": symbol_data.get("exchange_name", "Deriv")
                    }
            
            self.logger.info(f"Processed {len(self.instruments)} instruments")
            
        except Exception as e:
            self.logger.error(f"Error fetching active symbols: {str(e)}")
            raise
    
    async def reconnect_subscriptions(self, instrument_ids: List[str]):
        """
        Attempt to reconnect and resubscribe to the specified instruments.
        
        Args:
            instrument_ids: List of instrument IDs to reconnect
        """
        for instrument_id in instrument_ids:
            # Unsubscribe first to clean up state
            subscription = self.subscriptions.get(instrument_id)
            if subscription:
                # Just remove from our tracking - don't try to send unsubscribe messages
                # since the connection is already closed
                del self.subscriptions[instrument_id]
            
            # Wait a bit before reconnecting to avoid hammering the server
            await asyncio.sleep(1 + random.random() * 2)
            
            # Try to resubscribe
            await self.subscribe(instrument_id)
    
    async def monitor_connections(self):
        """Monitor connection health and reconnect as needed."""
        self.logger.debug("Starting connection monitor")
        
        while True:
            try:
                # Check each connection
                for endpoint, status in list(self.connection_status.items()):
                    if not status and endpoint in self.connections:
                        # Connection is marked as down but still in our connections dict
                        # This could happen if the ping handler hasn't cleaned up yet
                        await self.close_connection(endpoint)
                    
                    # Find subscriptions for this endpoint that need reconnecting
                    if not status:
                        active_subs = [
                            inst_id for inst_id, sub in self.subscriptions.items() 
                            if sub.get("endpoint") == endpoint
                        ]
                        
                        if active_subs:
                            self.logger.info(f"Attempting to reconnect for {len(active_subs)} subscriptions")
                            await self.reconnect_subscriptions(active_subs)
                
                # Check for stale subscriptions
                now = time.time()
                for instrument_id, subscription in list(self.subscriptions.items()):
                    last_tick_time = self.last_tick_time.get(instrument_id, 0)
                    
                    # If no tick received for 2 minutes, try to resubscribe
                    if now - last_tick_time > 120:
                        self.logger.warning(f"No data received for {instrument_id} in 2 minutes, resubscribing")
                        
                        # Unsubscribe and resubscribe
                        await self.unsubscribe(instrument_id)
                        await asyncio.sleep(1)  # Brief pause
                        await self.subscribe(instrument_id)
                
                # Sleep for a while before checking again
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Error in connection monitor: {str(e)}")
                await asyncio.sleep(60)  # Sleep longer on error
    
    async def metrics_reporter(self):
        """Periodically report metrics and state information."""
        self.logger.debug("Starting metrics reporter")
        
        while True:
            try:
                # Report connection status
                for endpoint, status in self.connection_status.items():
                    self.metrics.set_gauge(
                        "feed.connection_status", 
                        1 if status else 0, 
                        {"feed": "deriv", "endpoint": endpoint}
                    )
                
                # Report subscription count
                self.metrics.set_gauge(
                    "feed.subscriptions", 
                    len(self.subscriptions), 
                    {"feed": "deriv"}
                )
                
                # Report message and error counts
                for endpoint, count in self.message_counts.items():
                    self.metrics.set_counter(
                        "feed.messages_received",
                        count,
                        {"feed": "deriv", "endpoint": endpoint}
                    )
                
                for endpoint, count in self.error_counts.items():
                    self.metrics.set_counter(
                        "feed.errors",
                        count,
                        {"feed": "deriv", "endpoint": endpoint}
                    )
                
                # Report latency statistics
                for symbol, latencies in self.tick_latencies.items():
                    if latencies:
                        avg_latency = sum(latencies) / len(latencies)
                        self.metrics.record_value(
                            "feed.avg_latency",
                            avg_latency,
                            {"feed": "deriv", "symbol": symbol}
                        )
                
                # Log summary
                self.logger.info(
                    f"Deriv feed status: "
                    f"{sum(1 for s in self.connection_status.values() if s)}/{len(self.connection_status)} connections, "
                    f"{len(self.subscriptions)} subscriptions"
                )
                
                # Sleep for a while before reporting again
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in metrics reporter: {str(e)}")
                await asyncio.sleep(120)  # Sleep longer on error
    
    # Helper methods
    def normalize_tick_data(self, tick_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and enhance tick data with additional information.
        
        Args:
            tick_data: The raw tick data from the API
            
        Returns:
            Enhanced tick data with normalized fields
        """
        normalized = tick_data.copy()
        
        # Ensure timestamp is present (convert from epoch if needed)
        if "epoch" in normalized and "timestamp" not in normalized:
            normalized["timestamp"] = normalized["epoch"] * 1000  # Convert to milliseconds
        
        # Add bid/ask if only quote is available
        if "quote" in normalized and "bid" not in normalized:
            normalized["bid"] = normalized["quote"]
            normalized["ask"] = normalized["quote"]
        
        # Ensure symbol is lowercase for consistency
        if "symbol" in normalized:
            symbol = normalized["symbol"]
            
            # Add market metadata if available
            if symbol in self.active_symbols:
                symbol_data = self.active_symbols[symbol]
                normalized["market"] = symbol_data.get("market", "")
                normalized["submarket"] = symbol_data.get("submarket", "")
                normalized["is_synthetic"] = symbol_data.get("market", "") == "synthetic_index"
        
        # Add receive timestamp
        normalized["receive_time"] = time.time()
        
        return normalized
    
    def get_symbol_for_instrument(self, instrument_id: str) -> Optional[str]:
        """
        Get the platform-specific symbol for a normalized instrument ID.
        
        Args:
            instrument_id: The normalized instrument identifier
            
        Returns:
            The platform-specific symbol or None if not found
        """
        if instrument_id in self.instruments:
            return self.instruments[instrument_id]["symbol"]
        
        # Handle case where instrument_id is actually a symbol
        for symbol in self.active_symbols:
            if normalize_instrument_id("deriv", symbol) == instrument_id:
                return symbol
        
        return None
    
    def create_task(self, coro, name=None):
        """
        Create and track an asyncio task.
        
        Args:
            coro: The coroutine to run as a task
            name: Optional name for the task
        
        Returns:
            The created asyncio task
        """
        if name is None:
            name = create_task_name("deriv_feed")
            
        task = asyncio.create_task(coro, name=name)
        self.tasks.append(task)
        
        # Set up task completion callback to remove from our list
        task.add_done_callback(lambda t: self.tasks.remove(t) if t in self.tasks else None)
        
        return task


# Import for better type hinting
import random
