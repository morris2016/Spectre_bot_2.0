#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Arbitrage Loophole Detection Module

This module implements advanced arbitrage opportunities detection across various markets,
assets, and even within the same platform. It is designed to identify and exploit price
discrepancies with consistently high success rates.

Key capabilities:
- Cross-exchange arbitrage detection (Binance-Deriv and others)
- Triangular arbitrage within the same exchange
- Statistical arbitrage for correlated assets
- Latency arbitrage exploiting timing differences
- Funding rate arbitrage in perpetual futures
- Liquidity-based arbitrage opportunities
- Fee optimization for maximum net profit
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import asyncio
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import networkx as nx

from common.utils import calculate_profit_after_fees, execution_time_ms
from common.constants import (
    MIN_PROFITABLE_SPREAD_PERCENT, 
    MAX_EXECUTION_TIME_MS,
    ARBITRAGE_OPPORTUNITY_TYPES,
    MIN_LIQUIDITY_REQUIREMENTS,
    EXCHANGE_FEE_STRUCTURES,
    DEFAULT_CONFIDENCE_THRESHOLD
)
from common.exceptions import InsufficientDataError, ArbitrageValidationError
from common.metrics import MetricsCollector
from common.db_client import DBClient
from common.redis_client import RedisClient
from data_feeds.coordinator import FeedCoordinator
from feature_service.feature_extraction import FeatureExtractor

# Get logger
logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    """Represents a detected arbitrage opportunity with all relevant details."""
    
    # Basic information
    type: str  # Type of arbitrage (cross-exchange, triangular, etc.)
    timestamp: datetime
    
    # Assets/exchanges involved
    exchanges: List[str]
    symbols: List[str]
    path: List[Tuple[str, str, str]]  # (from_asset, to_asset, exchange)
    
    # Financial data
    raw_spread: float  # Raw price difference percentage
    net_profit_percent: float  # Expected profit after fees
    estimated_profit_usd: float
    required_capital: float
    
    # Execution details
    execution_time_estimate_ms: float
    price_volatility: float  # Estimated price volatility during execution
    slippage_estimate: float
    liquidity_confidence: float  # 0-1 scale for liquidity confidence
    
    # Opportunity quality assessment
    success_probability: float  # 0-1 scale
    risk_reward_ratio: float
    opportunity_score: float  # Overall score incorporating all factors
    
    # Additional metadata
    execution_steps: List[Dict[str, Any]]
    market_conditions: Dict[str, Any]
    historical_success_rate: float  # Success rate for similar opportunities
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert opportunity to dictionary format."""
        return {
            "type": self.type,
            "timestamp": self.timestamp.isoformat(),
            "exchanges": self.exchanges,
            "symbols": self.symbols,
            "path": self.path,
            "raw_spread": self.raw_spread,
            "net_profit_percent": self.net_profit_percent,
            "estimated_profit_usd": self.estimated_profit_usd,
            "required_capital": self.required_capital,
            "execution_time_estimate_ms": self.execution_time_estimate_ms,
            "price_volatility": self.price_volatility,
            "slippage_estimate": self.slippage_estimate,
            "liquidity_confidence": self.liquidity_confidence,
            "success_probability": self.success_probability,
            "risk_reward_ratio": self.risk_reward_ratio,
            "opportunity_score": self.opportunity_score,
            "execution_steps": self.execution_steps,
            "market_conditions": self.market_conditions,
            "historical_success_rate": self.historical_success_rate
        }


class ArbitrageDetector:
    """
    Advanced arbitrage opportunity detection system designed to identify
    profitable trading opportunities across multiple venues with high confidence.
    """
    
    def __init__(
        self, 
        feed_coordinator: FeedCoordinator, 
        db_client: DBClient, 
        redis_client: RedisClient,
        feature_extractor: FeatureExtractor,
        metrics_collector: MetricsCollector,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the arbitrage detector with necessary dependencies.
        
        Args:
            feed_coordinator: Coordinator for accessing market data feeds
            db_client: Database client for historical data and persistence
            redis_client: Redis client for high-speed cache access
            feature_extractor: Feature extraction service for market analysis
            metrics_collector: Metrics collection for performance tracking
            config: Configuration parameters for arbitrage detection
        """
        self.feed_coordinator = feed_coordinator
        self.db_client = db_client
        self.redis_client = redis_client
        self.feature_extractor = feature_extractor
        self.metrics_collector = metrics_collector
        
        # Set up configuration with defaults
        self.config = {
            "min_profitable_spread": MIN_PROFITABLE_SPREAD_PERCENT,
            "max_execution_time_ms": MAX_EXECUTION_TIME_MS,
            "min_success_probability": 0.80,  # Target high success rate
            "min_liquidity_requirements": MIN_LIQUIDITY_REQUIREMENTS,
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "enable_cross_exchange": True,
            "enable_triangular": True,
            "enable_statistical": True,
            "enable_latency": True,
            "enable_funding_rate": True,
            "max_path_length": 3,
            "historical_lookback_days": 30,
            "update_frequency_ms": 100,  # How often to refresh opportunities
            "max_active_opportunities": 10,  # Maximum number of active opportunities to track
            "blockchain_confirmation_requirements": {
                "BTC": 2,
                "ETH": 12,
                "default": 10
            }
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
            
        # Initialize opportunity tracking
        self.active_opportunities: List[ArbitrageOpportunity] = []
        self.opportunity_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Set up graph for triangular arbitrage
        self.market_graph = nx.DiGraph()
        
        # Initialize performance metrics
        self.metrics_collector.register_gauge("arbitrage_opportunities_count", 
                                             "Number of detected arbitrage opportunities")
        self.metrics_collector.register_histogram("arbitrage_profit_percent", 
                                                "Profit percentage of arbitrage opportunities")
        self.metrics_collector.register_gauge("arbitrage_success_rate", 
                                             "Success rate of executed arbitrage opportunities")
        
        # Cache for exchange orderbook and rate data
        self.market_data_cache = {}
        self.last_cache_update = datetime.now()
        self.cache_ttl = timedelta(milliseconds=self.config["update_frequency_ms"])
        
        logger.info(f"ArbitrageDetector initialized with {self.config}")
        
    async def start(self):
        """Start the arbitrage detector services."""
        logger.info("Starting arbitrage detector services")
        # Initialize historical performance data
        await self._load_historical_performance()
        # Start background tasks
        self._start_background_tasks()
        
    async def stop(self):
        """Stop the arbitrage detector services."""
        logger.info("Stopping arbitrage detector services")
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
            
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
    def _start_background_tasks(self):
        """Start background tasks for continuous monitoring."""
        self.background_tasks = [
            asyncio.create_task(self._monitor_cross_exchange_opportunities()),
            asyncio.create_task(self._monitor_triangular_opportunities()),
            asyncio.create_task(self._monitor_statistical_opportunities()),
            asyncio.create_task(self._monitor_latency_opportunities()),
            asyncio.create_task(self._monitor_funding_rate_opportunities()),
            asyncio.create_task(self._update_market_graph()),
            asyncio.create_task(self._track_performance_metrics())
        ]
        
    async def _load_historical_performance(self):
        """Load historical performance data for opportunity evaluation."""
        try:
            history = await self.db_client.get_collection("arbitrage_history").find(
                {"timestamp": {"$gte": datetime.now() - timedelta(days=self.config["historical_lookback_days"])}}
            ).to_list(None)
            
            # Group by opportunity type
            for item in history:
                opportunity_type = item.get("type", "unknown")
                if opportunity_type not in self.opportunity_history:
                    self.opportunity_history[opportunity_type] = []
                self.opportunity_history[opportunity_type].append(item)
                
            # Calculate success rates
            self.type_success_rates = {}
            for op_type, opportunities in self.opportunity_history.items():
                if not opportunities:
                    continue
                    
                successful = sum(1 for op in opportunities if op.get("executed", False) and op.get("successful", False))
                total_executed = sum(1 for op in opportunities if op.get("executed", False))
                
                if total_executed > 0:
                    self.type_success_rates[op_type] = successful / total_executed
                else:
                    self.type_success_rates[op_type] = 0.0
                    
            logger.info(f"Loaded historical performance data: {len(history)} records")
            logger.info(f"Historical success rates by type: {self.type_success_rates}")
            
        except Exception as e:
            logger.error(f"Failed to load historical performance data: {e}")
            # Initialize with empty data
            self.opportunity_history = {}
            self.type_success_rates = {}
    
    @execution_time_ms
    async def find_opportunities(self) -> List[ArbitrageOpportunity]:
        """
        Main method to detect all types of arbitrage opportunities.
        
        Returns:
            List of arbitrage opportunities sorted by opportunity score.
        """
        start_time = time.time()
        all_opportunities = []
        
        # Update cache if needed
        await self._update_market_data_cache()
        
        # Gather opportunities from all enabled detectors
        if self.config["enable_cross_exchange"]:
            cross_exchange_ops = await self.detect_cross_exchange_arbitrage()
            all_opportunities.extend(cross_exchange_ops)
            logger.debug(f"Found {len(cross_exchange_ops)} cross-exchange arbitrage opportunities")
            
        if self.config["enable_triangular"]:
            triangular_ops = await self.detect_triangular_arbitrage()
            all_opportunities.extend(triangular_ops)
            logger.debug(f"Found {len(triangular_ops)} triangular arbitrage opportunities")
            
        if self.config["enable_statistical"]:
            statistical_ops = await self.detect_statistical_arbitrage()
            all_opportunities.extend(statistical_ops)
            logger.debug(f"Found {len(statistical_ops)} statistical arbitrage opportunities")
            
        if self.config["enable_latency"]:
            latency_ops = await self.detect_latency_arbitrage()
            all_opportunities.extend(latency_ops)
            logger.debug(f"Found {len(latency_ops)} latency arbitrage opportunities")
            
        if self.config["enable_funding_rate"]:
            funding_ops = await self.detect_funding_rate_arbitrage()
            all_opportunities.extend(funding_ops)
            logger.debug(f"Found {len(funding_ops)} funding rate arbitrage opportunities")
        
        # Filter for minimum profitability and success probability
        filtered_opportunities = [
            op for op in all_opportunities
            if op.net_profit_percent >= self.config["min_profitable_spread"]
            and op.success_probability >= self.config["min_success_probability"]
            and op.liquidity_confidence >= self.config["confidence_threshold"]
        ]
        
        # Sort by opportunity score (highest first)
        sorted_opportunities = sorted(
            filtered_opportunities, 
            key=lambda x: x.opportunity_score,
            reverse=True
        )
        
        # Update active opportunities
        self.active_opportunities = sorted_opportunities[:self.config["max_active_opportunities"]]
        
        # Track metrics
        self.metrics_collector.set_gauge("arbitrage_opportunities_count", len(filtered_opportunities))
        if filtered_opportunities:
            self.metrics_collector.observe_histogram(
                "arbitrage_profit_percent", 
                [op.net_profit_percent for op in filtered_opportunities]
            )
        
        execution_time = (time.time() - start_time) * 1000
        logger.info(f"Found {len(filtered_opportunities)} profitable arbitrage opportunities "
                   f"in {execution_time:.2f}ms")
        
        return sorted_opportunities
    
    async def _update_market_data_cache(self):
        """Update the market data cache if it's expired."""
        now = datetime.now()
        if (now - self.last_cache_update) > self.cache_ttl:
            logger.debug("Updating market data cache")
            
            # Collect exchange data for relevant markets
            exchanges = ["binance", "deriv"]  # Add more exchanges as needed
            symbols = await self._get_common_symbols(exchanges)
            
            for exchange in exchanges:
                for symbol in symbols:
                    try:
                        # Get orderbook data
                        orderbook = await self.feed_coordinator.get_orderbook(exchange, symbol)
                        
                        # Get ticker data
                        ticker = await self.feed_coordinator.get_ticker(exchange, symbol)
                        
                        # Store in cache
                        cache_key = f"{exchange}:{symbol}"
                        self.market_data_cache[cache_key] = {
                            "orderbook": orderbook,
                            "ticker": ticker,
                            "timestamp": now
                        }
                    except Exception as e:
                        logger.warning(f"Failed to update cache for {exchange}:{symbol}: {e}")
                        
            # Update timestamp
            self.last_cache_update = now
            
    async def _get_common_symbols(self, exchanges: List[str]) -> List[str]:
        """Get common symbols available across specified exchanges."""
        all_symbols = {}
        
        for exchange in exchanges:
            try:
                exchange_symbols = await self.feed_coordinator.get_symbols(exchange)
                all_symbols[exchange] = set(exchange_symbols)
            except Exception as e:
                logger.warning(f"Failed to get symbols for {exchange}: {e}")
                all_symbols[exchange] = set()
                
        # Find common symbols across all exchanges
        if not all_symbols:
            return []
            
        common_symbols = set.intersection(*all_symbols.values())
        return list(common_symbols)
    
    async def detect_cross_exchange_arbitrage(self) -> List[ArbitrageOpportunity]:
        """
        Detect cross-exchange arbitrage opportunities by comparing prices
        of the same asset across different exchanges.
        """
        opportunities = []
        
        try:
            # Get common symbols across exchanges
            exchanges = ["binance", "deriv"]  # Add more exchanges as needed
            symbols = await self._get_common_symbols(exchanges)
            
            for symbol in symbols:
                # Collect bid/ask prices from each exchange
                prices = {}
                order_depths = {}
                
                for exchange in exchanges:
                    cache_key = f"{exchange}:{symbol}"
                    if cache_key not in self.market_data_cache:
                        continue
                        
                    cache_data = self.market_data_cache[cache_key]
                    orderbook = cache_data.get("orderbook", {})
                    
                    if not orderbook or "bids" not in orderbook or "asks" not in orderbook:
                        continue
                        
                    # Best bid and ask prices
                    best_bid = orderbook["bids"][0][0] if orderbook["bids"] else None
                    best_ask = orderbook["asks"][0][0] if orderbook["asks"] else None
                    
                    if best_bid is None or best_ask is None:
                        continue
                        
                    prices[exchange] = {
                        "bid": best_bid,
                        "ask": best_ask,
                        "mid": (best_bid + best_ask) / 2
                    }
                    
                    # Calculate order depth
                    bid_depth = sum(qty for _, qty in orderbook["bids"][:5])
                    ask_depth = sum(qty for _, qty in orderbook["asks"][:5])
                    order_depths[exchange] = {
                        "bid_depth": bid_depth,
                        "ask_depth": ask_depth
                    }
                
                # Check for opportunities between each exchange pair
                for buy_exchange in exchanges:
                    for sell_exchange in exchanges:
                        if buy_exchange == sell_exchange:
                            continue
                            
                        if buy_exchange not in prices or sell_exchange not in prices:
                            continue
                            
                        buy_price = prices[buy_exchange]["ask"]  # Price to buy at
                        sell_price = prices[sell_exchange]["bid"]  # Price to sell at
                        
                        # Calculate raw spread percentage
                        raw_spread = (sell_price - buy_price) / buy_price * 100
                        
                        if raw_spread <= 0:
                            continue  # No profitable opportunity
                            
                        # Estimate trade size based on available liquidity
                        buy_liquidity = order_depths[buy_exchange]["ask_depth"]
                        sell_liquidity = order_depths[sell_exchange]["bid_depth"]
                        max_trade_size = min(buy_liquidity, sell_liquidity)
                        
                        # Check if there's sufficient liquidity
                        min_required = self.config["min_liquidity_requirements"].get(
                            symbol, self.config["min_liquidity_requirements"]["default"]
                        )
                        
                        if max_trade_size < min_required:
                            continue  # Insufficient liquidity
                            
                        # Calculate profit after fees
                        buy_fee = EXCHANGE_FEE_STRUCTURES[buy_exchange]["taker"]
                        sell_fee = EXCHANGE_FEE_STRUCTURES[sell_exchange]["maker"]
                        
                        net_profit = calculate_profit_after_fees(
                            buy_price=buy_price,
                            sell_price=sell_price,
                            quantity=max_trade_size,
                            buy_fee_percent=buy_fee,
                            sell_fee_percent=sell_fee
                        )
                        
                        net_profit_percent = (net_profit / (buy_price * max_trade_size)) * 100
                        
                        if net_profit_percent <= self.config["min_profitable_spread"]:
                            continue  # Not profitable enough after fees
                            
                        # Calculate execution time estimate
                        execution_time_estimate = self._estimate_execution_time(
                            buy_exchange, sell_exchange, symbol
                        )
                        
                        if execution_time_estimate > self.config["max_execution_time_ms"]:
                            continue  # Too slow to execute safely
                            
                        # Estimate volatility risk during execution
                        price_volatility = await self._estimate_price_volatility(symbol, execution_time_estimate)
                        
                        # Estimate slippage
                        slippage_estimate = await self._estimate_slippage(
                            buy_exchange, sell_exchange, symbol, max_trade_size
                        )
                        
                        # Calculate liquidity confidence
                        liquidity_confidence = self._calculate_liquidity_confidence(
                            buy_exchange, sell_exchange, symbol, max_trade_size
                        )
                        
                        # Get market conditions
                        market_conditions = await self._get_market_conditions(symbol)
                        
                        # Calculate success probability
                        success_probability = self._calculate_success_probability(
                            opportunity_type="cross_exchange",
                            raw_spread=raw_spread,
                            net_profit_percent=net_profit_percent,
                            execution_time_ms=execution_time_estimate,
                            price_volatility=price_volatility,
                            slippage_estimate=slippage_estimate,
                            liquidity_confidence=liquidity_confidence,
                            market_conditions=market_conditions
                        )
                        
                        # Calculate risk/reward ratio
                        risk_reward_ratio = net_profit_percent / (price_volatility + slippage_estimate)
                        
                        # Calculate overall opportunity score
                        opportunity_score = self._calculate_opportunity_score(
                            net_profit_percent=net_profit_percent,
                            success_probability=success_probability,
                            liquidity_confidence=liquidity_confidence,
                            execution_time_ms=execution_time_estimate,
                            risk_reward_ratio=risk_reward_ratio
                        )
                        
                        # Get historical success rate for similar opportunities
                        historical_success_rate = self.type_success_rates.get("cross_exchange", 0.0)
                        
                        # Create opportunity object
                        opportunity = ArbitrageOpportunity(
                            type="cross_exchange",
                            timestamp=datetime.now(),
                            exchanges=[buy_exchange, sell_exchange],
                            symbols=[symbol],
                            path=[(symbol, symbol, buy_exchange), (symbol, symbol, sell_exchange)],
                            raw_spread=raw_spread,
                            net_profit_percent=net_profit_percent,
                            estimated_profit_usd=net_profit,
                            required_capital=buy_price * max_trade_size,
                            execution_time_estimate_ms=execution_time_estimate,
                            price_volatility=price_volatility,
                            slippage_estimate=slippage_estimate,
                            liquidity_confidence=liquidity_confidence,
                            success_probability=success_probability,
                            risk_reward_ratio=risk_reward_ratio,
                            opportunity_score=opportunity_score,
                            execution_steps=[
                                {
                                    "action": "buy",
                                    "exchange": buy_exchange,
                                    "symbol": symbol,
                                    "price": buy_price,
                                    "quantity": max_trade_size,
                                    "fee_percent": buy_fee
                                },
                                {
                                    "action": "sell",
                                    "exchange": sell_exchange,
                                    "symbol": symbol,
                                    "price": sell_price,
                                    "quantity": max_trade_size,
                                    "fee_percent": sell_fee
                                }
                            ],
                            market_conditions=market_conditions,
                            historical_success_rate=historical_success_rate
                        )
                        
                        opportunities.append(opportunity)
                        
            logger.debug(f"Detected {len(opportunities)} cross-exchange arbitrage opportunities")
            return opportunities
                        
        except Exception as e:
            logger.error(f"Error in cross-exchange arbitrage detection: {e}", exc_info=True)
            return []
        
    async def detect_triangular_arbitrage(self) -> List[ArbitrageOpportunity]:
        """
        Detect triangular arbitrage opportunities within a single exchange
        by finding profitable cycles between different trading pairs.
        """
        opportunities = []
        
        try:
            # Focus on one exchange at a time
            for exchange in ["binance", "deriv"]:
                # Update market graph for the exchange
                await self._update_exchange_market_graph(exchange)
                
                # Find all cycles in the graph up to max path length
                cycles = []
                for path_length in range(3, self.config["max_path_length"] + 1):
                    # Find simple cycles of specified length
                    simple_cycles = list(nx.simple_cycles(self.market_graph, path_length))
                    cycles.extend(simple_cycles)
                
                # Evaluate each cycle for arbitrage opportunity
                for cycle in cycles:
                    if len(cycle) < 3:
                        continue  # Need at least 3 nodes for triangular arbitrage
                    
                    # Add the first node again to close the cycle for calculation
                    full_cycle = cycle + [cycle[0]]
                    
                    # Calculate the product of all conversion rates
                    cumulative_rate = 1.0
                    cycle_valid = True
                    conversion_steps = []
                    path_edges = []
                    total_fees = 0.0
                    symbols_involved = []
                    
                    for i in range(len(full_cycle) - 1):
                        from_asset = full_cycle[i]
                        to_asset = full_cycle[i + 1]
                        
                        # Check if edge exists in graph
                        if not self.market_graph.has_edge(from_asset, to_asset):
                            cycle_valid = False
                            break
                            
                        # Get edge data
                        edge_data = self.market_graph.get_edge_data(from_asset, to_asset)
                        conversion_rate = edge_data.get("rate", 0)
                        fee_percent = edge_data.get("fee", 0)
                        market_symbol = edge_data.get("symbol", "")
                        
                        if conversion_rate <= 0:
                            cycle_valid = False
                            break
                            
                        # Apply fee to rate
                        effective_rate = conversion_rate * (1 - fee_percent / 100)
                        
                        # Update cumulative rate
                        cumulative_rate *= effective_rate
                        
                        # Track conversion steps
                        conversion_steps.append({
                            "from_asset": from_asset,
                            "to_asset": to_asset,
                            "rate": conversion_rate,
                            "fee_percent": fee_percent,
                            "effective_rate": effective_rate,
                            "symbol": market_symbol
                        })
                        
                        # Track path
                        path_edges.append((from_asset, to_asset, exchange))
                        
                        # Track total fees
                        total_fees += fee_percent
                        
                        # Track symbols
                        if market_symbol and market_symbol not in symbols_involved:
                            symbols_involved.append(market_symbol)
                    
                    if not cycle_valid:
                        continue
                    
                    # Calculate profit percentage (subtract 1 and convert to percentage)
                    raw_spread = (cumulative_rate - 1) * 100
                    
                    if raw_spread <= self.config["min_profitable_spread"]:
                        continue  # Not profitable enough
                    
                    # Evaluate execution metrics
                    execution_time_estimate = self._estimate_triangular_execution_time(
                        exchange, len(conversion_steps)
                    )
                    
                    if execution_time_estimate > self.config["max_execution_time_ms"]:
                        continue  # Too slow to execute safely
                    
                    # Estimate minimum liquidity across all steps
                    min_liquidity, liquidity_confidence = await self._evaluate_triangular_liquidity(
                        exchange, conversion_steps
                    )
                    
                    # Check minimum liquidity requirements
                    min_required = self.config["min_liquidity_requirements"]["default"]
                    if min_liquidity < min_required:
                        continue  # Insufficient liquidity
                    
                    # Get market volatility for involved symbols
                    avg_volatility = 0
                    for symbol in symbols_involved:
                        symbol_volatility = await self._estimate_price_volatility(
                            symbol, execution_time_estimate
                        )
                        avg_volatility += symbol_volatility
                    
                    if symbols_involved:
                        avg_volatility /= len(symbols_involved)
                    
                    # Estimate slippage
                    slippage_estimate = await self._estimate_triangular_slippage(
                        exchange, conversion_steps, min_liquidity
                    )
                    
                    # Get market conditions for primary symbol
                    primary_symbol = symbols_involved[0] if symbols_involved else ""
                    market_conditions = await self._get_market_conditions(primary_symbol)
                    
                    # Calculate net profit after slippage
                    net_profit_percent = raw_spread - slippage_estimate
                    
                    if net_profit_percent <= 0:
                        continue  # Not profitable after slippage
                    
                    # Calculate required capital in USD
                    required_capital = self._estimate_required_capital(exchange, full_cycle[0], min_liquidity)
                    
                    # Estimate profit in USD
                    estimated_profit_usd = required_capital * (net_profit_percent / 100)
                    
                    # Calculate success probability
                    success_probability = self._calculate_success_probability(
                        opportunity_type="triangular",
                        raw_spread=raw_spread,
                        net_profit_percent=net_profit_percent,
                        execution_time_ms=execution_time_estimate,
                        price_volatility=avg_volatility,
                        slippage_estimate=slippage_estimate,
                        liquidity_confidence=liquidity_confidence,
                        market_conditions=market_conditions
                    )
                    
                    # Calculate risk/reward ratio
                    risk_reward_ratio = net_profit_percent / (avg_volatility + slippage_estimate)
                    
                    # Calculate overall opportunity score
                    opportunity_score = self._calculate_opportunity_score(
                        net_profit_percent=net_profit_percent,
                        success_probability=success_probability,
                        liquidity_confidence=liquidity_confidence,
                        execution_time_ms=execution_time_estimate,
                        risk_reward_ratio=risk_reward_ratio
                    )
                    
                    # Get historical success rate for similar opportunities
                    historical_success_rate = self.type_success_rates.get("triangular", 0.0)
                    
                    # Create execution steps
                    execution_steps = []
                    for step in conversion_steps:
                        action = {
                            "action": "convert",
                            "exchange": exchange,
                            "from_asset": step["from_asset"],
                            "to_asset": step["to_asset"],
                            "rate": step["rate"],
                            "fee_percent": step["fee_percent"],
                            "symbol": step["symbol"]
                        }
                        execution_steps.append(action)
                    
                    # Create opportunity object
                    opportunity = ArbitrageOpportunity(
                        type="triangular",
                        timestamp=datetime.now(),
                        exchanges=[exchange],
                        symbols=symbols_involved,
                        path=path_edges,
                        raw_spread=raw_spread,
                        net_profit_percent=net_profit_percent,
                        estimated_profit_usd=estimated_profit_usd,
                        required_capital=required_capital,
                        execution_time_estimate_ms=execution_time_estimate,
                        price_volatility=avg_volatility,
                        slippage_estimate=slippage_estimate,
                        liquidity_confidence=liquidity_confidence,
                        success_probability=success_probability,
                        risk_reward_ratio=risk_reward_ratio,
                        opportunity_score=opportunity_score,
                        execution_steps=execution_steps,
                        market_conditions=market_conditions,
                        historical_success_rate=historical_success_rate
                    )
                    
                    opportunities.append(opportunity)
            
            logger.debug(f"Detected {len(opportunities)} triangular arbitrage opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Error in triangular arbitrage detection: {e}", exc_info=True)
            return []
    
    async def detect_statistical_arbitrage(self) -> List[ArbitrageOpportunity]:
        """
        Detect statistical arbitrage opportunities between correlated assets
        that have temporarily diverged from their typical relationship.
        """
        opportunities = []
        
        try:
            # Get pairs of correlated assets from feature service
            correlated_pairs = await self.feature_extractor.get_correlated_pairs(
                min_correlation=0.7,  # High correlation threshold
                lookback_days=30,
                stable_period_days=15
            )
            
            for pair in correlated_pairs:
                base_asset = pair["base_asset"]
                quote_asset = pair["quote_asset"]
                correlation = pair["correlation"]
                z_score = pair["z_score"]  # How many standard deviations from mean relationship
                mean_ratio = pair["mean_ratio"]
                current_ratio = pair["current_ratio"]
                exchange = pair["exchange"]
                
                # Only look for significant deviations (> 2 std dev) that are likely to revert
                if abs(z_score) < 2.0:
                    continue
                
                # Determine trade direction based on z-score
                # If z-score is positive, the ratio is higher than normal, so short base/long quote
                # If z-score is negative, the ratio is lower than normal, so long base/short quote
                direction = "mean_reversion"  # Statistical arbitrage is based on mean reversion
                trade_type = "short_base_long_quote" if z_score > 0 else "long_base_short_quote"
                
                # Get market data for both assets
                base_data = await self._get_asset_market_data(exchange, base_asset)
                quote_data = await self._get_asset_market_data(exchange, quote_asset)
                
                if not base_data or not quote_data:
                    continue
                
                # Calculate raw spread as deviation from mean
                raw_spread = abs(current_ratio - mean_ratio) / mean_ratio * 100
                
                # Check if spread is significant enough
                if raw_spread < self.config["min_profitable_spread"]:
                    continue
                
                # Estimate execution time
                execution_time_estimate = self._estimate_statistical_execution_time(exchange)
                
                if execution_time_estimate > self.config["max_execution_time_ms"]:
                    continue
                
                # Evaluate liquidity for both assets
                base_liquidity = self._get_asset_liquidity(base_data)
                quote_liquidity = self._get_asset_liquidity(quote_data)
                
                # Use the more conservative liquidity estimate
                min_liquidity = min(base_liquidity, quote_liquidity)
                liquidity_confidence = self._calculate_statistical_liquidity_confidence(
                    base_data, quote_data
                )
                
                # Check minimum liquidity
                min_required = self.config["min_liquidity_requirements"].get(
                    "statistical", self.config["min_liquidity_requirements"]["default"]
                )
                
                if min_liquidity < min_required:
                    continue
                
                # Calculate volatility
                base_volatility = await self._estimate_price_volatility(base_asset, execution_time_estimate)
                quote_volatility = await self._estimate_price_volatility(quote_asset, execution_time_estimate)
                avg_volatility = (base_volatility + quote_volatility) / 2
                
                # Estimate slippage
                slippage_estimate = await self._estimate_statistical_slippage(
                    exchange, base_asset, quote_asset, min_liquidity
                )
                
                # Calculate net profit after slippage
                net_profit_percent = raw_spread - slippage_estimate
                
                if net_profit_percent <= 0:
                    continue
                
                # Get combined market conditions
                market_conditions = await self._get_combined_market_conditions([base_asset, quote_asset])
                
                # Estimate required capital
                required_capital = self._estimate_statistical_capital(
                    exchange, base_asset, quote_asset, min_liquidity
                )
                
                # Estimate profit in USD
                estimated_profit_usd = required_capital * (net_profit_percent / 100)
                
                # Calculate success probability based on z-score and other factors
                # Statistical arbitrage has higher success with extreme z-scores
                z_score_factor = min(abs(z_score) / 4, 1.0)  # Normalize z-score influence
                
                success_probability = self._calculate_success_probability(
                    opportunity_type="statistical",
                    raw_spread=raw_spread,
                    net_profit_percent=net_profit_percent,
                    execution_time_ms=execution_time_estimate,
                    price_volatility=avg_volatility,
                    slippage_estimate=slippage_estimate,
                    liquidity_confidence=liquidity_confidence,
                    market_conditions=market_conditions,
                    additional_factors={"z_score_factor": z_score_factor, "correlation": correlation}
                )
                
                # Calculate risk/reward ratio
                risk_reward_ratio = net_profit_percent / (avg_volatility + slippage_estimate)
                
                # Calculate overall opportunity score
                opportunity_score = self._calculate_opportunity_score(
                    net_profit_percent=net_profit_percent,
                    success_probability=success_probability,
                    liquidity_confidence=liquidity_confidence,
                    execution_time_ms=execution_time_estimate,
                    risk_reward_ratio=risk_reward_ratio,
                    additional_factors={"z_score_factor": z_score_factor, "correlation": correlation}
                )
                
                # Get historical success rate for similar opportunities
                historical_success_rate = self.type_success_rates.get("statistical", 0.0)
                
                # Create execution steps based on trade type
                execution_steps = []
                if trade_type == "short_base_long_quote":
                    execution_steps = [
                        {
                            "action": "short",
                            "exchange": exchange,
                            "asset": base_asset,
                            "price": base_data["price"],
                            "quantity": min_liquidity / base_data["price"],
                            "fee_percent": EXCHANGE_FEE_STRUCTURES[exchange]["taker"]
                        },
                        {
                            "action": "long",
                            "exchange": exchange,
                            "asset": quote_asset,
                            "price": quote_data["price"],
                            "quantity": min_liquidity / quote_data["price"],
                            "fee_percent": EXCHANGE_FEE_STRUCTURES[exchange]["taker"]
                        }
                    ]
                else:  # long_base_short_quote
                    execution_steps = [
                        {
                            "action": "long",
                            "exchange": exchange,
                            "asset": base_asset,
                            "price": base_data["price"],
                            "quantity": min_liquidity / base_data["price"],
                            "fee_percent": EXCHANGE_FEE_STRUCTURES[exchange]["taker"]
                        },
                        {
                            "action": "short",
                            "exchange": exchange,
                            "asset": quote_asset,
                            "price": quote_data["price"],
                            "quantity": min_liquidity / quote_data["price"],
                            "fee_percent": EXCHANGE_FEE_STRUCTURES[exchange]["taker"]
                        }
                    ]
                
                # Create opportunity object
                opportunity = ArbitrageOpportunity(
                    type="statistical",
                    timestamp=datetime.now(),
                    exchanges=[exchange],
                    symbols=[base_asset, quote_asset],
                    path=[(base_asset, quote_asset, exchange)],
                    raw_spread=raw_spread,
                    net_profit_percent=net_profit_percent,
                    estimated_profit_usd=estimated_profit_usd,
                    required_capital=required_capital,
                    execution_time_estimate_ms=execution_time_estimate,
                    price_volatility=avg_volatility,
                    slippage_estimate=slippage_estimate,
                    liquidity_confidence=liquidity_confidence,
                    success_probability=success_probability,
                    risk_reward_ratio=risk_reward_ratio,
                    opportunity_score=opportunity_score,
                    execution_steps=execution_steps,
                    market_conditions=market_conditions,
                    historical_success_rate=historical_success_rate
                )
                
                opportunities.append(opportunity)
            
            logger.debug(f"Detected {len(opportunities)} statistical arbitrage opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Error in statistical arbitrage detection: {e}", exc_info=True)
            return []
    
    async def detect_latency_arbitrage(self) -> List[ArbitrageOpportunity]:
        """
        Detect latency arbitrage opportunities by identifying price updates
        that propagate with different speeds across exchanges or data feeds.
        """
        opportunities = []
        
        try:
            # Get common assets across exchanges
            exchanges = ["binance", "deriv"]  # Add more exchanges as needed
            symbols = await self._get_common_symbols(exchanges)
            
            # Get exchange latency metrics
            latency_metrics = await self._get_exchange_latency_metrics(exchanges)
            
            for symbol in symbols:
                # Skip if latency metrics not available for all exchanges
                if not all(exchange in latency_metrics for exchange in exchanges):
                    continue
                
                # Sort exchanges by data propagation latency (ascending)
                sorted_exchanges = sorted(
                    exchanges,
                    key=lambda x: latency_metrics.get(x, {}).get("propagation_latency_ms", float("inf"))
                )
                
                # We need at least two exchanges to compare
                if len(sorted_exchanges) < 2:
                    continue
                
                # Get latest price data for this symbol across exchanges
                prices = {}
                timestamps = {}
                
                for exchange in sorted_exchanges:
                    latest_data = await self.feed_coordinator.get_ticker(exchange, symbol)
                    
                    if not latest_data:
                        continue
                    
                    prices[exchange] = latest_data.get("price", 0)
                    timestamps[exchange] = latest_data.get("timestamp", 0)
                
                # Need at least two exchanges with price data
                if len(prices) < 2:
                    continue
                
                # Find the exchange with the lowest latency (fastest updates)
                fastest_exchange = sorted_exchanges[0]
                
                # Check opportunity against all slower exchanges
                for slower_exchange in sorted_exchanges[1:]:
                    if slower_exchange not in prices or fastest_exchange not in prices:
                        continue
                    
                    # Get price difference
                    fast_price = prices[fastest_exchange]
                    slow_price = prices[slower_exchange]
                    
                    # Calculate time difference in milliseconds
                    try:
                        fast_ts = timestamps[fastest_exchange]
                        slow_ts = timestamps[slower_exchange]
                        
                        # Convert to milliseconds if needed
                        if isinstance(fast_ts, datetime):
                            fast_ts = fast_ts.timestamp() * 1000
                        if isinstance(slow_ts, datetime):
                            slow_ts = slow_ts.timestamp() * 1000
                            
                        time_diff_ms = abs(slow_ts - fast_ts)
                    except (TypeError, ValueError):
                        # Skip if can't calculate time difference
                        continue
                    
                    # Skip if time difference is too small (likely no latency advantage)
                    min_latency_diff_ms = 50  # Minimum latency difference to exploit
                    if time_diff_ms < min_latency_diff_ms:
                        continue
                    
                    # Calculate price difference percentage
                    price_diff_pct = abs(fast_price - slow_price) / fast_price * 100
                    
                    # Skip if price difference is too small
                    if price_diff_pct < self.config["min_profitable_spread"]:
                        continue
                    
                    # Determine trade direction
                    buy_exchange = fastest_exchange if fast_price < slow_price else slower_exchange
                    sell_exchange = slower_exchange if fast_price < slow_price else fastest_exchange
                    
                    buy_price = prices[buy_exchange]
                    sell_price = prices[sell_exchange]
                    
                    # Calculate raw spread
                    raw_spread = (sell_price - buy_price) / buy_price * 100
                    
                    if raw_spread <= 0:
                        continue
                    
                    # Estimate execution time
                    execution_time_estimate = self._estimate_latency_execution_time(
                        buy_exchange, sell_exchange, time_diff_ms
                    )
                    
                    # Skip if execution time is too long
                    if execution_time_estimate > self.config["max_execution_time_ms"]:
                        continue
                    
                    # Get liquidity data for both exchanges
                    buy_liquidity = await self._get_exchange_liquidity(buy_exchange, symbol, "buy")
                    sell_liquidity = await self._get_exchange_liquidity(sell_exchange, symbol, "sell")
                    
                    # Use the more conservative estimate
                    min_liquidity = min(buy_liquidity, sell_liquidity)
                    
                    # Calculate liquidity confidence
                    liquidity_confidence = self._calculate_liquidity_confidence(
                        buy_exchange, sell_exchange, symbol, min_liquidity
                    )
                    
                    # Check minimum liquidity requirements
                    min_required = self.config["min_liquidity_requirements"].get(
                        symbol, self.config["min_liquidity_requirements"]["default"]
                    )
                    
                    if min_liquidity < min_required:
                        continue
                    
                    # Estimate price volatility
                    price_volatility = await self._estimate_price_volatility(symbol, execution_time_estimate)
                    
                    # Estimate slippage
                    slippage_estimate = await self._estimate_slippage(
                        buy_exchange, sell_exchange, symbol, min_liquidity
                    )
                    
                    # Calculate net profit after fees and slippage
                    buy_fee = EXCHANGE_FEE_STRUCTURES[buy_exchange]["taker"]
                    sell_fee = EXCHANGE_FEE_STRUCTURES[sell_exchange]["taker"]  # Note: Using taker fee for both
                    
                    total_fee_pct = buy_fee + sell_fee
                    net_profit_percent = raw_spread - total_fee_pct - slippage_estimate
                    
                    if net_profit_percent <= 0:
                        continue
                    
                    # Get market conditions
                    market_conditions = await self._get_market_conditions(symbol)
                    
                    # Calculate required capital
                    required_capital = buy_price * min_liquidity
                    
                    # Estimate profit in USD
                    estimated_profit_usd = required_capital * (net_profit_percent / 100)
                    
                    # Calculate success probability
                    # For latency arbitrage, time difference is a key factor
                    time_diff_factor = min(time_diff_ms / 1000, 1.0)  # Normalize time difference
                    
                    success_probability = self._calculate_success_probability(
                        opportunity_type="latency",
                        raw_spread=raw_spread,
                        net_profit_percent=net_profit_percent,
                        execution_time_ms=execution_time_estimate,
                        price_volatility=price_volatility,
                        slippage_estimate=slippage_estimate,
                        liquidity_confidence=liquidity_confidence,
                        market_conditions=market_conditions,
                        additional_factors={"time_diff_factor": time_diff_factor}
                    )
                    
                    # Calculate risk/reward ratio
                    risk_reward_ratio = net_profit_percent / (price_volatility + slippage_estimate)
                    
                    # Calculate overall opportunity score
                    opportunity_score = self._calculate_opportunity_score(
                        net_profit_percent=net_profit_percent,
                        success_probability=success_probability,
                        liquidity_confidence=liquidity_confidence,
                        execution_time_ms=execution_time_estimate,
                        risk_reward_ratio=risk_reward_ratio,
                        additional_factors={"time_diff_factor": time_diff_factor}
                    )
                    
                    # Get historical success rate for similar opportunities
                    historical_success_rate = self.type_success_rates.get("latency", 0.0)
                    
                    # Create execution steps
                    execution_steps = [
                        {
                            "action": "buy",
                            "exchange": buy_exchange,
                            "symbol": symbol,
                            "price": buy_price,
                            "quantity": min_liquidity,
                            "fee_percent": buy_fee
                        },
                        {
                            "action": "sell",
                            "exchange": sell_exchange,
                            "symbol": symbol,
                            "price": sell_price,
                            "quantity": min_liquidity,
                            "fee_percent": sell_fee
                        }
                    ]
                    
                    # Create opportunity object
                    opportunity = ArbitrageOpportunity(
                        type="latency",
                        timestamp=datetime.now(),
                        exchanges=[buy_exchange, sell_exchange],
                        symbols=[symbol],
                        path=[(symbol, symbol, buy_exchange), (symbol, symbol, sell_exchange)],
                        raw_spread=raw_spread,
                        net_profit_percent=net_profit_percent,
                        estimated_profit_usd=estimated_profit_usd,
                        required_capital=required_capital,
                        execution_time_estimate_ms=execution_time_estimate,
                        price_volatility=price_volatility,
                        slippage_estimate=slippage_estimate,
                        liquidity_confidence=liquidity_confidence,
                        success_probability=success_probability,
                        risk_reward_ratio=risk_reward_ratio,
                        opportunity_score=opportunity_score,
                        execution_steps=execution_steps,
                        market_conditions=market_conditions,
                        historical_success_rate=historical_success_rate
                    )
                    
                    opportunities.append(opportunity)
            
            logger.debug(f"Detected {len(opportunities)} latency arbitrage opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Error in latency arbitrage detection: {e}", exc_info=True)
            return []
    
    async def detect_funding_rate_arbitrage(self) -> List[ArbitrageOpportunity]:
        """
        Detect funding rate arbitrage opportunities in perpetual futures markets
        by exploiting significant differences between funding rates and spot markets.
        """
        opportunities = []
        
        try:
            # Currently only implemented for Binance as Deriv doesn't offer the same
            # type of perpetual futures with funding rates
            exchange = "binance"
            
            # Get all perpetual futures with funding rates
            perpetuals = await self.feed_coordinator.get_funding_rates(exchange)
            
            for perpetual in perpetuals:
                symbol = perpetual.get("symbol", "")
                funding_rate = perpetual.get("funding_rate", 0)  # As decimal (e.g., 0.001 = 0.1%)
                next_funding_time = perpetual.get("next_funding_time", None)
                
                # Skip if missing essential data
                if not symbol or funding_rate == 0 or not next_funding_time:
                    continue
                
                # Convert funding rate to annualized percentage
                # Typical funding occurs every 8 hours, so multiply by 3 for daily and 365 for annual
                annual_rate_pct = funding_rate * 3 * 365 * 100
                
                # Skip if funding rate is too small
                min_annual_rate = 5.0  # Minimum 5% annualized rate to be interesting
                if abs(annual_rate_pct) < min_annual_rate:
                    continue
                
                # Get spot symbol equivalent
                spot_symbol = symbol.replace("PERP", "").replace("_PERP", "")
                
                # Get prices for perpetual and spot
                perp_price = await self.feed_coordinator.get_price(exchange, symbol)
                spot_price = await self.feed_coordinator.get_price(exchange, spot_symbol)
                
                if not perp_price or not spot_price:
                    continue
                
                # Calculate time until next funding in hours
                now = datetime.now()
                if isinstance(next_funding_time, str):
                    try:
                        next_funding_time = datetime.fromisoformat(next_funding_time.replace("Z", "+00:00"))
                    except ValueError:
                        # Try timestamp in milliseconds
                        try:
                            next_funding_time = datetime.fromtimestamp(int(next_funding_time) / 1000)
                        except (ValueError, TypeError):
                            continue
                            
                time_to_funding_hours = (next_funding_time - now).total_seconds() / 3600
                
                # Skip if next funding is too far away
                if time_to_funding_hours > 8:
                    continue
                
                # Determine arbitrage strategy based on funding rate sign
                if funding_rate > 0:
                    # Positive funding rate: short perpetual, long spot
                    strategy = "short_perp_long_spot"
                    # Longs pay shorts, so we'll receive funding
                    funding_profit = abs(annual_rate_pct) * (time_to_funding_hours / (24 * 365))
                else:
                    # Negative funding rate: long perpetual, short spot
                    strategy = "long_perp_short_spot"
                    # Shorts pay longs, so we'll receive funding
                    funding_profit = abs(annual_rate_pct) * (time_to_funding_hours / (24 * 365))
                
                # Calculate price difference percentage
                price_diff_pct = abs(perp_price - spot_price) / spot_price * 100
                
                # Calculate raw spread (funding profit - price difference)
                # If price difference works against us, subtract it from funding profit
                raw_spread = funding_profit - price_diff_pct
                
                # Skip if not profitable
                if raw_spread <= 0:
                    continue
                
                # Estimate execution time
                execution_time_estimate = self._estimate_funding_execution_time(exchange)
                
                # Get liquidity data for both markets
                perp_liquidity = await self._get_exchange_liquidity(exchange, symbol, "both")
                spot_liquidity = await self._get_exchange_liquidity(exchange, spot_symbol, "both")
                
                # Use the more conservative estimate
                min_liquidity = min(perp_liquidity, spot_liquidity)
                
                # Calculate liquidity confidence
                liquidity_confidence = min(perp_liquidity, spot_liquidity) / max(perp_liquidity, spot_liquidity)
                
                # Check minimum liquidity requirements
                min_required = self.config["min_liquidity_requirements"].get(
                    "funding", self.config["min_liquidity_requirements"]["default"]
                )
                
                if min_liquidity < min_required:
                    continue
                
                # Estimate price volatility for both markets
                perp_volatility = await self._estimate_price_volatility(symbol, execution_time_estimate)
                spot_volatility = await self._estimate_price_volatility(spot_symbol, execution_time_estimate)
                avg_volatility = (perp_volatility + spot_volatility) / 2
                
                # Estimate slippage
                slippage_estimate = await self._estimate_funding_slippage(
                    exchange, symbol, spot_symbol, min_liquidity
                )
                
                # Calculate net profit after fees and slippage
                fee_pct = EXCHANGE_FEE_STRUCTURES[exchange]["taker"] * 2  # Need to trade both perp and spot
                net_profit_percent = raw_spread - fee_pct - slippage_estimate
                
                if net_profit_percent <= 0:
                    continue
                
                # Get market conditions
                market_conditions = await self._get_combined_market_conditions([symbol, spot_symbol])
                
                # Calculate required capital
                position_size = min_liquidity * spot_price  # In base currency value
                required_capital = position_size * 2  # Need capital for both positions
                
                # Estimate profit in USD
                estimated_profit_usd = position_size * (net_profit_percent / 100)
                
                # Calculate success probability
                funding_certainty = 0.95  # Funding is near-certain but can change in extreme conditions
                time_to_funding_factor = 1 - (time_to_funding_hours / 8)  # Closer to funding is better
                
                success_probability = self._calculate_success_probability(
                    opportunity_type="funding",
                    raw_spread=raw_spread,
                    net_profit_percent=net_profit_percent,
                    execution_time_ms=execution_time_estimate,
                    price_volatility=avg_volatility,
                    slippage_estimate=slippage_estimate,
                    liquidity_confidence=liquidity_confidence,
                    market_conditions=market_conditions,
                    additional_factors={
                        "funding_certainty": funding_certainty,
                        "time_to_funding_factor": time_to_funding_factor
                    }
                )
                
                # Calculate risk/reward ratio
                risk_reward_ratio = net_profit_percent / (avg_volatility + slippage_estimate)
                
                # Calculate overall opportunity score
                opportunity_score = self._calculate_opportunity_score(
                    net_profit_percent=net_profit_percent,
                    success_probability=success_probability,
                    liquidity_confidence=liquidity_confidence,
                    execution_time_ms=execution_time_estimate,
                    risk_reward_ratio=risk_reward_ratio,
                    additional_factors={
                        "funding_certainty": funding_certainty,
                        "time_to_funding_factor": time_to_funding_factor
                    }
                )
                
                # Get historical success rate for similar opportunities
                historical_success_rate = self.type_success_rates.get("funding", 0.0)
                
                # Create execution steps
                execution_steps = []
                if strategy == "short_perp_long_spot":
                    execution_steps = [
                        {
                            "action": "short",
                            "exchange": exchange,
                            "symbol": symbol,
                            "market": "perpetual",
                            "price": perp_price,
                            "quantity": min_liquidity,
                            "fee_percent": EXCHANGE_FEE_STRUCTURES[exchange]["taker"]
                        },
                        {
                            "action": "long",
                            "exchange": exchange,
                            "symbol": spot_symbol,
                            "market": "spot",
                            "price": spot_price,
                            "quantity": min_liquidity,
                            "fee_percent": EXCHANGE_FEE_STRUCTURES[exchange]["taker"]
                        },
                        {
                            "action": "collect_funding",
                            "exchange": exchange,
                            "symbol": symbol,
                            "funding_rate": funding_rate,
                            "next_funding_time": next_funding_time.isoformat(),
                            "expected_funding_profit": funding_profit
                        }
                    ]
                else:  # long_perp_short_spot
                    execution_steps = [
                        {
                            "action": "long",
                            "exchange": exchange,
                            "symbol": symbol,
                            "market": "perpetual",
                            "price": perp_price,
                            "quantity": min_liquidity,
                            "fee_percent": EXCHANGE_FEE_STRUCTURES[exchange]["taker"]
                        },
                        {
                            "action": "short",
                            "exchange": exchange,
                            "symbol": spot_symbol,
                            "market": "spot",
                            "price": spot_price,
                            "quantity": min_liquidity,
                            "fee_percent": EXCHANGE_FEE_STRUCTURES[exchange]["taker"]
                        },
                        {
                            "action": "collect_funding",
                            "exchange": exchange,
                            "symbol": symbol,
                            "funding_rate": funding_rate,
                            "next_funding_time": next_funding_time.isoformat(),
                            "expected_funding_profit": funding_profit
                        }
                    ]
                
                # Create opportunity object
                opportunity = ArbitrageOpportunity(
                    type="funding",
                    timestamp=datetime.now(),
                    exchanges=[exchange],
                    symbols=[symbol, spot_symbol],
                    path=[(symbol, spot_symbol, exchange)],
                    raw_spread=raw_spread,
                    net_profit_percent=net_profit_percent,
                    estimated_profit_usd=estimated_profit_usd,
                    required_capital=required_capital,
                    execution_time_estimate_ms=execution_time_estimate,
                    price_volatility=avg_volatility,
                    slippage_estimate=slippage_estimate,
                    liquidity_confidence=liquidity_confidence,
                    success_probability=success_probability,
                    risk_reward_ratio=risk_reward_ratio,
                    opportunity_score=opportunity_score,
                    execution_steps=execution_steps,
                    market_conditions=market_conditions,
                    historical_success_rate=historical_success_rate
                )
                
                opportunities.append(opportunity)
            
            logger.debug(f"Detected {len(opportunities)} funding rate arbitrage opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Error in funding rate arbitrage detection: {e}", exc_info=True)
            return []
    
    # Helper methods for background monitoring tasks
    
    async def _monitor_cross_exchange_opportunities(self):
        """Background task to continuously monitor cross-exchange opportunities."""
        while True:
            try:
                if self.config["enable_cross_exchange"]:
                    opportunities = await self.detect_cross_exchange_arbitrage()
                    # Store in Redis for quick access
                    if opportunities:
                        await self.redis_client.set(
                            "arbitrage:cross_exchange:latest", 
                            [op.to_dict() for op in opportunities],
                            ex=60  # Expire after 60 seconds
                        )
            except Exception as e:
                logger.error(f"Error in cross-exchange monitor: {e}")
            
            await asyncio.sleep(self.config["update_frequency_ms"] / 1000)
    
    async def _monitor_triangular_opportunities(self):
        """Background task to continuously monitor triangular opportunities."""
        while True:
            try:
                if self.config["enable_triangular"]:
                    opportunities = await self.detect_triangular_arbitrage()
                    # Store in Redis for quick access
                    if opportunities:
                        await self.redis_client.set(
                            "arbitrage:triangular:latest", 
                            [op.to_dict() for op in opportunities],
                            ex=60  # Expire after 60 seconds
                        )
            except Exception as e:
                logger.error(f"Error in triangular monitor: {e}")
            
            await asyncio.sleep(self.config["update_frequency_ms"] / 1000)
    
    async def _monitor_statistical_opportunities(self):
        """Background task to continuously monitor statistical opportunities."""
        while True:
            try:
                if self.config["enable_statistical"]:
                    opportunities = await self.detect_statistical_arbitrage()
                    # Store in Redis for quick access
                    if opportunities:
                        await self.redis_client.set(
                            "arbitrage:statistical:latest", 
                            [op.to_dict() for op in opportunities],
                            ex=300  # Expire after 5 minutes (statistical opportunities change more slowly)
                        )
            except Exception as e:
                logger.error(f"Error in statistical monitor: {e}")
            
            # Statistical arbitrage needs less frequent updates
            await asyncio.sleep(5)  # Check every 5 seconds
    
    async def _monitor_latency_opportunities(self):
        """Background task to continuously monitor latency opportunities."""
        while True:
            try:
                if self.config["enable_latency"]:
                    opportunities = await self.detect_latency_arbitrage()
                    # Store in Redis for quick access
                    if opportunities:
                        await self.redis_client.set(
                            "arbitrage:latency:latest", 
                            [op.to_dict() for op in opportunities],
                            ex=30  # Expire after 30 seconds (latency opportunities are very time-sensitive)
                        )
            except Exception as e:
                logger.error(f"Error in latency monitor: {e}")
            
            # Latency arbitrage needs very frequent updates
            await asyncio.sleep(0.1)  # Check every 100ms
    
    async def _monitor_funding_rate_opportunities(self):
        """Background task to continuously monitor funding rate opportunities."""
        while True:
            try:
                if self.config["enable_funding_rate"]:
                    opportunities = await self.detect_funding_rate_arbitrage()
                    # Store in Redis for quick access
                    if opportunities:
                        await self.redis_client.set(
                            "arbitrage:funding:latest", 
                            [op.to_dict() for op in opportunities],
                            ex=600  # Expire after 10 minutes (funding opportunities change slowly)
                        )
            except Exception as e:
                logger.error(f"Error in funding rate monitor: {e}")
            
            # Funding rate arbitrage needs less frequent updates
            await asyncio.sleep(60)  # Check every minute
    
    async def _update_market_graph(self):
        """Background task to keep the market graph updated."""
        while True:
            try:
                for exchange in ["binance", "deriv"]:
                    await self._update_exchange_market_graph(exchange)
            except Exception as e:
                logger.error(f"Error updating market graph: {e}")
            
            # Update every few seconds
            await asyncio.sleep(5)
    
    async def _track_performance_metrics(self):
        """Background task to track performance metrics."""
        while True:
            try:
                # Update success rates
                for op_type in ARBITRAGE_OPPORTUNITY_TYPES:
                    if op_type in self.type_success_rates:
                        self.metrics_collector.set_gauge(
                            f"arbitrage_success_rate_{op_type}", 
                            self.type_success_rates[op_type]
                        )
            except Exception as e:
                logger.error(f"Error tracking performance metrics: {e}")
            
            # Update every minute
            await asyncio.sleep(60)
    
    # Helper methods for triangular arbitrage
    
    async def _update_exchange_market_graph(self, exchange: str):
        """
        Update the market graph for a specific exchange by fetching latest
        trading pairs and their rates.
        """
        try:
            # Get all trading pairs for the exchange
            trading_pairs = await self.feed_coordinator.get_all_trading_pairs(exchange)
            
            # Create edges for each trading pair
            for pair in trading_pairs:
                base_asset = pair.get("base_asset", "")
                quote_asset = pair.get("quote_asset", "")
                symbol = pair.get("symbol", "")
                
                if not base_asset or not quote_asset or not symbol:
                    continue
                
                # Get latest ticker data
                ticker = await self.feed_coordinator.get_ticker(exchange, symbol)
                
                if not ticker:
                    continue
                
                # Get bid/ask prices
                bid_price = ticker.get("bid", 0)
                ask_price = ticker.get("ask", 0)
                
                if bid_price <= 0 or ask_price <= 0:
                    continue
                
                # Get trading fee
                fee = EXCHANGE_FEE_STRUCTURES[exchange]["taker"]
                
                # Add edges to the graph
                # Edge from base to quote (selling base for quote)
                self.market_graph.add_edge(
                    base_asset,
                    quote_asset,
                    rate=bid_price,
                    fee=fee,
                    symbol=symbol,
                    exchange=exchange,
                    direction="sell"
                )
                
                # Edge from quote to base (buying base with quote)
                inverse_rate = 1 / ask_price if ask_price > 0 else 0
                self.market_graph.add_edge(
                    quote_asset,
                    base_asset,
                    rate=inverse_rate,
                    fee=fee,
                    symbol=symbol,
                    exchange=exchange,
                    direction="buy"
                )
            
            logger.debug(f"Updated market graph for {exchange}: {self.market_graph.number_of_nodes()} nodes, "
                        f"{self.market_graph.number_of_edges()} edges")
                        
        except Exception as e:
            logger.error(f"Error updating market graph for {exchange}: {e}")
    
    # Additional helper methods
    
    def _estimate_execution_time(self, buy_exchange: str, sell_exchange: str, symbol: str) -> float:
        """Estimate the execution time for a cross-exchange arbitrage in milliseconds."""
        # Base execution times by exchange (in ms)
        base_times = {
            "binance": 150,
            "deriv": 200,
            # Add more exchanges as needed
        }
        
        # Asset-specific factors (multipliers)
        asset_factors = {
            "BTC": 1.0,
            "ETH": 1.1,
            "default": 1.2
        }
        
        buy_time = base_times.get(buy_exchange, 200)
        sell_time = base_times.get(sell_exchange, 200)
        
        # Get asset factor
        asset = symbol.split("/")[0] if "/" in symbol else symbol
        factor = asset_factors.get(asset, asset_factors["default"])
        
        # Calculate total time
        total_time = (buy_time + sell_time) * factor
        
        # Add transfer time if different exchanges
        if buy_exchange != sell_exchange:
            total_time += 50  # Additional coordination time
        
        return total_time
    
    def _estimate_triangular_execution_time(self, exchange: str, num_steps: int) -> float:
        """Estimate execution time for triangular arbitrage."""
        # Base time per step in ms
        base_time_per_step = 150
        
        # Exchange-specific factor
        exchange_factors = {
            "binance": 1.0,
            "deriv": 1.2,
            # Add more exchanges as needed
        }
        
        factor = exchange_factors.get(exchange, 1.0)
        
        # Calculate total time
        total_time = base_time_per_step * num_steps * factor
        
        return total_time
    
    def _estimate_statistical_execution_time(self, exchange: str) -> float:
        """Estimate execution time for statistical arbitrage."""
        # Base times by exchange (in ms)
        base_times = {
            "binance": 300,  # Statistical arbitrage involves more complex orders
            "deriv": 350,
            # Add more exchanges as needed
        }
        
        return base_times.get(exchange, 300)
    
    def _estimate_latency_execution_time(self, exchange1: str, exchange2: str, time_diff_ms: float) -> float:
        """Estimate execution time for latency arbitrage."""
        # Base times by exchange (in ms)
        base_times = {
            "binance": 100,  # Latency arbitrage requires fast execution
            "deriv": 150,
            # Add more exchanges as needed
        }
        
        # For latency arbitrage, we need to be faster than the observed time difference
        execution_time = base_times.get(exchange1, 150) + base_times.get(exchange2, 150)
        
        # Ensure we can execute within the observed time difference
        if execution_time >= time_diff_ms:
            # Add a penalty if we're likely to be too slow
            execution_time *= 1.5
        
        return execution_time
    
    def _estimate_funding_execution_time(self, exchange: str) -> float:
        """Estimate execution time for funding rate arbitrage."""
        # Base times by exchange (in ms)
        base_times = {
            "binance": 250,  # Funding arbitrage requires spot and perpetual execution
            "deriv": 300,
            # Add more exchanges as needed
        }
        
        return base_times.get(exchange, 250) * 2  # Multiply by 2 for two markets
    
    async def _estimate_price_volatility(self, symbol: str, execution_time_ms: float) -> float:
        """Estimate price volatility for the execution duration."""
        try:
            # Get historical volatility from feature service
            volatility_data = await self.feature_extractor.get_volatility_features(
                symbol, 
                timeframe="1m",
                include=["realized_volatility_1m"]
            )
            
            # Extract 1-minute volatility
            volatility_1m = volatility_data.get("realized_volatility_1m", 0.05)  # Default to 0.05% if not available
            
            # Scale volatility to execution time window
            # Convert execution_time_ms to minutes
            execution_time_minutes = execution_time_ms / (1000 * 60)
            
            # Scale volatility proportionally to square root of time
            scaled_volatility = volatility_1m * (execution_time_minutes ** 0.5)
            
            return scaled_volatility
            
        except Exception as e:
            logger.warning(f"Error estimating price volatility for {symbol}: {e}")
            return 0.1  # Default to 0.1% if estimation fails
    
    async def _estimate_slippage(self, buy_exchange: str, sell_exchange: str, symbol: str, 
                                quantity: float) -> float:
        """Estimate slippage for a trade of given quantity across exchanges."""
        try:
            # Get orderbook data for both exchanges
            buy_orderbook = await self.feed_coordinator.get_orderbook(buy_exchange, symbol)
            sell_orderbook = await self.feed_coordinator.get_orderbook(sell_exchange, symbol)
            
            if not buy_orderbook or not sell_orderbook:
                return 0.1  # Default slippage if orderbook data unavailable
            
            # Estimate buy slippage
            buy_slippage = self._calculate_orderbook_slippage(
                buy_orderbook.get("asks", []), 
                quantity, 
                "buy"
            )
            
            # Estimate sell slippage
            sell_slippage = self._calculate_orderbook_slippage(
                sell_orderbook.get("bids", []),
                quantity,
                "sell"
            )
            
            # Total slippage
            total_slippage = buy_slippage + sell_slippage
            
            return total_slippage
            
        except Exception as e:
            logger.warning(f"Error estimating slippage: {e}")
            return 0.15  # Default to 0.15% if estimation fails
    
    def _calculate_orderbook_slippage(self, orders: List[List[float]], quantity: float, 
                                      side: str) -> float:
        """Calculate expected slippage based on orderbook depth."""
        if not orders:
            return 0.1  # Default slippage if no orders
        
        # Get best price
        best_price = orders[0][0] if orders else 0
        
        if best_price == 0:
            return 0.1
        
        # Track filled quantity and total cost
        filled_qty = 0
        total_cost = 0
        
        for price, qty in orders:
            available_qty = min(qty, quantity - filled_qty)
            total_cost += available_qty * price
            filled_qty += available_qty
            
            if filled_qty >= quantity:
                break
        
        # If we couldn't fill the entire quantity
        if filled_qty < quantity:
            return 0.2  # Higher slippage due to insufficient liquidity
        
        # Calculate average execution price
        avg_price = total_cost / quantity
        
        # Calculate slippage as percentage
        if side == "buy":
            slippage = ((avg_price - best_price) / best_price) * 100
        else:  # sell
            slippage = ((best_price - avg_price) / best_price) * 100
        
        return max(slippage, 0)  # Ensure non-negative
    
    async def _estimate_triangular_slippage(self, exchange: str, steps: List[Dict], 
                                           quantity: float) -> float:
        """Estimate slippage for triangular arbitrage with multiple conversion steps."""
        try:
            total_slippage = 0
            
            for step in steps:
                symbol = step.get("symbol", "")
                if not symbol:
                    continue
                
                # Get orderbook for this symbol
                orderbook = await self.feed_coordinator.get_orderbook(exchange, symbol)
                
                if not orderbook:
                    total_slippage += 0.1  # Default slippage if orderbook unavailable
                    continue
                
                # Determine side
                side = step.get("direction", "")
                
                if side == "buy":
                    # Calculate slippage for buy
                    step_slippage = self._calculate_orderbook_slippage(
                        orderbook.get("asks", []),
                        quantity,
                        "buy"
                    )
                else:  # sell
                    # Calculate slippage for sell
                    step_slippage = self._calculate_orderbook_slippage(
                        orderbook.get("bids", []),
                        quantity,
                        "sell"
                    )
                
                total_slippage += step_slippage
            
            # Slippage tends to compound in triangular arbitrage
            total_slippage *= 1.1  # Add 10% extra for triangular-specific risks
            
            return total_slippage
            
        except Exception as e:
            logger.warning(f"Error estimating triangular slippage: {e}")
            return 0.2  # Default to 0.2% if estimation fails
    
    async def _estimate_statistical_slippage(self, exchange: str, base_asset: str, 
                                            quote_asset: str, quantity: float) -> float:
        """Estimate slippage for statistical arbitrage involving two assets."""
        try:
            # Statistical arbitrage often involves larger positions
            # which can result in more slippage
            base_slippage = await self._estimate_asset_slippage(exchange, base_asset, quantity)
            quote_slippage = await self._estimate_asset_slippage(exchange, quote_asset, quantity)
            
            # Use the higher slippage as a conservative estimate
            max_slippage = max(base_slippage, quote_slippage)
            
            # Statistical arbitrage is usually executed over a longer timeframe,
            # so add additional slippage for market impact
            total_slippage = max_slippage * 1.2  # Add 20% extra for statistical-specific risks
            
            return total_slippage
            
        except Exception as e:
            logger.warning(f"Error estimating statistical slippage: {e}")
            return 0.25  # Default to 0.25% if estimation fails
    
    async def _estimate_asset_slippage(self, exchange: str, asset: str, quantity: float) -> float:
        """Estimate slippage for a single asset."""
        try:
            # Convert asset to a symbol that we can query
            symbol = f"{asset}/USDT"  # Use USDT pair as a common denominator
            
            # Get orderbook
            orderbook = await self.feed_coordinator.get_orderbook(exchange, symbol)
            
            if not orderbook:
                return 0.15  # Default slippage if orderbook unavailable
            
            # Calculate both buy and sell slippage
            buy_slippage = self._calculate_orderbook_slippage(
                orderbook.get("asks", []),
                quantity,
                "buy"
            )
            
            sell_slippage = self._calculate_orderbook_slippage(
                orderbook.get("bids", []),
                quantity,
                "sell"
            )
            
            # Use the higher slippage as a conservative estimate
            return max(buy_slippage, sell_slippage)
            
        except Exception as e:
            logger.warning(f"Error estimating asset slippage for {asset}: {e}")
            return 0.15  # Default to 0.15% if estimation fails
    
    async def _estimate_funding_slippage(self, exchange: str, perp_symbol: str, 
                                       spot_symbol: str, quantity: float) -> float:
        """Estimate slippage for funding rate arbitrage between perp and spot."""
        try:
            # Get orderbooks
            perp_orderbook = await self.feed_coordinator.get_orderbook(exchange, perp_symbol)
            spot_orderbook = await self.feed_coordinator.get_orderbook(exchange, spot_symbol)
            
            if not perp_orderbook or not spot_orderbook:
                return 0.2  # Default slippage if orderbook unavailable
            
            # Calculate slippage for both markets (assume we need both buy and sell)
            perp_buy_slippage = self._calculate_orderbook_slippage(
                perp_orderbook.get("asks", []),
                quantity,
                "buy"
            )
            
            perp_sell_slippage = self._calculate_orderbook_slippage(
                perp_orderbook.get("bids", []),
                quantity,
                "sell"
            )
            
            spot_buy_slippage = self._calculate_orderbook_slippage(
                spot_orderbook.get("asks", []),
                quantity,
                "buy"
            )
            
            spot_sell_slippage = self._calculate_orderbook_slippage(
                spot_orderbook.get("bids", []),
                quantity,
                "sell"
            )
            
            # Take worst-case scenario for each market
            perp_slippage = max(perp_buy_slippage, perp_sell_slippage)
            spot_slippage = max(spot_buy_slippage, spot_sell_slippage)
            
            # Total slippage
            total_slippage = perp_slippage + spot_slippage
            
            return total_slippage
            
        except Exception as e:
            logger.warning(f"Error estimating funding slippage: {e}")
            return 0.2  # Default to 0.2% if estimation fails
    
    async def _get_exchange_latency_metrics(self, exchanges: List[str]) -> Dict[str, Dict[str, float]]:
        """Get latency metrics for exchanges."""
        result = {}
        
        for exchange in exchanges:
            try:
                # Get latency metrics from feed coordinator
                metrics = await self.feed_coordinator.get_exchange_metrics(exchange)
                
                if metrics:
                    result[exchange] = metrics
                else:
                    # Default metrics if not available
                    result[exchange] = {
                        "response_time_ms": 150,
                        "propagation_latency_ms": 200,
                        "update_frequency_ms": 100
                    }
            except Exception as e:
                logger.warning(f"Error getting latency metrics for {exchange}: {e}")
                # Default metrics
                result[exchange] = {
                    "response_time_ms": 150,
                    "propagation_latency_ms": 200,
                    "update_frequency_ms": 100
                }
        
        return result
    
    async def _get_exchange_liquidity(self, exchange: str, symbol: str, side: str) -> float:
        """Get available liquidity for a symbol on an exchange."""
        try:
            # Get orderbook data
            orderbook = await self.feed_coordinator.get_orderbook(exchange, symbol)
            
            if not orderbook:
                return 0
            
            # Calculate available liquidity for the specified side
            if side == "buy" or side == "both":
                # Sum up ask quantities for buying
                buy_liquidity = sum(qty for _, qty in orderbook.get("asks", [])[:5])
            else:
                buy_liquidity = 0
                
            if side == "sell" or side == "both":
                # Sum up bid quantities for selling
                sell_liquidity = sum(qty for _, qty in orderbook.get("bids", [])[:5])
            else:
                sell_liquidity = 0
            
            if side == "both":
                return min(buy_liquidity, sell_liquidity)
            elif side == "buy":
                return buy_liquidity
            else:  # sell
                return sell_liquidity
                
        except Exception as e:
            logger.warning(f"Error getting exchange liquidity for {exchange}:{symbol}: {e}")
            return 0
    
    async def _evaluate_triangular_liquidity(self, exchange: str, steps: List[Dict]) -> Tuple[float, float]:
        """Evaluate liquidity across all steps of a triangular arbitrage."""
        liquidity_values = []
        confidence_values = []
        
        for step in steps:
            symbol = step.get("symbol", "")
            if not symbol:
                continue
            
            # Get orderbook for this symbol
            orderbook = await self.feed_coordinator.get_orderbook(exchange, symbol)
            
            if not orderbook:
                liquidity_values.append(0)
                confidence_values.append(0)
                continue
            
            # Determine side
            side = step.get("direction", "")
            
            # Calculate available liquidity
            if side == "buy":
                # Sum up ask quantities
                liquidity = sum(qty for _, qty in orderbook.get("asks", [])[:5])
                
                # Calculate liquidity confidence
                total_asks = len(orderbook.get("asks", []))
                confidence = min(total_asks / 10, 1.0)  # Normalize to 0-1
            else:  # sell
                # Sum up bid quantities
                liquidity = sum(qty for _, qty in orderbook.get("bids", [])[:5])
                
                # Calculate liquidity confidence
                total_bids = len(orderbook.get("bids", []))
                confidence = min(total_bids / 10, 1.0)  # Normalize to 0-1
            
            liquidity_values.append(liquidity)
            confidence_values.append(confidence)
        
        # Use the minimum liquidity across all steps
        min_liquidity = min(liquidity_values) if liquidity_values else 0
        
        # Calculate overall confidence as the average
        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
        
        return min_liquidity, avg_confidence
    
    def _calculate_liquidity_confidence(self, buy_exchange: str, sell_exchange: str, 
                                       symbol: str, liquidity: float) -> float:
        """Calculate confidence in available liquidity."""
        # Get minimum required liquidity for this symbol
        min_required = self.config["min_liquidity_requirements"].get(
            symbol, self.config["min_liquidity_requirements"]["default"]
        )
        
        # Calculate confidence based on available vs. required liquidity
        confidence = min(liquidity / (min_required * 2), 1.0)
        
        # Adjust for exchange reliability
        exchange_factors = {
            "binance": 1.0,
            "deriv": 0.9,
            # Add more exchanges as needed
        }
        
        buy_factor = exchange_factors.get(buy_exchange, 0.8)
        sell_factor = exchange_factors.get(sell_exchange, 0.8)
        
        # Apply exchange factors
        confidence *= (buy_factor + sell_factor) / 2
        
        return confidence
    
    def _calculate_statistical_liquidity_confidence(self, base_data: Dict, quote_data: Dict) -> float:
        """Calculate liquidity confidence for statistical arbitrage."""
        # Extract order book depth
        base_depth = base_data.get("order_depth", 0)
        quote_depth = quote_data.get("order_depth", 0)
        
        # Extract trading volume
        base_volume = base_data.get("volume_24h", 0)
        quote_volume = quote_data.get("volume_24h", 0)
        
        # Calculate depth confidence
        depth_confidence = min(base_depth, quote_depth) / max(base_depth, quote_depth) if max(base_depth, quote_depth) > 0 else 0
        
        # Calculate volume confidence
        volume_confidence = min(base_volume, quote_volume) / max(base_volume, quote_volume) if max(base_volume, quote_volume) > 0 else 0
        
        # Combined confidence (weighted)
        combined_confidence = (depth_confidence * 0.6) + (volume_confidence * 0.4)
        
        return combined_confidence
    
    async def _get_asset_market_data(self, exchange: str, asset: str) -> Dict:
        """Get market data for an asset."""
        try:
            # Convert asset to a symbol that we can query
            symbol = f"{asset}/USDT"  # Use USDT pair as a common denominator
            
            # Get ticker data
            ticker = await self.feed_coordinator.get_ticker(exchange, symbol)
            
            if not ticker:
                return {}
            
            # Get orderbook
            orderbook = await self.feed_coordinator.get_orderbook(exchange, symbol)
            
            # Calculate order depth
            order_depth = 0
            if orderbook:
                bid_depth = sum(qty for _, qty in orderbook.get("bids", [])[:5])
                ask_depth = sum(qty for _, qty in orderbook.get("asks", [])[:5])
                order_depth = (bid_depth + ask_depth) / 2
            
            # Get 24h volume
            volume_24h = ticker.get("volume", 0)
            
            # Get current price
            price = ticker.get("price", 0)
            
            return {
                "price": price,
                "order_depth": order_depth,
                "volume_24h": volume_24h,
                "symbol": symbol,
                "exchange": exchange
            }
        
        except Exception as e:
            logger.warning(f"Error getting market data for {asset}: {e}")
            return {}
    
    def _get_asset_liquidity(self, asset_data: Dict) -> float:
        """Get liquidity for an asset from its market data."""
        order_depth = asset_data.get("order_depth", 0)
        price = asset_data.get("price", 0)
        
        if price > 0:
            return order_depth * price
        return 0
    
    async def _get_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Get current market conditions for a symbol."""
        try:
            # Get market state from feature service
            market_state = await self.feature_extractor.get_market_state(symbol)
            
            # Get market conditions features
            market_conditions = await self.feature_extractor.get_market_condition_features(symbol)
            
            # Combine data
            combined = {
                "state": market_state,
                **market_conditions
            }
            
            return combined
        
        except Exception as e:
            logger.warning(f"Error getting market conditions for {symbol}: {e}")
            return {"state": "unknown"}
    
    async def _get_combined_market_conditions(self, symbols: List[str]) -> Dict[str, Any]:
        """Get combined market conditions for multiple symbols."""
        all_conditions = {}
        
        for symbol in symbols:
            try:
                conditions = await self._get_market_conditions(symbol)
                all_conditions[symbol] = conditions
            except Exception as e:
                logger.warning(f"Error getting market conditions for {symbol}: {e}")
        
        # Derive combined state
        if not all_conditions:
            return {"state": "unknown"}
        
        # Count states
        states = [conditions.get("state", "unknown") for conditions in all_conditions.values()]
        state_counts = {}
        for state in states:
            state_counts[state] = state_counts.get(state, 0) + 1
        
        # Get most common state
        if state_counts:
            combined_state = max(state_counts.items(), key=lambda x: x[1])[0]
        else:
            combined_state = "unknown"
        
        # Create combined conditions
        combined = {
            "state": combined_state,
            "symbols": symbols,
            "individual_conditions": all_conditions
        }
        
        return combined
    
    def _estimate_required_capital(self, exchange: str, asset: str, quantity: float) -> float:
        """Estimate required capital for a position in USD."""
        try:
            # Get cached price data if available
            cache_key = f"{exchange}:{asset}/USDT"
            if cache_key in self.market_data_cache:
                ticker = self.market_data_cache[cache_key].get("ticker", {})
                price = ticker.get("price", 0)
                
                if price > 0:
                    return price * quantity
            
            # Default estimation if price not available
            return quantity * 100  # Assume $100 per unit as default
        
        except Exception as e:
            logger.warning(f"Error estimating required capital: {e}")
            return quantity * 100  # Default estimate
    
    def _estimate_statistical_capital(self, exchange: str, base_asset: str, 
                                    quote_asset: str, liquidity: float) -> float:
        """Estimate required capital for statistical arbitrage."""
        try:
            # Get cached price data for both assets
            base_key = f"{exchange}:{base_asset}/USDT"
            quote_key = f"{exchange}:{quote_asset}/USDT"
            
            base_price = 0
            quote_price = 0
            
            if base_key in self.market_data_cache:
                ticker = self.market_data_cache[base_key].get("ticker", {})
                base_price = ticker.get("price", 0)
            
            if quote_key in self.market_data_cache:
                ticker = self.market_data_cache[quote_key].get("ticker", {})
                quote_price = ticker.get("price", 0)
            
            # Statistical arbitrage typically requires positions in both assets
            base_position = liquidity * 0.5
            quote_position = liquidity * 0.5
            
            if base_price > 0 and quote_price > 0:
                return (base_position * base_price) + (quote_position * quote_price)
            
            # Default estimation if prices not available
            return liquidity
        
        except Exception as e:
            logger.warning(f"Error estimating statistical capital: {e}")
            return liquidity
    
    def _calculate_success_probability(self, opportunity_type: str, raw_spread: float,
                                     net_profit_percent: float, execution_time_ms: float,
                                     price_volatility: float, slippage_estimate: float,
                                     liquidity_confidence: float, market_conditions: Dict[str, Any],
                                     additional_factors: Dict[str, float] = None) -> float:
        """
        Calculate the probability of successful execution for an arbitrage opportunity.
        
        This sophisticated model considers multiple factors:
        - Profitability (raw spread and net profit)
        - Execution time
        - Price volatility
        - Slippage
        - Liquidity confidence
        - Market conditions
        - Opportunity-specific factors
        - Historical performance
        """
        # Base probability starts at maximum
        base_probability = 0.95
        
        # Get historical success rate for this opportunity type
        historical_rate = self.type_success_rates.get(opportunity_type, 0.8)
        
        # Factor 1: Profitability
        # Higher profit margin increases success probability
        profit_factor = min(net_profit_percent / (self.config["min_profitable_spread"] * 5), 1.0)
        
        # Factor 2: Execution Time
        # Faster execution increases success probability
        max_time = self.config["max_execution_time_ms"]
        time_factor = 1.0 - min(execution_time_ms / max_time, 1.0)
        
        # Factor 3: Price Volatility
        # Lower volatility increases success probability
        volatility_threshold = 0.5  # 0.5% is considered high volatility for short timeframes
        volatility_factor = 1.0 - min(price_volatility / volatility_threshold, 1.0)
        
        # Factor 4: Slippage
        # Lower slippage increases success probability
        slippage_threshold = self.config["min_profitable_spread"] / 2
        slippage_factor = 1.0 - min(slippage_estimate / slippage_threshold, 1.0)
        
        # Factor 5: Liquidity Confidence
        # Higher liquidity confidence increases success probability
        liquidity_factor = liquidity_confidence
        
        # Factor 6: Market Conditions
        # Favorable market conditions increase success probability
        market_state = market_conditions.get("state", "unknown")
        market_factors = {
            "trending": 0.9,
            "ranging": 0.8,
            "volatile": 0.6,
            "calm": 1.0,
            "unknown": 0.7
        }
        market_factor = market_factors.get(market_state, 0.7)
        
        # Factor 7: Opportunity-specific factors
        type_factors = {
            "cross_exchange": 0.85,
            "triangular": 0.80,
            "statistical": 0.75,
            "latency": 0.70,
            "funding": 0.90
        }
        opportunity_factor = type_factors.get(opportunity_type, 0.75)
        
        # Additional factors if provided
        additional_factor = 1.0
        if additional_factors:
            # Average all additional factors
            additional_values = [value for value in additional_factors.values() if isinstance(value, (int, float))]
            if additional_values:
                additional_factor = sum(additional_values) / len(additional_values)
        
        # Calculate weighted success probability
        weights = {
            "historical_rate": 0.25,
            "profit_factor": 0.15,
            "time_factor": 0.10,
            "volatility_factor": 0.10,
            "slippage_factor": 0.10,
            "liquidity_factor": 0.10,
            "market_factor": 0.05,
            "opportunity_factor": 0.10,
            "additional_factor": 0.05
        }
        
        weighted_probability = (
            (historical_rate * weights["historical_rate"]) +
            (profit_factor * weights["profit_factor"]) +
            (time_factor * weights["time_factor"]) +
            (volatility_factor * weights["volatility_factor"]) +
            (slippage_factor * weights["slippage_factor"]) +
            (liquidity_factor * weights["liquidity_factor"]) +
            (market_factor * weights["market_factor"]) +
            (opportunity_factor * weights["opportunity_factor"]) +
            (additional_factor * weights["additional_factor"])
        )
        
        # Apply base probability cap
        final_probability = min(weighted_probability, base_probability)
        
        # Ensure probability is in valid range
        final_probability = max(min(final_probability, 1.0), 0.0)
        
        return final_probability
    
    def _calculate_opportunity_score(self, net_profit_percent: float, success_probability: float,
                                   liquidity_confidence: float, execution_time_ms: float,
                                   risk_reward_ratio: float, additional_factors: Dict[str, float] = None) -> float:
        """
        Calculate overall opportunity score for prioritization.
        
        This score combines profitability, success probability, execution speed,
        and risk factors into a single metric for ranking opportunities.
        """
        # Normalize profit to 0-1 scale
        profit_score = min(net_profit_percent / 5.0, 1.0)  # Cap at 5% profit
        
        # Success probability is already 0-1
        probability_score = success_probability
        
        # Normalize execution time to 0-1 scale (faster is better)
        max_time = self.config["max_execution_time_ms"]
        time_score = 1.0 - min(execution_time_ms / max_time, 1.0)
        
        # Liquidity confidence is already 0-1
        liquidity_score = liquidity_confidence
        
        # Normalize risk-reward ratio to 0-1 scale
        risk_reward_score = min(risk_reward_ratio / 5.0, 1.0)  # Cap at 5:1 ratio
        
        # Calculate additional score if provided
        additional_score = 0.0
        if additional_factors:
            # Average all additional factors
            additional_values = [value for value in additional_factors.values() if isinstance(value, (int, float))]
            if additional_values:
                additional_score = sum(additional_values) / len(additional_values)
                additional_score = min(additional_score, 1.0)  # Ensure 0-1 scale
        
        # Calculate weighted score
        weights = {
            "profit": 0.30,
            "probability": 0.25,
            "time": 0.10,
            "liquidity": 0.15,
            "risk_reward": 0.15,
            "additional": 0.05
        }
        
        weighted_score = (
            (profit_score * weights["profit"]) +
            (probability_score * weights["probability"]) +
            (time_score * weights["time"]) +
            (liquidity_score * weights["liquidity"]) +
            (risk_reward_score * weights["risk_reward"]) +
            (additional_score * weights["additional"])
        )
        
        # Scale to 0-100 for easier interpretation
        final_score = weighted_score * 100
        
        return final_score
