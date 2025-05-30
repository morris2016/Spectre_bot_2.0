
#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Market Microstructure Analysis Module

This module provides sophisticated market microstructure analysis capabilities for optimal
trade execution. It analyzes order book dynamics, market impact, order flow imbalance,
and other microstructure patterns to gain execution advantages.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import asyncio
import time
from collections import deque
import logging
from dataclasses import dataclass
from enum import Enum, auto

from common.logger import get_logger
from common.constants import (
    TICK_SIZE_MAPPING, LIQUIDITY_THRESHOLDS, ORDER_BOOK_LEVELS,
    FLOW_IMBALANCE_WINDOW, SPREAD_ANALYSIS_WINDOW, 
    MICROSTRUCTURE_PATTERN_LOOKBACK
)
from common.utils import calculate_vwap, exponential_decay, calculate_trade_imbalance
from common.exceptions import MicrostructureAnalysisError
from data_feeds.base_feed import OrderBookData, TradeData
from execution_engine.order_manager import OrderManager

logger = get_logger("microstructure_analyzer")


class LiquidityState(Enum):
    """Enum representing the current liquidity state of the market."""
    VERY_THIN = auto()
    THIN = auto()
    MODERATE = auto()
    DEEP = auto()
    VERY_DEEP = auto()


class MarketImpactLevel(Enum):
    """Enum representing the expected market impact level for orders."""
    NEGLIGIBLE = auto()
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    SEVERE = auto()


@dataclass
class MicrostructureState:
    """Data class representing the current market microstructure state."""
    liquidity_state: LiquidityState
    spread_state: float  # Normalized spread relative to historical average
    order_book_imbalance: float  # Range: -1.0 (all asks) to 1.0 (all bids)
    flow_imbalance: float  # Range: -1.0 (all sells) to 1.0 (all buys)
    volatility_state: float  # Normalized tick-by-tick volatility
    expected_market_impact: Dict[float, MarketImpactLevel]  # Impact for different order sizes
    hidden_liquidity_estimate: float  # Estimated hidden liquidity as % of visible
    timestamp: float


class MicrostructureAnalyzer:
    """
    Analyzes market microstructure for optimal trade execution.
    
    This class provides sophisticated analysis of market microstructure elements
    including order book dynamics, order flow, spread behavior, and more to gain
    execution advantages.
    """
    
    def __init__(self, symbol: str, exchange: str):
        """
        Initialize the MicrostructureAnalyzer.
        
        Args:
            symbol: The trading symbol to analyze
            exchange: The exchange name
        """
        self.symbol = symbol
        self.exchange = exchange
        self.logger = logging.getLogger(f"microstructure_analyzer.{exchange}.{symbol}")
        
        # Order book data
        self.current_book: Optional[OrderBookData] = None
        self.historical_books = deque(maxlen=ORDER_BOOK_LEVELS)
        
        # Trade data
        self.recent_trades = deque(maxlen=FLOW_IMBALANCE_WINDOW)
        
        # Calculated metrics
        self.spread_history = deque(maxlen=SPREAD_ANALYSIS_WINDOW)
        self.tick_volatility = deque(maxlen=100)  # Last 100 ticks for volatility calc
        self.order_flow_imbalance = 0.0
        self.tick_size = self._get_tick_size()
        
        # Pattern recognition
        self.patterns = {}
        self.pattern_history = deque(maxlen=MICROSTRUCTURE_PATTERN_LOOKBACK)
        
        # Market impact model
        self.impact_model = self._initialize_impact_model()
        
        # State
        self.current_state: Optional[MicrostructureState] = None
        self.state_history = deque(maxlen=1000)
        
        self.logger.info(f"MicrostructureAnalyzer initialized for {exchange}:{symbol}")

    def _get_tick_size(self) -> float:
        """
        Get the tick size for the current symbol.
        
        Returns:
            The tick size value
        """
        if self.symbol in TICK_SIZE_MAPPING.get(self.exchange, {}):
            return TICK_SIZE_MAPPING[self.exchange][self.symbol]
        else:
            # Default to a small value if not found
            self.logger.warning(f"Tick size not defined for {self.exchange}:{self.symbol}, using default")
            return 0.0001

    def _initialize_impact_model(self) -> Dict:
        """
        Initialize the market impact model based on historical data.
        
        Returns:
            A dictionary containing the impact model parameters
        """
        # In a production system, this would load historical impact data
        # and train a model. Here we initialize with reasonable defaults.
        return {
            'base_impact': 0.2,  # Base impact in bps
            'power_law_exponent': 0.6,  # Typically between 0.5-0.7
            'decay_factor': 0.85,  # How quickly impact decays
            'liquidity_factor': 1.0,  # Multiplier based on current liquidity
            'model_version': 1.0
        }

    def update_order_book(self, order_book: OrderBookData) -> None:
        """
        Update internal state with new order book data.
        
        Args:
            order_book: New order book data
        """
        if self.current_book:
            self.historical_books.append(self.current_book)
        
        self.current_book = order_book
        
        # Calculate and update spread
        if order_book.asks and order_book.bids:
            best_ask = min([price for price, _ in order_book.asks])
            best_bid = max([price for price, _ in order_book.bids])
            current_spread = best_ask - best_bid
            self.spread_history.append(current_spread)
        
        # Update state
        self._update_state()

    def update_trades(self, trades: List[TradeData]) -> None:
        """
        Update internal state with new trade data.
        
        Args:
            trades: List of new trades
        """
        for trade in trades:
            self.recent_trades.append(trade)
            
            # Update tick volatility
            if len(self.tick_volatility) > 0:
                price_change = abs(trade.price - self.tick_volatility[-1])
                self.tick_volatility.append(trade.price)
            else:
                self.tick_volatility.append(trade.price)
        
        # Update order flow imbalance
        self._calculate_flow_imbalance()
        
        # Update state
        self._update_state()

    def _calculate_flow_imbalance(self) -> None:
        """Calculate and update the order flow imbalance."""
        if len(self.recent_trades) < 10:  # Need minimum sample
            self.order_flow_imbalance = 0.0
            return
            
        buy_volume = sum(t.quantity for t in self.recent_trades if t.is_buyer_maker is False)
        sell_volume = sum(t.quantity for t in self.recent_trades if t.is_buyer_maker is True)
        
        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            # Range from -1.0 (all sells) to 1.0 (all buys)
            self.order_flow_imbalance = (buy_volume - sell_volume) / total_volume
        else:
            self.order_flow_imbalance = 0.0

    def _assess_liquidity(self) -> LiquidityState:
        """
        Assess the current liquidity state of the market.
        
        Returns:
            A LiquidityState enum value
        """
        if not self.current_book:
            return LiquidityState.MODERATE
        
        # Calculate total liquidity within 1% of mid price
        mid_price = self._calculate_mid_price()
        if mid_price is None:
            return LiquidityState.MODERATE
            
        price_range = mid_price * 0.01  # 1% range
        
        bid_liquidity = sum(qty for price, qty in self.current_book.bids 
                           if price >= mid_price - price_range)
        ask_liquidity = sum(qty for price, qty in self.current_book.asks 
                           if price <= mid_price + price_range)
        
        total_liquidity = bid_liquidity + ask_liquidity
        
        # Determine state based on thresholds
        if total_liquidity < LIQUIDITY_THRESHOLDS.get(self.symbol, {}).get('VERY_THIN', 10):
            return LiquidityState.VERY_THIN
        elif total_liquidity < LIQUIDITY_THRESHOLDS.get(self.symbol, {}).get('THIN', 30):
            return LiquidityState.THIN
        elif total_liquidity < LIQUIDITY_THRESHOLDS.get(self.symbol, {}).get('MODERATE', 100):
            return LiquidityState.MODERATE
        elif total_liquidity < LIQUIDITY_THRESHOLDS.get(self.symbol, {}).get('DEEP', 300):
            return LiquidityState.DEEP
        else:
            return LiquidityState.VERY_DEEP

    def _calculate_mid_price(self) -> Optional[float]:
        """
        Calculate the current mid price.
        
        Returns:
            The mid price or None if order book data is insufficient
        """
        if not self.current_book or not self.current_book.asks or not self.current_book.bids:
            return None
            
        best_ask = min([price for price, _ in self.current_book.asks])
        best_bid = max([price for price, _ in self.current_book.bids])
        
        return (best_ask + best_bid) / 2.0

    def _calculate_order_book_imbalance(self) -> float:
        """
        Calculate the order book imbalance.
        
        Returns:
            A value from -1.0 (all asks) to 1.0 (all bids)
        """
        if not self.current_book or not self.current_book.asks or not self.current_book.bids:
            return 0.0
            
        # Calculate for top 5 levels
        top_bid_volume = sum(qty for _, qty in sorted(self.current_book.bids, 
                                                    key=lambda x: x[0], reverse=True)[:5])
        top_ask_volume = sum(qty for _, qty in sorted(self.current_book.asks, 
                                                    key=lambda x: x[0])[:5])
        
        total_volume = top_bid_volume + top_ask_volume
        if total_volume > 0:
            return (top_bid_volume - top_ask_volume) / total_volume
        else:
            return 0.0

    def _calculate_normalized_spread(self) -> float:
        """
        Calculate the normalized spread relative to historical average.
        
        Returns:
            Normalized spread value (1.0 = average, >1.0 = wider than average)
        """
        if not self.spread_history:
            return 1.0
            
        current_spread = self.spread_history[-1]
        avg_spread = sum(self.spread_history) / len(self.spread_history)
        
        if avg_spread > 0:
            return current_spread / avg_spread
        else:
            return 1.0

    def _calculate_volatility_state(self) -> float:
        """
        Calculate the current tick-by-tick volatility state.
        
        Returns:
            Normalized volatility value
        """
        if len(self.tick_volatility) < 10:
            return 0.5  # Default to moderate volatility
            
        # Use rolling standard deviation of price changes
        changes = [self.tick_volatility[i] - self.tick_volatility[i-1] 
                  for i in range(1, len(self.tick_volatility))]
        
        # Normalize by average price
        avg_price = sum(self.tick_volatility) / len(self.tick_volatility)
        if avg_price > 0:
            volatility = np.std(changes) / avg_price
            
            # Normalize to a reasonable range - higher is more volatile
            normalized = min(5.0, volatility * 1000)  # Cap at 5.0
            return normalized / 5.0  # Return as 0.0-1.0
        else:
            return 0.5

    def _estimate_hidden_liquidity(self) -> float:
        """
        Estimate hidden liquidity as a percentage of visible liquidity.
        
        Returns:
            Estimated hidden liquidity as fraction of visible
        """
        # In a real system, this would use historical trade vs order book analysis
        # Here we use a simplified model based on recent trade behavior
        
        if not self.recent_trades or not self.current_book:
            return 0.2  # Default assumption: 20% hidden
            
        # If trades are consistently larger than visible top-of-book, 
        # likely more hidden liquidity
        visible_top_sizes = []
        if self.current_book.bids:
            visible_top_sizes.append(max(self.current_book.bids, key=lambda x: x[0])[1])
        if self.current_book.asks:
            visible_top_sizes.append(min(self.current_book.asks, key=lambda x: x[0])[1])
            
        if not visible_top_sizes:
            return 0.2
            
        avg_visible = sum(visible_top_sizes) / len(visible_top_sizes)
        avg_trade_size = sum(t.quantity for t in self.recent_trades) / len(self.recent_trades)
        
        if avg_visible > 0:
            ratio = avg_trade_size / avg_visible
            # Bound between 0.05 (5%) and 0.8 (80%)
            return max(0.05, min(0.8, ratio * 0.25))
        else:
            return 0.2

    def _estimate_market_impact(self) -> Dict[float, MarketImpactLevel]:
        """
        Estimate market impact for different order sizes.
        
        Returns:
            Dictionary mapping order sizes to impact levels
        """
        mid_price = self._calculate_mid_price()
        if mid_price is None:
            return {
                0.1: MarketImpactLevel.NEGLIGIBLE,
                0.5: MarketImpactLevel.LOW,
                1.0: MarketImpactLevel.MODERATE,
                2.0: MarketImpactLevel.HIGH,
                5.0: MarketImpactLevel.SEVERE
            }
            
        # Calculate available liquidity at different levels
        liquidity_state = self._assess_liquidity()
        liquidity_factor = {
            LiquidityState.VERY_THIN: 2.5,
            LiquidityState.THIN: 1.8,
            LiquidityState.MODERATE: 1.0,
            LiquidityState.DEEP: 0.6,
            LiquidityState.VERY_DEEP: 0.3
        }[liquidity_state]
        
        # Size is in BTC or equivalent base currency
        impact_levels = {}
        
        # Define sizes as percentages of typical market liquidity
        typical_order_sizes = {
            0.1: MarketImpactLevel.NEGLIGIBLE,  # 0.1 BTC or equivalent
            0.5: MarketImpactLevel.LOW,         # 0.5 BTC
            1.0: MarketImpactLevel.MODERATE,    # 1.0 BTC
            2.0: MarketImpactLevel.HIGH,        # 2.0 BTC
            5.0: MarketImpactLevel.SEVERE       # 5.0 BTC
        }
        
        # Adjust based on current liquidity
        for size, base_impact in typical_order_sizes.items():
            # More liquid market can handle larger orders with less impact
            adjusted_impact_idx = min(len(MarketImpactLevel) - 1, 
                                    max(0, base_impact.value - 1 + int(liquidity_factor + 0.5)))
            impact_levels[size] = list(MarketImpactLevel)[adjusted_impact_idx]
            
        return impact_levels

    def _detect_microstructure_patterns(self) -> Dict[str, float]:
        """
        Detect market microstructure patterns in recent data.
        
        Returns:
            Dictionary of pattern names to confidence values (0.0-1.0)
        """
        patterns = {}
        
        # Detect iceberg orders
        patterns["iceberg_orders"] = self._detect_iceberg_orders()
        
        # Detect spoofing
        patterns["spoofing"] = self._detect_spoofing()
        
        # Detect momentum ignition
        patterns["momentum_ignition"] = self._detect_momentum_ignition()
        
        # Detect quote stuffing
        patterns["quote_stuffing"] = self._detect_quote_stuffing()
        
        # Detect layering
        patterns["layering"] = self._detect_layering()
        
        # Record to history
        self.pattern_history.append((time.time(), patterns))
        
        return patterns

    def _detect_iceberg_orders(self) -> float:
        """
        Detect the presence of iceberg orders.
        
        Returns:
            Confidence value from 0.0 to 1.0
        """
        # If not enough data, return low confidence
        if len(self.recent_trades) < 10 or not self.historical_books or not self.current_book:
            return 0.1
            
        # Look for repeated trades at the same price level with similar size
        price_groups = {}
        for trade in self.recent_trades:
            price = round(trade.price / self.tick_size) * self.tick_size  # Round to tick
            if price not in price_groups:
                price_groups[price] = []
            price_groups[price].append(trade.quantity)
            
        # Check for repeating patterns at same price levels
        iceberg_evidence = 0.0
        for price, quantities in price_groups.items():
            if len(quantities) < 3:
                continue
                
            # Check for similar sizes
            avg_qty = sum(quantities) / len(quantities)
            if avg_qty <= 0:
                continue
                
            similarity = sum(1 for q in quantities if 0.8 <= q/avg_qty <= 1.2) / len(quantities)
            
            # Higher similarity with more trades is stronger evidence
            evidence = similarity * min(1.0, len(quantities) / 10)
            iceberg_evidence = max(iceberg_evidence, evidence)
            
        return min(1.0, iceberg_evidence)

    def _detect_spoofing(self) -> float:
        """
        Detect potential spoofing behavior.
        
        Returns:
            Confidence value from 0.0 to 1.0
        """
        # Need multiple order book snapshots to detect spoofing
        if len(self.historical_books) < 5:
            return 0.0
            
        spoofing_evidence = 0.0
        
        # Look for large orders that appear and disappear without execution
        for i in range(1, len(self.historical_books)):
            prev_book = self.historical_books[i-1]
            curr_book = self.historical_books[i]
            
            # Check bids for disappeared orders
            for price, qty in prev_book.bids:
                # Find if this price level exists in current book
                matching_bids = [q for p, q in curr_book.bids if abs(p - price) < self.tick_size/2]
                
                if not matching_bids:
                    # Order disappeared - check if it was large relative to market
                    if len(prev_book.bids) >= 5:
                        avg_size = sum(q for _, q in prev_book.bids[:5]) / 5
                        if avg_size > 0 and qty > avg_size * 3:
                            # Large order disappeared without trade - potential spoofing
                            spoofing_evidence += min(1.0, qty / (avg_size * 10))
            
            # Check asks for disappeared orders
            for price, qty in prev_book.asks:
                matching_asks = [q for p, q in curr_book.asks if abs(p - price) < self.tick_size/2]
                
                if not matching_asks:
                    if len(prev_book.asks) >= 5:
                        avg_size = sum(q for _, q in prev_book.asks[:5]) / 5
                        if avg_size > 0 and qty > avg_size * 3:
                            spoofing_evidence += min(1.0, qty / (avg_size * 10))
        
        # Normalize by the number of comparisons
        if len(self.historical_books) > 1:
            spoofing_evidence /= (len(self.historical_books) - 1)
            
        return min(1.0, spoofing_evidence)

    def _detect_momentum_ignition(self) -> float:
        """
        Detect potential momentum ignition patterns.
        
        Returns:
            Confidence value from 0.0 to 1.0
        """
        # Need enough trades to detect momentum ignition
        if len(self.recent_trades) < 20:
            return 0.0
            
        # Look for a series of aggressive trades in same direction followed by reversal
        buy_volumes = []
        sell_volumes = []
        
        window_size = min(10, len(self.recent_trades) // 2)
        
        # Analyze in windows
        for i in range(0, len(self.recent_trades) - window_size, window_size):
            window = self.recent_trades[i:i+window_size]
            buy_vol = sum(t.quantity for t in window if not t.is_buyer_maker)
            sell_vol = sum(t.quantity for t in window if t.is_buyer_maker)
            
            buy_volumes.append(buy_vol)
            sell_volumes.append(sell_vol)
            
        if len(buy_volumes) < 2:
            return 0.0
            
        # Look for a spike in one direction followed by a reversal
        evidence = 0.0
        
        for i in range(1, len(buy_volumes)):
            # Check for buy spike followed by sell
            buy_increase = buy_volumes[i] / max(0.001, buy_volumes[i-1])
            if buy_increase > 3.0 and i < len(buy_volumes) - 1:
                # Check for subsequent reversal
                if sell_volumes[i+1] > buy_volumes[i] * 0.7:
                    evidence += min(1.0, (buy_increase - 3.0) / 7.0)
            
            # Check for sell spike followed by buy
            sell_increase = sell_volumes[i] / max(0.001, sell_volumes[i-1])
            if sell_increase > 3.0 and i < len(sell_volumes) - 1:
                if buy_volumes[i+1] > sell_volumes[i] * 0.7:
                    evidence += min(1.0, (sell_increase - 3.0) / 7.0)
                    
        return min(1.0, evidence)

    def _detect_quote_stuffing(self) -> float:
        """
        Detect potential quote stuffing behavior.
        
        Returns:
            Confidence value from 0.0 to 1.0
        """
        # Need enough order book history to detect quote stuffing
        if len(self.historical_books) < 10:
            return 0.0
            
        # Look for rapid changes in the number of orders
        order_counts = []
        
        for book in self.historical_books:
            bid_count = len(book.bids)
            ask_count = len(book.asks)
            order_counts.append(bid_count + ask_count)
            
        if not order_counts:
            return 0.0
            
        # Calculate rate of change
        changes = [abs(order_counts[i] - order_counts[i-1]) for i in range(1, len(order_counts))]
        
        if not changes:
            return 0.0
            
        # High and erratic change rates suggest quote stuffing
        avg_change = sum(changes) / len(changes)
        max_change = max(changes)
        
        # Normalize to 0.0-1.0 range
        normalized_avg = min(1.0, avg_change / 20.0)  # 20 orders per update is high
        normalized_max = min(1.0, max_change / 50.0)  # 50 orders per update is very high
        
        # Combine metrics with emphasis on maximum spikes
        evidence = normalized_avg * 0.3 + normalized_max * 0.7
        
        return evidence

    def _detect_layering(self) -> float:
        """
        Detect potential layering behavior.
        
        Returns:
            Confidence value from 0.0 to 1.0
        """
        # Need enough order book data to detect layering
        if not self.current_book or not self.current_book.bids or not self.current_book.asks:
            return 0.0
            
        # Layering involves multiple orders at different price levels on one side
        
        # Sort by price
        bids = sorted(self.current_book.bids, key=lambda x: x[0], reverse=True)
        asks = sorted(self.current_book.asks, key=lambda x: x[0])
        
        # Check for many similarly sized orders on bid side
        bid_layering = self._check_layering_pattern(bids)
        
        # Check for many similarly sized orders on ask side
        ask_layering = self._check_layering_pattern(asks)
        
        # Take the maximum evidence from either side
        return max(bid_layering, ask_layering)

    def _check_layering_pattern(self, orders: List[Tuple[float, float]]) -> float:
        """
        Check for layering pattern in a list of orders.
        
        Args:
            orders: List of (price, quantity) tuples
            
        Returns:
            Confidence value from 0.0 to 1.0
        """
        if len(orders) < 5:
            return 0.0
            
        # Check top 5 levels
        top_orders = orders[:5]
        
        # Calculate average size
        avg_size = sum(qty for _, qty in top_orders) / len(top_orders)
        
        if avg_size <= 0:
            return 0.0
            
        # Count orders with similar sizes
        similar_count = sum(1 for _, qty in top_orders if 0.7 <= qty/avg_size <= 1.3)
        
        # Calculate price spacing
        prices = [price for price, _ in top_orders]
        if len(prices) < 2:
            return 0.0
            
        price_diffs = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        avg_diff = sum(price_diffs) / len(price_diffs)
        
        if avg_diff <= 0:
            return 0.0
            
        # Check for consistent spacing
        consistent_spacing = sum(1 for diff in price_diffs if 0.5 <= diff/avg_diff <= 1.5)
        
        # Calculate evidence from size similarity and price spacing
        size_evidence = similar_count / len(top_orders)
        spacing_evidence = consistent_spacing / (len(prices) - 1) if len(prices) > 1 else 0
        
        # Combine evidence
        return (size_evidence * 0.6 + spacing_evidence * 0.4)

    def _update_state(self) -> None:
        """Update the current microstructure state."""
        if not self.current_book:
            return
            
        # Assess market state metrics
        liquidity_state = self._assess_liquidity()
        order_book_imbalance = self._calculate_order_book_imbalance()
        normalized_spread = self._calculate_normalized_spread()
        volatility_state = self._calculate_volatility_state()
        expected_impact = self._estimate_market_impact()
        hidden_liquidity = self._estimate_hidden_liquidity()
        
        # Update state object
        self.current_state = MicrostructureState(
            liquidity_state=liquidity_state,
            spread_state=normalized_spread,
            order_book_imbalance=order_book_imbalance,
            flow_imbalance=self.order_flow_imbalance,
            volatility_state=volatility_state,
            expected_market_impact=expected_impact,
            hidden_liquidity_estimate=hidden_liquidity,
            timestamp=time.time()
        )
        
        # Store in history
        self.state_history.append(self.current_state)
        
        # Detect patterns
        self.patterns = self._detect_microstructure_patterns()
        
        self.logger.debug(
            f"Updated microstructure state: liquidity={liquidity_state.name}, "
            f"spread={normalized_spread:.2f}, book_imbalance={order_book_imbalance:.2f}, "
            f"flow_imbalance={self.order_flow_imbalance:.2f}, volatility={volatility_state:.2f}"
        )

    def get_current_state(self) -> Optional[MicrostructureState]:
        """
        Get the current microstructure state.
        
        Returns:
            The current MicrostructureState or None if not available
        """
        return self.current_state

    def get_execution_advantage(self) -> Dict[str, Any]:
        """
        Get execution advantage recommendations based on current microstructure.
        
        Returns:
            Dictionary with execution advantage recommendations
        """
        if not self.current_state:
            return {"advantage": "none", "confidence": 0.0}
            
        advantages = []
        
        # Check for iceberg orders
        if self.patterns.get("iceberg_orders", 0) > 0.7:
            advantages.append({
                "type": "iceberg_orders",
                "action": "place_limit_at_iceberg_level",
                "confidence": self.patterns["iceberg_orders"]
            })
            
        # Check for strong order book imbalance
        if abs(self.current_state.order_book_imbalance) > 0.7:
            direction = "buy" if self.current_state.order_book_imbalance > 0 else "sell"
            advantages.append({
                "type": "book_imbalance",
                "action": f"aggressive_{direction}",
                "confidence": abs(self.current_state.order_book_imbalance)
            })
            
        # Check for low liquidity opportunity
        if self.current_state.liquidity_state in [LiquidityState.VERY_THIN, LiquidityState.THIN]:
            advantages.append({
                "type": "thin_liquidity",
                "action": "split_orders",
                "confidence": 0.9 if self.current_state.liquidity_state == LiquidityState.VERY_THIN else 0.7
            })
            
        # Check for wide spread opportunity
        if self.current_state.spread_state > 1.5:
            advantages.append({
                "type": "wide_spread",
                "action": "place_passive_orders",
                "confidence": min(1.0, (self.current_state.spread_state - 1.0) / 2.0)
            })
            
        # Select best advantage
        if advantages:
            return max(advantages, key=lambda x: x["confidence"])
        else:
            return {"advantage": "none", "confidence": 0.0}

    def estimate_optimal_execution_params(self, order_size: float, side: str) -> Dict[str, Any]:
        """
        Estimate optimal execution parameters based on current microstructure.
        
        Args:
            order_size: Size of the order to execute
            side: 'buy' or 'sell'
            
        Returns:
            Dictionary with optimal execution parameters
        """
        if not self.current_state:
            # Default conservative strategy
            return {
                "strategy": "twap",
                "num_slices": 5,
                "slice_interval_seconds": 60,
                "aggressive_factor": 0.3
            }
            
        # Assess market impact
        impact_level = None
        for size_threshold, level in sorted(self.current_state.expected_market_impact.items()):
            if order_size <= size_threshold:
                impact_level = level
                break
                
        if impact_level is None:
            impact_level = MarketImpactLevel.SEVERE
            
        # Determine execution strategy based on impact and market conditions
        if impact_level in [MarketImpactLevel.NEGLIGIBLE, MarketImpactLevel.LOW]:
            if abs(self.current_state.flow_imbalance) > 0.6:
                # Strong flow in one direction - act accordingly
                flow_direction = "buy" if self.current_state.flow_imbalance > 0 else "sell"
                if side == flow_direction:
                    # Going with the flow - be more aggressive
                    return {
                        "strategy": "aggressive",
                        "num_slices": 1,
                        "slice_interval_seconds": 0,
                        "aggressive_factor": 0.9
                    }
                else:
                    # Against the flow - be more passive
                    return {
                        "strategy": "iceberg",
                        "num_slices": 3,
                        "slice_interval_seconds": 30,
                        "aggressive_factor": 0.2
                    }
            else:
                # Balanced flow - standard approach
                return {
                    "strategy": "immediate",
                    "num_slices": 1,
                    "slice_interval_seconds": 0,
                    "aggressive_factor": 0.7
                }
                
        elif impact_level == MarketImpactLevel.MODERATE:
            # For moderate impact, use a sliced approach
            return {
                "strategy": "twap",
                "num_slices": 3,
                "slice_interval_seconds": 45,
                "aggressive_factor": 0.5
            }
            
        elif impact_level == MarketImpactLevel.HIGH:
            # For high impact, use more slices over longer time
            return {
                "strategy": "twap",
                "num_slices": 5,
                "slice_interval_seconds": 60,
                "aggressive_factor": 0.4
            }
            
        else:  # SEVERE
            # For severe impact, use many slices and more passive execution
            return {
                "strategy": "twap",
                "num_slices": 10,
                "slice_interval_seconds": 90,
                "aggressive_factor": 0.2
            }

    def adjust_order_params(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust order parameters based on current microstructure.
        
        Args:
            order_params: Original order parameters
            
        Returns:
            Adjusted order parameters
        """
        if not self.current_state:
            return order_params
            
        adjusted = order_params.copy()
        
        # Adjust based on spread
        if self.current_state.spread_state > 1.5:
            # Wider spread than usual - adjust limit price to be more conservative
            if "price" in adjusted and adjusted.get("type") == "limit":
                # For buys, lower the price, for sells, increase the price
                direction = -1 if adjusted.get("side") == "buy" else 1
                adjusted["price"] *= (1 + direction * 0.001 * (self.current_state.spread_state - 1.0))
                
        # Adjust based on volatility
        if self.current_state.volatility_state > 0.7:
            # Higher volatility - widen stops and take profits
            if "stop_price" in adjusted:
                direction = -1 if adjusted.get("side") == "buy" else 1
                adjusted["stop_price"] *= (1 + direction * 0.002 * self.current_state.volatility_state)
                
            if "take_profit_price" in adjusted:
                direction = 1 if adjusted.get("side") == "buy" else -1
                adjusted["take_profit_price"] *= (1 + direction * 0.002 * self.current_state.volatility_state)
                
        # Adjust based on liquidity
        if self.current_state.liquidity_state in [LiquidityState.VERY_THIN, LiquidityState.THIN]:
            # Low liquidity - reduce order size if possible
            if "quantity" in adjusted:
                liquidity_factor = 0.7 if self.current_state.liquidity_state == LiquidityState.VERY_THIN else 0.85
                adjusted["quantity"] *= liquidity_factor
                adjusted["quantity"] = max(adjusted["quantity"], self.tick_size)  # Ensure minimum size
                
        return adjusted

    def check_for_manipulation(self) -> Dict[str, float]:
        """
        Check for potential market manipulation patterns.
        
        Returns:
            Dictionary of manipulation types to confidence values (0.0-1.0)
        """
        manipulation = {}
        
        # Check for various manipulation patterns
        manipulation["spoofing"] = self.patterns.get("spoofing", 0.0)
        manipulation["layering"] = self.patterns.get("layering", 0.0)
        manipulation["momentum_ignition"] = self.patterns.get("momentum_ignition", 0.0)
        manipulation["quote_stuffing"] = self.patterns.get("quote_stuffing", 0.0)
        
        # Log high-confidence manipulation
        for pattern, confidence in manipulation.items():
            if confidence > 0.7:
                self.logger.warning(f"Potential {pattern} detected with {confidence:.2f} confidence")
                
        return manipulation

    def get_recommendations(self) -> Dict[str, Any]:
        """
        Get comprehensive microstructure-based recommendations.
        
        Returns:
            Dictionary with market microstructure recommendations
        """
        if not self.current_state:
            return {"status": "insufficient_data"}
            
        recommendations = {
            "timestamp": time.time(),
            "market_state": {
                "liquidity": self.current_state.liquidity_state.name,
                "spread": self.current_state.spread_state,
                "order_book_imbalance": self.current_state.order_book_imbalance,
                "flow_imbalance": self.current_state.flow_imbalance,
                "volatility": self.current_state.volatility_state,
                "hidden_liquidity": self.current_state.hidden_liquidity_estimate
            },
            "patterns": self.patterns,
            "manipulation_risk": self.check_for_manipulation(),
            "execution_advantage": self.get_execution_advantage(),
            "passive_order_recommendation": {},
            "aggressive_order_recommendation": {}
        }
        
        # Add passive order recommendation
        mid_price = self._calculate_mid_price()
        if mid_price:
            if self.current_state.order_book_imbalance > 0.3:
                # More bids than asks - buying pressure
                recommendations["passive_order_recommendation"] = {
                    "side": "sell",
                    "price_factor": 1.0 + min(0.005, self.current_state.order_book_imbalance * 0.001 * 
                                             (1 + self.current_state.volatility_state)),
                    "confidence": min(1.0, abs(self.current_state.order_book_imbalance) + 0.3)
                }
            elif self.current_state.order_book_imbalance < -0.3:
                # More asks than bids - selling pressure
                recommendations["passive_order_recommendation"] = {
                    "side": "buy",
                    "price_factor": 1.0 - min(0.005, abs(self.current_state.order_book_imbalance) * 0.001 *
                                             (1 + self.current_state.volatility_state)),
                    "confidence": min(1.0, abs(self.current_state.order_book_imbalance) + 0.3)
                }
            else:
                # Balanced book
                recommendations["passive_order_recommendation"] = {
                    "side": "neutral",
                    "price_factor": 1.0,
                    "confidence": 0.5
                }
                
        # Add aggressive order recommendation
        if abs(self.current_state.flow_imbalance) > 0.4:
            # Strong flow imbalance suggests momentum
            side = "buy" if self.current_state.flow_imbalance > 0 else "sell"
            recommendations["aggressive_order_recommendation"] = {
                "side": side,
                "urgency": min(1.0, abs(self.current_state.flow_imbalance) * 1.5),
                "confidence": min(1.0, abs(self.current_state.flow_imbalance) + 0.2)
            }
        else:
            recommendations["aggressive_order_recommendation"] = {
                "side": "neutral",
                "urgency": 0.0,
                "confidence": 0.5
            }
            
        return recommendations

    async def run_analysis_loop(self, interval: float = 1.0) -> None:
        """
        Run a continuous microstructure analysis loop.
        
        Args:
            interval: Analysis update interval in seconds
        """
        self.logger.info(f"Starting microstructure analysis loop with {interval}s interval")
        
        try:
            while True:
                # Update the state (this includes pattern detection)
                self._update_state()
                
                # Check for manipulation patterns
                manipulation = self.check_for_manipulation()
                
                # Log any high-confidence manipulation
                for pattern, confidence in manipulation.items():
                    if confidence > 0.8:
                        self.logger.warning(
                            f"High confidence {pattern} detected: {confidence:.2f}"
                        )
                
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            self.logger.info("Microstructure analysis loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in microstructure analysis loop: {e}")
            raise MicrostructureAnalysisError(f"Analysis loop failed: {str(e)}")
