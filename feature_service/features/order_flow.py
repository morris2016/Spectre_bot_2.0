#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Order Flow Analysis Features Module

This module provides advanced order flow analysis features for identifying volume
pressure, order book imbalances, and market microstructure patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
from scipy.stats import linregress
from collections import deque

from common.utils import rolling_apply, safe_divide, exponential_decay
from common.constants import TIME_FRAMES
from feature_service.features import register_feature
from feature_service.features.base_feature import BaseFeature
logger = logging.getLogger(__name__)

@dataclass
class OrderBookSnapshot:
    """Data class for order book snapshot at a particular time"""
    timestamp: int
    bids: List[Tuple[float, float]]  # price, volume pairs
    asks: List[Tuple[float, float]]  # price, volume pairs
    last_trade_price: float
    last_trade_volume: float


class OrderFlowFeatures(BaseFeature):
    """
    Advanced order flow analysis features for identifying market microstructure patterns.
    
    These features provide deep insights into the buying and selling pressure in the market
    by analyzing order book dynamics, trade flow, and market microstructure.
    """
    
    def __init__(self):
        super().__init__()
        # Historical snapshots for calculating changes over time
        self.order_book_history = deque(maxlen=1000)
        self.trade_history = deque(maxlen=5000)
        # Add this method to satisfy the abstract method requirement
    def calculate(self, data):
        """
        Calculate all order flow features.
        
        Args:
            data: Input data containing order book and trade information
            
        Returns:
            Dictionary with all calculated order flow features
        """
        # Process incoming data
        if 'order_book' in data:
            order_book = data['order_book']
            self.update_data(order_book, data.get('trades', []))
        
        # Calculate and return all features
        results = {
            "voi": self.calculate_volume_order_imbalance(),
            "pvi": self.calculate_price_volume_imbalance(),
            "liquidity_density": self.calculate_liquidity_density(),
            "bid_ask_imbalance": self.calculate_bid_ask_imbalance(),
            "market_pressure": self.calculate_market_pressure(),
            "absorption_ratio": self.calculate_absorption_ratio(),
            "smart_money_index": self.calculate_smart_money_index(),
            "whale_activity": self.calculate_whale_activity(),
            "liquidity_replenishment": self.calculate_liquidity_replenishment(),
            "institutional_activity": self.calculate_institutional_activity(),
            "spoofing_probability": self.detect_spoofing(),
            "exhaustion_move_probability": self.detect_exhaustion_moves(),
            "stop_hunt_probability": self.calculate_stop_hunt_probability()
        }
        
        # Add liquidity zones as separate features
        liquidity_zones = self.identify_liquidity_zones()
        for i, zone in enumerate(liquidity_zones):
            results[f"liquidity_zone_{i+1}"] = zone
            
        return results
        # Register all available features
        self._register_features()
        
    def _register_features(self):
        """Register all order flow features"""
        register_feature("order_flow.voi", self.calculate_volume_order_imbalance)
        register_feature("order_flow.pvi", self.calculate_price_volume_imbalance)
        register_feature("order_flow.liquidity_density", self.calculate_liquidity_density)
        register_feature("order_flow.bid_ask_imbalance", self.calculate_bid_ask_imbalance)
        register_feature("order_flow.market_pressure", self.calculate_market_pressure)
        register_feature("order_flow.absorption_ratio", self.calculate_absorption_ratio)
        register_feature("order_flow.smart_money_index", self.calculate_smart_money_index)
        register_feature("order_flow.whale_activity", self.calculate_whale_activity)
        register_feature("order_flow.liquidity_replenishment", self.calculate_liquidity_replenishment)
        register_feature("order_flow.price_impact", self.calculate_price_impact)
        register_feature("order_flow.institutional_activity", self.calculate_institutional_activity)
        register_feature("order_flow.spoofing_detection", self.detect_spoofing)
        register_feature("order_flow.liquidity_zones", self.identify_liquidity_zones)
        register_feature("order_flow.exhaustion_moves", self.detect_exhaustion_moves)
        register_feature("order_flow.stop_hunt_probability", self.calculate_stop_hunt_probability)
        
    def update_data(self, order_book_snapshot: OrderBookSnapshot, trades: List[Dict]) -> None:
        """
        Update internal state with new order book snapshot and trades.
        
        Args:
            order_book_snapshot: Current order book snapshot
            trades: List of recent trades
        """
        self.order_book_history.append(order_book_snapshot)
        for trade in trades:
            self.trade_history.append(trade)
        
    def calculate_volume_order_imbalance(self, depth: int = 10) -> float:
        """
        Calculate the Volume Order Imbalance (VOI) using order book data.
        
        VOI measures the imbalance between buy and sell orders in the market
        and can be an early indicator of price movements.
        
        Args:
            depth: Number of price levels to consider
            
        Returns:
            Volume Order Imbalance value between -1 and 1
        """
        if not self.order_book_history:
            return 0.0
        
        current_book = self.order_book_history[-1]
        
        # Limit to specified depth
        bids = current_book.bids[:depth] if len(current_book.bids) >= depth else current_book.bids
        asks = current_book.asks[:depth] if len(current_book.asks) >= depth else current_book.asks
        
        # Calculate bid and ask volumes
        bid_volume = sum(volume for _, volume in bids)
        ask_volume = sum(volume for _, volume in asks)
        
        # Calculate VOI
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            return (bid_volume - ask_volume) / total_volume
        return 0.0
    
    def calculate_price_volume_imbalance(self, depth: int = 10, decay_factor: float = 0.9) -> float:
        """
        Calculate Price-Volume Imbalance (PVI) which weights order imbalance by price distance.
        
        PVI gives more importance to orders closer to the current market price, providing
        a more nuanced view of order book pressure.
        
        Args:
            depth: Number of price levels to consider
            decay_factor: Weight decay factor for orders further from market price
            
        Returns:
            Price-Volume Imbalance value
        """
        if not self.order_book_history:
            return 0.0
        
        current_book = self.order_book_history[-1]
        mid_price = (current_book.bids[0][0] + current_book.asks[0][0]) / 2
        
        # Limit to specified depth
        bids = current_book.bids[:depth] if len(current_book.bids) >= depth else current_book.bids
        asks = current_book.asks[:depth] if len(current_book.asks) >= depth else current_book.asks
        
        # Calculate weighted bid and ask volumes
        weighted_bid_volume = 0
        for i, (price, volume) in enumerate(bids):
            price_distance = mid_price - price
            weight = decay_factor ** i  # Exponential decay by level
            weighted_bid_volume += volume * weight
            
        weighted_ask_volume = 0
        for i, (price, volume) in enumerate(asks):
            price_distance = price - mid_price
            weight = decay_factor ** i  # Exponential decay by level
            weighted_ask_volume += volume * weight
            
        # Calculate PVI
        total_weighted_volume = weighted_bid_volume + weighted_ask_volume
        if total_weighted_volume > 0:
            return (weighted_bid_volume - weighted_ask_volume) / total_weighted_volume
        return 0.0
    
    def calculate_liquidity_density(self, price_range_pct: float = 0.005) -> Dict[str, float]:
        """
        Calculate the density of liquidity at different price levels.
        
        High liquidity density areas often act as support/resistance levels.
        
        Args:
            price_range_pct: Price range percentage to consider
            
        Returns:
            Dictionary with liquidity density metrics
        """
        if not self.order_book_history:
            return {"bid_density": 0.0, "ask_density": 0.0, "total_density": 0.0}
        
        current_book = self.order_book_history[-1]
        mid_price = (current_book.bids[0][0] + current_book.asks[0][0]) / 2
        price_range = mid_price * price_range_pct
        
        # Calculate bid density
        bid_volume_in_range = 0
        for price, volume in current_book.bids:
            if mid_price - price <= price_range:
                bid_volume_in_range += volume
                
        # Calculate ask density
        ask_volume_in_range = 0
        for price, volume in current_book.asks:
            if price - mid_price <= price_range:
                ask_volume_in_range += volume
                
        total_volume_in_range = bid_volume_in_range + ask_volume_in_range
        
        return {
            "bid_density": bid_volume_in_range / price_range if price_range > 0 else 0,
            "ask_density": ask_volume_in_range / price_range if price_range > 0 else 0,
            "total_density": total_volume_in_range / price_range if price_range > 0 else 0
        }
    
    def calculate_bid_ask_imbalance(self, depth: int = 10, exponential_weights: bool = True) -> float:
        """
        Calculate bid-ask imbalance using bid and ask volume ratios.
        
        This feature can identify potential price direction based on order book imbalances.
        
        Args:
            depth: Number of price levels to consider
            exponential_weights: Whether to use exponential weighting by level
            
        Returns:
            Bid-ask imbalance ratio
        """
        if not self.order_book_history:
            return 0.0
        
        current_book = self.order_book_history[-1]
        
        # Limit to specified depth
        bids = current_book.bids[:depth] if len(current_book.bids) >= depth else current_book.bids
        asks = current_book.asks[:depth] if len(current_book.asks) >= depth else current_book.asks
        
        if exponential_weights:
            # Apply exponential weights to give more importance to prices closer to mid
            bid_volume = sum(volume * (0.9 ** i) for i, (_, volume) in enumerate(bids))
            ask_volume = sum(volume * (0.9 ** i) for i, (_, volume) in enumerate(asks))
        else:
            bid_volume = sum(volume for _, volume in bids)
            ask_volume = sum(volume for _, volume in asks)
            
        return safe_divide(bid_volume, ask_volume, 1.0) - 1.0
    
    def calculate_market_pressure(self, window: int = 100) -> float:
        """
        Calculate market pressure based on recent trade flow.
        
        This indicates buying or selling pressure based on trade direction and volume.
        
        Args:
            window: Number of recent trades to consider
            
        Returns:
            Market pressure indicator (-1 to 1)
        """
        if len(self.trade_history) < 10:
            return 0.0
        
        # Get recent trades up to window size
        recent_trades = list(self.trade_history)[-window:]
        
        # Calculate buy and sell volumes
        buy_volume = sum(trade['volume'] for trade in recent_trades if trade.get('side') == 'buy')
        sell_volume = sum(trade['volume'] for trade in recent_trades if trade.get('side') == 'sell')
        
        # Calculate market pressure
        total_volume = buy_volume + sell_volume
        if total_volume > 0:
            return (buy_volume - sell_volume) / total_volume
        return 0.0
    
    def calculate_absorption_ratio(self, window: int = 20) -> float:
        """
        Calculate order absorption ratio - how quickly limit orders are being filled.
        
        High absorption ratio indicates aggressive buying or selling that can predict price movement.
        
        Args:
            window: Number of order book snapshots to consider
            
        Returns:
            Absorption ratio
        """
        if len(self.order_book_history) < window:
            return 0.0
        
        # Get relevant history
        history = list(self.order_book_history)[-window:]
        
        # Calculate changes in order book
        changes = []
        for i in range(1, len(history)):
            prev_book = history[i-1]
            curr_book = history[i]
            
            # Compare top of book levels
            if len(prev_book.bids) > 0 and len(curr_book.bids) > 0:
                prev_top_bid_vol = prev_book.bids[0][1]
                curr_top_bid_vol = curr_book.bids[0][1]
                bid_change = prev_top_bid_vol - curr_top_bid_vol
                changes.append((bid_change, "bid"))
                
            if len(prev_book.asks) > 0 and len(curr_book.asks) > 0:
                prev_top_ask_vol = prev_book.asks[0][1]
                curr_top_ask_vol = curr_book.asks[0][1]
                ask_change = prev_top_ask_vol - curr_top_ask_vol
                changes.append((ask_change, "ask"))
        
        # Calculate positive changes (order absorption)
        absorbed_volume = sum(change for change, _ in changes if change > 0)
        total_change = sum(abs(change) for change, _ in changes)
        
        return safe_divide(absorbed_volume, total_change, 0.5)
    
    def calculate_smart_money_index(self, window: int = 50) -> float:
        """
        Calculate a smart money index based on trade size and timing.
        
        Helps identify institutional activity vs retail activity.
        
        Args:
            window: Number of trades to analyze
            
        Returns:
            Smart money index value
        """
        if len(self.trade_history) < window:
            return 0.0
        
        recent_trades = list(self.trade_history)[-window:]
        
        # Identify large trades (potential institutional activity)
        volumes = [trade['volume'] for trade in recent_trades]
        if not volumes:
            return 0.0
            
        median_volume = np.median(volumes)
        large_trade_threshold = median_volume * 5  # 5x median is considered large
        
        # Identify large trades and their direction
        large_buys = sum(
            trade['volume'] for trade in recent_trades 
            if trade.get('side') == 'buy' and trade['volume'] >= large_trade_threshold
        )
        large_sells = sum(
            trade['volume'] for trade in recent_trades 
            if trade.get('side') == 'sell' and trade['volume'] >= large_trade_threshold
        )
        
        # Calculate smaller retail trades
        small_buys = sum(
            trade['volume'] for trade in recent_trades 
            if trade.get('side') == 'buy' and trade['volume'] < large_trade_threshold
        )
        small_sells = sum(
            trade['volume'] for trade in recent_trades 
            if trade.get('side') == 'sell' and trade['volume'] < large_trade_threshold
        )
        
        # Smart money index combines institutional vs retail behavior
        large_imbalance = safe_divide(large_buys - large_sells, large_buys + large_sells, 0)
        small_imbalance = safe_divide(small_buys - small_sells, small_buys + small_sells, 0)
        
        # Smart money often moves counter to retail sentiment
        return large_imbalance - 0.5 * small_imbalance
    
    def calculate_whale_activity(self, percentile_threshold: float = 95) -> Dict[str, float]:
        """
        Detect and quantify whale activity in the market.
        
        Whales are large traders that can significantly move the market.
        
        Args:
            percentile_threshold: Percentile threshold for whale trade detection
            
        Returns:
            Dictionary with whale activity metrics
        """
        if len(self.trade_history) < 50:
            return {"whale_activity": 0.0, "whale_buy_pressure": 0.0, "whale_sell_pressure": 0.0}
        
        # Get recent trade volumes
        volumes = [trade['volume'] for trade in self.trade_history]
        whale_threshold = np.percentile(volumes, percentile_threshold)
        
        recent_trades = list(self.trade_history)[-200:]
        
        # Identify whale trades
        whale_trades = [
            trade for trade in recent_trades 
            if trade['volume'] >= whale_threshold
        ]
        
        if not whale_trades:
            return {"whale_activity": 0.0, "whale_buy_pressure": 0.0, "whale_sell_pressure": 0.0}
            
        # Calculate whale buy and sell volumes
        whale_buy_volume = sum(
            trade['volume'] for trade in whale_trades 
            if trade.get('side') == 'buy'
        )
        whale_sell_volume = sum(
            trade['volume'] for trade in whale_trades 
            if trade.get('side') == 'sell'
        )
        
        total_whale_volume = whale_buy_volume + whale_sell_volume
        whale_activity = len(whale_trades) / len(recent_trades)
        
        return {
            "whale_activity": whale_activity,
            "whale_buy_pressure": safe_divide(whale_buy_volume, total_whale_volume, 0.5),
            "whale_sell_pressure": safe_divide(whale_sell_volume, total_whale_volume, 0.5)
        }
    
    def calculate_liquidity_replenishment(self, depth: int = 5, window: int = 10) -> float:
        """
        Calculate how quickly liquidity is replenished after being absorbed.
        
        Fast replenishment can indicate strong support/resistance levels.
        
        Args:
            depth: Depth of order book to consider
            window: Number of snapshots to analyze
            
        Returns:
            Liquidity replenishment rate
        """
        if len(self.order_book_history) < window + 1:
            return 0.0
        
        history = list(self.order_book_history)[-(window+1):]
        replenishment_rates = []
        
        for i in range(1, len(history)):
            prev_book = history[i-1]
            curr_book = history[i]
            
            # Calculate liquidity at specified depth
            prev_bid_liquidity = sum(vol for _, vol in prev_book.bids[:depth])
            curr_bid_liquidity = sum(vol for _, vol in curr_book.bids[:depth])
            
            prev_ask_liquidity = sum(vol for _, vol in prev_book.asks[:depth])
            curr_ask_liquidity = sum(vol for _, vol in curr_book.asks[:depth])
            
            # Calculate net changes
            bid_change = curr_bid_liquidity - prev_bid_liquidity
            ask_change = curr_ask_liquidity - prev_ask_liquidity
            
            # Only consider replenishment (positive changes)
            if bid_change > 0:
                replenishment_rates.append(bid_change / prev_bid_liquidity if prev_bid_liquidity > 0 else 0)
            if ask_change > 0:
                replenishment_rates.append(ask_change / prev_ask_liquidity if prev_ask_liquidity > 0 else 0)
        
        return np.mean(replenishment_rates) if replenishment_rates else 0.0
    
    def calculate_price_impact(self, volume: float) -> float:
        """
        Estimate the price impact of a market order with the given volume.
        
        Helps determine market depth and liquidity.
        
        Args:
            volume: Volume of the hypothetical market order
            
        Returns:
            Estimated price impact as a percentage
        """
        if not self.order_book_history:
            return 0.0
        
        current_book = self.order_book_history[-1]
        mid_price = (current_book.bids[0][0] + current_book.asks[0][0]) / 2
        
        # Simulate market buy order
        remaining_volume = volume
        cost = 0
        
        for price, available_volume in current_book.asks:
            if remaining_volume <= 0:
                break
                
            volume_to_execute = min(remaining_volume, available_volume)
            cost += volume_to_execute * price
            remaining_volume -= volume_to_execute
            
        # Calculate average execution price
        if volume - remaining_volume > 0:
            avg_price = cost / (volume - remaining_volume)
            return (avg_price - mid_price) / mid_price
        return 0.0
    
    def calculate_institutional_activity(self, time_window: int = 3600) -> float:
        """
        Estimate institutional activity based on trade patterns and order book changes.
        
        Institutions often leave specific footprints in the market data.
        
        Args:
            time_window: Time window in seconds to analyze
            
        Returns:
            Institutional activity score (0-1)
        """
        if len(self.trade_history) < 50:
            return 0.0
        
        recent_trades = list(self.trade_history)[-200:]
        
        # Features that suggest institutional activity
        # 1. Large trade sizes
        volumes = [trade['volume'] for trade in recent_trades]
        large_trade_threshold = np.percentile(volumes, 90)
        large_trade_ratio = sum(1 for vol in volumes if vol >= large_trade_threshold) / len(volumes)
        
        # 2. Consistent buying/selling direction
        directions = [1 if trade.get('side') == 'buy' else -1 for trade in recent_trades]
        direction_consistency = abs(sum(directions)) / len(directions)
        
        # 3. Timing patterns (regular intervals)
        if len(recent_trades) >= 20:
            timestamps = [trade.get('timestamp', 0) for trade in recent_trades[-20:]]
            intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
            interval_std = np.std(intervals) if intervals else 0
            normalized_std = min(1.0, 1.0 / (1.0 + interval_std / 1000))  # Lower std = more regular
        else:
            normalized_std = 0.5
        
        # Combine signals (weighted average)
        inst_activity = (0.4 * large_trade_ratio + 
                         0.4 * direction_consistency + 
                         0.2 * normalized_std)
        
        return min(1.0, max(0.0, inst_activity))
    
    def detect_spoofing(self) -> float:
        """
        Detect potential spoofing activity in the order book.
        
        Spoofing involves placing and quickly canceling large orders to manipulate the market.
        
        Returns:
            Spoofing probability score (0-1)
        """
        if len(self.order_book_history) < 10:
            return 0.0
        
        recent_books = list(self.order_book_history)[-10:]
        spoofing_indicators = []
        
        for i in range(1, len(recent_books)):
            prev_book = recent_books[i-1]
            curr_book = recent_books[i]
            
            # Look for large orders that disappear without being executed
            # Bid side
            for prev_price, prev_vol in prev_book.bids:
                # Find matching price level in current book
                matching_bids = [(p, v) for p, v in curr_book.bids if abs(p - prev_price) < 1e-6]
                
                if not matching_bids:
                    # Order completely disappeared
                    if prev_vol > np.percentile([v for _, v in prev_book.bids], 90):
                        # Large order disappeared
                        spoofing_indicators.append(1.0)
                elif prev_vol > matching_bids[0][1]:
                    # Order size decreased without price movement
                    vol_change = prev_vol - matching_bids[0][1]
                    if vol_change > np.percentile([v for _, v in prev_book.bids], 80):
                        # Significant decrease
                        spoofing_indicators.append(0.7)
            
            # Ask side (similar logic)
            for prev_price, prev_vol in prev_book.asks:
                matching_asks = [(p, v) for p, v in curr_book.asks if abs(p - prev_price) < 1e-6]
                
                if not matching_asks:
                    if prev_vol > np.percentile([v for _, v in prev_book.asks], 90):
                        spoofing_indicators.append(1.0)
                elif prev_vol > matching_asks[0][1]:
                    vol_change = prev_vol - matching_asks[0][1]
                    if vol_change > np.percentile([v for _, v in prev_book.asks], 80):
                        spoofing_indicators.append(0.7)
        
        return np.mean(spoofing_indicators) if spoofing_indicators else 0.0
    
    def identify_liquidity_zones(self, num_zones: int = 3) -> List[Dict[str, Any]]:
        """
        Identify key liquidity zones from order book data.
        
        These zones often act as important support/resistance levels.
        
        Args:
            num_zones: Number of liquidity zones to identify
            
        Returns:
            List of dictionaries with liquidity zone information
        """
        if not self.order_book_history:
            return []
        
        current_book = self.order_book_history[-1]
        
        # Combine all price levels
        all_levels = []
        
        for price, volume in current_book.bids:
            all_levels.append({"price": price, "volume": volume, "side": "bid"})
            
        for price, volume in current_book.asks:
            all_levels.append({"price": price, "volume": volume, "side": "ask"})
            
        # No levels available
        if not all_levels:
            return []
            
        # Sort by volume (descending)
        all_levels.sort(key=lambda x: x["volume"], reverse=True)
        
        # Take top levels as liquidity zones
        top_levels = all_levels[:num_zones]
        
        # Format as liquidity zones
        liquidity_zones = []
        for level in top_levels:
            zone_type = "support" if level["side"] == "bid" else "resistance"
            strength = level["volume"] / max(x["volume"] for x in all_levels)
            
            liquidity_zones.append({
                "price": level["price"],
                "volume": level["volume"],
                "type": zone_type,
                "strength": strength
            })
            
        return liquidity_zones
    
    def detect_exhaustion_moves(self, volume_threshold: float = 3.0) -> float:
        """
        Detect potential exhaustion moves based on volume and price patterns.
        
        Exhaustion moves often indicate trend reversals.
        
        Args:
            volume_threshold: Volume multiple threshold compared to average
            
        Returns:
            Exhaustion move probability (0-1)
        """
        if len(self.trade_history) < 50:
            return 0.0
        
        recent_trades = list(self.trade_history)[-50:]
        
        # Calculate average volume
        volumes = [trade['volume'] for trade in recent_trades]
        avg_volume = np.mean(volumes)
        
        # Look for recent high volume trades
        last_10_trades = recent_trades[-10:]
        high_volume_trades = [
            trade for trade in last_10_trades 
            if trade['volume'] > avg_volume * volume_threshold
        ]
        
        if not high_volume_trades:
            return 0.0
            
        # Check if these high volume trades have consistent direction
        if len(high_volume_trades) >= 3:
            buy_count = sum(1 for trade in high_volume_trades if trade.get('side') == 'buy')
            sell_count = sum(1 for trade in high_volume_trades if trade.get('side') == 'sell')
            
            direction_consistency = max(buy_count, sell_count) / len(high_volume_trades)
            volume_ratio = sum(trade['volume'] for trade in high_volume_trades) / (avg_volume * len(high_volume_trades))
            
            # Combine signals
            exhaustion_score = direction_consistency * min(1.0, volume_ratio / 5.0)
            return exhaustion_score
        
        return 0.0
    
    def calculate_stop_hunt_probability(self) -> float:
        """
        Calculate the probability of a stop hunt based on price action and volume patterns.
        
        Stop hunts are characterized by sharp price movements followed by reversals.
        
        Returns:
            Stop hunt probability (0-1)
        """
        if len(self.trade_history) < 100 or len(self.order_book_history) < 10:
            return 0.0
        
        recent_trades = list(self.trade_history)[-100:]
        recent_books = list(self.order_book_history)[-10:]
        
        # Look for recent price spike in one direction
        prices = [trade.get('price', 0) for trade in recent_trades]
        returns = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
        
        # Check for significant moves
        if len(returns) < 20:
            return 0.0
            
        recent_returns = returns[-20:]
        max_up_move = max(recent_returns)
        max_down_move = abs(min(recent_returns))
        
        # Look for significant move
        significant_move = max(max_up_move, max_down_move) > 0.003  # 0.3% threshold
        
        if not significant_move:
            return 0.0
            
        # Check for direction reversal
        if max_up_move > max_down_move:
            # Upward spike, look for selling after
            direction = "up"
            recent_direction = sum(1 for r in recent_returns[-5:] if r < 0) > 2  # Mostly down after spike
        else:
            # Downward spike, look for buying after
            direction = "down"
            recent_direction = sum(1 for r in recent_returns[-5:] if r > 0) > 2  # Mostly up after spike
            
        direction_reversal = recent_direction
        
        # Check for large orders just beyond the move
        order_book_signal = 0.0
        
        if direction == "up":
            # Check for large ask walls that disappeared
            if len(recent_books) >= 3:
                prev_asks = recent_books[-3].asks
                current_asks = recent_books[-1].asks
                
                if prev_asks and current_asks:
                    # Look for large asks that disappeared
                    prev_ask_volumes = {price: vol for price, vol in prev_asks}
                    current_ask_volumes = {price: vol for price, vol in current_asks}
                    
                    disappeared_volume = 0
                    for price, vol in prev_ask_volumes.items():
                        if price not in current_ask_volumes and vol > np.percentile([v for _, v in prev_asks], 80):
                            disappeared_volume += vol
                            
                    if disappeared_volume > 0:
                        order_book_signal = min(1.0, disappeared_volume / (prev_asks[0][1] * 10))
        else:
            # Check for large bid walls that disappeared
            if len(recent_books) >= 3:
                prev_bids = recent_books[-3].bids
                current_bids = recent_books[-1].bids
                
                if prev_bids and current_bids:
                    # Look for large bids that disappeared
                    prev_bid_volumes = {price: vol for price, vol in prev_bids}
                    current_bid_volumes = {price: vol for price, vol in current_bids}
                    
                    disappeared_volume = 0
                    for price, vol in prev_bid_volumes.items():
                        if price not in current_bid_volumes and vol > np.percentile([v for _, v in prev_bids], 80):
                            disappeared_volume += vol
                            
                    if disappeared_volume > 0:
                        order_book_signal = min(1.0, disappeared_volume / (prev_bids[0][1] * 10))
                        
        # Calculate final probability
        magnitude = max(max_up_move, max_down_move) / 0.01  # Normalize to 1 at 1% move
        stop_hunt_probability = (0.4 * min(1.0, magnitude) + 
                               0.4 * (1.0 if direction_reversal else 0.0) + 
                               0.2 * order_book_signal)
        
        return min(1.0, stop_hunt_probability)

class OrderFlowAnalyzer:
    """
    Advanced order flow analyzer for market microstructure analysis.
    
    This class extends the OrderFlowFeatures class with specialized methods
    for analyzing order flow patterns, detecting market microstructure anomalies,
    and identifying exploitable trading opportunities.
    """
    
    def __init__(self):
        """Initialize the OrderFlowAnalyzer."""
        self.features = OrderFlowFeatures()
        self.order_book_history = deque(maxlen=1000)
        self.trade_history = deque(maxlen=5000)
        self.logger = logging.getLogger(__name__)
        
    async def analyze_order_flow(self, data):
        """
        Analyze order flow data to detect patterns and anomalies.
        
        Args:
            data: Dictionary containing order book and trade data
            
        Returns:
            Dictionary of analysis results
        """
        # Update internal data
        if 'order_book' in data:
            self.order_book_history.append(data['order_book'])
        if 'trades' in data:
            for trade in data['trades']:
                self.trade_history.append(trade)
                
        # Calculate basic features
        features = self.features.calculate(data)
        
        # Perform additional analysis
        analysis = {
            'features': features,
            'imbalance_score': await self.calculate_imbalance_score(),
            'aggressive_flow': await self.detect_aggressive_flow(),
            'liquidity_zones': await self.identify_liquidity_zones(),
            'absorption_rate': await self.calculate_absorption_rate(),
            'spoofing_probability': await self.detect_spoofing(),
            'iceberg_detection': await self.detect_iceberg_orders(),
        }
        
        return analysis
        
    async def calculate_imbalance_score(self):
        """Calculate order book imbalance score."""
        if not self.order_book_history:
            return 0.0
            
        current_book = self.order_book_history[-1]
        
        # Calculate bid and ask volumes
        bid_volume = sum(vol for _, vol in current_book.bids[:5])
        ask_volume = sum(vol for _, vol in current_book.asks[:5])
        
        # Calculate imbalance
        total_volume = bid_volume + ask_volume
        if total_volume > 0:
            return (bid_volume - ask_volume) / total_volume
        return 0.0
        
    async def detect_aggressive_flow(self):
        """Detect aggressive buying or selling in recent trades."""
        if len(self.trade_history) < 50:
            return {'is_aggressive': False, 'direction': 'neutral', 'score': 0.0}
            
        recent_trades = list(self.trade_history)[-50:]
        
        # Calculate buy and sell volumes
        buy_volume = sum(trade.get('amount', 0) for trade in recent_trades if trade.get('side') == 'buy')
        sell_volume = sum(trade.get('amount', 0) for trade in recent_trades if trade.get('side') == 'sell')
        
        # Calculate aggressiveness
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return {'is_aggressive': False, 'direction': 'neutral', 'score': 0.0}
            
        imbalance = (buy_volume - sell_volume) / total_volume
        
        # Determine if flow is aggressive
        is_aggressive = abs(imbalance) > 0.7
        direction = 'buy' if imbalance > 0 else 'sell'
        
        return {
            'is_aggressive': is_aggressive,
            'direction': direction if is_aggressive else 'neutral',
            'score': abs(imbalance)
        }
        
    async def identify_liquidity_zones(self):
        """Identify key liquidity zones in the order book."""
        if not self.order_book_history:
            return []
            
        current_book = self.order_book_history[-1]
        
        # Find clusters of liquidity
        bid_clusters = self._find_liquidity_clusters(current_book.bids)
        ask_clusters = self._find_liquidity_clusters(current_book.asks)
        
        # Format results
        zones = []
        for price, volume in bid_clusters:
            zones.append({
                'price': price,
                'volume': volume,
                'side': 'bid',
                'strength': min(1.0, volume / 100)  # Normalize strength
            })
            
        for price, volume in ask_clusters:
            zones.append({
                'price': price,
                'volume': volume,
                'side': 'ask',
                'strength': min(1.0, volume / 100)  # Normalize strength
            })
            
        return sorted(zones, key=lambda x: x['volume'], reverse=True)[:5]  # Top 5 zones
        
    def _find_liquidity_clusters(self, levels):
        """Find clusters of liquidity in order book levels."""
        if not levels:
            return []
            
        clusters = []
        current_cluster = {'price': levels[0][0], 'volume': levels[0][1]}
        
        for i in range(1, len(levels)):
            price, volume = levels[i]
            prev_price = levels[i-1][0]
            
            # If prices are close, merge into cluster
            if abs(price - prev_price) / prev_price < 0.001:
                current_cluster['volume'] += volume
            else:
                # Save current cluster and start new one
                clusters.append((current_cluster['price'], current_cluster['volume']))
                current_cluster = {'price': price, 'volume': volume}
                
        # Add final cluster
        clusters.append((current_cluster['price'], current_cluster['volume']))
        
        return sorted(clusters, key=lambda x: x[1], reverse=True)
        
    async def calculate_absorption_rate(self):
        """Calculate how quickly limit orders are being absorbed."""
        if len(self.order_book_history) < 10:
            return 0.0
            
        # Get recent snapshots
        snapshots = list(self.order_book_history)[-10:]
        
        # Track changes in top of book
        changes = []
        for i in range(1, len(snapshots)):
            prev_book = snapshots[i-1]
            curr_book = snapshots[i]
            
            # Compare top of book levels
            if prev_book.bids and curr_book.bids:
                prev_top_bid_vol = prev_book.bids[0][1]
                curr_top_bid_vol = curr_book.bids[0][1]
                bid_change = prev_top_bid_vol - curr_top_bid_vol
                changes.append(bid_change)
                
            if prev_book.asks and curr_book.asks:
                prev_top_ask_vol = prev_book.asks[0][1]
                curr_top_ask_vol = curr_book.asks[0][1]
                ask_change = prev_top_ask_vol - curr_top_ask_vol
                changes.append(ask_change)
                
        # Calculate absorption rate
        if not changes:
            return 0.0
            
        positive_changes = sum(max(0, change) for change in changes)
        total_change = sum(abs(change) for change in changes)
        
        if total_change == 0:
            return 0.0
            
        return positive_changes / total_change
        
    async def detect_spoofing(self):
        """Detect potential spoofing activity in the order book."""
        if len(self.order_book_history) < 5:
            return {'detected': False, 'probability': 0.0, 'side': None}
            
        recent_books = list(self.order_book_history)[-5:]
        
        # Look for large orders that disappear without being executed
        vanishing_bids = self._detect_vanishing_orders(recent_books, 'bids')
        vanishing_asks = self._detect_vanishing_orders(recent_books, 'asks')
        
        # Calculate spoofing probability
        bid_probability = vanishing_bids['probability'] if vanishing_bids else 0.0
        ask_probability = vanishing_asks['probability'] if vanishing_asks else 0.0
        
        if bid_probability > 0.7 or ask_probability > 0.7:
            side = 'bid' if bid_probability > ask_probability else 'ask'
            probability = max(bid_probability, ask_probability)
            return {'detected': True, 'probability': probability, 'side': side}
            
        return {'detected': False, 'probability': 0.0, 'side': None}
        
    def _detect_vanishing_orders(self, snapshots, side):
        """Detect orders that appear and quickly disappear."""
        if len(snapshots) < 3:
            return None
            
        # Look for large orders that appear and then disappear
        for i in range(len(snapshots) - 2):
            book1 = snapshots[i]
            book2 = snapshots[i+1]
            book3 = snapshots[i+2]
            
            # Get order book levels
            levels1 = getattr(book1, side, [])
            levels2 = getattr(book2, side, [])
            levels3 = getattr(book3, side, [])
            
            if not levels1 or not levels2 or not levels3:
                continue
                
            # Look for large orders in book2 that aren't in book1 or book3
            for price2, volume2 in levels2:
                # Skip if volume is small
                if volume2 < 10:
                    continue
                    
                # Check if this large order appeared suddenly and disappeared quickly
                existed_before = any(abs(price1 - price2) < 1e-6 for price1, _ in levels1)
                exists_after = any(abs(price3 - price2) < 1e-6 for price3, _ in levels3)
                
                if not existed_before and not exists_after:
                    # Calculate probability based on size relative to other orders
                    avg_size = sum(vol for _, vol in levels2) / len(levels2)
                    relative_size = volume2 / avg_size
                    probability = min(0.95, 0.5 + (relative_size - 1) / 10)
                    
                    return {
                        'price': price2,
                        'volume': volume2,
                        'probability': probability
                    }
                    
        return None
        
    async def detect_iceberg_orders(self):
        """Detect potential iceberg orders in the market."""
        if len(self.order_book_history) < 10 or len(self.trade_history) < 20:
            return {'detected': False, 'probability': 0.0, 'side': None}
            
        # Look for repeated fills at the same price level
        refills = self._detect_price_level_refills()
        
        if refills:
            return {
                'detected': True,
                'probability': refills['probability'],
                'side': refills['side'],
                'price': refills['price'],
                'estimated_size': refills['total_volume']
            }
            
        return {'detected': False, 'probability': 0.0, 'side': None}
        
    def _detect_price_level_refills(self):
        """Detect price levels that get repeatedly refilled after trades."""
        if len(self.order_book_history) < 5:
            return None
            
        recent_books = list(self.order_book_history)[-5:]
        
        # Track volumes at each price level
        bid_levels = defaultdict(list)
        ask_levels = defaultdict(list)
        
        for book in recent_books:
            for price, volume in book.bids:
                bid_levels[price].append(volume)
                
            for price, volume in book.asks:
                ask_levels[price].append(volume)
                
        # Look for levels with significant refills
        refill_candidates = []
        
        for side, levels in [('bid', bid_levels), ('ask', ask_levels)]:
            for price, volumes in levels.items():
                if len(volumes) < 3:
                    continue
                    
                # Check for pattern of decreasing then increasing
                decreases = sum(1 for i in range(1, len(volumes)) if volumes[i] < volumes[i-1])
                increases = sum(1 for i in range(1, len(volumes)) if volumes[i] > volumes[i-1])
                
                if decreases >= 1 and increases >= 1:
                    # Calculate refill ratio
                    total_decrease = sum(max(0, volumes[i-1] - volumes[i]) for i in range(1, len(volumes)))
                    total_increase = sum(max(0, volumes[i] - volumes[i-1]) for i in range(1, len(volumes)))
                    
                    if total_decrease > 0 and total_increase / total_decrease > 0.5:
                        refill_candidates.append({
                            'side': side,
                            'price': price,
                            'refill_ratio': total_increase / total_decrease,
                            'total_volume': sum(volumes),
                            'probability': min(0.9, 0.5 + (total_increase / total_decrease) / 2)
                        })
                        
        if refill_candidates:
            # Return the most likely candidate
            return max(refill_candidates, key=lambda x: x['probability'])
            
        return None

