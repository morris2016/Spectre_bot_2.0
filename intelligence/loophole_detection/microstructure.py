#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Market Microstructure Analysis and Exploitation Module

This module implements advanced market microstructure analysis to detect and exploit
short-term inefficiencies in order flow, price formation, and market mechanics.
It focuses on high-frequency patterns, order book imbalances, and execution anomalies
that can be leveraged for high-probability trades.
"""

import numpy as np
from typing import Any, Dict, List, Optional
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque, defaultdict

from common.logger import get_logger
from data_storage.market_data import MarketDataRepository
from feature_service.features.order_flow import OrderFlowAnalyzer
from feature_service.features.volume import VolumeProfileAnalyzer

logger = get_logger("intelligence.loophole_detection.microstructure")


@dataclass
class MicrostructureSignal:
    """Represents a microstructure-based trading signal."""
    timestamp: datetime
    symbol: str
    direction: str  # 'buy' or 'sell'
    signal_type: str
    confidence: float
    price_level: float
    expected_move: float
    time_validity: int  # in seconds
    metadata: Dict[str, Any]
    
    @property
    def is_valid(self) -> bool:
        """Check if the signal is still valid based on its creation time and validity period."""
        return datetime.utcnow() < self.timestamp + timedelta(seconds=self.time_validity)
    
    @property
    def time_to_expiry(self) -> int:
        """Calculate seconds until signal expiry."""
        if not self.is_valid:
            return 0
        return int((self.timestamp + timedelta(seconds=self.time_validity) - datetime.utcnow()).total_seconds())
    

class MicrostructureAnalyzer:
    """
    Advanced market microstructure analyzer for detecting exploitable market mechanics.
    
    This class focuses on order book dynamics, price formation mechanisms, and execution
    quirks that can be leveraged for high-probability short-term trades.
    """
    
    def __init__(
        self,
        market_data_repo: MarketDataRepository,
        order_flow_analyzer: OrderFlowAnalyzer,
        volume_analyzer: VolumeProfileAnalyzer,
        config: Dict[str, Any]
    ):
        """
        Initialize the MicrostructureAnalyzer.
        
        Args:
            market_data_repo: Repository for market data access
            order_flow_analyzer: Analyzer for order flow patterns
            volume_analyzer: Analyzer for volume profiles
            config: Configuration parameters for the analyzer
        """
        self.market_data_repo = market_data_repo
        self.order_flow_analyzer = order_flow_analyzer
        self.volume_analyzer = volume_analyzer
        self.config = config
        
        # Configuration parameters
        self.min_imbalance_ratio = config.get("min_imbalance_ratio", 3.0)
        self.iceberg_detection_threshold = config.get("iceberg_detection_threshold", 0.75)
        self.spoofing_detection_window = config.get("spoofing_detection_window", 5)
        self.liquidity_cliff_threshold = config.get("liquidity_cliff_threshold", 0.6)
        self.price_impact_threshold = config.get("price_impact_threshold", 0.0002)
        self.order_flow_momentum_window = config.get("order_flow_momentum_window", 20)
        self.tape_reading_window = config.get("tape_reading_window", 50)
        self.exhaust_volume_ratio = config.get("exhaust_volume_ratio", 2.5)
        self.liquidity_trap_threshold = config.get("liquidity_trap_threshold", 0.002)
        self.liquidity_trap_window = config.get("liquidity_trap_window", 20)
        self.trap_volume_ratio = config.get("trap_volume_ratio", 0.5)
        
        # State tracking
        self.order_book_snapshots = defaultdict(lambda: deque(maxlen=1000))
        self.trade_history = defaultdict(lambda: deque(maxlen=5000))
        self.detected_signals = defaultdict(list)
        self.active_signals = defaultdict(list)
        
        # Performance metrics
        self.signal_performance = defaultdict(lambda: {
            "total": 0,
            "successful": 0,
            "accuracy": 0.0,
            "avg_profit": 0.0,
            "avg_drawdown": 0.0
        })
        
        logger.info("MicrostructureAnalyzer initialized with configuration: %s", config)
    
    async def update_order_book(self, symbol: str, order_book_data: Dict[str, Any]) -> None:
        """
        Update the order book snapshots with new data.
        
        Args:
            symbol: The trading symbol
            order_book_data: The new order book data
        """
        self.order_book_snapshots[symbol].append({
            "timestamp": datetime.utcnow(),
            "data": order_book_data
        })
        
        # Perform real-time analytics on updated order book
        await self._analyze_order_book_update(symbol, order_book_data)
    
    async def update_trades(self, symbol: str, trades_data: List[Dict[str, Any]]) -> None:
        """
        Update the trade history with new trades.
        
        Args:
            symbol: The trading symbol
            trades_data: List of new trades
        """
        for trade in trades_data:
            self.trade_history[symbol].append({
                "timestamp": datetime.utcnow() if "timestamp" not in trade else trade["timestamp"],
                "data": trade
            })
        
        # Perform real-time analytics on new trades
        await self._analyze_trades_update(symbol, trades_data)
    
    async def analyze_microstructure(self, symbol: str) -> List[MicrostructureSignal]:
        """
        Perform comprehensive microstructure analysis for the given symbol.
        
        Args:
            symbol: The trading symbol to analyze
            
        Returns:
            List of MicrostructureSignal objects representing detected opportunities
        """
        signals = []
        
        # Perform different types of microstructure analysis
        order_book_signals = await self._detect_order_book_anomalies(symbol)
        trade_flow_signals = await self._detect_trade_flow_patterns(symbol)
        liquidity_signals = await self._detect_liquidity_anomalies(symbol)
        price_formation_signals = await self._detect_price_formation_quirks(symbol)
        
        # Combine all signals
        signals.extend(order_book_signals)
        signals.extend(trade_flow_signals)
        signals.extend(liquidity_signals)
        signals.extend(price_formation_signals)
        
        # Filter out expired signals
        valid_signals = [signal for signal in signals if signal.is_valid]
        
        # Update active signals
        self.active_signals[symbol] = valid_signals
        
        # Track new signals
        for signal in valid_signals:
            if signal not in self.detected_signals[symbol]:
                self.detected_signals[symbol].append(signal)
                logger.info(
                    "New microstructure signal detected for %s: %s, confidence: %.2f, price: %.6f",
                    symbol, signal.signal_type, signal.confidence, signal.price_level
                )
        
        return valid_signals
    
    async def _analyze_order_book_update(self, symbol: str, order_book_data: Dict[str, Any]) -> None:
        """
        Analyze a single order book update for potential signals.
        
        Args:
            symbol: The trading symbol
            order_book_data: The order book update data
        """
        # Calculate bid-ask imbalance
        bid_volume = sum(level["amount"] for level in order_book_data.get("bids", [])[:5])
        ask_volume = sum(level["amount"] for level in order_book_data.get("asks", [])[:5])
        
        if ask_volume > 0 and bid_volume > 0:
            imbalance_ratio = bid_volume / ask_volume
            
            # Detect significant imbalances
            if imbalance_ratio > self.min_imbalance_ratio:
                await self._create_imbalance_signal(symbol, "buy", imbalance_ratio, order_book_data)
            elif 1/imbalance_ratio > self.min_imbalance_ratio:
                await self._create_imbalance_signal(symbol, "sell", 1/imbalance_ratio, order_book_data)
        
        # Check for liquidity cliffs (large gaps between price levels)
        await self._check_liquidity_cliffs(symbol, order_book_data)
    
    async def _analyze_trades_update(self, symbol: str, trades_data: List[Dict[str, Any]]) -> None:
        """
        Analyze new trades for potential signals.
        
        Args:
            symbol: The trading symbol
            trades_data: List of new trades
        """
        # Analyze trade momentum
        if len(trades_data) >= 3:
            buy_volume = sum(trade["amount"] for trade in trades_data if trade.get("side") == "buy")
            sell_volume = sum(trade["amount"] for trade in trades_data if trade.get("side") == "sell")
            
            if buy_volume > 0 and sell_volume > 0:
                volume_ratio = buy_volume / sell_volume
                
                if volume_ratio > self.exhaust_volume_ratio:
                    await self._create_exhaustion_signal(symbol, "buy", volume_ratio, trades_data)
                elif 1/volume_ratio > self.exhaust_volume_ratio:
                    await self._create_exhaustion_signal(symbol, "sell", 1/volume_ratio, trades_data)
        
        # Detect large trades
        large_trades = [trade for trade in trades_data if self._is_large_trade(symbol, trade)]
        if large_trades:
            await self._analyze_large_trades(symbol, large_trades)
    
    async def _detect_order_book_anomalies(self, symbol: str) -> List[MicrostructureSignal]:
        """
        Detect anomalies in the order book structure.
        
        Args:
            symbol: The trading symbol
            
        Returns:
            List of MicrostructureSignal objects
        """
        signals = []
        
        if not self.order_book_snapshots[symbol]:
            return signals
        
        current_order_book = self.order_book_snapshots[symbol][-1]["data"]
        
        # Detect spoofing patterns
        spoofing_signals = await self._detect_spoofing(symbol, current_order_book)
        signals.extend(spoofing_signals)
        
        # Detect iceberg orders
        iceberg_signals = await self._detect_iceberg_orders(symbol, current_order_book)
        signals.extend(iceberg_signals)
        
        # Detect stop hunts
        stop_hunt_signals = await self._detect_stop_hunts(symbol)
        signals.extend(stop_hunt_signals)
        
        return signals
    
    async def _detect_trade_flow_patterns(self, symbol: str) -> List[MicrostructureSignal]:
        """
        Detect patterns in the flow of trades.
        
        Args:
            symbol: The trading symbol
            
        Returns:
            List of MicrostructureSignal objects
        """
        signals = []
        
        if len(self.trade_history[symbol]) < self.tape_reading_window:
            return signals
        
        recent_trades = list(self.trade_history[symbol])[-self.tape_reading_window:]
        
        # Detect aggressive buying/selling
        aggression_signals = await self._detect_trade_aggression(symbol, recent_trades)
        signals.extend(aggression_signals)
        
        # Detect trade clusters
        cluster_signals = await self._detect_trade_clusters(symbol, recent_trades)
        signals.extend(cluster_signals)
        
        # Detect unusual trade sequence patterns
        sequence_signals = await self._detect_unusual_sequences(symbol, recent_trades)
        signals.extend(sequence_signals)
        
        return signals
    
    async def _detect_liquidity_anomalies(self, symbol: str) -> List[MicrostructureSignal]:
        """
        Detect anomalies in market liquidity.
        
        Args:
            symbol: The trading symbol
            
        Returns:
            List of MicrostructureSignal objects
        """
        signals = []
        
        if not self.order_book_snapshots[symbol]:
            return signals
        
        # Get recent snapshots to analyze liquidity changes
        recent_snapshots = list(self.order_book_snapshots[symbol])[-10:]
        
        # Detect sudden liquidity drains
        drain_signals = await self._detect_liquidity_drains(symbol, recent_snapshots)
        signals.extend(drain_signals)
        
        # Detect one-sided liquidity
        one_sided_signals = await self._detect_one_sided_liquidity(symbol, recent_snapshots[-1]["data"])
        signals.extend(one_sided_signals)
        
        return signals
    
    async def _detect_price_formation_quirks(self, symbol: str) -> List[MicrostructureSignal]:
        """
        Detect quirks in the price formation process.
        
        Args:
            symbol: The trading symbol
            
        Returns:
            List of MicrostructureSignal objects
        """
        signals = []
        
        if not self.order_book_snapshots[symbol] or len(self.trade_history[symbol]) < 100:
            return signals
        
        # Detect micro price reversals
        reversal_signals = await self._detect_micro_reversals(symbol)
        signals.extend(reversal_signals)
        
        # Detect price rejection at specific levels
        rejection_signals = await self._detect_price_rejections(symbol)
        signals.extend(rejection_signals)
        
        # Detect abnormal bid-ask spread behavior
        spread_signals = await self._detect_abnormal_spreads(symbol)
        signals.extend(spread_signals)

        # Detect liquidity traps
        trap_signals = await self._detect_liquidity_traps(symbol)
        signals.extend(trap_signals)

        return signals
    
    async def _create_imbalance_signal(
        self, 
        symbol: str, 
        direction: str, 
        imbalance_ratio: float, 
        order_book_data: Dict[str, Any]
    ) -> None:
        """
        Create a signal based on order book imbalance.
        
        Args:
            symbol: The trading symbol
            direction: 'buy' or 'sell'
            imbalance_ratio: The calculated imbalance ratio
            order_book_data: The current order book data
        """
        confidence = min(0.95, 0.5 + (imbalance_ratio - self.min_imbalance_ratio) / 10)
        
        # Calculate expected price level based on direction
        if direction == "buy":
            price_level = float(order_book_data["asks"][0]["price"])
            expected_move = price_level * 0.0005 * min(5, imbalance_ratio / self.min_imbalance_ratio)
        else:
            price_level = float(order_book_data["bids"][0]["price"])
            expected_move = price_level * 0.0005 * min(5, imbalance_ratio / self.min_imbalance_ratio)
        
        signal = MicrostructureSignal(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            direction=direction,
            signal_type="order_book_imbalance",
            confidence=confidence,
            price_level=price_level,
            expected_move=expected_move,
            time_validity=30,  # Short-lived signal
            metadata={
                "imbalance_ratio": imbalance_ratio,
                "top_5_bid_volume": sum(level["amount"] for level in order_book_data.get("bids", [])[:5]),
                "top_5_ask_volume": sum(level["amount"] for level in order_book_data.get("asks", [])[:5])
            }
        )
        
        self.active_signals[symbol].append(signal)
        logger.debug(
            "Order book imbalance signal for %s: direction=%s, ratio=%.2f, confidence=%.2f", 
            symbol, direction, imbalance_ratio, confidence
        )
    
    async def _create_exhaustion_signal(
        self, 
        symbol: str, 
        direction: str, 
        volume_ratio: float, 
        trades_data: List[Dict[str, Any]]
    ) -> None:
        """
        Create a signal based on volume exhaustion.
        
        Args:
            symbol: The trading symbol
            direction: 'buy' or 'sell'
            volume_ratio: The calculated volume ratio
            trades_data: Recent trades data
        """
        # For exhaustion signals, we trade in the opposite direction of the exhaustion
        signal_direction = "sell" if direction == "buy" else "buy"
        
        confidence = min(0.90, 0.5 + (volume_ratio - self.exhaust_volume_ratio) / 10)
        
        # Calculate mean price from recent trades
        prices = [float(trade["price"]) for trade in trades_data]
        price_level = np.mean(prices) if prices else 0.0
        
        if price_level == 0.0:
            return
        
        # Expected move is smaller for exhaustion (reversal) signals
        expected_move = price_level * 0.0003 * min(3, volume_ratio / self.exhaust_volume_ratio)
        
        signal = MicrostructureSignal(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            direction=signal_direction,  # Opposite of exhaustion direction
            signal_type="volume_exhaustion",
            confidence=confidence,
            price_level=price_level,
            expected_move=expected_move,
            time_validity=60,  # Slightly longer validity
            metadata={
                "volume_ratio": volume_ratio,
                "buy_volume": sum(trade["amount"] for trade in trades_data if trade.get("side") == "buy"),
                "sell_volume": sum(trade["amount"] for trade in trades_data if trade.get("side") == "sell"),
                "trade_count": len(trades_data)
            }
        )
        
        self.active_signals[symbol].append(signal)
        logger.debug(
            "Volume exhaustion signal for %s: direction=%s, ratio=%.2f, confidence=%.2f", 
            symbol, signal_direction, volume_ratio, confidence
        )
    
    async def _check_liquidity_cliffs(self, symbol: str, order_book_data: Dict[str, Any]) -> None:
        """
        Check for liquidity cliffs (large gaps between price levels).
        
        Args:
            symbol: The trading symbol
            order_book_data: The current order book data
        """
        # Check ask side for liquidity cliffs
        if len(order_book_data.get("asks", [])) >= 3:
            ask_prices = [float(level["price"]) for level in order_book_data["asks"][:10]]
            ask_volumes = [float(level["amount"]) for level in order_book_data["asks"][:10]]
            
            for i in range(len(ask_prices) - 1):
                price_gap = (ask_prices[i+1] - ask_prices[i]) / ask_prices[i]
                volume_drop = 0
                
                if ask_volumes[i] > 0 and ask_volumes[i+1] > 0:
                    volume_drop = 1 - (ask_volumes[i+1] / ask_volumes[i])
                
                # Detect cliff based on price gap and volume drop
                if price_gap > self.price_impact_threshold and volume_drop > self.liquidity_cliff_threshold:
                    await self._create_liquidity_cliff_signal(
                        symbol, "buy", ask_prices[i], ask_prices[i+1], volume_drop
                    )
        
        # Check bid side for liquidity cliffs
        if len(order_book_data.get("bids", [])) >= 3:
            bid_prices = [float(level["price"]) for level in order_book_data["bids"][:10]]
            bid_volumes = [float(level["amount"]) for level in order_book_data["bids"][:10]]
            
            for i in range(len(bid_prices) - 1):
                price_gap = (bid_prices[i] - bid_prices[i+1]) / bid_prices[i]
                volume_drop = 0
                
                if bid_volumes[i] > 0 and bid_volumes[i+1] > 0:
                    volume_drop = 1 - (bid_volumes[i+1] / bid_volumes[i])
                
                # Detect cliff based on price gap and volume drop
                if price_gap > self.price_impact_threshold and volume_drop > self.liquidity_cliff_threshold:
                    await self._create_liquidity_cliff_signal(
                        symbol, "sell", bid_prices[i+1], bid_prices[i], volume_drop
                    )
    
    async def _create_liquidity_cliff_signal(
        self, 
        symbol: str, 
        direction: str, 
        price_before_cliff: float, 
        price_after_cliff: float,
        volume_drop: float
    ) -> None:
        """
        Create a signal based on a detected liquidity cliff.
        
        Args:
            symbol: The trading symbol
            direction: 'buy' or 'sell'
            price_before_cliff: Price level before the cliff
            price_after_cliff: Price level after the cliff
            volume_drop: The percentage drop in volume
        """
        cliff_size = abs(price_after_cliff - price_before_cliff) / price_before_cliff
        confidence = min(0.85, 0.5 + cliff_size * 100)
        
        # For a buy signal, target is after the cliff (resistance broken)
        # For a sell signal, target is before the cliff (support broken)
        price_level = price_before_cliff
        expected_move = abs(price_after_cliff - price_before_cliff) * 0.8
        
        signal = MicrostructureSignal(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            direction=direction,
            signal_type="liquidity_cliff",
            confidence=confidence,
            price_level=price_level,
            expected_move=expected_move,
            time_validity=120,  # Longer validity for structural signals
            metadata={
                "price_before_cliff": price_before_cliff,
                "price_after_cliff": price_after_cliff,
                "volume_drop": volume_drop,
                "cliff_size_pct": cliff_size * 100
            }
        )
        
        self.active_signals[symbol].append(signal)
        logger.debug(
            "Liquidity cliff signal for %s: direction=%s, cliff_size=%.4f%%, confidence=%.2f", 
            symbol, direction, cliff_size * 100, confidence
        )
    
    def _is_large_trade(self, symbol: str, trade: Dict[str, Any]) -> bool:
        """
        Determine if a trade is considered large for the given symbol.
        
        Args:
            symbol: The trading symbol
            trade: The trade data
            
        Returns:
            True if the trade is considered large, False otherwise
        """
        if len(self.trade_history[symbol]) < 100:
            return False
        
        # Calculate average trade size from recent history
        recent_trades = list(self.trade_history[symbol])[-100:]
        avg_trade_size = np.mean([t["data"].get("amount", 0) for t in recent_trades])
        
        # A trade is large if it's at least 5x the average size
        return trade.get("amount", 0) > avg_trade_size * 5
    
    async def _analyze_large_trades(self, symbol: str, large_trades: List[Dict[str, Any]]) -> None:
        """
        Analyze large trades for potential signals.
        
        Args:
            symbol: The trading symbol
            large_trades: List of large trades
        """
        for trade in large_trades:
            side = trade.get("side", "")
            if not side and "buyer_maker" in trade:
                # Convert buyer_maker flag to side
                side = "sell" if trade["buyer_maker"] else "buy"
            
            if side:
                # Potential signal in the same direction as large trade
                confidence = min(0.80, 0.5 + (trade["amount"] / (self._get_avg_trade_size(symbol) * 5) - 1) * 0.1)
                
                signal = MicrostructureSignal(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    direction=side,
                    signal_type="large_trade",
                    confidence=confidence,
                    price_level=float(trade["price"]),
                    expected_move=float(trade["price"]) * 0.0005,
                    time_validity=45,
                    metadata={
                        "trade_size": trade["amount"],
                        "avg_trade_size": self._get_avg_trade_size(symbol),
                        "relative_size": trade["amount"] / self._get_avg_trade_size(symbol)
                    }
                )
                
                self.active_signals[symbol].append(signal)
                logger.debug(
                    "Large trade signal for %s: direction=%s, size=%.6f, confidence=%.2f", 
                    symbol, side, trade["amount"], confidence
                )
    
    def _get_avg_trade_size(self, symbol: str) -> float:
        """
        Get the average trade size for the symbol.
        
        Args:
            symbol: The trading symbol
            
        Returns:
            Average trade size
        """
        if len(self.trade_history[symbol]) < 100:
            return 1.0  # Default if not enough history
        
        recent_trades = list(self.trade_history[symbol])[-100:]
        return np.mean([t["data"].get("amount", 0) for t in recent_trades]) or 1.0
    
    async def _detect_spoofing(
        self, 
        symbol: str, 
        current_order_book: Dict[str, Any]
    ) -> List[MicrostructureSignal]:
        """
        Detect potential spoofing patterns in the order book.
        
        Args:
            symbol: The trading symbol
            current_order_book: Current order book data
            
        Returns:
            List of detected signals
        """
        signals = []
        
        if len(self.order_book_snapshots[symbol]) < self.spoofing_detection_window:
            return signals
        
        # Get recent snapshots
        recent_snapshots = list(self.order_book_snapshots[symbol])[-self.spoofing_detection_window:]
        
        # Track large orders that suddenly disappear without execution
        vanished_bids = self._detect_vanishing_orders(recent_snapshots, "bids")
        vanished_asks = self._detect_vanishing_orders(recent_snapshots, "asks")
        
        # Create signals for detected spoofing
        if vanished_bids:
            # Vanishing bids could be sell spoofing
            signal = MicrostructureSignal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                direction="sell",
                signal_type="potential_spoofing",
                confidence=0.70,  # Moderate confidence as spoofing is hard to confirm
                price_level=float(current_order_book["bids"][0]["price"]),
                expected_move=float(current_order_book["bids"][0]["price"]) * 0.0008,
                time_validity=60,
                metadata={
                    "vanished_orders": vanished_bids,
                    "side": "bids"
                }
            )
            signals.append(signal)
        
        if vanished_asks:
            # Vanishing asks could be buy spoofing
            signal = MicrostructureSignal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                direction="buy",
                signal_type="potential_spoofing",
                confidence=0.70,
                price_level=float(current_order_book["asks"][0]["price"]),
                expected_move=float(current_order_book["asks"][0]["price"]) * 0.0008,
                time_validity=60,
                metadata={
                    "vanished_orders": vanished_asks,
                    "side": "asks"
                }
            )
            signals.append(signal)
        
        return signals
    
    def _detect_vanishing_orders(
        self, 
        snapshots: List[Dict[str, Any]], 
        side: str
    ) -> List[Dict[str, Any]]:
        """
        Detect orders that suddenly vanish without being executed.
        
        Args:
            snapshots: List of order book snapshots
            side: 'bids' or 'asks'
            
        Returns:
            List of vanished orders details
        """
        vanished_orders = []
        
        # Skip if not enough snapshots
        if len(snapshots) < 3:
            return vanished_orders
        
        # Start from the second newest snapshot (to compare with newest)
        for i in range(len(snapshots) - 2, 0, -1):
            prev_snapshot = snapshots[i]["data"]
            curr_snapshot = snapshots[i+1]["data"]
            
            if side not in prev_snapshot or side not in curr_snapshot:
                continue
            
            # Map price levels to volumes
            prev_levels = {level["price"]: level["amount"] for level in prev_snapshot[side][:10]}
            curr_levels = {level["price"]: level["amount"] for level in curr_snapshot[side][:10]}
            
            # Look for price levels with large volume that disappeared
            for price, amount in prev_levels.items():
                if (price not in curr_levels and 
                    amount > self._get_avg_level_size(snapshots, side) * 3):
                    
                    # Check if this was likely executed by looking at trades
                    if not self._was_likely_executed(snapshots[i]["timestamp"], 
                                                    snapshots[i+1]["timestamp"], 
                                                    float(price)):
                        vanished_orders.append({
                            "price": price,
                            "amount": amount,
                            "relative_size": amount / self._get_avg_level_size(snapshots, side),
                            "time": snapshots[i]["timestamp"].isoformat()
                        })
        
        return vanished_orders
    
    def _get_avg_level_size(self, snapshots: List[Dict[str, Any]], side: str) -> float:
        """
        Calculate average size of order book levels.
        
        Args:
            snapshots: List of order book snapshots
            side: 'bids' or 'asks'
            
        Returns:
            Average level size
        """
        level_sizes = []
        
        for snapshot in snapshots:
            if side in snapshot["data"]:
                for level in snapshot["data"][side][:10]:
                    level_sizes.append(float(level["amount"]))
        
        return np.mean(level_sizes) if level_sizes else 1.0
    
    def _was_likely_executed(
        self, 
        start_time: datetime, 
        end_time: datetime, 
        price: float
    ) -> bool:
        """
        Determine if an order was likely executed based on trade history.
        
        Args:
            start_time: Start of time window
            end_time: End of time window
            price: Price level to check
            
        Returns:
            True if order was likely executed, False otherwise
        """
        # This is a simplified check - a real implementation would track trades
        # and match them against disappearing orders more precisely
        return False
    
    async def _detect_iceberg_orders(
        self, 
        symbol: str, 
        current_order_book: Dict[str, Any]
    ) -> List[MicrostructureSignal]:
        """
        Detect potential iceberg orders (hidden liquidity).
        
        Args:
            symbol: The trading symbol
            current_order_book: Current order book data
            
        Returns:
            List of detected signals
        """
        signals = []
        
        # Need trade history to detect icebergs
        if len(self.trade_history[symbol]) < 50:
            return signals
        
        recent_trades = list(self.trade_history[symbol])[-50:]
        
        # Detect repeated trades at same price level that keep refilling
        bid_icebergs = self._detect_price_level_refills(recent_trades, "buy")
        ask_icebergs = self._detect_price_level_refills(recent_trades, "sell")
        
        # Create signals for detected icebergs
        for price_level, details in bid_icebergs.items():
            if details["refill_count"] >= 3:
                # Iceberg buy orders indicate support level - potential buy signal
                signal = MicrostructureSignal(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    direction="buy",
                    signal_type="iceberg_order",
                    confidence=min(0.85, 0.6 + details["refill_count"] * 0.05),
                    price_level=price_level,
                    expected_move=price_level * 0.001,
                    time_validity=180,  # Longer-lived signal as icebergs tend to persist
                    metadata={
                        "refill_count": details["refill_count"],
                        "total_volume": details["total_volume"],
                        "side": "buy"
                    }
                )
                signals.append(signal)
        
        for price_level, details in ask_icebergs.items():
            if details["refill_count"] >= 3:
                # Iceberg sell orders indicate resistance level - potential sell signal
                signal = MicrostructureSignal(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    direction="sell",
                    signal_type="iceberg_order",
                    confidence=min(0.85, 0.6 + details["refill_count"] * 0.05),
                    price_level=price_level,
                    expected_move=price_level * 0.001,
                    time_validity=180,
                    metadata={
                        "refill_count": details["refill_count"],
                        "total_volume": details["total_volume"],
                        "side": "sell"
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _detect_price_level_refills(
        self, 
        trades: List[Dict[str, Any]], 
        side: str
    ) -> Dict[float, Dict[str, Any]]:
        """
        Detect price levels that keep getting refilled (potential icebergs).
        
        Args:
            trades: List of recent trades
            side: 'buy' or 'sell'
            
        Returns:
            Dict mapping price levels to refill details
        """
        price_levels = defaultdict(lambda: {"trades": [], "refill_count": 0, "total_volume": 0})
        
        for trade_entry in trades:
            trade = trade_entry["data"]
            trade_side = trade.get("side", "")
            
            # If side not explicitly provided, try to infer from buyer_maker
            if not trade_side and "buyer_maker" in trade:
                trade_side = "sell" if trade["buyer_maker"] else "buy"
            
            if trade_side == side:
                price = float(trade["price"])
                amount = float(trade["amount"])
                
                price_levels[price]["trades"].append(trade)
                price_levels[price]["total_volume"] += amount
                
                # Check for refill pattern
                if len(price_levels[price]["trades"]) >= 2:
                    timestamps = [t.get("timestamp", 0) for t in price_levels[price]["trades"]]
                    
                    # Sort trades by timestamp if available
                    if all(timestamps):
                        sorted_trades = sorted(price_levels[price]["trades"], 
                                            key=lambda t: t.get("timestamp", 0))
                        price_levels[price]["trades"] = sorted_trades
                    
                    # Detect refill pattern by looking for time gaps
                    time_gaps = []
                    for i in range(1, len(price_levels[price]["trades"])):
                        if "timestamp" in price_levels[price]["trades"][i] and "timestamp" in price_levels[price]["trades"][i-1]:
                            t1 = price_levels[price]["trades"][i-1]["timestamp"]
                            t2 = price_levels[price]["trades"][i]["timestamp"]
                            time_gaps.append((t2 - t1).total_seconds())
                    
                    # If we have gaps and they're not too small or too large
                    if time_gaps and any(5 < gap < 60 for gap in time_gaps):
                        price_levels[price]["refill_count"] += 1
        
        return price_levels
    
    async def _detect_stop_hunts(self, symbol: str) -> List[MicrostructureSignal]:
        """
        Detect potential stop hunting patterns.
        
        Args:
            symbol: The trading symbol
            
        Returns:
            List of detected signals
        """
        signals = []
        
        # Need enough price history
        if len(self.trade_history[symbol]) < 200:
            return signals
        
        recent_trades = list(self.trade_history[symbol])[-200:]
        prices = [float(t["data"]["price"]) for t in recent_trades]
        
        # Look for quick price movements followed by reversals
        if len(prices) < 20:
            return signals
        
        for i in range(20, len(prices)):
            window = prices[i-20:i]
            pre_move = window[:10]
            post_move = window[10:]
            
            # Calculate quick moves
            pre_range = max(pre_move) - min(pre_move)
            post_range = max(post_move) - min(post_move)
            
            # Detect if there was a significant price movement
            if pre_range > 0:
                move_pct = post_range / pre_range
                
                # Detect rapid move followed by reversal
                if move_pct > 2.0:
                    # Check for reversal
                    pre_trend = pre_move[-1] - pre_move[0]
                    post_trend = post_move[-1] - post_move[0]
                    
                    if (pre_trend > 0 and post_trend < 0) or (pre_trend < 0 and post_trend > 0):
                        # This could be a stop hunt
                        direction = "buy" if post_trend > 0 else "sell"
                        
                        signal = MicrostructureSignal(
                            timestamp=datetime.utcnow(),
                            symbol=symbol,
                            direction=direction,
                            signal_type="potential_stop_hunt",
                            confidence=0.75,
                            price_level=prices[-1],
                            expected_move=pre_range * 0.5,
                            time_validity=120,
                            metadata={
                                "pre_range": pre_range,
                                "post_range": post_range,
                                "move_pct": move_pct,
                                "pre_trend": pre_trend,
                                "post_trend": post_trend
                            }
                        )
                        signals.append(signal)
        
        return signals
    
    async def _detect_trade_aggression(
        self, 
        symbol: str, 
        recent_trades: List[Dict[str, Any]]
    ) -> List[MicrostructureSignal]:
        """
        Detect aggressive buying or selling in recent trades.
        
        Args:
            symbol: The trading symbol
            recent_trades: List of recent trades
            
        Returns:
            List of detected signals
        """
        signals = []
        
        if len(recent_trades) < self.tape_reading_window:
            return signals
        
        # Count market vs limit orders
        market_buys = 0
        market_sells = 0
        limit_buys = 0
        limit_sells = 0
        
        for trade_entry in recent_trades:
            trade = trade_entry["data"]
            side = trade.get("side", "")
            
            # If side not explicitly provided, try to infer from buyer_maker
            if not side and "buyer_maker" in trade:
                side = "sell" if trade["buyer_maker"] else "buy"
                
            # If maker flag is available, use it to determine aggression
            is_market = trade.get("maker", False) is False
            
            if side == "buy":
                if is_market:
                    market_buys += 1
                else:
                    limit_buys += 1
            elif side == "sell":
                if is_market:
                    market_sells += 1
                else:
                    limit_sells += 1
        
        total_buys = market_buys + limit_buys
        total_sells = market_sells + limit_sells
        
        # Calculate aggression ratios
        buy_aggression = market_buys / total_buys if total_buys > 0 else 0
        sell_aggression = market_sells / total_sells if total_sells > 0 else 0
        
        # Generate signals for high aggression
        if buy_aggression > 0.7 and total_buys >= 10:
            # High buying aggression - potential upward movement
            price_level = float(recent_trades[-1]["data"]["price"])
            
            signal = MicrostructureSignal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                direction="buy",
                signal_type="aggressive_buying",
                confidence=min(0.90, 0.6 + buy_aggression * 0.3),
                price_level=price_level,
                expected_move=price_level * 0.001,
                time_validity=90,
                metadata={
                    "buy_aggression": buy_aggression,
                    "market_buys": market_buys,
                    "limit_buys": limit_buys,
                    "total_buys": total_buys
                }
            )
            signals.append(signal)
        
        if sell_aggression > 0.7 and total_sells >= 10:
            # High selling aggression - potential downward movement
            price_level = float(recent_trades[-1]["data"]["price"])
            
            signal = MicrostructureSignal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                direction="sell",
                signal_type="aggressive_selling",
                confidence=min(0.90, 0.6 + sell_aggression * 0.3),
                price_level=price_level,
                expected_move=price_level * 0.001,
                time_validity=90,
                metadata={
                    "sell_aggression": sell_aggression,
                    "market_sells": market_sells,
                    "limit_sells": limit_sells,
                    "total_sells": total_sells
                }
            )
            signals.append(signal)
        
        return signals
    
    async def _detect_trade_clusters(
        self, 
        symbol: str, 
        recent_trades: List[Dict[str, Any]]
    ) -> List[MicrostructureSignal]:
        """
        Detect clusters of trades at specific price levels.
        
        Args:
            symbol: The trading symbol
            recent_trades: List of recent trades
            
        Returns:
            List of detected signals
        """
        signals = []
        
        if len(recent_trades) < 30:
            return signals
        
        # Group trades by price level
        price_clusters = defaultdict(lambda: {"count": 0, "volume": 0, "buy_count": 0, "sell_count": 0})
        
        for trade_entry in recent_trades:
            trade = trade_entry["data"]
            price = float(trade["price"])
            amount = float(trade["amount"])
            
            # Round price to reduce noise in clustering
            rounded_price = round(price, 6)
            
            price_clusters[rounded_price]["count"] += 1
            price_clusters[rounded_price]["volume"] += amount
            
            # Track direction if available
            side = trade.get("side", "")
            if not side and "buyer_maker" in trade:
                side = "sell" if trade["buyer_maker"] else "buy"
                
            if side == "buy":
                price_clusters[rounded_price]["buy_count"] += 1
            elif side == "sell":
                price_clusters[rounded_price]["sell_count"] += 1
        
        # Find significant clusters
        avg_count = np.mean([c["count"] for c in price_clusters.values()])
        avg_volume = np.mean([c["volume"] for c in price_clusters.values()])
        
        for price, cluster in price_clusters.items():
            if cluster["count"] > max(5, avg_count * 2) and cluster["volume"] > avg_volume * 2:
                # This is a significant cluster
                
                # Determine direction based on buy/sell ratio
                if cluster["buy_count"] > cluster["sell_count"] * 1.5:
                    direction = "buy"
                elif cluster["sell_count"] > cluster["buy_count"] * 1.5:
                    direction = "sell"
                else:
                    # No clear direction, skip
                    continue
                
                signal = MicrostructureSignal(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    direction=direction,
                    signal_type="trade_cluster",
                    confidence=min(0.85, 0.6 + cluster["count"] / avg_count * 0.1),
                    price_level=price,
                    expected_move=price * 0.0008,
                    time_validity=150,
                    metadata={
                        "cluster_count": cluster["count"],
                        "cluster_volume": cluster["volume"],
                        "buy_count": cluster["buy_count"],
                        "sell_count": cluster["sell_count"],
                        "avg_count": avg_count,
                        "avg_volume": avg_volume
                    }
                )
                signals.append(signal)
        
        return signals
    
    async def _detect_unusual_sequences(
        self, 
        symbol: str, 
        recent_trades: List[Dict[str, Any]]
    ) -> List[MicrostructureSignal]:
        """
        Detect unusual sequences in trade patterns.
        
        Args:
            symbol: The trading symbol
            recent_trades: List of recent trades
            
        Returns:
            List of detected signals
        """
        signals = []
        
        if len(recent_trades) < 30:
            return signals
        
        # Extract sequence of trade sides and sizes
        sides = []
        sizes = []
        prices = []
        
        for trade_entry in recent_trades:
            trade = trade_entry["data"]
            
            side = trade.get("side", "")
            if not side and "buyer_maker" in trade:
                side = "sell" if trade["buyer_maker"] else "buy"
                
            if side:
                sides.append(side)
                sizes.append(float(trade["amount"]))
                prices.append(float(trade["price"]))
        
        # Skip if we couldn't extract enough sides
        if len(sides) < 20:
            return signals
        
        # Look for unusual repetitive patterns
        buy_streak = 0
        sell_streak = 0
        increasing_size_streak = 0
        
        for i in range(1, len(sides)):
            # Track buy/sell streaks
            if sides[i] == "buy":
                buy_streak += 1
                sell_streak = 0
            else:
                sell_streak += 1
                buy_streak = 0
            
            # Track increasing size streaks
            if sizes[i] > sizes[i-1]:
                increasing_size_streak += 1
            else:
                increasing_size_streak = 0
            
            # Generate signals for unusual streaks
            if buy_streak >= 10:
                # Unusual buying streak
                signal = MicrostructureSignal(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    direction="buy",
                    signal_type="buy_streak",
                    confidence=min(0.80, 0.5 + buy_streak * 0.02),
                    price_level=prices[-1],
                    expected_move=prices[-1] * 0.001,
                    time_validity=60,
                    metadata={
                        "streak_length": buy_streak,
                        "avg_size": np.mean(sizes[-buy_streak:]),
                        "price_change": (prices[-1] - prices[-buy_streak]) / prices[-buy_streak] * 100
                    }
                )
                signals.append(signal)
                
            if sell_streak >= 10:
                # Unusual selling streak
                signal = MicrostructureSignal(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    direction="sell",
                    signal_type="sell_streak",
                    confidence=min(0.80, 0.5 + sell_streak * 0.02),
                    price_level=prices[-1],
                    expected_move=prices[-1] * 0.001,
                    time_validity=60,
                    metadata={
                        "streak_length": sell_streak,
                        "avg_size": np.mean(sizes[-sell_streak:]),
                        "price_change": (prices[-1] - prices[-sell_streak]) / prices[-sell_streak] * 100
                    }
                )
                signals.append(signal)
                
            if increasing_size_streak >= 5:
                # Increasing trade size could indicate building momentum
                direction = sides[-1]
                
                signal = MicrostructureSignal(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    direction=direction,
                    signal_type="increasing_size",
                    confidence=min(0.75, 0.5 + increasing_size_streak * 0.03),
                    price_level=prices[-1],
                    expected_move=prices[-1] * 0.0007,
                    time_validity=45,
                    metadata={
                        "streak_length": increasing_size_streak,
                        "size_start": sizes[-increasing_size_streak],
                        "size_end": sizes[-1],
                        "growth_factor": sizes[-1] / sizes[-increasing_size_streak],
                        "side": direction
                    }
                )
                signals.append(signal)
        
        return signals
    
    async def _detect_liquidity_drains(
        self, 
        symbol: str, 
        recent_snapshots: List[Dict[str, Any]]
    ) -> List[MicrostructureSignal]:
        """
        Detect sudden drains in liquidity.
        
        Args:
            symbol: The trading symbol
            recent_snapshots: Recent order book snapshots
            
        Returns:
            List of detected signals
        """
        signals = []
        
        if len(recent_snapshots) < 3:
            return signals
        
        # Calculate liquidity for each snapshot
        bid_liquidity = []
        ask_liquidity = []
        
        for snapshot in recent_snapshots:
            if "data" not in snapshot:
                continue
                
            order_book = snapshot["data"]
            
            # Sum up liquidity in top N levels
            bid_sum = sum(float(level["amount"]) for level in order_book.get("bids", [])[:5])
            ask_sum = sum(float(level["amount"]) for level in order_book.get("asks", [])[:5])
            
            bid_liquidity.append(bid_sum)
            ask_liquidity.append(ask_sum)
        
        # Skip if not enough data points
        if len(bid_liquidity) < 3 or len(ask_liquidity) < 3:
            return signals
        
        # Calculate percentage changes in liquidity
        bid_changes = [(bid_liquidity[i] - bid_liquidity[i-1]) / bid_liquidity[i-1] 
                      if bid_liquidity[i-1] > 0 else 0 
                      for i in range(1, len(bid_liquidity))]
        
        ask_changes = [(ask_liquidity[i] - ask_liquidity[i-1]) / ask_liquidity[i-1] 
                      if ask_liquidity[i-1] > 0 else 0 
                      for i in range(1, len(ask_liquidity))]
        
        # Detect significant liquidity drains
        for i in range(len(bid_changes)):
            # Significant drop in bid liquidity could be bearish
            if bid_changes[i] < -0.4:  # 40% drop
                price_level = float(recent_snapshots[-1]["data"]["bids"][0]["price"])
                
                signal = MicrostructureSignal(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    direction="sell",
                    signal_type="bid_liquidity_drain",
                    confidence=min(0.85, 0.5 + abs(bid_changes[i]) * 0.5),
                    price_level=price_level,
                    expected_move=price_level * 0.001,
                    time_validity=90,
                    metadata={
                        "liquidity_change_pct": bid_changes[i] * 100,
                        "current_liquidity": bid_liquidity[-1],
                        "previous_liquidity": bid_liquidity[-2]
                    }
                )
                signals.append(signal)
        
        for i in range(len(ask_changes)):
            # Significant drop in ask liquidity could be bullish
            if ask_changes[i] < -0.4:  # 40% drop
                price_level = float(recent_snapshots[-1]["data"]["asks"][0]["price"])
                
                signal = MicrostructureSignal(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    direction="buy",
                    signal_type="ask_liquidity_drain",
                    confidence=min(0.85, 0.5 + abs(ask_changes[i]) * 0.5),
                    price_level=price_level,
                    expected_move=price_level * 0.001,
                    time_validity=90,
                    metadata={
                        "liquidity_change_pct": ask_changes[i] * 100,
                        "current_liquidity": ask_liquidity[-1],
                        "previous_liquidity": ask_liquidity[-2]
                    }
                )
                signals.append(signal)
        
        return signals
    
    async def _detect_one_sided_liquidity(
        self, 
        symbol: str, 
        order_book: Dict[str, Any]
    ) -> List[MicrostructureSignal]:
        """
        Detect one-sided liquidity imbalances.
        
        Args:
            symbol: The trading symbol
            order_book: Current order book data
            
        Returns:
            List of detected signals
        """
        signals = []
        
        if "bids" not in order_book or "asks" not in order_book:
            return signals
        
        # Calculate total liquidity on each side
        bid_liquidity = sum(float(level["amount"]) for level in order_book["bids"][:10])
        ask_liquidity = sum(float(level["amount"]) for level in order_book["asks"][:10])
        
        # Skip if either side has no liquidity
        if bid_liquidity == 0 or ask_liquidity == 0:
            return signals
        
        # Calculate imbalance ratio
        imbalance_ratio = bid_liquidity / ask_liquidity
        
        # Generate signals for significant imbalances
        if imbalance_ratio > 4.0:  # Bids 4x asks
            price_level = float(order_book["asks"][0]["price"])
            
            signal = MicrostructureSignal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                direction="buy",
                signal_type="buy_side_liquidity_imbalance",
                confidence=min(0.88, 0.6 + min(imbalance_ratio, 10) * 0.03),
                price_level=price_level,
                expected_move=price_level * 0.001,
                time_validity=120,
                metadata={
                    "imbalance_ratio": imbalance_ratio,
                    "bid_liquidity": bid_liquidity,
                    "ask_liquidity": ask_liquidity
                }
            )
            signals.append(signal)
            
        elif imbalance_ratio < 0.25:  # Asks 4x bids
            price_level = float(order_book["bids"][0]["price"])
            
            signal = MicrostructureSignal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                direction="sell",
                signal_type="sell_side_liquidity_imbalance",
                confidence=min(0.88, 0.6 + min(1/imbalance_ratio, 10) * 0.03),
                price_level=price_level,
                expected_move=price_level * 0.001,
                time_validity=120,
                metadata={
                    "imbalance_ratio": imbalance_ratio,
                    "bid_liquidity": bid_liquidity,
                    "ask_liquidity": ask_liquidity
                }
            )
            signals.append(signal)
        
        return signals
    
    async def _detect_micro_reversals(self, symbol: str) -> List[MicrostructureSignal]:
        """
        Detect micro-reversals in price action.
        
        Args:
            symbol: The trading symbol
            
        Returns:
            List of detected signals
        """
        signals = []
        
        if len(self.trade_history[symbol]) < 50:
            return signals
        
        # Get recent trades
        recent_trades = list(self.trade_history[symbol])[-50:]
        prices = [float(t["data"]["price"]) for t in recent_trades]
        
        # Skip if not enough prices
        if len(prices) < 30:
            return signals
        
        # Look for price exhaustion and reversal patterns
        for window_size in [10, 15, 20]:
            if len(prices) < window_size * 2:
                continue
                
            for i in range(window_size, len(prices) - window_size):
                pre_window = prices[i-window_size:i]
                post_window = prices[i:i+window_size]
                
                pre_min, pre_max = min(pre_window), max(pre_window)
                post_min, post_max = min(post_window), max(post_window)
                
                pre_range = pre_max - pre_min
                
                # Skip if range is too small
                if pre_range < prices[i] * 0.0005:
                    continue
                
                # Detect bullish reversal
                if prices[i] < pre_min + pre_range * 0.2 and post_max > pre_min + pre_range * 0.5:
                    signal = MicrostructureSignal(
                        timestamp=datetime.utcnow(),
                        symbol=symbol,
                        direction="buy",
                        signal_type="micro_reversal",
                        confidence=0.75,
                        price_level=prices[-1],
                        expected_move=pre_range * 0.3,
                        time_validity=60,
                        metadata={
                            "pre_range": pre_range,
                            "pre_min": pre_min,
                            "pre_max": pre_max,
                            "post_max": post_max,
                            "window_size": window_size
                        }
                    )
                    signals.append(signal)
                    
                # Detect bearish reversal
                if prices[i] > pre_max - pre_range * 0.2 and post_min < pre_max - pre_range * 0.5:
                    signal = MicrostructureSignal(
                        timestamp=datetime.utcnow(),
                        symbol=symbol,
                        direction="sell",
                        signal_type="micro_reversal",
                        confidence=0.75,
                        price_level=prices[-1],
                        expected_move=pre_range * 0.3,
                        time_validity=60,
                        metadata={
                            "pre_range": pre_range,
                            "pre_min": pre_min,
                            "pre_max": pre_max,
                            "post_min": post_min,
                            "window_size": window_size
                        }
                    )
                    signals.append(signal)
        
        return signals
    
    async def _detect_price_rejections(self, symbol: str) -> List[MicrostructureSignal]:
        """
        Detect price rejections at specific levels.
        
        Args:
            symbol: The trading symbol
            
        Returns:
            List of detected signals
        """
        signals = []
        
        if len(self.trade_history[symbol]) < 100:
            return signals
        
        # Get recent trades
        recent_trades = list(self.trade_history[symbol])[-100:]
        prices = [float(t["data"]["price"]) for t in recent_trades]
        
        # Skip if not enough prices
        if len(prices) < 50:
            return signals
        
        # Find levels that have been tested multiple times but not broken
        price_levels = {}
        
        # Group prices into levels
        for price in prices:
            # Round to reduce noise
            price_precision = 6  # Adjust based on symbol
            rounded_price = round(price, price_precision)
            
            if rounded_price not in price_levels:
                price_levels[rounded_price] = {"count": 0, "touches": []}
                
            price_levels[rounded_price]["count"] += 1
            price_levels[rounded_price]["touches"].append(price)
        
        # Find levels with multiple touches
        for level, data in price_levels.items():
            if data["count"] >= 4:  # At least 4 touches
                # Check if price moved away from this level significantly
                level_float = float(level)
                
                # Get prices after last touch
                last_touch_idx = len(prices) - 1 - prices[::-1].index(data["touches"][-1])
                prices_after = prices[last_touch_idx+1:]
                
                if prices_after:
                    max_after = max(prices_after)
                    min_after = min(prices_after)
                    
                    # Determine if this was resistance or support
                    if max_after < level_float and level_float - min_after > level_float * 0.001:
                        # Resistance rejection
                        signal = MicrostructureSignal(
                            timestamp=datetime.utcnow(),
                            symbol=symbol,
                            direction="sell",
                            signal_type="resistance_rejection",
                            confidence=min(0.85, 0.6 + data["count"] * 0.05),
                            price_level=level_float,
                            expected_move=level_float * 0.001,
                            time_validity=120,
                            metadata={
                                "touch_count": data["count"],
                                "rejection_level": level_float,
                                "max_after_touch": max_after,
                                "distance_from_level": (level_float - max_after) / level_float * 100
                            }
                        )
                        signals.append(signal)
                        
                    elif min_after > level_float and max_after - level_float > level_float * 0.001:
                        # Support rejection
                        signal = MicrostructureSignal(
                            timestamp=datetime.utcnow(),
                            symbol=symbol,
                            direction="buy",
                            signal_type="support_rejection",
                            confidence=min(0.85, 0.6 + data["count"] * 0.05),
                            price_level=level_float,
                            expected_move=level_float * 0.001,
                            time_validity=120,
                            metadata={
                                "touch_count": data["count"],
                                "rejection_level": level_float,
                                "min_after_touch": min_after,
                                "distance_from_level": (min_after - level_float) / level_float * 100
                            }
                        )
                        signals.append(signal)
        
        return signals
    
    async def _detect_abnormal_spreads(self, symbol: str) -> List[MicrostructureSignal]:
        """
        Detect abnormal bid-ask spread behavior.
        
        Args:
            symbol: The trading symbol
            
        Returns:
            List of detected signals
        """
        signals = []
        
        if len(self.order_book_snapshots[symbol]) < 20:
            return signals
        
        # Calculate spreads from recent snapshots
        spreads = []
        
        for snapshot in list(self.order_book_snapshots[symbol])[-20:]:
            order_book = snapshot["data"]
            
            if "bids" in order_book and "asks" in order_book and order_book["bids"] and order_book["asks"]:
                best_bid = float(order_book["bids"][0]["price"])
                best_ask = float(order_book["asks"][0]["price"])
                
                spread = best_ask - best_bid
                spread_pct = spread / best_bid * 100
                
                spreads.append(spread_pct)
        
        # Skip if not enough spreads calculated
        if len(spreads) < 10:
            return signals
        
        # Calculate average and standard deviation
        avg_spread = np.mean(spreads)
        std_spread = np.std(spreads)
        
        # Check if current spread is abnormal
        current_spread = spreads[-1]
        
        # Spread suddenly widening
        if current_spread > avg_spread + 2 * std_spread:
            # Widening spread often indicates increased volatility or uncertainty
            # This could be a sell signal as market makers pull liquidity
            
            current_order_book = self.order_book_snapshots[symbol][-1]["data"]
            price_level = float(current_order_book["bids"][0]["price"])
            
            signal = MicrostructureSignal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                direction="sell",
                signal_type="abnormal_spread_widening",
                confidence=min(0.80, 0.5 + (current_spread - avg_spread) / std_spread * 0.1),
                price_level=price_level,
                expected_move=price_level * 0.001,
                time_validity=45,
                metadata={
                    "current_spread_pct": current_spread,
                    "avg_spread_pct": avg_spread,
                    "std_spread_pct": std_spread,
                    "z_score": (current_spread - avg_spread) / std_spread
                }
            )
            signals.append(signal)
            
        # Spread suddenly narrowing
        elif current_spread < avg_spread - std_spread and avg_spread > 0.02:
            # Narrowing spread from a previously wide state could indicate
            # returning liquidity and potential price stabilization
            
            current_order_book = self.order_book_snapshots[symbol][-1]["data"]
            price_level = float(current_order_book["asks"][0]["price"])
            
            signal = MicrostructureSignal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                direction="buy",
                signal_type="abnormal_spread_narrowing",
                confidence=min(0.75, 0.5 + (avg_spread - current_spread) / std_spread * 0.1),
                price_level=price_level,
                expected_move=price_level * 0.0007,
                time_validity=60,
                metadata={
                    "current_spread_pct": current_spread,
                    "avg_spread_pct": avg_spread,
                    "std_spread_pct": std_spread,
                    "z_score": (current_spread - avg_spread) / std_spread
                }
            )
            signals.append(signal)
        
        return signals

    async def _detect_liquidity_traps(self, symbol: str) -> List[MicrostructureSignal]:
        """Detect potential liquidity trap patterns."""
        signals = []

        window = self.liquidity_trap_window

        if len(self.trade_history[symbol]) < window * 2:
            return signals

        recent_trades = list(self.trade_history[symbol])[-window * 2:]
        pre_trades = recent_trades[:window]
        trap_trades = recent_trades[window:]

        pre_prices = [float(t["data"]["price"]) for t in pre_trades]
        trap_prices = [float(t["data"]["price"]) for t in trap_trades]
        pre_vol = sum(float(t["data"].get("amount", 0)) for t in pre_trades)
        trap_vol = sum(float(t["data"].get("amount", 0)) for t in trap_trades)

        if not pre_prices or not trap_prices or pre_vol == 0:
            return signals

        pre_avg = np.mean(pre_prices)
        spike_range = max(trap_prices) - min(trap_prices)

        price_reverted = abs(trap_prices[-1] - pre_avg) / pre_avg < self.liquidity_trap_threshold
        spike_enough = spike_range / pre_avg > self.liquidity_trap_threshold * 2
        low_volume = trap_vol / pre_vol < self.trap_volume_ratio

        if price_reverted and spike_enough and low_volume:
            direction = "sell" if trap_prices[0] > pre_avg else "buy"
            signal = MicrostructureSignal(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                direction=direction,
                signal_type="liquidity_trap",
                confidence=0.7,
                price_level=trap_prices[-1],
                expected_move=pre_avg * 0.0005,
                time_validity=60,
                metadata={
                    "pre_avg_price": pre_avg,
                    "spike_range": spike_range,
                    "volume_ratio": trap_vol / pre_vol,
                },
            )
            signals.append(signal)

        return signals
            
    async def evaluate_signal_performance(self, symbol: str, signal: MicrostructureSignal, result: Dict[str, Any]) -> None:
        """
        Evaluate the performance of a signal after it has played out.
        
        Args:
            symbol: The trading symbol
            signal: The original signal
            result: Result data including price movements and outcome
        """
        # Update performance metrics
        if symbol not in self.signal_performance:
            self.signal_performance[symbol] = {
                "total": 0,
                "successful": 0,
                "accuracy": 0.0,
                "avg_profit": 0.0,
                "avg_drawdown": 0.0
            }
        
        self.signal_performance[symbol]["total"] += 1
        
        if result.get("successful", False):
            self.signal_performance[symbol]["successful"] += 1
        
        # Update accuracy
        if self.signal_performance[symbol]["total"] > 0:
            self.signal_performance[symbol]["accuracy"] = (
                self.signal_performance[symbol]["successful"] / 
                self.signal_performance[symbol]["total"]
            )
        
        # Update average profit
        profit_pct = result.get("profit_pct", 0.0)
        drawdown_pct = result.get("max_drawdown_pct", 0.0)
        
        # Incremental update of averages
        current_total = self.signal_performance[symbol]["total"]
        current_avg_profit = self.signal_performance[symbol]["avg_profit"]
        current_avg_drawdown = self.signal_performance[symbol]["avg_drawdown"]
        
        self.signal_performance[symbol]["avg_profit"] = (
            (current_avg_profit * (current_total - 1) + profit_pct) / current_total
        )
        
        self.signal_performance[symbol]["avg_drawdown"] = (
            (current_avg_drawdown * (current_total - 1) + drawdown_pct) / current_total
        )
        
        # Log results
        logger.info(
            "Signal evaluation for %s: %s signal %s (confidence: %.2f). Profit: %.2f%%, Success: %s",
            symbol, signal.signal_type, signal.direction, signal.confidence, 
            profit_pct, result.get("successful", False)
        )
        
        # Log current performance metrics
        logger.info(
            "Current performance for %s: Accuracy: %.2f%%, Avg Profit: %.2f%%, Avg Drawdown: %.2f%%",
            symbol, 
            self.signal_performance[symbol]["accuracy"] * 100,
            self.signal_performance[symbol]["avg_profit"],
            self.signal_performance[symbol]["avg_drawdown"]
        )
    
    def get_performance_metrics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for signals.
        
        Args:
            symbol: Optional symbol to get metrics for. If None, returns metrics for all symbols.
            
        Returns:
            Dictionary of performance metrics
        """
        if symbol:
            return self.signal_performance.get(symbol, {
                "total": 0,
                "successful": 0,
                "accuracy": 0.0,
                "avg_profit": 0.0,
                "avg_drawdown": 0.0
            })
        
        # Aggregate metrics across all symbols
        all_metrics = {
            "total": 0,
            "successful": 0,
            "accuracy": 0.0,
            "avg_profit": 0.0,
            "avg_drawdown": 0.0
        }
        
        for symbol_metrics in self.signal_performance.values():
            all_metrics["total"] += symbol_metrics["total"]
            all_metrics["successful"] += symbol_metrics["successful"]
            
            # Weighted average for profit and drawdown
            if all_metrics["total"] > 0:
                weight = symbol_metrics["total"] / all_metrics["total"]
                all_metrics["avg_profit"] += symbol_metrics["avg_profit"] * weight
                all_metrics["avg_drawdown"] += symbol_metrics["avg_drawdown"] * weight
        
        # Calculate overall accuracy
        if all_metrics["total"] > 0:
            all_metrics["accuracy"] = all_metrics["successful"] / all_metrics["total"]
        
        return all_metrics


class MicrostructureDetector:
    """
    Detector for market microstructure anomalies and exploitable patterns.
    
    This class wraps the MicrostructureAnalyzer to provide a standardized interface
    for the loophole detection system, focusing on detecting and exploiting
    short-term inefficiencies in market microstructure.
    """
    
    def __init__(
        self,
        market_data_repo: MarketDataRepository = None,
        order_flow_analyzer: OrderFlowAnalyzer = None,
        volume_profile_analyzer: VolumeProfileAnalyzer = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the MicrostructureDetector.
        
        Args:
            market_data_repo: Repository for market data access
            order_flow_analyzer: Analyzer for order flow patterns
            volume_profile_analyzer: Analyzer for volume profiles
            config: Configuration parameters for the detector
        """
        self.config = config or {}
        
        # Handle the case when analyzers are None
        if market_data_repo is None or order_flow_analyzer is None or volume_profile_analyzer is None:
            self.analyzer = None
            self.logger = get_logger("intelligence.loophole_detection.microstructure_detector")
            self.logger.warning("MicrostructureDetector initialized with missing dependencies")
        else:
            self.analyzer = MicrostructureAnalyzer(
                market_data_repo=market_data_repo,
                order_flow_analyzer=order_flow_analyzer,
                volume_analyzer=volume_profile_analyzer,
                config=self.config
            )
        self.active_symbols = set()
        self.logger = get_logger("intelligence.loophole_detection.microstructure_detector")
        self.logger.info("MicrostructureDetector initialized")
        
    async def detect(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Detect microstructure anomalies for the given symbol.
        
        Args:
            symbol: The trading symbol to analyze
            
        Returns:
            List of detected anomalies with details
        """
        if self.analyzer is None:
            self.logger.warning(f"Cannot detect microstructure anomalies for {symbol}: analyzer not initialized")
            return []
            
        self.active_symbols.add(symbol)
        signals = await self.analyzer.analyze_microstructure(symbol)
        
        # Convert signals to standardized format
        anomalies = []
        for signal in signals:
            anomaly = {
                "timestamp": signal.timestamp.isoformat(),
                "symbol": signal.symbol,
                "type": f"microstructure_{signal.signal_type}",
                "direction": signal.direction,
                "confidence": signal.confidence,
                "price_level": signal.price_level,
                "expected_move": signal.expected_move,
                "expiry": signal.time_to_expiry,
                "metadata": signal.metadata
            }
            anomalies.append(anomaly)
            
        return anomalies
        
    async def update_order_book(self, symbol: str, order_book_data: Dict[str, Any]) -> None:
        """
        Update the order book data for real-time analysis.
        
        Args:
            symbol: The trading symbol
            order_book_data: The order book data
        """
        if self.analyzer is None:
            self.logger.warning(f"Cannot update order book for {symbol}: analyzer not initialized")
            return
            
        await self.analyzer.update_order_book(symbol, order_book_data)
        
    async def update_trades(self, symbol: str, trades_data: List[Dict[str, Any]]) -> None:
        """
        Update the trades data for real-time analysis.
        
        Args:
            symbol: The trading symbol
            trades_data: List of trade data
        """
        if self.analyzer is None:
            self.logger.warning(f"Cannot update trades for {symbol}: analyzer not initialized")
            return
            
        await self.analyzer.update_trades(symbol, trades_data)
        
    def get_performance_metrics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for the detector.
        
        Args:
            symbol: Optional symbol to filter metrics
            
        Returns:
            Dictionary of performance metrics
        """
        if self.analyzer is None:
            self.logger.warning("Cannot get performance metrics: analyzer not initialized")
            return {
                "total": 0,
                "successful": 0,
                "accuracy": 0.0,
                "avg_profit": 0.0,
                "avg_drawdown": 0.0
            }
            
        return self.analyzer.get_performance_metrics(symbol)
