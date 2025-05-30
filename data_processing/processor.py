"""
Data Processor for the QuantumSpectre Elite Trading System.

This module contains the core data processing functionality, responsible for
transforming raw market data into standardized formats for further analysis.
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd

from common.config import settings
from common.logger import get_logger
from common.models.market_data import (
    MarketDepth, 
    Trade, 
    Candle, 
    Ticker, 
    OrderBook
)
from common.metrics import calculate_timing

logger = get_logger("data_processor")

class DataProcessor:
    """
    Advanced data processor for exchange market data.
    
    This class handles the processing of raw market data from various exchanges,
    converting it into standardized formats for further analysis and decision-making.
    It includes optimized algorithms for high-throughput processing and detection of
    significant market events.
    """
    
    def __init__(self):
        """Initialize data processor with caching and sequence tracking."""
        # Track sequence numbers for different data types to detect gaps
        self.last_sequence: Dict[str, int] = {}
        
        # Cache for order book snapshots and previous states
        self.orderbook_cache: Dict[str, Dict[str, Any]] = {}
        
        # Cache for recently processed data
        self.processed_cache: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.timing_stats: Dict[str, List[float]] = {
            "depth": [],
            "trade": [],
            "kline": [],
            "ticker": []
        }
        
        # Statistical anomaly detection thresholds
        self.volume_ma: Dict[str, float] = {}  # Moving average of volumes
        self.price_volatility: Dict[str, float] = {}  # Price volatility measures
        
        # Processing flags
        self.detect_anomalies = settings.DETECT_MARKET_ANOMALIES
        self.calculate_vwap = settings.CALCULATE_VWAP
        self.track_liquidations = settings.TRACK_LIQUIDATIONS
        
        logger.info("DataProcessor initialized with anomaly detection=%s, VWAP calculation=%s, liquidation tracking=%s",
                   self.detect_anomalies, self.calculate_vwap, self.track_liquidations)
    
    @calculate_timing
    def process_binance_ws(self, message: str, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Process Binance WebSocket message.
        
        Args:
            message: Raw WebSocket message
            symbol: Trading symbol
            
        Returns:
            Processed data or None if message can't be processed
        """
        try:
            data = json.loads(message)
            
            # Check message type based on keys
            if 'e' in data:  # Event type
                event_type = data['e']
                
                if event_type == 'depthUpdate':
                    return self._process_binance_depth(data, symbol)
                    
                elif event_type == 'trade':
                    return self._process_binance_trade(data, symbol)
                    
                elif event_type == 'kline':
                    return self._process_binance_kline(data, symbol)
                    
                elif event_type == '24hrTicker':
                    return self._process_binance_ticker(data, symbol)
                
                elif event_type == 'forceOrder':  # Liquidation event
                    return self._process_binance_liquidation(data, symbol)
                    
                else:
                    logger.debug(f"Unhandled Binance event type: {event_type}")
                    return None
            
            # Check if it's an order book snapshot
            elif 'lastUpdateId' in data and 'bids' in data and 'asks' in data:
                return self._process_binance_orderbook_snapshot(data, symbol)
                
            elif 'id' in data and 'result' in data:  # API response
                return self._process_binance_api_response(data, symbol)
                
            else:
                logger.debug(f"Unrecognized Binance message format: {message[:100]}...")
                return None
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in Binance message: {message[:100]}...")
            return None
            
        except Exception as e:
            logger.error(f"Error processing Binance message: {str(e)}")
            logger.debug(f"Problematic message: {message[:100]}...")
            return None
    
    @calculate_timing
    def process_deriv_ws(self, message: str, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Process Deriv WebSocket message.
        
        Args:
            message: Raw WebSocket message
            symbol: Trading symbol
            
        Returns:
            Processed data or None if message can't be processed
        """
        try:
            data = json.loads(message)
            
            # Check message type
            if 'tick' in data:
                return self._process_deriv_tick(data, symbol)
                
            elif 'ohlc' in data:
                return self._process_deriv_ohlc(data, symbol)
                
            elif 'history' in data:
                return self._process_deriv_history(data, symbol)
                
            elif 'candles' in data:
                return self._process_deriv_candles(data, symbol)
                
            elif 'ticks' in data:
                return self._process_deriv_ticks(data, symbol)
                
            elif 'error' in data:
                logger.warning(f"Deriv error message: {data['error'].get('message', 'Unknown error')}")
                return None
                
            elif 'ping' in data:
                # Just a ping message, no processing needed
                return None
                
            else:
                logger.debug(f"Unrecognized Deriv message format: {message[:100]}...")
                return None
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in Deriv message: {message[:100]}...")
            return None
            
        except Exception as e:
            logger.error(f"Error processing Deriv message: {str(e)}")
            logger.debug(f"Problematic message: {message[:100]}...")
            return None
    
    def _process_binance_depth(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Process Binance depth update."""
        last_update_id = data.get('u', 0)
        first_update_id = data.get('U', 0)
        event_time = data.get('E', int(time.time() * 1000))
        
        # Check for sequence gaps
        cache_key = f"binance:{symbol}:depth"
        if cache_key in self.last_sequence:
            last_seq = self.last_sequence[cache_key]
            if first_update_id > last_seq + 1:
                logger.warning(f"Sequence gap in Binance {symbol} depth: {last_seq} -> {first_update_id}")
                # We need to request a new snapshot as our data is stale
                return {
                    'type': 'depth_gap',
                    'symbol': symbol,
                    'exchange': 'binance',
                    'timestamp': event_time,
                    'last_sequence': last_seq,
                    'first_update_id': first_update_id,
                    'require_snapshot': True
                }
        
        # Update last sequence
        self.last_sequence[cache_key] = last_update_id
        
        # Update order book if we have a snapshot
        if cache_key in self.orderbook_cache:
            book = self.orderbook_cache[cache_key]
            
            # Verify sequence numbers
            if book['last_update_id'] < first_update_id - 1:
                logger.warning(f"Orderbook snapshot is stale for {symbol}, requesting new one")
                return {
                    'type': 'depth_stale',
                    'symbol': symbol,
                    'exchange': 'binance',
                    'timestamp': event_time,
                    'require_snapshot': True
                }
            
            # Apply updates to the cached order book
            for bid in data.get('b', []):
                price, qty = bid
                if float(qty) == 0:
                    if price in book['bids']:
                        del book['bids'][price]
                else:
                    book['bids'][price] = qty
            
            for ask in data.get('a', []):
                price, qty = ask
                if float(qty) == 0:
                    if price in book['asks']:
                        del book['asks'][price]
                else:
                    book['asks'][price] = qty
            
            # Update last update ID
            book['last_update_id'] = last_update_id
            
            # Calculate mid price and spread
            bids = sorted([(float(p), float(q)) for p, q in book['bids'].items()], key=lambda x: -x[0])
            asks = sorted([(float(p), float(q)) for p, q in book['asks'].items()], key=lambda x: x[0])
            
            if bids and asks:
                best_bid = bids[0][0]
                best_ask = asks[0][0]
                mid_price = (best_bid + best_ask) / 2
                spread = best_ask - best_bid
                spread_pct = (spread / mid_price) * 100
            else:
                mid_price = 0
                spread = 0
                spread_pct = 0
            
            # Calculate order book imbalance
            total_bid_value = sum(p * q for p, q in bids[:10])
            total_ask_value = sum(p * q for p, q in asks[:10])
            if total_bid_value + total_ask_value > 0:
                book_imbalance = (total_bid_value - total_ask_value) / (total_bid_value + total_ask_value)
            else:
                book_imbalance = 0
            
            # Build standardized format
            processed = {
                'type': 'depth',
                'symbol': symbol,
                'exchange': 'binance',
                'timestamp': event_time,
                'last_update_id': last_update_id,
                'first_update_id': first_update_id,
                'bids': [[float(price), float(qty)] for price, qty in data.get('b', [])],
                'asks': [[float(price), float(qty)] for price, qty in data.get('a', [])],
                'mid_price': mid_price,
                'spread': spread,
                'spread_pct': spread_pct,
                'book_imbalance': book_imbalance,
                'best_bid': best_bid if bids else None,
                'best_ask': best_ask if asks else None,
                'depth_bid_usd': total_bid_value,
                'depth_ask_usd': total_ask_value
            }
        else:
            # We don't have a snapshot yet
            # Build standardized format without additional metrics
            processed = {
                'type': 'depth',
                'symbol': symbol,
                'exchange': 'binance',
                'timestamp': event_time,
                'last_update_id': last_update_id,
                'first_update_id': first_update_id,
                'bids': [[float(price), float(qty)] for price, qty in data.get('b', [])],
                'asks': [[float(price), float(qty)] for price, qty in data.get('a', [])],
                'require_snapshot': True
            }
        
        # Check for significant imbalances or anomalies
        if self.detect_anomalies and 'book_imbalance' in processed:
            imbalance = processed['book_imbalance']
            if abs(imbalance) > settings.IMBALANCE_THRESHOLD:
                processed['anomaly'] = {
                    'type': 'book_imbalance',
                    'value': imbalance,
                    'threshold': settings.IMBALANCE_THRESHOLD,
                    'direction': 'buy' if imbalance > 0 else 'sell',
                    'severity': min(abs(imbalance) / settings.IMBALANCE_THRESHOLD, 1.0)
                }
        
        return processed
    
    def _process_binance_orderbook_snapshot(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Process Binance orderbook snapshot."""
        last_update_id = data.get('lastUpdateId', 0)
        
        # Save in cache for later updates
        cache_key = f"binance:{symbol}:depth"
        self.orderbook_cache[cache_key] = {
            'last_update_id': last_update_id,
            'bids': {price: qty for price, qty in data.get('bids', [])},
            'asks': {price: qty for price, qty in data.get('asks', [])}
        }
        
        # Update last sequence
        self.last_sequence[cache_key] = last_update_id
        
        # Calculate mid price and spread
        bids = sorted([(float(p), float(q)) for p, q in data.get('bids', [])], key=lambda x: -x[0])
        asks = sorted([(float(p), float(q)) for p, q in data.get('asks', [])], key=lambda x: x[0])
        
        if bids and asks:
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            spread_pct = (spread / mid_price) * 100
        else:
            mid_price = 0
            spread = 0
            spread_pct = 0
            best_bid = None
            best_ask = None
        
        # Calculate order book imbalance
        total_bid_value = sum(p * q for p, q in bids[:10])
        total_ask_value = sum(p * q for p, q in asks[:10])
        if total_bid_value + total_ask_value > 0:
            book_imbalance = (total_bid_value - total_ask_value) / (total_bid_value + total_ask_value)
        else:
            book_imbalance = 0
        
        # Build standardized format
        processed = {
            'type': 'orderbook',
            'symbol': symbol,
            'exchange': 'binance',
            'timestamp': int(time.time() * 1000),
            'last_update_id': last_update_id,
            'bids': [[float(price), float(qty)] for price, qty in data.get('bids', [])],
            'asks': [[float(price), float(qty)] for price, qty in data.get('asks', [])],
            'is_snapshot': True,
            'mid_price': mid_price,
            'spread': spread,
            'spread_pct': spread_pct,
            'book_imbalance': book_imbalance,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'depth_bid_usd': total_bid_value,
            'depth_ask_usd': total_ask_value
        }
        
        # Create a proper market data model
        order_book = OrderBook(
            symbol=symbol,
            exchange='binance',
            timestamp=int(time.time() * 1000),
            bids=[[float(price), float(qty)] for price, qty in data.get('bids', [])[:20]],
            asks=[[float(price), float(qty)] for price, qty in data.get('asks', [])[:20]],
            last_update_id=last_update_id,
            mid_price=mid_price,
            spread=spread,
            book_imbalance=book_imbalance
        )
        
        # Add model to the processed data
        processed['model'] = order_book
        
        return processed
    
    def _process_binance_trade(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Process Binance trade update."""
        event_time = data.get('E', int(time.time() * 1000))
        price = float(data.get('p', 0))
        quantity = float(data.get('q', 0))
        trade_time = data.get('T', 0)
        is_buyer_maker = data.get('m', False)
        
        # Track statistics for anomaly detection
        cache_key = f"binance:{symbol}:trade"
        if cache_key not in self.volume_ma:
            self.volume_ma[cache_key] = quantity
        else:
            # Exponential moving average of volume
            self.volume_ma[cache_key] = 0.9 * self.volume_ma[cache_key] + 0.1 * quantity
        
        # Calculate volume anomaly score
        volume_ratio = quantity / self.volume_ma[cache_key] if self.volume_ma[cache_key] > 0 else 1
        is_volume_anomaly = volume_ratio > settings.TRADE_VOLUME_ANOMALY_THRESHOLD
        
        # Calculate VWAP if enabled
        vwap = None
        if self.calculate_vwap:
            if cache_key not in self.processed_cache:
                self.processed_cache[cache_key] = {
                    'vwap_total_value': price * quantity,
                    'vwap_total_volume': quantity,
                    'last_price': price
                }
            else:
                cache = self.processed_cache[cache_key]
                cache['vwap_total_value'] += price * quantity
                cache['vwap_total_volume'] += quantity
                cache['last_price'] = price
                
                if cache['vwap_total_volume'] > 0:
                    vwap = cache['vwap_total_value'] / cache['vwap_total_volume']
        
        # Build standardized format
        processed = {
            'type': 'trade',
            'symbol': symbol,
            'exchange': 'binance',
            'timestamp': event_time,
            'trade_id': data.get('t', 0),
            'price': price,
            'quantity': quantity,
            'value': price * quantity,
            'is_buyer_maker': is_buyer_maker,
            'trade_time': trade_time,
            'side': 'sell' if is_buyer_maker else 'buy',
            'volume_ratio': volume_ratio
        }
        
        # Add VWAP if available
        if vwap:
            processed['vwap'] = vwap
            # Calculate price relative to VWAP
            processed['price_to_vwap'] = price / vwap if vwap > 0 else 1
        
        # Flag anomalies
        if self.detect_anomalies and is_volume_anomaly:
            processed['anomaly'] = {
                'type': 'large_trade',
                'value': volume_ratio,
                'threshold': settings.TRADE_VOLUME_ANOMALY_THRESHOLD,
                'direction': 'sell' if is_buyer_maker else 'buy',
                'severity': min(volume_ratio / settings.TRADE_VOLUME_ANOMALY_THRESHOLD, 1.0)
            }
        
        # Create a proper market data model
        trade = Trade(
            symbol=symbol,
            exchange='binance',
            timestamp=event_time,
            price=price,
            quantity=quantity,
            is_buyer_maker=is_buyer_maker,
            trade_id=data.get('t', 0),
            side='sell' if is_buyer_maker else 'buy'
        )
        
        # Add model to the processed data
        processed['model'] = trade
        
        return processed
    
    def _process_binance_kline(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Process Binance kline update."""
        event_time = data.get('E', int(time.time() * 1000))
        k = data.get('k', {})
        
        interval = k.get('i', '1m')
        open_time = k.get('t', 0)
        close_time = k.get('T', 0)
        open_price = float(k.get('o', 0))
        high_price = float(k.get('h', 0))
        low_price = float(k.get('l', 0))
        close_price = float(k.get('c', 0))
        volume = float(k.get('v', 0))
        quote_volume = float(k.get('q', 0))
        is_closed = k.get('x', False)
        
        # Calculate additional metrics
        price_change = close_price - open_price
        price_change_pct = (price_change / open_price * 100) if open_price > 0 else 0
        
        # For completed candles, check for anomalies
        anomaly = None
        if is_closed and self.detect_anomalies:
            # Check for abnormal volatility
            true_range = max(high_price - low_price, 
                            abs(high_price - close_price), 
                            abs(low_price - close_price))
            volatility_pct = (true_range / close_price * 100) if close_price > 0 else 0
            
            # Cache key for this symbol and timeframe
            vol_key = f"binance:{symbol}:{interval}:volatility"
            
            # Initialize or update volatility ma
            if vol_key not in self.price_volatility:
                self.price_volatility[vol_key] = volatility_pct
            else:
                # Exponential moving average of volatility
                self.price_volatility[vol_key] = 0.9 * self.price_volatility[vol_key] + 0.1 * volatility_pct
            
            # Check if current volatility is abnormally high
            vol_ratio = volatility_pct / self.price_volatility[vol_key] if self.price_volatility[vol_key] > 0 else 1
            is_volatility_anomaly = vol_ratio > settings.VOLATILITY_ANOMALY_THRESHOLD
            
            if is_volatility_anomaly:
                anomaly = {
                    'type': 'high_volatility',
                    'value': vol_ratio,
                    'threshold': settings.VOLATILITY_ANOMALY_THRESHOLD,
                    'direction': 'up' if price_change > 0 else 'down',
                    'severity': min(vol_ratio / settings.VOLATILITY_ANOMALY_THRESHOLD, 1.0)
                }
        
        # Build standardized format
        processed = {
            'type': 'kline',
            'symbol': symbol,
            'exchange': 'binance',
            'timestamp': event_time,
            'interval': interval,
            'open_time': open_time,
            'close_time': close_time,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'quote_volume': quote_volume,
            'num_trades': k.get('n', 0),
            'is_closed': is_closed,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'taker_buy_volume': float(k.get('V', 0)),
            'taker_buy_quote_volume': float(k.get('Q', 0))
        }
        
        # Add anomaly if detected
        if anomaly:
            processed['anomaly'] = anomaly
        
        # Create a proper market data model
        candle = Candle(
            symbol=symbol,
            exchange='binance',
            timestamp=event_time,
            interval=interval,
            open_time=open_time,
            close_time=close_time,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            is_closed=is_closed
        )
        
        # Add model to the processed data
        processed['model'] = candle
        
        return processed
    
    def _process_binance_ticker(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Process Binance ticker update."""
        event_time = data.get('E', int(time.time() * 1000))
        
        last_price = float(data.get('c', 0))
        bid_price = float(data.get('b', 0))
        ask_price = float(data.get('a', 0))
        
        # Build standardized format
        processed = {
            'type': 'ticker',
            'symbol': symbol,
            'exchange': 'binance',
            'timestamp': event_time,
            'price_change': float(data.get('p', 0)),
            'price_change_percent': float(data.get('P', 0)),
            'weighted_avg_price': float(data.get('w', 0)),
            'prev_close_price': float(data.get('x', 0)),
            'last_price': last_price,
            'bid_price': bid_price,
            'ask_price': ask_price,
            'open_price': float(data.get('o', 0)),
            'high_price': float(data.get('h', 0)),
            'low_price': float(data.get('l', 0)),
            'volume': float(data.get('v', 0)),
            'quote_volume': float(data.get('q', 0)),
            'open_time': data.get('O', 0),
            'close_time': data.get('C', 0),
            'first_trade_id': data.get('F', 0),
            'last_trade_id': data.get('L', 0),
            'trade_count': data.get('n', 0)
        }
        
        # Create a proper market data model
        ticker = Ticker(
            symbol=symbol,
            exchange='binance',
            timestamp=event_time,
            last_price=last_price,
            bid_price=bid_price,
            ask_price=ask_price,
            high_price=float(data.get('h', 0)),
            low_price=float(data.get('l', 0)),
            volume=float(data.get('v', 0)),
            price_change_percent=float(data.get('P', 0))
        )
        
        # Add model to the processed data
        processed['model'] = ticker
        
        return processed
    
    def _process_binance_liquidation(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Process Binance liquidation event."""
        event_time = data.get('E', int(time.time() * 1000))
        
        # Extract liquidation details
        o = data.get('o', {})
        symbol = o.get('s', symbol)
        side = o.get('S', '').lower()
        price = float(o.get('p', 0))
        quantity = float(o.get('q', 0))
        
        # Build standardized format
        processed = {
            'type': 'liquidation',
            'symbol': symbol,
            'exchange': 'binance',
            'timestamp': event_time,
            'side': side,
            'price': price,
            'quantity': quantity,
            'value': price * quantity,
            'order_type': o.get('o', '').lower(),
            'time_in_force': o.get('f', '').lower(),
            'order_status': o.get('X', '').lower()
        }
        
        # Flag as a significant market event
        if self.track_liquidations:
            if price * quantity > settings.LARGE_LIQUIDATION_THRESHOLD:
                processed['significant_event'] = {
                    'type': 'large_liquidation',
                    'value': price * quantity,
                    'threshold': settings.LARGE_LIQUIDATION_THRESHOLD,
                    'side': side,
                    'severity': min(price * quantity / settings.LARGE_LIQUIDATION_THRESHOLD, 1.0)
                }
                logger.info(f"Large liquidation detected for {symbol}: {side} {quantity} @ {price} ({price * quantity} USD)")
        
        return processed
    
    def _process_binance_api_response(self, data: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
        """Process Binance API response."""
        # Extract response ID
        response_id = data.get('id', 0)
        result = data.get('result', None)
        
        if result is None:
            logger.warning(f"Empty result in Binance API response: {data}")
            return None
        
        # For now, just pass through the result with minimal processing
        processed = {
            'type': 'api_response',
            'symbol': symbol,
            'exchange': 'binance',
            'timestamp': int(time.time() * 1000),
            'response_id': response_id,
            'result': result
        }
        
        return processed
    
    def _process_deriv_tick(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Process Deriv tick update."""
        tick_data = data.get('tick', {})
        
        # Extract tick data
        symbol = tick_data.get('symbol', symbol)
        quote = float(tick_data.get('quote', 0))
        bid = float(tick_data.get('bid', 0))
        ask = float(tick_data.get('ask', 0))
        
        # Build standardized format
        processed = {
            'type': 'tick',
            'symbol': symbol,
            'exchange': 'deriv',
            'timestamp': int(time.time() * 1000),
            'price': quote,
            'bid': bid,
            'ask': ask,
            'epoch': tick_data.get('epoch', 0),
            'pip_size': tick_data.get('pip_size', 0)
        }
        
        # Create a proper market data model
        ticker = Ticker(
            symbol=symbol,
            exchange='deriv',
            timestamp=int(time.time() * 1000),
            last_price=quote,
            bid_price=bid,
            ask_price=ask,
            high_price=None,
            low_price=None,
            volume=None,
            price_change_percent=None
        )
        
        # Add model to the processed data
        processed['model'] = ticker
        
        return processed
    
    def _process_deriv_ohlc(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Process Deriv OHLC update."""
        ohlc_data = data.get('ohlc', {})
        
        # Extract OHLC data
        symbol = ohlc_data.get('symbol', symbol)
        granularity = int(ohlc_data.get('granularity', 60))
        interval = f"{granularity}s"  # Format as string (60s, 300s, etc.)
        
        open_price = float(ohlc_data.get('open', 0))
        high_price = float(ohlc_data.get('high', 0))
        low_price = float(ohlc_data.get('low', 0))
        close_price = float(ohlc_data.get('close', 0))
        open_time = ohlc_data.get('open_time', 0) * 1000  # Convert to milliseconds
        epoch = ohlc_data.get('epoch', 0)
        
        # Calculate additional metrics
        price_change = close_price - open_price
        price_change_pct = (price_change / open_price * 100) if open_price > 0 else 0
        
        # Build standardized format
        processed = {
            'type': 'ohlc',
            'symbol': symbol,
            'exchange': 'deriv',
            'timestamp': int(time.time() * 1000),
            'interval': interval,
            'granularity': granularity,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'open_time': open_time,
            'epoch': epoch,
            'price_change': price_change,
            'price_change_pct': price_change_pct
        }
        
        # Create a proper market data model
        candle = Candle(
            symbol=symbol,
            exchange='deriv',
            timestamp=int(time.time() * 1000),
            interval=interval,
            open_time=open_time,
            close_time=int(time.time() * 1000),  # Use current time as close time
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=None,  # Deriv doesn't provide volume
            is_closed=False  # Real-time update, not closed yet
        )
        
        # Add model to the processed data
        processed['model'] = candle
        
        return processed
    
    def _process_deriv_history(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Process Deriv history response."""
        history = data.get('history', {})
        
        # Extract symbol from request id
        req_id = data.get('req_id', '')
        if ':' in req_id:
            symbol = req_id.split(':')[0]
        
        # Build standardized format
        processed = {
            'type': 'history',
            'symbol': symbol,
            'exchange': 'deriv',
            'timestamp': int(time.time() * 1000),
            'times': history.get('times', []),
            'prices': history.get('prices', []),
            'req_id': req_id
        }
        
        return processed
    
    def _process_deriv_candles(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Process Deriv candles response."""
        candles = data.get('candles', [])
        
        # Extract symbol and granularity from request id
        req_id = data.get('req_id', '')
        parts = req_id.split(':')
        if len(parts) >= 2:
            symbol = parts[0]
            granularity = parts[1] if len(parts) > 1 else '60'  # Default to 60s
        else:
            granularity = '60'  # Default granularity
        
        # Convert to standardized format
        processed_candles = []
        for candle in candles:
            # Convert candle data
            open_time = int(candle.get('epoch', 0)) * 1000  # Convert to ms
            close_time = open_time + (int(granularity) * 1000)  # Estimate close time
            open_price = float(candle.get('open', 0))
            high_price = float(candle.get('high', 0))
            low_price = float(candle.get('low', 0))
            close_price = float(candle.get('close', 0))
            
            # Create candle object
            processed_candle = {
                'symbol': symbol,
                'exchange': 'deriv',
                'timestamp': open_time,
                'interval': f"{granularity}s",
                'open_time': open_time,
                'close_time': close_time,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'is_closed': True  # Historical candles are closed
            }
            
            processed_candles.append(processed_candle)
        
        # Build standardized format for the whole response
        processed = {
            'type': 'candles',
            'symbol': symbol,
            'exchange': 'deriv',
            'timestamp': int(time.time() * 1000),
            'granularity': granularity,
            'candles': processed_candles,
            'req_id': req_id
        }
        
        return processed
    
    def _process_deriv_ticks(self, data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Process Deriv ticks response."""
        ticks = data.get('ticks', [])
        
        # Extract symbol from request id
        req_id = data.get('req_id', '')
        if ':' in req_id:
            symbol = req_id.split(':')[0]
        
        # Convert to standardized format
        processed_ticks = []
        for tick in ticks:
            timestamp = int(tick.get('epoch', 0)) * 1000  # Convert to ms
            
            processed_tick = {
                'symbol': symbol,
                'exchange': 'deriv',
                'timestamp': timestamp,
                'price': float(tick.get('quote', 0)),
                'epoch': tick.get('epoch', 0)
            }
            
            processed_ticks.append(processed_tick)
        
        # Build standardized format for the whole response
        processed = {
            'type': 'ticks',
            'symbol': symbol,
            'exchange': 'deriv',
            'timestamp': int(time.time() * 1000),
            'ticks': processed_ticks,
            'req_id': req_id
        }
        
        return processed
    
    def calculate_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance statistics for the processor.
        
        Returns:
            Dictionary of performance statistics by data type
        """
        stats = {}
        
        for data_type, timings in self.timing_stats.items():
            if not timings:
                stats[data_type] = {
                    'avg_ms': 0,
                    'max_ms': 0,
                    'min_ms': 0,
                    'count': 0
                }
                continue
                
            # Calculate statistics
            avg_time = sum(timings) / len(timings)
            max_time = max(timings)
            min_time = min(timings)
            
            stats[data_type] = {
                'avg_ms': avg_time * 1000,  # Convert to ms
                'max_ms': max_time * 1000,
                'min_ms': min_time * 1000,
                'count': len(timings)
            }
            
            # Reset timing stats to avoid unbounded growth
            if len(timings) > 1000:
                # Keep most recent 100 timings
                self.timing_stats[data_type] = timings[-100:]
        
        return stats
    
    def clear_caches(self) -> None:
        """Clear all internal caches."""
        self.orderbook_cache.clear()
        self.processed_cache.clear()
        self.last_sequence.clear()
        logger.info("Cleared all data processor caches")
    
    def reset_performance_stats(self) -> None:
        """Reset performance timing statistics."""
        for data_type in self.timing_stats:
            self.timing_stats[data_type] = []
        logger.info("Reset performance statistics")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status information for the processor.
        
        Returns:
            Dictionary with health status information
        """
        # Calculate cache sizes
        orderbook_cache_size = len(self.orderbook_cache)
        processed_cache_size = len(self.processed_cache)
        
        # Get performance stats
        perf_stats = self.calculate_performance_stats()
        
        return {
            'status': 'healthy',
            'timestamp': int(time.time() * 1000),
            'cache_sizes': {
                'orderbook_cache': orderbook_cache_size,
                'processed_cache': processed_cache_size
            },
            'performance_stats': perf_stats,
            'anomaly_detection_enabled': self.detect_anomalies,
            'vwap_calculation_enabled': self.calculate_vwap,
            'liquidation_tracking_enabled': self.track_liquidations
        }
    
    def process_binance_candles(self, candles: List[List[Any]], symbol: str, interval: str) -> List[Dict[str, Any]]:
        """
        Process Binance candlestick data from REST API.
        
        Args:
            candles: List of candlestick data
            symbol: Trading symbol
            interval: Candlestick interval
            
        Returns:
            List of processed candlestick data
        """
        processed_candles = []
        
        for candle_data in candles:
            if len(candle_data) < 6:
                logger.warning(f"Invalid candle data format: {candle_data}")
                continue
                
            try:
                open_time = candle_data[0]
                open_price = float(candle_data[1])
                high_price = float(candle_data[2])
                low_price = float(candle_data[3])
                close_price = float(candle_data[4])
                volume = float(candle_data[5])
                
                # Optional fields
                close_time = candle_data[6] if len(candle_data) > 6 else open_time + self._interval_to_ms(interval)
                quote_volume = float(candle_data[7]) if len(candle_data) > 7 else None
                num_trades = candle_data[8] if len(candle_data) > 8 else None
                
                # Calculate additional metrics
                price_change = close_price - open_price
                price_change_pct = (price_change / open_price * 100) if open_price > 0 else 0
                
                # Create standardized candle object
                processed_candle = {
                    'symbol': symbol,
                    'exchange': 'binance',
                    'timestamp': open_time,
                    'interval': interval,
                    'open_time': open_time,
                    'close_time': close_time,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume,
                    'is_closed': True,  # Historical data is always closed
                    'price_change': price_change,
                    'price_change_pct': price_change_pct
                }
                
                # Add optional fields if available
                if quote_volume is not None:
                    processed_candle['quote_volume'] = quote_volume
                
                if num_trades is not None:
                    processed_candle['num_trades'] = num_trades
                
                # Create a proper market data model
                candle = Candle(
                    symbol=symbol,
                    exchange='binance',
                    timestamp=open_time,
                    interval=interval,
                    open_time=open_time,
                    close_time=close_time,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    is_closed=True
                )
                
                # Add model to the processed data
                processed_candle['model'] = candle
                
                processed_candles.append(processed_candle)
                
            except (IndexError, ValueError) as e:
                logger.error(f"Error processing candle data: {str(e)}")
                continue
        
        return processed_candles
    
    def process_deriv_candles(self, candles: List[Dict[str, Any]], symbol: str, interval: str) -> List[Dict[str, Any]]:
        """
        Process Deriv candlestick data from REST API.
        
        Args:
            candles: List of candlestick data
            symbol: Trading symbol
            interval: Candlestick interval
            
        Returns:
            List of processed candlestick data
        """
        processed_candles = []
        
        # Convert interval to seconds
        interval_seconds = self._deriv_interval_to_seconds(interval)
        interval_str = f"{interval_seconds}s"
        
        for candle_data in candles:
            try:
                epoch = candle_data.get('epoch', 0)
                open_time = epoch * 1000  # Convert to ms
                close_time = open_time + (interval_seconds * 1000)
                
                open_price = float(candle_data.get('open', 0))
                high_price = float(candle_data.get('high', 0))
                low_price = float(candle_data.get('low', 0))
                close_price = float(candle_data.get('close', 0))
                
                # Calculate additional metrics
                price_change = close_price - open_price
                price_change_pct = (price_change / open_price * 100) if open_price > 0 else 0
                
                # Create standardized candle object
                processed_candle = {
                    'symbol': symbol,
                    'exchange': 'deriv',
                    'timestamp': open_time,
                    'interval': interval_str,
                    'open_time': open_time,
                    'close_time': close_time,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'is_closed': True,  # Historical data is always closed
                    'price_change': price_change,
                    'price_change_pct': price_change_pct,
                    'epoch': epoch
                }
                
                # Create a proper market data model
                candle = Candle(
                    symbol=symbol,
                    exchange='deriv',
                    timestamp=open_time,
                    interval=interval_str,
                    open_time=open_time,
                    close_time=close_time,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=None,  # Deriv doesn't provide volume
                    is_closed=True
                )
                
                # Add model to the processed data
                processed_candle['model'] = candle
                
                processed_candles.append(processed_candle)
                
            except (KeyError, ValueError) as e:
                logger.error(f"Error processing Deriv candle data: {str(e)}")
                continue
        
        return processed_candles
    
    def process_binance_ticker(self, ticker: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """
        Process Binance ticker data from REST API.
        
        Args:
            ticker: Ticker data
            symbol: Trading symbol
            
        Returns:
            Processed ticker data
        """
        try:
            # Extract ticker data
            last_price = float(ticker.get('lastPrice', 0))
            bid_price = float(ticker.get('bidPrice', 0))
            ask_price = float(ticker.get('askPrice', 0))
            high_price = float(ticker.get('highPrice', 0))
            low_price = float(ticker.get('lowPrice', 0))
            volume = float(ticker.get('volume', 0))
            price_change = float(ticker.get('priceChange', 0))
            price_change_pct = float(ticker.get('priceChangePercent', 0))
            
            # Build standardized format
            processed = {
                'type': 'ticker',
                'symbol': symbol,
                'exchange': 'binance',
                'timestamp': int(time.time() * 1000),
                'last_price': last_price,
                'bid_price': bid_price,
                'ask_price': ask_price,
                'high_price': high_price,
                'low_price': low_price,
                'volume': volume,
                'price_change': price_change,
                'price_change_percent': price_change_pct,
                'weighted_avg_price': float(ticker.get('weightedAvgPrice', 0)),
                'open_price': float(ticker.get('openPrice', 0)),
                'quote_volume': float(ticker.get('quoteVolume', 0)),
                'source': 'rest'
            }
            
            # Create a proper market data model
            ticker_model = Ticker(
                symbol=symbol,
                exchange='binance',
                timestamp=int(time.time() * 1000),
                last_price=last_price,
                bid_price=bid_price,
                ask_price=ask_price,
                high_price=high_price,
                low_price=low_price,
                volume=volume,
                price_change_percent=price_change_pct
            )
            
            # Add model to the processed data
            processed['model'] = ticker_model
            
            return processed
            
        except (KeyError, ValueError) as e:
            logger.error(f"Error processing Binance ticker data: {str(e)}")
            return {
                'type': 'ticker',
                'symbol': symbol,
                'exchange': 'binance',
                'timestamp': int(time.time() * 1000),
                'error': str(e),
                'source': 'rest'
            }
    
    def process_deriv_ticker(self, ticker: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """
        Process Deriv ticker data from REST API.
        
        Args:
            ticker: Ticker data
            symbol: Trading symbol
            
        Returns:
            Processed ticker data
        """
        try:
            # Extract ticker data
            quote = float(ticker.get('quote', 0))
            bid = float(ticker.get('bid', 0)) if 'bid' in ticker else None
            ask = float(ticker.get('ask', 0)) if 'ask' in ticker else None
            
            # Build standardized format
            processed = {
                'type': 'ticker',
                'symbol': symbol,
                'exchange': 'deriv',
                'timestamp': int(time.time() * 1000),
                'price': quote,
                'source': 'rest'
            }
            
            # Add bid/ask if available
            if bid is not None:
                processed['bid'] = bid
            
            if ask is not None:
                processed['ask'] = ask
            
            # Add epoch if available
            if 'epoch' in ticker:
                processed['epoch'] = ticker['epoch']
            
            # Create a proper market data model
            ticker_model = Ticker(
                symbol=symbol,
                exchange='deriv',
                timestamp=int(time.time() * 1000),
                last_price=quote,
                bid_price=bid,
                ask_price=ask,
                high_price=None,
                low_price=None,
                volume=None,
                price_change_percent=None
            )
            
            # Add model to the processed data
            processed['model'] = ticker_model
            
            return processed
            
        except (KeyError, ValueError) as e:
            logger.error(f"Error processing Deriv ticker data: {str(e)}")
            return {
                'type': 'ticker',
                'symbol': symbol,
                'exchange': 'deriv',
                'timestamp': int(time.time() * 1000),
                'error': str(e),
                'source': 'rest'
            }
    
    def _interval_to_ms(self, interval: str) -> int:
        """
        Convert a Binance interval string to milliseconds.
        
        Args:
            interval: Interval string (e.g., '1m', '1h', '1d')
            
        Returns:
            Milliseconds
        """
        # Extract number and unit
        if len(interval) < 2:
            return 60000  # Default to 1m
            
        try:
            num = int(interval[:-1])
            unit = interval[-1]
            
            if unit == 'm':
                return num * 60 * 1000
            elif unit == 'h':
                return num * 60 * 60 * 1000
            elif unit == 'd':
                return num * 24 * 60 * 60 * 1000
            elif unit == 'w':
                return num * 7 * 24 * 60 * 60 * 1000
            else:
                return 60000  # Default to 1m
                
        except (ValueError, IndexError):
            return 60000  # Default to 1m
    
    def _deriv_interval_to_seconds(self, interval: str) -> int:
        """
        Convert a Deriv interval string to seconds.
        
        Args:
            interval: Interval string (e.g., 'M1', 'H1', 'D1')
            
        Returns:
            Seconds
        """
        # Extract unit and number
        if len(interval) < 2:
            return 60  # Default to 1m
            
        try:
            unit = interval[0].upper()
            num = int(interval[1:])
            
            if unit == 'M':
                return num * 60
            elif unit == 'H':
                return num * 60 * 60
            elif unit == 'D':
                return num * 24 * 60 * 60
            else:
                return 60  # Default to 1m
                
        except (ValueError, IndexError):
            return 60  # Default to 1m
