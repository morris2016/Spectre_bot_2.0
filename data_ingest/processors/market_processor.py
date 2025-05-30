#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Market Data Processor

This module implements the Market Data Processor, which processes
market data such as OHLCV, order book, and trades.
"""

import time
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

from data_ingest.processor import DataProcessor
from common.exceptions import DataValidationError, DataProcessorError


class MarketDataProcessor(DataProcessor):
    """Processes market data (OHLCV, order book, trades)."""
    
    def __init__(self, config, logger=None):
        """Initialize the market data processor."""
        super().__init__(config, logger=logger)
        self.timeframes = config.get("timeframes", ["1m", "5m", "15m", "1h", "4h", "1d"])
        self.indicators = config.get("indicators", {})
        
    def initialize_validation_rules(self):
        """Initialize validation rules for market data."""
        self.validation_rules = {
            "type": {"required": True, "type": str},
            "exchange": {"required": True, "type": str},
            "symbol": {"required": True, "type": str},
            "timestamp": {"required": True, "type": (int, float)},
        }
        
        # Add specific validation rules based on data type
        data_type_rules = {
            "ohlcv": {
                "open": {"required": True, "type": (int, float)},
                "high": {"required": True, "type": (int, float)},
                "low": {"required": True, "type": (int, float)},
                "close": {"required": True, "type": (int, float)},
                "volume": {"required": True, "type": (int, float)},
                "timeframe": {"required": True, "type": str},
            },
            "order_book": {
                "bids": {"required": True, "type": list},
                "asks": {"required": True, "type": list},
            },
            "trade": {
                "price": {"required": True, "type": (int, float)},
                "amount": {"required": True, "type": (int, float)},
                "side": {"required": True, "type": str},
            }
        }
        
        # Update validation rules based on data type being processed
        for data_type, rules in data_type_rules.items():
            for field, rule in rules.items():
                self.validation_rules[f"{data_type}.{field}"] = rule
    
    async def _process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data."""
        data_type = data.get("type")
        
        # Choose processing method based on data type
        if data_type == "ohlcv":
            return await self._process_ohlcv(data)
        elif data_type == "order_book":
            return await self._process_order_book(data)
        elif data_type == "trade":
            return await self._process_trade(data)
        else:
            raise DataProcessorError(f"Unsupported market data type: {data_type}")
    
    async def _process_ohlcv(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process OHLCV (candle) data."""
        try:
            # Extract OHLCV data
            ohlcv = data.get("ohlcv", {})
            timeframe = data.get("timeframe", "1m")
            
            # Calculate basic indicators
            indicators = await self._calculate_indicators(ohlcv, timeframe)
            
            # Create processed data
            processed_data = {
                "type": "processed_ohlcv",
                "exchange": data.get("exchange"),
                "symbol": data.get("symbol"),
                "timestamp": data.get("timestamp"),
                "timeframe": timeframe,
                "ohlcv": ohlcv,
                "indicators": indicators,
                "source_data": data
            }
            
            self.metrics.increment("ohlcv.processed")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing OHLCV data: {str(e)}")
            self.metrics.increment("ohlcv.error")
            raise DataProcessorError(f"Failed to process OHLCV data: {str(e)}")
    
    async def _process_order_book(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process order book data."""
        try:
            # Extract order book data
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            
            # Calculate order book metrics
            bid_prices = [bid[0] for bid in bids]
            ask_prices = [ask[0] for ask in asks]
            bid_volumes = [bid[1] for bid in bids]
            ask_volumes = [ask[1] for ask in asks]
            
            # Calculate order book statistics
            if bid_prices and ask_prices:
                mid_price = (bid_prices[0] + ask_prices[0]) / 2
                spread = ask_prices[0] - bid_prices[0]
                spread_percent = (spread / mid_price) * 100
                bid_sum = sum(bid_volumes)
                ask_sum = sum(ask_volumes)
                bid_ask_ratio = bid_sum / ask_sum if ask_sum > 0 else float('inf')
                
                # Calculate price levels and volume distribution
                bid_levels = {}
                ask_levels = {}
                
                for i, (price, volume) in enumerate(bids):
                    level = i + 1
                    bid_levels[level] = {"price": price, "volume": volume}
                
                for i, (price, volume) in enumerate(asks):
                    level = i + 1
                    ask_levels[level] = {"price": price, "volume": volume}
                
                # Calculate cumulative volumes
                cumulative_bids = {}
                cumulative_volume = 0
                for level, data in bid_levels.items():
                    cumulative_volume += data["volume"]
                    cumulative_bids[level] = {
                        "price": data["price"],
                        "volume": data["volume"],
                        "cumulative_volume": cumulative_volume
                    }
                
                cumulative_asks = {}
                cumulative_volume = 0
                for level, data in ask_levels.items():
                    cumulative_volume += data["volume"]
                    cumulative_asks[level] = {
                        "price": data["price"],
                        "volume": data["volume"],
                        "cumulative_volume": cumulative_volume
                    }
                
                # Create order book metrics
                order_book_metrics = {
                    "mid_price": mid_price,
                    "spread": spread,
                    "spread_percent": spread_percent,
                    "bid_sum": bid_sum,
                    "ask_sum": ask_sum,
                    "bid_ask_ratio": bid_ask_ratio,
                    "bid_levels": len(bids),
                    "ask_levels": len(asks),
                    "top_bid": bid_prices[0] if bid_prices else None,
                    "top_ask": ask_prices[0] if ask_prices else None,
                }
            else:
                order_book_metrics = {}
            
            # Create processed data
            processed_data = {
                "type": "processed_order_book",
                "exchange": data.get("exchange"),
                "symbol": data.get("symbol"),
                "timestamp": data.get("timestamp"),
                "order_book": {
                    "bids": bids,
                    "asks": asks,
                    "bid_levels": bid_levels,
                    "ask_levels": ask_levels,
                    "cumulative_bids": cumulative_bids,
                    "cumulative_asks": cumulative_asks,
                },
                "metrics": order_book_metrics,
                "source_data": data
            }
            
            self.metrics.increment("order_book.processed")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing order book data: {str(e)}")
            self.metrics.increment("order_book.error")
            raise DataProcessorError(f"Failed to process order book data: {str(e)}")
    
    async def _process_trade(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process trade data."""
        try:
            # Extract trade data
            price = data.get("price")
            amount = data.get("amount")
            side = data.get("side")
            
            # Create processed trade data
            processed_data = {
                "type": "processed_trade",
                "exchange": data.get("exchange"),
                "symbol": data.get("symbol"),
                "timestamp": data.get("timestamp"),
                "trade": {
                    "price": price,
                    "amount": amount,
                    "side": side,
                    "value": price * amount
                },
                "source_data": data
            }
            
            self.metrics.increment("trade.processed")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing trade data: {str(e)}")
            self.metrics.increment("trade.error")
            raise DataProcessorError(f"Failed to process trade data: {str(e)}")
    
    async def _calculate_indicators(self, ohlcv, timeframe):
        """Calculate technical indicators for OHLCV data."""
        indicators = {}
        
        try:
            # Create DataFrame for indicator calculation
            if isinstance(ohlcv, dict):
                df = pd.DataFrame({
                    'open': [ohlcv.get('open')],
                    'high': [ohlcv.get('high')],
                    'low': [ohlcv.get('low')],
                    'close': [ohlcv.get('close')],
                    'volume': [ohlcv.get('volume')]
                })
            elif isinstance(ohlcv, list):
                # Assume list of OHLCV values
                df = pd.DataFrame(ohlcv, columns=['open', 'high', 'low', 'close', 'volume'])
            else:
                self.logger.warning(f"Unsupported OHLCV data format: {type(ohlcv)}")
                return indicators
            
            # Calculate basic statistics
            indicators['range'] = df['high'].iloc[-1] - df['low'].iloc[-1]
            indicators['range_percent'] = (indicators['range'] / df['open'].iloc[-1]) * 100
            indicators['body'] = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
            indicators['body_percent'] = (indicators['body'] / df['open'].iloc[-1]) * 100
            
            # Calculate simple moving averages if we have enough data
            if len(df) > 20:
                indicators['sma_5'] = df['close'].rolling(5).mean().iloc[-1]
                indicators['sma_10'] = df['close'].rolling(10).mean().iloc[-1]
                indicators['sma_20'] = df['close'].rolling(20).mean().iloc[-1]
                
                # Calculate RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                indicators['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
                
                # Calculate MACD
                ema_12 = df['close'].ewm(span=12, adjust=False).mean()
                ema_26 = df['close'].ewm(span=26, adjust=False).mean()
                macd_line = ema_12 - ema_26
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                indicators['macd'] = macd_line.iloc[-1]
                indicators['macd_signal'] = signal_line.iloc[-1]
                indicators['macd_histogram'] = macd_line.iloc[-1] - signal_line.iloc[-1]
            
            # Flag potential patterns (very simplified)
            # Bullish engulfing
            if len(df) >= 2:
                prev_body = abs(df['close'].iloc[-2] - df['open'].iloc[-2])
                curr_body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
                bullish_engulfing = (
                    df['open'].iloc[-1] < df['close'].iloc[-2] and
                    df['close'].iloc[-1] > df['open'].iloc[-2] and
                    df['close'].iloc[-1] > df['open'].iloc[-1] and
                    curr_body > prev_body
                )
                indicators['bullish_engulfing'] = bullish_engulfing
                
                # Bearish engulfing
                bearish_engulfing = (
                    df['open'].iloc[-1] > df['close'].iloc[-2] and
                    df['close'].iloc[-1] < df['open'].iloc[-2] and
                    df['close'].iloc[-1] < df['open'].iloc[-1] and
                    curr_body > prev_body
                )
                indicators['bearish_engulfing'] = bearish_engulfing
                
                # Hammer (simplified)
                lower_wick = df['close'].iloc[-1] - df['low'].iloc[-1] if df['close'].iloc[-1] > df['open'].iloc[-1] else df['open'].iloc[-1] - df['low'].iloc[-1]
                upper_wick = df['high'].iloc[-1] - df['close'].iloc[-1] if df['close'].iloc[-1] > df['open'].iloc[-1] else df['high'].iloc[-1] - df['open'].iloc[-1]
                
                hammer = (
                    lower_wick > 2 * indicators['body'] and
                    upper_wick < 0.1 * indicators['body'] and
                    indicators['body'] > 0
                )
                indicators['hammer'] = hammer
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return {}
