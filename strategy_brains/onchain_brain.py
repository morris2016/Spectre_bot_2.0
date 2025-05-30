#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
On-Chain Data Analysis Trading Strategy

This module implements a specialized trading brain that analyzes on-chain
blockchain data to make trading decisions based on network activity, whale
movements, and blockchain metrics.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import logging
import traceback
import requests
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque

# Internal imports
from common.logger import get_logger
from common.utils import calculate_sharpe_ratio, calculate_metrics
from common.constants import TIMEFRAMES, NETWORK_IDS
from common.exceptions import StrategyError, DataFetchError
from feature_service.features.technical import TechnicalFeatures
from strategy_brains.base_brain import BaseBrain
from data_feeds.onchain_feed import OnChainFeed

logger = get_logger("OnChainBrain")

class OnChainBrain(BaseBrain):
    """
    Trading strategy brain that analyzes on-chain blockchain data to make
    trading decisions based on network activity, wallet tracking, and
    blockchain metrics.
    """
    
    def __init__(self, 
                 name: str,
                 exchange: str,
                 symbol: str,
                 timeframe: str,
                 config: Dict[str, Any] = None):
        """
        Initialize the on-chain brain.
        
        Args:
            name: Name of the brain
            exchange: Exchange to trade on (e.g., 'binance')
            symbol: Symbol to trade (e.g., 'BTCUSDT')
            timeframe: Timeframe to use (e.g., '1h')
            config: Configuration dictionary
        """
        super().__init__(name, exchange, symbol, timeframe, config)
        
        # Default configuration
        self.default_config = {
            'network_id': None,  # Default to None, will be derived from symbol
            'data_sources': ['transactions', 'fees', 'addresses', 'whales', 'staking', 'hashrate'],
            'whale_threshold': 1000000,  # USD value to consider a whale
            'sentiment_impact': 0.5,  # Weight of sentiment in signal calculation
            'volume_impact': 0.8,  # Weight of volume changes in signal calculation
            'staking_impact': 0.6,  # Weight of staking metrics in signal calculation
            'network_health_impact': 0.7,  # Weight of network health metrics
            'cache_ttl': 3600,  # Cache time-to-live in seconds
            'whale_watch_addresses': [],  # Specific addresses to track
            'api_keys': {  # API keys for different blockchain data services
                'etherscan': None,
                'blockchair': None,
                'glassnode': None,
                'coinmetrics': None
            },
            'api_rate_limits': {  # Rate limits for API calls
                'default': 1.0,  # Default call per second
                'etherscan': 0.2,  # 5 second interval
                'blockchair': 0.5,  # 2 second interval
                'glassnode': 0.5,
                'coinmetrics': 0.5
            },
            'technical_features': ['rsi', 'macd', 'volume_profile'],
            'signal_threshold': 0.5,  # Threshold for generating a signal
            'model_path': './models/onchain',
            'backtest_data_path': './data/onchain',
            'bullish_metrics': [  # Metrics that indicate bullish sentiment when high
                'active_addresses',
                'new_addresses',
                'transaction_volume',
                'staking_rate', 
                'hashrate',
                'network_growth'
            ],
            'bearish_metrics': [  # Metrics that indicate bearish sentiment when high
                'exchange_inflow',
                'fees',
                'whale_selling_pressure'
            ],
            'signal_normalization': 'sigmoid',  # 'minmax', 'sigmoid', or 'tanh'
            'historical_lookback': 30,  # Days of historical data to use
            'data_aggregation': '1h',  # How to aggregate on-chain data
            'correlation_threshold': 0.3  # Minimum correlation to consider relevant
        }
        
        # Update with provided config
        self.config = {**self.default_config, **(config or {})}
        
        # Initialize on-chain data feed
        self.onchain_feed = OnChainFeed()
        
        # Initialize technical features calculator
        self.technical_features = TechnicalFeatures()
        
        # Auto-detect network if not provided
        if self.config['network_id'] is None:
            self.config['network_id'] = self._detect_network_from_symbol(symbol)
        
        # Initialize cache
        self.cache = {}
        self.cache_timestamps = {}
        
        # Initialize API rate limiting
        self.api_locks = {}
        self.last_api_calls = {}
        for api in self.config['api_rate_limits'].keys():
            self.api_locks[api] = threading.Lock()
            self.last_api_calls[api] = 0
        
        # Initialize signal history
        self.signal_history = deque(maxlen=100)
        self.metric_history = {}
        
        # Create model directory if needed
        os.makedirs(self.config['model_path'], exist_ok=True)
        os.makedirs(self.config['backtest_data_path'], exist_ok=True)
        
        logger.info(f"Initialized {self.__class__.__name__} for {exchange}:{symbol}:{timeframe} "
                   f"on network {self.config['network_id']}")
    
    def _detect_network_from_symbol(self, symbol: str) -> str:
        """
        Detect the appropriate blockchain network from the trading symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            network_id: Detected network ID
        """
        # Extract the base currency from the symbol
        base_currency = symbol.upper().split('USDT')[0].split('USD')[0].split('BTC')[0]
        
        # Map common symbols to networks
        network_mapping = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum', 
            'BNB': 'binance',
            'ADA': 'cardano',
            'SOL': 'solana',
            'DOT': 'polkadot',
            'AVAX': 'avalanche',
            'MATIC': 'polygon',
            'ATOM': 'cosmos',
            'ALGO': 'algorand',
            'NEAR': 'near'
        }
        
        network_id = network_mapping.get(base_currency, 'unknown')
        if network_id == 'unknown':
            logger.warning(f"Could not detect network for symbol {symbol}, using generic API endpoints")
        
        return network_id
    
    def _respect_rate_limit(self, api: str) -> None:
        """
        Ensure API rate limits are respected.
        
        Args:
            api: API name to enforce rate limiting for
        """
        api_name = api if api in self.api_locks else 'default'
        rate_limit = self.config['api_rate_limits'].get(api_name, self.config['api_rate_limits']['default'])
        min_interval = 1.0 / rate_limit
        
        with self.api_locks[api_name]:
            current_time = time.time()
            elapsed = current_time - self.last_api_calls.get(api_name, 0)
            
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            
            self.last_api_calls[api_name] = time.time()
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """
        Get data from cache if available and not expired.
        
        Args:
            key: Cache key
            
        Returns:
            data: Cached data or None
        """
        if key not in self.cache:
            return None
        
        # Check if cache is expired
        timestamp = self.cache_timestamps.get(key, 0)
        if time.time() - timestamp > self.config['cache_ttl']:
            return None
        
        return self.cache[key]
    
    def _set_in_cache(self, key: str, data: Any) -> None:
        """
        Store data in cache.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        self.cache[key] = data
        self.cache_timestamps[key] = time.time()
    
    def _fetch_transaction_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Fetch transaction metrics from the blockchain.
        
        Args:
            start_time: Start time for data fetch
            end_time: End time for data fetch
            
        Returns:
            metrics: Transaction metrics
        """
        cache_key = f"transactions_{self.config['network_id']}_{start_time.isoformat()}_{end_time.isoformat()}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
            
        try:
            # Respect API rate limits
            self._respect_rate_limit('default')
            
            # Fetch data using the on-chain feed
            transaction_data = self.onchain_feed.get_transaction_metrics(
                network=self.config['network_id'],
                start_time=start_time,
                end_time=end_time,
                interval=self.config['data_aggregation']
            )
            
            # Process the data
            metrics = {
                'total_transactions': transaction_data.get('total_transactions', 0),
                'avg_transaction_value': transaction_data.get('avg_transaction_value', 0),
                'transaction_growth': transaction_data.get('transaction_growth', 0),
                'transaction_volume': transaction_data.get('transaction_volume', 0),
                'transaction_fees': transaction_data.get('transaction_fees', 0),
                'timestamp': end_time.isoformat()
            }
            
            # Cache the results
            self._set_in_cache(cache_key, metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching transaction metrics: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return default values
            return {
                'total_transactions': 0,
                'avg_transaction_value': 0,
                'transaction_growth': 0,
                'transaction_volume': 0,
                'transaction_fees': 0,
                'timestamp': end_time.isoformat()
            }
    
    def _fetch_network_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Fetch network health metrics from the blockchain.
        
        Args:
            start_time: Start time for data fetch
            end_time: End time for data fetch
            
        Returns:
            metrics: Network health metrics
        """
        cache_key = f"network_{self.config['network_id']}_{start_time.isoformat()}_{end_time.isoformat()}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
            
        try:
            # Respect API rate limits
            self._respect_rate_limit('default')
            
            # Fetch data using the on-chain feed
            network_data = self.onchain_feed.get_network_health(
                network=self.config['network_id'],
                start_time=start_time,
                end_time=end_time,
                interval=self.config['data_aggregation']
            )
            
            # Process the data
            metrics = {
                'active_nodes': network_data.get('active_nodes', 0),
                'hashrate': network_data.get('hashrate', 0),
                'difficulty': network_data.get('difficulty', 0),
                'block_time': network_data.get('block_time', 0),
                'uncle_rate': network_data.get('uncle_rate', 0),
                'timestamp': end_time.isoformat()
            }
            
            # Cache the results
            self._set_in_cache(cache_key, metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching network metrics: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return default values
            return {
                'active_nodes': 0,
                'hashrate': 0,
                'difficulty': 0,
                'block_time': 0,
                'uncle_rate': 0,
                'timestamp': end_time.isoformat()
            }
    
    def _fetch_address_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Fetch address metrics from the blockchain.
        
        Args:
            start_time: Start time for data fetch
            end_time: End time for data fetch
            
        Returns:
            metrics: Address metrics
        """
        cache_key = f"addresses_{self.config['network_id']}_{start_time.isoformat()}_{end_time.isoformat()}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
            
        try:
            # Respect API rate limits
            self._respect_rate_limit('default')
            
            # Fetch data using the on-chain feed
            address_data = self.onchain_feed.get_address_metrics(
                network=self.config['network_id'],
                start_time=start_time,
                end_time=end_time,
                interval=self.config['data_aggregation']
            )
            
            # Process the data
            metrics = {
                'active_addresses': address_data.get('active_addresses', 0),
                'new_addresses': address_data.get('new_addresses', 0),
                'address_growth': address_data.get('address_growth', 0),
                'concentration': address_data.get('concentration', 0),
                'network_growth': address_data.get('network_growth', 0),
                'timestamp': end_time.isoformat()
            }
            
            # Cache the results
            self._set_in_cache(cache_key, metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching address metrics: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return default values
            return {
                'active_addresses': 0,
                'new_addresses': 0,
                'address_growth': 0,
                'concentration': 0,
                'network_growth': 0,
                'timestamp': end_time.isoformat()
            }
    
    def _fetch_whale_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Fetch whale activity metrics from the blockchain.
        
        Args:
            start_time: Start time for data fetch
            end_time: End time for data fetch
            
        Returns:
            metrics: Whale activity metrics
        """
        cache_key = f"whales_{self.config['network_id']}_{start_time.isoformat()}_{end_time.isoformat()}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
            
        try:
            # Respect API rate limits
            self._respect_rate_limit('default')
            
            # Fetch data using the on-chain feed
            whale_params = {
                'threshold': self.config['whale_threshold'],
                'watch_addresses': self.config['whale_watch_addresses']
            }
            
            whale_data = self.onchain_feed.get_whale_activity(
                network=self.config['network_id'],
                start_time=start_time,
                end_time=end_time,
                interval=self.config['data_aggregation'],
                params=whale_params
            )
            
            # Process the data
            metrics = {
                'whale_transaction_count': whale_data.get('whale_transaction_count', 0),
                'whale_inflow': whale_data.get('whale_inflow', 0),
                'whale_outflow': whale_data.get('whale_outflow', 0),
                'whale_netflow': whale_data.get('whale_netflow', 0),
                'whale_selling_pressure': whale_data.get('whale_selling_pressure', 0),
                'whale_buying_pressure': whale_data.get('whale_buying_pressure', 0),
                'timestamp': end_time.isoformat()
            }
            
            # Cache the results
            self._set_in_cache(cache_key, metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching whale metrics: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return default values
            return {
                'whale_transaction_count': 0,
                'whale_inflow': 0,
                'whale_outflow': 0,
                'whale_netflow': 0,
                'whale_selling_pressure': 0,
                'whale_buying_pressure': 0,
                'timestamp': end_time.isoformat()
            }
    
    def _fetch_staking_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Fetch staking metrics from the blockchain.
        
        Args:
            start_time: Start time for data fetch
            end_time: End time for data fetch
            
        Returns:
            metrics: Staking metrics
        """
        cache_key = f"staking_{self.config['network_id']}_{start_time.isoformat()}_{end_time.isoformat()}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
            
        try:
            # Respect API rate limits
            self._respect_rate_limit('default')
            
            # Fetch data using the on-chain feed
            staking_data = self.onchain_feed.get_staking_metrics(
                network=self.config['network_id'],
                start_time=start_time,
                end_time=end_time,
                interval=self.config['data_aggregation']
            )
            
            # Process the data
            metrics = {
                'staking_total': staking_data.get('staking_total', 0),
                'staking_rate': staking_data.get('staking_rate', 0),
                'avg_staking_duration': staking_data.get('avg_staking_duration', 0),
                'staking_rewards': staking_data.get('staking_rewards', 0),
                'staking_growth': staking_data.get('staking_growth', 0),
                'timestamp': end_time.isoformat()
            }
            
            # Cache the results
            self._set_in_cache(cache_key, metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching staking metrics: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return default values
            return {
                'staking_total': 0,
                'staking_rate': 0,
                'avg_staking_duration': 0,
                'staking_rewards': 0,
                'staking_growth': 0,
                'timestamp': end_time.isoformat()
            }
    
    def _fetch_exchange_flow_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Fetch exchange flow metrics from the blockchain.
        
        Args:
            start_time: Start time for data fetch
            end_time: End time for data fetch
            
        Returns:
            metrics: Exchange flow metrics
        """
        cache_key = f"exchange_flow_{self.config['network_id']}_{start_time.isoformat()}_{end_time.isoformat()}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
            
        try:
            # Respect API rate limits
            self._respect_rate_limit('default')
            
            # Fetch data using the on-chain feed
            exchange_data = self.onchain_feed.get_exchange_flow(
                network=self.config['network_id'],
                start_time=start_time,
                end_time=end_time,
                interval=self.config['data_aggregation']
            )
            
            # Process the data
            metrics = {
                'exchange_inflow': exchange_data.get('exchange_inflow', 0),
                'exchange_outflow': exchange_data.get('exchange_outflow', 0),
                'exchange_netflow': exchange_data.get('exchange_netflow', 0),
                'supply_on_exchanges': exchange_data.get('supply_on_exchanges', 0),
                'exchange_withdrawal_count': exchange_data.get('exchange_withdrawal_count', 0),
                'exchange_deposit_count': exchange_data.get('exchange_deposit_count', 0),
                'timestamp': end_time.isoformat()
            }
            
            # Cache the results
            self._set_in_cache(cache_key, metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching exchange flow metrics: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return default values
            return {
                'exchange_inflow': 0,
                'exchange_outflow': 0,
                'exchange_netflow': 0,
                'supply_on_exchanges': 0,
                'exchange_withdrawal_count': 0,
                'exchange_deposit_count': 0,
                'timestamp': end_time.isoformat()
            }
    
    def _get_all_metrics(self, end_time: datetime) -> Dict[str, Any]:
        """
        Fetch all relevant on-chain metrics.
        
        Args:
            end_time: End time for data fetch
            
        Returns:
            combined_metrics: All fetched metrics combined
        """
        # Define the start time based on lookback period
        start_time = end_time - timedelta(days=self.config['historical_lookback'])
        
        # Initialize pool of workers
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit tasks to fetch different metrics in parallel
            futures = {
                'transactions': executor.submit(self._fetch_transaction_metrics, start_time, end_time),
                'network': executor.submit(self._fetch_network_metrics, start_time, end_time),
                'addresses': executor.submit(self._fetch_address_metrics, start_time, end_time),
                'whales': executor.submit(self._fetch_whale_metrics, start_time, end_time),
                'staking': executor.submit(self._fetch_staking_metrics, start_time, end_time),
                'exchange_flow': executor.submit(self._fetch_exchange_flow_metrics, start_time, end_time)
            }
            
            # Collect results
            combined_metrics = {
                'timestamp': end_time.isoformat(),
                'network_id': self.config['network_id'],
            }
            
            for metric_type, future in futures.items():
                try:
                    metrics = future.result()
                    # Flatten the metrics into the combined dictionary
                    for key, value in metrics.items():
                        if key != 'timestamp':  # Avoid duplicate timestamp
                            combined_metrics[key] = value
                except Exception as e:
                    logger.error(f"Error getting {metric_type} metrics: {str(e)}")
        
        # Update metric history
        for key, value in combined_metrics.items():
            if key not in ['timestamp', 'network_id']:
                if key not in self.metric_history:
                    self.metric_history[key] = deque(maxlen=100)
                self.metric_history[key].append(value)
        
        return combined_metrics
    
    def _calculate_on_chain_signal(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate trading signal from on-chain metrics.
        
        Args:
            metrics: On-chain metrics
            
        Returns:
            signal_data: Signal information
        """
        # Initialize signal components
        bullish_signals = []
        bearish_signals = []
        
        # Calculate normalized metric changes
        metric_changes = {}
        for key in metrics.keys():
            if key in ['timestamp', 'network_id']:
                continue
                
            # If we have historical data, calculate percent change
            if key in self.metric_history and len(self.metric_history[key]) > 1:
                # Get current value and historical values
                current = metrics[key]
                history = list(self.metric_history[key])[:-1]  # Exclude current value
                average = np.mean(history) if history else current
                
                # Calculate percent change
                if average != 0:
                    change = (current - average) / average
                else:
                    change = 0
                    
                metric_changes[key] = change
        
        # Process bullish metrics
        for metric in self.config['bullish_metrics']:
            if metric in metric_changes:
                change = metric_changes[metric]
                # Positive change in bullish metric is good
                if change > 0:
                    bullish_signals.append(change)
                # Negative change in bullish metric is bad
                else:
                    bearish_signals.append(-change)
        
        # Process bearish metrics
        for metric in self.config['bearish_metrics']:
            if metric in metric_changes:
                change = metric_changes[metric]
                # Positive change in bearish metric is bad
                if change > 0:
                    bearish_signals.append(change)
                # Negative change in bearish metric is good
                else:
                    bullish_signals.append(-change)
        
        # Calculate composite signals
        bullish_strength = np.mean(bullish_signals) if bullish_signals else 0
        bearish_strength = np.mean(bearish_signals) if bearish_signals else 0
        
        # Apply sentiment weights
        transaction_metrics = ['total_transactions', 'avg_transaction_value', 'transaction_growth']
        transaction_sentiment = 0
        for metric in transaction_metrics:
            if metric in metric_changes:
                transaction_sentiment += metric_changes[metric]
        transaction_sentiment = transaction_sentiment / len(transaction_metrics) if transaction_metrics else 0
        
        volume_metrics = ['transaction_volume', 'whale_transaction_count']
        volume_sentiment = 0
        for metric in volume_metrics:
            if metric in metric_changes:
                volume_sentiment += metric_changes[metric]
        volume_sentiment = volume_sentiment / len(volume_metrics) if volume_metrics else 0
        
        network_health_metrics = ['active_nodes', 'hashrate', 'network_growth']
        network_sentiment = 0
        for metric in network_health_metrics:
            if metric in metric_changes:
                network_sentiment += metric_changes[metric]
        network_sentiment = network_sentiment / len(network_health_metrics) if network_health_metrics else 0
        
        # Weighted sentiment score
        sentiment_score = (
            transaction_sentiment * self.config['sentiment_impact'] +
            volume_sentiment * self.config['volume_impact'] +
            (bullish_strength - bearish_strength) +
            network_sentiment * self.config['network_health_impact']
        ) / (1 + self.config['sentiment_impact'] + self.config['volume_impact'] + self.config['network_health_impact'])
        
        # Normalize signal
        if self.config['signal_normalization'] == 'minmax':
            # Simple min-max normalization to [-1, 1]
            signal = max(min(sentiment_score, 1.0), -1.0)
        elif self.config['signal_normalization'] == 'sigmoid':
            # Sigmoid function to compress to [-1, 1]
            signal = (2 / (1 + np.exp(-2 * sentiment_score))) - 1
        else:  # tanh
            # Hyperbolic tangent
            signal = np.tanh(sentiment_score)
        
        # Calculate confidence based on signal strength and consistency
        confidence = min(abs(signal) * 2, 1.0)
        
        # Determine discrete signal (-1, 0, 1)
        discrete_signal = 0
        if signal > self.config['signal_threshold']:
            discrete_signal = 1
        elif signal < -self.config['signal_threshold']:
            discrete_signal = -1
        
        # Store signal in history
        self.signal_history.append(signal)
        
        # Return signal data
        signal_data = {
            'timestamp': metrics['timestamp'],
            'raw_signal': sentiment_score,
            'normalized_signal': signal,
            'discrete_signal': discrete_signal,
            'confidence': confidence,
            'bullish_strength': bullish_strength,
            'bearish_strength': bearish_strength,
            'metric_changes': metric_changes,
            'top_bullish_metrics': self._get_top_metrics(metric_changes, self.config['bullish_metrics'], 3, True),
            'top_bearish_metrics': self._get_top_metrics(metric_changes, self.config['bearish_metrics'], 3, False)
        }
        
        return signal_data
    
    def _get_top_metrics(self, 
                         metric_changes: Dict[str, float], 
                         metrics_list: List[str], 
                         n: int = 3, 
                         is_bullish: bool = True) -> List[Dict[str, Any]]:
        """
        Get the top n metrics with the most significant changes.
        
        Args:
            metric_changes: Dictionary of metric changes
            metrics_list: List of metrics to consider
            n: Number of top metrics to return
            is_bullish: Whether to look for bullish or bearish metrics
            
        Returns:
            top_metrics: List of top metrics with their values
        """
        relevant_metrics = []
        for metric in metrics_list:
            if metric in metric_changes:
                change = metric_changes[metric]
                # For bullish metrics, positive change is good
                # For bearish metrics, negative change is good
                if is_bullish:
                    relevant_metrics.append((metric, change))
                else:
                    relevant_metrics.append((metric, -change))
        
        # Sort by change value (descending)
        relevant_metrics.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n
        top_n = relevant_metrics[:n]
        result = []
        for metric, change in top_n:
            result.append({
                'metric': metric,
                'change': change,
                'impact': 'positive' if change > 0 else 'negative'
            })
        
        return result
    
    def _check_for_specific_patterns(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for specific on-chain patterns that might indicate market movements.
        
        Args:
            metrics: On-chain metrics
            
        Returns:
            patterns: Detected patterns
        """
        patterns = {
            'detected': False,
            'pattern_type': None,
            'description': None,
            'confidence': 0.0,
            'expected_impact': 0
        }
        
        # Check for accumulation pattern (increasing whale buying, decreasing exchange balance)
        if ('whale_buying_pressure' in metrics and 'exchange_netflow' in metrics and
            metrics['whale_buying_pressure'] > 0 and metrics['exchange_netflow'] < 0):
            # Check if this has been consistent
            if (len(self.metric_history.get('whale_buying_pressure', [])) > 3 and 
                len(self.metric_history.get('exchange_netflow', [])) > 3):
                # Check last 3 values
                whale_buying = list(self.metric_history['whale_buying_pressure'])[-3:]
                exchange_flow = list(self.metric_history['exchange_netflow'])[-3:]
                
                if all(wb > 0 for wb in whale_buying) and all(ef < 0 for ef in exchange_flow):
                    patterns = {
                        'detected': True,
                        'pattern_type': 'accumulation',
                        'description': 'Whales are accumulating while coins are moving off exchanges',
                        'confidence': min(abs(metrics['whale_buying_pressure']) * 3, 1.0),
                        'expected_impact': 1  # Bullish
                    }
        
        # Check for distribution pattern (increasing whale selling, increasing exchange inflow)
        elif ('whale_selling_pressure' in metrics and 'exchange_inflow' in metrics and
              metrics['whale_selling_pressure'] > 0 and metrics['exchange_inflow'] > 0):
            # Check if this has been consistent
            if (len(self.metric_history.get('whale_selling_pressure', [])) > 3 and 
                len(self.metric_history.get('exchange_inflow', [])) > 3):
                # Check last 3 values
                whale_selling = list(self.metric_history['whale_selling_pressure'])[-3:]
                exchange_inflow = list(self.metric_history['exchange_inflow'])[-3:]
                
                if all(ws > 0 for ws in whale_selling) and all(ei > 0 for ei in exchange_inflow):
                    patterns = {
                        'detected': True,
                        'pattern_type': 'distribution',
                        'description': 'Whales are selling while coins are moving to exchanges',
                        'confidence': min(abs(metrics['whale_selling_pressure']) * 3, 1.0),
                        'expected_impact': -1  # Bearish
                    }
        
        # Check for network growth pattern (increasing addresses, increasing transactions)
        elif ('address_growth' in metrics and 'transaction_growth' in metrics and
              metrics['address_growth'] > 0 and metrics['transaction_growth'] > 0):
            # Check if this has been consistent
            if (len(self.metric_history.get('address_growth', [])) > 3 and 
                len(self.metric_history.get('transaction_growth', [])) > 3):
                # Check last 3 values
                addr_growth = list(self.metric_history['address_growth'])[-3:]
                tx_growth = list(self.metric_history['transaction_growth'])[-3:]
                
                if all(ag > 0 for ag in addr_growth) and all(tg > 0 for tg in tx_growth):
                    patterns = {
                        'detected': True,
                        'pattern_type': 'network_growth',
                        'description': 'Network is growing with new addresses and increasing activity',
                        'confidence': min((metrics['address_growth'] + metrics['transaction_growth']) / 2, 1.0),
                        'expected_impact': 1  # Bullish
                    }
        
        # Check for staking increase pattern (increasing staking rate)
        elif 'staking_growth' in metrics and metrics['staking_growth'] > 0:
            # Check if this has been consistent
            if len(self.metric_history.get('staking_growth', [])) > 3:
                # Check last 3 values
                staking_growth = list(self.metric_history['staking_growth'])[-3:]
                
                if all(sg > 0 for sg in staking_growth):
                    patterns = {
                        'detected': True,
                        'pattern_type': 'staking_increase',
                        'description': 'Increasing staking rate indicating long-term confidence',
                        'confidence': min(metrics['staking_growth'] * 2, 1.0),
                        'expected_impact': 1  # Bullish
                    }
        
        # Check for network stress pattern (increasing fees, slower block times)
        elif ('transaction_fees' in metrics and 'block_time' in metrics and
              metrics['transaction_fees'] > 0 and metrics.get('block_time', 0) > 0):
            # Check if this has been consistent
            if (len(self.metric_history.get('transaction_fees', [])) > 3 and 
                len(self.metric_history.get('block_time', [])) > 3):
                # Check last 3 values
                fees = list(self.metric_history['transaction_fees'])[-3:]
                block_times = list(self.metric_history.get('block_time', [0, 0, 0]))[-3:]
                
                if all(f > 0 for f in fees) and all(bt > 0 for bt in block_times):
                    patterns = {
                        'detected': True,
                        'pattern_type': 'network_stress',
                        'description': 'Network congestion with increasing fees and slower blocks',
                        'confidence': min((metrics['transaction_fees'] + metrics.get('block_time', 0)) / 2, 1.0),
                        'expected_impact': -1  # Generally bearish
                    }
        
        return patterns
    
    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze market data using on-chain metrics.
        
        Args:
            data: OHLCV data to analyze
            
        Returns:
            analysis: DataFrame with analysis results
        """
        # Clone the input data to avoid modifying the original
        analysis = data.copy()
        
        # Add technical features for correlation analysis
        for feature in self.config['technical_features']:
            try:
                feature_df = getattr(self.technical_features, feature)(data)
                # Merge only new columns to avoid duplicates
                for col in feature_df.columns:
                    if col not in analysis.columns:
                        analysis[col] = feature_df[col]
            except Exception as e:
                logger.error(f"Error calculating {feature}: {str(e)}")
        
        # Initialize on-chain signal columns
        analysis['onchain_signal'] = 0.0
        analysis['onchain_confidence'] = 0.0
        analysis['onchain_pattern'] = None
        
        # Process each candle for on-chain signals
        for i in range(len(analysis)):
            # Get timestamp for this candle
            timestamp = analysis.index[i]
            if isinstance(timestamp, pd.Timestamp):
                end_time = timestamp.to_pydatetime()
            else:
                end_time = datetime.fromtimestamp(timestamp / 1000)  # Assume milliseconds
            
            # Get on-chain metrics
            metrics = self._get_all_metrics(end_time)
            
            # Calculate signal from metrics
            signal_data = self._calculate_on_chain_signal(metrics)
            
            # Check for specific patterns
            patterns = self._check_for_specific_patterns(metrics)
            
            # Store results in the analysis dataframe
            analysis.loc[analysis.index[i], 'onchain_signal'] = signal_data['normalized_signal']
            analysis.loc[analysis.index[i], 'onchain_confidence'] = signal_data['confidence']
            analysis.loc[analysis.index[i], 'onchain_pattern'] = patterns['pattern_type'] if patterns['detected'] else None
            
            # Add pattern impact if detected
            if patterns['detected']:
                analysis.loc[analysis.index[i], 'pattern_impact'] = patterns['expected_impact']
                analysis.loc[analysis.index[i], 'pattern_confidence'] = patterns['confidence']
            else:
                analysis.loc[analysis.index[i], 'pattern_impact'] = 0
                analysis.loc[analysis.index[i], 'pattern_confidence'] = 0
        
        # Calculate correlation between on-chain signals and price movements
        if len(analysis) > 1:
            # Calculate returns
            analysis['returns'] = analysis['close'].pct_change()
            
            # Calculate correlation
            correlation = analysis['onchain_signal'].corr(analysis['returns'])
            
            # Add correlation info to all rows for reference
            analysis['onchain_price_correlation'] = correlation
        
        return analysis
    
    def update(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Update the brain with new data.
        
        Args:
            data: New OHLCV data
            
        Returns:
            update_info: Information about the update
        """
        # Get timestamp for latest data
        latest_timestamp = data.index[-1]
        if isinstance(latest_timestamp, pd.Timestamp):
            end_time = latest_timestamp.to_pydatetime()
        else:
            end_time = datetime.fromtimestamp(latest_timestamp / 1000)  # Assume milliseconds
        
        # Get on-chain metrics
        metrics = self._get_all_metrics(end_time)
        
        # Calculate signal from metrics
        signal_data = self._calculate_on_chain_signal(metrics)
        
        # Check for specific patterns
        patterns = self._check_for_specific_patterns(metrics)
        
        # Return update information
        update_info = {
            'timestamp': end_time.isoformat(),
            'signal': signal_data['normalized_signal'],
            'discrete_signal': signal_data['discrete_signal'],
            'confidence': signal_data['confidence'],
            'pattern_detected': patterns['detected'],
            'pattern_type': patterns['pattern_type'],
            'pattern_description': patterns['description'],
            'pattern_confidence': patterns['confidence'] if patterns['detected'] else 0.0,
            'top_bullish_metrics': signal_data['top_bullish_metrics'],
            'top_bearish_metrics': signal_data['top_bearish_metrics'],
            'metrics_processed': len(metrics)
        }
        
        return update_info
    
    def get_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a trading signal from on-chain metrics and market data.
        
        Args:
            data: Current OHLCV data
            
        Returns:
            signal_info: Trading signal information
        """
        # Update brain with new data
        update_info = self.update(data)
        
        # Calculate technical factors from price data
        technical_signal = 0.0
        try:
            # Calculate RSI as a simple technical factor
            if 'rsi' in self.config['technical_features']:
                rsi_df = self.technical_features.rsi(data)
                latest_rsi = rsi_df['rsi'].iloc[-1]
                
                # Convert RSI to a signal
                if latest_rsi > 70:
                    technical_signal = -0.5  # Overbought
                elif latest_rsi < 30:
                    technical_signal = 0.5  # Oversold
            
            # Calculate MACD
            if 'macd' in self.config['technical_features']:
                macd_df = self.technical_features.macd(data)
                macd = macd_df['macd'].iloc[-1]
                signal = macd_df['signal'].iloc[-1]
                
                # MACD crossing above signal is bullish
                if macd > signal:
                    technical_signal += 0.3
                # MACD crossing below signal is bearish
                elif macd < signal:
                    technical_signal -= 0.3
                
        except Exception as e:
            logger.error(f"Error calculating technical factors: {str(e)}")
        
        # Combine on-chain signal with technical factors
        # Weight on-chain signal more heavily
        combined_signal = (update_info['signal'] * 0.7) + (technical_signal * 0.3)
        
        # If a strong pattern is detected, give it more weight
        if update_info['pattern_detected'] and update_info['pattern_confidence'] > 0.6:
            pattern_signal = update_info['pattern_confidence'] * update_info.get('pattern_impact', 0)
            combined_signal = (combined_signal * 0.6) + (pattern_signal * 0.4)
        
        # Normalize the combined signal
        if self.config['signal_normalization'] == 'minmax':
            # Simple min-max normalization to [-1, 1]
            normalized_signal = max(min(combined_signal, 1.0), -1.0)
        elif self.config['signal_normalization'] == 'sigmoid':
            # Sigmoid function to compress to [-1, 1]
            normalized_signal = (2 / (1 + np.exp(-2 * combined_signal))) - 1
        else:  # tanh
            # Hyperbolic tangent
            normalized_signal = np.tanh(combined_signal)
        
        # Calculate confidence based on signal strength and on-chain confidence
        confidence = min(abs(normalized_signal) * update_info['confidence'] * 1.5, 1.0)
        
        # Determine discrete signal (-1, 0, 1)
        discrete_signal = 0
        if normalized_signal > self.config['signal_threshold']:
            discrete_signal = 1
        elif normalized_signal < -self.config['signal_threshold']:
            discrete_signal = -1
        
        # Enhanced signal information
        signal_info = {
            'signal': normalized_signal,
            'discrete_signal': discrete_signal,
            'confidence': confidence,
            'timestamp': update_info['timestamp'],
            'source': self.name,
            'exchange': self.exchange,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'strategy': 'onchain_analysis',
            'onchain_signal': update_info['signal'],
            'technical_signal': technical_signal,
            'pattern_detected': update_info['pattern_detected'],
            'pattern_type': update_info['pattern_type'],
            'pattern_description': update_info['pattern_description'],
            'top_bullish_factors': update_info['top_bullish_metrics'],
            'top_bearish_factors': update_info['top_bearish_metrics']
        }
        
        # Add risk analysis
        try:
            # Calculate potential risk/reward
            last_close = data['close'].iloc[-1]
            
            # If we have volatility data, use it for risk estimation
            if 'atr' in data.columns:
                atr = data['atr'].iloc[-1]
                risk_info = {
                    'stop_loss': last_close - (atr * 2) if discrete_signal > 0 else last_close + (atr * 2),
                    'take_profit': last_close + (atr * 3) if discrete_signal > 0 else last_close - (atr * 3),
                    'risk_reward_ratio': 1.5  # 3/2
                }
            else:
                # Use a simple percentage-based approach if ATR not available
                risk_info = {
                    'stop_loss': last_close * 0.98 if discrete_signal > 0 else last_close * 1.02,
                    'take_profit': last_close * 1.03 if discrete_signal > 0 else last_close * 0.97,
                    'risk_reward_ratio': 1.5  # 3%/2%
                }
            
            signal_info['risk_analysis'] = risk_info
            
        except Exception as e:
            logger.error(f"Error calculating risk analysis: {str(e)}")
        
        return signal_info
    
    def save(self) -> bool:
        """
        Save the brain's state to disk.
        
        Returns:
            success: Whether the save was successful
        """
        try:
            # Create a dictionary with brain state
            state = {
                'name': self.name,
                'exchange': self.exchange,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'config': self.config,
                'signal_history': list(self.signal_history),
                'metric_history': {k: list(v) for k, v in self.metric_history.items()},
                'timestamp': datetime.now().isoformat()
            }
            
            # Create filename based on brain details
            filename = f"{self.exchange}_{self.symbol}_{self.timeframe}_{self.name}.json"
            filepath = os.path.join(self.config['model_path'], filename)
            
            # Save to disk
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved brain state to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save brain state: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def load(self) -> bool:
        """
        Load the brain's state from disk.
        
        Returns:
            success: Whether the load was successful
        """
        try:
            # Create filename based on brain details
            filename = f"{self.exchange}_{self.symbol}_{self.timeframe}_{self.name}.json"
            filepath = os.path.join(self.config['model_path'], filename)
            
            # Check if file exists
            if not os.path.exists(filepath):
                logger.warning(f"No saved state found at {filepath}")
                return False
            
            # Load from disk
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Validate state
            if (state['exchange'] != self.exchange or 
                state['symbol'] != self.symbol or 
                state['timeframe'] != self.timeframe):
                logger.warning(f"Saved state does not match current brain parameters")
                return False
            
            # Update brain state
            self.config = state['config']
            self.signal_history = deque(state['signal_history'], maxlen=100)
            
            # Convert metric history lists back to deques
            self.metric_history = {}
            for k, v in state['metric_history'].items():
                self.metric_history[k] = deque(v, maxlen=100)
            
            logger.info(f"Loaded brain state from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load brain state: {str(e)}")
            logger.error(traceback.format_exc())
            return False


if __name__ == "__main__":
    # Test code
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create sample OHLCV data
    n = 1000
    np.random.seed(42)
    dates = [datetime.now() - timedelta(minutes=i) for i in range(n, 0, -1)]
    
    # Create a trending market with some noise
    close = np.cumsum(np.random.normal(0, 1, n)) + 1000
    # Add a trend
    close = close + np.linspace(0, 50, n)
    # Add seasonality
    close = close + 20 * np.sin(np.linspace(0, 8 * np.pi, n))
    
    high = close + np.random.normal(0, 5, n)
    low = close - np.random.normal(0, 5, n)
    open_price = close - np.random.normal(0, 2, n)
    volume = np.random.normal(1000, 100, n) + 200 * np.sin(np.linspace(0, 6 * np.pi, n)) + 1000
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df.set_index('timestamp', inplace=True)
    
    # Initialize and test the on-chain brain
    config = {
        'network_id': 'bitcoin',
        'historical_lookback': 5,  # Reduce for testing
        'model_path': './test_models',
        'backtest_data_path': './test_data'
    }
    
    brain = OnChainBrain(
        name="TestOnChainBrain",
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1h",
        config=config
    )
    
    # Get a signal
    signal_info = brain.get_signal(df)
    print(f"Signal: {signal_info}")
    
    # Save brain state
    brain.save()

        
