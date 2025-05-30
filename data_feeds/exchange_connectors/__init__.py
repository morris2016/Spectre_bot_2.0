"""
Data Feeds Module for QuantumSpectre Trading System.

This module handles various external data sources including market data,
news, social media, and on-chain data.
"""

__version__ = '1.0.0'

from .base_feed import BaseFeed
from .binance_feed import BinanceFeed
from .deriv_feed import DerivFeed
from .news_feed import NewsFeed, stream_twitter
from .onchain_feed import OnchainFeed, fetch_onchain

__all__ = [
    'BaseFeed',
    'BinanceFeed',
    'DerivFeed',
    'NewsFeed',
    'OnchainFeed',
    'stream_twitter',
    'fetch_onchain',
]
