#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Backtester Data Provider

This module provides a comprehensive data provider for the backtesting engine,
allowing access to historical market data, order book data, and other related
information for accurate simulation.
"""

import asyncio
import datetime
import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from common.logger import get_logger
from common.constants import TIME_FRAMES, DataSourcePreference

DATA_SOURCES = DataSourcePreference
DataProvider = DataSourcePreference

from common.exceptions import (
    DataSourceError, DataInsufficientError, InvalidAssetError
)
from common.db_client import DBClient
from common.redis_client import RedisClient
from common.utils import validate_timeframe, parallelize

from data_storage.time_series import TimeSeriesDB
from data_storage.market_data import MarketDataRepository
from data_feeds.binance_feed import BinanceFeed
from data_feeds.deriv_feed import DerivFeed

logger = get_logger(__name__)

class BacktestDataProvider:
    """
    Advanced data provider for backtesting, providing access to historical data
    with sophisticated caching, interpolation, and multi-source capabilities.
    
    Features:
    - Multi-source data aggregation (exchange, external APIs, local files)
    - On-demand data loading with intelligent caching
    - Seamless timeframe conversion and alignment
    - Support for tick, order book, and volume profile data
    - Market microstructure simulation for realistic backtesting
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        db_client: Optional[DBClient] = None,
        redis_client: Optional[RedisClient] = None,
        feed_clients: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the backtesting data provider.
        
        Args:
            config: Configuration dictionary
            db_client: Database client for persistent storage
            redis_client: Redis client for cache
            feed_clients: Optional dictionary of feed clients for data retrieval
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{config.get('backtest_id', 'default')}")
        
        # Clients
        self.db_client = db_client or DBClient()
        self.redis_client = redis_client or RedisClient()
        
        # Data repositories
        self.market_data_repo = MarketDataRepository(self.db_client)
        self.time_series_db = TimeSeriesDB(self.db_client, self.redis_client)
        
        # Feed clients
        self.feed_clients = feed_clients or {}
        if not self.feed_clients:
            # Initialize default feed clients if not provided
            self.feed_clients = {
                'binance': BinanceFeed(real_time=False, db_client=self.db_client),
                'deriv': DerivFeed(real_time=False, db_client=self.db_client)
            }
        
        # Data cache
        self.ohlcv_cache = {}
        self.orderbook_cache = {}
        self.tick_cache = {}
        self.news_cache = {}
        self.sentiment_cache = {}
        
        # State tracking
        self.data_loaded = False
        self.available_assets = set()
        self.available_timeframes = set()
        self.start_date = None
        self.end_date = None
        
        # Parallelization
        self.executor = ThreadPoolExecutor(
            max_workers=config.get('max_workers', 10)
        )
        
        self.logger.info(f"BacktestDataProvider initialized")
    
    async def load_data(
        self,
        assets: List[str],
        timeframes: List[str],
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        include_volume: bool = True,
        include_orderbook: bool = False,
        include_ticks: bool = False,
        include_sentiment: bool = False,
        data_source: DataSourcePreference = DATA_SOURCES.DB_FIRST

    ) -> bool:
        """
        Load historical data for the specified assets, timeframes, and date range.
        
        Args:
            assets: List of asset symbols
            timeframes: List of timeframe strings
            start_date: Start date for data
            end_date: End date for data
            include_volume: Whether to include volume data
            include_orderbook: Whether to include order book data
            include_ticks: Whether to include tick data
            include_sentiment: Whether to include sentiment data
            data_source: Source preference for data retrieval
            
        Returns:
            bool: True if data loaded successfully
        """
        self.logger.info(
            f"Loading data for {len(assets)} assets, {len(timeframes)} timeframes "
            f"from {start_date} to {end_date}"
        )
        
        self.start_date = start_date
        self.end_date = end_date
        
        # Validate inputs
        for timeframe in timeframes:
            if not validate_timeframe(timeframe):
                raise ValueError(f"Invalid timeframe: {timeframe}")
                
        # Track loading errors
        errors = []
        
        # Load OHLCV data for each asset and timeframe
        for asset in assets:
            for timeframe in timeframes:
                try:
                    await self._load_ohlcv(
                        asset, 
                        timeframe, 
                        start_date, 
                        end_date, 
                        data_source
                    )
                    self.available_assets.add(asset)
                    self.available_timeframes.add(timeframe)
                    
                except Exception as e:
                    errors.append(f"Error loading OHLCV for {asset} {timeframe}: {str(e)}")
                    self.logger.error(f"Failed to load OHLCV for {asset} {timeframe}: {str(e)}")
        
        # Load order book data if required
        if include_orderbook:
            for asset in assets:
                try:
                    await self._load_orderbook(
                        asset, 
                        start_date, 
                        end_date,
                        data_source
                    )
                except Exception as e:
                    errors.append(f"Error loading orderbook for {asset}: {str(e)}")
                    self.logger.error(f"Failed to load orderbook for {asset}: {str(e)}")
        
        # Load tick data if required
        if include_ticks:
            for asset in assets:
                try:
                    await self._load_ticks(
                        asset, 
                        start_date, 
                        end_date,
                        data_source
                    )
                except Exception as e:
                    errors.append(f"Error loading ticks for {asset}: {str(e)}")
                    self.logger.error(f"Failed to load ticks for {asset}: {str(e)}")
        
        # Load sentiment data if required
        if include_sentiment:
            for asset in assets:
                try:
                    await self._load_sentiment(
                        asset, 
                        start_date, 
                        end_date,
                        data_source
                    )
                except Exception as e:
                    errors.append(f"Error loading sentiment for {asset}: {str(e)}")
                    self.logger.error(f"Failed to load sentiment for {asset}: {str(e)}")
        
        # Check if we have sufficient data
        if not self.available_assets or not self.available_timeframes:
            error_msg = "Failed to load any data. Errors: " + "; ".join(errors)
            self.logger.error(error_msg)
            raise DataInsufficientError(error_msg)
        
        # Set data loaded flag
        self.data_loaded = True
        
        self.logger.info(
            f"Data loaded successfully for {len(self.available_assets)} assets "
            f"and {len(self.available_timeframes)} timeframes"
        )
        
        return True
    
    async def _load_ohlcv(
        self, 
        asset: str, 
        timeframe: str, 
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        data_source: DataSourcePreference = DATA_SOURCES.DB_FIRST

    ) -> pd.DataFrame:
        """
        Load OHLCV data for a specific asset and timeframe.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe string
            start_date: Start date
            end_date: End date
            data_source: Source preference
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        # Generate cache key
        cache_key = f"{asset}_{timeframe}"
        
        # If already in cache, return it
        if cache_key in self.ohlcv_cache:
            return self.ohlcv_cache[cache_key]
        
        # Adjust dates for potential gaps and weekend data
        padded_start = start_date - pd.Timedelta(days=5)
        padded_end = end_date + pd.Timedelta(days=5)
        
        # Try different data sources based on preference
        df = None
        
        if data_source == DataSourcePreference.DB_FIRST:
            # Try database first
            try:
                df = await self.time_series_db.get_ohlcv(
                    asset=asset,
                    timeframe=timeframe,
                    start_date=padded_start,
                    end_date=padded_end
                )
                
                if df is not None and not df.empty:
                    self.logger.debug(
                        f"Loaded {len(df)} OHLCV records for {asset} {timeframe} from database"
                    )
                    
            except Exception as e:
                self.logger.warning(
                    f"Database retrieval failed for {asset} {timeframe}: {str(e)}"
                )
            
            # If DB failed, try feed client
            if df is None or df.empty:
                try:
                    # Determine appropriate feed based on asset
                    feed_client = self._get_feed_for_asset(asset)
                    
                    df = await feed_client.get_historical_ohlcv(
                        asset=asset,
                        timeframe=timeframe,
                        start_date=padded_start,
                        end_date=padded_end
                    )
                    
                    if df is not None and not df.empty:
                        self.logger.debug(
                            f"Loaded {len(df)} OHLCV records for {asset} {timeframe} from feed"
                        )
                        
                        # Store in database for future use
                        await self.time_series_db.store_ohlcv(
                            asset=asset,
                            timeframe=timeframe,
                            df=df
                        )
                
                except Exception as e:
                    self.logger.warning(
                        f"Feed retrieval failed for {asset} {timeframe}: {str(e)}"
                    )
        
        elif data_source == DataSourcePreference.FEED_FIRST:
            # Try feed client first
            try:
                feed_client = self._get_feed_for_asset(asset)
                
                df = await feed_client.get_historical_ohlcv(
                    asset=asset,
                    timeframe=timeframe,
                    start_date=padded_start,
                    end_date=padded_end
                )
                
                if df is not None and not df.empty:
                    self.logger.debug(
                        f"Loaded {len(df)} OHLCV records for {asset} {timeframe} from feed"
                    )
                    
                    # Store in database for future use
                    await self.time_series_db.store_ohlcv(
                        asset=asset,
                        timeframe=timeframe,
                        df=df
                    )
                    
            except Exception as e:
                self.logger.warning(
                    f"Feed retrieval failed for {asset} {timeframe}: {str(e)}"
                )
            
            # If feed failed, try database
            if df is None or df.empty:
                try:
                    df = await self.time_series_db.get_ohlcv(
                        asset=asset,
                        timeframe=timeframe,
                        start_date=padded_start,
                        end_date=padded_end
                    )
                    
                    if df is not None and not df.empty:
                        self.logger.debug(
                            f"Loaded {len(df)} OHLCV records for {asset} {timeframe} from database"
                        )
                        
                except Exception as e:
                    self.logger.warning(
                        f"Database retrieval failed for {asset} {timeframe}: {str(e)}"
                    )
        
        # If still no data, raise error
        if df is None or df.empty:
            raise DataInsufficientError(
                f"Could not retrieve OHLCV data for {asset} {timeframe}"
            )
        
        # Ensure data is properly indexed and sorted
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            else:
                raise ValueError(f"OHLCV data for {asset} {timeframe} has no timestamp column")
        
        df.sort_index(inplace=True)
        
        # Convert column names to standard format if needed
        standard_columns = ['open', 'high', 'low', 'close', 'volume']
        rename_map = {}
        
        for std_col in standard_columns:
            if std_col not in df.columns:
                for col in df.columns:
                    if col.lower() == std_col:
                        rename_map[col] = std_col
                        break
        
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
        
        # Ensure required columns exist
        for col in ['open', 'high', 'low', 'close']:
            if col not in df.columns:
                raise ValueError(f"OHLCV data for {asset} {timeframe} missing {col} column")
        
        # Add volume column if missing
        if 'volume' not in df.columns:
            df['volume'] = 0
        
        # Filter to requested date range
        df = df[(df.index >= padded_start) & (df.index <= padded_end)]
        
        # Store in cache
        self.ohlcv_cache[cache_key] = df
        
        return df
    
    async def _load_orderbook(
        self, 
        asset: str, 
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        data_source: str = DataSourcePreference.DB_FIRST
    ) -> Dict[pd.Timestamp, pd.DataFrame]:
        """
        Load order book snapshots for a specific asset.
        
        Args:
            asset: Asset symbol
            start_date: Start date
            end_date: End date
            data_source: Source preference
            
        Returns:
            Dict[pd.Timestamp, pd.DataFrame]: Order book snapshots
        """
        # Generate cache key
        cache_key = f"{asset}_orderbook"
        
        # If already in cache, return it
        if cache_key in self.orderbook_cache:
            return self.orderbook_cache[cache_key]
        
        # Try different data sources based on preference
        orderbooks = None
        
        if data_source == DataSourcePreference.DB_FIRST:
            # Try database first
            try:
                orderbooks = await self.time_series_db.get_orderbook_snapshots(
                    asset=asset,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if orderbooks:
                    self.logger.debug(
                        f"Loaded {len(orderbooks)} orderbook snapshots for {asset} from database"
                    )
                    
            except Exception as e:
                self.logger.warning(
                    f"Database retrieval failed for {asset} orderbooks: {str(e)}"
                )
            
            # If DB failed, try feed client
            if not orderbooks:
                try:
                    feed_client = self._get_feed_for_asset(asset)
                    
                    orderbooks = await feed_client.get_historical_orderbooks(
                        asset=asset,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if orderbooks:
                        self.logger.debug(
                            f"Loaded {len(orderbooks)} orderbook snapshots for {asset} from feed"
                        )
                        
                        # Store in database for future use
                        await self.time_series_db.store_orderbook_snapshots(
                            asset=asset,
                            orderbooks=orderbooks
                        )
                
                except Exception as e:
                    self.logger.warning(
                        f"Feed retrieval failed for {asset} orderbooks: {str(e)}"
                    )
        
        elif data_source == DataSourcePreference.FEED_FIRST:
            # Try feed client first
            try:
                feed_client = self._get_feed_for_asset(asset)
                
                orderbooks = await feed_client.get_historical_orderbooks(
                    asset=asset,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if orderbooks:
                    self.logger.debug(
                        f"Loaded {len(orderbooks)} orderbook snapshots for {asset} from feed"
                    )
                    
                    # Store in database for future use
                    await self.time_series_db.store_orderbook_snapshots(
                        asset=asset,
                        orderbooks=orderbooks
                    )
                    
            except Exception as e:
                self.logger.warning(
                    f"Feed retrieval failed for {asset} orderbooks: {str(e)}"
                )
            
            # If feed failed, try database
            if not orderbooks:
                try:
                    orderbooks = await self.time_series_db.get_orderbook_snapshots(
                        asset=asset,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if orderbooks:
                        self.logger.debug(
                            f"Loaded {len(orderbooks)} orderbook snapshots for {asset} from database"
                        )
                        
                except Exception as e:
                    self.logger.warning(
                        f"Database retrieval failed for {asset} orderbooks: {str(e)}"
                    )
        
        # If still no data, return empty dict but don't raise error
        # Order book data is optional
        if not orderbooks:
            self.logger.warning(f"No orderbook data available for {asset}")
            orderbooks = {}
        
        # Store in cache
        self.orderbook_cache[cache_key] = orderbooks
        
        return orderbooks
    
    async def _load_ticks(
        self, 
        asset: str, 
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        data_source: str = DataSourcePreference.DB_FIRST
    ) -> pd.DataFrame:
        """
        Load tick data for a specific asset.
        
        Args:
            asset: Asset symbol
            start_date: Start date
            end_date: End date
            data_source: Source preference
            
        Returns:
            pd.DataFrame: Tick data
        """
        # Generate cache key
        cache_key = f"{asset}_ticks"
        
        # If already in cache, return it
        if cache_key in self.tick_cache:
            return self.tick_cache[cache_key]
        
        # Try different data sources based on preference
        ticks = None
        
        if data_source == DataSourcePreference.DB_FIRST:
            # Try database first
            try:
                ticks = await self.time_series_db.get_ticks(
                    asset=asset,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if ticks is not None and not ticks.empty:
                    self.logger.debug(
                        f"Loaded {len(ticks)} ticks for {asset} from database"
                    )
                    
            except Exception as e:
                self.logger.warning(
                    f"Database retrieval failed for {asset} ticks: {str(e)}"
                )
            
            # If DB failed, try feed client
            if ticks is None or ticks.empty:
                try:
                    feed_client = self._get_feed_for_asset(asset)
                    
                    ticks = await feed_client.get_historical_ticks(
                        asset=asset,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if ticks is not None and not ticks.empty:
                        self.logger.debug(
                            f"Loaded {len(ticks)} ticks for {asset} from feed"
                        )
                        
                        # Store in database for future use
                        await self.time_series_db.store_ticks(
                            asset=asset,
                            ticks=ticks
                        )
                
                except Exception as e:
                    self.logger.warning(
                        f"Feed retrieval failed for {asset} ticks: {str(e)}"
                    )
        
        elif data_source == DataSourcePreference.FEED_FIRST:
            # Try feed client first
            try:
                feed_client = self._get_feed_for_asset(asset)
                
                ticks = await feed_client.get_historical_ticks(
                    asset=asset,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if ticks is not None and not ticks.empty:
                    self.logger.debug(
                        f"Loaded {len(ticks)} ticks for {asset} from feed"
                    )
                    
                    # Store in database for future use
                    await self.time_series_db.store_ticks(
                        asset=asset,
                        ticks=ticks
                    )
                    
            except Exception as e:
                self.logger.warning(
                    f"Feed retrieval failed for {asset} ticks: {str(e)}"
                )
            
            # If feed failed, try database
            if ticks is None or ticks.empty:
                try:
                    ticks = await self.time_series_db.get_ticks(
                        asset=asset,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if ticks is not None and not ticks.empty:
                        self.logger.debug(
                            f"Loaded {len(ticks)} ticks for {asset} from database"
                        )
                        
                except Exception as e:
                    self.logger.warning(
                        f"Database retrieval failed for {asset} ticks: {str(e)}"
                    )
        
        # If still no data, create empty dataframe but don't raise error
        # Tick data is optional
        if ticks is None or ticks.empty:
            self.logger.warning(f"No tick data available for {asset}")
            ticks = pd.DataFrame(columns=['timestamp', 'price', 'volume', 'side'])
            ticks.set_index('timestamp', inplace=True)
        
        # Ensure data is properly indexed and sorted
        if not isinstance(ticks.index, pd.DatetimeIndex):
            if 'timestamp' in ticks.columns:
                ticks.set_index('timestamp', inplace=True)
            else:
                # Create timestamp index
                ticks['timestamp'] = pd.date_range(
                    start=start_date, 
                    end=end_date, 
                    periods=len(ticks)
                )
                ticks.set_index('timestamp', inplace=True)
        
        ticks.sort_index(inplace=True)
        
        # Store in cache
        self.tick_cache[cache_key] = ticks
        
        return ticks
    
    async def _load_sentiment(
        self, 
        asset: str, 
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        data_source: str = DataSourcePreference.DB_FIRST
    ) -> pd.DataFrame:
        """
        Load sentiment data for a specific asset.
        
        Args:
            asset: Asset symbol
            start_date: Start date
            end_date: End date
            data_source: Source preference
            
        Returns:
            pd.DataFrame: Sentiment data
        """
        # Generate cache key
        cache_key = f"{asset}_sentiment"
        
        # If already in cache, return it
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        # Try to load from database
        sentiment = None
        
        try:
            sentiment = await self.time_series_db.get_sentiment(
                asset=asset,
                start_date=start_date,
                end_date=end_date
            )
            
            if sentiment is not None and not sentiment.empty:
                self.logger.debug(
                    f"Loaded {len(sentiment)} sentiment records for {asset} from database"
                )
                
        except Exception as e:
            self.logger.warning(
                f"Database retrieval failed for {asset} sentiment: {str(e)}"
            )
        
        # If no data, create synthetic sentiment
        if sentiment is None or sentiment.empty:
            self.logger.warning(
                f"No sentiment data available for {asset}, creating synthetic data"
            )
            
            # Create synthetic sentiment data based on price movements
            try:
                # Load OHLCV data
                ohlcv = await self._load_ohlcv(
                    asset=asset,
                    timeframe='1d',  # Daily timeframe for sentiment
                    start_date=start_date,
                    end_date=end_date,
                    data_source=data_source
                )
                
                if ohlcv is not None and not ohlcv.empty:
                    # Generate synthetic sentiment based on price movements
                    # This is a simplified model for backtesting purposes
                    sentiment = pd.DataFrame(index=ohlcv.index)
                    
                    # Calculate returns
                    returns = ohlcv['close'].pct_change()
                    
                    # Calculate moving averages
                    ma5 = ohlcv['close'].rolling(5).mean()
                    ma20 = ohlcv['close'].rolling(20).mean()
                    
                    # Generate sentiment score (-1 to 1)
                    sentiment['score'] = np.zeros(len(sentiment))
                    
                    # Price above MA20 -> slightly positive sentiment
                    sentiment.loc[ohlcv['close'] > ma20, 'score'] += 0.2
                    
                    # Price below MA20 -> slightly negative sentiment
                    sentiment.loc[ohlcv['close'] < ma20, 'score'] -= 0.2
                    
                    # Price above MA5 -> additional positive sentiment
                    sentiment.loc[ohlcv['close'] > ma5, 'score'] += 0.1
                    
                    # Price below MA5 -> additional negative sentiment
                    sentiment.loc[ohlcv['close'] < ma5, 'score'] -= 0.1
                    
                    # Strong positive returns -> additional positive sentiment
                    sentiment.loc[returns > 0.03, 'score'] += 0.3
                    
                    # Strong negative returns -> additional negative sentiment
                    sentiment.loc[returns < -0.03, 'score'] -= 0.3
                    
                    # Add some random noise
                    np.random.seed(42)  # For reproducibility
                    sentiment['score'] += np.random.normal(0, 0.1, len(sentiment))
                    
                    # Clip values to [-1, 1] range
                    sentiment['score'] = sentiment['score'].clip(-1, 1)
                    
                    # Add volume component
                    sentiment['volume'] = (ohlcv['volume'] / ohlcv['volume'].mean()).clip(0.1, 10)
                    
                    # Fill missing values
                    sentiment.fillna(method='ffill', inplace=True)
                    sentiment.fillna(0, inplace=True)
                    
                    self.logger.debug(
                        f"Created synthetic sentiment data for {asset} with {len(sentiment)} records"
                    )
                
            except Exception as e:
                self.logger.warning(
                    f"Failed to create synthetic sentiment for {asset}: {str(e)}"
                )
                
                # Create empty dataframe
                sentiment = pd.DataFrame(
                    columns=['timestamp', 'score', 'volume', 'source'],
                    index=pd.date_range(start=start_date, end=end_date, freq='1D')
                )
                sentiment['score'] = 0
                sentiment['volume'] = 1
                sentiment['source'] = 'synthetic'
        
        # Ensure data has proper columns
        if 'score' not in sentiment.columns:
            sentiment['score'] = 0
            
        if 'volume' not in sentiment.columns:
            sentiment['volume'] = 1
            
        if 'source' not in sentiment.columns:
            sentiment['source'] = 'unknown'
        
        # Store in cache
        self.sentiment_cache[cache_key] = sentiment
        
        return sentiment
    
    def _get_feed_for_asset(self, asset: str):
        """
        Determine the appropriate feed client for a given asset.
        
        Args:
            asset: Asset symbol
            
        Returns:
            Feed client instance
        """
        # Check asset prefix to determine feed
        if asset.startswith(('BTC', 'ETH', 'BNB', 'XRP')) or '/' in asset or 'USDT' in asset:
            return self.feed_clients.get('binance', next(iter(self.feed_clients.values())))
        elif asset.startswith(('R_', 'BOOM', 'CRASH')):
            return self.feed_clients.get('deriv', next(iter(self.feed_clients.values())))
        else:
            # Default to first available feed
            return next(iter(self.feed_clients.values()))
    
    def is_data_loaded(self) -> bool:
        """
        Check if data has been loaded.
        
        Returns:
            bool: True if data is loaded
        """
        return self.data_loaded
    
    def get_available_assets(self) -> List[str]:
        """
        Get list of available assets.
        
        Returns:
            List[str]: Available assets
        """
        return list(self.available_assets)
    
    def get_available_timeframes(self) -> List[str]:
        """
        Get list of available timeframes.
        
        Returns:
            List[str]: Available timeframes
        """
        return list(self.available_timeframes)
    
    def get_ohlcv(
        self, 
        asset: str, 
        timeframe: str, 
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV data for a specific asset and timeframe.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe string
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Optional limit for number of records
            
        Returns:
            pd.DataFrame: OHLCV data
        """
        # Check if data is loaded
        if not self.data_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
            
        # Check if asset and timeframe are available
        if asset not in self.available_assets:
            raise ValueError(f"Asset {asset} not available")
            
        if timeframe not in self.available_timeframes:
            raise ValueError(f"Timeframe {timeframe} not available")
            
        # Generate cache key
        cache_key = f"{asset}_{timeframe}"
        
        # Check if data is in cache
        if cache_key not in self.ohlcv_cache:
            raise RuntimeError(f"OHLCV data for {asset} {timeframe} not loaded")
            
        # Get data from cache
        df = self.ohlcv_cache[cache_key].copy()
        
        # Apply filters
        if start_date is not None:
            df = df[df.index >= start_date]
            
        if end_date is not None:
            df = df[df.index <= end_date]
            
        if limit is not None:
            df = df.iloc[-limit:]
            
        return df
    
    def get_ohlcv_until(
        self, 
        asset: str, 
        timeframe: str, 
        timestamp: datetime.datetime
    ) -> pd.DataFrame:
        """
        Get OHLCV data for a specific asset and timeframe up to a specific timestamp.
        
        This is useful for backtesting to ensure we only use data available at a specific time.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe string
            timestamp: Cutoff timestamp
            
        Returns:
            pd.DataFrame: OHLCV data up to the timestamp
        """
        # Get full data
        df = self.get_ohlcv(asset, timeframe)
        
        # Filter to timestamp
        df = df[df.index <= timestamp]
        
        return df
    
    def get_orderbook(
        self, 
        asset: str, 
        timestamp: datetime.datetime,
        tolerance_seconds: int = 60
    ) -> Optional[pd.DataFrame]:
        """
        Get order book snapshot closest to a specific timestamp.
        
        Args:
            asset: Asset symbol
            timestamp: Target timestamp
            tolerance_seconds: Tolerance in seconds for timestamp matching
            
        Returns:
            Optional[pd.DataFrame]: Order book snapshot or None
        """
        # Check if data is loaded
        if not self.data_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
            
        # Check if asset is available
        if asset not in self.available_assets:
            raise ValueError(f"Asset {asset} not available")
            
        # Generate cache key
        cache_key = f"{asset}_orderbook"
        
        # Check if data is in cache
        if cache_key not in self.orderbook_cache:
            return None
            
        # Get data from cache
        orderbooks = self.orderbook_cache[cache_key]
        
        if not orderbooks:
            return None
            
        # Find closest timestamp
        timestamps = list(orderbooks.keys())
        closest_ts = min(timestamps, key=lambda ts: abs((ts - timestamp).total_seconds()))
        
        # Check if within tolerance
        if abs((closest_ts - timestamp).total_seconds()) > tolerance_seconds:
            return None
            
        return orderbooks[closest_ts]
    
    def get_ticks(
        self, 
        asset: str, 
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get tick data for a specific asset.
        
        Args:
            asset: Asset symbol
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Optional limit for number of records
            
        Returns:
            pd.DataFrame: Tick data
        """
        # Check if data is loaded
        if not self.data_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
            
        # Check if asset is available
        if asset not in self.available_assets:
            raise ValueError(f"Asset {asset} not available")
            
        # Generate cache key
        cache_key = f"{asset}_ticks"
        
        # Check if data is in cache
        if cache_key not in self.tick_cache:
            return pd.DataFrame(columns=['price', 'volume', 'side'])
            
        # Get data from cache
        df = self.tick_cache[cache_key].copy()
        
        # Apply filters
        if start_date is not None:
            df = df[df.index >= start_date]
            
        if end_date is not None:
            df = df[df.index <= end_date]
            
        if limit is not None:
            df = df.iloc[-limit:]
            
        return df
    
    def get_ticks_until(
        self, 
        asset: str, 
        timestamp: datetime.datetime
    ) -> pd.DataFrame:
        """
        Get tick data for a specific asset up to a specific timestamp.
        
        Args:
            asset: Asset symbol
            timestamp: Cutoff timestamp
            
        Returns:
            pd.DataFrame: Tick data up to the timestamp
        """
        # Get full data
        df = self.get_ticks(asset)
        
        # Filter to timestamp
        df = df[df.index <= timestamp]
        
        return df
    
    def get_sentiment(
        self, 
        asset: str, 
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None
    ) -> pd.DataFrame:
        """
        Get sentiment data for a specific asset.
        
        Args:
            asset: Asset symbol
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            pd.DataFrame: Sentiment data
        """
        # Check if data is loaded
        if not self.data_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
            
        # Check if asset is available
        if asset not in self.available_assets:
            raise ValueError(f"Asset {asset} not available")
            
        # Generate cache key
        cache_key = f"{asset}_sentiment"
        
        # Check if data is in cache
        if cache_key not in self.sentiment_cache:
            return pd.DataFrame(columns=['score', 'volume', 'source'])
            
        # Get data from cache
        df = self.sentiment_cache[cache_key].copy()
        
        # Apply filters
        if start_date is not None:
            df = df[df.index >= start_date]
            
        if end_date is not None:
            df = df[df.index <= end_date]
            
        return df
    
    def get_sentiment_at(
        self, 
        asset: str, 
        timestamp: datetime.datetime,
        window_days: int = 1
    ) -> Dict[str, float]:
        """
        Get sentiment data for a specific asset at a specific timestamp.
        
        Args:
            asset: Asset symbol
            timestamp: Target timestamp
            window_days: Look-back window in days
            
        Returns:
            Dict[str, float]: Sentiment data with score, volume, etc.
        """
        # Generate window
        start_date = timestamp - pd.Timedelta(days=window_days)
        end_date = timestamp
        
        # Get sentiment data
        df = self.get_sentiment(
            asset=asset,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            return {
                'score': 0,
                'volume': 1,
                'source': 'none'
            }
            
        # Calculate weighted average sentiment
        avg_score = (df['score'] * df['volume']).sum() / df['volume'].sum()
        avg_volume = df['volume'].mean()
        
        # Get most common source
        if 'source' in df.columns:
            sources = df['source'].value_counts()
            main_source = sources.index[0] if not sources.empty else 'unknown'
        else:
            main_source = 'unknown'
        
        return {
            'score': avg_score,
            'volume': avg_volume,
            'source': main_source
        }
    
    def get_converted_timeframe(
        self, 
        asset: str, 
        source_timeframe: str, 
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Convert OHLCV data from one timeframe to another.
        
        Args:
            asset: Asset symbol
            source_timeframe: Source timeframe string
            target_timeframe: Target timeframe string
            
        Returns:
            pd.DataFrame: Converted OHLCV data
        """
        # Check if data is loaded
        if not self.data_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
            
        # Check if asset and source timeframe are available
        if asset not in self.available_assets:
            raise ValueError(f"Asset {asset} not available")
            
        if source_timeframe not in self.available_timeframes:
            raise ValueError(f"Timeframe {source_timeframe} not available")
            
        # Generate cache key for source
        source_key = f"{asset}_{source_timeframe}"
        
        # Check if source data is in cache
        if source_key not in self.ohlcv_cache:
            raise RuntimeError(f"OHLCV data for {asset} {source_timeframe} not loaded")
            
        # Get source data
        source_df = self.ohlcv_cache[source_key].copy()
        
        # If target equals source, return source
        if source_timeframe == target_timeframe:
            return source_df
            
        # Generate cache key for target
        target_key = f"{asset}_{target_timeframe}"
        
        # If target data is already in cache, return it
        if target_key in self.ohlcv_cache:
            return self.ohlcv_cache[target_key].copy()
            
        # Get timeframe values in minutes
        source_minutes = TIME_FRAMES[source_timeframe]
        target_minutes = TIME_FRAMES[target_timeframe]
        
        # Handle conversion based on relationship
        if target_minutes > source_minutes:
            # Upsampling (e.g., 1m to 5m)
            # Determine resampling rule
            if target_minutes < 60:
                # Minutes
                rule = f"{target_minutes}T"
            elif target_minutes < 1440:
                # Hours
                rule = f"{target_minutes // 60}H"
            else:
                # Days
                rule = f"{target_minutes // 1440}D"
                
            # Resample
            resampled = source_df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # Cache and return
            self.ohlcv_cache[target_key] = resampled
            return resampled
            
        else:
            # Downsampling not supported - need higher resolution data
            raise ValueError(
                f"Cannot convert from {source_timeframe} to {target_timeframe}. "
                f"Need higher resolution data."
            )
    
    async def regenerate_ohlcv(
        self, 
        asset: str, 
        timeframe: str, 
        from_ticks: bool = True
    ) -> pd.DataFrame:
        """
        Regenerate OHLCV data from tick data or by conversion from lower timeframe.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe string
            from_ticks: Whether to use tick data (True) or lowest timeframe OHLCV (False)
            
        Returns:
            pd.DataFrame: Regenerated OHLCV data
        """
        # Check if data is loaded
        if not self.data_loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")
            
        # Check if asset is available
        if asset not in self.available_assets:
            raise ValueError(f"Asset {asset} not available")
            
        # Generate cache key
        cache_key = f"{asset}_{timeframe}"
        
        if from_ticks:
            # Generate from tick data
            tick_key = f"{asset}_ticks"
            
            if tick_key not in self.tick_cache:
                raise RuntimeError(f"Tick data for {asset} not loaded")
                
            # Get tick data
            ticks = self.tick_cache[tick_key].copy()
            
            # Determine resampling rule
            tf_minutes = TIME_FRAMES[timeframe]
            
            if tf_minutes < 60:
                # Minutes
                rule = f"{tf_minutes}T"
            elif tf_minutes < 1440:
                # Hours
                rule = f"{tf_minutes // 60}H"
            else:
                # Days
                rule = f"{tf_minutes // 1440}D"
                
            # Resample ticks to OHLCV
            ohlcv = ticks.resample(rule).agg({
                'price': ['first', 'max', 'min', 'last'],
                'volume': 'sum'
            })
            
            # Flatten multi-index columns
            ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
            
        else:
            # Find lowest available timeframe
            available_tfs = sorted(
                self.available_timeframes,
                key=lambda tf: TIME_FRAMES[tf]
            )
            
            if not available_tfs:
                raise RuntimeError("No timeframes available")
                
            lowest_tf = available_tfs[0]
            
            # Get lowest timeframe data
            lowest_key = f"{asset}_{lowest_tf}"
            
            if lowest_key not in self.ohlcv_cache:
                raise RuntimeError(f"OHLCV data for {asset} {lowest_tf} not loaded")
                
            # Convert to target timeframe
            ohlcv = self.get_converted_timeframe(
                asset=asset,
                source_timeframe=lowest_tf,
                target_timeframe=timeframe
            )
        
        # Cache and return
        self.ohlcv_cache[cache_key] = ohlcv
        return ohlcv
    
    def clear_cache(self):
        """Clear all cached data."""
        self.ohlcv_cache = {}
        self.orderbook_cache = {}
        self.tick_cache = {}
        self.news_cache = {}
        self.sentiment_cache = {}
        self.data_loaded = False
        self.logger.info("Cache cleared")


# Backwards compatibility
DataProvider = BacktestDataProvider

