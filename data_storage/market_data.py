#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Market Data Storage Module

This module provides specialized functionality for storing, retrieving, and
managing market data with high performance and reliability. It includes
optimized methods for time series data, order book snapshots, and derived
market metrics.
"""

import os
import time
import json
import zlib
import pickle
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    PYARROW_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    pa = None  # type: ignore
    pq = None  # type: ignore
    PYARROW_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "pyarrow not available; parquet storage disabled"
    )
from pandas.tseries.frequencies import to_offset
import asyncio
try:
    import aiofiles  # type: ignore
    AIOFILES_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    aiofiles = None  # type: ignore
    AIOFILES_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "aiofiles not available; falling back to synchronous file I/O"
    )

if not AIOFILES_AVAILABLE:
    class _AsyncFile:
        def __init__(self, path: str, mode: str):
            self._file = open(path, mode)

        async def __aenter__(self):
            return self._file

        async def __aexit__(self, exc_type, exc, tb):
            self._file.close()
            return False

    def aio_open(path: str, mode: str = "r"):
        return _AsyncFile(path, mode)
else:
    aio_open = aiofiles.open
try:
    import redis  # type: ignore
    REDIS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    redis = None  # type: ignore
    REDIS_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "redis package not available; caching disabled"
    )
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select

# Internal imports
from common.logger import get_logger
from common.utils import (
    TimeFrame, 
    generate_uid, 
    timestamp_to_datetime, 
    datetime_to_timestamp,
    timeit,
    create_batches
)
from common.constants import (
    COMPRESSION_LEVEL,
    MARKET_DATA_RETENTION_POLICY,
    MARKET_DATA_CHUNK_SIZE,
    MARKET_DATA_MAX_WORKERS,
    STORAGE_ROOT_PATH
)
from common.exceptions import (
    StorageError,
    DataIntegrityError,
    DataNotFoundError
)
from common.redis_client import RedisClient
from common.db_client import DatabaseClient, get_db_client
from data_storage.database import DatabaseManager
from data_storage.time_series import TimeSeriesStorage
from data_storage.models.market_data import (
    OHLCVData,
    OrderBookSnapshot,
    TradeData,
    MarketMetrics,
    LiquidityData,
    SentimentData,
    TechnicalIndicators
)
logger = get_logger(__name__)


class MarketDataRepository:
    """
    Repository class for accessing and managing market data from the database.
    Provides a higher-level interface for working with market data models.
    """
    
    def __init__(self, db_session=None):
        """
        Initialize the market data repository.
        
        Args:
            db_session: SQLAlchemy database session
        """
        from data_storage.database import db_session as default_session
        self.db_session = db_session or default_session
        self.logger = get_logger(__name__)
        
    async def get_ohlcv_data(self, asset_id, timeframe, start_time=None, end_time=None, limit=1000):
        """
        Get OHLCV data for an asset within a time range.
        
        Args:
            asset_id: Asset identifier
            timeframe: Timeframe string (e.g., '1h', '1d')
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            limit: Maximum number of records to return
            
        Returns:
            List of OHLCVData objects
        """
        query = select(OHLCVData).filter(
            OHLCVData.asset_id == asset_id,
            OHLCVData.timeframe == timeframe
        )
        
        if start_time:
            query = query.filter(OHLCVData.timestamp >= start_time)
        if end_time:
            query = query.filter(OHLCVData.timestamp <= end_time)
            
        query = query.order_by(OHLCVData.timestamp.desc()).limit(limit)
        
        try:
            result = await self.db_session.execute(query)
            return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data: {str(e)}")
            return []
            
    async def get_technical_indicators(self, asset_id, timeframe, timestamp=None):
        """
        Get technical indicators for an asset at a specific time.
        
        Args:
            asset_id: Asset identifier
            timeframe: Timeframe string
            timestamp: Timestamp (optional, defaults to latest)
            
        Returns:
            TechnicalIndicators object or None
        """
        query = select(TechnicalIndicators).filter(
            TechnicalIndicators.asset_id == asset_id,
            TechnicalIndicators.timeframe == timeframe
        )
        
        if timestamp:
            query = query.filter(TechnicalIndicators.timestamp == timestamp)
        else:
            query = query.order_by(TechnicalIndicators.timestamp.desc()).limit(1)
            
        try:
            result = await self.db_session.execute(query)
            return result.scalars().first()
        except Exception as e:
            self.logger.error(f"Error fetching technical indicators: {str(e)}")
            return None
            
    async def get_sentiment_data(self, asset_id, timeframe, start_time=None, end_time=None):
        """
        Get sentiment data for an asset within a time range.
        
        Args:
            asset_id: Asset identifier
            timeframe: Timeframe string
            start_time: Start timestamp (optional)
            end_time: End timestamp (optional)
            
        Returns:
            List of SentimentData objects
        """
        query = select(SentimentData).filter(
            SentimentData.asset_id == asset_id,
            SentimentData.timeframe == timeframe
        )
        
        if start_time:
            query = query.filter(SentimentData.timestamp >= start_time)
        if end_time:
            query = query.filter(SentimentData.timestamp <= end_time)
            
        query = query.order_by(SentimentData.timestamp.desc())
        
        try:
            result = await self.db_session.execute(query)
            return result.scalars().all()
        except Exception as e:
            self.logger.error(f"Error fetching sentiment data: {str(e)}")
            return []



class MarketDataStore:
    """
    Advanced market data storage system with multi-tiered architecture for different
    data freshness and access patterns:
    
    1. Hot data (recent) - In-memory and Redis
    2. Warm data (medium-term) - Time series optimized database
    3. Cold data (historical) - Compressed parquet files
    """
    
    def __init__(
        self, 
        redis_client: Optional[RedisClient] = None,
        db_client: Optional[DatabaseClient] = None,
        root_path: Optional[str] = None,
        max_workers: int = MARKET_DATA_MAX_WORKERS,
        compression_level: int = COMPRESSION_LEVEL
    ):
        """
        Initialize the market data store with multiple storage backends.
        
        Args:
            redis_client: Redis client for hot data storage
            db_client: Database client for warm data storage
            root_path: Root path for cold data storage
            max_workers: Maximum number of worker threads
            compression_level: Compression level for data storage
        """
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Initializing market data store")
        
        # Set up storage paths
        self.root_path = root_path or os.path.join(STORAGE_ROOT_PATH, 'market_data')
        os.makedirs(self.root_path, exist_ok=True)
        
        # Set up clients
        self.redis_client = redis_client or RedisClient()
        self.db_client = db_client
        self.db_manager = None
        self.ts_store = None
        
        # Runtime settings
        self.max_workers = max_workers
        self.compression_level = compression_level
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Cache settings
        self._cache = {}
        self._cache_stats = {'hits': 0, 'misses': 0}
        self._memory_data = {}  # Ultra-hot data stored directly in memory
        
        # Data settings
        self.retention_policy = MARKET_DATA_RETENTION_POLICY
        self.chunk_size = MARKET_DATA_CHUNK_SIZE
        
        # Initialize storage paths
        self._init_storage_paths()
        
        # Performance metrics
        self.performance_metrics = {
            'write_time': [],
            'read_time': [],
            'compression_ratio': [],
            'query_time': []
        }

        self.logger.info("Market data store initialized")


    async def initialize(self, db_connector: Optional[DatabaseClient] = None) -> None:
        """Initialize database resources for the market data store."""
        if db_connector is not None:
            self.db_client = db_connector
        if self.db_client is None:
            self.db_client = await get_db_client()
        if getattr(self.db_client, "pool", None) is None:
            await self.db_client.initialize()
        self.db_manager = DatabaseManager(self.db_client)
        self.ts_store = TimeSeriesStore(self.db_client)
        await self.db_client.create_tables()

    
    def _init_storage_paths(self):
        """Initialize storage paths for different data types"""
        data_types = [
            'ohlcv', 'order_book', 'trades', 'metrics', 
            'liquidity', 'sentiment', 'indicators'
        ]
        
        # Create directories for each exchange and data type
        for data_type in data_types:
            path = os.path.join(self.root_path, data_type)
            os.makedirs(path, exist_ok=True)
            
            # Create exchange-specific subdirectories
            exchange_dirs = ['binance', 'deriv', 'aggregated']
            for exchange in exchange_dirs:
                exchange_path = os.path.join(path, exchange)
                os.makedirs(exchange_path, exist_ok=True)
    
    @timeit
    async def store_ohlcv_data(
        self, 
        exchange: str,
        symbol: str, 
        timeframe: Union[str, TimeFrame],
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        metadata: Optional[Dict[str, Any]] = None,
        store_cold: bool = True
    ) -> str:
        """
        Store OHLCV data for a symbol and timeframe.
        
        Args:
            exchange: The exchange name
            symbol: The trading symbol
            timeframe: The candle timeframe
            data: DataFrame or list of dictionaries with OHLCV data
            metadata: Additional metadata to store
            store_cold: Whether to store cold data in parquet files
            
        Returns:
            uid: Unique identifier for the stored data
        """
        start_time = time.time()
        
        # Convert timeframe to string if it's an enum
        if isinstance(timeframe, TimeFrame):
            timeframe = timeframe.value
            
        # Convert data to DataFrame if it's a list
        if isinstance(data, list):
            data = pd.DataFrame(data)
        
        # Validate input data
        self._validate_ohlcv_data(data)
        
        # Generate unique ID for this data set
        uid = generate_uid(f"{exchange}_{symbol}_{timeframe}_{datetime.now().isoformat()}")
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
            
        metadata.update({
            'exchange': exchange,
            'symbol': symbol,
            'timeframe': timeframe,
            'start_time': data['timestamp'].min(),
            'end_time': data['timestamp'].max(),
            'num_candles': len(data),
            'stored_at': datetime.now().isoformat(),
            'uid': uid
        })
        
        # Store hot data in Redis
        redis_key = f"ohlcv:{exchange}:{symbol}:{timeframe}:latest"
        latest_data = data.tail(100).to_dict(orient='records')
        await self.redis_client.set(
            redis_key, 
            pickle.dumps(latest_data), 
            expire=self.retention_policy['hot']
        )
        
        # Store hot metadata
        await self.redis_client.set(
            f"{redis_key}:metadata", 
            json.dumps(metadata),
            expire=self.retention_policy['hot']
        )
        
        # Store warm data in time series database
        async with self.db_client.get_session() as session:
            for idx, row in data.iterrows():
                ohlcv_entry = OHLCVData(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=timestamp_to_datetime(row['timestamp']),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume']),
                    uid=uid,
                    metadata=metadata
                )
                session.add(ohlcv_entry)
            await session.commit()
                
        # Store cold data in parquet files if requested
        if store_cold:
            # Get the cold storage path
            date_str = datetime.now().strftime('%Y%m%d')
            cold_path = os.path.join(
                self.root_path, 
                'ohlcv', 
                exchange,
                f"{symbol}_{timeframe}",
            )
            os.makedirs(cold_path, exist_ok=True)
            
            file_path = os.path.join(cold_path, f"{date_str}_{uid}.parquet")
            
            # Optimize data for storage
            storage_data = data.copy()
            
            # Store as parquet file
            table = pa.Table.from_pandas(storage_data)
            
            # Compression options
            compression = 'snappy'  # Fast and decent compression ratio
            
            # Write with metadata
            pq.write_table(
                table, 
                file_path, 
                compression=compression,
                metadata={k: str(v) for k, v in metadata.items()}
            )
            
            # Calculate and log compression ratio
            original_size = storage_data.memory_usage(deep=True).sum()
            compressed_size = os.path.getsize(file_path)
            compression_ratio = original_size / compressed_size
            
            self.performance_metrics['compression_ratio'].append(compression_ratio)
            
            self.logger.debug(
                f"Stored cold OHLCV data for {exchange}:{symbol}:{timeframe} "
                f"with compression ratio {compression_ratio:.2f}"
            )
        
        elapsed = time.time() - start_time
        self.performance_metrics['write_time'].append(elapsed)
        
        self.logger.info(
            f"Stored OHLCV data for {exchange}:{symbol}:{timeframe} "
            f"with {len(data)} candles in {elapsed:.3f}s"
        )
        
        return uid
    
    @timeit
    async def get_ohlcv_data(
        self,
        exchange: str,
        symbol: str,
        timeframe: Union[str, TimeFrame],
        start_time: Optional[Union[int, datetime]] = None,
        end_time: Optional[Union[int, datetime]] = None,
        limit: Optional[int] = None,
        include_metadata: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """
        Retrieve OHLCV data for a symbol and timeframe.
        
        Args:
            exchange: The exchange name
            symbol: The trading symbol
            timeframe: The candle timeframe
            start_time: Start time (timestamp or datetime)
            end_time: End time (timestamp or datetime)
            limit: Maximum number of candles to return
            include_metadata: Whether to include metadata in the result
            
        Returns:
            data: DataFrame with OHLCV data
            metadata: Metadata if include_metadata is True
        """
        start_timer = time.time()
        
        # Convert timeframe to string if it's an enum
        if isinstance(timeframe, TimeFrame):
            timeframe = timeframe.value
        
        # Convert datetime to timestamp if necessary
        if start_time is not None and isinstance(start_time, datetime):
            start_time = datetime_to_timestamp(start_time)
        
        if end_time is not None and isinstance(end_time, datetime):
            end_time = datetime_to_timestamp(end_time)
        
        # Default end_time to now if not provided
        if end_time is None:
            end_time = datetime_to_timestamp(datetime.now())
        
        # If we're querying recent data, try to get it from Redis
        if (start_time is None or 
            (end_time - start_time) < 60 * 60 * 24):  # Less than 1 day of data
            
            redis_key = f"ohlcv:{exchange}:{symbol}:{timeframe}:latest"
            cached_data = await self.redis_client.get(redis_key)
            
            if cached_data:
                self._cache_stats['hits'] += 1
                data = pickle.loads(cached_data)
                df = pd.DataFrame(data)
                
                # Apply filters if provided
                if start_time is not None:
                    df = df[df['timestamp'] >= start_time]
                
                df = df[df['timestamp'] <= end_time]
                
                if limit is not None:
                    df = df.tail(limit)
                
                # Retrieve metadata if requested
                if include_metadata:
                    metadata_key = f"{redis_key}:metadata"
                    metadata_json = await self.redis_client.get(metadata_key)
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    
                    elapsed = time.time() - start_timer
                    self.performance_metrics['read_time'].append(elapsed)
                    
                    self.logger.debug(
                        f"Retrieved OHLCV data from Redis for {exchange}:{symbol}:{timeframe} "
                        f"with {len(df)} candles in {elapsed:.3f}s"
                    )
                    
                    return df, metadata
                
                elapsed = time.time() - start_timer
                self.performance_metrics['read_time'].append(elapsed)
                
                self.logger.debug(
                    f"Retrieved OHLCV data from Redis for {exchange}:{symbol}:{timeframe} "
                    f"with {len(df)} candles in {elapsed:.3f}s"
                )
                
                return df
            
            self._cache_stats['misses'] += 1
        
        # Convert timestamps to datetime for database query
        start_datetime = timestamp_to_datetime(start_time) if start_time else None
        end_datetime = timestamp_to_datetime(end_time)
        
        # Query from database
        async with self.db_client.get_session() as session:
            query = select(OHLCVData).where(
                OHLCVData.exchange == exchange,
                OHLCVData.symbol == symbol,
                OHLCVData.timeframe == timeframe,
                OHLCVData.timestamp <= end_datetime
            )
            
            if start_datetime:
                query = query.where(OHLCVData.timestamp >= start_datetime)
            
            if limit:
                query = query.order_by(OHLCVData.timestamp.desc()).limit(limit)
            else:
                query = query.order_by(OHLCVData.timestamp.asc())
            
            result = await session.execute(query)
            ohlcv_records = result.scalars().all()
            
            if not ohlcv_records:
                # If no data in database, try loading from cold storage
                try:
                    df = await self._load_cold_ohlcv_data(
                        exchange, symbol, timeframe, start_time, end_time, limit
                    )
                    
                    if include_metadata:
                        # Extract metadata from the first record, if available
                        metadata = {}
                        if not df.empty:
                            cold_path = os.path.join(
                                self.root_path, 
                                'ohlcv', 
                                exchange,
                                f"{symbol}_{timeframe}"
                            )
                            parquet_files = [f for f in os.listdir(cold_path) if f.endswith('.parquet')]
                            if parquet_files:
                                # Get metadata from the most recent file
                                parquet_file = sorted(parquet_files)[-1]
                                file_path = os.path.join(cold_path, parquet_file)
                                parquet_metadata = pq.read_metadata(file_path).metadata
                                metadata = {k.decode(): v.decode() for k, v in parquet_metadata.items()}
                        
                        elapsed = time.time() - start_timer
                        self.performance_metrics['read_time'].append(elapsed)
                        
                        self.logger.debug(
                            f"Retrieved OHLCV data from cold storage for {exchange}:{symbol}:{timeframe} "
                            f"with {len(df)} candles in {elapsed:.3f}s"
                        )
                        
                        return df, metadata
                    
                    elapsed = time.time() - start_timer
                    self.performance_metrics['read_time'].append(elapsed)
                    
                    self.logger.debug(
                        f"Retrieved OHLCV data from cold storage for {exchange}:{symbol}:{timeframe} "
                        f"with {len(df)} candles in {elapsed:.3f}s"
                    )
                    
                    return df
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load cold OHLCV data: {str(e)}")
                    raise DataNotFoundError(
                        f"No OHLCV data found for {exchange}:{symbol}:{timeframe}"
                    )
            
            # Convert to DataFrame
            data = []
            metadata = {}
            
            for record in ohlcv_records:
                data.append({
                    'timestamp': datetime_to_timestamp(record.timestamp),
                    'open': float(record.open),
                    'high': float(record.high),
                    'low': float(record.low),
                    'close': float(record.close),
                    'volume': float(record.volume)
                })
                
                # Use the most recent metadata
                if record.metadata:
                    metadata = record.metadata
            
            df = pd.DataFrame(data)
            
            # Apply limit again in case the query didn't correctly limit
            if limit and len(df) > limit:
                df = df.tail(limit)
            
            # Update Redis cache for future queries
            if not df.empty:
                redis_key = f"ohlcv:{exchange}:{symbol}:{timeframe}:latest"
                latest_data = df.tail(100).to_dict(orient='records')
                await self.redis_client.set(
                    redis_key, 
                    pickle.dumps(latest_data), 
                    expire=self.retention_policy['hot']
                )
                
                # Store metadata
                await self.redis_client.set(
                    f"{redis_key}:metadata", 
                    json.dumps(metadata),
                    expire=self.retention_policy['hot']
                )
        
        elapsed = time.time() - start_timer
        self.performance_metrics['read_time'].append(elapsed)
        
        self.logger.debug(
            f"Retrieved OHLCV data from database for {exchange}:{symbol}:{timeframe} "
            f"with {len(df)} candles in {elapsed:.3f}s"
        )
        
        if include_metadata:
            return df, metadata
        
        return df
    
    async def _load_cold_ohlcv_data(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load OHLCV data from cold storage.
        
        Args:
            exchange: The exchange name
            symbol: The trading symbol
            timeframe: The candle timeframe
            start_time: Start time timestamp
            end_time: End time timestamp
            limit: Maximum number of candles to return
            
        Returns:
            data: DataFrame with OHLCV data
        """
        cold_path = os.path.join(
            self.root_path, 
            'ohlcv', 
            exchange,
            f"{symbol}_{timeframe}"
        )
        
        if not os.path.exists(cold_path):
            return pd.DataFrame()
        
        parquet_files = [f for f in os.listdir(cold_path) if f.endswith('.parquet')]
        
        if not parquet_files:
            return pd.DataFrame()
        
        # Read and concatenate all relevant parquet files
        dfs = []
        
        for parquet_file in sorted(parquet_files):
            file_path = os.path.join(cold_path, parquet_file)
            
            # Read parquet file
            try:
                # Use PyArrow to read only necessary columns and rows
                table = pq.read_table(file_path)
                temp_df = table.to_pandas()
                
                # Apply filters
                if start_time is not None:
                    temp_df = temp_df[temp_df['timestamp'] >= start_time]
                
                if end_time is not None:
                    temp_df = temp_df[temp_df['timestamp'] <= end_time]
                
                if not temp_df.empty:
                    dfs.append(temp_df)
            except Exception as e:
                self.logger.warning(f"Error reading parquet file {file_path}: {str(e)}")
        
        if not dfs:
            return pd.DataFrame()
        
        # Concatenate and sort by timestamp
        result_df = pd.concat(dfs).sort_values('timestamp')
        
        # Remove duplicates
        result_df = result_df.drop_duplicates(subset=['timestamp'])
        
        # Apply limit
        if limit is not None and len(result_df) > limit:
            result_df = result_df.tail(limit)
        
        return result_df
    
    @timeit
    async def store_order_book_snapshot(
        self,
        exchange: str,
        symbol: str,
        timestamp: Union[int, datetime],
        bids: List[List[float]],
        asks: List[List[float]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store order book snapshot data.
        
        Args:
            exchange: The exchange name
            symbol: The trading symbol
            timestamp: Snapshot timestamp
            bids: List of [price, quantity] bid entries
            asks: List of [price, quantity] ask entries
            metadata: Additional metadata to store
            
        Returns:
            uid: Unique identifier for the stored data
        """
        start_time = time.time()
        
        # Convert datetime to timestamp if necessary
        if isinstance(timestamp, datetime):
            timestamp = datetime_to_timestamp(timestamp)
        
        # Generate unique ID for this data set
        uid = generate_uid(f"{exchange}_{symbol}_orderbook_{timestamp}")
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
            
        metadata.update({
            'exchange': exchange,
            'symbol': symbol,
            'timestamp': timestamp,
            'num_bids': len(bids),
            'num_asks': len(asks),
            'stored_at': datetime.now().isoformat(),
            'uid': uid
        })
        
        # Convert lists to compressed binary format for efficient storage
        bids_data = zlib.compress(pickle.dumps(bids), level=self.compression_level)
        asks_data = zlib.compress(pickle.dumps(asks), level=self.compression_level)
        
        # Store latest snapshot in Redis
        redis_key = f"orderbook:{exchange}:{symbol}:latest"
        snapshot_data = {
            'timestamp': timestamp,
            'bids': bids,
            'asks': asks,
            'metadata': metadata
        }
        
        await self.redis_client.set(
            redis_key,
            pickle.dumps(snapshot_data),
            expire=self.retention_policy['hot']
        )
        
        # Store in database
        datetime_ts = timestamp_to_datetime(timestamp)
        
        async with self.db_client.get_session() as session:
            order_book = OrderBookSnapshot(
                exchange=exchange,
                symbol=symbol,
                timestamp=datetime_ts,
                bids_data=bids_data,
                asks_data=asks_data,
                uid=uid,
                metadata=metadata
            )
            session.add(order_book)
            await session.commit()
        
        # Store cold data in files if timestamp is not too recent
        now = datetime.now()
        snapshot_time = timestamp_to_datetime(timestamp)
        
        if (now - snapshot_time) > timedelta(minutes=60):
            date_str = snapshot_time.strftime('%Y%m%d')
            hour_str = snapshot_time.strftime('%H')
            
            cold_path = os.path.join(
                self.root_path,
                'order_book',
                exchange,
                symbol,
                date_str
            )
            os.makedirs(cold_path, exist_ok=True)
            
            file_path = os.path.join(cold_path, f"{hour_str}_{timestamp}.data")
            
            # Create a compressed data package
            data_package = {
                'bids': bids,
                'asks': asks,
                'metadata': metadata
            }
            
            compressed_data = zlib.compress(
                pickle.dumps(data_package),
                level=self.compression_level
            )
            
            # Write to file
            async with aio_open(file_path, 'wb') as f:
                await f.write(compressed_data)
        
        elapsed = time.time() - start_time
        self.performance_metrics['write_time'].append(elapsed)
        
        self.logger.debug(
            f"Stored order book snapshot for {exchange}:{symbol} "
            f"with {len(bids)} bids and {len(asks)} asks in {elapsed:.3f}s"
        )
        
        return uid
    
    @timeit
    async def get_order_book_snapshot(
        self,
        exchange: str,
        symbol: str,
        timestamp: Optional[Union[int, datetime]] = None,
        include_metadata: bool = False
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Retrieve order book snapshot data.
        
        Args:
            exchange: The exchange name
            symbol: The trading symbol
            timestamp: Specific timestamp to retrieve (None for latest)
            include_metadata: Whether to include metadata in the result
            
        Returns:
            snapshot: Dictionary with timestamp, bids, and asks
            metadata: Metadata if include_metadata is True
        """
        start_timer = time.time()
        
        # If timestamp is None, get the latest snapshot from Redis
        if timestamp is None:
            redis_key = f"orderbook:{exchange}:{symbol}:latest"
            cached_data = await self.redis_client.get(redis_key)
            
            if cached_data:
                self._cache_stats['hits'] += 1
                snapshot_data = pickle.loads(cached_data)
                
                elapsed = time.time() - start_timer
                self.performance_metrics['read_time'].append(elapsed)
                
                self.logger.debug(
                    f"Retrieved latest order book snapshot for {exchange}:{symbol} "
                    f"from Redis in {elapsed:.3f}s"
                )
                
                if include_metadata:
                    return {
                        'timestamp': snapshot_data['timestamp'],
                        'bids': snapshot_data['bids'],
                        'asks': snapshot_data['asks']
                    }, snapshot_data['metadata']
                
                return {
                    'timestamp': snapshot_data['timestamp'],
                    'bids': snapshot_data['bids'],
                    'asks': snapshot_data['asks']
                }
            
            self._cache_stats['misses'] += 1
        
        # Convert timestamp to datetime for database query if provided
        datetime_ts = None
        if timestamp is not None:
            if isinstance(timestamp, int):
                datetime_ts = timestamp_to_datetime(timestamp)
            else:
                datetime_ts = timestamp
        
        # Query from database
        async with self.db_client.get_session() as session:
            if datetime_ts is not None:
                # Query specific timestamp
                query = select(OrderBookSnapshot).where(
                    OrderBookSnapshot.exchange == exchange,
                    OrderBookSnapshot.symbol == symbol,
                    OrderBookSnapshot.timestamp == datetime_ts
                )
            else:
                # Query latest
                query = select(OrderBookSnapshot).where(
                    OrderBookSnapshot.exchange == exchange,
                    OrderBookSnapshot.symbol == symbol
                ).order_by(OrderBookSnapshot.timestamp.desc()).limit(1)
            
            result = await session.execute(query)
            record = result.scalars().first()
            
            if not record:
                # If no data in database, try loading from cold storage
                if timestamp is not None:
                    try:
                        snapshot = await self._load_cold_order_book_snapshot(
                            exchange, symbol, timestamp
                        )
                        
                        if snapshot:
                            elapsed = time.time() - start_timer
                            self.performance_metrics['read_time'].append(elapsed)
                            
                            self.logger.debug(
                                f"Retrieved order book snapshot for {exchange}:{symbol} "
                                f"at {timestamp} from cold storage in {elapsed:.3f}s"
                            )
                            
                            if include_metadata:
                                return {
                                    'timestamp': snapshot['timestamp'],
                                    'bids': snapshot['bids'],
                                    'asks': snapshot['asks']
                                }, snapshot['metadata']
                            
                            return {
                                'timestamp': snapshot['timestamp'],
                                'bids': snapshot['bids'],
                                'asks': snapshot['asks']
                            }
                    except Exception as e:
                        self.logger.warning(f"Failed to load cold order book data: {str(e)}")
                
                raise DataNotFoundError(
                    f"No order book snapshot found for {exchange}:{symbol}" +
                    (f" at {timestamp}" if timestamp else "")
                )
            
            # Decompress binary data
            bids = pickle.loads(zlib.decompress(record.bids_data))
            asks = pickle.loads(zlib.decompress(record.asks_data))
            
            snapshot = {
                'timestamp': datetime_to_timestamp(record.timestamp),
                'bids': bids,
                'asks': asks
            }
            
            # Update Redis cache for future queries
            redis_key = f"orderbook:{exchange}:{symbol}:latest"
            snapshot_data = {
                'timestamp': snapshot['timestamp'],
                'bids': bids,
                'asks': asks,
                'metadata': record.metadata
            }
            
            await self.redis_client.set(
                redis_key,
                pickle.dumps(snapshot_data),
                expire=self.retention_policy['hot']
            )
            
            elapsed = time.time() - start_timer
            self.performance_metrics['read_time'].append(elapsed)
            
            self.logger.debug(
                f"Retrieved order book snapshot for {exchange}:{symbol} "
                f"from database in {elapsed:.3f}s"
            )
            
            if include_metadata:
                return snapshot, record.metadata
            
            return snapshot
    
    async def _load_cold_order_book_snapshot(
        self,
        exchange: str,
        symbol: str,
        timestamp: Union[int, datetime]
    ) -> Optional[Dict[str, Any]]:
        """
        Load order book snapshot from cold storage.
        
        Args:
            exchange: The exchange name
            symbol: The trading symbol
            timestamp: Snapshot timestamp
            
        Returns:
            snapshot: Dictionary with timestamp, bids, and asks
        """
        # Convert datetime to timestamp if necessary
        if isinstance(timestamp, datetime):
            timestamp = datetime_to_timestamp(timestamp)
        
        # Convert timestamp to datetime for folder structure
        datetime_ts = timestamp_to_datetime(timestamp)
        date_str = datetime_ts.strftime('%Y%m%d')
        hour_str = datetime_ts.strftime('%H')
        
        cold_path = os.path.join(
            self.root_path,
            'order_book',
            exchange,
            symbol,
            date_str
        )
        
        if not os.path.exists(cold_path):
            return None
        
        # Try exact timestamp first
        exact_file = os.path.join(cold_path, f"{hour_str}_{timestamp}.data")
        if os.path.exists(exact_file):
            async with aio_open(exact_file, 'rb') as f:
                compressed_data = await f.read()
                data_package = pickle.loads(zlib.decompress(compressed_data))
                
                return {
                    'timestamp': timestamp,
                    'bids': data_package['bids'],
                    'asks': data_package['asks'],
                    'metadata': data_package.get('metadata', {})
                }
        
        # If exact file not found, try to find closest timestamp
        files = [f for f in os.listdir(cold_path) if f.startswith(f"{hour_str}_") and f.endswith(".data")]
        
        if not files:
            return None
        
        # Extract timestamps from filenames
        timestamps = [int(f.split('_')[1].split('.')[0]) for f in files]
        
        # Find the closest timestamp
        closest_ts = min(timestamps, key=lambda x: abs(x - timestamp))
        closest_file = os.path.join(cold_path, f"{hour_str}_{closest_ts}.data")
        
        async with aio_open(closest_file, 'rb') as f:
            compressed_data = await f.read()
            data_package = pickle.loads(zlib.decompress(compressed_data))
            
            return {
                'timestamp': closest_ts,
                'bids': data_package['bids'],
                'asks': data_package['asks'],
                'metadata': data_package.get('metadata', {})
            }
    
    @timeit
    async def store_trades(
        self,
        exchange: str,
        symbol: str,
        trades: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store trade data for a symbol.
        
        Args:
            exchange: The exchange name
            symbol: The trading symbol
            trades: List of trade dictionaries with id, timestamp, price, amount, side
            metadata: Additional metadata to store
            
        Returns:
            uid: Unique identifier for the stored data
        """
        start_time = time.time()
        
        if not trades:
            self.logger.warning(f"Empty trades list for {exchange}:{symbol}")
            return generate_uid(f"{exchange}_{symbol}_empty_{datetime.now().isoformat()}")
        
        # Generate unique ID for this data set
        uid = generate_uid(f"{exchange}_{symbol}_trades_{datetime.now().isoformat()}")
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
            
        metadata.update({
            'exchange': exchange,
            'symbol': symbol,
            'start_time': min(t['timestamp'] for t in trades),
            'end_time': max(t['timestamp'] for t in trades),
            'num_trades': len(trades),
            'stored_at': datetime.now().isoformat(),
            'uid': uid
        })
        
        # Store recent trades in Redis
        redis_key = f"trades:{exchange}:{symbol}:recent"
        recent_trades = sorted(trades, key=lambda x: x['timestamp'], reverse=True)[:100]
        
        await self.redis_client.set(
            redis_key,
            pickle.dumps(recent_trades),
            expire=self.retention_policy['hot']
        )
        
        # Store metadata
        await self.redis_client.set(
            f"{redis_key}:metadata",
            json.dumps(metadata),
            expire=self.retention_policy['hot']
        )
        
        # Store in database
        async with self.db_client.get_session() as session:
            # Process in batches to avoid overwhelming the database
            for trade in trades:
                trade_entry = TradeData(
                    exchange=exchange,
                    symbol=symbol,
                    trade_id=str(trade.get('id', '')),
                    timestamp=timestamp_to_datetime(trade['timestamp']),
                    price=float(trade['price']),
                    amount=float(trade['amount']),
                    side=trade.get('side', 'unknown'),
                    uid=uid,
                    metadata=trade.get('metadata', {})
                )
                session.add(trade_entry)
            
            await session.commit()
        
        # Store cold data in files
        # Group trades by hour for efficient storage
        trades_by_hour = {}
        for trade in trades:
            trade_dt = timestamp_to_datetime(trade['timestamp'])
            hour_key = trade_dt.strftime('%Y%m%d_%H')
            
            if hour_key not in trades_by_hour:
                trades_by_hour[hour_key] = []
                
            trades_by_hour[hour_key].append(trade)
        
        # Save each hour's trades to a separate file
        for hour_key, hour_trades in trades_by_hour.items():
            date_str, hour_str = hour_key.split('_')
            
            cold_path = os.path.join(
                self.root_path,
                'trades',
                exchange,
                symbol,
                date_str
            )
            os.makedirs(cold_path, exist_ok=True)
            
            file_path = os.path.join(cold_path, f"{hour_str}_{uid}.data")
            
            # Create a compressed data package
            data_package = {
                'trades': hour_trades,
                'metadata': metadata
            }
            
            compressed_data = zlib.compress(
                pickle.dumps(data_package),
                level=self.compression_level
            )
            
            # Write to file
            async with aio_open(file_path, 'wb') as f:
                await f.write(compressed_data)
        
        elapsed = time.time() - start_time
        self.performance_metrics['write_time'].append(elapsed)
        
        self.logger.debug(
            f"Stored {len(trades)} trades for {exchange}:{symbol} in {elapsed:.3f}s"
        )
        
        return uid
    
    @timeit
    async def get_trades(
        self,
        exchange: str,
        symbol: str,
        start_time: Optional[Union[int, datetime]] = None,
        end_time: Optional[Union[int, datetime]] = None,
        limit: Optional[int] = None,
        include_metadata: bool = False
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        """
        Retrieve trade data for a symbol.
        
        Args:
            exchange: The exchange name
            symbol: The trading symbol
            start_time: Start time (timestamp or datetime)
            end_time: End time (timestamp or datetime)
            limit: Maximum number of trades to return
            include_metadata: Whether to include metadata in the result
            
        Returns:
            trades: List of trade dictionaries
            metadata: Metadata if include_metadata is True
        """
        start_timer = time.time()
        
        # Convert datetime to timestamp if necessary
        if start_time is not None and isinstance(start_time, datetime):
            start_time = datetime_to_timestamp(start_time)
        
        if end_time is not None and isinstance(end_time, datetime):
            end_time = datetime_to_timestamp(end_time)
        
        # Default end_time to now if not provided
        if end_time is None:
            end_time = datetime_to_timestamp(datetime.now())
        
        # If we're querying recent data and no start_time is provided, 
        # try to get from Redis
        if start_time is None and limit is not None:
            redis_key = f"trades:{exchange}:{symbol}:recent"
            cached_data = await self.redis_client.get(redis_key)
            
            if cached_data:
                self._cache_stats['hits'] += 1
                trades = pickle.loads(cached_data)
                
                # Apply filters
                trades = [t for t in trades if t['timestamp'] <= end_time]
                
                if limit is not None and len(trades) > limit:
                    trades = trades[:limit]
                
                # Retrieve metadata if requested
                if include_metadata:
                    metadata_key = f"{redis_key}:metadata"
                    metadata_json = await self.redis_client.get(metadata_key)
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    
                    elapsed = time.time() - start_timer
                    self.performance_metrics['read_time'].append(elapsed)
                    
                    self.logger.debug(
                        f"Retrieved trades from Redis for {exchange}:{symbol} "
                        f"with {len(trades)} trades in {elapsed:.3f}s"
                    )
                    
                    return trades, metadata
                
                elapsed = time.time() - start_timer
                self.performance_metrics['read_time'].append(elapsed)
                
                self.logger.debug(
                    f"Retrieved trades from Redis for {exchange}:{symbol} "
                    f"with {len(trades)} trades in {elapsed:.3f}s"
                )
                
                return trades
            
            self._cache_stats['misses'] += 1
        
        # Convert timestamps to datetime for database query
        start_datetime = timestamp_to_datetime(start_time) if start_time else None
        end_datetime = timestamp_to_datetime(end_time)
        
        # Query from database
        async with self.db_client.get_session() as session:
            query = select(TradeData).where(
                TradeData.exchange == exchange,
                TradeData.symbol == symbol,
                TradeData.timestamp <= end_datetime
            )
            
            if start_datetime:
                query = query.where(TradeData.timestamp >= start_datetime)
            
            if limit:
                query = query.order_by(TradeData.timestamp.desc()).limit(limit)
            else:
                query = query.order_by(TradeData.timestamp.asc())
            
            result = await session.execute(query)
            trade_records = result.scalars().all()
            
            if not trade_records:
                # If no data in database, try loading from cold storage
                try:
                    trades_result = await self._load_cold_trades(
                        exchange, symbol, start_time, end_time, limit
                    )
                    
                    if include_metadata:
                        trades, metadata = trades_result
                        
                        elapsed = time.time() - start_timer
                        self.performance_metrics['read_time'].append(elapsed)
                        
                        self.logger.debug(
                            f"Retrieved trades from cold storage for {exchange}:{symbol} "
                            f"with {len(trades)} trades in {elapsed:.3f}s"
                        )
                        
                        return trades, metadata
                    else:
                        trades = trades_result
                        
                        elapsed = time.time() - start_timer
                        self.performance_metrics['read_time'].append(elapsed)
                        
                        self.logger.debug(
                            f"Retrieved trades from cold storage for {exchange}:{symbol} "
                            f"with {len(trades)} trades in {elapsed:.3f}s"
                        )
                        
                        return trades
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load cold trade data: {str(e)}")
                    return [] if not include_metadata else ([], {})
            
            # Convert to list of dictionaries
            trades = []
            metadata = {}
            
            for record in trade_records:
                trades.append({
                    'id': record.trade_id,
                    'timestamp': datetime_to_timestamp(record.timestamp),
                    'price': float(record.price),
                    'amount': float(record.amount),
                    'side': record.side,
                    'metadata': record.metadata
                })
                
                # Use the most recent metadata
                if record.metadata:
                    metadata = record.metadata
            
            # Apply limit again in case the query didn't correctly limit
            if limit and len(trades) > limit:
                trades = trades[:limit]
            
            # Update Redis cache for future queries
            if trades:
                redis_key = f"trades:{exchange}:{symbol}:recent"
                recent_trades = sorted(trades, key=lambda x: x['timestamp'], reverse=True)[:100]
                
                await self.redis_client.set(
                    redis_key,
                    pickle.dumps(recent_trades),
                    expire=self.retention_policy['hot']
                )
                
                # Store metadata
                await self.redis_client.set(
                    f"{redis_key}:metadata",
                    json.dumps(metadata),
                    expire=self.retention_policy['hot']
                )
            
            elapsed = time.time() - start_timer
            self.performance_metrics['read_time'].append(elapsed)
            
            self.logger.debug(
                f"Retrieved trades from database for {exchange}:{symbol} "
                f"with {len(trades)} trades in {elapsed:.3f}s"
            )
            
            if include_metadata:
                return trades, metadata
            
            return trades
    
    async def _load_cold_trades(
        self,
        exchange: str,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        """
        Load trade data from cold storage.
        
        Args:
            exchange: The exchange name
            symbol: The trading symbol
            start_time: Start time timestamp
            end_time: End time timestamp
            limit: Maximum number of trades to return
            
        Returns:
            trades: List of trade dictionaries
            metadata: Metadata dictionary
        """
        # Determine date range to search
        if start_time is None:
            # If no start time, use last 24 hours
            start_datetime = timestamp_to_datetime(end_time) - timedelta(days=1)
        else:
            start_datetime = timestamp_to_datetime(start_time)
            
        end_datetime = timestamp_to_datetime(end_time)
        
        # Generate list of dates to search
        current_date = start_datetime.date()
        end_date = end_datetime.date()
        date_list = []
        
        while current_date <= end_date:
            date_list.append(current_date.strftime('%Y%m%d'))
            current_date += timedelta(days=1)
        
        # Search for trade files in each date directory
        all_trades = []
        metadata = {}
        
        for date_str in date_list:
            date_path = os.path.join(
                self.root_path,
                'trades',
                exchange,
                symbol,
                date_str
            )
            
            if not os.path.exists(date_path):
                continue
            
            # Get all hour files for this date
            hour_files = [f for f in os.listdir(date_path) if f.endswith('.data')]
            
            for hour_file in hour_files:
                file_path = os.path.join(date_path, hour_file)
                
                try:
                    async with aio_open(file_path, 'rb') as f:
                        compressed_data = await f.read()
                        data_package = pickle.loads(zlib.decompress(compressed_data))
                        
                        file_trades = data_package['trades']
                        file_metadata = data_package.get('metadata', {})
                        
                        # Apply time filters
                        if start_time is not None:
                            file_trades = [t for t in file_trades if t['timestamp'] >= start_time]
                        
                        file_trades = [t for t in file_trades if t['timestamp'] <= end_time]
                        
                        all_trades.extend(file_trades)
                        
                        # Use the most recent metadata
                        if file_metadata:
                            metadata = file_metadata
                except Exception as e:
                    self.logger.warning(f"Error reading trade file {file_path}: {str(e)}")
        
        # Sort by timestamp
        all_trades.sort(key=lambda x: x['timestamp'])
        
        # Apply limit
        if limit is not None and len(all_trades) > limit:
            all_trades = all_trades[:limit]
        
        return all_trades, metadata
    
    def _validate_ohlcv_data(self, data: pd.DataFrame) -> None:
        """
        Validate OHLCV data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Raises:
            DataIntegrityError: If data is invalid
        """
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(data.columns)
        
        if missing_columns:
            raise DataIntegrityError(
                f"Missing required columns: {missing_columns}"
            )
        
        # Validate data types
        for col in required_columns:
            if col == 'timestamp':
                if not np.issubdtype(data[col].dtype, np.number):
                    raise DataIntegrityError(
                        f"Column {col} must be numeric, got {data[col].dtype}"
                    )
            else:
                if not np.issubdtype(data[col].dtype, np.number):
                    raise DataIntegrityError(
                        f"Column {col} must be numeric, got {data[col].dtype}"
                    )
        
        # Validate high >= low
        if not all(data['high'] >= data['low']):
            raise DataIntegrityError("High must be greater than or equal to Low")
        
        # Validate high >= open, high >= close
        if not all(data['high'] >= data['open']) or not all(data['high'] >= data['close']):
            raise DataIntegrityError("High must be greater than or equal to Open and Close")
        
        # Validate low <= open, low <= close
        if not all(data['low'] <= data['open']) or not all(data['low'] <= data['close']):
            raise DataIntegrityError("Low must be less than or equal to Open and Close")
        
        # Validate volume >= 0
        if not all(data['volume'] >= 0):
            raise DataIntegrityError("Volume must be non-negative")
        
        # Validate timestamps are sorted
        if not all(data['timestamp'].diff().fillna(1) > 0):
            raise DataIntegrityError("Timestamps must be sorted in ascending order")
    
    @timeit
    async def store_technical_indicators(
        self,
        exchange: str,
        symbol: str,
        timeframe: Union[str, TimeFrame],
        timestamp: Union[int, datetime],
        indicators: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store technical indicators.
        
        Args:
            exchange: The exchange name
            symbol: The trading symbol
            timeframe: The indicator timeframe
            timestamp: Indicator timestamp
            indicators: Dictionary of indicator values
            metadata: Additional metadata to store
            
        Returns:
            uid: Unique identifier for the stored data
        """
        start_time = time.time()
        
        # Convert timeframe to string if it's an enum
        if isinstance(timeframe, TimeFrame):
            timeframe = timeframe.value
            
        # Convert datetime to timestamp if necessary
        if isinstance(timestamp, datetime):
            timestamp = datetime_to_timestamp(timestamp)
        
        # Generate unique ID for this data set
        uid = generate_uid(f"{exchange}_{symbol}_{timeframe}_indicators_{timestamp}")
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
            
        metadata.update({
            'exchange': exchange,
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': timestamp,
            'indicator_count': len(indicators),
            'stored_at': datetime.now().isoformat(),
            'uid': uid
        })
        
        # Compress indicators for storage
        indicators_data = zlib.compress(
            pickle.dumps(indicators),
            level=self.compression_level
        )
        
        # Store in Redis for fast access
        redis_key = f"indicators:{exchange}:{symbol}:{timeframe}:latest"
        redis_data = {
            'timestamp': timestamp,
            'indicators': indicators,
            'metadata': metadata
        }
        
        await self.redis_client.set(
            redis_key,
            pickle.dumps(redis_data),
            expire=self.retention_policy['hot']
        )
        
        # Store in database
        datetime_ts = timestamp_to_datetime(timestamp)
        
        async with self.db_client.get_session() as session:
            indicator_entry = TechnicalIndicators(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime_ts,
                indicators_data=indicators_data,
                uid=uid,
                metadata=metadata
            )
            session.add(indicator_entry)
            await session.commit()
        
        elapsed = time.time() - start_time
        self.performance_metrics['write_time'].append(elapsed)
        
        self.logger.debug(
            f"Stored technical indicators for {exchange}:{symbol}:{timeframe} "
            f"with {len(indicators)} indicators in {elapsed:.3f}s"
        )
        
        return uid
    
    @timeit
    async def get_technical_indicators(
        self,
        exchange: str,
        symbol: str,
        timeframe: Union[str, TimeFrame],
        timestamp: Optional[Union[int, datetime]] = None,
        indicators: Optional[List[str]] = None,
        include_metadata: bool = False
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Retrieve technical indicators.
        
        Args:
            exchange: The exchange name
            symbol: The trading symbol
            timeframe: The indicator timeframe
            timestamp: Specific timestamp to retrieve (None for latest)
            indicators: List of specific indicators to retrieve (None for all)
            include_metadata: Whether to include metadata in the result
            
        Returns:
            indicators: Dictionary of indicator values
            metadata: Metadata if include_metadata is True
        """
        start_timer = time.time()
        
        # Convert timeframe to string if it's an enum
        if isinstance(timeframe, TimeFrame):
            timeframe = timeframe.value
        
        # If timestamp is None, get the latest from Redis
        if timestamp is None:
            redis_key = f"indicators:{exchange}:{symbol}:{timeframe}:latest"
            cached_data = await self.redis_client.get(redis_key)
            
            if cached_data:
                self._cache_stats['hits'] += 1
                redis_data = pickle.loads(cached_data)
                
                result_indicators = redis_data['indicators']
                
                # Filter indicators if requested
                if indicators is not None:
                    result_indicators = {k: v for k, v in result_indicators.items() if k in indicators}
                
                elapsed = time.time() - start_timer
                self.performance_metrics['read_time'].append(elapsed)
                
                self.logger.debug(
                    f"Retrieved technical indicators from Redis for {exchange}:{symbol}:{timeframe} "
                    f"with {len(result_indicators)} indicators in {elapsed:.3f}s"
                )
                
                if include_metadata:
                    return result_indicators, redis_data['metadata']
                
                return result_indicators
            
            self._cache_stats['misses'] += 1
        
        # Convert timestamp to datetime for database query if provided
        datetime_ts = None
        if timestamp is not None:
            if isinstance(timestamp, int):
                datetime_ts = timestamp_to_datetime(timestamp)
            else:
                datetime_ts = timestamp
        
        # Query from database
        async with self.db_client.get_session() as session:
            if datetime_ts is not None:
                # Query specific timestamp
                query = select(TechnicalIndicators).where(
                    TechnicalIndicators.exchange == exchange,
                    TechnicalIndicators.symbol == symbol,
                    TechnicalIndicators.timeframe == timeframe,
                    TechnicalIndicators.timestamp == datetime_ts
                )
            else:
                # Query latest
                query = select(TechnicalIndicators).where(
                    TechnicalIndicators.exchange == exchange,
                    TechnicalIndicators.symbol == symbol,
                    TechnicalIndicators.timeframe == timeframe
                ).order_by(TechnicalIndicators.timestamp.desc()).limit(1)
            
            result = await session.execute(query)
            record = result.scalars().first()
            
            if not record:
                raise DataNotFoundError(
                    f"No technical indicators found for {exchange}:{symbol}:{timeframe}" +
                    (f" at {timestamp}" if timestamp else "")
                )
            
            # Decompress indicators data
            all_indicators = pickle.loads(zlib.decompress(record.indicators_data))
            
            # Filter indicators if requested
            if indicators is not None:
                all_indicators = {k: v for k, v in all_indicators.items() if k in indicators}
            
            # Update Redis cache for future queries
            redis_key = f"indicators:{exchange}:{symbol}:{timeframe}:latest"
            redis_data = {
                'timestamp': datetime_to_timestamp(record.timestamp),
                'indicators': all_indicators,
                'metadata': record.metadata
            }
            
            await self.redis_client.set(
                redis_key,
                pickle.dumps(redis_data),
                expire=self.retention_policy['hot']
            )
            
            elapsed = time.time() - start_timer
            self.performance_metrics['read_time'].append(elapsed)
            
            self.logger.debug(
                f"Retrieved technical indicators from database for {exchange}:{symbol}:{timeframe} "
                f"with {len(all_indicators)} indicators in {elapsed:.3f}s"
            )
            
            if include_metadata:
                return all_indicators, record.metadata
            
            return all_indicators
    
    async def purge_old_data(self, older_than_days: int = 30) -> int:
        """
        Purge old data from storage.
        
        Args:
            older_than_days: Purge data older than this many days
            
        Returns:
            count: Number of items purged
        """
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        purge_count = 0
        
        self.logger.info(f"Purging data older than {cutoff_date}")
        
        # Purge from database
        async with self.db_client.get_session() as session:
            # Purge OHLCV data
            result = await session.execute(
                select(OHLCVData).where(OHLCVData.timestamp < cutoff_date)
            )
            ohlcv_records = result.scalars().all()
            
            for record in ohlcv_records:
                session.delete(record)
                purge_count += 1
            
            # Purge order book data
            result = await session.execute(
                select(OrderBookSnapshot).where(OrderBookSnapshot.timestamp < cutoff_date)
            )
            orderbook_records = result.scalars().all()
            
            for record in orderbook_records:
                session.delete(record)
                purge_count += 1
            
            # Purge trade data
            result = await session.execute(
                select(TradeData).where(TradeData.timestamp < cutoff_date)
            )
            trade_records = result.scalars().all()
            
            for record in trade_records:
                session.delete(record)
                purge_count += 1
            
            # Purge technical indicators
            result = await session.execute(
                select(TechnicalIndicators).where(TechnicalIndicators.timestamp < cutoff_date)
            )
            indicator_records = result.scalars().all()
            
            for record in indicator_records:
                session.delete(record)
                purge_count += 1
            
            await session.commit()
        
        self.logger.info(f"Purged {purge_count} records from database")
        
        return purge_count
    
    async def close(self):
        """Close all connections and resources"""
        self.logger.info("Closing market data store")
        
        # Close executor
        self.executor.shutdown()
        
        # Close Redis client
        await self.redis_client.close()
        
        # Close database client
        await self.db_client.close()

        self.logger.info("Market data store closed")

# Backward compatibility for older modules
MarketDataStorage = MarketDataStore

__all__ = [
    "MarketDataStore",
    "MarketDataStorage",
]
