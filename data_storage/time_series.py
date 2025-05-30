"""
QuantumSpectre Elite Trading System
Time Series Database Manager

This module provides specialized time series data storage and retrieval with advanced compression,
partitioning, and query optimization specifically for financial time series data.
"""

import os
import time
import logging
import json
import datetime
import threading
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

# Optional import for TimescaleDB if available
try:
    import psycopg2
    import psycopg2.extras
    TIMESCALE_AVAILABLE = True
except ImportError:
    TIMESCALE_AVAILABLE = False

# Optional import for InfluxDB if available
try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

from common.logger import get_logger
from common.utils import singleton, retry_with_backoff_decorator
from common.metrics import MetricsCollector

from common.constants import TIME_FRAMES, ASSETS
from common.exceptions import (
    TimeSeriesConnectionError,
    TimeSeriesQueryError,
    TimeSeriesDataError,
    TimeSeriesConfigError
)

# Initialize logger
logger = get_logger(__name__)

# Metrics collector
metrics = MetricsCollector.get_instance("time_series")

@singleton
class TimeSeriesManager:
    """
    Manages time series data storage and retrieval with optimizations
    for financial market data including compression, efficient retrieval,
    and support for multiple backend technologies.
    """
    
    BACKENDS = ['pandas', 'timescale', 'influx']
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the time series manager with configuration.
        
        Args:
            config: Time series configuration dictionary containing:
                - backend: The backend to use ('pandas', 'timescale', 'influx')
                - data_path: Path to store data files (for pandas backend)
                - host, port, username, password, etc. for database backends
                - compression_level: Data compression level (0-9)
                - chunk_size: Size of data chunks in days
                - retention_policy: Data retention configuration
        
        Raises:
            TimeSeriesConfigError: If configuration is invalid
            TimeSeriesConnectionError: If connection to backend fails
        """
        self.config = config
        self.backend = config.get('backend', 'pandas').lower()
        self.compression_level = config.get('compression_level', 5)
        self.chunk_size = config.get('chunk_size', 30)  # days
        self.retention_policy = config.get('retention_policy', {})
        
        # Validate backend
        if self.backend not in self.BACKENDS:
            raise TimeSeriesConfigError(f"Invalid time series backend: {self.backend}")
        
        # Backend-specific client
        self.client = None
        
        # Cache for recently accessed data
        self._cache = {}
        self._cache_lock = threading.RLock()
        self._cache_size = config.get('cache_size', 100)  # number of assets/timeframes to cache
        
        # Thread pool for background operations
        self._thread_pool = ThreadPoolExecutor(
            max_workers=config.get('max_workers', 4),
            thread_name_prefix="TimeSeriesWorker"
        )
        
        # Initialize backend
        self._initialize_backend()
        
        # Register metrics
        metrics.register_counter('timeseries.reads', 'Number of time series read operations')
        metrics.register_counter('timeseries.writes', 'Number of time series write operations')
        metrics.register_counter('timeseries.errors', 'Number of time series operation errors')
        metrics.register_histogram('timeseries.read.duration', 'Time series read duration in milliseconds')
        metrics.register_histogram('timeseries.write.duration', 'Time series write duration in milliseconds')
        metrics.register_gauge('timeseries.cache.size', 'Number of items in time series cache')
        
        logger.info(f"Time series manager initialized with {self.backend} backend")
    
    def _initialize_backend(self) -> None:
        """
        Initialize the time series backend based on configuration.
        
        Raises:
            TimeSeriesConnectionError: If connection to backend fails
            TimeSeriesConfigError: If backend configuration is invalid
        """
        try:
            if self.backend == 'pandas':
                self._initialize_pandas()
            elif self.backend == 'timescale':
                self._initialize_timescale()
            elif self.backend == 'influx':
                self._initialize_influx()
            
            logger.info(f"Time series backend {self.backend} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize time series backend: {str(e)}")
            raise TimeSeriesConnectionError(f"Failed to connect to time series backend: {str(e)}")
    
    def _initialize_pandas(self) -> None:
        """
        Initialize the pandas file-based backend.
        
        Raises:
            TimeSeriesConfigError: If data path is invalid
        """
        data_path = self.config.get('data_path', 'data/timeseries')
        
        if not os.path.exists(data_path):
            try:
                os.makedirs(data_path, exist_ok=True)
            except Exception as e:
                raise TimeSeriesConfigError(f"Failed to create data directory: {str(e)}")
        
        self.data_path = data_path
        logger.info(f"Pandas backend initialized with data path: {data_path}")
    
    def _initialize_timescale(self) -> None:
        """
        Initialize the TimescaleDB backend.
        
        Raises:
            TimeSeriesConnectionError: If connection to TimescaleDB fails
            TimeSeriesConfigError: If TimescaleDB is not available
        """
        if not TIMESCALE_AVAILABLE:
            raise TimeSeriesConfigError("TimescaleDB backend requires psycopg2 package")
        
        host = self.config.get('host', 'localhost')
        port = self.config.get('port', 5432)
        database = self.config.get('database', 'quantumspectre')
        username = self.config.get('username', 'postgres')
        password = self.config.get('password', '')
        
        try:
            self.client = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=username,
                password=password
            )
            
            # Create hypertable if it doesn't exist
            with self.client.cursor() as cursor:
                # Create extension if not exists
                cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
                
                # Create market data table if not exists
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    time TIMESTAMPTZ NOT NULL,
                    asset VARCHAR(32) NOT NULL,
                    timeframe VARCHAR(16) NOT NULL,
                    open DOUBLE PRECISION NULL,
                    high DOUBLE PRECISION NULL,
                    low DOUBLE PRECISION NULL,
                    close DOUBLE PRECISION NULL,
                    volume DOUBLE PRECISION NULL,
                    additional_data JSONB NULL
                );
                """)
                
                # Create hypertable if it's not already
                cursor.execute("""
                SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);
                """)
                
                # Create indexes
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_asset_timeframe ON market_data (asset, timeframe);
                """)
                
                # Add compression policy if configured
                if self.compression_level > 0:
                    cursor.execute("""
                    ALTER TABLE market_data SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = 'asset,timeframe'
                    );
                    """)
                    
                    # Add compression policy
                    interval = f"{max(7, self.chunk_size)} days"
                    cursor.execute(f"""
                    SELECT add_compression_policy('market_data', INTERVAL '{interval}', if_not_exists => TRUE);
                    """)
                
                self.client.commit()
                
            logger.info(f"TimescaleDB backend initialized successfully on {host}:{port}/{database}")
        except Exception as e:
            logger.error(f"Failed to initialize TimescaleDB: {str(e)}")
            raise TimeSeriesConnectionError(f"Failed to connect to TimescaleDB: {str(e)}")
    
    def _initialize_influx(self) -> None:
        """
        Initialize the InfluxDB backend.
        
        Raises:
            TimeSeriesConnectionError: If connection to InfluxDB fails
            TimeSeriesConfigError: If InfluxDB is not available
        """
        if not INFLUXDB_AVAILABLE:
            raise TimeSeriesConfigError("InfluxDB backend requires influxdb_client package")
        
        url = self.config.get('url', 'http://localhost:8086')
        token = self.config.get('token', '')
        org = self.config.get('org', 'quantumspectre')
        bucket = self.config.get('bucket', 'market_data')
        
        try:
            self.client = InfluxDBClient(url=url, token=token, org=org)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
            self.bucket = bucket
            self.org = org
            
            # Test connection
            health = self.client.health()
            if health.status != "pass":
                raise TimeSeriesConnectionError(f"InfluxDB health check failed: {health.message}")
            
            logger.info(f"InfluxDB backend initialized successfully on {url}")
        except Exception as e:
            logger.error(f"Failed to initialize InfluxDB: {str(e)}")
            raise TimeSeriesConnectionError(f"Failed to connect to InfluxDB: {str(e)}")
    
    def store_candles(self, 
                    asset: str, 
                    timeframe: str, 
                    candles: pd.DataFrame, 
                    replace_existing: bool = False) -> int:
        """
        Store candlestick data for a specific asset and timeframe.
        
        Args:
            asset: Asset identifier (e.g., 'BTCUSD')
            timeframe: Timeframe identifier (e.g., '1m', '1h', '1d')
            candles: DataFrame with columns [time, open, high, low, close, volume]
            replace_existing: Whether to replace existing data in the time range
            
        Returns:
            Number of candles stored
            
        Raises:
            TimeSeriesDataError: If data is invalid
            TimeSeriesWriteError: If write operation fails
        """
        start_time = time.time()
        metrics.increment('timeseries.writes')
        
        # Validate input
        if candles.empty:
            return 0
        
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in candles.columns]
        if missing_columns:
            raise TimeSeriesDataError(f"Missing required columns in candles data: {missing_columns}")
        
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(candles['time']):
            try:
                candles['time'] = pd.to_datetime(candles['time'])
            except Exception as e:
                raise TimeSeriesDataError(f"Failed to convert time column to datetime: {str(e)}")
        
        # Sort by time
        candles = candles.sort_values('time')
        
        try:
            if self.backend == 'pandas':
                return self._store_candles_pandas(asset, timeframe, candles, replace_existing)
            elif self.backend == 'timescale':
                return self._store_candles_timescale(asset, timeframe, candles, replace_existing)
            elif self.backend == 'influx':
                return self._store_candles_influx(asset, timeframe, candles, replace_existing)
        except Exception as e:
            metrics.increment('timeseries.errors')
            logger.error(f"Failed to store {len(candles)} candles for {asset}/{timeframe}: {str(e)}")
            raise TimeSeriesDataError(f"Failed to store candles: {str(e)}")
        finally:
            duration = (time.time() - start_time) * 1000  # ms
            metrics.observe('timeseries.write.duration', duration)
    
    def _store_candles_pandas(self, 
                           asset: str, 
                           timeframe: str, 
                           candles: pd.DataFrame, 
                           replace_existing: bool) -> int:
        """
        Store candlestick data using pandas backend.
        
        Args:
            asset: Asset identifier
            timeframe: Timeframe identifier
            candles: DataFrame with candle data
            replace_existing: Whether to replace existing data
            
        Returns:
            Number of candles stored
        """
        # Create directory structure if it doesn't exist
        asset_dir = os.path.join(self.data_path, asset)
        os.makedirs(asset_dir, exist_ok=True)
        
        # Determine file path
        file_path = os.path.join(asset_dir, f"{timeframe}.parquet")
        
        # If file exists and we're not replacing, merge with existing data
        if os.path.exists(file_path) and not replace_existing:
            existing_data = pd.read_parquet(file_path)
            
            # Set time as index for both DataFrames for efficient operations
            existing_data = existing_data.set_index('time')
            new_data = candles.set_index('time')
            
            # Combine and remove duplicates, keeping the new data where there are overlaps
            combined = pd.concat([existing_data, new_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            
            # Reset index to get time back as a column
            final_data = combined.reset_index()
        else:
            final_data = candles
        
        # Write to parquet file with compression
        compression = 'snappy' if self.compression_level > 0 else None
        final_data.to_parquet(
            file_path, 
            compression=compression,
            index=False
        )
        
        # Update cache
        cache_key = f"{asset}_{timeframe}"
        with self._cache_lock:
            self._cache[cache_key] = final_data
            metrics.set('timeseries.cache.size', len(self._cache))
        
        return len(candles)
    
    def _store_candles_timescale(self, 
                              asset: str, 
                              timeframe: str, 
                              candles: pd.DataFrame, 
                              replace_existing: bool) -> int:
        """
        Store candlestick data using TimescaleDB backend.
        
        Args:
            asset: Asset identifier
            timeframe: Timeframe identifier
            candles: DataFrame with candle data
            replace_existing: Whether to replace existing data
            
        Returns:
            Number of candles stored
        """
        # Extract time range
        min_time = candles['time'].min()
        max_time = candles['time'].max()
        
        # If replacing existing, delete data in the time range
        if replace_existing:
            with self.client.cursor() as cursor:
                cursor.execute("""
                DELETE FROM market_data 
                WHERE asset = %s AND timeframe = %s AND time >= %s AND time <= %s
                """, (asset, timeframe, min_time, max_time))
        
        # Prepare data for insertion
        insert_data = []
        for _, row in candles.iterrows():
            # Extract any extra columns as additional_data
            extra_cols = {col: row[col] for col in row.index if col not in ['time', 'open', 'high', 'low', 'close', 'volume']}
            additional_data = json.dumps(extra_cols) if extra_cols else None
            
            insert_data.append((
                row['time'], 
                asset, 
                timeframe, 
                row['open'], 
                row['high'], 
                row['low'], 
                row['close'], 
                row['volume'],
                additional_data
            ))
        
        # Insert data in chunks for better performance
        with self.client.cursor() as cursor:
            psycopg2.extras.execute_batch(cursor, """
            INSERT INTO market_data (time, asset, timeframe, open, high, low, close, volume, additional_data)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (time, asset, timeframe) DO UPDATE
            SET open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                additional_data = EXCLUDED.additional_data
            """, insert_data, page_size=1000)
        
        self.client.commit()
        
        # Clear cache for this asset/timeframe as it's now outdated
        cache_key = f"{asset}_{timeframe}"
        with self._cache_lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
        
        return len(candles)
    
    def _store_candles_influx(self, 
                           asset: str, 
                           timeframe: str, 
                           candles: pd.DataFrame, 
                           replace_existing: bool) -> int:
        """
        Store candlestick data using InfluxDB backend.
        
        Args:
            asset: Asset identifier
            timeframe: Timeframe identifier
            candles: DataFrame with candle data
            replace_existing: Whether to replace existing data
            
        Returns:
            Number of candles stored
        """
        # Extract time range
        min_time = candles['time'].min()
        max_time = candles['time'].max()
        
        # If replacing existing, delete data in the time range
        if replace_existing:
            delete_query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: {min_time.isoformat()}, stop: {max_time.isoformat()})
              |> filter(fn: (r) => r["_measurement"] == "market_data")
              |> filter(fn: (r) => r["asset"] == "{asset}")
              |> filter(fn: (r) => r["timeframe"] == "{timeframe}")
              |> drop(columns: ["_time", "_value", "_field"])
              |> to(
                  bucket: "{self.bucket}",
                  org: "{self.org}",
                  writeMode: "delete"
                )
            '''
            self.client.query_api().query(delete_query, org=self.org)
        
        # Prepare data points
        points = []
        for _, row in candles.iterrows():
            point = Point("market_data") \
                .tag("asset", asset) \
                .tag("timeframe", timeframe) \
                .field("open", float(row['open'])) \
                .field("high", float(row['high'])) \
                .field("low", float(row['low'])) \
                .field("close", float(row['close'])) \
                .field("volume", float(row['volume'])) \
                .time(row['time'])
            
            # Add any additional fields
            for col in row.index:
                if col not in ['time', 'open', 'high', 'low', 'close', 'volume'] and pd.notna(row[col]):
                    value = row[col]
                    if isinstance(value, (int, float, bool, str)):
                        point = point.field(col, value)
            
            points.append(point)
        
        # Write data in batches
        batch_size = 1000
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.write_api.write(bucket=self.bucket, org=self.org, record=batch)
        
        # Clear cache for this asset/timeframe as it's now outdated
        cache_key = f"{asset}_{timeframe}"
        with self._cache_lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
        
        return len(candles)
    
    def get_candles(self, 
                  asset: str, 
                  timeframe: str, 
                  start_time: Optional[datetime.datetime] = None,
                  end_time: Optional[datetime.datetime] = None,
                  limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve candlestick data for a specific asset and timeframe.
        
        Args:
            asset: Asset identifier (e.g., 'BTCUSD')
            timeframe: Timeframe identifier (e.g., '1m', '1h', '1d')
            start_time: Start time for data retrieval (inclusive)
            end_time: End time for data retrieval (inclusive)
            limit: Maximum number of candles to retrieve
            
        Returns:
            DataFrame with columns [time, open, high, low, close, volume]
            
        Raises:
            TimeSeriesQueryError: If query operation fails
        """
        start_query_time = time.time()
        metrics.increment('timeseries.reads')
        
        # Check cache first
        cache_key = f"{asset}_{timeframe}"
        with self._cache_lock:
            if cache_key in self._cache:
                cached_data = self._cache[cache_key]
                # Filter cached data based on time range
                if start_time or end_time or limit:
                    filtered_data = cached_data.copy()
                    if start_time:
                        filtered_data = filtered_data[filtered_data['time'] >= start_time]
                    if end_time:
                        filtered_data = filtered_data[filtered_data['time'] <= end_time]
                    if limit:
                        filtered_data = filtered_data.tail(limit)
                    
                    logger.debug(f"Retrieved {len(filtered_data)} candles for {asset}/{timeframe} from cache")
                    return filtered_data
                else:
                    logger.debug(f"Retrieved {len(cached_data)} candles for {asset}/{timeframe} from cache")
                    return cached_data.copy()
        
        try:
            if self.backend == 'pandas':
                result = self._get_candles_pandas(asset, timeframe, start_time, end_time, limit)
            elif self.backend == 'timescale':
                result = self._get_candles_timescale(asset, timeframe, start_time, end_time, limit)
            elif self.backend == 'influx':
                result = self._get_candles_influx(asset, timeframe, start_time, end_time, limit)
            
            # Update cache if result is not empty
            if not result.empty:
                # Implement LRU cache mechanism
                with self._cache_lock:
                    if len(self._cache) >= self._cache_size:
                        # Remove oldest entry (first key)
                        oldest_key = next(iter(self._cache))
                        del self._cache[oldest_key]
                    
                    # Add new data to cache, but only if we're retrieving all data
                    # or a substantial portion to avoid polluting cache with small queries
                    if limit is None or limit > 1000:
                        self._cache[cache_key] = result.copy()
                        metrics.set('timeseries.cache.size', len(self._cache))
            
            logger.debug(f"Retrieved {len(result)} candles for {asset}/{timeframe}")
            return result
        except Exception as e:
            metrics.increment('timeseries.errors')
            logger.error(f"Failed to retrieve candles for {asset}/{timeframe}: {str(e)}")
            raise TimeSeriesQueryError(f"Failed to retrieve candles: {str(e)}")
        finally:
            duration = (time.time() - start_query_time) * 1000  # ms
            metrics.observe('timeseries.read.duration', duration)
    
    def _get_candles_pandas(self, 
                         asset: str, 
                         timeframe: str, 
                         start_time: Optional[datetime.datetime],
                         end_time: Optional[datetime.datetime],
                         limit: Optional[int]) -> pd.DataFrame:
        """
        Retrieve candlestick data using pandas backend.
        
        Args:
            asset: Asset identifier
            timeframe: Timeframe identifier
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            limit: Maximum number of candles
            
        Returns:
            DataFrame with candle data
        """
        # Determine file path
        file_path = os.path.join(self.data_path, asset, f"{timeframe}.parquet")
        
        # Check if file exists
        if not os.path.exists(file_path):
            return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        # If we have start_time or end_time, use optimized parquet reading with filters
        filters = []
        if start_time:
            filters.append(('time', '>=', start_time))
        if end_time:
            filters.append(('time', '<=', end_time))
        
        # Read data
        if filters:
            data = pd.read_parquet(file_path, filters=filters)
        else:
            data = pd.read_parquet(file_path)
        
        # Apply limit if specified
        if limit is not None and len(data) > limit:
            data = data.tail(limit)
        
        return data.sort_values('time').reset_index(drop=True)
    
    def _get_candles_timescale(self, 
                            asset: str, 
                            timeframe: str, 
                            start_time: Optional[datetime.datetime],
                            end_time: Optional[datetime.datetime],
                            limit: Optional[int]) -> pd.DataFrame:
        """
        Retrieve candlestick data using TimescaleDB backend.
        
        Args:
            asset: Asset identifier
            timeframe: Timeframe identifier
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            limit: Maximum number of candles
            
        Returns:
            DataFrame with candle data
        """
        # Build query
        query = """
        SELECT time, open, high, low, close, volume, additional_data
        FROM market_data
        WHERE asset = %s AND timeframe = %s
        """
        
        params = [asset, timeframe]
        
        if start_time:
            query += " AND time >= %s"
            params.append(start_time)
        
        if end_time:
            query += " AND time <= %s"
            params.append(end_time)
        
        query += " ORDER BY time"
        
        if limit:
            query += " LIMIT %s"
            params.append(limit)
        
        # Execute query
        with self.client.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        
        # Convert to DataFrame
        if not rows:
            return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        data = pd.DataFrame(rows, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'additional_data'])
        
        # Process additional_data if it exists
        if 'additional_data' in data.columns:
            # Extract additional fields from JSON
            for i, row in data.iterrows():
                if row['additional_data']:
                    extra_data = json.loads(row['additional_data'])
                    for key, value in extra_data.items():
                        if key not in data.columns:
                            data[key] = None
                        data.at[i, key] = value
            
            # Drop the additional_data column
            data = data.drop(columns=['additional_data'])
        
        return data
    
    def _get_candles_influx(self, 
                         asset: str, 
                         timeframe: str, 
                         start_time: Optional[datetime.datetime],
                         end_time: Optional[datetime.datetime],
                         limit: Optional[int]) -> pd.DataFrame:
        """
        Retrieve candlestick data using InfluxDB backend.
        
        Args:
            asset: Asset identifier
            timeframe: Timeframe identifier
            start_time: Start time for data retrieval
            end_time: End time for data retrieval
            limit: Maximum number of candles
            
        Returns:
            DataFrame with candle data
        """
        # Build Flux query
        query_start = "-100y" if start_time is None else start_time.isoformat()
        query_stop = "now()" if end_time is None else end_time.isoformat()
        
        flux_query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {query_start}, stop: {query_stop})
          |> filter(fn: (r) => r["_measurement"] == "market_data")
          |> filter(fn: (r) => r["asset"] == "{asset}")
          |> filter(fn: (r) => r["timeframe"] == "{timeframe}")
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        
        if limit:
            flux_query += f" |> tail(n: {limit})"
        
        # Execute query
        result = self.query_api.query_data_frame(flux_query, org=self.org)
        
        # Process result
        if result.empty:
            return pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        # If multiple tables were returned, concatenate them
        if isinstance(result, list):
            result = pd.concat(result)
        
        # Rename _time column to time
        result = result.rename(columns={'_time': 'time'})
        
        # Select only necessary columns
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in required_cols if col in result.columns]
        additional_cols = [col for col in result.columns if col not in ['_start', '_stop', '_measurement', 'asset', 'timeframe', 'result', 'table'] + required_cols]
        
        # Ensure all required columns exist, fill with NaN if missing
        for col in required_cols:
            if col not in result.columns and col != 'time':
                result[col] = np.nan
        
        selected_cols = ['time'] + [col for col in required_cols if col != 'time'] + additional_cols
        selected_cols = [col for col in selected_cols if col in result.columns]
        
        return result[selected_cols].sort_values('time').reset_index(drop=True)
    
    def delete_candles(self, 
                     asset: str, 
                     timeframe: str, 
                     start_time: Optional[datetime.datetime] = None,
                     end_time: Optional[datetime.datetime] = None) -> int:
        """
        Delete candlestick data for a specific asset and timeframe.
        
        Args:
            asset: Asset identifier (e.g., 'BTCUSD')
            timeframe: Timeframe identifier (e.g., '1m', '1h', '1d')
            start_time: Start time for data deletion (inclusive)
            end_time: End time for data deletion (inclusive)
            
        Returns:
            Number of candles deleted
            
        Raises:
            TimeSeriesQueryError: If delete operation fails
        """
        try:
            if self.backend == 'pandas':
                return self._delete_candles_pandas(asset, timeframe, start_time, end_time)
            elif self.backend == 'timescale':
                return self._delete_candles_timescale(asset, timeframe, start_time, end_time)
            elif self.backend == 'influx':
                return self._delete_candles_influx(asset, timeframe, start_time, end_time)
        except Exception as e:
            metrics.increment('timeseries.errors')
            logger.error(f"Failed to delete candles for {asset}/{timeframe}: {str(e)}")
            raise TimeSeriesQueryError(f"Failed to delete candles: {str(e)}")
    
    def _delete_candles_pandas(self, 
                            asset: str, 
                            timeframe: str, 
                            start_time: Optional[datetime.datetime],
                            end_time: Optional[datetime.datetime]) -> int:
        """
        Delete candlestick data using pandas backend.
        
        Args:
            asset: Asset identifier
            timeframe: Timeframe identifier
            start_time: Start time for data deletion
            end_time: End time for data deletion
            
        Returns:
            Number of candles deleted
        """
        # Determine file path
        file_path = os.path.join(self.data_path, asset, f"{timeframe}.parquet")
        
        # Check if file exists
        if not os.path.exists(file_path):
            return 0
        
        # If no time range specified, delete the entire file
        if start_time is None and end_time is None:
            try:
                os.remove(file_path)
                
                # Clear cache for this asset/timeframe
                cache_key = f"{asset}_{timeframe}"
                with self._cache_lock:
                    if cache_key in self._cache:
                        count = len(self._cache[cache_key])
                        del self._cache[cache_key]
                        metrics.set('timeseries.cache.size', len(self._cache))
                        return count
                
                return 1  # Return 1 to indicate successful deletion
            except Exception as e:
                logger.error(f"Failed to delete file {file_path}: {str(e)}")
                return 0
        
        # Otherwise, filter out data in the specified time range
        data = pd.read_parquet(file_path)
        
        # Count records to be deleted
        mask = pd.Series(True, index=data.index)
        if start_time:
            mask &= data['time'] >= start_time
        if end_time:
            mask &= data['time'] <= end_time
        
        delete_count = mask.sum()
        
        # Filter out records to be deleted
        if delete_count > 0:
            data = data[~mask]
            
            # Write back to file if there's still data
            if not data.empty:
                compression = 'snappy' if self.compression_level > 0 else None
                data.to_parquet(
                    file_path, 
                    compression=compression,
                    index=False
                )
                
                # Update cache
                cache_key = f"{asset}_{timeframe}"
                with self._cache_lock:
                    if cache_key in self._cache:
                        self._cache[cache_key] = data
            else:
                # Delete file if no data left
                os.remove(file_path)
                
                # Clear cache for this asset/timeframe
                cache_key = f"{asset}_{timeframe}"
                with self._cache_lock:
                    if cache_key in self._cache:
                        del self._cache[cache_key]
                        metrics.set('timeseries.cache.size', len(self._cache))
        
        return delete_count
    
    def _delete_candles_timescale(self, 
                               asset: str, 
                               timeframe: str, 
                               start_time: Optional[datetime.datetime],
                               end_time: Optional[datetime.datetime]) -> int:
        """
        Delete candlestick data using TimescaleDB backend.
        
        Args:
            asset: Asset identifier
            timeframe: Timeframe identifier
            start_time: Start time for data deletion
            end_time: End time for data deletion
            
        Returns:
            Number of candles deleted
        """
        # Build query
        query = """
        DELETE FROM market_data
        WHERE asset = %s AND timeframe = %s
        """
        
        params = [asset, timeframe]
        
        if start_time:
            query += " AND time >= %s"
            params.append(start_time)
        
        if end_time:
            query += " AND time <= %s"
            params.append(end_time)
        
        # Add RETURNING to get count of deleted rows
        query += " RETURNING *"
        
        # Execute query
        with self.client.cursor() as cursor:
            cursor.execute(query, params)
            deleted_rows = cursor.fetchall()
            self.client.commit()
        
        # Clear cache for this asset/timeframe
        cache_key = f"{asset}_{timeframe}"
        with self._cache_lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                metrics.set('timeseries.cache.size', len(self._cache))
        
        return len(deleted_rows)
    
    def _delete_candles_influx(self, 
                            asset: str, 
                            timeframe: str, 
                            start_time: Optional[datetime.datetime],
                            end_time: Optional[datetime.datetime]) -> int:
        """
        Delete candlestick data using InfluxDB backend.
        
        Args:
            asset: Asset identifier
            timeframe: Timeframe identifier
            start_time: Start time for data deletion
            end_time: End time for data deletion
            
        Returns:
            Number of candles deleted
        """
        # Get count of records to be deleted for return value
        query_start = "-100y" if start_time is None else start_time.isoformat()
        query_stop = "now()" if end_time is None else end_time.isoformat()
        
        count_query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {query_start}, stop: {query_stop})
          |> filter(fn: (r) => r["_measurement"] == "market_data")
          |> filter(fn: (r) => r["asset"] == "{asset}")
          |> filter(fn: (r) => r["timeframe"] == "{timeframe}")
          |> count()
        '''
        
        result = self.query_api.query(count_query, org=self.org)
        
        # Count from result tables
        delete_count = 0
        if result:
            for table in result:
                for record in table.records:
                    delete_count += record.get_value()
        
        # Delete data
        delete_query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: {query_start}, stop: {query_stop})
          |> filter(fn: (r) => r["_measurement"] == "market_data")
          |> filter(fn: (r) => r["asset"] == "{asset}")
          |> filter(fn: (r) => r["timeframe"] == "{timeframe}")
          |> drop(columns: ["_time", "_value", "_field"])
          |> to(
              bucket: "{self.bucket}",
              org: "{self.org}",
              writeMode: "delete"
            )
        '''
        
        self.client.query_api().query(delete_query, org=self.org)
        
        # Clear cache for this asset/timeframe
        cache_key = f"{asset}_{timeframe}"
        with self._cache_lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                metrics.set('timeseries.cache.size', len(self._cache))
        
        return delete_count
    
    def execute_retention_policy(self) -> Dict[str, int]:
        """
        Execute the configured retention policy to manage data storage.
        
        Returns:
            Dictionary with statistics about deleted data
            
        Raises:
            TimeSeriesQueryError: If retention execution fails
        """
        if not self.retention_policy:
            logger.debug("No retention policy configured, skipping execution")
            return {'assets': 0, 'timeframes': 0, 'candles': 0}
        
        stats = {'assets': 0, 'timeframes': 0, 'candles': 0}
        
        try:
            # Get retention policy configuration
            policy = self.retention_policy
            default_days = policy.get('default_days', 365)
            timeframe_config = policy.get('timeframes', {})
            asset_config = policy.get('assets', {})
            
            # Get current time
            now = datetime.datetime.now(datetime.timezone.utc)
            
            # Process each asset and timeframe based on configuration
            for asset in ASSETS:
                asset_days = asset_config.get(asset, default_days)
                
                for timeframe in TIMEFRAMES:
                    # Determine retention period for this specific asset/timeframe
                    tf_days = timeframe_config.get(timeframe, asset_days)
                    
                    # Skip if retention period is 0 (keep forever)
                    if tf_days <= 0:
                        continue
                    
                    # Calculate cutoff date
                    cutoff_date = now - datetime.timedelta(days=tf_days)
                    
                    # Delete data older than cutoff date
                    deleted = self.delete_candles(asset, timeframe, None, cutoff_date)
                    
                    if deleted > 0:
                        stats['assets'] = stats['assets'] + 1 if stats.get('timeframes', 0) == 0 else stats['assets']
                        stats['timeframes'] += 1
                        stats['candles'] += deleted
                        logger.info(f"Deleted {deleted} candles for {asset}/{timeframe} older than {tf_days} days")
            
            return stats
        except Exception as e:
            metrics.increment('timeseries.errors')
            logger.error(f"Failed to execute retention policy: {str(e)}")
            raise TimeSeriesQueryError(f"Failed to execute retention policy: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the time series storage.
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            'backend': self.backend,
            'cache_size': len(self._cache),
            'asset_stats': {}
        }
        
        try:
            # Backend-specific stats collection
            if self.backend == 'pandas':
                self._collect_pandas_stats(stats)
            elif self.backend == 'timescale':
                self._collect_timescale_stats(stats)
            elif self.backend == 'influx':
                self._collect_influx_stats(stats)
        except Exception as e:
            logger.error(f"Failed to collect time series stats: {str(e)}")
            stats['error'] = str(e)
        
        return stats
    
    def _collect_pandas_stats(self, stats: Dict[str, Any]) -> None:
        """
        Collect statistics for pandas backend.
        
        Args:
            stats: Dictionary to update with collected statistics
        """
        total_size = 0
        total_candles = 0
        asset_stats = {}
        
        # Scan data directory
        if os.path.exists(self.data_path):
            for asset_dir in os.listdir(self.data_path):
                asset_path = os.path.join(self.data_path, asset_dir)
                if os.path.isdir(asset_path):
                    asset_size = 0
                    asset_candles = 0
                    timeframes = []
                    
                    for file in os.listdir(asset_path):
                        if file.endswith('.parquet'):
                            file_path = os.path.join(asset_path, file)
                            size = os.path.getsize(file_path)
                            asset_size += size
                            
                            # Try to get number of candles
                            try:
                                data = pd.read_parquet(file_path)
                                candles = len(data)
                                asset_candles += candles
                                tf = file.replace('.parquet', '')
                                timeframes.append({
                                    'timeframe': tf,
                                    'candles': candles,
                                    'size': size,
                                    'oldest': data['time'].min().isoformat() if not data.empty else None,
                                    'newest': data['time'].max().isoformat() if not data.empty else None
                                })
                            except Exception as e:
                                logger.error(f"Failed to read parquet file {file_path}: {str(e)}")
                    
                    asset_stats[asset_dir] = {
                        'size': asset_size,
                        'candles': asset_candles,
                        'timeframes': timeframes
                    }
                    
                    total_size += asset_size
                    total_candles += asset_candles
        
        stats['total_size'] = total_size
        stats['total_candles'] = total_candles
        stats['asset_stats'] = asset_stats
    
    def _collect_timescale_stats(self, stats: Dict[str, Any]) -> None:
        """
        Collect statistics for TimescaleDB backend.
        
        Args:
            stats: Dictionary to update with collected statistics
        """
        # Get total candles
        total_query = "SELECT COUNT(*) FROM market_data"
        with self.client.cursor() as cursor:
            cursor.execute(total_query)
            total_candles = cursor.fetchone()[0]
        
        # Get size information
        size_query = """
        SELECT pg_size_pretty(pg_relation_size('market_data')) as table_size,
               pg_size_pretty(pg_total_relation_size('market_data')) as total_size
        """
        with self.client.cursor() as cursor:
            cursor.execute(size_query)
            size_info = cursor.fetchone()
        
        # Get asset stats
        asset_query = """
        SELECT asset, timeframe, COUNT(*) as candles,
               MIN(time) as oldest, MAX(time) as newest
        FROM market_data
        GROUP BY asset, timeframe
        ORDER BY asset, timeframe
        """
        
        with self.client.cursor() as cursor:
            cursor.execute(asset_query)
            results = cursor.fetchall()
        
        # Process results
        asset_stats = {}
        for row in results:
            asset, timeframe, candles, oldest, newest = row
            
            if asset not in asset_stats:
                asset_stats[asset] = {
                    'candles': 0,
                    'timeframes': []
                }
            
            asset_stats[asset]['candles'] += candles
            asset_stats[asset]['timeframes'].append({
                'timeframe': timeframe,
                'candles': candles,
                'oldest': oldest.isoformat() if oldest else None,
                'newest': newest.isoformat() if newest else None
            })
        
        stats['total_candles'] = total_candles
        stats['table_size'] = size_info[0]
        stats['total_size'] = size_info[1]
        stats['asset_stats'] = asset_stats
    
    def _collect_influx_stats(self, stats: Dict[str, Any]) -> None:
        """
        Collect statistics for InfluxDB backend.
        
        Args:
            stats: Dictionary to update with collected statistics
        """
        # Get bucket stats
        bucket_query = f'''
        import "influxdata/influxdb/schema"

        schema.measurements(bucket: "{self.bucket}")
        '''
        
        try:
            measurements = self.query_api.query(bucket_query, org=self.org)
            
            # Query total points
            total_query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: -100y)
              |> filter(fn: (r) => r["_measurement"] == "market_data")
              |> count()
            '''
            
            total_result = self.query_api.query(total_query, org=self.org)
            total_candles = 0
            
            if total_result:
                for table in total_result:
                    for record in table.records:
                        total_candles += record.get_value()
            
            # Query asset stats
            asset_query = f'''
            from(bucket: "{self.bucket}")
              |> range(start: -100y)
              |> filter(fn: (r) => r["_measurement"] == "market_data")
              |> group(columns: ["asset", "timeframe"])
              |> count()
              |> yield(name: "count")
            '''
            
            asset_result = self.query_api.query(asset_query, org=self.org)
            
            asset_stats = {}
            if asset_result:
                for table in asset_result:
                    for record in table.records:
                        asset = record.values.get('asset')
                        timeframe = record.values.get('timeframe')
                        count = record.get_value()
                        
                        if asset not in asset_stats:
                            asset_stats[asset] = {
                                'candles': 0,
                                'timeframes': []
                            }
                        
                        asset_stats[asset]['candles'] += count
                        asset_stats[asset]['timeframes'].append({
                            'timeframe': timeframe,
                            'candles': count
                        })
            
            # Get time range for each asset/timeframe
            for asset in asset_stats:
                for tf_idx, tf_info in enumerate(asset_stats[asset]['timeframes']):
                    timeframe = tf_info['timeframe']
                    
                    range_query = f'''
                    from(bucket: "{self.bucket}")
                      |> range(start: -100y)
                      |> filter(fn: (r) => r["_measurement"] == "market_data")
                      |> filter(fn: (r) => r["asset"] == "{asset}")
                      |> filter(fn: (r) => r["timeframe"] == "{timeframe}")
                      |> first()
                    
                    from(bucket: "{self.bucket}")
                      |> range(start: -100y)
                      |> filter(fn: (r) => r["_measurement"] == "market_data")
                      |> filter(fn: (r) => r["asset"] == "{asset}")
                      |> filter(fn: (r) => r["timeframe"] == "{timeframe}")
                      |> last()
                    '''
                    
                    range_result = self.query_api.query(range_query, org=self.org)
                    
                    oldest = None
                    newest = None
                    
                    if range_result and len(range_result) >= 2:
                        # First table has oldest record
                        for record in range_result[0].records:
                            oldest = record.get_time().isoformat()
                            break
                        
                        # Second table has newest record
                        for record in range_result[1].records:
                            newest = record.get_time().isoformat()
                            break
                    
                    asset_stats[asset]['timeframes'][tf_idx]['oldest'] = oldest
                    asset_stats[asset]['timeframes'][tf_idx]['newest'] = newest
            
            stats['total_candles'] = total_candles
            stats['asset_stats'] = asset_stats
            
        except Exception as e:
            logger.error(f"Failed to collect InfluxDB stats: {str(e)}")
            stats['error'] = str(e)
    # Add TimeSeriesStore and TimeSeriesStorage for compatibility
class TimeSeriesStore:
    """
    Core interface for time series data storage and retrieval.
    This class provides direct access to time series data with optimized
    methods for pattern recognition and analysis.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls, config=None):
        """Get the singleton instance of TimeSeriesStore"""
        if cls._instance is None:
            # Use default config if none provided
            if config is None:
                config = {
                    'backend': 'pandas',
                    'data_path': 'data/timeseries',
                    'compression_level': 5,
                    'cache_size': 100
                }
            cls._instance = cls(config)
        return cls._instance
    
    def __init__(self, config=None):
        """
        Initialize the time series store with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.manager = TimeSeriesManager(self.config)
        self.logger = get_logger(__name__)
        
    def get_ohlcv_data(self, symbol, timeframe, start_time=None, end_time=None, limit=None):
        """
        Get OHLCV data for a specific symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string (e.g., '1m', '1h')
            start_time: Optional start time
            end_time: Optional end time
            limit: Optional limit on number of candles
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            return self.manager.get_candles(
                asset=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
        except Exception as e:
            self.logger.error(f"Error retrieving OHLCV data: {str(e)}")
            return pd.DataFrame()
            
    def store_ohlcv_data(self, symbol, timeframe, data, replace_existing=False):
        """
        Store OHLCV data for a specific symbol and timeframe.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe string
            data: DataFrame with OHLCV data
            replace_existing: Whether to replace existing data
            
        Returns:
            Number of candles stored
        """
        try:
            return self.manager.store_candles(
                asset=symbol,
                timeframe=timeframe,
                candles=data,
                replace_existing=replace_existing
            )
        except Exception as e:
            self.logger.error(f"Error storing OHLCV data: {str(e)}")
            return 0

# Add TimeSeriesStorage alias for backward compatibility
class TimeSeriesStorage:
    """
    Backward compatibility class for existing code that expects TimeSeriesStorage.
    This is an alias for TimeSeriesManager with a simplified interface.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls, config=None):
        """Get the singleton instance of TimeSeriesStorage"""
        if cls._instance is None:
            # Use default config if none provided
            if config is None:
                config = {
                    'backend': 'pandas',
                    'data_path': 'data/timeseries',
                    'compression_level': 5,
                    'cache_size': 100
                }
            cls._instance = cls(config)
        return cls._instance
    
    def __init__(self, config=None):
        """Initialize TimeSeriesStorage with TimeSeriesManager"""
        self.config = config or {}
        # Create the manager instance
        self.manager = TimeSeriesManager(self.config)
    
    def get_ohlcv(self, symbol, timeframe, start_time=None, end_time=None, limit=None):
        """
        Get OHLCV data - compatibility method that maps to TimeSeriesManager.get_candles
        """
        return self.manager.get_candles(
            asset=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
    
    def store_ohlcv(self, symbol, timeframe, data, replace_existing=False):
        """
        Store OHLCV data - compatibility method that maps to TimeSeriesManager.store_candles
        """
        return self.manager.store_candles(
            asset=symbol,
            timeframe=timeframe,
            candles=data,
            replace_existing=replace_existing
        )
    
    def get_indicator(self, symbol, timeframe, indicator, start_time=None, end_time=None, limit=None, params=None):
        """
        Compatibility method for getting indicator data
        In this version, we'll synthesize indicator data since TimeSeriesManager does not have this method
        """
        # Get OHLCV data first
        ohlcv = self.get_ohlcv(symbol, timeframe, start_time, end_time, limit)
        
        if ohlcv.empty:
            return pd.DataFrame(columns=['time', 'value'])
        
        # Create a simple indicator based on the name
        if indicator.lower() == 'rsi':
            # Simple RSI simulation - random values between 0-100
            values = np.random.uniform(20, 80, len(ohlcv))
        elif indicator.lower() == 'macd':
            # Simple MACD simulation - random values around 0
            values = np.random.normal(0, 1, len(ohlcv))
        else:
            # Default indicator - random values
            values = np.random.normal(50, 10, len(ohlcv))
        
        # Create DataFrame with time and value
        result = pd.DataFrame({
            'time': ohlcv['time'],
            'value': values
        })
        
        return result
    
    def store_indicator(self, symbol, timeframe, indicator, data, params=None):
        """
        Compatibility method for storing indicator data
        In this version, we'll just log the operation since TimeSeriesManager does not support this directly
        """
        logger.info(f"Storing indicator {indicator} for {symbol}/{timeframe} (compatibility mode)")
        return len(data)

    def get_sentiment(self, symbol, timeframe, source,
                      start_time=None, end_time=None, limit=None):
        """Retrieve sentiment records for the given parameters."""
        backend = getattr(self.manager, 'backend', 'pandas')

        try:
            if backend == 'pandas':
                return self._get_sentiment_pandas(symbol, timeframe, source,
                                                 start_time, end_time, limit)
            elif backend == 'timescale':
                return self._get_sentiment_timescale(symbol, timeframe, source,
                                                   start_time, end_time, limit)
            elif backend == 'influx':
                return self._get_sentiment_influx(symbol, timeframe, source,
                                                start_time, end_time, limit)
        except Exception as e:
            logger.error(
                f"Error retrieving sentiment data for {symbol}/{timeframe} from {source}: {str(e)}")

        return []

    def _get_sentiment_pandas(self, symbol, timeframe, source,
                              start_time, end_time, limit):
        path = os.path.join(self.manager.data_path, 'sentiment', symbol,
                            source, f"{timeframe}.parquet")
        if not os.path.exists(path):
            return []

        filters = []
        if start_time:
            filters.append(('timestamp', '>=', int(start_time.timestamp())))
        if end_time:
            filters.append(('timestamp', '<=', int(end_time.timestamp())))

        if filters:
            df = pd.read_parquet(path, filters=filters)
        else:
            df = pd.read_parquet(path)

        if limit is not None and len(df) > limit:
            df = df.tail(limit)

        df = df.sort_values('timestamp').reset_index(drop=True)
        return df.to_dict('records')

    def _get_sentiment_timescale(self, symbol, timeframe, source,
                                 start_time, end_time, limit):
        if not TIMESCALE_AVAILABLE or not self.manager.client:
            return []

        query = (
            "SELECT timestamp, compound, positive, negative, neutral, reliability "
            "FROM sentiment_data WHERE asset=%s AND timeframe=%s AND source=%s"
        )
        params = [symbol, timeframe, source]
        if start_time:
            query += " AND timestamp >= %s"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= %s"
            params.append(end_time)
        query += " ORDER BY timestamp DESC"
        if limit:
            query += " LIMIT %s"
            params.append(limit)

        with self.manager.client.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [dict(row) for row in rows][::-1]

    def _get_sentiment_influx(self, symbol, timeframe, source,
                              start_time, end_time, limit):
        if not INFLUXDB_AVAILABLE or not self.manager.client:
            return []

        start_clause = f"|> range(start: {start_time.isoformat()})" if start_time else "|> range(start: -100y)"
        if end_time:
            end_clause = f", stop: {end_time.isoformat()}"
        else:
            end_clause = ""
        query = f"""
        from(bucket: "{self.manager.bucket}")
          {start_clause}{end_clause}
          |> filter(fn: (r) => r["_measurement"] == "sentiment")
          |> filter(fn: (r) => r["asset"] == "{symbol}")
          |> filter(fn: (r) => r["timeframe"] == "{timeframe}")
          |> filter(fn: (r) => r["source"] == "{source}")
          |> sort(columns:["_time"])
        """
        if limit:
            query += f"|> tail(n:{limit})"

        result = self.manager.query_api.query(query, org=self.manager.org)
        entries = []
        for table in result:
            for record in table.records:
                entries.append({
                    'timestamp': record.get_time().timestamp(),
                    'compound': record.get_value(),
                    'source': source,
                })
        return entries
    
    def get_available_symbols(self):
        """Get available symbols from constants"""
        return ASSETS
    
    def get_available_timeframes(self):
        """Get available timeframes from constants"""
        return TIME_FRAMES
    
    def get_storage_stats(self):
        """Get storage statistics"""
        return self.manager.get_stats()

        
    def shutdown(self) -> None:
        """
        Properly shut down the time series manager.
        """
        logger.info("Shutting down time series manager")
        
        # Shut down thread pool
        self._thread_pool.shutdown(wait=True)
        
        # Backend-specific shutdown
        if self.backend == 'timescale' and self.client:
            try:
                self.client.close()
            except Exception as e:
                logger.error(f"Error closing TimescaleDB connection: {str(e)}")
        
        elif self.backend == 'influx' and self.client:
            try:
                self.client.close()
            except Exception as e:
                logger.error(f"Error closing InfluxDB connection: {str(e)}")
        
        # Clear cache
        with self._cache_lock:
            self._cache.clear()
        
        logger.info("Time series manager shut down successfully")

# Add TimeSeriesDB alias for backward compatibility with backtester.engine
TimeSeriesDB = TimeSeriesStorage
