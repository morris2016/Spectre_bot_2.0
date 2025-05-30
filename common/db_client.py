#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Database Client Module

This module provides a client for database operations, supporting connection pooling,
transactions, and migrations.
"""

from __future__ import annotations

import os
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    import asyncpg  # type: ignore
    DB_LIB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    asyncpg = None  # type: ignore
    DB_LIB_AVAILABLE = False

from common.logger import get_logger
from common.exceptions import (
    DatabaseError, DatabaseConnectionError, DatabaseQueryError
)

_db_client: Optional[DatabaseClient] = None


async def get_db_client(**kwargs) -> DatabaseClient:
    """Get a shared :class:`DatabaseClient` instance."""
    global _db_client
    if _db_client is None:
        _db_client = DatabaseClient(**kwargs)
        await _db_client.initialize()
    return _db_client

class DatabaseClient:
    """Client for database operations."""
    
    def __init__(self, db_type="postgresql", host="localhost", port=5432, 
                 username="postgres", password="", database="quantumspectre",
                 pool_size=10, ssl=False, timeout=30):
        """
        Initialize database client.
        
        Args:
            db_type: Database type (currently only "postgresql" is supported)
            host: Database host
            port: Database port
            username: Database username
            password: Database password
            database: Database name
            pool_size: Connection pool size
            ssl: Whether to use SSL
            timeout: Query timeout in seconds
        """
        self.db_type = db_type
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.pool_size = pool_size
        self.ssl = ssl
        self.timeout = timeout
        self.pool = None
        self.logger = get_logger("DatabaseClient")
        
    async def initialize(self):
        """
        Initialize the database connection pool.
        
        Raises:
            DatabaseConnectionError: If connection fails
        """
        try:
            if self.db_type == "postgresql":
                self.pool = await asyncpg.create_pool(
                    user=self.username,
                    password=self.password,
                    database=self.database,
                    host=self.host,
                    port=self.port,
                    max_size=self.pool_size,
                    ssl=self.ssl,
                    command_timeout=self.timeout
                )
                self.logger.info(f"Connected to PostgreSQL database: {self.database}")
            else:
                raise DatabaseError(f"Unsupported database type: {self.db_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database connection: {str(e)}")
            raise DatabaseConnectionError(f"Failed to connect to database: {str(e)}")
            
    async def close(self):
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()
            self.logger.info("Database connection pool closed")
            
    async def execute(self, query, *args, timeout=None):
        """
        Execute a database query.
        
        Args:
            query: SQL query
            *args: Query arguments
            timeout: Query timeout in seconds (overrides default)
            
        Returns:
            Query result
            
        Raises:
            DatabaseQueryError: If query execution fails
        """
        if not self.pool:
            raise DatabaseError("Database not initialized")
            
        try:
            return await self.pool.execute(
                query, *args, timeout=timeout or self.timeout
            )
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            self.logger.debug(f"Query: {query}, Args: {args}")
            raise DatabaseQueryError(f"Query execution failed: {str(e)}")
            
    async def fetch(self, query, *args, timeout=None):
        """
        Fetch rows from database.
        
        Args:
            query: SQL query
            *args: Query arguments
            timeout: Query timeout in seconds (overrides default)
            
        Returns:
            List of rows
            
        Raises:
            DatabaseQueryError: If query execution fails
        """
        if not self.pool:
            raise DatabaseError("Database not initialized")
            
        try:
            return await self.pool.fetch(
                query, *args, timeout=timeout or self.timeout
            )
        except Exception as e:
            self.logger.error(f"Query fetch failed: {str(e)}")
            self.logger.debug(f"Query: {query}, Args: {args}")
            raise DatabaseQueryError(f"Query fetch failed: {str(e)}")
            
    async def fetchrow(self, query, *args, timeout=None):
        """
        Fetch a single row from database.
        
        Args:
            query: SQL query
            *args: Query arguments
            timeout: Query timeout in seconds (overrides default)
            
        Returns:
            Row or None
            
        Raises:
            DatabaseQueryError: If query execution fails
        """
        if not self.pool:
            raise DatabaseError("Database not initialized")
            
        try:
            return await self.pool.fetchrow(
                query, *args, timeout=timeout or self.timeout
            )
        except Exception as e:
            self.logger.error(f"Query fetchrow failed: {str(e)}")
            self.logger.debug(f"Query: {query}, Args: {args}")
            raise DatabaseQueryError(f"Query fetchrow failed: {str(e)}")
            
    async def fetchval(self, query, *args, column=0, timeout=None):
        """
        Fetch a single value from database.
        
        Args:
            query: SQL query
            *args: Query arguments
            column: Column index
            timeout: Query timeout in seconds (overrides default)
            
        Returns:
            Value or None
            
        Raises:
            DatabaseQueryError: If query execution fails
        """
        if not self.pool:
            raise DatabaseError("Database not initialized")
            
        try:
            return await self.pool.fetchval(
                query, *args, column=column, timeout=timeout or self.timeout
            )
        except Exception as e:
            self.logger.error(f"Query fetchval failed: {str(e)}")
            self.logger.debug(f"Query: {query}, Args: {args}")
            raise DatabaseQueryError(f"Query fetchval failed: {str(e)}")

    def query(self, query: str, params: Optional[Union[List[Any], Tuple[Any, ...]]] = None,
              timeout: Optional[int] = None) -> List[asyncpg.Record]:
        """Synchronously fetch rows from the database.

        This helper allows code that is not running in an asynchronous
        context to execute read-only queries using the existing
        :meth:`fetch` coroutine.

        Args:
            query: SQL query to execute.
            params: Optional query parameters.
            timeout: Optional query timeout in seconds.

        Returns:
            List of records returned by the query.

        Raises:
            DatabaseQueryError: If query execution fails
        """
        if params is None:
            params = []

        coro = self.fetch(query, *params, timeout=timeout)
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
        except RuntimeError:
            # No running event loop
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)
            finally:
                asyncio.set_event_loop(None)
                loop.close()
            
    async def transaction(self):
        """
        Start a transaction.

        Returns:
            Transaction object
            
        Raises:
            DatabaseError: If database not initialized
        """
        if not self.pool:
            raise DatabaseError("Database not initialized")
            
        return self.pool.transaction()

    async def commit(self):
        """Commit the current transaction if one exists."""
        if not self.pool:
            raise DatabaseError("Database not initialized")

        async with self.pool.acquire() as connection:
            try:
                await connection.execute("COMMIT")
            except Exception as e:
                self.logger.error(f"Commit failed: {str(e)}")
                raise DatabaseError(f"Commit failed: {str(e)}")
        
    async def run_migrations(self, migrations_dir="./migrations"):
        """
        Run database migrations.
        
        Args:
            migrations_dir: Directory containing migration files
            
        Raises:
            DatabaseError: If migrations fail
        """
        if not self.pool:
            raise DatabaseError("Database not initialized")
            
        # Create migrations table if it doesn't exist
        await self.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                applied_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
        
        # Get applied migrations
        applied_migrations = await self.fetch("SELECT name FROM migrations")
        applied_migration_names = {row['name'] for row in applied_migrations}
        
        # Find migration files
        try:
            migration_files = sorted([
                f for f in os.listdir(migrations_dir)
                if f.endswith('.sql') and f not in applied_migration_names
            ])
        except FileNotFoundError:
            self.logger.warning(f"Migrations directory not found: {migrations_dir}")
            return
            
        for migration_file in migration_files:
            try:
                with open(os.path.join(migrations_dir, migration_file), 'r') as f:
                    migration_sql = f.read()
                    
                # Execute migration within a transaction
                async with self.transaction():
                    self.logger.info(f"Applying migration: {migration_file}")
                    await self.execute(migration_sql)
                    await self.execute(
                        "INSERT INTO migrations (name) VALUES ($1)",
                        migration_file
                    )
                    
            except Exception as e:
                self.logger.error(f"Migration failed: {migration_file} - {str(e)}")
                raise DatabaseError(f"Migration failed: {migration_file} - {str(e)}")
                
        self.logger.info(f"Database migrations complete. Applied {len(migration_files)} migrations.")
        
    async def create_tables(self):
        """
        Create database tables.
        
        Raises:
            DatabaseError: If table creation fails
        """
        if not self.pool:
            raise DatabaseError("Database not initialized")
            
        try:
            # Create market data table
            await self.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp BIGINT NOT NULL,
                    open DOUBLE PRECISION NOT NULL,
                    high DOUBLE PRECISION NOT NULL,
                    low DOUBLE PRECISION NOT NULL,
                    close DOUBLE PRECISION NOT NULL,
                    volume DOUBLE PRECISION NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    UNIQUE (exchange, symbol, timeframe, timestamp)
                )
            """)
            
            # Create trade table
            await self.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    strategy TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity DOUBLE PRECISION NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    exit_price DOUBLE PRECISION,
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    profit_loss DOUBLE PRECISION,
                    profit_loss_percent DOUBLE PRECISION,
                    status TEXT NOT NULL,
                    metadata JSONB
                )
            """)
            
            # Create order table
            await self.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id SERIAL PRIMARY KEY,
                    exchange_order_id TEXT,
                    trade_id INTEGER REFERENCES trades(id),
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity DOUBLE PRECISION NOT NULL,
                    price DOUBLE PRECISION,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    metadata JSONB
                )
            """)
            
            # Create signals table
            await self.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    strategy TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    timestamp BIGINT NOT NULL,
                    confidence DOUBLE PRECISION,
                    processed BOOLEAN NOT NULL DEFAULT FALSE,
                    metadata JSONB,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """)
            
            # Create strategy performance table
            await self.execute("""
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id SERIAL PRIMARY KEY,
                    strategy TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    period_start TIMESTAMP NOT NULL,
                    period_end TIMESTAMP NOT NULL,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    losing_trades INTEGER NOT NULL,
                    profit_loss DOUBLE PRECISION NOT NULL,
                    profit_loss_percent DOUBLE PRECISION NOT NULL,
                    max_drawdown DOUBLE PRECISION NOT NULL,
                    sharpe_ratio DOUBLE PRECISION,
                    metrics JSONB,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """)
            
            # Create system events table
            await self.execute("""
                CREATE TABLE IF NOT EXISTS system_events (
                    id SERIAL PRIMARY KEY,
                    service TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                    metadata JSONB
                )
            """)
            
            self.logger.info("Database tables created")
            
        except Exception as e:
            self.logger.error(f"Failed to create tables: {str(e)}")
            raise DatabaseError(f"Failed to create tables: {str(e)}")
            
    async def insert_market_data(self, exchange, symbol, timeframe, timestamp, open_price, high, low, close, volume):
        """
        Insert market data.
        
        Args:
            exchange: Exchange name
            symbol: Symbol name
            timeframe: Timeframe
            timestamp: Timestamp in milliseconds
            open_price: Opening price
            high: High price
            low: Low price
            close: Close price
            volume: Volume
            
        Returns:
            Inserted record ID
            
        Raises:
            DatabaseQueryError: If insertion fails
        """
        try:
            record_id = await self.fetchval("""
                INSERT INTO market_data (exchange, symbol, timeframe, timestamp, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (exchange, symbol, timeframe, timestamp) DO UPDATE
                SET open = $5, high = $6, low = $7, close = $8, volume = $9
                RETURNING id
            """, exchange, symbol, timeframe, timestamp, open_price, high, low, close, volume)
            
            return record_id
            
        except Exception as e:
            self.logger.error(f"Failed to insert market data: {str(e)}")
            raise DatabaseQueryError(f"Failed to insert market data: {str(e)}")
            
    async def get_market_data(self, exchange, symbol, timeframe, start_time, end_time, limit=None):
        """
        Get market data.
        
        Args:
            exchange: Exchange name
            symbol: Symbol name
            timeframe: Timeframe
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds
            limit: Maximum number of records to return
            
        Returns:
            List of market data records
            
        Raises:
            DatabaseQueryError: If query fails
        """
        try:
            query = """
                SELECT * FROM market_data
                WHERE exchange = $1 AND symbol = $2 AND timeframe = $3
                AND timestamp >= $4 AND timestamp <= $5
                ORDER BY timestamp ASC
            """
            
            args = [exchange, symbol, timeframe, start_time, end_time]
            
            if limit is not None:
                query += " LIMIT $6"
                args.append(limit)
                
            return await self.fetch(query, *args)
            
        except Exception as e:
            self.logger.error(f"Failed to get market data: {str(e)}")
            raise DatabaseQueryError(f"Failed to get market data: {str(e)}")
            
    async def insert_trade(self, strategy, exchange, symbol, side, quantity, entry_price, entry_time, status, metadata=None):
        """
        Insert a trade.
        
        Args:
            strategy: Strategy name
            exchange: Exchange name
            symbol: Symbol name
            side: Trade side (buy/sell)
            quantity: Trade quantity
            entry_price: Entry price
            entry_time: Entry time
            status: Trade status
            metadata: Optional metadata
            
        Returns:
            Inserted trade ID
            
        Raises:
            DatabaseQueryError: If insertion fails
        """
        try:
            trade_id = await self.fetchval("""
                INSERT INTO trades (strategy, exchange, symbol, side, quantity, entry_price, entry_time, status, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
            """, strategy, exchange, symbol, side, quantity, entry_price, entry_time, status, 
                json.dumps(metadata) if metadata else None)
            
            return trade_id
            
        except Exception as e:
            self.logger.error(f"Failed to insert trade: {str(e)}")
            raise DatabaseQueryError(f"Failed to insert trade: {str(e)}")
            
    async def update_trade(self, trade_id, exit_price=None, exit_time=None, profit_loss=None, 
                          profit_loss_percent=None, status=None, metadata=None):
        """
        Update a trade.
        
        Args:
            trade_id: Trade ID
            exit_price: Exit price
            exit_time: Exit time
            profit_loss: Profit/loss
            profit_loss_percent: Profit/loss percentage
            status: Trade status
            metadata: Optional metadata
            
        Raises:
            DatabaseQueryError: If update fails
        """
        try:
            # Build update query and arguments
            update_parts = []
            args = [trade_id]
            arg_index = 2
            
            if exit_price is not None:
                update_parts.append(f"exit_price = ${arg_index}")
                args.append(exit_price)
                arg_index += 1
                
            if exit_time is not None:
                update_parts.append(f"exit_time = ${arg_index}")
                args.append(exit_time)
                arg_index += 1
                
            if profit_loss is not None:
                update_parts.append(f"profit_loss = ${arg_index}")
                args.append(profit_loss)
                arg_index += 1
                
            if profit_loss_percent is not None:
                update_parts.append(f"profit_loss_percent = ${arg_index}")
                args.append(profit_loss_percent)
                arg_index += 1
                
            if status is not None:
                update_parts.append(f"status = ${arg_index}")
                args.append(status)
                arg_index += 1
                
            if metadata is not None:
                update_parts.append(f"metadata = ${arg_index}")
                args.append(json.dumps(metadata))
                arg_index += 1
                
            if not update_parts:
                return  # Nothing to update
                
            query = f"""
                UPDATE trades
                SET {', '.join(update_parts)}
                WHERE id = $1
            """
            
            await self.execute(query, *args)
            
        except Exception as e:
            self.logger.error(f"Failed to update trade: {str(e)}")
            raise DatabaseQueryError(f"Failed to update trade: {str(e)}")
            
    async def insert_order(self, exchange_order_id, trade_id, exchange, symbol, order_type, side, 
                          quantity, price, status, created_at, metadata=None):
        """
        Insert an order.
        
        Args:
            exchange_order_id: Exchange order ID
            trade_id: Trade ID
            exchange: Exchange name
            symbol: Symbol name
            order_type: Order type
            side: Order side
            quantity: Order quantity
            price: Order price
            status: Order status
            created_at: Creation timestamp
            metadata: Optional metadata
            
        Returns:
            Inserted order ID
            
        Raises:
            DatabaseQueryError: If insertion fails
        """
        try:
            order_id = await self.fetchval("""
                INSERT INTO orders (
                    exchange_order_id, trade_id, exchange, symbol, order_type, side, 
                    quantity, price, status, created_at, updated_at, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $10, $11)
                RETURNING id
            """, exchange_order_id, trade_id, exchange, symbol, order_type, side, 
                quantity, price, status, created_at, 
                json.dumps(metadata) if metadata else None)
            
            return order_id
            
        except Exception as e:
            self.logger.error(f"Failed to insert order: {str(e)}")
            raise DatabaseQueryError(f"Failed to insert order: {str(e)}")
            
    async def update_order(self, order_id, status=None, updated_at=None, metadata=None):
        """
        Update an order.
        
        Args:
            order_id: Order ID
            status: Order status
            updated_at: Update timestamp
            metadata: Optional metadata
            
        Raises:
            DatabaseQueryError: If update fails
        """
        try:
            # Build update query and arguments
            update_parts = []
            args = [order_id]
            arg_index = 2
            
            if status is not None:
                update_parts.append(f"status = ${arg_index}")
                args.append(status)
                arg_index += 1
                
            if updated_at is not None:
                update_parts.append(f"updated_at = ${arg_index}")
                args.append(updated_at)
                arg_index += 1
                
            if metadata is not None:
                update_parts.append(f"metadata = ${arg_index}")
                args.append(json.dumps(metadata))
                arg_index += 1
                
            if not update_parts:
                return  # Nothing to update
                
            query = f"""
                UPDATE orders
                SET {', '.join(update_parts)}
                WHERE id = $1
            """
            
            await self.execute(query, *args)
            
        except Exception as e:
            self.logger.error(f"Failed to update order: {str(e)}")
            raise DatabaseQueryError(f"Failed to update order: {str(e)}")
            
    async def insert_signal(self, strategy, exchange, symbol, timeframe, signal_type, direction, 
                           timestamp, confidence=None, metadata=None):
        """
        Insert a signal.
        
        Args:
            strategy: Strategy name
            exchange: Exchange name
            symbol: Symbol name
            timeframe: Timeframe
            signal_type: Signal type
            direction: Signal direction
            timestamp: Timestamp in milliseconds
            confidence: Signal confidence
            metadata: Optional metadata
            
        Returns:
            Inserted signal ID
            
        Raises:
            DatabaseQueryError: If insertion fails
        """
        try:
            signal_id = await self.fetchval("""
                INSERT INTO signals (
                    strategy, exchange, symbol, timeframe, signal_type, direction, 
                    timestamp, confidence, metadata
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
            """, strategy, exchange, symbol, timeframe, signal_type, direction, 
                timestamp, confidence, json.dumps(metadata) if metadata else None)
            
            return signal_id
            
        except Exception as e:
            self.logger.error(f"Failed to insert signal: {str(e)}")
            raise DatabaseQueryError(f"Failed to insert signal: {str(e)}")
            
    async def mark_signal_processed(self, signal_id):
        """
        Mark a signal as processed.
        
        Args:
            signal_id: Signal ID
            
        Raises:
            DatabaseQueryError: If update fails
        """
        try:
            await self.execute("""
                UPDATE signals
                SET processed = TRUE
                WHERE id = $1
            """, signal_id)
            
        except Exception as e:
            self.logger.error(f"Failed to mark signal as processed: {str(e)}")
            raise DatabaseQueryError(f"Failed to mark signal as processed: {str(e)}")
            
    async def insert_system_event(self, service, event_type, message, severity, metadata=None):
        """
        Insert a system event.
        
        Args:
            service: Service name
            event_type: Event type
            message: Event message
            severity: Event severity
            metadata: Optional metadata
            
        Returns:
            Inserted event ID
            
        Raises:
            DatabaseQueryError: If insertion fails
        """
        try:
            event_id = await self.fetchval("""
                INSERT INTO system_events (service, event_type, message, severity, metadata)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """, service, event_type, message, severity, 
                json.dumps(metadata) if metadata else None)
            
            return event_id
            
        except Exception as e:
            self.logger.error(f"Failed to insert system event: {str(e)}")
            raise DatabaseQueryError(f"Failed to insert system event: {str(e)}")
            
    async def insert_strategy_performance(self, strategy, timeframe, period_start, period_end,
                                         total_trades, winning_trades, losing_trades,
                                         profit_loss, profit_loss_percent, max_drawdown,
                                         sharpe_ratio=None, metrics=None):
        """
        Insert strategy performance.
        
        Args:
            strategy: Strategy name
            timeframe: Timeframe
            period_start: Period start timestamp
            period_end: Period end timestamp
            total_trades: Total number of trades
            winning_trades: Number of winning trades
            losing_trades: Number of losing trades
            profit_loss: Profit/loss
            profit_loss_percent: Profit/loss percentage
            max_drawdown: Maximum drawdown
            sharpe_ratio: Sharpe ratio
            metrics: Optional metrics
            
        Returns:
            Inserted record ID
            
        Raises:
            DatabaseQueryError: If insertion fails
        """
        try:
            record_id = await self.fetchval("""
                INSERT INTO strategy_performance (
                    strategy, timeframe, period_start, period_end,
                    total_trades, winning_trades, losing_trades,
                    profit_loss, profit_loss_percent, max_drawdown,
                    sharpe_ratio, metrics
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                RETURNING id
            """, strategy, timeframe, period_start, period_end,
                total_trades, winning_trades, losing_trades,
                profit_loss, profit_loss_percent, max_drawdown,
                sharpe_ratio, json.dumps(metrics) if metrics else None)
            
            return record_id
            
        except Exception as e:
            self.logger.error(f"Failed to insert strategy performance: {str(e)}")
            raise DatabaseQueryError(f"Failed to insert strategy performance: {str(e)}")
            
    async def get_open_trades(self):
        """
        Get open trades.
        
        Returns:
            List of open trades
            
        Raises:
            DatabaseQueryError: If query fails
        """
        try:
            return await self.fetch("""
                SELECT * FROM trades
                WHERE status = 'open'
                ORDER BY entry_time DESC
            """)
            
        except Exception as e:
            self.logger.error(f"Failed to get open trades: {str(e)}")
            raise DatabaseQueryError(f"Failed to get open trades: {str(e)}")
            
    async def get_trade_by_id(self, trade_id):
        """
        Get a trade by ID.
        
        Args:
            trade_id: Trade ID
            
        Returns:
            Trade record or None
            
        Raises:
            DatabaseQueryError: If query fails
        """
        try:
            return await self.fetchrow("""
                SELECT * FROM trades
                WHERE id = $1
            """, trade_id)
            
        except Exception as e:
            self.logger.error(f"Failed to get trade by ID: {str(e)}")
            raise DatabaseQueryError(f"Failed to get trade by ID: {str(e)}")
            
    async def get_orders_by_trade_id(self, trade_id):
        """
        Get orders for a trade.
        
        Args:
            trade_id: Trade ID
            
        Returns:
            List of orders
            
        Raises:
            DatabaseQueryError: If query fails
        """
        try:
            return await self.fetch("""
                SELECT * FROM orders
                WHERE trade_id = $1
                ORDER BY created_at ASC
            """, trade_id)
            
        except Exception as e:
            self.logger.error(f"Failed to get orders by trade ID: {str(e)}")
            raise DatabaseQueryError(f"Failed to get orders by trade ID: {str(e)}")
            
    async def get_unprocessed_signals(self):
        """
        Get unprocessed signals.
        
        Returns:
            List of unprocessed signals
            
        Raises:
            DatabaseQueryError: If query fails
        """
        try:
            return await self.fetch("""
                SELECT * FROM signals
                WHERE processed = FALSE
                ORDER BY timestamp ASC
            """)
            
        except Exception as e:
            self.logger.error(f"Failed to get unprocessed signals: {str(e)}")
            raise DatabaseQueryError(f"Failed to get unprocessed signals: {str(e)}")
            
    async def get_system_events(self, service=None, event_type=None, severity=None, limit=100):
        """
        Get system events.
        
        Args:
            service: Optional service filter
            event_type: Optional event type filter
            severity: Optional severity filter
            limit: Maximum number of events to return
            
        Returns:
            List of system events
            
        Raises:
            DatabaseQueryError: If query fails
        """
        try:
            query = "SELECT * FROM system_events"
            conditions = []
            args = []
            
            if service:
                conditions.append(f"service = ${len(args) + 1}")
                args.append(service)
                
            if event_type:
                conditions.append(f"event_type = ${len(args) + 1}")
                args.append(event_type)
                
            if severity:
                conditions.append(f"severity = ${len(args) + 1}")
                args.append(severity)
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            query += f" ORDER BY timestamp DESC LIMIT ${len(args) + 1}"
            args.append(limit)
            
            return await self.fetch(query, *args)
            
        except Exception as e:
            self.logger.error(f"Failed to get system events: {str(e)}")
            raise DatabaseQueryError(f"Failed to get system events: {str(e)}")
            
    async def get_strategy_performance(self, strategy=None, timeframe=None, limit=100):
        """
        Get strategy performance.
        
        Args:
            strategy: Optional strategy filter
            timeframe: Optional timeframe filter
            limit: Maximum number of records to return
            
        Returns:
            List of strategy performance records
            
        Raises:
            DatabaseQueryError: If query fails
        """
        try:
            query = "SELECT * FROM strategy_performance"
            conditions = []
            args = []
            
            if strategy:
                conditions.append(f"strategy = ${len(args) + 1}")
                args.append(strategy)
                
            if timeframe:
                conditions.append(f"timeframe = ${len(args) + 1}")
                args.append(timeframe)
                
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            query += f" ORDER BY period_end DESC LIMIT ${len(args) + 1}"
            args.append(limit)
            
            return await self.fetch(query, *args)
            
        except Exception as e:
            self.logger.error(f"Failed to get strategy performance: {str(e)}")
            raise DatabaseQueryError(f"Failed to get strategy performance: {str(e)}")

# Create an alias for DatabaseClient as DBClient for backward compatibility
DBClient = DatabaseClient
