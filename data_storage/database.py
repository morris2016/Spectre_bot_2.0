"""
QuantumSpectre Elite Trading System
Database Manager

This module provides the database management capabilities for the QuantumSpectre Elite Trading System,
handling connections, queries, and optimizations for various database backends.
"""

import os
import time
import logging
import threading
import queue
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from contextlib import contextmanager

import sqlalchemy
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, relationship
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError

from common.logger import get_logger
from common.utils import singleton, retry_with_backoff_decorator
from common.metrics import MetricsCollector
from common.exceptions import (
    DatabaseConnectionError, 
    DatabaseQueryError,
    DatabaseIntegrityError,
    DatabaseTimeoutError
)

# Initialize logger
logger = get_logger(__name__)

# Base class for all ORM models
Base = declarative_base()

# Metrics collector
metrics = MetricsCollector.get_instance()

# Global session variable that will be initialized by DatabaseManager
db_session = None

@singleton
class DatabaseManager:
    """
    Manages database connections and provides query optimization,
    connection pooling, and transaction management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the database manager with configuration.
        
        Args:
            config: Database configuration dictionary containing:
                - url: Database connection URL
                - pool_size: Connection pool size
                - max_overflow: Maximum overflow connections
                - pool_timeout: Pool timeout in seconds
                - pool_recycle: Connection recycle time in seconds
                - echo: Whether to echo SQL statements
                - query_timeout: Default query timeout in seconds
        
        Raises:
            DatabaseConnectionError: If database connection fails
        """
        self.config = config
        self.url = config.get('url', 'sqlite:///quantum_spectre.db')
        self.query_timeout = config.get('query_timeout', 30)
        self.metadata = MetaData()
        self.session_factory = None
        self.engine = None
        self._init_lock = threading.Lock()
        self._initialized = False
        self._query_stats = {}
        
        # Initialize database connection
        self._initialize_connection()
        
        # Register metrics
        metrics.register_gauge('db.pool.connections', 'Number of active database connections')
        metrics.register_counter('db.queries.total', 'Total number of database queries executed')
        metrics.register_counter('db.queries.errors', 'Number of database query errors')
        metrics.register_histogram('db.query.duration', 'Database query duration in milliseconds')
        
        logger.info(f"Database manager initialized with {self.url}")
    
    @retry_with_backoff_decorator(
        max_retries=5, 
        base_delay=1, 
        max_delay=30, 
        exceptions=(OperationalError,)
    )
    def _initialize_connection(self) -> None:
        """
        Initialize the database connection with retry logic.
        
        Raises:
            DatabaseConnectionError: If connection initialization fails after retries
        """
        with self._init_lock:
            if self._initialized:
                return
                
            try:
                self.engine = create_engine(
                    self.url,
                    pool_size=self.config.get('pool_size', 10),
                    max_overflow=self.config.get('max_overflow', 20),
                    pool_timeout=self.config.get('pool_timeout', 30),
                    pool_recycle=self.config.get('pool_recycle', 1800),
                    echo=self.config.get('echo', False),
                    connect_args={'timeout': self.query_timeout}
                )
                
                # Create session factory
                session_factory = sessionmaker(bind=self.engine)
                self.session_factory = scoped_session(session_factory)
                
                # Create a global db_session for modules that need direct access
                global db_session
                db_session = self.session_factory
                
                # If using SQLite, enable foreign keys
                if self.url.startswith('sqlite'):
                    @sqlalchemy.event.listens_for(self.engine, "connect")
                    def set_sqlite_pragma(dbapi_connection, connection_record):
                        cursor = dbapi_connection.cursor()
                        cursor.execute("PRAGMA foreign_keys=ON")
                        cursor.execute("PRAGMA journal_mode=WAL")
                        cursor.execute("PRAGMA synchronous=NORMAL")
                        cursor.execute("PRAGMA temp_store=MEMORY")
                        cursor.execute("PRAGMA mmap_size=30000000000")
                        cursor.close()
                
                self._initialized = True
                logger.info("Database connection initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize database connection: {str(e)}")
                raise DatabaseConnectionError(f"Failed to connect to database: {str(e)}")
    
    @contextmanager
    def session(self):
        """
        Provide a transactional session scope.
        
        Yields:
            SQLAlchemy session object
            
        Raises:
            DatabaseConnectionError: If session creation fails
            DatabaseIntegrityError: If integrity error occurs
            DatabaseQueryError: If query error occurs
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except IntegrityError as e:
            session.rollback()
            logger.error(f"Database integrity error: {str(e)}")
            metrics.increment('db.queries.errors')
            raise DatabaseIntegrityError(f"Database integrity error: {str(e)}")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database query error: {str(e)}")
            metrics.increment('db.queries.errors')
            raise DatabaseQueryError(f"Database query error: {str(e)}")
        finally:
            session.close()
    
    def execute_query(self, query: str, params: Dict[str, Any] = None, timeout: int = None) -> List[Dict[str, Any]]:
        """
        Execute a raw SQL query with timing and metrics.
        
        Args:
            query: SQL query string
            params: Query parameters
            timeout: Query timeout in seconds (overrides default)
            
        Returns:
            List of dictionaries containing query results
            
        Raises:
            DatabaseQueryError: If query execution fails
            DatabaseTimeoutError: If query times out
        """
        if not self._initialized:
            self._initialize_connection()
        
        start_time = time.time()
        metrics.increment('db.queries.total')
        
        actual_timeout = timeout or self.query_timeout
        query_key = query[:100]  # Use first 100 chars as key for stats
        
        try:
            with self.engine.connect() as connection:
                # Set statement timeout if supported by database
                if hasattr(connection, 'execution_options'):
                    connection = connection.execution_options(timeout=actual_timeout)
                
                result = connection.execute(text(query), params or {})
                rows = [dict(row._mapping) for row in result]
                
                # Record query stats
                duration = (time.time() - start_time) * 1000  # ms
                metrics.observe('db.query.duration', duration)
                
                if query_key not in self._query_stats:
                    self._query_stats[query_key] = {
                        'count': 0,
                        'total_time': 0,
                        'min_time': float('inf'),
                        'max_time': 0
                    }
                
                stats = self._query_stats[query_key]
                stats['count'] += 1
                stats['total_time'] += duration
                stats['min_time'] = min(stats['min_time'], duration)
                stats['max_time'] = max(stats['max_time'], duration)
                
                return rows
        except sqlalchemy.exc.TimeoutError as e:
            metrics.increment('db.queries.errors')
            logger.error(f"Query timeout after {actual_timeout}s: {query}")
            raise DatabaseTimeoutError(f"Query timed out after {actual_timeout} seconds")
        except Exception as e:
            metrics.increment('db.queries.errors')
            logger.error(f"Query error: {str(e)}, Query: {query}")
            raise DatabaseQueryError(f"Query execution failed: {str(e)}")
    
    def create_tables(self) -> None:
        """
        Create all tables defined in the Base metadata.
        
        Raises:
            DatabaseConnectionError: If table creation fails
        """
        if not self._initialized:
            self._initialize_connection()
            
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {str(e)}")
            raise DatabaseConnectionError(f"Failed to create tables: {str(e)}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get database connection pool statistics.
        
        Returns:
            Dictionary containing connection pool statistics
        """
        if not self._initialized or not self.engine:
            return {'status': 'not_initialized'}
            
        try:
            pool = self.engine.pool
            stats = {
                'size': pool.size(),
                'checkedin': pool.checkedin(),
                'checkedout': pool.checkedout(),
                'overflow': pool.overflow(),
                'query_stats': {k: v for k, v in sorted(
                    self._query_stats.items(), 
                    key=lambda item: item[1]['total_time'], 
                    reverse=True
                )[:10]}  # Top 10 queries by total time
            }
            
            # Update metrics
            metrics.set('db.pool.connections', stats['checkedout'])
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get connection stats: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def optimize_for_query(self, table_name: str, columns: List[str]) -> None:
        """
        Create an index for the specified columns to optimize query performance.
        
        Args:
            table_name: Name of the table
            columns: List of columns to index
            
        Raises:
            DatabaseQueryError: If index creation fails
        """
        if not self._initialized:
            self._initialize_connection()
            
        index_name = f"idx_{'_'.join(columns)}_{int(time.time())}"
        columns_str = ', '.join(columns)
        
        query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({columns_str})"
        
        try:
            self.execute_query(query)
            logger.info(f"Created index {index_name} on {table_name}({columns_str})")
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            raise DatabaseQueryError(f"Failed to create index: {str(e)}")
    
    def shutdown(self) -> None:
        """
        Properly shut down the database connection pool.
        """
        if self.engine:
            logger.info("Shutting down database connection pool")
            self.engine.dispose()
            self._initialized = False
            logger.info("Database connection pool shut down successfully")
