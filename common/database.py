"""
Database Management for QuantumSpectre Elite Trading System.

This module handles database connections, session management, and ORM integration
using SQLAlchemy. It provides connection pooling, transaction management, and
migration support.
"""

import os
import time
import asyncio
from typing import Dict, Any, Optional, List, Union
import logging
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, inspect, event, text, Column, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.engine import Engine
from sqlalchemy.sql import func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncEngine

from alembic.config import Config
from alembic import command

from common.logger import get_logger

logger = get_logger('database')

# Base class for all ORM models
Base = declarative_base()


class BaseMixin:
    """Reusable columns for all tables."""
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# Global database engine and session factory
engine = None
async_engine = None
SessionLocal = None
AsyncSessionLocal = None
db_manager = None

class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(
        self, 
        db_uri: str,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_recycle: int = 3600,
        echo: bool = False
    ):
        """
        Initialize database manager.
        
        Args:
            db_uri: Database connection URI
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
            pool_recycle: Connection recycle time in seconds
            echo: Whether to echo SQL statements
        """
        self.db_uri = db_uri
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_recycle = pool_recycle
        self.echo = echo
        
        # Create engines and session factories
        self._setup_engines()
        
        # Connection pool monitoring
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'pool_checkouts': 0,
            'pool_checkins': 0,
            'pool_recycles': 0,
        }
    
    def _setup_engines(self) -> None:
        """Set up database engines and session factories."""
        global engine, async_engine, SessionLocal, AsyncSessionLocal
        
        # Create synchronous engine
        engine_args = {
            'pool_size': self.pool_size,
            'max_overflow': self.max_overflow,
            'pool_recycle': self.pool_recycle,
            'pool_pre_ping': True,
            'echo': self.echo,
        }
        
        # Handle SQLite special case
        is_sqlite = self.db_uri.startswith('sqlite')
        if is_sqlite:
            # SQLite doesn't support pool_size and max_overflow
            engine_args.pop('pool_size', None)
            engine_args.pop('max_overflow', None)
            
            # Ensure directory exists for SQLite file
            if self.db_uri.startswith('sqlite:///') and not self.db_uri.startswith('sqlite:////'):
                db_path = self.db_uri.replace('sqlite:///', '')
                os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        # Create engines
        engine = create_engine(self.db_uri, **engine_args)
        
        # Create async engine if not using SQLite
        if not is_sqlite:
            async_uri = self.db_uri.replace('postgresql://', 'postgresql+asyncpg://')
            async_engine = create_async_engine(async_uri, **engine_args)
        else:
            # For SQLite, we'll use the synchronous engine for both
            logger.warning("SQLite does not support async operations, using sync engine")
            async_engine = engine
        
        # Create session factories
        SessionLocal = scoped_session(sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        ))
        
        if not is_sqlite:
            AsyncSessionLocal = async_sessionmaker(
                async_engine,
                expire_on_commit=False,
                class_=AsyncSession
            )
        else:
            # For SQLite, we'll use the synchronous session for both
            AsyncSessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=engine
            )
        
        # Set up connection pool event listeners
        self._setup_event_listeners()
    
    def _setup_event_listeners(self) -> None:
        """Set up event listeners for connection pool monitoring."""
        # Only for non-async engine
        @event.listens_for(engine, 'checkout')
        def receive_checkout(dbapi_conn, conn_record, conn_proxy):
            self.stats['pool_checkouts'] += 1
            self.stats['active_connections'] += 1
        
        @event.listens_for(engine, 'checkin')
        def receive_checkin(dbapi_conn, conn_record):
            self.stats['pool_checkins'] += 1
            self.stats['active_connections'] -= 1
        
        @event.listens_for(engine, 'connect')
        def receive_connect(dbapi_conn, conn_record):
            self.stats['total_connections'] += 1
        
        @event.listens_for(engine, 'reset')
        def receive_reset(dbapi_conn, conn_record):
            self.stats['pool_recycles'] += 1
    
    @property
    def is_sqlite(self) -> bool:
        """Check if database is SQLite."""
        return self.db_uri.startswith('sqlite')
    
    def get_session(self):
        """
        Get a new database session.
        
        Returns:
            New database session
        """
        return SessionLocal()
    
    @asynccontextmanager
    async def get_async_session(self):
        """
        Get an async database session.
        
        Yields:
            Async database session
        """
        session = AsyncSessionLocal()
        try:
            yield session
        finally:
            await session.close()
    
    def create_all(self) -> None:
        """Create all tables defined in ORM models."""
        Base.metadata.create_all(bind=engine)
    
    async def create_all_async(self) -> None:
        """Create all tables defined in ORM models (async version)."""
        if not self.is_sqlite:
            async with async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        else:
            # For SQLite, use synchronous version
            self.create_all()
    
    def drop_all(self) -> None:
        """Drop all tables defined in ORM models."""
        Base.metadata.drop_all(bind=engine)
    
    async def drop_all_async(self) -> None:
        """Drop all tables defined in ORM models (async version)."""
        if not self.is_sqlite:
            async with async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
        else:
            # For SQLite, use synchronous version
            self.drop_all()
    
    def run_migrations(self, alembic_cfg_path: str = 'alembic.ini') -> None:
        """
        Run database migrations using Alembic.
        
        Args:
            alembic_cfg_path: Path to Alembic configuration file
        """
        try:
            logger.info("Running database migrations")
            
            # Check if config file exists
            if not os.path.exists(alembic_cfg_path):
                logger.error(f"Alembic config file not found: {alembic_cfg_path}")
                return
            
            # Load Alembic config
            alembic_cfg = Config(alembic_cfg_path)
            
            # Run migrations
            command.upgrade(alembic_cfg, 'head')
            
            logger.info("Database migrations completed successfully")
        except Exception as e:
            logger.error(f"Error running database migrations: {str(e)}")
    
    def check_connection(self) -> bool:
        """
        Check database connection.
        
        Returns:
            Connection status
        """
        try:
            # Try to connect and run a simple query
            with engine.connect() as conn:
                conn.execute(text('SELECT 1'))
            return True
        except Exception as e:
            logger.error(f"Database connection check failed: {str(e)}")
            return False
    
    async def check_connection_async(self) -> bool:
        """
        Check database connection (async version).
        
        Returns:
            Connection status
        """
        try:
            # Try to connect and run a simple query
            if not self.is_sqlite:
                async with async_engine.connect() as conn:
                    await conn.execute(text('SELECT 1'))
            else:
                # For SQLite, use synchronous version
                return self.check_connection()
            return True
        except Exception as e:
            logger.error(f"Async database connection check failed: {str(e)}")
            return False
    
    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get connection pool status.
        
        Returns:
            Pool status information
        """
        return {
            'total_connections': self.stats['total_connections'],
            'active_connections': self.stats['active_connections'],
            'pool_checkouts': self.stats['pool_checkouts'],
            'pool_checkins': self.stats['pool_checkins'],
            'pool_recycles': self.stats['pool_recycles'],
            'pool_size': self.pool_size,
            'max_overflow': self.max_overflow,
        }
    
    async def close(self) -> None:
        """Close all database connections."""
        global engine, async_engine
        
        if engine:
            engine.dispose()
        
        if async_engine and not self.is_sqlite:
            await async_engine.dispose()
        
        logger.info("Database connections closed")

async def initialize_database(
    db_uri: str,
    pool_size: int = 5,
    max_overflow: int = 10
) -> DatabaseManager:
    """
    Initialize database connection.
    
    Args:
        db_uri: Database connection URI
        pool_size: Connection pool size
        max_overflow: Maximum overflow connections
        
    Returns:
        Database manager instance
    """
    global db_manager
    
    try:
        logger.info(f"Initializing database connection: {db_uri}")
        
        # Create database manager
        db_manager = DatabaseManager(
            db_uri=db_uri,
            pool_size=pool_size,
            max_overflow=max_overflow
        )
        
        # Check connection
        if not await db_manager.check_connection_async():
            logger.critical("Failed to connect to database")
            return None
        
        # Create tables if they don't exist
        await db_manager.create_all_async()
        
        # Run migrations if alembic.ini exists
        if os.path.exists('alembic.ini'):
            db_manager.run_migrations()
        
        logger.info("Database initialization completed successfully")
        return db_manager
        
    except Exception as e:
        logger.critical(f"Error initializing database: {str(e)}")
        return None

def get_db():
    """
    Get database session for dependency injection.
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_async_db():
    """
    Get async database session for dependency injection.
    
    Yields:
        Async database session
    """
    async with AsyncSessionLocal() as session:
        yield session
