#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Database Migration Module

This module provides database migration capabilities using Alembic.
It handles schema evolution and version management for the system's databases.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Union, Any

from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError

from config import Config
from common.logger import get_logger
from common.exceptions import MigrationError

logger = get_logger(__name__)

class MigrationManager:
    """
    Manages database migrations using Alembic.
    
    This class provides methods to initialize, upgrade, downgrade, and check the 
    status of database migrations. It handles multiple database connections and
    ensures data integrity during schema changes.
    """
    
    def __init__(self, config: Config, alembic_location: str = 'data_storage/migrations'):
        """
        Initialize the migration manager.
        
        Args:
            config: System configuration
            alembic_location: Path to the alembic migrations directory
        """
        self.config = config
        self.alembic_location = alembic_location
        self.alembic_cfg = self._create_alembic_config()
        self.engines = {}
        self._initialize_engines()
        logger.info("Migration manager initialized")
    
    def _create_alembic_config(self) -> AlembicConfig:
        """
        Create an Alembic configuration object.
        
        Returns:
            AlembicConfig: Configured alembic config object
        """
        alembic_cfg = AlembicConfig()
        alembic_cfg.set_main_option("script_location", self.alembic_location)
        alembic_cfg.set_main_option("sqlalchemy.url", self.config.get("database.main_url"))
        
        # Set additional options for alembic
        alembic_cfg.set_main_option("file_template", "%%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d%%(second).2d_%%(rev)s_%%(slug)s")
        alembic_cfg.set_main_option("timezone", "UTC")
        
        return alembic_cfg
    
    def _initialize_engines(self) -> None:
        """
        Initialize database engines for all configured databases.
        
        This method creates SQLAlchemy engine instances for each database
        defined in the configuration.
        """
        try:
            main_url = self.config.get("database.main_url")
            self.engines["main"] = create_engine(main_url, pool_pre_ping=True)
            
            # Initialize additional databases if configured
            for db_name, db_url in self.config.get("database.additional_urls", {}).items():
                self.engines[db_name] = create_engine(db_url, pool_pre_ping=True)
                
            logger.info(f"Initialized database engines for {len(self.engines)} databases")
        except Exception as e:
            logger.error(f"Failed to initialize database engines: {e}")
            raise MigrationError(f"Database engine initialization failed: {e}")
    
    def check_database_exists(self, db_name: str = "main") -> bool:
        """
        Check if the database exists and is accessible.
        
        Args:
            db_name: Name of the database to check
            
        Returns:
            bool: True if the database exists and is accessible
        """
        if db_name not in self.engines:
            logger.warning(f"Database '{db_name}' not found in configured engines")
            return False
            
        try:
            conn = self.engines[db_name].connect()
            conn.close()
            logger.debug(f"Successfully connected to database '{db_name}'")
            return True
        except SQLAlchemyError as e:
            logger.warning(f"Failed to connect to database '{db_name}': {e}")
            return False
    
    def initialize_database(self, db_name: str = "main") -> None:
        """
        Initialize a new database with the current schema version.
        
        Args:
            db_name: Name of the database to initialize
        
        Raises:
            MigrationError: If database initialization fails
        """
        if not self.check_database_exists(db_name):
            raise MigrationError(f"Database '{db_name}' does not exist or is not accessible")
        
        try:
            # Check if the database already has migrations
            if self.has_migration_history(db_name):
                logger.info(f"Database '{db_name}' already has migration history, skipping initialization")
                return
                
            logger.info(f"Initializing database '{db_name}' with migrations")
            
            # Use the appropriate URL for the selected database
            if db_name != "main":
                original_url = self.alembic_cfg.get_main_option("sqlalchemy.url")
                self.alembic_cfg.set_main_option("sqlalchemy.url", 
                                                self.config.get(f"database.additional_urls.{db_name}"))
            
            # Initialize the migration history table
            command.stamp(self.alembic_cfg, "base")
            
            # Upgrade to the latest version
            command.upgrade(self.alembic_cfg, "head")
            
            # Restore the original URL if it was changed
            if db_name != "main":
                self.alembic_cfg.set_main_option("sqlalchemy.url", original_url)
                
            logger.info(f"Successfully initialized database '{db_name}' with migrations")
        except Exception as e:
            logger.error(f"Failed to initialize database '{db_name}': {e}")
            raise MigrationError(f"Database initialization failed: {e}")
    
    def has_migration_history(self, db_name: str = "main") -> bool:
        """
        Check if the database has a migration history table.
        
        Args:
            db_name: Name of the database to check
            
        Returns:
            bool: True if the database has migration history
        """
        try:
            inspector = inspect(self.engines[db_name])
            has_alembic_table = 'alembic_version' in inspector.get_table_names()
            logger.debug(f"Database '{db_name}' {'has' if has_alembic_table else 'does not have'} migration history")
            return has_alembic_table
        except SQLAlchemyError as e:
            logger.warning(f"Error checking migration history for '{db_name}': {e}")
            return False
    
    def get_current_revision(self, db_name: str = "main") -> Optional[str]:
        """
        Get the current migration revision of the database.
        
        Args:
            db_name: Name of the database to check
            
        Returns:
            Optional[str]: The current revision or None if no migrations are applied
        """
        if not self.has_migration_history(db_name):
            logger.debug(f"No migration history found for database '{db_name}'")
            return None
            
        try:
            with self.engines[db_name].connect() as conn:
                result = conn.execute("SELECT version_num FROM alembic_version").scalar()
                logger.debug(f"Current migration revision for database '{db_name}': {result}")
                return result
        except SQLAlchemyError as e:
            logger.warning(f"Error retrieving current revision for '{db_name}': {e}")
            return None
    
    def upgrade_database(self, target: str = "head", db_name: str = "main") -> None:
        """
        Upgrade the database to the specified revision.
        
        Args:
            target: Target revision (default: 'head' for latest)
            db_name: Name of the database to upgrade
            
        Raises:
            MigrationError: If database upgrade fails
        """
        try:
            # Use the appropriate URL for the selected database
            if db_name != "main":
                original_url = self.alembic_cfg.get_main_option("sqlalchemy.url")
                self.alembic_cfg.set_main_option("sqlalchemy.url", 
                                                self.config.get(f"database.additional_urls.{db_name}"))
            
            logger.info(f"Upgrading database '{db_name}' to revision '{target}'")
            command.upgrade(self.alembic_cfg, target)
            
            # Restore the original URL if it was changed
            if db_name != "main":
                self.alembic_cfg.set_main_option("sqlalchemy.url", original_url)
                
            logger.info(f"Successfully upgraded database '{db_name}' to revision '{target}'")
        except Exception as e:
            logger.error(f"Failed to upgrade database '{db_name}': {e}")
            raise MigrationError(f"Database upgrade failed: {e}")
    
    def downgrade_database(self, target: str, db_name: str = "main") -> None:
        """
        Downgrade the database to the specified revision.
        
        Args:
            target: Target revision
            db_name: Name of the database to downgrade
            
        Raises:
            MigrationError: If database downgrade fails
        """
        try:
            # Use the appropriate URL for the selected database
            if db_name != "main":
                original_url = self.alembic_cfg.get_main_option("sqlalchemy.url")
                self.alembic_cfg.set_main_option("sqlalchemy.url", 
                                                self.config.get(f"database.additional_urls.{db_name}"))
            
            logger.info(f"Downgrading database '{db_name}' to revision '{target}'")
            command.downgrade(self.alembic_cfg, target)
            
            # Restore the original URL if it was changed
            if db_name != "main":
                self.alembic_cfg.set_main_option("sqlalchemy.url", original_url)
                
            logger.info(f"Successfully downgraded database '{db_name}' to revision '{target}'")
        except Exception as e:
            logger.error(f"Failed to downgrade database '{db_name}': {e}")
            raise MigrationError(f"Database downgrade failed: {e}")
    
    def create_migration(self, message: str) -> None:
        """
        Create a new migration with the given message.
        
        Args:
            message: Description of the migration
            
        Raises:
            MigrationError: If migration creation fails
        """
        try:
            logger.info(f"Creating new migration: '{message}'")
            command.revision(self.alembic_cfg, message=message, autogenerate=True)
            logger.info(f"Successfully created new migration: '{message}'")
        except Exception as e:
            logger.error(f"Failed to create migration '{message}': {e}")
            raise MigrationError(f"Migration creation failed: {e}")
    
    def get_migration_history(self, db_name: str = "main") -> List[Dict[str, Any]]:
        """
        Get the migration history of the database.
        
        Args:
            db_name: Name of the database to check
            
        Returns:
            List[Dict[str, Any]]: List of migration history entries
        """
        if not self.has_migration_history(db_name):
            logger.debug(f"No migration history found for database '{db_name}'")
            return []
            
        try:
            from alembic.script import ScriptDirectory
            from alembic.runtime.migration import MigrationContext
            
            # Get the script directory
            script = ScriptDirectory.from_config(self.alembic_cfg)
            
            # Get the current revision
            current_rev = self.get_current_revision(db_name)
            
            # Get all revisions
            revisions = []
            for rev in script.walk_revisions():
                is_current = rev.revision == current_rev
                revisions.append({
                    "revision": rev.revision,
                    "down_revision": rev.down_revision,
                    "description": rev.doc,
                    "is_current": is_current,
                    "created_date": rev.path.stem.split("_")[0]
                })
            
            # Sort by date (the revision filename starts with the date)
            revisions.sort(key=lambda x: x["created_date"], reverse=True)
            
            logger.debug(f"Retrieved {len(revisions)} migration history entries for database '{db_name}'")
            return revisions
        except Exception as e:
            logger.warning(f"Error retrieving migration history for '{db_name}': {e}")
            return []


# Initialize migration manager instance when module is imported
migration_manager = None

def initialize(config: Config) -> MigrationManager:
    """
    Initialize the migration manager.
    
    Args:
        config: System configuration
        
    Returns:
        MigrationManager: Initialized migration manager
    """
    global migration_manager
    if migration_manager is None:
        migration_manager = MigrationManager(config)
    return migration_manager

def get_migration_manager() -> MigrationManager:
    """
    Get the migration manager instance.
    
    Returns:
        MigrationManager: Migration manager instance
        
    Raises:
        RuntimeError: If migration manager is not initialized
    """
    global migration_manager
    if migration_manager is None:
        raise RuntimeError("Migration manager not initialized. Call initialize() first.")
    return migration_manager
