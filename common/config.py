"""
Configuration Management for QuantumSpectre Elite Trading System.

This module handles loading and managing system configuration from various sources
including environment variables, YAML files, and command line arguments.
"""

import os
import sys
import yaml
import logging
from typing import Dict, Any, Optional, List, Union
from common.logger import get_logger
from dataclasses import dataclass, field, asdict
import json
import socket
import pathlib
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

logger = get_logger(__name__)

@dataclass
class SystemSettings:
    """System configuration settings."""
    
    # General settings
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"  # Options: development, production, test
    DEBUG_MODE: bool = False
    USE_TESTNET: bool = True
    
    # Path settings
    BASE_DIR: str = str(pathlib.Path(__file__).parent.parent.absolute())
    DATA_DIR: str = field(default_factory=lambda: os.path.join(str(pathlib.Path(__file__).parent.parent.absolute()), "data"))
    LOG_DIR: str = field(default_factory=lambda: os.path.join(str(pathlib.Path(__file__).parent.parent.absolute()), "logs"))
    MODEL_DIR: str = field(default_factory=lambda: os.path.join(str(pathlib.Path(__file__).parent.parent.absolute()), "models"))
    TEMP_DIR: str = field(default_factory=lambda: os.path.join(str(pathlib.Path(__file__).parent.parent.absolute()), "temp"))
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE_MAX_SIZE: int = 10485760  # 10MB
    LOG_FILE_BACKUP_COUNT: int = 10
    LOG_TO_CONSOLE: bool = True
    
    # Database settings
    DATABASE_URI: str = "sqlite:///data/quantum_spectre.db"
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_POOL_RECYCLE: int = 3600
    
    # Cache settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    
    # Security settings
    SECRET_KEY: str = "change_this_in_production"
    SECRET_MANAGER_TYPE: str = "file"  # Options: file, aws, vault
    ENCRYPTION_KEY_PATH: str = "secrets/encryption_key.key"
    JWT_SECRET_KEY: str = "change_this_in_production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_TIMEOUT: int = 60
    API_CORS_ORIGINS: List[str] = field(default_factory=lambda: ["*"])
    
    # WebSocket settings
    WS_HOST: str = "0.0.0.0"
    WS_PORT: int = 8001
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_MAX_CLIENTS: int = 1000
    
    # Dashboard settings
    DASHBOARD_HOST: str = "0.0.0.0"
    DASHBOARD_PORT: int = 8002
    
    # Exchange API settings
    BINANCE_API_KEY: str = ""
    BINANCE_API_SECRET: str = ""
    BINANCE_API_URL: str = "https://testnet.binance.vision/api"  # Testnet by default
    BINANCE_SYMBOLS: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
    
    DERIV_API_KEY: str = ""
    DERIV_APP_ID: str = ""
    DERIV_API_URL: str = "https://api.deriv.com"
    DERIV_SYMBOLS: List[str] = field(default_factory=lambda: ["R_10", "R_25", "R_50", "R_75", "R_100"])
    
    # Connectivity settings
    CONNECTION_TIMEOUT: int = 30
    RETRY_ATTEMPTS: int = 3
    RETRY_DELAY: int = 5
    HTTP_TIMEOUT: int = 30
    HTTP_MAX_RETRIES: int = 3
    HTTP_RETRY_DELAY: int = 3
    
    # Performance and hardware settings
    USE_GPU: bool = False
    NUM_WORKERS: int = 4
    MODEL_CACHE_SIZE: int = 10
    BATCH_SIZE: int = 32
    
    # Trading settings
    DEFAULT_QUOTE_CURRENCY: str = "USDT"
    DEFAULT_LEVERAGE: int = 1
    MAX_LEVERAGE: int = 10
    MAX_RISK_PER_TRADE: float = 0.02  # 2% of account balance
    MAX_DRAWDOWN: float = 0.15  # 15% maximum drawdown
    STOP_LOSS_PERCENT: float = 0.05  # 5% stop loss
    TAKE_PROFIT_PERCENT: float = 0.10  # 10% take profit
    ENABLE_TRAILING_STOP: bool = True
    
    # Evolution settings
    EVOLUTION_POPULATION_SIZE: int = 100
    EVOLUTION_GENERATIONS: int = 50
    EVOLUTION_MUTATION_RATE: float = 0.05
    EVOLUTION_CROSSOVER_RATE: float = 0.7
    EVOLUTION_TOURNAMENT_SIZE: int = 5
    
    # Monitoring settings
    ENABLE_PROMETHEUS: bool = True
    METRICS_PORT: int = 9090
    HEALTH_CHECK_INTERVAL: int = 60
    ALERT_EMAIL: str = ""
    
    # App information
    APP_NAME: str = "QuantumSpectre Elite"
    APP_DESCRIPTION: str = "Advanced AI-Driven Trading System"
    APP_VERSION: str = "1.0.0"
    HOSTNAME: str = field(default_factory=socket.gethostname)
    
    # Feature flags
    ENABLE_DARKWEB_MONITORING: bool = False
    ENABLE_NEWS_SENTIMENT: bool = True
    ENABLE_SOCIAL_MEDIA_TRACKING: bool = True
    ENABLE_PATTERN_RECOGNITION: bool = True
    ENABLE_LOOPHOLE_DETECTION: bool = True
    ENABLE_SELF_EVOLUTION: bool = True
    
    # Custom parameters
    CUSTOM_PARAMS: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, settings_dict: Dict[str, Any]) -> None:
        """
        Update settings from dictionary.
        
        Args:
            settings_dict: Dictionary with settings to update
        """
        for key, value in settings_dict.items():
            if hasattr(self, key):
                # Handle special cases for directory paths
                if key.endswith('_DIR') and not os.path.isabs(value):
                    value = os.path.join(self.BASE_DIR, value)
                
                # Update the attribute
                setattr(self, key, value)
            else:
                # Store unknown settings in CUSTOM_PARAMS
                self.CUSTOM_PARAMS[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary.
        
        Returns:
            Dictionary representation of settings
        """
        return asdict(self)
    
    def to_json(self) -> str:
        """
        Convert settings to JSON string.
        
        Returns:
            JSON string representation of settings
        """
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def create_directories(self) -> None:
        """Create required directories if they don't exist."""
        for dir_attr in ('DATA_DIR', 'LOG_DIR', 'MODEL_DIR', 'TEMP_DIR'):
            dir_path = getattr(self, dir_attr)
            os.makedirs(dir_path, exist_ok=True)
            
    def validate(self) -> List[str]:
        """
        Validate settings.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check if directories exist or can be created
        for dir_attr in ('DATA_DIR', 'LOG_DIR', 'MODEL_DIR', 'TEMP_DIR'):
            dir_path = getattr(self, dir_attr)
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create {dir_attr} directory: {str(e)}")
        
        # Validate log level
        valid_log_levels = ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        if self.LOG_LEVEL not in valid_log_levels:
            errors.append(f"Invalid LOG_LEVEL: {self.LOG_LEVEL}. Must be one of {valid_log_levels}")
        
        # Validate environment
        valid_envs = ('development', 'production', 'test')
        if self.ENVIRONMENT not in valid_envs:
            errors.append(f"Invalid ENVIRONMENT: {self.ENVIRONMENT}. Must be one of {valid_envs}")
            
        # Validate trading parameters
        if not 0 < self.MAX_RISK_PER_TRADE < 1:
            errors.append(f"Invalid MAX_RISK_PER_TRADE: {self.MAX_RISK_PER_TRADE}. Must be between 0 and 1")
            
        if not 0 < self.MAX_DRAWDOWN < 1:
            errors.append(f"Invalid MAX_DRAWDOWN: {self.MAX_DRAWDOWN}. Must be between 0 and 1")
        
        return errors

# Initialize default settings
settings = SystemSettings()

def load_config(config_path: str) -> bool:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Success status
    """
    try:
        # Check if file exists
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            return False
        
        # Load configuration from file
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_data = yaml.safe_load(f)
            elif config_path.endswith('.json'):
                config_data = json.load(f)
            else:
                logger.error(f"Unsupported configuration file format: {config_path}")
                return False
                
        # Update settings with environment variables
        env_vars = {}
        for key in dir(settings):
            if key.isupper():  # Only consider uppercase attributes
                env_var = f"QUANTUM_SPECTRE_{key}"
                if env_var in os.environ:
                    # Handle type conversion
                    orig_value = getattr(settings, key)
                    env_value = os.environ[env_var]
                    
                    # Convert to appropriate type
                    if isinstance(orig_value, bool):
                        env_vars[key] = env_value.lower() in ('true', 'yes', '1', 'y')
                    elif isinstance(orig_value, int):
                        env_vars[key] = int(env_value)
                    elif isinstance(orig_value, float):
                        env_vars[key] = float(env_value)
                    elif isinstance(orig_value, list):
                        env_vars[key] = env_value.split(',')
                    else:
                        env_vars[key] = env_value
        
        # Update settings
        if config_data:
            settings.update(config_data)
        if env_vars:
            settings.update(env_vars)
        
        # Create required directories
        settings.create_directories()
        
        # Validate settings
        errors = settings.validate()
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return False
        
def get_settings() -> SystemSettings:
    """
    Get current settings.
    
    Returns:
        Current settings
    """
    return settings
