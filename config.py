#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Configuration Module

This module handles loading, validating, and providing access to system configuration.
It supports multiple configuration formats, environment variable overrides, and validation.
"""

import os
import sys
import json
import yaml
try:
    import tomllib as toml_parser  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - older Pythons
    import tomli as toml_parser
try:  # optional writer support
    import tomli_w
except ModuleNotFoundError:  # pragma: no cover - writer is optional
    tomli_w = None
import logging
import pkgutil
import importlib
import traceback
import copy
from typing import Dict, List, Any, Optional, Union, Callable, ClassVar
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from common.exceptions import ConfigurationError
from common.constants import (
    DEFAULT_CONFIG_PATH, CONFIG_SCHEMA_VERSION, SYSTEM_NAME,
    SERVICE_NAMES, SUPPORTED_PLATFORMS, SUPPORTED_ASSETS,
    DEFAULT_LOG_LEVEL, DEFAULT_EXCHANGE_CONFIGS
)

# Default configurations
DEFAULT_CONFIG = {
    "schema_version": CONFIG_SCHEMA_VERSION,
    "system": {
        "name": SYSTEM_NAME,
        "environment": "development",
        "debug": False,
        "auto_migrate": True,
        "timezone": "UTC",
        "gpu_enabled": True,
        "data_dir": "./data",
        "max_workers": 8,
        "pid_file": f"/tmp/{SYSTEM_NAME}.pid"
    },
    "services": {
        service: {"enabled": True, "auto_restart": True, "critical": False}
        for service in SERVICE_NAMES
    },
    "logging": {
        "level": DEFAULT_LOG_LEVEL,
        "file": None,
        "max_size": 10485760,  # 10MB
        "backup_count": 5,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "console": True,
        "sentry_dsn": None,
    },
    "database": {
        "type": "postgresql",
        "host": "localhost",
        "port": 5432,
        "username": "postgres",
        "password": "",
        "database": "quantumspectre",
        "pool_size": 10,
        "ssl": False,
        "timeout": 30
    },
    "redis": {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": None,
        "ssl": False,
        "timeout": 10,
        "max_connections": 50
    },
    "security": {
        "encryption_key": None,
        "key_file": "./.encryption_key",
        "storage_method": "file",
        "storage_path": "./credentials",
        "auto_generate": True,
        "api_token_expiry": 86400,  # 24 hours
        "password_hash_algorithm": "argon2",
        "encryption_algorithm": "AES-256-GCM",
        "allowed_origins": ["http://localhost:3000"]
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "ssl": False,
        "ssl_cert": None,
        "ssl_key": None,
        "workers": 4,
        "cors_enabled": True,
        "rate_limit": {
            "enabled": True,
            "limit": 100,
            "period": 60
        },
        "timeout": 30,
        "prefix": "/api/v1"
    },
    "ui": {
        "host": "0.0.0.0",
        "port": 3000,
        "assets_dir": "./ui/assets",
        "template_dir": "./ui/templates",
        "cache_dir": "./ui/cache",
        "static_dir": "./ui/dist",
        "index_file": "index.html",
        "debug": False,
        "ssl": False,
        "ssl_cert": None,
        "ssl_key": None,
        "session_secret": None,
        "session_timeout": 86400  # 24 hours
    },
    "exchanges": DEFAULT_EXCHANGE_CONFIGS,
    "data_ingest": {
        "processors": {},
        "sources": {},
        "store_processed_data": False
    },
    "voice_assistant": {
        "enabled": True,
        "model": "local",  # "local" or "cloud"
        "voice": "en-US-Standard-D",
        "language": "en-US",
        "volume": 1.0,
        "rate": 1.0,
        "pitch": 0.0,
        "notification_sounds": True,
        "voice_commands": True,
        "confidence_threshold": 0.75,
        "auto_mute_during_calls": True,
        "cloud_api_key": None,
        "local_model_path": "./models/voice",
        "max_message_length": 1000,
        "summarize_long_messages": True
    },
    "backtesting": {
        "data_source": "database",  # "database", "file", or "api"
        "default_timeframe": "1h",
        "default_start_date": "2020-01-01",
        "default_end_date": "now",
        "commission": {
            "binance": {
                "maker": 0.001,
                "taker": 0.001
            },
            "deriv": {
                "maker": 0.003,
                "taker": 0.003
            }
        },
        "slippage": 0.0005,  # 0.05%
        "initial_capital": 10.0,  # $10
        "cache_results": True,
        "cache_dir": "./cache/backtest",
        "parallel_runs": 4
    },
    "trading": {
        "enabled": False,  # Default to disabled for safety
        "mode": "paper",  # "paper", "live"
        "default_platform": "binance",
        "default_asset": "BTC/USD",
        "max_open_positions": 5,
        "default_position_size": 0.01,  # 1% of account
        "max_position_size": 0.1,  # 10% of account
        "use_leverage": False,
        "max_leverage": 1,
        "min_profit_threshold": 0.005,  # 0.5%
        "stop_loss": {
            "enabled": True,
            "default": 0.02,  # 2%
            "trailing": {
                "enabled": True,
                "activation": 0.01,  # 1%
                "distance": 0.005  # 0.5%
            }
        },
        "take_profit": {
            "enabled": True,
            "default": 0.03,  # 3%
            "partial": {
                "enabled": True,
                "levels": [
                    {"target": 0.01, "size": 0.3},  # At 1% profit, take 30% off
                    {"target": 0.02, "size": 0.3},  # At 2% profit, take another 30% off
                    {"target": 0.03, "size": 0.4}   # At 3% profit, take remaining 40% off
                ]
            }
        },
        "risk_management": {
            "max_daily_drawdown": 0.05,  # 5%
            "max_weekly_drawdown": 0.1,  # 10%
            "max_drawdown": 0.2,  # 20%
            "max_daily_trades": 10,
            "max_concurrent_trades": 3,
            "correlation_limit": 0.7,  # Limit correlated assets
            "required_confidence": 0.75,  # Minimum signal confidence
            "stop_trading_after_consecutive_losses": 5
        },
        "session_times": {
            "enabled": False,
            "timezone": "UTC",
            "sessions": [
                {"days": [0, 1, 2, 3, 4], "start": "08:00", "end": "16:00"},  # Weekdays 8am-4pm
                {"days": [5, 6], "enabled": False}  # Disabled on weekends
            ]
        }
    },
    "machine_learning": {
        "enabled": True,
        "model_dir": "./models/ml",
        "cache_dir": "./cache/ml",
        "gpu_memory_limit": 0.8,  # Use 80% of available GPU memory
        "training": {
            "epochs": 100,
            "batch_size": 64,
            "validation_split": 0.2,
            "early_stopping": True,
            "early_stopping_patience": 10,
            "save_best_only": True,
            "shuffle": True,
            "automatic_hyperparameter_tuning": True,
            "cross_validation_folds": 5
        },
        "prediction": {
            "ensemble": True,
            "threshold": 0.6,
            "batch_size": 128,
            "cache_predictions": True,
            "cache_ttl": 300  # 5 minutes
        },
        "feature_importance": {
            "enabled": True,
            "method": "shap",  # shap, permutation, or gradient
            "n_repeats": 5
        },
        "retraining": {
            "auto_retrain": True,
            "schedule": "daily",  # hourly, daily, weekly
            "min_samples": 1000,
            "performance_threshold": 0.05  # Retrain if performance drops by 5%
        }
    },
    "ml_models": {
        "type": "supervised",
        "rl_algorithm": "dqn",
        "rl_episodes": 10,
        "rl_features": []
    },
    "intelligence": {
        "loophole_detection": {
            "enabled": True,
            "sensitivity": 0.8,
            "min_confidence": 0.75,
            "scan_interval": 300,  # 5 minutes
            "exploit_verified_only": True,
            "max_concurrent_exploits": 2
        },
        "pattern_recognition": {
            "enabled": True,
            "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
            "min_pattern_quality": 0.7,
            "max_patterns_per_asset": 5,
            "confirmation_required": True
        },
        "market_regime": {
            "detection_method": "hmm",  # hmm, clustering, or threshold
            "update_interval": 3600,  # 1 hour
            "history_length": 30,  # days
            "num_regimes": 4,  # trending up, trending down, range bound, volatile
            "auto_adapt": True
        },
        "adaptive_learning": {
            "reinforcement": {
                "enabled": True,
                "learning_rate": 0.001,
                "discount_factor": 0.95,
                "exploration_rate": 0.1,
                "min_exploration_rate": 0.01,
                "exploration_decay": 0.995
            },
            "genetic": {
                "enabled": True,
                "population_size": 50,
                "generations": 20,
                "mutation_rate": 0.1,
                "crossover_rate": 0.7,
                "elite_size": 5
            }
        }
    },
    "data_feeds": {
        "binance": {
            "enabled": True,
            "api_key": "",
            "api_secret": "",
            "testnet": True,
            "timeout": 10,
            "rate_limit_buffer": 0.8,  # Use 80% of rate limit
            "reconnect_attempts": 5,
            "reconnect_delay": 5,
            "websocket": {
                "enabled": True,
                "snapshot_interval": 3600,  # 1 hour
                "ping_interval": 30,
                "ping_timeout": 10
            }
        },
        "deriv": {
            "enabled": True,
            "app_id": "",
            "api_token": "",
            "demo": True,
            "timeout": 10,
            "reconnect_attempts": 5,
            "reconnect_delay": 5,
            "websocket": {
                "enabled": True,
                "ping_interval": 30,
                "ping_timeout": 10
            }
        },
        "news": {
            "enabled": True,
            "sources": ["cryptopanic", "newsapi", "twitter", "reddit"],
            "update_interval": 300,  # 5 minutes
            "sentiment_analysis": True,
            "relevance_threshold": 0.7,
            "rate_limit_buffer": 0.8
        },
        "social": {
            "enabled": True,
            "sources": ["twitter", "reddit", "telegram"],
            "update_interval": 300,  # 5 minutes
            "sentiment_analysis": True,
            "influence_weighting": True,
            "rate_limit_buffer": 0.8
        },
        "onchain": {
            "enabled": True,
            "networks": ["bitcoin", "ethereum", "binance-smart-chain"],
            "update_interval": 600,  # 10 minutes
            "metrics": ["transactions", "fees", "whales", "token_transfers"],
            "archive_node": False,
            "api_keys": {}
        },
        "dark_web": {
            "enabled": False,  # Disabled by default for ethical/legal reasons
            "tor_proxy": "socks5h://localhost:9050",
            "update_interval": 3600,  # 1 hour
            "anonymity_level": "high",
            "keywords": ["cryptocurrency", "exchange", "hack", "exploit"],
            "retention_period": 7  # days
        }
    },
    "feature_service": {
        "enabled": True,
        "cache_ttl": 300,  # 5 minutes
        "parallel_processing": True,
        "max_workers": 4,
        "batch_size": 1024, # Added default batch size
        "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
        "real_time": True,
        "cross_asset_features": True,
        "history_length": {
            "1m": 24,    # 24 hours
            "5m": 72,    # 3 days
            "15m": 168,  # 7 days
            "1h": 720,   # 30 days
            "4h": 1440,  # 60 days
            "1d": 3650   # 10 years
        },
        "feature_groups": {
            "technical": True,
            "volatility": True,
            "volume": True,
            "sentiment": True,
            "market_structure": True,
            "order_flow": True,
            "pattern": True,
            "cross_asset": True
        }
    },
    "brain_council": {
        "enabled": True,
        "voting_method": "weighted",  # simple, weighted, or confidence
        "council_types": {
            "timeframe": True,
            "asset": True,
            "regime": True,
            "master": True
        },
        "performance_tracking": {
            "window": 100,  # trades
            "decay_factor": 0.98  # Weight recent performance more
        },
        "weighting": {
            "initial": "equal",  # equal, performance, or custom
            "min_weight": 0.05,
            "max_weight": 0.5,
            "auto_adjust": True,
            "adjustment_interval": 10  # trades
        },
        "signal_generation": {
            "min_confidence": 0.6,
            "min_consensus": 0.5,  # Minimum 50% of brains must agree
            "strength_threshold": 0.7,
            "signal_ttl": 300  # 5 minutes
        }
    },
    "monitoring": {
        "enabled": True,
        "metrics_interval": 10,  # seconds
        "metrics_retention": 86400,  # 1 day
        "system_health_check_interval": 60,  # 1 minute
        "alerting": {
            "enabled": True,
            "channels": ["console", "log", "email"],
            "email": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "use_tls": True,
                "username": "",
                "password": "",
                "from_email": "",
                "to_email": ""
            },
            "performance_alerts": {
                "enabled": True,
                "drawdown_threshold": 0.1,  # 10%
                "profit_threshold": 0.1,  # 10%
                "consecutive_losses": 3
            },
            "system_alerts": {
                "enabled": True,
                "cpu_threshold": 90,  # %
                "memory_threshold": 90,  # %
                "disk_threshold": 90,  # %
                "service_restart_threshold": 3
            }
        },
        "log_analysis": {
            "enabled": True,
            "error_pattern_detection": True,
            "anomaly_detection": True,
            "scan_interval": 300  # 5 minutes
        }
    }
}

@dataclass
class Config:
    """Configuration container with validation and access methods."""

    data: Dict[str, Any] = None
    _loaded_config: ClassVar[Optional['Config']] = None
    
    def __init__(self, data: Dict[str, Any] = None):
        """
        Initialize configuration with provided data or defaults.
        
        Args:
            data: Configuration data dictionary
        """
        self.data = copy.deepcopy(DEFAULT_CONFIG) if data is None else copy.deepcopy(data)
        self._environment_override()
    
    def __getattr__(self, name: str) -> Any:
        """
        Allow attribute-style access to configuration sections.
        
        Args:
            name: Section name
            
        Returns:
            Configuration section
            
        Raises:
            AttributeError: If section doesn't exist
        """
        if name in self.data:
            return self.data[name]
        raise AttributeError(f"Configuration has no section '{name}'")

    @classmethod
    def get_section(cls, section_name: str, default: Any = None) -> Any:
        """Return a top-level configuration section.

        If a configuration file has been loaded via :func:`load_config`, the
        section will be read from that configuration instance. Otherwise the
        value from :data:`DEFAULT_CONFIG` is returned. In other words, when no
        config file has been loaded this method behaves like::

            DEFAULT_CONFIG.get(section_name, default)
        """
        if cls._loaded_config is None:
            return DEFAULT_CONFIG.get(section_name, default)
        return cls._loaded_config.data.get(section_name, default)
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation path.
        
        Args:
            path: Configuration path (e.g. "database.host")
            default: Default value if path doesn't exist
            
        Returns:
            Configuration value or default
        """
        current = self.data
        for part in path.split('.'):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current
    
    def set(self, path: str, value: Any) -> None:
        """
        Set configuration value using dot notation path.
        
        Args:
            path: Configuration path (e.g. "database.host")
            value: Value to set
        """
        parts = path.split('.')
        current = self.data
        
        # Navigate to the right level
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Set the value
        current[parts[-1]] = value
    
    def _environment_override(self) -> None:
        """
        Override configuration with environment variables.
        
        Environment variables should be in the format:
        QUANTUMSPECTRE_SECTION_KEY=value
        
        For example:
        QUANTUMSPECTRE_DATABASE_HOST=localhost
        QUANTUMSPECTRE_TRADING_ENABLED=true
        """
        prefix = "QUANTUMSPECTRE_"
        
        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                # Remove prefix and split into parts
                path = env_var[len(prefix):].lower().replace("_", ".")
                
                # Convert value to appropriate type
                if value.lower() in ("true", "yes", "1", "on"):
                    value = True
                elif value.lower() in ("false", "no", "0", "off"):
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace(".", "", 1).isdigit() and value.count(".") == 1:
                    value = float(value)
                
                # Set the value
                self.set(path, value)
    
    def validate(self) -> None:
        """
        Validate configuration for required fields and appropriate values.
        
        Raises:
            ConfigurationError: If validation fails
        """
        # Check schema version
        schema_version = self.data.get("schema_version")
        if schema_version != CONFIG_SCHEMA_VERSION:
            raise ConfigurationError(
                f"Configuration schema version mismatch. Expected {CONFIG_SCHEMA_VERSION}, got {schema_version}."
            )
            
        # Validate system section
        system = self.data.get("system", {})
        if not system.get("name"):
            raise ConfigurationError("System name must be provided")
            
        # Validate database configuration if enabled
        db = self.data.get("database", {})
        if not db.get("host"):
            raise ConfigurationError("Database host must be provided")
        if not db.get("database"):
            raise ConfigurationError("Database name must be provided")
            
        # Validate Redis configuration if enabled
        redis = self.data.get("redis", {})
        if not redis.get("host"):
            raise ConfigurationError("Redis host must be provided")
            
        # Validate API configuration if enabled
        api = self.data.get("api", {})
        if api.get("ssl") and (not api.get("ssl_cert") or not api.get("ssl_key")):
            raise ConfigurationError("API SSL certificate and key must be provided when SSL is enabled")
            
        # Validate UI configuration if enabled
        ui = self.data.get("ui", {})
        if ui.get("ssl") and (not ui.get("ssl_cert") or not ui.get("ssl_key")):
            raise ConfigurationError("UI SSL certificate and key must be provided when SSL is enabled")
            
        # Validate exchange configurations
        exchanges = self.data.get("exchanges", {})
        for exchange_name, exchange_config in exchanges.items():
            if exchange_config.get("enabled", False):
                # Check for required API credentials if live trading
                if exchange_config.get("mode") == "live":
                    if not exchange_config.get("api_key"):
                        raise ConfigurationError(f"API key must be provided for live trading on {exchange_name}")
                    if not exchange_config.get("api_secret"):
                        raise ConfigurationError(f"API secret must be provided for live trading on {exchange_name}")
                        
                # Validate supported assets
                for asset in exchange_config.get("assets", []):
                    if asset not in SUPPORTED_ASSETS.get(exchange_name, []):
                        raise ConfigurationError(f"Unsupported asset {asset} for exchange {exchange_name}")
                        
        # Validate trading configuration
        trading = self.data.get("trading", {})
        if trading.get("enabled"):
            # Check if it's set to live mode
            if trading.get("mode") == "live":
                # Ensure default platform is valid
                default_platform = trading.get("default_platform")
                if default_platform not in SUPPORTED_PLATFORMS:
                    raise ConfigurationError(f"Unsupported default platform: {default_platform}")
                    
                # Ensure default asset is valid for the platform
                default_asset = trading.get("default_asset")
                if default_asset not in SUPPORTED_ASSETS.get(default_platform, []):
                    raise ConfigurationError(f"Unsupported default asset {default_asset} for platform {default_platform}")
                    
                # Validate risk management settings
                risk = trading.get("risk_management", {})
                if not risk.get("max_drawdown") or risk.get("max_drawdown") > 0.5:
                    raise ConfigurationError("A reasonable max_drawdown (< 50%) must be set for live trading")
                    
        # Validate backtesting configuration
        backtesting = self.data.get("backtesting", {})
        if backtesting.get("data_source") not in ["database", "file", "api"]:
            raise ConfigurationError(f"Invalid backtesting data source: {backtesting.get('data_source')}")
            
        # Additional validations can be added as needed
        
        # All validations passed
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self.data.copy()
    
    def to_json(self, pretty: bool = False) -> str:
        """
        Convert configuration to JSON string.
        
        Args:
            pretty: Whether to format the JSON with indentation
            
        Returns:
            JSON string
        """
        indent = 2 if pretty else None
        return json.dumps(self.data, indent=indent)
    
    def save(self, path: str, format: str = "json") -> None:
        """
        Save configuration to file.
        
        Args:
            path: File path
            format: File format (json, yaml, toml)
            
        Raises:
            ConfigurationError: If format is unsupported or save fails
        """
        try:
            mode = 'wb' if format.lower() == "toml" else 'w'
            with open(path, mode) as f:
                if format.lower() == "json":
                    json.dump(self.data, f, indent=2)
                elif format.lower() == "yaml":
                    yaml.dump(self.data, f)
                elif format.lower() == "toml":
                    if tomli_w is None:
                        raise ConfigurationError(
                            "Saving to TOML requires the optional tomli-w package"
                        )
                    tomli_w.dump(self.data, f)
                else:
                    raise ConfigurationError(f"Unsupported configuration format: {format}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """
        Create configuration from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            Configuration instance
        """
        return cls(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Config':
        """
        Create configuration from JSON string.
        
        Args:
            json_str: JSON string
            
        Returns:
            Configuration instance
            
        Raises:
            ConfigurationError: If JSON parsing fails
        """
        try:
            data = json.loads(json_str)
            return cls(data)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON: {str(e)}")
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'Config':
        """
        Create configuration from YAML string.
        
        Args:
            yaml_str: YAML string
            
        Returns:
            Configuration instance
            
        Raises:
            ConfigurationError: If YAML parsing fails
        """
        try:
            data = yaml.safe_load(yaml_str)
            return cls(data)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML: {str(e)}")
    
    @classmethod
    def from_toml(cls, toml_str: str) -> 'Config':
        """
        Create configuration from TOML string.
        
        Args:
            toml_str: TOML string
            
        Returns:
            Configuration instance
            
        Raises:
            ConfigurationError: If TOML parsing fails
        """
        try:
            data = toml_parser.loads(toml_str)
            return cls(data)
        except toml_parser.TOMLDecodeError as e:
            raise ConfigurationError(f"Invalid TOML: {str(e)}")

def load_config(path: str) -> Config:
    """
    Load configuration from file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Configuration instance
        
    Raises:
        ConfigurationError: If file can't be loaded or parsed
    """
    try:
        path = Path(path)
        with open(path, 'r') as f:
            content = f.read()
            
        # Determine format based on file extension
        ext = path.suffix.lower()
        if ext == '.json':
            config = Config.from_json(content)
        elif ext in ('.yml', '.yaml'):
            config = Config.from_yaml(content)
        elif ext == '.toml':
            config = Config.from_toml(content)
        else:
            raise ConfigurationError(f"Unsupported configuration file format: {ext}")
        Config._loaded_config = config
        return config
    except FileNotFoundError:
        # If file doesn't exist, create a new default config
        config = Config()
        
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # Save default config
        config.save(path)

        Config._loaded_config = config
        return config
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {str(e)}")


def save_config(config: Config, path: str = None) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration object to save
        path: Path to save configuration to (defaults to the path it was loaded from)
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    
    # Determine format based on file extension
    format = "yaml"
    if path.endswith(".json"):
        format = "json"
    elif path.endswith(".toml"):
        format = "toml"
    
    # Save configuration
    config.save(path, format)
