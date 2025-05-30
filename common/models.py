"""
Database models for QuantumSpectre Elite Trading System.

This module defines the SQLAlchemy ORM models for the main entities
in the system, including users, strategies, signals, and other data.
"""

import enum
import uuid
import datetime
from typing import Dict, List, Any, Optional, Union, Set
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON, Enum, Table
from sqlalchemy.orm import relationship
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.sql import func

from common.database import Base, BaseMixin

# Enums for model fields
class UserRole(enum.Enum):
    """User role enum."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API = "api"

class SignalType(enum.Enum):
    """Trading signal type enum."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

class OrderType(enum.Enum):
    """Order type enum."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderStatus(enum.Enum):
    """Order status enum."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class ExchangeType(enum.Enum):
    """Exchange type enum."""
    BINANCE = "binance"
    DERIV = "deriv"

class AssetType(enum.Enum):
    """Asset type enum."""
    CRYPTO = "crypto"
    FOREX = "forex"
    STOCK = "stock"
    COMMODITY = "commodity"
    INDEX = "index"

class TimeframeType(enum.Enum):
    """Timeframe type enum."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

class StrategyType(enum.Enum):
    """Strategy type enum."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    BREAKOUT = "breakout"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    PATTERN_RECOGNITION = "pattern_recognition"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    MACHINE_LEARNING = "machine_learning"
    CUSTOM = "custom"

# User model
class User(Base, BaseMixin):
    """User model for authentication and permissions."""
    
    __tablename__ = "users"
    
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    salt = Column(String(32), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_login = Column(DateTime, nullable=True)
    mfa_secret = Column(String(32), nullable=True)
    mfa_enabled = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    settings = relationship("UserSetting", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f""

# API key model
class ApiKey(Base, BaseMixin):
    """API key model for external API access."""
    
    __tablename__ = "api_keys"
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key = Column(String(64), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    permissions = Column(JSON, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    last_used = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    def __repr__(self):
        return f""

# User settings model
class UserSetting(Base, BaseMixin):
    """User settings model for user preferences."""
    
    __tablename__ = "user_settings"
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    key = Column(String(100), nullable=False)
    value = Column(Text, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="settings")
    
    __table_args__ = (
        {'sqlite_autoincrement': True},
    )
    
    def __repr__(self):
        return f""

# Exchange account model
class ExchangeAccount(Base, BaseMixin):
    """Exchange account model for API connections."""
    
    __tablename__ = "exchange_accounts"
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    exchange = Column(Enum(ExchangeType), nullable=False)
    name = Column(String(100), nullable=False)
    api_key = Column(String(255), nullable=False)
    api_secret = Column(String(255), nullable=False)
    additional_params = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_testnet = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    user = relationship("User")
    orders = relationship("Order", back_populates="exchange_account", cascade="all, delete-orphan")
    
    __table_args__ = (
        {'sqlite_autoincrement': True},
    )
    
    def __repr__(self):
        return f""

# Asset model
class Asset(Base, BaseMixin):
    """Asset model for tradable assets."""
    
    __tablename__ = "assets"
    
    symbol = Column(String(20), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    exchange = Column(Enum(ExchangeType), nullable=False)
    asset_type = Column(Enum(AssetType), nullable=False)
    precision = Column(Integer, default=8, nullable=False)
    min_quantity = Column(Float, nullable=True)
    max_quantity = Column(Float, nullable=True)
    min_price = Column(Float, nullable=True)
    max_price = Column(Float, nullable=True)
    min_notional = Column(Float, nullable=True)
    base_asset = Column(String(10), nullable=True)
    quote_asset = Column(String(10), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    meta = Column(JSON, nullable=True)
    
    # Relationships
    signals = relationship("Signal", back_populates="asset")
    orders = relationship("Order", back_populates="asset")
    
    def __repr__(self):
        return f""

# Signal model
class Signal(Base, BaseMixin):
    """Trading signal model for strategy outputs."""
    
    __tablename__ = "signals"
    
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    signal_type = Column(Enum(SignalType), nullable=False)
    confidence = Column(Float, nullable=False)
    price = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    timeframe = Column(Enum(TimeframeType), nullable=False)
    expiry = Column(DateTime, nullable=True)
    meta = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    executed = Column(Boolean, default=False, nullable=False)
    executed_at = Column(DateTime, nullable=True)
    execution_price = Column(Float, nullable=True)
    
    # Relationships
    asset = relationship("Asset", back_populates="signals")
    strategy = relationship("Strategy", back_populates="signals")
    orders = relationship("Order", back_populates="signal")
    
    def __repr__(self):
        return f""

# Strategy model
class Strategy(Base, BaseMixin):
    """Trading strategy model for signal generators."""
    
    __tablename__ = "strategies"
    
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    strategy_type = Column(Enum(StrategyType), nullable=False)
    code = Column(Text, nullable=True)
    parameters = Column(JSON, nullable=True)
    creator_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_public = Column(Boolean, default=False, nullable=False)
    performance_metrics = Column(JSON, nullable=True)
    version = Column(String(20), default="1.0.0", nullable=False)
    
    # Relationships
    signals = relationship("Signal", back_populates="strategy")
    creator = relationship("User")
    
    def __repr__(self):
        return f""

# Order model
class Order(Base, BaseMixin):
    """Order model for exchange orders."""
    
    __tablename__ = "orders"
    
    exchange_account_id = Column(Integer, ForeignKey("exchange_accounts.id"), nullable=False)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    signal_id = Column(Integer, ForeignKey("signals.id"), nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    order_type = Column(Enum(OrderType), nullable=False)
    side = Column(Enum(SignalType), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)
    status = Column(Enum(OrderStatus), nullable=False)
    
    external_id = Column(String(100), nullable=True)
    filled_quantity = Column(Float, default=0.0, nullable=False)
    filled_price = Column(Float, nullable=True)
    filled_notional = Column(Float, nullable=True)
    
    fees = Column(Float, nullable=True)
    fee_asset = Column(String(10), nullable=True)
    
    is_test = Column(Boolean, default=False, nullable=False)
    meta = Column(JSON, nullable=True)
    
    executed_at = Column(DateTime, nullable=True)
    closed_at = Column(DateTime, nullable=True)
    
    # Relationships
    exchange_account = relationship("ExchangeAccount", back_populates="orders")
    asset = relationship("Asset", back_populates="orders")
    signal = relationship("Signal", back_populates="orders")
    user = relationship("User")
    
    def __repr__(self):
        return f""

# Brain council model
class BrainCouncil(Base, BaseMixin):
    """Brain council model for coordinating multiple strategies."""
    
    __tablename__ = "brain_councils"
    
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    timeframe = Column(Enum(TimeframeType), nullable=False)
    parameters = Column(JSON, nullable=True)
    min_consensus = Column(Float, default=0.7, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    performance_metrics = Column(JSON, nullable=True)
    
    # Relationships
    members = relationship("BrainCouncilMember", back_populates="council", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f""

# Brain council member model
class BrainCouncilMember(Base, BaseMixin):
    """Brain council member model for strategy weighting."""
    
    __tablename__ = "brain_council_members"
    
    council_id = Column(Integer, ForeignKey("brain_councils.id"), nullable=False)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    weight = Column(Float, default=1.0, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    performance_metrics = Column(JSON, nullable=True)
    
    # Relationships
    council = relationship("BrainCouncil", back_populates="members")
    strategy = relationship("Strategy")
    
    def __repr__(self):
        return f""

# Performance record model
class PerformanceRecord(Base, BaseMixin):
    """Performance record model for tracking strategy and council performance."""
    
    __tablename__ = "performance_records"
    
    entity_type = Column(String(50), nullable=False)  # 'strategy', 'council', 'system'
    entity_id = Column(Integer, nullable=False)
    timeframe = Column(Enum(TimeframeType), nullable=False)
    
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    
    win_count = Column(Integer, default=0, nullable=False)
    loss_count = Column(Integer, default=0, nullable=False)
    win_rate = Column(Float, nullable=True)
    
    profit_pct = Column(Float, nullable=True)
    profit_absolute = Column(Float, nullable=True)
    
    max_drawdown = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    
    meta = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f""

# ML Model record
class MLModel(Base, BaseMixin):
    """Machine learning model record."""
    
    __tablename__ = "ml_models"
    
    name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=True)
    version = Column(String(20), nullable=False)
    
    file_path = Column(String(255), nullable=False)
    input_features = Column(JSON, nullable=False)
    output_features = Column(JSON, nullable=False)
    
    training_start = Column(DateTime, nullable=False)
    training_end = Column(DateTime, nullable=False)
    
    training_accuracy = Column(Float, nullable=True)
    validation_accuracy = Column(Float, nullable=True)
    test_accuracy = Column(Float, nullable=True)
    
    is_active = Column(Boolean, default=True, nullable=False)
    meta = Column(JSON, nullable=True)
    
    # Relationships
    asset = relationship("Asset")
    
    def __repr__(self):
        return f""

# System log model
class SystemLog(Base, BaseMixin):
    """System log record for important events."""
    
    __tablename__ = "system_logs"
    
    level = Column(String(20), nullable=False)
    component = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f""

# Asset price record
class AssetPrice(Base, BaseMixin):
    """Asset price record for historical data."""
    
    __tablename__ = "asset_prices"
    
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    timeframe = Column(Enum(TimeframeType), nullable=False)
    
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Relationships
    asset = relationship("Asset")
    
    __table_args__ = (
        {'sqlite_autoincrement': True},
    )
    
    def __repr__(self):
        return f""

# News item model
class NewsItem(Base, BaseMixin):
    """News item record for news data."""
    
    __tablename__ = "news_items"
    
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    source = Column(String(100), nullable=False)
    url = Column(String(255), nullable=True)
    published_at = Column(DateTime, nullable=False)
    sentiment_score = Column(Float, nullable=True)
    entities = Column(JSON, nullable=True)
    keywords = Column(JSON, nullable=True)
    relevance = Column(Float, nullable=True)
    
    def __repr__(self):
        return f""

# Asset news relation
class AssetNews(Base, BaseMixin):
    """Association between assets and news items."""
    
    __tablename__ = "asset_news"
    
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=False)
    news_id = Column(Integer, ForeignKey("news_items.id"), nullable=False)
    relevance_score = Column(Float, nullable=True)
    
    # Relationships
    asset = relationship("Asset")
    news = relationship("NewsItem")
    
    __table_args__ = (
        {'sqlite_autoincrement': True},
    )
    
    def __repr__(self):
        return f""

# Trading pattern model
class TradingPattern(Base, BaseMixin):
    """Trading pattern record for pattern recognition."""
    
    __tablename__ = "trading_patterns"
    
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    asset_id = Column(Integer, ForeignKey("assets.id"), nullable=True)
    timeframe = Column(Enum(TimeframeType), nullable=False)
    
    pattern_data = Column(JSON, nullable=False)
    match_threshold = Column(Float, default=0.8, nullable=False)
    
    success_count = Column(Integer, default=0, nullable=False)
    failure_count = Column(Integer, default=0, nullable=False)
    success_rate = Column(Float, nullable=True)
    
    signal_type = Column(Enum(SignalType), nullable=False)
    average_profit = Column(Float, nullable=True)
    
    is_active = Column(Boolean, default=True, nullable=False)
    is_system = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    asset = relationship("Asset")
    
    def __repr__(self):
        return f""
