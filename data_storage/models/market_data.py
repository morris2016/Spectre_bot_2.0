#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Market Data Models

This module defines the database models for storing market data including
price history, order book snapshots, market events, and trading signals.
These models are optimized for high-performance time-series queries with
advanced indexing and partitioning strategies.
"""

import uuid
import json
import enum
import datetime
from typing import Dict, List, Optional, Union, Any, TypeVar, Generic, Tuple
from dataclasses import dataclass, field, asdict

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, JSON, Text,
    ForeignKey, Index, Table, Enum, UniqueConstraint, TIMESTAMP, LargeBinary
)
from sqlalchemy.dialects.postgresql import JSONB, UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func
from sqlalchemy.schema import MetaData

from common.utils import TimestampUtils, UuidUtils, HashUtils
from common.constants import (
    TIMEFRAMES, PLATFORMS, ASSET_CLASSES, ORDER_TYPES,
    ORDER_SIDES, SIGNAL_TYPES, SIGNAL_STRENGTHS
)
from common.exceptions import ModelValidationError

# Base metadata with naming conventions for constraints
db_metadata = MetaData(naming_convention={  # Renamed from metadata to avoid conflicts
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
})

Base = declarative_base(metadata=db_metadata)  # Use renamed metadata

# Enumeration types for market data
class MarketEventType(enum.Enum):
    """Types of market events to track."""
    PRICE_SPIKE = "price_spike"
    VOLUME_SURGE = "volume_surge"
    LIQUIDITY_CHANGE = "liquidity_change"
    NEWS_IMPACT = "news_impact"
    PATTERN_FORMATION = "pattern_formation"
    SUPPORT_RESISTANCE_TEST = "support_resistance_test"
    REGIME_CHANGE = "regime_change"
    CORRELATION_SHIFT = "correlation_shift"
    VOLATILITY_CHANGE = "volatility_change"
    MARKET_STRUCTURE_CHANGE = "market_structure_change"
    DARK_POOL_ACTIVITY = "dark_pool_activity"
    INSIDER_ACTIVITY = "insider_activity"
    MANIPULATION_DETECTED = "manipulation_detected"
    SENTIMENT_SHIFT = "sentiment_shift"
    WHALE_MOVEMENT = "whale_movement"
    SMART_MONEY_FLOW = "smart_money_flow"
    OPTIONS_ACTIVITY = "options_activity"
    FUNDING_RATE_CHANGE = "funding_rate_change"
    LIQUIDATION_CASCADE = "liquidation_cascade"
    TECHNICAL_BREAKOUT = "technical_breakout"

class DataSource(enum.Enum):
    """Sources of market data."""
    EXCHANGE_API = "exchange_api"
    WEBSOCKET_FEED = "websocket_feed"
    NEWS_FEED = "news_feed"
    SOCIAL_FEED = "social_feed"
    ONCHAIN_DATA = "onchain_data"
    DARK_WEB = "dark_web"
    CALCULATED = "calculated"
    USER_INPUT = "user_input"
    BACKTESTING = "backtesting"
    SIMULATION = "simulation"
    THIRD_PARTY_API = "third_party_api"
    HISTORICAL_DATA = "historical_data"
    
class PatternType(enum.Enum):
    """Types of patterns detected in market data."""
    HARMONIC = "harmonic"
    CANDLESTICK = "candlestick"
    CHART = "chart"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    ORDER_FLOW = "order_flow"
    SUPPORT_RESISTANCE = "support_resistance"
    MARKET_STRUCTURE = "market_structure"
    CORRELATION = "correlation"
    DIVERGENCE = "divergence"
    SENTIMENT = "sentiment"
    FIBONACCI = "fibonacci"
    ELLIOT_WAVE = "elliot_wave"
    WYCKOFF = "wyckoff"
    SUPPLY_DEMAND = "supply_demand"
    VWAP = "vwap"
    FOOTPRINT = "footprint"
    LIQUIDITY = "liquidity"
    SMART_MONEY = "smart_money"

# Base classes and mixins
class TimestampMixin:
    """Mixin for adding timestamp fields to models."""
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), 
                        onupdate=func.now(), nullable=False)

class Auditable(TimestampMixin):
    """Mixin for adding audit fields to models."""
    created_by = Column(String(50), nullable=True)
    updated_by = Column(String(50), nullable=True)
    version = Column(Integer, default=1, nullable=False)
    is_deleted = Column(Boolean, default=False, nullable=False)
    
    def update_version(self):
        """Increment version on update."""
        self.version += 1
        
class SoftDeleteMixin:
    """Mixin for soft delete functionality."""
    deleted_at = Column(TIMESTAMP(timezone=True), nullable=True)
    deleted_by = Column(String(50), nullable=True)
    
    def soft_delete(self, user_id=None):
        """Mark record as deleted."""
        self.deleted_at = func.now()
        self.deleted_by = user_id
        self.is_deleted = True

# Association tables
asset_tag_association = Table(
    'asset_tag_association',
    db_metadata,  # Use renamed metadata
    Column('asset_id', UUID(as_uuid=True), ForeignKey('assets.id'), primary_key=True),
    Column('tag_id', UUID(as_uuid=True), ForeignKey('tags.id'), primary_key=True)
)

# Market data models
class Asset(Base, Auditable, SoftDeleteMixin):
    """Asset model for representing tradable assets."""
    __tablename__ = 'assets'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(30), nullable=False)
    name = Column(String(100), nullable=True)
    platform = Column(Enum(*PLATFORMS, name='platform_enum'), nullable=False)
    asset_class = Column(Enum(*ASSET_CLASSES, name='asset_class_enum'), nullable=False)
    base_currency = Column(String(20), nullable=True)
    quote_currency = Column(String(20), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    meta_data = Column(JSONB, nullable=True)  # Renamed from metadata to avoid SQLAlchemy conflict
    
    # Relationships
    price_data = relationship("PriceData", back_populates="asset", cascade="all, delete-orphan")
    market_events = relationship("MarketEvent", back_populates="asset")
    order_book_snapshots = relationship("OrderBookSnapshot", back_populates="asset")
    signals = relationship("TradingSignal", back_populates="asset")
    tags = relationship("Tag", secondary=asset_tag_association, back_populates="assets")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('symbol', 'platform', name='uq_asset_symbol_platform'),
        Index('ix_assets_symbol', 'symbol'),
        Index('ix_assets_platform', 'platform'),
        Index('ix_assets_asset_class', 'asset_class'),
        Index('ix_assets_is_active', 'is_active'),
    )
    
    def __repr__(self):
        return f""
    
    @property
    def full_symbol(self):
        """Get fully qualified symbol with platform."""
        return f"{self.platform}:{self.symbol}"
    
    @classmethod
    def get_or_create(cls, session, symbol, platform, **kwargs):
        """Get existing asset or create a new one."""
        asset = session.query(cls).filter_by(
            symbol=symbol, platform=platform
        ).first()
        
        if not asset:
            asset = cls(symbol=symbol, platform=platform, **kwargs)
            session.add(asset)
            session.flush()
        
        return asset

class Tag(Base, TimestampMixin):
    """Tag model for categorizing assets."""
    __tablename__ = 'tags'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(50), nullable=False, unique=True)
    description = Column(String(255), nullable=True)
    
    # Relationships
    assets = relationship("Asset", secondary=asset_tag_association, back_populates="tags")
    
    def __repr__(self):
        return f""

class PriceData(Base, TimestampMixin):
    """Model for storing OHLCV price data for assets."""
    __tablename__ = 'price_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id = Column(UUID(as_uuid=True), ForeignKey('assets.id'), nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    timeframe = Column(Enum(*TIMEFRAMES, name='timeframe_enum'), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    num_trades = Column(Integer, nullable=True)
    vwap = Column(Float, nullable=True)
    buy_volume = Column(Float, nullable=True)
    sell_volume = Column(Float, nullable=True)
    source = Column(Enum(DataSource), nullable=False, default=DataSource.EXCHANGE_API)
    is_complete = Column(Boolean, default=True, nullable=False)
    meta_data = Column(JSONB, nullable=True)  # Renamed from metadata to avoid SQLAlchemy conflict
    
    # Relationships
    asset = relationship("Asset", back_populates="price_data")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('asset_id', 'timestamp', 'timeframe', name='uq_price_data_asset_time_tf'),
        Index('ix_price_data_asset_id', 'asset_id'),
        Index('ix_price_data_timestamp', 'timestamp'),
        Index('ix_price_data_timeframe', 'timeframe'),
        # Composite index for range queries
        Index('ix_price_data_asset_tf_time', 'asset_id', 'timeframe', 'timestamp'),
    )
    
    def __repr__(self):
        return (f"")
    
    @property
    def body_size(self):
        """Calculate the size of the candle body."""
        return abs(self.close - self.open)
    
    @property
    def range_size(self):
        """Calculate the total range of the candle."""
        return self.high - self.low
    
    @property
    def is_bullish(self):
        """Determine if the candle is bullish."""
        return self.close > self.open
    
    @property
    def upper_shadow(self):
        """Calculate the size of the upper shadow."""
        return self.high - (self.close if self.is_bullish else self.open)
    
    @property
    def lower_shadow(self):
        """Calculate the size of the lower shadow."""
        return (self.open if self.is_bullish else self.close) - self.low
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'id': str(self.id),
            'asset_id': str(self.asset_id),
            'timestamp': self.timestamp.isoformat(),
            'timeframe': self.timeframe,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'num_trades': self.num_trades,
            'vwap': self.vwap,
            'buy_volume': self.buy_volume,
            'sell_volume': self.sell_volume,
            'is_complete': self.is_complete
        }
    
    @classmethod
    def from_dict(cls, data_dict, session=None):
        """Create a PriceData instance from a dictionary."""
        if 'asset_id' not in data_dict and 'asset_symbol' in data_dict and 'platform' in data_dict:
            if not session:
                raise ValueError("Session required to look up asset_id")
            
            asset = session.query(Asset).filter_by(
                symbol=data_dict['asset_symbol'],
                platform=data_dict['platform']
            ).first()
            
            if not asset:
                raise ValueError(f"Asset not found: {data_dict['asset_symbol']} on {data_dict['platform']}")
            
            data_dict['asset_id'] = asset.id
        
        # Convert timestamp if it's a string
        if isinstance(data_dict.get('timestamp'), str):
            data_dict['timestamp'] = datetime.datetime.fromisoformat(
                data_dict['timestamp'].replace('Z', '+00:00')
            )
        
        return cls(**data_dict)

class OrderBookSnapshot(Base, TimestampMixin):
    """Model for storing order book snapshots."""
    __tablename__ = 'order_book_snapshots'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id = Column(UUID(as_uuid=True), ForeignKey('assets.id'), nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    bids = Column(JSONB, nullable=False)  # List of [price, quantity] pairs
    asks = Column(JSONB, nullable=False)  # List of [price, quantity] pairs
    update_id = Column(String(50), nullable=True)  # Exchange-specific update ID
    source = Column(Enum(DataSource), nullable=False, default=DataSource.EXCHANGE_API)
    depth = Column(Integer, nullable=True)  # Depth of the order book
    is_complete = Column(Boolean, default=True, nullable=False)
    meta_data = Column(JSONB, nullable=True)  # Renamed from metadata to avoid SQLAlchemy conflict
    
    # Relationships
    asset = relationship("Asset", back_populates="order_book_snapshots")
    
    # Constraints
    __table_args__ = (
        Index('ix_order_book_asset_id', 'asset_id'),
        Index('ix_order_book_timestamp', 'timestamp'),
        Index('ix_order_book_asset_time', 'asset_id', 'timestamp'),
    )
    
    def __repr__(self):
        return (f"")
    
    @property
    def spread(self):
        """Calculate the bid-ask spread."""
        if not self.bids or not self.asks:
            return None
        
        best_bid = self.bids[0][0] if isinstance(self.bids, list) else max(self.bids.keys())
        best_ask = self.asks[0][0] if isinstance(self.asks, list) else min(self.asks.keys())
        
        return best_ask - best_bid
    
    @property
    def midpoint(self):
        """Calculate the midpoint price."""
        if not self.bids or not self.asks:
            return None
        
        best_bid = self.bids[0][0] if isinstance(self.bids, list) else max(self.bids.keys())
        best_ask = self.asks[0][0] if isinstance(self.asks, list) else min(self.asks.keys())
        
        return (best_bid + best_ask) / 2
    
    @property
    def imbalance(self):
        """Calculate order book imbalance."""
        if not self.bids or not self.asks:
            return None
        
        bid_volume = sum(q for _, q in self.bids) if isinstance(self.bids, list) else sum(self.bids.values())
        ask_volume = sum(q for _, q in self.asks) if isinstance(self.asks, list) else sum(self.asks.values())
        
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0
        
        return (bid_volume - ask_volume) / total_volume

class MarketEvent(Base, TimestampMixin):
    """Model for storing significant market events."""
    __tablename__ = 'market_events'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id = Column(UUID(as_uuid=True), ForeignKey('assets.id'), nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    event_type = Column(Enum(MarketEventType), nullable=False)
    severity = Column(Integer, nullable=False)  # 1-10 scale
    description = Column(Text, nullable=True)
    details = Column(JSONB, nullable=True)
    source = Column(Enum(DataSource), nullable=False)
    confidence = Column(Float, nullable=False)  # 0-1 scale
    duration = Column(Integer, nullable=True)  # Duration in seconds
    is_active = Column(Boolean, default=True, nullable=False)
    meta_data = Column(JSONB, nullable=True)  # Renamed from metadata to avoid SQLAlchemy conflict
    
    # Relationships
    asset = relationship("Asset", back_populates="market_events")
    
    # Constraints
    __table_args__ = (
        Index('ix_market_events_asset_id', 'asset_id'),
        Index('ix_market_events_timestamp', 'timestamp'),
        Index('ix_market_events_event_type', 'event_type'),
        Index('ix_market_events_severity', 'severity'),
        Index('ix_market_events_is_active', 'is_active'),
        Index('ix_market_events_asset_time', 'asset_id', 'timestamp'),
    )
    
    def __repr__(self):
        return (f"")

class TradingSignal(Base, TimestampMixin):
    """Model for storing trading signals generated by the system."""
    __tablename__ = 'trading_signals'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id = Column(UUID(as_uuid=True), ForeignKey('assets.id'), nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    signal_type = Column(Enum(*SIGNAL_TYPES, name='signal_type_enum'), nullable=False)
    direction = Column(Enum(*ORDER_SIDES, name='direction_enum'), nullable=False)
    timeframe = Column(Enum(*TIMEFRAMES, name='signal_timeframe_enum'), nullable=False)
    strength = Column(Enum(*SIGNAL_STRENGTHS, name='signal_strength_enum'), nullable=False)
    price = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    risk_reward_ratio = Column(Float, nullable=True)
    confidence = Column(Float, nullable=False)  # 0-1 scale
    expiration = Column(TIMESTAMP(timezone=True), nullable=True)
    entry_type = Column(String(50), nullable=True)  # Market, Limit, etc.
    rationale = Column(Text, nullable=True)
    source_strategy = Column(String(100), nullable=True)
    meta_data = Column(JSONB, nullable=True)  # Renamed from metadata to avoid SQLAlchemy conflict
    
    # Execution status
    is_executed = Column(Boolean, default=False, nullable=False)
    executed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    execution_price = Column(Float, nullable=True)
    position_id = Column(UUID(as_uuid=True), nullable=True)
    success = Column(Boolean, nullable=True)  # Was the signal profitable
    pnl = Column(Float, nullable=True)  # Profit/loss if executed
    
    # Relationships
    asset = relationship("Asset", back_populates="signals")
    
    # Constraints
    __table_args__ = (
        Index('ix_trading_signals_asset_id', 'asset_id'),
        Index('ix_trading_signals_timestamp', 'timestamp'),
        Index('ix_trading_signals_signal_type', 'signal_type'),
        Index('ix_trading_signals_direction', 'direction'),
        Index('ix_trading_signals_timeframe', 'timeframe'),
        Index('ix_trading_signals_strength', 'strength'),
        Index('ix_trading_signals_is_executed', 'is_executed'),
        Index('ix_trading_signals_confidence', 'confidence'),
        Index('ix_trading_signals_success', 'success'),
        Index('ix_trading_signals_asset_time', 'asset_id', 'timestamp'),
    )
    
    def __repr__(self):
        return (f"")
    
    @property
    def age(self):
        """Calculate age of signal in seconds."""
        now = datetime.datetime.now(datetime.timezone.utc)
        return (now - self.timestamp).total_seconds()
    
    @property
    def is_expired(self):
        """Check if signal has expired."""
        if not self.expiration:
            return False
        
        now = datetime.datetime.now(datetime.timezone.utc)
        return now > self.expiration
    
    @property
    def time_to_expiration(self):
        """Calculate time until expiration in seconds."""
        if not self.expiration:
            return None
        
        now = datetime.datetime.now(datetime.timezone.utc)
        return max(0, (self.expiration - now).total_seconds())
    
    def mark_executed(self, execution_price, position_id=None):
        """Mark signal as executed."""
        self.is_executed = True
        self.executed_at = datetime.datetime.now(datetime.timezone.utc)
        self.execution_price = execution_price
        self.position_id = position_id
    
    def record_result(self, exit_price, exit_timestamp=None):
        """Record the result of the signal."""
        if not self.is_executed:
            raise ValueError("Cannot record result for non-executed signal")
        
        if not exit_timestamp:
            exit_timestamp = datetime.datetime.now(datetime.timezone.utc)
        
        # Calculate P&L
        if self.direction == 'BUY':
            self.pnl = (exit_price - self.execution_price) / self.execution_price
        else:
            self.pnl = (self.execution_price - exit_price) / self.execution_price
        
        # Determine success
        self.success = self.pnl > 0
        
        return self.pnl, self.success
        
@dataclass
class OHLCVData:
    """Data class for OHLCV (Open, High, Low, Close, Volume) market data."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str
    asset_id: str = None
    source: str = "exchange"
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'timeframe': self.timeframe,
            'asset_id': self.asset_id,
            'source': self.source,
            'additional_data': self.additional_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OHLCVData':
        """Create instance from dictionary."""
        return cls(**data)

class Pattern(Base, TimestampMixin):
    """Model for storing detected market patterns."""
    __tablename__ = 'patterns'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id = Column(UUID(as_uuid=True), ForeignKey('assets.id'), nullable=False)
    start_timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    end_timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    timeframe = Column(Enum(*TIMEFRAMES, name='pattern_timeframe_enum'), nullable=False)
    pattern_type = Column(Enum(PatternType), nullable=False)
    pattern_name = Column(String(100), nullable=False)
    direction = Column(Enum(*ORDER_SIDES, name='pattern_direction_enum'), nullable=True)
    strength = Column(Float, nullable=False)  # 0-1 scale
    completion = Column(Float, nullable=False)  # 0-1 scale
    target_price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)
    probability = Column(Float, nullable=False)  # 0-1 scale
    description = Column(Text, nullable=True)
    points = Column(JSONB, nullable=True)  # Key points in the pattern
    meta_data = Column(JSONB, nullable=True)  # Renamed from metadata to avoid SQLAlchemy conflict
    
    # Validation status
    is_validated = Column(Boolean, default=False, nullable=False)
    validated_at = Column(TIMESTAMP(timezone=True), nullable=True)
    validation_price = Column(Float, nullable=True)
    success = Column(Boolean, nullable=True)  # Did pattern play out as expected
    
    # Constraints
    __table_args__ = (
        Index('ix_patterns_asset_id', 'asset_id'),
        Index('ix_patterns_start_timestamp', 'start_timestamp'),
        Index('ix_patterns_end_timestamp', 'end_timestamp'),
        Index('ix_patterns_timeframe', 'timeframe'),
        Index('ix_patterns_pattern_type', 'pattern_type'),
        Index('ix_patterns_pattern_name', 'pattern_name'),
        Index('ix_patterns_direction', 'direction'),
        Index('ix_patterns_strength', 'strength'),
        Index('ix_patterns_probability', 'probability'),
        Index('ix_patterns_is_validated', 'is_validated'),
        Index('ix_patterns_success', 'success'),
        Index('ix_patterns_asset_start_end', 'asset_id', 'start_timestamp', 'end_timestamp'),
    )
    
    def __repr__(self):
        return (f"")
    
    @property
    def duration(self):
        """Calculate pattern duration in seconds."""
        return (self.end_timestamp - self.start_timestamp).total_seconds()
    
    def mark_validated(self, validation_price, success):
        """Mark pattern as validated with outcome."""
        self.is_validated = True
        self.validated_at = datetime.datetime.now(datetime.timezone.utc)
        self.validation_price = validation_price
        self.success = success
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'id': str(self.id),
            'asset_id': str(self.asset_id),
            'start_timestamp': self.start_timestamp.isoformat(),
            'end_timestamp': self.end_timestamp.isoformat(),
            'timeframe': self.timeframe,
            'pattern_type': self.pattern_type.value,
            'pattern_name': self.pattern_name,
            'direction': self.direction,
            'strength': self.strength,
            'completion': self.completion,
            'target_price': self.target_price,
            'stop_price': self.stop_price,
            'probability': self.probability,
            'description': self.description,
            'points': self.points,
            'is_validated': self.is_validated,
            'success': self.success
        }

class MarketRegime(Base, TimestampMixin):
    """Model for storing market regime classifications."""
    __tablename__ = 'market_regimes'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id = Column(UUID(as_uuid=True), ForeignKey('assets.id'), nullable=False)
    start_timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    end_timestamp = Column(TIMESTAMP(timezone=True), nullable=True)  # NULL for ongoing
    regime_type = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)  # 0-1 scale
    volatility = Column(Float, nullable=True)
    trend_strength = Column(Float, nullable=True)
    description = Column(Text, nullable=True)
    meta_data = Column(JSONB, nullable=True)  # Renamed from metadata to avoid SQLAlchemy conflict
    
    # Constraints
    __table_args__ = (
        Index('ix_market_regimes_asset_id', 'asset_id'),
        Index('ix_market_regimes_start_timestamp', 'start_timestamp'),
        Index('ix_market_regimes_end_timestamp', 'end_timestamp'),
        Index('ix_market_regimes_regime_type', 'regime_type'),
        Index('ix_market_regimes_asset_start', 'asset_id', 'start_timestamp'),
    )
    
    def __repr__(self):
        return (f"")
    
    @property
    def is_active(self):
        """Check if regime is currently active."""
        return self.end_timestamp is None
    
    @property
    def duration(self):
        """Calculate regime duration in seconds."""
        if not self.end_timestamp:
            now = datetime.datetime.now(datetime.timezone.utc)
            return (now - self.start_timestamp).total_seconds()
        
        return (self.end_timestamp - self.start_timestamp).total_seconds()
    
    def close(self, end_timestamp=None):
        """Close the regime period."""
        if not end_timestamp:
            end_timestamp = datetime.datetime.now(datetime.timezone.utc)
        
        self.end_timestamp = end_timestamp

class MarketStructure(Base, TimestampMixin):
    """Model for storing market structure analysis."""
    __tablename__ = 'market_structures'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id = Column(UUID(as_uuid=True), ForeignKey('assets.id'), nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    timeframe = Column(Enum(*TIMEFRAMES, name='structure_timeframe_enum'), nullable=False)
    
    # Market structure features
    trend_direction = Column(String(20), nullable=True)  # UP, DOWN, SIDEWAYS
    structure_type = Column(String(50), nullable=True)  # TREND, RANGE, REVERSAL, etc.
    swing_points = Column(JSONB, nullable=True)  # Highs and lows with timestamps
    key_levels = Column(JSONB, nullable=True)  # Support/resistance levels
    fibonacci_levels = Column(JSONB, nullable=True)
    recent_volume_profile = Column(JSONB, nullable=True)
    liquidity_zones = Column(JSONB, nullable=True)
    fair_value_gap = Column(JSONB, nullable=True)
    order_blocks = Column(JSONB, nullable=True)
    
    # Analysis
    narrative = Column(Text, nullable=True)  # Textual description of structure
    confidence = Column(Float, nullable=False)  # 0-1 scale
    meta_data = Column(JSONB, nullable=True)  # Renamed from metadata to avoid SQLAlchemy conflict
    
    # Constraints
    __table_args__ = (
        Index('ix_market_structures_asset_id', 'asset_id'),
        Index('ix_market_structures_timestamp', 'timestamp'),
        Index('ix_market_structures_timeframe', 'timeframe'),
        Index('ix_market_structures_trend_direction', 'trend_direction'),
        Index('ix_market_structures_structure_type', 'structure_type'),
        Index('ix_market_structures_asset_time_tf', 'asset_id', 'timestamp', 'timeframe'),
    )
    
    def __repr__(self):
        return (f"")

# Create specialized views and materialized views
def create_views(engine):
    """Create database views for frequently accessed data patterns."""
    from sqlalchemy import text
    
    # Recent price data view
    recent_price_view = text("""
    CREATE MATERIALIZED VIEW IF NOT EXISTS recent_price_data AS
    SELECT pd.*
    FROM price_data pd
    JOIN (
        SELECT asset_id, timeframe, MAX(timestamp) as max_timestamp
        FROM price_data
        WHERE timestamp >= NOW() - INTERVAL '7 days'
        GROUP BY asset_id, timeframe
    ) recent ON pd.asset_id = recent.asset_id 
             AND pd.timeframe = recent.timeframe 
             AND pd.timestamp >= recent.max_timestamp - INTERVAL '24 hours'
    ORDER BY pd.asset_id, pd.timeframe, pd.timestamp;
    
    CREATE UNIQUE INDEX IF NOT EXISTS ix_recent_price_data_pk 
    ON recent_price_data (id);
    
    CREATE INDEX IF NOT EXISTS ix_recent_price_data_asset_tf_time 
    ON recent_price_data (asset_id, timeframe, timestamp);
    """)
    
    # Active signals view
    active_signals_view = text("""
    CREATE MATERIALIZED VIEW IF NOT EXISTS active_trading_signals AS
    SELECT ts.*
    FROM trading_signals ts
    WHERE (ts.expiration IS NULL OR ts.expiration > NOW())
      AND ts.is_executed = FALSE
      AND ts.timestamp >= NOW() - INTERVAL '1 day'
    ORDER BY ts.timestamp DESC;
    
    CREATE UNIQUE INDEX IF NOT EXISTS ix_active_trading_signals_pk 
    ON active_trading_signals (id);
    
    CREATE INDEX IF NOT EXISTS ix_active_trading_signals_asset_conf 
    ON active_trading_signals (asset_id, confidence DESC);
    """)
    
    # Recent patterns view
    recent_patterns_view = text("""
    CREATE MATERIALIZED VIEW IF NOT EXISTS recent_patterns AS
    SELECT p.*
    FROM patterns p
    WHERE p.end_timestamp >= NOW() - INTERVAL '7 days'
      AND p.completion >= 0.8
    ORDER BY p.end_timestamp DESC;
    
    CREATE UNIQUE INDEX IF NOT EXISTS ix_recent_patterns_pk 
    ON recent_patterns (id);
    
    CREATE INDEX IF NOT EXISTS ix_recent_patterns_asset_prob
    ON recent_patterns (asset_id, probability DESC);
    """)
    
    # Execute view creation
    with engine.begin() as conn:
        conn.execute(recent_price_view)
        conn.execute(active_signals_view)
        conn.execute(recent_patterns_view)
        
        # Create refresh function
        conn.execute(text("""
        CREATE OR REPLACE FUNCTION refresh_materialized_views()
        RETURNS VOID
        LANGUAGE plpgsql
        AS $$
        BEGIN
            REFRESH MATERIALIZED VIEW CONCURRENTLY recent_price_data;
            REFRESH MATERIALIZED VIEW CONCURRENTLY active_trading_signals;
            REFRESH MATERIALIZED VIEW CONCURRENTLY recent_patterns;
        END;
        $$;
        """))
    
    return True

# Function to ensure all required tables exist
def initialize_tables(engine, create_views_flag=True):
    """Initialize all market data tables."""
    Base.metadata.create_all(engine)
    
    if create_views_flag:
        create_views(engine)
    
    return True

@dataclass
class TradeData:
    """Data class for individual trade data."""
    id: str
    asset_id: str
    timestamp: int
    price: float
    amount: float
    side: str  # 'buy' or 'sell'
    trade_id: str = None
    maker: bool = False
    taker: bool = True
    fee: float = 0.0
    fee_currency: str = None
    order_id: str = None
    source: str = "exchange"
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'asset_id': self.asset_id,
            'timestamp': self.timestamp,
            'price': self.price,
            'amount': self.amount,
            'side': self.side,
            'trade_id': self.trade_id,
            'maker': self.maker,
            'taker': self.taker,
            'fee': self.fee,
            'fee_currency': self.fee_currency,
            'order_id': self.order_id,
            'source': self.source,
            'additional_data': self.additional_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeData':
        """Create instance from dictionary."""
        return cls(**data)
    
    @property
    def value(self) -> float:
        """Calculate the total value of the trade."""
        return self.price * self.amount
    
    @property
    def net_value(self) -> float:
        """Calculate the net value after fees."""
        return self.value - self.fee

@dataclass
class SentimentData:
    """Data class for market sentiment information."""
    asset_id: str
    timestamp: int
    timeframe: str
    
    # Sentiment scores
    sentiment_score: float = 0.0  # -1.0 to 1.0, negative to positive
    sentiment_magnitude: float = 0.0  # 0.0 to 1.0, strength of sentiment
    
    # Source breakdown
    social_sentiment: float = 0.0
    news_sentiment: float = 0.0
    analyst_sentiment: float = 0.0
    
    # Volume and momentum indicators
    volume_sentiment: float = 0.0
    momentum_sentiment: float = 0.0
    
    # Sentiment trends
    sentiment_change_1h: float = 0.0
    sentiment_change_24h: float = 0.0
    sentiment_change_7d: float = 0.0
    
    # Additional metrics
    bullish_signals: int = 0
    bearish_signals: int = 0
    neutral_signals: int = 0
    
    # Source data
    sources: List[Dict[str, Any]] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SentimentData':
        """Create instance from dictionary."""
        return cls(**data)
    
    @property
    def is_bullish(self) -> bool:
        """Check if sentiment is bullish."""
        return self.sentiment_score > 0.3
        
    @property
    def is_bearish(self) -> bool:
        """Check if sentiment is bearish."""
        return self.sentiment_score < -0.3
        
    @property
    def is_neutral(self) -> bool:
        """Check if sentiment is neutral."""
        return -0.3 <= self.sentiment_score <= 0.3

@dataclass
class PatternOccurrence:
    """Data class for storing pattern occurrences."""
    pattern_id: str
    pattern_type: str
    direction: str
    asset: str
    timeframe: str
    completion_timestamp: int
    quality_score: float
    success: bool = False
    profit_percentage: float = 0.0
    
    def save(self):
        """Save pattern occurrence to database."""
        # Implementation would connect to database and save
        pass
    
    @classmethod
    def get_or_create(cls, **kwargs):
        """Get existing pattern metrics or create new ones."""
        # Implementation would fetch from database or create new
        return cls(**kwargs)

@dataclass
class PatternMetrics:
    """Data class for storing pattern performance metrics."""
    pattern_type: str
    direction: str
    asset: str
    timeframe: str
    success_count: int = 0
    failure_count: int = 0
    total_profit: float = 0.0
    avg_profit: float = 0.0
    success_rate: float = 0.0
    
    def save(self):
        """Save pattern metrics to database."""
        # Implementation would connect to database and save
        pass
    
    @classmethod
    def get_or_create(cls, **kwargs):
        """Get existing pattern metrics or create new ones."""
        # Implementation would fetch from database or create new
        return cls(**kwargs)
    
    @classmethod
    def get_aggregated(cls, pattern_type=None):
        """Get aggregated metrics for a pattern type."""
        # Implementation would aggregate metrics from database
        return {
            "success_rate": 0.0,
            "avg_profit": 0.0,
            "total_count": 0
        }

@dataclass
class LiquidityData:
    """Data class for market liquidity information."""
    asset_id: str
    timestamp: int
    timeframe: str
    
    # Order book data
    bids: List[List[float]] = field(default_factory=list)  # List of [price, volume] pairs
    asks: List[List[float]] = field(default_factory=list)  # List of [price, volume] pairs
    
    # Liquidity metrics
    bid_ask_spread: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    mid_price: float = 0.0
    weighted_mid_price: float = 0.0
    
    # Imbalance metrics
    book_imbalance: float = 0.0  # -1.0 to 1.0, negative means more asks
    pressure_imbalance: float = 0.0  # -1.0 to 1.0, negative means selling pressure
    
    # Market impact estimates
    estimated_impact_buy: Dict[float, float] = field(default_factory=dict)  # size -> price impact
    estimated_impact_sell: Dict[float, float] = field(default_factory=dict)  # size -> price impact
    
    # Liquidity zones
    support_zones: List[Dict[str, float]] = field(default_factory=list)
    resistance_zones: List[Dict[str, float]] = field(default_factory=list)
    
    # Additional metrics
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LiquidityData':
        """Create instance from dictionary."""
        return cls(**data)
    
    @property
    def is_liquid(self) -> bool:
        """Check if market is liquid based on spread and depth."""
        # Simple heuristic - can be customized based on asset type
        return (self.bid_ask_spread < 0.005 and  # Less than 0.5% spread
                self.bid_depth > 100000 and      # At least $100k depth on bid
                self.ask_depth > 100000)         # At least $100k depth on ask
    
    def get_optimal_execution_price(self, size: float, side: str) -> float:
        """
        Calculate optimal execution price for a given size and side.
        
        Args:
            size: Size to execute
            side: 'buy' or 'sell'
            
        Returns:
            Estimated execution price
        """
        if side.lower() == 'buy':
            if size in self.estimated_impact_buy:
                return self.mid_price * (1 + self.estimated_impact_buy[size])
            # Interpolate or extrapolate if exact size not available
            return self.mid_price * (1 + self.bid_ask_spread * 0.5 + (size / 10000) * 0.001)
        else:  # sell
            if size in self.estimated_impact_sell:
                return self.mid_price * (1 - self.estimated_impact_sell[size])
            # Interpolate or extrapolate if exact size not available
            return self.mid_price * (1 - self.bid_ask_spread * 0.5 - (size / 10000) * 0.001)

@dataclass
class TechnicalIndicators:
    """Data class for technical indicators."""
    asset_id: str
    timestamp: int
    timeframe: str
    
    # Trend indicators
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    
    # Momentum indicators
    rsi_14: float = 0.0
    stoch_k_14: float = 0.0
    stoch_d_14: float = 0.0
    cci_20: float = 0.0
    
    # Volatility indicators
    bollinger_upper: float = 0.0
    bollinger_middle: float = 0.0
    bollinger_lower: float = 0.0
    atr_14: float = 0.0
    
    # Volume indicators
    obv: float = 0.0
    vwap: float = 0.0
    
    # Additional indicators
    ichimoku_tenkan: float = 0.0
    ichimoku_kijun: float = 0.0
    ichimoku_senkou_a: float = 0.0
    ichimoku_senkou_b: float = 0.0
    
    # Custom indicators
    custom_indicators: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TechnicalIndicators':
        """Create instance from dictionary."""
        return cls(**data)
    
    @property
    def is_golden_cross(self) -> bool:
        """Check if there's a golden cross (SMA 50 crosses above SMA 200)."""
        return self.sma_50 > self.sma_200 and self.sma_50 - self.sma_200 < self.sma_50 * 0.01
    
    @property
    def is_death_cross(self) -> bool:
        """Check if there's a death cross (SMA 50 crosses below SMA 200)."""
        return self.sma_50 < self.sma_200 and self.sma_200 - self.sma_50 < self.sma_200 * 0.01
    
    @property
    def is_overbought(self) -> bool:
        """Check if the asset is overbought based on RSI."""
        return self.rsi_14 > 70
    
    @property
    def is_oversold(self) -> bool:
        """Check if the asset is oversold based on RSI."""
        return self.rsi_14 < 30

@dataclass
class MarketMetrics:
    """Data class for market metrics and statistics."""
    asset_id: str
    timestamp: int
    timeframe: str
    
    # Price metrics
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Volatility metrics
    volatility: float = 0.0
    atr: float = 0.0
    
    # Trend metrics
    trend_strength: float = 0.0
    trend_direction: str = "neutral"
    
    # Volume metrics
    relative_volume: float = 1.0
    volume_profile: Dict[str, Any] = field(default_factory=dict)
    
    # Market structure
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)
    
    # Liquidity metrics
    liquidity_score: float = 0.0
    bid_ask_spread: float = 0.0
    
    # Sentiment and correlation
    sentiment_score: float = 0.0
    correlation_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Additional metrics
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketMetrics':
        """Create instance from dictionary."""
        return cls(**data)
    
    @property
    def is_bullish(self) -> bool:
        """Check if market is bullish based on metrics."""
        return (self.trend_direction == "bullish" and
                self.trend_strength > 0.5 and
                self.sentiment_score > 0.0)
    
    @property
    def is_bearish(self) -> bool:
        """Check if market is bearish based on metrics."""
        return (self.trend_direction == "bearish" and
                self.trend_strength > 0.5 and
                self.sentiment_score < 0.0)


@dataclass
class SignalRecord:
    """Simple record of generated trading signals."""

    symbol: str
    timeframe: str
    action: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    reasoning: str
    timestamp: datetime.datetime
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignalRecord":
        """Create instance from dictionary."""
        return cls(**data)

    def save(self) -> None:
        """Persist record to storage (placeholder)."""
        pass

    @classmethod
    def create(cls, **kwargs) -> "SignalRecord":
        """Instantiate and store a new record."""
        record = cls(**kwargs)
        record.save()
        return record


@dataclass
class MarketRegimeData:
    """Data about detected market regimes."""

    asset_id: str
    platform: str
    timeframe: str
    regime: str
    confidence: float
    detected_at: datetime.datetime
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketRegimeData":
        """Create instance from dictionary."""
        return cls(**data)

    async def save(self) -> None:
        """Persist regime data asynchronously (placeholder)."""
        return None

    @classmethod
    async def get_history(cls, **query) -> List["MarketRegimeData"]:
        """Retrieve historical regime data (placeholder)."""
        return []


@dataclass
class AssetCharacteristics:
    """Summary of asset-specific behaviour and statistics."""

    asset_id: str
    platform: str
    volatility_profile: Dict[str, Any]
    liquidity_profile: Dict[str, Any]
    correlation_data: Dict[str, Any]
    seasonal_patterns: Dict[str, Any]
    behavioral_traits: Dict[str, Any]
    created_at: datetime.datetime
    updated_at: datetime.datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssetCharacteristics":
        """Create instance from dictionary."""
        return cls(**data)

    async def save(self) -> None:
        """Persist characteristics asynchronously (placeholder)."""
        return None

    @classmethod
    async def get_by_asset_id(cls, asset_id: str, platform: str) -> Optional["AssetCharacteristics"]:
        """Retrieve characteristics for an asset (placeholder)."""
        return None
