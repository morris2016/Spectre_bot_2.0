#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
User Data Models

This module defines the database models for storing user data, configurations,
preferences, trading history, and performance metrics. These models support
the user-facing aspects of the system including configuration management,
strategy customization, and performance tracking.
"""

import uuid
import json
import enum
import datetime
import secrets
import time
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

from common.utils import TimestampUtils, UuidUtils, HashUtils, SecurityUtils
from common.constants import (
    TIMEFRAMES, PLATFORMS, ASSET_CLASSES, ORDER_TYPES,
    ORDER_SIDES, SIGNAL_TYPES, SIGNAL_STRENGTHS, USER_ROLES
)
from common.exceptions import ModelValidationError
from data_storage.models.market_data import Base, TimestampMixin, Auditable, SoftDeleteMixin


@dataclass
class UserPreferences:
    """User-level capital and risk preferences."""

    user_id: str
    risk_profile: str = "moderate"
    max_drawdown_threshold: float = 20.0
    risk_per_trade: float = 1.0
    leverage_preference: float = 2.0
    recovery_aggressiveness: float = 0.5
    auto_compound: bool = True
    capital_allocation_strategy: str = "dynamic"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class UserCapitalSettings:
    """Configuration for capital allocation limits."""

    user_id: str
    max_position_size_percentage: float = 10.0
    min_position_size: float = 0.01
    kelly_criterion_modifier: float = 0.5
    max_correlated_exposure: float = 25.0
    reserve_percentage: float = 10.0
    profit_distribution: Dict[str, float] = field(default_factory=lambda: {"reinvest": 80.0, "reserve": 20.0})
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# User preference category enum
class PreferenceCategory(enum.Enum):
    """Categories for user preferences."""
    GENERAL = "general"
    TRADING = "trading"
    NOTIFICATIONS = "notifications"
    UI = "ui"
    SECURITY = "security"
    STRATEGY = "strategy"
    RISK_MANAGEMENT = "risk_management"
    PLATFORM_SPECIFIC = "platform_specific"
    VOICE_ASSISTANT = "voice_assistant"
    ADVANCED = "advanced"

# Trade status enum
class TradeStatus(enum.Enum):
    """Status values for trades."""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    EXPIRED = "expired"

# User models
class User(Base, Auditable, SoftDeleteMixin):
    """Model for storing user information."""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), nullable=False, unique=True)
    email = Column(String(255), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    role = Column(Enum(*USER_ROLES, name='user_role_enum'), nullable=False, default='user')
    is_active = Column(Boolean, default=True, nullable=False)
    
    # User verification
    email_verified = Column(Boolean, default=False, nullable=False)
    email_verification_token = Column(String(64), nullable=True)
    email_verification_sent_at = Column(TIMESTAMP(timezone=True), nullable=True)
    
    # Security
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    last_login_at = Column(TIMESTAMP(timezone=True), nullable=True)
    last_login_ip = Column(String(45), nullable=True)
    password_changed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    
    # Two-factor authentication
    two_factor_enabled = Column(Boolean, default=False, nullable=False)
    two_factor_secret = Column(String(255), nullable=True)
    backup_codes = Column(JSONB, nullable=True)
    
    # Account limits
    max_bots = Column(Integer, nullable=True)
    max_strategies = Column(Integer, nullable=True)
    max_concurrent_trades = Column(Integer, nullable=True)
    
    # User metadata
    last_seen_at = Column(TIMESTAMP(timezone=True), nullable=True)
    onboarding_completed = Column(Boolean, default=False, nullable=False)
    user_metadata = Column(JSONB, nullable=True)  # Renamed from metadata to avoid SQLAlchemy conflict
    
    # Relationships
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    trading_accounts = relationship("TradingAccount", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    strategies = relationship("UserStrategy", back_populates="user", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        Index('ix_users_username', 'username'),
        Index('ix_users_email', 'email'),
        Index('ix_users_role', 'role'),
        Index('ix_users_is_active', 'is_active'),
        {'extend_existing': True},
    )
    
    def __repr__(self):
        return f""
    
    def set_password(self, password):
        """Set the password hash from a plaintext password."""
        self.password_hash = SecurityUtils.hash_password(password)
        self.password_changed_at = datetime.datetime.now(datetime.timezone.utc)
    
    def verify_password(self, password):
        """Verify a plaintext password against the stored hash."""
        return SecurityUtils.verify_password(password, self.password_hash)
    
    def generate_email_verification_token(self):
        """Generate a new email verification token."""
        self.email_verification_token = secrets.token_urlsafe(48)
        self.email_verification_sent_at = datetime.datetime.now(datetime.timezone.utc)
        return self.email_verification_token
    
    def verify_email(self, token):
        """Verify the user's email with the provided token."""
        if not self.email_verification_token or self.email_verified:
            return False
        
        if self.email_verification_token != token:
            return False
        
        self.email_verified = True
        self.email_verification_token = None
        return True
    
    def generate_two_factor_secret(self):
        """Generate a new two-factor authentication secret."""
        self.two_factor_secret = SecurityUtils.generate_totp_secret()
        return self.two_factor_secret
    
    def verify_two_factor(self, code):
        """Verify a two-factor authentication code."""
        if not self.two_factor_enabled or not self.two_factor_secret:
            return True
            
        return SecurityUtils.verify_totp(code, self.two_factor_secret)
    
    def generate_backup_codes(self, count=10):
        """Generate backup codes for 2FA recovery."""
        if not self.two_factor_enabled:
            return []
            
        codes = [secrets.token_hex(5).upper() for _ in range(count)]
        self.backup_codes = [{"code": code, "used": False} for code in codes]
        return codes
    
    def use_backup_code(self, code):
        """Use a backup code for 2FA recovery."""
        if not self.backup_codes:
            return False
            
        for backup in self.backup_codes:
            if backup["code"] == code and not backup["used"]:
                backup["used"] = True
                return True
                
        return False
    
    def to_dict(self, include_sensitive=False):
        """Convert user to dictionary for API responses."""
        result = {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role,
            "is_active": self.is_active,
            "email_verified": self.email_verified,
            "two_factor_enabled": self.two_factor_enabled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
            "onboarding_completed": self.onboarding_completed
        }
        
        if include_sensitive:
            result.update({
                "failed_login_attempts": self.failed_login_attempts,
                "last_login_ip": self.last_login_ip,
                "max_bots": self.max_bots,
                "max_strategies": self.max_strategies,
                "max_concurrent_trades": self.max_concurrent_trades
            })
            
        return result

class ApiKey(Base, TimestampMixin):
    """Model for storing user API keys for external platforms."""
    __tablename__ = 'api_keys'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    platform = Column(Enum(*PLATFORMS, name='api_platform_enum'), nullable=False)
    name = Column(String(100), nullable=False)
    api_key = Column(String(255), nullable=False)
    api_secret = Column(String(255), nullable=False)
    passphrase = Column(String(255), nullable=True)  # For platforms that require it
    is_active = Column(Boolean, default=True, nullable=False)
    permissions = Column(JSONB, nullable=True)  # Specific permissions granted
    expires_at = Column(TIMESTAMP(timezone=True), nullable=True)
    last_used_at = Column(TIMESTAMP(timezone=True), nullable=True)
    key_metadata = Column(JSONB, nullable=True)  # Renamed from metadata to avoid SQLAlchemy conflict
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    trading_accounts = relationship("TradingAccount", back_populates="api_key")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'platform', 'name', name='uq_user_platform_keyname'),
        Index('ix_api_keys_user_id', 'user_id'),
        Index('ix_api_keys_platform', 'platform'),
        Index('ix_api_keys_is_active', 'is_active'),
    )
    
    def __repr__(self):
        return f""
    
    def mask_secrets(self):
        """Return a masked version of sensitive fields for display."""
        masked_key = self.api_key[:4] + "*" * (len(self.api_key) - 8) + self.api_key[-4:]
        masked_secret = "*" * 16
        masked_passphrase = "*" * 8 if self.passphrase else None
        
        return {
            "id": str(self.id),
            "platform": self.platform,
            "name": self.name,
            "api_key": masked_key,
            "api_secret": masked_secret,
            "passphrase": masked_passphrase,
            "is_active": self.is_active,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None
        }
    
    def update_last_used(self):
        """Update the last used timestamp."""
        self.last_used_at = datetime.datetime.now(datetime.timezone.utc)
    
    @property
    def is_expired(self):
        """Check if the API key has expired."""
        if not self.expires_at:
            return False
            
        return self.expires_at < datetime.datetime.now(datetime.timezone.utc)
    
    @classmethod
    def encrypt_secrets(cls, api_secret, passphrase=None):
        """Encrypt API secrets before storage."""
        encrypted_secret = SecurityUtils.encrypt_sensitive_data(api_secret)
        encrypted_passphrase = None
        if passphrase:
            encrypted_passphrase = SecurityUtils.encrypt_sensitive_data(passphrase)
            
        return encrypted_secret, encrypted_passphrase
    
    @classmethod
    def decrypt_secrets(cls, encrypted_secret, encrypted_passphrase=None):
        """Decrypt API secrets for use."""
        api_secret = SecurityUtils.decrypt_sensitive_data(encrypted_secret)
        passphrase = None
        if encrypted_passphrase:
            passphrase = SecurityUtils.decrypt_sensitive_data(encrypted_passphrase)
            
        return api_secret, passphrase

class TradingAccount(Base, Auditable):
    """Model for storing user trading accounts."""
    __tablename__ = 'trading_accounts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    api_key_id = Column(UUID(as_uuid=True), ForeignKey('api_keys.id'), nullable=False)
    platform = Column(Enum(*PLATFORMS, name='account_platform_enum'), nullable=False)
    account_id = Column(String(100), nullable=False)  # Platform-specific account ID
    name = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_demo = Column(Boolean, default=False, nullable=False)
    balance = Column(Float, nullable=True)
    equity = Column(Float, nullable=True)
    currency = Column(String(10), nullable=True)
    leverage = Column(Float, nullable=True)
    margin_mode = Column(String(20), nullable=True)  # ISOLATED, CROSS, etc.
    last_sync_at = Column(TIMESTAMP(timezone=True), nullable=True)
    account_metadata = Column(JSONB, nullable=True)  # Renamed from metadata to avoid SQLAlchemy conflict
    
    # Risk management settings
    max_risk_per_trade = Column(Float, nullable=True)  # Percentage of account
    max_daily_drawdown = Column(Float, nullable=True)  # Percentage of account
    max_position_size = Column(Float, nullable=True)
    circuit_breaker_enabled = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="trading_accounts")
    api_key = relationship("ApiKey", back_populates="trading_accounts")
    trades = relationship("Trade", back_populates="trading_account", cascade="all, delete-orphan")
    positions = relationship("Position", back_populates="trading_account", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'platform', 'account_id', name='uq_user_platform_account'),
        Index('ix_trading_accounts_user_id', 'user_id'),
        Index('ix_trading_accounts_api_key_id', 'api_key_id'),
        Index('ix_trading_accounts_platform', 'platform'),
        Index('ix_trading_accounts_is_active', 'is_active'),
        Index('ix_trading_accounts_is_demo', 'is_demo'),
    )
    
    def __repr__(self):
        return (f"")
    
    def update_balance(self, balance, equity=None):
        """Update account balance and equity."""
        self.balance = balance
        if equity is not None:
            self.equity = equity
        self.last_sync_at = datetime.datetime.now(datetime.timezone.utc)
    
    @property
    def unrealized_pnl(self):
        """Calculate unrealized P&L as the difference between equity and balance."""
        if self.equity is None or self.balance is None:
            return None
        return self.equity - self.balance
    
    @property
    def margin_used(self):
        """Calculate margin currently in use."""
        return sum(position.margin for position in self.positions 
                  if position.status == TradeStatus.OPEN.value)
    
    @property
    def free_margin(self):
        """Calculate available margin."""
        if self.margin_used is None or self.balance is None:
            return None
        return self.balance - self.margin_used
    
    @property
    def margin_level(self):
        """Calculate margin level as percentage."""
        if self.margin_used is None or self.margin_used == 0 or self.equity is None:
            return None
        return (self.equity / self.margin_used) * 100
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "api_key_id": str(self.api_key_id),
            "platform": self.platform,
            "account_id": self.account_id,
            "name": self.name,
            "is_active": self.is_active,
            "is_demo": self.is_demo,
            "balance": self.balance,
            "equity": self.equity,
            "currency": self.currency,
            "leverage": self.leverage,
            "margin_mode": self.margin_mode,
            "last_sync_at": self.last_sync_at.isoformat() if self.last_sync_at else None,
            "max_risk_per_trade": self.max_risk_per_trade,
            "max_daily_drawdown": self.max_daily_drawdown,
            "max_position_size": self.max_position_size,
            "circuit_breaker_enabled": self.circuit_breaker_enabled,
            "unrealized_pnl": self.unrealized_pnl,
            "margin_used": self.margin_used,
            "free_margin": self.free_margin,
            "margin_level": self.margin_level
        }

class UserPreference(Base, TimestampMixin):
    """Model for storing user preferences."""
    __tablename__ = 'user_preferences'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    category = Column(Enum(PreferenceCategory), nullable=False)
    key = Column(String(100), nullable=False)
    value = Column(JSONB, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="preferences")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'category', 'key', name='uq_user_pref_category_key'),
        Index('ix_user_preferences_user_id', 'user_id'),
        Index('ix_user_preferences_category', 'category'),
        Index('ix_user_preferences_key', 'key'),
    )
    
    def __repr__(self):
        return (f"")
    
    @classmethod
    def get_preference(cls, session, user_id, category, key, default=None):
        """Get a user preference value with default fallback."""
        pref = session.query(cls).filter_by(
            user_id=user_id, 
            category=category,
            key=key
        ).first()
        
        if not pref:
            return default
            
        return pref.value
    
    @classmethod
    def set_preference(cls, session, user_id, category, key, value):
        """Set a user preference value, creating if it doesn't exist."""
        pref = session.query(cls).filter_by(
            user_id=user_id, 
            category=category,
            key=key
        ).first()
        
        if not pref:
            pref = cls(
                user_id=user_id,
                category=category,
                key=key,
                value=value
            )
            session.add(pref)
        else:
            pref.value = value
            
        return pref

class UserSession(Base, TimestampMixin):
    """Model for tracking user login sessions."""
    __tablename__ = 'user_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    session_token = Column(String(255), nullable=False, unique=True)
    ip_address = Column(String(45), nullable=False)
    user_agent = Column(String(255), nullable=True)
    device_info = Column(JSONB, nullable=True)
    expires_at = Column(TIMESTAMP(timezone=True), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_activity_at = Column(TIMESTAMP(timezone=True), nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    # Constraints
    __table_args__ = (
        Index('ix_user_sessions_user_id', 'user_id'),
        Index('ix_user_sessions_session_token', 'session_token'),
        Index('ix_user_sessions_is_active', 'is_active'),
        Index('ix_user_sessions_expires_at', 'expires_at'),
    )
    
    def __repr__(self):
        return f""
    
    @property
    def is_expired(self):
        """Check if the session has expired."""
        now = datetime.datetime.now(datetime.timezone.utc)
        return now > self.expires_at
    
    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity_at = datetime.datetime.now(datetime.timezone.utc)
    
    def extend_expiration(self, hours=24):
        """Extend the session expiration time."""
        self.expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=hours)
    
    def terminate(self):
        """Terminate the session."""
        self.is_active = False
    
    @classmethod
    def create_session(cls, user_id, ip_address, user_agent=None, device_info=None, hours=24):
        """Create a new user session."""
        session = cls(
            user_id=user_id,
            session_token=secrets.token_urlsafe(48),
            ip_address=ip_address,
            user_agent=user_agent,
            device_info=device_info,
            expires_at=datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=hours),
            last_activity_at=datetime.datetime.now(datetime.timezone.utc)
        )
        return session
    
    @classmethod
    def get_active_session(cls, session, token):
        """Get an active session by token."""
        user_session = session.query(cls).filter_by(
            session_token=token,
            is_active=True
        ).first()
        
        if not user_session or user_session.is_expired:
            return None
            
        user_session.update_activity()
        return user_session

class Notification(Base, TimestampMixin):
    """Model for storing user notifications."""
    __tablename__ = 'notifications'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    notification_type = Column(String(50), nullable=False)
    priority = Column(Integer, nullable=False, default=0)  # 0-5, with 5 being highest
    read = Column(Boolean, default=False, nullable=False)
    read_at = Column(TIMESTAMP(timezone=True), nullable=True)
    expires_at = Column(TIMESTAMP(timezone=True), nullable=True)
    data = Column(JSONB, nullable=True)  # Additional structured data
    action_url = Column(String(255), nullable=True)  # Optional URL for action
    
    # Relationships
    user = relationship("User", back_populates="notifications")
    
    # Constraints
    __table_args__ = (
        Index('ix_notifications_user_id', 'user_id'),
        Index('ix_notifications_notification_type', 'notification_type'),
        Index('ix_notifications_read', 'read'),
        Index('ix_notifications_priority', 'priority'),
        Index('ix_notifications_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return (f"")
    
    def mark_as_read(self):
        """Mark the notification as read."""
        self.read = True
        self.read_at = datetime.datetime.now(datetime.timezone.utc)
    
    @property
    def is_expired(self):
        """Check if the notification has expired."""
        if not self.expires_at:
            return False
            
        return datetime.datetime.now(datetime.timezone.utc) > self.expires_at
    
    @classmethod
    def create_notification(cls, user_id, title, message, notification_type, 
                          priority=0, expires_in_hours=None, data=None, action_url=None):
        """Create a new notification."""
        expires_at = None
        if expires_in_hours is not None:
            expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=expires_in_hours)
            
        notification = cls(
            user_id=user_id,
            title=title,
            message=message,
            notification_type=notification_type,
            priority=priority,
            expires_at=expires_at,
            data=data,
            action_url=action_url
        )
        return notification

class UserStrategy(Base, Auditable, SoftDeleteMixin):
    """Model for storing user-customized trading strategies."""
    __tablename__ = 'user_strategies'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    base_strategy = Column(String(100), nullable=False)  # Reference to system strategy
    is_active = Column(Boolean, default=True, nullable=False)
    is_favorite = Column(Boolean, default=False, nullable=False)
    
    # Strategy parameters
    parameters = Column(JSONB, nullable=False)  # Customized parameters
    timeframes = Column(JSONB, nullable=False)  # Timeframes this strategy works with
    asset_classes = Column(JSONB, nullable=False)  # Asset classes this strategy works with
    risk_profile = Column(String(20), nullable=False)  # LOW, MEDIUM, HIGH
    
    # Performance metrics
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    avg_trade_duration = Column(Integer, nullable=True)  # In seconds
    total_trades = Column(Integer, nullable=True)
    last_performance_update = Column(TIMESTAMP(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="strategies")
    assets = relationship("UserStrategyAsset", back_populates="strategy", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'name', name='uq_user_strategy_name'),
        Index('ix_user_strategies_user_id', 'user_id'),
        Index('ix_user_strategies_base_strategy', 'base_strategy'),
        Index('ix_user_strategies_is_active', 'is_active'),
        Index('ix_user_strategies_risk_profile', 'risk_profile'),
    )
    
    def __repr__(self):
        return (f"")
    
    def update_performance(self, session):
        """Update strategy performance metrics from trade history."""
        from sqlalchemy import func, and_
        
        # Get all trades for this strategy
        trades_query = session.query(Trade).filter(
            Trade.user_strategy_id == self.id
        )
        
        total_trades = trades_query.count()
        if total_trades == 0:
            return False
            
        # Calculate win rate
        winning_trades = trades_query.filter(Trade.profit > 0).count()
        self.win_rate = (winning_trades / total_trades) if total_trades > 0 else 0
        
        # Calculate profit factor
        profit_sum = trades_query.filter(Trade.profit > 0).with_entities(func.sum(Trade.profit)).scalar() or 0
        loss_sum = abs(trades_query.filter(Trade.profit < 0).with_entities(func.sum(Trade.profit)).scalar() or 0)
        self.profit_factor = (profit_sum / loss_sum) if loss_sum > 0 else 0
        
        # Calculate average trade duration
        duration_avg = trades_query.filter(
            Trade.close_time != None
        ).with_entities(
            func.avg(func.extract('epoch', Trade.close_time - Trade.open_time))
        ).scalar()
        
        self.avg_trade_duration = int(duration_avg) if duration_avg else 0
        self.total_trades = total_trades
        self.last_performance_update = datetime.datetime.now(datetime.timezone.utc)
        
        return True
    
    def to_dict(self, include_assets=False):
        """Convert to dictionary for API responses."""
        result = {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "name": self.name,
            "description": self.description,
            "base_strategy": self.base_strategy,
            "is_active": self.is_active,
            "is_favorite": self.is_favorite,
            "parameters": self.parameters,
            "timeframes": self.timeframes,
            "asset_classes": self.asset_classes,
            "risk_profile": self.risk_profile,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "avg_trade_duration": self.avg_trade_duration,
            "total_trades": self.total_trades,
            "last_performance_update": self.last_performance_update.isoformat() if self.last_performance_update else None
        }
        
        if include_assets and hasattr(self, 'assets'):
            result["assets"] = [asset.to_dict() for asset in self.assets]
            
        return result

class UserStrategyAsset(Base, TimestampMixin):
    """Model for associating assets with user strategies with specific parameters."""
    __tablename__ = 'user_strategy_assets'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_strategy_id = Column(UUID(as_uuid=True), ForeignKey('user_strategies.id'), nullable=False)
    asset_id = Column(UUID(as_uuid=True), ForeignKey('assets.id'), nullable=False)
    
    # Asset-specific strategy parameters
    is_active = Column(Boolean, default=True, nullable=False)
    parameters = Column(JSONB, nullable=True)  # Override strategy parameters for this asset
    weight = Column(Float, default=1.0, nullable=False)  # For portfolio allocation
    performance_override = Column(Boolean, default=False, nullable=False)
    
    # Asset-specific performance metrics
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)
    last_trade_at = Column(TIMESTAMP(timezone=True), nullable=True)
    
    # Relationships
    strategy = relationship("UserStrategy", back_populates="assets")
    asset = relationship("Asset")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_strategy_id', 'asset_id', name='uq_strategy_asset'),
        Index('ix_user_strategy_assets_strategy_id', 'user_strategy_id'),
        Index('ix_user_strategy_assets_asset_id', 'asset_id'),
        Index('ix_user_strategy_assets_is_active', 'is_active'),
    )
    
    def __repr__(self):
        return (f"")
    
    def to_dict(self, include_asset_details=False):
        """Convert to dictionary for API responses."""
        result = {
            "id": str(self.id),
            "user_strategy_id": str(self.user_strategy_id),
            "asset_id": str(self.asset_id),
            "is_active": self.is_active,
            "parameters": self.parameters,
            "weight": self.weight,
            "performance_override": self.performance_override,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "last_trade_at": self.last_trade_at.isoformat() if self.last_trade_at else None
        }
        
        if include_asset_details and hasattr(self, 'asset'):
            result["asset"] = {
                "symbol": self.asset.symbol,
                "platform": self.asset.platform,
                "name": self.asset.name,
                "asset_class": self.asset.asset_class
            }
            
        return result

class Trade(Base, TimestampMixin):
    """Model for storing trade records."""
    __tablename__ = 'trades'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    trading_account_id = Column(UUID(as_uuid=True), ForeignKey('trading_accounts.id'), nullable=False)
    position_id = Column(UUID(as_uuid=True), ForeignKey('positions.id'), nullable=True)
    asset_id = Column(UUID(as_uuid=True), ForeignKey('assets.id'), nullable=False)
    user_strategy_id = Column(UUID(as_uuid=True), ForeignKey('user_strategies.id'), nullable=True)
    
    # Trade details
    external_id = Column(String(100), nullable=True)  # ID from exchange
    trade_type = Column(String(20), nullable=False)  # MARKET, LIMIT, etc.
    side = Column(Enum(*ORDER_SIDES, name='trade_side_enum'), nullable=False)
    size = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    
    # Trade lifecycle
    status = Column(Enum(TradeStatus), nullable=False, default=TradeStatus.OPEN)
    open_time = Column(TIMESTAMP(timezone=True), nullable=False)
    close_time = Column(TIMESTAMP(timezone=True), nullable=True)
    
    # Performance
    profit = Column(Float, nullable=True)  # In account currency
    profit_percentage = Column(Float, nullable=True)  # As percentage of position size
    fees = Column(Float, nullable=True)
    slippage = Column(Float, nullable=True)
    
    # Trade source
    signal_id = Column(UUID(as_uuid=True), nullable=True)  # Reference to trading signal
    is_manual = Column(Boolean, default=False, nullable=False)
    is_automated = Column(Boolean, default=True, nullable=False)
    entry_comment = Column(String(255), nullable=True)
    exit_comment = Column(String(255), nullable=True)
    
    # Trade analytics
    entry_indicators = Column(JSONB, nullable=True)
    exit_indicators = Column(JSONB, nullable=True)
    trade_screenshot = Column(String(255), nullable=True)  # URL to screenshot
    market_conditions = Column(JSONB, nullable=True)
    tags = Column(JSONB, nullable=True)  # User-defined tags
    
    # Relationships
    user = relationship("User")
    trading_account = relationship("TradingAccount", back_populates="trades")
    position = relationship("Position", back_populates="trades")
    asset = relationship("Asset")
    
    # Constraints
    __table_args__ = (
        Index('ix_trades_user_id', 'user_id'),
        Index('ix_trades_trading_account_id', 'trading_account_id'),
        Index('ix_trades_position_id', 'position_id'),
        Index('ix_trades_asset_id', 'asset_id'),
        Index('ix_trades_user_strategy_id', 'user_strategy_id'),
        Index('ix_trades_status', 'status'),
        Index('ix_trades_open_time', 'open_time'),
        Index('ix_trades_close_time', 'close_time'),
        Index('ix_trades_is_manual', 'is_manual'),
        Index('ix_trades_profit', 'profit'),
    )
    
    def __repr__(self):
        return (f"")
    
    def close(self, exit_price, close_time=None, comment=None):
        """Close the trade with an exit price."""
        if self.status != TradeStatus.OPEN:
            raise ValueError("Cannot close a trade that is not open")
            
        self.exit_price = exit_price
        self.close_time = close_time or datetime.datetime.now(datetime.timezone.utc)
        self.status = TradeStatus.CLOSED
        self.exit_comment = comment
        
        # Calculate profit
        if self.side == 'BUY':
            price_diff = self.exit_price - self.entry_price
        else:
            price_diff = self.entry_price - self.exit_price
            
        self.profit = price_diff * self.size
        self.profit_percentage = (price_diff / self.entry_price) * 100
        
        return self.profit, self.profit_percentage
    
    @property
    def duration(self):
        """Calculate trade duration in seconds."""
        if not self.close_time or not self.open_time:
            return None
            
        return (self.close_time - self.open_time).total_seconds()
    
    @property
    def unrealized_profit(self):
        """Calculate unrealized profit for open trades."""
        if self.status != TradeStatus.OPEN or not hasattr(self, 'asset') or not self.asset:
            return None
            
        # This would need to get current price from market data
        # For now, returning None as a placeholder
        return None
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "trading_account_id": str(self.trading_account_id),
            "position_id": str(self.position_id) if self.position_id else None,
            "asset_id": str(self.asset_id),
            "user_strategy_id": str(self.user_strategy_id) if self.user_strategy_id else None,
            "external_id": self.external_id,
            "trade_type": self.trade_type,
            "side": self.side,
            "size": self.size,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "status": self.status.value,
            "open_time": self.open_time.isoformat() if self.open_time else None,
            "close_time": self.close_time.isoformat() if self.close_time else None,
            "profit": self.profit,
            "profit_percentage": self.profit_percentage,
            "fees": self.fees,
            "slippage": self.slippage,
            "signal_id": str(self.signal_id) if self.signal_id else None,
            "is_manual": self.is_manual,
            "is_automated": self.is_automated,
            "entry_comment": self.entry_comment,
            "exit_comment": self.exit_comment,
            "duration": self.duration,
            "tags": self.tags
        }

class Position(Base, TimestampMixin):
    """Model for storing trading positions (groups of related trades)."""
    __tablename__ = 'positions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    trading_account_id = Column(UUID(as_uuid=True), ForeignKey('trading_accounts.id'), nullable=False)
    asset_id = Column(UUID(as_uuid=True), ForeignKey('assets.id'), nullable=False)
    user_strategy_id = Column(UUID(as_uuid=True), ForeignKey('user_strategies.id'), nullable=True)
    
    # Position details
    name = Column(String(100), nullable=True)
    side = Column(Enum(*ORDER_SIDES, name='position_side_enum'), nullable=False)
    size = Column(Float, nullable=False)
    average_entry = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    status = Column(Enum(TradeStatus), nullable=False, default=TradeStatus.OPEN)
    
    # Risk management
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    risk_amount = Column(Float, nullable=True)  # In account currency
    risk_percentage = Column(Float, nullable=True)  # As percentage of account
    
    # Position lifecycle
    open_time = Column(TIMESTAMP(timezone=True), nullable=False)
    close_time = Column(TIMESTAMP(timezone=True), nullable=True)
    last_update_time = Column(TIMESTAMP(timezone=True), nullable=False)
    
    # Performance
    profit = Column(Float, nullable=True)  # In account currency
    profit_percentage = Column(Float, nullable=True)  # As percentage of position size
    max_profit = Column(Float, nullable=True)  # Maximum profit reached
    max_loss = Column(Float, nullable=True)  # Maximum loss reached
    
    # Position source and metadata
    is_manual = Column(Boolean, default=False, nullable=False)
    is_hedged = Column(Boolean, default=False, nullable=False)
    margin = Column(Float, nullable=True)
    leverage = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    position_metadata = Column(JSONB, nullable=True)

    
    # Relationships
    user = relationship("User")
    trading_account = relationship("TradingAccount", back_populates="positions")
    asset = relationship("Asset")
    trades = relationship("Trade", back_populates="position")
    
    # Constraints
    __table_args__ = (
        Index('ix_positions_user_id', 'user_id'),
        Index('ix_positions_trading_account_id', 'trading_account_id'),
        Index('ix_positions_asset_id', 'asset_id'),
        Index('ix_positions_user_strategy_id', 'user_strategy_id'),
        Index('ix_positions_status', 'status'),
        Index('ix_positions_open_time', 'open_time'),
        Index('ix_positions_close_time', 'close_time'),
        {'extend_existing': True},
    )
    
    def __repr__(self):
        return (f"")
    
    def update_average_entry(self, session):
        """Update average entry price based on constituent trades."""
        trades = session.query(Trade).filter(
            Trade.position_id == self.id
        ).all()
        
        if not trades:
            return False
            
        total_size = sum(trade.size for trade in trades)
        weighted_price = sum(trade.size * trade.entry_price for trade in trades)
        
        if total_size > 0:
            self.average_entry = weighted_price / total_size
            self.size = total_size
            self.last_update_time = datetime.datetime.now(datetime.timezone.utc)
            return True
            
        return False
    
    def update_current_price(self, price):
        """Update current price and calculate unrealized profit."""
        self.current_price = price
        self.last_update_time = datetime.datetime.now(datetime.timezone.utc)
        
        if self.side == 'BUY':
            price_diff = self.current_price - self.average_entry
        else:
            price_diff = self.average_entry - self.current_price
            
        unrealized_profit = price_diff * self.size
        unrealized_profit_percentage = (price_diff / self.average_entry) * 100
        
        # Update max profit/loss
        if self.max_profit is None or unrealized_profit > self.max_profit:
            self.max_profit = unrealized_profit
            
        if self.max_loss is None or unrealized_profit < self.max_loss:
            self.max_loss = unrealized_profit
            
        return unrealized_profit, unrealized_profit_percentage
    
    def close(self, close_price, close_time=None):
        """Close the position."""
        if self.status != TradeStatus.OPEN:
            raise ValueError("Cannot close a position that is not open")
            
        self.current_price = close_price
        self.close_time = close_time or datetime.datetime.now(datetime.timezone.utc)
        self.status = TradeStatus.CLOSED
        self.last_update_time = self.close_time
        
        # Calculate final profit
        if self.side == 'BUY':
            price_diff = self.current_price - self.average_entry
        else:
            price_diff = self.average_entry - self.current_price
            
        self.profit = price_diff * self.size
        self.profit_percentage = (price_diff / self.average_entry) * 100
        
        return self.profit, self.profit_percentage
    
    @property
    def duration(self):
        """Calculate position duration in seconds."""
        if not self.close_time:
            now = datetime.datetime.now(datetime.timezone.utc)
            return (now - self.open_time).total_seconds()
            
        return (self.close_time - self.open_time).total_seconds()
    
    @property
    def unrealized_profit(self):
        """Calculate unrealized profit based on current price."""
        if self.status != TradeStatus.OPEN or not self.current_price:
            return None
            
        if self.side == 'BUY':
            price_diff = self.current_price - self.average_entry
        else:
            price_diff = self.average_entry - self.current_price
            
        return price_diff * self.size
    
    @property
    def unrealized_profit_percentage(self):
        """Calculate unrealized profit percentage based on current price."""
        if self.status != TradeStatus.OPEN or not self.current_price or not self.average_entry:
            return None
            
        if self.side == 'BUY':
            price_diff = self.current_price - self.average_entry
        else:
            price_diff = self.average_entry - self.current_price
            
        return (price_diff / self.average_entry) * 100
    
    def to_dict(self, include_trades=False):
        """Convert to dictionary for API responses."""
        result = {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "trading_account_id": str(self.trading_account_id),
            "asset_id": str(self.asset_id),
            "user_strategy_id": str(self.user_strategy_id) if self.user_strategy_id else None,
            "name": self.name,
            "side": self.side,
            "size": self.size,
            "average_entry": self.average_entry,
            "current_price": self.current_price,
            "status": self.status.value,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_amount": self.risk_amount,
            "risk_percentage": self.risk_percentage,
            "open_time": self.open_time.isoformat() if self.open_time else None,
            "close_time": self.close_time.isoformat() if self.close_time else None,
            "last_update_time": self.last_update_time.isoformat() if self.last_update_time else None,
            "profit": self.profit,
            "profit_percentage": self.profit_percentage,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "is_manual": self.is_manual,
            "is_hedged": self.is_hedged,
            "margin": self.margin,
            "leverage": self.leverage,
            "notes": self.notes,
            "duration": self.duration,
            "unrealized_profit": self.unrealized_profit,
            "unrealized_profit_percentage": self.unrealized_profit_percentage
        }
        
        if include_trades and hasattr(self, 'trades'):
            result["trades"] = [trade.to_dict() for trade in self.trades]
            
        return result

class PerformanceSnapshot(Base, TimestampMixin):
    """Model for storing trading account performance snapshots."""
    __tablename__ = 'performance_snapshots'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    trading_account_id = Column(UUID(as_uuid=True), ForeignKey('trading_accounts.id'), nullable=False)
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    
    # Balance metrics
    balance = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    open_positions = Column(Integer, nullable=False)
    margin_used = Column(Float, nullable=True)
    free_margin = Column(Float, nullable=True)
    
    # Performance metrics
    daily_profit = Column(Float, nullable=True)
    daily_profit_percentage = Column(Float, nullable=True)
    weekly_profit = Column(Float, nullable=True)
    weekly_profit_percentage = Column(Float, nullable=True)
    monthly_profit = Column(Float, nullable=True)
    monthly_profit_percentage = Column(Float, nullable=True)
    
    # Trade metrics
    trades_today = Column(Integer, nullable=True)
    wins_today = Column(Integer, nullable=True)
    losses_today = Column(Integer, nullable=True)
    
    # Advanced metrics
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    current_drawdown = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    
    # Constraints
    __table_args__ = (
        Index('ix_performance_snapshots_user_id', 'user_id'),
        Index('ix_performance_snapshots_trading_account_id', 'trading_account_id'),
        Index('ix_performance_snapshots_timestamp', 'timestamp'),
        Index('ix_performance_snapshots_account_time', 'trading_account_id', 'timestamp'),
    )
    
    def __repr__(self):
        return (f"")

    @classmethod
    def create_snapshot(cls, session, trading_account):
        """Create a performance snapshot for a trading account."""
        from sqlalchemy import func, and_
        
        # Calculate trade metrics for today
        today_start = datetime.datetime.now(datetime.timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        
        trades_today_query = session.query(Trade).filter(
            Trade.trading_account_id == trading_account.id,
            Trade.open_time >= today_start
        )
        
        trades_today = trades_today_query.count()
        wins_today = trades_today_query.filter(Trade.profit > 0).count()
        losses_today = trades_today_query.filter(Trade.profit < 0).count()
        
        # Calculate profit metrics
        day_ago = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)
        week_ago = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)
        month_ago = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=30)
        
        daily_profit_query = session.query(func.sum(Trade.profit)).filter(
            Trade.trading_account_id == trading_account.id,
            Trade.close_time >= day_ago
        )
        
        weekly_profit_query = session.query(func.sum(Trade.profit)).filter(
            Trade.trading_account_id == trading_account.id,
            Trade.close_time >= week_ago
        )
        
        monthly_profit_query = session.query(func.sum(Trade.profit)).filter(
            Trade.trading_account_id == trading_account.id,
            Trade.close_time >= month_ago
        )
        
        daily_profit = daily_profit_query.scalar() or 0
        weekly_profit = weekly_profit_query.scalar() or 0
        monthly_profit = monthly_profit_query.scalar() or 0
        
        # Get previous snapshot for comparison
        prev_snapshot = session.query(cls).filter(
            cls.trading_account_id == trading_account.id
        ).order_by(cls.timestamp.desc()).first()
        
        prev_balance = prev_snapshot.balance if prev_snapshot else trading_account.balance
        
        # Calculate percentages
        if prev_balance and prev_balance > 0:
            daily_profit_percentage = (daily_profit / prev_balance) * 100
            weekly_profit_percentage = (weekly_profit / prev_balance) * 100
            monthly_profit_percentage = (monthly_profit / prev_balance) * 100
        else:
            daily_profit_percentage = 0
            weekly_profit_percentage = 0
            monthly_profit_percentage = 0
            
        # Calculate win rate and profit factor
        all_closed_trades = session.query(Trade).filter(
            Trade.trading_account_id == trading_account.id,
            Trade.status == TradeStatus.CLOSED
        )
        
        total_trades = all_closed_trades.count()
        winning_trades = all_closed_trades.filter(Trade.profit > 0).count()
        
        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0
        
        profit_sum = all_closed_trades.filter(
            Trade.profit > 0
        ).with_entities(func.sum(Trade.profit)).scalar() or 0
        
        loss_sum = abs(all_closed_trades.filter(
            Trade.profit < 0
        ).with_entities(func.sum(Trade.profit)).scalar() or 0)
        
        profit_factor = (profit_sum / loss_sum) if loss_sum > 0 else 0
        
        # Create the snapshot
        snapshot = cls(
            user_id=trading_account.user_id,
            trading_account_id=trading_account.id,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            balance=trading_account.balance,
            equity=trading_account.equity or trading_account.balance,
            open_positions=session.query(Position).filter(
                Position.trading_account_id == trading_account.id,
                Position.status == TradeStatus.OPEN
            ).count(),
            margin_used=trading_account.margin_used,
            free_margin=trading_account.free_margin,
            daily_profit=daily_profit,
            daily_profit_percentage=daily_profit_percentage,
            weekly_profit=weekly_profit,
            weekly_profit_percentage=weekly_profit_percentage,
            monthly_profit=monthly_profit,
            monthly_profit_percentage=monthly_profit_percentage,
            trades_today=trades_today,
            wins_today=wins_today,
            losses_today=losses_today,
            win_rate=win_rate,
            profit_factor=profit_factor
            # Note: Sharpe, Sortino, max drawdown would need more complex calculations
        )
        
        return snapshot
        
# Define functions to initialize the database
def initialize_user_tables(engine):
    """Initialize all user-related tables."""
    Base.metadata.create_all(engine)
    return True
