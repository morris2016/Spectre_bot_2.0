#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
System Constants and Enumerations

This module provides system-wide constants, enumerations, and configuration defaults
used throughout the QuantumSpectre Elite Trading System.
"""

import os
import enum
from pathlib import Path


# ======================================
# System Core Constants
# ======================================

VERSION = "1.0.0"
CONFIG_SCHEMA_VERSION = 1
SYSTEM_NAME = "QuantumSpectre Elite Trading System"
AUTHOR = "QuantumSpectre Team"
LICENSE = "MIT"

# Environment settings
ENV_PRODUCTION = "production"
ENV_DEVELOPMENT = "development"
ENV_TESTING = "testing"

# Default configuration paths
DEFAULT_CONFIG_PATH = os.environ.get(
    "QUANTUM_SPECTRE_CONFIG", str(Path.home() / ".quantumspectre" / "config.yml")
)
DEFAULT_DATA_DIR = os.environ.get(
    "QUANTUM_SPECTRE_DATA", str(Path.home() / ".quantumspectre" / "data")
)
STORAGE_ROOT_PATH = os.environ.get(
    "QUANTUM_SPECTRE_STORAGE", str(Path.home() / ".quantumspectre" / "storage")
)
DEFAULT_LOG_DIR = os.environ.get(
    "QUANTUM_SPECTRE_LOGS", str(Path.home() / ".quantumspectre" / "logs")
)
DEFAULT_MODEL_DIR = os.environ.get(
    "QUANTUM_SPECTRE_MODELS", str(Path.home() / ".quantumspectre" / "models")
)


# ======================================
# System Architecture Configuration
# ======================================

SERVICE_NAMES = {
    "data_ingest": "Data Ingestion Service",
    "data_feeds": "Data Feeds Service",
    "feature_service": "Feature Service",
    "intelligence": "Intelligence Service",
    "ml_models": "ML Models Service",
    "strategy_brains": "Strategy Brains Service",
    "brain_council": "Brain Council Service",
    "execution_engine": "Execution Engine Service",
    "risk_manager": "Risk Manager Service",
    "backtester": "Backtester Service",
    "monitoring": "Monitoring Service",
    "api_gateway": "API Gateway Service",
    "ui": "UI Service",
}

SERVICE_DEPENDENCIES = {
    "data_ingest": [],
    "data_feeds": ["data_ingest"],
    "feature_service": ["data_feeds"],
    "intelligence": ["feature_service"],
    "ml_models": ["feature_service"],
    "strategy_brains": ["intelligence", "ml_models"],
    "brain_council": ["strategy_brains"],
    "execution_engine": ["brain_council", "risk_manager"],
    "risk_manager": ["data_feeds"],
    "backtester": ["feature_service", "strategy_brains", "risk_manager"],
    "monitoring": [],
    "api_gateway": ["brain_council", "execution_engine", "monitoring"],
    "ui": ["api_gateway"],
}

SERVICE_STARTUP_ORDER = [
    "data_ingest", "data_feeds", "feature_service", "intelligence", "ml_models",
    "strategy_brains", "risk_manager", "brain_council", "execution_engine",
    "backtester", "monitoring", "api_gateway", "ui"
]

DATA_INGEST_METRICS_PREFIX = "data_ingest"


# ======================================
# Resource Management
# ======================================

DEFAULT_THREAD_POOL_SIZE = 10
MAX_THREAD_POOL_SIZE = 100
DEFAULT_PROCESS_POOL_SIZE = 4
MAX_PROCESS_POOL_SIZE = 16
MARKET_DATA_MAX_WORKERS = 16

MEMORY_WARNING_THRESHOLD = 0.85
MEMORY_CRITICAL_THRESHOLD = 0.95

LOG_LEVELS = {
    "CRITICAL": 50, "ERROR": 40, "WARNING": 30, "INFO": 20, "DEBUG": 10, "NOTSET": 0,
}
DEFAULT_LOG_LEVEL = "INFO"


# ======================================
# Network Configuration
# ======================================

# API rate limits (requests per minute)
API_RATE_LIMIT_DEFAULT = 100
API_RATE_LIMIT_TRADING = 20
API_RATE_LIMIT_AUTH = 10

# WebSocket configuration
WEBSOCKET_MAX_CONNECTIONS = 10000
WEBSOCKET_PING_INTERVAL = 30
WEBSOCKET_PING_TIMEOUT = 10
WEBSOCKET_CLOSE_TIMEOUT = 5

# HTTP configuration
HTTP_TIMEOUT_DEFAULT = 10
HTTP_TIMEOUT_FEED = 30
HTTP_TIMEOUT_LONG = 120
HTTP_MAX_RETRIES = 3
HTTP_RETRY_BACKOFF = 2.0


# ======================================
# Security Configuration
# ======================================

TOKEN_EXPIRY_ACCESS = 3600  # 1 hour
TOKEN_EXPIRY_REFRESH = 2592000  # 30 days
PASSWORD_MIN_LENGTH = 10
PASSWORD_HASH_ALGORITHM = "pbkdf2_sha256"
PASSWORD_SALT_LENGTH = 32
PASSWORD_HASH_ITERATIONS = 200000


# ======================================
# Database Configuration
# ======================================

DATABASE_POOL_MIN_SIZE = 5
DATABASE_POOL_MAX_SIZE = 20
DATABASE_MAX_QUERIES = 50000
DATABASE_CONNECTION_TIMEOUT = 60
DATABASE_COMMAND_TIMEOUT = 60


# ======================================
# Cache Configuration
# ======================================

CACHE_DEFAULT_TTL = 300
CACHE_LONG_TTL = 3600
CACHE_VERY_LONG_TTL = 86400


# ======================================
DEFAULT_FEATURE_PARAMS = {}  # Default parameters for feature calculations
# Exchange and Trading Constants
# ======================================

# Supported exchanges

# Exchange and Trading Enums
# ======================================

class Exchange(enum.Enum):
    """Supported trading exchanges."""
    BINANCE = "binance"
    DERIV = "deriv"
    BACKTEST = "backtest"


# Supported asset classes

class AssetClass(enum.Enum):
    """Supported asset classes."""
    CRYPTO = "crypto"
    FOREX = "forex"
    STOCK = "stock"
    INDEX = "index"
    COMMODITY = "commodity"
    SYNTHETIC = "synthetic"


class Timeframe(enum.Enum):
    """Supported trading timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"

TIME_FRAMES = TIMEFRAMES = [tf.value for tf in Timeframe]


class OrderType(enum.Enum):
    """Order types for trading."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"


# Alias for backward compatibility
ORDER_TYPE = OrderType

# Order sides
class OrderSide(enum.Enum):
    """Order sides for trading."""
    BUY = "buy"
    SELL = "sell"

# Position side (alias for backward compatibility with PositionType)
class PositionSide(enum.Enum):
    LONG = "long"
    SHORT = "short"

POSITION_SIDES = [ps.value for ps in PositionSide]

# Backward compatibility
POSITION_SIDE = PositionSide

# Position types (deprecated, use PositionSide)
class PositionType(enum.Enum):
    LONG = "long"
    SHORT = "short"


class OrderStatus(enum.Enum):
    """Order status states."""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    PENDING_CANCEL = "pending_cancel"
    REJECTED = "rejected"
    EXPIRED = "expired"

# Backwards compatibility
ORDER_STATUS = OrderStatus


class PositionStatus(enum.Enum):
    """Position status states."""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_CLOSED = "partially_closed"
    CLOSED = "closed"
    FAILED = "failed"

POSITION_STATUSES = [ps.value for ps in PositionStatus]

# Backward compatibility
POSITION_STATUS = PositionStatus


class TriggerType(enum.Enum):
    """Trigger types for stop and take profit orders."""
    PRICE = "price"           # Regular price based trigger
    MARK_PRICE = "mark_price" # Mark price trigger (for futures)
    INDEX_PRICE = "index_price" # Index price trigger


class TimeInForce(enum.Enum):
    """Time in force options."""
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date

TIME_IN_FORCE = TimeInForce


class SignalDirection(enum.Enum):
    """Trade signal directions."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SignalStrength(enum.Enum):
    """Signal strength levels."""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


class MarketRegime(enum.Enum):
    """Market condition types."""
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CHOPPY = "choppy"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


class RiskLevel(enum.Enum):
    """Risk assessment levels."""
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5


class RegimeTypes(enum.Enum):
    """Detailed market regime classifications used by strategy brains."""
    STRONG_TREND = "strong_trend"
    WEAK_TREND = "weak_trend"
    VOLATILE_TREND = "volatile_trend"
    RANGING = "ranging"
    VOLATILE_RANGE = "volatile_range"
    CHOPPY = "choppy"

# Regime types for strategy selection
REGIME_TYPES = [rt.value for rt in RegimeTypes]

# Weighting methods for signal aggregation
WEIGHTING_METHODS = {
    "EQUAL": "equal",
    "PERFORMANCE_BASED": "performance_based",
    "CONFIDENCE_BASED": "confidence_based",
    "ADAPTIVE": "adaptive",
    "REGIME_SPECIFIC": "regime_specific",
    "VOLATILITY_ADJUSTED": "volatility_adjusted"
}

# Default feature importance threshold
DEFAULT_FEATURE_IMPORTANCE_THRESHOLD = 0.01

# Default voting configuration
DEFAULT_VOTING_CONFIG = {
    "min_votes_required": 3,
    "consensus_threshold": 0.6,
    "confidence_threshold": 0.5,
    "weighting_method": "performance_based",
    "tie_breaker": "highest_confidence"
}

# ML model voting weights
ML_MODEL_VOTING_WEIGHTS = {
    "gradient_boosting": 1.0,
    "random_forest": 0.9,
    "neural_network": 0.8,
    "lstm": 1.0,
    "ensemble": 1.2
}

# Default prediction horizons for ML models
DEFAULT_PREDICTION_HORIZONS = {
    "short_term": 12,  # 12 periods
    "medium_term": 24,  # 24 periods
    "long_term": 48    # 48 periods
}

# Default confidence scaling factor for ML predictions
DEFAULT_CONFIDENCE_SCALING_FACTOR = 0.8

# Notification priority levels
NOTIFICATION_PRIORITY_LOW = 1
NOTIFICATION_PRIORITY_MEDIUM = 2
NOTIFICATION_PRIORITY_HIGH = 3
NOTIFICATION_PRIORITY_CRITICAL = 4

# Metric collection frequency in seconds
METRIC_COLLECTION_FREQUENCY = 60
# Maximum number of metrics to keep in history
MAX_METRIC_HISTORY = 1000


# Fee types
class FeeType(enum.Enum):
    MAKER = "maker"
    TAKER = "taker"
    FUNDING = "funding"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"


# ======================================
# Feature and Pattern Constants
# ======================================

# Technical indicator categories
class IndicatorCategory(enum.Enum):
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OSCILLATOR = "oscillator"
    SUPPORT_RESISTANCE = "support_resistance"
    PATTERN = "pattern"
    CUSTOM = "custom"


# Candlestick pattern types
class CandlestickPattern(enum.Enum):
    DOJI = "doji"
    HAMMER = "hammer"
    SHOOTING_STAR = "shooting_star"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    PIERCING_LINE = "piercing_line"
    DARK_CLOUD_COVER = "dark_cloud_cover"
    HARAMI_BULLISH = "harami_bullish"
    HARAMI_BEARISH = "harami_bearish"
    MARUBOZU = "marubozu"


# Chart pattern types
class ChartPattern(enum.Enum):
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    CHANNEL_UP = "channel_up"
    CHANNEL_DOWN = "channel_down"
    RECTANGLE = "rectangle"
    CUP_AND_HANDLE = "cup_and_handle"
    ROUNDING_BOTTOM = "rounding_bottom"
    ROUNDING_TOP = "rounding_top"


# Harmonic pattern types
class HarmonicPattern(enum.Enum):
    GARTLEY = "gartley"
    BUTTERFLY = "butterfly"
    BAT = "bat"
    CRAB = "crab"
    SHARK = "shark"
    CYPHER = "cypher"
    THREE_DRIVES = "three_drives"
    FIVE_ZERO = "five_zero"
    ABCD = "abcd"


# Fibonacci ratios
FIBONACCI_RATIOS = {
    "0": 0.0,
    "23.6": 0.236,
    "38.2": 0.382,
    "50": 0.5,
    "61.8": 0.618,
    "76.4": 0.764,
    "78.6": 0.786,
    "100": 1.0,
    "127.2": 1.272,
    "138.2": 1.382,
    "150": 1.5,
    "161.8": 1.618,
    "200": 2.0,
    "223.6": 2.236,
    "261.8": 2.618,
    "361.8": 3.618,
    "423.6": 4.236
}

# Common indicator parameters
SMA_PERIODS = [10, 20, 50, 100, 200]
EMA_PERIODS = [9, 12, 26, 50, 200]
RSI_PERIODS = [7, 14, 21]
MACD_PARAMS = {
    "FAST": 12,
    "SLOW": 26,
    "SIGNAL": 9
}
BOLLINGER_BANDS_PARAMS = {
    "PERIOD": 20,
    "STD_DEV": 2
}
STOCHASTIC_PARAMS = {
    "K_PERIOD": 14,
    "K_SLOWING": 3,
    "D_PERIOD": 3
}

# ======================================
# Machine Learning Constants
# ======================================

# ML model types
class ModelType(enum.Enum):
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    LSTM = "lstm"
    GRU = "gru"
    CNN = "cnn"
    ARIMA = "arima"
    PROPHET = "prophet"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ENSEMBLE = "ensemble"
# ======================================
# Trading Platform Configuration
# ======================================

SUPPORTED_PLATFORMS = ["deriv", "binance"]

SUPPORTED_ASSETS = {
    "binance": [
        "BTC", "ETH", "USDT", "BNB", "ADA", "XRP", "DOGE", "SOL", "DOT", "LTC",
        "BCH", "LINK", "MATIC", "ATOM", "AVAX", "TRX", "XLM", "NEAR", "FIL",
        "EOS", "AAVE", "UNI", "SAND", "MANA", "SHIB", "ALGO", "FTM", "ETC",
        "ZIL", "VET", "THETA", "XTZ", "GRT", "CHZ", "ENJ", "BAT", "ZRX",
        "1INCH", "COMP", "SNX", "YFI", "CRV", "KSM", "DASH", "OMG", "QTUM",
        "ICX", "ONT", "WAVES", "LRC", "BTT", "HOT", "NANO", "SC", "ZEN",
        "STMX", "ANKR", "CELR", "CVC", "DENT", "IOST", "KAVA", "NKN", "OCEAN",
        "RLC", "STORJ", "TOMO", "WRX", "XEM", "ZEC"
    ],
    "deriv": ["BTC", "ETH", "LTC", "USDC", "USDT", "XRP"]
}

ASSET_TYPES = [
    "crypto", "forex", "stocks", "indices", "commodities", "futures", "options",
]

# Feature scaling methods
class ScalingMethod(enum.Enum):
    STANDARD = "standard"  # Z-score normalization
    MINMAX = "minmax"      # Min-max scaling to [0,1]
    ROBUST = "robust"      # Scaling using quantiles
    MAXABS = "maxabs"      # Scaling by maximum absolute value
    NONE = "none"          # No scaling


# Hyperparameter optimization methods
class HyperparamOptMethod(enum.Enum):
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    TPE = "tpe"            # Tree-structured Parzen Estimator


# Cross-validation strategies
class CrossValidationStrategy(enum.Enum):
    K_FOLD = "k_fold"
    STRATIFIED_K_FOLD = "stratified_k_fold"
    TIME_SERIES_SPLIT = "time_series_split"
    GROUP_K_FOLD = "group_k_fold"
    PURGED_K_FOLD = "purged_k_fold"  # Specific for financial data


# ======================================
# Loophole Detection Constants
# ======================================

# Loophole types
class LoopholeType(enum.Enum):
    ARBITRAGE = "arbitrage"
    MARKET_INEFFICIENCY = "market_inefficiency"
    LIQUIDITY_IMBALANCE = "liquidity_imbalance"
    PREDICTABLE_PATTERN = "predictable_pattern"
    PLATFORM_QUIRK = "platform_quirk"
    ORDER_BOOK_ANOMALY = "order_book_anomaly"

# Loophole detection thresholds
MIN_INEFFICIENCY_SCORE = 0.65  # Minimum score for market inefficiency detection
ANOMALY_CONTAMINATION_FACTOR = 0.05  # Contamination factor for anomaly detection


# Arbitrage types
class ArbitrageType(enum.Enum):
    SPATIAL = "spatial"           # Same asset, different venues
    TRIANGULAR = "triangular"     # Three related assets on same venue
    STATISTICAL = "statistical"   # Correlated assets
    FUTURES_SPOT = "futures_spot" # Futures vs spot price difference
    FUNDING_RATE = "funding_rate" # Funding rate arbitrage in perpetuals
    INDEX_TRACKING = "index_tracking" # Index vs constituents

# Arbitrage opportunity types
ARBITRAGE_OPPORTUNITY_TYPES = [
    "cross_exchange",
    "triangular",
    "statistical",
    "futures_spot",
    "funding_rate",
    "index_tracking"
]
STRATEGY_TYPES = [
    "trend_following", "mean_reversion", "breakout", "momentum", "statistical_arbitrage",
    "market_making", "sentiment_based", "machine_learning", "pattern_recognition",
    "volatility_based", "order_flow", "market_structure", "multi_timeframe",
    "adaptive", "ensemble", "reinforcement_learning", "regime_based",
]

STRATEGY_CATEGORIES = {
    "TREND": "trend",
    "MEAN_REVERSION": "mean_reversion",
    "MOMENTUM": "momentum",
    "BREAKOUT": "breakout",
    "VOLATILITY": "volatility",
    "PATTERN": "pattern",
    "SENTIMENT": "sentiment",
    "ARBITRAGE": "arbitrage",
    "MARKET_MAKING": "market_making",
    "REINFORCEMENT": "reinforcement",
    "ENSEMBLE": "ensemble",
    "ADAPTIVE": "adaptive"
}

EXECUTION_MODES = [
    "live", "paper", "backtest", "simulation", "optimization", "stress_test",
]

SLIPPAGE_MODELS = [
    "fixed", "percentage", "volume_based", "volatility_based", "orderbook_based", "impact_based",
]


# ======================================
# Risk Management Constants
# Risk Management Configuration
# ======================================

class RiskControlMethod(enum.Enum):
    """Risk control methods."""
    FIXED_STOP_LOSS = "fixed_stop_loss"
    TRAILING_STOP = "trailing_stop"
    ATR_STOP = "atr_stop"
    VOLATILITY_STOP = "volatility_stop"
    SUPPORT_RESISTANCE_STOP = "support_resistance_stop"
    TIME_STOP = "time_stop"
    EQUITY_STOP = "equity_stop"
    DRAWDOWN_STOP = "drawdown_stop"


class PositionSizingMethod(enum.Enum):
    """Position sizing methods."""
    FIXED_SIZE = "fixed_size"
    FIXED_VALUE = "fixed_value"
    FIXED_PERCENT = "fixed_percent"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY_CRITERION = "kelly_criterion"
    OPTIMAL_F = "optimal_f"
    RISK_PARITY = "risk_parity"


class ExecutionAlgorithm(enum.Enum):
    """Trade execution algorithms."""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    PEG = "peg"
    SNIPER = "sniper"
    ADAPTIVE = "adaptive"


# Default risk parameters
DEFAULT_RISK_PERCENT_PER_TRADE = 1.0
DEFAULT_MAX_OPEN_TRADES = 5
DEFAULT_MAX_CORRELATED_TRADES = 2
DEFAULT_MAX_DRAWDOWN_PERCENT = 20.0
DEFAULT_PROFIT_FACTOR_THRESHOLD = 1.5
DEFAULT_WIN_RATE_THRESHOLD = 65.0
DEFAULT_TRAILING_STOP_ACTIVATION = 1.0
DEFAULT_KELLY_FRACTION = 0.5
DEFAULT_GROWTH_FACTOR = 1.05
DEFAULT_FIXED_STOP_PERCENTAGE = 2.0
DEFAULT_MIN_STOP_DISTANCE = 0.005
DEFAULT_STOP_LOSS_MULTIPLIER = 1.5
DEFAULT_TAKE_PROFIT_MULTIPLIER = 2.0

# Position management
PARTIAL_CLOSE_LEVELS = [0.25, 0.5, 0.75]
POSITION_SIZE_PRECISION = 4
MAX_LEVERAGE_BINANCE = 125
MAX_LEVERAGE_DERIV = 100


# ======================================
# Technical Analysis Configuration
# ======================================

FIBONACCI_RATIOS = {
    "0": 0.0, "23.6": 0.236, "38.2": 0.382, "50": 0.5, "61.8": 0.618,
    "76.4": 0.764, "78.6": 0.786, "100": 1.0, "127.2": 1.272, "138.2": 1.382,
    "150": 1.5, "161.8": 1.618, "200": 2.0, "223.6": 2.236, "261.8": 2.618,
    "361.8": 3.618, "423.6": 4.236
}

SMA_PERIODS = [10, 20, 50, 100, 200]
EMA_PERIODS = [9, 12, 26, 50, 200]
RSI_PERIODS = [7, 14, 21]
MACD_PARAMS = {"FAST": 12, "SLOW": 26, "SIGNAL": 9}
BOLLINGER_BANDS_PARAMS = {"PERIOD": 20, "STD_DEV": 2}
STOCHASTIC_PARAMS = {"K_PERIOD": 14, "K_SLOWING": 3, "D_PERIOD": 3}

# ======================================
# Notification Constants
# ======================================


# ======================================
# Machine Learning Configuration
# ======================================

class ModelType(enum.Enum):
    """Machine learning model types."""
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    LSTM = "lstm"
    GRU = "gru"
    CNN = "cnn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ENSEMBLE = "ensemble"


class ScalingMethod(enum.Enum):
    """Feature scaling methods."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    MAXABS = "maxabs"
    NONE = "none"


# ======================================
# Notification Configuration
# ======================================

class NotificationType(enum.Enum):
    """Notification types."""
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    STOP_LOSS_HIT = "stop_loss_hit"
    TAKE_PROFIT_HIT = "take_profit_hit"
    NEW_SIGNAL = "new_signal"
    PATTERN_DETECTED = "pattern_detected"
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"


class NotificationPriority(enum.Enum):
    """Notification priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class NotificationChannel(enum.Enum):
    """Notification channels."""
    INTERNAL = "internal"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"


# ======================================
# User Interface Constants
# ======================================

# UI themes
class UITheme(enum.Enum):
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"
    CUSTOM = "custom"


# Chart types
class ChartType(enum.Enum):
    CANDLESTICK = "candlestick"
    LINE = "line"
    AREA = "area"
    BAR = "bar"
    HEIKIN_ASHI = "heikin_ashi"
    RENKO = "renko"
    POINT_AND_FIGURE = "point_and_figure"
    KAGI = "kagi"


# Dashboard layouts
class DashboardLayout(enum.Enum):
    SINGLE = "single"
    DUAL = "dual"
    QUAD = "quad"
    CUSTOM = "custom"
    TRADING_FOCUS = "trading_focus"
    ANALYSIS_FOCUS = "analysis_focus"
    MONITORING_FOCUS = "monitoring_focus"


# Default dashboard components
DEFAULT_DASHBOARD_COMPONENTS = [
    "asset_selector",
    "timeframe_selector",
    "chart_main",
    "order_book",
    "position_summary",
    "recent_trades",
    "strategy_performance",
    "signals_panel",
    "risk_metrics",
    "market_overview"
]

# Export format types
class ExportFormat(enum.Enum):
    CSV = "csv"
    JSON = "json"
    EXCEL = "excel"
    PDF = "pdf"
    PNG = "png"
    HTML = "html"


# Default UI configuration
DEFAULT_UI_CONFIG = {
    "theme": "dark",
    "chart_type": "candlestick",
    "default_timeframe": "1h",
    "default_exchange": "binance",
    "default_layout": "trading_focus",
    "auto_refresh_interval": 5,  # seconds
    "enable_sound_alerts": True,
    "enable_voice_assistant": True,
    "trading_hotkeys_enabled": False,
    "confirm_orders": True,
    "log_level_ui": "info"
}
# ======================================
# Deriv-specific Constants
# ======================================
# Deriv-specific Configuration
# ======================================

MAX_RECONNECT_ATTEMPTS = 5
INITIAL_RECONNECT_DELAY = 1.0
MAX_RECONNECT_DELAY = 60.0
DEFAULT_SUBSCRIPTION_TIMEOUT = 30.0
DEFAULT_PING_INTERVAL = 30.0
MARKET_ORDER_BOOK_DEPTH = 10
DERIV_PRICE_REFRESH_RATE = 1.0

DERIV_ENDPOINTS = {
    "websocket": "wss://ws.binaryws.com/websockets/v3",
    "oauth": "https://oauth.deriv.com",
    "api": "https://api.deriv.com",
}

DERIV_ASSET_CLASSES = {
    "forex": "forex",
    "indices": "indices",
    "commodities": "commodities",
    "synthetic": "synthetic_index"
}

# Deriv markets
DERIV_MARKETS = {
    "forex": ["AUD/JPY", "AUD/USD", "EUR/AUD", "EUR/GBP", "EUR/JPY", "EUR/USD", 
              "GBP/AUD", "GBP/JPY", "GBP/USD", "USD/CAD", "USD/CHF", "USD/JPY"],
    "indices": ["AUS200", "EUROPE50", "FRANCE40", "GERMANY40", "HONGKONG50", 
                "JAPAN225", "NETHERLAND25", "SPAIN35", "UK100", "US30", "US500", "USTECH100"],
    "commodities": ["ALUMINIUM", "COPPER", "GOLD", "PALLADIUM", "PLATINUM", "SILVER"],
    "synthetic": ["BOOM1000", "BOOM500", "CRASH1000", "CRASH500", "CRYPTO", 
                  "JUMP10", "JUMP25", "JUMP50", "JUMP75", "JUMP100", "RANGE_BREAK_100", 
                  "RANGE_BREAK_200", "STEP_INDEX", "VOLATILITY10", "VOLATILITY25", 
                  "VOLATILITY50", "VOLATILITY75", "VOLATILITY100"]
}

# ======================================
# Social Media Constants
# ======================================
# Supported social media platforms
SOCIAL_PLATFORMS = ["twitter", "reddit", "telegram", "discord", "stocktwits", "tradingview"]

# Social media query parameters
SOCIAL_QUERY_PARAMS = {
    "twitter": {
        "max_results": 100,
        "result_type": "mixed",
        "lang": "en",
        "include_entities": True
    },
    "reddit": {
        "limit": 100,
        "sort": "relevance",
        "time_filter": "day"
    },
    "telegram": {
        "limit": 50,
        "offset": 0
    },
    "discord": {
        "limit": 100,
        "before": None,
        "after": None
    },
    "stocktwits": {
        "limit": 30,
        "filter": "all"
    },
    "tradingview": {
        "limit": 50,
        "sort": "recent"
    }
}

SOCIAL_API_KEYS = {
    "twitter": {
        "api_key": "",
        "api_secret": "",
        "bearer_token": ""
    },
    "reddit": {
        "client_id": "",
        "client_secret": "",
        "user_agent": "QuantumSpectre/1.0"
    },
    "telegram": {
        "bot_token": ""
    },
    "discord": {
        "bot_token": ""
    },
    "synthetic": "synthetic_index",
}

SOCIAL_UPDATE_INTERVALS = {
    "twitter": 60,
    "reddit": 300,
    "telegram": 120,
    "discord": 120,
    "stocktwits": 180,
    "tradingview": 300,
}

NLP_MODELS = {
    "sentiment": "ProsusAI/finbert",
    "crypto_sentiment": "ElKulako/cryptobert",
    "ner": "en_core_web_sm",
}

ASSET_KEYWORDS = {
    "BTC/USD": ["bitcoin", "btc", "bitcoin/usd", "btcusd"],
    "ETH/USD": ["ethereum", "eth", "ethereum/usd", "ethusd"],
    "EUR/USD": ["euro", "eur/usd", "eurusd", "euro dollar"],
    "GBP/USD": ["gbp", "pound", "cable", "gbp/usd", "gbpusd"],
    "USD/JPY": ["yen", "jpy", "usdjpy", "dollar yen"],
}


# ======================================
# Feature Engineering
# ======================================

FEATURE_PRIORITY_LEVELS = ["high", "normal", "low"]
DEFAULT_FEATURE_PARAMS = {}


# ======================================
# Helper Lists for Runtime Use
# ======================================

EXCHANGE_TYPES = [ex.value for ex in Exchange]
TIME_FRAMES = [tf.value for tf in Timeframe]
TIMEFRAMES = TIME_FRAMES
ORDER_TYPES = [ot.value for ot in OrderType]
ORDER_SIDES = [side.value for side in OrderSide]
POSITION_SIDES = [ps.value for ps in PositionSide]
ORDER_STATUSES = [ps.value for ps in OrderStatus]
POSITION_STATUSES = [ps.value for ps in PositionStatus]
SIGNAL_STRENGTHS = [ss.value for ss in SignalStrength]


# ======================================
# Export Interface
# ======================================

__all__ = [
    # System constants
    "VERSION", "CONFIG_SCHEMA_VERSION", "SYSTEM_NAME", "AUTHOR", "LICENSE",
    "ENV_PRODUCTION", "ENV_DEVELOPMENT", "ENV_TESTING",
    "DEFAULT_CONFIG_PATH", "DEFAULT_DATA_DIR", "STORAGE_ROOT_PATH",
    "DEFAULT_LOG_DIR", "DEFAULT_MODEL_DIR",
    
    # Exchange and trading enums
    'Exchange', 'AssetClass', 'Timeframe', 'OrderType', 'OrderSide',
    'PositionSide', 'PositionType', 'OrderStatus', 'PositionStatus',
    'TriggerType', 'TimeInForce',
    'SignalDirection', 'SignalStrength', 'MarketRegime', 'StrategyType',
    'RiskLevel', 'FeeType','EXCHANGE_TYPES', 'TIME_FRAMES', 'TIMEFRAMES', 'ORDER_TYPES', 'ORDER_SIDES',
    'ORDER_TYPE', 'ORDER_STATUS', 'TIME_IN_FORCE',
    'POSITION_SIDES', 'ORDER_STATUSES', 'POSITION_STATUSES', 'POSITION_SIDE', 'POSITION_STATUS',


    # Service configuration
    "SERVICE_NAMES", "SERVICE_DEPENDENCIES", "SERVICE_STARTUP_ORDER",
    "DATA_INGEST_METRICS_PREFIX",

    
    # Resource management
    "DEFAULT_THREAD_POOL_SIZE", "MAX_THREAD_POOL_SIZE", "DEFAULT_PROCESS_POOL_SIZE",
    "MAX_PROCESS_POOL_SIZE", "MARKET_DATA_MAX_WORKERS",
    "MEMORY_WARNING_THRESHOLD", "MEMORY_CRITICAL_THRESHOLD",
    "LOG_LEVELS", "DEFAULT_LOG_LEVEL",
    
    # Network configuration
    "API_RATE_LIMIT_DEFAULT", "API_RATE_LIMIT_TRADING", "API_RATE_LIMIT_AUTH",
    "WEBSOCKET_MAX_CONNECTIONS", "WEBSOCKET_PING_INTERVAL", "WEBSOCKET_PING_TIMEOUT",
    "WEBSOCKET_CLOSE_TIMEOUT", "HTTP_TIMEOUT_DEFAULT", "HTTP_TIMEOUT_FEED",
    "HTTP_TIMEOUT_LONG", "HTTP_MAX_RETRIES", "HTTP_RETRY_BACKOFF",
    
    # Security configuration
    "TOKEN_EXPIRY_ACCESS", "TOKEN_EXPIRY_REFRESH", "PASSWORD_MIN_LENGTH",
    "PASSWORD_HASH_ALGORITHM", "PASSWORD_SALT_LENGTH", "PASSWORD_HASH_ITERATIONS",
    
    # Risk management enums
    'RiskControlMethod', 'PositionSizingMethod', 'ExecutionAlgorithm',
    'DEFAULT_RISK_PERCENT_PER_TRADE', 'DEFAULT_MAX_OPEN_TRADES',
    'DEFAULT_MAX_CORRELATED_TRADES', 'DEFAULT_MAX_DRAWDOWN_PERCENT',
    'DEFAULT_PROFIT_FACTOR_THRESHOLD', 'DEFAULT_WIN_RATE_THRESHOLD',
    'DEFAULT_TRAILING_STOP_ACTIVATION', 'DEFAULT_KELLY_FRACTION',
    'DEFAULT_STOP_LOSS_MULTIPLIER', 'DEFAULT_TAKE_PROFIT_MULTIPLIER',
    'POSITION_SIZE_PRECISION', 'DEFAULT_GROWTH_FACTOR',
    'PARTIAL_CLOSE_LEVELS', 'DEFAULT_FIXED_STOP_PERCENTAGE',
    'DEFAULT_MIN_STOP_DISTANCE',
    'MAX_LEVERAGE_BINANCE', 'MAX_LEVERAGE_DERIV',
    'POSITION_SIZE_PRECISION', 'MAX_LEVERAGE_BINANCE', 'MAX_LEVERAGE_DERIV',
    'DEFAULT_GROWTH_FACTOR', 'PARTIAL_CLOSE_LEVELS',
    'DEFAULT_FIXED_STOP_PERCENTAGE', 'DEFAULT_MIN_STOP_DISTANCE',

    'DEFAULT_TRAILING_ACTIVATION_PERCENTAGE', 'DEFAULT_TRAILING_CALLBACK_RATE',
    'MAX_STOP_LEVELS', 'DEFAULT_CHANDELIER_EXIT_MULTIPLIER',


    # Database configuration
    "DATABASE_POOL_MIN_SIZE", "DATABASE_POOL_MAX_SIZE", "DATABASE_MAX_QUERIES",
    "DATABASE_CONNECTION_TIMEOUT", "DATABASE_COMMAND_TIMEOUT",

    
    # Cache configuration
    "CACHE_DEFAULT_TTL", "CACHE_LONG_TTL", "CACHE_VERY_LONG_TTL",
    
    # Trading enums
    "Exchange", "AssetClass", "Timeframe", "OrderType", "OrderSide", "PositionSide",
    "OrderStatus", "PositionStatus", "TimeInForce", "SignalDirection", "SignalStrength",
    "MarketRegime", "RiskLevel",
    
    # Trading platform configuration
    "SUPPORTED_PLATFORMS", "SUPPORTED_ASSETS", "ASSET_TYPES", "STRATEGY_TYPES",
    "EXECUTION_MODES", "SLIPPAGE_MODELS",
    
    # Social media constants
    'SOCIAL_PLATFORMS', 'SOCIAL_API_KEYS', 'SOCIAL_QUERY_PARAMS',
    'SOCIAL_UPDATE_INTERVALS', 'NLP_MODELS', 'ASSET_KEYWORDS',
]

# Markets and assets
MARKETS_OF_INTEREST = [
    "crypto",
    "forex",
    "equities",
    "commodities",
    "indices"
]

# Assets of interest
ASSETS_OF_INTEREST = [
    "BTC/USD",
    "ETH/USD",
    "XRP/USD",
    "ADA/USD",
    "SOL/USD",
    "BNB/USD",
    "USDT/USD",
    "USDC/USD",
    "EUR/USD",
    "GBP/USD",
    "JPY/USD",
    "GOLD/USD",
    "SILVER/USD",
    "OIL/USD",
    "SP500/USD",
    "NASDAQ/USD",
    "DOW/USD"
]

# News sources configuration
NEWS_SOURCES = [
    {"name": "Bloomberg", "url": "https://www.bloomberg.com", "weight": 0.9},
    {"name": "Reuters", "url": "https://www.reuters.com", "weight": 0.9},
    {"name": "Wall Street Journal", "url": "https://www.wsj.com", "weight": 0.85},
    {"name": "Financial Times", "url": "https://www.ft.com", "weight": 0.85},
    {"name": "CNBC", "url": "https://www.cnbc.com", "weight": 0.8},
    {"name": "Yahoo Finance", "url": "https://finance.yahoo.com", "weight": 0.75},
    {"name": "Investopedia", "url": "https://www.investopedia.com", "weight": 0.7},
    {"name": "CoinDesk", "url": "https://www.coindesk.com", "weight": 0.75},
    {"name": "CoinTelegraph", "url": "https://cointelegraph.com", "weight": 0.7}
]

# Dark web forums to monitor
DARK_WEB_FORUMS = [
    {
        "name": "CryptoTalk",
        "url": "cryptotalk_forum.onion",
        "priority": "high",
        "categories": ["trading", "exploits", "leaks", "general"]
    },
    {
        "name": "DarknetMarkets",
        "url": "darknet_markets_forum.onion",
        "priority": "medium",
        "categories": ["marketplace", "reviews", "security"]
    },
    {
        "name": "HackForums",
        "url": "hackforums_darkweb.onion",
        "priority": "high",
        "categories": ["hacking", "data breaches", "exploits", "general"]
    },
    {
        "name": "BlockchainUnderground",
        "url": "blockchain_underground.onion",
        "priority": "high",
        "categories": ["cryptocurrency", "trading", "mining", "security"]
    },
    {
        "name": "TradingSecrets",
        "url": "trading_secrets_forum.onion",
        "priority": "medium",
        "categories": ["insider info", "market manipulation", "trading strategies"]
    }
]
# Dark web feed configuration
DARK_WEB_FEED_CONFIG = {
    "scan_interval": 3600,  # 1 hour in seconds
    "max_posts_per_scan": 100,
    "min_credibility_score": 0.5,
    "target_keywords": [
        "crypto hack", "exchange hack", "stolen coins", "vulnerability", 
        "zero day", "data leak", "market manipulation", "insider trading"
    ],
    "high_priority_keywords": [
        "exchange hack", "stolen coins", "zero day"
    ]
}

# Market impact phrases for sentiment analysis
MARKET_IMPACT_PHRASES = {
    "positive": [
        "bullish", "rally", "surge", "jump", "soar", "gain", "rise", "climb",
        "outperform", "beat expectations", "exceeded forecast", "record high",
        "strong growth", "upgrade", "positive outlook", "buy rating", "recovery",
        "momentum", "breakthrough", "innovative", "partnership", "acquisition",
        "strategic investment", "dividend increase", "stock buyback", "expansion"
    ],
    "negative": [
        "bearish", "plunge", "crash", "tumble", "slump", "drop", "fall", "decline",
        "underperform", "miss expectations", "below forecast", "record low",
        "weak growth", "downgrade", "negative outlook", "sell rating", "recession",
        "slowdown", "layoffs", "cost-cutting", "bankruptcy", "debt concerns",
        "investigation", "lawsuit", "regulatory issues", "dividend cut", "loss"
    ],
    "uncertainty": [
        "volatile", "uncertainty", "unclear", "mixed signals", "cautious",
        "monitoring", "watching closely", "potential impact", "reviewing options",
        "reassessing", "under consideration", "evaluating", "pending decision",
        "regulatory review", "awaiting approval", "conditional", "tentative",
        "experimental", "trial", "testing", "preliminary", "proposed", "expected"
    ]
}
DARK_WEB_SITES = [
    {
        "name": "SecureCrypto Forum",
        "type": "forum",
        "priority": "high",
        "entry_points": ["secure_crypto_hash123.onion"],
        "requires_auth": True
    },
    {
        "name": "DarkLeaks",
        "type": "marketplace",
        "priority": "medium",
        "entry_points": ["darkleaks_data224.onion"],
        "requires_auth": False
    },
    {
        "name": "Shadow Intel",
        "type": "intelligence",
        "priority": "high",
        "entry_points": ["shadow_intel876.onion"],
        "requires_auth": True
    },
    {
        "name": "BlackMarket",
        "type": "marketplace",
        "priority": "medium",
        "entry_points": ["black_market_crypto553.onion"],
        "requires_auth": True
    },
    {
        "name": "Hack Archives",
        "type": "forum",
        "priority": "low",
        "entry_points": ["hack_archives991.onion"],
        "requires_auth": False
    }
]

# Dark web markets to monitor
DARK_WEB_MARKETS = [
    {
        "name": "CryptoMarket",
        "url": "crypto_market_xyz.onion",
        "priority": "high",
        "categories": ["cryptocurrency", "accounts", "data"]
    },
    {
        "name": "BlackMarket",
        "url": "black_market_xyz.onion",
        "priority": "medium",
        "categories": ["cryptocurrency", "accounts", "cards"]
    },
    {
        "name": "DarkTrade",
        "url": "dark_trade_xyz.onion",
        "priority": "high",
        "categories": ["cryptocurrency", "accounts", "data", "exploits"]
    },
    {
        "name": "ShadowSales",
        "url": "shadow_sales_xyz.onion",
        "priority": "medium",
        "categories": ["cryptocurrency", "accounts", "services"]
    },
    {
        "name": "UndergroundMarket",
        "url": "underground_market_xyz.onion",
        "priority": "low",
        "categories": ["cryptocurrency", "accounts", "data", "services"]
    }
]


# Keywords to determine relevance of dark web content
DARK_WEB_RELEVANCE_KEYWORDS = {
    "high_priority": [
        "exchange hack", "zero day", "exploit", "stolen coins", "private key",
        "data breach", "leaked database", "insider trading", "market manipulation",
        "pump and dump scheme", "flash crash", "database dump"
    ],
    "medium_priority": [
        "ransomware", "malware", "phishing", "scam", "vulnerability",
        "sensitive data", "backdoor", "credential", "kyc data", "trading bot",
        "algorithm", "trading strategy", "market making", "arbitrage", "high frequency"
    ],
    "low_priority": [
        "cryptocurrency", "blockchain", "bitcoin", "ethereum", "trading",
        "wallet", "exchange", "defi", "nft", "mining", "token", "ico",
        "altcoin", "whale", "liquidity", "volatility"
    ]
}

# Dark web scanning settings
DARK_WEB_SCAN_INTERVAL = 3600  # 1 hour in seconds

# Sentiment analysis constants
SENTIMENT_SOURCE_WEIGHTS = {
    "news": 1.0,
    "social_media": 0.8,
    "forum": 0.6,
    "blog": 0.7,
    "analyst_report": 1.2,
    "company_announcement": 1.3,
    "regulatory_filing": 1.5,
    "dark_web": 0.5
}

# Volume analysis constants
VOLUME_ZONE_SIGNIFICANCE = {
    "high": 0.8,
    "medium": 0.5,
    "low": 0.2
}

# User agent strings for web requests
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
]
# Dark web scanning settings
DARK_WEB_SCAN_INTERVAL = 3600  # 1 hour in seconds
SCAN_THREAD_COUNT = 5  # Number of threads for scanning

# Sentiment analysis constants
SENTIMENT_SOURCE_WEIGHTS = {
    "news": 1.0,
    "social_media": 0.8,
    "forum": 0.6,
    "blog": 0.7,
    "analyst_report": 1.2,
    "company_announcement": 1.3,
    "regulatory_filing": 1.5,
    "dark_web": 0.5
}
SENTIMENT_IMPACT_WINDOW = {
    "breaking_news": 24,  # hours
    "regular_news": 48,
    "social_media": 12,
    "analyst_report": 72,
    "earnings_report": 120,
    "regulatory_announcement": 168
}

# Sentiment entities for tracking
SENTIMENT_ENTITIES = [
    "market", "economy", "company", "sector", "industry", "product", 
    "regulation", "policy", "central_bank", "interest_rates", "inflation",
    "growth", "recession", "earnings", "revenue", "profit", "loss", 
    "merger", "acquisition", "ipo", "bankruptcy", "scandal", "lawsuit"
]

# Technical indicator parameters
TECHNICAL_INDICATOR_PARAMS = {
    "sma": {"windows": [5, 10, 20, 50, 100, 200]},
    "ema": {"windows": [5, 10, 20, 50, 100, 200]},
    "rsi": {"window": 14, "overbought": 70, "oversold": 30},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger_bands": {"window": 20, "num_std_dev": 2},
    "stochastic": {"k_period": 14, "d_period": 3, "slowing": 3},
    "adx": {"window": 14, "threshold": 25},
    "atr": {"window": 14},
    "cci": {"window": 20},
    "obv": {},
    "ichimoku": {
        "tenkan_period": 9,
        "kijun_period": 26,
        "senkou_b_period": 52,
        "chikou_period": 26
    }
}

# Volatility indicator parameters
VOLATILITY_INDICATOR_PARAMS = {
    "atr": {"window": 14},
    "bollinger_bandwidth": {"window": 20, "num_std_dev": 2},
    "historical_volatility": {"window": 20, "trading_periods": 252},
    "keltner_channels": {"ema_period": 20, "atr_period": 10, "atr_multiplier": 2},
    "true_range": {},
    "average_true_range_percent": {"window": 14}
}

# Volume analysis constants
VOLUME_PROFILE_BINS = 20  # Number of bins for volume profile calculation
VOLUME_ZONE_SIGNIFICANCE = {
    "high": 0.8,
    "medium": 0.5,
    "low": 0.2
}

# If ASSETS is also required, add it:
ASSETS = [
    'BTC/USDT',
    'ETH/USDT',
    'BNB/USDT',
    # Add other assets as needed
]
# Sentiment analysis constants
SENTIMENT_DECAY_FACTOR = 0.9  # Decay factor for sentiment over time

# Volume analysis constants
VOLUME_PROFILE_BINS = 20  # Number of bins for volume profile calculation
# Maximum number of regimes to consider
MAX_REGIMES = 5
# General retry configurations
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # Initial delay in seconds
# Define regime scan interval (in seconds)
REGIME_SCAN_INTERVAL = 3600  # Default to hourly scans
REGIME_LOOKBACK_PERIODS = 90  # Number of periods to look back for regime detection
# Default logging level
DEFAULT_LOG_LEVEL = "INFO"

DEFAULT_EXCHANGE_CONFIGS = {
    "binance": {
        "enabled": True,
        "rate_limit": 1200,
        "mode": "paper",
        "assets": SUPPORTED_ASSETS["binance"]
    },
    "deriv": {
        "enabled": True,
        "rate_limit": 60,
        "mode": "paper",
        "assets": SUPPORTED_ASSETS["deriv"]
    }
}

# Intelligence module configuration
INTELLIGENCE_MODULES = [
    "pattern_recognition",
    "loophole_detection",
    "adaptive_learning"
]

DEFAULT_MODULE_CONFIG = {
    "enabled": True,
    "priority": "normal",
    "max_threads": 4,
    "max_memory_mb": 1024,
    "cache_ttl_seconds": 300
}

# Additional constants needed by intelligence/app.py
PATTERN_RECOGNITION_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
LOOPHOLE_DETECTION_INTERVAL = 300  # seconds
ASSET_BLACKLIST = []
ADAPTIVE_LEARNING_SCHEDULE = {
    "interval_minutes": 60,
    "full_retrain_hours": 24
}
ONLINE_LEARNING_INTERVAL = 300  # seconds between online learning updates
MODEL_SAVE_INTERVAL = 3600      # seconds between model persistence operations
MIN_SAMPLES_FOR_TRAINING = 50
FEATURE_IMPORTANCE_THRESHOLD = 0.01
LEARNING_RATES = {"default": 0.001, "fine_tune": 0.0001}
REINFORCEMENT_DECAY_FACTOR = 0.99
BATCH_SIZES = {"default": 32, "large": 128}
DEFAULT_SERVICE_TIMEOUT = 60  # seconds
MARKET_REGIMES = ["trending_bullish", "trending_bearish", "ranging", "volatile", "choppy", "breakout", "reversal"]
SIGNAL_CONFIDENCE_LEVELS = {
    "VERY_LOW": 1,
    "LOW": 2,
    "MEDIUM": 3,
    "HIGH": 4,
    "VERY_HIGH": 5,
    "EXCEPTIONAL": 6
}

# Bayesian optimization kernels for adaptive learning
BAYESIAN_OPT_KERNELS = {
    "RBF": "rbf",
    "MATERN": "matern",
    "RATIONAL_QUADRATIC": "rational_quadratic",
    "EXP_SINE_SQUARED": "exp_sine_squared",
    "DOT_PRODUCT": "dot_product",
    "CONSTANT": "constant",
    "WHITE": "white"
}

# Default Gaussian Process kernel for adaptive learning
DEFAULT_GP_KERNEL = "matern"

# Acquisition functions for Bayesian optimization
ACQUISITION_FUNCTIONS = {
    "UCB": "upper_confidence_bound",
    "EI": "expected_improvement",
    "PI": "probability_improvement",
    "LCB": "lower_confidence_bound",
    "RANDOM": "random"
}

# Default acquisition function for Bayesian optimization
DEFAULT_ACQUISITION = "EI"

# Compression settings
COMPRESSION_LEVEL = 9  # Maximum compression level (0-9)

# Pattern recognition constants
PATTERN_COMPLETION_THRESHOLD = 0.75  # Threshold for pattern completion (0.0-1.0)
PATTERN_STRENGTH_LEVELS = {
    "WEAK": 0.3,
    "MODERATE": 0.5,
    "STRONG": 0.7,
    "VERY_STRONG": 0.9
}
MIN_PATTERN_BARS = 5  # Minimum number of bars for pattern detection
MAX_PATTERN_BARS = 100  # Maximum number of bars for pattern detection

# Support and resistance constants
SUPPORT_RESISTANCE_METHODS = [
    "price_levels",
    "moving_average",
    "fibonacci",
    "pivot_points",
    "volume_profile",
    "fractal",
    "regression"
]
ZONE_CONFIDENCE_LEVELS = {
    "LOW": 0.3,
    "MEDIUM": 0.6,
    "HIGH": 0.85,
    "VERY_HIGH": 0.95
}
ZONE_TYPES = [
    "support",
    "resistance",
    "dynamic_support",
    "dynamic_resistance",
    "demand",
    "supply"
]

# Optimization constants
OPTIMIZATION_DIRECTION = {
    "MAXIMIZE": 1,
    "MINIMIZE": -1
}
DEFAULT_PARAM_BOUNDS = {
    "continuous": (0.0, 1.0),
    "integer": (1, 100),
    "categorical": None
}
MAX_PARALLEL_EVALUATIONS = 8
GP_RANDOM_RESTARTS = 5

# Platform constants
PLATFORMS = ["windows", "linux", "macos"]

# Data retention policies
MARKET_DATA_RETENTION_POLICY = {
    "1m": 7,    # 7 days for 1-minute data
    "5m": 30,   # 30 days for 5-minute data
    "15m": 60,  # 60 days for 15-minute data
    "1h": 180,  # 180 days for 1-hour data
    "4h": 365,  # 365 days for 4-hour data
    "1d": 1825, # 5 years for daily data
    "1w": 3650  # 10 years for weekly data
}

# Intelligence module constants
INTELLIGENCE_MODULES = [
    "pattern_recognition",
    "loophole_detection",
    "adaptive_learning"
]

DEFAULT_MODULE_CONFIG = {
    "enabled": True,
    "priority": "normal",
    "max_threads": 4,
    "max_memory_mb": 1024,
    "cache_ttl_seconds": 300
}

PATTERN_RECOGNITION_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# Signal types and directions
SIGNAL_TYPES = {
    "ENTRY": "entry",
    "EXIT": "exit",
    "STOP_LOSS": "stop_loss",
    "TAKE_PROFIT": "take_profit",
    "TRAILING_STOP": "trailing_stop",
    "POSITION_SIZE": "position_size",
    "RISK_ADJUSTMENT": "risk_adjustment"
}

# Trading action constants
ACTION_BUY = "buy"
ACTION_SELL = "sell"
ACTION_HOLD = "hold"
ACTION_CLOSE = "close"

POSITION_DIRECTION = {
    "LONG": "long",
    "SHORT": "short",
    "NEUTRAL": "neutral"
}

# Risk management constants
RISK_MANAGER_UPDATE_INTERVAL = 60  # seconds
CIRCUIT_BREAKER_THRESHOLDS = {
    "VOLATILITY": 3.0,  # Standard deviations above normal
    "DRAWDOWN": 0.15,   # 15% drawdown
    "LOSS_STREAK": 5,   # 5 consecutive losses
    "PROFIT_DEVIATION": 0.3  # 30% deviation from expected profit
}

DRAWDOWN_PROTECTION_LEVELS = {
    "WARNING": 0.10,    # 10% drawdown
    "REDUCE_RISK": 0.15, # 15% drawdown
    "STOP_TRADING": 0.20 # 20% drawdown
}

# Default risk allocation used by drawdown protection
DEFAULT_RISK_PERCENTAGE = 0.02  # 2% risk per trade

# Thresholds for adjusting risk during recovery modes
RECOVERY_MODE_THRESHOLDS = {
    "CONSERVATIVE": 0.25,
    "BALANCED": 0.50,
    "AGGRESSIVE": 0.75,
}

MAX_ALLOWED_DRAWDOWN = 0.25  # 25% maximum allowed drawdown

EXPOSURE_LIMITS = {
    "MAX_PER_ASSET": 0.20,      # 20% of portfolio per asset
    "MAX_PER_SECTOR": 0.40,     # 40% of portfolio per sector
    "MAX_CORRELATED_ASSETS": 0.60, # 60% of portfolio in correlated assets
    "MAX_LEVERAGE": 3.0         # 3x maximum leverage
}

DEFAULT_MAX_RISK_PER_TRADE = 0.02  # 2% of account per trade
DEFAULT_BASE_POSITION_SIZE = 0.01  # 1% of account as base position size
MAX_LEVERAGE_BINANCE = 125
MAX_LEVERAGE_DERIV = 100
DEFAULT_STOP_LOSS_MULTIPLIER = 1.5
DEFAULT_TAKE_PROFIT_MULTIPLIER = 2.0
POSITION_SIZE_PRECISION = 4
DEFAULT_GROWTH_FACTOR = 1.0
PARTIAL_CLOSE_LEVELS = [0.25, 0.5, 0.75]
DEFAULT_FIXED_STOP_PERCENTAGE = 1.0
DEFAULT_MIN_STOP_DISTANCE = 0.5

DEFAULT_GROWTH_FACTOR = 1.05
PARTIAL_CLOSE_LEVELS = [0.25, 0.5, 0.75]
DEFAULT_FIXED_STOP_PERCENTAGE = 0.02
DEFAULT_MIN_STOP_DISTANCE = 0.001
MAX_POSITION_CORRELATION = 0.7  # Maximum allowed correlation between positions
CORRELATION_LOOKBACK_PERIODS = 100  # Periods to look back for correlation calculation

DEFAULT_ATR_PERIODS = 14  # Default periods for ATR calculation
DEFAULT_ATR_MULTIPLIER = 2.0  # Default multiplier for ATR-based stops

# Additional risk parameters
DEFAULT_GROWTH_FACTOR = 1.2
PARTIAL_CLOSE_LEVELS = [0.5, 0.75]
DEFAULT_FIXED_STOP_PERCENTAGE = 1.0  # percent
DEFAULT_MIN_STOP_DISTANCE = 0.005  # 0.5% of entry price
DEFAULT_TRAILING_ACTIVATION_PERCENTAGE = 0.5
DEFAULT_TRAILING_CALLBACK_RATE = 0.25
MAX_STOP_LEVELS = 3
DEFAULT_CHANDELIER_EXIT_MULTIPLIER = 3.0

RECOVERY_STRATEGIES = {
    "REDUCE_POSITION_SIZE": "reduce_position_size",
    "INCREASE_WIN_RATE": "increase_win_rate",
    "REDUCE_TRADING_FREQUENCY": "reduce_trading_frequency",
    "SWITCH_STRATEGY": "switch_strategy",
    "PAUSE_TRADING": "pause_trading"
}

ACCOUNT_STATES = {
    "HEALTHY": "healthy",
    "WARNING": "warning",
    "CRITICAL": "critical",
    "RECOVERY": "recovery"
}

# Order and position constants
ORDER_TYPE_MAP = {
    "MARKET": "market",
    "LIMIT": "limit",
    "STOP": "stop",
    "STOP_LIMIT": "stop_limit",
    "TAKE_PROFIT": "take_profit",
    "TAKE_PROFIT_LIMIT": "take_profit_limit",
    "TRAILING_STOP": "trailing_stop"
}

ORDER_SIDE_MAP = {
    "BUY": "buy",
    "SELL": "sell"
}

ORDER_STATUS_MAP = {
    "NEW": "new",
    "PARTIALLY_FILLED": "partially_filled",
    "FILLED": "filled",
    "CANCELED": "canceled",
    "REJECTED": "rejected",
    "EXPIRED": "expired"
}

POSITION_SIDE_MAP = {
    "LONG": "long",
    "SHORT": "short"
}

POSITION_STATUS_MAP = {
    "OPEN": "open",
    "CLOSED": "closed",
    "PARTIALLY_CLOSED": "partially_closed"
}

TIME_IN_FORCE_MAP = {
    "GTC": "gtc",  # Good Till Canceled
    "IOC": "ioc",  # Immediate Or Cancel
    "FOK": "fok",  # Fill Or Kill
    "GTD": "gtd"   # Good Till Date
}

# Execution engine constants
EXECUTION_COOLDOWN_MS = 100  # Milliseconds between execution attempts
MAX_EXECUTION_TIME_MS = 5000  # Maximum execution time in milliseconds
MAX_RETRY_ATTEMPTS = 3  # Maximum number of retry attempts for execution

TICK_SIZE_MAPPING = {
    "BTC/USD": 0.5,
    "ETH/USD": 0.05,
    "EUR/USD": 0.00001,
    "GBP/USD": 0.00001,
    "USD/JPY": 0.001,
    "DEFAULT": 0.00001
}

LIQUIDITY_THRESHOLDS = {
    "LOW": 10000,
    "MEDIUM": 100000,
    "HIGH": 1000000
}

ORDER_BOOK_LEVELS = 10  # Number of order book levels to track

FLOW_IMBALANCE_WINDOW = 100  # Number of recent trades to measure flow imbalance
SPREAD_ANALYSIS_WINDOW = 100  # Window size for spread behavior analysis
MICROSTRUCTURE_PATTERN_LOOKBACK = 500  # Lookback period for pattern detection

# Machine learning constants
ML_MODEL_TYPES = {
    "CLASSIFICATION": "classification",
    "REGRESSION": "regression",
    "REINFORCEMENT": "reinforcement",
    "UNSUPERVISED": "unsupervised",
    "ENSEMBLE": "ensemble"
}

FEATURE_IMPORTANCE_METHODS = {
    "PERMUTATION": "permutation",
    "SHAP": "shap",
    "FEATURE_IMPORTANCE": "feature_importance"
}

MODEL_SAVE_PATH = "./models"
TRADING_CUTOFFS = {
    "MIN_CONFIDENCE": 0.65,
    "MIN_ACCURACY": 0.60,
    "MIN_SHARPE": 0.5
}

MODEL_REGISTRY_CONFIG = {
    "path": "./models/registry",
    "backup_frequency": 24,  # hours
    "max_versions": 5
}
MODEL_REGISTRY_KEYS = ["name", "version", "path"]

FEATURE_IMPORTANCE_CONFIG = {
    "n_permutations": 10,
    "n_repeats": 3,
    "random_state": 42
}

DEFAULT_LOOKBACK_PERIODS = 100
DEFAULT_CONFIDENCE_LEVELS = {
    "LOW": 0.3,
    "MEDIUM": 0.5,
    "HIGH": 0.7,
    "VERY_HIGH": 0.9
}

DEFAULT_MODEL_TYPES = ["gradient_boosting", "random_forest", "neural_network"]
DEFAULT_PAIR_CORR_THRESHOLD = 0.7

REWARD_FUNCTIONS = {
    "SHARPE": "sharpe",
    "SORTINO": "sortino",
    "PROFIT": "profit",
    "CALMAR": "calmar"
}

# Genetic algorithm constants
GENETIC_POPULATION_SIZE = 50
GENETIC_GENERATIONS = 30

# GPU settings
GPU_MEMORY_LIMIT = 0.8  # 80% of GPU memory
GPU_MEMORY_GROWTH = True
DEFAULT_GPU_ID = 0

# Strategy constants
RISK_LEVELS = {
    "VERY_LOW": 1,
    "LOW": 2,
    "MEDIUM": 3,
    "HIGH": 4,
    "VERY_HIGH": 5
}

TREND_FOLLOWING_CONFIG = {
    "fast_period": 20,
    "slow_period": 50,
    "signal_period": 9,
    "atr_period": 14,
    "atr_multiplier": 2.0
}

DIRECTIONAL_BIAS_THRESHOLD = 0.6
FILTER_STRENGTH_LEVELS = {
    "WEAK": 0.3,
    "MODERATE": 0.5,
    "STRONG": 0.7,
    "VERY_STRONG": 0.9
}

SWING_TRADING_CONFIG = {
    "swing_detection_periods": 20,
    "min_swing_size": 0.01,
    "max_swing_lookback": 100
}

MAX_SCALP_DURATION = 60 * 60  # 1 hour in seconds

# Monitoring constants
MONITORING_CONFIG = {
    "log_level": "info",
    "metrics_interval": 60,  # seconds
    "health_check_interval": 300,  # seconds
    "alert_channels": ["console", "email"]
}

SERVICE_STATUS = {
    "STARTING": "starting",
    "RUNNING": "running",
    "DEGRADED": "degraded",
    "STOPPED": "stopped",
    "ERROR": "error"
}

DASHBOARD_SECTIONS = [
    "overview",
    "performance",
    "risk",
    "signals",
    "positions",
    "system"
]

METRICS_CATEGORIES = {
    "PERFORMANCE": "performance",
    "RISK": "risk",
    "SYSTEM": "system",
    "TRADING": "trading"
}

TRADING_METRIC_THRESHOLDS = {
    "win_rate": {
        "warning": 0.4,
        "critical": 0.3
    },
    "profit_factor": {
        "warning": 1.2,
        "critical": 1.0
    },
    "drawdown": {
        "warning": 0.15,
        "critical": 0.25
    }
}

METRIC_TYPES = {
    "COUNTER": "counter",
    "GAUGE": "gauge",
    "HISTOGRAM": "histogram",
    "SUMMARY": "summary"
}

PERFORMANCE_METRICS = [
    "win_rate",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "average_profit",
    "average_loss"
]

SYSTEM_METRICS = [
    "cpu_usage",
    "memory_usage",
    "disk_usage",
    "network_latency",
    "request_count",
    "error_count"
]

METRIC_PRIORITIES = {
    "HIGH": "high",
    "MEDIUM": "medium",
    "LOW": "low"
}

LOG_PATTERNS = {
    "ERROR": r"(?i)(error|exception|fail|traceback)",
    "WARNING": r"(?i)(warning|warn|deprecated)",
    "CRITICAL": r"(?i)(critical|fatal|crash)"
}

# Alerting constants
ALERT_LEVELS = {
    "INFO": "info",
    "WARNING": "warning",
    "ERROR": "error",
    "CRITICAL": "critical"
}

ALERT_TYPES = {
    "SYSTEM": "system",
    "TRADING": "trading",
    "SECURITY": "security",
    "PERFORMANCE": "performance"
}

ALERT_CHANNELS = {
    "CONSOLE": "console",
    "EMAIL": "email",
    "SMS": "sms",
    "WEBHOOK": "webhook",
    "PUSH": "push"
}

# Loophole detection constants
INEFFICIENCY_DETECTION_THRESHOLDS = {
    "price_deviation": 0.02,  # 2% price deviation
    "volume_spike": 3.0,      # 3x normal volume
    "bid_ask_spread": 0.01,   # 1% bid-ask spread
    "order_book_imbalance": 3.0  # 3x imbalance
}

MAX_INEFFICIENCY_AGE = 3600  # Maximum age of inefficiency in seconds

MIN_PROFITABLE_SPREAD_PERCENT = 0.5  # 0.5% minimum profitable spread
MIN_LIQUIDITY_REQUIREMENTS = {
    "BTC/USD": 5.0,    # 5 BTC minimum liquidity
    "ETH/USD": 50.0,   # 50 ETH minimum liquidity
    "EUR/USD": 100000, # 100K EUR minimum liquidity
    "DEFAULT": 1000    # Default minimum liquidity
}

# Exchange fee structures
EXCHANGE_FEE_STRUCTURES = {
    "binance": {
        "maker": 0.001,  # 0.1% maker fee
        "taker": 0.001,  # 0.1% taker fee
        "withdrawal": {
            "BTC": 0.0005,
            "ETH": 0.005,
            "USDT": 25.0,
            "DEFAULT": 0.01
        }
    },
    "deriv": {
        "maker": 0.0,    # 0% maker fee
        "taker": 0.0025, # 0.25% taker fee
        "withdrawal": {
            "BTC": 0.0008,
            "ETH": 0.01,
            "DEFAULT": 0.02
        }
    }
}

# Default confidence threshold for pattern detection
DEFAULT_CONFIDENCE_THRESHOLD = 0.65

# Genetic algorithm parameters
GENETIC_MUTATION_RATE = 0.05
GENETIC_CROSSOVER_RATE = 0.7
GENETIC_POPULATION_SIZE = 100
GENETIC_ELITE_SIZE = 5
GENETIC_MAX_GENERATIONS = 50
GENETIC_SELECTION_PRESSURE = 1.5
GENETIC_ELITISM_RATE = 0.1
GENETIC_TOURNAMENT_SIZE = 3
MAX_THREADS = 8
MIN_FITNESS = 0.0

# Network constants
NETWORK_IDS = {
    "ETHEREUM": 1,
    "BINANCE_SMART_CHAIN": 56,
    "POLYGON": 137,
    "ARBITRUM": 42161,
    "OPTIMISM": 10,
    "AVALANCHE": 43114
}

# HTTP constants
HTTP_SUCCESS_CODES = [200, 201, 202, 203, 204, 205, 206]
HTTP_RETRY_CODES = [408, 429, 500, 502, 503, 504]
HTTP_FATAL_CODES = [400, 401, 403, 404, 405, 406, 409, 410]

# WebSocket constants
WS_DEFAULT_PING_INTERVAL = 30  # seconds
WS_DEFAULT_PING_TIMEOUT = 10   # seconds

# Feed constants
EXCHANGE_NAMES = [
    "binance",
    "coinbase",
    "kraken",
    "deriv",
    "oanda"
    # Risk management
    "RiskControlMethod", "PositionSizingMethod", "ExecutionAlgorithm",
    "DEFAULT_RISK_PERCENT_PER_TRADE", "DEFAULT_MAX_OPEN_TRADES", "DEFAULT_MAX_CORRELATED_TRADES",
    "DEFAULT_MAX_DRAWDOWN_PERCENT", "DEFAULT_PROFIT_FACTOR_THRESHOLD", "DEFAULT_WIN_RATE_THRESHOLD",
    "DEFAULT_TRAILING_STOP_ACTIVATION", "DEFAULT_KELLY_FRACTION", "DEFAULT_GROWTH_FACTOR",
    "DEFAULT_FIXED_STOP_PERCENTAGE", "DEFAULT_MIN_STOP_DISTANCE",
    "DEFAULT_STOP_LOSS_MULTIPLIER", "DEFAULT_TAKE_PROFIT_MULTIPLIER",
    "PARTIAL_CLOSE_LEVELS", "POSITION_SIZE_PRECISION", "MAX_LEVERAGE_BINANCE", "MAX_LEVERAGE_DERIV",
    
    # Technical analysis
    "FIBONACCI_RATIOS", "SMA_PERIODS", "EMA_PERIODS", "RSI_PERIODS",
    "MACD_PARAMS", "BOLLINGER_BANDS_PARAMS", "STOCHASTIC_PARAMS",
    
    # Machine learning
    "ModelType", "ScalingMethod",
    
    # Notifications
    "NotificationType", "NotificationPriority", "NotificationChannel",
    
    # Deriv configuration
    "MAX_RECONNECT_ATTEMPTS", "INITIAL_RECONNECT_DELAY", "MAX_RECONNECT_DELAY",
    "DEFAULT_SUBSCRIPTION_TIMEOUT", "DEFAULT_PING_INTERVAL", "MARKET_ORDER_BOOK_DEPTH",
    "DERIV_PRICE_REFRESH_RATE", "DERIV_ENDPOINTS", "DERIV_ASSET_CLASSES",
    
    # Feature engineering
    "FEATURE_PRIORITY_LEVELS", "DEFAULT_FEATURE_PARAMS",
    
    # Helper lists
    "EXCHANGE_TYPES", "TIME_FRAMES", "ORDER_TYPES", "ORDER_SIDES", "POSITION_SIDES",
    "ORDER_STATUSES", "POSITION_STATUSES", "SIGNAL_STRENGTHS",
]

FEED_TYPES = {
    "MARKET_DATA": "market_data",
    "NEWS": "news",
    "SOCIAL": "social",
    "ONCHAIN": "onchain",
    "DARK_WEB": "dark_web",
    "REGIME": "regime"
}

FEED_STATUS = {
    "CONNECTED": "connected",
    "DISCONNECTED": "disconnected",
    "RECONNECTING": "reconnecting",
    "ERROR": "error"
}


class DataSourcePreference(enum.Enum):
    """Preference order for selecting data sources."""

    EXCHANGE_ONLY = "exchange_only"
    DB_ONLY = "db_only"
    DB_FIRST = "db_first"
    API_ONLY = "api_only"
    SIMULATION = "simulation"


DataProvider = DataSourcePreference

# Data constants
DATA_SOURCES = {
    "EXCHANGE": "exchange",
    "DATABASE": "database",
    "FILE": "file",
    "API": "api",
    "SIMULATION": "simulation"
}


class DataSourcePreference(enum.Enum):
    """Preferred order for fetching historical data."""

    DB_FIRST = "db_first"
    FEED_FIRST = "feed_first"

SYSTEM_COMPONENT_TYPES = {
    "SERVICE": "service",
    "DATABASE": "database",
    "CACHE": "cache",
    "API": "api",
    "UI": "ui"
}

# Voting and confidence constants
VOTE_THRESHOLDS = {
    "STRONG_CONSENSUS": 0.8,
    "CONSENSUS": 0.6,
    "MAJORITY": 0.5,
    "PLURALITY": 0.4
}

COUNCIL_WEIGHTS = {
    "MASTER": 1.0,
    "ASSET": 0.8,
    "REGIME": 0.9,
    "TIMEFRAME": 0.7
}

CONFIDENCE_LEVELS = {
    "VERY_LOW": 0.2,
    "LOW": 0.4,
    "MEDIUM": 0.6,
    "HIGH": 0.8,
    "VERY_HIGH": 0.95
}

MIN_CONFIDENCE_THRESHOLD = 0.5

# Scenario constants
VOLATILITY_LEVELS = {
    "VERY_LOW": 0.5,
    "LOW": 0.75,
    "NORMAL": 1.0,
    "HIGH": 1.5,
    "VERY_HIGH": 2.0,
    "EXTREME": 3.0
}

SCENARIO_TYPES = {
    "HISTORICAL": "historical",
    "SYNTHETIC": "synthetic",
    "STRESS_TEST": "stress_test",
    "MONTE_CARLO": "monte_carlo",
    "CUSTOM": "custom"
}

# Voice advisor constants
VOICE_ADVISOR_MODES = {
    "CONCISE": "concise",
    "DETAILED": "detailed",
    "TECHNICAL": "technical",
    "BEGINNER": "beginner"
}

# Window size constants
DEFAULT_WINDOW_SIZES = {
    "VERY_SHORT": 5,
    "SHORT": 20,
    "MEDIUM": 50,
    "LONG": 100,
    "VERY_LONG": 200
}

# Feature transformer constants
FEATURE_TRANSFORMERS = {
    "NORMALIZER": "normalizer",
    "SCALER": "scaler",
    "ENCODER": "encoder",
    "FILTER": "filter",
    "AGGREGATOR": "aggregator"
}

# Signal confidence constants
SignalConfidence = {
    "VERY_LOW": 0.2,
    "LOW": 0.4,
    "MEDIUM": 0.6,
    "HIGH": 0.8,
    "VERY_HIGH": 0.95
}

# Signal type enum
SignalType = {
    "ENTRY": "entry",
    "EXIT": "exit",
    "STOP_LOSS": "stop_loss",
    "TAKE_PROFIT": "take_profit"
}

# Order type mapping
ORDER_TYPE_LEGACY_MAP = {
    "MARKET": "market",
    "LIMIT": "limit",
    "STOP": "stop",
    "STOP_LIMIT": "stop_limit",
}

ORDER_TYPE_DICT = ORDER_TYPE_LEGACY_MAP

# Order side mapping
ORDER_SIDE_LEGACY_MAP = {
    "BUY": "buy",
    "SELL": "sell",
}

ORDER_SIDE_DICT = ORDER_SIDE_LEGACY_MAP

# Asset classes
ASSET_CLASSES = {
    "CRYPTO": "crypto",
    "FOREX": "forex",
    "STOCKS": "stocks",
    "INDICES": "indices",
    "COMMODITIES": "commodities",
    "FUTURES": "futures",
    "OPTIONS": "options"
}

# Data processing constants
MARKET_DATA_CHUNK_SIZE = 1000  # Number of candles to process in one chunk

# Signal strength constants
SIGNAL_STRENGTHS = {
    "VERY_WEAK": 0.2,
    "WEAK": 0.4,
    "MODERATE": 0.6,
    "STRONG": 0.8,
    "VERY_STRONG": 0.95
}

# Data processing worker constants
MARKET_DATA_MAX_WORKERS = 8

# User roles for authentication and authorization
USER_ROLES = {
    "ADMIN": "admin",
    "TRADER": "trader",
    "ANALYST": "analyst",
    "VIEWER": "viewer",
    "SYSTEM": "system",
    "API": "api"
}

# Data priority levels for processing
DATA_PRIORITY_LEVELS = {
    "CRITICAL": 0,
    "HIGH": 1,
    "NORMAL": 2,
    "LOW": 3,
    "BACKGROUND": 4
}

# Exchange-specific order types
BINANCE_ORDER_TYPES = {
    "MARKET": "MARKET",
    "LIMIT": "LIMIT",
    "STOP_LOSS": "STOP_LOSS",
    "STOP_LOSS_LIMIT": "STOP_LOSS_LIMIT",
    "TAKE_PROFIT": "TAKE_PROFIT",
    "TAKE_PROFIT_LIMIT": "TAKE_PROFIT_LIMIT",
    "LIMIT_MAKER": "LIMIT_MAKER"
}

DERIV_ORDER_TYPES = {
    "MARKET": "MARKET",
    "LIMIT": "LIMIT",
    "STOP": "STOP",
    "STOP_LIMIT": "STOP_LIMIT"
}

# Order status mappings
BINANCE_ORDER_STATUS_MAP = {
    "NEW": OrderStatus.NEW.value,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED.value,
    "FILLED": OrderStatus.FILLED.value,
    "CANCELED": OrderStatus.CANCELED.value,
    "PENDING_CANCEL": OrderStatus.PENDING_CANCEL.value,
    "REJECTED": OrderStatus.REJECTED.value,
    "EXPIRED": OrderStatus.EXPIRED.value
}

DERIV_ORDER_STATUS_MAP = {
    "open": OrderStatus.NEW.value,
    "pending": OrderStatus.NEW.value,
    "filled": OrderStatus.FILLED.value,
    "partially_filled": OrderStatus.PARTIALLY_FILLED.value,
    "cancelled": OrderStatus.CANCELED.value,
    "rejected": OrderStatus.REJECTED.value,
    "expired": OrderStatus.EXPIRED.value
}

# Execution parameters
MAX_SLIPPAGE_PERCENT = 0.5  # Maximum allowed slippage in percent
MAX_RETRY_ATTEMPTS = 3      # Maximum number of retry attempts for failed orders

