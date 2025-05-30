#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Exception Hierarchy

This module provides a comprehensive exception hierarchy for the QuantumSpectre Elite Trading System,
with specialized exceptions for different error scenarios.
"""


# ======================================
# Base Exception Classes
# ======================================

class QuantumSpectreError(Exception):
    """Base exception for all QuantumSpectre system errors."""
    pass


class SystemCriticalError(QuantumSpectreError):
    """Raised for critical system errors that require immediate shutdown."""
    pass


class ConfigurationError(QuantumSpectreError):
    """Raised when there is an error in the system configuration."""
    pass


class TimeoutError(QuantumSpectreError):
    """Raised when an operation times out."""
    pass


class NetworkError(QuantumSpectreError):
    """Raised when there is a network-related error."""
    pass


class InvalidParameterError(QuantumSpectreError):
    """Raised when an invalid parameter is provided to a function or method."""
    pass


class CalculationError(QuantumSpectreError):
    """Raised when a numerical calculation fails."""
    pass


class AnalysisError(QuantumSpectreError):
    """Raised for general errors during analysis tasks."""
    pass


class OptimizationError(QuantumSpectreError):
    """Raised for errors during optimization processes."""
    pass


class EnvironmentError(QuantumSpectreError):
    """Raised for errors related to the trading environment."""
    pass


class InvalidActionError(QuantumSpectreError):
    """Raised when an invalid action is taken in the environment."""
    pass


# ======================================
# Service Management Exceptions
# ======================================

class ServiceError(QuantumSpectreError):
    """Base class for service-related errors."""
    pass


class ServiceStartupError(ServiceError):
    """Raised when a service fails to start."""
    pass


class ServiceShutdownError(ServiceError):
    """Raised when a service fails to shut down properly."""
    pass


class ServiceConnectionError(ServiceError):
    """Raised for errors connecting to an internal or external service."""
    pass


class ServiceUnavailableError(ServiceError):
    """Raised when a required service is unavailable."""
    pass


class CircuitBreakerTrippedException(SystemCriticalError):
    """Raised when a circuit breaker is tripped."""
    pass


class TTSEngineError(ServiceError):
    """Raised for errors in the Text-to-Speech engine."""
    pass


# ======================================
# Resource Management Exceptions
# ======================================

class ResourceError(QuantumSpectreError):
    """Base class for resource-related errors."""
    pass


class ResourceExhaustionError(ResourceError):
    """Raised when a system resource (e.g., memory, disk, GPU) is exhausted."""
    pass


class GPUNotAvailableError(ResourceError):
    """Raised when GPU resources are requested but not available."""
    pass


class HardwareError(ResourceError):
    """Raised for errors related to hardware interaction or failure."""
    pass


class HardwareAccelerationError(HardwareError):
    """Raised when a hardware acceleration step fails."""

    pass


class RedundancyFailureError(ResourceError):
    """Raised when a redundancy mechanism fails."""
    pass


# ======================================
# Data Management Exceptions
# ======================================

class DataError(QuantumSpectreError):
    """Base class for data-related errors."""
    pass


class DataIngestionError(DataError):
    """Raised when there is an error during data ingestion."""
    pass


class DataProcessorError(DataError):
    """Raised when there is an error in a data processor."""
    pass


class ProcessorNotFoundError(DataProcessorError):
    """Raised when a requested data processor is not found."""
    pass


class DataTransformationError(DataProcessorError):
    """Raised for errors during data transformation."""
    pass


class SourceNotFoundError(DataError):
    """Raised when a requested data source is not found."""
    pass


class DataValidationError(DataError):
    """Raised when data validation fails."""
    pass


class InvalidDataError(DataValidationError):
    """Raised when provided data is invalid for the operation."""
    pass


class DataQualityError(DataValidationError):
    """Raised for issues related to data quality."""
    pass


class InvalidAssetError(DataValidationError):
    """Raised when an invalid asset is specified."""
    pass




class RiskError(QuantumSpectreError):
    """Base class for risk management errors."""
class InvalidTimeRangeError(DataValidationError):
    """Raised when an invalid time range is specified."""
    pass


class InvalidTimeframeError(QuantumSpectreError):
    """Raised when an invalid timeframe is provided."""
    pass


class RiskExceededError(RiskError):
    """Raised when a risk threshold is breached."""
    pass


class RiskExceededError(RiskError):
    """Raised when overall risk exposure is exceeded."""
    pass


class InvalidSignalError(QuantumSpectreError):
    """Raised when a trading signal is invalid or malformed."""
    pass


class NoConsensusError(QuantumSpectreError):
    """Raised when no consensus can be reached among voting components."""
    pass


class TakeProfitError(QuantumSpectreError):
    """Raised when there is an error with take profit order placement."""
    pass


class OrderValidationError(QuantumSpectreError):
    """Raised when an order fails validation."""
    pass


class OrderExecutionError(QuantumSpectreError):
    """Raised when an order fails to execute."""
    pass


class CouncilInitializationError(QuantumSpectreError):
    """Raised when a council fails to initialize."""
    pass


class ExposureError(QuantumSpectreError):
    """Raised when exposure limits are exceeded."""
    pass


class CircuitBreakerError(QuantumSpectreError):
    """Raised when a circuit breaker is triggered."""
    pass


class PositionError(QuantumSpectreError):
    """Raised when there is an error with a position."""
    pass


class DrawdownError(QuantumSpectreError):
    """Raised when drawdown limits are exceeded."""
    pass


class ExchangeError(QuantumSpectreError):
    """Raised when there is an error with an exchange."""
    pass



class InsufficientBalanceError(RiskError):
    """Raised when an account balance is insufficient to open a position."""
    pass


class BacktestError(QuantumSpectreError):
    """Base class for backtesting errors."""
class DataSourceError(DataError):
    """Raised when there is an error with a data source."""
    pass


class DataParsingError(DataError):
    """Raised when there is an error parsing data."""
    pass


class ParsingError(DataError):
    """Raised when there is a general parsing error."""
    pass


class DataAlignmentError(DataError):
    """Raised for errors aligning data from different sources or timeframes."""
    pass


class DataIntegrityError(DataError):
    """Raised when data integrity is compromised."""
    pass




class StrategyError(QuantumSpectreError):
    """Base class for strategy errors."""

class InsufficientDataError(DataError):
    """Raised when there is insufficient data for an operation."""
    pass


class DataInsufficientError(InsufficientDataError):
    """Raised when data is present but insufficient in quantity for an operation."""
    pass


class DataNotFoundError(DataError):
    """Raised when expected data is not found."""
    pass


class EncodingError(DataError):
    """Raised for errors during data encoding or decoding."""
    pass


class SamplingError(DataError):
    """Raised for errors during data sampling."""
    pass


class MarketDataError(DataError):
    """Raised when market data retrieval fails or is invalid."""
    pass


# ======================================
# Database and Storage Exceptions
# ======================================

class DatabaseError(QuantumSpectreError):
    """Base class for database-related errors."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when a database connection fails."""
    pass


class DatabaseQueryError(DatabaseError):
    """Raised when a database query fails."""
    pass


class DatabaseTimeoutError(DatabaseError):
    """Raised when a database operation times out."""
    pass


class DatabaseIntegrityError(DatabaseError):
    """Raised when database integrity is compromised."""
    pass


class DataStoreError(DatabaseError):
    """Raised for errors interacting with a generic data store."""
    pass


class StorageError(DatabaseError):
    """Raised for errors related to data storage operations."""
    pass


class MigrationError(DatabaseError):
    """Raised for errors during database migrations."""
    pass


class RedisError(QuantumSpectreError):
    """Base class for Redis-related errors."""
    pass


class RedisConnectionError(RedisError):
    """Raised when a Redis connection fails."""
    pass


# ======================================
# Time Series Database Exceptions
# ======================================

class TimeSeriesConnectionError(DatabaseError):
    """Exception raised when a connection to the time series database fails."""
    
    def __init__(self, message="Failed to connect to time series database", details=None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class TimeSeriesQueryError(DatabaseError):
    """Exception raised when a query to the time series database fails."""
    
    def __init__(self, message="Failed to execute time series query", query=None, details=None):
        self.message = message
        self.query = query
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        result = self.message
        if self.query:
            result += f"\nQuery: {self.query}"
        if self.details:
            result += f"\nDetails: {self.details}"
        return result


class TimeSeriesDataError(DataError):
    """Exception raised when there's an issue with time series data."""
    
    def __init__(self, message="Time series data error", data_info=None, details=None):
        self.message = message
        self.data_info = data_info
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        result = self.message
        if self.data_info:
            result += f"\nData info: {self.data_info}"
        if self.details:
            result += f"\nDetails: {self.details}"
        return result


class TimeSeriesConfigError(ConfigurationError):
    """Exception raised when there's a configuration error with the time series database."""
    
    def __init__(self, message="Time series configuration error", config_key=None, details=None):
        self.message = message
        self.config_key = config_key
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        result = self.message
        if self.config_key:
            result += f"\nConfig key: {self.config_key}"
        if self.details:
            result += f"\nDetails: {self.details}"
        return result


# ======================================
# Feed Management Exceptions
# ======================================

class FeedError(QuantumSpectreError):
    """Base class for feed-related errors."""
    pass


class FeedConnectionError(FeedError):
    """Raised when a feed connection fails."""
    pass


class BlockchainConnectionError(FeedConnectionError):
    """Raised specifically for blockchain node connection errors."""
    pass


class WebSocketError(FeedConnectionError):
    """Raised for errors related to WebSocket connections."""
    pass


class FeedDisconnectedError(FeedError):
    """Raised when a feed unexpectedly disconnects."""
    pass


class FeedTimeoutError(FeedError):
    """Raised when a feed operation times out."""
    pass


class FeedRateLimitError(FeedError):
    """Raised when a feed rate limit is exceeded."""
    pass


class FeedPriorityError(FeedError):
    """Raised when there is an error with feed priority handling."""
    pass


class FeedNotFoundError(FeedError):
    """Raised when a requested feed is not found."""
    pass


class FeedInitializationError(FeedError):
    """Raised when feed initialization fails."""
    pass


class FeedAuthenticationError(FeedError):
    """Raised when authentication with a feed service fails."""
    pass


class FeedSubscriptionError(FeedError):
    """Raised when there is an error with feed subscription."""
    pass


class FeedDataError(FeedError):
    """Raised when there is an error with feed data."""
    pass


class DataFeedConnectionError(FeedConnectionError):
    """Raised when there is a connection error with a data feed."""
    pass


class FeedCoordinationError(FeedError):
    """Raised for errors in coordinating multiple data feeds."""
    pass


class SubscriptionError(FeedError):
    """Raised for errors subscribing to data feeds or topics."""
    pass


class DataFetchError(FeedError):
    """Raised for errors fetching data from external sources."""
    pass


class RESTClientError(FeedError):
    """Raised for errors in a REST API client."""
    pass


class RequestError(FeedError):
    """Raised for general errors making external requests."""
    pass


class NewsFeedError(FeedError):
    """Raised when there is an error in the news feed module."""
    pass


class NewsParsingError(DataParsingError):
    """Raised when there is an error parsing news data."""
    pass


class NewsSourceUnavailableError(FeedError):
    """Raised when a news source is unavailable."""
    pass


# ======================================
# Rate Limiting Exceptions
# ======================================

class RateLimitError(QuantumSpectreError):
    """Raised when a system or API rate limit is exceeded."""
    
    def __init__(self, message="Rate limit exceeded", retry_after=None):
        self.message = message
        self.retry_after = retry_after
        super().__init__(self.message)

    def __str__(self):
        if self.retry_after:
            return f"{self.message}. Retry after {self.retry_after} seconds."
        return self.message


# ======================================
# Security Exceptions
# ======================================

class SecurityError(QuantumSpectreError):
    """Base class for security-related errors."""
    pass


class APIKeyError(SecurityError):
    """Raised when there is an issue with API key validation."""
    pass


class AuthenticationError(SecurityError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(SecurityError):
    """Raised when authorization fails."""
    pass


class PermissionDeniedError(AuthorizationError):
    """Raised when an action is denied due to insufficient permissions."""
    pass


class OperationNotPermittedError(SecurityError):
    """Raised when an operation is not permitted for the current user/context."""
    pass


class CredentialError(SecurityError):
    """Raised when there is an error with credentials."""
    pass


class SecurityViolationError(SecurityError):
    """Raised when a security violation is detected."""
    pass


# ======================================
# Trading Execution Exceptions
# ======================================

class ExecutionError(QuantumSpectreError):
    """Base class for execution engine errors."""
    pass


class EnvironmentError(ExecutionError):
    """Raised for errors related to trading or simulation environments."""
    pass


class InvalidActionError(EnvironmentError):
    """Raised when an invalid action is taken in an environment."""
    pass

class PositionError(ExecutionError):
    """Raised for position management errors."""
    pass

class PositionExecutionError(PositionError):
    """Raised when executing a position fails."""
    pass

class InvalidPositionStateError(PositionError):
    """Raised when a position is in an invalid state."""
    pass

class InsufficientBalanceError(PositionError):
    """Raised when the account balance is insufficient."""
    pass

class MarginCallError(PositionError):
    """Raised when a margin call is triggered."""
    pass

class PositionLiquidationError(PositionError):
    """Raised when a position is forcibly liquidated."""
    pass

class RiskExceededError(RiskError):
    """Raised when a trade exceeds defined risk parameters."""
    pass



class MaxDrawdownExceededError(RiskError):
    """Raised when maximum drawdown is exceeded."""
    pass


class MarginCallError(RiskError):
    """Raised when a margin call occurs."""
    pass


class PositionLiquidationError(RiskError):
    """Raised when a position is forcefully liquidated."""
    pass



class OrderError(ExecutionError):
    """Base class for order-related errors."""
    pass


class OrderRejectedError(OrderError):
    """Raised when an order is rejected by an exchange."""
    pass


class OrderTimeoutError(OrderError):
    """Raised when an order times out."""
    pass


class OrderExecutionError(OrderError):
    """Raised when there is an error executing an order."""
    pass


class InvalidOrderError(OrderError):
    """Raised when an order is invalid or has invalid parameters."""
    pass


class OrderCancellationError(OrderError):
    """Raised when there is an error cancelling an order."""
    pass


class SlippageExceededError(OrderError):
    """Raised when slippage exceeds the allowed threshold."""
    pass


class InsufficientFundsError(OrderError):
    """Raised when there are insufficient funds for an order."""
    pass


class StopLossError(OrderError):
    """Raised for errors related to stop-loss order management."""
    pass


class PositionError(ExecutionError):
    """Base class for position-related errors."""
    pass


class PositionExecutionError(PositionError):
    """Raised when a position operation fails during execution."""
    pass


class InvalidPositionStateError(PositionError):
    """Raised when a position is in an unexpected state."""
    pass


class MarginCallError(PositionError):
    """Raised when a margin call occurs on a position."""
    pass


class PositionLiquidationError(PositionError):
    """Raised when a position is forcibly liquidated."""
    pass


class InsufficientLiquidityError(ExecutionError):
    """Raised when there is insufficient liquidity to execute a trade."""
    pass


class ExchangeError(QuantumSpectreError):
    """Raised when there is a problem related to exchange operations."""
    pass


# ======================================
# Risk Management Exceptions
# ======================================

class RiskError(QuantumSpectreError):
    """Base class for risk management errors."""
    pass


class InsufficientBalanceError(RiskError):
    """Raised when account balance is insufficient for an operation."""
    pass


class RiskLimitExceededError(RiskError):
    """Raised when a risk limit is exceeded."""
    pass


class RiskExceededError(RiskError):
    """Raised when calculated risk exceeds configured threshold."""
    pass


class MaxDrawdownExceededError(RiskLimitExceededError):
    """Raised when maximum allowed drawdown is exceeded."""
    pass


class DrawdownLimitExceededException(RiskLimitExceededError):
    """Raised when a drawdown limit is exceeded."""
    pass

class MaxDrawdownExceededError(RiskLimitExceededError):
    """Raised when the maximum allowed drawdown is exceeded."""
    pass


class RiskManagerError(RiskError):
    """Raised for errors specific to the Risk Manager service."""
    pass


class PositionSizingError(RiskError):
    """Raised for errors in position sizing calculations."""
    pass


class RiskManagementException(RiskError):
    """General exception for risk management issues."""
    pass


class CapitalManagementError(RiskError):
    """Raised for errors in capital management."""
    pass


# ======================================
# Machine Learning Exceptions
# ======================================

class ModelError(QuantumSpectreError):
    """Base class for ML model errors."""
    pass


class ModelRegistrationError(ModelError):
    """Raised when registration of an ML model fails."""
    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""
    pass


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""
    pass


class ModelSaveError(ModelError):
    """Raised when saving a model fails."""
    pass


class ModelLoadError(ModelError):
    """Raised when there is an error loading a machine learning model."""
    pass


class ModelLoadingError(ModelLoadError):
    """Alias for backward compatibility."""

    pass


class InvalidModelStateError(ModelError):
    """Raised when a model is in an invalid state for the requested operation."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested ML model is not found."""
    pass


class ModelPersistenceError(ModelError):
    """Raised when loading or persisting a model fails."""
    pass


class ModelValidationError(ModelError):
    """Raised for validation errors related to models."""
    pass


class ModelVersionError(ModelError):
    """Raised for issues related to model versioning."""
    pass


class ModelNotSupportedError(ModelError):
    """Raised when a model type or version is not supported."""
    pass


class HyperparameterOptimizationError(ModelError):
    """Raised when hyperparameter optimization fails."""
    pass


class EnsembleConfigError(ModelError):
    """Raised for configuration errors in ensemble models."""
    pass


class TrainingError(ModelTrainingError):
    """General error during a training process."""
    pass


class PredictionError(ModelPredictionError):
    """General error during a prediction process."""
    pass


class InferenceError(ModelPredictionError):
    """Raised for errors during model inference."""
    pass


# ======================================
# Feature Engineering Exceptions
# ======================================

class FeatureServiceError(QuantumSpectreError):
    """Base class for feature service errors."""
    pass


class FeatureEngineeringError(FeatureServiceError):
    """Raised when feature engineering fails."""
    pass


class FeatureNotFoundError(FeatureServiceError):
    """Raised when a requested feature is not found."""
    pass


class FeatureCalculationError(FeatureServiceError):
    """Raised when there is an error calculating a feature."""
    pass


class InvalidFeatureDefinitionError(FeatureCalculationError):
    """Raised when a feature definition is invalid or malformed."""
    pass


class InvalidFeatureFormatError(FeatureCalculationError):
    """Raised when feature data has an invalid format."""
    pass


class FeatureTimeoutError(FeatureServiceError):
    """Raised when a feature calculation operation times out."""
    pass


class CorrelationCalculationError(FeatureCalculationError):
    """Raised for errors during correlation calculations."""
    pass


class PatternRecognitionError(FeatureCalculationError):
    """Raised for errors during pattern recognition."""
    pass


class PatternDetectionError(PatternRecognitionError):
    """Raised when there is an error detecting patterns in data."""
    pass


class PatternNotFoundError(FeatureNotFoundError):
    """Raised when a specific pattern is not found."""
    pass


class MicrostructureAnalysisError(FeatureCalculationError):
    """Raised for errors in market microstructure analysis."""
    pass


class SentimentAnalysisError(FeatureCalculationError):
    """Raised for errors during sentiment analysis."""
    pass


# ======================================
# Intelligence System Exceptions
# ======================================

class IntelligenceError(QuantumSpectreError):
    """Base class for intelligence system errors."""
    pass


class IntelligenceServiceError(IntelligenceError):
    """Raised when there is an error in the intelligence service."""
    pass


class RegimeDetectionError(IntelligenceError):
    """Raised when there is an error in regime detection."""
    pass


class LoopholeDetectionError(IntelligenceError):
    """Raised when there is an error in loophole detection."""
    pass


class AdaptiveLearningError(IntelligenceError):
    """Raised when there is an error in adaptive learning."""
    pass


class InvalidPopulationError(AdaptiveLearningError):
    """Raised when a genetic algorithm population is invalid."""
    pass


class ConvergenceError(AdaptiveLearningError):
    """Raised when an algorithm fails to converge."""
    pass


class GeneticOperationError(AdaptiveLearningError):
    """Raised when a genetic algorithm operation fails."""
    pass


class EvolutionError(AdaptiveLearningError):
    """Raised for errors during genetic algorithm evolution."""
    pass


# ======================================
# Strategy Exceptions
# ======================================

class StrategyError(QuantumSpectreError):
    """Base class for strategy errors."""
    pass


class SignalGenerationError(StrategyError):
    """Raised when signal generation fails."""
    pass


class InvalidStrategyError(StrategyError):
    """Raised when an invalid strategy is encountered or configured."""
    pass


class StrategyExecutionError(StrategyError):
    """Raised for errors during strategy execution."""
    pass


class AdaptationError(StrategyError):
    """Raised for errors during adaptive learning or strategy adaptation."""
    pass


class DecisionError(StrategyError):
    """Raised for errors in decision-making processes."""
    pass


class ArbitrageOpportunityExpiredError(StrategyError):
    """Raised when an arbitrage opportunity is no longer valid."""
    pass


class ArbitrageValidationError(StrategyError):
    """Raised for validation errors in arbitrage strategies."""
    pass


class RecoveryStrategyError(StrategyError):
    """Raised for errors in recovery strategies."""
    pass


class BrainNotFoundError(StrategyError):
    """Raised when a required Strategy Brain is not found."""
    pass


# ======================================
# Council System Exceptions
# ======================================

class CouncilError(QuantumSpectreError):
    """Raised for general errors in a Council."""
    pass


class AssetCouncilError(CouncilError):
    """Raised for errors in the Asset Council."""
    pass


class RegimeCouncilError(CouncilError):
    """Raised for errors in the Regime Council."""
    pass


class VotingError(QuantumSpectreError):
    """Raised for errors in voting systems."""
    pass


class WeightingSystemError(QuantumSpectreError):
    """Raised for errors in weighting systems."""
    pass


# ======================================
# Backtesting Exceptions
# ======================================

class BacktestError(QuantumSpectreError):
    """Base class for backtesting errors."""
    pass


class BacktestConfigError(BacktestError):
    """Raised for configuration errors in the backtester."""
    pass


class BacktestDataError(BacktestError):
    """Raised for data-related errors in the backtester."""
    pass


class BacktestStrategyError(BacktestError):
    """Raised for strategy-related errors in the backtester."""
    pass


class BacktestScenarioError(BacktestError):
    """Raised for errors related to backtesting scenarios."""
    pass


class SimulationError(BacktestError):
    """Raised for errors during backtest simulations."""
    pass


class BacktestExecutionError(BacktestError):
    """Raised when a backtest execution fails or is cancelled."""
    pass


class BacktestOptimizationError(BacktestError):
    """Raised for errors during backtest optimization."""
    pass


# ======================================
# Monitoring and Alerting Exceptions
# ======================================

class MonitoringError(QuantumSpectreError):
    """Base class for monitoring errors."""
    pass


class AlertError(MonitoringError):
    """Raised when alert generation or delivery fails."""
    pass


class AlertDeliveryError(AlertError):
    """Raised when alert delivery fails."""
    pass


class AlertConfigurationError(AlertError):
    """Raised when there is an error in alert configuration."""
    pass


class MetricCollectionError(MonitoringError):
    """Raised for errors during metric collection."""
    pass


class LogAnalysisError(MonitoringError):
    """Raised for errors during log analysis."""
    pass


class PerformanceTrackerError(MonitoringError):
    """Raised for errors in the performance tracker."""
    pass


# ======================================
# User Interface Exceptions
# ======================================

class DashboardError(QuantumSpectreError):
    """Raised for errors related to dashboard operations."""
    pass


class VoiceAdvisorError(QuantumSpectreError):
    """Raised for errors in the Voice Advisor system."""
    pass


class TimeSeriesConnectionError(Exception):
    """
    Exception raised when a connection to the time series database fails
    """
    def __init__(self, message="Failed to connect to time series database", details=None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message

class TimeSeriesQueryError(Exception):
    """
    Exception raised when a query to the time series database fails
    """
    def __init__(self, message="Failed to execute time series query", query=None, details=None):
        self.message = message
        self.query = query
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        result = self.message
        if self.query:
            result += f"\nQuery: {self.query}"
        if self.details:
            result += f"\nDetails: {self.details}"
        return result

class TimeSeriesDataError(Exception):
    """
    Exception raised when there's an issue with time series data
    (missing data, corrupted data, etc.)
    """
    def __init__(self, message="Time series data error", data_info=None, details=None):
        self.message = message
        self.data_info = data_info
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        result = self.message
        if self.data_info:
            result += f"\nData info: {self.data_info}"
        if self.details:
            result += f"\nDetails: {self.details}"
        return result

class TimeSeriesConfigError(Exception):
    """
    Exception raised when there's a configuration error with the time series database
    (invalid settings, connection parameters, etc.)
    """
    def __init__(self, message="Time series configuration error", config_key=None, details=None):
        self.message = message
        self.config_key = config_key
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        result = self.message
        if self.config_key:
            result += f"\nConfig key: {self.config_key}"
        if self.details:
            result += f"\nDetails: {self.details}"
        return result
class ReportGenerationError(QuantumSpectreError):
    """Raised for errors during report generation."""
    pass


# ======================================
# Export Interface
# ======================================

__all__ = [
    'QuantumSpectreError', 'ConfigurationError', 'ServiceError', 'ServiceStartupError',
    'ServiceShutdownError', 'SystemCriticalError', 'DataError', 'DataIngestionError',
    'DataProcessorError', 'ProcessorNotFoundError', 'SourceNotFoundError', 'DataValidationError',
    'FeedError', 'FeedConnectionError', 'BlockchainConnectionError', 'FeedDisconnectedError',
    'FeedTimeoutError', 'FeedRateLimitError', 'DatabaseError', 'DatabaseConnectionError',
    'DatabaseQueryError', 'RedisError', 'RedisConnectionError', 'SecurityError',
    'APIKeyError', 'AuthenticationError', 'AuthorizationError', 'ExecutionError',
    'EnvironmentError', 'InvalidActionError',
    'OrderError', 'OrderRejectedError', 'OrderTimeoutError', 'InsufficientFundsError',
    'InsufficientBalanceError',
    'InvalidOrderError', 'OrderCancellationError', 'SlippageExceededError', 'NetworkError',
    'RiskError', 'RiskLimitExceededError', 'RiskExceededError', 'BacktestError', 'ModelError',
    'ModelTrainingError', 'ModelPredictionError', 'StrategyError', 'SignalGenerationError',
    'PositionError', 'InsufficientBalanceError', 'PositionExecutionError',
    'InvalidOrderError', 'OrderCancellationError', 'SlippageExceededError', 'NetworkError',
    'RiskError', 'RiskLimitExceededError', 'BacktestError', 'ModelError',
    'RiskExceededError', 'ModelTrainingError', 'ModelPredictionError',
    'ModelRegistrationError', 'InvalidModelStateError',
    'StrategyError', 'SignalGenerationError',
    'MonitoringError', 'AlertError', 'ResourceError', 'ResourceExhaustionError',
    'TimeoutError', 'ExchangeError', 'RateLimitError', 'FeedNotFoundError',
    'FeedInitializationError', 'FeedAuthenticationError', 'DataSourceError',
    'FeedSubscriptionError', 'FeedDataError', 'ParsingError', 'DataFeedConnectionError',
    'ModelLoadError', 'ModelLoadingError', 'DataParsingError', 'CredentialError', 'SecurityViolationError',
    'RegimeDetectionError', 'NewsFeedError', 'NewsParsingError', 'NewsSourceUnavailableError',
    'FeatureNotFoundError', 'FeatureCalculationError', 'FeatureServiceError',
    'InvalidTimeframeError', 'InvalidParameterError', 'AlertDeliveryError',
    'AlertConfigurationError', 'RiskManagerError', 'PositionSizingError',
    'InvalidPositionStateError', 'MaxDrawdownExceededError',
    'MarginCallError', 'PositionLiquidationError', 'StopLossError',

    'AlertConfigurationError', 'RiskManagerError', 'PositionSizingError', 'StopLossError',
    'PositionError', 'PositionExecutionError', 'InvalidPositionStateError',
    'InsufficientBalanceError', 'MarginCallError', 'PositionLiquidationError',
    'RiskExceededError', 'ModelRegistrationError',
    'ModelNotFoundError', 'DashboardError', 'InsufficientLiquidityError',
    'ArbitrageOpportunityExpiredError', 'DrawdownLimitExceededException', 'MaxDrawdownExceededError',
    'RiskManagementException', 'ModelVersionError', 'LogAnalysisError',
    'InsufficientDataError', 'EncodingError', 'MetricCollectionError',
    'ServiceConnectionError', 'DataStoreError', 'InvalidDataError', 'TrainingError',
    'ArbitrageValidationError', 'PredictionError', 'AnalysisError', 'RecoveryStrategyError',
    'OptimizationError', 'CorrelationCalculationError', 'SamplingError', 'DataQualityError',
    'HardwareError', 'HardwareAccelerationError', 'EnsembleConfigError', 'ServiceUnavailableError', 'ModelNotSupportedError',
    'InvalidFeatureFormatError', 'StrategyExecutionError', 'AdaptationError', 'InferenceError',
    'CircuitBreakerTrippedException', 'PatternRecognitionError', 'PatternNotFoundError',
    'DataAlignmentError', 'RESTClientError', 'RequestError', 'DataTransformationError',
    'CapitalManagementError', 'MicrostructureAnalysisError', 'MigrationError',
    'ModelValidationError', 'WebSocketError', 'SubscriptionError', 'DataFetchError',
    'BacktestConfigError', 'BacktestDataError', 'BacktestStrategyError', 'AssetCouncilError',
    'BrainNotFoundError', 'CouncilError', 'DecisionError', 'PerformanceTrackerError',
    'InvalidStrategyError', 'SimulationError', 'BacktestExecutionError', 'BacktestOptimizationError', 'SentimentAnalysisError', 'RegimeCouncilError',
    'ReportGenerationError', 'OperationNotPermittedError', 'BacktestScenarioError',
    'DataNotFoundError', 'VoiceAdvisorError', 'TTSEngineError', 'VotingError',
    'PermissionDeniedError', 'WeightingSystemError', 'FeedCoordinationError',
    'DataInsufficientError', 'InvalidAssetError', 'InvalidTimeRangeError',
    'InvalidFeatureDefinitionError', 'FeatureTimeoutError',
    'TimeSeriesConnectionError', 'TimeSeriesQueryError', 'TimeSeriesDataError',
    'TimeSeriesConfigError'

    # Base exceptions
    "QuantumSpectreError", "SystemCriticalError", "ConfigurationError", "TimeoutError",
    "NetworkError", "InvalidParameterError", "CalculationError", "AnalysisError", "OptimizationError",
    "EnvironmentError", "InvalidActionError",
    
    # Service management
    "ServiceError", "ServiceStartupError", "ServiceShutdownError", "ServiceConnectionError",
    "ServiceUnavailableError", "CircuitBreakerTrippedException", "TTSEngineError",
    
    # Resource management
    "ResourceError", "ResourceExhaustionError", "GPUNotAvailableError", "HardwareError", "HardwareAccelerationError",
    "RedundancyFailureError",
    
    # Data management
    "DataError", "DataIngestionError", "DataProcessorError", "ProcessorNotFoundError",
    "DataTransformationError", "SourceNotFoundError", "DataValidationError", "InvalidDataError",
    "DataQualityError", "InvalidAssetError", "InvalidTimeRangeError", "InvalidTimeframeError",
    "DataSourceError", "DataParsingError", "ParsingError", "DataAlignmentError", "DataIntegrityError",
    "InsufficientDataError", "DataInsufficientError", "DataNotFoundError", "EncodingError",
    "SamplingError", "MarketDataError",
    
    # Database and storage
    "DatabaseError", "DatabaseConnectionError", "DatabaseQueryError", "DatabaseTimeoutError",
    "DatabaseIntegrityError", "DataStoreError", "StorageError", "MigrationError",
    "RedisError", "RedisConnectionError",
    
    # Time series database
    "TimeSeriesConnectionError", "TimeSeriesQueryError", "TimeSeriesDataError", "TimeSeriesConfigError",
    
    # Feed management
    "FeedError", "FeedConnectionError", "BlockchainConnectionError", "WebSocketError",
    "FeedDisconnectedError", "FeedTimeoutError", "FeedRateLimitError", "FeedPriorityError",
    "FeedNotFoundError", "FeedInitializationError", "FeedAuthenticationError", "FeedSubscriptionError",
    "FeedDataError", "DataFeedConnectionError", "FeedCoordinationError", "SubscriptionError",
    "DataFetchError", "RESTClientError", "RequestError", "NewsFeedError", "NewsParsingError",
    "NewsSourceUnavailableError",
    
    # Rate limiting
    "RateLimitError",
    
    # Security
    "SecurityError", "APIKeyError", "AuthenticationError", "AuthorizationError",
    "PermissionDeniedError", "OperationNotPermittedError", "CredentialError", "SecurityViolationError",
    
    # Trading execution
    "ExecutionError", "OrderError", "OrderRejectedError", "OrderTimeoutError", "OrderExecutionError",
    "InvalidOrderError", "OrderCancellationError", "SlippageExceededError", "InsufficientFundsError",
    "StopLossError", "PositionError", "PositionExecutionError", "InvalidPositionStateError",
    "MarginCallError", "PositionLiquidationError", "InsufficientLiquidityError", "ExchangeError",
    
    # Risk management
    "RiskError", "InsufficientBalanceError", "RiskLimitExceededError", "RiskExceededError",
    "MaxDrawdownExceededError", "DrawdownLimitExceededException", "RiskManagerError",
]
