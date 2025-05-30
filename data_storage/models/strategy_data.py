#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Strategy Data Models

This module defines SQLAlchemy models for storing strategy configurations,
performance metrics, and evolution history. These models enable the system
to track strategy performance, evolve strategies over time, and maintain
a comprehensive history of trading decisions.
"""

import enum
import uuid
import json
import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text,
    ForeignKey, Index, Enum, JSON, UniqueConstraint, Table, LargeBinary
)
from sqlalchemy.orm import relationship, backref
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY

from data_storage.models import Base
from common.constants import TIMEFRAMES, ASSET_TYPES, STRATEGY_TYPES


class StrategyStatus(enum.Enum):
    """Enum for strategy status"""
    DRAFT = "draft"
    BACKTEST = "backtest"
    PAPER_TRADING = "paper_trading"
    LIVE = "live"
    PAUSED = "paused"
    ARCHIVED = "archived"
    EVOLVING = "evolving"


class StrategyEvolutionMethod(enum.Enum):
    """Enum for strategy evolution methods"""
    MANUAL = "manual"
    GENETIC = "genetic"
    REINFORCEMENT = "reinforcement"
    BAYESIAN = "bayesian"
    META_LEARNING = "meta_learning"
    ENSEMBLE = "ensemble"


# Association table for strategy-asset pairs
strategy_asset_association = Table(
    'strategy_asset_association',
    Base.metadata,
    Column('strategy_id', UUID(as_uuid=True), ForeignKey('strategies.id'), primary_key=True),
    Column('asset_id', UUID(as_uuid=True), ForeignKey('assets.id'), primary_key=True),
    Column('created_at', DateTime, default=datetime.datetime.utcnow),
    Column('updated_at', DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow),
    Column('status', Enum(StrategyStatus), default=StrategyStatus.DRAFT),
    Column('confidence', Float, default=0.0),
    Column('allocation_weight', Float, default=1.0),
    Column('performance_score', Float, default=0.0),
    Column('win_rate', Float, default=0.0),
    Column('custom_params', JSONB, default={}),
)

# Association table for strategy-timeframe pairs
strategy_timeframe_association = Table(
    'strategy_timeframe_association',
    Base.metadata,
    Column('strategy_id', UUID(as_uuid=True), ForeignKey('strategies.id'), primary_key=True),
    Column('timeframe', String(20), primary_key=True),
    Column('created_at', DateTime, default=datetime.datetime.utcnow),
    Column('updated_at', DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow),
    Column('weight', Float, default=1.0),
    Column('performance_score', Float, default=0.0),
    Column('custom_params', JSONB, default={}),
)


class Strategy(Base):
    """Strategy model for storing trading strategy configurations"""
    __tablename__ = 'strategies'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    strategy_type = Column(String(50), nullable=False)
    status = Column(Enum(StrategyStatus), default=StrategyStatus.DRAFT)
    
    # Core parameters
    parameters = Column(JSONB, default={})
    entry_conditions = Column(JSONB, default=[])
    exit_conditions = Column(JSONB, default=[])
    risk_management = Column(JSONB, default={})
    
    # Performance metrics
    win_rate = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    sortino_ratio = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    avg_profit_per_trade = Column(Float, default=0.0)
    avg_loss_per_trade = Column(Float, default=0.0)
    avg_hold_time = Column(Float, default=0.0)
    performance_score = Column(Float, default=0.0)
    
    # Evolution metadata
    version = Column(Integer, default=1)
    parent_id = Column(UUID(as_uuid=True), ForeignKey('strategies.id'), nullable=True)
    evolution_method = Column(Enum(StrategyEvolutionMethod), nullable=True)
    evolution_history = Column(JSONB, default=[])
    evolution_score = Column(Float, default=0.0)
    generation = Column(Integer, default=0)
    
    # Code representations
    python_code = Column(Text, nullable=True)
    compiled_code = Column(LargeBinary, nullable=True)
    
    # Timestamps and metadata
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    last_trained_at = Column(DateTime, nullable=True)
    creator_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    
    # Relationships
    assets = relationship("Asset", secondary=strategy_asset_association, 
                          backref=backref("strategies", lazy="dynamic"))
    performance_records = relationship("StrategyPerformance", back_populates="strategy",
                                     cascade="all, delete-orphan")
    backtests = relationship("BacktestRun", back_populates="strategy",
                           cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="strategy")
    children = relationship("Strategy", backref=backref('parent', remote_side=[id]))
    
    # Create indices for common queries
    __table_args__ = (
        Index('ix_strategies_type_status', strategy_type, status),
        Index('ix_strategies_performance', performance_score.desc()),
        Index('ix_strategies_created_at', created_at),
        UniqueConstraint('name', 'version', name='uq_strategy_name_version'),
    )
    
    def __repr__(self):
        return f""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary representation."""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'strategy_type': self.strategy_type,
            'status': self.status.value if self.status else None,
            'parameters': self.parameters,
            'entry_conditions': self.entry_conditions,
            'exit_conditions': self.exit_conditions,
            'risk_management': self.risk_management,
            'win_rate': self.win_rate,
            'performance_score': self.performance_score,
            'version': self.version,
            'generation': self.generation,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
    
    @classmethod
    def create_from_dict(cls, data: Dict[str, Any]) -> 'Strategy':
        """Create a strategy from dictionary data."""
        required_fields = ['name', 'strategy_type']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Handle enum conversions
        if 'status' in data and data['status']:
            data['status'] = StrategyStatus(data['status'])
        if 'evolution_method' in data and data['evolution_method']:
            data['evolution_method'] = StrategyEvolutionMethod(data['evolution_method'])
        
        return cls(**data)
    
    def update_performance(self, new_metrics: Dict[str, float]) -> None:
        """Update strategy performance metrics."""
        for key, value in new_metrics.items():
            if hasattr(self, key) and isinstance(value, (int, float)):
                setattr(self, key, value)
        
        # Calculate composite performance score
        self.performance_score = self._calculate_performance_score()
    
    def _calculate_performance_score(self) -> float:
        """Calculate a composite performance score from multiple metrics."""
        # Weights for different performance metrics
        weights = {
            'win_rate': 0.25,
            'profit_factor': 0.20,
            'sharpe_ratio': 0.15,
            'sortino_ratio': 0.15,
            'max_drawdown': 0.15,
            'avg_profit_per_trade': 0.10,
        }
        
        # Normalize max_drawdown (lower is better)
        normalized_drawdown = 1.0 - min(abs(self.max_drawdown) / 0.5, 1.0) if self.max_drawdown else 0
        
        # Calculate weighted score
        score = (
            weights['win_rate'] * self.win_rate +
            weights['profit_factor'] * min(self.profit_factor / 3.0, 1.0) +
            weights['sharpe_ratio'] * min(self.sharpe_ratio / 3.0, 1.0) +
            weights['sortino_ratio'] * min(self.sortino_ratio / 3.0, 1.0) +
            weights['max_drawdown'] * normalized_drawdown +
            weights['avg_profit_per_trade'] * min(self.avg_profit_per_trade / 0.05, 1.0)
        )
        
        return min(max(score, 0.0), 1.0)
    
    def clone(self, new_name: Optional[str] = None) -> 'Strategy':
        """Create a clone of this strategy with incremented version."""
        new_name = new_name or f"{self.name} v{self.version + 1}"
        
        # Create a new strategy with copied attributes
        clone = Strategy(
            name=new_name,
            description=self.description,
            strategy_type=self.strategy_type,
            status=StrategyStatus.DRAFT,
            parameters=self.parameters.copy() if self.parameters else {},
            entry_conditions=self.entry_conditions.copy() if self.entry_conditions else [],
            exit_conditions=self.exit_conditions.copy() if self.exit_conditions else [],
            risk_management=self.risk_management.copy() if self.risk_management else {},
            python_code=self.python_code,
            version=self.version + 1,
            parent_id=self.id,
            generation=self.generation + 1
        )
        
        return clone
    
    def record_evolution_step(self, method: StrategyEvolutionMethod, changes: Dict[str, Any], 
                              performance_delta: float) -> None:
        """Record an evolution step in the strategy's history."""
        if not self.evolution_history:
            self.evolution_history = []
            
        step = {
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'method': method.value,
            'changes': changes,
            'performance_before': self.performance_score - performance_delta,
            'performance_after': self.performance_score,
            'performance_delta': performance_delta,
            'generation': self.generation
        }
        
        self.evolution_history.append(step)
        self.evolution_method = method


class StrategyPerformance(Base):
    """Model for tracking strategy performance over time."""
    __tablename__ = 'strategy_performances'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey('strategies.id'), nullable=False)
    asset_id = Column(UUID(as_uuid=True), ForeignKey('assets.id'), nullable=True)
    timeframe = Column(String(20), nullable=True)
    
    # Time period
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    
    # Performance metrics
    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    total_profit = Column(Float, default=0.0)
    total_loss = Column(Float, default=0.0)
    net_profit = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    sortino_ratio = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    max_runup = Column(Float, default=0.0)
    avg_profit_per_trade = Column(Float, default=0.0)
    avg_loss_per_trade = Column(Float, default=0.0)
    largest_win = Column(Float, default=0.0)
    largest_loss = Column(Float, default=0.0)
    avg_hold_time_winning = Column(Float, default=0.0)
    avg_hold_time_losing = Column(Float, default=0.0)
    performance_score = Column(Float, default=0.0)
    
    # Detailed metrics
    monthly_returns = Column(JSONB, default={})
    drawdown_periods = Column(JSONB, default=[])
    trade_distribution = Column(JSONB, default={})
    
    # Market conditions
    market_regime = Column(String(50), nullable=True)
    volatility_level = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    strategy = relationship("Strategy", back_populates="performance_records")
    
    # Indices
    __table_args__ = (
        Index('ix_strategy_performance_strategy_id_timeframe', strategy_id, timeframe),
        Index('ix_strategy_performance_time_period', start_time, end_time),
    )
    
    def __repr__(self):
        return (f"")
    
    @classmethod
    def create_from_backtest(cls, strategy_id: uuid.UUID, backtest_result: Dict[str, Any]) -> 'StrategyPerformance':
        """Create a performance record from backtest results."""
        required_fields = ['start_time', 'end_time', 'asset_id', 'timeframe']
        for field in required_fields:
            if field not in backtest_result:
                raise ValueError(f"Missing required field in backtest results: {field}")
        
        # Extract metrics from backtest results
        metrics = backtest_result.get('metrics', {})
        
        # Create the performance record
        return cls(
            strategy_id=strategy_id,
            asset_id=backtest_result['asset_id'],
            timeframe=backtest_result['timeframe'],
            start_time=backtest_result['start_time'],
            end_time=backtest_result['end_time'],
            win_count=metrics.get('win_count', 0),
            loss_count=metrics.get('loss_count', 0),
            win_rate=metrics.get('win_rate', 0.0),
            profit_factor=metrics.get('profit_factor', 0.0),
            total_profit=metrics.get('total_profit', 0.0),
            total_loss=metrics.get('total_loss', 0.0),
            net_profit=metrics.get('net_profit', 0.0),
            sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
            sortino_ratio=metrics.get('sortino_ratio', 0.0),
            max_drawdown=metrics.get('max_drawdown', 0.0),
            max_runup=metrics.get('max_runup', 0.0),
            avg_profit_per_trade=metrics.get('avg_profit_per_trade', 0.0),
            avg_loss_per_trade=metrics.get('avg_loss_per_trade', 0.0),
            largest_win=metrics.get('largest_win', 0.0),
            largest_loss=metrics.get('largest_loss', 0.0),
            avg_hold_time_winning=metrics.get('avg_hold_time_winning', 0.0),
            avg_hold_time_losing=metrics.get('avg_hold_time_losing', 0.0),
            performance_score=metrics.get('performance_score', 0.0),
            monthly_returns=metrics.get('monthly_returns', {}),
            drawdown_periods=metrics.get('drawdown_periods', []),
            trade_distribution=metrics.get('trade_distribution', {}),
            market_regime=backtest_result.get('market_regime'),
            volatility_level=backtest_result.get('volatility_level'),
        )


class BacktestRun(Base):
    """Model for storing backtest run information."""
    __tablename__ = 'backtest_runs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey('strategies.id'), nullable=False)
    
    # Configuration
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    assets = Column(JSONB, default=[])  # List of asset IDs
    timeframes = Column(ARRAY(String), default=[])
    initial_capital = Column(Float, nullable=False)
    parameters = Column(JSONB, default={})
    
    # Results summary
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    net_profit = Column(Float, default=0.0)
    net_profit_percent = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    sortino_ratio = Column(Float, default=0.0)
    
    # Detailed results stored separately
    trades_file_path = Column(String(255), nullable=True)
    equity_curve_file_path = Column(String(255), nullable=True)
    detailed_metrics = Column(JSONB, default={})
    
    # Status
    status = Column(String(50), default="completed")
    error_message = Column(Text, nullable=True)
    execution_time = Column(Float, default=0.0)  # seconds
    
    # Version tracking
    strategy_version = Column(Integer, nullable=False)
    system_version = Column(String(50), nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    strategy = relationship("Strategy", back_populates="backtests")
    
    __table_args__ = (
        Index('ix_backtest_runs_strategy_id', strategy_id),
        Index('ix_backtest_runs_created_at', created_at),
    )
    
    def __repr__(self):
        return (f"")
    
    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the backtest results."""
        return {
            'id': str(self.id),
            'strategy_id': str(self.strategy_id),
            'strategy_name': self.strategy.name if self.strategy else None,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'net_profit_percent': self.net_profit_percent,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class StrategyEvolveTask(Base):
    """Model for tracking strategy evolution tasks."""
    __tablename__ = 'strategy_evolve_tasks'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey('strategies.id'), nullable=False)
    
    # Evolution parameters
    evolution_method = Column(Enum(StrategyEvolutionMethod), nullable=False)
    parameters = Column(JSONB, default={})
    assets = Column(JSONB, default=[])  # List of asset IDs to optimize for
    timeframes = Column(ARRAY(String), default=[])
    target_metrics = Column(JSONB, default={})  # Metrics to optimize
    constraints = Column(JSONB, default={})  # Constraints on parameters
    
    # Status tracking
    status = Column(String(50), default="pending")
    progress = Column(Float, default=0.0)
    current_generation = Column(Integer, default=0)
    total_generations = Column(Integer, default=0)
    iterations_completed = Column(Integer, default=0)
    total_iterations = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    
    # Results
    result_strategy_id = Column(UUID(as_uuid=True), ForeignKey('strategies.id'), nullable=True)
    performance_improvement = Column(Float, default=0.0)
    evolution_history = Column(JSONB, default=[])
    
    # Timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    strategy = relationship("Strategy", foreign_keys=[strategy_id])
    result_strategy = relationship("Strategy", foreign_keys=[result_strategy_id])
    
    __table_args__ = (
        Index('ix_strategy_evolve_tasks_strategy_id', strategy_id),
        Index('ix_strategy_evolve_tasks_status', status),
    )
    
    def __repr__(self):
        return (f"")
    
    def update_progress(self, progress: float, current_generation: int = None, 
                        iterations_completed: int = None) -> None:
        """Update the task progress."""
        self.progress = min(max(progress, 0.0), 1.0)
        
        if current_generation is not None:
            self.current_generation = current_generation
            
        if iterations_completed is not None:
            self.iterations_completed = iterations_completed
    
    def mark_started(self) -> None:
        """Mark the task as started."""
        self.status = "running"
        self.started_at = datetime.datetime.utcnow()
    
    def mark_completed(self, result_strategy_id: uuid.UUID, performance_improvement: float) -> None:
        """Mark the task as completed with results."""
        self.status = "completed"
        self.completed_at = datetime.datetime.utcnow()
        self.result_strategy_id = result_strategy_id
        self.performance_improvement = performance_improvement
        self.progress = 1.0
    
    def mark_failed(self, error_message: str) -> None:
        """Mark the task as failed with an error message."""
        self.status = "failed"
        self.completed_at = datetime.datetime.utcnow()
        self.error_message = error_message


class StrategyGeneticHistory(Base):
    """Model for tracking genetic algorithm evolution history for strategies."""
    __tablename__ = 'strategy_genetic_history'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey('strategies.id'), nullable=False)
    
    # Genetic algorithm metadata
    generation = Column(Integer, nullable=False)
    individual_id = Column(String(50), nullable=False)
    parent_ids = Column(ARRAY(String), default=[])
    mutation_rate = Column(Float, nullable=True)
    crossover_rate = Column(Float, nullable=True)
    
    # Genetic representation
    chromosome = Column(JSONB, nullable=False)  # Parameter values
    fitness = Column(Float, nullable=False)
    rank = Column(Integer, nullable=True)
    
    # Performance metrics
    win_rate = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    net_profit = Column(Float, default=0.0)
    
    # Evolution details
    mutation_details = Column(JSONB, default={})
    crossover_details = Column(JSONB, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    strategy = relationship("Strategy", foreign_keys=[strategy_id])
    
    __table_args__ = (
        Index('ix_strategy_genetic_history_strategy_id', strategy_id),
        Index('ix_strategy_genetic_history_generation', generation),
        Index('ix_strategy_genetic_history_fitness', fitness.desc()),
    )
    
    def __repr__(self):
        return f"StrategyGeneticHistory(id={self.id}, strategy_id={self.strategy_id}, generation={self.generation}, fitness={self.fitness})"
    
    @classmethod
    def create_from_individual(cls, strategy_id: uuid.UUID, generation: int,
                              individual_id: str, chromosome: Dict[str, Any],
                              fitness: float, metrics: Dict[str, float],
                              parent_ids: List[str] = None,
                              mutation_details: Dict[str, Any] = None,
                              crossover_details: Dict[str, Any] = None) -> 'StrategyGeneticHistory':
        """Create a genetic history record from an individual in the population."""
        return cls(
            strategy_id=strategy_id,
            generation=generation,
            individual_id=individual_id,
            parent_ids=parent_ids or [],
            chromosome=chromosome,
            fitness=fitness,
            win_rate=metrics.get('win_rate', 0.0),
            profit_factor=metrics.get('profit_factor', 0.0),
            sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
            max_drawdown=metrics.get('max_drawdown', 0.0),
            net_profit=metrics.get('net_profit', 0.0),
            mutation_details=mutation_details or {},
            crossover_details=crossover_details or {}
        )


class PositionModel(Base):
    """Simplified model for executed trading positions."""

    __tablename__ = 'strategy_positions'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey('strategies.id'))
    symbol = Column(String(20), nullable=False)
    strategy_id = Column(UUID(as_uuid=True), ForeignKey('strategies.id'), nullable=False)
    symbol = Column(String(50), nullable=False)
    side = Column(String(10), nullable=False)
    quantity = Column(Float, default=0.0)
    entry_price = Column(Float, default=0.0)
    asset_id = Column(UUID(as_uuid=True), ForeignKey('assets.id'), nullable=False)
    side = Column(String(10), nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    opened_at = Column(DateTime, default=datetime.datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)

    strategy = relationship('Strategy', backref=backref('positions', lazy='dynamic'))

    status = Column(String(20), default='open')
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    strategy = relationship('Strategy', back_populates='trades')
