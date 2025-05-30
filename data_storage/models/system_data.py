

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
System Data Models

This module defines SQLAlchemy models for storing system configuration,
state, and operational data. These models enable the system to maintain
its state across restarts, track performance, and store system-wide settings.
"""

import enum
import uuid
import json
import datetime
from typing import Dict, List, Optional, Any, Union, Set
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text,
    ForeignKey, Index, Enum, JSON, UniqueConstraint, Table, LargeBinary
)
from sqlalchemy.orm import relationship, backref
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY

from data_storage.models import Base
from common.constants import SYSTEM_COMPONENT_TYPES


class SystemComponentStatus(enum.Enum):
    """Enum for system component status"""
    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    ERROR = "error"
    STOPPED = "stopped"
    MAINTENANCE = "maintenance"


class SystemNotificationSeverity(enum.Enum):
    """Enum for notification severity levels"""
    DEBUG = "debug"
    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    ALERT = "alert"
    EMERGENCY = "emergency"


class SystemComponent(Base):
    """Model for tracking system components and their status"""
    __tablename__ = 'system_components'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    component_type = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    
    # Current status
    status = Column(Enum(SystemComponentStatus), default=SystemComponentStatus.UNKNOWN)
    last_heartbeat = Column(DateTime, nullable=True)
    version = Column(String(50), nullable=True)
    health_score = Column(Float, default=0.0)
    
    # Runtime info
    pid = Column(Integer, nullable=True)
    host = Column(String(255), nullable=True)
    port = Column(Integer, nullable=True)
    startup_time = Column(DateTime, nullable=True)
    uptime_seconds = Column(Integer, default=0)
    
    # Performance metrics
    cpu_usage = Column(Float, default=0.0)
    memory_usage = Column(Float, default=0.0)
    disk_usage = Column(Float, default=0.0)
    
    # Connectivity info
    dependencies = Column(JSONB, default=[])
    endpoints = Column(JSONB, default={})
    
    # Configuration
    config = Column(JSONB, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    logs = relationship("ComponentLog", back_populates="component",
                        cascade="all, delete-orphan")
    metrics = relationship("ComponentMetric", back_populates="component",
                          cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_system_components_status', status),
        Index('ix_system_components_type', component_type),
        Index('ix_system_components_health', health_score),
    )
    
    def __repr__(self):
        return f""
    
    def update_heartbeat(self) -> None:
        """Update the component heartbeat"""
        self.last_heartbeat = datetime.datetime.utcnow()
        
        if self.startup_time:
            self.uptime_seconds = int((datetime.datetime.utcnow() - self.startup_time).total_seconds())
    
    def update_metrics(self, cpu: float, memory: float, disk: float) -> None:
        """Update the component performance metrics"""
        self.cpu_usage = cpu
        self.memory_usage = memory
        self.disk_usage = disk
        self.updated_at = datetime.datetime.utcnow()
    
    def update_status(self, status: SystemComponentStatus, recalculate_health: bool = True) -> None:
        """Update the component status"""
        self.status = status
        self.updated_at = datetime.datetime.utcnow()
        
        if recalculate_health:
            self._recalculate_health_score()
    
    def _recalculate_health_score(self) -> None:
        """Calculate a health score based on status and metrics"""
        # Base score from status
        status_scores = {
            SystemComponentStatus.RUNNING: 1.0,
            SystemComponentStatus.STARTING: 0.6,
            SystemComponentStatus.DEGRADED: 0.4,
            SystemComponentStatus.MAINTENANCE: 0.5,
            SystemComponentStatus.ERROR: 0.1,
            SystemComponentStatus.STOPPED: 0.0,
            SystemComponentStatus.UNKNOWN: 0.0,
        }
        
        status_score = status_scores.get(self.status, 0.0)
        
        # Adjust based on metrics (if running)
        metric_score = 1.0
        if self.status == SystemComponentStatus.RUNNING:
            # Penalize for high resource usage
            if self.cpu_usage > 90:
                metric_score -= 0.2
            elif self.cpu_usage > 75:
                metric_score -= 0.1
                
            if self.memory_usage > 90:
                metric_score -= 0.2
            elif self.memory_usage > 75:
                metric_score -= 0.1
                
            if self.disk_usage > 90:
                metric_score -= 0.2
            elif self.disk_usage > 75:
                metric_score -= 0.1
        
        # Calculate final score (60% status, 40% metrics)
        self.health_score = (status_score * 0.6) + (metric_score * 0.4)


class ComponentLog(Base):
    """Model for storing component logs"""
    __tablename__ = 'component_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    component_id = Column(UUID(as_uuid=True), ForeignKey('system_components.id'), nullable=False)
    
    # Log details
    level = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(JSONB, nullable=True)
    source_file = Column(String(255), nullable=True)
    source_line = Column(Integer, nullable=True)
    traceback = Column(Text, nullable=True)
    
    # Context
    context = Column(JSONB, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    
    # Relationships
    component = relationship("SystemComponent", back_populates="logs")
    
    __table_args__ = (
        Index('ix_component_logs_level', level),
        Index('ix_component_logs_component_created', component_id, created_at.desc()),
    )
    
    def __repr__(self):
        return f""


class ComponentMetric(Base):
    """Model for storing component metrics over time"""
    __tablename__ = 'component_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    component_id = Column(UUID(as_uuid=True), ForeignKey('system_components.id'), nullable=False)
    
    # Metric details
    name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    units = Column(String(50), nullable=True)
    
    # Context
    tags = Column(JSONB, default={})
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    component = relationship("SystemComponent", back_populates="metrics")
    
    __table_args__ = (
        Index('ix_component_metrics_component_name_time', 
              component_id, name, timestamp.desc()),
    )
    
    def __repr__(self):
        return f""


class SystemNotification(Base):
    """Model for system notifications and alerts"""
    __tablename__ = 'system_notifications'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Notification details
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    severity = Column(Enum(SystemNotificationSeverity), nullable=False)
    source = Column(String(100), nullable=True)
    details = Column(JSONB, nullable=True)
    
    # Status tracking
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    acknowledged_at = Column(DateTime, nullable=True)
    
    # Action tracking
    requires_action = Column(Boolean, default=False)
    action_taken = Column(Boolean, default=False)
    action_details = Column(JSONB, nullable=True)
    
    # Related components
    component_id = Column(UUID(as_uuid=True), ForeignKey('system_components.id'), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    
    # Relationships
    component = relationship("SystemComponent")
    acknowledger = relationship("User", foreign_keys=[acknowledged_by])
    
    __table_args__ = (
        Index('ix_system_notifications_severity', severity),
        Index('ix_system_notifications_created', created_at.desc()),
        Index('ix_system_notifications_acknowledged', acknowledged),
    )
    
    def __repr__(self):
        return (f"")
    
    def acknowledge(self, user_id: uuid.UUID) -> None:
        """Acknowledge the notification"""
        self.acknowledged = True
        self.acknowledged_by = user_id
        self.acknowledged_at = datetime.datetime.utcnow()
    
    def record_action(self, action_details: Dict[str, Any]) -> None:
        """Record an action taken on this notification"""
        self.action_taken = True
        self.action_details = action_details
        self.updated_at = datetime.datetime.utcnow()
    
    def is_expired(self) -> bool:
        """Check if the notification has expired"""
        if not self.expires_at:
            return False
        return datetime.datetime.utcnow() > self.expires_at


class SystemConfig(Base):
    """Model for system-wide configuration settings"""
    __tablename__ = 'system_config'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Config identification
    name = Column(String(255), nullable=False, unique=True)
    category = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Value storage - supports different types
    value_string = Column(Text, nullable=True)
    value_int = Column(Integer, nullable=True)
    value_float = Column(Float, nullable=True)
    value_bool = Column(Boolean, nullable=True)
    value_json = Column(JSONB, nullable=True)
    
    # Constraints and validation
    data_type = Column(String(50), nullable=False)
    is_required = Column(Boolean, default=False)
    default_value = Column(Text, nullable=True)
    value_options = Column(JSONB, nullable=True)  # For enums/choices
    validation_regex = Column(String(255), nullable=True)
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)
    
    # Access control
    is_sensitive = Column(Boolean, default=False)
    requires_restart = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    updated_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    
    # Relationships
    creator = relationship("User", foreign_keys=[created_by])
    updater = relationship("User", foreign_keys=[updated_by])
    
    __table_args__ = (
        Index('ix_system_config_category', category),
    )
    
    def __repr__(self):
        return f""
    
    @property
    def value(self) -> Any:
        """Get the config value with proper type"""
        if self.data_type == 'string':
            return self.value_string
        elif self.data_type == 'integer':
            return self.value_int
        elif self.data_type == 'float':
            return self.value_float
        elif self.data_type == 'boolean':
            return self.value_bool
        elif self.data_type == 'json':
            return self.value_json
        return None
    
    @value.setter
    def value(self, val: Any) -> None:
        """Set the config value with proper type validation"""
        if self.data_type == 'string':
            self.value_string = str(val) if val is not None else None
        elif self.data_type == 'integer':
            self.value_int = int(val) if val is not None else None
        elif self.data_type == 'float':
            self.value_float = float(val) if val is not None else None
        elif self.data_type == 'boolean':
            self.value_bool = bool(val) if val is not None else None
        elif self.data_type == 'json':
            self.value_json = val
    
    def validate(self) -> bool:
        """Validate the current value against constraints"""
        if self.is_required and self.value is None:
            return False
            
        if self.value is None:
            return True
            
        # Type-specific validation
        if self.data_type in ('integer', 'float'):
            if self.min_value is not None and self.value < self.min_value:
                return False
            if self.max_value is not None and self.value > self.max_value:
                return False
                
        elif self.data_type == 'string' and self.validation_regex:
            import re
            if not re.match(self.validation_regex, self.value_string or ''):
                return False
                
        # Validate against options
        if self.value_options and self.value not in self.value_options:
            return False
            
        return True


class SystemState(Base):
    """Model for storing system state information"""
    __tablename__ = 'system_state'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # State identification
    key = Column(String(255), nullable=False, unique=True)
    category = Column(String(100), nullable=False)
    
    # Value storage
    value = Column(JSONB, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    
    __table_args__ = (
        Index('ix_system_state_category', category),
    )
    
    def __repr__(self):
        return f""
    
    def is_expired(self) -> bool:
        """Check if the state entry has expired"""
        if not self.expires_at:
            return False
        return datetime.datetime.utcnow() > self.expires_at


class SystemTask(Base):
    """Model for tracking system tasks and jobs"""
    __tablename__ = 'system_tasks'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Task identification
    name = Column(String(255), nullable=False)
    task_type = Column(String(100), nullable=False)
    
    # Task details
    description = Column(Text, nullable=True)
    parameters = Column(JSONB, default={})
    
    # Status tracking
    status = Column(String(50), default='pending')
    progress = Column(Float, default=0.0)
    result = Column(JSONB, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Scheduling
    scheduled_at = Column(DateTime, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Task control
    priority = Column(Integer, default=0)
    max_attempts = Column(Integer, default=1)
    attempt_count = Column(Integer, default=0)
    last_attempt_at = Column(DateTime, nullable=True)
    timeout_seconds = Column(Integer, nullable=True)
    
    # Associated component
    component_id = Column(UUID(as_uuid=True), ForeignKey('system_components.id'), nullable=True)
    
    # User association
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    component = relationship("SystemComponent")
    user = relationship("User")
    
    __table_args__ = (
        Index('ix_system_tasks_status', status),
        Index('ix_system_tasks_priority_status', priority, status),
        Index('ix_system_tasks_scheduled_at', scheduled_at),
    )
    
    def __repr__(self):
        return f""
    
    def mark_started(self) -> None:
        """Mark the task as started"""
        self.status = 'running'
        self.started_at = datetime.datetime.utcnow()
        self.attempt_count += 1
        self.last_attempt_at = datetime.datetime.utcnow()
    
    def mark_completed(self, result: Dict[str, Any] = None) -> None:
        """Mark the task as completed with results"""
        self.status = 'completed'
        self.completed_at = datetime.datetime.utcnow()
        self.progress = 1.0
        if result is not None:
            self.result = result
    
    def mark_failed(self, error_message: str) -> None:
        """Mark the task as failed with error details"""
        if self.attempt_count >= self.max_attempts:
            self.status = 'failed'
        else:
            self.status = 'pending'  # Will be retried
            
        self.error_message = error_message
        self.completed_at = datetime.datetime.utcnow() if self.status == 'failed' else None
    
    def can_be_retried(self) -> bool:
        """Check if the task can be retried"""
        return self.status in ('pending', 'failed') and self.attempt_count < self.max_attempts
    
    def update_progress(self, progress: float) -> None:
        """Update the task progress"""
        self.progress = min(max(progress, 0.0), 1.0)


class SystemBackup(Base):
    """Model for tracking system backups"""
    __tablename__ = 'system_backups'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Backup details
    name = Column(String(255), nullable=False)
    backup_type = Column(String(50), nullable=False)  # full, incremental, config-only, etc.
    storage_path = Column(String(255), nullable=False)
    size_bytes = Column(Integer, nullable=True)
    components = Column(JSONB, default=[])  # Components included in backup
    
    # Metadata
    description = Column(Text, nullable=True)
    version = Column(String(50), nullable=True)
    checksum = Column(String(64), nullable=True)
    
    # Status
    status = Column(String(50), default='pending')
    error_message = Column(Text, nullable=True)
    
    # Scheduling/execution
    scheduled_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    executed_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    scheduled_at = Column(DateTime, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    
    # Relationships
    scheduler = relationship("User", foreign_keys=[scheduled_by])
    executor = relationship("User", foreign_keys=[executed_by])
    
    __table_args__ = (
        Index('ix_system_backups_status', status),
        Index('ix_system_backups_created_at', created_at.desc()),
    )
    
    def __repr__(self):
        return f""


class SystemMaintenance(Base):
    """Model for tracking system maintenance activities"""
    __tablename__ = 'system_maintenance'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Maintenance details
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    maintenance_type = Column(String(50), nullable=False)  # update, optimization, repair, etc.
    affected_components = Column(JSONB, default=[])
    
    # Planned execution
    planned_start_time = Column(DateTime, nullable=True)
    estimated_duration_minutes = Column(Integer, nullable=True)
    planned_end_time = Column(DateTime, nullable=True)
    
    # Actual execution
    actual_start_time = Column(DateTime, nullable=True)
    actual_end_time = Column(DateTime, nullable=True)
    
    # Status tracking
    status = Column(String(50), default='scheduled')
    progress = Column(Float, default=0.0)
    result = Column(JSONB, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Notification tracking
    notification_sent = Column(Boolean, default=False)
    notification_time = Column(DateTime, nullable=True)
    
    # User association
    scheduled_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    executed_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    scheduler = relationship("User", foreign_keys=[scheduled_by])
    executor = relationship("User", foreign_keys=[executed_by])
    
    __table_args__ = (
        Index('ix_system_maintenance_status', status),
        Index('ix_system_maintenance_planned', planned_start_time),
    )
    
    def __repr__(self):
        return f""
    
    def mark_started(self, user_id: uuid.UUID = None) -> None:
        """Mark maintenance as started"""
        self.status = 'in_progress'
        self.actual_start_time = datetime.datetime.utcnow()
        if user_id:
            self.executed_by = user_id
    
    def mark_completed(self, result: Dict[str, Any] = None) -> None:
        """Mark maintenance as completed"""
        self.status = 'completed'
        self.actual_end_time = datetime.datetime.utcnow()
        self.progress = 1.0
        if result:
            self.result = result
    
    def mark_failed(self, error_message: str) -> None:
        """Mark maintenance as failed"""
        self.status = 'failed'
        self.actual_end_time = datetime.datetime.utcnow()
        self.error_message = error_message
    
    def update_progress(self, progress: float) -> None:
        """Update maintenance progress"""
        self.progress = min(max(progress, 0.0), 1.0)

