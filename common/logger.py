#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Logging Module

This module provides a centralized logging system for the QuantumSpectre Elite Trading System.
It supports console and file logging, with different formats and log levels.
"""

import os
import sys
import json
import time
import logging
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Dict, Any, Optional, List, Union

# ANSI color codes for terminal output
class Colors:
    RESET = '\033[0m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'

class ColorFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.BLUE,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.RED + Colors.BOLD
    }
    
    def format(self, record):
        # Save original format
        format_orig = self._style._fmt
        
        # Apply color to level name
        levelname = record.levelname
        record.levelname = f"{self.LEVEL_COLORS.get(record.levelno, Colors.RESET)}{levelname}{Colors.RESET}"
        
        # Apply color to the whole message for critical level
        if record.levelno == logging.CRITICAL:
            self._style._fmt = f"{Colors.RED}{format_orig}{Colors.RESET}"
            
        # Format the record
        result = logging.Formatter.format(self, record)
        
        # Restore original format
        self._style._fmt = format_orig
        
        return result

class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON strings for structured logging."""
    
    def format(self, record):
        log_record = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'process': record.process,
            'thread': record.thread
        }
        
        # Add exception info if available
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__,
                'value': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
            
        # Add extra fields from record
        if hasattr(record, 'extra') and record.extra:
            log_record.update(record.extra)
            
        return json.dumps(log_record)

class PerformanceHandler(logging.Handler):
    """Handler that collects performance metrics from logs."""
    
    def __init__(self):
        super().__init__()
        self.metrics = {}
        self.operations = {}
        
    def emit(self, record):
        if not hasattr(record, 'operation'):
            return
            
        operation = record.operation
        duration = getattr(record, 'duration', None)
        
        if operation not in self.operations:
            self.operations[operation] = {
                'count': 0,
                'total_duration': 0,
                'min_duration': float('inf'),
                'max_duration': 0,
                'average_duration': 0
            }
            
        if duration:
            self.operations[operation]['count'] += 1
            self.operations[operation]['total_duration'] += duration
            self.operations[operation]['min_duration'] = min(duration, self.operations[operation]['min_duration'])
            self.operations[operation]['max_duration'] = max(duration, self.operations[operation]['max_duration'])
            self.operations[operation]['average_duration'] = (
                self.operations[operation]['total_duration'] / self.operations[operation]['count']
            )

class LogContextAdapter(logging.LoggerAdapter):
    """Logger adapter that adds context to log records."""
    
    def process(self, msg, kwargs):
        # Add extra context to the record
        kwargs.setdefault('extra', {})
        
        if self.extra:
            # Merge our extra context with any existing extra
            for key, value in self.extra.items():
                if key not in kwargs['extra']:
                    kwargs['extra'][key] = value
                    
        return msg, kwargs
        
    def operation(self, operation_name):
        """Create a context manager for timing operations."""
        return OperationContext(self, operation_name)

class OperationContext:
    """Context manager for timing and logging operations."""
    
    def __init__(self, logger, operation_name):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(f"Starting operation: {self.operation_name}", 
                         extra={'operation': self.operation_name})
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type:
            self.logger.error(
                f"Operation {self.operation_name} failed after {duration:.3f}s",
                exc_info=(exc_type, exc_val, exc_tb),
                extra={'operation': self.operation_name, 'duration': duration}
            )
        else:
            self.logger.debug(
                f"Operation {self.operation_name} completed in {duration:.3f}s",
                extra={'operation': self.operation_name, 'duration': duration}
            )

def setup_logging(level=logging.INFO, log_file=None, max_size=10485760, backup_count=5, 
                 json_format=False, performance_tracking=True, console=True):
    """
    Set up the logging system.
    
    Args:
        level: Log level (default: logging.INFO)
        log_file: Path to log file (default: None, logs to console only)
        max_size: Maximum log file size in bytes (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
        json_format: Whether to use JSON format for logs (default: False)
        performance_tracking: Whether to track performance metrics (default: True)
        console: Whether to log to console (default: True)
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Create formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(log_format, date_format)
        
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColorFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S"
        ))
        root_logger.addHandler(console_handler)
        
    # Add file handler if log file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Set up rotating file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
    # Add performance handler if requested
    if performance_tracking:
        performance_handler = PerformanceHandler()
        performance_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(performance_handler)
        
    # Log setup completion
    logging.getLogger("logging").info(f"Logging system initialized. Level: {logging.getLevelName(level)}")

def get_logger(name, context=None):
    """
    Get a logger with the specified name and optional context.
    
    Args:
        name: Logger name
        context: Optional context dictionary to be added to log records
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if context:
        return LogContextAdapter(logger, context)
        
    return logger

def get_performance_metrics():
    """
    Get performance metrics from the performance handler.
    
    Returns:
        Dictionary of performance metrics
    """
    for handler in logging.getLogger().handlers:
        if isinstance(handler, PerformanceHandler):
            return handler.operations
            
    return {}

# Export required by system for backward compatibility
performance_log = OperationContext
