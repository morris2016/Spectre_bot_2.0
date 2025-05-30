
#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Log Analyzer Module

This module implements sophisticated log analysis and pattern detection capabilities
for identifying system issues, potential opportunities, and performance bottlenecks.
"""

import os
import re
import time
import json
import gzip
import datetime
from collections import defaultdict, Counter, deque
from typing import Dict, List, Set, Tuple, Any, Optional, Union, Callable
import asyncio
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from dataclasses import dataclass, field

from common.logger import get_logger
from common.async_utils import run_in_threadpool
from common.utils import (
    timeit, generate_uuid, timestamp_to_datetime, datetime_to_timestamp, 
    format_timestamp, get_human_readable_time
)
from common.redis_client import RedisClient
from common.constants import LOG_LEVELS, LOG_PATTERNS
from common.exceptions import LogAnalysisError

logger = get_logger("log_analyzer")


@dataclass
class LogEvent:
    """Structured representation of a log event"""
    timestamp: float
    level: str
    service: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    source_file: str = ""
    line_number: int = 0
    thread_id: str = ""
    parsed: bool = False
    hash_id: str = ""


class LogPattern:
    """Represents a pattern to detect in logs"""
    
    def __init__(
        self, 
        name: str, 
        pattern: str, 
        level: str = "INFO", 
        importance: int = 1,
        description: str = "",
        actions: List[str] = None,
        correlation_patterns: List[str] = None
    ):
        self.name = name
        self.pattern = re.compile(pattern)
        self.level = level
        self.importance = importance
        self.description = description
        self.actions = actions or []
        self.correlation_patterns = correlation_patterns or []
        self.count = 0
        self.last_seen = 0
        self.first_seen = 0
        self.examples = deque(maxlen=5)
    
    def matches(self, log_event: LogEvent) -> bool:
        """Check if the log event matches this pattern"""
        if log_event.level != self.level and self.level != "*":
            return False
        
        match = self.pattern.search(log_event.message)
        if match:
            self.count += 1
            self.last_seen = log_event.timestamp
            if self.first_seen == 0:
                self.first_seen = log_event.timestamp
            
            if len(self.examples) < 5:
                self.examples.append(log_event)
            return True
        
        return False


class LogAnalyzer:
    """
    Analyzes system logs to identify patterns, anomalies, and correlations.
    Provides insights into system performance and potential issues.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the log analyzer with configuration
        
        Args:
            config: Configuration dictionary for log analyzer
        """
        self.config = config
        self.redis_client = RedisClient(config["redis"])
        self.log_dir = config.get("log_dir", "/var/log/quantumspectre")
        self.max_logs_to_analyze = config.get("max_logs_to_analyze", 100000)
        self.window_size = config.get("window_size", 3600)  # Default 1 hour
        self.patterns = self._load_patterns()
        self.service_metrics = defaultdict(lambda: {
            "error_count": 0,
            "warning_count": 0,
            "info_count": 0,
            "pattern_matches": defaultdict(int),
            "timeouts": 0,
            "exceptions": Counter(),
        })
        self.recent_events = deque(maxlen=1000)
        self.thread_pool = ThreadPoolExecutor(max_workers=config.get("max_workers", 4))
        self.alert_threshold = config.get("alert_threshold", 10)
        self.anomaly_models = {}
        self._initialize_anomaly_models()
        
        # Real-time tracking
        self.realtime_enabled = config.get("realtime_enabled", True)
        self.realtime_window = deque(maxlen=config.get("realtime_window_size", 10000))
        self.realtime_pattern_counts = defaultdict(int)
        self.running = False
        logger.info("Log analyzer initialized with configuration", extra={"config": config})
    
    def _load_patterns(self) -> List[LogPattern]:
        """Load predefined and custom log patterns"""
        patterns = []
        
        # Load system defined patterns
        for name, pattern_info in LOG_PATTERNS.items():
            pattern = LogPattern(
                name=name,
                pattern=pattern_info["pattern"],
                level=pattern_info.get("level", "INFO"),
                importance=pattern_info.get("importance", 1),
                description=pattern_info.get("description", ""),
                actions=pattern_info.get("actions", []),
                correlation_patterns=pattern_info.get("correlation_patterns", [])
            )
            patterns.append(pattern)
        
        # Load user-defined patterns from config
        for pattern_info in self.config.get("custom_patterns", []):
            pattern = LogPattern(
                name=pattern_info["name"],
                pattern=pattern_info["pattern"],
                level=pattern_info.get("level", "INFO"),
                importance=pattern_info.get("importance", 1),
                description=pattern_info.get("description", ""),
                actions=pattern_info.get("actions", []),
                correlation_patterns=pattern_info.get("correlation_patterns", [])
            )
            patterns.append(pattern)
        
        logger.info(f"Loaded {len(patterns)} log patterns")
        return patterns
    
    def _initialize_anomaly_models(self) -> None:
        """Initialize anomaly detection models for different metrics"""
        try:
            # Basic anomaly detection using statistical methods
            # For production, we would use more sophisticated ML models
            for service in self.config.get("monitored_services", []):
                self.anomaly_models[service] = {
                    "error_rate": {
                        "mean": 0,
                        "std": 0,
                        "samples": [],
                        "max_samples": 100
                    },
                    "response_time": {
                        "mean": 0,
                        "std": 0,
                        "samples": [],
                        "max_samples": 100
                    }
                }
            logger.info("Initialized anomaly detection models")
        except Exception as e:
            logger.error(f"Failed to initialize anomaly detection models: {str(e)}")
            raise LogAnalysisError(f"Failed to initialize anomaly models: {str(e)}")
    
    def parse_log_line(self, line: str) -> Optional[LogEvent]:
        """
        Parse a raw log line into a structured LogEvent object
        
        Args:
            line: Raw log line from file
            
        Returns:
            LogEvent object if parsing succeeds, None otherwise
        """
        try:
            # Handle JSON formatted logs
            if line.startswith('{') and line.endswith('}'):
                try:
                    data = json.loads(line)
                    return LogEvent(
                        timestamp=data.get("timestamp", time.time()),
                        level=data.get("level", "INFO"),
                        service=data.get("service", "unknown"),
                        message=data.get("message", ""),
                        context=data.get("extra", {}),
                        source_file=data.get("file", ""),
                        line_number=data.get("line", 0),
                        thread_id=data.get("thread", ""),
                        parsed=True,
                        hash_id=generate_uuid()
                    )
                except json.JSONDecodeError:
                    pass
            
            # Try various log format patterns
            # Pattern 1: [TIME] [LEVEL] [SERVICE] message
            pattern1 = r'\[([\d\-\:\. ]+)\]\s+\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]\s+\[([^\]]+)\]\s+(.+)'
            match = re.match(pattern1, line)
            if match:
                timestamp_str, level, service, message = match.groups()
                try:
                    dt = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                    timestamp = dt.timestamp()
                except ValueError:
                    timestamp = time.time()
                
                return LogEvent(
                    timestamp=timestamp,
                    level=level,
                    service=service,
                    message=message,
                    parsed=True,
                    hash_id=generate_uuid()
                )
            
            # Pattern 2: TIME LEVEL [SERVICE] message
            pattern2 = r'([\d\-\:\. ]+)\s+(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+\[([^\]]+)\]\s+(.+)'
            match = re.match(pattern2, line)
            if match:
                timestamp_str, level, service, message = match.groups()
                try:
                    dt = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                    timestamp = dt.timestamp()
                except ValueError:
                    timestamp = time.time()
                
                return LogEvent(
                    timestamp=timestamp,
                    level=level,
                    service=service,
                    message=message,
                    parsed=True,
                    hash_id=generate_uuid()
                )
            
            # If we can't parse it with our patterns, just create a basic event
            return LogEvent(
                timestamp=time.time(),
                level="INFO",
                service="unknown",
                message=line,
                parsed=False,
                hash_id=generate_uuid()
            )
        except Exception as e:
            logger.warning(f"Failed to parse log line: {str(e)}", extra={"line": line[:200]})
            return None
    
    async def analyze_log_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a single log file and extract patterns, events, and metrics
        
        Args:
            file_path: Path to the log file
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            "file": file_path,
            "events_processed": 0,
            "patterns_detected": defaultdict(int),
            "services": defaultdict(lambda: defaultdict(int)),
            "levels": defaultdict(int),
            "start_time": None,
            "end_time": None,
            "duration": 0,
            "exceptions": [],
            "anomalies": []
        }
        
        try:
            logger.info(f"Analyzing log file: {file_path}")
            open_func = gzip.open if file_path.endswith('.gz') else open
            mode = 'rt' if file_path.endswith('.gz') else 'r'
            
            with open_func(file_path, mode) as f:
                for i, line in enumerate(f):
                    if i >= self.max_logs_to_analyze:
                        logger.warning(f"Reached maximum log analysis limit ({self.max_logs_to_analyze})")
                        break
                    
                    if not line.strip():
                        continue
                    
                    event = await run_in_threadpool(self.parse_log_line, line)
                    if not event:
                        continue
                    
                    # Update timestamps
                    if results["start_time"] is None or event.timestamp < results["start_time"]:
                        results["start_time"] = event.timestamp
                    
                    if results["end_time"] is None or event.timestamp > results["end_time"]:
                        results["end_time"] = event.timestamp
                    
                    # Track metrics
                    results["events_processed"] += 1
                    results["levels"][event.level] += 1
                    results["services"][event.service]["total"] += 1
                    results["services"][event.service][event.level] += 1
                    
                    # Check for exceptions
                    if "exception" in event.message.lower() or "error" in event.message.lower():
                        if len(results["exceptions"]) < 100:  # Limit to avoid memory issues
                            results["exceptions"].append({
                                "timestamp": event.timestamp,
                                "service": event.service,
                                "message": event.message,
                                "level": event.level
                            })
                    
                    # Match patterns
                    for pattern in self.patterns:
                        if pattern.matches(event):
                            results["patterns_detected"][pattern.name] += 1
                            
                            # Store significant events
                            if pattern.importance >= 3:
                                self.recent_events.append(event)
                    
                    # Update service metrics
                    self.service_metrics[event.service][f"{event.level.lower()}_count"] += 1
                    
                    # Track exceptions
                    if event.level == "ERROR":
                        # Try to extract exception type
                        exception_match = re.search(r'([\w\.]+Exception|Error):', event.message)
                        if exception_match:
                            exception_type = exception_match.group(1)
                            self.service_metrics[event.service]["exceptions"][exception_type] += 1
                    
                    # Check for timeout patterns
                    if "timeout" in event.message.lower():
                        self.service_metrics[event.service]["timeouts"] += 1
            
            if results["start_time"] and results["end_time"]:
                results["duration"] = results["end_time"] - results["start_time"]
            
            # Convert defaultdicts to regular dicts for JSON serialization
            results["patterns_detected"] = dict(results["patterns_detected"])
            results["services"] = {k: dict(v) for k, v in results["services"].items()}
            results["levels"] = dict(results["levels"])
            
            logger.info(f"Completed analysis of {file_path}",
                     extra={"events": results["events_processed"],
                           "patterns": len(results["patterns_detected"])})
            
            return results
        
        except Exception as e:
            logger.error(f"Error analyzing log file {file_path}: {str(e)}")
            raise LogAnalysisError(f"Failed to analyze log file {file_path}: {str(e)}")
    
    async def analyze_recent_logs(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze logs from the past specified hours
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Analysis results
        """
        try:
            cutoff_time = time.time() - (hours * 3600)
            log_files = []
            
            # Find all relevant log files
            for root, _, files in os.walk(self.log_dir):
                for file in files:
                    if file.endswith(".log") or file.endswith(".log.gz"):
                        file_path = os.path.join(root, file)
                        file_time = os.path.getmtime(file_path)
                        
                        if file_time >= cutoff_time:
                            log_files.append(file_path)
            
            log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            logger.info(f"Found {len(log_files)} log files within the last {hours} hours")
            
            # Analyze files in parallel
            tasks = []
            for file_path in log_files:
                tasks.append(self.analyze_log_file(file_path))
            
            results = await asyncio.gather(*tasks)
            
            # Combine results
            combined_results = {
                "timeframe": f"Last {hours} hours",
                "files_analyzed": len(results),
                "total_events": sum(r["events_processed"] for r in results),
                "patterns_detected": defaultdict(int),
                "services": defaultdict(lambda: defaultdict(int)),
                "levels": defaultdict(int),
                "start_time": min((r["start_time"] for r in results if r["start_time"] is not None), default=None),
                "end_time": max((r["end_time"] for r in results if r["end_time"] is not None), default=None),
                "exceptions": [],
                "anomalies": [],
                "service_metrics": dict(self.service_metrics)
            }
            
            # Aggregate pattern detections, service counts, and levels
            for r in results:
                for pattern, count in r["patterns_detected"].items():
                    combined_results["patterns_detected"][pattern] += count
                
                for service, metrics in r["services"].items():
                    for metric, value in metrics.items():
                        combined_results["services"][service][metric] += value
                
                for level, count in r["levels"].items():
                    combined_results["levels"][level] += count
                
                # Collect exceptions (limited to avoid memory issues)
                if len(combined_results["exceptions"]) < 1000:
                    combined_results["exceptions"].extend(r["exceptions"][:100])
            
            # Calculate duration
            if combined_results["start_time"] and combined_results["end_time"]:
                combined_results["duration"] = combined_results["end_time"] - combined_results["start_time"]
            else:
                combined_results["duration"] = 0
            
            # Convert defaultdicts to regular dicts for JSON serialization
            combined_results["patterns_detected"] = dict(combined_results["patterns_detected"])
            combined_results["services"] = {k: dict(v) for k, v in combined_results["services"].items()}
            combined_results["levels"] = dict(combined_results["levels"])
            
            # Sort exceptions by timestamp
            combined_results["exceptions"].sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Detect anomalies
            combined_results["anomalies"] = self.detect_anomalies(combined_results)
            
            # Identify potential issues
            combined_results["potential_issues"] = self.identify_potential_issues(combined_results)
            
            return combined_results
        
        except Exception as e:
            logger.error(f"Error analyzing recent logs: {str(e)}")
            raise LogAnalysisError(f"Failed to analyze recent logs: {str(e)}")
    
    def detect_anomalies(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in log patterns and metrics
        
        Args:
            results: Analysis results
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        try:
            # Check for unusual error rates
            for service, metrics in results["services"].items():
                if "ERROR" in metrics and "total" in metrics:
                    error_rate = metrics["ERROR"] / metrics["total"] if metrics["total"] > 0 else 0
                    
                    # If error rate is over 5%, flag as anomaly
                    if error_rate > 0.05 and metrics["total"] > 10:
                        anomalies.append({
                            "type": "high_error_rate",
                            "service": service,
                            "value": error_rate,
                            "threshold": 0.05,
                            "description": f"High error rate of {error_rate:.2%} for service {service}"
                        })
            
            # Check for unusual pattern frequencies
            for pattern, count in results["patterns_detected"].items():
                pattern_obj = next((p for p in self.patterns if p.name == pattern), None)
                if pattern_obj and pattern_obj.importance >= 3 and count > self.alert_threshold:
                    anomalies.append({
                        "type": "high_pattern_frequency",
                        "pattern": pattern,
                        "count": count,
                        "threshold": self.alert_threshold,
                        "description": f"High frequency of pattern '{pattern}' ({count} occurrences)"
                    })
            
            # Check for sudden increases in timeouts
            for service, metrics in self.service_metrics.items():
                if metrics["timeouts"] > 5:
                    anomalies.append({
                        "type": "high_timeout_count",
                        "service": service,
                        "count": metrics["timeouts"],
                        "threshold": 5,
                        "description": f"High number of timeouts ({metrics['timeouts']}) for service {service}"
                    })
            
            return anomalies
        
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []
    
    def identify_potential_issues(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify potential issues based on log analysis
        
        Args:
            results: Analysis results
            
        Returns:
            List of potential issues with descriptions and recommended actions
        """
        issues = []
        
        try:
            # Check for services with high error rates
            for service, metrics in results["services"].items():
                error_count = metrics.get("ERROR", 0)
                total_count = metrics.get("total", 0)
                
                if error_count > 10 and total_count > 0:
                    error_rate = error_count / total_count
                    if error_rate > 0.1:  # More than 10% errors
                        issues.append({
                            "type": "service_health",
                            "service": service,
                            "severity": "high" if error_rate > 0.3 else "medium",
                            "description": f"Service {service} has a high error rate of {error_rate:.2%}",
                            "recommended_actions": [
                                "Check service logs for specific errors",
                                "Check service dependencies and connections",
                                "Verify service configuration"
                            ]
                        })
            
            # Check for pattern correlations that might indicate issues
            correlated_patterns = self.find_correlated_patterns(results["patterns_detected"])
            for correlation in correlated_patterns:
                issues.append({
                    "type": "pattern_correlation",
                    "patterns": correlation["patterns"],
                    "severity": "medium",
                    "description": correlation["description"],
                    "recommended_actions": correlation["actions"]
                })
            
            # Check for exceptions that might indicate issues
            exception_types = Counter()
            for exception in results["exceptions"]:
                exception_match = re.search(r'([\w\.]+Exception|Error):', exception["message"])
                if exception_match:
                    exception_type = exception_match.group(1)
                    exception_types[exception_type] += 1
            
            for exception_type, count in exception_types.most_common(5):
                if count > 5:
                    issues.append({
                        "type": "recurring_exception",
                        "exception_type": exception_type,
                        "count": count,
                        "severity": "high" if count > 20 else "medium",
                        "description": f"Recurring exception of type {exception_type} ({count} occurrences)",
                        "recommended_actions": [
                            "Check application code handling this exception type",
                            "Verify error handling and recovery mechanisms",
                            "Check if any recent deployments introduced this issue"
                        ]
                    })
            
            return issues
        
        except Exception as e:
            logger.error(f"Error identifying potential issues: {str(e)}")
            return []
    
    def find_correlated_patterns(self, pattern_counts: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        Find correlated patterns that might indicate system issues
        
        Args:
            pattern_counts: Counter of pattern occurrences
            
        Returns:
            List of correlated pattern groups with descriptions
        """
        correlations = []
        
        # Define known correlation groups that indicate issues
        correlation_groups = [
            {
                "patterns": ["database_timeout", "connection_error"],
                "description": "Database connection issues detected",
                "actions": [
                    "Check database server status",
                    "Verify database connection settings",
                    "Check network connectivity to database"
                ]
            },
            {
                "patterns": ["api_rate_limit", "external_service_error"],
                "description": "External API service issues detected",
                "actions": [
                    "Check external service status",
                    "Implement rate limiting and backoff strategies",
                    "Verify API authentication credentials"
                ]
            },
            {
                "patterns": ["memory_warning", "high_cpu_usage", "slow_response"],
                "description": "System resource constraints detected",
                "actions": [
                    "Check system resources (memory, CPU)",
                    "Consider scaling up resources",
                    "Optimize resource-intensive operations"
                ]
            }
        ]
        
        for group in correlation_groups:
            patterns_present = [p for p in group["patterns"] if p in pattern_counts and pattern_counts[p] > 3]
            if len(patterns_present) >= 2:
                correlations.append({
                    "patterns": patterns_present,
                    "description": group["description"],
                    "actions": group["actions"]
                })
        
        return correlations
    
    async def start_realtime_monitoring(self) -> None:
        """Start real-time log monitoring"""
        if not self.realtime_enabled:
            logger.info("Real-time monitoring is disabled in configuration")
            return
        
        if self.running:
            logger.warning("Real-time monitoring is already running")
            return
        
        self.running = True
        logger.info("Starting real-time log monitoring")
        
        try:
            # Subscribe to log channels in Redis
            pubsub = self.redis_client.client.pubsub()
            pubsub.subscribe("logs:all")
            
            # Start the processing loop
            while self.running:
                message = pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    try:
                        log_data = json.loads(message["data"])
                        event = LogEvent(
                            timestamp=log_data.get("timestamp", time.time()),
                            level=log_data.get("level", "INFO"),
                            service=log_data.get("service", "unknown"),
                            message=log_data.get("message", ""),
                            context=log_data.get("extra", {}),
                            source_file=log_data.get("file", ""),
                            line_number=log_data.get("line", 0),
                            thread_id=log_data.get("thread", ""),
                            parsed=True,
                            hash_id=generate_uuid()
                        )
                        
                        # Process the event
                        await self.process_realtime_event(event)
                    except json.JSONDecodeError:
                        logger.warning("Received invalid JSON data in log message")
                
                # Sleep briefly to avoid busy waiting
                await asyncio.sleep(0.01)
        
        except Exception as e:
            logger.error(f"Error in real-time log monitoring: {str(e)}")
            self.running = False
            raise LogAnalysisError(f"Real-time monitoring error: {str(e)}")
        finally:
            self.running = False
            pubsub.unsubscribe()
            logger.info("Stopped real-time log monitoring")
    
    async def process_realtime_event(self, event: LogEvent) -> None:
        """
        Process a real-time log event
        
        Args:
            event: Log event to process
        """
        # Add to recent window
        self.realtime_window.append(event)
        
        # Update service metrics
        self.service_metrics[event.service][f"{event.level.lower()}_count"] += 1
        
        # Check patterns
        for pattern in self.patterns:
            if pattern.matches(event):
                self.realtime_pattern_counts[pattern.name] += 1
                
                # Store significant events
                if pattern.importance >= 3:
                    self.recent_events.append(event)
                    
                    # Publish alert if threshold exceeded
                    if self.realtime_pattern_counts[pattern.name] > self.alert_threshold:
                        await self.publish_alert(pattern, event)
        
        # Update anomaly detection
        if event.service in self.anomaly_models:
            if event.level == "ERROR":
                self._update_anomaly_model(event.service, "error_rate", 1)
            else:
                self._update_anomaly_model(event.service, "error_rate", 0)
            
            # Check for response time info
            if "response_time" in event.context:
                self._update_anomaly_model(
                    event.service, "response_time", float(event.context["response_time"])
                )
    
    def _update_anomaly_model(self, service: str, metric: str, value: float) -> None:
        """Update anomaly detection model with new data point"""
        model = self.anomaly_models[service][metric]
        samples = model["samples"]
        
        # Add new sample
        samples.append(value)
        
        # Trim samples list if needed
        if len(samples) > model["max_samples"]:
            samples.pop(0)
        
        # Update mean and std
        if samples:
            model["mean"] = np.mean(samples)
            model["std"] = np.std(samples) if len(samples) > 1 else 0
    
    async def publish_alert(self, pattern: LogPattern, event: LogEvent) -> None:
        """
        Publish an alert for a detected pattern
        
        Args:
            pattern: The detected pattern
            event: The event that triggered the alert
        """
        try:
            alert = {
                "timestamp": time.time(),
                "pattern": pattern.name,
                "service": event.service,
                "description": pattern.description,
                "importance": pattern.importance,
                "actions": pattern.actions,
                "count": self.realtime_pattern_counts[pattern.name],
                "sample_message": event.message,
                "context": event.context
            }
            
            # Publish to Redis channel
            alert_json = json.dumps(alert)
            await run_in_threadpool(
                self.redis_client.client.publish, "alerts:logs", alert_json
            )
            
            logger.info(f"Published alert for pattern {pattern.name}", 
                       extra={"pattern": pattern.name, "count": self.realtime_pattern_counts[pattern.name]})
        
        except Exception as e:
            logger.error(f"Error publishing alert: {str(e)}")
    
    def stop_realtime_monitoring(self) -> None:
        """Stop real-time log monitoring"""
        self.running = False
        logger.info("Stopping real-time log monitoring")
    
    async def get_recent_log_summary(self, minutes: int = 5) -> Dict[str, Any]:
        """
        Get a summary of recent logs
        
        Args:
            minutes: Number of minutes to look back
            
        Returns:
            Summary of recent logs
        """
        cutoff_time = time.time() - (minutes * 60)
        recent_logs = [e for e in self.realtime_window if e.timestamp >= cutoff_time]
        
        summary = {
            "timeframe": f"Last {minutes} minutes",
            "total_events": len(recent_logs),
            "error_count": sum(1 for e in recent_logs if e.level == "ERROR"),
            "warning_count": sum(1 for e in recent_logs if e.level == "WARNING"),
            "info_count": sum(1 for e in recent_logs if e.level == "INFO"),
            "service_counts": defaultdict(int),
            "pattern_counts": dict(self.realtime_pattern_counts),
            "significant_events": []
        }
        
        # Count by service
        for event in recent_logs:
            summary["service_counts"][event.service] += 1
        
        # Find significant events
        significant_events = [e for e in self.recent_events 
                            if e.timestamp >= cutoff_time]
        
        # Sort by timestamp and limit
        significant_events.sort(key=lambda x: x.timestamp, reverse=True)
        for event in significant_events[:10]:  # Limit to 10 events
            summary["significant_events"].append({
                "timestamp": event.timestamp,
                "formatted_time": format_timestamp(event.timestamp),
                "level": event.level,
                "service": event.service,
                "message": event.message
            })
        
        # Convert defaultdicts to dicts
        summary["service_counts"] = dict(summary["service_counts"])
        
        return summary
    
    async def analyze_service_health(self, service_name: str) -> Dict[str, Any]:
        """
        Analyze the health of a specific service
        
        Args:
            service_name: Name of service to analyze
            
        Returns:
            Health metrics and analysis for the service
        """
        # Get recent logs for this service
        service_logs = [e for e in self.realtime_window 
                     if e.service == service_name]
        
        # Calculate metrics
        total_events = len(service_logs)
        error_count = sum(1 for e in service_logs if e.level == "ERROR")
        warning_count = sum(1 for e in service_logs if e.level == "WARNING")
        
        # Get service metrics
        metrics = self.service_metrics.get(service_name, {})
        error_rate = error_count / total_events if total_events > 0 else 0
        
        # Determine health status
        health_status = "healthy"
        if error_rate > 0.1:
            health_status = "degraded"
        if error_rate > 0.3:
            health_status = "unhealthy"
        
        # Get recent errors
        recent_errors = [e for e in service_logs 
                        if e.level == "ERROR"]
        recent_errors.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Format recent errors
        formatted_errors = []
        for error in recent_errors[:5]:  # Limit to 5 errors
            formatted_errors.append({
                "timestamp": error.timestamp,
                "formatted_time": format_timestamp(error.timestamp),
                "message": error.message,
                "context": error.context
            })
        
        # Calculate response time if available
        response_times = [float(e.context.get("response_time", 0)) 
                        for e in service_logs 
                        if "response_time" in e.context]
        
        avg_response_time = np.mean(response_times) if response_times else 0
        
        # Get pattern matches
        pattern_matches = {}
        for pattern in self.patterns:
            pattern_matches[pattern.name] = sum(1 for e in service_logs 
                                             if pattern.matches(e))
        
        # Compile health report
        health_report = {
            "service": service_name,
            "health_status": health_status,
            "total_events": total_events,
            "error_count": error_count,
            "warning_count": warning_count,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "timeouts": metrics.get("timeouts", 0),
            "pattern_matches": {k: v for k, v in pattern_matches.items() if v > 0},
            "recent_errors": formatted_errors,
            "recommendations": []
        }
        
        # Generate recommendations
        if error_rate > 0.1:
            health_report["recommendations"].append(
                "Investigate the high error rate - check service dependencies and configurations"
            )
        
        if metrics.get("timeouts", 0) > 5:
            health_report["recommendations"].append(
                "Service has multiple timeouts - check network connectivity and dependent service health"
            )
        
        # Add pattern-specific recommendations
        for pattern_name, count in pattern_matches.items():
            if count > 5:
                pattern_obj = next((p for p in self.patterns if p.name == pattern_name), None)
                if pattern_obj and pattern_obj.actions:
                    for action in pattern_obj.actions:
                        if action not in health_report["recommendations"]:
                            health_report["recommendations"].append(action)
        
        return health_report
    
    async def generate_insights(self) -> Dict[str, Any]:
        """
        Generate insights from collected log data
        
        Returns:
            Dictionary with insights and recommendations
        """
        insights = {
            "summary": "System insights based on log analysis",
            "generated_at": time.time(),
            "formatted_time": format_timestamp(time.time()),
            "overall_health": "unknown",
            "service_health": {},
            "pattern_insights": [],
            "potential_issues": [],
            "recommendations": []
        }
        
        try:
            # Analyze recent logs (last hour)
            recent_analysis = await self.analyze_recent_logs(hours=1)
            
            # Determine overall health
            error_rate = recent_analysis["levels"].get("ERROR", 0) / recent_analysis["total_events"] if recent_analysis["total_events"] > 0 else 0
            if error_rate < 0.01:
                insights["overall_health"] = "healthy"
            elif error_rate < 0.05:
                insights["overall_health"] = "good"
            elif error_rate < 0.1:
                insights["overall_health"] = "degraded"
            else:
                insights["overall_health"] = "unhealthy"
            
            # Service health
            for service, metrics in recent_analysis["services"].items():
                service_error_rate = metrics.get("ERROR", 0) / metrics.get("total", 1)
                health_status = "healthy"
                if service_error_rate > 0.05:
                    health_status = "degraded"
                if service_error_rate > 0.1:
                    health_status = "unhealthy"
                
                insights["service_health"][service] = {
                    "status": health_status,
                    "error_rate": service_error_rate,
                    "event_count": metrics.get("total", 0)
                }
            
            # Pattern insights
            for pattern_name, count in sorted(
                recent_analysis["patterns_detected"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]:
                pattern_obj = next((p for p in self.patterns if p.name == pattern_name), None)
                if pattern_obj:
                    insights["pattern_insights"].append({
                        "pattern": pattern_name,
                        "count": count,
                        "description": pattern_obj.description,
                        "importance": pattern_obj.importance
                    })
            
            # Add potential issues
            insights["potential_issues"] = recent_analysis.get("potential_issues", [])
            
            # Generate recommendations
            if insights["overall_health"] in ["degraded", "unhealthy"]:
                insights["recommendations"].append(
                    "The system is experiencing higher than normal error rates - investigate service health"
                )
            
            # Add service-specific recommendations
            for service, health in insights["service_health"].items():
                if health["status"] == "unhealthy":
                    insights["recommendations"].append(
                        f"Service '{service}' is unhealthy with high error rate - check logs and dependencies"
                    )
            
            # Add anomaly recommendations
            for anomaly in recent_analysis.get("anomalies", []):
                if "description" in anomaly:
                    insights["recommendations"].append(
                        f"Investigate: {anomaly['description']}"
                    )
            
            return insights
        
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            raise LogAnalysisError(f"Failed to generate insights: {str(e)}")

    async def get_log_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get statistics about logs over a period of time
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Log statistics
        """
        try:
            # Calculate start time
            start_time = time.time() - (days * 86400)
            
            # Query for log files in the period
            log_files = []
            for root, _, files in os.walk(self.log_dir):
                for file in files:
                    if file.endswith(".log") or file.endswith(".log.gz"):
                        file_path = os.path.join(root, file)
                        file_time = os.path.getmtime(file_path)
                        
                        if file_time >= start_time:
                            log_files.append(file_path)
            
            # Get file sizes
            total_size = sum(os.path.getsize(f) for f in log_files)
            
            # Sample a subset of log files
            sampled_files = log_files
            if len(log_files) > 10:
                # Sort by modification time and take newest ones
                sampled_files = sorted(log_files, key=os.path.getmtime, reverse=True)[:10]
            
            # Analyze samples
            tasks = []
            for file_path in sampled_files:
                tasks.append(self.analyze_log_file(file_path))
            
            sample_results = await asyncio.gather(*tasks)
            
            # Aggregate sample results
            sample_events = sum(r["events_processed"] for r in sample_results)
            sample_size = sum(os.path.getsize(f) for f in sampled_files)
            
            # Estimate total events
            estimated_events = int(sample_events * (total_size / sample_size)) if sample_size > 0 else 0
            
            # Calculate daily averages
            daily_avg_size = total_size / days
            daily_avg_events = estimated_events / days
            
            # Count log files by type
            file_types = defaultdict(int)
            for file in log_files:
                service_match = re.search(r'([^/]+)\.log', os.path.basename(file))
                if service_match:
                    service = service_match.group(1)
                    file_types[service] += 1
                else:
                    file_types["unknown"] += 1
            
            # Prepare statistics
            statistics = {
                "period": f"Last {days} days",
                "total_files": len(log_files),
                "total_size_bytes": total_size,
                "total_size_human": get_human_readable_time(total_size),
                "estimated_events": estimated_events,
                "daily_average": {
                    "size_bytes": daily_avg_size,
                    "size_human": get_human_readable_time(daily_avg_size),
                    "events": daily_avg_events
                },
                "file_types": dict(file_types),
                "sample_coverage": len(sampled_files) / len(log_files) if log_files else 0
            }
            
            return statistics
        
        except Exception as e:
            logger.error(f"Error getting log statistics: {str(e)}")
            raise LogAnalysisError(f"Failed to get log statistics: {str(e)}")
