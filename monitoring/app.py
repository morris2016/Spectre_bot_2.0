
#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Monitoring System Application

This module provides the main monitoring service application that orchestrates
all monitoring components, metrics collection, alerting, and reporting.
"""

import os
import sys
import time
import signal
import asyncio
import logging
import threading
import datetime
from typing import Dict, List, Set, Any, Optional, Callable, Awaitable
import concurrent.futures

# Internal imports
from common.logger import get_logger
from common.exceptions import ServiceStartupError, ServiceShutdownError
from common.constants import MONITORING_CONFIG, SERVICE_STATUS
from common.redis_client import RedisClient
from common.db_client import DatabaseClient, get_db_client
from common.async_utils import create_task_with_error_handling, run_in_executor
from common.utils import chunked_iterable, merge_configs

import monitoring
from monitoring.metrics_collector import MetricsCollector
from monitoring.alerting import AlertingSystem
from monitoring.performance_tracker import PerformanceTracker
from monitoring.system_health import SystemHealthMonitor
from monitoring.log_analyzer import LogAnalyzer

# Constants
DEFAULT_MONITORING_INTERVAL_SEC = 10
DEFAULT_HEALTH_CHECK_INTERVAL_SEC = 30
DEFAULT_METRICS_FLUSH_INTERVAL_SEC = 60
DEFAULT_ANOMALY_DETECTION_INTERVAL_SEC = 300
DEFAULT_LOG_ANALYSIS_INTERVAL_SEC = 600

class MonitoringService:
    """
    Main monitoring service that coordinates all monitoring activities.
    
    This service is responsible for:
    - Collecting system and performance metrics
    - Monitoring system health
    - Generating alerts for anomalies and issues
    - Tracking trading performance
    - Analyzing logs for patterns and issues
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the monitoring service.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = get_logger("monitoring.service")
        self.config = merge_configs(MONITORING_CONFIG, config or {})
        self.is_running = False
        self.tasks = []
        self.lock = threading.RLock()
        self.event_loop = None
        
        # Monitoring components
        self.metrics_collector = None
        self.alerting_system = None
        self.performance_tracker = None
        self.system_health_monitor = None
        self.log_analyzer = None
        
        # Database clients
        self.redis_client = None
        self.db_client = None
        
        # Service references
        self.service_refs = {}
        
        # Status
        self.last_error = None
        self.start_time = None
        self.status = SERVICE_STATUS.INITIALIZED
        
        # Intervals
        self.monitoring_interval = self.config.get("monitoring_interval", DEFAULT_MONITORING_INTERVAL_SEC)
        self.health_check_interval = self.config.get("health_check_interval", DEFAULT_HEALTH_CHECK_INTERVAL_SEC)
        self.metrics_flush_interval = self.config.get("metrics_flush_interval", DEFAULT_METRICS_FLUSH_INTERVAL_SEC)
        self.anomaly_detection_interval = self.config.get("anomaly_detection_interval", DEFAULT_ANOMALY_DETECTION_INTERVAL_SEC)
        self.log_analysis_interval = self.config.get("log_analysis_interval", DEFAULT_LOG_ANALYSIS_INTERVAL_SEC)
        
        self.logger.info("Monitoring service initialized with configuration")
    
    async def initialize(self, db_connector: Optional[DatabaseClient] = None) -> None:
        """Initialize the monitoring service and all its components."""
        self.logger.info("Initializing monitoring service...")
        try:
            self.event_loop = asyncio.get_running_loop()

            # Initialize database connections
            self.redis_client = RedisClient(
                host=self.config.get("redis", {}).get("host", "localhost"),
                port=self.config.get("redis", {}).get("port", 6379),
                db=self.config.get("redis", {}).get("db", 0),
                password=self.config.get("redis", {}).get("password", None),
            )
            await self.redis_client.initialize()

            if db_connector is not None:
                self.db_client = db_connector
            if self.db_client is None:
                self.db_client = await get_db_client(
                    db_type=self.config.get("database", {}).get("type", "postgresql"),
                    host=self.config.get("database", {}).get("host", "localhost"),
                    port=self.config.get("database", {}).get("port", 5432),
                    username=self.config.get("database", {}).get("username", "postgres"),
                    password=self.config.get("database", {}).get("password", ""),
                    database=self.config.get("database", {}).get("database", "quantumspectre"),
                    pool_size=self.config.get("database", {}).get("pool_size", 10),
                    ssl=self.config.get("database", {}).get("ssl", False),
                    timeout=self.config.get("database", {}).get("timeout", 30),
                )
            if getattr(self.db_client, "pool", None) is None:
                await self.db_client.initialize()
                await self.db_client.create_tables()
            
            # Initialize monitoring components
            self.metrics_collector = monitoring.get_component("metrics_collector")
            self.alerting_system = monitoring.get_component("alerting_system")
            self.performance_tracker = monitoring.get_component("performance_tracker")
            self.system_health_monitor = monitoring.get_component("system_health_monitor")
            self.log_analyzer = monitoring.get_component("log_analyzer")
            
            # Configure components
            await monitoring.initialize_monitoring(self.config)
            monitoring.register_exporters(self.config)
            monitoring.register_alert_handlers(self.config)
            
            # Set service references
            self.system_health_monitor.register_services(self.service_refs)
            
            # Register common metrics
            self._register_common_metrics()
            
            self.status = SERVICE_STATUS.INITIALIZED
            self.logger.info("Monitoring service initialization complete")
        except Exception as e:
            self.status = SERVICE_STATUS.ERROR
            self.last_error = str(e)
            self.logger.error(f"Error initializing monitoring service: {str(e)}")
            raise ServiceStartupError(f"Failed to initialize monitoring service: {str(e)}") from e
    
    def _register_common_metrics(self) -> None:
        """Register common metrics that should be tracked by the system."""
        # System metrics
        self.metrics_collector.register_gauge("system.cpu.usage", "CPU usage percentage")
        self.metrics_collector.register_gauge("system.memory.usage", "Memory usage percentage")
        self.metrics_collector.register_gauge("system.disk.usage", "Disk usage percentage")
        self.metrics_collector.register_gauge("system.network.bytes_sent", "Network bytes sent")
        self.metrics_collector.register_gauge("system.network.bytes_received", "Network bytes received")
        
        # Application metrics
        self.metrics_collector.register_gauge("app.threads.active", "Number of active threads")
        self.metrics_collector.register_gauge("app.event_loop.lag", "Event loop lag in milliseconds")
        self.metrics_collector.register_gauge("app.memory.usage", "Application memory usage")
        
        # Database metrics
        self.metrics_collector.register_gauge("db.connections.active", "Number of active database connections")
        self.metrics_collector.register_gauge("db.connections.idle", "Number of idle database connections")
        self.metrics_collector.register_gauge("db.query.latency", "Database query latency in milliseconds")
        
        # Redis metrics
        self.metrics_collector.register_gauge("redis.connections.active", "Number of active Redis connections")
        self.metrics_collector.register_gauge("redis.memory.usage", "Redis memory usage")
        self.metrics_collector.register_gauge("redis.commands.processed", "Number of Redis commands processed")
        
        # Trading metrics
        self.metrics_collector.register_counter("trading.signals.generated", "Number of trading signals generated")
        self.metrics_collector.register_counter("trading.orders.placed", "Number of orders placed")
        self.metrics_collector.register_counter("trading.orders.filled", "Number of orders filled")
        self.metrics_collector.register_counter("trading.orders.canceled", "Number of orders canceled")
        self.metrics_collector.register_gauge("trading.positions.open", "Number of open positions")
        self.metrics_collector.register_gauge("trading.balance", "Account balance")
        self.metrics_collector.register_gauge("trading.pnl.daily", "Daily profit and loss")
        self.metrics_collector.register_gauge("trading.pnl.total", "Total profit and loss")
        self.metrics_collector.register_gauge("trading.win_rate", "Trading win rate")
        
        # Strategy metrics
        self.metrics_collector.register_gauge("strategy.confidence", "Strategy confidence level")
        self.metrics_collector.register_counter("strategy.brain_council.votes", "Number of brain council votes processed")
        self.metrics_collector.register_gauge("strategy.performance.sharpe", "Strategy Sharpe ratio")
        
        # Alert metrics
        self.metrics_collector.register_counter("alerts.generated", "Number of alerts generated")
        self.metrics_collector.register_counter("alerts.critical", "Number of critical alerts")
        self.metrics_collector.register_counter("alerts.warning", "Number of warning alerts")
        self.metrics_collector.register_counter("alerts.info", "Number of info alerts")
    
    def register_service(self, service_name: str, service_instance: Any) -> None:
        """
        Register a service for health monitoring.
        
        Args:
            service_name: Name of the service
            service_instance: Service instance to monitor
        """
        with self.lock:
            self.service_refs[service_name] = service_instance
            if self.system_health_monitor:
                self.system_health_monitor.register_service(service_name, service_instance)
    
    async def start(self) -> None:
        """
        Start the monitoring service and all scheduled tasks.
        
        Raises:
            ServiceStartupError: If startup fails
        """
        self.logger.info("Starting monitoring service...")
        
        if self.is_running:
            self.logger.warning("Monitoring service already running, ignoring start request")
            return
        
        try:
            if self.status != SERVICE_STATUS.INITIALIZED:
                await self.initialize()
            
            self.start_time = datetime.datetime.now()
            self.is_running = True
            self.status = SERVICE_STATUS.STARTING
            
            # Start monitoring tasks
            self.tasks = [
                create_task_with_error_handling(
                    self._collect_metrics_task(), 
                    "metrics_collection", 
                    self._handle_task_error
                ),
                create_task_with_error_handling(
                    self._check_system_health_task(), 
                    "health_check", 
                    self._handle_task_error
                ),
                create_task_with_error_handling(
                    self._flush_metrics_task(), 
                    "metrics_flush", 
                    self._handle_task_error
                ),
                create_task_with_error_handling(
                    self._detect_anomalies_task(), 
                    "anomaly_detection", 
                    self._handle_task_error
                ),
                create_task_with_error_handling(
                    self._analyze_logs_task(), 
                    "log_analysis", 
                    self._handle_task_error
                ),
                create_task_with_error_handling(
                    self._track_performance_task(), 
                    "performance_tracking", 
                    self._handle_task_error
                )
            ]
            
            self.logger.info("All monitoring tasks started")
            self.status = SERVICE_STATUS.RUNNING
            
            # Report startup metrics
            self.metrics_collector.increment("monitoring.service.starts")
            self.metrics_collector.gauge("monitoring.uptime_seconds", 0)
            
            # Send startup alert
            self.alerting_system.send_alert(
                "Monitoring service started",
                "The QuantumSpectre monitoring service has started successfully.",
                "info",
                {"startup_time": self.start_time.isoformat()}
            )
            
            self.logger.info("Monitoring service started successfully")
        except Exception as e:
            self.is_running = False
            self.status = SERVICE_STATUS.ERROR
            self.last_error = str(e)
            self.logger.error(f"Error starting monitoring service: {str(e)}")
            raise ServiceStartupError(f"Failed to start monitoring service: {str(e)}") from e
    
    async def stop(self) -> None:
        """
        Stop the monitoring service and all scheduled tasks.
        
        Raises:
            ServiceShutdownError: If shutdown fails
        """
        self.logger.info("Stopping monitoring service...")
        
        if not self.is_running:
            self.logger.warning("Monitoring service not running, ignoring stop request")
            return
        
        try:
            self.status = SERVICE_STATUS.STOPPING
            self.is_running = False
            
            # Cancel all tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete or cancel
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            
            # Shutdown monitoring components
            monitoring.shutdown_monitoring()
            
            # Close database connections
            if self.redis_client:
                await self.redis_client.close()

            if self.db_client:
                await self.db_client.close()
            
            # Record uptime before shutdown
            if self.start_time:
                uptime = (datetime.datetime.now() - self.start_time).total_seconds()
                self.metrics_collector.gauge("monitoring.uptime_seconds", uptime)
            
            # Flush metrics one last time
            await run_in_executor(self.metrics_collector.flush_all)
            
            # Send shutdown alert
            self.alerting_system.send_alert(
                "Monitoring service stopped",
                "The QuantumSpectre monitoring service has been stopped.",
                "info",
                {}
            )
            
            self.tasks = []
            self.status = SERVICE_STATUS.STOPPED
            self.logger.info("Monitoring service stopped successfully")
        except Exception as e:
            self.status = SERVICE_STATUS.ERROR
            self.last_error = str(e)
            self.logger.error(f"Error stopping monitoring service: {str(e)}")
            raise ServiceShutdownError(f"Failed to stop monitoring service: {str(e)}") from e
    
    def _handle_task_error(self, task_name: str, exception: Exception) -> None:
        """
        Handle errors in monitoring tasks.
        
        Args:
            task_name: Name of the task that failed
            exception: Exception that occurred
        """
        self.logger.error(f"Error in monitoring task '{task_name}': {str(exception)}")
        self.metrics_collector.increment("monitoring.task.errors", {"task": task_name})
        
        if self.is_running:
            # Restart the task if it's a critical one
            if task_name in ["metrics_collection", "health_check"]:
                self.logger.info(f"Restarting critical monitoring task: {task_name}")
                if task_name == "metrics_collection":
                    new_task = create_task_with_error_handling(
                        self._collect_metrics_task(), 
                        task_name, 
                        self._handle_task_error
                    )
                elif task_name == "health_check":
                    new_task = create_task_with_error_handling(
                        self._check_system_health_task(), 
                        task_name, 
                        self._handle_task_error
                    )
                
                # Replace the failed task in the tasks list
                with self.lock:
                    self.tasks = [t for t in self.tasks if t.get_name() != task_name]
                    self.tasks.append(new_task)
    
    async def _collect_metrics_task(self) -> None:
        """Task to periodically collect system and application metrics."""
        self.logger.info("Starting metrics collection task")
        
        while self.is_running:
            try:
                # Collect system metrics
                await run_in_executor(self._collect_system_metrics)
                
                # Collect application metrics
                await run_in_executor(self._collect_application_metrics)
                
                # Collect trading metrics
                await run_in_executor(self._collect_trading_metrics)
                
                # Update uptime
                if self.start_time:
                    uptime = (datetime.datetime.now() - self.start_time).total_seconds()
                    self.metrics_collector.gauge("monitoring.uptime_seconds", uptime)
                
                # Success metric
                self.metrics_collector.increment("monitoring.metrics_collection.success")
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {str(e)}")
                self.metrics_collector.increment("monitoring.metrics_collection.errors")
            
            # Wait for next collection interval
            await asyncio.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            # CPU usage (simulated for now, would use psutil in production)
            cpu_usage = 50  # Replace with actual CPU usage measurement
            self.metrics_collector.gauge("system.cpu.usage", cpu_usage)
            
            # Memory usage (simulated for now, would use psutil in production)
            memory_usage = 40  # Replace with actual memory usage measurement
            self.metrics_collector.gauge("system.memory.usage", memory_usage)
            
            # Disk usage (simulated for now, would use psutil in production)
            disk_usage = 30  # Replace with actual disk usage measurement
            self.metrics_collector.gauge("system.disk.usage", disk_usage)
            
            # Network usage (simulated for now, would use psutil in production)
            net_sent = 1024 * 1024  # Replace with actual bytes sent
            net_recv = 2048 * 1024  # Replace with actual bytes received
            self.metrics_collector.gauge("system.network.bytes_sent", net_sent)
            self.metrics_collector.gauge("system.network.bytes_received", net_recv)
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {str(e)}")
            raise
    
    def _collect_application_metrics(self) -> None:
        """Collect application-level metrics."""
        try:
            # Thread count
            thread_count = threading.active_count()
            self.metrics_collector.gauge("app.threads.active", thread_count)
            
            # Event loop lag (simulated for now)
            event_loop_lag = 5  # Replace with actual event loop lag measurement in ms
            self.metrics_collector.gauge("app.event_loop.lag", event_loop_lag)
            
            # Application memory usage (simulated for now)
            app_memory = 500 * 1024 * 1024  # Replace with actual memory usage
            self.metrics_collector.gauge("app.memory.usage", app_memory)
            
            # Database metrics (retrieve from db client)
            if self.db_client:
                db_stats = self.db_client.get_stats()
                self.metrics_collector.gauge("db.connections.active", db_stats.get("active_connections", 0))
                self.metrics_collector.gauge("db.connections.idle", db_stats.get("idle_connections", 0))
                self.metrics_collector.gauge("db.query.latency", db_stats.get("average_query_time_ms", 0))
            
            # Redis metrics (retrieve from redis client)
            if self.redis_client:
                redis_stats = self.redis_client.get_stats()
                self.metrics_collector.gauge("redis.connections.active", redis_stats.get("active_connections", 0))
                self.metrics_collector.gauge("redis.memory.usage", redis_stats.get("memory_usage", 0))
                self.metrics_collector.gauge("redis.commands.processed", redis_stats.get("commands_processed", 0))
        except Exception as e:
            self.logger.error(f"Error collecting application metrics: {str(e)}")
            raise
    
    def _collect_trading_metrics(self) -> None:
        """Collect trading-related metrics from various services."""
        try:
            # Trading performance metrics would be gathered from other services
            # This is a placeholder for demonstration
            
            # Example signal metrics (would come from brain council)
            signals_generated = 100  # Example value
            self.metrics_collector.gauge("trading.signals.generated", signals_generated)
            
            # Example order metrics (would come from execution engine)
            orders_placed = 80  # Example value
            orders_filled = 75  # Example value
            orders_canceled = 5  # Example value
            self.metrics_collector.gauge("trading.orders.placed", orders_placed)
            self.metrics_collector.gauge("trading.orders.filled", orders_filled)
            self.metrics_collector.gauge("trading.orders.canceled", orders_canceled)
            
            # Example position metrics (would come from position manager)
            open_positions = 10  # Example value
            account_balance = 1050.25  # Example value
            self.metrics_collector.gauge("trading.positions.open", open_positions)
            self.metrics_collector.gauge("trading.balance", account_balance)
            
            # Example PnL metrics (would come from performance tracker)
            daily_pnl = 25.75  # Example value
            total_pnl = 150.25  # Example value
            win_rate = 82.5  # Example value (percentage)
            self.metrics_collector.gauge("trading.pnl.daily", daily_pnl)
            self.metrics_collector.gauge("trading.pnl.total", total_pnl)
            self.metrics_collector.gauge("trading.win_rate", win_rate)
            
            # Example strategy metrics (would come from brain council)
            strategy_confidence = 0.85  # Example value
            brain_council_votes = 50  # Example value
            strategy_sharpe = 1.85  # Example value
            self.metrics_collector.gauge("strategy.confidence", strategy_confidence)
            self.metrics_collector.gauge("strategy.brain_council.votes", brain_council_votes)
            self.metrics_collector.gauge("strategy.performance.sharpe", strategy_sharpe)
        except Exception as e:
            self.logger.error(f"Error collecting trading metrics: {str(e)}")
            raise
    
    async def _check_system_health_task(self) -> None:
        """Task to periodically check system health."""
        self.logger.info("Starting system health check task")
        
        while self.is_running:
            try:
                # Run health checks
                health_results = await run_in_executor(
                    self.system_health_monitor.check_all_services
                )
                
                # Process health check results
                for service_name, health_status in health_results.items():
                    self.metrics_collector.gauge(
                        "service.health", 
                        1 if health_status.get("healthy", False) else 0,
                        {"service": service_name}
                    )
                    
                    # Generate alerts for unhealthy services
                    if not health_status.get("healthy", False):
                        self.alerting_system.send_alert(
                            f"Service health check failed: {service_name}",
                            health_status.get("message", "Unknown error"),
                            "warning",
                            {"service": service_name, "details": health_status}
                        )
                
                # Success metric
                self.metrics_collector.increment("monitoring.health_check.success")
            except Exception as e:
                self.logger.error(f"Error checking system health: {str(e)}")
                self.metrics_collector.increment("monitoring.health_check.errors")
            
            # Wait for next health check interval
            await asyncio.sleep(self.health_check_interval)
    
    async def _flush_metrics_task(self) -> None:
        """Task to periodically flush metrics to external systems."""
        self.logger.info("Starting metrics flush task")
        
        while self.is_running:
            try:
                # Flush metrics to all configured exporters
                await run_in_executor(self.metrics_collector.flush_all)
                
                # Success metric
                self.metrics_collector.increment("monitoring.metrics_flush.success")
            except Exception as e:
                self.logger.error(f"Error flushing metrics: {str(e)}")
                self.metrics_collector.increment("monitoring.metrics_flush.errors")
            
            # Wait for next flush interval
            await asyncio.sleep(self.metrics_flush_interval)
    
    async def _detect_anomalies_task(self) -> None:
        """Task to periodically detect anomalies in metrics and system behavior."""
        self.logger.info("Starting anomaly detection task")
        
        while self.is_running:
            try:
                # Run anomaly detection on metrics
                anomalies = await run_in_executor(
                    self.system_health_monitor.detect_anomalies
                )
                
                # Process detected anomalies
                for anomaly in anomalies:
                    self.metrics_collector.increment(
                        "monitoring.anomalies_detected",
                        {"metric": anomaly.get("metric", "unknown")}
                    )
                    
                    # Generate alerts for detected anomalies
                    self.alerting_system.send_alert(
                        f"Anomaly detected: {anomaly.get('metric', 'unknown')}",
                        anomaly.get("description", "Unusual metric behavior detected"),
                        anomaly.get("severity", "warning"),
                        {"details": anomaly}
                    )
                
                # Success metric
                self.metrics_collector.increment("monitoring.anomaly_detection.success")
            except Exception as e:
                self.logger.error(f"Error detecting anomalies: {str(e)}")
                self.metrics_collector.increment("monitoring.anomaly_detection.errors")
            
            # Wait for next anomaly detection interval
            await asyncio.sleep(self.anomaly_detection_interval)
    
    async def _analyze_logs_task(self) -> None:
        """Task to periodically analyze logs for patterns and issues."""
        self.logger.info("Starting log analysis task")
        
        while self.is_running:
            try:
                # Analyze logs for patterns and issues
                log_findings = await run_in_executor(
                    self.log_analyzer.analyze_recent_logs
                )
                
                # Process log analysis findings
                for finding in log_findings:
                    self.metrics_collector.increment(
                        "monitoring.log_findings",
                        {"type": finding.get("type", "unknown")}
                    )
                    
                    # Generate alerts for significant log findings
                    if finding.get("severity", "info") in ["warning", "error", "critical"]:
                        self.alerting_system.send_alert(
                            f"Log analysis finding: {finding.get('type', 'unknown')}",
                            finding.get("description", "Interesting pattern detected in logs"),
                            finding.get("severity", "info"),
                            {"details": finding}
                        )
                
                # Success metric
                self.metrics_collector.increment("monitoring.log_analysis.success")
            except Exception as e:
                self.logger.error(f"Error analyzing logs: {str(e)}")
                self.metrics_collector.increment("monitoring.log_analysis.errors")
            
            # Wait for next log analysis interval
            await asyncio.sleep(self.log_analysis_interval)
    
    async def _track_performance_task(self) -> None:
        """Task to track trading performance metrics."""
        self.logger.info("Starting performance tracking task")
        
        while self.is_running:
            try:
                # Update performance metrics
                await run_in_executor(
                    self.performance_tracker.update_all_metrics
                )
                
                # Get latest performance summary
                performance_summary = self.performance_tracker.get_performance_summary()
                
                # Check for performance thresholds and generate alerts
                if performance_summary.get("win_rate", 0) < 70:
                    self.alerting_system.send_alert(
                        "Win rate below threshold",
                        f"Current win rate is {performance_summary.get('win_rate')}%, which is below the target threshold of 70%",
                        "warning",
                        {"performance": performance_summary}
                    )
                
                # Check for significant profit/loss
                daily_pnl = performance_summary.get("daily_pnl", 0)
                if abs(daily_pnl) > 100:  # Example threshold
                    severity = "info" if daily_pnl > 0 else "warning"
                    self.alerting_system.send_alert(
                        f"Significant {'profit' if daily_pnl > 0 else 'loss'} today",
                        f"Daily P&L is {daily_pnl}, which is a significant {'profit' if daily_pnl > 0 else 'loss'}",
                        severity,
                        {"performance": performance_summary}
                    )
                
                # Success metric
                self.metrics_collector.increment("monitoring.performance_tracking.success")
            except Exception as e:
                self.logger.error(f"Error tracking performance: {str(e)}")
                self.metrics_collector.increment("monitoring.performance_tracking.errors")
            
            # Wait for next performance tracking interval
            await asyncio.sleep(self.monitoring_interval)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the monitoring service.
        
        Returns:
            Dictionary with service status information
        """
        status_info = {
            "status": self.status,
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": (datetime.datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "last_error": self.last_error,
            "active_tasks": len([t for t in self.tasks if not t.done()]) if self.tasks else 0,
            "metrics_collection_interval": self.monitoring_interval,
            "health_check_interval": self.health_check_interval,
            "metrics_flush_interval": self.metrics_flush_interval,
            "anomaly_detection_interval": self.anomaly_detection_interval,
            "log_analysis_interval": self.log_analysis_interval,
            "components": {
                "metrics_collector": self.metrics_collector is not None,
                "alerting_system": self.alerting_system is not None,
                "performance_tracker": self.performance_tracker is not None,
                "system_health_monitor": self.system_health_monitor is not None,
                "log_analyzer": self.log_analyzer is not None
            }
        }
        return status_info


# Singleton instance
_service_instance = None

def get_monitoring_service(config: Dict[str, Any] = None) -> MonitoringService:
    """
    Get or create the singleton MonitoringService instance.
    
    Args:
        config: Optional configuration dictionary
    
    Returns:
        MonitoringService instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = MonitoringService(config)
    return _service_instance


if __name__ == "__main__":
    """Run the monitoring service directly for testing or standalone mode."""
    import argparse
    parser = argparse.ArgumentParser(description="QuantumSpectre Monitoring Service")
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration if provided
    config = {}
    if args.config:
        import json
        with open(args.config, "r") as f:
            config = json.load(f)
    
    # Run the service
    async def main():
        service = get_monitoring_service(config)
        
        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(service.stop()))
        
        try:
            await service.initialize()
            await service.start()
            
            # Keep running until stopped
            while service.is_running:
                await asyncio.sleep(1)
        except Exception as e:
            logging.error(f"Error running monitoring service: {str(e)}")
        finally:
            if service.is_running:
                await service.stop()
    
    asyncio.run(main())

