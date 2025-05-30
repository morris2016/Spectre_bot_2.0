
#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Metrics Collection System

This module handles the collection, processing, and storage of system and trading metrics
to provide real-time monitoring and historical performance analysis.
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from common.logger import get_logger
from common.utils import generate_id, truncate_float
from common.constants import (
    METRIC_TYPES, PERFORMANCE_METRICS, SYSTEM_METRICS, METRIC_PRIORITIES,
    METRIC_COLLECTION_FREQUENCY, SERVICE_NAMES, MAX_METRIC_HISTORY
)
from common.db_client import DatabaseClient, get_db_client
from common.redis_client import RedisClient
from common.exceptions import (
    MetricCollectionError, ServiceConnectionError, DataStoreError
)

logger = get_logger("metrics_collector")

class MetricsCollector:
    """
    Advanced metrics collection system that gathers, processes, and stores
    performance and system metrics from all components of the QuantumSpectre Elite
    Trading System.
    """
    
    def __init__(self, config: Dict[str, Any], db_client: DatabaseClient = None,
                 redis_client: RedisClient = None):
        """
        Initialize the MetricsCollector with configuration and database connections.
        
        Args:
            config: Configuration dictionary
            db_client: Database client for persistent storage
            redis_client: Redis client for real-time metrics
        """
        self.config = config
        self.db_client = db_client
        self._db_params = config
        self.redis_client = redis_client or RedisClient(config)
        
        # Collection settings
        self.collection_enabled = config.get("metrics", {}).get("enabled", True)
        self.collection_frequency = config.get("metrics", {}).get("collection_frequency", 
                                               METRIC_COLLECTION_FREQUENCY)
        self.persist_interval = config.get("metrics", {}).get("persist_interval", 300)  # 5 minutes
        
        # In-memory cache for real-time access
        self.metrics_cache = {
            metric_type: {} for metric_type in METRIC_TYPES
        }
        
        # Collection tasks
        self.collection_tasks = {}
        self.aggregation_task = None
        self.persistence_task = None
        
        # State tracking
        self.is_running = False
        self.last_persistence_time = time.time()

        logger.info(f"MetricsCollector initialized with frequency: {self.collection_frequency}s")

    async def initialize(self, db_connector: Optional[DatabaseClient] = None) -> None:
        """Obtain a database client and create required tables."""
        if db_connector is not None:
            self.db_client = db_connector
        if self.db_client is None:
            self.db_client = await get_db_client(**self._db_params)
        if getattr(self.db_client, "pool", None) is None:
            await self.db_client.initialize()
            await self.db_client.create_tables()
    
    async def start(self) -> None:
        """Start the metrics collection system."""
        if self.is_running:
            logger.warning("MetricsCollector is already running")
            return

        logger.info("Starting MetricsCollector...")
        await self.initialize()
        self.is_running = True
        
        # Initialize collection tasks for each service
        for service_name in SERVICE_NAMES:
            self.collection_tasks[service_name] = asyncio.create_task(
                self._collect_service_metrics(service_name)
            )
        
        # Initialize aggregation task
        self.aggregation_task = asyncio.create_task(self._aggregate_metrics())
        
        # Initialize persistence task
        self.persistence_task = asyncio.create_task(self._persist_metrics())
        
        logger.info("MetricsCollector started successfully")
    
    async def stop(self) -> None:
        """Stop the metrics collection system."""
        if not self.is_running:
            logger.warning("MetricsCollector is not running")
            return
        
        logger.info("Stopping MetricsCollector...")
        self.is_running = False
        
        # Cancel all tasks
        for task in self.collection_tasks.values():
            task.cancel()
        
        if self.aggregation_task:
            self.aggregation_task.cancel()
        
        if self.persistence_task:
            self.persistence_task.cancel()
        
        # Final persistence of all metrics
        await self._force_persist_metrics()
        
        logger.info("MetricsCollector stopped successfully")
    
    async def _collect_service_metrics(self, service_name: str) -> None:
        """
        Continuously collect metrics from a specific service.
        
        Args:
            service_name: Name of the service to collect metrics from
        """
        logger.info(f"Starting metrics collection for service: {service_name}")
        
        while self.is_running:
            try:
                # Get service-specific metrics
                metrics = await self._get_service_metrics(service_name)
                
                # Process and store metrics
                self._process_metrics(service_name, metrics)
                
                # Publish high-priority metrics to Redis for real-time access
                await self._publish_real_time_metrics(service_name, metrics)
                
            except ServiceConnectionError as e:
                logger.error(f"Failed to connect to service {service_name}: {str(e)}")
            except MetricCollectionError as e:
                logger.error(f"Error collecting metrics from {service_name}: {str(e)}")
            except Exception as e:
                logger.exception(f"Unexpected error in metrics collection for {service_name}: {str(e)}")
            
            # Wait for next collection cycle
            await asyncio.sleep(self.collection_frequency)
    
    async def _get_service_metrics(self, service_name: str) -> Dict[str, Any]:
        """
        Get metrics from a specific service.
        
        Args:
            service_name: Name of the service to get metrics from
            
        Returns:
            Dictionary of metrics from the service
        """
        try:
            # Service endpoint construction
            service_config = self.config.get("services", {}).get(service_name, {})
            service_host = service_config.get("host", "localhost")
            service_port = service_config.get("port", 8000)
            service_endpoint = f"http://{service_host}:{service_port}/metrics"
            
            # For this implementation, we'll simulate getting metrics
            # In a real system, you would make an HTTP request to the service
            metrics = self._simulate_service_metrics(service_name)
            
            return metrics
        
        except Exception as e:
            raise ServiceConnectionError(f"Failed to get metrics from {service_name}: {str(e)}")
    
    def _simulate_service_metrics(self, service_name: str) -> Dict[str, Any]:
        """
        Simulate metrics for development and testing purposes.
        In production, this would be replaced with actual metrics from the services.
        
        Args:
            service_name: Name of the service to simulate metrics for
            
        Returns:
            Dictionary of simulated metrics
        """
        timestamp = datetime.now().isoformat()
        
        # Base metrics for all services
        base_metrics = {
            "timestamp": timestamp,
            "service": service_name,
            "cpu_usage": round(np.random.uniform(0.5, 35.0), 2),
            "memory_usage": round(np.random.uniform(50.0, 500.0), 2),
            "heap_size": round(np.random.uniform(30.0, 300.0), 2),
            "garbage_collection_time": round(np.random.uniform(0.0, 100.0), 2),
            "request_count": int(np.random.uniform(10, 1000)),
            "request_latency_ms": round(np.random.uniform(1.0, 100.0), 2),
            "error_count": int(np.random.uniform(0, 10)),
        }
        
        # Add service-specific metrics
        if service_name == "data_ingest":
            base_metrics.update({
                "data_points_processed": int(np.random.uniform(1000, 10000)),
                "ingest_latency_ms": round(np.random.uniform(5.0, 50.0), 2),
                "queue_size": int(np.random.uniform(0, 500)),
                "stream_connected": True,
                "sources_online": int(np.random.uniform(5, 20)),
            })
        
        elif service_name == "intelligence":
            base_metrics.update({
                "patterns_detected": int(np.random.uniform(0, 100)),
                "prediction_confidence": round(np.random.uniform(0.6, 0.99), 3),
                "model_inference_time_ms": round(np.random.uniform(5.0, 50.0), 2),
                "active_strategies": int(np.random.uniform(5, 20)),
            })
        
        elif service_name == "brain_council":
            base_metrics.update({
                "signals_generated": int(np.random.uniform(0, 50)),
                "signal_strength_avg": round(np.random.uniform(0.7, 0.95), 3),
                "decision_time_ms": round(np.random.uniform(10.0, 100.0), 2),
                "active_brains": int(np.random.uniform(10, 50)),
                "council_consensus_level": round(np.random.uniform(0.6, 0.99), 3),
            })
        
        elif service_name == "execution_engine":
            base_metrics.update({
                "orders_placed": int(np.random.uniform(0, 30)),
                "orders_filled": int(np.random.uniform(0, 30)),
                "execution_latency_ms": round(np.random.uniform(100.0, 500.0), 2),
                "slippage_bps": round(np.random.uniform(0.0, 10.0), 2),
                "active_positions": int(np.random.uniform(0, 20)),
            })
        
        elif service_name == "risk_manager":
            base_metrics.update({
                "risk_exposure": round(np.random.uniform(0.1, 0.5), 2),
                "max_drawdown": round(np.random.uniform(0.01, 0.1), 3),
                "portfolio_volatility": round(np.random.uniform(0.005, 0.05), 4),
                "risk_checks_performed": int(np.random.uniform(50, 500)),
                "risk_alerts": int(np.random.uniform(0, 5)),
            })
        
        return base_metrics
    
    def _process_metrics(self, service_name: str, metrics: Dict[str, Any]) -> None:
        """
        Process and store metrics in the in-memory cache.
        
        Args:
            service_name: Name of the service the metrics came from
            metrics: Dictionary of metrics to process
        """
        timestamp = metrics.get("timestamp", datetime.now().isoformat())
        
        # Separate metrics by type
        system_metrics = {k: v for k, v in metrics.items() if k in SYSTEM_METRICS}
        performance_metrics = {k: v for k, v in metrics.items() if k in PERFORMANCE_METRICS}
        
        # Add to in-memory cache
        if system_metrics:
            if service_name not in self.metrics_cache["system"]:
                self.metrics_cache["system"][service_name] = []
            
            self.metrics_cache["system"][service_name].append({
                "timestamp": timestamp,
                "metrics": system_metrics
            })
            
            # Limit the size of the cache
            if len(self.metrics_cache["system"][service_name]) > MAX_METRIC_HISTORY:
                self.metrics_cache["system"][service_name].pop(0)
        
        if performance_metrics:
            if service_name not in self.metrics_cache["performance"]:
                self.metrics_cache["performance"][service_name] = []
            
            self.metrics_cache["performance"][service_name].append({
                "timestamp": timestamp,
                "metrics": performance_metrics
            })
            
            # Limit the size of the cache
            if len(self.metrics_cache["performance"][service_name]) > MAX_METRIC_HISTORY:
                self.metrics_cache["performance"][service_name].pop(0)
    
    async def _publish_real_time_metrics(self, service_name: str, metrics: Dict[str, Any]) -> None:
        """
        Publish high-priority metrics to Redis for real-time access.
        
        Args:
            service_name: Name of the service the metrics came from
            metrics: Dictionary of metrics to publish
        """
        try:
            # Filter for high-priority metrics
            high_priority_metrics = {
                k: v for k, v in metrics.items() 
                if k in METRIC_PRIORITIES and METRIC_PRIORITIES[k] == "high"
            }
            
            if not high_priority_metrics:
                return
            
            # Add timestamp and service name
            high_priority_metrics["timestamp"] = metrics.get("timestamp", datetime.now().isoformat())
            high_priority_metrics["service"] = service_name
            
            # Publish to Redis
            channel = f"metrics:{service_name}:high_priority"
            await self.redis_client.publish(channel, high_priority_metrics)
            
            # Also set the latest metrics for direct access
            key = f"metrics:latest:{service_name}"
            await self.redis_client.set(key, high_priority_metrics, ex=self.collection_frequency * 3)
            
        except Exception as e:
            logger.error(f"Failed to publish real-time metrics for {service_name}: {str(e)}")
    
    async def _aggregate_metrics(self) -> None:
        """Continuously aggregate metrics for system-wide analysis."""
        logger.info("Starting metrics aggregation task")
        
        while self.is_running:
            try:
                # Aggregate system metrics
                system_summary = self._compute_system_summary()
                
                # Aggregate performance metrics
                performance_summary = self._compute_performance_summary()
                
                # Store aggregated metrics
                timestamp = datetime.now().isoformat()
                
                if system_summary:
                    if "summary" not in self.metrics_cache["system"]:
                        self.metrics_cache["system"]["summary"] = []
                    
                    self.metrics_cache["system"]["summary"].append({
                        "timestamp": timestamp,
                        "metrics": system_summary
                    })
                    
                    # Limit the size of the cache
                    if len(self.metrics_cache["system"]["summary"]) > MAX_METRIC_HISTORY:
                        self.metrics_cache["system"]["summary"].pop(0)
                
                if performance_summary:
                    if "summary" not in self.metrics_cache["performance"]:
                        self.metrics_cache["performance"]["summary"] = []
                    
                    self.metrics_cache["performance"]["summary"].append({
                        "timestamp": timestamp,
                        "metrics": performance_summary
                    })
                    
                    # Limit the size of the cache
                    if len(self.metrics_cache["performance"]["summary"]) > MAX_METRIC_HISTORY:
                        self.metrics_cache["performance"]["summary"].pop(0)
                
                # Publish summaries to Redis
                await self._publish_summaries(system_summary, performance_summary)
                
            except Exception as e:
                logger.exception(f"Error in metrics aggregation: {str(e)}")
            
            # Wait for next aggregation cycle (typically longer than collection cycle)
            await asyncio.sleep(self.collection_frequency * 2)
    
    def _compute_system_summary(self) -> Dict[str, Any]:
        """
        Compute a summary of system metrics across all services.
        
        Returns:
            Dictionary of aggregated system metrics
        """
        # Extract the latest system metrics for each service
        latest_metrics = {}
        
        for service_name, service_metrics in self.metrics_cache["system"].items():
            if service_name == "summary" or not service_metrics:
                continue
            
            latest_metrics[service_name] = service_metrics[-1]["metrics"]
        
        if not latest_metrics:
            return {}
        
        # Compute aggregates
        summary = {
            "total_cpu_usage": sum(m.get("cpu_usage", 0) for m in latest_metrics.values()),
            "total_memory_usage_mb": sum(m.get("memory_usage", 0) for m in latest_metrics.values()),
            "total_heap_size_mb": sum(m.get("heap_size", 0) for m in latest_metrics.values()),
            "avg_request_latency_ms": np.mean([m.get("request_latency_ms", 0) for m in latest_metrics.values() if "request_latency_ms" in m]),
            "total_error_count": sum(m.get("error_count", 0) for m in latest_metrics.values()),
            "service_count": len(latest_metrics),
            "healthy_services": sum(1 for m in latest_metrics.values() if m.get("error_count", 0) < 5)
        }
        
        # Add system health score (simple heuristic)
        error_ratio = summary["total_error_count"] / max(1, sum(m.get("request_count", 0) for m in latest_metrics.values()))
        cpu_health = 1.0 - (summary["total_cpu_usage"] / (100.0 * len(latest_metrics)))
        memory_health = 1.0 - (summary["total_memory_usage_mb"] / (1000.0 * len(latest_metrics)))
        service_health = summary["healthy_services"] / summary["service_count"]
        
        health_score = (0.4 * (1.0 - error_ratio)) + (0.2 * cpu_health) + (0.2 * memory_health) + (0.2 * service_health)
        summary["system_health_score"] = round(max(0.0, min(1.0, health_score)), 3)
        
        return summary
    
    def _compute_performance_summary(self) -> Dict[str, Any]:
        """
        Compute a summary of performance metrics across all services.
        
        Returns:
            Dictionary of aggregated performance metrics
        """
        # Extract performance metrics for relevant services
        trading_metrics = {}
        
        # Focus on services that contribute to trading performance
        relevant_services = ["brain_council", "execution_engine", "risk_manager"]
        
        for service_name in relevant_services:
            if service_name in self.metrics_cache["performance"] and self.metrics_cache["performance"][service_name]:
                trading_metrics[service_name] = self.metrics_cache["performance"][service_name][-1]["metrics"]
        
        if not trading_metrics:
            return {}
        
        # Compute aggregates focusing on trading performance
        summary = {}
        
        # Brain council metrics
        if "brain_council" in trading_metrics:
            bc_metrics = trading_metrics["brain_council"]
            summary["signals_generated"] = bc_metrics.get("signals_generated", 0)
            summary["signal_strength_avg"] = bc_metrics.get("signal_strength_avg", 0)
            summary["council_consensus_level"] = bc_metrics.get("council_consensus_level", 0)
        
        # Execution metrics
        if "execution_engine" in trading_metrics:
            exec_metrics = trading_metrics["execution_engine"]
            summary["orders_placed"] = exec_metrics.get("orders_placed", 0)
            summary["orders_filled"] = exec_metrics.get("orders_filled", 0)
            summary["fill_rate"] = exec_metrics.get("orders_filled", 0) / max(1, exec_metrics.get("orders_placed", 1))
            summary["avg_slippage_bps"] = exec_metrics.get("slippage_bps", 0)
            summary["active_positions"] = exec_metrics.get("active_positions", 0)
        
        # Risk metrics
        if "risk_manager" in trading_metrics:
            risk_metrics = trading_metrics["risk_manager"]
            summary["risk_exposure"] = risk_metrics.get("risk_exposure", 0)
            summary["max_drawdown"] = risk_metrics.get("max_drawdown", 0)
            summary["portfolio_volatility"] = risk_metrics.get("portfolio_volatility", 0)
        
        return summary
    
    async def _publish_summaries(self, system_summary: Dict[str, Any], performance_summary: Dict[str, Any]) -> None:
        """
        Publish metric summaries to Redis for real-time access.
        
        Args:
            system_summary: Dictionary of aggregated system metrics
            performance_summary: Dictionary of aggregated performance metrics
        """
        try:
            timestamp = datetime.now().isoformat()
            
            if system_summary:
                system_summary["timestamp"] = timestamp
                await self.redis_client.set("metrics:summary:system", system_summary, ex=self.collection_frequency * 6)
            
            if performance_summary:
                performance_summary["timestamp"] = timestamp
                await self.redis_client.set("metrics:summary:performance", performance_summary, ex=self.collection_frequency * 6)
            
        except Exception as e:
            logger.error(f"Failed to publish metric summaries: {str(e)}")
    
    async def _persist_metrics(self) -> None:
        """Continuously persist metrics to the database for long-term storage."""
        logger.info("Starting metrics persistence task")
        
        while self.is_running:
            try:
                # Check if it's time to persist
                current_time = time.time()
                if (current_time - self.last_persistence_time) >= self.persist_interval:
                    await self._perform_persistence()
                    self.last_persistence_time = current_time
            
            except Exception as e:
                logger.exception(f"Error in metrics persistence: {str(e)}")
            
            await asyncio.sleep(min(30, self.persist_interval / 10))  # Check more frequently than the interval
    
    async def _perform_persistence(self) -> None:
        """Persist metrics to the database."""
        logger.info("Persisting metrics to database...")
        
        try:
            # Prepare data for bulk insertion
            system_metrics_list = []
            performance_metrics_list = []
            
            # Process system metrics
            for service_name, metrics_list in self.metrics_cache["system"].items():
                if not metrics_list:
                    continue
                
                # Only persist metrics since the last persistence
                cutoff_time = datetime.fromtimestamp(self.last_persistence_time)
                
                for metric_entry in metrics_list:
                    try:
                        metric_time = datetime.fromisoformat(metric_entry["timestamp"])
                        if metric_time > cutoff_time:
                            system_metrics_list.append({
                                "service_name": service_name,
                                "timestamp": metric_entry["timestamp"],
                                "metrics": metric_entry["metrics"]
                            })
                    except ValueError:
                        # Skip entries with invalid timestamps
                        continue
            
            # Process performance metrics
            for service_name, metrics_list in self.metrics_cache["performance"].items():
                if not metrics_list:
                    continue
                
                # Only persist metrics since the last persistence
                cutoff_time = datetime.fromtimestamp(self.last_persistence_time)
                
                for metric_entry in metrics_list:
                    try:
                        metric_time = datetime.fromisoformat(metric_entry["timestamp"])
                        if metric_time > cutoff_time:
                            performance_metrics_list.append({
                                "service_name": service_name,
                                "timestamp": metric_entry["timestamp"],
                                "metrics": metric_entry["metrics"]
                            })
                    except ValueError:
                        # Skip entries with invalid timestamps
                        continue
            
            # Persist to database in bulk
            if system_metrics_list:
                await self.db_client.insert_many("system_metrics", system_metrics_list)
                logger.info(f"Persisted {len(system_metrics_list)} system metrics")
            
            if performance_metrics_list:
                await self.db_client.insert_many("performance_metrics", performance_metrics_list)
                logger.info(f"Persisted {len(performance_metrics_list)} performance metrics")
            
        except DataStoreError as e:
            logger.error(f"Failed to persist metrics: {str(e)}")
        except Exception as e:
            logger.exception(f"Unexpected error in metrics persistence: {str(e)}")
    
    async def _force_persist_metrics(self) -> None:
        """Force persistence of all metrics, typically called during shutdown."""
        logger.info("Forcing persistence of all metrics...")
        self.last_persistence_time = 0  # Reset to ensure all metrics are persisted
        await self._perform_persistence()
    
    # Public API methods
    
    async def get_latest_metrics(self, service_name: Optional[str] = None, 
                                metric_type: str = "system") -> Dict[str, Any]:
        """
        Get the latest metrics for a specific service or all services.
        
        Args:
            service_name: Name of the service to get metrics for, or None for all services
            metric_type: Type of metrics to get (system or performance)
            
        Returns:
            Dictionary of the latest metrics
        """
        if metric_type not in self.metrics_cache:
            return {}
        
        if service_name is not None:
            if service_name not in self.metrics_cache[metric_type]:
                return {}
            
            metrics_list = self.metrics_cache[metric_type][service_name]
            if not metrics_list:
                return {}
            
            return {
                "service_name": service_name,
                "timestamp": metrics_list[-1]["timestamp"],
                "metrics": metrics_list[-1]["metrics"]
            }
        
        # Get latest metrics for all services
        result = {}
        
        for svc_name, metrics_list in self.metrics_cache[metric_type].items():
            if not metrics_list:
                continue
            
            result[svc_name] = {
                "timestamp": metrics_list[-1]["timestamp"],
                "metrics": metrics_list[-1]["metrics"]
            }
        
        return result
    
    async def get_metrics_history(self, service_name: str, metric_name: str, 
                                 time_range: int = 3600, metric_type: str = "system") -> List[Dict[str, Any]]:
        """
        Get historical metrics for a specific service and metric.
        
        Args:
            service_name: Name of the service to get metrics for
            metric_name: Name of the specific metric to get history for
            time_range: Time range in seconds to get history for
            metric_type: Type of metrics to get (system or performance)
            
        Returns:
            List of historical metric values with timestamps
        """
        if metric_type not in self.metrics_cache:
            return []
        
        if service_name not in self.metrics_cache[metric_type]:
            return []
        
        metrics_list = self.metrics_cache[metric_type][service_name]
        if not metrics_list:
            return []
        
        # Filter by time range and extract specific metric
        cutoff_time = datetime.now() - timedelta(seconds=time_range)
        result = []
        
        for entry in metrics_list:
            try:
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time < cutoff_time:
                    continue
                
                if metric_name in entry["metrics"]:
                    result.append({
                        "timestamp": entry["timestamp"],
                        "value": entry["metrics"][metric_name]
                    })
            except ValueError:
                # Skip entries with invalid timestamps
                continue
        
        return result
    
    async def get_system_health(self) -> Dict[str, Any]:
        """
        Get the current system health status.
        
        Returns:
            Dictionary with system health information
        """
        try:
            # Try to get from Redis first for real-time data
            health_summary = await self.redis_client.get("metrics:summary:system")
            
            if health_summary:
                return health_summary
            
            # Otherwise compute from cache
            return self._compute_system_summary()
            
        except Exception as e:
            logger.error(f"Failed to get system health: {str(e)}")
            return {}
    
    async def get_trading_performance(self) -> Dict[str, Any]:
        """
        Get the current trading performance metrics.
        
        Returns:
            Dictionary with trading performance information
        """
        try:
            # Try to get from Redis first for real-time data
            performance_summary = await self.redis_client.get("metrics:summary:performance")
            
            if performance_summary:
                return performance_summary
            
            # Otherwise compute from cache
            return self._compute_performance_summary()
            
        except Exception as e:
            logger.error(f"Failed to get trading performance: {str(e)}")
            return {}

# Singleton instance for application-wide use
_metrics_collector = None

def get_metrics_collector(config: Optional[Dict[str, Any]] = None) -> MetricsCollector:
    """
    Get or create the singleton MetricsCollector instance.
    
    Args:
        config: Configuration dictionary (only used if creating a new instance)
        
    Returns:
        The singleton MetricsCollector instance
    """
    global _metrics_collector
    
    if _metrics_collector is None and config is not None:
        _metrics_collector = MetricsCollector(config)
    
    if _metrics_collector is None:
        raise RuntimeError("MetricsCollector not initialized. Provide config on first call.")
    
    return _metrics_collector

# End of File
