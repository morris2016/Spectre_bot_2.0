
#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Dashboard Module

This module provides real-time monitoring dashboard capabilities for system metrics,
trading performance, and health status with advanced visualization.
"""

import os
import re
import time
import json
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from collections import defaultdict, deque

from common.logger import get_logger
from common.utils import (
    timeit, generate_uuid, timestamp_to_datetime, datetime_to_timestamp, 
    format_timestamp, get_human_readable_time
)
from common.redis_client import RedisClient
from common.constants import (
    DASHBOARD_SECTIONS, METRICS_CATEGORIES, TRADING_METRIC_THRESHOLDS
)
from common.exceptions import DashboardError
from common.async_utils import run_in_threadpool

logger = get_logger("dashboard")


class DashboardMetricProvider:
    """
    Provides metrics for the monitoring dashboard from various system components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dashboard metric provider
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.redis_client = RedisClient(config["redis"])
        self.cache_ttl = config.get("cache_ttl", 30)  # Cache TTL in seconds
        self.cache = {}
        self.cache_timestamps = {}
        self.channel_subscriptions = {}
        self.event_buffer = defaultdict(lambda: deque(maxlen=1000))
        logger.info("Dashboard metric provider initialized")
    
    async def start_event_collector(self) -> None:
        """Start collecting events from Redis channels for real-time metrics"""
        if self.channel_subscriptions:
            logger.warning("Event collector is already running")
            return
        
        try:
            logger.info("Starting dashboard event collector")
            pubsub = self.redis_client.client.pubsub()
            
            # Subscribe to relevant channels
            channels = [
                "metrics:system",
                "metrics:trading",
                "metrics:performance",
                "alerts:system",
                "alerts:trading",
                "alerts:security"
            ]
            
            await run_in_threadpool(pubsub.subscribe, *channels)
            self.channel_subscriptions["pubsub"] = pubsub
            self.channel_subscriptions["running"] = True
            
            # Start the background task
            asyncio.create_task(self._event_collector_loop(pubsub))
        
        except Exception as e:
            logger.error(f"Failed to start event collector: {str(e)}")
            raise DashboardError(f"Failed to start dashboard event collector: {str(e)}")
    
    async def _event_collector_loop(self, pubsub) -> None:
        """
        Background loop for collecting events from Redis
        
        Args:
            pubsub: Redis PubSub object for subscriptions
        """
        try:
            while self.channel_subscriptions.get("running", False):
                # Get message with a timeout
                message = await run_in_threadpool(pubsub.get_message, timeout=0.1)
                
                if message and message["type"] == "message":
                    channel = message["channel"].decode("utf-8")
                    try:
                        data = json.loads(message["data"])
                        
                        # Add timestamp if not present
                        if "timestamp" not in data:
                            data["timestamp"] = time.time()
                        
                        # Store in the appropriate buffer
                        self.event_buffer[channel].append(data)
                    except json.JSONDecodeError:
                        logger.warning(f"Received invalid JSON data on channel {channel}")
                
                # Small sleep to avoid busy-waiting
                await asyncio.sleep(0.01)
        
        except Exception as e:
            logger.error(f"Error in event collector loop: {str(e)}")
        finally:
            logger.info("Dashboard event collector loop stopped")
    
    def stop_event_collector(self) -> None:
        """Stop the event collector"""
        if "running" in self.channel_subscriptions:
            self.channel_subscriptions["running"] = False
            logger.info("Stopping dashboard event collector")
    
    async def _get_cached_or_fetch(self, key: str, fetch_func: Callable, *args, **kwargs) -> Any:
        """
        Get data from cache or fetch it using the provided function
        
        Args:
            key: Cache key
            fetch_func: Function to fetch data if not in cache
            *args, **kwargs: Arguments to pass to fetch_func
            
        Returns:
            Cached or freshly fetched data
        """
        current_time = time.time()
        
        # Check if we have a cached value that's still valid
        if key in self.cache:
            cache_time = self.cache_timestamps.get(key, 0)
            if current_time - cache_time < self.cache_ttl:
                return self.cache[key]
        
        # Fetch fresh data
        data = await fetch_func(*args, **kwargs)
        
        # Cache the result
        self.cache[key] = data
        self.cache_timestamps[key] = current_time
        
        return data
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """
        Get an overview of the system status
        
        Returns:
            System overview data
        """
        return await self._get_cached_or_fetch(
            "system_overview", self._fetch_system_overview
        )
    
    async def _fetch_system_overview(self) -> Dict[str, Any]:
        """
        Fetch system overview data
        
        Returns:
            System overview data
        """
        try:
            # Get system metrics from Redis
            system_metrics_key = "metrics:system:latest"
            system_metrics_data = await run_in_threadpool(
                self.redis_client.client.get, system_metrics_key
            )
            
            if system_metrics_data:
                system_metrics = json.loads(system_metrics_data)
            else:
                system_metrics = {
                    "cpu_usage": 0,
                    "memory_usage": 0,
                    "disk_usage": 0,
                    "network_in": 0,
                    "network_out": 0,
                    "timestamp": time.time()
                }
            
            # Get service status
            service_status_key = "status:services"
            service_status_data = await run_in_threadpool(
                self.redis_client.client.get, service_status_key
            )
            
            if service_status_data:
                service_status = json.loads(service_status_data)
            else:
                service_status = {}
            
            # Count services by status
            status_counts = defaultdict(int)
            for service, status in service_status.items():
                status_counts[status["status"]] += 1
            
            # Get recent alerts
            recent_alerts = []
            for alert in list(self.event_buffer.get("alerts:system", []))[-5:]:
                recent_alerts.append({
                    "timestamp": alert.get("timestamp", 0),
                    "formatted_time": format_timestamp(alert.get("timestamp", 0)),
                    "level": alert.get("level", "INFO"),
                    "message": alert.get("message", ""),
                    "source": alert.get("source", "")
                })
            
            # Compile overview
            overview = {
                "timestamp": time.time(),
                "formatted_time": format_timestamp(time.time()),
                "system_metrics": {
                    "cpu_usage": system_metrics.get("cpu_usage", 0),
                    "memory_usage": system_metrics.get("memory_usage", 0),
                    "disk_usage": system_metrics.get("disk_usage", 0),
                    "network_in": system_metrics.get("network_in", 0),
                    "network_out": system_metrics.get("network_out", 0)
                },
                "services": {
                    "total": len(service_status),
                    "running": status_counts.get("running", 0),
                    "degraded": status_counts.get("degraded", 0),
                    "stopped": status_counts.get("stopped", 0),
                    "unknown": status_counts.get("unknown", 0)
                },
                "recent_alerts": recent_alerts,
                "system_health": self._calculate_system_health(system_metrics, status_counts)
            }
            
            return overview
        
        except Exception as e:
            logger.error(f"Error fetching system overview: {str(e)}")
            raise DashboardError(f"Failed to fetch system overview: {str(e)}")
    
    def _calculate_system_health(
        self, metrics: Dict[str, Any], service_status: Dict[str, int]
    ) -> str:
        """
        Calculate overall system health based on metrics and service status
        
        Args:
            metrics: System metrics
            service_status: Service status counts
            
        Returns:
            Health status string: "healthy", "warning", "critical", or "unknown"
        """
        try:
            # Start with assumption of health
            health = "healthy"
            
            # Check resource usage
            cpu_usage = metrics.get("cpu_usage", 0)
            memory_usage = metrics.get("memory_usage", 0)
            disk_usage = metrics.get("disk_usage", 0)
            
            # Resource thresholds for warning
            if cpu_usage > 80 or memory_usage > 80 or disk_usage > 85:
                health = "warning"
            
            # Resource thresholds for critical
            if cpu_usage > 95 or memory_usage > 95 or disk_usage > 95:
                health = "critical"
            
            # Service status affects health
            total_services = sum(service_status.values())
            if total_services > 0:
                running_pct = service_status.get("running", 0) / total_services
                
                # If less than 90% of services are running, it's at least a warning
                if running_pct < 0.9:
                    health = "warning"
                
                # If less than 75% of services are running, it's critical
                if running_pct < 0.75:
                    health = "critical"
                
                # If any services are in degraded state
                if service_status.get("degraded", 0) > 0 and health == "healthy":
                    health = "warning"
            
            return health
        
        except Exception as e:
            logger.error(f"Error calculating system health: {str(e)}")
            return "unknown"
    
    async def get_service_status(self) -> Dict[str, Any]:
        """
        Get detailed status of all services
        
        Returns:
            Service status data
        """
        return await self._get_cached_or_fetch(
            "service_status", self._fetch_service_status
        )
    
    async def _fetch_service_status(self) -> Dict[str, Any]:
        """
        Fetch service status data
        
        Returns:
            Service status data
        """
        try:
            # Get service status
            service_status_key = "status:services"
            service_status_data = await run_in_threadpool(
                self.redis_client.client.get, service_status_key
            )
            
            if service_status_data:
                service_status = json.loads(service_status_data)
            else:
                service_status = {}
            
            # Get service metrics
            service_metrics_key = "metrics:services"
            service_metrics_data = await run_in_threadpool(
                self.redis_client.client.get, service_metrics_key
            )
            
            if service_metrics_data:
                service_metrics = json.loads(service_metrics_data)
            else:
                service_metrics = {}
            
            # Combine status and metrics
            result = {
                "timestamp": time.time(),
                "formatted_time": format_timestamp(time.time()),
                "services": {}
            }
            
            for service, status in service_status.items():
                result["services"][service] = {
                    "status": status.get("status", "unknown"),
                    "uptime": status.get("uptime", 0),
                    "last_updated": status.get("last_updated", 0),
                    "metrics": service_metrics.get(service, {}),
                    "health": status.get("health", "unknown"),
                    "issues": status.get("issues", [])
                }
            
            return result
        
        except Exception as e:
            logger.error(f"Error fetching service status: {str(e)}")
            raise DashboardError(f"Failed to fetch service status: {str(e)}")
    
    async def get_trading_performance(self) -> Dict[str, Any]:
        """
        Get trading performance metrics
        
        Returns:
            Trading performance data
        """
        return await self._get_cached_or_fetch(
            "trading_performance", self._fetch_trading_performance
        )
    
    async def _fetch_trading_performance(self) -> Dict[str, Any]:
        """
        Fetch trading performance data
        
        Returns:
            Trading performance data
        """
        try:
            # Get overall trading metrics
            trading_metrics_key = "metrics:trading:overview"
            trading_metrics_data = await run_in_threadpool(
                self.redis_client.client.get, trading_metrics_key
            )
            
            if trading_metrics_data:
                trading_metrics = json.loads(trading_metrics_data)
            else:
                trading_metrics = {
                    "win_rate": 0,
                    "profit_factor": 0,
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_profit": 0,
                    "timestamp": time.time()
                }
            
            # Get platform specific metrics
            platforms = ["binance", "deriv"]
            platform_metrics = {}
            
            for platform in platforms:
                platform_key = f"metrics:trading:{platform}"
                platform_data = await run_in_threadpool(
                    self.redis_client.client.get, platform_key
                )
                
                if platform_data:
                    platform_metrics[platform] = json.loads(platform_data)
                else:
                    platform_metrics[platform] = {
                        "win_rate": 0,
                        "profit_factor": 0,
                        "total_trades": 0
                    }
            
            # Get recent trades
            recent_trades_key = "trading:recent_trades"
            recent_trades_data = await run_in_threadpool(
                self.redis_client.client.hgetall, recent_trades_key
            )
            
            recent_trades = []
            for _, trade_data in recent_trades_data.items():
                trade = json.loads(trade_data)
                recent_trades.append(trade)
            
            # Sort by timestamp
            recent_trades.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            
            # Limit to 10 most recent
            recent_trades = recent_trades[:10]
            
            # Format trades for display
            formatted_trades = []
            for trade in recent_trades:
                formatted_trades.append({
                    "id": trade.get("id", ""),
                    "timestamp": trade.get("timestamp", 0),
                    "formatted_time": format_timestamp(trade.get("timestamp", 0)),
                    "platform": trade.get("platform", ""),
                    "asset": trade.get("asset", ""),
                    "type": trade.get("type", ""),
                    "direction": trade.get("direction", ""),
                    "entry_price": trade.get("entry_price", 0),
                    "exit_price": trade.get("exit_price", 0),
                    "profit_loss": trade.get("profit_loss", 0),
                    "profit_percentage": trade.get("profit_percentage", 0),
                    "status": trade.get("status", ""),
                    "strategy": trade.get("strategy", "")
                })
            
            # Get asset performance
            asset_performance_key = "metrics:trading:assets"
            asset_performance_data = await run_in_threadpool(
                self.redis_client.client.get, asset_performance_key
            )
            
            if asset_performance_data:
                asset_performance = json.loads(asset_performance_data)
            else:
                asset_performance = {}
            
            # Compile performance data
            performance = {
                "timestamp": time.time(),
                "formatted_time": format_timestamp(time.time()),
                "overall": {
                    "win_rate": trading_metrics.get("win_rate", 0),
                    "profit_factor": trading_metrics.get("profit_factor", 0),
                    "total_trades": trading_metrics.get("total_trades", 0),
                    "winning_trades": trading_metrics.get("winning_trades", 0),
                    "losing_trades": trading_metrics.get("losing_trades", 0),
                    "total_profit": trading_metrics.get("total_profit", 0),
                    "average_profit": trading_metrics.get("average_profit", 0),
                    "average_loss": trading_metrics.get("average_loss", 0),
                    "max_drawdown": trading_metrics.get("max_drawdown", 0),
                    "sharpe_ratio": trading_metrics.get("sharpe_ratio", 0)
                },
                "platforms": platform_metrics,
                "assets": asset_performance,
                "recent_trades": formatted_trades,
                "health": self._calculate_trading_health(trading_metrics)
            }
            
            return performance
        
        except Exception as e:
            logger.error(f"Error fetching trading performance: {str(e)}")
            raise DashboardError(f"Failed to fetch trading performance: {str(e)}")
    
    def _calculate_trading_health(self, metrics: Dict[str, Any]) -> str:
        """
        Calculate trading system health based on performance metrics
        
        Args:
            metrics: Trading performance metrics
            
        Returns:
            Health status string: "excellent", "good", "average", "poor", or "unknown"
        """
        try:
            # Define thresholds
            thresholds = TRADING_METRIC_THRESHOLDS
            
            # Check win rate
            win_rate = metrics.get("win_rate", 0)
            profit_factor = metrics.get("profit_factor", 0)
            max_drawdown = metrics.get("max_drawdown", 0)
            
            # Determine health based on win rate and profit factor
            if win_rate >= thresholds["excellent"]["win_rate"] and profit_factor >= thresholds["excellent"]["profit_factor"]:
                health = "excellent"
            elif win_rate >= thresholds["good"]["win_rate"] and profit_factor >= thresholds["good"]["profit_factor"]:
                health = "good"
            elif win_rate >= thresholds["average"]["win_rate"] and profit_factor >= thresholds["average"]["profit_factor"]:
                health = "average"
            else:
                health = "poor"
            
            # Adjust based on drawdown
            if abs(max_drawdown) > thresholds["poor"]["max_drawdown"] and health != "poor":
                health = "average"  # Downgrade if drawdown is too high
            
            # Check if we have enough trades for a meaningful assessment
            total_trades = metrics.get("total_trades", 0)
            if total_trades < 20:
                health = "insufficient_data"
            
            return health
        
        except Exception as e:
            logger.error(f"Error calculating trading health: {str(e)}")
            return "unknown"
    
    async def get_strategy_performance(self) -> Dict[str, Any]:
        """
        Get performance metrics for different trading strategies
        
        Returns:
            Strategy performance data
        """
        return await self._get_cached_or_fetch(
            "strategy_performance", self._fetch_strategy_performance
        )
    
    async def _fetch_strategy_performance(self) -> Dict[str, Any]:
        """
        Fetch strategy performance data
        
        Returns:
            Strategy performance data
        """
        try:
            # Get strategy metrics
            strategy_metrics_key = "metrics:trading:strategies"
            strategy_metrics_data = await run_in_threadpool(
                self.redis_client.client.get, strategy_metrics_key
            )
            
            if strategy_metrics_data:
                strategy_metrics = json.loads(strategy_metrics_data)
            else:
                strategy_metrics = {}
            
            # Get brain performance
            brain_metrics_key = "metrics:trading:brains"
            brain_metrics_data = await run_in_threadpool(
                self.redis_client.client.get, brain_metrics_key
            )
            
            if brain_metrics_data:
                brain_metrics = json.loads(brain_metrics_data)
            else:
                brain_metrics = {}
            
            # Compile performance data
            performance = {
                "timestamp": time.time(),
                "formatted_time": format_timestamp(time.time()),
                "strategies": {},
                "brains": {},
                "best_performing": {
                    "strategy": None,
                    "brain": None
                }
            }
            
            # Process strategy metrics
            best_strategy = None
            best_strategy_win_rate = 0
            
            for strategy, metrics in strategy_metrics.items():
                performance["strategies"][strategy] = {
                    "win_rate": metrics.get("win_rate", 0),
                    "profit_factor": metrics.get("profit_factor", 0),
                    "total_trades": metrics.get("total_trades", 0),
                    "total_profit": metrics.get("total_profit", 0),
                    "average_profit": metrics.get("average_profit", 0),
                    "max_drawdown": metrics.get("max_drawdown", 0)
                }
                
                # Track best performing strategy
                if metrics.get("win_rate", 0) > best_strategy_win_rate and metrics.get("total_trades", 0) >= 10:
                    best_strategy_win_rate = metrics.get("win_rate", 0)
                    best_strategy = strategy
            
            # Process brain metrics
            best_brain = None
            best_brain_win_rate = 0
            
            for brain, metrics in brain_metrics.items():
                performance["brains"][brain] = {
                    "win_rate": metrics.get("win_rate", 0),
                    "profit_factor": metrics.get("profit_factor", 0),
                    "total_trades": metrics.get("total_trades", 0),
                    "total_profit": metrics.get("total_profit", 0),
                    "average_profit": metrics.get("average_profit", 0),
                    "confidence": metrics.get("confidence", 0)
                }
                
                # Track best performing brain
                if metrics.get("win_rate", 0) > best_brain_win_rate and metrics.get("total_trades", 0) >= 10:
                    best_brain_win_rate = metrics.get("win_rate", 0)
                    best_brain = brain
            
            # Set best performers
            performance["best_performing"]["strategy"] = best_strategy
            performance["best_performing"]["brain"] = best_brain
            
            return performance
        
        except Exception as e:
            logger.error(f"Error fetching strategy performance: {str(e)}")
            raise DashboardError(f"Failed to fetch strategy performance: {str(e)}")
    
    async def get_system_alerts(self, limit: int = 100) -> Dict[str, Any]:
        """
        Get recent system alerts
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            System alerts data
        """
        return await self._get_cached_or_fetch(
            "system_alerts", self._fetch_system_alerts, limit
        )
    
    async def _fetch_system_alerts(self, limit: int = 100) -> Dict[str, Any]:
        """
        Fetch system alerts
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            System alerts data
        """
        try:
            # Collect alerts from different channels
            all_alerts = []
            
            # System alerts
            for alert in list(self.event_buffer.get("alerts:system", [])):
                all_alerts.append({
                    "timestamp": alert.get("timestamp", 0),
                    "category": "system",
                    "level": alert.get("level", "INFO"),
                    "message": alert.get("message", ""),
                    "source": alert.get("source", ""),
                    "details": alert.get("details", {})
                })
            
            # Trading alerts
            for alert in list(self.event_buffer.get("alerts:trading", [])):
                all_alerts.append({
                    "timestamp": alert.get("timestamp", 0),
                    "category": "trading",
                    "level": alert.get("level", "INFO"),
                    "message": alert.get("message", ""),
                    "source": alert.get("source", ""),
                    "details": alert.get("details", {})
                })
            
            # Security alerts
            for alert in list(self.event_buffer.get("alerts:security", [])):
                all_alerts.append({
                    "timestamp": alert.get("timestamp", 0),
                    "category": "security",
                    "level": alert.get("level", "INFO"),
                    "message": alert.get("message", ""),
                    "source": alert.get("source", ""),
                    "details": alert.get("details", {})
                })
            
            # Sort by timestamp (newest first)
            all_alerts.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            
            # Limit number of alerts
            all_alerts = all_alerts[:limit]
            
            # Format timestamps
            for alert in all_alerts:
                alert["formatted_time"] = format_timestamp(alert.get("timestamp", 0))
            
            # Group alerts by category
            categorized_alerts = defaultdict(list)
            for alert in all_alerts:
                categorized_alerts[alert["category"]].append(alert)
            
            # Count alerts by level
            level_counts = defaultdict(int)
            for alert in all_alerts:
                level_counts[alert["level"]] += 1
            
            # Compile result
            result = {
                "timestamp": time.time(),
                "formatted_time": format_timestamp(time.time()),
                "total_alerts": len(all_alerts),
                "alerts_by_level": dict(level_counts),
                "alerts_by_category": {k: len(v) for k, v in categorized_alerts.items()},
                "alerts": all_alerts
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error fetching system alerts: {str(e)}")
            raise DashboardError(f"Failed to fetch system alerts: {str(e)}")
    
    async def get_resource_usage_history(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get historical resource usage data
        
        Args:
            hours: Number of hours of history to retrieve
            
        Returns:
            Resource usage history data
        """
        return await self._get_cached_or_fetch(
            f"resource_history_{hours}", self._fetch_resource_usage_history, hours
        )
    
    async def _fetch_resource_usage_history(self, hours: int = 24) -> Dict[str, Any]:
        """
        Fetch historical resource usage data
        
        Args:
            hours: Number of hours of history to retrieve
            
        Returns:
            Resource usage history data
        """
        try:
            # Calculate start time
            start_time = time.time() - (hours * 3600)
            
            # Get resource history from Redis
            history_key = "metrics:system:history"
            history_data = await run_in_threadpool(
                self.redis_client.client.zrangebyscore,
                history_key, start_time, "+inf"
            )
            
            # Parse metrics
            metrics_history = {
                "timestamps": [],
                "cpu_usage": [],
                "memory_usage": [],
                "disk_usage": [],
                "network_in": [],
                "network_out": []
            }
            
            for data_point in history_data:
                try:
                    metrics = json.loads(data_point)
                    
                    metrics_history["timestamps"].append(metrics.get("timestamp", 0))
                    metrics_history["cpu_usage"].append(metrics.get("cpu_usage", 0))
                    metrics_history["memory_usage"].append(metrics.get("memory_usage", 0))
                    metrics_history["disk_usage"].append(metrics.get("disk_usage", 0))
                    metrics_history["network_in"].append(metrics.get("network_in", 0))
                    metrics_history["network_out"].append(metrics.get("network_out", 0))
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON data in resource history")
            
            # Calculate averages
            avg_cpu = sum(metrics_history["cpu_usage"]) / len(metrics_history["cpu_usage"]) if metrics_history["cpu_usage"] else 0
            avg_memory = sum(metrics_history["memory_usage"]) / len(metrics_history["memory_usage"]) if metrics_history["memory_usage"] else 0
            avg_disk = sum(metrics_history["disk_usage"]) / len(metrics_history["disk_usage"]) if metrics_history["disk_usage"] else 0
            
            # Calculate peak usage
            peak_cpu = max(metrics_history["cpu_usage"]) if metrics_history["cpu_usage"] else 0
            peak_memory = max(metrics_history["memory_usage"]) if metrics_history["memory_usage"] else 0
            peak_disk = max(metrics_history["disk_usage"]) if metrics_history["disk_usage"] else 0
            
            # Compile result
            result = {
                "timestamp": time.time(),
                "formatted_time": format_timestamp(time.time()),
                "period": f"Last {hours} hours",
                "data_points": len(metrics_history["timestamps"]),
                "history": metrics_history,
                "averages": {
                    "cpu_usage": avg_cpu,
                    "memory_usage": avg_memory,
                    "disk_usage": avg_disk
                },
                "peaks": {
                    "cpu_usage": peak_cpu,
                    "memory_usage": peak_memory,
                    "disk_usage": peak_disk
                }
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error fetching resource usage history: {str(e)}")
            raise DashboardError(f"Failed to fetch resource usage history: {str(e)}")
    
    async def get_trading_history(self, days: int = 30) -> Dict[str, Any]:
        """
        Get historical trading performance data
        
        Args:
            days: Number of days of history to retrieve
            
        Returns:
            Trading history data
        """
        return await self._get_cached_or_fetch(
            f"trading_history_{days}", self._fetch_trading_history, days
        )
    
    async def _fetch_trading_history(self, days: int = 30) -> Dict[str, Any]:
        """
        Fetch historical trading performance data
        
        Args:
            days: Number of days of history to retrieve
            
        Returns:
            Trading history data
        """
        try:
            # Calculate start time
            start_time = time.time() - (days * 86400)
            
            # Get trading history from Redis
            history_key = "metrics:trading:daily"
            history_data = await run_in_threadpool(
                self.redis_client.client.zrangebyscore,
                history_key, start_time, "+inf"
            )
            
            # Parse metrics
            daily_metrics = []
            
            for data_point in history_data:
                try:
                    metrics = json.loads(data_point)
                    daily_metrics.append(metrics)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON data in trading history")
            
            # Sort by timestamp
            daily_metrics.sort(key=lambda x: x.get("timestamp", 0))
            
            # Extract time series data
            history = {
                "timestamps": [],
                "win_rates": [],
                "profit_factors": [],
                "total_profits": [],
                "trade_counts": [],
                "equity_curve": []
            }
            
            # Track cumulative equity
            equity = 100  # Start with base 100
            
            for metrics in daily_metrics:
                timestamp = metrics.get("timestamp", 0)
                win_rate = metrics.get("win_rate", 0)
                profit_factor = metrics.get("profit_factor", 0)
                daily_profit = metrics.get("total_profit", 0)
                trade_count = metrics.get("total_trades", 0)
                
                # Update equity curve
                equity += daily_profit
                
                # Append to time series
                history["timestamps"].append(timestamp)
                history["win_rates"].append(win_rate)
                history["profit_factors"].append(profit_factor)
                history["total_profits"].append(daily_profit)
                history["trade_counts"].append(trade_count)
                history["equity_curve"].append(equity)
            
            # Calculate summary metrics
            total_trades = sum(history["trade_counts"])
            total_profit = sum(history["total_profits"])
            avg_win_rate = sum(history["win_rates"]) / len(history["win_rates"]) if history["win_rates"] else 0
            
            # Calculate growth metrics
            starting_equity = 100
            final_equity = history["equity_curve"][-1] if history["equity_curve"] else 100
            total_return = ((final_equity / starting_equity) - 1) * 100
            
            # Compile result
            result = {
                "timestamp": time.time(),
                "formatted_time": format_timestamp(time.time()),
                "period": f"Last {days} days",
                "days_with_trades": len(daily_metrics),
                "history": history,
                "summary": {
                    "total_trades": total_trades,
                    "total_profit": total_profit,
                    "average_win_rate": avg_win_rate,
                    "total_return_percentage": total_return,
                    "starting_equity": starting_equity,
                    "final_equity": final_equity
                }
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error fetching trading history: {str(e)}")
            raise DashboardError(f"Failed to fetch trading history: {str(e)}")
    
    async def get_loophole_detector_status(self) -> Dict[str, Any]:
        """
        Get status of loophole detection systems
        
        Returns:
            Loophole detector status data
        """
        return await self._get_cached_or_fetch(
            "loophole_detector_status", self._fetch_loophole_detector_status
        )
    
    async def _fetch_loophole_detector_status(self) -> Dict[str, Any]:
        """
        Fetch loophole detector status data
        
        Returns:
            Loophole detector status data
        """
        try:
            # Get loophole detector status
            detector_status_key = "status:loophole_detectors"
            detector_status_data = await run_in_threadpool(
                self.redis_client.client.get, detector_status_key
            )
            
            if detector_status_data:
                detector_status = json.loads(detector_status_data)
            else:
                detector_status = {}
            
            # Get recent loopholes discovered
            loopholes_key = "loopholes:recent"
            loopholes_data = await run_in_threadpool(
                self.redis_client.client.lrange, loopholes_key, 0, 9
            )
            
            loopholes = []
            for loophole_data in loopholes_data:
                try:
                    loophole = json.loads(loophole_data)
                    loopholes.append(loophole)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON data in loopholes list")
            
            # Format loopholes
            formatted_loopholes = []
            for loophole in loopholes:
                formatted_loopholes.append({
                    "id": loophole.get("id", ""),
                    "timestamp": loophole.get("timestamp", 0),
                    "formatted_time": format_timestamp(loophole.get("timestamp", 0)),
                    "platform": loophole.get("platform", ""),
                    "asset": loophole.get("asset", ""),
                    "type": loophole.get("type", ""),
                    "description": loophole.get("description", ""),
                    "confidence": loophole.get("confidence", 0),
                    "exploitation_status": loophole.get("exploitation_status", ""),
                    "profit_potential": loophole.get("profit_potential", 0)
                })
            
            # Get loophole exploitation metrics
            exploitation_key = "metrics:loopholes:exploitation"
            exploitation_data = await run_in_threadpool(
                self.redis_client.client.get, exploitation_key
            )
            
            if exploitation_data:
                exploitation_metrics = json.loads(exploitation_data)
            else:
                exploitation_metrics = {
                    "total_detected": 0,
                    "total_exploited": 0,
                    "successful_exploits": 0,
                    "total_profit": 0
                }
            
            # Compile result
            result = {
                "timestamp": time.time(),
                "formatted_time": format_timestamp(time.time()),
                "detectors": {},
                "recent_loopholes": formatted_loopholes,
                "exploitation_metrics": exploitation_metrics
            }
            
            # Process detector status
            for detector, status in detector_status.items():
                result["detectors"][detector] = {
                    "status": status.get("status", "unknown"),
                    "last_detection": status.get("last_detection", 0),
                    "total_detections": status.get("total_detections", 0),
                    "success_rate": status.get("success_rate", 0),
                    "asset_focus": status.get("asset_focus", [])
                }
            
            return result
        
        except Exception as e:
            logger.error(f"Error fetching loophole detector status: {str(e)}")
            raise DashboardError(f"Failed to fetch loophole detector status: {str(e)}")
    
    async def get_brain_council_status(self) -> Dict[str, Any]:
        """
        Get status of brain council and decision-making systems
        
        Returns:
            Brain council status data
        """
        return await self._get_cached_or_fetch(
            "brain_council_status", self._fetch_brain_council_status
        )
    
    async def _fetch_brain_council_status(self) -> Dict[str, Any]:
        """
        Fetch brain council status data
        
        Returns:
            Brain council status data
        """
        try:
            # Get brain council status
            council_status_key = "status:brain_council"
            council_status_data = await run_in_threadpool(
                self.redis_client.client.get, council_status_key
            )
            
            if council_status_data:
                council_status = json.loads(council_status_data)
            else:
                council_status = {}
            
            # Get recent council decisions
            decisions_key = "brain_council:recent_decisions"
            decisions_data = await run_in_threadpool(
                self.redis_client.client.lrange, decisions_key, 0, 9
            )
            
            decisions = []
            for decision_data in decisions_data:
                try:
                    decision = json.loads(decision_data)
                    decisions.append(decision)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON data in decisions list")
            
            # Format decisions
            formatted_decisions = []
            for decision in decisions:
                formatted_decisions.append({
                    "id": decision.get("id", ""),
                    "timestamp": decision.get("timestamp", 0),
                    "formatted_time": format_timestamp(decision.get("timestamp", 0)),
                    "asset": decision.get("asset", ""),
                    "platform": decision.get("platform", ""),
                    "direction": decision.get("direction", ""),
                    "confidence": decision.get("confidence", 0),
                    "contributing_brains": decision.get("contributing_brains", []),
                    "result": decision.get("result", ""),
                    "profit_loss": decision.get("profit_loss", 0)
                })
            
            # Get brain council metrics
            metrics_key = "metrics:brain_council"
            metrics_data = await run_in_threadpool(
                self.redis_client.client.get, metrics_key
            )
            
            if metrics_data:
                council_metrics = json.loads(metrics_data)
            else:
                council_metrics = {
                    "total_decisions": 0,
                    "winning_decisions": 0,
                    "win_rate": 0,
                    "average_confidence": 0
                }
            
            # Get brain weights
            weights_key = "brain_council:weights"
            weights_data = await run_in_threadpool(
                self.redis_client.client.get, weights_key
            )
            
            if weights_data:
                brain_weights = json.loads(weights_data)
            else:
                brain_weights = {}
            
            # Compile result
            result = {
                "timestamp": time.time(),
                "formatted_time": format_timestamp(time.time()),
                "councils": {},
                "recent_decisions": formatted_decisions,
                "metrics": council_metrics,
                "brain_weights": brain_weights
            }
            
            # Process council status
            for council, status in council_status.items():
                result["councils"][council] = {
                    "status": status.get("status", "unknown"),
                    "active_brains": status.get("active_brains", 0),
                    "last_decision": status.get("last_decision", 0),
                    "decision_rate": status.get("decision_rate", 0),
                    "focus": status.get("focus", "")
                }
            
            return result
        
        except Exception as e:
            logger.error(f"Error fetching brain council status: {str(e)}")
            raise DashboardError(f"Failed to fetch brain council status: {str(e)}")
    
    async def get_voice_advisor_logs(self, limit: int = 20) -> Dict[str, Any]:
        """
        Get recent voice advisor logs and notifications
        
        Args:
            limit: Maximum number of logs to return
            
        Returns:
            Voice advisor logs data
        """
        return await self._get_cached_or_fetch(
            "voice_advisor_logs", self._fetch_voice_advisor_logs, limit
        )
    
    async def _fetch_voice_advisor_logs(self, limit: int = 20) -> Dict[str, Any]:
        """
        Fetch voice advisor logs
        
        Args:
            limit: Maximum number of logs to return
            
        Returns:
            Voice advisor logs data
        """
        try:
            # Get voice advisor logs
            logs_key = "logs:voice_advisor"
            logs_data = await run_in_threadpool(
                self.redis_client.client.lrange, logs_key, 0, limit - 1
            )
            
            logs = []
            for log_data in logs_data:
                try:
                    log = json.loads(log_data)
                    logs.append(log)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON data in voice advisor logs")
            
            # Format logs
            formatted_logs = []
            for log in logs:
                formatted_logs.append({
                    "id": log.get("id", ""),
                    "timestamp": log.get("timestamp", 0),
                    "formatted_time": format_timestamp(log.get("timestamp", 0)),
                    "type": log.get("type", ""),
                    "message": log.get("message", ""),
                    "confidence": log.get("confidence", 0),
                    "category": log.get("category", ""),
                    "asset": log.get("asset", ""),
                    "platform": log.get("platform", ""),
                    "action_taken": log.get("action_taken", False)
                })
            
            # Sort by timestamp (newest first)
            formatted_logs.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            
            # Get advisor status
            status_key = "status:voice_advisor"
            status_data = await run_in_threadpool(
                self.redis_client.client.get, status_key
            )
            
            if status_data:
                advisor_status = json.loads(status_data)
            else:
                advisor_status = {
                    "status": "unknown",
                    "notifications_enabled": True,
                    "last_notification": 0
                }
            
            # Compile result
            result = {
                "timestamp": time.time(),
                "formatted_time": format_timestamp(time.time()),
                "advisor_status": advisor_status,
                "logs": formatted_logs,
                "logs_by_type": defaultdict(int)
            }
            
            # Count logs by type
            for log in formatted_logs:
                result["logs_by_type"][log["type"]] += 1
            
            # Convert defaultdict to regular dict
            result["logs_by_type"] = dict(result["logs_by_type"])
            
            return result
        
        except Exception as e:
            logger.error(f"Error fetching voice advisor logs: {str(e)}")
            raise DashboardError(f"Failed to fetch voice advisor logs: {str(e)}")
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive data for the dashboard
        
        Returns:
            Complete dashboard data
        """
        try:
            logger.info("Fetching comprehensive dashboard data")
            
            # Fetch all data components in parallel
            tasks = [
                self.get_system_overview(),
                self.get_service_status(),
                self.get_trading_performance(),
                self.get_strategy_performance(),
                self.get_system_alerts(limit=10),
                self.get_loophole_detector_status(),
                self.get_brain_council_status(),
                self.get_voice_advisor_logs(limit=5)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Compile comprehensive dashboard data
            dashboard_data = {
                "timestamp": time.time(),
                "formatted_time": format_timestamp(time.time()),
                "system_overview": results[0],
                "service_status": results[1],
                "trading_performance": results[2],
                "strategy_performance": results[3],
                "recent_alerts": results[4],
                "loophole_detector": results[5],
                "brain_council": results[6],
                "voice_advisor": results[7]
            }
            
            # Generate system health summary
            dashboard_data["system_health"] = {
                "overall": results[0].get("system_health", "unknown"),
                "trading": results[2].get("health", "unknown"),
                "services_running": results[1]["services"].keys(),
                "alert_count": results[4]["total_alerts"],
                "high_priority_alerts": sum(1 for a in results[4]["alerts"] if a.get("level") in ["ERROR", "CRITICAL"])
            }
            
            logger.info("Successfully assembled comprehensive dashboard data")
            return dashboard_data
        
        except Exception as e:
            logger.error(f"Error getting comprehensive dashboard data: {str(e)}")
            raise DashboardError(f"Failed to get dashboard data: {str(e)}")
    
    async def get_historical_dashboard_data(self, days: int = 7) -> Dict[str, Any]:
        """
        Get historical dashboard data for trend analysis
        
        Args:
            days: Number of days of history to retrieve
            
        Returns:
            Historical dashboard data
        """
        try:
            logger.info(f"Fetching historical dashboard data for {days} days")
            
            # Fetch trading and resource history in parallel
            tasks = [
                self.get_trading_history(days),
                self.get_resource_usage_history(hours=days * 24)
            ]
            
            results = await asyncio.gather(*tasks)
            
            trading_history = results[0]
            resource_history = results[1]
            
            # Get archived alerts
            archived_alerts_key = "metrics:alerts:history"
            archived_alerts_data = await run_in_threadpool(
                self.redis_client.client.zrangebyscore,
                archived_alerts_key, time.time() - (days * 86400), "+inf"
            )
            
            # Parse archived alerts
            alerts_by_day = defaultdict(lambda: defaultdict(int))
            
            for alert_data in archived_alerts_data:
                try:
                    alert = json.loads(alert_data)
                    timestamp = alert.get("timestamp", 0)
                    level = alert.get("level", "INFO")
                    
                    # Convert timestamp to date string (YYYY-MM-DD)
                    date_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                    
                    # Increment counter for this day and level
                    alerts_by_day[date_str][level] += 1
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON data in archived alerts")
            
            # Format alert history for time series
            alert_history = {
                "dates": [],
                "info_counts": [],
                "warning_counts": [],
                "error_counts": [],
                "critical_counts": []
            }
            
            # Sort dates
            sorted_dates = sorted(alerts_by_day.keys())
            
            for date in sorted_dates:
                counts = alerts_by_day[date]
                alert_history["dates"].append(date)
                alert_history["info_counts"].append(counts.get("INFO", 0))
                alert_history["warning_counts"].append(counts.get("WARNING", 0))
                alert_history["error_counts"].append(counts.get("ERROR", 0))
                alert_history["critical_counts"].append(counts.get("CRITICAL", 0))
            
            # Compile historical data
            historical_data = {
                "timestamp": time.time(),
                "formatted_time": format_timestamp(time.time()),
                "period": f"Last {days} days",
                "trading_history": trading_history,
                "resource_history": resource_history,
                "alert_history": alert_history
            }
            
            # Generate trend analysis
            if trading_history["days_with_trades"] > 0:
                win_rates = trading_history["history"]["win_rates"]
                profit_factors = trading_history["history"]["profit_factors"]
                
                # Calculate win rate trend (simple linear regression)
                if len(win_rates) >= 3:
                    x = list(range(len(win_rates)))
                    y = win_rates
                    n = len(x)
                    
                    sum_x = sum(x)
                    sum_y = sum(y)
                    sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y))
                    sum_xx = sum(x_i * x_i for x_i in x)
                    
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x) if (n * sum_xx - sum_x * sum_x) != 0 else 0
                    
                    win_rate_trend = "improving" if slope > 0.005 else "declining" if slope < -0.005 else "stable"
                else:
                    win_rate_trend = "insufficient_data"
                
                # Analyze equity curve
                equity_curve = trading_history["history"]["equity_curve"]
                if len(equity_curve) >= 2:
                    equity_trend = "improving" if equity_curve[-1] > equity_curve[0] else "declining"
                else:
                    equity_trend = "insufficient_data"
                
                historical_data["trend_analysis"] = {
                    "win_rate_trend": win_rate_trend,
                    "equity_trend": equity_trend,
                    "win_rate_slope": slope if 'slope' in locals() else 0,
                    "total_return": trading_history["summary"]["total_return_percentage"]
                }
            
            logger.info("Successfully assembled historical dashboard data")
            return historical_data
        
        except Exception as e:
            logger.error(f"Error getting historical dashboard data: {str(e)}")
            raise DashboardError(f"Failed to get historical dashboard data: {str(e)}")


class Dashboard:
    """
    Real-time monitoring dashboard with advanced visualization capabilities.
    Provides a comprehensive view of system status, trading performance,
    and health metrics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the dashboard with configuration
        
        Args:
            config: Configuration dictionary for dashboard
        """
        self.config = config
        self.metric_provider = DashboardMetricProvider(config)
        logger.info("Dashboard initialized with configuration", extra={"config": config})
    
    async def start(self) -> None:
        """Start the dashboard and metric collection"""
        try:
            logger.info("Starting dashboard services")
            await self.metric_provider.start_event_collector()
            logger.info("Dashboard services started successfully")
        except Exception as e:
            logger.error(f"Failed to start dashboard: {str(e)}")
            raise DashboardError(f"Failed to start dashboard: {str(e)}")
    
    async def stop(self) -> None:
        """Stop the dashboard and metric collection"""
        try:
            logger.info("Stopping dashboard services")
            self.metric_provider.stop_event_collector()
            logger.info("Dashboard services stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping dashboard: {str(e)}")
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive dashboard data for the UI
        
        Returns:
            Complete dashboard data
        """
        return await self.metric_provider.get_dashboard_data()
    
    async def get_section_data(self, section: str, **kwargs) -> Dict[str, Any]:
        """
        Get data for a specific dashboard section
        
        Args:
            section: Section name
            **kwargs: Additional parameters for the section
            
        Returns:
            Section-specific data
        """
        try:
            if section == "system_overview":
                return await self.metric_provider.get_system_overview()
            elif section == "service_status":
                return await self.metric_provider.get_service_status()
            elif section == "trading_performance":
                return await self.metric_provider.get_trading_performance()
            elif section == "strategy_performance":
                return await self.metric_provider.get_strategy_performance()
            elif section == "alerts":
                limit = kwargs.get("limit", 100)
                return await self.metric_provider.get_system_alerts(limit=limit)
            elif section == "resource_history":
                hours = kwargs.get("hours", 24)
                return await self.metric_provider.get_resource_usage_history(hours=hours)
            elif section == "trading_history":
                days = kwargs.get("days", 30)
                return await self.metric_provider.get_trading_history(days=days)
            elif section == "loophole_detector":
                return await self.metric_provider.get_loophole_detector_status()
            elif section == "brain_council":
                return await self.metric_provider.get_brain_council_status()
            elif section == "voice_advisor":
                limit = kwargs.get("limit", 20)
                return await self.metric_provider.get_voice_advisor_logs(limit=limit)
            elif section == "historical_dashboard":
                days = kwargs.get("days", 7)
                return await self.metric_provider.get_historical_dashboard_data(days=days)
            else:
                raise DashboardError(f"Unknown dashboard section: {section}")
        
        except Exception as e:
            logger.error(f"Error fetching section data for {section}: {str(e)}")
            raise DashboardError(f"Failed to fetch data for section {section}: {str(e)}")
    
    async def get_service_details(self, service_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific service
        
        Args:
            service_name: Name of the service
            
        Returns:
            Detailed service information
        """
        try:
            # Get service status
            service_status = await self.metric_provider.get_service_status()
            
            # Check if service exists
            if service_name not in service_status["services"]:
                raise DashboardError(f"Service not found: {service_name}")
            
            service_data = service_status["services"][service_name]
            
            # Get service-specific metrics
            service_metrics_key = f"metrics:services:{service_name}"
            service_metrics_data = await run_in_threadpool(
                self.metric_provider.redis_client.client.get, service_metrics_key
            )
            
            if service_metrics_data:
                detailed_metrics = json.loads(service_metrics_data)
            else:
                detailed_metrics = {}
            
            # Get recent logs for this service
            service_logs_key = f"logs:services:{service_name}"
            service_logs_data = await run_in_threadpool(
                self.metric_provider.redis_client.client.lrange, 
                service_logs_key, 0, 19  # Get last 20 logs
            )
            
            recent_logs = []
            for log_data in service_logs_data:
                try:
                    log = json.loads(log_data)
                    recent_logs.append(log)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON data in service logs for {service_name}")
            
            # Format logs
            formatted_logs = []
            for log in recent_logs:
                formatted_logs.append({
                    "timestamp": log.get("timestamp", 0),
                    "formatted_time": format_timestamp(log.get("timestamp", 0)),
                    "level": log.get("level", "INFO"),
                    "message": log.get("message", ""),
                    "context": log.get("context", {})
                })
            
            # Sort logs by timestamp (newest first)
            formatted_logs.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            
            # Get dependency information
            dependencies_key = f"services:{service_name}:dependencies"
            dependencies_data = await run_in_threadpool(
                self.metric_provider.redis_client.client.get, dependencies_key
            )
            
            if dependencies_data:
                dependencies = json.loads(dependencies_data)
            else:
                dependencies = []
            
            # Compile detailed service information
            details = {
                "timestamp": time.time(),
                "formatted_time": format_timestamp(time.time()),
                "service": service_name,
                "status": service_data.get("status", "unknown"),
                "health": service_data.get("health", "unknown"),
                "uptime": service_data.get("uptime", 0),
                "formatted_uptime": get_human_readable_time(service_data.get("uptime", 0)),
                "last_updated": service_data.get("last_updated", 0),
                "formatted_last_updated": format_timestamp(service_data.get("last_updated", 0)),
                "detailed_metrics": detailed_metrics,
                "recent_logs": formatted_logs,
                "dependencies": dependencies,
                "issues": service_data.get("issues", [])
            }
            
            return details
        
        except Exception as e:
            logger.error(f"Error fetching service details for {service_name}: {str(e)}")
            raise DashboardError(f"Failed to fetch details for service {service_name}: {str(e)}")
    
    async def get_asset_details(self, asset: str, platform: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific trading asset
        
        Args:
            asset: Asset symbol (e.g., "BTC/USD")
            platform: Trading platform (e.g., "binance" or "deriv")
            
        Returns:
            Detailed asset information
        """
        try:
            # Get asset metrics
            asset_metrics_key = f"metrics:trading:assets:{platform}:{asset}"
            asset_metrics_data = await run_in_threadpool(
                self.metric_provider.redis_client.client.get, asset_metrics_key
            )
            
            if asset_metrics_data:
                asset_metrics = json.loads(asset_metrics_data)
            else:
                asset_metrics = {}
            
            # Get recent trades for this asset
            asset_trades_key = f"trading:assets:{platform}:{asset}:recent_trades"
            asset_trades_data = await run_in_threadpool(
                self.metric_provider.redis_client.client.lrange, 
                asset_trades_key, 0, 19  # Get last 20 trades
            )
            
            recent_trades = []
            for trade_data in asset_trades_data:
                try:
                    trade = json.loads(trade_data)
                    recent_trades.append(trade)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON data in asset trades for {asset}")
            
            # Format trades
            formatted_trades = []
            for trade in recent_trades:
                formatted_trades.append({
                    "id": trade.get("id", ""),
                    "timestamp": trade.get("timestamp", 0),
                    "formatted_time": format_timestamp(trade.get("timestamp", 0)),
                    "direction": trade.get("direction", ""),
                    "entry_price": trade.get("entry_price", 0),
                    "exit_price": trade.get("exit_price", 0),
                    "profit_loss": trade.get("profit_loss", 0),
                    "profit_percentage": trade.get("profit_percentage", 0),
                    "status": trade.get("status", ""),
                    "strategy": trade.get("strategy", ""),
                    "brain": trade.get("brain", "")
                })
            
            # Sort trades by timestamp (newest first)
            formatted_trades.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            
            # Get brain performance for this asset
            brain_metrics_key = f"metrics:trading:brains:assets:{platform}:{asset}"
            brain_metrics_data = await run_in_threadpool(
                self.metric_provider.redis_client.client.get, brain_metrics_key
            )
            
            if brain_metrics_data:
                brain_metrics = json.loads(brain_metrics_data)
            else:
                brain_metrics = {}
            
            # Get recent loopholes for this asset
            asset_loopholes_key = f"loopholes:assets:{platform}:{asset}"
            asset_loopholes_data = await run_in_threadpool(
                self.metric_provider.redis_client.client.lrange, 
                asset_loopholes_key, 0, 9  # Get last 10 loopholes
            )
            
            recent_loopholes = []
            for loophole_data in asset_loopholes_data:
                try:
                    loophole = json.loads(loophole_data)
                    recent_loopholes.append(loophole)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON data in asset loopholes for {asset}")
            
            # Format loopholes
            formatted_loopholes = []
            for loophole in recent_loopholes:
                formatted_loopholes.append({
                    "id": loophole.get("id", ""),
                    "timestamp": loophole.get("timestamp", 0),
                    "formatted_time": format_timestamp(loophole.get("timestamp", 0)),
                    "type": loophole.get("type", ""),
                    "description": loophole.get("description", ""),
                    "confidence": loophole.get("confidence", 0),
                    "exploitation_status": loophole.get("exploitation_status", ""),
                    "profit_potential": loophole.get("profit_potential", 0)
                })
            
            # Sort loopholes by timestamp (newest first)
            formatted_loopholes.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            
            # Compile detailed asset information
            details = {
                "timestamp": time.time(),
                "formatted_time": format_timestamp(time.time()),
                "asset": asset,
                "platform": platform,
                "metrics": {
                    "win_rate": asset_metrics.get("win_rate", 0),
                    "profit_factor": asset_metrics.get("profit_factor", 0),
                    "total_trades": asset_metrics.get("total_trades", 0),
                    "winning_trades": asset_metrics.get("winning_trades", 0),
                    "losing_trades": asset_metrics.get("losing_trades", 0),
                    "total_profit": asset_metrics.get("total_profit", 0),
                    "average_profit": asset_metrics.get("average_profit", 0),
                    "average_loss": asset_metrics.get("average_loss", 0),
                    "max_drawdown": asset_metrics.get("max_drawdown", 0)
                },
                "recent_trades": formatted_trades,
                "brain_metrics": brain_metrics,
                "recent_loopholes": formatted_loopholes,
                "market_regime": asset_metrics.get("market_regime", "unknown"),
                "trading_status": asset_metrics.get("trading_status", "unknown")
            }
            
            # Calculate overall health
            win_rate = details["metrics"]["win_rate"]
            if win_rate >= 0.8:
                details["health"] = "excellent"
            elif win_rate >= 0.65:
                details["health"] = "good"
            elif win_rate >= 0.5:
                details["health"] = "average"
            else:
                details["health"] = "poor"
            
            # Check if enough trades to evaluate
            if details["metrics"]["total_trades"] < 10:
                details["health"] = "insufficient_data"
            
            return details
        
        except Exception as e:
            logger.error(f"Error fetching asset details for {asset} on {platform}: {str(e)}")
            raise DashboardError(f"Failed to fetch details for asset {asset}: {str(e)}")
    
    async def get_brain_details(self, brain_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific trading brain
        
        Args:
            brain_name: Name of the trading brain
            
        Returns:
            Detailed brain information
        """
        try:
            # Get brain metrics
            brain_metrics_key = f"metrics:trading:brains:{brain_name}"
            brain_metrics_data = await run_in_threadpool(
                self.metric_provider.redis_client.client.get, brain_metrics_key
            )
            
            if brain_metrics_data:
                brain_metrics = json.loads(brain_metrics_data)
            else:
                brain_metrics = {}
            
            # Get recent decisions by this brain
            brain_decisions_key = f"brain:{brain_name}:recent_decisions"
            brain_decisions_data = await run_in_threadpool(
                self.metric_provider.redis_client.client.lrange, 
                brain_decisions_key, 0, 19  # Get last 20 decisions
            )
            
            recent_decisions = []
            for decision_data in brain_decisions_data:
                try:
                    decision = json.loads(decision_data)
                    recent_decisions.append(decision)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON data in brain decisions for {brain_name}")
            
            # Format decisions
            formatted_decisions = []
            for decision in recent_decisions:
                formatted_decisions.append({
                    "id": decision.get("id", ""),
                    "timestamp": decision.get("timestamp", 0),
                    "formatted_time": format_timestamp(decision.get("timestamp", 0)),
                    "asset": decision.get("asset", ""),
                    "platform": decision.get("platform", ""),
                    "direction": decision.get("direction", ""),
                    "confidence": decision.get("confidence", 0),
                    "result": decision.get("result", ""),
                    "profit_loss": decision.get("profit_loss", 0),
                    "council_accepted": decision.get("council_accepted", False)
                })
            
            # Sort decisions by timestamp (newest first)
            formatted_decisions.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            
            # Get performance by asset
            asset_performance_key = f"metrics:trading:brains:{brain_name}:assets"
            asset_performance_data = await run_in_threadpool(
                self.metric_provider.redis_client.client.get, asset_performance_key
            )
            
            if asset_performance_data:
                asset_performance = json.loads(asset_performance_data)
            else:
                asset_performance = {}
            
            # Get brain status
            brain_status_key = f"status:brains:{brain_name}"
            brain_status_data = await run_in_threadpool(
                self.metric_provider.redis_client.client.get, brain_status_key
            )
            
            if brain_status_data:
                brain_status = json.loads(brain_status_data)
            else:
                brain_status = {
                    "status": "unknown",
                    "type": "unknown",
                    "specialization": "unknown",
                    "active": False
                }
            
            # Get evolution metrics
            evolution_key = f"metrics:brains:{brain_name}:evolution"
            evolution_data = await run_in_threadpool(
                self.metric_provider.redis_client.client.get, evolution_key
            )
            
            if evolution_data:
                evolution_metrics = json.loads(evolution_data)
            else:
                evolution_metrics = {
                    "generations": 0,
                    "improvements": 0,
                    "last_evolution": 0
                }
            
            # Compile detailed brain information
            details = {
                "timestamp": time.time(),
                "formatted_time": format_timestamp(time.time()),
                "brain": brain_name,
                "type": brain_status.get("type", "unknown"),
                "specialization": brain_status.get("specialization", "unknown"),
                "status": brain_status.get("status", "unknown"),
                "active": brain_status.get("active", False),
                "metrics": {
                    "win_rate": brain_metrics.get("win_rate", 0),
                    "profit_factor": brain_metrics.get("profit_factor", 0),
                    "total_decisions": brain_metrics.get("total_decisions", 0),
                    "correct_decisions": brain_metrics.get("correct_decisions", 0),
                    "incorrect_decisions": brain_metrics.get("incorrect_decisions", 0),
                    "total_profit": brain_metrics.get("total_profit", 0),
                    "average_confidence": brain_metrics.get("average_confidence", 0),
                    "council_acceptance_rate": brain_metrics.get("council_acceptance_rate", 0)
                },
                "recent_decisions": formatted_decisions,
                "asset_performance": asset_performance,
                "evolution": evolution_metrics
            }
            
            # Calculate overall health
            win_rate = details["metrics"]["win_rate"]
            if win_rate >= 0.8:
                details["health"] = "excellent"
            elif win_rate >= 0.65:
                details["health"] = "good"
            elif win_rate >= 0.5:
                details["health"] = "average"
            else:
                details["health"] = "poor"
            
            # Check if enough decisions to evaluate
            if details["metrics"]["total_decisions"] < 10:
                details["health"] = "insufficient_data"
            
            return details
        
        except Exception as e:
            logger.error(f"Error fetching brain details for {brain_name}: {str(e)}")
            raise DashboardError(f"Failed to fetch details for brain {brain_name}: {str(e)}")
    
    async def get_trading_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive trading performance report
        
        Returns:
            Trading performance report
        """
        try:
            # Get trading performance
            trading_perf = await self.metric_provider.get_trading_performance()
            
            # Get strategy performance
            strategy_perf = await self.metric_provider.get_strategy_performance()
            
            # Get trading history (last 30 days)
            trading_history = await self.metric_provider.get_trading_history(days=30)
            
            # Generate report
            report = {
                "timestamp": time.time(),
                "formatted_time": format_timestamp(time.time()),
                "period": "Last 30 days",
                "summary": {
                    "overall_win_rate": trading_perf["overall"]["win_rate"],
                    "overall_profit": trading_perf["overall"]["total_profit"],
                    "total_trades": trading_perf["overall"]["total_trades"],
                    "profit_factor": trading_perf["overall"]["profit_factor"],
                    "max_drawdown": trading_perf["overall"]["max_drawdown"],
                    "sharpe_ratio": trading_perf["overall"]["sharpe_ratio"],
                    "health": trading_perf["health"]
                },
                "platforms": trading_perf["platforms"],
                "top_assets": {},
                "top_strategies": {},
                "top_brains": {},
                "performance_trend": {
                    "win_rate_trend": [],
                    "profit_trend": [],
                    "equity_curve": trading_history["history"]["equity_curve"]
                }
            }
            
            # Get top performing assets
            assets = []
            for asset, metrics in trading_perf["assets"].items():
                if metrics.get("total_trades", 0) >= 5:  # Minimum trades for consideration
                    assets.append({
                        "asset": asset,
                        "win_rate": metrics.get("win_rate", 0),
                        "profit": metrics.get("total_profit", 0),
                        "trades": metrics.get("total_trades", 0)
                    })
            
            # Sort by win rate and take top 5
            top_assets_by_win_rate = sorted(
                assets, key=lambda x: x["win_rate"], reverse=True
            )[:5]
            
            # Sort by profit and take top 5
            top_assets_by_profit = sorted(
                assets, key=lambda x: x["profit"], reverse=True
            )[:5]
            
            report["top_assets"] = {
                "by_win_rate": top_assets_by_win_rate,
                "by_profit": top_assets_by_profit
            }
            
            # Get top performing strategies
            strategies = []
            for strategy, metrics in strategy_perf["strategies"].items():
                if metrics.get("total_trades", 0) >= 5:  # Minimum trades for consideration
                    strategies.append({
                        "strategy": strategy,
                        "win_rate": metrics.get("win_rate", 0),
                        "profit": metrics.get("total_profit", 0),
                        "trades": metrics.get("total_trades", 0)
                    })
            
            # Sort by win rate and take top 5
            top_strategies_by_win_rate = sorted(
                strategies, key=lambda x: x["win_rate"], reverse=True
            )[:5]
            
            # Sort by profit and take top 5
            top_strategies_by_profit = sorted(
                strategies, key=lambda x: x["profit"], reverse=True
            )[:5]
            
            report["top_strategies"] = {
                "by_win_rate": top_strategies_by_win_rate,
                "by_profit": top_strategies_by_profit
            }
            
            # Get top performing brains
            brains = []
            for brain, metrics in strategy_perf["brains"].items():
                if metrics.get("total_trades", 0) >= 5:  # Minimum trades for consideration
                    brains.append({
                        "brain": brain,
                        "win_rate": metrics.get("win_rate", 0),
                        "profit": metrics.get("total_profit", 0),
                        "trades": metrics.get("total_trades", 0),
                        "confidence": metrics.get("confidence", 0)
                    })
            

            # Sort by win rate and take top 5
            top_brains_by_win_rate = sorted(
                brains, key=lambda x: x["win_rate"], reverse=True
            )[:5]

            # Sort by profit and take top 5
            top_brains_by_profit = sorted(
                brains, key=lambda x: x["profit"], reverse=True
            )[:5]

            report["top_brains"] = {
                "by_win_rate": top_brains_by_win_rate,
                "by_profit": top_brains_by_profit,
            }

            logger.info("Successfully assembled trading performance report")
            return report

        except Exception as e:
            logger.error(f"Error getting trading performance report: {str(e)}")
            raise DashboardError(
                f"Failed to get trading performance report: {str(e)}"
            )
    
    
