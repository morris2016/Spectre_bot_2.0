
#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
System Health Module

This module provides comprehensive system health monitoring, including
resource usage tracking, component status checks, and automated recovery
procedures for maintaining optimal system performance.
"""

import os
import sys
import time
import psutil
import platform
import asyncio
import logging
import socket
import json
import traceback
import datetime
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import subprocess
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from common.logger import get_logger
from common.redis_client import RedisClient
from common.db_client import DatabaseClient, get_db_client
from common.constants import (
    SERVICE_NAMES, SERVICE_DEPENDENCIES, SERVICE_STARTUP_ORDER,
    RESOURCE_THRESHOLDS, HEALTH_CHECK_INTERVALS
)
from common.exceptions import (
    ServiceUnavailableError, ResourceExhaustionError, 
    ComponentFailureError, NetworkError
)

logger = get_logger("system_health")

class SystemHealth:
    """
    Comprehensive system health monitoring and management system for
    ensuring optimal operation of the QuantumSpectre Elite Trading System.
    """
    
    def __init__(self, 
                 config: Dict[str, Any], 
                 redis_client: Optional[RedisClient] = None,
                 db_client: Optional[DatabaseClient] = None):
        """
        Initialize the SystemHealth monitor with the specified configuration.
        
        Args:
            config: Configuration dictionary
            redis_client: Optional Redis client for real-time data
            db_client: Optional database client for persistent storage
        """
        self.config = config
        self.redis_client = redis_client or RedisClient(config.get('redis', {}))
        self.db_client = db_client
        self._db_params = config.get('database', {})
        
        # System information
        self.system_info = self.get_system_info()
        
        # Component health status
        self.component_status = {
            name: {"status": "unknown", "last_check": 0, "failures": 0, "message": ""}
            for name in SERVICE_NAMES
        }
        
        # Resource usage history
        self.resource_history = {
            'cpu': [],
            'memory': [],
            'disk': [],
            'network': [],
            'gpu': [],
            'timestamps': []
        }
        
        # History length
        self.history_length = config.get('health_history_length', 1440)  # Default: 24 hours at 1-minute intervals
        
        # Current resource usage
        self.current_resources = {}
        
        # Alert status
        self.alerts = {
            'cpu': False,
            'memory': False,
            'disk': False,
            'network': False,
            'gpu': False,
            'component_failures': set()
        }
        
        # Thresholds
        self.thresholds = config.get('resource_thresholds', RESOURCE_THRESHOLDS)
        
        # Check intervals
        self.check_intervals = config.get('health_check_intervals', HEALTH_CHECK_INTERVALS)
        
        # Recovery actions log
        self.recovery_actions = []
        
        # Recovery in progress flags
        self.recovery_in_progress = {}
        
        # Performance metrics
        self.performance_metrics = {
            'latency': {},
            'throughput': {},
            'error_rate': {},
            'response_time': {}
        }
        
        # Health check tasks
        self.health_check_tasks = {}
        
        # Initialize monitoring
        self.initialize_monitoring()

        logger.info("System health monitor initialized successfully")

    async def initialize(self, db_connector: Optional[DatabaseClient] = None) -> None:
        """Obtain a database client and ensure tables exist."""
        if db_connector is not None:
            self.db_client = db_connector
        if self.db_client is None:
            self.db_client = await get_db_client(**self._db_params)
        if getattr(self.db_client, "pool", None) is None:
            await self.db_client.initialize()
            await self.db_client.create_tables()
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Collect and return system information.
        
        Returns:
            Dictionary containing system information
        """
        try:
            uname = platform.uname()
            
            # Get total memory
            mem = psutil.virtual_memory()
            
            # Get disk information
            disk = psutil.disk_usage('/')
            
            # Get GPU information if available
            gpu_info = self.get_gpu_info()
            
            # Get network interfaces
            net_if = psutil.net_if_addrs()
            network_interfaces = {}
            
            for interface, addrs in net_if.items():
                addresses = []
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        addresses.append(addr.address)
                if addresses:
                    network_interfaces[interface] = addresses
            
            # Build system info dictionary
            system_info = {
                'system': uname.system,
                'node': uname.node,
                'release': uname.release,
                'version': uname.version,
                'machine': uname.machine,
                'processor': uname.processor,
                'python_version': sys.version,
                'memory_total': mem.total,
                'memory_available': mem.available,
                'disk_total': disk.total,
                'disk_free': disk.free,
                'cpu_cores': psutil.cpu_count(logical=False),
                'cpu_threads': psutil.cpu_count(logical=True),
                'network_interfaces': network_interfaces,
                'gpu': gpu_info
            }
            
            return system_info
        except Exception as e:
            logger.error(f"Error collecting system information: {str(e)}")
            return {'error': str(e)}
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU information if available.
        
        Returns:
            Dictionary containing GPU information or empty dict if no GPU
        """
        gpu_info = {}
        
        try:
            # Try to import NVIDIA management library
            import pynvml
            
            try:
                pynvml.nvmlInit()
                gpu_count = pynvml.nvmlDeviceGetCount()
                
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    gpu_info[i] = {
                        'name': name,
                        'total_memory': memory_info.total,
                        'free_memory': memory_info.free,
                        'used_memory': memory_info.used
                    }
                
                pynvml.nvmlShutdown()
            except Exception as e:
                logger.warning(f"Could not get NVIDIA GPU information: {str(e)}")
                
                # Try alternative method for NVIDIA GPUs
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        for i, line in enumerate(lines):
                            parts = line.split(',')
                            if len(parts) >= 4:
                                gpu_info[i] = {
                                    'name': parts[0].strip(),
                                    'total_memory': int(parts[1].strip()) * 1024 * 1024,  # Convert to bytes
                                    'free_memory': int(parts[2].strip()) * 1024 * 1024,
                                    'used_memory': int(parts[3].strip()) * 1024 * 1024
                                }
                except Exception as e2:
                    logger.warning(f"Could not get GPU information using nvidia-smi: {str(e2)}")
        except ImportError:
            logger.debug("NVIDIA Management Library (pynvml) not available")
        
        return gpu_info
    
    def initialize_monitoring(self):
        """Initialize all health monitoring tasks"""
        # Start resource monitoring
        asyncio.create_task(self.monitor_resources())
        
        # Start component monitoring
        for component in SERVICE_NAMES:
            self.health_check_tasks[component] = asyncio.create_task(
                self.monitor_component(component)
            )
        
        # Start performance monitoring
        asyncio.create_task(self.monitor_performance())
        
        # Start periodic system status update
        asyncio.create_task(self.update_system_status())
        
        logger.info("Health monitoring tasks initialized")
    
    async def monitor_resources(self):
        """Monitor system resource usage at regular intervals"""
        interval = self.check_intervals.get('resources', 60)  # Default: check every 60 seconds
        
        while True:
            try:
                # Collect current resource usage
                resources = self.collect_resource_usage()
                
                # Update current resources
                self.current_resources = resources
                
                # Add to history
                self.update_resource_history(resources)
                
                # Check for threshold violations
                await self.check_resource_thresholds(resources)
                
                # Publish resource metrics to Redis
                self.publish_resource_metrics(resources)
                
                # Wait for next check
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {str(e)}")
                await asyncio.sleep(interval)
    
    def collect_resource_usage(self) -> Dict[str, Any]:
        """
        Collect current resource usage metrics.
        
        Returns:
            Dictionary containing resource usage metrics
        """
        try:
            timestamp = int(time.time())
            
            # CPU usage (percent)
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent
            }
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            }
            
            # Network I/O
            network_before = psutil.net_io_counters()
            time.sleep(1)  # Measure for 1 second
            network_after = psutil.net_io_counters()
            
            network_usage = {
                'bytes_sent': network_after.bytes_sent,
                'bytes_recv': network_after.bytes_recv,
                'packets_sent': network_after.packets_sent,
                'packets_recv': network_after.packets_recv,
                'bytes_sent_per_sec': network_after.bytes_sent - network_before.bytes_sent,
                'bytes_recv_per_sec': network_after.bytes_recv - network_before.bytes_recv
            }
            
            # GPU usage if available
            gpu_usage = self.collect_gpu_usage()
            
            return {
                'timestamp': timestamp,
                'cpu': cpu_usage,
                'memory': memory_usage,
                'disk': disk_usage,
                'network': network_usage,
                'gpu': gpu_usage
            }
        except Exception as e:
            logger.error(f"Error collecting resource usage: {str(e)}")
            return {
                'timestamp': int(time.time()),
                'error': str(e)
            }
    
    def collect_gpu_usage(self) -> Dict[str, Any]:
        """
        Collect GPU usage metrics if available.
        
        Returns:
            Dictionary containing GPU usage metrics or empty dict if no GPU
        """
        gpu_usage = {}
        
        try:
            # Try to import NVIDIA management library
            import pynvml
            
            try:
                pynvml.nvmlInit()
                gpu_count = pynvml.nvmlDeviceGetCount()
                
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    gpu_usage[i] = {
                        'gpu_util': utilization.gpu,
                        'memory_util': utilization.memory,
                        'memory_total': memory_info.total,
                        'memory_free': memory_info.free,
                        'memory_used': memory_info.used,
                        'memory_percent': (memory_info.used / memory_info.total) * 100
                    }
                
                pynvml.nvmlShutdown()
            except Exception as e:
                logger.warning(f"Could not get NVIDIA GPU usage: {str(e)}")
                
                # Try alternative method
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total', '--format=csv,noheader,nounits'],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        for i, line in enumerate(lines):
                            parts = line.split(',')
                            if len(parts) >= 4:
                                gpu_util = int(parts[0].strip())
                                memory_util = int(parts[1].strip())
                                memory_used = int(parts[2].strip()) * 1024 * 1024  # Convert to bytes
                                memory_total = int(parts[3].strip()) * 1024 * 1024
                                
                                gpu_usage[i] = {
                                    'gpu_util': gpu_util,
                                    'memory_util': memory_util,
                                    'memory_total': memory_total,
                                    'memory_used': memory_used,
                                    'memory_free': memory_total - memory_used,
                                    'memory_percent': (memory_used / memory_total) * 100 if memory_total > 0 else 0
                                }
                except Exception as e2:
                    logger.warning(f"Could not get GPU usage using nvidia-smi: {str(e2)}")
        except ImportError:
            logger.debug("NVIDIA Management Library (pynvml) not available")
        
        return gpu_usage
    
    def update_resource_history(self, resources: Dict[str, Any]):
        """
        Update resource usage history with the latest measurements.
        
        Args:
            resources: Dictionary containing resource usage metrics
        """
        timestamp = resources.get('timestamp')
        
        if timestamp:
            # Add timestamp to history
            self.resource_history['timestamps'].append(timestamp)
            
            # Add CPU usage to history
            self.resource_history['cpu'].append(resources.get('cpu', 0))
            
            # Add memory usage to history
            memory = resources.get('memory', {})
            self.resource_history['memory'].append(memory.get('percent', 0))
            
            # Add disk usage to history
            disk = resources.get('disk', {})
            self.resource_history['disk'].append(disk.get('percent', 0))
            
            # Add network usage to history
            network = resources.get('network', {})
            network_usage = network.get('bytes_sent_per_sec', 0) + network.get('bytes_recv_per_sec', 0)
            self.resource_history['network'].append(network_usage)
            
            # Add GPU usage to history if available
            gpu = resources.get('gpu', {})
            gpu_usage = 0
            if gpu:
                gpu_utils = [g.get('gpu_util', 0) for g in gpu.values()]
                gpu_usage = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
            self.resource_history['gpu'].append(gpu_usage)
            
            # Trim history if it exceeds the configured length
            if len(self.resource_history['timestamps']) > self.history_length:
                for key in self.resource_history:
                    self.resource_history[key] = self.resource_history[key][-self.history_length:]
    
    async def check_resource_thresholds(self, resources: Dict[str, Any]):
        """
        Check if any resource usage exceeds configured thresholds.
        
        Args:
            resources: Dictionary containing resource usage metrics
        """
        # Check CPU
        cpu_usage = resources.get('cpu', 0)
        cpu_threshold = self.thresholds.get('cpu', 90)
        if cpu_usage > cpu_threshold:
            if not self.alerts['cpu']:
                self.alerts['cpu'] = True
                await self.log_alert('cpu', f"CPU usage exceeds threshold: {cpu_usage:.1f}% > {cpu_threshold}%")
        else:
            self.alerts['cpu'] = False
        
        # Check memory
        memory = resources.get('memory', {})
        memory_percent = memory.get('percent', 0)
        memory_threshold = self.thresholds.get('memory', 90)
        if memory_percent > memory_threshold:
            if not self.alerts['memory']:
                self.alerts['memory'] = True
                await self.log_alert('memory', f"Memory usage exceeds threshold: {memory_percent:.1f}% > {memory_threshold}%")
        else:
            self.alerts['memory'] = False
        
        # Check disk
        disk = resources.get('disk', {})
        disk_percent = disk.get('percent', 0)
        disk_threshold = self.thresholds.get('disk', 90)
        if disk_percent > disk_threshold:
            if not self.alerts['disk']:
                self.alerts['disk'] = True
                await self.log_alert('disk', f"Disk usage exceeds threshold: {disk_percent:.1f}% > {disk_threshold}%")
        else:
            self.alerts['disk'] = False
        
        # Check GPU if available
        gpu = resources.get('gpu', {})
        if gpu:
            for gpu_id, gpu_stats in gpu.items():
                gpu_memory_percent = gpu_stats.get('memory_percent', 0)
                gpu_threshold = self.thresholds.get('gpu', 90)
                if gpu_memory_percent > gpu_threshold:
                    if not self.alerts['gpu']:
                        self.alerts['gpu'] = True
                        await self.log_alert('gpu', f"GPU {gpu_id} memory usage exceeds threshold: {gpu_memory_percent:.1f}% > {gpu_threshold}%")
                    break
            else:
                self.alerts['gpu'] = False
    
    async def log_alert(self, resource_type: str, message: str):
        """
        Log a resource usage alert.
        
        Args:
            resource_type: Type of resource (cpu, memory, etc.)
            message: Alert message
        """
        logger.warning(f"RESOURCE ALERT - {resource_type.upper()}: {message}")
        
        # Store alert in database
        try:
            await self.db_client.execute(
                """
                INSERT INTO system_alerts (
                    timestamp, type, resource, message, acknowledged
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (int(time.time()), 'resource', resource_type, message, 0)
            )
        except Exception as e:
            logger.error(f"Failed to store alert in database: {str(e)}")
        
        # Publish alert to Redis
        try:
            alert_data = {
                'timestamp': int(time.time()),
                'type': 'resource',
                'resource': resource_type,
                'message': message,
                'acknowledged': False
            }
            self.redis_client.publish('system:alerts', json.dumps(alert_data))
        except Exception as e:
            logger.error(f"Failed to publish alert to Redis: {str(e)}")
    
    async def monitor_component(self, component_name: str):
        """
        Monitor a specific system component at regular intervals.
        
        Args:
            component_name: Name of the component to monitor
        """
        # Get check interval for this component
        interval = self.check_intervals.get('components', 30)  # Default: check every 30 seconds
        
        # Get component-specific settings
        component_config = self.config.get('components', {}).get(component_name, {})
        component_interval = component_config.get('check_interval', interval)
        
        while True:
            try:
                # Check component health
                status, message = await self.check_component_health(component_name)
                
                # Update component status
                self.update_component_status(component_name, status, message)
                
                # Take recovery action if needed
                if status != "healthy" and not self.recovery_in_progress.get(component_name, False):
                    asyncio.create_task(self.handle_component_failure(component_name, status, message))
                
                # Wait for next check
                await asyncio.sleep(component_interval)
            except Exception as e:
                logger.error(f"Error monitoring component {component_name}: {str(e)}")
                self.update_component_status(component_name, "error", f"Monitoring error: {str(e)}")
                await asyncio.sleep(component_interval)
    
    async def check_component_health(self, component_name: str) -> Tuple[str, str]:
        """
        Check the health of a specific component.
        
        Args:
            component_name: Name of the component to check
            
        Returns:
            Tuple containing status (healthy/degraded/critical/error) and message
        """
        # Check if component is registered in Redis
        try:
            component_key = f"service:{component_name}"
            component_data = self.redis_client.get(component_key)
            
            if not component_data:
                return "error", f"Component not registered: {component_name}"
            
            component_info = json.loads(component_data)
            heartbeat_time = component_info.get('heartbeat', 0)
            status = component_info.get('status', 'unknown')
            
            # Check if heartbeat is recent
            current_time = int(time.time())
            heartbeat_age = current_time - heartbeat_time
            
            # Get heartbeat threshold for this component
            heartbeat_threshold = self.config.get('components', {}).get(component_name, {}).get('heartbeat_threshold', 60)
            
            if heartbeat_age > heartbeat_threshold:
                return "critical", f"Component heartbeat too old: {heartbeat_age} seconds"
            
            # Check component's self-reported status
            if status == 'healthy':
                return "healthy", "Component reports healthy status"
            elif status == 'degraded':
                return "degraded", f"Component self-reports as degraded: {component_info.get('message', 'No details')}"
            elif status == 'error':
                return "critical", f"Component self-reports error: {component_info.get('message', 'No details')}"
            else:
                # Additional health checks if needed
                # These would be component-specific checks
                return "healthy", "Component appears to be functioning"
        except Exception as e:
            logger.error(f"Error checking health of component {component_name}: {str(e)}")
            return "error", f"Health check error: {str(e)}"
    
    def update_component_status(self, component_name: str, status: str, message: str):
        """
        Update the status of a component.
        
        Args:
            component_name: Name of the component
            status: Status of the component (healthy/degraded/critical/error)
            message: Status message or details
        """
        current_time = int(time.time())
        
        # Get previous status
        previous_status = self.component_status.get(component_name, {}).get('status', 'unknown')
        
        # Update component status
        if component_name not in self.component_status:
            self.component_status[component_name] = {}
        
        self.component_status[component_name].update({
            'status': status,
            'last_check': current_time,
            'message': message
        })
        
        # Increment failure count if status is not healthy
        if status != 'healthy':
            failures = self.component_status[component_name].get('failures', 0)
            self.component_status[component_name]['failures'] = failures + 1
        else:
            # Reset failures if component is healthy now
            self.component_status[component_name]['failures'] = 0
        
        # Log status change
        if status != previous_status:
            if status == 'healthy':
                logger.info(f"Component {component_name} is now healthy: {message}")
            elif status == 'degraded':
                logger.warning(f"Component {component_name} is degraded: {message}")
            elif status == 'critical':
                logger.error(f"Component {component_name} is in critical state: {message}")
            elif status == 'error':
                logger.error(f"Component {component_name} has an error: {message}")
            
            # Update component failures set for alerts
            if status != 'healthy':
                self.alerts['component_failures'].add(component_name)
            else:
                if component_name in self.alerts['component_failures']:
                    self.alerts['component_failures'].remove(component_name)
        
        # Publish component status to Redis
        try:
            component_status_data = {
                'name': component_name,
                'status': status,
                'message': message,
                'last_check': current_time,
                'failures': self.component_status[component_name].get('failures', 0)
            }
            self.redis_client.hset(
                'system:component_status',
                component_name,
                json.dumps(component_status_data)
            )
        except Exception as e:
            logger.error(f"Failed to publish component status to Redis: {str(e)}")
    
    async def handle_component_failure(self, component_name: str, status: str, message: str):
        """
        Handle a component failure by taking appropriate recovery actions.
        
        Args:
            component_name: Name of the failed component
            status: Current status of the component
            message: Failure message or details
        """
        try:
            # Mark recovery as in progress for this component
            self.recovery_in_progress[component_name] = True
            
            # Log the failure
            logger.warning(f"Handling component failure: {component_name} ({status}) - {message}")
            
            # Get recovery configuration for this component
            recovery_config = self.config.get('components', {}).get(component_name, {}).get('recovery', {})
            
            # Get current failure count
            failures = self.component_status[component_name].get('failures', 0)
            
            # Get recovery strategy based on failure count
            recovery_strategy = None
            if failures <= 3:
                recovery_strategy = recovery_config.get('low', 'restart')
            elif failures <= 6:
                recovery_strategy = recovery_config.get('medium', 'restart_with_deps')
            else:
                recovery_strategy = recovery_config.get('high', 'alert_only')
            
            # Log recovery action
            action_time = int(time.time())
            action_record = {
                'timestamp': action_time,
                'component': component_name,
                'status': status,
                'failures': failures,
                'message': message,
                'action': recovery_strategy
            }
            
            # Add to recovery actions log
            self.recovery_actions.append(action_record)
            
            # Truncate log if necessary
            max_actions = self.config.get('max_recovery_actions', 100)
            if len(self.recovery_actions) > max_actions:
                self.recovery_actions = self.recovery_actions[-max_actions:]
            
            # Store in database
            try:
                await self.db_client.execute(
                    """
                    INSERT INTO recovery_actions (
                        timestamp, component, status, failures, message, action
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (action_time, component_name, status, failures, message, recovery_strategy)
                )
            except Exception as e:
                logger.error(f"Failed to store recovery action in database: {str(e)}")
            
            # Take recovery action
            if recovery_strategy == 'restart':
                await self.restart_component(component_name)
            elif recovery_strategy == 'restart_with_deps':
                await self.restart_component_with_dependencies(component_name)
            elif recovery_strategy == 'alert_only':
                await self.send_critical_alert(component_name, status, message, failures)
            elif recovery_strategy == 'custom':
                await self.execute_custom_recovery(component_name, recovery_config.get('custom_action'))
            else:
                logger.warning(f"Unknown recovery strategy '{recovery_strategy}' for component {component_name}")
        except Exception as e:
            logger.error(f"Error handling component failure for {component_name}: {str(e)}")
        finally:
            # Mark recovery as no longer in progress
            self.recovery_in_progress[component_name] = False
    
    async def restart_component(self, component_name: str):
        """
        Restart a specific component.
        
        Args:
            component_name: Name of the component to restart
        """
        logger.info(f"Attempting to restart component: {component_name}")
        
        try:
            # Send restart command via Redis
            restart_command = {
                'action': 'restart',
                'component': component_name,
                'timestamp': int(time.time())
            }
            self.redis_client.publish('system:commands', json.dumps(restart_command))
            
            # Wait for component to restart
            restart_timeout = self.config.get('components', {}).get(component_name, {}).get('restart_timeout', 30)
            await asyncio.sleep(restart_timeout)
            
            # Check if restart was successful
            status, message = await self.check_component_health(component_name)
            
            if status == 'healthy':
                logger.info(f"Successfully restarted component: {component_name}")
                return True
            else:
                logger.warning(f"Component restart may have failed for {component_name}: {status} - {message}")
                return False
        except Exception as e:
            logger.error(f"Error restarting component {component_name}: {str(e)}")
            return False
    
    async def restart_component_with_dependencies(self, component_name: str):
        """
        Restart a component and its dependencies in the correct order.
        
        Args:
            component_name: Name of the component to restart
        """
        logger.info(f"Attempting to restart component with dependencies: {component_name}")
        
        try:
            # Get dependencies for this component
            dependencies = SERVICE_DEPENDENCIES.get(component_name, [])
            
            # Stop the component and its dependencies in reverse order
            components_to_restart = dependencies + [component_name]
            
            for component in reversed(components_to_restart):
                stop_command = {
                    'action': 'stop',
                    'component': component,
                    'timestamp': int(time.time())
                }
                self.redis_client.publish('system:commands', json.dumps(stop_command))
                
                # Wait for component to stop
                await asyncio.sleep(5)
            
            # Start components in dependency order
            for component in components_to_restart:
                start_command = {
                    'action': 'start',
                    'component': component,
                    'timestamp': int(time.time())
                }
                self.redis_client.publish('system:commands', json.dumps(start_command))
                
                # Wait for component to start
                restart_timeout = self.config.get('components', {}).get(component, {}).get('restart_timeout', 30)
                await asyncio.sleep(restart_timeout)
            
            # Check if restart was successful
            status, message = await self.check_component_health(component_name)
            
            if status == 'healthy':
                logger.info(f"Successfully restarted component with dependencies: {component_name}")
                return True
            else:
                logger.warning(f"Component restart with dependencies may have failed for {component_name}: {status} - {message}")
                return False
        except Exception as e:
            logger.error(f"Error restarting component with dependencies {component_name}: {str(e)}")
            return False
    
    async def send_critical_alert(self, component_name: str, status: str, message: str, failures: int):
        """
        Send a critical alert for a component failure.
        
        Args:
            component_name: Name of the failed component
            status: Status of the component
            message: Failure message or details
            failures: Number of consecutive failures
        """
        alert_message = f"CRITICAL ALERT: Component {component_name} has failed {failures} times. Status: {status}. Details: {message}"
        
        logger.critical(alert_message)
        
        # Store alert in database
        try:
            await self.db_client.execute(
                """
                INSERT INTO system_alerts (
                    timestamp, type, resource, message, acknowledged, severity
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (int(time.time()), 'component', component_name, alert_message, 0, 'critical')
            )
        except Exception as e:
            logger.error(f"Failed to store critical alert in database: {str(e)}")
        
        # Publish alert to Redis
        try:
            alert_data = {
                'timestamp': int(time.time()),
                'type': 'component',
                'resource': component_name,
                'message': alert_message,
                'acknowledged': False,
                'severity': 'critical',
                'failures': failures
            }
            self.redis_client.publish('system:alerts', json.dumps(alert_data))
        except Exception as e:
            logger.error(f"Failed to publish critical alert to Redis: {str(e)}")
        
        # Send external notification if configured
        notification_config = self.config.get('notifications', {})
        if notification_config.get('enabled', False):
            asyncio.create_task(self.send_external_notification(alert_message, 'critical'))
    
    async def execute_custom_recovery(self, component_name: str, custom_action: str):
        """
        Execute a custom recovery action script for a component.
        
        Args:
            component_name: Name of the failed component
            custom_action: Custom recovery action to execute
        """
        logger.info(f"Executing custom recovery action for component {component_name}: {custom_action}")
        
        try:
            # Get the recovery script path
            recovery_scripts_dir = self.config.get('recovery_scripts_dir', './recovery_scripts')
            script_path = os.path.join(recovery_scripts_dir, custom_action)
            
            # Check if script exists
            if not os.path.exists(script_path):
                logger.error(f"Custom recovery script not found: {script_path}")
                return False
            
            # Execute the script
            process = await asyncio.create_subprocess_exec(
                script_path, component_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info(f"Custom recovery action successful for {component_name}: {stdout.decode().strip()}")
                return True
            else:
                logger.error(f"Custom recovery action failed for {component_name}: {stderr.decode().strip()}")
                return False
        except Exception as e:
            logger.error(f"Error executing custom recovery action for {component_name}: {str(e)}")
            return False
    
    async def send_external_notification(self, message: str, severity: str):
        """
        Send an external notification for a critical alert.
        
        Args:
            message: Alert message
            severity: Alert severity (info/warning/error/critical)
        """
        notification_config = self.config.get('notifications', {})
        
        # Skip if notifications are disabled or if severity doesn't meet threshold
        severity_levels = {'info': 0, 'warning': 1, 'error': 2, 'critical': 3}
        threshold_level = severity_levels.get(notification_config.get('min_severity', 'error'), 2)
        current_level = severity_levels.get(severity, 0)
        
        if not notification_config.get('enabled', False) or current_level < threshold_level:
            return
        
        logger.info(f"Sending external notification ({severity}): {message}")
        
        try:
            # Send email notification if configured
            if notification_config.get('email', {}).get('enabled', False):
                await self.send_email_notification(message, severity)
            
            # Send webhook notification if configured
            if notification_config.get('webhook', {}).get('enabled', False):
                await self.send_webhook_notification(message, severity)
        except Exception as e:
            logger.error(f"Error sending external notification: {str(e)}")
    
    async def send_email_notification(self, message: str, severity: str):
        """
        Send an email notification.
        
        Args:
            message: Notification message
            severity: Alert severity
        """
        from email.message import EmailMessage
        import smtplib
        
        email_config = self.config.get('notifications', {}).get('email', {})
        
        try:
            # Prepare email message
            msg = EmailMessage()
            msg['Subject'] = f"QuantumSpectre Alert ({severity.upper()})"
            msg['From'] = email_config.get('sender', 'alerts@quantumspectre.com')
            msg['To'] = email_config.get('recipients', [])
            msg.set_content(message)
            
            # Send the email
            smtp_server = email_config.get('smtp_server', 'localhost')
            smtp_port = email_config.get('smtp_port', 25)
            smtp_user = email_config.get('smtp_user')
            smtp_password = email_config.get('smtp_password')
            
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if email_config.get('use_tls', False):
                    server.starttls()
                
                if smtp_user and smtp_password:
                    server.login(smtp_user, smtp_password)
                
                server.send_message(msg)
            
            logger.info(f"Sent email notification ({severity})")
        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")
    
    async def send_webhook_notification(self, message: str, severity: str):
        """
        Send a webhook notification.
        
        Args:
            message: Notification message
            severity: Alert severity
        """
        import aiohttp
        
        webhook_config = self.config.get('notifications', {}).get('webhook', {})
        webhook_url = webhook_config.get('url')
        
        if not webhook_url:
            logger.error("Webhook URL not configured")
            return
        
        try:
            # Prepare webhook payload
            payload = {
                'timestamp': int(time.time()),
                'message': message,
                'severity': severity,
                'system': platform.node(),
                'source': 'QuantumSpectre Elite Trading System'
            }
            
            # Add custom fields if configured
            custom_fields = webhook_config.get('custom_fields', {})
            payload.update(custom_fields)
            
            # Send the webhook request
            headers = webhook_config.get('headers', {})
            headers.setdefault('Content-Type', 'application/json')
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, headers=headers) as response:
                    if response.status >= 200 and response.status < 300:
                        logger.info(f"Sent webhook notification ({severity})")
                    else:
                        logger.error(f"Webhook notification failed with status {response.status}: {await response.text()}")
        except Exception as e:
            logger.error(f"Error sending webhook notification: {str(e)}")
    
    async def monitor_performance(self):
        """Monitor system performance metrics at regular intervals"""
        interval = self.check_intervals.get('performance', 60)  # Default: check every 60 seconds
        
        while True:
            try:
                # Collect performance metrics
                latency_metrics = await self.collect_latency_metrics()
                throughput_metrics = await self.collect_throughput_metrics()
                error_rate_metrics = await self.collect_error_rate_metrics()
                response_time_metrics = await self.collect_response_time_metrics()
                
                # Update performance metrics
                self.performance_metrics['latency'] = latency_metrics
                self.performance_metrics['throughput'] = throughput_metrics
                self.performance_metrics['error_rate'] = error_rate_metrics
                self.performance_metrics['response_time'] = response_time_metrics
                
                # Publish performance metrics to Redis
                self.publish_performance_metrics()
                
                # Wait for next check
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
                await asyncio.sleep(interval)
    
    async def collect_latency_metrics(self) -> Dict[str, float]:
        """
        Collect system latency metrics.
        
        Returns:
            Dictionary containing latency metrics
        """
        latency_metrics = {}
        
        try:
            # Measure database latency
            db_latency = await self.measure_database_latency()
            latency_metrics['database'] = db_latency
            
            # Measure Redis latency
            redis_latency = await self.measure_redis_latency()
            latency_metrics['redis'] = redis_latency
            
            # Measure network latency to exchanges
            binance_latency = await self.measure_network_latency('api.binance.com')
            deriv_latency = await self.measure_network_latency('api.deriv.com')
            
            latency_metrics['binance'] = binance_latency
            latency_metrics['deriv'] = deriv_latency
        except Exception as e:
            logger.error(f"Error collecting latency metrics: {str(e)}")
        
        return latency_metrics
    
    async def measure_database_latency(self) -> float:
        """
        Measure database query latency.
        
        Returns:
            Latency in milliseconds
        """
        try:
            start_time = time.time()
            await self.db_client.execute("SELECT 1")
            end_time = time.time()
            
            return (end_time - start_time) * 1000  # Convert to milliseconds
        except Exception as e:
            logger.error(f"Error measuring database latency: {str(e)}")
            return -1
    
    async def measure_redis_latency(self) -> float:
        """
        Measure Redis operation latency.
        
        Returns:
            Latency in milliseconds
        """
        try:
            start_time = time.time()
            self.redis_client.ping()
            end_time = time.time()
            
            return (end_time - start_time) * 1000  # Convert to milliseconds
        except Exception as e:
            logger.error(f"Error measuring Redis latency: {str(e)}")
            return -1
    
    async def measure_network_latency(self, host: str) -> float:
        """
        Measure network latency to a host.
        
        Args:
            host: Hostname to ping
            
        Returns:
            Latency in milliseconds
        """
        try:
            import socket
            import time
            
            start_time = time.time()
            socket.gethostbyname(host)
            end_time = time.time()
            
            return (end_time - start_time) * 1000  # Convert to milliseconds
        except Exception as e:
            logger.error(f"Error measuring network latency to {host}: {str(e)}")
            return -1
    
    async def collect_throughput_metrics(self) -> Dict[str, float]:
        """
        Collect system throughput metrics.
        
        Returns:
            Dictionary containing throughput metrics
        """
        throughput_metrics = {}
        
        try:
            # Get Redis throughput
            redis_throughput = self.redis_client.info().get('instantaneous_ops_per_sec', 0)
            throughput_metrics['redis_ops'] = redis_throughput
            
            # Get database throughput (approximation based on recent queries)
            db_queries_key = "system:stats:db_queries"
            db_query_count = self.redis_client.get(db_queries_key)
            
            if db_query_count:
                throughput_metrics['db_queries'] = float(db_query_count)
            else:
                throughput_metrics['db_queries'] = 0
            
            # Get trading throughput
            trades_key = "system:stats:trades_per_minute"
            trades_count = self.redis_client.get(trades_key)
            
            if trades_count:
                throughput_metrics['trades'] = float(trades_count)
            else:
                throughput_metrics['trades'] = 0
        except Exception as e:
            logger.error(f"Error collecting throughput metrics: {str(e)}")
        
        return throughput_metrics
    
    async def collect_error_rate_metrics(self) -> Dict[str, float]:
        """
        Collect system error rate metrics.
        
        Returns:
            Dictionary containing error rate metrics
        """
        error_metrics = {}
        
        try:
            # Get error rates from Redis stats
            error_stats_key = "system:stats:errors"
            error_stats = self.redis_client.hgetall(error_stats_key)
            
            for error_type, count in error_stats.items():
                error_metrics[error_type.decode('utf-8')] = float(count)
            
            # Calculate error rates
            request_count_key = "system:stats:request_count"
            request_count = self.redis_client.get(request_count_key)
            
            if request_count:
                request_count = float(request_count)
                total_errors = sum(error_metrics.values())
                error_metrics['error_rate'] = (total_errors / request_count) if request_count > 0 else 0
            else:
                error_metrics['error_rate'] = 0
        except Exception as e:
            logger.error(f"Error collecting error rate metrics: {str(e)}")
        
        return error_metrics
    
    async def collect_response_time_metrics(self) -> Dict[str, float]:
        """
        Collect system response time metrics.
        
        Returns:
            Dictionary containing response time metrics
        """
        response_metrics = {}
        
        try:
            # Get response times from Redis stats
            response_stats_key = "system:stats:response_times"
            response_stats = self.redis_client.hgetall(response_stats_key)
            
            for endpoint, time_data in response_stats.items():
                response_metrics[endpoint.decode('utf-8')] = float(time_data)
        except Exception as e:
            logger.error(f"Error collecting response time metrics: {str(e)}")
        
        return response_metrics
    
    def publish_resource_metrics(self, resources: Dict[str, Any]):
        """
        Publish resource metrics to Redis for UI access.
        
        Args:
            resources: Dictionary containing resource metrics
        """
        try:
            # Publish current resource usage
            resources_key = "system:resources:current"
            self.redis_client.set(resources_key, json.dumps(resources))
            
            # Publish resource history (recent)
            recent_history = {
                'cpu': self.resource_history['cpu'][-60:],
                'memory': self.resource_history['memory'][-60:],
                'disk': self.resource_history['disk'][-60:],
                'network': self.resource_history['network'][-60:],
                'gpu': self.resource_history['gpu'][-60:],
                'timestamps': self.resource_history['timestamps'][-60:]
            }
            
            history_key = "system:resources:history"
            self.redis_client.set(history_key, json.dumps(recent_history))
            
            # Publish alert status
            alerts_key = "system:resources:alerts"
            alerts_data = {
                'cpu': self.alerts['cpu'],
                'memory': self.alerts['memory'],
                'disk': self.alerts['disk'],
                'network': self.alerts['network'],
                'gpu': self.alerts['gpu'],
                'component_failures': list(self.alerts['component_failures'])
            }
            self.redis_client.set(alerts_key, json.dumps(alerts_data))
        except Exception as e:
            logger.error(f"Error publishing resource metrics to Redis: {str(e)}")
    
    def publish_performance_metrics(self):
        """Publish performance metrics to Redis for UI access"""
        try:
            # Publish performance metrics
            performance_key = "system:performance"
            self.redis_client.set(performance_key, json.dumps(self.performance_metrics))
        except Exception as e:
            logger.error(f"Error publishing performance metrics to Redis: {str(e)}")
    
    async def update_system_status(self):
        """Update overall system status at regular intervals"""
        interval = self.check_intervals.get('system_status', 60)  # Default: update every 60 seconds
        
        while True:
            try:
                # Calculate overall system health
                system_status = self.calculate_system_health()
                
                # Update system info
                self.system_info = self.get_system_info()
                
                # Publish system status to Redis
                status_key = "system:status"
                self.redis_client.set(status_key, json.dumps(system_status))
                
                # Publish system info to Redis
                info_key = "system:info"
                self.redis_client.set(info_key, json.dumps(self.system_info))
                
                # Wait for next update
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error updating system status: {str(e)}")
                await asyncio.sleep(interval)
    
    def calculate_system_health(self) -> Dict[str, Any]:
        """
        Calculate overall system health based on component status and resource usage.
        
        Returns:
            Dictionary containing system health information
        """
        # Count components by status
        status_counts = {'healthy': 0, 'degraded': 0, 'critical': 0, 'error': 0, 'unknown': 0}
        
        for component, status_data in self.component_status.items():
            status = status_data.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Determine overall status
        overall_status = 'healthy'
        status_message = 'All systems operational'
        
        if status_counts.get('critical', 0) > 0:
            overall_status = 'critical'
            status_message = f"{status_counts['critical']} components in critical state"
        elif status_counts.get('error', 0) > 0:
            overall_status = 'error'
            status_message = f"{status_counts['error']} components with errors"
        elif status_counts.get('degraded', 0) > 0:
            overall_status = 'degraded'
            status_message = f"{status_counts['degraded']} components in degraded state"
        
        # Check resource alerts
        resource_alerts = []
        for resource_type, alert_status in self.alerts.items():
            if resource_type != 'component_failures' and alert_status:
                resource_alerts.append(resource_type)
        
        if resource_alerts and overall_status == 'healthy':
            overall_status = 'degraded'
            status_message = f"Resource alerts: {', '.join(resource_alerts)}"
        
        # Calculate health score (0-100)
        component_weight = 0.7  # 70% of score based on component health
        resource_weight = 0.3  # 30% of score based on resource usage
        
        component_score = 100
        if status_counts.get('total', 0) > 0:
            total_components = sum(status_counts.values())
            component_score = (
                (status_counts.get('healthy', 0) * 100) +
                (status_counts.get('degraded', 0) * 50) +
                (status_counts.get('critical', 0) * 10) +
                (status_counts.get('error', 0) * 0) +
                (status_counts.get('unknown', 0) * 0)
            ) / total_components
        
        resource_score = 100
        resource_alerts_count = len(resource_alerts)
        if resource_alerts_count > 0:
            resource_score = max(0, 100 - (resource_alerts_count * 25))
        
        health_score = (component_score * component_weight) + (resource_score * resource_weight)
        
        # Build system health object
        system_health = {
            'status': overall_status,
            'message': status_message,
            'health_score': health_score,
            'component_stats': status_counts,
            'resource_alerts': resource_alerts,
            'timestamp': int(time.time()),
            'uptime': self.get_system_uptime()
        }
        
        return system_health
    
    def get_system_uptime(self) -> float:
        """
        Get system uptime in seconds.
        
        Returns:
            System uptime in seconds
        """
        try:
            return psutil.boot_time()
        except Exception:
            return 0
    
    async def get_alerts(self, limit: int = 100, offset: int = 0,
                         severity: Optional[str] = None,
                         acknowledged: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Get system alerts from the database.
        
        Args:
            limit: Maximum number of alerts to return
            offset: Offset for pagination
            severity: Filter by severity (critical, error, warning, info)
            acknowledged: Filter by acknowledgement status
            
        Returns:
            List of alert dictionaries
        """
        try:
            # Build query based on filters
            query = "SELECT * FROM system_alerts"
            params = []
            
            where_clauses = []
            
            if severity:
                where_clauses.append("severity = ?")
                params.append(severity)
            
            if acknowledged is not None:
                where_clauses.append("acknowledged = ?")
                params.append(1 if acknowledged else 0)
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Execute query
            results = await self.db_client.fetch(query, *params)
            
            alerts = []
            for row in results:
                alert = {
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'type': row['type'],
                    'resource': row['resource'],
                    'message': row['message'],
                    'severity': row.get('severity', 'error'),
                    'acknowledged': bool(row['acknowledged'])
                }
                alerts.append(alert)
            
            return alerts
        except Exception as e:
            logger.error(f"Error retrieving alerts: {str(e)}")
            return []
    
    async def acknowledge_alert(self, alert_id: int) -> bool:
        """
        Acknowledge a system alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await self.db_client.execute(
                "UPDATE system_alerts SET acknowledged = 1 WHERE id = ?",
                (alert_id,)
            )
            logger.info(f"Alert {alert_id} acknowledged")
            return True
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {str(e)}")
            return False
    
    async def get_recovery_actions(self, limit: int = 100, offset: int = 0,
                                   component: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get system recovery actions from the database.
        
        Args:
            limit: Maximum number of actions to return
            offset: Offset for pagination
            component: Filter by component name
            
        Returns:
            List of recovery action dictionaries
        """
        try:
            # Build query based on filters
            query = "SELECT * FROM recovery_actions"
            params = []
            
            if component:
                query += " WHERE component = ?"
                params.append(component)
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Execute query
            results = await self.db_client.fetch(query, *params)
            
            actions = []
            for row in results:
                action = {
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'component': row['component'],
                    'status': row['status'],
                    'failures': row['failures'],
                    'message': row['message'],
                    'action': row['action']
                }
                actions.append(action)
            
            return actions
        except Exception as e:
            logger.error(f"Error retrieving recovery actions: {str(e)}")
            return []
    
    def get_component_status(self, component_name: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get status for a specific component or all components.
        
        Args:
            component_name: Name of component or None for all components
            
        Returns:
            Dictionary or list of dictionaries containing component status
        """
        if component_name:
            # Return status for specific component
            return self.component_status.get(component_name, {
                'status': 'unknown',
                'last_check': 0,
                'failures': 0,
                'message': 'Component not found'
            })
        else:
            # Return status for all components
            return [
                {'name': name, **status}
                for name, status in self.component_status.items()
            ]
    
    def get_resource_usage_history(self, resource_type: str, 
                                duration: str = '1h') -> Dict[str, List[float]]:
        """
        Get historical resource usage data.
        
        Args:
            resource_type: Type of resource (cpu, memory, disk, network, gpu)
            duration: Duration of history to return (1h, 6h, 24h, 7d)
            
        Returns:
            Dictionary containing resource usage history
        """
        try:
            # Get points based on duration
            points = {
                '1h': 60,      # 1 minute intervals for 1 hour
                '6h': 360,     # 1 minute intervals for 6 hours
                '24h': 1440,   # 1 minute intervals for 24 hours
                '7d': 10080    # 1 minute intervals for 7 days
            }.get(duration, 60)
            
            # Limit points to available history
            points = min(points, len(self.resource_history.get('timestamps', [])))
            
            if points == 0:
                return {'timestamps': [], 'values': []}
            
            # Get resource history
            timestamps = self.resource_history.get('timestamps', [])[-points:]
            values = self.resource_history.get(resource_type, [])[-points:]
            
            return {
                'timestamps': timestamps,
                'values': values
            }
        except Exception as e:
            logger.error(f"Error getting resource usage history: {str(e)}")
            return {'timestamps': [], 'values': []}
    
    async def cleanup(self):
        """Cleanup resources and cancel monitoring tasks"""
        logger.info("Cleaning up system health monitor...")
        
        # Cancel all monitoring tasks
        for task in self.health_check_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        for task in self.health_check_tasks.values():
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        logger.info("System health monitor cleanup completed")
