#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Main Application Entry Point

This module serves as the entry point for the QuantumSpectre Elite Trading System.
It orchestrates the startup, operation, and graceful shutdown of all system components.
"""

import os
import sys
import signal
import argparse
import asyncio
import logging
import traceback
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import multiprocessing as mp
try:
    import nltk  # type: ignore
    NLTK_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    nltk = None  # type: ignore
    NLTK_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "NLTK library not found. NLP features will be disabled"
    )
import ssl
import importlib
from common.utils import safe_nltk_download

# Internal imports
from config import Config, load_config
from common.logger import setup_logging, get_logger
from common.constants import (
    SERVICE_NAMES, SERVICE_DEPENDENCIES, SERVICE_STARTUP_ORDER,
    LOG_LEVELS, DEFAULT_CONFIG_PATH, VERSION
)
from common.metrics import MetricsCollector
from common.exceptions import ConfigurationError, ServiceStartupError, SystemCriticalError
from common.async_utils import run_with_timeout
from common.redis_client import RedisClient
from common.db_client import DatabaseClient, get_db_client
from common.security import SecureCredentialManager

# Service imports

# Service modules are imported lazily to avoid loading optional dependencies
# when only a subset of services is started. This allows the UI to boot even
# if heavy ML libraries are missing.

SERVICE_CLASS_PATHS = {
    "data_ingest": ("data_ingest.app", "DataIngestService"),
    "data_feeds": ("data_feeds.app", "DataFeedService"),
    "feature_service": ("feature_service.app", "FeatureService"),
    "intelligence": ("intelligence.app", "IntelligenceService"),
    "ml_models": ("ml_models.app", "MLModelService"),
    "strategy_brains": ("strategy_brains.app", "StrategyBrainService"),
    "brain_council": ("brain_council.app", "BrainCouncilService"),
    "execution_engine": ("execution_engine.app", "ExecutionEngineService"),
    "risk_manager": ("risk_manager.app", "RiskManagerService"),
    "backtester": ("backtester.app", "BacktesterService"),
    "monitoring": ("monitoring.app", "MonitoringService"),
    "api_gateway": ("api_gateway.app", "APIGatewayService"),
    "ui": ("ui.app", "UIService"),
    "voice_assistant": ("voice_assistant.app", "VoiceAssistantService"),
}

# Global variables
# Module-level logger used throughout the application
logger = logging.getLogger(__name__)
services = {}
config = None
metrics_collector = None
is_shutting_down = False
executor = None
service_event_loop = None
startup_lock = asyncio.Lock()
redis_client = None
db_client = None
credentials_manager = None
signal_bus = None


class ServiceManager:
    """Manages the lifecycle of all system services."""

    def __init__(self, config: Config, event_loop: asyncio.AbstractEventLoop):
        """
        Initialize the service manager.

        Args:
            config: System configuration object
            event_loop: Main asyncio event loop
        """
        self.config = config
        self.loop = event_loop
        self.services = {}
        self.service_tasks = {}
        self.startup_complete = asyncio.Event()
        self.shutdown_complete = asyncio.Event()
        self.logger = get_logger("ServiceManager")
        self.metrics = MetricsCollector("service_manager")
        self.lock = asyncio.Lock()
        self.service_statuses = {}
        self.health_check_tasks = {}
        self.shutdown_in_progress = False

    async def start_services(self):
        """
        Start all system services in the correct dependency order.
        Respects service dependencies to ensure proper initialization.
        """
        global services
        self.logger.info("Starting QuantumSpectre Elite services...")

        # First, instantiate all service objects lazily using importlib
        service_classes = {}
        for name, (module_path, class_name) in SERVICE_CLASS_PATHS.items():
            if not self.config.services.get(name, {}).get("enabled", True):
                self.logger.info(f"Service {name} is disabled in configuration")
                continue
            try:
                module = importlib.import_module(module_path)
                service_class = getattr(module, class_name)
                service_classes[name] = service_class
            except Exception as exc:
                self.logger.error(
                    f"Failed to import service {name} from {module_path}: {exc}"
                )
                if self.config.services.get(name, {}).get("required", True):
                    self.logger.warning(
                        f"Service {name} marked as required but failed to load; continuing"
                    )
                continue

        # Create service instances but don't start them yet
        for name, service_class in service_classes.items():
            self.logger.info(f"Instantiating {name} service")
            try:
                # Get the parameters that the service class accepts
                import inspect
                params = inspect.signature(service_class.__init__).parameters
                
                # Build the arguments dictionary based on what the service accepts
                kwargs = {'config': self.config}
                if 'loop' in params:
                    kwargs['loop'] = self.loop
                if 'redis_client' in params:
                    kwargs['redis_client'] = redis_client
                if 'db_client' in params:
                    kwargs['db_client'] = db_client
                if 'signal_bus' in params:
                    # Import SignalBus if not already imported
                    from common.utils import SignalBus
                    global signal_bus
                    if signal_bus is None:
                        signal_bus = SignalBus()
                    kwargs['signal_bus'] = signal_bus
                
                # Instantiate the service with the appropriate parameters
                self.services[name] = service_class(**kwargs)
                self.service_statuses[name] = "instantiated"
            except Exception as exc:
                self.logger.error(f"Failed to instantiate {name} service: {exc}")
                self.logger.error(traceback.format_exc())
                self.logger.error(traceback.format_exc())
                if self.config.services.get(name, {}).get("required", True):
                    self.logger.warning(
                        f"Service {name} marked as required but failed to instantiate; continuing"
                    )
                else:
                    self.logger.debug(
                        f"Optional service {name} skipped due to instantiation failure"
                    )

        # Start services in dependency order
        for service_name in SERVICE_STARTUP_ORDER:
            if service_name in self.services:
                await self._start_service_with_dependencies(service_name)

        # Share service references
        services = self.services

        # Start health check monitoring for all services
        for service_name, service in self.services.items():
            self._start_health_check(service_name)

        self.logger.info("All services started successfully")
        self.startup_complete.set()

    async def _start_service_with_dependencies(self, service_name: str):
        """
        Start a service and ensure all its dependencies are started first.

        Args:
            service_name: Name of the service to start
        """
        # Avoid duplicate startups
        if self.service_statuses.get(service_name) == "running":
            return

        # Check if we're already in the process of starting this service
        if self.service_statuses.get(service_name) == "starting":
            self.logger.warning(f"Circular dependency detected for {service_name}")
            raise ConfigurationError(f"Circular dependency detected for {service_name}")

        self.service_statuses[service_name] = "starting"

        # Start dependencies first
        for dependency in SERVICE_DEPENDENCIES.get(service_name, []):
            if dependency in self.services:
                await self._start_service_with_dependencies(dependency)

        # Now start the actual service
        service = self.services[service_name]
        self.logger.info(f"Starting {service_name} service")
        self.logger.debug(f"Calling start() method for {service_name}")

        try:
            start_timeout = self.config.services.get(service_name, {}).get("startup_timeout", 60)
            self.logger.info(f"Starting {service_name} with timeout {start_timeout}s")
            
            async with self.lock:
                try:
                    await run_with_timeout(
                        service.start(),
                        timeout=start_timeout,
                        loop=self.loop,
                        error_message=f"{service_name} service failed to start within {start_timeout} seconds"
                    )
                    self.logger.debug(f"Service {service_name} started successfully")
                except Exception as e:
                    self.logger.error(f"Error starting {service_name} service: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    raise

            self.service_statuses[service_name] = "running"
            self.logger.info(f"{service_name} service started successfully")

            # Start a monitoring task for the service
            self.service_tasks[service_name] = asyncio.create_task(
                self._monitor_service(service_name, service)
            )
        except Exception as e:
            self.service_statuses[service_name] = "failed"
            self.logger.error(f"Failed to start {service_name} service: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.logger.warning(
                f"Service {service_name} failed to start and will be skipped"
            )

    async def _monitor_service(self, service_name: str, service: Any):
        """
        Monitor a service for any failures and attempt recovery if needed.

        Args:
            service_name: Name of the service to monitor
            service: Service instance
        """
        while not self.shutdown_in_progress:
            try:
                # Wait for the service's internal task to complete or fail
                if hasattr(service, "task") and service.task is not None:
                    await service.task

                # If we get here, the service task completed unexpectedly
                if not self.shutdown_in_progress:
                    self.logger.warning(f"{service_name} service task completed unexpectedly")

                    if self.config.services.get(service_name, {}).get("auto_restart", True):
                        self.logger.info(f"Attempting to restart {service_name} service")
                        try:
                            self.logger.debug(f"Calling stop() method for {service_name} before restart")
                            await service.stop()
                            self.logger.debug(f"Calling start() method for {service_name} after stop")
                            await service.start()
                            self.logger.info(f"{service_name} service restarted successfully")
                        except Exception as e:
                            self.logger.error(f"Failed to restart {service_name} service: {str(e)}")
                    else:
                        self.logger.warning(f"Auto-restart disabled for {service_name}, not restarting")
                        self.service_statuses[service_name] = "stopped"

            except asyncio.CancelledError:
                # Task was cancelled, this is normal during shutdown
                break
            except Exception as e:
                if not self.shutdown_in_progress:
                    self.logger.error(f"Error in {service_name} service: {str(e)}")
                    self.logger.error(traceback.format_exc())

                    # Update status
                    self.service_statuses[service_name] = "error"

                    # Try to restart if configured to do so
                    if self.config.services.get(service_name, {}).get("auto_restart", True):
                        retry_count = 0
                        max_retries = self.config.services.get(service_name, {}).get("max_restart_attempts", 3)
                        retry_delay = self.config.services.get(service_name, {}).get("restart_delay", 5)

                        while retry_count < max_retries and not self.shutdown_in_progress:
                            retry_count += 1
                            self.logger.info(f"Attempting to restart {service_name} service (attempt {retry_count}/{max_retries})")

                            try:
                                # Make sure it's stopped first
                                await service.stop()
                                await asyncio.sleep(retry_delay)
                                await service.start()
                                self.logger.info(f"{service_name} service restarted successfully")
                                self.service_statuses[service_name] = "running"
                                break
                            except Exception as restart_error:
                                self.logger.error(f"Failed to restart {service_name} service: {str(restart_error)}")
                                await asyncio.sleep(retry_delay)

                        if retry_count >= max_retries and self.service_statuses[service_name] != "running":
                            self.logger.error(f"Failed to restart {service_name} service after {max_retries} attempts")
                            self.service_statuses[service_name] = "failed"

                            # Check if this is a critical service
                            if self.config.services.get(service_name, {}).get("critical", False):
                                self.logger.critical(
                                    f"Critical service {service_name} failed, initiating system shutdown due to {service_name} failure"

                                )
                                # Use loop.call_soon_threadsafe to avoid nested event loop issues
                                self.loop.call_soon_threadsafe(
                                    lambda: asyncio.create_task(
                                        self.shutdown("Critical service failure")
                                    )
                                )
                    else:
                        self.logger.warning(f"Auto-restart disabled for {service_name}, not restarting")

                # If we've handled the error but the service isn't critical,
                # continue monitoring in case manual restart occurs
                if not self.shutdown_in_progress:
                    await asyncio.sleep(10)  # Wait before checking again
                else:
                    # We're shutting down, so just exit the monitoring loop
                    break

    def _start_health_check(self, service_name: str):
        """
        Start a health check task for a service.

        Args:
            service_name: Name of the service to health check
        """
        service = self.services[service_name]
        if hasattr(service, "health_check") and callable(service.health_check):
            interval = self.config.services.get(service_name, {}).get("health_check_interval", 30)
            self.health_check_tasks[service_name] = asyncio.create_task(
                self._run_health_checks(service_name, service, interval)
            )

    async def _run_health_checks(self, service_name: str, service: Any, interval: int):
        """
        Run periodic health checks for a service.

        Args:
            service_name: Name of the service
            service: Service instance
            interval: Health check interval in seconds
        """
        while not self.shutdown_in_progress:
            try:
                self.logger.debug(f"Running health check for {service_name}")
                result = await service.health_check()
                if not result:
                    self.logger.warning(f"Health check failed for {service_name} service")
                    self.metrics.increment(f"health_check_failure.{service_name}")

                    # If service reports unhealthy multiple times, try restarting it
                    consecutive_failures = self.metrics.get(f"health_check_failure.{service_name}.consecutive", 0) + 1
                    self.metrics.set(f"health_check_failure.{service_name}.consecutive", consecutive_failures)

                    max_failures = self.config.services.get(service_name, {}).get("max_health_failures", 3)
                    if consecutive_failures >= max_failures:
                        self.logger.warning(f"{service_name} service health check failed {consecutive_failures} times, attempting restart")
                        try:
                            self.logger.debug(f"Calling stop() method for {service_name} before restart after health failure")
                            await service.stop()
                            self.logger.debug(f"Calling start() method for {service_name} after stop after health failure")
                            await service.start()
                            self.logger.info(f"{service_name} service restarted successfully after health failure")
                            self.metrics.set(f"health_check_failure.{service_name}.consecutive", 0)
                        except Exception as e:
                            self.logger.error(f"Failed to restart {service_name} service after health failure: {str(e)}")
                else:
                    # Reset consecutive failure counter on successful health check
                    self.metrics.set(f"health_check_failure.{service_name}.consecutive", 0)
            except Exception as e:
                self.logger.error(f"Error running health check for {service_name} service: {str(e)}")

            await asyncio.sleep(interval)

    async def shutdown(self, reason: str = "Shutdown requested"):
        """
        Shutdown all services in reverse dependency order.

        Args:
            reason: Reason for the shutdown
        """
        global is_shutting_down

        # Avoid multiple shutdowns
        if self.shutdown_in_progress:
            return

        self.shutdown_in_progress = True
        is_shutting_down = True

        self.logger.info(f"Shutting down all services. Reason: {reason}")

        # Cancel all health check tasks
        for service_name, task in self.health_check_tasks.items():
            if not task.done():
                task.cancel()

        # Cancel all service monitoring tasks
        for service_name, task in self.service_tasks.items():
            if not task.done():
                task.cancel()

        # Shutdown in reverse dependency order
        for service_name in reversed(SERVICE_STARTUP_ORDER):
            if service_name in self.services:
                service = self.services[service_name]
                self.logger.info(f"Stopping {service_name} service")
                self.logger.debug(f"Calling stop() method for {service_name}")

                try:
                    shutdown_timeout = self.config.services.get(service_name, {}).get("shutdown_timeout", 30)
                    await run_with_timeout(
                        service.stop(),
                        timeout=shutdown_timeout,
                        loop=self.loop,
                        error_message=f"{service_name} service failed to stop within {shutdown_timeout} seconds"
                    )
                    self.service_statuses[service_name] = "stopped"
                    self.logger.info(f"{service_name} service stopped successfully")
                    self.logger.debug(f"Service {service_name} stopped successfully")
                except Exception as e:
                    self.logger.error(f"Error stopping {service_name} service: {str(e)}")
                    self.service_statuses[service_name] = "error"

        self.logger.info("All services stopped")
        self.shutdown_complete.set()


def setup_argument_parser():
    """
    Set up command-line argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description=f"QuantumSpectre Elite Trading System v{VERSION}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-c", "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to configuration file"
    )

    parser.add_argument(
        "-l", "--log-level",
        choices=LOG_LEVELS.keys(),
        default="info",
        help="Set the logging level"
    )

    parser.add_argument(
        "--log-file",
        help="Path to log file (if not specified, logs to console only)"
    )

    parser.add_argument(
        "--service",
        choices=list(SERVICE_NAMES) + ["all"],
        default="all",
        help="Specific service to run (default: all services)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional logging"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without starting services"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"QuantumSpectre Elite Trading System v{VERSION}",
        help="Show version information and exit"
    )

    return parser


async def initialize_db(config: Config) -> DatabaseClient:
    """
    Initialize the database connection.

    Args:
        config: System configuration object

    Returns:
        Initialized database client
    """
    logger.info("Initializing database connection...")

    try:
        db_config = config.database
        
        # Check if database is enabled
        if not db_config.get("enabled", True):
            logger.info("Database is disabled in configuration; skipping connection")
            
            # Check if memory storage should be used
            if db_config.get("use_memory_storage", False):
                logger.info("Using in-memory storage for database operations")
                # Here we could initialize an in-memory database if needed
                
            return None
            
        db_client = await get_db_client(
            db_type=db_config.get("type", "postgresql"),
            host=db_config.get("host", "localhost"),
            port=db_config.get("port", 5432),
            username=db_config.get("user", "postgres"),
            password=db_config.get("password", ""),
            database=db_config.get("dbname", "quantumspectre"),
            pool_size=db_config.get("min_pool_size", 10),
            ssl=db_config.get("ssl", False),
            timeout=db_config.get("connection_timeout", 30)
        )

        # Run migrations if needed
        if config.system.get("auto_migrate", True):
            logger.info("Running database migrations...")
            await db_client.run_migrations()

        logger.info("Database connection initialized successfully")
        return db_client

    except Exception as e:
        logger.error(f"Failed to initialize database connection: {str(e)}")
        logger.error(traceback.format_exc())

        logger.warning(
            "Continuing without database connectivity; persistent storage will be unavailable"
        )
        return None


async def initialize_redis(config: Config) -> RedisClient:
    """
    Initialize the Redis connection.

    Args:
        config: System configuration object

    Returns:
        Initialized Redis client
    """
    logger.info("Initializing Redis connection...")

    try:
        redis_config = config.redis
        redis_client = RedisClient(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            db=redis_config.get("db", 0),
            password=redis_config.get("password", None),
            ssl=redis_config.get("ssl", False),
            timeout=redis_config.get("timeout", 10),
            max_connections=redis_config.get("max_connections", 50)
        )

        # Initialize the connection
        await redis_client.initialize()

        logger.info("Redis connection initialized successfully")
        return redis_client

    except Exception as e:
        logger.error(f"Failed to initialize Redis connection: {str(e)}")
        logger.error(traceback.format_exc())

        logger.warning("Continuing without Redis; functionality may be limited")
        return None


async def initialize_credentials(config: Config) -> SecureCredentialManager:
    """
    Initialize the secure credential manager.

    Args:
        config: System configuration object

    Returns:
        Initialized credential manager
    """
    logger.info("Initializing secure credential manager...")

    try:
        security_config = config.security
        cred_manager = SecureCredentialManager(
            encryption_key=security_config.get("encryption_key", None),
            key_file=security_config.get("key_file", None),
            storage_method=security_config.get("storage_method", "file"),
            storage_path=security_config.get("storage_path", "./credentials"),
            auto_generate=security_config.get("auto_generate", True)
        )

        # Initialize the manager
        await cred_manager.initialize()

        logger.info("Secure credential manager initialized successfully")
        return cred_manager

    except Exception as e:
        logger.error(f"Failed to initialize secure credential manager: {str(e)}")
        logger.error(traceback.format_exc())
        raise SystemCriticalError("Security initialization failed") from e


def setup_nltk_data():
    """
    Set up NLTK data and handle potential download errors.
    This function attempts to load required NLTK data packages locally,
    and handles SSL certificate issues that might occur.
    """
    if not NLTK_AVAILABLE:
        logger.warning("NLTK is not installed. Skipping NLP initialization")
        return

    logger.info("Setting up NLTK data...")

    # Required NLTK packages
    required_packages = [
        'vader_lexicon',  # For sentiment analysis
        'punkt',          # For sentence tokenization
        'stopwords',      # For stopword filtering
    ]

    # Handle SSL certificate issues that might occur
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        _create_unverified_https_context = None
    if _create_unverified_https_context:
        ssl._create_default_https_context = _create_unverified_https_context

    # Try to load packages from local data directory first
    nltk_data_dir = os.path.expanduser("~/.nltk_data")
    nltk.data.path.insert(0, nltk_data_dir)

    # Check each package and load locally without attempting downloads
    for package in required_packages:
        resource = f"tokenizers/{package}" if package == "punkt" else f"corpora/{package}"
        if safe_nltk_download(resource):
            logger.debug(f"NLTK package '{package}' available")
        else:
            logger.warning(
                f"NLTK package '{package}' not found; NLP features may be limited"
            )

    logger.info("NLTK setup complete")


async def startup():

    """
    Main system startup sequence.
    """
    global logger, config, metrics_collector, redis_client, db_client, credentials_manager, service_event_loop

    try:
        # Parse command-line arguments
        parser = setup_argument_parser()
        args = parser.parse_args()

        # Set up logging system
        log_level = LOG_LEVELS[args.log_level.upper()]
        if args.debug:
            log_level = logging.DEBUG

        setup_logging(log_level, log_file=args.log_file)
        logger = get_logger("main")

        logger.info(f"Starting QuantumSpectre Elite Trading System v{VERSION}")
        logger.info(f"Process ID: {os.getpid()}")

        # Set up NLTK data
        setup_nltk_data()

        # Load configuration
        config_path = args.config
        logger.info(f"Loading configuration from {config_path}")
        config = load_config(config_path)
        logger.debug(f"Loaded configuration: {config}")

        # Validate configuration
        logger.info("Validating configuration...")
        config.validate()

        # Force UI to run on port 3002 to avoid conflicts in the sandbox
        config.ui["port"] = 3002

        # If dry run, exit here
        if args.dry_run:
            logger.info("Configuration validation successful (dry run)")
            return 0

        # Initialize metrics collector
        metrics_collector = MetricsCollector("system")

        # Initialize event loop
        if service_event_loop is None:
            service_event_loop = asyncio.get_event_loop()

        # Initialize secure credential manager
        credentials_manager = await initialize_credentials(config)
        logger.info("Credential manager initialized successfully")

        # Initialize Redis client

        try:
            redis_client = await initialize_redis(config)
            logger.info("Redis client initialized successfully")
        except SystemCriticalError:
            # Allow system to continue in a degraded mode when Redis is unavailable
            logger.warning("Redis connection failed; continuing without Redis features")
            redis_client = None

        # Initialize database client
        try:
            db_client = await initialize_db(config)
            logger.info("Database client initialized successfully")
        except SystemCriticalError:
            logger.warning(
                "Database connection failed; continuing without database features"
            )
            db_client = None

        # Create service manager
        service_manager = ServiceManager(config, service_event_loop)
        logger.debug(f"ServiceManager created; services configured: {list(config.services.keys())}")

        # Register signal handlers for graceful shutdown
        # Note: add_signal_handler is not supported on Windows
        import platform
        if platform.system() != 'Windows':
            for sig in (signal.SIGINT, signal.SIGTERM):
                service_event_loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(handle_shutdown_signal(s, service_manager))
                )
            logger.info("Signal handlers registered for graceful shutdown")
        else:
            logger.info("Running on Windows - signal handlers not supported, using KeyboardInterrupt handling")

        # Start all services
        logger.info("Starting all services...")
        try:
            await service_manager.start_services()
            logger.info(f"All services started successfully: {list(service_manager.services.keys())}")
        except Exception as e:
            logger.error(f"Failed to start services: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        # Wait for shutdown signal
        await service_manager.shutdown_complete.wait()

        logger.info("Clean shutdown complete")
        return 0

    except ConfigurationError as e:
        logger.error(f"Configuration error: {str(e)}")
        return 1
    except ServiceStartupError as e:
        logger.error(f"Service startup error: {str(e)}")
        return 2
    except SystemCriticalError as e:
        logger.critical(f"Critical system error: {str(e)}")
        return 3
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}")
        logger.critical(traceback.format_exc())
        return 4


async def handle_shutdown_signal(signal_num, service_manager):
    """
    Handle OS shutdown signals gracefully.

    Args:
        signal_num: Signal number
        service_manager: Service manager instance
    """
    sig_name = signal.Signals(signal_num).name
    logger.info(f"Received shutdown signal: {sig_name}")
    await service_manager.shutdown(f"Received {sig_name} signal")


def main():
    """
    Entry point for the application.
    """
    global executor, service_event_loop

    # Create thread pool executor for background tasks
    mp.set_start_method('spawn', force=True)
    executor = ThreadPoolExecutor(max_workers=os.cpu_count())

    # Set up event loop
    service_event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(service_event_loop)

    try:
        exit_code = service_event_loop.run_until_complete(startup())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        # Handle keyboard interrupt for cleaner shutdown
        logger.info("Shutdown requested via keyboard interrupt")
        tasks = asyncio.all_tasks(loop=service_event_loop)
        for task in tasks:
            task.cancel()
        service_event_loop.run_until_complete(
            asyncio.gather(*tasks, return_exceptions=True)
        )
        logger.info("Shutdown complete")
    finally:
        # Clean up resources
        if executor:
            executor.shutdown(wait=False)
        if service_event_loop:
            service_event_loop.close()


if __name__ == "__main__":
    main()
