#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
API Gateway Initialization

This module initializes the API Gateway component that serves as the interface between
the frontend UI and the backend services of the QuantumSpectre Elite Trading System.
"""

import logging
from typing import Dict, List, Any
from api_gateway.app import (
    ApiGatewayService,
    initialize_api,
    register_route,
    register_middleware,
    register_auth_provider,
    register_ws_handler,
)

# Setup package metadata
__version__ = '1.0.0'
__author__ = 'QuantumSpectre Team'
__description__ = 'API Gateway for QuantumSpectre Elite Trading System'

# Initialize the global registry for API routes
registered_routes: Dict[str, Dict[str, Any]] = {
    'GET': {},
    'POST': {},
    'PUT': {},
    'DELETE': {},
    'WS': {}  # WebSocket routes
}

# Initialize the global registry for API middlewares
registered_middlewares: List[Any] = []

# Initialize authentication providers
auth_providers = {}

# Initialize WebSocket connection handlers
ws_connections = {}
ws_channels = {}

# Initialize rate limiting configuration
rate_limit_config = {
    'default': {
        'requests': 100,
        'period': 60  # seconds
    },
    'trading': {
        'requests': 10,
        'period': 60  # seconds
    },
    'data': {
        'requests': 300,
        'period': 60  # seconds
    }
}

# Configure logging
logger = logging.getLogger('api_gateway')

# Export public interface
__all__ = [
    'ApiGatewayService',
    'initialize_api',
    'register_route',
    'register_middleware',
    'register_auth_provider',
    'register_ws_handler',
    'registered_routes',
    'registered_middlewares',
    'auth_providers',
    'ws_connections',
    'ws_channels',
    'rate_limit_config',
    '__version__'
]
