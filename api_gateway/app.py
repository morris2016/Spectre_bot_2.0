#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
API Gateway Main Application

This module implements the main API Gateway service that handles HTTP and WebSocket
requests, authentication, routing, and communication with backend services.
"""
import os
import json
import time
import asyncio
import traceback
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager


import jwt  # PyJWT

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# from fastapi.staticfiles import StaticFiles # Not used in this snippet
from fastapi.responses import JSONResponse  # HTMLResponse, FileResponse not used here
# from fastapi.encoders import jsonable_encoder # Not used in this snippet

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.websockets import WebSocketState

# Internal imports - Assuming these paths are correct relative to the project structure
# It's crucial that the Python path is set up correctly for these imports to work.
# For example, the root directory of 'Spectre-bot-master' should be in PYTHONPATH.

# Assuming 'common' is a top-level package or accessible in PYTHONPATH
from common.logger import get_logger


from common.metrics import MetricsCollector
# from common.async_utils import AsyncLimiter, AsyncRetry, AsyncCircuitBreaker # Not used in this snippet
from common.redis_client import RedisClient  # Async-compatible Redis client

# Assuming 'config.py' at the root level provides the Config class/instance
# If 'config.py' provides an instance: from config import app_config as Config (or similar)
from config import Config  # This might need adjustment based on actual project structure


# Import service clients - Assuming these exist and are correctly structured
# These paths suggest that 'brain_council', 'execution_engine', etc., are top-level packages
# or are structured such that these imports are valid.
from brain_council.client import BrainCouncilClient
from execution_engine.client import ExecutionEngineClient
from backtester.client import BacktesterClient
from risk_manager.client import RiskManagerClient
from data_ingest.client import DataIngestClient
from feature_service.client import FeatureServiceClient
from intelligence.client import IntelligenceClient
from monitoring.client import MonitoringClient
from ml_models.client import MLModelsClient
from strategy_brains.client import StrategyBrainsClient

# Import __version__ and rate_limit_config from the package init
from api_gateway import __version__, rate_limit_config


logger = get_logger('api_gateway')
metrics = MetricsCollector('api_gateway')  # Assuming MetricsCollector is appropriately defined
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")  # tokenUrl is relative to the app's root

redis_client: Optional[RedisClient] = None  # Initialized in startup

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware to prevent API abuse."""

    async def dispatch(self, request: Request, call_next):
        if redis_client is None:
            logger.error("Redis client not initialized for RateLimitMiddleware.")
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "Rate limiting service is unavailable."}
            )

        # Get client IP
        client_ip = request.client.host if request.client else "unknown_client"

        # Determine route category (trading, data, etc.)
        path = request.url.path
        category = 'default'
        if '/trading/' in path:
            category = 'trading'
        elif '/data/' in path:
            category = 'data'

        # Get rate limit config for category
        # Ensure rate_limit_config is accessible and structured as expected
        current_rate_limit_config = rate_limit_config.get(category, rate_limit_config.get('default', {'requests': 100, 'period': 60}))
        max_requests = current_rate_limit_config['requests']
        time_period = current_rate_limit_config['period']

        # Check rate limit
        key = f"ratelimit:{category}:{client_ip}"
        current = await redis_client.incr(key)
        if current == 1:
            await redis_client.expire(key, time_period)

        if current > max_requests:
            metrics.increment('rate_limit_exceeded', tags={'category': category}) # Ensure increment method handles tags
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please try again later."}
            )

        # Process the request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        # Record metrics
        # Ensure timing method handles tags
        metrics.record_timing('request_duration', process_time * 1000, tags={'path': path, 'method': request.method})


        response.headers["X-Process-Time"] = str(process_time)
        return response

class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Generate request ID for tracking
        request_id = f"{int(time.time())}-{os.urandom(4).hex()}"
        request.state.request_id = request_id # type: ignore

        # Log request
        logger.info(f"Request {request_id} started: {request.method} {request.url.path}")

        # Process request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Log response
            logger.info(
                f"Request {request_id} completed: {response.status_code} in {process_time:.3f}s"
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request {request_id} failed after {process_time:.3f}s: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            # It's generally better to re-raise the exception or let FastAPI's default error handling
            # convert it to a 500, unless specific error formatting is needed here.
            # If this middleware is before FastAPI's exception handlers, this response will be sent.
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error", "request_id": request_id}
            )

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager for startup and shutdown."""
    # Startup logic
    global redis_client
    # Ensure Config() call is correct based on how your config is structured/loaded.
    # This assumes Config is a class that can be instantiated and provides attributes.
    config_instance = Config() # Or however your global config is accessed

    try:
        # Initialize Redis client
        # Ensure RedisClient is async and await its initialization if it's an async method
        redis_client = RedisClient(
            host=config_instance.get("REDIS_HOST", "localhost"), # Adjusted to use .get() for robustness
            port=int(config_instance.get("REDIS_PORT", 6379)),
            password=config_instance.get("REDIS_PASSWORD", None),
            db=int(config_instance.get("REDIS_DB", 0))
        )
        await redis_client.initialize()


        # Initialize service clients
        logger.info("Initializing service clients...")

        # These URLs should come from the config_instance
        # Example: config_instance.get("BRAIN_COUNCIL_URL")
        # For now, using placeholders or assuming they are attributes of config_instance
        app.state.brain_council_client = BrainCouncilClient(config_instance.get("BRAIN_COUNCIL_URL", "http://localhost:8001")) # type: ignore
        app.state.execution_engine_client = ExecutionEngineClient(config_instance.get("EXECUTION_ENGINE_URL", "http://localhost:8002")) # type: ignore
        app.state.backtester_client = BacktesterClient(config_instance.get("BACKTESTER_URL", "http://localhost:8003")) # type: ignore
        app.state.risk_manager_client = RiskManagerClient(config_instance.get("RISK_MANAGER_URL", "http://localhost:8004")) # type: ignore
        app.state.data_ingest_client = DataIngestClient(config_instance.get("DATA_INGEST_URL", "http://localhost:8005")) # type: ignore
        app.state.feature_service_client = FeatureServiceClient(config_instance.get("FEATURE_SERVICE_URL", "http://localhost:8006")) # type: ignore
        app.state.intelligence_client = IntelligenceClient(config_instance.get("INTELLIGENCE_URL", "http://localhost:8007")) # type: ignore
        app.state.monitoring_client = MonitoringClient(config_instance.get("MONITORING_URL", "http://localhost:8008")) # type: ignore
        app.state.ml_models_client = MLModelsClient(config_instance.get("ML_MODELS_URL", "http://localhost:8009")) # type: ignore
        app.state.strategy_brains_client = StrategyBrainsClient(config_instance.get("STRATEGY_BRAINS_URL", "http://localhost:8010")) # type: ignore


        # Start service health check task
        # Ensure health_check_services is defined and app is passed correctly
        app.state.health_check_task = asyncio.create_task(health_check_services(app)) # type: ignore

        logger.info("API Gateway startup complete")
        yield
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}\n{traceback.format_exc()}")
        # Depending on severity, you might want to exit or handle differently
        raise
    finally:
        # Shutdown logic
        logger.info("Shutting down API Gateway...")
        if hasattr(app.state, 'health_check_task') and app.state.health_check_task: # type: ignore
            app.state.health_check_task.cancel() # type: ignore
            try:
                await app.state.health_check_task # type: ignore
            except asyncio.CancelledError:
                logger.info("Health check task cancelled.")
        if redis_client:
            await redis_client.close()
        logger.info("API Gateway shutdown complete")

async def health_check_services(app: FastAPI):
    """Periodically check health of backend services."""
    while True:
        try:
            # Ensure services are initialized on app.state
            services_to_check = [
                ('brain_council', getattr(app.state, 'brain_council_client', None)),
                ('execution_engine', getattr(app.state, 'execution_engine_client', None)),
                ('backtester', getattr(app.state, 'backtester_client', None)),
                ('risk_manager', getattr(app.state, 'risk_manager_client', None)),
                ('data_ingest', getattr(app.state, 'data_ingest_client', None)),
                ('feature_service', getattr(app.state, 'feature_service_client', None)),
                ('intelligence', getattr(app.state, 'intelligence_client', None)),
                ('monitoring', getattr(app.state, 'monitoring_client', None)),
                ('ml_models', getattr(app.state, 'ml_models_client', None)),
                ('strategy_brains', getattr(app.state, 'strategy_brains_client', None))
            ]

            for name, client in services_to_check:
                if client is None or redis_client is None: # Check redis_client as well
                    logger.warning(f"Service client or Redis client not available for health check: {name}")
                    continue
                try:
                    # Assuming client has an async health_check method
                    status_payload = await client.health_check() # type: ignore
                    # Ensure metrics.gauge handles tags correctly
                    metrics.gauge(f'service_health.{name}', 1 if status_payload.get('status') == 'ok' else 0, tags={'service_name': name}) # type: ignore
                    await redis_client.set(f'service_health:{name}', json.dumps(status_payload))
                except Exception as e:
                    logger.error(f"Health check failed for {name}: {str(e)}")
                    metrics.gauge(f'service_health.{name}', 0, tags={'service_name': name}) # type: ignore
                    await redis_client.set(f'service_health:{name}', json.dumps({
                        'status': 'error',
                        'message': str(e),
                        'timestamp': datetime.utcnow().isoformat()
                    }))
        except Exception as e:
            logger.error(f"Error in health check task: {str(e)}")
        
        # Check every 30 seconds
        await asyncio.sleep(30)


# Create FastAPI app with lifespan
app = FastAPI(
    title="QuantumSpectre Elite Trading System - API Gateway",
    description="API Gateway for QuantumSpectre Elite Trading System",
    version=__version__, # Use __version__ from api_gateway.__init__
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Authentication
async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Validate JWT token and return user info."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if redis_client is None:
        logger.error("Redis client not available for get_current_user")
        raise credentials_exception


    try:
        # Get secret key from config
        config_instance = Config() # Or however your global config is accessed
        secret_key = config_instance.get("JWT_SECRET_KEY", "defaultfallbacksecret") # Provide a default

        # Decode JWT token
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception

        # Get user info from Redis
        user_data_str = await redis_client.get(f"user:{username}")
        if not user_data_str:
            raise credentials_exception
        
        user = json.loads(user_data_str) # Assuming user_data_str is a JSON string
        return user
    except jwt.PyJWTError:
        logger.warning("JWT validation error", exc_info=True)
        raise credentials_exception
    except json.JSONDecodeError:
        logger.error(f"Failed to decode user data from Redis for user: {username}", exc_info=True) # type: ignore
        raise credentials_exception


# Login endpoint
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Generate JWT token for user authentication."""
    if redis_client is None:
        logger.error("Redis client not available for login")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Authentication service unavailable.")

    # Get config
    config_instance = Config()

    # Authenticate user
    user_data_str = await redis_client.get(f"user:{form_data.username}")
    if not user_data_str:
        metrics.increment('failed_login_attempts') # Ensure increment method exists
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        user = json.loads(user_data_str)
    except json.JSONDecodeError:
        logger.error(f"Failed to decode user data from Redis for user: {form_data.username}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error processing user data.")


    # In a real system, you'd use proper password verification (e.g., bcrypt, argon2)
    # This is a simplified example and insecure for production.
    # Assuming password stored in plain text or a reversible way for this example.
    # For a real system, you'd hash the form_data.password and compare with stored hash.
    if form_data.password != user.get("password"): # INSECURE: Replace with hashed password comparison
        metrics.increment('failed_login_attempts')
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expire_minutes_str = config_instance.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60")
    access_token_expire_minutes = int(access_token_expire_minutes_str)
    access_token_expires = timedelta(minutes=access_token_expire_minutes)
    expire = datetime.utcnow() + access_token_expires

    token_data = {
        "sub": user["username"],
        "exp": expire,
        "permissions": user.get("permissions", [])
    }
    
    secret_key = config_instance.get("JWT_SECRET_KEY", "defaultfallbacksecret") # Provide a default
    access_token = jwt.encode(token_data, secret_key, algorithm="HS256")
    metrics.increment('successful_logins')
    return {"access_token": access_token, "token_type": "bearer"}

# Health check endpoint
@app.get("/health")
async def health_check_endpoint(): # Renamed from health_check to avoid conflict
    """API Gateway health check endpoint."""
    return {
        "status": "ok",
        "version": __version__, # Use __version__ from api_gateway.__init__
        "timestamp": datetime.utcnow().isoformat()
    }

# API routes for service status
@app.get("/api/v1/system/status")
async def system_status(request: Request, current_user: Dict = Depends(get_current_user)):
    """Get status of all backend services."""
    if redis_client is None:
        logger.error("Redis client not available for system_status")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="System status service unavailable.")
    try:
        services = [
            'brain_council', 'execution_engine', 'backtester', 'risk_manager',
            'data_ingest', 'feature_service', 'intelligence', 'monitoring',
            'ml_models', 'strategy_brains'
        ]

        status_payload = {} # Renamed from status to avoid conflict with FastAPI status
        for service in services:
            service_status_str = await redis_client.get(f'service_health:{service}')
            if service_status_str:
                status_payload[service] = json.loads(service_status_str)
            else:
                status_payload[service] = {"status": "unknown"}

        return {
            "status": "ok",
            "services": status_payload,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system status"
        )

# WebSocket connection for real-time updates
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time data streaming."""
    await websocket.accept()
    if redis_client is None:
        logger.error("Redis client not available for WebSocket endpoint.")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    # Register the connection
    if not hasattr(app.state, 'connections'):
        app.state.connections = {} # type: ignore
    app.state.connections[client_id] = websocket # type: ignore

    logger.info(f"WebSocket connection established for client {client_id}")
    try:
        # Send initial data
        await websocket.send_json({
            "event": "connected",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Listen for messages
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)

                # Handle subscription requests
                if message.get("type") == "subscribe":
                    topic = message.get("topic")
                    if topic:
                        # Store subscription in Redis
                        await redis_client.sadd(f"subscriptions:{topic}", client_id)
                        await websocket.send_json({
                            "event": "subscribed",
                            "topic": topic,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        logger.info(f"Client {client_id} subscribed to {topic}")

                # Handle unsubscribe requests
                elif message.get("type") == "unsubscribe":
                    topic = message.get("topic")
                    if topic:
                        await redis_client.srem(f"subscriptions:{topic}", client_id)
                        await websocket.send_json({
                            "event": "unsubscribed",
                            "topic": topic,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        logger.info(f"Client {client_id} unsubscribed from {topic}")
            
            except json.JSONDecodeError:
                logger.warning(f"Received invalid JSON from client {client_id}")
                await websocket.send_json({
                    "event": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat()
                })
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for client {client_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint for client {client_id}: {str(e)}", exc_info=True)
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR) # type: ignore
    finally:
        # Remove client from connections
        if hasattr(app.state, 'connections') and client_id in app.state.connections: # type: ignore
            del app.state.connections[client_id] # type: ignore

        # Clean up subscriptions
        async def cleanup_subscriptions():
            if redis_client is None: return
            # Get all subscription keys
            # Ensure redis_client.keys() is an async method or wrapped appropriately
            keys_bytes = await redis_client.keys("subscriptions:*") # type: ignore
            keys = [k.decode('utf-8') for k in keys_bytes] if keys_bytes else []

            for key in keys:
                await redis_client.srem(key, client_id)
        
        asyncio.create_task(cleanup_subscriptions())


async def broadcast_to_topic(topic: str, message: Dict):
    """Broadcast a message to all clients subscribed to a topic."""
    if redis_client is None or not hasattr(app.state, 'connections'):
        logger.error("Redis client or app connections not available for broadcast.")
        return
    try:
        # Get subscribers
        subscribers_bytes = await redis_client.smembers(f"subscriptions:{topic}") # type: ignore
        subscribers = [s.decode('utf-8') for s in subscribers_bytes] if subscribers_bytes else []


        if not subscribers:
            return

        # Prepare message
        if not isinstance(message, dict):
            message = {"data": message} # type: ignore

        message["topic"] = topic # type: ignore
        message["timestamp"] = datetime.utcnow().isoformat() # type: ignore

        # Send to all subscribers
        for client_id_str in subscribers:
            client_id = str(client_id_str) # Ensure it's a string
            if client_id in app.state.connections: # type: ignore
                websocket = app.state.connections[client_id] # type: ignore
                if websocket.client_state == WebSocketState.CONNECTED: # type: ignore
                    try:
                        await websocket.send_json(message) # type: ignore
                    except Exception as e:
                        logger.error(f"Failed to send message to client {client_id}: {str(e)}")
    except Exception as e:
        logger.error(f"Error broadcasting to topic {topic}: {str(e)}", exc_info=True)

# Trading operations API
@app.post("/api/v1/trading/signal")
async def create_trading_signal(request: Request, current_user: Dict = Depends(get_current_user)):
    """Create a manual trading signal."""
    try:
        # Check permissions
        if "trade:create" not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )

        # Parse request body
        data = await request.json()

        # Validate required fields
        required_fields = ["instrument", "direction", "size", "type"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required field: {field}"
                )

        # Forward to brain council
        brain_council_client = request.app.state.brain_council_client
        execution_engine_client = request.app.state.execution_engine_client

        # Get signal confidence from brain council
        signal_result = await brain_council_client.evaluate_manual_signal(
            instrument=data["instrument"],
            direction=data["direction"],
            current_user=current_user["username"]
        )

        # Combine with execution parameters
        execution_params = {
            "instrument": data["instrument"],
            "direction": data["direction"],
            "size": data["size"],
            "type": data["type"],
            "confidence": signal_result.get("confidence", 0),
            "user": current_user["username"],
            "source": "manual"
        }

        # Add optional parameters
        if "stop_loss" in data:
            execution_params["stop_loss"] = data["stop_loss"]
        if "take_profit" in data:
            execution_params["take_profit"] = data["take_profit"]

        # Execute the trade
        result = await execution_engine_client.execute_trade(execution_params)

        # Log the action
        logger.info(f"Manual trade signal created by {current_user['username']}: {data['instrument']} {data['direction']}")

        # Record metrics
        metrics.increment('trade_signals', tags={ # type: ignore
            'user': current_user['username'],
            'instrument': data['instrument'],
            'direction': data['direction'],
            'source': 'manual'
        })

        # Broadcast the signal to subscribed clients
        await broadcast_to_topic("trade_signals", {
            "event": "new_trade_signal",
            "signal": {
                "instrument": data["instrument"],
                "direction": data["direction"],
                "confidence": signal_result.get("confidence", 0),
                "source": "manual",
                "user": current_user["username"]
            }
        })

        return {
            "status": "success",
            "trade_id": result.get("trade_id"),
            "message": "Trade signal created successfully",
            "confidence": signal_result.get("confidence"),
            "execution_status": result.get("status")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating trade signal: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create trade signal"
        )

@app.get("/api/v1/trading/active")
async def get_active_trades(request: Request, current_user: Dict = Depends(get_current_user)):
    """Get all active trades."""
    try:
        # Check permissions
        if "trade:view" not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )

        # Get active trades from execution engine
        execution_engine_client = request.app.state.execution_engine_client
        active_trades = await execution_engine_client.get_active_trades(
            user=current_user["username"]
        )

        return {
            "status": "success",
            "trades": active_trades,
            "count": len(active_trades)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting active trades: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve active trades"
        )

@app.post("/api/v1/trading/close/{trade_id}")
async def close_trade(trade_id: str, request: Request, current_user: Dict = Depends(get_current_user)):
    """Close an active trade."""
    try:
        # Check permissions
        if "trade:execute" not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )

        # Close trade via execution engine
        execution_engine_client = request.app.state.execution_engine_client
        result = await execution_engine_client.close_trade(
            trade_id=trade_id,
            user=current_user["username"]
        )

        # Log the action
        logger.info(f"Trade {trade_id} closed by {current_user['username']}")

        # Record metrics
        metrics.increment('trades_closed', tags={'user': current_user['username']}) # type: ignore

        # Broadcast the trade closure to subscribed clients
        await broadcast_to_topic("trade_updates", {
            "event": "trade_closed",
            "trade_id": trade_id,
            "user": current_user["username"]
        })

        return {
            "status": "success",
            "message": "Trade closed successfully",
            "result": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing trade {trade_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to close trade {trade_id}"
        )

# Auto-trading configuration
@app.post("/api/v1/autotrading/configure")
async def configure_autotrading(request: Request, current_user: Dict = Depends(get_current_user)):
    """Configure auto-trading settings."""
    if redis_client is None:
        logger.error("Redis client not available for autotrading configuration.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Autotrading service unavailable.")
    try:
        # Check permissions
        if "autotrade:configure" not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )

        # Parse request body
        data = await request.json()

        # Validate required fields
        required_fields = ["platform", "assets", "risk_level"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required field: {field}"
                )

        # Validate platform
        if data["platform"] not in ["binance", "deriv"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid platform. Must be 'binance' or 'deriv'"
            )

        # Validate risk level
        if not (1 <= data["risk_level"] <= 10):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Risk level must be between 1 and 10"
            )

        # Validate assets format
        if not isinstance(data["assets"], list) or len(data["assets"]) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Assets must be a non-empty list"
            )

        # Store configuration in Redis
        config_key = f"autotrading:config:{current_user['username']}"
        config_data_payload = {
            "platform": data["platform"],
            "assets": data["assets"],
            "risk_level": data["risk_level"],
            "updated_at": datetime.utcnow().isoformat(),
            "updated_by": current_user["username"]
        }

        # Add optional fields
        optional_fields = ["max_trades", "max_drawdown", "strategy_preference"]
        for field in optional_fields:
            if field in data:
                config_data_payload[field] = data[field]

        await redis_client.set(config_key, json.dumps(config_data_payload))

        # Notify brain council of configuration changes
        brain_council_client = request.app.state.brain_council_client
        await brain_council_client.update_autotrading_config(
            user=current_user["username"],
            platform=data["platform"],
            assets=data["assets"],
            risk_level=data["risk_level"],
            **{k: data[k] for k in optional_fields if k in data}
        )

        # Log the action
        logger.info(f"Auto-trading configured by {current_user['username']} for {data['platform']}")

        return {
            "status": "success",
            "message": "Auto-trading configuration updated successfully",
            "config": config_data_payload
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring auto-trading: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to configure auto-trading"
        )

@app.post("/api/v1/autotrading/toggle")
async def toggle_autotrading(request: Request, current_user: Dict = Depends(get_current_user)):
    """Enable or disable auto-trading."""
    if redis_client is None:
        logger.error("Redis client not available for toggling autotrading.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Autotrading service unavailable.")
    try:
        # Check permissions
        if "autotrade:toggle" not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )

        # Parse request body
        data = await request.json()

        # Validate required fields
        if "enabled" not in data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required field: enabled"
            )

        enabled = bool(data["enabled"])

        # Check if configuration exists
        config_key = f"autotrading:config:{current_user['username']}"
        config_data_str = await redis_client.get(config_key)

        if not config_data_str and enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Auto-trading must be configured before enabling"
            )
        
        config_data = json.loads(config_data_str) if config_data_str else {}


        # Update enabled status
        status_key = f"autotrading:status:{current_user['username']}"
        status_data_payload = {
            "enabled": enabled,
            "updated_at": datetime.utcnow().isoformat(),
        }

        await redis_client.set(status_key, json.dumps(status_data_payload))

        # Notify brain council and execution engine
        brain_council_client = request.app.state.brain_council_client
        execution_engine_client = request.app.state.execution_engine_client

        if enabled:
            await brain_council_client.enable_autotrading(
                user=current_user["username"],
                platform=config_data.get("platform"), # Use .get for safety
                assets=config_data.get("assets")
            )
            await execution_engine_client.enable_autotrading(
                user=current_user["username"],
                platform=config_data.get("platform")
            )
        else:
            await brain_council_client.disable_autotrading(
                user=current_user["username"]
            )
            await execution_engine_client.disable_autotrading(
                user=current_user["username"]
            )

        # Log the action
        action = "enabled" if enabled else "disabled"
        logger.info(f"Auto-trading {action} by {current_user['username']}")

        # Record metrics
        metrics.increment(f'autotrading_{action}', tags={'user': current_user['username']}) # type: ignore

        # Broadcast the status change to subscribed clients
        await broadcast_to_topic("autotrading_updates", {
            "event": "autotrading_status_changed",
            "enabled": enabled,
            "user": current_user["username"]
        })

        return {
            "status": "success",
            "message": f"Auto-trading {action} successfully",
            "enabled": enabled
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling auto-trading: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to toggle auto-trading"
        )

# Data endpoints
@app.get("/api/v1/market-data/{platform}/{asset}")
async def get_market_data(
    platform: str,
    asset: str,
    timeframe: str = "1h",
    limit: int = 1000,
    request: Request = Request, # Provide default for Request if it's not always passed by FastAPI
    current_user: Dict = Depends(get_current_user)
):
    """Get historical market data for a specific asset."""
    try:
        # Check permissions
        if "data:view" not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )

        # Validate parameters
        valid_platforms = ["binance", "deriv"]
        if platform not in valid_platforms:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid platform. Must be one of: {', '.join(valid_platforms)}"
            )

        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        if timeframe not in valid_timeframes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}"
            )

        # Get data from data ingest service
        data_ingest_client = request.app.state.data_ingest_client
        market_data = await data_ingest_client.get_historical_data(
            platform=platform,
            asset=asset,
            timeframe=timeframe,
            limit=limit
        )

        # Record metrics
        metrics.increment('market_data_requests', tags={ # type: ignore
            'platform': platform,
            'asset': asset,
            'timeframe': timeframe
        })

        return {
            "status": "success",
            "platform": platform,
            "asset": asset,
            "timeframe": timeframe,
            "count": len(market_data),
            "data": market_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting market data: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve market data"
        )

@app.get("/api/v1/pattern-analysis/{platform}/{asset}")
async def get_pattern_analysis(
    platform: str,
    asset: str,
    timeframe: str = "1h",
    request: Request = Request, # Provide default
    current_user: Dict = Depends(get_current_user)
):
    """Get pattern analysis for a specific asset."""
    try:
        # Check permissions
        if "analysis:view" not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )

        # Validate parameters
        valid_platforms = ["binance", "deriv"]
        if platform not in valid_platforms:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid platform. Must be one of: {', '.join(valid_platforms)}"
            )

        # Get pattern analysis from intelligence service
        intelligence_client = request.app.state.intelligence_client
        patterns = await intelligence_client.analyze_patterns(
            platform=platform,
            asset=asset,
            timeframe=timeframe
        )

        # Record metrics
        metrics.increment('pattern_analysis_requests', tags={ # type: ignore
            'platform': platform,
            'asset': asset,
            'timeframe': timeframe
        })

        return {
            "status": "success",
            "platform": platform,
            "asset": asset,
            "timeframe": timeframe,
            "patterns": patterns
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting pattern analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve pattern analysis"
        )

# Brain insights endpoints
@app.get("/api/v1/insights/{platform}/{asset}")
async def get_brain_insights(
    platform: str,
    asset: str,
    request: Request = Request, # Provide default
    current_user: Dict = Depends(get_current_user)
):
    """Get brain insights for a specific asset."""
    try:
        # Check permissions
        if "insights:view" not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )

        # Get insights from brain council
        brain_council_client = request.app.state.brain_council_client
        insights = await brain_council_client.get_insights(
            platform=platform,
            asset=asset
        )

        # Record metrics
        metrics.increment('insight_requests', tags={ # type: ignore
            'platform': platform,
            'asset': asset
        })

        return {
            "status": "success",
            "platform": platform,
            "asset": asset,
            "insights": insights
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting brain insights: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve brain insights"
        )

# Voice advisor settings
@app.post("/api/v1/voice-advisor/settings")
async def update_voice_advisor_settings(request: Request, current_user: Dict = Depends(get_current_user)):
    """Update voice advisor settings."""
    if redis_client is None:
        logger.error("Redis client not available for voice advisor settings.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Voice advisor service unavailable.")
    try:
        # Check permissions
        if "settings:edit" not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )

        # Parse request body
        data = await request.json()

        # Validate required fields
        required_fields = ["enabled", "voice_type", "verbosity_level", "notification_types"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required field: {field}"
                )

        # Validate voice type
        valid_voices = ["male", "female", "neutral", "system"]
        if data["voice_type"] not in valid_voices:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid voice type. Must be one of: {', '.join(valid_voices)}"
            )

        # Validate verbosity level
        if not (1 <= data["verbosity_level"] <= 5):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Verbosity level must be between 1 and 5"
            )

        # Store settings in Redis
        settings_key = f"voice_advisor:settings:{current_user['username']}"
        settings_data_payload = {
            "enabled": data["enabled"],
            "voice_type": data["voice_type"],
            "verbosity_level": data["verbosity_level"],
            "notification_types": data["notification_types"],
            "updated_at": datetime.utcnow().isoformat()
        }

        # Add optional fields
        optional_fields = ["language", "volume", "speed", "quiet_hours"]
        for field in optional_fields:
            if field in data:
                settings_data_payload[field] = data[field]

        await redis_client.set(settings_key, json.dumps(settings_data_payload))

        # Notify brain council of voice advisor settings
        brain_council_client = request.app.state.brain_council_client
        await brain_council_client.update_voice_advisor_settings(
            user=current_user["username"],
            settings=settings_data_payload
        )

        # Log the action
        logger.info(f"Voice advisor settings updated by {current_user['username']}")

        return {
            "status": "success",
            "message": "Voice advisor settings updated successfully",
            "settings": settings_data_payload
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating voice advisor settings: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update voice advisor settings"
        )

# Backtesting endpoints
@app.post("/api/v1/backtest")
async def run_backtest(request: Request, current_user: Dict = Depends(get_current_user)):
    """Run a backtest with specified parameters."""
    try:
        # Check permissions
        if "backtest:run" not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )

        # Parse request body
        data = await request.json()

        # Validate required fields
        required_fields = ["platform", "asset", "timeframe", "strategy", "start_date", "end_date"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required field: {field}"
                )

        # Submit backtest to backtester service
        backtester_client = request.app.state.backtester_client
        backtest_id = await backtester_client.submit_backtest(
            user=current_user["username"],
            platform=data["platform"],
            asset=data["asset"],
            timeframe=data["timeframe"],
            strategy=data["strategy"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            initial_capital=data.get("initial_capital", 1000),
            parameters=data.get("parameters", {})
        )

        # Log the action
        logger.info(f"Backtest submitted by {current_user['username']} for {data['platform']} {data['asset']}")

        # Record metrics
        metrics.increment('backtest_submissions', tags={ # type: ignore
            'user': current_user['username'],
            'platform': data['platform'],
            'asset': data['asset'],
            'strategy': data['strategy']
        })

        return {
            "status": "success",
            "message": "Backtest submitted successfully",
            "backtest_id": backtest_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting backtest: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit backtest"
        )

@app.get("/api/v1/backtest/{backtest_id}")
async def get_backtest_results(
    backtest_id: str,
    request: Request = Request, # Provide default
    current_user: Dict = Depends(get_current_user)
):
    """Get results of a specific backtest."""
    try:
        # Check permissions
        if "backtest:view" not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied"
            )

        # Get backtest results from backtester service
        backtester_client = request.app.state.backtester_client
        results = await backtester_client.get_backtest_results(
            backtest_id=backtest_id,
            user=current_user["username"]
        )

        # Check if results exist
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Backtest with ID {backtest_id} not found"
            )

        # Check if backtest belongs to user (assuming 'user' field in results)
        if results.get("user") != current_user["username"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this backtest"
            )

        return {
            "status": "success",
            "backtest_id": backtest_id,
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting backtest results for {backtest_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve backtest results"
        )


# Run the application if executed directly
if __name__ == "__main__":
    # Get configuration from environment or use defaults
    # This assumes that the Config object is already initialized if this script is run directly,
    # or that these environment variables are sufficient for uvicorn.
    # For a full application, config loading should happen before this.
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8000))
    reload_flag = os.environ.get("API_RELOAD", "False").lower() == "true" # Adjusted for FastAPI/Uvicorn
    log_level_str = os.environ.get("API_LOG_LEVEL", "info")

    uvicorn.run(
        "api_gateway.app:app", # Points to the 'app' instance in 'api_gateway.app' module
        host=host,
        port=port,
        reload=reload_flag,
        log_level=log_level_str.lower()
    )
