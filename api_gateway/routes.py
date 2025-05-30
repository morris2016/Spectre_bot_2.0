#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
API Gateway - Routes

This module defines all API routes and endpoints for the QuantumSpectre Elite Trading System.
It handles request routing, validation, and response formatting.
"""

import time
import json
import asyncio
from typing import Dict, List, Any, Optional, Union

# Import route modules
from .routes import (
    auth, users, market_data, trading, backtesting,
    system, strategy, monitoring, ml_models, brain_council
)
from fastapi import APIRouter, FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect, Request, status, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from .authentication import (
    get_current_user, 
    CredentialsSchema, 
    JWTBearer, 
    UserModel, 
    TokenResponse
)
from common.logger import get_logger
from common.exceptions import ValidationError, OperationNotPermittedError
from config import Config, save_config

# Create logger
logger = get_logger(__name__)

# Database configuration model
class DatabaseConfigModel(BaseModel):
    host: str = Field(..., description="Database host")
    port: int = Field(..., description="Database port")
    user: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")
    dbname: str = Field(..., description="Database name")
    min_pool_size: int = Field(5, description="Minimum connection pool size")
    max_pool_size: int = Field(20, description="Maximum connection pool size")
    connection_timeout: int = Field(60, description="Connection timeout in seconds")
    command_timeout: int = Field(60, description="Command timeout in seconds")

# Create routers
main_router = APIRouter()
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])
user_router = APIRouter(prefix="/user", tags=["User Management"])
system_router = APIRouter(prefix="/system", tags=["System Management"])
market_router = APIRouter(prefix="/market", tags=["Market Data"])
trading_router = APIRouter(prefix="/trading", tags=["Trading"])
strategy_router = APIRouter(prefix="/strategy", tags=["Strategy Management"])
brain_router = APIRouter(prefix="/brain", tags=["Brain Management"])
backtest_router = APIRouter(prefix="/backtest", tags=["Backtesting"])
monitoring_router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str, channel: str):
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = []
        self.active_connections[channel].append(websocket)
        logger.info(f"Client {client_id} connected to {channel}. Active connections: {sum(len(conns) for conns in self.active_connections.values())}")
    
    def disconnect(self, websocket: WebSocket, channel: str):
        if channel in self.active_connections:
            if websocket in self.active_connections[channel]:
                self.active_connections[channel].remove(websocket)
                logger.info(f"Client disconnected from {channel}. Remaining connections: {sum(len(conns) for conns in self.active_connections.values())}")
    
    async def broadcast(self, message: Dict[str, Any], channel: str):
        if channel in self.active_connections:
            dead_connections = []
            for connection in self.active_connections[channel]:
                try:
                    await connection.send_json(message)
                except RuntimeError:
                    dead_connections.append(connection)
            
            # Remove dead connections
            for dead in dead_connections:
                self.disconnect(dead, channel)

manager = ConnectionManager()

# Request models
class PlatformSelection(BaseModel):
    platform: str = Field(..., description="Trading platform (binance or deriv)")
    
    @validator('platform')
    def validate_platform(cls, v):
        if v not in ['binance', 'deriv']:
            raise ValueError('Platform must be either "binance" or "deriv"')
        return v

class AssetSelection(BaseModel):
    platform: str = Field(..., description="Trading platform (binance or deriv)")
    asset: str = Field(..., description="Asset symbol to trade")

class TradingParams(BaseModel):
    platform: str = Field(..., description="Trading platform (binance or deriv)")
    asset: str = Field(..., description="Asset symbol to trade")
    timeframe: str = Field(..., description="Trading timeframe")
    trade_mode: str = Field(..., description="Trading mode (auto or manual)")
    risk_level: float = Field(..., ge=0, le=10, description="Risk level (0-10)")
    
    @validator('platform')
    def validate_platform(cls, v):
        if v not in ['binance', 'deriv']:
            raise ValueError('Platform must be either "binance" or "deriv"')
        return v
    
    @validator('trade_mode')
    def validate_trade_mode(cls, v):
        if v not in ['auto', 'manual']:
            raise ValueError('Trade mode must be either "auto" or "manual"')
        return v

class OrderRequest(BaseModel):
    platform: str = Field(..., description="Trading platform (binance or deriv)")
    asset: str = Field(..., description="Asset symbol to trade")
    side: str = Field(..., description="Order side (buy or sell)")
    order_type: str = Field(..., description="Order type")
    quantity: float = Field(..., gt=0, description="Order quantity")
    price: Optional[float] = Field(None, description="Order price (for limit orders)")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    
    @validator('side')
    def validate_side(cls, v):
        if v not in ['buy', 'sell']:
            raise ValueError('Side must be either "buy" or "sell"')
        return v
    
    @validator('order_type')
    def validate_order_type(cls, v):
        valid_types = ['market', 'limit', 'stop', 'stop_limit']
        if v not in valid_types:
            raise ValueError(f'Order type must be one of {", ".join(valid_types)}')
        return v

class StrategyConfig(BaseModel):
    strategy_id: str = Field(..., description="Strategy identifier")
    parameters: Dict[str, Any] = Field(..., description="Strategy parameters")
    enabled: bool = Field(True, description="Whether the strategy is enabled")

# Authentication routes
@auth_router.post("/login", response_model=TokenResponse)
async def login(credentials: CredentialsSchema):
    """
    Authenticate user and return JWT token
    """
    try:
        return await get_current_user.login(credentials.username, credentials.password)
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

@auth_router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: Request):
    """
    Refresh JWT token
    """
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = auth_header.split(" ")[1]
        return await get_current_user.refresh(token)
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

# User routes
@user_router.get("/profile", dependencies=[Depends(JWTBearer())])
async def get_user_profile(request: Request):
    """
    Get current user profile
    """
    try:
        user = await get_current_user(request)
        return {
            "username": user.username,
            "email": user.email,
            "created_at": user.created_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "preferences": user.preferences
        }
    except Exception as e:
        logger.error(f"Get profile error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )

@user_router.put("/preferences", dependencies=[Depends(JWTBearer())])
async def update_preferences(request: Request, preferences: Dict[str, Any]):
    """
    Update user preferences
    """
    try:
        user = await get_current_user(request)
        # Update user preferences in database
        # This is a placeholder - actual implementation would update the database
        return {"status": "success", "message": "Preferences updated successfully"}
    except Exception as e:
        logger.error(f"Update preferences error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update preferences"
        )

# System routes
@system_router.get("/status", dependencies=[Depends(JWTBearer())])
async def get_system_status():
    """
    Get current system status
    """
    # This would call into other services to gather status information
    return {
        "status": "operational",
        "services": {
            "data_ingest": {"status": "healthy", "uptime": "10h 23m"},
            "feature_service": {"status": "healthy", "uptime": "10h 20m"},
            "intelligence": {"status": "healthy", "uptime": "10h 19m"},
            "ml_models": {"status": "healthy", "uptime": "10h 18m"},
            "strategy_brains": {"status": "healthy", "uptime": "10h 17m"},
            "brain_council": {"status": "healthy", "uptime": "10h 16m"},
            "execution_engine": {"status": "healthy", "uptime": "10h 15m"},
            "risk_manager": {"status": "healthy", "uptime": "10h 14m"},
            "monitoring": {"status": "healthy", "uptime": "10h 13m"}
        },
        "version": "1.0.0",
        "uptime": "10h 25m",
        "last_update": "2023-07-20T15:30:45Z"
    }

@system_router.get("/config", dependencies=[Depends(JWTBearer())])
async def get_system_config():
    """
    Get system configuration
    """
    # Return sanitized configuration (no sensitive values)
    config = Config.get_public_config()
    return config

@system_router.post("/database/config", dependencies=[Depends(JWTBearer())])
async def update_database_config(db_config: DatabaseConfigModel, request: Request):
    """
    Update database configuration
    """
    try:
        user = await get_current_user(request)
        
        # Check if user has admin permissions
        if "admin" not in user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to update database configuration"
            )
        
        # Update config.yml with new database settings
        config = Config()
        config.database = db_config.dict()
        save_config(config)
        
        # Restart database connection
        from common.db_client import get_db_client
        
        try:
            # Close existing connection if any
            from main import db_client as current_db_client
            if current_db_client:
                await current_db_client.close()
            
            # Create new connection
            db_config_dict = db_config.dict()
            new_db_client = await get_db_client(
                db_type=db_config_dict.get("type", "postgresql"),
                host=db_config_dict.get("host", "localhost"),
                port=db_config_dict.get("port", 5432),
                username=db_config_dict.get("user", "postgres"),
                password=db_config_dict.get("password", ""),
                database=db_config_dict.get("dbname", "quantumspectre"),
                pool_size=db_config_dict.get("min_pool_size", 10),
                ssl=False,
                timeout=db_config_dict.get("connection_timeout", 30)
            )
            
            # Update global db_client
            import main
            main.db_client = new_db_client
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            
        if db_client is None:
            return {
                "status": "error",
                "message": "Failed to connect to database with new configuration"
            }
        
        return {
            "status": "success",
            "message": "Database configuration updated successfully"
        }
    except Exception as e:
        logger.error(f"Error updating database configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update database configuration: {str(e)}"
        )
@system_router.post("/restart", dependencies=[Depends(JWTBearer())])
async def restart_system():
    """
    Restart the system or specific components
    """
    # This would actually trigger a restart process
    return {"status": "success", "message": "System restart initiated"}

# Market data routes
@market_router.post("/select-platform")
async def select_trading_platform(platform_data: PlatformSelection, request: Request):
    """
    Select trading platform and get available assets
    """
    try:
        # This would retrieve available assets for the selected platform
        if platform_data.platform == "binance":
            assets = [
                {"symbol": "BTCUSDT", "name": "Bitcoin", "type": "crypto"},
                {"symbol": "ETHUSDT", "name": "Ethereum", "type": "crypto"},
                {"symbol": "BNBUSDT", "name": "Binance Coin", "type": "crypto"},
                {"symbol": "ADAUSDT", "name": "Cardano", "type": "crypto"},
                {"symbol": "DOGEUSDT", "name": "Dogecoin", "type": "crypto"}
            ]
        else:  # deriv
            assets = [
                {"symbol": "R_10", "name": "Volatility 10 Index", "type": "synthetic"},
                {"symbol": "R_25", "name": "Volatility 25 Index", "type": "synthetic"},
                {"symbol": "R_50", "name": "Volatility 50 Index", "type": "synthetic"},
                {"symbol": "R_75", "name": "Volatility 75 Index", "type": "synthetic"},
                {"symbol": "R_100", "name": "Volatility 100 Index", "type": "synthetic"}
            ]
        
        return {"platform": platform_data.platform, "assets": assets}
    except Exception as e:
        logger.error(f"Platform selection error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve assets for selected platform"
        )

@market_router.get("/ohlc/{platform}/{asset}/{timeframe}")
async def get_ohlc_data(platform: str, asset: str, timeframe: str, limit: int = 100):
    """
    Get OHLC data for a specific asset
    """
    try:
        # Validate platform
        if platform not in ["binance", "deriv"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Platform must be 'binance' or 'deriv'"
            )
        
        # Validate timeframe
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        if timeframe not in valid_timeframes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Timeframe must be one of: {', '.join(valid_timeframes)}"
            )
        
        # This would fetch the actual OHLC data from the data service
        # For now, we'll return mock data
        candles = []
        current_time = int(time.time() * 1000)
        interval_ms = {
            "1m": 60000,
            "5m": 300000,
            "15m": 900000,
            "30m": 1800000,
            "1h": 3600000,
            "4h": 14400000,
            "1d": 86400000,
            "1w": 604800000
        }
        
        interval = interval_ms[timeframe]
        
        # Generate mock data
        import random
        base_price = 100.0 if platform == "deriv" else 50000.0
        price = base_price
        for i in range(limit):
            open_price = price
            high_price = open_price * (1 + random.uniform(0, 0.02))
            low_price = open_price * (1 - random.uniform(0, 0.02))
            close_price = random.uniform(low_price, high_price)
            volume = random.uniform(1, 100)
            
            candles.insert(0, {
                "timestamp": current_time - (interval * (limit - i)),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            })
            
            price = close_price
        
        return {
            "platform": platform,
            "asset": asset,
            "timeframe": timeframe,
            "data": candles
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OHLC data error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve OHLC data"
        )

@market_router.get("/depth/{platform}/{asset}")
async def get_order_book(platform: str, asset: str, limit: int = 10):
    """
    Get order book data for a specific asset
    """
    try:
        # Validate platform
        if platform not in ["binance", "deriv"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Platform must be 'binance' or 'deriv'"
            )
        
        # This would fetch the actual order book data from the market data service
        # For now, we'll return mock data
        import random
        
        base_price = 100.0 if platform == "deriv" else 50000.0
        
        bids = []
        asks = []
        
        for i in range(limit):
            bid_price = base_price * (1 - (i * 0.001))
            ask_price = base_price * (1 + (i * 0.001))
            bid_quantity = random.uniform(0.1, 10.0)
            ask_quantity = random.uniform(0.1, 10.0)
            
            bids.append([bid_price, bid_quantity])
            asks.append([ask_price, ask_quantity])
        
        return {
            "platform": platform,
            "asset": asset,
            "bids": bids,
            "asks": asks,
            "timestamp": int(time.time() * 1000)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Order book error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve order book data"
        )

# Trading routes
@trading_router.post("/setup", dependencies=[Depends(JWTBearer())])
async def setup_trading(params: TradingParams, request: Request):
    """
    Set up trading for a specific asset
    """
    try:
        user = await get_current_user(request)
        # This would actually configure the trading system for the selected asset
        # and retrieve initial state information
        
        return {
            "status": "success",
            "message": f"Trading setup for {params.asset} on {params.platform}",
            "config": {
                "asset": params.asset,
                "platform": params.platform,
                "timeframe": params.timeframe,
                "trade_mode": params.trade_mode,
                "risk_level": params.risk_level,
                "active_brains": [
                    {"id": "pattern_brain", "name": "Pattern Brain", "confidence": 0.85},
                    {"id": "trend_brain", "name": "Trend Brain", "confidence": 0.78},
                    {"id": "volatility_brain", "name": "Volatility Brain", "confidence": 0.72}
                ],
                "status": "ready"
            }
        }
    except Exception as e:
        logger.error(f"Trading setup error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to set up trading"
        )

@trading_router.post("/orders", dependencies=[Depends(JWTBearer())])
async def create_order(order: OrderRequest, request: Request):
    """
    Create a new trading order
    """
    try:
        user = await get_current_user(request)
        # This would actually create a new order via the execution engine
        
        order_id = f"ORD-{int(time.time())}-{hash(order.asset) % 10000}"
        
        return {
            "status": "success",
            "message": f"Order created successfully",
            "order": {
                "id": order_id,
                "platform": order.platform,
                "asset": order.asset,
                "side": order.side,
                "type": order.order_type,
                "quantity": order.quantity,
                "price": order.price,
                "stop_loss": order.stop_loss,
                "take_profit": order.take_profit,
                "status": "pending",
                "created_at": int(time.time() * 1000)
            }
        }
    except Exception as e:
        logger.error(f"Order creation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create order"
        )

@trading_router.get("/orders", dependencies=[Depends(JWTBearer())])
async def get_orders(
    request: Request,
    platform: Optional[str] = None,
    asset: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    Get list of orders with optional filtering
    """
    try:
        user = await get_current_user(request)
        # This would fetch orders from the database with the specified filters
        
        # Mock data
        orders = []
        for i in range(limit):
            order_id = f"ORD-{int(time.time())}-{(i + offset) % 10000}"
            created_at = int(time.time() * 1000) - (i + offset) * 60000
            
            # Skip if platform filter is set and doesn't match
            if platform and (i % 2 == 0 and platform != "binance" or i % 2 == 1 and platform != "deriv"):
                continue
                
            # Skip if status filter is set and doesn't match
            if status:
                statuses = ["filled", "pending", "cancelled"]
                if status != statuses[i % 3]:
                    continue
            
            side = "buy" if i % 2 == 0 else "sell"
            
            order = {
                "id": order_id,
                "platform": "binance" if i % 2 == 0 else "deriv",
                "asset": "BTCUSDT" if i % 5 == 0 else "ETHUSDT" if i % 5 == 1 else "R_10" if i % 5 == 2 else "R_25" if i % 5 == 3 else "R_50",
                "side": side,
                "type": "market" if i % 3 == 0 else "limit" if i % 3 == 1 else "stop",
                "quantity": 0.01 * (i + 1),
                "price": 50000 - (i * 100) if i % 2 == 0 else 100 - (i * 0.2),
                "status": "filled" if i % 3 == 0 else "pending" if i % 3 == 1 else "cancelled",
                "created_at": created_at,
                "updated_at": created_at + 30000 if i % 3 == 0 else None,
                "pnl": 125.50 if i % 3 == 0 and side == "buy" else -75.25 if i % 3 == 0 and side == "sell" else None
            }
            
            # Skip if asset filter is set and doesn't match
            if asset and order["asset"] != asset:
                continue
                
            orders.append(order)
        
        return {
            "status": "success",
            "orders": orders,
            "total": 250,  # Total number of orders matching filter (for pagination)
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Get orders error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve orders"
        )

@trading_router.get("/orders/{order_id}", dependencies=[Depends(JWTBearer())])
async def get_order(order_id: str, request: Request):
    """
    Get details of a specific order
    """
    try:
        user = await get_current_user(request)
        # This would fetch the order details from the database
        
        # Mock data
        created_at = int(time.time() * 1000) - 3600000
        platform = "binance" if int(order_id.split("-")[1]) % 2 == 0 else "deriv"
        asset = "BTCUSDT" if platform == "binance" else "R_10"
        side = "buy" if int(order_id.split("-")[2]) % 2 == 0 else "sell"
        status = "filled"
        
        order = {
            "id": order_id,
            "platform": platform,
            "asset": asset,
            "side": side,
            "type": "market",
            "quantity": 0.01,
            "price": 50000 if platform == "binance" else 100,
            "status": status,
            "created_at": created_at,
            "updated_at": created_at + 30000,
            "pnl": 125.50 if side == "buy" else -75.25,
            "trade_history": [
                {"time": created_at, "status": "created", "message": "Order created"},
                {"time": created_at + 1000, "status": "pending", "message": "Order submitted to exchange"},
                {"time": created_at + 5000, "status": "executed", "message": "Order executed at price 50000"},
                {"time": created_at + 30000, "status": "closed", "message": "Position closed with PnL 125.50"}
            ]
        }
        
        return {
            "status": "success",
            "order": order
        }
    except Exception as e:
        logger.error(f"Get order error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve order details"
        )

@trading_router.delete("/orders/{order_id}", dependencies=[Depends(JWTBearer())])
async def cancel_order(order_id: str, request: Request):
    """
    Cancel an open order
    """
    try:
        user = await get_current_user(request)
        # This would actually cancel the order via the execution engine
        
        return {
            "status": "success",
            "message": f"Order {order_id} cancelled successfully"
        }
    except Exception as e:
        logger.error(f"Cancel order error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel order"
        )

@trading_router.get("/performance", dependencies=[Depends(JWTBearer())])
async def get_trading_performance(
    request: Request,
    platform: Optional[str] = None,
    asset: Optional[str] = None,
    timeframe: str = "1d",
    start_time: Optional[int] = None,
    end_time: Optional[int] = None
):
    """
    Get trading performance metrics
    """
    try:
        user = await get_current_user(request)
        # This would fetch performance metrics from the database
        
        # Default to last 30 days if not specified
        current_time = int(time.time() * 1000)
        if not end_time:
            end_time = current_time
        if not start_time:
            start_time = end_time - (30 * 24 * 60 * 60 * 1000)  # 30 days in milliseconds
        
        # Calculate daily performance for the requested period
        import random
        from datetime import datetime, timedelta
        
        days = (end_time - start_time) // (24 * 60 * 60 * 1000)
        daily_data = []
        
        account_value = 1000.0  # Starting account value
        
        for i in range(days):
            day_timestamp = start_time + (i * 24 * 60 * 60 * 1000)
            day_date = datetime.fromtimestamp(day_timestamp / 1000).strftime('%Y-%m-%d')
            
            # Generate random performance data
            trades_count = random.randint(5, 20)
            win_count = int(trades_count * random.uniform(0.6, 0.9))  # 60-90% win rate
            lose_count = trades_count - win_count
            
            avg_win = random.uniform(15, 30)
            avg_loss = random.uniform(10, 20)
            
            day_pnl = (win_count * avg_win) - (lose_count * avg_loss)
            account_value += day_pnl
            
            daily_data.append({
                "date": day_date,
                "timestamp": day_timestamp,
                "trades": trades_count,
                "wins": win_count,
                "losses": lose_count,
                "win_rate": win_count / trades_count if trades_count > 0 else 0,
                "pnl": day_pnl,
                "account_value": account_value
            })
        
        # Calculate overall metrics
        total_trades = sum(day["trades"] for day in daily_data)
        total_wins = sum(day["wins"] for day in daily_data)
        total_losses = sum(day["losses"] for day in daily_data)
        total_pnl = sum(day["pnl"] for day in daily_data)
        
        # Calculate max drawdown
        max_account = account_value
        max_drawdown = 0
        max_drawdown_pct = 0
        
        prev_max = 1000.0  # Starting value
        for day in daily_data:
            if day["account_value"] > prev_max:
                prev_max = day["account_value"]
            drawdown = prev_max - day["account_value"]
            drawdown_pct = drawdown / prev_max if prev_max > 0 else 0
            
            if drawdown_pct > max_drawdown_pct:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct
        
        return {
            "status": "success",
            "summary": {
                "start_date": datetime.fromtimestamp(start_time / 1000).strftime('%Y-%m-%d'),
                "end_date": datetime.fromtimestamp(end_time / 1000).strftime('%Y-%m-%d'),
                "total_trades": total_trades,
                "win_rate": total_wins / total_trades if total_trades > 0 else 0,
                "profit_factor": sum(day["wins"] * day["pnl"] / day["trades"] for day in daily_data if day["trades"] > 0) / 
                               abs(sum(day["losses"] * day["pnl"] / day["trades"] for day in daily_data if day["trades"] > 0)) 
                               if sum(day["losses"] * day["pnl"] / day["trades"] for day in daily_data if day["trades"] > 0) != 0 else 0,
                "total_pnl": total_pnl,
                "pnl_percentage": (total_pnl / 1000) * 100,  # Percentage of initial account value
                "max_drawdown": max_drawdown,
                "max_drawdown_percentage": max_drawdown_pct * 100
            },
            "daily_data": daily_data
        }
    except Exception as e:
        logger.error(f"Performance data error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance metrics"
        )

# Strategy routes
@strategy_router.get("/list", dependencies=[Depends(JWTBearer())])
async def list_strategies():
    """
    List available strategies
    """
    try:
        # This would fetch the list of available strategies from the strategy service
        
        strategies = [
            {
                "id": "momentum_brain",
                "name": "Momentum Brain",
                "description": "Trades based on price momentum indicators",
                "parameters": [
                    {"name": "rsi_period", "type": "int", "default": 14, "min": 2, "max": 30},
                    {"name": "rsi_overbought", "type": "float", "default": 70, "min": 50, "max": 90},
                    {"name": "rsi_oversold", "type": "float", "default": 30, "min": 10, "max": 50},
                    {"name": "ma_fast_period", "type": "int", "default": 12, "min": 5, "max": 30},
                    {"name": "ma_slow_period", "type": "int", "default": 26, "min": 10, "max": 50}
                ],
                "performance": {"win_rate": 0.78, "profit_factor": 1.85}
            },
            {
                "id": "mean_reversion_brain",
                "name": "Mean Reversion Brain",
                "description": "Trades based on price reversion to the mean",
                "parameters": [
                    {"name": "bollinger_period", "type": "int", "default": 20, "min": 10, "max": 50},
                    {"name": "bollinger_stddev", "type": "float", "default": 2.0, "min": 1.0, "max": 4.0},
                    {"name": "mean_period", "type": "int", "default": 50, "min": 20, "max": 200}
                ],
                "performance": {"win_rate": 0.75, "profit_factor": 1.65}
            },
            {
                "id": "breakout_brain",
                "name": "Breakout Brain",
                "description": "Identifies and trades breakouts from ranges",
                "parameters": [
                    {"name": "atr_period", "type": "int", "default": 14, "min": 5, "max": 30},
                    {"name": "breakout_threshold", "type": "float", "default": 1.5, "min": 0.5, "max": 3.0},
                    {"name": "consolidation_bars", "type": "int", "default": 5, "min": 3, "max": 20}
                ],
                "performance": {"win_rate": 0.72, "profit_factor": 1.90}
            }
        ]
        
        return {
            "status": "success",
            "strategies": strategies
        }
    except Exception as e:
        logger.error(f"Strategy list error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve strategies"
        )

@strategy_router.get("/{strategy_id}", dependencies=[Depends(JWTBearer())])
async def get_strategy(strategy_id: str):
    """
    Get details of a specific strategy
    """
    try:
        # This would fetch the details of the specified strategy
        
        strategies = {
            "momentum_brain": {
                "id": "momentum_brain",
                "name": "Momentum Brain",
                "description": "Trades based on price momentum indicators",
                "parameters": [
                    {"name": "rsi_period", "type": "int", "default": 14, "min": 2, "max": 30},
                    {"name": "rsi_overbought", "type": "float", "default": 70, "min": 50, "max": 90},
                    {"name": "rsi_oversold", "type": "float", "default": 30, "min": 10, "max": 50},
                    {"name": "ma_fast_period", "type": "int", "default": 12, "min": 5, "max": 30},
                    {"name": "ma_slow_period", "type": "int", "default": 26, "min": 10, "max": 50}
                ],
                "performance": {"win_rate": 0.78, "profit_factor": 1.85}
            },
            "mean_reversion_brain": {
                "id": "mean_reversion_brain",
                "name": "Mean Reversion Brain",
                "description": "Trades based on price reversion to the mean",
                "parameters": [
                    {"name": "bollinger_period", "type": "int", "default": 20, "min": 10, "max": 50},
                    {"name": "bollinger_stddev", "type": "float", "default": 2.0, "min": 1.0, "max": 4.0},
                    {"name": "mean_period", "type": "int", "default": 50, "min": 20, "max": 200}
                ],
                "performance": {"win_rate": 0.75, "profit_factor": 1.65}
            }
        }
        
        if strategy_id not in strategies:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Strategy not found"
            )
        
        return {
            "status": "success",
            "strategy": strategies[strategy_id]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get strategy error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve strategy"
        )

@strategy_router.post("/configure", dependencies=[Depends(JWTBearer())])
async def configure_strategy(config: StrategyConfig, request: Request):
    """
    Configure a strategy with specific parameters
    """
    try:
        user = await get_current_user(request)
        # This would configure the specified strategy with the provided parameters
        
        return {
            "status": "success",
            "message": f"Strategy {config.strategy_id} configured successfully",
            "config": {
                "strategy_id": config.strategy_id,
                "parameters": config.parameters,
                "enabled": config.enabled
            }
        }
    except Exception as e:
        logger.error(f"Strategy configuration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to configure strategy"
        )

# Brain management routes
@brain_router.get("/status", dependencies=[Depends(JWTBearer())])
async def get_brain_status(platform: Optional[str] = None, asset: Optional[str] = None):
    """
    Get status of all trading brains or specific ones
    """
    try:
        # This would fetch the status of the trading brains
        
        brains = [
            {
                "id": "momentum_brain",
                "name": "Momentum Brain",
                "platform": "binance",
                "asset": "BTCUSDT",
                "status": "active",
                "confidence": 0.85,
                "last_signal": "buy",
                "last_signal_time": int(time.time() * 1000) - 300000,
                "accuracy": 0.78
            },
            {
                "id": "mean_reversion_brain",
                "name": "Mean Reversion Brain",
                "platform": "binance",
                "asset": "ETHUSDT",
                "status": "active",
                "confidence": 0.72,
                "last_signal": "sell",
                "last_signal_time": int(time.time() * 1000) - 600000,
                "accuracy": 0.75
            },
            {
                "id": "trend_brain",
                "name": "Trend Brain",
                "platform": "deriv",
                "asset": "R_10",
                "status": "active",
                "confidence": 0.90,
                "last_signal": "buy",
                "last_signal_time": int(time.time() * 1000) - 120000,
                "accuracy": 0.82
            }
        ]
        
        # Filter by platform and asset if specified
        if platform:
            brains = [brain for brain in brains if brain["platform"] == platform]
        if asset:
            brains = [brain for brain in brains if brain["asset"] == asset]
        
        return {
            "status": "success",
            "brains": brains
        }
    except Exception as e:
        logger.error(f"Brain status error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve brain status"
        )

@brain_router.post("/activate", dependencies=[Depends(JWTBearer())])
async def activate_brain(request: Request, brain_id: str, platform: str, asset: str):
    """
    Activate a specific brain for trading
    """
    try:
        user = await get_current_user(request)
        # This would activate the specified brain for trading
        
        return {
            "status": "success",
            "message": f"Brain {brain_id} activated for {asset} on {platform}"
        }
    except Exception as e:
        logger.error(f"Brain activation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to activate brain"
        )

@brain_router.post("/deactivate", dependencies=[Depends(JWTBearer())])
async def deactivate_brain(request: Request, brain_id: str, platform: str, asset: str):
    """
    Deactivate a specific brain for trading
    """
    try:
        user = await get_current_user(request)
        # This would deactivate the specified brain for trading
        
        return {
            "status": "success",
            "message": f"Brain {brain_id} deactivated for {asset} on {platform}"
        }
    except Exception as e:
        logger.error(f"Brain deactivation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deactivate brain"
        )

# Backtesting routes
@backtest_router.post("/run", dependencies=[Depends(JWTBearer())])
async def run_backtest(
    request: Request,
    platform: str,
    asset: str,
    strategy_id: str,
    start_time: int,
    end_time: int,
    parameters: Optional[Dict[str, Any]] = None
):
    """
    Run a backtest for a specific strategy
    """
    try:
        user = await get_current_user(request)
        # This would run the backtest with the specified parameters
        
        # Generate a backtest ID
        backtest_id = f"BT-{int(time.time())}-{hash(asset) % 10000}"
        
        # Add the backtest to the queue and return immediately
        return {
            "status": "success",
            "message": "Backtest queued successfully",
            "backtest_id": backtest_id,
            "estimated_completion_time": int(time.time() * 1000) + 300000  # 5 minutes from now
        }
    except Exception as e:
        logger.error(f"Backtest error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue backtest"
        )

@backtest_router.get("/status/{backtest_id}", dependencies=[Depends(JWTBearer())])
async def get_backtest_status(backtest_id: str, request: Request):
    """
    Get status of a backtest
    """
    try:
        user = await get_current_user(request)
        # This would fetch the status of the specified backtest
        
        # Mock data based on the backtest ID
        status_options = ["queued", "running", "completed", "failed"]
        backtest_hash = int(backtest_id.split("-")[1])
        status = status_options[backtest_hash % 4]
        progress = 100 if status == "completed" else 0 if status == "queued" else backtest_hash % 100
        
        return {
            "status": "success",
            "backtest_status": {
                "id": backtest_id,
                "status": status,
                "progress": progress,
                "start_time": int(backtest_id.split("-")[1]) * 1000,
                "estimated_completion_time": (int(backtest_id.split("-")[1]) + 300) * 1000 if status != "completed" else None,
                "completion_time": (int(backtest_id.split("-")[1]) + 300) * 1000 if status == "completed" else None
            }
        }
    except Exception as e:
        logger.error(f"Backtest status error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve backtest status"
        )

@backtest_router.get("/results/{backtest_id}", dependencies=[Depends(JWTBearer())])
async def get_backtest_results(backtest_id: str, request: Request):
    """
    Get results of a completed backtest
    """
    try:
        user = await get_current_user(request)
        # This would fetch the results of the specified backtest
        
        # Check if the backtest is completed
        backtest_hash = int(backtest_id.split("-")[1])
        backtest_status = "completed" if backtest_hash % 4 == 2 else "running" if backtest_hash % 4 == 1 else "queued" if backtest_hash % 4 == 0 else "failed"
        
        if backtest_status != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Backtest is not completed yet. Current status: {backtest_status}"
            )
        
        # Generate mock backtest results
        import random
        
        trades_count = random.randint(50, 200)
        win_rate = random.uniform(0.6, 0.9)
        win_count = int(trades_count * win_rate)
        lose_count = trades_count - win_count
        
        avg_win = random.uniform(15, 30)
        avg_loss = random.uniform(10, 20)
        
        total_pnl = (win_count * avg_win) - (lose_count * avg_loss)
        
        # Generate daily performance data
        start_time = int(backtest_id.split("-")[1]) * 1000 - (30 * 24 * 60 * 60 * 1000)
        end_time = int(backtest_id.split("-")[1]) * 1000
        
        days = (end_time - start_time) // (24 * 60 * 60 * 1000)
        daily_data = []
        
        account_value = 1000.0  # Starting account value
        
        for i in range(days):
            day_timestamp = start_time + (i * 24 * 60 * 60 * 1000)
            
            # Generate random performance data
            day_trades_count = random.randint(0, 10)
            day_win_count = int(day_trades_count * random.uniform(0.6, 0.9))
            day_lose_count = day_trades_count - day_win_count
            
            day_pnl = (day_win_count * avg_win) - (day_lose_count * avg_loss)
            account_value += day_pnl
            
            daily_data.append({
                "date": day_timestamp,
                "trades": day_trades_count,
                "wins": day_win_count,
                "losses": day_lose_count,
                "pnl": day_pnl,
                "account_value": account_value
            })
        
        # Calculate max drawdown
        max_account = max(day["account_value"] for day in daily_data)
        max_drawdown = max_account - min(day["account_value"] for day in daily_data)
        max_drawdown_pct = max_drawdown / max_account if max_account > 0 else 0
        
        return {
            "status": "success",
            "backtest_results": {
                "id": backtest_id,
                "platform": "binance" if backtest_hash % 2 == 0 else "deriv",
                "asset": "BTCUSDT" if backtest_hash % 5 == 0 else "ETHUSDT" if backtest_hash % 5 == 1 else "R_10" if backtest_hash % 5 == 2 else "R_25" if backtest_hash % 5 == 3 else "R_50",
                "strategy_id": "momentum_brain" if backtest_hash % 3 == 0 else "mean_reversion_brain" if backtest_hash % 3 == 1 else "breakout_brain",
                "start_time": start_time,
                "end_time": end_time,
                "parameters": {
                    "param1": 14,
                    "param2": 70,
                    "param3": 30
                },
                "summary": {
                    "total_trades": trades_count,
                    "win_rate": win_rate,
                    "profit_factor": (win_count * avg_win) / (lose_count * avg_loss) if lose_count * avg_loss > 0 else 0,
                    "total_pnl": total_pnl,
                    "pnl_percentage": (total_pnl / 1000) * 100,
                    "max_drawdown": max_drawdown,
                    "max_drawdown_percentage": max_drawdown_pct * 100,
                    "sharpe_ratio": random.uniform(1.0, 3.0)
                },
                "daily_data": daily_data
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest results error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve backtest results"
        )

# WebSocket routes
@main_router.websocket("/ws/{channel}/{client_id}")
async def websocket_endpoint(websocket: WebSocket, channel: str, client_id: str):
    """
    WebSocket endpoint for real-time updates
    """
    await manager.connect(websocket, client_id, channel)
    try:
        while True:
            # Keep the connection alive but don't expect messages from the client
            data = await websocket.receive_text()
            # If the client sends a message, echo it back
            await websocket.send_json({"type": "echo", "message": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket, channel)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket, channel)

# Monitoring WebSocket for updates
async def send_monitoring_updates():
    """
    Send monitoring updates to connected WebSocket clients
    """
    while True:
        await asyncio.sleep(1)  # Send updates every 1 second
        
        # Generate mock monitoring data
        import random
        
        cpu_usage = random.uniform(10, 90)
        memory_usage = random.uniform(20, 80)
        disk_usage = random.uniform(30, 70)
        
        await manager.broadcast(
            {
                "type": "system_stats",
                "timestamp": int(time.time() * 1000),
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "disk_usage": disk_usage
            },
            "monitoring"
        )

# Market data WebSocket for updates
async def send_market_updates():
    """
    Send market data updates to connected WebSocket clients
    """
    while True:
        await asyncio.sleep(0.5)  # Send updates every 0.5 seconds
        
        # Generate mock market data
        import random
        
        # Binance assets
        binance_assets = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT"]
        for asset in binance_assets:
            price = 50000.0 if asset == "BTCUSDT" else 3000.0 if asset == "ETHUSDT" else 400.0 if asset == "BNBUSDT" else 2.0 if asset == "ADAUSDT" else 0.25
            price_change = price * random.uniform(-0.005, 0.005)
            
            await manager.broadcast(
                {
                    "type": "price_update",
                    "platform": "binance",
                    "asset": asset,
                    "timestamp": int(time.time() * 1000),
                    "price": price + price_change,
                    "volume": random.uniform(100, 1000)
                },
                f"market_binance_{asset}"
            )
        
        # Deriv assets
        deriv_assets = ["R_10", "R_25", "R_50", "R_75", "R_100"]
        for asset in deriv_assets:
            price = 100.0 if asset == "R_10" else 250.0 if asset == "R_25" else 500.0 if asset == "R_50" else 750.0 if asset == "R_75" else 1000.0
            price_change = price * random.uniform(-0.01, 0.01)
            
            await manager.broadcast(
                {
                    "type": "price_update",
                    "platform": "deriv",
                    "asset": asset,
                    "timestamp": int(time.time() * 1000),
                    "price": price + price_change,
                    "volume": random.uniform(50, 500)
                },
                f"market_deriv_{asset}"
            )

# Trading signal WebSocket for updates
async def send_trading_signals():
    """
    Send trading signals to connected WebSocket clients
    """
    while True:
        await asyncio.sleep(5)  # Send updates every 5 seconds
        
        # Generate mock trading signals
        import random
        
        # List of assets
        assets = [
            {"platform": "binance", "asset": "BTCUSDT"},
            {"platform": "binance", "asset": "ETHUSDT"},
            {"platform": "deriv", "asset": "R_10"},
            {"platform": "deriv", "asset": "R_25"}
        ]
        
        # Randomly select an asset to send a signal for
        asset_info = random.choice(assets)
        
        # Only send a signal 20% of the time to avoid flooding
        if random.random() < 0.2:
            signal_type = random.choice(["buy", "sell", "hold"])
            confidence = random.uniform(0.6, 0.95)
            
            # Higher confidence for buy/sell signals
            if signal_type != "hold":
                confidence = random.uniform(0.75, 0.95)
            
            # Create the signal
            signal = {
                "type": "trading_signal",
                "timestamp": int(time.time() * 1000),
                "platform": asset_info["platform"],
                "asset": asset_info["asset"],
                "signal": signal_type,
                "confidence": confidence,
                "reason": "Multiple brain consensus",
                "timeframe": random.choice(["1m", "5m", "15m", "1h"]),
                "brains": [
                    {
                        "id": "pattern_brain",
                        "name": "Pattern Brain",
                        "signal": signal_type,
                        "confidence": confidence + random.uniform(-0.05, 0.05)
                    },
                    {
                        "id": "trend_brain",
                        "name": "Trend Brain",
                        "signal": signal_type,
                        "confidence": confidence + random.uniform(-0.08, 0.08)
                    }
                ]
            }
            
            # Send to asset-specific channel
            await manager.broadcast(signal, f"signal_{asset_info['platform']}_{asset_info['asset']}")
            
            # Also send to the all_signals channel
            await manager.broadcast(signal, "all_signals")

def setup_routes(app: FastAPI):
    """
    Set up all API routes
    """
    app.include_router(main_router)
    app.include_router(auth_router)
    app.include_router(user_router)
    app.include_router(system_router)
    app.include_router(market_router)
    app.include_router(trading_router)
    app.include_router(strategy_router)
    app.include_router(brain_router)
    app.include_router(backtest_router)
    app.include_router(monitoring_router)
    app.include_router(brain_council.router)
    
    # Start background tasks
    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(send_monitoring_updates())
        asyncio.create_task(send_market_updates())
        asyncio.create_task(send_trading_signals())
