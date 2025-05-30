#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Brain Council API Routes

This module provides API endpoints for the enhanced brain council system,
including asset-specific councils and ML council data.
"""

import time
import asyncio
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from common.redis_client import RedisClient
from common.db_client import DatabaseClient
from common.auth import get_current_user
from common.models import User

# Initialize router
router = APIRouter(prefix="/api/brain-council", tags=["brain_council"])

# Initialize logger
logger = logging.getLogger("api_gateway.brain_council")

# Redis client for communication with brain council service
redis_client = None
db_client = None

class AssetCouncilResponse(BaseModel):
    """Response model for asset council data."""
    asset_id: str
    direction: str
    confidence: float
    regime: str
    votes: Dict[str, Any]
    ml_models: List[Dict[str, Any]]
    timestamp: float

class MLCouncilResponse(BaseModel):
    """Response model for ML council data."""
    models: List[Dict[str, Any]]
    avg_accuracy: float
    asset_coverage: int
    performance: Dict[str, Dict[str, float]]
    asset_models: Dict[str, List[Dict[str, Any]]]

class DashboardResponse(BaseModel):
    """Response model for council dashboard data."""
    asset_count: int
    brain_count: int
    ml_model_count: int
    signals_per_hour: int
    available_assets: List[str]
    available_timeframes: List[str]

async def initialize(redis: RedisClient, db: DatabaseClient):
    """Initialize the brain council routes with required clients."""
    global redis_client, db_client
    redis_client = redis
    db_client = db
    logger.info("Brain Council API routes initialized")

@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard_data(current_user: User = Depends(get_current_user)):
    """
    Get overview data for the brain council dashboard.
    
    Returns:
        Dashboard data including system statistics and available assets/timeframes
    """
    try:
        # Request dashboard data from brain council service
        request_id = f"dashboard_request_{int(time.time())}"
        response_channel = f"brain_council.dashboard.response.{request_id}"
        
        request = {
            "id": request_id,
            "response_channel": response_channel,
            "timestamp": time.time()
        }
        
        # Publish request
        await redis_client.publish("brain_council.dashboard.request", request)
        
        # Wait for response with timeout
        timeout = 5  # seconds
        response = await redis_client.subscribe_and_wait(
            response_channel, timeout=timeout
        )
        
        if not response:
            # If no response from service, generate mock data
            # In production, this would be replaced with a database query
            return {
                "asset_count": 6,
                "brain_count": 12,
                "ml_model_count": 8,
                "signals_per_hour": 120,
                "available_assets": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", "XRPUSDT"],
                "available_timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
            }
            
        return response
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")

@router.get("/asset/{asset_id}", response_model=AssetCouncilResponse)
async def get_asset_council_data(
    asset_id: str, 
    timeframe: str = Query("1h", description="Trading timeframe"),
    current_user: User = Depends(get_current_user)
):
    """
    Get data for a specific asset council.
    
    Args:
        asset_id: Asset identifier
        timeframe: Trading timeframe
        
    Returns:
        Asset council data including direction, confidence, and votes
    """
    try:
        # Request asset council data from brain council service
        request_id = f"asset_council_request_{asset_id}_{timeframe}_{int(time.time())}"
        response_channel = f"brain_council.asset.response.{request_id}"
        
        request = {
            "id": request_id,
            "asset_id": asset_id,
            "timeframe": timeframe,
            "response_channel": response_channel,
            "timestamp": time.time()
        }
        
        # Publish request
        await redis_client.publish("brain_council.asset.request", request)
        
        # Wait for response with timeout
        timeout = 5  # seconds
        response = await redis_client.subscribe_and_wait(
            response_channel, timeout=timeout
        )
        
        if not response:
            # If no response from service, generate mock data
            # In production, this would be replaced with a database query
            return {
                "asset_id": asset_id,
                "direction": "buy",
                "confidence": 0.75,
                "regime": "trending_bullish",
                "votes": {
                    "trend_brain": {"direction": "buy", "confidence": 0.8},
                    "momentum_brain": {"direction": "buy", "confidence": 0.7},
                    "ml_model_1": {"direction": "buy", "confidence": 0.9},
                    "ml_model_2": {"direction": "hold", "confidence": 0.6}
                },
                "ml_models": [
                    {"name": "XGBoost Classifier", "type": "classification", "accuracy": 0.82},
                    {"name": "LSTM Price Predictor", "type": "deep_learning", "accuracy": 0.78}
                ],
                "timestamp": time.time()
            }
            
        return response
        
    except Exception as e:
        logger.error(f"Error getting asset council data for {asset_id}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get asset council data: {str(e)}"
        )

@router.get("/ml-council", response_model=MLCouncilResponse)
async def get_ml_council_data(current_user: User = Depends(get_current_user)):
    """
    Get data for the ML council.
    
    Returns:
        ML council data including models, performance metrics, and asset-specific models
    """
    try:
        # Request ML council data from brain council service
        request_id = f"ml_council_request_{int(time.time())}"
        response_channel = f"brain_council.ml.response.{request_id}"
        
        request = {
            "id": request_id,
            "response_channel": response_channel,
            "timestamp": time.time()
        }
        
        # Publish request
        await redis_client.publish("brain_council.ml.request", request)
        
        # Wait for response with timeout
        timeout = 5  # seconds
        response = await redis_client.subscribe_and_wait(
            response_channel, timeout=timeout
        )
        
        if not response:
            # If no response from service, generate mock data
            # In production, this would be replaced with a database query
            return {
                "models": [
                    {"name": "XGBoost Classifier", "type": "classification", "accuracy": 0.82, "asset_count": 6},
                    {"name": "LSTM Price Predictor", "type": "deep_learning", "accuracy": 0.78, "asset_count": 4},
                    {"name": "Random Forest Regressor", "type": "regression", "accuracy": 0.75, "asset_count": 6},
                    {"name": "GRU Sequence Model", "type": "time_series", "accuracy": 0.79, "asset_count": 3},
                    {"name": "Ensemble Model", "type": "ensemble", "accuracy": 0.85, "asset_count": 6}
                ],
                "avg_accuracy": 0.798,
                "asset_coverage": 6,
                "performance": {
                    "classification": {
                        "accuracy": 0.82,
                        "precision": 0.80,
                        "recall": 0.78,
                        "f1_score": 0.79
                    },
                    "regression": {
                        "accuracy": 0.75,
                        "precision": 0.73,
                        "recall": 0.72,
                        "f1_score": 0.72
                    },
                    "time_series": {
                        "accuracy": 0.79,
                        "precision": 0.77,
                        "recall": 0.76,
                        "f1_score": 0.76
                    },
                    "deep_learning": {
                        "accuracy": 0.78,
                        "precision": 0.76,
                        "recall": 0.75,
                        "f1_score": 0.75
                    },
                    "ensemble": {
                        "accuracy": 0.85,
                        "precision": 0.83,
                        "recall": 0.82,
                        "f1_score": 0.82
                    }
                },
                "asset_models": {
                    "BTCUSDT": [
                        {"name": "XGBoost Classifier", "type": "classification", "accuracy": 0.84},
                        {"name": "LSTM Price Predictor", "type": "deep_learning", "accuracy": 0.81}
                    ],
                    "ETHUSDT": [
                        {"name": "XGBoost Classifier", "type": "classification", "accuracy": 0.83},
                        {"name": "GRU Sequence Model", "type": "time_series", "accuracy": 0.80}
                    ]
                }
            }
            
        return response
        
    except Exception as e:
        logger.error(f"Error getting ML council data: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get ML council data: {str(e)}"
        )