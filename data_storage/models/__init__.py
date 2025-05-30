#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Database Models Package

This package contains the SQLAlchemy ORM models used for data persistence
across the system, including market data, user configurations, and system state.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, LargeBinary, Text, JSON, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship, backref
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Import Base from market_data to avoid duplicate declarations
from data_storage.models.market_data import Base

# Import all models to ensure they're registered with the Base
# Note: We're importing these after Base to avoid circular imports

# We'll import these models in a way that avoids circular imports
# The models will still be registered with Base when they're imported elsewhere

# Export only the Base class
__all__ = ['Base']
