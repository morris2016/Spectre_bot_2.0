#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
API Gateway - Authentication

This module handles authentication and authorization for the QuantumSpectre Elite Trading System.
It provides JWT-based authentication, user management, and secure access control.
"""

import os
import time
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import jwt
from passlib.context import CryptContext
from fastapi import Request, HTTPException, status, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator

from common.logger import get_logger
from common.exceptions import AuthenticationError
from common.security import hash_password, verify_password, generate_salt
from config import Config

# Create logger
logger = get_logger(__name__)

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "super_secret_key_change_in_production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION = 24 * 60 * 60  # 24 hours in seconds

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# User Database - In a real system, this would be stored in a database
# This is just a placeholder for development
USERS_DB = {
    "admin": {
        "username": "admin",
        "password_hash": "$2b$12$Fa8GYgRkJj.JeF0DPKvKZ.PWq1oJp56FXHKLI1wPPPANnihMjxlSy",  # hashed "admin123"
        "email": "admin@quantumspectre.com",
        "created_at": datetime(2023, 1, 1),
        "last_login": None,
        "preferences": {
            "default_platform": "binance",
            "default_timeframe": "15m",
            "theme": "dark",
            "notifications_enabled": True
        }
    },
    "demo": {
        "username": "demo",
        "password_hash": "$2b$12$AoHxQQUs2p7bAQFGC7TLFu4bZKdEgdJvgZT0x8n.7EAWBCVE4mB4q",  # hashed "demo123"
        "email": "demo@quantumspectre.com",
        "created_at": datetime(2023, 1, 15),
        "last_login": None,
        "preferences": {
            "default_platform": "deriv",
            "default_timeframe": "5m",
            "theme": "light",
            "notifications_enabled": True
        }
    }
}

# Models
class UserModel(BaseModel):
    """User model for authentication"""
    username: str
    email: str
    created_at: datetime
    last_login: Optional[datetime] = None
    preferences: Dict = Field(default_factory=dict)

class CredentialsSchema(BaseModel):
    """Credentials schema for login"""
    username: str
    password: str

class TokenResponse(BaseModel):
    """Token response for successful authentication"""
    access_token: str
    token_type: str
    expires_at: int
    user: UserModel

class TokenData(BaseModel):
    """Token data for JWT payload"""
    username: str
    exp: int

class JWTBearer(HTTPBearer):
    """JWT Bearer authentication"""
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail="Invalid authentication credentials"
            )
            
        if credentials.scheme != "Bearer":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail="Invalid authentication scheme"
            )
            
        # Validate the token
        try:
            payload = jwt.decode(
                credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM]
            )
            
            # Check if token is expired
            if payload.get("exp") < time.time():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired",
                    headers={"WWW-Authenticate": "Bearer"},
                )
                
            # Check if username exists and is valid
            username = payload.get("username")
            if username is None or username not in USERS_DB:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid user",
                    headers={"WWW-Authenticate": "Bearer"},
                )
                
            return credentials.credentials
            
        except jwt.PyJWTError as e:
            logger.error(f"JWT validation error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

class UserAuth:
    """User authentication service"""
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Get password hash"""
        return pwd_context.hash(password)
        
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
        
    @staticmethod
    def create_access_token(username: str) -> str:
        """Create JWT access token"""
        expires_at = int(time.time()) + JWT_EXPIRATION
        payload = {
            "username": username,
            "exp": expires_at
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    @staticmethod
    def decode_token(token: str) -> dict:
        """Decode JWT token"""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.PyJWTError as e:
            logger.error(f"Failed to decode token: {str(e)}")
            raise AuthenticationError("Invalid token")
    
    @staticmethod
    async def login(username: str, password: str) -> TokenResponse:
        """Authenticate user and return token"""
        # Check if user exists
        if username not in USERS_DB:
            logger.warning(f"Login attempt with non-existent user: {username}")
            raise AuthenticationError("Invalid username or password")
            
        user = USERS_DB[username]
        
        # Verify password
        if not UserAuth.verify_password(password, user["password_hash"]):
            logger.warning(f"Login attempt with incorrect password for user: {username}")
            raise AuthenticationError("Invalid username or password")
            
        # Create access token
        access_token = UserAuth.create_access_token(username)
        
        # Update last login
        USERS_DB[username]["last_login"] = datetime.now()
        
        # Create user model
        user_model = UserModel(
            username=user["username"],
            email=user["email"],
            created_at=user["created_at"],
            last_login=user["last_login"],
            preferences=user["preferences"]
        )
        
        # Create token response
        return TokenResponse(
            access_token=access_token,
            token_type="Bearer",
            expires_at=int(time.time()) + JWT_EXPIRATION,
            user=user_model
        )
    
    @staticmethod
    async def refresh(token: str) -> TokenResponse:
        """Refresh JWT token"""
        try:
            # Decode token
            payload = UserAuth.decode_token(token)
            
            # Check if token is valid
            username = payload.get("username")
            if username not in USERS_DB:
                logger.warning(f"Token refresh attempt for non-existent user: {username}")
                raise AuthenticationError("Invalid token")
                
            # Create new token
            access_token = UserAuth.create_access_token(username)
            
            # Get user data
            user = USERS_DB[username]
            
            # Create user model
            user_model = UserModel(
                username=user["username"],
                email=user["email"],
                created_at=user["created_at"],
                last_login=user["last_login"],
                preferences=user["preferences"]
            )
            
            # Create token response
            return TokenResponse(
                access_token=access_token,
                token_type="Bearer",
                expires_at=int(time.time()) + JWT_EXPIRATION,
                user=user_model
            )
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Token refresh error: {str(e)}")
            raise AuthenticationError("Failed to refresh token")
    
    @staticmethod
    async def get_current_user(token: str) -> UserModel:
        """Get current user from token"""
        try:
            # Decode token
            payload = UserAuth.decode_token(token)
            
            # Check if token is valid
            username = payload.get("username")
            if username not in USERS_DB:
                logger.warning(f"Token validation attempt for non-existent user: {username}")
                raise AuthenticationError("Invalid token")
                
            # Get user data
            user = USERS_DB[username]
            
            # Create user model
            return UserModel(
                username=user["username"],
                email=user["email"],
                created_at=user["created_at"],
                last_login=user["last_login"],
                preferences=user["preferences"]
            )
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Get current user error: {str(e)}")
            raise AuthenticationError("Failed to get current user")

# Create instance of user authentication service
get_current_user = UserAuth()

async def get_current_user_from_request(request: Request) -> UserModel:
    """Get current user from request"""
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        token = auth_header.split(" ")[1]
        return await get_current_user.get_current_user(token)
    except Exception as e:
        logger.error(f"Get current user from request error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

# User management functions (these would use a database in production)
async def create_user(username: str, password: str, email: str) -> UserModel:
    """Create a new user"""
    try:
        # Check if user already exists
        if username in USERS_DB:
            logger.warning(f"Attempt to create duplicate user: {username}")
            raise ValueError("Username already exists")
            
        # Hash password
        password_hash = UserAuth.get_password_hash(password)
        
        # Create user
        USERS_DB[username] = {
            "username": username,
            "password_hash": password_hash,
            "email": email,
            "created_at": datetime.now(),
            "last_login": None,
            "preferences": {
                "default_platform": "binance",
                "default_timeframe": "15m",
                "theme": "dark",
                "notifications_enabled": True
            }
        }
        
        # Return user model
        return UserModel(
            username=username,
            email=email,
            created_at=USERS_DB[username]["created_at"],
            last_login=None,
            preferences=USERS_DB[username]["preferences"]
        )
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Create user error: {str(e)}")
        raise ValueError("Failed to create user")

async def update_user_preferences(username: str, preferences: Dict) -> UserModel:
    """Update user preferences"""
    try:
        # Check if user exists
        if username not in USERS_DB:
            logger.warning(f"Attempt to update non-existent user: {username}")
            raise ValueError("User not found")
            
        # Update preferences
        USERS_DB[username]["preferences"].update(preferences)
        
        # Return user model
        return UserModel(
            username=username,
            email=USERS_DB[username]["email"],
            created_at=USERS_DB[username]["created_at"],
            last_login=USERS_DB[username]["last_login"],
            preferences=USERS_DB[username]["preferences"]
        )
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Update user preferences error: {str(e)}")
        raise ValueError("Failed to update user preferences")

async def change_password(username: str, old_password: str, new_password: str) -> bool:
    """Change user password"""
    try:
        # Check if user exists
        if username not in USERS_DB:
            logger.warning(f"Attempt to change password for non-existent user: {username}")
            raise ValueError("User not found")
            
        # Verify old password
        if not UserAuth.verify_password(old_password, USERS_DB[username]["password_hash"]):
            logger.warning(f"Attempt to change password with incorrect old password for user: {username}")
            raise ValueError("Incorrect old password")
            
        # Hash new password
        password_hash = UserAuth.get_password_hash(new_password)
        
        # Update password
        USERS_DB[username]["password_hash"] = password_hash
        
        return True
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Change password error: {str(e)}")
        raise ValueError("Failed to change password")
