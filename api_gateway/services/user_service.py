#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
User Service

This module provides user management functionality for the API Gateway,
including user registration, authentication, profile management, and
user preferences.
"""

import os
import json
import time
import hashlib
import secrets
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
import logging

import jwt
from pydantic import BaseModel, EmailStr, validator, Field

from config import Config
from common.logger import get_logger
from common.metrics import MetricsCollector
from common.redis_client import RedisClient
from common.db_client import DatabaseClient
from common.exceptions import (
    AuthenticationError, PermissionDeniedError, NotFoundError,
    ValidationError, ServiceUnavailableError
)
from api_gateway.services.base_service import BaseService

# Initialize logger
logger = get_logger(__name__)

class UserCreateModel(BaseModel):
    """Model for user creation."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    
    @validator("username")
    def username_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError("Username must be alphanumeric")
        return v
    
    @validator("password")
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c.isalpha() for c in v):
            raise ValueError("Password must contain at least one letter")
        return v

class UserUpdateModel(BaseModel):
    """Model for user updates."""
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    
    @validator("password")
    def password_strength(cls, v):
        if v is None:
            return v
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c.isalpha() for c in v):
            raise ValueError("Password must contain at least one letter")
        return v

class UserService(BaseService):
    """User management service."""
    
    def __init__(self, config: Config, redis_client: RedisClient, db_client: DatabaseClient):
        """Initialize the user service."""
        super().__init__("user_service", redis_client, db_client)
        self.config = config
        self.jwt_secret = config.get("security.jwt_secret", os.urandom(32).hex())
        self.jwt_algorithm = config.get("security.jwt_algorithm", "HS256")
        self.token_expiry = config.get("security.token_expiry", 86400)  # 24 hours
        self.refresh_token_expiry = config.get("security.refresh_token_expiry", 2592000)  # 30 days
        self.password_salt = config.get("security.password_salt", os.urandom(16).hex())
        self.start_time = time.time()
    
    async def start(self):
        """Start the user service."""
        await super().start()
        
        # Subscribe to user-related messages
        await self.subscribe("user:*", self._handle_user_message)
    
    async def register_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new user.
        
        Args:
            user_data: User registration data
            
        Returns:
            Created user object
            
        Raises:
            ValidationError: If data is invalid
            ServiceUnavailableError: If registration fails
        """
        try:
            # Validate user data
            user_create = UserCreateModel(**user_data)
            
            # Check if username or email already exists
            existing_user = await self._get_user_by_username(user_create.username)
            if existing_user:
                raise ValidationError("Username already exists")
            
            existing_user = await self._get_user_by_email(user_create.email)
            if existing_user:
                raise ValidationError("Email already exists")
            
            # Hash password
            hashed_password = self._hash_password(user_create.password)
            
            # Prepare user object
            user = {
                "username": user_create.username,
                "email": user_create.email,
                "password_hash": hashed_password,
                "full_name": user_create.full_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "last_login": None,
                "is_active": True,
                "is_verified": False,
                "roles": ["user"]
            }
            
            # Store user in database
            result = await self.db_client.insert_one("users", user)
            user_id = str(result.inserted_id)
            user["id"] = user_id
            
            # Remove sensitive data
            user.pop("password_hash", None)
            
            # Create verification token (for email verification)
            verification_token = secrets.token_urlsafe(32)
            await self.redis_client.set(
                f"verification:{verification_token}",
                user_id,
                ex=86400  # 24 hours
            )
            
            # Publish user registered event
            await self.publish("user:registered", {
                "user_id": user_id,
                "username": user["username"],
                "email": user["email"],
                "verification_token": verification_token
            })
            
            # Track metrics
            self.metrics.increment("user_registrations_total")
            
            logger.info(f"User registered: id={user_id} username={user['username']}")
            
            return user
            
        except ValidationError as e:
            logger.warning(f"User registration validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"User registration error: {str(e)}")
            logger.error(traceback.format_exc())
            raise ServiceUnavailableError("User registration failed") from e
    
    async def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticate a user and generate tokens.
        
        Args:
            username: Username or email
            password: User password
            
        Returns:
            Authentication tokens and user info
            
        Raises:
            AuthenticationError: If authentication fails
            ServiceUnavailableError: If authentication service fails
        """
        try:
            # Find user by username or email
            user = None
            if "@" in username:
                # Looks like an email
                user = await self._get_user_by_email(username)
            else:
                # Looks like a username
                user = await self._get_user_by_username(username)
            
            if not user:
                # User not found, but return generic error for security
                raise AuthenticationError("Invalid username or password")
            
            # Check if user is active
            if not user.get("is_active", True):
                raise AuthenticationError("Account is inactive")
            
            # Verify password
            is_valid = self._verify_password(password, user["password_hash"])
            if not is_valid:
                # Invalid password, but return generic error for security
                raise AuthenticationError("Invalid username or password")
            
            # Generate tokens
            user_id = str(user["_id"])
            access_token = await self.create_access_token(user_id, user.get("roles", ["user"]))
            refresh_token = await self.create_refresh_token(user_id)
            
            # Update last login time
            await self.db_client.update_one(
                "users",
                {"_id": user["_id"]},
                {"$set": {"last_login": datetime.now(timezone.utc).isoformat()}}
            )
            
            # Remove sensitive data
            user_info = {k: v for k, v in user.items() if k != "password_hash"}
            user_info["id"] = user_id
            
            # Publish login event
            await self.publish("user:login", {
                "user_id": user_id,
                "username": user["username"]
            })
            
            # Track metrics
            self.metrics.increment("user_logins_total")
            
            logger.info(f"User authenticated: id={user_id} username={user['username']}")
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": self.token_expiry,
                "user": user_info
            }
            
        except AuthenticationError:
            # Track failed logins
            self.metrics.increment("user_login_failures_total")
            raise
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            logger.error(traceback.format_exc())
            raise ServiceUnavailableError("Authentication failed") from e
    
    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh access token using a refresh token.
        
        Args:
            refresh_token: The refresh token
            
        Returns:
            New access token and refresh token
            
        Raises:
            AuthenticationError: If token is invalid
            ServiceUnavailableError: If token refresh fails
        """
        try:
            # Get token data from Redis
            token_key = f"refresh_token:{refresh_token}"
            user_id = await self.redis_client.get(token_key)
            
            if not user_id:
                raise AuthenticationError("Invalid refresh token")
            
            # Get user from database
            user = await self._get_user_by_id(user_id)
            if not user:
                raise AuthenticationError("User not found")
            
            # Check if user is active
            if not user.get("is_active", True):
                raise AuthenticationError("Account is inactive")
            
            # Generate new tokens
            new_access_token = await self.create_access_token(user_id, user.get("roles", ["user"]))
            new_refresh_token = await self.create_refresh_token(user_id)
            
            # Invalidate old refresh token
            await self.redis_client.delete(token_key)
            
            # Publish token refresh event
            await self.publish("user:token_refresh", {
                "user_id": user_id
            })
            
            # Track metrics
            self.metrics.increment("token_refreshes_total")
            
            logger.info(f"Token refreshed for user id={user_id}")
            
            return {
                "access_token": new_access_token,
                "refresh_token": new_refresh_token,
                "token_type": "bearer",
                "expires_in": self.token_expiry
            }
            
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Token refresh error: {str(e)}")
            logger.error(traceback.format_exc())
            raise ServiceUnavailableError("Token refresh failed") from e
    
    async def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate an access token.
        
        Args:
            token: The access token
            
        Returns:
            Token payload if valid
            
        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            # Decode and verify token
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )
            
            # Check if token is blacklisted
            is_blacklisted = await self.redis_client.exists(f"blacklist:{token}")
            if is_blacklisted:
                raise AuthenticationError("Token is blacklisted")
            
            # Check token expiration
            if "exp" in payload and datetime.fromtimestamp(payload["exp"]) < datetime.now():
                raise AuthenticationError("Token expired")
            
            # Check token type
            if payload.get("type") != "access":
                raise AuthenticationError("Invalid token type")
            
            return payload
            
        except jwt.DecodeError:
            raise AuthenticationError("Invalid token signature")
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token expired")
        except Exception as e:
            logger.error(f"Token validation error: {str(e)}")
            logger.error(traceback.format_exc())
            raise AuthenticationError("Invalid token") from e
    
    async def create_access_token(self, user_id: str, roles: List[str]) -> str:
        """
        Create an access token for a user.
        
        Args:
            user_id: The user ID
            roles: User roles
            
        Returns:
            JWT access token
        """
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(seconds=self.token_expiry)
        
        payload = {
            "sub": user_id,
            "roles": roles,
            "type": "access",
            "iat": int(now.timestamp()),
            "exp": int(expiry.timestamp())
        }
        
        return jwt.encode(
            payload,
            self.jwt_secret,
            algorithm=self.jwt_algorithm
        )
    
    async def create_refresh_token(self, user_id: str) -> str:
        """
        Create a refresh token for a user.
        
        Args:
            user_id: The user ID
            
        Returns:
            Refresh token
        """
        # Generate secure random token
        refresh_token = secrets.token_urlsafe(32)
        
        # Store in Redis with expiry
        token_key = f"refresh_token:{refresh_token}"
        await self.redis_client.set(token_key, user_id, ex=self.refresh_token_expiry)
        
        return refresh_token
    
    async def revoke_token(self, token: str) -> bool:
        """
        Revoke an access token.
        
        Args:
            token: The access token
            
        Returns:
            True if token was revoked, False otherwise
        """
        try:
            # Decode token without verification to get expiration
            payload = jwt.decode(
                token,
                options={"verify_signature": False}
            )
            
            # Get token expiration
            exp = payload.get("exp")
            if not exp:
                return False
            
            # Calculate remaining time
            now = int(time.time())
            ttl = max(0, exp - now)
            
            # Add to blacklist with TTL matching token expiration
            await self.redis_client.set(f"blacklist:{token}", "1", ex=ttl)
            
            # Track metrics
            self.metrics.increment("tokens_revoked_total")
            
            logger.info(f"Token revoked: user_id={payload.get('sub')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Token revocation error: {str(e)}")
            return False
    
    async def get_user(self, user_id: str) -> Dict[str, Any]:
        """
        Get user by ID.
        
        Args:
            user_id: The user ID
            
        Returns:
            User object
            
        Raises:
            NotFoundError: If user not found
        """
        user = await self._get_user_by_id(user_id)
        if not user:
            raise NotFoundError("User not found")
        
        # Remove sensitive data
        user_info = {k: v for k, v in user.items() if k != "password_hash"}
        user_info["id"] = str(user["_id"])
        
        return user_info
    
    async def update_user(self, user_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user information.
        
        Args:
            user_id: The user ID
            update_data: Data to update
            
        Returns:
            Updated user object
            
        Raises:
            NotFoundError: If user not found
            ValidationError: If data is invalid
        """
        try:
            # Validate update data
            user_update = UserUpdateModel(**update_data)
            
            # Get user
            user = await self._get_user_by_id(user_id)
            if not user:
                raise NotFoundError("User not found")
            
            # Prepare update object
            update_dict = {}
            
            # Update email
            if user_update.email is not None and user_update.email != user.get("email"):
                # Check if email already exists
                existing_user = await self._get_user_by_email(user_update.email)
                if existing_user and str(existing_user["_id"]) != user_id:
                    raise ValidationError("Email already exists")
                update_dict["email"] = user_update.email
            
            # Update full name
            if user_update.full_name is not None:
                update_dict["full_name"] = user_update.full_name
            
            # Update password
            if user_update.password is not None:
                update_dict["password_hash"] = self._hash_password(user_update.password)
            
            # Add updated timestamp
            update_dict["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            # Update user in database
            if update_dict:
                await self.db_client.update_one(
                    "users",
                    {"_id": user["_id"]},
                    {"$set": update_dict}
                )
            
            # Get updated user
            updated_user = await self._get_user_by_id(user_id)
            if not updated_user:
                raise ServiceUnavailableError("Failed to retrieve updated user")
            
            # Remove sensitive data
            user_info = {k: v for k, v in updated_user.items() if k != "password_hash"}
            user_info["id"] = user_id
            
            # Publish user updated event
            await self.publish("user:updated", {
                "user_id": user_id,
                "fields_updated": list(update_dict.keys())
            })
            
            logger.info(f"User updated: id={user_id} fields={list(update_dict.keys())}")
            
            return user_info
            
        except NotFoundError:
            raise
        except ValidationError as e:
            logger.warning(f"User update validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"User update error: {str(e)}")
            logger.error(traceback.format_exc())
            raise ServiceUnavailableError("User update failed") from e
    
    async def _get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user by ID.
        
        Args:
            user_id: The user ID
            
        Returns:
            User object or None if not found
        """
        try:
            return await self.db_client.find_one("users", {"_id": user_id})
        except Exception as e:
            logger.error(f"Database error getting user by ID: {str(e)}")
            return None
    
    async def _get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user by username.
        
        Args:
            username: The username
            
        Returns:
            User object or None if not found
        """
        try:
            return await self.db_client.find_one("users", {"username": username})
        except Exception as e:
            logger.error(f"Database error getting user by username: {str(e)}")
            return None
    
    async def _get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Get user by email.
        
        Args:
            email: The email
            
        Returns:
            User object or None if not found
        """
        try:
            return await self.db_client.find_one("users", {"email": email})
        except Exception as e:
            logger.error(f"Database error getting user by email: {str(e)}")
            return None
    
    async def _handle_user_message(self, channel: str, message: Dict[str, Any]):
        """
        Handle user-related Redis messages.
        
        Args:
            channel: Message channel
            message: Message data
        """
        try:
            # Handle different message types based on channel
            if channel == "user:password_reset_request":
                await self._handle_password_reset_request(message)
            elif channel == "user:verify_email":
                await self._handle_email_verification(message)
        except Exception as e:
            logger.error(f"Error handling user message on channel {channel}: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _handle_password_reset_request(self, message: Dict[str, Any]):
        """
        Handle password reset request.
        
        Args:
            message: Message data with email
        """
        email = message.get("email")
        if not email:
            logger.error("Password reset request missing email")
            return
        
        try:
            # Find user by email
            user = await self._get_user_by_email(email)
            if not user:
                logger.info(f"Password reset requested for non-existent email: {email}")
                return
            
            # Generate reset token
            reset_token = secrets.token_urlsafe(32)
            user_id = str(user["_id"])
            
            # Store token in Redis with expiration
            await self.redis_client.set(
                f"password_reset:{reset_token}",
                user_id,
                ex=3600  # 1 hour
            )
            
            # Publish event for email sending
            await self.publish("email:send", {
                "template": "password_reset",
                "recipient": email,
                "data": {
                    "username": user["username"],
                    "reset_token": reset_token,
                    "reset_link": f"{self.config.get('app.frontend_url', '')}/reset-password?token={reset_token}"
                }
            })
            
            logger.info(f"Password reset token generated for user: {user_id}")
            
        except Exception as e:
            logger.error(f"Error handling password reset request: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _handle_email_verification(self, message: Dict[str, Any]):
        """
        Handle email verification.
        
        Args:
            message: Message data with token
        """
        token = message.get("token")
        if not token:
            logger.error("Email verification missing token")
            return
        
        try:
            # Get user ID from token
            user_id = await self.redis_client.get(f"verification:{token}")
            if not user_id:
                logger.warning(f"Invalid or expired verification token: {token}")
                return
            
            # Mark user as verified
            await self.db_client.update_one(
                "users",
                {"_id": user_id},
                {"$set": {"is_verified": True}}
            )
            
            # Delete verification token
            await self.redis_client.delete(f"verification:{token}")
            
            # Get user for email notification
            user = await self._get_user_by_id(user_id)
            if user:
                # Publish email verification success event
                await self.publish("email:send", {
                    "template": "verification_success",
                    "recipient": user["email"],
                    "data": {
                        "username": user["username"]
                    }
                })
            
            logger.info(f"Email verified for user: {user_id}")
            
        except Exception as e:
            logger.error(f"Error handling email verification: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _hash_password(self, password: str) -> str:
        """
        Hash a password with salt.
        
        Args:
            password: The password to hash
            
        Returns:
            Hashed password
        """
        salted = password + self.password_salt
        return hashlib.sha256(salted.encode()).hexdigest()
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """
        Verify a password against a hash.
        
        Args:
            password: The password to verify
            hashed: The hashed password
            
        Returns:
            True if password is correct, False otherwise
        """
        return self._hash_password(password) == hashed
