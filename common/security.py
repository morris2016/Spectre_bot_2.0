#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Security Module

This module provides security utilities for API keys, credential management,
encryption/decryption, and secure storage.
"""

import os
import json
import time
import base64
import hashlib
import hmac
import secrets
import string
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
try:  # pragma: no cover - optional dependency
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    import logging

    CRYPTO_AVAILABLE = False

    logging.getLogger(__name__).warning(
        "cryptography package not available; using insecure fallback"
    )

    def _xor_bytes(data: bytes, key: bytes) -> bytes:
        return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))

    class Fernet:  # type: ignore
        """Simplistic XOR-based fallback for missing cryptography."""

        def __init__(self, key: bytes):
            self.key = base64.urlsafe_b64decode(key)

        @staticmethod
        def generate_key() -> bytes:
            return base64.urlsafe_b64encode(os.urandom(32))

        def encrypt(self, data: bytes) -> bytes:
            return base64.urlsafe_b64encode(_xor_bytes(data, self.key))

        def decrypt(self, token: bytes) -> bytes:
            return _xor_bytes(base64.urlsafe_b64decode(token), self.key)

from common.logger import get_logger
from common.exceptions import SecurityError, APIKeyError, AuthenticationError

class SecureCredentialManager:
    """Manages secure storage and retrieval of sensitive credentials."""
    
    def __init__(self, encryption_key=None, key_file=None, storage_method="file", 
                storage_path="./credentials", auto_generate=True):
        """
        Initialize credential manager.
        
        Args:
            encryption_key: Optional encryption key (base64 encoded)
            key_file: Path to file containing encryption key
            storage_method: Storage method ("file" or "memory")
            storage_path: Path to credentials storage directory (if using file storage)
            auto_generate: Whether to auto-generate encryption key if not provided
        """
        self.encryption_key = encryption_key
        self.key_file = key_file
        self.storage_method = storage_method
        self.storage_path = storage_path
        self.auto_generate = auto_generate
        self.cipher = None
        self.credentials = {}
        self.logger = get_logger("SecureCredentialManager")
        
    async def initialize(self):
        """
        Initialize the credential manager.
        
        Raises:
            SecurityError: If initialization fails
        """
        try:
            # Set up encryption key
            if not self.encryption_key:
                if self.key_file and os.path.exists(self.key_file):
                    # Load key from file
                    with open(self.key_file, 'rb') as f:
                        self.encryption_key = f.read().strip()
                elif self.auto_generate:
                    # Generate new key
                    self.encryption_key = Fernet.generate_key()
                    
                    if self.key_file:
                        # Save key to file
                        key_dir = os.path.dirname(self.key_file)
                        if key_dir and not os.path.exists(key_dir):
                            os.makedirs(key_dir)
                            
                        with open(self.key_file, 'wb') as f:
                            f.write(self.encryption_key)
                            
                        # Secure the key file
                        os.chmod(self.key_file, 0o600)
                else:
                    raise SecurityError("No encryption key provided and auto_generate is disabled")
                    
            # Initialize Fernet cipher
            self.cipher = Fernet(self.encryption_key)
            
            # Set up storage
            if self.storage_method == "file":
                if not os.path.exists(self.storage_path):
                    os.makedirs(self.storage_path)
                    
                # Load existing credentials
                await self._load_credentials()
                
            self.logger.info("Secure credential manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize secure credential manager: {str(e)}")
            raise SecurityError(f"Failed to initialize secure credential manager: {str(e)}")
            
    async def _load_credentials(self):
        """
        Load credentials from storage.
        
        Raises:
            SecurityError: If loading fails
        """
        if self.storage_method != "file":
            return
            
        try:
            credential_path = Path(self.storage_path)
            if not credential_path.exists():
                return
                
            for credential_file in credential_path.glob("*.enc"):
                credential_type = credential_file.stem
                
                with open(credential_file, 'rb') as f:
                    encrypted_data = f.read()
                    
                decrypted_data = self.cipher.decrypt(encrypted_data)
                self.credentials[credential_type] = json.loads(decrypted_data.decode('utf-8'))
                
        except Exception as e:
            self.logger.error(f"Failed to load credentials: {str(e)}")
            raise SecurityError(f"Failed to load credentials: {str(e)}")
            
    async def _save_credentials(self, credential_type: str):
        """
        Save credentials to storage.
        
        Args:
            credential_type: Credential type to save
            
        Raises:
            SecurityError: If saving fails
        """
        if self.storage_method != "file" or credential_type not in self.credentials:
            return
            
        try:
            credential_path = Path(self.storage_path)
            if not credential_path.exists():
                credential_path.mkdir(parents=True)
                
            data = json.dumps(self.credentials[credential_type]).encode('utf-8')
            encrypted_data = self.cipher.encrypt(data)
            
            with open(credential_path / f"{credential_type}.enc", 'wb') as f:
                f.write(encrypted_data)
                
        except Exception as e:
            self.logger.error(f"Failed to save credentials for {credential_type}: {str(e)}")
            raise SecurityError(f"Failed to save credentials: {str(e)}")
            
    async def set_credentials(self, credential_type: str, credentials: Dict[str, Any]):
        """
        Set credentials.
        
        Args:
            credential_type: Credential type
            credentials: Credential data
            
        Raises:
            SecurityError: If setting fails
        """
        if not self.cipher:
            raise SecurityError("Credential manager not initialized")
            
        try:
            self.credentials[credential_type] = credentials
            
            if self.storage_method == "file":
                await self._save_credentials(credential_type)
                
            self.logger.info(f"Credentials set for {credential_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to set credentials for {credential_type}: {str(e)}")
            raise SecurityError(f"Failed to set credentials: {str(e)}")
            
    async def get_credentials(self, credential_type: str) -> Dict[str, Any]:
        """
        Get credentials.
        
        Args:
            credential_type: Credential type
            
        Returns:
            Credential data
            
        Raises:
            SecurityError: If retrieval fails
        """
        if not self.cipher:
            raise SecurityError("Credential manager not initialized")
            
        if credential_type not in self.credentials:
            self.logger.warning(f"Credentials not found for {credential_type}")
            return {}
            
        return self.credentials[credential_type]
        
    async def delete_credentials(self, credential_type: str):
        """
        Delete credentials.
        
        Args:
            credential_type: Credential type
            
        Raises:
            SecurityError: If deletion fails
        """
        if not self.cipher:
            raise SecurityError("Credential manager not initialized")
            
        try:
            if credential_type in self.credentials:
                del self.credentials[credential_type]
                
            if self.storage_method == "file":
                credential_path = Path(self.storage_path) / f"{credential_type}.enc"
                if credential_path.exists():
                    credential_path.unlink()
                    
            self.logger.info(f"Credentials deleted for {credential_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to delete credentials for {credential_type}: {str(e)}")
            raise SecurityError(f"Failed to delete credentials: {str(e)}")
            
    async def encrypt(self, data: Union[str, bytes]) -> str:
        """
        Encrypt data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Base64-encoded encrypted data
            
        Raises:
            SecurityError: If encryption fails
        """
        if not self.cipher:
            raise SecurityError("Credential manager not initialized")
            
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
                
            encrypted_data = self.cipher.encrypt(data)
            return base64.b64encode(encrypted_data).decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Encryption failed: {str(e)}")
            raise SecurityError(f"Encryption failed: {str(e)}")
            
    async def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Base64-encoded encrypted data
            
        Returns:
            Decrypted data
            
        Raises:
            SecurityError: If decryption fails
        """
        if not self.cipher:
            raise SecurityError("Credential manager not initialized")
            
        try:
            encrypted_bytes = base64.b64decode(encrypted_data)
            decrypted_data = self.cipher.decrypt(encrypted_bytes)
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            self.logger.error(f"Decryption failed: {str(e)}")
            raise SecurityError(f"Decryption failed: {str(e)}")
            
    def generate_password(self, length=16, include_uppercase=True, include_digits=True, 
                         include_special=True):
        """
        Generate a secure random password.
        
        Args:
            length: Password length
            include_uppercase: Whether to include uppercase letters
            include_digits: Whether to include digits
            include_special: Whether to include special characters
            
        Returns:
            Generated password
        """
        # Define character sets
        chars = string.ascii_lowercase
        if include_uppercase:
            chars += string.ascii_uppercase
        if include_digits:
            chars += string.digits
        if include_special:
            chars += string.punctuation
            
        # Generate password
        return ''.join(secrets.choice(chars) for _ in range(length))
        
    def generate_api_key(self, prefix=None, length=32):
        """
        Generate a secure API key.
        
        Args:
            prefix: Optional prefix for the key
            length: Key length (excluding prefix)
            
        Returns:
            Generated API key
        """
        api_key = base64.b32encode(os.urandom(length)).decode('utf-8').rstrip('=')
        if prefix:
            api_key = f"{prefix}_{api_key}"
        return api_key
        
    def generate_api_secret(self, length=64):
        """
        Generate a secure API secret.
        
        Args:
            length: Secret length
            
        Returns:
            Generated API secret
        """
        return base64.b64encode(os.urandom(length)).decode('utf-8')
        
    def generate_hmac_signature(self, key: str, message: str, algorithm="sha256"):
        """
        Generate an HMAC signature.
        
        Args:
            key: Secret key
            message: Message to sign
            algorithm: Hash algorithm (sha256, sha512)
            
        Returns:
            Hexadecimal signature
            
        Raises:
            SecurityError: If signature generation fails
        """
        try:
            if isinstance(key, str):
                key = key.encode('utf-8')
                
            if isinstance(message, str):
                message = message.encode('utf-8')
                
            if algorithm == "sha256":
                digest = hmac.new(key, message, hashlib.sha256).hexdigest()
            elif algorithm == "sha512":
                digest = hmac.new(key, message, hashlib.sha512).hexdigest()
            else:
                raise SecurityError(f"Unsupported algorithm: {algorithm}")
                
            return digest
            
        except Exception as e:
            self.logger.error(f"Failed to generate HMAC signature: {str(e)}")
            raise SecurityError(f"Failed to generate HMAC signature: {str(e)}")



    def generate_secure_token(self, length=64):
        """
        Generate a secure token.
        
        Args:
            length: Token length
            
        Returns:
            Generated token
        """
        return secrets.token_urlsafe(length)
        
    def verify_password(self, hashed_password: str, password: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            hashed_password: Hashed password
            password: Plain password
            
        Returns:
            True if password is correct
            
        Raises:
            SecurityError: If verification fails
        """
        try:
            # Split the stored hash to get the algorithm, salt, and hash
            parts = hashed_password.split('$')
            if len(parts) != 4:
                raise SecurityError("Invalid hash format")
                
            _, algorithm, salt, stored_hash = parts
            
            # Compute hash of the provided password
            if algorithm == "pbkdf2_sha256":
                computed_hash = self._hash_password_pbkdf2(password, salt)
            else:
                raise SecurityError(f"Unsupported hash algorithm: {algorithm}")
                
            # Compare hashes (constant-time comparison to prevent timing attacks)
            return secrets.compare_digest(stored_hash, computed_hash)
            
        except Exception as e:
            self.logger.error(f"Password verification failed: {str(e)}")
            raise SecurityError(f"Password verification failed: {str(e)}")
            
    def hash_password(self, password: str) -> str:
        """
        Hash a password.
        
        Args:
            password: Password to hash
            
        Returns:
            Hashed password
            
        Raises:
            SecurityError: If hashing fails
        """
        try:
            salt = base64.b64encode(os.urandom(16)).decode('utf-8')
            password_hash = self._hash_password_pbkdf2(password, salt)
            
            # Format: $algorithm$salt$hash
            return f"$pbkdf2_sha256${salt}${password_hash}"
            
        except Exception as e:
            self.logger.error(f"Password hashing failed: {str(e)}")
            raise SecurityError(f"Password hashing failed: {str(e)}")
            
    def _hash_password_pbkdf2(self, password: str, salt: str) -> str:
        """
        Hash a password using PBKDF2.
        
        Args:
            password: Password to hash
            salt: Salt (base64 encoded)
            
        Returns:
            Hashed password (base64 encoded)
        """
        salt_bytes = base64.b64decode(salt)

        if CRYPTO_AVAILABLE:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt_bytes,
                iterations=100000,
            )
            key = kdf.derive(password.encode("utf-8"))
        else:
            key = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), salt_bytes, 100000, dklen=32
            )

        return base64.b64encode(key).decode("utf-8")
        
    async def audit_log(self, action: str, user: str, resource: str, status: str, details: Optional[Dict] = None):
        """
        Log a security audit event.
        
        Args:
            action: Action being performed
            user: User performing the action
            resource: Resource being accessed
            status: Status of the action (success, failure)
            details: Optional details
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "user": user,
            "resource": resource,
            "status": status,
            "details": details or {}
        }
        
        self.logger.info(f"Security audit: {json.dumps(log_entry)}")

# NEW CODE STARTS HERE — OUTSIDE THE CLASS
secure_manager_instance = SecureCredentialManager()

async def encrypt_data(data: Union[str, bytes]) -> str:
    if not secure_manager_instance.cipher:
        await secure_manager_instance.initialize()
    return await secure_manager_instance.encrypt(data)

async def decrypt_data(encrypted: str) -> str:
    if not secure_manager_instance.cipher:
        await secure_manager_instance.initialize()
    return await secure_manager_instance.decrypt(encrypted)

def hash_password(password: str) -> str:
    return secure_manager_instance.hash_password(password)

def verify_password(hashed_password: str, password: str) -> bool:
    return secure_manager_instance.verify_password(hashed_password, password)


# This should be outside the SecureCredentialManager class, with other standalone functions
def hash_content(content: Union[str, bytes]) -> str:
    """
    Creates a secure hash of the provided content.
    
    Args:
        content: Content to hash (string or bytes)
        
    Returns:
        Hexadecimal hash string
    """
    if isinstance(content, str):
        content = content.encode('utf-8')
        
    return hashlib.sha256(content).hexdigest()

async def decrypt_credentials(encrypted_credentials: str) -> Dict[str, Any]:
    """
    Decrypt credentials.
    
    Args:
        encrypted_credentials: Encrypted credentials string
        
    Returns:
        Decrypted credentials as dictionary
        
    Raises:
        SecurityError: If decryption fails
    """
    try:
        decrypted_data = await decrypt_data(encrypted_credentials)
        return json.loads(decrypted_data)
    except Exception as e:
        logger = get_logger("security")
        logger.error(f"Failed to decrypt credentials: {str(e)}")
        raise SecurityError(f"Failed to decrypt credentials: {str(e)}")
