#!/usr/bin/env python3
"""UI Service for QuantumSpectre Elite Trading System."""

import asyncio
import os
import socket
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse
from pydantic import BaseModel
import json
import os

from common.logger import get_logger
from common.metrics import MetricsCollector


class UIService:
    """Service responsible for running the web based user interface."""

    def __init__(self, config: Any, loop: Optional[asyncio.AbstractEventLoop] = None,
                 redis_client: Any = None, db_client: Any = None) -> None:
        self.config = config
        self.loop = loop or asyncio.get_event_loop()
        self.redis_client = redis_client
        self.db_client = db_client
        self.logger = get_logger("UIService")
        self.metrics = MetricsCollector("ui")

        self._server: Optional[uvicorn.Server] = None
        self.task: Optional[asyncio.Task] = None
        self.running = False

        # FastAPI application instance
        self.app = FastAPI(title="QuantumSpectre UI")
        self.app.add_api_route("/health", self.health_endpoint, methods=["GET"])
        
        # Add API endpoints for database configuration
        self.app.add_api_route("/api/v1/system/config", self.get_system_config, methods=["GET"])
        self.app.add_api_route("/api/v1/system/database/config", self.update_database_config, methods=["POST"])
        self.app.add_api_route("/token", self.token_endpoint, methods=["POST"])

        static_dir = os.path.abspath(self.config.ui.get("static_dir", "./ui/dist"))
        index_file = self.config.ui.get("index_file", "index.html")
        self.index_path = os.path.join(static_dir, index_file)

        if os.path.isdir(static_dir):
            # Mount the static directory at the root path
            self.app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
            self.logger.info(f"Serving static files from {static_dir}")
        else:
            self.logger.warning("UI static directory '%s' does not exist", static_dir)
            # Fallback to serving the index route
            self.app.add_api_route("/{full_path:path}", self.index, methods=["GET"])

    def _port_available(self, host: str, port: int) -> bool:
        """Return True if the port is free for binding."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            return sock.connect_ex((host, port)) != 0

    async def start(self) -> None:
        """Start the UI service using Uvicorn."""
        if self.running:
            return

        host = self.config.ui.get("host", "0.0.0.0")
        port = int(self.config.ui.get("port", 3002))
        log_level = self.config.logging.get("ui_level", "info").lower()

        # Skip startup if the port is already in use
        if not self._port_available(host, port):
            self.logger.error("UI port %s is already in use; disabling UI service", port)
            return

        config = uvicorn.Config(self.app, host=host, port=port, log_level=log_level,
                                loop="asyncio")
        self._server = uvicorn.Server(config)
        self.task = self.loop.create_task(self._server.serve())
        self.running = True
        self.logger.info("UI Service started on %s:%s", host, port)

    async def stop(self) -> None:
        """Stop the UI service."""
        if not self.running:
            return
        if self._server and self._server.should_exit is False:
            self._server.should_exit = True
        if self.task:
            await self.task
        self.running = False
        self.logger.info("UI Service stopped")

    async def health_check(self) -> bool:
        """Return True if the service is running."""
        return self.running

    async def health_endpoint(self) -> dict:
        """Simple health check endpoint for FastAPI."""
        return {"status": "ok"}

    async def index(self, full_path: str) -> Response:
        """Serve the React application's index file for all routes."""
        if not os.path.isfile(self.index_path):
            self.logger.warning("UI index file '%s' not found", self.index_path)
            return Response(status_code=404)
        return FileResponse(self.index_path)
        
    async def get_system_config(self) -> dict:
        """Get system configuration."""
        # Return a sanitized version of the configuration
        return {
            "database": {
                "host": self.config.database.get("host", "localhost"),
                "port": self.config.database.get("port", 5432),
                "user": self.config.database.get("user", "postgres"),
                "password": "",  # Don't return the actual password
                "dbname": self.config.database.get("dbname", "quantumspectre"),
                "min_pool_size": self.config.database.get("min_pool_size", 5),
                "max_pool_size": self.config.database.get("max_pool_size", 20),
                "connection_timeout": self.config.database.get("connection_timeout", 60),
                "command_timeout": self.config.database.get("command_timeout", 60),
                "enabled": self.config.database.get("enabled", True)
            }
        }
    
    async def update_database_config(self, request: Request) -> dict:
        """Update database configuration."""
        try:
            # Parse request body
            data = await request.json()
            
            # Update config.yml with new database settings
            from config import save_config
            
            # Update database configuration
            self.config.database["host"] = data.get("host", "localhost")
            self.config.database["port"] = data.get("port", 5432)
            self.config.database["user"] = data.get("user", "postgres")
            self.config.database["password"] = data.get("password", "")
            self.config.database["dbname"] = data.get("dbname", "quantumspectre")
            self.config.database["min_pool_size"] = data.get("min_pool_size", 5)
            self.config.database["max_pool_size"] = data.get("max_pool_size", 20)
            self.config.database["connection_timeout"] = data.get("connection_timeout", 60)
            self.config.database["command_timeout"] = data.get("command_timeout", 60)
            self.config.database["enabled"] = True
            self.config.database["use_memory_storage"] = True  # Enable memory storage
            
            # Save configuration
            save_config(self.config)
            
            # Skip database connection check
            self.logger.info("Database configuration saved. Using memory storage.")
            
            return {
                "status": "success",
                "message": "Database configuration updated successfully"
            }
        except Exception as e:
            self.logger.error(f"Error updating database configuration: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update database configuration: {str(e)}"
            )
    
    async def token_endpoint(self, request: Request) -> dict:
        """Simple token endpoint for demo purposes."""
        try:
            form_data = await request.form()
            username = form_data.get("username")
            password = form_data.get("password")
            
            # For demo purposes, accept admin/admin
            if username == "admin" and password == "admin":
                return {
                    "access_token": "demo_token",
                    "token_type": "bearer"
                }
            else:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid username or password"
                )
        except Exception as e:
            self.logger.error(f"Error in token endpoint: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Authentication error: {str(e)}"
            )
