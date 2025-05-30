#!/usr/bin/env python3
"""
Run the UI service standalone.
"""

import asyncio
import os
import sys
import logging

from config import Config, load_config
from common.logger import setup_logging, get_logger
from ui.app import UIService

# Set up logging
setup_logging(logging.INFO)
logger = get_logger("run_ui")

async def main():
    """Run the UI service."""
    # Load configuration
    config_path = os.environ.get("CONFIG_PATH", "config.yml")
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Force UI to run on port 9000
    config.ui["port"] = 9000
    
    # Create and start UI service
    ui_service = UIService(config)
    await ui_service.start()
    
    logger.info(f"UI service started on {config.ui.get('host', '0.0.0.0')}:{config.ui.get('port', 8080)}")
    
    # Keep the service running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down UI service...")
        await ui_service.stop()
        logger.info("UI service stopped")

if __name__ == "__main__":
    asyncio.run(main())