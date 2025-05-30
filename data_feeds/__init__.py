

#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Data Feeds Package

This package provides a collection of data feed implementations for
connecting to various data sources including exchanges, news providers,
social media, dark web, and on-chain data sources.

Each feed is optimized for its specific data source with specialized
capabilities for handling the unique characteristics of that source.
"""

import logging
from typing import Dict, List, Any, Type, Optional
import importlib

from common.logger import get_logger
from common.utils import generate_uuid
from common.exceptions import FeedNotFoundError, FeedInitializationError

# Version of the data_feeds package
__version__ = '1.0.0'

# Registry of all available feed classes
FEED_REGISTRY = {}

# Feed categories
CATEGORY_EXCHANGE = 'exchange'
CATEGORY_NEWS = 'news'
CATEGORY_SOCIAL = 'social'
CATEGORY_ONCHAIN = 'onchain'
CATEGORY_DARKWEB = 'darkweb'
CATEGORY_REGIME = 'regime'

# Logger for this package
logger = get_logger('data_feeds')


def register_feed(feed_name: str, feed_class: Type, category: str) -> None:
    """
    Register a feed class in the global registry.
    
    Args:
        feed_name: Unique name for the feed
        feed_class: The feed class to register
        category: Category of the feed
    """
    if feed_name in FEED_REGISTRY:
        logger.warning(f"Overwriting existing feed registration for {feed_name}")
        
    FEED_REGISTRY[feed_name] = {
        'class': feed_class,
        'category': category
    }
    logger.debug(f"Registered feed: {feed_name} ({category})")


def get_feed_class(feed_name: str) -> Type:
    """
    Get a feed class by name from the registry.
    
    Args:
        feed_name: Name of the feed to retrieve
        
    Returns:
        The feed class
        
    Raises:
        FeedNotFoundError: If the feed is not registered
    """
    if feed_name not in FEED_REGISTRY:
        raise FeedNotFoundError(f"Feed not found: {feed_name}")
        
    return FEED_REGISTRY[feed_name]['class']


def get_feeds_by_category(category: str) -> Dict[str, Type]:
    """
    Get all feeds of a specific category.
    
    Args:
        category: Category to filter by
        
    Returns:
        Dictionary of feed_name -> feed_class for the specified category
    """
    return {
        name: info['class'] 
        for name, info in FEED_REGISTRY.items() 
        if info['category'] == category
    }


def create_feed(feed_name: str, config: Dict[str, Any]) -> Any:
    """
    Create a feed instance by name with the provided configuration.
    
    Args:
        feed_name: Name of the feed to create
        config: Configuration for the feed
        
    Returns:
        Initialized feed instance
        
    Raises:
        FeedNotFoundError: If the feed is not registered
        FeedInitializationError: If the feed fails to initialize
    """
    try:
        feed_class = get_feed_class(feed_name)
        return feed_class(config)
    except FeedNotFoundError:
        raise
    except Exception as e:
        raise FeedInitializationError(f"Failed to initialize feed {feed_name}: {str(e)}") from e


def list_available_feeds() -> List[Dict[str, Any]]:
    """
    List all available feeds with their categories.
    
    Returns:
        List of dictionaries containing feed information
    """
    return [
        {
            'name': name,
            'category': info['category'],
            'class': info['class'].__name__
        }
        for name, info in FEED_REGISTRY.items()
    ]


def discover_feeds() -> None:
    """
    Discover and register all feed implementations in the package.
    
    This function searches for all modules in the data_feeds package and
    attempts to load and register any feed classes they contain.
    """
    logger.info("Discovering feed implementations...")
    
    # Define modules to search
    feed_modules = [
        'data_feeds.binance_feed',
        'data_feeds.deriv_feed',
        'data_feeds.news_feed',
        'data_feeds.social_feed',
        'data_feeds.onchain_feed',
        'data_feeds.dark_web_feed',
        'data_feeds.regime_feed'
    ]
    
    # Try to import each module and register its feeds
    for module_name in feed_modules:
        try:
            module = importlib.import_module(module_name)
            # If the module has a register_feeds function, call it
            if hasattr(module, 'register_feeds'):
                module.register_feeds()
            
            logger.debug(f"Loaded feed module: {module_name}")
        except ImportError as e:
            logger.warning(f"Failed to import feed module {module_name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading feed module {module_name}: {str(e)}")
    
    logger.info(f"Discovered {len(FEED_REGISTRY)} feed implementations")


# Automatically discover and register feeds when the package is imported
discover_feeds()


# Set up an app for the feeds service
def create_app(config: Dict[str, Any]) -> 'DataFeedService':
    """
    Create the DataFeedService application.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized DataFeedService instance
    """
    # Import here to avoid circular imports
    from data_feeds.app import DataFeedService
    from common.event_bus import EventBus

    return DataFeedService(config, event_bus=EventBus.get_instance())

