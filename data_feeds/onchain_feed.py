#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Onchain Data Feed Module

This module handles blockchain data gathering, processing, and analysis
for the QuantumSpectre Elite Trading System. It monitors multiple blockchains
to track on-chain metrics, whale movements, smart contract interactions,
and other blockchain data points that could affect asset prices.
"""

import inspect
import os
import re
import time
import json
import asyncio
import datetime
from typing import Dict, List, Set, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import threading
import queue
import logging
from concurrent.futures import ThreadPoolExecutor
import traceback
import hmac
import hashlib
from decimal import Decimal
from enum import Enum

# Data gathering and processing
import httpx
import websockets
import pandas as pd
import numpy as np

# Blockchain specific libraries
from bitcoinrpc.authproxy import AuthServiceProxy
from etherscan import Etherscan
from eth_utils import to_checksum_address
from eth_abi import decode as eth_decode  # noqa: F401  # imported for potential ABI decoding

# Internal imports
# Add compatibility for inspect.getargspec
# from data_feeds.inspect import getargspec
from config import Config
from common.logger import get_logger
from common.utils import retry_with_backoff_decorator, rate_limit, SafeDict, hash_content
# Default values will be defined in __init__ or loaded from config
# from common.constants import (
#     BLOCKCHAIN_ENDPOINTS, BLOCKCHAIN_API_KEYS, BLOCKCHAIN_UPDATE_INTERVALS,
#     WHALE_THRESHOLDS, CONTRACT_ADDRESSES, ERC20_ABI
# )
from common.metrics import MetricsCollector
from common.exceptions import (
    DataSourceError, RateLimitError, AuthenticationError,
    ParsingError, BlockchainConnectionError
)
from common.db_client import DatabaseClient
from common.redis_client import RedisClient
from common.async_utils import gather_with_concurrency, timed_cache

from data_feeds.base_feed import BaseDataFeed

# Initialize logger
logger = get_logger(__name__)


class EthereumRPCClient:
    """Minimal JSON-RPC client for Ethereum-compatible chains using httpx."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self._http = httpx.Client()
        self.eth = self.Eth(self)

    class Eth:
        def __init__(self, parent: "EthereumRPCClient"):
            self._parent = parent

        def _call(self, method: str, params=None):
            return self._parent._call(method, params)

        @property
        def block_number(self) -> int:
            return int(self._call("eth_blockNumber"), 16)

        @property
        def gas_price(self) -> int:
            return int(self._call("eth_gasPrice"), 16)

        def get_block(self, block_number: Union[int, str], full_transactions: bool = True) -> Dict[str, Any]:
            if isinstance(block_number, int):
                block_number = hex(block_number)
            return self._call("eth_getBlockByNumber", [block_number, full_transactions])

        def get_transaction_receipt(self, tx_hash: str) -> Dict[str, Any]:
            return self._call("eth_getTransactionReceipt", [tx_hash])

        def get_code(self, address: str) -> str:
            return self._call("eth_getCode", [address, "latest"])

    def _call(self, method: str, params=None):
        if params is None:
            params = []
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        resp = self._http.post(self.endpoint, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        return data["result"]

    def is_connected(self) -> bool:
        try:
            self.eth.block_number
            return True
        except Exception:
            return False

class BlockchainNetwork(Enum):
    """Enum for supported blockchain networks"""
    BITCOIN = "bitcoin"
    ETHEREUM = "ethereum"
    BINANCE_SMART_CHAIN = "bsc"
    POLYGON = "polygon"
    SOLANA = "solana"
    AVALANCHE = "avalanche"

@dataclass
class OnchainTransaction:
    """Data class for standardized blockchain transaction data"""
    blockchain: BlockchainNetwork
    tx_hash: str
    block_number: int
    timestamp: datetime.datetime
    from_address: str
    to_address: str
    value: Decimal
    asset: str  # BTC, ETH, token symbol, etc.
    gas_used: int = 0
    gas_price: int = 0
    is_contract_interaction: bool = False
    contract_address: Optional[str] = None
    method_id: Optional[str] = None
    method_name: Optional[str] = None
    is_token_transfer: bool = False
    token_address: Optional[str] = None
    token_id: Optional[int] = None  # For NFTs
    status: bool = True  # True if transaction succeeded
    usd_value: Optional[Decimal] = None
    is_whale_tx: bool = False
    confidence_score: float = 1.0  # For probabilistic metrics
    
    def __post_init__(self):
        """Post initialization processing"""
        # Ensure timestamp is datetime object
        if isinstance(self.timestamp, (int, float)):
            self.timestamp = datetime.datetime.fromtimestamp(self.timestamp, tz=datetime.timezone.utc)
        elif isinstance(self.timestamp, str):
            self.timestamp = datetime.datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
        
        # Calculate USD value if not provided but value and a price is available
        if self.usd_value is None and hasattr(self, 'asset_price') and self.value is not None:
            self.usd_value = self.value * getattr(self, 'asset_price', Decimal(0))
        
        # Calculate fee
        if self.gas_used and self.gas_price:
            self.fee = Decimal(self.gas_used * self.gas_price) / Decimal(10**18)  # in native token
        else:
            self.fee = Decimal(0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        result = {k: v for k, v in self.__dict__.items()}
        # Convert Enum to string
        if isinstance(result['blockchain'], BlockchainNetwork):
            result['blockchain'] = result['blockchain'].value
        # Convert datetime to ISO format string
        if isinstance(result['timestamp'], datetime.datetime):
            result['timestamp'] = result['timestamp'].isoformat()
        # Convert Decimal to string for JSON serialization
        for k, v in result.items():
            if isinstance(v, Decimal):
                result[k] = str(v)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OnchainTransaction':
        """Create instance from dictionary"""
        # Convert string to Enum
        if 'blockchain' in data and isinstance(data['blockchain'], str):
            data['blockchain'] = BlockchainNetwork(data['blockchain'])
        # Convert string to Decimal
        for k, v in data.items():
            if k in ('value', 'usd_value', 'fee') and isinstance(v, str):
                data[k] = Decimal(v)
        return cls(**data)


@dataclass
class WhaleAlert:
    """Data class for whale movement alerts"""
    blockchain: BlockchainNetwork
    tx_hash: str
    timestamp: datetime.datetime
    from_address: str
    to_address: str
    asset: str
    value: Decimal
    usd_value: Decimal
    from_label: Optional[str] = None  # Exchange/wallet label if known
    to_label: Optional[str] = None    # Exchange/wallet label if known
    alert_type: str = "movement"  # movement, accumulation, distribution
    confidence: float = 1.0
    related_assets: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        result = {k: v for k, v in self.__dict__.items()}
        # Convert Enum to string
        if isinstance(result['blockchain'], BlockchainNetwork):
            result['blockchain'] = result['blockchain'].value
        # Convert datetime to ISO format string
        if isinstance(result['timestamp'], datetime.datetime):
            result['timestamp'] = result['timestamp'].isoformat()
        # Convert Decimal to string for JSON serialization
        for k, v in result.items():
            if isinstance(v, Decimal):
                result[k] = str(v)
        return result


@dataclass
class BlockchainMetric:
    """Data class for blockchain metrics"""
    blockchain: BlockchainNetwork
    metric_name: str  # network_hashrate, gas_price, active_addresses, etc.
    timestamp: datetime.datetime
    value: Any
    interval: str = "1h"  # Time interval this metric represents
    related_assets: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        result = {k: v for k, v in self.__dict__.items()}
        # Convert Enum to string
        if isinstance(result['blockchain'], BlockchainNetwork):
            result['blockchain'] = result['blockchain'].value
        # Convert datetime to ISO format string
        if isinstance(result['timestamp'], datetime.datetime):
            result['timestamp'] = result['timestamp'].isoformat()
        # Convert Decimal to string for JSON serialization
        if isinstance(result['value'], Decimal):
            result['value'] = str(result['value'])
        return result


class OnchainDataFeed(BaseDataFeed):
    """
    Onchain Data Feed for gathering and analyzing blockchain data relevant to trading.
    
    This class monitors various blockchains to track on-chain metrics, whale movements,
    smart contract interactions, and other blockchain data points that could affect asset prices.
    """
    
    def __init__(self, config: Config, db_client: DatabaseClient, redis_client: RedisClient):
        """
        Initialize the Onchain Data Feed.
        
        Args:
            config: Application configuration
            db_client: Database client for persistent storage
            redis_client: Redis client for real-time data and caching
        """
        super().__init__(name="onchain_data_feed", config=config)
        self.db_client = db_client
        self.redis_client = redis_client
        self.metrics = MetricsCollector(namespace="onchain_feed")
        
        # Define default values for constants previously imported
        DEFAULT_BLOCKCHAIN_ENDPOINTS = {
            "ethereum": {"rpc": "http://localhost:8545"},
            "bitcoin": {"rpc": "http://localhost:8332"},
            "bsc": {"rpc": "http://localhost:8546"},
            "polygon": {"rpc": "http://localhost:8547"}
        }
        DEFAULT_BLOCKCHAIN_API_KEYS = {} # Users should configure this
        DEFAULT_BLOCKCHAIN_UPDATE_INTERVALS = {
            "ethereum": 60, "bitcoin": 300, "bsc": 60, "polygon": 60
        }
        DEFAULT_WHALE_THRESHOLDS = {
            "BTC": 100, "ETH": 1000
        }
        DEFAULT_CONTRACT_ADDRESSES = {} # Users should configure this
        # ERC20_ABI is more complex, usually a JSON string or dict.
        # For now, let's assume it's not strictly needed for startup or is handled elsewhere if critical.
        # If it's needed, it should be loaded from a file or defined as a large constant string/dict.

        # Configure blockchains to monitor
        self.blockchains = config.get("onchain_feed.blockchains", [e.value for e in BlockchainNetwork])
        self.endpoints = config.get("onchain_feed.endpoints", DEFAULT_BLOCKCHAIN_ENDPOINTS)
        self.api_keys = config.get("onchain_feed.api_keys", DEFAULT_BLOCKCHAIN_API_KEYS)
        self.update_intervals = config.get("onchain_feed.update_intervals", DEFAULT_BLOCKCHAIN_UPDATE_INTERVALS)
        
        # Whale monitoring configuration
        self.whale_thresholds = config.get("onchain_feed.whale_thresholds", DEFAULT_WHALE_THRESHOLDS)
        
        # Smart contract monitoring configuration
        self.contract_addresses = config.get("onchain_feed.contract_addresses", DEFAULT_CONTRACT_ADDRESSES)
        
        # Blockchain clients
        self.blockchain_clients = {}
        self.api_clients = {}
        
        # Asset to blockchain mapping
        self.asset_blockchain_map = {
            "BTC/USD": BlockchainNetwork.BITCOIN,
            "ETH/USD": BlockchainNetwork.ETHEREUM,
            "BNB/USD": BlockchainNetwork.BINANCE_SMART_CHAIN,
            "MATIC/USD": BlockchainNetwork.POLYGON,
            "SOL/USD": BlockchainNetwork.SOLANA,
            "AVAX/USD": BlockchainNetwork.AVALANCHE,
            # Add token mappings
            "USDT/USD": [BlockchainNetwork.ETHEREUM, BlockchainNetwork.BINANCE_SMART_CHAIN],
            "USDC/USD": [BlockchainNetwork.ETHEREUM, BlockchainNetwork.BINANCE_SMART_CHAIN],
        }
        
        # Data processing queues
        self.tx_queue = asyncio.Queue()
        self.metric_queue = asyncio.Queue()
        self.whale_queue = asyncio.Queue()
        
        # Blockchain state tracking
        self.last_block_heights = {}
        self.blockchain_metrics = {}
        
        # Address labeling cache
        self.address_labels = {}
        
        # Transaction processing pipeline
        self.tx_processing_pipeline = [
            self._enrich_transaction,
            self._analyze_transaction,
            self._detect_whale_movement,
            self._detect_contract_interaction,
            self._detect_pattern
        ]
        
        logger.info(f"Onchain Data Feed initialized with {len(self.blockchains)} blockchains")
    
    async def start(self):
        """Start the onchain data feed collection and processing"""
        logger.info("Starting Onchain Data Feed service")
        
        # Initialize blockchain clients
        await self._initialize_clients()
        
        # Initialize worker tasks
        tasks = []
        
        # Start blockchain-specific collectors
        for blockchain in self.blockchains:
            collector_method = getattr(self, f"_collect_{blockchain}", None)
            if collector_method:
                interval = self.update_intervals.get(blockchain, 60)
                tasks.append(self._periodic_collector(collector_method, interval))
                logger.info(f"Started collector for {blockchain} with interval {interval}s")
            else:
                logger.warning(f"No collector method found for blockchain: {blockchain}")
        
        # Start blockchain metrics collectors
        tasks.append(self._periodic_collector(self._collect_blockchain_metrics, 300))  # Every 5 minutes
        
        # Start data processors
        tasks.append(self._transaction_processor())
        tasks.append(self._publish_whale_alerts())
        tasks.append(self._publish_metrics())
        
        # Run all tasks concurrently
        self.running = True
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Stop the onchain data feed service"""
        logger.info("Stopping Onchain Data Feed service")
        self.running = False
        
        # Close blockchain clients
        for client in self.blockchain_clients.values():
            if hasattr(client, "close"):
                if asyncio.iscoroutinefunction(client.close):
                    await client.close()
                else:
                    client.close()
        
        # Close API clients
        for client in self.api_clients.values():
            if hasattr(client, "close"):
                await client.close()
        
        logger.info("Onchain Data Feed service stopped")
    
    async def _initialize_clients(self):
        """Initialize blockchain clients and API clients"""
        logger.info("Starting client initialization...")
        # Initialize HTTP client for API requests
        timeout = httpx.Timeout(30.0, connect=10.0)
        self.http_client = httpx.AsyncClient(timeout=timeout)
        self.api_clients['http'] = self.http_client
        logger.debug(f"HTTP client initialized with timeout {timeout}")
        
        # Initialize blockchain-specific clients
        logger.debug(f"Initializing blockchain clients for networks: {self.blockchains}")
        for blockchain in self.blockchains:
            logger.debug(f"Attempting to initialize client for {blockchain}")
            try:
                init_method = getattr(self, f"_init_{blockchain}_client", None)
                if init_method:
                    await init_method()
                    logger.info(f"Successfully initialized {blockchain} client")
                    logger.debug(f"{blockchain} endpoint configuration: {self.endpoints.get(blockchain, {})}; API keys present: {bool(self.api_keys.get(blockchain))}")
                else:
                    logger.warning(f"No initialization method found for {blockchain}")
            except Exception as e:
                logger.error(f"Error initializing {blockchain} client: {str(e)}")
                logger.error(traceback.format_exc())
        logger.info("Client initialization process completed.")
    
    async def _init_ethereum_client(self):
        """Initialize Ethereum client and related APIs"""
        blockchain = BlockchainNetwork.ETHEREUM
        logger.info(f"Initializing {blockchain.value} client...")
        
        # Get RPC endpoint
        endpoint = self.endpoints.get(blockchain.value, {}).get('rpc')
        logger.debug(f"Using {blockchain.value} RPC endpoint: {endpoint}")
        if not endpoint:
            logger.error(f"No RPC endpoint configured for {blockchain.value}")
            raise BlockchainConnectionError(f"No RPC endpoint configured for {blockchain.value}")
        
        # Initialize JSON-RPC client
        try:
            w3 = EthereumRPCClient(endpoint)
        except Exception as e:
            logger.error(f"Error creating RPC client for {blockchain.value} at {endpoint}: {e}")
            raise
        
        # Placeholder for POA chain adjustments if needed
        
        # Test connection
        logger.debug(f"Testing connection to {blockchain.value} node at {endpoint}...")
        if not w3.is_connected():
            logger.error(f"Failed to connect to {blockchain.value} node at {endpoint}")
            raise BlockchainConnectionError(f"Failed to connect to Ethereum node at {endpoint}")
        logger.info(f"Successfully connected to {blockchain.value} node at {endpoint}")
        
        self.blockchain_clients[blockchain.value] = w3
        
        # Initialize Etherscan API client if key is available
        etherscan_key = self.api_keys.get(blockchain.value, {}).get('etherscan')
        if etherscan_key:
            logger.debug(f"Initializing Etherscan client for {blockchain.value} with provided API key.")
            etherscan = Etherscan(etherscan_key)
            self.api_clients[f"{blockchain.value}_etherscan"] = etherscan
            logger.info(f"Etherscan client for {blockchain.value} initialized.")
        else:
            logger.debug(f"No Etherscan API key found for {blockchain.value}.")
        
        # Get initial block height
        self.last_block_heights[blockchain.value] = w3.eth.block_number
        logger.info(f"{blockchain.value} client initialization complete. Last block: {self.last_block_heights[blockchain.value]}")
    
    async def _init_bitcoin_client(self):
        """Initialize Bitcoin client and related APIs"""
        blockchain = BlockchainNetwork.BITCOIN
        logger.info(f"Initializing {blockchain.value} client...")
        
        # Get RPC endpoint details
        endpoint_config = self.endpoints.get(blockchain.value, {})
        rpc_uri = endpoint_config.get('rpc')
        logger.debug(f"Using {blockchain.value} RPC URI: {rpc_uri}")
        
        if not rpc_uri:
            logger.error(f"No RPC endpoint configured for {blockchain.value}")
            raise BlockchainConnectionError(f"No RPC endpoint configured for {blockchain.value}")
        
        # Create RPC client
        rpc_user = endpoint_config.get('rpc_user', '')
        rpc_password = endpoint_config.get('rpc_password', '')
        logger.debug(f"Using {blockchain.value} RPC user: {'present' if rpc_user else 'not present'}")
        
        # Construct service URL
        service_url = f"http://{rpc_user}:{rpc_password}@{rpc_uri.split('://')[-1]}"
        logger.debug(f"Constructed {blockchain.value} service URL: {service_url.replace(rpc_password, '********') if rpc_password else service_url}")
        
        # Initialize Bitcoin RPC client
        try:
            btc_client = AuthServiceProxy(service_url, timeout=30) # Added timeout
        except Exception as e:
            logger.error(f"Error creating AuthServiceProxy for {blockchain.value}: {e}")
            raise

        # Test connection
        logger.debug(f"Testing connection to {blockchain.value} node...")
        try:
            info = btc_client.getblockchaininfo()
            self.last_block_heights[blockchain.value] = info['blocks']
            logger.info(f"Successfully connected to {blockchain.value} node. Current block: {info['blocks']}")
        except Exception as e:
            logger.error(f"Failed to connect to {blockchain.value} node: {str(e)}")
            raise BlockchainConnectionError(f"Failed to connect to Bitcoin node: {str(e)}")
        
        self.blockchain_clients[blockchain.value] = btc_client
        
        # Initialize Blockstream API client (no key required)
        self.api_clients[f"{blockchain.value}_blockstream"] = None  # Use HTTP client directly
        logger.debug(f"Blockstream API client for {blockchain.value} will use the shared HTTP client.")
        logger.info(f"{blockchain.value} client initialization complete.")
    
    async def _init_binance_smart_chain_client(self):
        """Initialize Binance Smart Chain client and related APIs"""
        blockchain = BlockchainNetwork.BINANCE_SMART_CHAIN
        logger.info(f"Initializing {blockchain.value} client...")
        
        # Get RPC endpoint
        endpoint = self.endpoints.get(blockchain.value, {}).get('rpc')
        logger.debug(f"Using {blockchain.value} RPC endpoint: {endpoint}")
        if not endpoint:
            logger.error(f"No RPC endpoint configured for {blockchain.value}")
            raise BlockchainConnectionError(f"No RPC endpoint configured for {blockchain.value}")
        
        # Initialize JSON-RPC client
        try:
            w3 = EthereumRPCClient(endpoint)
        except Exception as e:
            logger.error(f"Error creating RPC client for {blockchain.value} at {endpoint}: {e}")
            raise

        # Test connection
        logger.debug(f"Testing connection to {blockchain.value} node at {endpoint}...")
        if not w3.is_connected():
            logger.error(f"Failed to connect to {blockchain.value} node at {endpoint}")
            raise BlockchainConnectionError(f"Failed to connect to Binance Smart Chain node at {endpoint}")
        logger.info(f"Successfully connected to {blockchain.value} node at {endpoint}")
        
        self.blockchain_clients[blockchain.value] = w3
        
        # Initialize BscScan API client if key is available
        bscscan_key = self.api_keys.get(blockchain.value, {}).get('bscscan')
        if bscscan_key:
            logger.debug(f"Initializing BscScan client for {blockchain.value} with provided API key.")
            # BscScan API uses same format as Etherscan
            self.api_clients[f"{blockchain.value}_bscscan"] = bscscan_key  # Store key for HTTP requests
            logger.info(f"BscScan client for {blockchain.value} initialized (key stored).")
        else:
            logger.debug(f"No BscScan API key found for {blockchain.value}.")
        
        # Get initial block height
        self.last_block_heights[blockchain.value] = w3.eth.block_number
        logger.info(f"{blockchain.value} client initialization complete. Last block: {self.last_block_heights[blockchain.value]}")
    
    async def _init_polygon_client(self):
        """Initialize Polygon client and related APIs"""
        blockchain = BlockchainNetwork.POLYGON
        logger.info(f"Initializing {blockchain.value} client...")
        
        # Similar to Ethereum initialization
        endpoint = self.endpoints.get(blockchain.value, {}).get('rpc')
        logger.debug(f"Using {blockchain.value} RPC endpoint: {endpoint}")
        if not endpoint:
            logger.error(f"No RPC endpoint configured for {blockchain.value}")
            raise BlockchainConnectionError(f"No RPC endpoint configured for {blockchain.value}")
        
        # Initialize JSON-RPC client
        try:
            w3 = EthereumRPCClient(endpoint)
        except Exception as e:
            logger.error(f"Error creating RPC client for {blockchain.value} at {endpoint}: {e}")
            raise

        # Placeholder for POA chain adjustments if needed
        
        # Test connection
        logger.debug(f"Testing connection to {blockchain.value} node at {endpoint}...")
        if not w3.is_connected():
            logger.error(f"Failed to connect to {blockchain.value} node at {endpoint}")
            raise BlockchainConnectionError(f"Failed to connect to Polygon node at {endpoint}")
        logger.info(f"Successfully connected to {blockchain.value} node at {endpoint}")
        
        self.blockchain_clients[blockchain.value] = w3
        
        # Initialize PolygonScan API client if key is available
        polygonscan_key = self.api_keys.get(blockchain.value, {}).get('polygonscan')
        if polygonscan_key:
            logger.debug(f"Initializing PolygonScan client for {blockchain.value} with provided API key.")
            # PolygonScan API uses same format as Etherscan
            self.api_clients[f"{blockchain.value}_polygonscan"] = polygonscan_key  # Store key for HTTP requests
            logger.info(f"PolygonScan client for {blockchain.value} initialized (key stored).")
        else:
            logger.debug(f"No PolygonScan API key found for {blockchain.value}.")
        
        # Get initial block height
        self.last_block_heights[blockchain.value] = w3.eth.block_number
    
    async def _periodic_collector(self, collector_method: Callable, interval: int):
        """
        Run a collector method periodically at specified interval.
        
        Args:
            collector_method: The collector method to run
            interval: Collection interval in seconds
        """
        while self.running:
            try:
                start_time = time.time()
                await collector_method()
                
                # Calculate sleep time, ensuring at least 1 second between runs
                elapsed = time.time() - start_time
                sleep_time = max(1, interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in collector {collector_method.__name__}: {str(e)}")
                logger.error(traceback.format_exc())
                # Add exponential backoff for repeated errors
                await asyncio.sleep(min(300, interval * 2))
    
    async def _transaction_processor(self):
        """Process blockchain transactions from the queue"""
        while self.running:
            try:
                # Get transaction from queue with timeout
                tx = await asyncio.wait_for(self.tx_queue.get(), timeout=1.0)
                
                # Skip if None (can happen when queue is empty and timeout)
                if tx is None:
                    continue
                
                # Process the transaction through the pipeline
                processed_tx = tx
                for process_step in self.tx_processing_pipeline:
                    processed_tx = await process_step(processed_tx)
                    # If processing step returned None, skip further processing
                    if processed_tx is None:
                        break
                
                # If we have processed transaction, store it
                if processed_tx is not None:
                    await self._store_transaction(processed_tx)
                
                # Mark task as done
                self.tx_queue.task_done()
            
            except asyncio.TimeoutError:
                # This is expected when the queue is empty
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in transaction processor: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(1)  # Prevent tight loop if there's an error
    
    async def _publish_whale_alerts(self):
        """Publish whale alerts from the queue"""
        while self.running:
            try:
                # Get whale alert from queue with timeout
                alert = await asyncio.wait_for(self.whale_queue.get(), timeout=1.0)
                
                # Skip if None
                if alert is None:
                    continue
                
                # Convert to dictionary for storage and publication
                alert_dict = alert.to_dict()
                
                # Publish to Redis for real-time consumers
                channel = f"onchain:whale_alerts:{alert.blockchain.value}"
                await self.redis_client.publish(channel, json.dumps(alert_dict))
                
                # Also publish to a general whale alerts channel
                await self.redis_client.publish("onchain:whale_alerts", json.dumps(alert_dict))
                
                # Store in database for historical analysis
                collection = "onchain_whale_alerts"
                await self.db_client.insert_one(collection, alert_dict)
                
                # Mark task as done
                self.whale_queue.task_done()
                
            except asyncio.TimeoutError:
                # This is expected when the queue is empty
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in whale alert publisher: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(1)  # Prevent tight loop if there's an error
    
    async def _publish_metrics(self):
        """Publish blockchain metrics from the queue"""
        while self.running:
            try:
                # Get metric from queue with timeout
                metric = await asyncio.wait_for(self.metric_queue.get(), timeout=1.0)
                
                # Skip if None
                if metric is None:
                    continue
                
                # Convert to dictionary for storage and publication
                metric_dict = metric.to_dict()
                
                # Publish to Redis for real-time consumers
                channel = f"onchain:metrics:{metric.blockchain.value}:{metric.metric_name}"
                await self.redis_client.publish(channel, json.dumps(metric_dict))
                
                # Store in database for historical analysis
                collection = f"onchain_metrics_{metric.blockchain.value}"
                await self.db_client.insert_one(collection, metric_dict)
                
                # Also store in time series for each related asset
                for asset in metric.related_assets:
                    # Create key for time series
                    ts_key = f"ts:onchain:metrics:{asset}:{metric.metric_name}"
                    
                    # Store as time series point (timestamp => value)
                    timestamp = int(metric.timestamp.timestamp())
                    value = metric.value
                    if isinstance(value, Decimal):
                        value = float(value)
                    elif not isinstance(value, (int, float)):
                        value = str(value)
                    
                    await self.redis_client.ts_add(ts_key, timestamp, value)
                
                # Store current value in hash for quick access
                hash_key = f"onchain:metrics:{metric.blockchain.value}"
                field = metric.metric_name
                value = metric.value
                if isinstance(value, Decimal):
                    value = str(value)
                elif isinstance(value, (datetime.datetime, datetime.date)):
                    value = value.isoformat()
                
                await self.redis_client.hset(hash_key, field, value)
                
                # Mark task as done
                self.metric_queue.task_done()
                
            except asyncio.TimeoutError:
                # This is expected when the queue is empty
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics publisher: {str(e)}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(1)  # Prevent tight loop if there's an error
    
    async def _store_transaction(self, tx: OnchainTransaction):
        """Store transaction in database and publish to Redis"""
        try:
            # Convert to dictionary for storage
            tx_dict = tx.to_dict()
            
            # Store in database
            collection = f"onchain_transactions_{tx.blockchain.value}"
            await self.db_client.insert_one(collection, tx_dict)
            
            # Publish to Redis for real-time consumers
            channel = f"onchain:transactions:{tx.blockchain.value}"
            await self.redis_client.publish(channel, json.dumps(tx_dict))
            
            # For contract interactions, publish to specific channel
            if tx.is_contract_interaction and tx.contract_address:
                contract_channel = f"onchain:contract:{tx.blockchain.value}:{tx.contract_address}"
                await self.redis_client.publish(contract_channel, json.dumps(tx_dict))
            
            # For token transfers, publish to token-specific channel
            if tx.is_token_transfer and tx.token_address:
                token_channel = f"onchain:token:{tx.blockchain.value}:{tx.token_address}"
                await self.redis_client.publish(token_channel, json.dumps(tx_dict))
            
            # Update address activity counters
            if tx.from_address:
                await self.redis_client.hincrby(f"onchain:address:{tx.blockchain.value}:{tx.from_address}", "tx_count", 1)
                await self.redis_client.hincrby(f"onchain:address:{tx.blockchain.value}:{tx.from_address}", "tx_out_count", 1)
            
            if tx.to_address:
                await self.redis_client.hincrby(f"onchain:address:{tx.blockchain.value}:{tx.to_address}", "tx_count", 1)
                await self.redis_client.hincrby(f"onchain:address:{tx.blockchain.value}:{tx.to_address}", "tx_in_count", 1)
            
            logger.debug(f"Stored transaction {tx.tx_hash} from blockchain {tx.blockchain.value}")
            
        except Exception as e:
            logger.error(f"Error storing transaction: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Transaction Processing Pipeline
    
    async def _enrich_transaction(self, tx: OnchainTransaction) -> OnchainTransaction:
        """
        Enrich transaction with additional data from external sources.
        
        Args:
            tx: The blockchain transaction
            
        Returns:
            Enriched transaction
        """
        if not tx:
            return tx
        
        try:
            # Get token price for USD value calculation if not already set
            if tx.usd_value is None and tx.value is not None:
                price_key = f"price:{tx.asset}"
                price_str = await self.redis_client.get(price_key)
                
                if price_str:
                    try:
                        price = Decimal(price_str)
                        tx.asset_price = price
                        tx.usd_value = tx.value * price
                    except (ValueError, TypeError, decimal.InvalidOperation):
                        logger.warning(f"Invalid price format for {tx.asset}: {price_str}")
            
            # Add address labels if available
            if tx.from_address:
                from_label = await self._get_address_label(tx.blockchain, tx.from_address)
                if from_label:
                    tx.from_label = from_label
            
            if tx.to_address:
                to_label = await self._get_address_label(tx.blockchain, tx.to_address)
                if to_label:
                    tx.to_label = to_label
            
            # Add method name for contract interactions if available (Ethereum-like chains)
            if tx.is_contract_interaction and tx.method_id and not tx.method_name:
                method_name = await self._get_method_name(tx.blockchain, tx.method_id)
                if method_name:
                    tx.method_name = method_name
            
            return tx
            
        except Exception as e:
            logger.warning(f"Error enriching transaction {tx.tx_hash}: {str(e)}")
            return tx  # Return original transaction if enrichment fails
    
    async def _analyze_transaction(self, tx: OnchainTransaction) -> OnchainTransaction:
        """
        Analyze transaction for trading relevance.
        
        Args:
            tx: The blockchain transaction
            
        Returns:
            Analyzed transaction or None if not relevant
        """
        if not tx:
            return None
        
        try:
            # Skip transactions with zero value unless they're contract interactions
            if tx.value == 0 and not tx.is_contract_interaction:
                return None
            
            # Determine trading-related assets
            for asset, blockchain_info in self.asset_blockchain_map.items():
                blockchains = blockchain_info if isinstance(blockchain_info, list) else [blockchain_info]
                
                if tx.blockchain in blockchains:
                    # Direct blockchain match (like BTC, ETH)
                    if isinstance(blockchain_info, BlockchainNetwork) and tx.asset == asset.split('/')[0]:
                        if 'related_assets' not in tx.__dict__:
                            tx.related_assets = []
                        tx.related_assets.append(asset)
                    
                    # Token matches (like USDT, USDC)
                    elif tx.is_token_transfer and tx.token_address:
                        # Check if token address matches any we're tracking
                        token_match = False
                        for contract_info in self.contract_addresses.get(tx.blockchain.value, []):
                            if contract_info.get('address', '').lower() == tx.token_address.lower():
                                if contract_info.get('symbol') == asset.split('/')[0]:
                                    token_match = True
                                    break
                        
                        if token_match:
                            if 'related_assets' not in tx.__dict__:
                                tx.related_assets = []
                            tx.related_assets.append(asset)
            
            # Skip if no related assets found
            if not hasattr(tx, 'related_assets') or not tx.related_assets:
                return None
            
            return tx
            
        except Exception as e:
            logger.warning(f"Error analyzing transaction {tx.tx_hash}: {str(e)}")
            return tx  # Return original transaction if analysis fails
    
    async def _detect_whale_movement(self, tx: OnchainTransaction) -> OnchainTransaction:
        """
        Detect whale movements in transaction.
        
        Args:
            tx: The blockchain transaction
            
        Returns:
            Transaction with whale movement detection
        """
        if not tx or not hasattr(tx, 'related_assets') or not tx.related_assets:
            return tx
        
        try:
            # Get whale threshold for this asset/blockchain
            whale_threshold = None
            for asset in tx.related_assets:
                asset_base = asset.split('/')[0]
                threshold = self.whale_thresholds.get(asset_base)
                if threshold:
                    whale_threshold = threshold
                    break
            
            # If no specific threshold, use blockchain default
            if not whale_threshold:
                whale_threshold = self.whale_thresholds.get(tx.blockchain.value, 1000000)  # $1M default
            
            # Check if this is a whale transaction
            if tx.usd_value and tx.usd_value >= Decimal(whale_threshold):
                tx.is_whale_tx = True
                
                # Create whale alert
                alert = WhaleAlert(
                    blockchain=tx.blockchain,
                    tx_hash=tx.tx_hash,
                    timestamp=tx.timestamp,
                    from_address=tx.from_address,
                    to_address=tx.to_address,
                    asset=tx.asset,
                    value=tx.value,
                    usd_value=tx.usd_value,
                    from_label=getattr(tx, 'from_label', None),
                    to_label=getattr(tx, 'to_label', None),
                    alert_type="movement",
                    related_assets=tx.related_assets
                )
                
                # Determine alert type more specifically
                if tx.from_label and "exchange" in tx.from_label.lower() and not (tx.to_label and "exchange" in tx.to_label.lower()):
                    alert.alert_type = "withdrawal"
                elif tx.to_label and "exchange" in tx.to_label.lower() and not (tx.from_label and "exchange" in tx.from_label.lower()):
                    alert.alert_type = "deposit"
                
                # Add to whale alert queue
                await self.whale_queue.put(alert)
                
                logger.info(f"Detected whale movement of {tx.value} {tx.asset} (${tx.usd_value}) in tx {tx.tx_hash}")
            
            return tx
            
        except Exception as e:
            logger.warning(f"Error detecting whale movement for {tx.tx_hash}: {str(e)}")
            return tx  # Return original transaction if detection fails
    
    async def _detect_contract_interaction(self, tx: OnchainTransaction) -> OnchainTransaction:
        """
        Analyze contract interactions for trading significance.
        
        Args:
            tx: The blockchain transaction
            
        Returns:
            Transaction with contract analysis
        """
        if not tx or not tx.is_contract_interaction:
            return tx
        
        try:
            # Skip if no contract address
            if not tx.contract_address:
                return tx
            
            # Check if this is a known contract we're monitoring
            contract_info = None
            for contract in self.contract_addresses.get(tx.blockchain.value, []):
                if contract.get('address', '').lower() == tx.contract_address.lower():
                    contract_info = contract
                    break
            
            # If not a specifically monitored contract, just return
            if not contract_info:
                return tx
            
            # Add contract metadata
            tx.contract_name = contract_info.get('name')
            tx.contract_type = contract_info.get('type')
            tx.contract_symbol = contract_info.get('symbol')
            
            # Check for specific method interactions
            if tx.method_id and contract_info.get('methods'):
                for method in contract_info['methods']:
                    if method.get('id') == tx.method_id:
                        tx.method_name = method.get('name')
                        tx.method_description = method.get('description')
                        
                        # If the method is marked as significant, add additional processing
                        if method.get('significant', False):
                            # Create a special alert or metric for significant contract interactions
                            metric = BlockchainMetric(
                                blockchain=tx.blockchain,
                                metric_name=f"contract_{tx.contract_name.lower().replace(' ', '_')}_{tx.method_name.lower()}",
                                timestamp=tx.timestamp,
                                value=1,  # Count of interactions
                                interval="event",
                                related_assets=tx.related_assets
                            )
                            
                            await self.metric_queue.put(metric)
                            
                            logger.info(f"Detected significant contract interaction: {tx.contract_name}.{tx.method_name} in tx {tx.tx_hash}")
                        
                        break
            
            return tx
            
        except Exception as e:
            logger.warning(f"Error analyzing contract interaction for {tx.tx_hash}: {str(e)}")
            return tx  # Return original transaction if analysis fails
    
    async def _detect_pattern(self, tx: OnchainTransaction) -> OnchainTransaction:
        """
        Detect patterns in transaction activity that could signal trading opportunities.
        
        Args:
            tx: The blockchain transaction
            
        Returns:
            Transaction with pattern detection
        """
        if not tx:
            return tx
        
        try:
            # This would implement more sophisticated pattern detection in production
            # For now, we just tag some basic patterns
            
            # Check for exchange withdrawal to exchange
            if (getattr(tx, 'from_label', None) and "exchange" in tx.from_label.lower() and 
                getattr(tx, 'to_label', None) and "exchange" in tx.to_label.lower()):
                tx.pattern_type = "exchange_to_exchange"
                tx.pattern_description = f"Transfer from {tx.from_label} to {tx.to_label}"
                tx.pattern_significance = 0.7
                
                logger.info(f"Detected exchange-to-exchange transfer in {tx.tx_hash}: {tx.from_label} -> {tx.to_label}")
            
            # Check for contract deployment
            elif tx.is_contract_interaction and not tx.to_address:
                tx.pattern_type = "contract_deployment"
                tx.pattern_description = "New contract deployment"
                tx.pattern_significance = 0.8
                
                logger.info(f"Detected contract deployment in {tx.tx_hash}")
            
            # Check for large token burns
            elif (tx.is_token_transfer and 
                  (tx.to_address.lower() == "0x0000000000000000000000000000000000000000" or 
                   tx.to_address.lower() == "0x000000000000000000000000000000000000dead")):
                tx.pattern_type = "token_burn"
                tx.pattern_description = f"Token burn of {tx.value} {tx.asset}"
                tx.pattern_significance = 0.9 if tx.usd_value and tx.usd_value > 100000 else 0.5
                
                logger.info(f"Detected token burn in {tx.tx_hash}: {tx.value} {tx.asset}")
            
            return tx
            
        except Exception as e:
            logger.warning(f"Error detecting patterns for {tx.tx_hash}: {str(e)}")
            return tx  # Return original transaction if detection fails
    
    # Helper methods
    
    async def _get_address_label(self, blockchain: BlockchainNetwork, address: str) -> Optional[str]:
        """
        Get label for an address from cache or external sources.
        
        Args:
            blockchain: The blockchain network
            address: The address to look up
            
        Returns:
            Label if available, None otherwise
        """
        # Check cache first
        cache_key = f"{blockchain.value}:{address.lower()}"
        if cache_key in self.address_labels:
            return self.address_labels[cache_key]
        
        # Check Redis
        label = await self.redis_client.get(f"onchain:address_labels:{cache_key}")
        if label:
            self.address_labels[cache_key] = label
            return label
        
        # If no cached label, try to get from external sources based on blockchain
        label = None
        
        try:
            if blockchain == BlockchainNetwork.ETHEREUM:
                # Try Etherscan if available
                etherscan = self.api_clients.get(f"{blockchain.value}_etherscan")
                if etherscan:
                    try:
                        # This is a simplified example; real implementation would use Etherscan API
                        # with proper rate limiting and error handling
                        pass
                    except Exception as e:
                        logger.warning(f"Etherscan address lookup error: {str(e)}")
            
            elif blockchain == BlockchainNetwork.BITCOIN:
                # For Bitcoin, check against known exchange addresses
                # This would typically use a dedicated address database or API
                pass
            
            # If no label found but address looks like a contract, mark it as such
            if not label and blockchain in [BlockchainNetwork.ETHEREUM, BlockchainNetwork.BINANCE_SMART_CHAIN, BlockchainNetwork.POLYGON]:
                w3 = self.blockchain_clients.get(blockchain.value)
                if w3:
                    code = w3.eth.get_code(to_checksum_address(address))
                    if code and code != '0x':  # Has contract code
                        label = "Contract"
            
            # If we found a label, cache it
            if label:
                self.address_labels[cache_key] = label
                await self.redis_client.set(f"onchain:address_labels:{cache_key}", label, ex=86400*7)  # Cache for 7 days
            
            return label
            
        except Exception as e:
            logger.warning(f"Error getting address label for {address}: {str(e)}")
            return None
    
    async def _get_method_name(self, blockchain: BlockchainNetwork, method_id: str) -> Optional[str]:
        """
        Get name for a contract method ID.
        
        Args:
            blockchain: The blockchain network
            method_id: The method ID (first 4 bytes of keccak hash of the function signature)
            
        Returns:
            Method name if available, None otherwise
        """
        # Check Redis cache
        cache_key = f"{blockchain.value}:{method_id}"
        method_name = await self.redis_client.get(f"onchain:method_names:{cache_key}")
        
        if method_name:
            return method_name
        
        # Common method IDs (could be expanded or loaded from a database)
        common_methods = {
            "0xa9059cbb": "transfer",
            "0x23b872dd": "transferFrom",
            "0x095ea7b3": "approve",
            "0x70a08231": "balanceOf",
            "0x18160ddd": "totalSupply",
            "0x313ce567": "decimals",
            "0x06fdde03": "name",
            "0x95d89b41": "symbol",
            "0x022c0d9f": "swap",
            "0x3593564c": "execute",
            "0x7ff36ab5": "swapExactETHForTokens",
            "0xb6f9de95": "swapExactTokensForETH",
            "0x5c11d795": "swapExactTokensForTokens",
            "0xfb3bdb41": "swapETHForExactTokens",
            "0x791ac947": "swapTokensForExactETH",
            "0x8803dbee": "swapTokensForExactTokens",
            "0xe8e33700": "addLiquidity",
            "0xf305d719": "addLiquidityETH",
            "0x0d295980": "stake",
            "0x2e1a7d4d": "withdraw",
            "0xbd9caa4d": "claim",
            "0xa694fc3a": "harvest",
            "0x60806040": "constructor"
        }
        
        if method_id in common_methods:
            method_name = common_methods[method_id]
            # Cache it
            await self.redis_client.set(f"onchain:method_names:{cache_key}", method_name, ex=86400*30)  # Cache for 30 days
            return method_name
        
        return None
    
    # Blockchain data collectors
    
    async def _collect_ethereum(self):
        """Collect data from the Ethereum blockchain"""
        blockchain = BlockchainNetwork.ETHEREUM
        logger.debug(f"Collecting data from {blockchain.value}")
        
        try:
            # Get Ethereum client
            w3 = self.blockchain_clients.get(blockchain.value)
            if not w3:
                logger.warning(f"No client available for {blockchain.value}")
                return
            
            # Get current block height
            current_block = w3.eth.block_number
            last_block = self.last_block_heights.get(blockchain.value, current_block - 10)
            
            # Don't process too many blocks at once
            max_blocks = 10
            if current_block - last_block > max_blocks:
                last_block = current_block - max_blocks
            
            # Process new blocks
            for block_num in range(last_block + 1, current_block + 1):
                try:
                    # Get block data
                    block = w3.eth.get_block(block_num, full_transactions=True)
                    
                    # Process transactions in the block
                    for tx_data in block.transactions:
                        try:
                            # Convert from AttributeDict to regular dict
                            tx_dict = dict(tx_data)
                            
                            # Get receipt for additional information
                            receipt = w3.eth.get_transaction_receipt(tx_dict['hash'])
                            
                            # Extract basic transaction information
                            value_eth = Decimal(tx_dict.get('value', 0)) / Decimal(10**18)
                            gas_price_gwei = Decimal(tx_dict.get('gasPrice', 0)) / Decimal(10**9)
                            
                            # Determine if this is a contract interaction
                            to_address = tx_dict.get('to')
                            is_contract = False
                            contract_address = None
                            method_id = None
                            
                            # Check if this is a contract creation
                            if to_address is None and 'creates' in tx_dict:
                                is_contract = True
                                contract_address = tx_dict['creates']
                            # Check if this is a contract interaction
                            elif to_address is not None:
                                # Get code at the address
                                code = w3.eth.get_code(to_address)
                                if code and code != '0x':
                                    is_contract = True
                                    contract_address = to_address
                                    
                                    # Extract method ID from input data
                                    input_data = tx_dict.get('input', '0x')
                                    if len(input_data) >= 10:  # 0x + 8 chars for method ID
                                        method_id = input_data[0:10]  # First 4 bytes after 0x
                            
                            # Determine if this is a token transfer (ERC20)
                            is_token_transfer = False
                            token_address = None
                            
                            # Check log events for ERC20 Transfer
                            if receipt and receipt.get('logs'):
                                for log in receipt['logs']:
                                    # ERC20 Transfer event has 3 topics:
                                    # 0: keccak(Transfer(address,address,uint256))
                                    # 1: from address
                                    # 2: to address
                                    topics = log.get('topics', [])
                                    if len(topics) == 3 and topics[0].hex() == '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef':
                                        is_token_transfer = True
                                        token_address = log.get('address')
                                        break
                            
                            # Create transaction object
                            tx = OnchainTransaction(
                                blockchain=blockchain,
                                tx_hash=tx_dict['hash'].hex(),
                                block_number=block_num,
                                timestamp=datetime.datetime.fromtimestamp(block.timestamp, tz=datetime.timezone.utc),
                                from_address=tx_dict.get('from', ''),
                                to_address=to_address if to_address else '',
                                value=value_eth,
                                asset="ETH",
                                gas_used=receipt.get('gasUsed', 0) if receipt else 0,
                                gas_price=tx_dict.get('gasPrice', 0),
                                is_contract_interaction=is_contract,
                                contract_address=contract_address,
                                method_id=method_id,
                                is_token_transfer=is_token_transfer,
                                token_address=token_address,
                                status=receipt.get('status', 1) == 1 if receipt else True
                            )
                            
                            # Add to transaction queue
                            await self.tx_queue.put(tx)
                            
                        except Exception as e:
                            logger.warning(f"Error processing Ethereum transaction: {str(e)}")
                            continue
                    
                    # Update last processed block
                    self.last_block_heights[blockchain.value] = block_num
                    
                except Exception as e:
                    logger.error(f"Error processing Ethereum block {block_num}: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error collecting Ethereum data: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _collect_bitcoin(self):
        """Collect data from the Bitcoin blockchain"""
        blockchain = BlockchainNetwork.BITCOIN
        logger.debug(f"Collecting data from {blockchain.value}")
        
        try:
            # Get Bitcoin client
            btc_client = self.blockchain_clients.get(blockchain.value)
            if not btc_client:
                logger.warning(f"No client available for {blockchain.value}")
                return
            
            # Get current block height
            current_info = btc_client.getblockchaininfo()
            current_block = current_info['blocks']
            last_block = self.last_block_heights.get(blockchain.value, current_block - 6)
            
            # Don't process too many blocks at once
            max_blocks = 6
            if current_block - last_block > max_blocks:
                last_block = current_block - max_blocks
            
            # Process new blocks
            for block_num in range(last_block + 1, current_block + 1):
                try:
                    # Get block hash
                    block_hash = btc_client.getblockhash(block_num)
                    
                    # Get block data with transaction details
                    block = btc_client.getblock(block_hash, 2)
                    
                    # Process transactions in the block
                    for tx_data in block['tx']:
                        try:
                            # Skip coinbase transactions
                            if 'coinbase' in tx_data['vin'][0]:
                                continue
                            
                            # Process inputs and outputs
                            total_input = Decimal(0)
                            total_output = Decimal(0)
                            
                            # Calculate input value
                            for vin in tx_data['vin']:
                                if 'txid' in vin and 'vout' in vin:
                                    prev_tx = btc_client.getrawtransaction(vin['txid'], True)
                                    prev_vout = prev_tx['vout'][vin['vout']]
                                    total_input += Decimal(prev_vout['value'])
                            
                            # Calculate output value and identify recipients
                            for vout in tx_data['vout']:
                                total_output += Decimal(vout['value'])
                            
                            # Calculate fee
                            fee = total_input - total_output
                            
                            # Since Bitcoin transactions can have multiple inputs and outputs,
                            # we'll create a simplified representation with the primary input and output
                            
                            # Get primary input (first one)
                            from_address = ""
                            if tx_data['vin'] and 'scriptSig' in tx_data['vin'][0] and 'addresses' in tx_data['vin'][0].get('scriptSig', {}):
                                from_address = tx_data['vin'][0]['scriptSig']['addresses'][0]
                            
                            # Get primary output (largest one that's not change)
                            to_address = ""
                            largest_output = Decimal(0)
                            
                            for vout in tx_data['vout']:
                                if 'scriptPubKey' in vout and 'addresses' in vout['scriptPubKey'] and vout['value'] > largest_output:
                                    # Skip likely change outputs (matching input address)
                                    output_address = vout['scriptPubKey']['addresses'][0]
                                    if output_address != from_address:
                                        largest_output = Decimal(vout['value'])
                                        to_address = output_address
                            
                            # If we couldn't identify a clear recipient, use the first output
                            if not to_address and tx_data['vout'] and 'scriptPubKey' in tx_data['vout'][0] and 'addresses' in tx_data['vout'][0]['scriptPubKey']:
                                to_address = tx_data['vout'][0]['scriptPubKey']['addresses'][0]
                            
                            # Create transaction object
                            tx = OnchainTransaction(
                                blockchain=blockchain,
                                tx_hash=tx_data['txid'],
                                block_number=block_num,
                                timestamp=datetime.datetime.fromtimestamp(block['time'], tz=datetime.timezone.utc),
                                from_address=from_address,
                                to_address=to_address,
                                value=largest_output,  # Use largest output as primary value
                                asset="BTC",
                                gas_used=0,  # N/A for Bitcoin
                                gas_price=0,  # N/A for Bitcoin
                                is_contract_interaction=False,  # N/A for Bitcoin
                            )
                            
                            # Add transaction fee
                            tx.fee = fee
                            
                            # Add additional Bitcoin-specific fields
                            tx.total_input = total_input
                            tx.total_output = total_output
                            tx.input_count = len(tx_data['vin'])
                            tx.output_count = len(tx_data['vout'])
                            
                            # Add to transaction queue
                            await self.tx_queue.put(tx)
                            
                        except Exception as e:
                            logger.warning(f"Error processing Bitcoin transaction: {str(e)}")
                            continue
                    
                    # Update last processed block
                    self.last_block_heights[blockchain.value] = block_num
                    
                except Exception as e:
                    logger.error(f"Error processing Bitcoin block {block_num}: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error collecting Bitcoin data: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _collect_binance_smart_chain(self):
        """Collect data from Binance Smart Chain"""
        # Implementation is very similar to Ethereum
        blockchain = BlockchainNetwork.BINANCE_SMART_CHAIN
        logger.debug(f"Collecting data from {blockchain.value}")
        
        try:
            # Get BSC client
            w3 = self.blockchain_clients.get(blockchain.value)
            if not w3:
                logger.warning(f"No client available for {blockchain.value}")
                return
            
            # Get current block height
            current_block = w3.eth.block_number
            last_block = self.last_block_heights.get(blockchain.value, current_block - 10)
            
            # Don't process too many blocks at once
            max_blocks = 10
            if current_block - last_block > max_blocks:
                last_block = current_block - max_blocks
            
            # Process new blocks - implementation similar to Ethereum
            for block_num in range(last_block + 1, current_block + 1):
                try:
                    # Get block data
                    block = w3.eth.get_block(block_num, full_transactions=True)
                    
                    # Process transactions in the block - similar to Ethereum
                    for tx_data in block.transactions:
                        try:
                            # Convert from AttributeDict to regular dict
                            tx_dict = dict(tx_data)
                            
                            # Get receipt for additional information
                            receipt = w3.eth.get_transaction_receipt(tx_dict['hash'])
                            
                            # Extract basic transaction information
                            value_bnb = Decimal(tx_dict.get('value', 0)) / Decimal(10**18)
                            gas_price_gwei = Decimal(tx_dict.get('gasPrice', 0)) / Decimal(10**9)
                            
                            # Determine if this is a contract interaction
                            to_address = tx_dict.get('to')
                            is_contract = False
                            contract_address = None
                            method_id = None
                            
                            # Check if this is a contract creation
                            if to_address is None and 'creates' in tx_dict:
                                is_contract = True
                                contract_address = tx_dict['creates']
                            # Check if this is a contract interaction
                            elif to_address is not None:
                                # Get code at the address
                                code = w3.eth.get_code(to_address)
                                if code and code != '0x':
                                    is_contract = True
                                    contract_address = to_address
                                    
                                    # Extract method ID from input data
                                    input_data = tx_dict.get('input', '0x')
                                    if len(input_data) >= 10:  # 0x + 8 chars for method ID
                                        method_id = input_data[0:10]  # First 4 bytes after 0x
                            
                            # Determine if this is a token transfer (ERC20)
                            is_token_transfer = False
                            token_address = None
                            
                            # Check log events for ERC20 Transfer
                            if receipt and receipt.get('logs'):
                                for log in receipt['logs']:
                                    # ERC20 Transfer event signature
                                    topics = log.get('topics', [])
                                    if len(topics) == 3 and topics[0].hex() == '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef':
                                        is_token_transfer = True
                                        token_address = log.get('address')
                                        break
                            
                            # Create transaction object
                            tx = OnchainTransaction(
                                blockchain=blockchain,
                                tx_hash=tx_dict['hash'].hex(),
                                block_number=block_num,
                                timestamp=datetime.datetime.fromtimestamp(block.timestamp, tz=datetime.timezone.utc),
                                from_address=tx_dict.get('from', ''),
                                to_address=to_address if to_address else '',
                                value=value_bnb,
                                asset="BNB",
                                gas_used=receipt.get('gasUsed', 0) if receipt else 0,
                                gas_price=tx_dict.get('gasPrice', 0),
                                is_contract_interaction=is_contract,
                                contract_address=contract_address,
                                method_id=method_id,
                                is_token_transfer=is_token_transfer,
                                token_address=token_address,
                                status=receipt.get('status', 1) == 1 if receipt else True
                            )
                            
                            # Add to transaction queue
                            await self.tx_queue.put(tx)
                            
                        except Exception as e:
                            logger.warning(f"Error processing BSC transaction: {str(e)}")
                            continue
                    
                    # Update last processed block
                    self.last_block_heights[blockchain.value] = block_num
                    
                except Exception as e:
                    logger.error(f"Error processing BSC block {block_num}: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error collecting BSC data: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _collect_polygon(self):
        """Collect data from the Polygon blockchain"""
        # Implementation is very similar to Ethereum - with minor differences
        blockchain = BlockchainNetwork.POLYGON
        logger.debug(f"Collecting data from {blockchain.value}")
        
        try:
            # Get Polygon client
            w3 = self.blockchain_clients.get(blockchain.value)
            if not w3:
                logger.warning(f"No client available for {blockchain.value}")
                return
            
            # Get current block height
            current_block = w3.eth.block_number
            last_block = self.last_block_heights.get(blockchain.value, current_block - 10)
            
            # Don't process too many blocks at once
            max_blocks = 10
            if current_block - last_block > max_blocks:
                last_block = current_block - max_blocks
            
            # Process new blocks
            for block_num in range(last_block + 1, current_block + 1):
                try:
                    # Get block data
                    block = w3.eth.get_block(block_num, full_transactions=True)
                    
                    # Process transactions in the block
                    for tx_data in block.transactions:
                        try:
                            # Process similar to Ethereum
                            tx_dict = dict(tx_data)
                            receipt = w3.eth.get_transaction_receipt(tx_dict['hash'])
                            
                            value_matic = Decimal(tx_dict.get('value', 0)) / Decimal(10**18)
                            
                            # Determine if contract interaction
                            to_address = tx_dict.get('to')
                            is_contract = False
                            contract_address = None
                            method_id = None
                            
                            if to_address is None and 'creates' in tx_dict:
                                is_contract = True
                                contract_address = tx_dict['creates']
                            elif to_address is not None:
                                code = w3.eth.get_code(to_address)
                                if code and code != '0x':
                                    is_contract = True
                                    contract_address = to_address
                                    
                                    input_data = tx_dict.get('input', '0x')
                                    if len(input_data) >= 10:
                                        method_id = input_data[0:10]
                            
                            # Check for ERC20 token transfers
                            is_token_transfer = False
                            token_address = None
                            
                            if receipt and receipt.get('logs'):
                                for log in receipt['logs']:
                                    topics = log.get('topics', [])
                                    if len(topics) == 3 and topics[0].hex() == '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef':
                                        is_token_transfer = True
                                        token_address = log.get('address')
                                        break
                            
                            # Create transaction object
                            tx = OnchainTransaction(
                                blockchain=blockchain,
                                tx_hash=tx_dict['hash'].hex(),
                                block_number=block_num,
                                timestamp=datetime.datetime.fromtimestamp(block.timestamp, tz=datetime.timezone.utc),
                                from_address=tx_dict.get('from', ''),
                                to_address=to_address if to_address else '',
                                value=value_matic,
                                asset="MATIC",
                                gas_used=receipt.get('gasUsed', 0) if receipt else 0,
                                gas_price=tx_dict.get('gasPrice', 0),
                                is_contract_interaction=is_contract,
                                contract_address=contract_address,
                                method_id=method_id,
                                is_token_transfer=is_token_transfer,
                                token_address=token_address,
                                status=receipt.get('status', 1) == 1 if receipt else True
                            )
                            
                            # Add to transaction queue
                            await self.tx_queue.put(tx)
                            
                        except Exception as e:
                            logger.warning(f"Error processing Polygon transaction: {str(e)}")
                            continue
                    
                    # Update last processed block
                    self.last_block_heights[blockchain.value] = block_num
                    
                except Exception as e:
                    logger.error(f"Error processing Polygon block {block_num}: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error collecting Polygon data: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _collect_solana(self):
        """Collect data from the Solana blockchain"""
        blockchain = BlockchainNetwork.SOLANA
        logger.debug(f"Collecting data from {blockchain.value}")
        
        try:
            # In a production implementation, we would:
            # 1. Connect to Solana RPC API
            # 2. Get recent transaction signatures
            # 3. Get transaction details for each signature
            # 4. Parse and standardize the data
            # 5. Add to transaction queue
            
            # For this implementation, we'll keep it as a placeholder
            # since Solana's architecture is quite different from EVM-based chains
            logger.info("Solana data collection is implemented as a placeholder")
            
            # Track that we ran the collector
            self.last_collect_time = datetime.datetime.now()
            
        except Exception as e:
            logger.error(f"Error collecting Solana data: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _collect_avalanche(self):
        """Collect data from the Avalanche blockchain"""
        # Implementation similar to Ethereum - Avalanche C-Chain is EVM compatible
        blockchain = BlockchainNetwork.AVALANCHE
        logger.debug(f"Collecting data from {blockchain.value}")
        
        try:
            # Get Avalanche client
            w3 = self.blockchain_clients.get(blockchain.value)
            if not w3:
                logger.warning(f"No client available for {blockchain.value}")
                return
            
            # Get current block height
            current_block = w3.eth.block_number
            last_block = self.last_block_heights.get(blockchain.value, current_block - 10)
            
            # Don't process too many blocks at once
            max_blocks = 10
            if current_block - last_block > max_blocks:
                last_block = current_block - max_blocks
            
            # Process new blocks - similar to Ethereum implementation
            for block_num in range(last_block + 1, current_block + 1):
                try:
                    # Get block data
                    block = w3.eth.get_block(block_num, full_transactions=True)
                    
                    # Process transactions in the block (similar to Ethereum implementation)
                    for tx_data in block.transactions:
                        try:
                            tx_dict = dict(tx_data)
                            receipt = w3.eth.get_transaction_receipt(tx_dict['hash'])
                            
                            value_avax = Decimal(tx_dict.get('value', 0)) / Decimal(10**18)
                            
                            # Determine if this is a contract interaction
                            to_address = tx_dict.get('to')
                            is_contract = False
                            contract_address = None
                            method_id = None
                            
                            if to_address is None and 'creates' in tx_dict:
                                is_contract = True
                                contract_address = tx_dict['creates']
                            elif to_address is not None:
                                code = w3.eth.get_code(to_address)
                                if code and code != '0x':
                                    is_contract = True
                                    contract_address = to_address
                                    
                                    input_data = tx_dict.get('input', '0x')
                                    if len(input_data) >= 10:
                                        method_id = input_data[0:10]
                            
                            # Check for token transfers
                            is_token_transfer = False
                            token_address = None
                            
                            if receipt and receipt.get('logs'):
                                for log in receipt['logs']:
                                    topics = log.get('topics', [])
                                    if len(topics) == 3 and topics[0].hex() == '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef':
                                        is_token_transfer = True
                                        token_address = log.get('address')
                                        break
                            
                            # Create transaction object
                            tx = OnchainTransaction(
                                blockchain=blockchain,
                                tx_hash=tx_dict['hash'].hex(),
                                block_number=block_num,
                                timestamp=datetime.datetime.fromtimestamp(block.timestamp, tz=datetime.timezone.utc),
                                from_address=tx_dict.get('from', ''),
                                to_address=to_address if to_address else '',
                                value=value_avax,
                                asset="AVAX",
                                gas_used=receipt.get('gasUsed', 0) if receipt else 0,
                                gas_price=tx_dict.get('gasPrice', 0),
                                is_contract_interaction=is_contract,
                                contract_address=contract_address,
                                method_id=method_id,
                                is_token_transfer=is_token_transfer,
                                token_address=token_address,
                                status=receipt.get('status', 1) == 1 if receipt else True
                            )
                            
                            # Add to transaction queue
                            await self.tx_queue.put(tx)
                            
                        except Exception as e:
                            logger.warning(f"Error processing Avalanche transaction: {str(e)}")
                            continue
                    
                    # Update last processed block
                    self.last_block_heights[blockchain.value] = block_num
                    
                except Exception as e:
                    logger.error(f"Error processing Avalanche block {block_num}: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Error collecting Avalanche data: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _collect_blockchain_metrics(self):
        """Collect general blockchain metrics for all monitored chains"""
        logger.debug("Collecting blockchain metrics")
        
        # Get current time
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        
        # Collect metrics for each supported blockchain
        for blockchain in self.blockchains:
            try:
                # Get appropriate collector method
                collector_method = getattr(self, f"_collect_{blockchain}_metrics", None)
                if collector_method:
                    await collector_method(now)
                else:
                    # Generic metrics collection if no specific method
                    await self._collect_generic_metrics(blockchain, now)
            except Exception as e:
                logger.error(f"Error collecting metrics for {blockchain}: {str(e)}")
                logger.error(traceback.format_exc())
    
    async def _collect_ethereum_metrics(self, timestamp):
        """Collect Ethereum-specific metrics"""
        blockchain = BlockchainNetwork.ETHEREUM
        
        try:
            # Get Ethereum client
            w3 = self.blockchain_clients.get(blockchain.value)
            if not w3:
                logger.warning(f"No client available for {blockchain.value}")
                return
            
            # Gas price metrics
            gas_price_wei = w3.eth.gas_price
            gas_price_gwei = Decimal(gas_price_wei) / Decimal(10**9)
            
            gas_metric = BlockchainMetric(
                blockchain=blockchain,
                metric_name="gas_price_gwei",
                timestamp=timestamp,
                value=gas_price_gwei,
                interval="current",
                related_assets=["ETH/USD"]
            )
            
            await self.metric_queue.put(gas_metric)
            
            # Current block details
            block_number = w3.eth.block_number
            block = w3.eth.get_block(block_number)
            
            # Block time (time since last block)
            if block_number > 0:
                previous_block = w3.eth.get_block(block_number - 1)
                block_time = block.timestamp - previous_block.timestamp
                
                block_time_metric = BlockchainMetric(
                    blockchain=blockchain,
                    metric_name="block_time",
                    timestamp=timestamp,
                    value=block_time,
                    interval="current",
                    related_assets=["ETH/USD"]
                )
                
                await self.metric_queue.put(block_time_metric)
            
            # Transactions in last block
            tx_count = len(block.transactions)
            
            tx_count_metric = BlockchainMetric(
                blockchain=blockchain,
                metric_name="tx_count_per_block",
                timestamp=timestamp,
                value=tx_count,
                interval="current",
                related_assets=["ETH/USD"]
            )
            
            await self.metric_queue.put(tx_count_metric)
            
            # Try to get additional metrics from Etherscan if available
            etherscan = self.api_clients.get(f"{blockchain.value}_etherscan")
            if etherscan:
                # This would use the Etherscan API to get additional metrics
                # For example, pending transactions, network hashrate, etc.
                # For this implementation, we'll skip as it requires API calls
                pass
                
        except Exception as e:
            logger.error(f"Error collecting Ethereum metrics: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _collect_bitcoin_metrics(self, timestamp):
        """Collect Bitcoin-specific metrics"""
        blockchain = BlockchainNetwork.BITCOIN
        
        try:
            # Get Bitcoin client
            btc_client = self.blockchain_clients.get(blockchain.value)
            if not btc_client:
                logger.warning(f"No client available for {blockchain.value}")
                return
            
            # Blockchain info
            blockchain_info = btc_client.getblockchaininfo()
            
            # Network hashrate
            network_info = btc_client.getnetworkinfo()
            mining_info = btc_client.getmininginfo()
            
            # Difficulty metric
            difficulty_metric = BlockchainMetric(
                blockchain=blockchain,
                metric_name="difficulty",
                timestamp=timestamp,
                value=Decimal(str(blockchain_info['difficulty'])),
                interval="current",
                related_assets=["BTC/USD"]
            )
            
            await self.metric_queue.put(difficulty_metric)
            
            # Hashrate metric
            if 'networkhashps' in mining_info:
                hashrate_metric = BlockchainMetric(
                    blockchain=blockchain,
                    metric_name="network_hashrate",
                    timestamp=timestamp,
                    value=Decimal(str(mining_info['networkhashps'])),
                    interval="current",
                    related_assets=["BTC/USD"]
                )
                
                await self.metric_queue.put(hashrate_metric)
            
            # Mempool metrics
            mempool_info = btc_client.getmempoolinfo()
            
            mempool_size_metric = BlockchainMetric(
                blockchain=blockchain,
                metric_name="mempool_size",
                timestamp=timestamp,
                value=mempool_info['size'],
                interval="current",
                related_assets=["BTC/USD"]
            )
            
            await self.metric_queue.put(mempool_size_metric)
            
            mempool_bytes_metric = BlockchainMetric(
                blockchain=blockchain,
                metric_name="mempool_bytes",
                timestamp=timestamp,
                value=mempool_info['bytes'],
                interval="current",
                related_assets=["BTC/USD"]
            )
            
            await self.metric_queue.put(mempool_bytes_metric)
            
        except Exception as e:
            logger.error(f"Error collecting Bitcoin metrics: {str(e)}")
            logger.error(traceback.format_exc())
    
    async def _collect_generic_metrics(self, blockchain, timestamp):
        """Collect generic metrics for any blockchain"""
        blockchain_enum = BlockchainNetwork(blockchain)
        
        try:
            # For EVM-compatible chains
            if blockchain_enum in [BlockchainNetwork.ETHEREUM, BlockchainNetwork.BINANCE_SMART_CHAIN, 
                                 BlockchainNetwork.POLYGON, BlockchainNetwork.AVALANCHE]:
                
                w3 = self.blockchain_clients.get(blockchain)
                if not w3:
                    return
                
                # Block number metric
                block_number = w3.eth.block_number
                
                block_number_metric = BlockchainMetric(
                    blockchain=blockchain_enum,
                    metric_name="block_number",
                    timestamp=timestamp,
                    value=block_number,
                    interval="current",
                    related_assets=[f"{blockchain_enum.name}/USD"]
                )
                
                await self.metric_queue.put(block_number_metric)
                
                # Gas price metric
                gas_price_wei = w3.eth.gas_price
                gas_price_gwei = Decimal(gas_price_wei) / Decimal(10**9)
                
                gas_metric = BlockchainMetric(
                    blockchain=blockchain_enum,
                    metric_name="gas_price_gwei",
                    timestamp=timestamp,
                    value=gas_price_gwei,
                    interval="current",
                    related_assets=[f"{blockchain_enum.name}/USD"]
                )
                
                await self.metric_queue.put(gas_metric)
            
        except Exception as e:
            logger.error(f"Error collecting generic metrics for {blockchain}: {str(e)}")
            logger.error(traceback.format_exc())
