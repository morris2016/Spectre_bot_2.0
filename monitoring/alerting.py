
#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Alerting System

This module handles the creation, management, and delivery of alerts for system,
trading, and security events.
"""

import os
import json
import asyncio
import smtplib
import requests
import time
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from common.logger import get_logger
from common.utils import generate_id, truncate_float, escape_html
from common.constants import (
    ALERT_LEVELS, ALERT_TYPES, ALERT_CHANNELS, 
    TRADING_ALERT_THRESHOLDS, SYSTEM_ALERT_THRESHOLDS,
    MAX_ALERT_HISTORY, ALERT_COOLDOWN_PERIODS
)
from common.db_client import DatabaseClient, get_db_client
from common.redis_client import RedisClient
from common.exceptions import (
    AlertDeliveryError, AlertConfigurationError, 
    TemplateRenderError, ServiceConnectionError
)

logger = get_logger("alerting")

class AlertingSystem:
    """
    Comprehensive alerting system that detects conditions requiring attention,
    manages alert states, and delivers notifications through multiple channels.
    """
    
    def __init__(self, config: Dict[str, Any], db_client: DatabaseClient = None, 
                 redis_client: RedisClient = None):
        """
        Initialize the AlertingSystem with configuration and clients.
        
        Args:
            config: Configuration dictionary
            db_client: Database client for alert storage and history
            redis_client: Redis client for real-time alerts and state
        """
        self.config = config
        self.db_client = db_client
        self._db_params = config
        self.redis_client = redis_client or RedisClient(config)
        
        # Alert configuration
        self.alert_config = config.get("alerting", {})
        self.enabled_channels = self.alert_config.get("enabled_channels", ["console"])
        self.alert_cooldown = self.alert_config.get("cooldown_periods", ALERT_COOLDOWN_PERIODS)
        
        # Voice advisor configuration
        self.voice_advisor_enabled = self.alert_config.get("voice_advisor", {}).get("enabled", False)
        self.voice_advisor_host = self.alert_config.get("voice_advisor", {}).get("host", "localhost")
        self.voice_advisor_port = self.alert_config.get("voice_advisor", {}).get("port", 8085)
        
        # Email configuration
        self.email_config = self.alert_config.get("email", {})
        self.smtp_server = self.email_config.get("smtp_server")
        self.smtp_port = self.email_config.get("smtp_port", 587)
        self.smtp_username = self.email_config.get("username")
        self.smtp_password = self.email_config.get("password")
        self.email_from = self.email_config.get("from_address")
        self.email_recipients = self.email_config.get("recipients", [])
        
        # Webhook configuration
        self.webhook_config = self.alert_config.get("webhook", {})
        self.webhook_urls = self.webhook_config.get("urls", {})
        
        # SMS configuration
        self.sms_config = self.alert_config.get("sms", {})
        self.sms_provider = self.sms_config.get("provider")
        self.sms_api_key = self.sms_config.get("api_key")
        self.sms_from = self.sms_config.get("from_number")
        self.sms_recipients = self.sms_config.get("recipients", [])
        
        # Alert templates
        self.templates = self._load_templates()
        
        # In-memory alert state
        self.active_alerts = {}
        self.alert_history = []
        self.last_alert_times = {}

        # Alert tasks
        self.alert_task = None
        self.is_running = False

        logger.info("AlertingSystem initialized")

    async def initialize(self, db_connector: Optional[DatabaseClient] = None) -> None:
        """Obtain a database client and ensure tables exist."""
        if db_connector is not None:
            self.db_client = db_connector
        if self.db_client is None:
            self.db_client = await get_db_client(**self._db_params)
        if getattr(self.db_client, "pool", None) is None:
            await self.db_client.initialize()
            await self.db_client.create_tables()
    
    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """
        Load alert templates for different channels and alert types.
        
        Returns:
            Dictionary of templates organized by type and channel
        """
        templates = {}
        
        # Default templates for different alert types and channels
        for alert_type in ALERT_TYPES:
            templates[alert_type] = {}
            
            for channel in ALERT_CHANNELS:
                # Load from config if available
                template_config = self.alert_config.get("templates", {}).get(alert_type, {}).get(channel)
                
                if template_config:
                    templates[alert_type][channel] = template_config
                else:
                    # Default templates
                    if channel == "email":
                        templates[alert_type][channel] = {
                            "subject": f"QuantumSpectre Elite: {alert_type.upper()} Alert",
                            "body": "Alert: {{alert_message}}\nLevel: {{alert_level}}\nTime: {{timestamp}}\nDetails: {{details}}"
                        }
                    elif channel == "sms":
                        templates[alert_type][channel] = {
                            "body": "QuantumSpectre: {{alert_level}} {{alert_type}} - {{alert_message}}"
                        }
                    elif channel == "webhook":
                        templates[alert_type][channel] = {
                            "format": "json"
                        }
                    elif channel == "voice":
                        templates[alert_type][channel] = {
                            "script": "Alert: {{alert_message}}. Level: {{alert_level}}."
                        }
                    else:  # console and others
                        templates[alert_type][channel] = {
                            "format": "text"
                        }
        
        return templates
    
    async def start(self) -> None:
        """Start the alerting system."""
        if self.is_running:
            logger.warning("AlertingSystem is already running")
            return
        
        logger.info("Starting AlertingSystem...")
        await self.initialize()
        self.is_running = True
        
        # Load active alerts from storage
        await self._load_active_alerts()
        
        # Start background alert monitor task
        self.alert_task = asyncio.create_task(self._monitor_alerts())
        
        logger.info("AlertingSystem started successfully")
    
    async def stop(self) -> None:
        """Stop the alerting system."""
        if not self.is_running:
            logger.warning("AlertingSystem is not running")
            return
        
        logger.info("Stopping AlertingSystem...")
        self.is_running = False
        
        # Cancel background task
        if self.alert_task:
            self.alert_task.cancel()
        
        # Persist alerts to database
        await self._persist_alerts()
        
        logger.info("AlertingSystem stopped successfully")
    
    async def _load_active_alerts(self) -> None:
        """Load active alerts from the database."""
        try:
            # Load alerts that haven't been resolved
            active_alerts = await self.db_client.find("alerts", {"resolved": False})
            
            for alert in active_alerts:
                alert_id = alert.get("alert_id")
                if alert_id:
                    self.active_alerts[alert_id] = alert
            
            logger.info(f"Loaded {len(self.active_alerts)} active alerts from database")
            
            # Load recent alert history
            cutoff_time = datetime.now() - timedelta(days=1)
            query = {"timestamp": {"$gte": cutoff_time.isoformat()}}
            recent_alerts = await self.db_client.find("alerts", query, sort=[("timestamp", -1)], limit=MAX_ALERT_HISTORY)
            
            self.alert_history = recent_alerts
            
            # Load last alert times for cooldown
            for alert in recent_alerts:
                key = f"{alert.get('alert_type')}:{alert.get('source')}:{alert.get('alert_level')}"
                timestamp = alert.get("timestamp")
                
                if key and timestamp:
                    try:
                        alert_time = datetime.fromisoformat(timestamp)
                        self.last_alert_times[key] = alert_time.timestamp()
                    except ValueError:
                        continue
            
        except Exception as e:
            logger.error(f"Failed to load active alerts: {str(e)}")
    
    async def _persist_alerts(self) -> None:
        """Persist active alerts to the database."""
        try:
            # Only persist alerts that have been modified
            for alert_id, alert in self.active_alerts.items():
                if alert.get("modified", False):
                    await self.db_client.update_one(
                        "alerts", 
                        {"alert_id": alert_id}, 
                        {"$set": alert},
                        upsert=True
                    )
            
            logger.info(f"Persisted active alerts to database")
            
        except Exception as e:
            logger.error(f"Failed to persist alerts: {str(e)}")
    
    async def _monitor_alerts(self) -> None:
        """Monitor and process alerts in the background."""
        logger.info("Starting alert monitoring task")
        
        while self.is_running:
            try:
                # Process active alerts
                await self._process_active_alerts()
                
                # Check for alert expiration
                await self._check_alert_expiration()
                
                # Periodically persist alerts
                if np.random.random() < 0.1:  # 10% chance each cycle to reduce DB load
                    await self._persist_alerts()
                
            except Exception as e:
                logger.exception(f"Error in alert monitoring: {str(e)}")
            
            await asyncio.sleep(5)  # Check every 5 seconds
    
    async def _process_active_alerts(self) -> None:
        """Process active alerts, check for updates or resolution."""
        for alert_id, alert in list(self.active_alerts.items()):
            try:
                # Skip already resolved alerts
                if alert.get("resolved", False):
                    continue
                
                # Check if auto-resolution is enabled for this alert
                auto_resolve = alert.get("auto_resolve", False)
                
                if auto_resolve:
                    # Check conditions for auto-resolution
                    if alert.get("alert_type") == "system":
                        # For system alerts, check if the condition is still true
                        resolved = await self._check_system_alert_resolved(alert)
                    elif alert.get("alert_type") == "trading":
                        # For trading alerts, check if the condition is still true
                        resolved = await self._check_trading_alert_resolved(alert)
                    else:
                        # Default to not auto-resolving
                        resolved = False
                    
                    if resolved:
                        await self.resolve_alert(alert_id, "Auto-resolved: condition no longer true")
                
                # Check for escalation
                await self._check_alert_escalation(alert_id, alert)
                
            except Exception as e:
                logger.error(f"Error processing alert {alert_id}: {str(e)}")
    
    async def _check_system_alert_resolved(self, alert: Dict[str, Any]) -> bool:
        """
        Check if a system alert condition is no longer true.
        
        Args:
            alert: The alert to check
            
        Returns:
            True if the alert should be resolved, False otherwise
        """
        try:
            source = alert.get("source")
            metric = alert.get("details", {}).get("metric")
            threshold = alert.get("details", {}).get("threshold")
            
            if not all([source, metric, threshold]):
                return False
            
            # Get current metric value
            metrics = await self._get_current_metrics(source)
            
            if not metrics or metric not in metrics:
                return False
            
            current_value = metrics[metric]
            comparison = alert.get("details", {}).get("comparison", ">")
            
            # Check if the condition is no longer true
            if comparison == ">" and current_value <= threshold:
                return True
            elif comparison == ">=" and current_value < threshold:
                return True
            elif comparison == "<" and current_value >= threshold:
                return True
            elif comparison == "<=" and current_value > threshold:
                return True
            elif comparison == "==" and current_value != threshold:
                return True
            elif comparison == "!=" and current_value == threshold:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking system alert resolution: {str(e)}")
            return False
    
    async def _check_trading_alert_resolved(self, alert: Dict[str, Any]) -> bool:
        """
        Check if a trading alert condition is no longer true.
        
        Args:
            alert: The alert to check
            
        Returns:
            True if the alert should be resolved, False otherwise
        """
        try:
            # For pattern alerts, they're usually one-time notifications
            if alert.get("details", {}).get("pattern"):
                # Time-based auto-resolution
                alert_time = datetime.fromisoformat(alert.get("timestamp", datetime.now().isoformat()))
                current_time = datetime.now()
                
                # Auto-resolve pattern alerts after 30 minutes
                if (current_time - alert_time).total_seconds() > 1800:
                    return True
            
            # For position alerts, check if the position is still active
            position_id = alert.get("details", {}).get("position_id")
            if position_id:
                # Check if position is still active (implementation depends on your system)
                position_active = await self._check_position_active(position_id)
                return not position_active
            
            # For other trading alerts, time-based resolution
            alert_time = datetime.fromisoformat(alert.get("timestamp", datetime.now().isoformat()))
            current_time = datetime.now()
            
            # Auto-resolve after 1 hour by default
            if (current_time - alert_time).total_seconds() > 3600:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking trading alert resolution: {str(e)}")
            return False
    
    async def _check_position_active(self, position_id: str) -> bool:
        """
        Check if a trading position is still active.
        
        Args:
            position_id: ID of the position to check
            
        Returns:
            True if the position is active, False otherwise
        """
        try:
            # This would connect to your position management system
            # For demonstration, we'll simulate a lookup
            position = await self.db_client.find_one("positions", {"position_id": position_id})
            
            if not position:
                return False
            
            return position.get("status") == "active"
            
        except Exception as e:
            logger.error(f"Error checking position status: {str(e)}")
            return False  # Default to closed/not active
    
    async def _get_current_metrics(self, service_name: str) -> Dict[str, Any]:
        """
        Get current metrics for a service.
        
        Args:
            service_name: Name of the service to get metrics for
            
        Returns:
            Dictionary of current metrics
        """
        try:
            # Try to get from Redis for real-time data
            key = f"metrics:latest:{service_name}"
            metrics = await self.redis_client.get(key)
            
            if metrics:
                return metrics.get("metrics", {})
            
            # Fallback to database
            latest_metrics = await self.db_client.find_one(
                "system_metrics",
                {"service_name": service_name},
                sort=[("timestamp", -1)]
            )
            
            if latest_metrics:
                return latest_metrics.get("metrics", {})
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting current metrics: {str(e)}")
            return {}
    
    async def _check_alert_escalation(self, alert_id: str, alert: Dict[str, Any]) -> None:
        """
        Check if an alert should be escalated based on duration and lack of attention.
        
        Args:
            alert_id: ID of the alert to check
            alert: Alert data
        """
        # Skip if alert doesn't have auto-escalation
        if not alert.get("auto_escalate", False):
            return
        
        try:
            # Check how long the alert has been active
            alert_time = datetime.fromisoformat(alert.get("timestamp", datetime.now().isoformat()))
            current_time = datetime.now()
            alert_duration = (current_time - alert_time).total_seconds()
            
            # Get current level and max level
            current_level = alert.get("alert_level", "info")
            current_level_idx = ALERT_LEVELS.index(current_level) if current_level in ALERT_LEVELS else 0
            max_level_idx = len(ALERT_LEVELS) - 1
            
            # Escalation thresholds based on alert type
            escalation_intervals = {
                "system": [300, 900, 1800],  # 5min, 15min, 30min
                "trading": [60, 300, 600],   # 1min, 5min, 10min
                "security": [60, 300, 600]   # 1min, 5min, 10min
            }
            
            alert_type = alert.get("alert_type", "system")
            intervals = escalation_intervals.get(alert_type, [300, 900, 1800])
            
            # Determine if escalation is needed
            for i, threshold in enumerate(intervals):
                escalation_level = min(current_level_idx + i + 1, max_level_idx)
                
                if alert_duration > threshold and current_level_idx < escalation_level:
                    # Escalate to the new level
                    new_level = ALERT_LEVELS[escalation_level]
                    
                    # Update the alert
                    alert["alert_level"] = new_level
                    alert["modified"] = True
                    alert["escalation_history"] = alert.get("escalation_history", []) + [{
                        "timestamp": current_time.isoformat(),
                        "from_level": current_level,
                        "to_level": new_level,
                        "reason": f"Auto-escalated after {alert_duration:.0f} seconds"
                    }]
                    
                    # Send notification about escalation
                    escalation_message = f"Alert escalated from {current_level} to {new_level} after {alert_duration:.0f} seconds"
                    await self._send_alert_notification(alert, escalation_message)
                    
                    logger.info(f"Escalated alert {alert_id} from {current_level} to {new_level}")
                    break
            
        except Exception as e:
            logger.error(f"Error checking alert escalation for {alert_id}: {str(e)}")
    
    async def _check_alert_expiration(self) -> None:
        """Check for expired alerts and mark them as resolved."""
        current_time = datetime.now()
        
        for alert_id, alert in list(self.active_alerts.items()):
            try:
                # Skip already resolved alerts
                if alert.get("resolved", False):
                    continue
                
                # Check if alert has expiration
                expiration = alert.get("expiration")
                if not expiration:
                    continue
                
                try:
                    expiration_time = datetime.fromisoformat(expiration)
                    if current_time >= expiration_time:
                        await self.resolve_alert(alert_id, "Expired automatically")
                except ValueError:
                    logger.error(f"Invalid expiration format for alert {alert_id}")
                
            except Exception as e:
                logger.error(f"Error checking alert expiration for {alert_id}: {str(e)}")
    
    def _render_template(self, template: Dict[str, Any], alert: Dict[str, Any]) -> Dict[str, str]:
        """
        Render an alert template with the alert data.
        
        Args:
            template: Template configuration
            alert: Alert data
            
        Returns:
            Dictionary with rendered template fields
        """
        try:
            result = {}
            
            for field, template_str in template.items():
                if not isinstance(template_str, str):
                    result[field] = template_str
                    continue
                
                # Basic template rendering
                rendered = template_str
                
                # Replace alert fields
                for key, value in alert.items():
                    if isinstance(value, (str, int, float, bool)):
                        placeholder = f"{{{{{key}}}}}"
                        rendered = rendered.replace(placeholder, str(value))
                
                # Replace nested fields in details
                if "details" in alert and isinstance(alert["details"], dict):
                    for key, value in alert["details"].items():
                        if isinstance(value, (str, int, float, bool)):
                            placeholder = f"{{{{details.{key}}}}}"
                            rendered = rendered.replace(placeholder, str(value))
                
                result[field] = rendered
            
            return result
            
        except Exception as e:
            logger.error(f"Error rendering template: {str(e)}")
            raise TemplateRenderError(f"Failed to render template: {str(e)}")
    
    async def _send_email_alert(self, alert: Dict[str, Any], template: Dict[str, str]) -> bool:
        """
        Send an alert via email.
        
        Args:
            alert: Alert data
            template: Rendered template
            
        Returns:
            True if the email was sent successfully, False otherwise
        """
        if not self.smtp_server or not self.smtp_username or not self.smtp_password or not self.email_recipients:
            logger.warning("Email configuration incomplete, skipping email alert")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = template.get("subject", f"QuantumSpectre Alert: {alert.get('alert_level', 'Alert')}")
            msg["From"] = self.email_from
            msg["To"] = ", ".join(self.email_recipients)
            
            # Create HTML version
            html_body = f"""
            
              
              
                QuantumSpectre Elite Trading System - {escape_html(alert.get('alert_level', 'Alert').upper())}
                Type: {escape_html(alert.get('alert_type', 'System'))}
                Message: {escape_html(alert.get('alert_message', ''))}
                Time: {escape_html(alert.get('timestamp', ''))}
                Source: {escape_html(alert.get('source', ''))}
                Details:
                
            """
            
            # Add details
            if "details" in alert and isinstance(alert["details"], dict):
                for key, value in alert["details"].items():
                    html_body += f"{escape_html(key)}: {escape_html(str(value))}"
            
            html_body += """
                
                This is an automated alert from the QuantumSpectre Elite Trading System.
              
            
            """
            
            # Attach parts
            msg.attach(MIMEText(template.get("body", "Alert notification"), "plain"))
            msg.attach(MIMEText(html_body, "html"))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Sent email alert: {alert.get('alert_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            return False
    
    async def _send_sms_alert(self, alert: Dict[str, Any], template: Dict[str, str]) -> bool:
        """
        Send an alert via SMS.
        
        Args:
            alert: Alert data
            template: Rendered template
            
        Returns:
            True if the SMS was sent successfully, False otherwise
        """
        if not self.sms_provider or not self.sms_api_key or not self.sms_recipients:
            logger.warning("SMS configuration incomplete, skipping SMS alert")
            return False
        
        try:
            # Message body
            body = template.get("body", f"QuantumSpectre: {alert.get('alert_level')} - {alert.get('alert_message')}")
            
            if self.sms_provider == "twilio":
                await self._send_twilio_sms(body)
            elif self.sms_provider == "nexmo":
                await self._send_nexmo_sms(body)
            else:
                logger.warning(f"Unsupported SMS provider: {self.sms_provider}")
                return False
            
            logger.info(f"Sent SMS alert: {alert.get('alert_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send SMS alert: {str(e)}")
            return False
    
    async def _send_twilio_sms(self, body: str) -> None:
        """
        Send SMS using Twilio.
        
        Args:
            body: SMS message body
        """
        try:
            # This is a simplified implementation, in production you'd use the Twilio SDK
            account_sid = self.sms_api_key.split(":")[0]
            auth_token = self.sms_api_key.split(":")[1]
            
            for recipient in self.sms_recipients:
                url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
                
                data = {
                    "From": self.sms_from,
                    "To": recipient,
                    "Body": body
                }
                
                response = requests.post(
                    url, 
                    data=data,
                    auth=(account_sid, auth_token)
                )
                
                if response.status_code != 201:
                    logger.error(f"Twilio API error: {response.text}")
                
        except Exception as e:
            logger.error(f"Twilio SMS error: {str(e)}")
            raise
    
    async def _send_nexmo_sms(self, body: str) -> None:
        """
        Send SMS using Nexmo/Vonage.
        
        Args:
            body: SMS message body
        """
        try:
            # This is a simplified implementation, in production you'd use the Nexmo SDK
            api_key = self.sms_api_key.split(":")[0]
            api_secret = self.sms_api_key.split(":")[1]
            
            for recipient in self.sms_recipients:
                url = "https://rest.nexmo.com/sms/json"
                
                data = {
                    "api_key": api_key,
                    "api_secret": api_secret,
                    "from": self.sms_from,
                    "to": recipient,
                    "text": body
                }
                
                response = requests.post(url, data=data)
                
                if response.status_code != 200:
                    logger.error(f"Nexmo API error: {response.text}")
                
        except Exception as e:
            logger.error(f"Nexmo SMS error: {str(e)}")
            raise
    
    async def _send_webhook_alert(self, alert: Dict[str, Any], template: Dict[str, str]) -> bool:
        """
        Send an alert via webhook.
        
        Args:
            alert: Alert data
            template: Rendered template
            
        Returns:
            True if the webhook was called successfully, False otherwise
        """
        if not self.webhook_urls:
            logger.warning("Webhook configuration incomplete, skipping webhook alert")
            return False
        
        try:
            # Get appropriate webhook URL based on alert level and type
            webhook_url = None
            alert_level = alert.get("alert_level", "info")
            alert_type = alert.get("alert_type", "system")
            
            # Try specific URL first, then fall back to general URLs
            specific_key = f"{alert_type}_{alert_level}"
            if specific_key in self.webhook_urls:
                webhook_url = self.webhook_urls[specific_key]
            elif alert_type in self.webhook_urls:
                webhook_url = self.webhook_urls[alert_type]
            elif alert_level in self.webhook_urls:
                webhook_url = self.webhook_urls[alert_level]
            elif "default" in self.webhook_urls:
                webhook_url = self.webhook_urls["default"]
            
            if not webhook_url:
                logger.warning(f"No webhook URL defined for {alert_type} {alert_level}")
                return False
            
            # Prepare webhook payload
            if template.get("format", "json") == "json":
                # Full JSON payload
                payload = {
                    "alert_id": alert.get("alert_id"),
                    "alert_type": alert.get("alert_type"),
                    "alert_level": alert.get("alert_level"),
                    "alert_message": alert.get("alert_message"),
                    "source": alert.get("source"),
                    "timestamp": alert.get("timestamp"),
                    "details": alert.get("details", {})
                }
                
                headers = {"Content-Type": "application/json"}
                response = requests.post(webhook_url, json=payload, headers=headers)
            else:
                # Text payload
                body = template.get("body", f"QuantumSpectre: {alert.get('alert_level')} - {alert.get('alert_message')}")
                
                headers = {"Content-Type": "text/plain"}
                response = requests.post(webhook_url, data=body, headers=headers)
            
            if response.status_code not in (200, 201, 202, 204):
                logger.error(f"Webhook error: {response.status_code} {response.text}")
                return False
            
            logger.info(f"Sent webhook alert: {alert.get('alert_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {str(e)}")
            return False
    
    async def _send_voice_alert(self, alert: Dict[str, Any], template: Dict[str, str]) -> bool:
        """
        Send an alert via voice advisor.
        
        Args:
            alert: Alert data
            template: Rendered template
            
        Returns:
            True if the voice alert was triggered successfully, False otherwise
        """
        if not self.voice_advisor_enabled:
            logger.warning("Voice advisor not enabled, skipping voice alert")
            return False
        
        try:
            # Prepare voice script
            script = template.get("script", f"Alert: {alert.get('alert_message')}. Level: {alert.get('alert_level')}.")
            
            # Adjust priority based on alert level
            alert_level = alert.get("alert_level", "info")
            priority_map = {
                "critical": "immediate",
                "error": "high",
                "warning": "medium",
                "info": "low"
            }
            priority = priority_map.get(alert_level, "medium")
            
            # Call voice advisor service
            url = f"http://{self.voice_advisor_host}:{self.voice_advisor_port}/api/speak"
            
            payload = {
                "text": script,
                "priority": priority,
                "alert_id": alert.get("alert_id"),
                "alert_type": alert.get("alert_type"),
                "metadata": {
                    "source": alert.get("source"),
                    "timestamp": alert.get("timestamp"),
                    "details": alert.get("details", {})
                }
            }
            
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code not in (200, 201, 202, 204):
                logger.error(f"Voice advisor error: {response.status_code} {response.text}")
                return False
            
            logger.info(f"Sent voice alert: {alert.get('alert_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send voice alert: {str(e)}")
            return False
    
    async def _send_console_alert(self, alert: Dict[str, Any], template: Dict[str, str]) -> bool:
        """
        Send an alert to the console.
        
        Args:
            alert: Alert data
            template: Rendered template
            
        Returns:
            True always as console alerts don't fail
        """
        try:
            # Format alert based on level
            alert_level = alert.get("alert_level", "info")
            
            level_formats = {
                "critical": "\033[1;31m",  # Bold Red
                "error": "\033[31m",       # Red
                "warning": "\033[33m",     # Yellow
                "info": "\033[36m",        # Cyan
                "debug": "\033[37m"        # Light Gray
            }
            
            reset = "\033[0m"
            level_format = level_formats.get(alert_level, level_formats["info"])
            
            # Format message
            timestamp = alert.get("timestamp", datetime.now().isoformat())
            alert_type = alert.get("alert_type", "system").upper()
            source = alert.get("source", "system")
            message = alert.get("alert_message", "")
            
            console_msg = f"{level_format}[{timestamp}] {alert_level.upper()} {alert_type} from {source}: {message}{reset}"
            
            # Log with appropriate level
            if alert_level == "critical":
                logger.critical(console_msg)
            elif alert_level == "error":
                logger.error(console_msg)
            elif alert_level == "warning":
                logger.warning(console_msg)
            elif alert_level == "debug":
                logger.debug(console_msg)
            else:  # info and others
                logger.info(console_msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send console alert: {str(e)}")
            return False
    
    async def _send_alert_notification(self, alert: Dict[str, Any], override_message: Optional[str] = None) -> None:
        """
        Send alert notifications through all enabled channels.
        
        Args:
            alert: Alert data
            override_message: Optional message to override the alert message
        """
        if override_message:
            alert = alert.copy()
            alert["alert_message"] = override_message
        
        alert_type = alert.get("alert_type", "system")
        
        # Get templates for this alert type
        templates = self.templates.get(alert_type, {})
        
        # Send notifications through enabled channels
        tasks = []
        
        for channel in self.enabled_channels:
            if channel not in ALERT_CHANNELS:
                logger.warning(f"Unknown alert channel: {channel}")
                continue
            
            # Get and render template
            channel_template = templates.get(channel, {})
            try:
                rendered_template = self._render_template(channel_template, alert)
            except TemplateRenderError as e:
                logger.error(f"Failed to render template for {channel}: {str(e)}")
                continue
            
            # Send through appropriate channel
            if channel == "email":
                tasks.append(self._send_email_alert(alert, rendered_template))
            elif channel == "sms":
                tasks.append(self._send_sms_alert(alert, rendered_template))
            elif channel == "webhook":
                tasks.append(self._send_webhook_alert(alert, rendered_template))
            elif channel == "voice":
                tasks.append(self._send_voice_alert(alert, rendered_template))
            elif channel == "console":
                tasks.append(self._send_console_alert(alert, rendered_template))
        
        # Wait for all notifications to be sent
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    # Public API methods
    
    async def create_alert(self, alert_type: str, alert_level: str, message: str, source: str, 
                           details: Optional[Dict[str, Any]] = None, 
                           auto_resolve: bool = False,
                           auto_escalate: bool = False,
                           expiration: Optional[str] = None) -> str:
        """
        Create a new alert and send notifications.
        
        Args:
            alert_type: Type of alert (system, trading, security)
            alert_level: Severity level (critical, error, warning, info)
            message: Alert message
            source: Component that generated the alert
            details: Additional details and context
            auto_resolve: Whether to automatically resolve the alert when conditions are fixed
            auto_escalate: Whether to automatically escalate the alert if not resolved
            expiration: Optional timestamp when the alert expires
            
        Returns:
            ID of the created alert
        """
        # Validate inputs
        if alert_type not in ALERT_TYPES:
            raise AlertConfigurationError(f"Invalid alert type: {alert_type}")
        
        if alert_level not in ALERT_LEVELS:
            raise AlertConfigurationError(f"Invalid alert level: {alert_level}")
        
        # Check cooldown to prevent alert storms
        cooldown_key = f"{alert_type}:{source}:{alert_level}"
        now = time.time()
        
        if cooldown_key in self.last_alert_times:
            last_time = self.last_alert_times[cooldown_key]
            cooldown = self.alert_cooldown.get(alert_level, 60)
            
            if (now - last_time) < cooldown:
                logger.info(f"Alert suppressed due to cooldown: {cooldown_key}")
                return "suppressed"
        
        # Create alert
        alert_id = generate_id()
        timestamp = datetime.now().isoformat()
        
        alert = {
            "alert_id": alert_id,
            "alert_type": alert_type,
            "alert_level": alert_level,
            "alert_message": message,
            "source": source,
            "timestamp": timestamp,
            "details": details or {},
            "auto_resolve": auto_resolve,
            "auto_escalate": auto_escalate,
            "expiration": expiration,
            "resolved": False,
            "modified": True
        }
        
        # Store alert
        self.active_alerts[alert_id] = alert
        
        # Update cooldown time
        self.last_alert_times[cooldown_key] = now
        
        # Add to history
        self.alert_history.insert(0, alert)
        if len(self.alert_history) > MAX_ALERT_HISTORY:
            self.alert_history.pop()
        
        # Send notifications
        await self._send_alert_notification(alert)
        
        # Store in database
        try:
            await self.db_client.insert_one("alerts", alert)
        except Exception as e:
            logger.error(f"Failed to store alert in database: {str(e)}")
        
        logger.info(f"Created alert: {alert_id} - {alert_level} {alert_type} from {source}")
        return alert_id
    
    async def resolve_alert(self, alert_id: str, resolution_message: str) -> bool:
        """
        Resolve an active alert.
        
        Args:
            alert_id: ID of the alert to resolve
            resolution_message: Message explaining the resolution
            
        Returns:
            True if the alert was resolved, False if not found
        """
        if alert_id not in self.active_alerts:
            logger.warning(f"Alert not found for resolution: {alert_id}")
            return False
        
        alert = self.active_alerts[alert_id]
        
        # Skip if already resolved
        if alert.get("resolved", False):
            return True
        
        # Update alert
        alert["resolved"] = True
        alert["resolution_message"] = resolution_message
        alert["resolution_timestamp"] = datetime.now().isoformat()
        alert["modified"] = True
        
        # Send resolution notification
        resolution_alert = alert.copy()
        resolution_alert["alert_message"] = f"Alert resolved: {resolution_message}"
        resolution_alert["alert_level"] = "info"  # Downgrade level for resolution notifications
        
        await self._send_alert_notification(resolution_alert)
        
        # Update in database
        try:
            await self.db_client.update_one(
                "alerts", 
                {"alert_id": alert_id}, 
                {"$set": {
                    "resolved": True,
                    "resolution_message": resolution_message,
                    "resolution_timestamp": alert["resolution_timestamp"]
                }}
            )
        except Exception as e:
            logger.error(f"Failed to update alert in database: {str(e)}")
        
        logger.info(f"Resolved alert: {alert_id} - {resolution_message}")
        return True
    
    async def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an active alert.
        
        Args:
            alert_id: ID of the alert to update
            updates: Dictionary of fields to update
            
        Returns:
            True if the alert was updated, False if not found
        """
        if alert_id not in self.active_alerts:
            logger.warning(f"Alert not found for update: {alert_id}")
            return False
        
        alert = self.active_alerts[alert_id]
        
        # Apply updates
        for key, value in updates.items():
            if key not in ["alert_id", "timestamp"]:  # Protect immutable fields
                alert[key] = value
        
        alert["modified"] = True
        
        # Update in database
        try:
            await self.db_client.update_one(
                "alerts", 
                {"alert_id": alert_id}, 
                {"$set": updates}
            )
        except Exception as e:
            logger.error(f"Failed to update alert in database: {str(e)}")
        
        logger.info(f"Updated alert: {alert_id} - {list(updates.keys())}")
        return True
    
    async def get_active_alerts(self, alert_type: Optional[str] = None, 
                              alert_level: Optional[str] = None,
                              source: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get active (unresolved) alerts, optionally filtered.
        
        Args:
            alert_type: Optional type filter
            alert_level: Optional level filter
            source: Optional source filter
            
        Returns:
            List of matching active alerts
        """
        filtered_alerts = []
        
        for alert in self.active_alerts.values():
            if alert.get("resolved", False):
                continue
            
            if alert_type and alert.get("alert_type") != alert_type:
                continue
            
            if alert_level and alert.get("alert_level") != alert_level:
                continue
            
            if source and alert.get("source") != source:
                continue
            
            filtered_alerts.append(alert)
        
        # Sort by timestamp, newest first
        filtered_alerts.sort(key=lambda a: a.get("timestamp", ""), reverse=True)
        
        return filtered_alerts
    
    async def get_alert_history(self, limit: int = 100, 
                               alert_type: Optional[str] = None,
                               alert_level: Optional[str] = None,
                               source: Optional[str] = None,
                               include_resolved: bool = True) -> List[Dict[str, Any]]:
        """
        Get alert history, optionally filtered.
        
        Args:
            limit: Maximum number of alerts to return
            alert_type: Optional type filter
            alert_level: Optional level filter
            source: Optional source filter
            include_resolved: Whether to include resolved alerts
            
        Returns:
            List of matching alerts from history
        """
        filtered_alerts = []
        
        for alert in self.alert_history:
            if not include_resolved and alert.get("resolved", False):
                continue
            
            if alert_type and alert.get("alert_type") != alert_type:
                continue
            
            if alert_level and alert.get("alert_level") != alert_level:
                continue
            
            if source and alert.get("source") != source:
                continue
            
            filtered_alerts.append(alert)
            
            if len(filtered_alerts) >= limit:
                break
        
        return filtered_alerts
    
    async def get_alert_stats(self) -> Dict[str, Any]:
        """
        Get statistics about alerts.
        
        Returns:
            Dictionary with alert statistics
        """
        stats = {
            "total_active": 0,
            "by_level": {},
            "by_type": {},
            "by_source": {}
        }
        
        # Initialize counters
        for level in ALERT_LEVELS:
            stats["by_level"][level] = 0
        
        for alert_type in ALERT_TYPES:
            stats["by_type"][alert_type] = 0
        
        # Count active alerts
        for alert in self.active_alerts.values():
            if alert.get("resolved", False):
                continue
            
            stats["total_active"] += 1
            
            level = alert.get("alert_level", "info")
            stats["by_level"][level] = stats["by_level"].get(level, 0) + 1
            
            alert_type = alert.get("alert_type", "system")
            stats["by_type"][alert_type] = stats["by_type"].get(alert_type, 0) + 1
            
            source = alert.get("source", "unknown")
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
        
        # Add timestamp
        stats["timestamp"] = datetime.now().isoformat()
        
        return stats
    
    async def check_system_metrics(self, service_name: str, metrics: Dict[str, Any]) -> None:
        """
        Check system metrics against thresholds and create alerts if needed.
        
        Args:
            service_name: Name of the service the metrics are from
            metrics: Dictionary of metrics to check
        """
        try:
            # Check each metric against thresholds
            for metric_name, metric_value in metrics.items():
                if metric_name not in SYSTEM_ALERT_THRESHOLDS:
                    continue
                
                thresholds = SYSTEM_ALERT_THRESHOLDS[metric_name]
                
                # Check each threshold
                for level, threshold_config in thresholds.items():
                    threshold = threshold_config["value"]
                    comparison = threshold_config.get("comparison", ">")
                    
                    # Check if threshold is exceeded
                    condition_met = False
                    
                    if comparison == ">" and metric_value > threshold:
                        condition_met = True
                    elif comparison == ">=" and metric_value >= threshold:
                        condition_met = True
                    elif comparison == "<" and metric_value < threshold:
                        condition_met = True
                    elif comparison == "<=" and metric_value <= threshold:
                        condition_met = True
                    elif comparison == "==" and metric_value == threshold:
                        condition_met = True
                    elif comparison == "!=" and metric_value != threshold:
                        condition_met = True
                    
                    if condition_met:
                        # Create alert
                        message = threshold_config.get("message", f"{metric_name} {comparison} {threshold}")
                        details = {
                            "metric": metric_name,
                            "value": metric_value,
                            "threshold": threshold,
                            "comparison": comparison
                        }
                        
                        await self.create_alert(
                            alert_type="system",
                            alert_level=level,
                            message=message,
                            source=service_name,
                            details=details,
                            auto_resolve=True
                        )
            
        except Exception as e:
            logger.error(f"Error checking system metrics: {str(e)}")
    
    async def check_trading_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Check trading metrics against thresholds and create alerts if needed.
        
        Args:
            metrics: Dictionary of trading metrics to check
        """
        try:
            # Check each metric against thresholds
            for metric_name, metric_value in metrics.items():
                if metric_name not in TRADING_ALERT_THRESHOLDS:
                    continue
                
                thresholds = TRADING_ALERT_THRESHOLDS[metric_name]
                
                # Check each threshold
                for level, threshold_config in thresholds.items():
                    threshold = threshold_config["value"]
                    comparison = threshold_config.get("comparison", ">")
                    
                    # Check if threshold is exceeded
                    condition_met = False
                    
                    if comparison == ">" and metric_value > threshold:
                        condition_met = True
                    elif comparison == ">=" and metric_value >= threshold:
                        condition_met = True
                    elif comparison == "<" and metric_value < threshold:
                        condition_met = True
                    elif comparison == "<=" and metric_value <= threshold:
                        condition_met = True
                    elif comparison == "==" and metric_value == threshold:
                        condition_met = True
                    elif comparison == "!=" and metric_value != threshold:
                        condition_met = True
                    
                    if condition_met:
                        # Create alert
                        message = threshold_config.get("message", f"{metric_name} {comparison} {threshold}")
                        details = {
                            "metric": metric_name,
                            "value": metric_value,
                            "threshold": threshold,
                            "comparison": comparison
                        }
                        
                        # Add position details if available
                        if "position_id" in metrics:
                            details["position_id"] = metrics["position_id"]
                        
                        if "symbol" in metrics:
                            details["symbol"] = metrics["symbol"]
                        
                        await self.create_alert(
                            alert_type="trading",
                            alert_level=level,
                            message=message,
                            source=metrics.get("source", "trading_engine"),
                            details=details,
                            auto_resolve=True,
                            auto_escalate=True
                        )
            
        except Exception as e:
            logger.error(f"Error checking trading metrics: {str(e)}")
    
    async def create_pattern_alert(self, pattern_name: str, symbol: str, 
                                timeframe: str, confidence: float,
                                direction: str, details: Dict[str, Any]) -> str:
        """
        Create an alert for a detected trading pattern.
        
        Args:
            pattern_name: Name of the detected pattern
            symbol: Trading symbol where the pattern was detected
            timeframe: Timeframe of the detection
            confidence: Confidence level (0-1)
            direction: Expected price direction (up/down)
            details: Additional details about the pattern
            
        Returns:
            ID of the created alert
        """
        try:
            # Determine alert level based on confidence
            alert_level = "info"
            if confidence >= 0.9:
                alert_level = "critical"  # Highest confidence patterns
            elif confidence >= 0.8:
                alert_level = "error"     # High confidence patterns
            elif confidence >= 0.7:
                alert_level = "warning"   # Medium confidence patterns
            
            # Create message
            direction_arrow = "⬆️" if direction.lower() == "up" else "⬇️"
            message = f"{pattern_name} pattern detected on {symbol} ({timeframe}) with {confidence:.1%} confidence {direction_arrow}"
            
            # Prepare details
            pattern_details = {
                "pattern": pattern_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "confidence": confidence,
                "direction": direction,
                "detection_time": datetime.now().isoformat()
            }
            
            # Add any additional details
            pattern_details.update(details)
            
            # Create the alert
            alert_id = await self.create_alert(
                alert_type="trading",
                alert_level=alert_level,
                message=message,
                source="pattern_recognition",
                details=pattern_details,
                auto_resolve=True,  # Pattern alerts auto-resolve after a time
                expiration=None  # Will use default expiration logic
            )
            
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating pattern alert: {str(e)}")
            return "error"
    
    async def create_security_alert(self, security_issue: str, source: str, severity: str,
                                  details: Dict[str, Any], auto_escalate: bool = True) -> str:
        """
        Create an alert for a security issue.
        
        Args:
            security_issue: Description of the security issue
            source: Component that detected the issue
            severity: Severity level (critical, error, warning, info)
            details: Additional details about the security issue
            auto_escalate: Whether to automatically escalate the alert if not resolved
            
        Returns:
            ID of the created alert
        """
        try:
            # Map severity to alert level
            alert_level = severity if severity in ALERT_LEVELS else "warning"
            
            # Create message
            message = f"Security issue detected: {security_issue}"
            
            # Create the alert
            alert_id = await self.create_alert(
                alert_type="security",
                alert_level=alert_level,
                message=message,
                source=source,
                details=details,
                auto_resolve=False,  # Security alerts typically need manual resolution
                auto_escalate=auto_escalate
            )
            
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating security alert: {str(e)}")
            return "error"

# Singleton instance for application-wide use
_alerting_system = None

def get_alerting_system(config: Optional[Dict[str, Any]] = None) -> AlertingSystem:
    """
    Get or create the singleton AlertingSystem instance.
    
    Args:
        config: Configuration dictionary (only used if creating a new instance)
        
    Returns:
        The singleton AlertingSystem instance
    """
    global _alerting_system
    
    if _alerting_system is None and config is not None:
        _alerting_system = AlertingSystem(config)
    
    if _alerting_system is None:
        raise RuntimeError("AlertingSystem not initialized. Provide config on first call.")
    
    return _alerting_system

# End of File

