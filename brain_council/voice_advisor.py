#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Brain Council - Voice Advisor

This module implements an advanced voice advisory system that provides real-time
trading suggestions, pattern notifications, and market insights through
both text and voice communication channels.
"""

import os
import json
import logging
import tempfile
import asyncio
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue

# TTS and speech libraries
import numpy as np
from scipy.io import wavfile
import sounddevice as sd
import soundfile as sf

# Internal imports
from common.logger import get_logger
from common.utils import format_price, format_percentage, get_user_preference
from common.constants import (
    SIGNAL_TYPES, TIMEFRAMES, VOICE_ADVISOR_MODES,
    ACTION_BUY, ACTION_SELL, ACTION_HOLD, ACTION_CLOSE,
    NOTIFICATION_PRIORITY_LOW, NOTIFICATION_PRIORITY_MEDIUM, NOTIFICATION_PRIORITY_HIGH
)
from common.exceptions import VoiceAdvisorError, TTSEngineError
from ml_models.models.deep_learning import TextToSpeechModel
from data_storage.models.user_data import UserPreference, NotificationRecord


class VoiceAdvisor:
    """
    Advanced voice advisor that provides real-time trading suggestions,
    pattern notifications, and market insights through voice and text.
    
    The advisor uses a sophisticated text-to-speech engine optimized for
    trading terminology and can adapt its communication style based on
    user preferences and signal importance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Voice Advisor.
        
        Args:
            config: Configuration for the voice advisor including voice settings,
                   notification preferences, and TTS parameters.
        """
        self.logger = get_logger("brain_council.voice_advisor")
        self.config = config
        
        # Voice configuration
        self.voice_config = config.get("voice", {})
        self.voice_enabled = self.voice_config.get("enabled", True)
        self.voice_speed = self.voice_config.get("speed", 1.0)
        self.voice_pitch = self.voice_config.get("pitch", 1.0)
        self.voice_volume = self.voice_config.get("volume", 1.0)
        self.voice_type = self.voice_config.get("type", "neutral")
        
        # Notification configuration
        self.notification_config = config.get("notification", {})
        self.min_priority = self.notification_config.get("min_priority", NOTIFICATION_PRIORITY_LOW)
        self.notification_cooldown = self.notification_config.get("cooldown_seconds", 30)
        
        # Initialize TTS model
        self.tts_model = None
        self.initialize_tts_model()
        
        # Queue for voice notifications
        self.voice_queue = queue.Queue()
        self.voice_thread = None
        
        # Last notification time tracking
        self.last_notification_time = {}
        
        # Statistics
        self.notifications_sent = 0
        self.notifications_skipped = 0
        
        self.logger.info("Voice Advisor initialized with voice %s enabled", 
                      "is" if self.voice_enabled else "is not")
        
        # Start voice processing thread if enabled
        if self.voice_enabled:
            self.start_voice_thread()
    
    def initialize_tts_model(self) -> None:
        """Initialize the text-to-speech model."""
        try:
            tts_config = self.config.get("tts_model", {})
            model_path = tts_config.get("model_path", "models/tts/quantumspectre_tts.pt")
            
            self.logger.info("Initializing TTS model from %s", model_path)
            self.tts_model = TextToSpeechModel(
                model_path=model_path,
                device=tts_config.get("device", "cuda" if self.config.get("use_gpu", True) else "cpu"),
                precision=tts_config.get("precision", "float16"),
                cache_dir=tts_config.get("cache_dir", "./tts_cache")
            )
            self.logger.info("TTS model initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize TTS model: %s", str(e), exc_info=True)
            # Fallback to simpler TTS if available
            try:
                from gtts import gTTS
                self.logger.info("Falling back to gTTS for text-to-speech")
                self.tts_model = None  # We'll use gTTS directly when needed
            except ImportError:
                self.logger.warning("gTTS not available for fallback, voice output disabled")
                self.voice_enabled = False
    
    def start_voice_thread(self) -> None:
        """Start the voice processing thread."""
        if self.voice_thread is None or not self.voice_thread.is_alive():
            self.voice_thread = threading.Thread(
                target=self._voice_worker, 
                daemon=True
            )
            self.voice_thread.start()
            self.logger.info("Voice processing thread started")
    
    def _voice_worker(self) -> None:
        """Worker thread for processing and playing voice notifications."""
        while True:
            try:
                # Get the next item from the queue
                text, priority, audio_path = self.voice_queue.get()
                
                # Play the audio if available
                if audio_path and os.path.exists(audio_path):
                    self._play_audio(audio_path)
                    # Clean up temp file if it's a temp file
                    if audio_path.startswith(tempfile.gettempdir()):
                        try:
                            os.remove(audio_path)
                        except Exception as e:
                            self.logger.warning("Failed to remove temp audio file: %s", str(e))
                
                # Mark task as done
                self.voice_queue.task_done()
                
                # Brief pause to prevent CPU overuse
                time.sleep(0.1)
            except Exception as e:
                self.logger.error("Error in voice worker thread: %s", str(e), exc_info=True)
                time.sleep(1)  # Sleep longer on error
    
    def _play_audio(self, audio_path: str) -> None:
        """
        Play an audio file.
        
        Args:
            audio_path: Path to the audio file
        """
        try:
            # Read audio file
            data, fs = sf.read(audio_path)
            
            # Adjust volume
            data = data * self.voice_volume
            
            # Play audio
            sd.play(data, fs)
            sd.wait()
        except Exception as e:
            self.logger.error("Failed to play audio: %s", str(e), exc_info=True)
    
    def _generate_audio(self, text: str, priority: int) -> Optional[str]:
        """
        Generate audio from text using TTS.
        
        Args:
            text: Text to convert to speech
            priority: Priority level of the notification
            
        Returns:
            Path to the generated audio file or None if generation failed
        """
        try:
            # Adjust voice parameters based on priority
            speed = self.voice_speed
            pitch = self.voice_pitch
            
            if priority == NOTIFICATION_PRIORITY_HIGH:
                speed = min(1.2, speed * 1.1)  # Slightly faster for urgent messages
                pitch = min(1.2, pitch * 1.1)  # Slightly higher pitch
            
            # Generate temp file path
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_file.close()
            output_path = temp_file.name
            
            # Use our TTS model if available
            if self.tts_model is not None:
                self.tts_model.generate_speech(
                    text=text,
                    output_path=output_path,
                    voice_type=self.voice_type,
                    speed=speed,
                    pitch=pitch
                )
            else:
                # Fallback to gTTS
                from gtts import gTTS
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(output_path)
            
            return output_path
        except Exception as e:
            self.logger.error("Failed to generate audio: %s", str(e), exc_info=True)
            return None
    
    async def notify_signal(self, signal: Dict[str, Any], 
                     market_data: Dict[str, Any]) -> bool:
        """
        Notify user about a trading signal through voice and text.
        
        Args:
            signal: Trading signal information
            market_data: Current market data
            
        Returns:
            True if notification was sent, False otherwise
        """
        symbol = signal["symbol"]
        action = signal["action"]
        confidence = signal["confidence"]
        
        # Get priority based on confidence and action
        if confidence >= 0.85:
            priority = NOTIFICATION_PRIORITY_HIGH
        elif confidence >= 0.7:
            priority = NOTIFICATION_PRIORITY_MEDIUM
        else:
            priority = NOTIFICATION_PRIORITY_LOW
        
        # Check if we should skip due to priority or cooldown
        if priority < self.min_priority:
            self.logger.debug("Skipping notification due to low priority: %d", priority)
            self.notifications_skipped += 1
            return False
        
        # Check cooldown
        now = datetime.utcnow()
        last_time = self.last_notification_time.get(symbol, datetime.min)
        if (now - last_time).total_seconds() < self.notification_cooldown:
            self.logger.debug("Skipping notification due to cooldown for %s", symbol)
            self.notifications_skipped += 1
            return False
        
        # Format notification text
        action_text = self._format_action(action)
        entry_price = format_price(signal["entry_price"], symbol)
        risk_reward = signal["risk_reward"]
        
        notification_text = f"Signal: {action_text} {symbol} "
        notification_text += f"at {entry_price} with confidence {confidence:.0%}. "
        
        # Add risk/reward
        notification_text += f"Risk-reward ratio is {risk_reward:.1f}. "
        
        # Add reasoning if available
        if "reasoning" in signal and len(signal["reasoning"]) > 0:
            # Extract a short summary from the reasoning
            reasoning_summary = self._summarize_reasoning(signal["reasoning"])
            notification_text += f"{reasoning_summary} "
        
        # Add market context if available
        if market_data and "market_regime" in market_data:
            notification_text += f"Current market regime is {market_data['market_regime']}. "
        
        # Generate more detailed message for text notification
        detailed_text = notification_text + "\n\n"
        detailed_text += f"Entry: {entry_price}\n"
        detailed_text += f"Stop Loss: {format_price(signal['stop_loss'], symbol)}\n"
        detailed_text += f"Take Profit: {format_price(signal['take_profit'], symbol)}\n"
        detailed_text += f"Risk/Reward: {risk_reward:.2f}\n"
        detailed_text += f"Confidence: {confidence:.2f}\n"
        if "expected_duration" in signal:
            duration = signal["expected_duration"]
            hours = duration.get("hours", 0)
            minutes = duration.get("minutes", 0)
            detailed_text += f"Expected Duration: {hours}h {minutes}m\n"
        
        # Send text notification
        await self._send_text_notification(
            symbol=symbol,
            title=f"{action_text.upper()} {symbol}",
            message=detailed_text,
            priority=priority
        )
        
        # Send voice notification if enabled
        if self.voice_enabled:
            await self._send_voice_notification(notification_text, priority)
        
        # Update last notification time
        self.last_notification_time[symbol] = now
        self.notifications_sent += 1
        
        return True
    
    async def notify_pattern(self, pattern_data: Dict[str, Any]) -> bool:
        """
        Notify user about a detected pattern through voice and text.
        
        Args:
            pattern_data: Pattern information
            
        Returns:
            True if notification was sent, False otherwise
        """
        symbol = pattern_data["symbol"]
        pattern_name = pattern_data["pattern_name"]
        confidence = pattern_data.get("confidence", 0.6)
        timeframe = pattern_data.get("timeframe", "unknown")
        
        # Get priority based on confidence and pattern reliability
        pattern_reliability = pattern_data.get("historical_reliability", 0.6)
        combined_score = (confidence + pattern_reliability) / 2
        
        if combined_score >= 0.8:
            priority = NOTIFICATION_PRIORITY_HIGH
        elif combined_score >= 0.65:
            priority = NOTIFICATION_PRIORITY_MEDIUM
        else:
            priority = NOTIFICATION_PRIORITY_LOW
        
        # Check if we should skip due to priority or cooldown
        if priority < self.min_priority:
            self.logger.debug("Skipping pattern notification due to low priority: %d", priority)
            self.notifications_skipped += 1
            return False
        
        # Check cooldown (special key for patterns)
        pattern_key = f"{symbol}_{pattern_name}_{timeframe}"
        now = datetime.utcnow()
        last_time = self.last_notification_time.get(pattern_key, datetime.min)
        if (now - last_time).total_seconds() < self.notification_cooldown:
            self.logger.debug("Skipping pattern notification due to cooldown for %s", pattern_key)
            self.notifications_skipped += 1
            return False
        
        # Construct message text
        action_hint = pattern_data.get("action_hint", "indicates potential opportunity")
        price = pattern_data.get("price", None)
        price_str = f"at {format_price(price, symbol)}" if price else ""
        
        notification_text = f"Pattern detected: {pattern_name} on {symbol} {timeframe} chart {price_str}. "
        notification_text += f"This {action_hint} with {confidence:.0%} confidence. "
        
        # Add target projection if available
        if "target_projection" in pattern_data:
            target = pattern_data["target_projection"]
            current = pattern_data.get("current_price", 0)
            if current > 0 and target > 0:
                percent_change = ((target - current) / current) * 100
                notification_text += f"Target projection suggests a move to {format_price(target, symbol)} "
                notification_text += f"({format_percentage(percent_change)}). "
        
        # Generate more detailed message for text notification
        detailed_text = notification_text + "\n\n"
        detailed_text += f"Pattern: {pattern_name}\n"
        detailed_text += f"Symbol: {symbol}\n"
        detailed_text += f"Timeframe: {timeframe}\n"
        detailed_text += f"Confidence: {confidence:.2f}\n"
        detailed_text += f"Historical Reliability: {pattern_reliability:.2f}\n"
        
        if "key_levels" in pattern_data:
            detailed_text += "\nKey Levels:\n"
            for level_name, level_value in pattern_data["key_levels"].items():
                detailed_text += f"- {level_name}: {format_price(level_value, symbol)}\n"
        
        # Send text notification
        await self._send_text_notification(
            symbol=symbol,
            title=f"Pattern: {pattern_name} on {symbol}",
            message=detailed_text,
            priority=priority
        )
        
        # Send voice notification if enabled
        if self.voice_enabled:
            await self._send_voice_notification(notification_text, priority)
        
        # Update last notification time
        self.last_notification_time[pattern_key] = now
        self.notifications_sent += 1
        
        return True
    
    async def notify_market_insight(self, insight_data: Dict[str, Any]) -> bool:
        """
        Notify user about a market insight through voice and text.
        
        Args:
            insight_data: Market insight information
            
        Returns:
            True if notification was sent, False otherwise
        """
        insight_type = insight_data["type"]
        importance = insight_data.get("importance", 0.5)
        symbols_affected = insight_data.get("symbols_affected", [])
        
        # Get priority based on importance
        if importance >= 0.75:
            priority = NOTIFICATION_PRIORITY_HIGH
        elif importance >= 0.5:
            priority = NOTIFICATION_PRIORITY_MEDIUM
        else:
            priority = NOTIFICATION_PRIORITY_LOW
        
        # Check if we should skip due to priority
        if priority < self.min_priority:
            self.logger.debug("Skipping insight notification due to low priority: %d", priority)
            self.notifications_skipped += 1
            return False
        
        # Check cooldown
        insight_key = f"insight_{insight_type}"
        now = datetime.utcnow()
        last_time = self.last_notification_time.get(insight_key, datetime.min)
        if (now - last_time).total_seconds() < self.notification_cooldown:
            self.logger.debug("Skipping insight notification due to cooldown for %s", insight_key)
            self.notifications_skipped += 1
            return False
        
        # Construct message text based on insight type
        title = f"Market Insight: {insight_type.replace('_', ' ').title()}"
        
        notification_text = f"Market insight: {insight_data.get('summary', insight_type)}. "
        
        if symbols_affected:
            symbols_text = ", ".join(symbols_affected[:3])
            if len(symbols_affected) > 3:
                symbols_text += f" and {len(symbols_affected) - 3} more"
            notification_text += f"Affects {symbols_text}. "
        
        notification_text += insight_data.get("details", "")
        
        # Generate more detailed message for text notification
        detailed_text = notification_text + "\n\n"
        
        if "data_points" in insight_data:
            detailed_text += "Key Data Points:\n"
            for key, value in insight_data["data_points"].items():
                detailed_text += f"- {key}: {value}\n"
        
        # Send text notification
        await self._send_text_notification(
            symbol=symbols_affected[0] if symbols_affected else "MARKET",
            title=title,
            message=detailed_text,
            priority=priority
        )
        
        # Send voice notification if enabled
        if self.voice_enabled:
            await self._send_voice_notification(notification_text, priority)
        
        # Update last notification time
        self.last_notification_time[insight_key] = now
        self.notifications_sent += 1
        
        return True
    
    async def notify_loophole(self, loophole_data: Dict[str, Any]) -> bool:
        """
        Notify user about a detected market loophole or inefficiency.
        
        Args:
            loophole_data: Information about the detected loophole
            
        Returns:
            True if notification was sent, False otherwise
        """
        loophole_type = loophole_data["type"]
        confidence = loophole_data.get("confidence", 0.7)
        symbol = loophole_data.get("symbol", "MARKET")
        expected_profit = loophole_data.get("expected_profit", 0.0)
        
        # Loophole notifications are always high priority
        priority = NOTIFICATION_PRIORITY_HIGH
        
        # Check cooldown (more specific key for loopholes)
        loophole_key = f"loophole_{symbol}_{loophole_type}"
        now = datetime.utcnow()
        last_time = self.last_notification_time.get(loophole_key, datetime.min)
        if (now - last_time).total_seconds() < self.notification_cooldown:
            self.logger.debug("Skipping loophole notification due to cooldown for %s", loophole_key)
            self.notifications_skipped += 1
            return False
        
        # Construct message text
        title = f"LOOPHOLE DETECTED: {loophole_type.replace('_', ' ').upper()}"
        
        notification_text = f"Critical loophole detected: {loophole_type} on {symbol}. "
        notification_text += f"Confidence: {confidence:.0%}. "
        
        if expected_profit > 0:
            notification_text += f"Expected profit: {format_percentage(expected_profit)}. "
        
        if "action_required" in loophole_data:
            notification_text += f"Action required: {loophole_data['action_required']}. "
        
        if "time_sensitivity" in loophole_data:
            sensitivity = loophole_data["time_sensitivity"]
            if sensitivity == "high":
                notification_text += "This opportunity is highly time-sensitive! "
            elif sensitivity == "medium":
                notification_text += "This opportunity has medium time sensitivity. "
        
        # Generate more detailed message for text notification
        detailed_text = notification_text + "\n\n"
        detailed_text += f"Loophole Type: {loophole_type}\n"
        detailed_text += f"Symbol: {symbol}\n"
        detailed_text += f"Confidence: {confidence:.2f}\n"
        detailed_text += f"Expected Profit: {format_percentage(expected_profit)}\n"
        
        if "execution_steps" in loophole_data:
            detailed_text += "\nExecution Steps:\n"
            for i, step in enumerate(loophole_data["execution_steps"], 1):
                detailed_text += f"{i}. {step}\n"
        
        if "risk_factors" in loophole_data:
            detailed_text += "\nRisk Factors:\n"
            for risk in loophole_data["risk_factors"]:
                detailed_text += f"- {risk}\n"
        
        # Send text notification
        await self._send_text_notification(
            symbol=symbol,
            title=title,
            message=detailed_text,
            priority=priority
        )
        
        # Send voice notification if enabled - higher volume for loopholes
        if self.voice_enabled:
            temp_volume = self.voice_volume
            self.voice_volume = min(1.0, self.voice_volume * 1.3)  # Increase volume temporarily
            await self._send_voice_notification(notification_text, priority)
            self.voice_volume = temp_volume  # Restore original volume
        
        # Update last notification time
        self.last_notification_time[loophole_key] = now
        self.notifications_sent += 1
        
        return True
    
    async def _send_text_notification(self, symbol: str, title: str, 
                              message: str, priority: int) -> None:
        """
        Send a text notification to the user.
        
        Args:
            symbol: The trading symbol related to this notification
            title: Notification title
            message: Notification message
            priority: Notification priority
        """
        try:
            # Store notification in database
            notification = NotificationRecord.create(
                symbol=symbol,
                title=title,
                message=message,
                priority=priority,
                timestamp=datetime.utcnow(),
                is_read=False
            )
            
            self.logger.info("Text notification sent: %s", title)
            
            # Could also integrate with external notification services here
            # Such as email, Telegram, Discord, desktop notifications, etc.
            
        except Exception as e:
            self.logger.error("Failed to send text notification: %s", str(e), exc_info=True)
    
    async def _send_voice_notification(self, text: str, priority: int) -> None:
        """
        Send a voice notification to the user.
        
        Args:
            text: Text to speak
            priority: Notification priority
        """
        try:
            # Generate audio in a separate thread to avoid blocking
            audio_path = await asyncio.to_thread(self._generate_audio, text, priority)
            
            if audio_path:
                # Add to voice queue for playback
                self.voice_queue.put((text, priority, audio_path))
                self.logger.debug("Added voice notification to queue: %s", text[:50])
            else:
                self.logger.warning("Failed to generate audio for notification")
                
        except Exception as e:
            self.logger.error("Failed to send voice notification: %s", str(e), exc_info=True)
    
    def _format_action(self, action: str) -> str:
        """
        Format an action string for notifications.
        
        Args:
            action: Action constant
            
        Returns:
            Formatted action string
        """
        action_map = {
            ACTION_BUY: "buy",
            ACTION_SELL: "sell",
            ACTION_HOLD: "hold",
            ACTION_CLOSE: "close position on"
        }
        return action_map.get(action, action)
    
    def _summarize_reasoning(self, reasoning: str, max_length: int = 100) -> str:
        """
        Create a concise summary of signal reasoning.
        
        Args:
            reasoning: Full reasoning text
            max_length: Maximum length for summary
            
        Returns:
            Summarized reasoning
        """
        if len(reasoning) <= max_length:
            return reasoning
        
        # Simple summarization - get first part and truncate
        summary = reasoning.split("|")[0].strip()
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the voice advisor.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "notifications_sent": self.notifications_sent,
            "notifications_skipped": self.notifications_skipped,
            "voice_enabled": self.voice_enabled,
            "voice_queue_size": self.voice_queue.qsize() if self.voice_enabled else 0
        }
    
    def reset_statistics(self) -> None:
        """Reset the voice advisor statistics."""
        self.notifications_sent = 0
        self.notifications_skipped = 0
