#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Pattern Recognition Module

This module provides advanced pattern recognition capabilities for identifying
market patterns across multiple timeframes, including classic chart patterns,
candlestick patterns, harmonic patterns, and custom patterns.
"""

import os
import importlib
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import logging

# Internal imports
from common.logger import get_logger
from common.utils import get_submodules

logger = get_logger(__name__)

# Dictionary to hold pattern detector classes
PATTERN_DETECTORS = {}

def register_pattern_detector(name: str, detector_class: Any) -> None:
    """
    Register a pattern detector class.
    
    Args:
        name: Name of the pattern detector
        detector_class: The pattern detector class
    """
    if name in PATTERN_DETECTORS:
        logger.warning(f"Pattern detector '{name}' already registered, overwriting")
    
    PATTERN_DETECTORS[name] = detector_class
    logger.debug(f"Registered pattern detector '{name}'")

def get_pattern_detector(name: str) -> Optional[Any]:
    """
    Get a pattern detector by name.
    
    Args:
        name: Name of the pattern detector
    
    Returns:
        The pattern detector class or None if not found
    """
    return PATTERN_DETECTORS.get(name)

def list_pattern_detectors() -> List[str]:
    """
    List all registered pattern detectors.
    
    Returns:
        List of pattern detector names
    """
    return list(PATTERN_DETECTORS.keys())

# Import pattern detector modules automatically
PATTERN_MODULES = [
    'harmonic_patterns',
    'candlestick_patterns',
    'chart_patterns',
    'support_resistance',
    'volume_patterns'
]

# Load all pattern modules
for module_name in PATTERN_MODULES:
    try:
        # Import the module
        module = importlib.import_module(f"intelligence.pattern_recognition.{module_name}")
        
        # Check if module has register_detectors function and call it
        if hasattr(module, 'register_detectors'):
            module.register_detectors()
        
        logger.debug(f"Loaded pattern module '{module_name}'")
    except Exception as e:
        logger.error(f"Error loading pattern module '{module_name}': {str(e)}")

class PatternRecognition:
    """
    Pattern Recognition Service
    
    This class coordinates pattern detection across various types of patterns
    and provides a unified interface for pattern detection.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the pattern recognition service.
        
        Args:
            config: Configuration for pattern recognition
        """
        self.config = config
        self.detectors = {}
        
        # Initialize pattern detectors
        self._initialize_detectors()
        
        logger.info("Pattern Recognition initialized with %d detectors", len(self.detectors))
    
    def _initialize_detectors(self) -> None:
        """Initialize all pattern detectors."""
        for name, detector_class in PATTERN_DETECTORS.items():
            try:
                # Create an instance of the detector
                detector_config = self.config.get(name, {})
                detector = detector_class(detector_config)
                
                # Store the detector instance
                self.detectors[name] = detector
                
                logger.debug(f"Initialized pattern detector '{name}'")
            except Exception as e:
                logger.error(f"Error initializing pattern detector '{name}': {str(e)}")
    
    async def detect_patterns(
        self, 
        data: Dict[str, Any], 
        pattern_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect patterns in market data.
        
        Args:
            data: Market data for pattern detection
            pattern_types: Optional list of pattern types to detect, or None for all
        
        Returns:
            Dictionary of detected patterns
        """
        results = {}
        
        # If no pattern types specified, use all available detectors
        if pattern_types is None:
            pattern_types = list(self.detectors.keys())
        
        # Run each specified detector
        for pattern_type in pattern_types:
            if pattern_type in self.detectors:
                try:
                    detector = self.detectors[pattern_type]
                    patterns = await detector.detect(data)
                    results[pattern_type] = patterns
                except Exception as e:
                    logger.error(f"Error detecting patterns with '{pattern_type}': {str(e)}")
                    results[pattern_type] = []
            else:
                logger.warning(f"Pattern detector '{pattern_type}' not found")
                results[pattern_type] = []
        
        return results
    
    async def detect_pattern(
        self, 
        data: Dict[str, Any], 
        pattern_type: str, 
        pattern_name: str
    ) -> Dict[str, Any]:
        """
        Detect a specific pattern in market data.
        
        Args:
            data: Market data for pattern detection
            pattern_type: Type of pattern to detect
            pattern_name: Name of the specific pattern
        
        Returns:
            Dictionary with detection results
        """
        if pattern_type not in self.detectors:
            logger.warning(f"Pattern detector '{pattern_type}' not found")
            return {'detected': False, 'pattern': pattern_name, 'confidence': 0.0}
        
        try:
            detector = self.detectors[pattern_type]
            result = await detector.detect_specific(data, pattern_name)
            return result
        except Exception as e:
            logger.error(f"Error detecting pattern '{pattern_name}' with '{pattern_type}': {str(e)}")
            return {'detected': False, 'pattern': pattern_name, 'confidence': 0.0}
    
    def get_available_patterns(self) -> Dict[str, List[str]]:
        """
        Get all available patterns by detector.
        
        Returns:
            Dictionary mapping detector names to lists of pattern names
        """
        patterns = {}
        
        for name, detector in self.detectors.items():
            if hasattr(detector, 'get_supported_patterns'):
                patterns[name] = detector.get_supported_patterns()
            else:
                patterns[name] = []
        
        return patterns
    
    async def analyze_patterns(
        self, 
        data: Dict[str, Any],
        pattern_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive pattern analysis on market data.
        
        This method goes beyond simple detection to provide analysis of the patterns,
        including potential implications, reliability statistics, and confluence with
        other market factors.
        
        Args:
            data: Market data for pattern analysis
            pattern_types: Optional list of pattern types to analyze, or None for all
        
        Returns:
            Comprehensive analysis of detected patterns
        """
        # First detect patterns
        patterns = await self.detect_patterns(data, pattern_types)
        
        # Analyze the detected patterns
        analysis = {
            'patterns': patterns,
            'summary': {},
            'confluence': {},
            'implications': {},
            'reliability': {}
        }
        
        # Generate summary
        pattern_count = sum(len(p) for p in patterns.values())
        analysis['summary']['total_patterns'] = pattern_count
        
        if pattern_count > 0:
            # Analyze pattern confluence
            analysis['confluence'] = self._analyze_confluence(patterns)
            
            # Analyze pattern implications
            analysis['implications'] = self._analyze_implications(patterns, data)
            
            # Analyze pattern reliability
            analysis['reliability'] = self._analyze_reliability(patterns)
        
        return analysis
    
    def _analyze_confluence(self, patterns: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze pattern confluence to identify areas with multiple pattern confirmations.
        
        Args:
            patterns: Detected patterns
        
        Returns:
            Confluence analysis
        """
        # Simple implementation - in a full system this would be much more sophisticated
        confluence = {
            'areas': [],
            'score': 0.0
        }
        
        # Extract price levels from patterns
        price_levels = []
        for pattern_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if 'price_levels' in pattern:
                    for level_type, level_value in pattern['price_levels'].items():
                        price_levels.append({
                            'value': level_value,
                            'type': level_type,
                            'pattern': pattern.get('name', 'Unknown'),
                            'pattern_type': pattern_type,
                            'confidence': pattern.get('confidence', 0.5)
                        })
        
        # Group close price levels
        if price_levels:
            from sklearn.cluster import DBSCAN
            import numpy as np
            
            # Extract values for clustering
            values = np.array([[level['value']] for level in price_levels])
            
            # Use DBSCAN for clustering close price levels
            # Epsilon (eps) should be adjusted based on the price scale
            eps = np.mean(values) * 0.01  # 1% of mean price as epsilon
            clustering = DBSCAN(eps=eps, min_samples=2).fit(values)
            
            # Extract clusters
            labels = clustering.labels_
            unique_labels = set(labels)
            
            # Create confluence areas
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                
                # Get levels in this cluster
                cluster_indices = np.where(labels == label)[0]
                cluster_levels = [price_levels[i] for i in cluster_indices]
                
                # Calculate average price for the cluster
                avg_price = np.mean([level['value'] for level in cluster_levels])
                
                # Calculate confidence score
                confidence_sum = sum(level['confidence'] for level in cluster_levels)
                
                confluence['areas'].append({
                    'price': float(avg_price),
                    'levels': cluster_levels,
                    'count': len(cluster_levels),
                    'confidence': float(confidence_sum / len(cluster_levels))
                })
        
        # Calculate overall confluence score
        if confluence['areas']:
            max_confidence = max(area['confidence'] for area in confluence['areas'])
            max_count = max(area['count'] for area in confluence['areas'])
            confluence['score'] = float(0.5 * max_confidence + 0.5 * (max_count / (max_count + 2)))
        
        return confluence
    
    def _analyze_implications(
        self, 
        patterns: Dict[str, List[Dict[str, Any]]], 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze pattern implications for future price movement.
        
        Args:
            patterns: Detected patterns
            data: Market data
        
        Returns:
            Implications analysis
        """
        # Count bullish vs bearish patterns
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        bullish_confidence = 0.0
        bearish_confidence = 0.0
        
        for pattern_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                bias = pattern.get('bias', 'neutral').lower()
                confidence = pattern.get('confidence', 0.5)
                
                if bias == 'bullish':
                    bullish_count += 1
                    bullish_confidence += confidence
                elif bias == 'bearish':
                    bearish_count += 1
                    bearish_confidence += confidence
                else:
                    neutral_count += 1
        
        # Calculate average confidence
        avg_bullish_confidence = bullish_confidence / bullish_count if bullish_count > 0 else 0.0
        avg_bearish_confidence = bearish_confidence / bearish_count if bearish_count > 0 else 0.0
        
        # Determine overall bias
        total_count = bullish_count + bearish_count + neutral_count
        if total_count == 0:
            overall_bias = 'neutral'
            bias_score = 0.0
        else:
            bullish_weight = bullish_count * avg_bullish_confidence
            bearish_weight = bearish_count * avg_bearish_confidence
            
            if bullish_weight > bearish_weight:
                overall_bias = 'bullish'
                bias_score = bullish_weight / (bullish_weight + bearish_weight + 0.0001)
            elif bearish_weight > bullish_weight:
                overall_bias = 'bearish'
                bias_score = bearish_weight / (bullish_weight + bearish_weight + 0.0001)
            else:
                overall_bias = 'neutral'
                bias_score = 0.0
        
        return {
            'bias': overall_bias,
            'bias_score': float(bias_score),
            'bullish_patterns': bullish_count,
            'bearish_patterns': bearish_count,
            'neutral_patterns': neutral_count,
            'bullish_confidence': float(avg_bullish_confidence),
            'bearish_confidence': float(avg_bearish_confidence)
        }
    
    def _analyze_reliability(self, patterns: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze pattern reliability based on historical performance.
        
        Args:
            patterns: Detected patterns
        
        Returns:
            Reliability analysis
        """
        # This would normally come from a historical pattern performance database
        # For now, we'll use a simplified approach
        
        # Default reliability scores for different pattern types
        reliability_scores = {
            'harmonic_patterns': 0.75,
            'candlestick_patterns': 0.65,
            'chart_patterns': 0.70,
            'support_resistance': 0.80,
            'volume_patterns': 0.60
        }
        
        # Calculate average reliability
        total_reliability = 0.0
        total_patterns = 0
        
        # Track high reliability patterns
        high_reliability = []
        
        for pattern_type, pattern_list in patterns.items():
            type_reliability = reliability_scores.get(pattern_type, 0.5)
            
            for pattern in pattern_list:
                # Pattern-specific reliability would come from a database
                # For now, use the type reliability with a slight adjustment based on confidence
                confidence = pattern.get('confidence', 0.5)
                pattern_reliability = type_reliability * (0.8 + 0.4 * confidence)
                
                # Add to totals
                total_reliability += pattern_reliability
                total_patterns += 1
                
                # Track high reliability patterns
                if pattern_reliability > 0.7:
                    high_reliability.append({
                        'name': pattern.get('name', 'Unknown'),
                        'type': pattern_type,
                        'reliability': float(pattern_reliability),
                        'confidence': float(confidence)
                    })
        
        # Calculate average
        avg_reliability = total_reliability / total_patterns if total_patterns > 0 else 0.0
        
        return {
            'average_reliability': float(avg_reliability),
            'high_reliability_patterns': high_reliability,
            'pattern_count': total_patterns
        }


# Create a singleton instance of the pattern recognition manager
pattern_recognition_manager = PatternRecognition({
    'harmonic_patterns': {
        'enabled': True,
        'min_confidence': 0.6,
        'max_patterns': 10
    },
    'candlestick_patterns': {
        'enabled': True,
        'min_confidence': 0.7,
        'max_patterns': 15
    },
    'chart_patterns': {
        'enabled': True,
        'min_confidence': 0.65,
        'max_patterns': 8
    },
    'support_resistance': {
        'enabled': True,
        'min_confidence': 0.75,
        'max_levels': 10
    },
    'volume_patterns': {
        'enabled': True,
        'min_confidence': 0.6,
        'max_patterns': 5
    }
})
