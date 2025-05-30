#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Voting System Module

This module implements the VotingSystem class that provides sophisticated voting 
mechanisms for the Brain Council to consolidate signals from multiple strategy brains 
and specialized councils into high-confidence trading decisions.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import asyncio

from common.logger import get_logger
from common.exceptions import VotingError
from common.utils import normalize_weights, sigmoid
from common.constants import (
    VOTE_THRESHOLDS, SIGNAL_TYPES, CONFIDENCE_LEVELS,
    DEFAULT_VOTING_CONFIG
)


@dataclass
class VotingResult:
    """Result of a voting process with direction and confidence."""
    
    direction: str
    confidence: float
    agreement_level: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'direction': self.direction,
            'confidence': self.confidence,
            'agreement_level': self.agreement_level,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

@dataclass
class VoterPerformance:
    """Performance metrics for a single voter."""

    total_votes: int = 0
    correct_votes: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    pnl_sum: float = 0.0
    confidence_accuracy_correlation: float = 0.0
    _confidences: List[float] = field(default_factory=list, repr=False)
    _successes: List[int] = field(default_factory=list, repr=False)

    def update(self, vote: Dict[str, Any], decision: Dict[str, Any],
               outcome_successful: bool, outcome_pnl: float) -> None:
        """Update metrics based on a single vote."""
        self.total_votes += 1
        if vote['direction'] == decision['direction'] and outcome_successful:
            self.correct_votes += 1
        self.pnl_sum += outcome_pnl
        self._confidences.append(vote.get('confidence', 0))
        self._successes.append(
            1 if vote['direction'] == decision['direction'] and outcome_successful else 0
        )

    def finalize(self) -> None:
        """Calculate derived metrics after all votes have been processed."""
        if self.total_votes:
            self.win_rate = self.correct_votes / self.total_votes
            self.avg_pnl = self.pnl_sum / self.total_votes

        if len(set(self._successes)) > 1 and len(self._confidences) > 1:
            self.confidence_accuracy_correlation = float(
                np.corrcoef(self._confidences, self._successes)[0, 1]
            )
        else:
            self.confidence_accuracy_correlation = 0.0

        # Clear temporary data
        self._confidences.clear()
        self._successes.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable representation of the metrics."""
        return {
            'total_votes': self.total_votes,
            'correct_votes': self.correct_votes,
            'win_rate': self.win_rate,
            'avg_pnl': self.avg_pnl,
            'pnl_sum': self.pnl_sum,
            'confidence_accuracy_correlation': self.confidence_accuracy_correlation,
        }

class VotingSystem:
    """
    VotingSystem implements sophisticated voting algorithms to consolidate 
    signals from multiple sources into unified high-confidence trading decisions.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the voting system with configuration parameters.
        
        Args:
            config: Configuration dictionary for the voting system
        """
        self.logger = get_logger("brain_council.voting_system")
        
        # Extract voting-specific config or use defaults
        voting_config = config.get("voting_system", DEFAULT_VOTING_CONFIG)
        
        # Thresholds for voting decisions
        self.confidence_threshold = voting_config.get("confidence_threshold", 0.6)
        self.agreement_threshold = voting_config.get("agreement_threshold", 0.65)
        self.min_voters_required = voting_config.get("min_voters_required", 2)
        
        # Voting methods and parameters
        self.primary_method = voting_config.get("primary_method", "weighted_confidence")
        self.secondary_method = voting_config.get("secondary_method", "majority_vote")
        self.tie_breaking_method = voting_config.get("tie_breaking_method", "highest_confidence")
        
        # Advanced parameters
        self.use_conviction_weighting = voting_config.get("use_conviction_weighting", True)
        self.use_historical_accuracy = voting_config.get("use_historical_accuracy", True)
        self.use_confidence_boosting = voting_config.get("use_confidence_boosting", True)
        self.conflict_resolution_strategy = voting_config.get("conflict_resolution_strategy", "weighted_average")
        
        # History tracking for adaptive voting
        self.voting_history = []
        self.history_max_size = voting_config.get("history_max_size", 1000)
        
        self.logger.info(f"Voting system initialized with {self.primary_method} as primary method")
    
    async def generate_decision(
        self, 
        votes: Dict[str, Dict[str, Any]], 
        weights: Dict[str, float],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a consolidated decision from multiple votes.
        
        Args:
            votes: Dictionary of votes from different sources
            weights: Dictionary of weights for each source
            context: Optional contextual information for decision enhancement
            
        Returns:
            Decision dictionary with direction, confidence, and supporting data
        """
        try:
            self.logger.debug(f"Generating decision from {len(votes)} votes")
            
            # Standardize and validate votes
            standardized_votes = self._standardize_votes(votes)
            
            # Check if we have enough valid votes
            if len(standardized_votes) < self.min_voters_required:
                self.logger.warning(f"Insufficient voters: {len(standardized_votes)} < {self.min_voters_required}")
                return self._generate_no_decision_signal(context)
            
            # Normalize weights to sum to 1.0
            normalized_weights = normalize_weights({k: v for k, v in weights.items() if k in standardized_votes})
            
            # Apply historical accuracy adjustments if enabled
            if self.use_historical_accuracy:
                normalized_weights = self._apply_historical_accuracy(normalized_weights, standardized_votes)
            
            # Execute primary voting method
            if self.primary_method == "weighted_confidence":
                decision = self._weighted_confidence_vote(standardized_votes, normalized_weights)
            elif self.primary_method == "conviction_vote":
                decision = self._conviction_vote(standardized_votes, normalized_weights)
            elif self.primary_method == "bayesian_vote":
                decision = self._bayesian_vote(standardized_votes, normalized_weights)
            else:
                # Default to simple majority vote
                decision = self._majority_vote(standardized_votes, normalized_weights)
            
            # Check if we have a clear decision
            if decision['confidence'] < self.confidence_threshold:
                self.logger.info(f"Decision confidence {decision['confidence']} below threshold {self.confidence_threshold}")
                
                # Try secondary method if primary didn't yield a confident result
                if self.secondary_method != self.primary_method:
                    if self.secondary_method == "weighted_confidence":
                        backup_decision = self._weighted_confidence_vote(standardized_votes, normalized_weights)
                    elif self.secondary_method == "conviction_vote":
                        backup_decision = self._conviction_vote(standardized_votes, normalized_weights)
                    elif self.secondary_method == "bayesian_vote":
                        backup_decision = self._bayesian_vote(standardized_votes, normalized_weights)
                    else:
                        backup_decision = self._majority_vote(standardized_votes, normalized_weights)
                    
                    # If backup method is more confident, use that instead
                    if backup_decision['confidence'] > decision['confidence']:
                        decision = backup_decision
                        self.logger.info(f"Using secondary method with higher confidence: {decision['confidence']}")
                
                # If still below threshold, return a 'hold' or 'no decision' signal
                if decision['confidence'] < self.confidence_threshold:
                    return self._generate_no_decision_signal(context)
            
            # Apply confidence boosting if enabled
            if self.use_confidence_boosting:
                decision = self._boost_confidence(decision, standardized_votes, normalized_weights)
            
            # Add metadata about the voting process
            decision['voting_metadata'] = {
                'method_used': self.primary_method,
                'voters_count': len(standardized_votes),
                'agreement_level': decision.get('agreement_level', 0),
                'weights': normalized_weights,
                'timestamp': time.time()
            }
            
            # Add contextual information if provided
            if context:
                decision['context'] = context
            
            # Record this decision for historical tracking
            self._record_decision(decision, standardized_votes, normalized_weights)
            
            self.logger.info(
                f"Generated {decision['direction']} decision with "
                f"{decision['confidence']:.2f} confidence"
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error generating decision: {str(e)}")
            raise VotingError(f"Failed to generate decision: {str(e)}")
    
    def _standardize_votes(self, votes: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Standardize the format of votes for consistent processing.
        
        Args:
            votes: Raw votes from various sources
            
        Returns:
            Standardized votes dictionary
        """
        standardized = {}
        
        for source, vote in votes.items():
            # Skip empty or None votes
            if not vote:
                continue
            
            # Ensure required fields
            if 'direction' not in vote:
                self.logger.warning(f"Vote from {source} missing direction, skipping")
                continue
            
            # Standardize direction to one of: buy, sell, hold
            direction = vote['direction'].lower()
            if direction not in ['buy', 'sell', 'hold']:
                self.logger.warning(f"Unknown direction '{direction}' from {source}, skipping")
                continue
            
            # Ensure confidence is a float between 0 and 1
            confidence = vote.get('confidence', 0.5)
            try:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid confidence value from {source}, using default 0.5")
                confidence = 0.5
            
            # Create standardized vote entry
            standardized[source] = {
                'direction': direction,
                'confidence': confidence,
                'timestamp': vote.get('timestamp', time.time()),
                'signals': vote.get('signals', {}),
                'metadata': vote.get('metadata', {})
            }
        
        return standardized
    
    def _majority_vote(
        self, 
        votes: Dict[str, Dict[str, Any]], 
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Perform a simple majority vote, optionally weighted by source importance.
        
        Args:
            votes: Standardized votes from different sources
            weights: Normalized weights for each source
            
        Returns:
            Decision dictionary with direction and confidence
        """
        # Count votes for each direction
        vote_counts = {'buy': 0, 'sell': 0, 'hold': 0}
        
        # With weighting
        for source, vote in votes.items():
            direction = vote['direction']
            weight = weights.get(source, 1.0 / len(votes))  # Default to equal weight
            vote_counts[direction] += weight
        
        # Find the winner
        winner_direction = max(vote_counts.items(), key=lambda x: x[1])[0]
        total_votes = sum(vote_counts.values())
        
        # Calculate agreement level and confidence
        agreement_level = vote_counts[winner_direction] / total_votes if total_votes > 0 else 0
        
        # Confidence based on agreement level
        # Scaled to ensure 100% agreement = 1.0 confidence, 33% agreement = 0.0 confidence
        base_confidence = (agreement_level - 1/3) / (2/3) if agreement_level > 1/3 else 0
        
        # Calculate average conviction for the winning direction
        conviction_sum = 0
        conviction_count = 0
        
        for source, vote in votes.items():
            if vote['direction'] == winner_direction:
                conviction_sum += vote['confidence']
                conviction_count += 1
        
        avg_conviction = conviction_sum / conviction_count if conviction_count > 0 else 0.5
        
        # Blend agreement-based confidence with average conviction
        confidence = 0.7 * base_confidence + 0.3 * avg_conviction
        confidence = max(0.0, min(1.0, confidence))  # Ensure in [0,1]
        
        return {
            'direction': winner_direction,
            'confidence': confidence,
            'agreement_level': agreement_level,
            'vote_counts': vote_counts
        }
    
    def _weighted_confidence_vote(
        self, 
        votes: Dict[str, Dict[str, Any]], 
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Perform a weighted vote that considers both source weight and signal confidence.
        
        Args:
            votes: Standardized votes from different sources
            weights: Normalized weights for each source
            
        Returns:
            Decision dictionary with direction and confidence
        """
        # Calculate weighted votes for each direction
        weighted_votes = {'buy': 0, 'sell': 0, 'hold': 0}
        
        for source, vote in votes.items():
            direction = vote['direction']
            source_weight = weights.get(source, 1.0 / len(votes))
            
            # Weight by both source importance and signal confidence
            signal_weight = source_weight * vote['confidence']
            weighted_votes[direction] += signal_weight
        
        # Find the winner
        winner_direction = max(weighted_votes.items(), key=lambda x: x[1])[0]
        total_weighted_votes = sum(weighted_votes.values())
        
        # Calculate agreement level
        agreement_level = weighted_votes[winner_direction] / total_weighted_votes if total_weighted_votes > 0 else 0
        
        # Calculate confidence based on agreement and conviction
        # More sophisticated than majority vote - scales better with weighted inputs
        confidence = weighted_votes[winner_direction] / total_weighted_votes if total_weighted_votes > 0 else 0
        
        # Apply sigmoid transformation to create more separation between confident and uncertain signals
        confidence = sigmoid(confidence * 5 - 2.5)
        
        return {
            'direction': winner_direction,
            'confidence': confidence,
            'agreement_level': agreement_level,
            'weighted_votes': weighted_votes
        }
    
    def _conviction_vote(
        self, 
        votes: Dict[str, Dict[str, Any]], 
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Perform a conviction-based vote where both agreement and conviction matter.
        
        Args:
            votes: Standardized votes from different sources
            weights: Normalized weights for each source
            
        Returns:
            Decision dictionary with direction and confidence
        """
        # Similar to weighted confidence but with emphasis on strong convictions
        direction_weights = {'buy': 0, 'sell': 0, 'hold': 0}
        direction_convictions = {'buy': [], 'sell': [], 'hold': []}
        
        for source, vote in votes.items():
            direction = vote['direction']
            source_weight = weights.get(source, 1.0 / len(votes))
            
            # Square the confidence to emphasize strong convictions
            conviction = vote['confidence'] ** 2
            
            # Add to direction weights
            direction_weights[direction] += source_weight
            
            # Record conviction for this direction
            direction_convictions[direction].append((conviction, source_weight))
        
        # Find the winning direction based on weights
        winner_direction = max(direction_weights.items(), key=lambda x: x[1])[0]
        total_weight = sum(direction_weights.values())
        
        # Calculate agreement level
        agreement_level = direction_weights[winner_direction] / total_weight if total_weight > 0 else 0
        
        # Calculate average conviction for the winning direction, weighted by source
        if direction_convictions[winner_direction]:
            total_conviction_weight = 0
            weighted_conviction_sum = 0
            
            for conviction, source_weight in direction_convictions[winner_direction]:
                weighted_conviction_sum += conviction * source_weight
                total_conviction_weight += source_weight
            
            avg_conviction = weighted_conviction_sum / total_conviction_weight
        else:
            avg_conviction = 0.5  # Default
        
        # Final confidence is a blend of agreement and conviction
        if agreement_level >= self.agreement_threshold:
            # High agreement - conviction becomes more important
            confidence = 0.4 * agreement_level + 0.6 * avg_conviction
        else:
            # Low agreement - prioritize agreement level
            confidence = 0.7 * agreement_level + 0.3 * avg_conviction
        
        # Ensure confidence is in [0,1]
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            'direction': winner_direction,
            'confidence': confidence,
            'agreement_level': agreement_level,
            'direction_weights': direction_weights,
            'avg_conviction': avg_conviction
        }
    
    def _bayesian_vote(
        self, 
        votes: Dict[str, Dict[str, Any]], 
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Perform a Bayesian-inspired vote that treats each vote as evidence.
        
        Args:
            votes: Standardized votes from different sources
            weights: Normalized weights for each source
            
        Returns:
            Decision dictionary with direction and confidence
        """
        # Start with equal prior probabilities
        priors = {'buy': 1/3, 'sell': 1/3, 'hold': 1/3}
        
        # Calculate posterior probabilities using votes as evidence
        posteriors = priors.copy()
        
        for source, vote in votes.items():
            direction = vote['direction']
            source_weight = weights.get(source, 1.0 / len(votes))
            confidence = vote['confidence']
            
            # Modify weight based on confidence
            effective_weight = source_weight * confidence
            
            # Simple Bayesian update rule (simplified for this context)
            # Increase probability of the voted direction
            posteriors[direction] *= (1 + effective_weight)
            
            # Normalize to maintain valid probability distribution
            total = sum(posteriors.values())
            for dir in posteriors:
                posteriors[dir] /= total
        
        # Find direction with highest posterior probability
        winner_direction = max(posteriors.items(), key=lambda x: x[1])[0]
        
        # Confidence is the posterior probability
        confidence = posteriors[winner_direction]
        
        # Calculate a measure of certainty/agreement
        # Higher certainty means more probability mass concentrated on one direction
        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in posteriors.values())
        max_entropy = -np.log(1/3)  # Maximum entropy with 3 equal options
        certainty = 1 - (entropy / max_entropy)
        
        return {
            'direction': winner_direction,
            'confidence': confidence,
            'agreement_level': certainty,
            'posteriors': posteriors
        }
    
    def _boost_confidence(
        self, 
        decision: Dict[str, Any], 
        votes: Dict[str, Dict[str, Any]], 
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Apply confidence boosting based on additional factors.
        
        Args:
            decision: Preliminary decision dictionary
            votes: Standardized votes from different sources
            weights: Normalized weights for each source
            
        Returns:
            Updated decision with potentially boosted confidence
        """
        direction = decision['direction']
        base_confidence = decision['confidence']
        boost_factors = []
        
        # Factor 1: Agreement across different source types
        # Check if we have votes from different source types
        source_types = set()
        for source in votes:
            # Extract source type from source id (e.g., "timeframe_council" -> "timeframe")
            source_type = source.split('_')[0] if '_' in source else source
            source_types.add(source_type)
        
        # Boost if multiple source types agree
        source_type_count = len(source_types)
        agreement_boost = 0
        if source_type_count >= 3 and decision.get('agreement_level', 0) > 0.8:
            agreement_boost = 0.1
        elif source_type_count >= 2 and decision.get('agreement_level', 0) > 0.9:
            agreement_boost = 0.05
        
        boost_factors.append(agreement_boost)
        
        # Factor 2: Conviction strength - boost if the winning signals have very high conviction
        conviction_boost = 0
        high_conviction_count = sum(1 for vote in votes.values() 
                                   if vote['direction'] == direction and vote['confidence'] > 0.85)
        
        if high_conviction_count >= 3:
            conviction_boost = 0.15
        elif high_conviction_count >= 2:
            conviction_boost = 0.1
        elif high_conviction_count >= 1:
            conviction_boost = 0.05
            
        boost_factors.append(conviction_boost)
        
        # Factor 3: Weight of strongest voter - if a very high-weight source is confident
        max_weight_source = max(weights.items(), key=lambda x: x[1])[0]
        max_weight = weights[max_weight_source]
        max_weight_vote = votes.get(max_weight_source, {})
        
        weight_boost = 0
        if (max_weight > 0.4 and 
            max_weight_vote.get('direction') == direction and 
            max_weight_vote.get('confidence', 0) > 0.8):
            weight_boost = 0.1
            
        boost_factors.append(weight_boost)
        
        # Apply the highest boost factor (don't stack them all)
        max_boost = max(boost_factors)
        boosted_confidence = min(base_confidence + max_boost, 1.0)
        
        # Update the decision with boosted confidence
        decision['confidence'] = boosted_confidence
        decision['confidence_boost'] = max_boost
        decision['boost_factors'] = {
            'agreement_boost': agreement_boost,
            'conviction_boost': conviction_boost,
            'weight_boost': weight_boost
        }
        
        return decision
    
    def _apply_historical_accuracy(
        self, 
        weights: Dict[str, float], 
        votes: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Adjust weights based on historical accuracy of each voter.
        
        Args:
            weights: Original normalized weights
            votes: Current votes from different sources
            
        Returns:
            Adjusted weights dictionary
        """
        # Not enough history yet
        if len(self.voting_history) < 10:
            return weights
        
        adjusted_weights = weights.copy()
        recent_history = self.voting_history[-100:]  # Use recent history
        
        # Calculate historical accuracy for each source
        source_accuracy = {}
        for source in votes:
            correct_votes = 0
            total_votes = 0
            
            for decision in recent_history:
                # Check if this source contributed to this historical decision
                if source in decision.get('votes', {}):
                    total_votes += 1
                    
                    # Check if this source's direction matched the final outcome
                    if (decision['votes'][source]['direction'] == decision.get('outcome_direction') and
                        decision.get('outcome_successful', False)):
                        correct_votes += 1
            
            # Calculate accuracy if we have enough samples
            if total_votes >= 5:
                accuracy = correct_votes / total_votes
                source_accuracy[source] = accuracy
        
        # Adjust weights based on accuracy
        for source, accuracy in source_accuracy.items():
            # Adjust weight based on historical accuracy
            # Scale factor determines how much history influences weights
            scale_factor = 0.5
            adjusted_weights[source] = weights[source] * (1 + (accuracy - 0.5) * scale_factor)
        
        # Renormalize weights
        return normalize_weights(adjusted_weights)
    
    def _generate_no_decision_signal(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a 'no decision' or 'hold' signal.
        
        Args:
            context: Optional contextual information
            
        Returns:
            Hold signal with low confidence
        """
        signal = {
            'direction': 'hold',
            'confidence': 0.3,  # Low confidence
            'agreement_level': 0,
            'reason': 'insufficient_confidence',
            'timestamp': time.time()
        }
        
        if context:
            signal['context'] = context
            
        return signal
    
    def _record_decision(
        self, 
        decision: Dict[str, Any], 
        votes: Dict[str, Dict[str, Any]], 
        weights: Dict[str, float]
    ) -> None:
        """
        Record a decision for historical tracking and learning.
        
        Args:
            decision: The final decision
            votes: Votes that contributed to the decision
            weights: Weights used for each source
        """
        # Create a record including the decision and contributing votes
        record = {
            'decision': decision.copy(),
            'votes': votes.copy(),
            'weights': weights.copy(),
            'timestamp': time.time(),
            'outcome_successful': None,  # To be filled later when outcome is known
            'outcome_direction': None,
            'outcome_pnl': None
        }
        
        # Add to history
        self.voting_history.append(record)
        
        # Trim history if it exceeds maximum size
        if len(self.voting_history) > self.history_max_size:
            self.voting_history = self.voting_history[-self.history_max_size:]
    
    async def update_decision_outcome(
        self, 
        decision_timestamp: float, 
        successful: bool, 
        actual_direction: Optional[str] = None, 
        pnl: Optional[float] = None
    ) -> bool:
        """
        Update a previous decision with its actual outcome for learning.
        
        Args:
            decision_timestamp: Timestamp of the decision to update
            successful: Whether the decision led to a successful trade
            actual_direction: The direction that would have been correct
            pnl: Profit/loss from the trade
            
        Returns:
            Boolean indicating whether the update was successful
        """
        # Find the decision in history
        for record in reversed(self.voting_history):
            if abs(record['timestamp'] - decision_timestamp) < 1.0:  # Allow for small timestamp differences
                # Update the record
                record['outcome_successful'] = successful
                if actual_direction:
                    record['outcome_direction'] = actual_direction
                if pnl is not None:
                    record['outcome_pnl'] = pnl
                
                self.logger.debug(f"Updated decision outcome: success={successful}, pnl={pnl}")
                return True
        
        self.logger.warning(f"Could not find decision with timestamp {decision_timestamp} to update")
        return False
    
    async def get_voter_performance(self) -> Dict[str, Dict[str, Any]]:
        """Return performance metrics for each voter."""
        if len(self.voting_history) < 10:
            return {}

        voter_metrics = defaultdict(VoterPerformance)
        
        # Track metrics for each voter
        voter_metrics = defaultdict(lambda: {
            'total_votes': 0,
            'correct_votes': 0,
            'win_rate': 0,
            'avg_pnl': 0,
            'pnl_sum': 0,
            'confidence_accuracy_correlation': 0,
            'confidences': [],
            'successes': []
        })
        
        # Process historical decisions

        for record in self.voting_history:
            if record.get('outcome_successful') is None:
                continue

            outcome_successful = record['outcome_successful']
            outcome_pnl = record.get('outcome_pnl', 0)

            for voter, vote in record.get('votes', {}).items():

                voter_metrics[voter].update(
                    vote,
                    record['decision'],
                    outcome_successful,
                    outcome_pnl,
                )
        
        for metrics in voter_metrics.values():
            metrics.finalize()

        return {voter: m.to_dict() for voter, m in voter_metrics.items()}
        
        # Calculate derived metrics
        for voter, metrics in voter_metrics.items():
            if metrics['total_votes'] > 0:
                metrics['win_rate'] = metrics['correct_votes'] / metrics['total_votes']
                metrics['avg_pnl'] = metrics['pnl_sum'] / metrics['total_votes']

                # Calculate correlation between voter confidence and success
                if len(set(metrics['successes'])) > 1 and len(metrics['confidences']) > 1:
                    metrics['confidence_accuracy_correlation'] = float(
                        np.corrcoef(metrics['confidences'], metrics['successes'])[0, 1]
                    )
                else:
                    metrics['confidence_accuracy_correlation'] = 0.0

                # Remove temporary lists before returning
                del metrics['confidences']
                del metrics['successes']
        
        return dict(voter_metrics)

    
    def get_voting_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the voting system's performance.
        
        Returns:
            Dictionary with voting performance statistics
        """
        if len(self.voting_history) < 10:
            return {'status': 'insufficient_data'}
        
        # Count successful decisions
        completed_decisions = [d for d in self.voting_history if d.get('outcome_successful') is not None]
        successful_decisions = [d for d in completed_decisions if d.get('outcome_successful', False)]
        
        # Calculate win rate
        win_rate = len(successful_decisions) / len(completed_decisions) if completed_decisions else 0
        
        # Calculate average PnL
        pnl_values = [d.get('outcome_pnl', 0) for d in completed_decisions if d.get('outcome_pnl') is not None]
        avg_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else 0
        
        # Calculate method success rates
        method_stats = defaultdict(lambda: {'count': 0, 'successful': 0, 'win_rate': 0})
        
        for decision in completed_decisions:
            method = decision['decision'].get('voting_metadata', {}).get('method_used', 'unknown')
            method_stats[method]['count'] += 1
            
            if decision.get('outcome_successful', False):
                method_stats[method]['successful'] += 1
        
        # Calculate win rates for each method
        for method, stats in method_stats.items():
            if stats['count'] > 0:
                stats['win_rate'] = stats['successful'] / stats['count']
        
        # Assess confidence accuracy
        confidence_bins = defaultdict(lambda: {'count': 0, 'successful': 0, 'win_rate': 0})
        
        for decision in completed_decisions:
            confidence = decision['decision'].get('confidence', 0)
            bin_key = int(confidence * 10) / 10  # Round to nearest 0.1
            
            confidence_bins[bin_key]['count'] += 1
            if decision.get('outcome_successful', False):
                confidence_bins[bin_key]['successful'] += 1
        
        # Calculate win rates for each confidence bin
        for bin_key, stats in confidence_bins.items():
            if stats['count'] > 0:
                stats['win_rate'] = stats['successful'] / stats['count']
        
        return {
            'overall_win_rate': win_rate,
            'sample_size': len(completed_decisions),
            'avg_pnl': avg_pnl,
            'method_stats': dict(method_stats),
            'confidence_calibration': dict(confidence_bins)
        }
