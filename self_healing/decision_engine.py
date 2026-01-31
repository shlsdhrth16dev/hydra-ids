"""
Enhanced Decision Engine for Self-Healing Actions.

Provides intelligent, confidence-based decision making with
policy support, rate limiting, and comprehensive reasoning.
"""

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime, timedelta
from collections import deque
import logging

from .config import DecisionEngineConfig
from .exceptions import DecisionEngineError


logger = logging.getLogger(__name__)


ActionType = Literal["no_action", "monitor", "alert", "investigate", "retrain", "rollback"]


class DecisionEngine:
    """
    Intelligent decision engine for self-healing actions.
    
    Features:
    - Multi-level actions (monitor, alert, investigate, retrain, rollback)
    - Confidence scoring for decisions
    - Policy-based decision framework (conservative, balanced, aggressive)
    - Rate limiting to prevent excessive retraining
    - Cooldown periods and backoff strategies
    - Decision history and audit trail
    - Explainable decision reasoning
    
    Args:
        config: DecisionEngineConfig instance
    """
    
    def __init__(self, config: Optional[DecisionEngineConfig] = None):
        if config is None:
            config = DecisionEngineConfig()
        
        self.config = config
        
        # Decision history
        self.decision_history: deque = deque(maxlen=100)
        
        # Rate limiting tracking
        self.retrain_timestamps: List[datetime] = []
        self.last_action_time: Optional[datetime] = None
        
        logger.info(
            "DecisionEngine initialized with policy=%s, max_retrains=%d/day",
            config.policy, config.max_retrains_per_day
        )
    
    def decide(
        self,
        health_status: Dict[str, Any],
        drift_status: Dict[str, Any],
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make intelligent decision on self-healing action.
        
        Args:
            health_status: Health monitoring results
            drift_status: Drift detection results
            additional_context: Optional additional context for decision making
        
        Returns:
            Decision dictionary with action, confidence, and reasoning
        
        Raises:
            DecisionEngineError: If decision making fails
        """
        try:
            timestamp = datetime.now()
            
            # Extract key indicators
            is_healthy = health_status.get('is_healthy', True)
            drift_detected = drift_status.get('drift_detected', False)
            
            # Get severity levels
            health_severity = self._assess_health_severity(health_status)
            drift_severity = self._assess_drift_severity(drift_status)
            
            # Check rate limiting
            can_retrain = self._check_rate_limit()
            in_cooldown = self._check_cooldown()
            
            # Make decision based on policy
            action, confidence, reasoning = self._make_decision(
                is_healthy=is_healthy,
                drift_detected=drift_detected,
                health_severity=health_severity,
                drift_severity=drift_severity,
                can_retrain=can_retrain,
                in_cooldown=in_cooldown
            )
            
            # Check confidence threshold
            if confidence < self.config.min_confidence_for_action and action not in ["no_action", "monitor"]:
                logger.warning(
                    "Decision confidence %.2f below threshold %.2f, downgrading to monitor",
                    confidence, self.config.min_confidence_for_action
                )
                reasoning.append(f"Low confidence ({confidence:.2f}), downgraded action")
                action = "monitor"
            
            # Build decision report
            decision = {
                'timestamp': timestamp.isoformat(),
                'action': action,
                'confidence': confidence,
                'reasoning': reasoning,
                'policy': self.config.policy,
                'health_severity': health_severity,
                'drift_severity': drift_severity,
                'is_healthy': is_healthy,
                'drift_detected': drift_detected,
                'can_retrain': can_retrain,
                'in_cooldown': in_cooldown,
                'metadata': {
                    'recent_retrains': len(self.retrain_timestamps),
                    'last_action_time': self.last_action_time.isoformat() if self.last_action_time else None,
                }
            }
            
            # Record decision
            self.decision_history.append(decision)
            
            # Update tracking
            if action == "retrain":
                self.retrain_timestamps.append(timestamp)
            self.last_action_time = timestamp
            
            # Log decision
            logger.info(
                "Decision made: action=%s, confidence=%.2f, reasoning='%s'",
                action, confidence, '; '.join(reasoning)
            )
            
            if action in ["retrain", "rollback"]:
                logger.warning("Critical action recommended: %s", action)
            
            return decision
            
        except Exception as e:
            logger.error("Decision making failed: %s", str(e), exc_info=True)
            raise DecisionEngineError(f"Decision making failed: {e}") from e
    
    def _assess_health_severity(self, health_status: Dict[str, Any]) -> str:
        """Assess health severity level."""
        metrics = health_status.get('metrics', {})
        
        # Get worst metric
        key_metrics = ['recall', 'precision', 'f1', 'accuracy']
        available_metrics = [metrics.get(m, 1.0) for m in key_metrics if m in metrics]
        
        if not available_metrics:
            return "unknown"
        
        worst_metric = min(available_metrics)
        
        if worst_metric < self.config.critical_health_threshold:
            return "critical"
        elif worst_metric < self.config.warning_health_threshold:
            return "warning"
        else:
            return "healthy"
    
    def _assess_drift_severity(self, drift_status: Dict[str, Any]) -> str:
        """Assess drift severity level."""
        drift_ratio = drift_status.get('drift_ratio', 0.0)
        
        if drift_ratio >= self.config.critical_drift_threshold:
            return "critical"
        elif drift_ratio >= self.config.warning_drift_threshold:
            return "warning"
        else:
            return "minimal"
    
    def _check_rate_limit(self) -> bool:
        """Check if retraining is allowed under rate limits."""
        if not self.config.enable_rate_limiting:
            return True
        
        # Clean old timestamps
        cutoff = datetime.now() - timedelta(days=1)
        self.retrain_timestamps = [
            ts for ts in self.retrain_timestamps if ts > cutoff
        ]
        
        return len(self.retrain_timestamps) < self.config.max_retrains_per_day
    
    def _check_cooldown(self) -> bool:
        """Check if we're in cooldown period."""
        if not self.config.enable_rate_limiting or not self.last_action_time:
            return False
        
        time_since_last = datetime.now() - self.last_action_time
        cooldown_period = timedelta(hours=self.config.cooldown_hours)
        
        return time_since_last < cooldown_period
    
    def _make_decision(
        self,
        is_healthy: bool,
        drift_detected: bool,
        health_severity: str,
        drift_severity: str,
        can_retrain: bool,
        in_cooldown: bool
    ) -> tuple[ActionType, float, List[str]]:
        """
        Core decision logic based on policy and conditions.
        
        Returns:
            (action, confidence, reasoning)
        """
        reasoning = []
        
        # Policy-specific logic
        if self.config.policy == "conservative":
            return self._conservative_policy(
                is_healthy, drift_detected, health_severity, drift_severity,
                can_retrain, in_cooldown, reasoning
            )
        elif self.config.policy == "aggressive":
            return self._aggressive_policy(
                is_healthy, drift_detected, health_severity, drift_severity,
                can_retrain, in_cooldown, reasoning
            )
        else:  # balanced
            return self._balanced_policy(
                is_healthy, drift_detected, health_severity, drift_severity,
                can_retrain, in_cooldown, reasoning
            )
    
    def _conservative_policy(
        self, is_healthy: bool, drift_detected: bool,
        health_severity: str, drift_severity: str,
        can_retrain: bool, in_cooldown: bool, reasoning: List[str]
    ) -> tuple[ActionType, float, List[str]]:
        """Conservative policy: only act when absolutely necessary."""
        
        # Critical health + drift = retrain (if allowed)
        if health_severity == "critical" and drift_detected:
            if can_retrain and not in_cooldown:
                reasoning.append("Critical health with drift detected")
                return "retrain", 0.95, reasoning
            else:
                reasoning.append("Would retrain but rate limited")
                return "alert", 0.90, reasoning
        
        # Critical health without drift = investigate (possible data issue)
        if health_severity == "critical":
            reasoning.append("Critical health without drift - possible data issue")
            return "investigate", 0.85, reasoning
        
        # Warning level = alert
        if health_severity == "warning" or drift_severity == "warning":
            reasoning.append(f"Warning level: health={health_severity}, drift={drift_severity}")
            return "alert", 0.70, reasoning
        
        # Everything else = monitor
        reasoning.append("System within acceptable parameters")
        return "monitor", 0.95, reasoning
    
    def _aggressive_policy(
        self, is_healthy: bool, drift_detected: bool,
        health_severity: str, drift_severity: str,
        can_retrain: bool, in_cooldown: bool, reasoning: List[str]
    ) -> tuple[ActionType, float, List[str]]:
        """Aggressive policy: proactive retraining."""
        
        # Any drift or health issue = retrain
        if (not is_healthy or drift_detected) and can_retrain and not in_cooldown:
            reasoning.append("Proactive retraining on any degradation signal")
            confidence = 0.90 if health_severity == "critical" else 0.75
            return "retrain", confidence, reasoning
        
        # Rate limited = alert
        if not is_healthy or drift_detected:
            reasoning.append("Would retrain but rate limited")
            return "alert", 0.80, reasoning
        
        # Monitor for everything else
        reasoning.append("No issues detected")
        return "monitor", 0.95, reasoning
    
    def _balanced_policy(
        self, is_healthy: bool, drift_detected: bool,
        health_severity: str, drift_severity: str,
        can_retrain: bool, in_cooldown: bool, reasoning: List[str]
    ) -> tuple[ActionType, float, List[str]]:
        """Balanced policy: moderate approach."""
        
        # Critical health + drift = retrain
        if health_severity == "critical" and drift_detected:
            if can_retrain and not in_cooldown:
                reasoning.append("Critical health with drift")
                return "retrain", 0.95, reasoning
            else:
                reasoning.append("Critical but rate limited")
                return "rollback", 0.85, reasoning
        
        # Critical health without drift = rollback
        if health_severity == "critical":
            reasoning.append("Critical health without drift - rolling back")
            return "rollback", 0.90, reasoning
        
        # Warning health + warning drift = investigate
        if health_severity == "warning" and drift_severity == "warning":
            reasoning.append("Both health and drift at warning levels")
            return "investigate", 0.75, reasoning
        
        # Any warning = alert
        if health_severity == "warning" or drift_severity == "warning":
            reasoning.append(f"Warning detected: health={health_severity}, drift={drift_severity}")
            return "alert", 0.80, reasoning
        
        # Drift only = monitor closely
        if drift_detected:
            reasoning.append("Drift detected but health is good")
            return "monitor", 0.85, reasoning
        
        # All good
        reasoning.append("System healthy")
        return "no_action", 0.98, reasoning
    
    def get_decision_summary(self) -> Dict[str, Any]:
        """Get summary of decision history."""
        if not self.decision_history:
            return {'decisions_made': 0}
        
        # Count actions
        action_counts = {}
        for decision in self.decision_history:
            action = decision['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            'decisions_made': len(self.decision_history),
            'action_counts': action_counts,
            'recent_decisions': list(self.decision_history)[-5:],
            'avg_confidence': sum(d['confidence'] for d in self.decision_history) / len(self.decision_history),
            'retrains_last_24h': len(self.retrain_timestamps),
        }
    
    def reset_rate_limits(self) -> None:
        """Reset rate limiting counters."""
        self.retrain_timestamps.clear()
        self.last_action_time = None
        logger.info("Rate limits reset")
