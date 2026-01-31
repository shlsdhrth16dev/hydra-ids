"""
Enhanced Health Monitor for IDS Model Performance.

Provides comprehensive monitoring with multi-metric tracking,
trend analysis, adaptive thresholds, and detailed logging.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    recall_score, accuracy_score, precision_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from collections import deque
from datetime import datetime
import logging

from .config import HealthMonitorConfig
from .exceptions import HealthMonitorError


logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Advanced health monitoring for ML models.
    
    Features:
    - Multi-metric tracking (recall, precision, F1, accuracy, ROC-AUC, PR-AUC)
    - Trend analysis with rolling windows
    - Performance degradation detection
    - Adaptive threshold adjustment
    - Support for multi-class classification
    - Detailed logging and reporting
    
    Args:
        config: HealthMonitorConfig instance
        recall_threshold: Deprecated, use config instead
    """
    
    def __init__(
        self,
        config: Optional[HealthMonitorConfig] = None,
        recall_threshold: Optional[float] = None  # Backward compatibility
    ):
        # Handle backward compatibility
        if config is None:
            config = HealthMonitorConfig()
            if recall_threshold is not None:
                config.recall_threshold = recall_threshold
        
        self.config = config
        
        # Metric history for trend tracking
        self.history: Dict[str, deque] = {
            'recall': deque(maxlen=config.trend_window_size),
            'precision': deque(maxlen=config.trend_window_size),
            'f1': deque(maxlen=config.trend_window_size),
            'accuracy': deque(maxlen=config.trend_window_size),
        }
        
        # Adaptive thresholds
        self.adaptive_thresholds: Dict[str, float] = {
            'recall': config.recall_threshold,
            'precision': config.precision_threshold,
            'f1': config.f1_threshold,
            'accuracy': config.accuracy_threshold,
        }
        
        # Evaluation counter
        self.evaluation_count = 0
        
        logger.info("HealthMonitor initialized with config: %s", config.model_dump())
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model health with comprehensive metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (for ROC-AUC, PR-AUC)
        
        Returns:
            Dictionary with health metrics and status
        
        Raises:
            HealthMonitorError: If evaluation fails
        """
        try:
            self.evaluation_count += 1
            timestamp = datetime.now().isoformat()
            
            # Calculate base metrics
            metrics = self._calculate_metrics(y_true, y_pred, y_pred_proba)
            
            # Update history
            if self.config.enable_trend_tracking:
                self._update_history(metrics)
            
            # Detect degradation
            degradation = self._detect_degradation() if self.config.enable_trend_tracking else None
            
            # Update adaptive thresholds
            if self.config.enable_adaptive_thresholds:
                self._update_adaptive_thresholds(metrics)
            
            # Determine health status
            is_healthy = self._determine_health_status(metrics)
            
            # Calculate confidence matrix
            conf_matrix = confusion_matrix(y_true, y_pred).tolist()
            
            # Build comprehensive health report
            health_report = {
                'timestamp': timestamp,
                'evaluation_count': self.evaluation_count,
                'metrics': metrics,
                'is_healthy': is_healthy,
                'thresholds': {
                    'recall': self.adaptive_thresholds['recall'],
                    'precision': self.adaptive_thresholds['precision'],
                    'f1': self.adaptive_thresholds['f1'],
                    'accuracy': self.adaptive_thresholds['accuracy'],
                },
                'confusion_matrix': conf_matrix,
                'degradation': degradation,
                'samples_evaluated': len(y_true),
            }
            
            # Log results
            logger.info(
                "Health evaluation #%d: is_healthy=%s, recall=%.4f, precision=%.4f, f1=%.4f",
                self.evaluation_count, is_healthy,
                metrics['recall'], metrics['precision'], metrics['f1']
            )
            
            if not is_healthy:
                logger.warning(
                    "Model health degraded! Failed metrics: %s",
                    [k for k, v in metrics.items() 
                     if k in self.adaptive_thresholds and v < self.adaptive_thresholds[k]]
                )
            
            return health_report
            
        except Exception as e:
            logger.error("Health evaluation failed: %s", str(e), exc_info=True)
            raise HealthMonitorError(f"Health evaluation failed: {e}") from e
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate all performance metrics."""
        # Detect if classification is binary or multiclass
        n_classes = len(np.unique(y_true))
        average_strategy = 'binary' if n_classes == 2 else 'weighted'
        
        metrics = {
            'recall': float(recall_score(y_true, y_pred, average=average_strategy, zero_division=0)),
            'precision': float(precision_score(y_true, y_pred, average=average_strategy, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average=average_strategy, zero_division=0)),
            'accuracy': float(accuracy_score(y_true, y_pred)),
        }
        
        # Add probabilistic metrics if available
        if y_pred_proba is not None:
            try:
                # Handle binary classification
                if n_classes == 2:
                    if y_pred_proba.ndim == 1 or y_pred_proba.shape[1] == 1:
                        proba = y_pred_proba.ravel()
                    else:
                        proba = y_pred_proba[:, 1]
                    
                    metrics['roc_auc'] = float(roc_auc_score(y_true, proba))
                    metrics['pr_auc'] = float(average_precision_score(y_true, proba))
                else:
                    # Multiclass - use OVR strategy
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba, 
                                                             multi_class='ovr', average='weighted'))
                    # PR-AUC for multiclass is more complex, skip for now
                    logger.debug("Skipping PR-AUC for multiclass classification")
            except Exception as e:
                logger.warning("Could not calculate probabilistic metrics: %s", str(e))
        
        return metrics
    
    def _update_history(self, metrics: Dict[str, float]) -> None:
        """Update metric history for trend tracking."""
        for key in ['recall', 'precision', 'f1', 'accuracy']:
            if key in metrics:
                self.history[key].append(metrics[key])
    
    def _detect_degradation(self) -> Optional[Dict[str, Any]]:
        """Detect performance degradation trends."""
        if self.evaluation_count < 2:
            return None
        
        degradation_info = {}
        
        for metric_name, values in self.history.items():
            if len(values) < 2:
                continue
            
            # Calculate trend
            recent_avg = np.mean(list(values)[-3:]) if len(values) >= 3 else values[-1]
            older_avg = np.mean(list(values)[:-3]) if len(values) > 3 else values[0]
            
            degradation = older_avg - recent_avg
            
            if degradation > self.config.degradation_threshold:
                degradation_info[metric_name] = {
                    'degradation': float(degradation),
                    'recent_avg': float(recent_avg),
                    'older_avg': float(older_avg),
                    'is_degraded': True
                }
                logger.warning(
                    "Performance degradation detected in %s: %.4f drop",
                    metric_name, degradation
                )
        
        return degradation_info if degradation_info else None
    
    def _update_adaptive_thresholds(self, metrics: Dict[str, float]) -> None:
        """Update thresholds based on historical performance."""
        if self.evaluation_count < 5:  # Need baseline
            return
        
        for metric_name in ['recall', 'precision', 'f1', 'accuracy']:
            if metric_name not in self.history or len(self.history[metric_name]) == 0:
                continue
            
            # Calculate historical average
            hist_avg = np.mean(self.history[metric_name])
            
            # Move threshold towards historical average
            current_threshold = self.adaptive_thresholds[metric_name]
            new_threshold = (
                current_threshold * (1 - self.config.adaptation_rate) +
                hist_avg * self.config.adaptation_rate
            )
            
            self.adaptive_thresholds[metric_name] = float(new_threshold)
    
    def _determine_health_status(self, metrics: Dict[str, float]) -> bool:
        """Determine if model is healthy based on all metrics."""
        # Check all primary metrics against thresholds
        checks = {
            'recall': metrics.get('recall', 0) >= self.adaptive_thresholds['recall'],
            'precision': metrics.get('precision', 0) >= self.adaptive_thresholds['precision'],
            'f1': metrics.get('f1', 0) >= self.adaptive_thresholds['f1'],
            'accuracy': metrics.get('accuracy', 0) >= self.adaptive_thresholds['accuracy'],
        }
        
        # Model is healthy if all checks pass
        return all(checks.values())
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of health monitoring state."""
        summary = {
            'evaluation_count': self.evaluation_count,
            'current_thresholds': self.adaptive_thresholds.copy(),
            'history_size': {k: len(v) for k, v in self.history.items()},
        }
        
        # Add recent metrics if available
        if self.evaluation_count > 0:
            summary['recent_metrics'] = {
                k: float(list(v)[-1]) if len(v) > 0 else None
                for k, v in self.history.items()
            }
            
            # Add average metrics
            summary['average_metrics'] = {
                k: float(np.mean(list(v))) if len(v) > 0 else 0.0
                for k, v in self.history.items()
            }
        
        return summary
    
    def reset_history(self) -> None:
        """Reset all tracking history."""
        for key in self.history:
            self.history[key].clear()
        self.evaluation_count = 0
        logger.info("Health monitor history reset")
