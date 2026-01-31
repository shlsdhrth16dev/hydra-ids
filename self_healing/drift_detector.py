"""
Enhanced Drift Detector for IDS Data Streams.

Provides comprehensive drift detection using multiple statistical tests,
ensemble voting, feature importance weighting, and detailed reporting.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from datetime import datetime
import logging

from .config import DriftDetectorConfig
from .exceptions import DriftDetectionError


logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Advanced drift detection for ML features.
    
    Features:
    - Multiple drift detection methods (KS test, PSI, Wasserstein distance)
    - Ensemble voting across multiple tests
    - Feature importance weighting
    - Per-feature drift scoring
    - Comprehensive drift reporting
    
    Args:
        config: DriftDetectorConfig instance
        p_value_threshold: Deprecated, use config instead
    """
    
    def __init__(
        self,
        config: Optional[DriftDetectorConfig] = None,
        p_value_threshold: Optional[float] = None  # Backward compatibility
    ):
        # Handle backward compatibility
        if config is None:
            config = DriftDetectorConfig()
            if p_value_threshold is not None:
                config.ks_p_value_threshold = p_value_threshold
        
        self.config = config
        self.feature_importance: Optional[np.ndarray] = None
        self.drift_history: List[Dict[str, Any]] = []
        
        logger.info("DriftDetector initialized with methods: %s", config.methods)
    
    def set_feature_importance(self, importance: np.ndarray) -> None:
        """
        Set feature importance weights for weighted drift scoring.
        
        Args:
            importance: Array of feature importances (same length as features)
        """
        self.feature_importance = importance / importance.sum()  # Normalize
        logger.info("Feature importance set for %d features", len(importance))
    
    def detect(
        self,
        X_reference: pd.DataFrame,
        X_current: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect drift between reference and current data.
        
        Args:
            X_reference: Reference (baseline) data
            X_current: Current (production) data
            feature_names: Optional feature names for reporting
        
        Returns:
            Dictionary with drift detection results
        
        Raises:
            DriftDetectionError: If drift detection fails
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # Validate inputs
            self._validate_inputs(X_reference, X_current)
            
            if feature_names is None:
                feature_names = list(X_reference.columns)
            
            # Run drift detection methods
            drift_results = {}
            per_feature_scores = {}
            
            if "ks" in self.config.methods:
                drift_results['ks'] = self._detect_ks(X_reference, X_current, feature_names)
                per_feature_scores['ks'] = drift_results['ks']['per_feature_scores']
            
            if "psi" in self.config.methods:
                drift_results['psi'] = self._detect_psi(X_reference, X_current, feature_names)
                per_feature_scores['psi'] = drift_results['psi']['per_feature_scores']
            
            if "wasserstein" in self.config.methods:
                drift_results['wasserstein'] = self._detect_wasserstein(
                    X_reference, X_current, feature_names
                )
                per_feature_scores['wasserstein'] = drift_results['wasserstein']['per_feature_scores']
            
            # Ensemble voting if multiple methods
            if self.config.enable_ensemble and len(self.config.methods) > 1:
                ensemble_result = self._ensemble_vote(drift_results, feature_names)
            else:
                # Use first method's result
                first_method = list(drift_results.values())[0]
                ensemble_result = {
                    'drift_detected': first_method['drift_detected'],
                    'drift_ratio': first_method['drift_ratio'],
                }
            
            # Calculate weighted drift score if feature importance available
            weighted_score = None
            if self.config.enable_feature_importance_weighting and self.feature_importance is not None:
                weighted_score = self._calculate_weighted_drift(per_feature_scores)
            
            # Build comprehensive report
            report = {
                'timestamp': timestamp,
                'drift_detected': ensemble_result['drift_detected'],
                'drift_ratio': ensemble_result['drift_ratio'],
                'weighted_drift_score': weighted_score,
                'methods_used': self.config.methods,
                'individual_methods': drift_results,
                'ensemble_voting': ensemble_result.get('voting_details'),
                'drifted_features': self._get_drifted_features(drift_results, feature_names),
                'reference_samples': len(X_reference),
                'current_samples': len(X_current),
                'num_features': X_reference.shape[1],
            }
            
            # Store in history
            self.drift_history.append(report)
            
            # Log results
            logger.info(
                "Drift detection complete: drift_detected=%s, drift_ratio=%.4f, methods=%s",
                report['drift_detected'], report['drift_ratio'], self.config.methods
            )
            
            if report['drift_detected']:
                logger.warning(
                    "Drift detected! %d/%d features drifted (%.2f%%)",
                    len(report['drifted_features']),
                    X_reference.shape[1],
                    report['drift_ratio'] * 100
                )
            
            return report
            
        except Exception as e:
            logger.error("Drift detection failed: %s", str(e), exc_info=True)
            raise DriftDetectionError(f"Drift detection failed: {e}") from e
    
    def _validate_inputs(self, X_ref: pd.DataFrame, X_cur: pd.DataFrame) -> None:
        """Validate input dataframes."""
        if X_ref.shape[1] != X_cur.shape[1]:
            raise DriftDetectionError(
                f"Feature count mismatch: reference has {X_ref.shape[1]}, "
                f"current has {X_cur.shape[1]}"
            )
        
        if not all(X_ref.columns == X_cur.columns):
            raise DriftDetectionError("Feature names do not match between reference and current data")
    
    def _detect_ks(
        self,
        X_ref: pd.DataFrame,
        X_cur: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Kolmogorov-Smirnov test for drift detection."""
        drifted_features = []
        per_feature_scores = {}
        
        for i, col in enumerate(X_ref.columns):
            stat, p_value = ks_2samp(X_ref[col], X_cur[col])
            
            per_feature_scores[feature_names[i]] = {
                'statistic': float(stat),
                'p_value': float(p_value),
                'drifted': p_value < self.config.ks_p_value_threshold
            }
            
            if p_value < self.config.ks_p_value_threshold:
                drifted_features.append(feature_names[i])
        
        drift_ratio = len(drifted_features) / X_ref.shape[1]
        
        return {
            'method': 'ks_test',
            'drift_detected': drift_ratio > self.config.drift_ratio_threshold,
            'drift_ratio': float(drift_ratio),
            'drifted_features': drifted_features,
            'per_feature_scores': per_feature_scores,
        }
    
    def _detect_psi(
        self,
        X_ref: pd.DataFrame,
        X_cur: pd.DataFrame,
        feature_names: List[str],
        num_bins: int = 10
    ) -> Dict[str, Any]:
        """Population Stability Index for drift detection."""
        drifted_features = []
        per_feature_scores = {}
        
        for i, col in enumerate(X_ref.columns):
            psi = self._calculate_psi(
                X_ref[col].values,
                X_cur[col].values,
                num_bins=num_bins
            )
            
            per_feature_scores[feature_names[i]] = {
                'psi_score': float(psi),
                'drifted': psi > self.config.psi_threshold
            }
            
            if psi > self.config.psi_threshold:
                drifted_features.append(feature_names[i])
        
        drift_ratio = len(drifted_features) / X_ref.shape[1]
        
        return {
            'method': 'psi',
            'drift_detected': drift_ratio > self.config.drift_ratio_threshold,
            'drift_ratio': float(drift_ratio),
            'drifted_features': drifted_features,
            'per_feature_scores': per_feature_scores,
        }
    
    def _calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        num_bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index.
        
        PSI = sum((actual_% - expected_%) * ln(actual_% / expected_%))
        """
        # Create bins based on expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, num_bins + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates
        
        if len(breakpoints) <= 1:
            return 0.0  # No variation
        
        # Calculate distributions
        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
        
        # Avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # Calculate PSI
        psi = np.sum((actual_percents - expected_percents) * 
                     np.log(actual_percents / expected_percents))
        
        return float(psi)
    
    def _detect_wasserstein(
        self,
        X_ref: pd.DataFrame,
        X_cur: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Wasserstein distance for drift detection."""
        drifted_features = []
        per_feature_scores = {}
        
        for i, col in enumerate(X_ref.columns):
            distance = wasserstein_distance(X_ref[col], X_cur[col])
            
            per_feature_scores[feature_names[i]] = {
                'distance': float(distance),
                'drifted': distance > self.config.wasserstein_threshold
            }
            
            if distance > self.config.wasserstein_threshold:
                drifted_features.append(feature_names[i])
        
        drift_ratio = len(drifted_features) / X_ref.shape[1]
        
        return {
            'method': 'wasserstein',
            'drift_detected': drift_ratio > self.config.drift_ratio_threshold,
            'drift_ratio': float(drift_ratio),
            'drifted_features': drifted_features,
            'per_feature_scores': per_feature_scores,
        }
    
    def _ensemble_vote(
        self,
        drift_results: Dict[str, Dict[str, Any]],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Ensemble voting across multiple drift detection methods."""
        num_features = len(feature_names)
        
        # Count votes per feature
        feature_votes = {name: 0 for name in feature_names}
        
        for method_result in drift_results.values():
            for feature in method_result['drifted_features']:
                feature_votes[feature] += 1
        
        # Features with min_tests_agree votes are considered drifted
        drifted_features = [
            feature for feature, votes in feature_votes.items()
            if votes >= self.config.min_tests_agree
        ]
        
        drift_ratio = len(drifted_features) / num_features
        
        # Calculate average drift ratio across methods
        avg_drift_ratio = np.mean([r['drift_ratio'] for r in drift_results.values()])
        
        return {
            'drift_detected': drift_ratio > self.config.drift_ratio_threshold,
            'drift_ratio': float(drift_ratio),
            'avg_method_drift_ratio': float(avg_drift_ratio),
            'voting_details': {
                'feature_votes': feature_votes,
                'min_tests_agree': self.config.min_tests_agree,
                'num_methods': len(drift_results),
            },
        }
    
    def _calculate_weighted_drift(
        self,
        per_feature_scores: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> float:
        """Calculate weighted drift score using feature importance."""
        if self.feature_importance is None:
            return 0.0
        
        # Get drift flags from first method
        first_method = list(per_feature_scores.keys())[0]
        feature_names = list(per_feature_scores[first_method].keys())
        
        # Count how many methods detected drift for each feature
        weighted_score = 0.0
        for i, feature in enumerate(feature_names):
            drift_count = sum(
                1 for method_scores in per_feature_scores.values()
                if method_scores[feature].get('drifted', False)
            )
            
            # Weight by feature importance
            feature_drift = drift_count / len(per_feature_scores)
            weighted_score += feature_drift * self.feature_importance[i]
        
        return float(weighted_score)
    
    def _get_drifted_features(
        self,
        drift_results: Dict[str, Dict[str, Any]],
        feature_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Get detailed list of drifted features across all methods."""
        drifted_info = []
        
        # Aggregate drifted features
        all_drifted = set()
        for result in drift_results.values():
            all_drifted.update(result['drifted_features'])
        
        for feature in sorted(all_drifted):
            feature_info = {
                'feature_name': feature,
                'methods_detecting_drift': []
            }
            
            for method_name, result in drift_results.items():
                if feature in result['drifted_features']:
                    feature_info['methods_detecting_drift'].append(method_name)
            
            drifted_info.append(feature_info)
        
        return drifted_info
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection history."""
        if not self.drift_history:
            return {'detections_performed': 0}
        
        return {
            'detections_performed': len(self.drift_history),
            'total_drifts_detected': sum(1 for d in self.drift_history if d['drift_detected']),
            'latest_detection': self.drift_history[-1] if self.drift_history else None,
            'avg_drift_ratio': float(np.mean([d['drift_ratio'] for d in self.drift_history])),
        }
