"""
Attack Controller - Production Grade

Orchestrates multiple attack types with configuration validation,
history tracking, and comprehensive reporting.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import pandas as pd

from attacks.poisoning import label_flipping_attack, feature_noise_attack
from attacks.drift import gradual_mean_shift, covariate_drift
from attacks.corruption import drop_features, inject_missing_values, inject_outliers
from attacks.evasion import evasion_noise, targeted_evasion
from attacks.attack_validator import validate_config, validate_fraction, ValidationError
from attacks.attack_metrics import AttackMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AttackController:
    """
    Production-grade attack orchestration with:
    - Configuration validation
    - Attack history tracking
    - Metrics calculation
    - Rollback support
    - Comprehensive reporting
    """
    
    # Configuration schema
    CONFIG_SCHEMA = {
        'label_flip_fraction': (float, lambda v, n: validate_fraction(v, n) if v is not None else None),
        'feature_noise': (float, lambda v, n: None),
        'drift_strength': (float, lambda v, n: None),
        'drop_fraction': (float, lambda v, n: validate_fraction(v, n) if v is not None else None),
        'epsilon': (float, lambda v, n: None),
        'missing_fraction': (float, lambda v, n: validate_fraction(v, n) if v is not None else None),
        'outlier_fraction': (float, lambda v, n: validate_fraction(v, n) if v is not None else None),
        'outlier_magnitude': (float, lambda v, n: None),
    }
    
    def __init__(self, config: dict, model=None, scaler=None, track_history: bool = True):
        """
        Initialize Attack Controller.
        
        Args:
            config: Attack configuration dictionary
            model: Optional ML model for metrics calculation
            scaler: Optional scaler for preprocessing
            track_history: If True, maintain attack history
        """
        try:
            # Validate config
            validate_config(config, self.CONFIG_SCHEMA)
            self.config = config
            
            # Initialize components
            self.model = model
            self.scaler = scaler
            self.track_history = track_history
            
            # Attack history
            self.attack_history: List[Dict[str, Any]] = []
            self.original_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None
            
            # Metrics calculator
            if model:
                self.metrics_calculator = AttackMetrics(model, scaler)
            else:
                self.metrics_calculator = None
            
            logger.info(f"AttackController initialized with config: {config}")
            
        except Exception as e:
            logger.error(f"AttackController initialization failed: {e}", exc_info=True)
            raise
    
    def set_baseline(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Store original data for rollback capability.
        
        Args:
            X: Original features
            y: Original labels (optional)
        """
        self.original_data = (X.copy(), y.copy() if y is not None else None)
        logger.info(f"Baseline data stored: {X.shape}")
    
    def rollback(self) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Rollback to original data.
        
        Returns:
            Tuple of (original_X, original_y)
        """
        if self.original_data is None:
            raise RuntimeError("No baseline data available. Call set_baseline() first.")
        
        logger.info("Rolling back to baseline data")
        # History is preserved for reporting - don't clear it
        # self.attack_history.clear()  # ← REMOVED: This was wiping attack data
        return self.original_data[0].copy(), self.original_data[1].copy() if self.original_data[1] is not None else None
    
    def apply_poisoning(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        evaluate: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Apply poisoning attacks (label flipping + feature noise).
        
        Args:
            X: Features
            y: Labels
            evaluate: If True and model available, calculate metrics
            
        Returns:
            Tuple of (X_attacked, y_attacked, metadata)
        """
        try:
            logger.info("Applying poisoning attack...")
            
            # Label flipping
            y_poisoned, label_meta = label_flipping_attack(
                y,
                flip_fraction=self.config.get("label_flip_fraction", 0.1),
                random_state=self.config.get("random_state", 42)
            )
            
            # Feature noise
            X_poisoned, feature_meta = feature_noise_attack(
                X,
                noise_level=self.config.get("feature_noise", 0.1),
                random_state=self.config.get("random_state", 42)
            )
            
            # Combine metadata
            metadata = {
                'attack_combination': 'poisoning',
                'timestamp': datetime.now().isoformat(),
                'label_attack': label_meta,
                'feature_attack': feature_meta
            }
            
            # Evaluate if requested
            if evaluate and self.metrics_calculator and self.original_data:
                X_orig, y_orig = self.original_data
                eval_results = self.metrics_calculator.evaluate_attack_impact(
                    X_orig, X_poisoned, y_orig, metadata
                )
                metadata['evaluation'] = eval_results
            
            # Track history
            if self.track_history:
                self.attack_history.append(metadata)
            
            logger.info("Poisoning attack complete")
            return X_poisoned, y_poisoned, metadata
            
        except Exception as e:
            logger.error(f"Poisoning attack failed: {e}", exc_info=True)
            raise
    
    def apply_drift(
        self,
        X: pd.DataFrame,
        evaluate: bool = True,
        drift_type: str = 'gradual'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply drift attack.
        
        Args:
            X: Features
            evaluate: If True and model available, calculate metrics
            drift_type: Type of drift ('gradual', 'sudden', 'covariate')
            
        Returns:
            Tuple of (X_drifted, metadata)
        """
        try:
            logger.info(f"Applying {drift_type} drift attack...")
            
            if drift_type == 'covariate':
                X_drifted, metadata = covariate_drift(
                    X,
                    drift_magnitude=self.config.get("drift_strength", 0.2),
                    random_state=self.config.get("random_state", 42)
                )
            else:
                X_drifted, metadata = gradual_mean_shift(
                    X,
                    shift_strength=self.config.get("drift_strength", 0.2),
                    drift_type=drift_type,
                    random_state=self.config.get("random_state", 42)
                )
            
            metadata['timestamp'] = datetime.now().isoformat()
            
            # Evaluate if requested
            if evaluate and self.metrics_calculator and self.original_data:
                X_orig, y_orig = self.original_data
                eval_results = self.metrics_calculator.evaluate_attack_impact(
                    X_orig, X_drifted, y_orig, metadata
                )
                metadata['evaluation'] = eval_results
            
            # Track history
            if self.track_history:
                self.attack_history.append(metadata)
            
            logger.info("Drift attack complete")
            return X_drifted, metadata
            
        except Exception as e:
            logger.error(f"Drift attack failed: {e}", exc_info=True)
            raise
    
    def apply_corruption(
        self,
        X: pd.DataFrame,
        evaluate: bool = True,
        corruption_type: str = 'drop_features'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply corruption attack.
        
        Args:
            X: Features
            evaluate: If True and model available, calculate metrics
            corruption_type: Type ('drop_features', 'missing_values', 'outliers')
            
        Returns:
            Tuple of (X_corrupted, metadata)
        """
        try:
            logger.info(f"Applying {corruption_type} corruption attack...")
            
            if corruption_type == 'drop_features':
                X_corrupted, metadata = drop_features(
                    X,
                    drop_fraction=self.config.get("drop_fraction", 0.2),
                    random_state=self.config.get("random_state", 42)
                )
            elif corruption_type == 'missing_values':
                X_corrupted, metadata = inject_missing_values(
                    X,
                    missing_fraction=self.config.get("missing_fraction", 0.1),
                    random_state=self.config.get("random_state", 42)
                )
            else:  # outliers
                X_corrupted, metadata = inject_outliers(
                    X,
                    outlier_fraction=self.config.get("outlier_fraction", 0.05),
                    outlier_magnitude=self.config.get("outlier_magnitude", 5.0),
                    random_state=self.config.get("random_state", 42)
                )
            
            metadata['timestamp'] = datetime.now().isoformat()
            
            # Evaluate if requested
            if evaluate and self.metrics_calculator and self.original_data:
                X_orig, y_orig = self.original_data
                eval_results = self.metrics_calculator.evaluate_attack_impact(
                    X_orig, X_corrupted, y_orig, metadata
                )
                metadata['evaluation'] = eval_results
            
            # Track history
            if self.track_history:
                self.attack_history.append(metadata)
            
            logger.info("Corruption attack complete")
            return X_corrupted, metadata
            
        except Exception as e:
            logger.error(f"Corruption attack failed: {e}", exc_info=True)
            raise
    
    def apply_evasion(
        self,
        X: pd.DataFrame,
        evaluate: bool = True,
        strategy: str = 'random_sign'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply evasion attack.
        
        Args:
            X: Features
            evaluate: If True and model available, calculate metrics
            strategy: Evasion strategy
            
        Returns:
            Tuple of (X_evaded, metadata)
        """
        try:
            logger.info(f"Applying evasion attack with strategy={strategy}...")
            
            X_evaded, metadata = evasion_noise(
                X,
                epsilon=self.config.get("epsilon", 0.05),
                strategy=strategy,
                random_state=self.config.get("random_state", 42)
            )
            
            metadata['timestamp'] = datetime.now().isoformat()
            
            # Evaluate if requested
            if evaluate and self.metrics_calculator and self.original_data:
                X_orig, y_orig = self.original_data
                eval_results = self.metrics_calculator.evaluate_attack_impact(
                    X_orig, X_evaded, y_orig, metadata
                )
                metadata['evaluation'] = eval_results
            
            # Track history
            if self.track_history:
                self.attack_history.append(metadata)
            
            logger.info("Evasion attack complete")
            return X_evaded, metadata
            
        except Exception as e:
            logger.error(f"Evasion attack failed: {e}", exc_info=True)
            raise
    
    def apply_attack_chain(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        attack_sequence: List[str] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], List[Dict]]:
        """
        Apply multiple attacks in sequence.
        
        Args:
            X: Features
            y: Labels (optional)
            attack_sequence: List of attack names to apply
            
        Returns:
            Tuple of (X_final, y_final, metadata_list)
        """
        if attack_sequence is None:
            attack_sequence = ['drift', 'corruption', 'evasion']
        
        logger.info(f"Applying attack chain: {attack_sequence}")
        
        X_current = X.copy()
        y_current = y.copy() if y is not None else None
        metadata_list = []
        
        for attack in attack_sequence:
            if attack == 'poisoning' and y_current is not None:
                X_current, y_current, meta = self.apply_poisoning(X_current, y_current, evaluate=False)
            elif attack == 'drift':
                X_current, meta = self.apply_drift(X_current, evaluate=False)
            elif attack == 'corruption':
                X_current, meta = self.apply_corruption(X_current, evaluate=False)
            elif attack == 'evasion':
                X_current, meta = self.apply_evasion(X_current, evaluate=False)
            else:
                logger.warning(f"Unknown attack type: {attack}, skipping")
                continue
            
            metadata_list.append(meta)
        
        # Final evaluation
        if self.metrics_calculator and self.original_data:
            X_orig, y_orig = self.original_data
            combined_meta = {
                'attack_combination': 'chain',
                'attack_sequence': attack_sequence,
                'individual_attacks': metadata_list
            }
            eval_results = self.metrics_calculator.evaluate_attack_impact(
                X_orig, X_current, y_orig, combined_meta
            )
            metadata_list.append({'chain_evaluation': eval_results})
        
        logger.info("Attack chain complete")
        return X_current, y_current, metadata_list
    
    def export_report(self, output_path: Path):
        """
        Export comprehensive attack report.
        
        Args:
            output_path: Path to save JSON report
        """
        def json_serializer(obj):
            """Custom serializer for non-JSON-serializable objects."""
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                # Skip model/scaler objects
                return str(type(obj).__name__)
            else:
                return str(obj)
        
        report = {
            'config': self.config,
            'attack_history': self.attack_history,
            'total_attacks': len(self.attack_history),
            'timestamp': datetime.now().isoformat()
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=json_serializer)
            logger.info(f"Attack report exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            # Try with minimal data
            minimal_report = {
                'config': self.config,
                'total_attacks': len(self.attack_history),
                'timestamp': datetime.now().isoformat(),
                'note': 'Full history could not be serialized'
            }
            with open(output_path, 'w') as f:
                json.dump(minimal_report, f, indent=2, default=str)
            logger.warning(f"Exported minimal report due to serialization error")



if __name__ == "__main__":
    # Test
    config = {
        'label_flip_fraction': 0.2,
        'feature_noise': 0.15,
        'drift_strength': 0.3,
        'drop_fraction': 0.25,
        'epsilon': 0.05
    }
    
    controller = AttackController(config)
    print("AttackController initialized successfully")
