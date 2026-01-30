"""
Concept Drift Simulation for Model Testing

Implements gradual and sudden distribution shift attacks with
comprehensive validation, logging, and metrics tracking.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional, List
from attacks.attack_validator import (
    validate_dataframe,
    validate_fraction,
    ValidationError
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def gradual_mean_shift(
    X: pd.DataFrame,
    shift_strength: float = 0.2,
    fraction: float = 0.3,
    random_state: int = 42,
    drift_type: str = 'gradual',
    target_features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Shifts feature distributions to simulate concept drift.
    
    Args:
        X: Feature DataFrame
        shift_strength: Magnitude of shift as fraction of mean
        fraction: Fraction of samples to affect
        random_state: Random seed for reproducibility
        drift_type: 'gradual' (linear ramp), 'sudden' (step), or 'incremental' (multi-step)
        target_features: If specified, only drift these features
        
    Returns:
        Tuple of (drifted_features, attack_metadata)
        
    Raises:
        ValidationError: If inputs are invalid
    """
    try:
        # Validation
        validate_dataframe(X, name="X", numeric_only=True)
        validate_fraction(fraction, name="fraction")
        
        if shift_strength < 0:
            raise ValidationError(f"shift_strength must be non-negative, got {shift_strength}")
        
        valid_drift_types = ['gradual', 'sudden', 'incremental']
        if drift_type not in valid_drift_types:
            raise ValidationError(f"drift_type must be one of {valid_drift_types}, got {drift_type}")
        
        logger.info(f"Starting drift attack: shift_strength={shift_strength}, "
                   f"fraction={fraction}, drift_type={drift_type}")
        
        # Initialize
        rng = np.random.default_rng(random_state)
        X_drifted = X.copy()
        n_shift = int(len(X) * fraction)
        
        if n_shift == 0:
            logger.warning(f"fraction={fraction} results in 0 samples. Setting n_shift=1")
            n_shift = min(1, len(X))
        
        # Determine features to drift
        if target_features:
            missing = set(target_features) - set(X.columns)
            if missing:
                raise ValidationError(f"target_features not in X: {missing}")
            features_to_drift = target_features
        else:
            features_to_drift = X.columns.tolist()
        
        # Calculate shift vector
        feature_means = X[features_to_drift].mean()
        shift_vector = feature_means * shift_strength
        
        # Apply drift based on type
        if drift_type == 'sudden':
            # Sudden shift: step function
            X_drifted.iloc[:n_shift, X_drifted.columns.get_indexer(features_to_drift)] += shift_vector.values
            
        elif drift_type == 'gradual':
            # Gradual shift: linear ramp from 0 to shift_vector
            for i in range(n_shift):
                weight = (i + 1) / n_shift  # 0 to 1
                X_drifted.iloc[i, X_drifted.columns.get_indexer(features_to_drift)] += shift_vector.values * weight
                
        else:  # incremental
            # Multi-step drift: divide into 5 steps
            n_steps = 5
            step_size = n_shift // n_steps
            for step in range(n_steps):
                start = step * step_size
                end = start + step_size if step < n_steps - 1 else n_shift
                weight = (step + 1) / n_steps
                X_drifted.iloc[start:end, X_drifted.columns.get_indexer(features_to_drift)] += shift_vector.values * weight
        
        # Calculate drift statistics
        diff = X_drifted[features_to_drift] - X[features_to_drift]
        affected_samples = diff[diff.any(axis=1)]
        
        metadata = {
            'attack_type': 'drift',
            'drift_type': drift_type,
            'shift_strength': shift_strength,
            'fraction': fraction,
            'n_samples_drifted': len(affected_samples),
            'n_features_drifted': len(features_to_drift),
            'drifted_features': features_to_drift,
            'shift_vector': shift_vector.to_dict(),
            'drift_statistics': {
                'mean_shift': float(diff.values.mean()),
                'std_shift': float(diff.values.std()),
                'max_shift': float(diff.values.max()),
                'min_shift': float(diff.values.min())
            }
        }
        
        logger.info(f"Drift attack complete: {len(affected_samples)} samples affected")
        logger.debug(f"Drift stats: {metadata['drift_statistics']}")
        
        return X_drifted, metadata
        
    except Exception as e:
        logger.error(f"Drift attack failed: {e}", exc_info=True)
        raise


def covariate_drift(
    X: pd.DataFrame,
    drift_magnitude: float = 0.3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Simulate covariate drift by shifting feature distributions.
    
    Adds feature-specific drift based on statistical properties.
    
    Args:
        X: Feature DataFrame
        drift_magnitude: Magnitude of distribution shift
        random_state: Random seed
        
    Returns:
        Tuple of (drifted_features, attack_metadata)
    """
    try:
        validate_dataframe(X, name="X", numeric_only=True)
        
        logger.info(f"Starting covariate drift: drift_magnitude={drift_magnitude}")
        
        rng = np.random.default_rng(random_state)
        X_drifted = X.copy()
        
        # Apply feature-specific drift
        drift_info = {}
        for col in X.columns:
            # Shift mean
            original_mean = X[col].mean()
            original_std = X[col].std()
            
            mean_shift = original_std * drift_magnitude * rng.choice([-1, 1])
            X_drifted[col] = X_drifted[col] + mean_shift
            
            drift_info[col] = {
                'original_mean': float(original_mean),
                'shift': float(mean_shift),
                'new_mean': float(X_drifted[col].mean())
            }
        
        metadata = {
            'attack_type': 'covariate_drift',
            'drift_magnitude': drift_magnitude,
            'n_features': len(X.columns),
            'feature_drift_info': drift_info
        }
        
        logger.info(f"Covariate drift complete: {len(X.columns)} features drifted")
        
        return X_drifted, metadata
        
    except Exception as e:
        logger.error(f"Covariate drift attack failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Test
    X_test = pd.DataFrame(np.random.randn(100, 5) * 10 + 50, 
                         columns=['a', 'b', 'c', 'd', 'e'])
    
    # Test gradual drift
    X_drifted, meta = gradual_mean_shift(X_test, shift_strength=0.2, 
                                         drift_type='gradual', fraction=0.3)
    print(f"Gradual drift metadata: {meta}")
    
    # Test sudden drift
    X_drifted, meta = gradual_mean_shift(X_test, shift_strength=0.3, 
                                         drift_type='sudden', fraction=0.5)
    print(f"\nSudden drift metadata: {meta}")
    
    # Test covariate drift
    X_drifted, meta = covariate_drift(X_test, drift_magnitude=0.4)
    print(f"\nCovariate drift metadata: {meta}")
    
    print("\nDrift tests passed!")
