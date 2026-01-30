"""
Feature Corruption Attacks for Model Testing

Implements various data corruption strategies with
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


def drop_features(
    X: pd.DataFrame,
    drop_fraction: float = 0.2,
    random_state: int = 42,
    strategy: str = 'random',
    feature_importance: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Randomly corrupts a fraction of features by zeroing them out.
    
    Strategies:
    - 'random': Random feature selection
    - 'important': Target most important features (requires feature_importance)
    - 'unimportant': Target least important features (requires feature_importance)
    
    Args:
        X: Feature DataFrame to corrupt
        drop_fraction: Fraction of features to zero out (0-1)
        random_state: Random seed for reproducibility
        strategy: Feature selection strategy
        feature_importance: Dict mapping feature names to importance scores
        
    Returns:
        Tuple of (corrupted_features, attack_metadata)
        
    Raises:
        ValidationError: If inputs are invalid
    """
    try:
        # Validation
        validate_dataframe(X, name="X", numeric_only=True)
        validate_fraction(drop_fraction, name="drop_fraction")
        
        valid_strategies = ['random', 'important', 'unimportant']
        if strategy not in valid_strategies:
            raise ValidationError(f"strategy must be one of {valid_strategies}, got {strategy}")
        
        if strategy in ['important', 'unimportant'] and not feature_importance:
            raise ValidationError(f"strategy='{strategy}' requires feature_importance dict")
        
        logger.info(f"Starting feature corruption: drop_fraction={drop_fraction}, "
                   f"strategy={strategy}")
        
        # Initialize
        rng = np.random.default_rng(random_state)
        X_corrupted = X.copy()
        n_drop = max(1, int(X.shape[1] * drop_fraction))
        
        # Select features to drop
        if strategy == 'random':
            drop_cols = rng.choice(X.columns, size=n_drop, replace=False)
            
        else:
            # Importance-based selection
            if not set(feature_importance.keys()).issuperset(set(X.columns)):
                logger.warning("feature_importance missing some columns, using available")
            
            # Sort by importance
            sorted_features = sorted(
                [(col, feature_importance.get(col, 0)) for col in X.columns],
                key=lambda x: x[1],
                reverse=(strategy == 'important')
            )
            drop_cols = [col for col, _ in sorted_features[:n_drop]]
        
        # Apply corruption (zero out)
        original_values = X_corrupted[drop_cols].copy()
        X_corrupted[drop_cols] = 0.0
        
        # Calculate statistics
        metadata = {
            'attack_type': 'feature_corruption',
            'strategy': strategy,
            'drop_fraction': drop_fraction,
            'n_features_corrupted': len(drop_cols),
            'corrupted_features': list(drop_cols),
            'corruption_statistics': {
                'original_mean': float(original_values.values.mean()),
                'original_std': float(original_values.values.std()),
                'original_min': float(original_values.values.min()),
                'original_max': float(original_values.values.max())
            }
        }
        
        logger.info(f"Feature corruption complete: {len(drop_cols)} features zeroed")
        logger.debug(f"Corrupted features: {drop_cols}")
        
        return X_corrupted, metadata
        
    except Exception as e:
        logger.error(f"Feature corruption attack failed: {e}", exc_info=True)
        raise


def inject_missing_values(
    X: pd.DataFrame,
    missing_fraction: float = 0.1,
    random_state: int = 42,
    per_feature: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    """
    Inject missing values (NaN) into the dataset.
    
    Args:
        X: Feature DataFrame
        missing_fraction: Fraction of values to set as NaN
        random_state: Random seed
        per_feature: If True, apply fraction to each feature independently
        
    Returns:
        Tuple of (corrupted_features, attack_metadata)
    """
    try:
        validate_dataframe(X, name="X", numeric_only=True)
        validate_fraction(missing_fraction, name="missing_fraction")
        
        logger.info(f"Injecting missing values: missing_fraction={missing_fraction}, "
                   f"per_feature={per_feature}")
        
        rng = np.random.default_rng(random_state)
        X_corrupted = X.copy()
        
        if per_feature:
            # Apply to each feature independently
            missing_counts = {}
            for col in X.columns:
                n_missing = int(len(X) * missing_fraction)
                if n_missing > 0:
                    missing_idx = rng.choice(len(X), size=n_missing, replace=False)
                    X_corrupted.iloc[missing_idx, X_corrupted.columns.get_loc(col)] = np.nan
                    missing_counts[col] = n_missing
        else:
            # Apply globally
            total_values = X.size
            n_missing = int(total_values * missing_fraction)
            flat_idx = rng.choice(total_values, size=n_missing, replace=False)
            
            rows = flat_idx // X.shape[1]
            cols = flat_idx % X.shape[1]
            
            for r, c in zip(rows, cols):
                X_corrupted.iloc[r, c] = np.nan
            
            missing_counts = X_corrupted.isnull().sum().to_dict()
        
        total_missing = X_corrupted.isnull().sum().sum()
        
        metadata = {
            'attack_type': 'missing_values',
            'missing_fraction': missing_fraction,
            'per_feature': per_feature,
            'total_missing': int(total_missing),
            'missing_per_feature': {k: int(v) for k, v in missing_counts.items() if v > 0}
        }
        
        logger.info(f"Missing values injected: {total_missing} total")
        
        return X_corrupted, metadata
        
    except Exception as e:
        logger.error(f"Missing values injection failed: {e}", exc_info=True)
        raise


def inject_outliers(
    X: pd.DataFrame,
    outlier_fraction: float = 0.05,
    outlier_magnitude: float = 5.0,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Inject outliers into the dataset.
    
    Args:
        X: Feature DataFrame
        outlier_fraction: Fraction of samples to corrupt
        outlier_magnitude: How many standard deviations away from mean
        random_state: Random seed
        
    Returns:
        Tuple of (corrupted_features, attack_metadata)
    """
    try:
        validate_dataframe(X, name="X", numeric_only=True)
        validate_fraction(outlier_fraction, name="outlier_fraction")
        
        logger.info(f"Injecting outliers: outlier_fraction={outlier_fraction}, "
                   f"magnitude={outlier_magnitude}")
        
        rng = np.random.default_rng(random_state)
        X_corrupted = X.copy()
        
        n_outliers = max(1, int(len(X) * outlier_fraction))
        outlier_indices = rng.choice(len(X), size=n_outliers, replace=False)
        
        # For each outlier sample, perturb all features
        for idx in outlier_indices:
            for col in X.columns:
                mean = X[col].mean()
                std = X[col].std()
                direction = rng.choice([-1, 1])
                X_corrupted.loc[X.index[idx], col] = mean + (direction * outlier_magnitude * std)
        
        metadata = {
            'attack_type': 'outliers',
            'outlier_fraction': outlier_fraction,
            'outlier_magnitude': outlier_magnitude,
            'n_outliers': n_outliers,
            'outlier_indices': outlier_indices.tolist()
        }
        
        logger.info(f"Outliers injected: {n_outliers} samples")
        
        return X_corrupted, metadata
        
    except Exception as e:
        logger.error(f"Outlier injection failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Test
    X_test = pd.DataFrame(np.random.randn(100, 5) * 10 + 50,
                         columns=['a', 'b', 'c', 'd', 'e'])
    
    # Test feature dropping
    X_corrupted, meta = drop_features(X_test, drop_fraction=0.2, strategy='random')
    print(f"Feature drop metadata: {meta}")
    
    # Test missing values
    X_corrupted, meta = inject_missing_values(X_test, missing_fraction=0.1)
    print(f"\nMissing values metadata: {meta}")
    
    # Test outliers
    X_corrupted, meta = inject_outliers(X_test, outlier_fraction=0.05, outlier_magnitude=5.0)
    print(f"\nOutliers metadata: {meta}")
    
    print("\nCorruption tests passed!")
