"""
Data Poisoning Attacks for ML Model Testing

Implements label flipping and feature noise injection attacks with
comprehensive validation, logging, and metrics tracking.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Optional
from attacks.attack_validator import (
    validate_dataframe,
    validate_series,
    validate_fraction,
    ValidationError
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def label_flipping_attack(
    y: pd.Series,
    flip_fraction: float = 0.1,
    random_state: int = 42,
    target_class: Optional[int] = None,
    flip_to_class: Optional[int] = None
) -> Tuple[pd.Series, Dict]:
    """
    Flips a fraction of labels in the dataset.
    
    Supports both binary and multi-class label flipping with targeted attacks.
    
    Args:
        y: Label series to poison
        flip_fraction: Fraction of labels to flip (0-1)
        random_state: Random seed for reproducibility
        target_class: If specified, only flip labels of this class
        flip_to_class: If specified, flip to this class (otherwise random/binary flip)
        
    Returns:
        Tuple of (poisoned_labels, attack_metadata)
        
    Raises:
        ValidationError: If inputs are invalid
    """
    try:
        # Validation
        validate_series(y, name="y", min_length=1)
        validate_fraction(flip_fraction, name="flip_fraction")
        
        if len(y) == 0:
            raise ValidationError("Label series is empty")
        
        # Check for valid class values
        unique_classes = y.unique()
        if len(unique_classes) == 0:
            raise ValidationError("No unique classes found")
        
        logger.info(f"Starting label flipping attack: flip_fraction={flip_fraction}, "
                   f"target_class={target_class}, flip_to_class={flip_to_class}")
        
        # Initialize
        rng = np.random.default_rng(random_state)
        y_poisoned = y.copy()
        
        # Determine indices to flip
        if target_class is not None:
            # Only flip specific class
            if target_class not in unique_classes:
                raise ValidationError(f"target_class {target_class} not found in labels")
            candidate_indices = y[y == target_class].index
            logger.info(f"Targeting class {target_class}: {len(candidate_indices)} samples")
        else:
            # Flip any class
            candidate_indices = y.index
        
        n_flip = int(len(candidate_indices) * flip_fraction)
        if n_flip == 0:
            logger.warning(f"flip_fraction={flip_fraction} results in 0 flips. Flipping at least 1.")
            n_flip = min(1, len(candidate_indices))
        
        flip_indices = rng.choice(candidate_indices, size=n_flip, replace=False)
        
        # Perform flipping
        if flip_to_class is not None:
            # Flip to specific class
            if flip_to_class not in unique_classes and flip_to_class != -1:
                logger.warning(f"flip_to_class {flip_to_class} not in original classes")
            y_poisoned.loc[flip_indices] = flip_to_class
            flip_strategy = f"targeted to class {flip_to_class}"
        elif len(unique_classes) == 2:
            # Binary flip
            y_poisoned.loc[flip_indices] = 1 - y_poisoned.loc[flip_indices]
            flip_strategy = "binary flip"
        else:
            # Multi-class: flip to random other class
            for idx in flip_indices:
                original = y_poisoned.loc[idx]
                other_classes = [c for c in unique_classes if c != original]
                y_poisoned.loc[idx] = rng.choice(other_classes)
            flip_strategy = "random other class"
        
        # Calculate statistics
        n_changed = (y != y_poisoned).sum()
        
        metadata = {
            'attack_type': 'label_flipping',
            'flip_fraction': flip_fraction,
            'n_flipped': int(n_changed),
            'flip_indices': flip_indices.tolist(),
            'target_class': target_class,
            'flip_to_class': flip_to_class,
            'flip_strategy': flip_strategy,
            'original_distribution': y.value_counts().to_dict(),
            'poisoned_distribution': y_poisoned.value_counts().to_dict()
        }
        
        logger.info(f"Label flipping complete: {n_changed} labels modified ({flip_strategy})")
        logger.debug(f"Original distribution: {metadata['original_distribution']}")
        logger.debug(f"Poisoned distribution: {metadata['poisoned_distribution']}")
        
        return y_poisoned, metadata
        
    except Exception as e:
        logger.error(f"Label flipping attack failed: {e}", exc_info=True)
        raise


def feature_noise_attack(
    X: pd.DataFrame,
    noise_level: float = 0.1,
    random_state: int = 42,
    noise_type: str = 'gaussian',
    target_features: Optional[list] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Injects noise into feature space.
    
    Args:
        X: Feature DataFrame to poison
        noise_level: Standard deviation of noise for gaussian, or amplitude for uniform
        random_state: Random seed for reproducibility
        noise_type: Type of noise ('gaussian' or 'uniform')
        target_features: If specified, only add noise to these features
        
    Returns:
        Tuple of (poisoned_features, attack_metadata)
        
    Raises:
        ValidationError: If inputs are invalid
    """
    try:
        # Validation
        validate_dataframe(X, name="X", numeric_only=True)
        
        if noise_level < 0:
            raise ValidationError(f"noise_level must be non-negative, got {noise_level}")
        
        if noise_type not in ['gaussian', 'uniform']:
            raise ValidationError(f"noise_type must be 'gaussian' or 'uniform', got {noise_type}")
        
        logger.info(f"Starting feature noise attack: noise_level={noise_level}, "
                   f"noise_type={noise_type}, target_features={target_features}")
        
        # Initialize
        rng = np.random.default_rng(random_state)
        X_poisoned = X.copy()
        
        # Determine features to poison
        if target_features:
            missing = set(target_features) - set(X.columns)
            if missing:
                raise ValidationError(f"target_features not in X: {missing}")
            features_to_poison = target_features
        else:
            features_to_poison = X.columns.tolist()
        
        # Generate noise
        noise_shape = (len(X), len(features_to_poison))
        
        if noise_type == 'gaussian':
            noise = rng.normal(0, noise_level, noise_shape)
        else:  # uniform
            noise = rng.uniform(-noise_level, noise_level, noise_shape)
        
        # Apply noise
        X_poisoned[features_to_poison] = X_poisoned[features_to_poison] + noise
        
        # Calculate statistics
        diff = X_poisoned[features_to_poison] - X[features_to_poison]
        
        metadata = {
            'attack_type': 'feature_noise',
            'noise_level': noise_level,
            'noise_type': noise_type,
            'n_features_poisoned': len(features_to_poison),
            'poisoned_features': features_to_poison,
            'noise_statistics': {
                'mean': float(diff.values.mean()),
                'std': float(diff.values.std()),
                'min': float(diff.values.min()),
                'max': float(diff.values.max())
            }
        }
        
        logger.info(f"Feature noise attack complete: {len(features_to_poison)} features modified")
        logger.debug(f"Noise stats: {metadata['noise_statistics']}")
        
        return X_poisoned, metadata
        
    except Exception as e:
        logger.error(f"Feature noise attack failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Test
    import sys
    
    # Test label flipping
    y_test = pd.Series([0, 0, 1, 1, 2, 2, 2, 2])
    y_flipped, meta = label_flipping_attack(y_test, flip_fraction=0.25)
    print(f"Original: {y_test.tolist()}")
    print(f"Flipped: {y_flipped.tolist()}")
    print(f"Metadata: {meta}")
    
    # Test feature noise
    X_test = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])
    X_noisy, meta = feature_noise_attack(X_test, noise_level=0.1)
    print(f"\nNoise metadata: {meta}")
    
    sys.exit(0)
