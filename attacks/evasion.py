"""
Evasion Attacks for IDS Testing

Implements adversarial perturbations to evade detection with
comprehensive validation, logging, and metrics tracking.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Optional
from attacks.attack_validator import (
    validate_dataframe,
    ValidationError
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evasion_noise(
    X: pd.DataFrame,
    epsilon: float = 0.05,
    random_state: int = 42,
    strategy: str = 'random_sign',
    target_features: Optional[list] = None,
    clip_to_valid: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply small perturbations to bypass detection while maintaining validity.
    
    Strategies:
    - 'random_sign': Random ±epsilon perturbations
    - 'additive': Always positive epsilon perturbations
    - 'multiplicative': Multiply features by (1 ± epsilon)
    
    Args:
        X: Feature DataFrame to perturb
        epsilon: Perturbation magnitude
        random_state: Random seed for reproducibility
        strategy: Perturbation strategy
        target_features: If specified, only perturb these features
        clip_to_valid: If True, clip negative values to 0 (for count features)
        
    Returns:
        Tuple of (perturbed_features, attack_metadata)
        
    Raises:
        ValidationError: If inputs are invalid
    """
    try:
        # Validation
        validate_dataframe(X, name="X", numeric_only=True)
        
        if epsilon < 0:
            raise ValidationError(f"epsilon must be non-negative, got {epsilon}")
        
        valid_strategies = ['random_sign', 'additive', 'multiplicative']
        if strategy not in valid_strategies:
            raise ValidationError(f"strategy must be one of {valid_strategies}, got {strategy}")
        
        logger.info(f"Starting evasion attack: epsilon={epsilon}, strategy={strategy}, "
                   f"target_features={target_features}, clip_to_valid={clip_to_valid}")
        
        # Initialize
        rng = np.random.default_rng(random_state)
        X_evaded = X.copy()
        
        # Determine features to perturb
        if target_features:
            missing = set(target_features) - set(X.columns)
            if missing:
                raise ValidationError(f"target_features not in X: {missing}")
            features_to_perturb = target_features
        else:
            features_to_perturb = X.columns.tolist()
        
        # Apply perturbations
        if strategy == 'random_sign':
            # Random ±epsilon
            signs = rng.choice([-1, 1], size=(len(X), len(features_to_perturb)))
            perturbation = signs * epsilon
            X_evaded[features_to_perturb] = X_evaded[features_to_perturb] + perturbation
            
        elif strategy == 'additive':
            # Always add epsilon
            X_evaded[features_to_perturb] = X_evaded[features_to_perturb] + epsilon
            
        else:  # multiplicative
            # Multiply by (1 ± epsilon)
            signs = rng.choice([-1, 1], size=(len(X), len(features_to_perturb)))
            multiplier = 1 + (signs * epsilon)
            X_evaded[features_to_perturb] = X_evaded[features_to_perturb] * multiplier
        
        # Clip negative values for count features
        if clip_to_valid:
            original_negatives = (X_evaded[features_to_perturb] < 0).sum().sum()
            if original_negatives > 0:
                X_evaded[features_to_perturb] = X_evaded[features_to_perturb].clip(lower=0)
                logger.info(f"Clipped {original_negatives} negative values to 0")
        
        # Calculate statistics
        diff = X_evaded[features_to_perturb] - X[features_to_perturb]
        
        metadata = {
            'attack_type': 'evasion',
            'epsilon': epsilon,
            'strategy': strategy,
            'n_features_perturbed': len(features_to_perturb),
            'perturbed_features': features_to_perturb,
            'clip_to_valid': clip_to_valid,
            'perturbation_statistics': {
                'mean': float(diff.values.mean()),
                'std': float(diff.values.std()),
                'min': float(diff.values.min()),
                'max': float(diff.values.max()),
                'l2_norm': float(np.linalg.norm(diff.values))
            }
        }
        
        logger.info(f"Evasion attack complete: {len(features_to_perturb)} features perturbed")
        logger.debug(f"Perturbation stats: {metadata['perturbation_statistics']}")
        
        return X_evaded, metadata
        
    except Exception as e:
        logger.error(f"Evasion attack failed: {e}", exc_info=True)
        raise


def targeted_evasion(
    X: pd.DataFrame,
    model,
    scaler,
    target_class: int,
    max_iterations: int = 100,
    step_size: float = 0.01,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict]:
    """
    Targeted evasion attack using gradient-based approach.
    
    Iteratively perturbs features to push predictions toward target_class.
    
    Args:
        X: Feature DataFrame to perturb
        model: Trained model with predict_proba method
        scaler: Fitted scaler for preprocessing
        target_class: Class to evade toward
        max_iterations: Maximum perturbation iterations
        step_size: Step size for gradient descent
        random_state: Random seed
        
    Returns:
        Tuple of (perturbed_features, attack_metadata)
    """
    try:
        validate_dataframe(X, name="X", numeric_only=True)
        
        logger.info(f"Starting targeted evasion: target_class={target_class}, "
                   f"max_iterations={max_iterations}, step_size={step_size}")
        
        X_evaded = X.copy()
        X_scaled = scaler.transform(X)
        
        # Simplified gradient-free approach: random search
        rng = np.random.default_rng(random_state)
        best_X = X_scaled.copy()
        best_prob = model.predict_proba(X_scaled)[:, target_class].mean()
        
        for i in range(max_iterations):
            # Random perturbation
            perturbation = rng.normal(0, step_size, X_scaled.shape)
            X_candidate = X_scaled + perturbation
            
            # Evaluate
            prob = model.predict_proba(X_candidate)[:, target_class].mean()
            
            if prob > best_prob:
                best_X = X_candidate
                best_prob = prob
        
        # Inverse transform
        X_evaded = pd.DataFrame(
            scaler.inverse_transform(best_X),
            columns=X.columns,
            index=X.index
        )
        
        metadata = {
            'attack_type': 'targeted_evasion',
            'target_class': target_class,
            'max_iterations': max_iterations,
            'final_probability': float(best_prob),
            'perturbation_l2': float(np.linalg.norm(best_X - scaler.transform(X)))
        }
        
        logger.info(f"Targeted evasion complete: final prob={best_prob:.4f}")
        
        return X_evaded, metadata
        
    except Exception as e:
        logger.error(f"Targeted evasion attack failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Test
    X_test = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', 'c', 'd', 'e'])
    
    # Test random_sign
    X_evaded, meta = evasion_noise(X_test, epsilon=0.05, strategy='random_sign')
    print(f"Random sign metadata: {meta}")
    
    # Test additive
    X_evaded, meta = evasion_noise(X_test, epsilon=0.05, strategy='additive')
    print(f"\nAdditive metadata: {meta}")
    
    print("\nEvasion tests passed!")