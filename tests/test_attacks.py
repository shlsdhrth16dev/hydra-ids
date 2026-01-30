"""
Test Suite for Attack Simulation Framework

Run with: python -m pytest tests/test_attacks.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Import attack modules
from attacks.poisoning import label_flipping_attack, feature_noise_attack
from attacks.evasion import evasion_noise
from attacks.drift import gradual_mean_shift, covariate_drift
from attacks.corruption import drop_features, inject_missing_values, inject_outliers
from attacks.attack_validator import (
    validate_dataframe,
    validate_series,
    validate_fraction,
    ValidationError
)
from attacks.attack_controller import AttackController


# Fixtures
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(100, 10) * 10 + 50,
        columns=[f'feature_{i}' for i in range(10)]
    )
    y = pd.Series(np.random.choice([0, 1, 2], size=100))
    return X, y


@pytest.fixture
def attack_config():
    """Default attack configuration."""
    return {
        'label_flip_fraction': 0.2,
        'feature_noise': 0.1,
        'drift_strength': 0.3,
        'drop_fraction': 0.25,
        'epsilon': 0.05,
        'random_state': 42
    }


# Validation Tests
class TestValidation:
    def test_validate_dataframe_valid(self, sample_data):
        X, _ = sample_data
        validate_dataframe(X, name="test")  # Should not raise
    
    def test_validate_dataframe_invalid_type(self):
        with pytest.raises(ValidationError):
            validate_dataframe("not a dataframe", name="test")
    
    def test_validate_dataframe_non_numeric(self):
        df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
        with pytest.raises(ValidationError):
            validate_dataframe(df, name="test", numeric_only=True)
    
    def test_validate_series_valid(self, sample_data):
        _, y = sample_data
        validate_series(y, name="test")  # Should not raise
    
    def test_validate_fraction_valid(self):
        validate_fraction(0.5, name="test")  # Should not raise
    
    def test_validate_fraction_invalid(self):
        with pytest.raises(ValidationError):
            validate_fraction(1.5, name="test")


# Poisoning Tests
class TestPoisoning:
    def test_label_flipping_basic(self, sample_data):
        _, y = sample_data
        y_poisoned, meta = label_flipping_attack(y, flip_fraction=0.2)
        
        assert len(y_poisoned) == len(y)
        assert meta['attack_type'] == 'label_flipping'
        assert meta['n_flipped'] > 0
        assert meta['n_flipped'] <= len(y) * 0.2 + 1
    
    def test_label_flipping_targeted(self, sample_data):
        _, y = sample_data
        y_poisoned, meta = label_flipping_attack(
            y, flip_fraction=0.3, target_class=0, flip_to_class=1
        )
        
        assert meta['target_class'] == 0
        assert meta['flip_to_class'] == 1
        assert meta['n_flipped'] > 0
    
    def test_feature_noise_basic(self, sample_data):
        X, _ = sample_data
        X_poisoned, meta = feature_noise_attack(X, noise_level=0.1)
        
        assert X_poisoned.shape == X.shape
        assert meta['attack_type'] == 'feature_noise'
        assert not np.allclose(X.values, X_poisoned.values)
    
    def test_feature_noise_reproducibility(self, sample_data):
        X, _ = sample_data
        X_p1, _ = feature_noise_attack(X, noise_level=0.1, random_state=42)
        X_p2, _ = feature_noise_attack(X, noise_level=0.1, random_state=42)
        
        assert np.allclose(X_p1.values, X_p2.values)


# Evasion Tests
class TestEvasion:
    def test_evasion_noise_basic(self, sample_data):
        X, _ = sample_data
        X_evaded, meta = evasion_noise(X, epsilon=0.05, strategy='random_sign')
        
        assert X_evaded.shape == X.shape
        assert meta['attack_type'] == 'evasion'
        assert meta['strategy'] == 'random_sign'
    
    def test_evasion_strategies(self, sample_data):
        X, _ = sample_data
        strategies = ['random_sign', 'additive', 'multiplicative']
        
        for strategy in strategies:
            X_evaded, meta = evasion_noise(X, epsilon=0.05, strategy=strategy)
            assert meta['strategy'] == strategy
            assert not np.allclose(X.values, X_evaded.values)
    
    def test_evasion_clip_to_valid(self, sample_data):
        X, _ = sample_data
        X_evaded, _ = evasion_noise(X, epsilon=100, clip_to_valid=True)
        
        # All values should be >= 0
        assert (X_evaded.values >= 0).all()


# Drift Tests
class TestDrift:
    def test_gradual_drift(self, sample_data):
        X, _ = sample_data
        X_drifted, meta = gradual_mean_shift(
            X, shift_strength=0.2, drift_type='gradual'
        )
        
        assert X_drifted.shape == X.shape
        assert meta['drift_type'] == 'gradual'
        assert meta['attack_type'] == 'drift'
    
    def test_sudden_drift(self, sample_data):
        X, _ = sample_data
        X_drifted, meta = gradual_mean_shift(
            X, shift_strength=0.3, drift_type='sudden'
        )
        
        assert meta['drift_type'] == 'sudden'
    
    def test_covariate_drift(self, sample_data):
        X, _ = sample_data
        X_drifted, meta = covariate_drift(X, drift_magnitude=0.3)
        
        assert X_drifted.shape == X.shape
        assert meta['attack_type'] == 'covariate_drift'


# Corruption Tests
class TestCorruption:
    def test_drop_features(self, sample_data):
        X, _ = sample_data
        X_corrupted, meta = drop_features(X, drop_fraction=0.2)
        
        assert X_corrupted.shape == X.shape
        assert meta['attack_type'] == 'feature_corruption'
        assert len(meta['corrupted_features']) > 0
        
        # Check features are zeroed
        for col in meta['corrupted_features']:
            assert (X_corrupted[col] == 0).all()
    
    def test_missing_values(self, sample_data):
        X, _ = sample_data
        X_corrupted, meta = inject_missing_values(X, missing_fraction=0.1)
        
        assert X_corrupted.shape == X.shape
        assert meta['total_missing'] > 0
        assert X_corrupted.isnull().sum().sum() > 0
    
    def test_inject_outliers(self, sample_data):
        X, _ = sample_data
        X_corrupted, meta = inject_outliers(
            X, outlier_fraction=0.05, outlier_magnitude=5.0
        )
        
        assert meta['attack_type'] == 'outliers'
        assert meta['n_outliers'] > 0


# Controller Tests
class TestAttackController:
    def test_controller_init(self, attack_config):
        controller = AttackController(attack_config)
        assert controller.config == attack_config
    
    def test_controller_baseline(self, attack_config, sample_data):
        X, y = sample_data
        controller = AttackController(attack_config)
        controller.set_baseline(X, y)
        
        X_rollback, y_rollback = controller.rollback()
        assert X_rollback.equals(X)
        assert y_rollback.equals(y)
    
    def test_controller_apply_drift(self, attack_config, sample_data):
        X, y = sample_data
        controller = AttackController(attack_config)
        controller.set_baseline(X, y)
        
        X_drifted, meta = controller.apply_drift(X, evaluate=False)
        assert X_drifted.shape == X.shape
        assert 'timestamp' in meta
    
    def test_controller_apply_corruption(self, attack_config, sample_data):
        X, y = sample_data
        controller = AttackController(attack_config)
        
        X_corrupted, meta = controller.apply_corruption(X, evaluate=False)
        assert X_corrupted.shape == X.shape
    
    def test_controller_history_tracking(self, attack_config, sample_data):
        X, y = sample_data
        controller = AttackController(attack_config, track_history=True)
        controller.set_baseline(X, y)
        
        controller.apply_drift(X, evaluate=False)
        controller.apply_corruption(X, evaluate=False)
        
        assert len(controller.attack_history) == 2


# Integration Tests
class TestIntegration:
    def test_full_attack_pipeline(self, attack_config, sample_data):
        """Test complete attack workflow."""
        X, y = sample_data
        controller = AttackController(attack_config)
        controller.set_baseline(X, y)
        
        # Apply multiple attacks
        X_drifted, _ = controller.apply_drift(X, evaluate=False)
        X_corrupted, _ = controller.apply_corruption(X_drifted, evaluate=False)
        X_evaded, _ = controller.apply_evasion(X_corrupted, evaluate=False)
        
        # Verify shape maintained
        assert X_evaded.shape == X.shape
        
        # Verify changes made
        assert not np.allclose(X.values, X_evaded.values)
    
    def test_attack_chain(self, attack_config, sample_data):
        """Test attack chaining functionality."""
        X, y = sample_data
        controller = AttackController(attack_config)
        controller.set_baseline(X, y)
        
        X_final, y_final, meta_list = controller.apply_attack_chain(
            X, y,
            attack_sequence=['drift', 'corruption', 'evasion']
        )
        
        assert X_final.shape == X.shape
        assert len(meta_list) >= 3  # At least 3 attacks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
