# Attack Simulation Framework - User Guide

## Overview

The Attack Simulation Framework provides production-grade adversarial testing for ML-based Intrusion Detection Systems. It includes multiple attack types with comprehensive metrics tracking and reporting.

## Features

- **Multiple Attack Types**: Poisoning, Evasion, Drift, and Corruption attacks
- **Automatic Metrics**: Model degradation tracking and attack effectiveness scoring
- **Attack History**: Full tracking with rollback capability
- **Attack Chaining**: Combine multiple attacks for worst-case testing
- **Comprehensive Validation**: Input validation and error handling
- **Professional Logging**: Detailed logging for debugging and monitoring

## Quick Start

### 1. Basic Usage

```python
from attacks import AttackController
import pandas as pd
import joblib

# Load your data and model
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
model = joblib.load('models/xgboost/xgboost_v1.joblib')
scaler = joblib.load('models/preprocessing/scaler.joblib')

# Configure attacks
config = {
    'drift_strength': 0.3,
    'drop_fraction': 0.2,
    'epsilon': 0.05,
    'random_state': 42
}

# Initialize controller
controller = AttackController(
    config=config,
    model=model,
    scaler=scaler,
    track_history=True
)

# Set baseline for rollback
controller.set_baseline(X_test, y_test)

# Apply attack
X_drifted, metadata = controller.apply_drift(X_test, evaluate=True)

# View results
print(f"Accuracy drop: {metadata['evaluation']['model_degradation']['accuracy_drop']:.4f}")
```

### 2. Attack Types

#### Drift Attack
```python
X_drifted, meta = controller.apply_drift(
    X, 
    evaluate=True,
    drift_type='gradual'  # Options: 'gradual', 'sudden', 'covariate'
)
```

#### Corruption Attack
```python
X_corrupted, meta = controller.apply_corruption(
    X,
    evaluate=True,
    corruption_type='drop_features'  # Options: 'drop_features', 'missing_values', 'outliers'
)
```

#### Evasion Attack
```python
X_evaded, meta = controller.apply_evasion(
    X,
    evaluate=True,
    strategy='random_sign'  # Options: 'random_sign', 'additive', 'multiplicative'
)
```

#### Poisoning Attack
```python
X_poisoned, y_poisoned, meta = controller.apply_poisoning(
    X, y,
    evaluate=True
)
```

### 3. Attack Chaining

Apply multiple attacks in sequence:

```python
X_final, y_final, meta_list = controller.apply_attack_chain(
    X, y,
    attack_sequence=['drift', 'corruption', 'evasion']
)

# Get combined metrics
chain_eval = meta_list[-1]['chain_evaluation']
print(f"Combined effectiveness: {chain_eval['attack_effectiveness']:.3f}")
```

### 4. Rollback

```python
# Apply attacks
controller.apply_drift(X)
controller.apply_corruption(X)

# Rollback to original data
X_clean, y_clean = controller.rollback()
```

### 5. Export Report

```python
from pathlib import Path

controller.export_report(Path('data/logs/attack_report.json'))
```

## Configuration Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `label_flip_fraction` | float | Fraction of labels to flip (0-1) | 0.1 |
| `feature_noise` | float | Gaussian noise std for features | 0.1 |
| `drift_strength` | float | Drift magnitude as % of mean | 0.2 |
| `drop_fraction` | float | Fraction of features to drop | 0.2 |
| `epsilon` | float | Evasion perturbation magnitude | 0.05 |
| `missing_fraction` | float | Fraction of values to set as NaN | 0.1 |
| `outlier_fraction` | float | Fraction of samples to corrupt | 0.05 |
| `outlier_magnitude` | float | Outlier distance in std deviations | 5.0 |
| `random_state` | int | Random seed for reproducibility | 42 |

## Metrics Explanation

### Attack Effectiveness Score (0-1)
Weighted combination of accuracy, F1, precision, and recall degradation. Higher score = more effective attack.

### Model Degradation Metrics
- **Accuracy Drop**: Decrease in classification accuracy
- **F1 Drop**: Decrease in weighted F1 score
- **Precision/Recall Drop**: Changes in precision and recall

### Prediction Analysis
- **Prediction Flip Rate**: Fraction of predictions that changed
- **Evasion Success Rate**: Fraction of attacks misclassified as benign
- **False Negatives Increase**: Increase in undetected attacks

## Running Tests

```bash
# Run all tests
python -m pytest tests/test_attacks.py -v

# Run specific test class
python -m pytest tests/test_attacks.py::TestPoisoning -v

# Run with coverage
python -m pytest tests/test_attacks.py --cov=attacks --cov-report=html
```

## Troubleshooting

### ValidationError
- Check that X isDataFrame with numeric values only
- Ensure fractions are between 0 and 1
- Verify required columns exist

### Memory Issues
- Use smaller sample size for testing
- Enable chunked loading in AttackSimulator
- Run attacks sequentially instead of chaining

### Label Encoding Mismatch
- Ensure labels are numeric (0, 1, 2, ...), not strings
- Use label_names.json from preprocessing artifacts
- Check LabelEncoder mapping in models/preprocessing/

## Best Practices

1. **Always set baseline** before running attacks for rollback capability
2. **Use random_state** for reproducible results
3. **Monitor metrics** to understand attack impact
4. **Start small** - test with samples before full dataset
5. **Export reports** for documentation and analysis
6. **Enable history tracking** for debugging

## Advanced Usage

### Custom Attack Strategies

```python
from attacks.poisoning import feature_noise_attack

# Targeted feature noise
X_poisoned, meta = feature_noise_attack(
    X,
    noise_level=0.15,
    target_features=['flow_duration', 'packet_count'],
    noise_type='gaussian'
)
```

### Importance-Based Corruption

```python
from attacks.corruption import drop_features

# Require feature importance dict
feature_importance = {'feature_0': 0.5, 'feature_1': 0.3, ...}

X_corrupted, meta = drop_features(
    X,
    drop_fraction=0.2,
    strategy='important',  # Target most important features
    feature_importance=feature_importance
)
```

## Support

For issues or questions, refer to:
- Test suite: `tests/test_attacks.py`
- Example notebook: `notebooks/attack_simulation.ipynb`
- Module docstrings for detailed API documentation
