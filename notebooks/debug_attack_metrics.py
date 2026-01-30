"""
Debug script to understand why attack metrics are zero.

This script will help diagnose the evaluation issue.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from attacks import AttackController, AttackMetrics

# Load data and model
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'

print("=" * 70)
print("ATTACK METRICS DEBUG")
print("=" * 70)

# Load
X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
y_test = pd.read_csv(DATA_DIR / 'y_test.csv').squeeze()
model = joblib.load(MODEL_DIR / 'xgboost/xgboost_v1.joblib')
scaler = joblib.load(MODEL_DIR / 'preprocessing/scaler.joblib')

# Small sample
SAMPLE_SIZE = 1000
np.random.seed(42)
sample_idx = np.random.choice(len(X_test), size=min(SAMPLE_SIZE, len(X_test)), replace=False)
X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
y_sample = y_test.iloc[sample_idx].reset_index(drop=True)

print(f"\n1. Sample Info:")
print(f"   Shape: {X_sample.shape}")
print(f"   Label distribution:\n{y_sample.value_counts().head()}")

# Initialize
config = {
    'epsilon': 0.3,  # Increase epsilon significantly  
    'drift_strength': 0.5,
    'drop_fraction': 0.3,
    'random_state': 42
}

controller = AttackController(config, model, scaler)
controller.set_baseline(X_sample, y_sample)

# Test clean performance first
print(f"\n2. Clean Performance:")
X_scaled = scaler.transform(X_sample)
y_pred_clean = model.predict(X_scaled)
from sklearn.metrics import accuracy_score
clean_acc = accuracy_score(y_sample, y_pred_clean)
print(f"   Clean accuracy: {clean_acc:.4f}")
print(f"   Clean predictions unique: {np.unique(y_pred_clean)}")

# Apply single drift attack with increased strength
print(f"\n3. Applying Drift Attack (strength=0.5)...")
X_drifted, drift_meta = controller.apply_drift(X_sample, evaluate=True, drift_type='gradual')

print(f"\n4. Drift Attack Results:")
print(f"   Data changed: {not X_drifted.equals(X_sample)}")
print(f"   Max difference: {(X_drifted - X_sample).abs().max().max():.6f}")
print(f"   Features drifted: {len(drift_meta.get('drifted_features', []))}")

# Check attacked performance manually
X_drifted_scaled = scaler.transform(X_drifted)
y_pred_attacked = model.predict(X_drifted_scaled)
attacked_acc = accuracy_score(y_sample, y_pred_attacked)

print(f"\n5. Manual Evaluation:")
print(f"   Attacked accuracy: {attacked_acc:.4f}")
print(f"   Accuracy drop: {clean_acc - attacked_acc:.4f}")
print(f"   Predictions changed: {(y_pred_clean != y_pred_attacked).sum()} / {len(y_sample)}")

# Check what's in evaluation
if 'evaluation' in drift_meta:
    eval_data = drift_meta['evaluation']
    print(f"\n6. Stored Evaluation Data:")
    print(f"   Clean acc (stored): {eval_data['clean_performance']['accuracy']:.4f}")
    print(f"   Attacked acc (stored): {eval_data['attacked_performance']['accuracy']:.4f}")
    print(f"   Accuracy drop (stored): {eval_data['model_degradation']['accuracy_drop']:.4f}")
    print(f"   Effectiveness: {eval_data['attack_effectiveness']:.4f}")
else:
    print(f"\n6. ❌ NO EVALUATION DATA FOUND!")

# Try stronger evasion
print(f"\n7. Applying Strong Evasion Attack (epsilon=0.3)...")
X_evaded, evasion_meta = controller.apply_evasion(X_sample, evaluate=True, strategy='random_sign')

X_evaded_scaled = scaler.transform(X_evaded)
y_pred_evaded = model.predict(X_evaded_scaled)
evaded_acc = accuracy_score(y_sample, y_pred_evaded)

print(f"   Evaded accuracy: {evaded_acc:.4f}")
print(f"   Accuracy drop: {clean_acc - evaded_acc:.4f}")
print(f"   Max perturbation: {(X_evaded - X_sample).abs().max().max():.6f}")

if 'evaluation' in evasion_meta:
    print(f"   Stored accuracy drop: {evasion_meta['evaluation']['model_degradation']['accuracy_drop']:.4f}")

print("\n" + "=" * 70)
print("DEBUG COMPLETE")
print("=" * 70)
