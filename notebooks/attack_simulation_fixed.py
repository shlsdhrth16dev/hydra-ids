"""
Attack Simulation - Production Demo

This script demonstrates the upgraded attack simulation framework with:
- Multiple attack types
- Automatic metrics calculation
- Attack effectiveness comparison
- Comprehensive reporting

Run this in a Jupyter notebook or as a standalone script.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import attack framework
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from attacks import AttackController, AttackMetrics

# Configure plotting
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 60)
print("ATTACK SIMULATION - PRODUCTION DEMO")
print("=" * 60)

# ============================================================================
# 1. Load Data and Model
# ============================================================================
print("\n[1/7] Loading data and model...")

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODEL_DIR = PROJECT_ROOT / 'models'

X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
y_test = pd.read_csv(DATA_DIR / 'y_test.csv').squeeze()

# Load model and preprocessing
model = joblib.load(MODEL_DIR / 'xgboost/xgboost_v1.joblib')
scaler = joblib.load(MODEL_DIR / 'preprocessing/scaler.joblib')

# Load label names
with open(MODEL_DIR / 'preprocessing/label_names.json', 'r') as f:
    label_names = json.load(f)

print(f"✓ Data shape: {X_test.shape}")
print(f"✓ Labels: {len(label_names)} classes")
print(f"✓ Model: {type(model).__name__}")

# Sample subset for faster testing
SAMPLE_SIZE = 10000
np.random.seed(42)
sample_idx = np.random.choice(len(X_test), size=min(SAMPLE_SIZE, len(X_test)), replace=False)

X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
y_sample = y_test.iloc[sample_idx].reset_index(drop=True)

print(f"✓ Using {len(X_sample)} samples for testing")

# ============================================================================
# 2. Initialize Attack Controller
# ============================================================================
print("\n[2/7] Initializing Attack Controller...")

# Configure attacks
attack_config = {
    'label_flip_fraction': 0.15,
    'feature_noise': 0.10,
    'drift_strength': 0.25,
    'drop_fraction': 0.20,
    'epsilon': 0.05,
    'missing_fraction': 0.10,
    'outlier_fraction': 0.05,
    'outlier_magnitude': 4.0,
    'random_state': 42
}

# Initialize controller with model for metrics
controller = AttackController(
    config=attack_config,
    model=model,
    scaler=scaler,
    track_history=True
)

# Set baseline for rollback
controller.set_baseline(X_sample, y_sample)

print("✓ Attack controller initialized")

# ============================================================================
# 3. Run Individual Attacks
# ============================================================================
print("\n[3/7] Testing individual attacks...")

# Test 1: Drift Attack
print("\n=== DRIFT ATTACK ===")
X_drifted, drift_meta = controller.apply_drift(X_sample, evaluate=True, drift_type='gradual')

if 'evaluation' in drift_meta:
    eval_data = drift_meta['evaluation']
    print(f"Accuracy drop: {eval_data['model_degradation']['accuracy_drop']:.4f}")
    print(f"Attack effectiveness: {eval_data['attack_effectiveness']:.4f}")
    print(f"Prediction flip rate: {eval_data['prediction_analysis']['prediction_flip_rate']:.4f}")

# Test 2: Corruption Attack
print("\n=== CORRUPTION ATTACK (Feature Drop) ===")
X_corrupted, corruption_meta = controller.apply_corruption(
    X_sample, 
    evaluate=True, 
    corruption_type='drop_features'
)

if 'evaluation' in corruption_meta:
    eval_data = corruption_meta['evaluation']
    print(f"Accuracy drop: {eval_data['model_degradation']['accuracy_drop']:.4f}")
    print(f"Attack effectiveness: {eval_data['attack_effectiveness']:.4f}")

# Test 3: Evasion Attack
print("\n=== EVASION ATTACK ===")
X_evaded, evasion_meta = controller.apply_evasion(
    X_sample, 
    evaluate=True, 
    strategy='random_sign'
)

if 'evaluation' in evasion_meta:
    eval_data = evasion_meta['evaluation']
    print(f"Accuracy drop: {eval_data['model_degradation']['accuracy_drop']:.4f}")
    print(f"Evasion success rate: {eval_data['prediction_analysis']['evasion_success_rate']:.4f}")
    print(f"Attack effectiveness: {eval_data['attack_effectiveness']:.4f}")

# ============================================================================
# 4. Attack Chain (Combined)
# ============================================================================
print("\n[4/7] Testing attack chain...")

# Reset to baseline
X_clean, y_clean = controller.rollback()

# Apply attack chain
print("\n=== ATTACK CHAIN ===")
X_final, y_final, chain_meta = controller.apply_attack_chain(
    X_clean,
    y_clean,
    attack_sequence=['drift', 'corruption', 'evasion']
)

print(f"Applied {len(chain_meta) - 1} attacks in sequence")

# Extract chain evaluation
if 'chain_evaluation' in chain_meta[-1]:
    eval_data = chain_meta[-1]['chain_evaluation']
    print(f"\nCombined Attack Impact:")
    print(f"  Accuracy drop: {eval_data['model_degradation']['accuracy_drop']:.4f}")
    print(f"  F1 drop: {eval_data['model_degradation']['f1_drop']:.4f}")
    print(f"  Evasion success: {eval_data['prediction_analysis']['evasion_success_rate']:.4f}")
    print(f"  Attack effectiveness: {eval_data['attack_effectiveness']:.4f}")

# ============================================================================
# 5. Comparison and Visualization
# ============================================================================
print("\n[5/7] Creating attack comparison...")

# Compare all attacks
metrics_calc = AttackMetrics(model, scaler)

attack_results = {}
for attack_info in controller.attack_history:
    # Skip chain evaluation summaries
    if 'chain_evaluation' in attack_info:
        continue
    # Only add attacks with evaluation data
    if 'evaluation' in attack_info:
        attack_type = attack_info.get('attack_type') or attack_info.get('attack_combination', 'unknown')
        attack_results[attack_type] = attack_info['evaluation']

# Only create comparison if we have results
if attack_results:
    comparison_df = metrics_calc.compare_attacks(attack_results)
    print("\n=== ATTACK COMPARISON ===")
    print(comparison_df.to_string(index=False))
    
    # ========================================================================
    # 6. Visualization
    # ========================================================================
    print("\n[6/7] Generating visualizations...")
    
    # Plot attack effectiveness
    if len(comparison_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Effectiveness scores
        ax1 = axes[0]
        comparison_df.plot(x='attack', y='effectiveness', kind='barh', ax=ax1, color='coral')
        ax1.set_xlabel('Effectiveness Score')
        ax1.set_ylabel('Attack Type')
        ax1.set_title('Attack Effectiveness Comparison')
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Metric degradation
        ax2 = axes[1]
        metrics_to_plot = ['accuracy_drop', 'f1_drop', 'evasion_success_rate']
        comparison_df[metrics_to_plot].plot(kind='bar', ax=ax2)
        ax2.set_xlabel('Attack Index')
        ax2.set_ylabel('Score')
        ax2.set_title('Model Degradation Metrics')
        ax2.legend(loc='best')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(DATA_DIR.parent / 'logs' / 'attack_comparison.png', dpi=150, bbox_inches='tight')
        print("✓ Plots saved to data/logs/attack_comparison.png")
        plt.show()
else:
    print("\n⚠ No attack evaluations found in history")

# ============================================================================
# 7. Export Report
# ============================================================================
print("\n[7/7] Exporting report...")

# Export comprehensive report
REPORT_PATH = Path('../data/logs/attack_report.json')
controller.export_report(REPORT_PATH)

print(f"\n✓ Report exported to {REPORT_PATH}")
print(f"✓ Total attacks executed: {len(controller.attack_history)}")

print("\n" + "=" * 60)
print("ATTACK SIMULATION COMPLETE")
print("=" * 60)
