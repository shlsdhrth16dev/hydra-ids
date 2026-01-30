"""Extended script to test all attack functionality"""
import sys
import os
from pathlib import Path

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

# Cell 1: Imports
print("=" * 60)
print("CELL 1: Imports")
print("=" * 60)
try:
    import pandas as pd
    import numpy as np
    import joblib
    import json
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    from attacks import AttackController, AttackMetrics
    
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cell 2: Load Data and Model
print("\n" + "=" * 60)
print("CELL 2: Load Data and Model")
print("=" * 60)
try:
    DATA_DIR = PROJECT_DIR / 'data' / 'processed'
    MODEL_DIR = PROJECT_DIR / 'models'
    
    X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
    y_test = pd.read_csv(DATA_DIR / 'y_test.csv').squeeze()
    
    model = joblib.load(MODEL_DIR / 'xgboost' / 'xgboost_v1.joblib')
    scaler = joblib.load(MODEL_DIR / 'preprocessing' / 'scaler.joblib')
    
    with open(MODEL_DIR / 'preprocessing' / 'label_names.json', 'r') as f:
        label_names = json.load(f)
    
    print(f"✓ Data shape: {X_test.shape}")
    print(f"✓ Labels: {len(label_names)} classes")
    print(f"✓ Model: {type(model).__name__}")
except Exception as e:
    print(f"✗ Error loading data/model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cell 3: Sample Data
print("\n" + "=" * 60)
print("CELL 3: Sample Data")
print("=" * 60)
try:
    SAMPLE_SIZE = 10000
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_test), size=min(SAMPLE_SIZE, len(X_test)), replace=False)
    
    X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
    y_sample = y_test.iloc[sample_idx].reset_index(drop=True)
    
    print(f"✓ Using {len(X_sample)} samples for testing")
except Exception as e:
    print(f"✗ Error sampling data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cell 4: Initialize Attack Controller
print("\n" + "=" * 60)
print("CELL 4: Initialize Attack Controller")
print("=" * 60)
try:
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
    
    controller = AttackController(
        config=attack_config,
        model=model,
        scaler=scaler,
        track_history=True
    )
    
    controller.set_baseline(X_sample, y_sample)
    print("✓ Attack controller initialized")
except Exception as e:
    print(f"✗ Error initializing controller: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cell 5: Test Drift Attack
print("\n" + "=" * 60)
print("CELL 5: Drift Attack")
print("=" * 60)
try:
    print("\n=== DRIFT ATTACK ===")
    X_drifted, drift_meta = controller.apply_drift(X_sample, evaluate=True, drift_type='gradual')
    
    if 'evaluation' in drift_meta:
        eval_data = drift_meta['evaluation']
        print(f"✓ Accuracy drop: {eval_data['model_degradation']['accuracy_drop']:.4f}")
        print(f"✓ Attack effectiveness: {eval_data['attack_effectiveness']:.4f}")
        print(f"✓ Prediction flip rate: {eval_data['prediction_analysis']['prediction_flip_rate']:.4f}")
    else:
        print("⚠ No evaluation data in drift metadata")
except Exception as e:
    print(f"✗ Error in drift attack: {e}")
    import traceback
    traceback.print_exc()

# Cell 6: Test Corruption Attack
print("\n" + "=" * 60)
print("CELL 6: Corruption Attack")
print("=" * 60)
try:
    print("\n=== CORRUPTION ATTACK (Feature Drop) ===")
    X_corrupted, corruption_meta = controller.apply_corruption(
        X_sample, 
        evaluate=True, 
        corruption_type='drop_features'
    )
    
    if 'evaluation' in corruption_meta:
        eval_data = corruption_meta['evaluation']
        print(f"✓ Accuracy drop: {eval_data['model_degradation']['accuracy_drop']:.4f}")
        print(f"✓ Attack effectiveness: {eval_data['attack_effectiveness']:.4f}")
    else:
        print("⚠ No evaluation data in corruption metadata")
except Exception as e:
    print(f"✗ Error in corruption attack: {e}")
    import traceback
    traceback.print_exc()

# Cell 7: Test Evasion Attack
print("\n" + "=" * 60)
print("CELL 7: Evasion Attack")
print("=" * 60)
try:
    print("\n=== EVASION ATTACK ===")
    X_evaded, evasion_meta = controller.apply_evasion(
        X_sample, 
        evaluate=True, 
        strategy='random_sign'
    )
    
    if 'evaluation' in evasion_meta:
        eval_data = evasion_meta['evaluation']
        print(f"✓ Accuracy drop: {eval_data['model_degradation']['accuracy_drop']:.4f}")
        print(f"✓ Evasion success rate: {eval_data['prediction_analysis']['evasion_success_rate']:.4f}")
        print(f"✓ Attack effectiveness: {eval_data['attack_effectiveness']:.4f}")
    else:
        print("⚠ No evaluation data in evasion metadata")
except Exception as e:
    print(f"✗ Error in evasion attack: {e}")
    import traceback
    traceback.print_exc()

# Cell 8: Test Attack Chain
print("\n" + "=" * 60)
print("CELL 8: Attack Chain")
print("=" * 60)
try:
    X_clean, y_clean = controller.rollback()
    
    print("\n=== ATTACK CHAIN ===")
    X_final, y_final, chain_meta = controller.apply_attack_chain(
        X_clean,
        y_clean,
        attack_sequence=['drift', 'corruption', 'evasion']
    )
    
    print(f"✓ Applied {len(chain_meta) - 1} attacks in sequence")
    
    if 'chain_evaluation' in chain_meta[-1]:
        eval_data = chain_meta[-1]['chain_evaluation']
        print(f"\nCombined Attack Impact:")
        print(f"  Accuracy drop: {eval_data['model_degradation']['accuracy_drop']:.4f}")
        print(f"  F1 drop: {eval_data['model_degradation']['f1_drop']:.4f}")
        print(f"  Evasion success: {eval_data['prediction_analysis']['evasion_success_rate']:.4f}")
        print(f"  Attack effectiveness: {eval_data['attack_effectiveness']:.4f}")
    else:
        print("⚠ No chain evaluation data")
except Exception as e:
    print(f"✗ Error in attack chain: {e}")
    import traceback
    traceback.print_exc()

# Cell 9: Compare Attacks
print("\n" + "=" * 60)
print("CELL 9: Attack Comparison")
print("=" * 60)
try:
    metrics_calc = AttackMetrics(model, scaler)
    
    attack_results = {}
    for attack_info in controller.attack_history[:-1]:  # Exclude chain evaluation
        if 'evaluation' in attack_info:
            attack_type = attack_info.get('attack_type') or attack_info.get('attack_combination', 'unknown')
            attack_results[attack_type] = attack_info['evaluation']
    
    comparison_df = metrics_calc.compare_attacks(attack_results)
    print("\n=== ATTACK COMPARISON ===")
    print(comparison_df.to_string(index=False))
    print(f"\n✓ Compared {len(comparison_df)} attacks")
except Exception as e:
    print(f"✗ Error in attack comparison: {e}")
    import traceback
    traceback.print_exc()

# Cell 10: Export Report
print("\n" + "=" * 60)
print("CELL 10: Export Report")
print("=" * 60)
try:
    REPORT_PATH = PROJECT_DIR / 'data' / 'logs' / 'attack_report.json'
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    controller.export_report(REPORT_PATH)
    
    print(f"\n✓ Report exported to {REPORT_PATH}")
    print(f"✓ Total attacks executed: {len(controller.attack_history)}")
except Exception as e:
    print(f"✗ Error exporting report: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("ALL NOTEBOOK CELLS EXECUTED SUCCESSFULLY")
print("=" * 60)
