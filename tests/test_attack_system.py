"""
Test Attack Simulation System

Comprehensive test to verify all attack components work correctly.
"""

import sys
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from attacks import AttackController, AttackMetrics

def test_attack_system():
    """Test the complete attack simulation system."""
    
    print("=" * 60)
    print("ATTACK SIMULATION SYSTEM TEST")
    print("=" * 60)
    
    # 1. Load data and model
    print("\n[1/6] Loading data and model...")
    DATA_DIR = Path(__file__).parent.parent / 'data' / 'processed'
    MODEL_DIR = Path(__file__).parent.parent / 'models'
    
    try:
        X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
        y_test = pd.read_csv(DATA_DIR / 'y_test.csv').squeeze()
        model = joblib.load(MODEL_DIR / 'xgboost' / 'xgboost_v1.joblib')
        scaler = joblib.load(MODEL_DIR / 'preprocessing' / 'scaler.joblib')
        
        with open(MODEL_DIR / 'preprocessing' / 'label_names.json', 'r') as f:
            label_names = json.load(f)
        
        print(f"   ✓ Data shape: {X_test.shape}")
        print(f"   ✓ Labels: {len(label_names)} classes")
        print(f"   ✓ Model: {type(model).__name__}")
    except Exception as e:
        print(f"   ✗ Failed to load data/model: {e}")
        return False
    
    # 2. Sample data for faster testing
    print("\n[2/6] Sampling test data...")
    SAMPLE_SIZE = 5000
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_test), size=min(SAMPLE_SIZE, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
    y_sample = y_test.iloc[sample_idx].reset_index(drop=True)
    print(f"   ✓ Using {len(X_sample)} samples")
    
    # 3. Initialize AttackController
    print("\n[3/6] Initializing AttackController...")
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
    
    try:
        controller = AttackController(
            config=attack_config,
            model=model,
            scaler=scaler,
            track_history=True
        )
        controller.set_baseline(X_sample, y_sample)
        print("   ✓ AttackController initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize controller: {e}")
        return False
    
    # 4. Test individual attacks
    print("\n[4/6] Testing individual attacks...")
    
    # Test Drift Attack
    try:
        print("   Testing drift attack...")
        X_drifted, drift_meta = controller.apply_drift(X_sample, evaluate=True, drift_type='gradual')
        if 'evaluation' in drift_meta:
            eval_data = drift_meta['evaluation']
            print(f"      ✓ Accuracy drop: {eval_data['model_degradation']['accuracy_drop']:.4f}")
            print(f"      ✓ Attack effectiveness: {eval_data['attack_effectiveness']:.4f}")
        else:
            print("      ⚠ No evaluation data in metadata")
    except Exception as e:
        print(f"      ✗ Drift attack failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Corruption Attack
    try:
        print("   Testing corruption attack...")
        X_corrupted, corruption_meta = controller.apply_corruption(
            X_sample, 
            evaluate=True, 
            corruption_type='drop_features'
        )
        if 'evaluation' in corruption_meta:
            eval_data = corruption_meta['evaluation']
            print(f"      ✓ Accuracy drop: {eval_data['model_degradation']['accuracy_drop']:.4f}")
            print(f"      ✓ Attack effectiveness: {eval_data['attack_effectiveness']:.4f}")
        else:
            print("      ⚠ No evaluation data in metadata")
    except Exception as e:
        print(f"      ✗ Corruption attack failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Evasion Attack
    try:
        print("   Testing evasion attack...")
        X_evaded, evasion_meta = controller.apply_evasion(
            X_sample, 
            evaluate=True, 
            strategy='random_sign'
        )
        if 'evaluation' in evasion_meta:
            eval_data = evasion_meta['evaluation']
            print(f"      ✓ Accuracy drop: {eval_data['model_degradation']['accuracy_drop']:.4f}")
            print(f"      ✓ Evasion success: {eval_data['prediction_analysis']['evasion_success_rate']:.4f}")
            print(f"      ✓ Attack effectiveness: {eval_data['attack_effectiveness']:.4f}")
        else:
            print("      ⚠ No evaluation data in metadata")
    except Exception as e:
        print(f"      ✗ Evasion attack failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Test attack chain
    print("\n[5/6] Testing attack chain...")
    try:
        X_clean, y_clean = controller.rollback()
        X_final, y_final, chain_meta = controller.apply_attack_chain(
            X_clean,
            y_clean,
            attack_sequence=['drift', 'corruption', 'evasion']
        )
        print(f"   ✓ Applied {len(chain_meta) - 1} attacks in sequence")
        
        if 'chain_evaluation' in chain_meta[-1]:
            eval_data = chain_meta[-1]['chain_evaluation']
            print(f"   ✓ Combined accuracy drop: {eval_data['model_degradation']['accuracy_drop']:.4f}")
            print(f"   ✓ Combined effectiveness: {eval_data['attack_effectiveness']:.4f}")
    except Exception as e:
        print(f"   ✗ Attack chain failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Test comparison and reporting
    print("\n[6/6] Testing comparison and reporting...")
    try:
        metrics_calc = AttackMetrics(model, scaler)
        
        # Build attack results dict
        attack_results = {}
        for attack_info in controller.attack_history:
            if 'evaluation' in attack_info:
                attack_type = attack_info.get('attack_type') or attack_info.get('attack_combination', 'unknown')
                attack_results[attack_type] = attack_info['evaluation']
        
        print(f"   Found {len(attack_results)} attacks with evaluation data")
        
        if len(attack_results) > 0:
            comparison_df = metrics_calc.compare_attacks(attack_results)
            print(f"   ✓ Comparison DataFrame created: {comparison_df.shape}")
            print("\n   Attack Comparison:")
            print(comparison_df.to_string(index=False))
        else:
            print("   ⚠ No attacks with evaluation data found")
        
        # Export report
        REPORT_PATH = Path(__file__).parent.parent / 'data' / 'logs' / 'test_attack_report.json'
        controller.export_report(REPORT_PATH)
        print(f"\n   ✓ Report exported to {REPORT_PATH}")
        
    except Exception as e:
        print(f"   ✗ Comparison/reporting failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_attack_system()
    sys.exit(0 if success else 1)
