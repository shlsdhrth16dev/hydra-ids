"""Script to execute notebook cells and check for errors"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Cell 1: Imports
print("=" * 60)
print("CELL 1: Imports")
print("=" * 60)
try:
    import pandas as pd
    import numpy as np
    import joblib
    import json
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    from attacks import AttackController, AttackMetrics
    
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Cell 2: Load Data and Model
print("\n" + "=" * 60)
print("CELL 2: Load Data and Model")
print("=" * 60)
try:
    # Use absolute paths based on script location
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_DIR = SCRIPT_DIR.parent
    DATA_DIR = PROJECT_DIR / 'data' / 'processed'
    MODEL_DIR = PROJECT_DIR / 'models'
    
    # Check if files exist
    files_to_check = [
        DATA_DIR / 'X_test.csv',
        DATA_DIR / 'y_test.csv',
        MODEL_DIR / 'xgboost/xgboost_v1.joblib',
        MODEL_DIR / 'preprocessing/scaler.joblib',
        MODEL_DIR / 'preprocessing/label_names.json'
    ]
    
    missing_files = []
    for file_path in files_to_check:
        if not file_path.exists():
            missing_files.append(str(file_path))
            print(f"✗ Missing: {file_path}")
        else:
            print(f"✓ Found: {file_path}")
    
    if missing_files:
        print(f"\n✗ Missing {len(missing_files)} required files")
        sys.exit(1)
    
    # Load data
    X_test = pd.read_csv(DATA_DIR / 'X_test.csv')
    y_test = pd.read_csv(DATA_DIR / 'y_test.csv').squeeze()
    
    # Load model and preprocessing
    model = joblib.load(MODEL_DIR / 'xgboost/xgboost_v1.joblib')
    scaler = joblib.load(MODEL_DIR / 'preprocessing/scaler.joblib')
    
    # Load label names
    with open(MODEL_DIR / 'preprocessing/label_names.json', 'r') as f:
        label_names = json.load(f)
    
    print(f"\n✓ Data shape: {X_test.shape}")
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

print("\n" + "=" * 60)
print("ALL CRITICAL CELLS EXECUTED SUCCESSFULLY")
print("=" * 60)
