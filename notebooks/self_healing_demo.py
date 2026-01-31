"""
Enhanced Self-Healing System Demonstration.

This script demonstrates the complete self-healing workflow with all new features.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import self-healing components
from self_healing import (
    SelfHealingOrchestrator,
    SelfHealingConfig,
    load_config
)
from attacks.attack_controller import AttackController


def main():
    """Run enhanced self-healing demonstration."""
    
    print("="*80)
    print("ENHANCED SELF-HEALING SYSTEM DEMONSTRATION")
    print("="*80)
    print()
    
    # Load configuration
    print("1. Loading configuration...")
    config = SelfHealingConfig()  # Use defaults
    
    # Or load from file:
    # config = load_config("config/self_healing.yaml")
    
    print(f"   Policy: {config.decision_engine.policy}")
    print(f"   Drift methods: {config.drift_detector.methods}")
    print(f"   Max retrains/day: {config.decision_engine.max_retrains_per_day}")
    print()
    
    # Initialize orchestrator
    print("2. Initializing self-healing orchestrator...")
    orchestrator = SelfHealingOrchestrator(config)
    print("   ✓ Orchestrator initialized with all components")
    print()
    
    # Load data
    print("3. Loading data...")
    try:
        X_train = pd.read_csv("data/processed/X_train.csv")
        X_test = pd.read_csv("data/processed/X_test.csv")
        y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
        y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
        
        # Convert to binary
        y_train_bin = (y_train != "benign").astype(int)
        y_test_bin = (y_test != "benign").astype(int)
        
        print(f"   Training: {X_train.shape}, Test: {X_test.shape}")
        print()
    except FileNotFoundError as e:
        print(f"   ✗ Error loading data: {e}")
        print("   Please ensure data files exist in data/processed/")
        return
    
    # Load model
    print("4. Loading model...")
    try:
        model = joblib.load("models/random_forest/random_forest_v1.joblib")
        print(f"   ✓ Loaded model: {type(model).__name__}")
        print()
    except FileNotFoundError:
        print("   ✗ Model not found. Trying alternative path...")
        try:
            model = joblib.load("models/xgboost/xgboost_v1.joblib")
            print(f"   ✓ Loaded model: {type(model).__name__}")
            print()
        except FileNotFoundError:
            print("   ✗ No models found. Please train a baseline model first.")
            return
    
    # Simulate attack to trigger self-healing
    print("5. Simulating adversarial attack...")
    attacker = AttackController({"drift_strength": 0.4})
    X_attacked = attacker.apply_drift(X_test)
    print(f"   ✓ Applied drift attack with strength=0.4")
    print()
    
    # Run self-healing workflow
    print("6. Running self-healing workflow...")
    print("-" * 80)
    
    workflow_report = orchestrator.run_healing_workflow(
        X_reference=X_test,  # Clean reference data
        X_current=X_attacked,  # Attacked/drifted data
        y_current=y_test_bin,
        current_model=model,
        X_train=X_train,
        y_train=y_train_bin,
        dry_run=False  # Set to True to simulate without changes
    )
    
    print("-" * 80)
    print()
    
    # Display results
    print("7. Workflow Results:")
    print(f"   Success: {workflow_report['success']}")
    print(f"   Duration: {workflow_report['duration_seconds']:.2f}s")
    print(f"   Final Action: {workflow_report['final_action']}")
    print()
    
    # Stage details
    print("8. Stage Details:")
    for stage_name, stage_info in workflow_report['stages'].items():
        status = "✓" if stage_info['success'] else "✗"
        print(f"   {status} {stage_name}")
        
        if stage_name == 'health_monitoring' and stage_info['success']:
            result = stage_info['result']
            print(f"      - Healthy: {result['is_healthy']}")
            print(f"      - Recall: {result['metrics']['recall']:.4f}")
            print(f"      - Precision: {result['metrics']['precision']:.4f}")
            print(f"      - F1: {result['metrics']['f1']:.4f}")
        
        elif stage_name == 'drift_detection' and stage_info['success']:
            result = stage_info['result']
            print(f"      - Drift Detected: {result['drift_detected']}")
            print(f"      - Drift Ratio: {result['drift_ratio']:.4f}")
            print(f"      - Methods: {', '.join(result['methods_used'])}")
        
        elif stage_name == 'decision_making' and stage_info['success']:
            result = stage_info['result']
            print(f"      - Action: {result['action']}")
            print(f"      - Confidence: {result['confidence']:.2f}")
            print(f"      - Reasoning: {'; '.join(result['reasoning'])}")
    
    print()
    
    # Component summaries
    print("9. Component Summaries:")
    
    health_summary = orchestrator.health_monitor.get_health_summary()
    print(f"   Health Monitor:")
    print(f"      - Evaluations: {health_summary['evaluation_count']}")
    print(f"      - Adaptive Thresholds: {health_summary['current_thresholds']}")
    
    drift_summary = orchestrator.drift_detector.get_drift_summary()
    print(f"   Drift Detector:")
    print(f"      - Detections: {drift_summary['detections_performed']}")
    print(f"      - Drifts Found: {drift_summary.get('total_drifts_detected', 0)}")
    
    decision_summary = orchestrator.decision_engine.get_decision_summary()
    print(f"   Decision Engine:")
    print(f"      - Decisions: {decision_summary['decisions_made']}")
    print(f"      - Actions: {decision_summary.get('action_counts', {})}")
    
    if orchestrator.retrainer.training_history:
        retrain_summary = orchestrator.retrainer.get_training_summary()
        print(f"   Retrainer:")
        print(f"      - Trainings: {retrain_summary['trainings_performed']}")
        print(f"      - Avg Train Score: {retrain_summary['avg_train_score']:.4f}")
    
    alert_summary = orchestrator.alert_system.get_alert_summary()
    print(f"   Alert System:")
    print(f"      - Alerts Sent: {alert_summary['alerts_sent']}")
    print(f"      - By Severity: {alert_summary.get('by_severity', {})}")
    
    print()
    
    # Workflow summary
    workflow_summary = orchestrator.get_workflow_summary()
    print("10. Overall Performance:")
    print(f"    Workflows Executed: {workflow_summary['workflows_executed']}")
    print(f"    Success Rate: {workflow_summary['success_rate']:.1%}")
    print(f"    Avg Duration: {workflow_summary['avg_duration']:.2f}s")
    print()
    
    print("="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
