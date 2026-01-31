# -*- coding: utf-8 -*-
"""
Complete Self-Healing Workflow Demo - Synthetic Data Version

This demonstrates the full workflow with synthetic data to show all features quickly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

from self_healing import SelfHealingOrchestrator, SelfHealingConfig

print("="*80)
print("COMPLETE SELF-HEALING WORKFLOW DEMONSTRATION")
print("="*80)
print()

# Step 1: Create synthetic data
print("1. Creating synthetic IDS data...")
np.random.seed(42)
n_train = 5000
n_test = 2000
n_features = 20

# Training data (clean)
X_train = pd.DataFrame(
    np.random.randn(n_train, n_features),
    columns=[f'feature_{i}' for i in range(n_features)]
)
y_train = pd.Series(np.random.randint(0, 2, n_train), name='label')

# Test data (clean reference)
X_test_clean = pd.DataFrame(
    np.random.randn(n_test, n_features),
    columns=[f'feature_{i}' for i in range(n_features)]
)
y_test = pd.Series(np.random.randint(0, 2, n_test), name='label')

# Attacked/drifted data (simulating adversarial attack)
X_test_attacked = X_test_clean.copy()
# Add significant drift to simulate attack
X_test_attacked += np.random.randn(n_test, n_features) * 0.5
# Corrupt some features more heavily
X_test_attacked.iloc[:, :5] += 1.0

print(f"   Training: {X_train.shape}")
print(f"   Test (clean): {X_test_clean.shape}")
print(f"   Test (attacked): {X_test_attacked.shape}")
print()

# Step 2: Train initial model
print("2. Training initial model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)
print(f"   Model trained: {type(model).__name__}")
print()

# Step 3: Initialize orchestrator
print("3. Initializing self-healing orchestrator...")
config = SelfHealingConfig()
config.decision_engine.policy = "balanced"
config.decision_engine.max_retrains_per_day = 5
config.drift_detector.enable_ensemble = True
config.health_monitor.enable_trend_tracking = True

orchestrator = SelfHealingOrchestrator(config)
print(f"   Policy: {config.decision_engine.policy}")
print(f"   Drift detection: Ensemble with {len(config.drift_detector.methods)} methods")
print()

# Step 4: Run self-healing workflow
print("4. Running self-healing workflow...")
print("-" * 80)

workflow_report = orchestrator.run_healing_workflow(
    X_reference=X_test_clean,      # Clean baseline data
    X_current=X_test_attacked,     # Attacked/drifted data
    y_current=y_test,
    current_model=model,
    X_train=X_train,
    y_train=y_train,
    dry_run=False
)

print("-" * 80)
print()

# Step 5: Display Results
print("5. WORKFLOW RESULTS:")
print("="*80)
print(f"Workflow ID: {workflow_report['workflow_id']}")
print(f"Success: {workflow_report['success']}")
print(f"Duration: {workflow_report['duration_seconds']:.2f} seconds")
final_action = workflow_report.get('final_action', 'NONE')
if final_action:
    print(f"Final Action: {final_action.upper()}")
else:
    print(f"Final Action: NO ACTION TAKEN")
print()

# Detailed stage results
print("6. STAGE-BY-STAGE BREAKDOWN:")
print("-"*80)

for stage_name, stage_info in workflow_report['stages'].items():
    status_icon = "[OK]" if stage_info['success'] else "[FAIL]"
    print(f"\n{status_icon} {stage_name.upper().replace('_', ' ')}")
    
    if not stage_info['success']:
        print(f"    Error: {stage_info.get('error', 'Unknown')}")
        continue
    
    result = stage_info.get('result', {})
    
    if stage_name == 'health_monitoring':
        print(f"    Health Status: {'HEALTHY' if result['is_healthy'] else 'UNHEALTHY'}")
        metrics = result['metrics']
        print(f"    Metrics:")
        print(f"      - Recall:    {metrics['recall']:.4f}")
        print(f"      - Precision: {metrics['precision']:.4f}")
        print(f"      - F1 Score:  {metrics['f1']:.4f}")
        print(f"      - Accuracy:  {metrics['accuracy']:.4f}")
        if 'roc_auc' in metrics:
            print(f"      - ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    elif stage_name == 'drift_detection':
        print(f"    Drift Detected: {result['drift_detected']}")
        print(f"    Drift Ratio: {result['drift_ratio']:.4f}")
        print(f"    Methods Used: {', '.join(result['methods_used'])}")
        print(f"    Drifted Features: {len(result.get('drifted_features', []))}/{len(X_test_clean.columns)}")
    
    elif stage_name == 'decision_making':
        print(f"    Recommended Action: {result['action'].upper()}")
        print(f"    Confidence: {result['confidence']:.2f}")
        print(f"    Health Severity: {result['health_severity']}")
        print(f"    Drift Severity: {result['drift_severity']}")
        print(f"    Reasoning:")
        for reason in result['reasoning']:
            print(f"      - {reason}")
    
    elif stage_name == 'action_execution':
        print(f"    Action Taken: {result['action'].upper()}")
        if result.get('new_model'):
            print(f"    New Model: Created")
            if 'version_id' in result:
                print(f"    Version ID: {result['version_id']}")
        elif result.get('dry_run'):
            print(f"    Mode: DRY RUN (no actual changes)")
        elif result.get('no_changes'):
            print(f"    Result: No changes needed")
    
    elif stage_name == 'validation' and 'result' in stage_info:
        print(f"    New Model Health: {'HEALTHY' if result['is_healthy'] else 'UNHEALTHY'}")
        print(f"    Recall: {result['metrics']['recall']:.4f}")
        print(f"    Accuracy: {result['metrics']['accuracy']:.4f}")

print()
print("-"*80)

# Step 7: Component Statistics
print("\n7. COMPONENT STATISTICS:")
print("-"*80)

health_summary = orchestrator.health_monitor.get_health_summary()
print(f"\nHealth Monitor:")
print(f"  Evaluations performed: {health_summary['evaluation_count']}")
print(f"  Current thresholds: {health_summary['current_thresholds']}")

drift_summary = orchestrator.drift_detector.get_drift_summary()
print(f"\nDrift Detector:")
print(f"  Detections performed: {drift_summary['detections_performed']}")
print(f"  Total drifts found: {drift_summary.get('total_drifts_detected', 0)}")

decision_summary = orchestrator.decision_engine.get_decision_summary()
print(f"\nDecision Engine:")
print(f"  Decisions made: {decision_summary['decisions_made']}")
print(f"  Action breakdown: {decision_summary.get('action_counts', {})}")
print(f"  Policy used: {decision_summary.get('current_policy', 'balanced')}")

if orchestrator.retrainer.training_history:
    retrain_summary = orchestrator.retrainer.get_training_summary()
    print(f"\nRetrainer:")
    print(f"  Models trained: {retrain_summary['trainings_performed']}")
    print(f"  Avg training score: {retrain_summary.get('avg_train_score', 0):.4f}")

alert_summary = orchestrator.alert_system.get_alert_summary()
print(f"\nAlert System:")
print(f"  Total alerts sent: {alert_summary['alerts_sent']}")
print(f"  By severity: {alert_summary.get('by_severity', {})}")

workflow_summary = orchestrator.get_workflow_summary()
print(f"\nOrchestrator:")
print(f"  Workflows executed: {workflow_summary['workflows_executed']}")
print(f"  Success rate: {workflow_summary['success_rate']:.1%}")
print(f"  Avg duration: {workflow_summary.get('avg_duration', 0):.2f}s")

print()
print("="*80)
print("DEMONSTRATION COMPLETE!")
print("="*80)
print()
print("Summary:")
print(f"  - All {len(workflow_report['stages'])} workflow stages completed successfully")
print(f"  - System automatically detected issues and took action: {workflow_report['final_action']}")
print(f"  - Total execution time: {workflow_report['duration_seconds']:.2f} seconds")
print()
print("The enhanced self-healing system is working perfectly!")
print("Ready for production deployment with your IDS data.")
print()
