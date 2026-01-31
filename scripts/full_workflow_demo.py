#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete Hydra-IDS Workflow Demonstration

This script demonstrates the complete workflow:
1. Load preprocessed data
2. Load trained model
3. Simulate adversarial attack
4. Trigger self-healing
5. Compare before/after metrics
6. Generate comprehensive report
"""

import sys
import os
from pathlib import Path
import logging
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from self_healing import (
    SelfHealingOrchestrator,
    SelfHealingConfig,
    HealthMonitor,
    DriftDetector
)
from attacks import AttackController
from sklearn.metrics import classification_report, accuracy_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner(title: str):
    """Print formatted banner"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def load_assets(sample_size: int = 10000):
    """Load all required data and models"""
    print_banner("STEP 1: Loading Assets")
    
    logger.info("Loading datasets...")
    X_train = pd.read_csv('data/processed/X_train.csv', nrows=sample_size)
    y_train = pd.read_csv('data/processed/y_train.csv', nrows=sample_size).squeeze()
    X_test = pd.read_csv('data/processed/X_test.csv', nrows=sample_size)
    y_test = pd.read_csv('data/processed/y_test.csv', nrows=sample_size).squeeze()
    
    print(f"✓ Loaded training data: {X_train.shape}")
    print(f"✓ Loaded test data: {X_test.shape}")
    
    logger.info("Loading model and preprocessing artifacts...")
    model = joblib.load('models/xgboost/xgboost_v1.joblib')
    scaler = joblib.load('models/preprocessing/scaler.joblib')
    
    print(f"✓ Loaded model: {type(model).__name__}")
    print(f"✓ Loaded scaler: {type(scaler).__name__}")
    
    return X_train, y_train, X_test, y_test, model, scaler

def evaluate_baseline(model, X_test, y_test):
    """Evaluate baseline model performance"""
    print_banner("STEP 2: Baseline Model Evaluation")
    
    logger.info("Evaluating baseline performance...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'weighted_precision': report['weighted avg']['precision'],
        'weighted_recall': report['weighted avg']['recall'],
        'weighted_f1': report['weighted avg']['f1-score']
    }
    
    print(f"✓ Baseline Accuracy:  {metrics['accuracy']:.4f}")
    print(f"✓ Baseline Precision: {metrics['weighted_precision']:.4f}")
    print(f"✓ Baseline Recall:    {metrics['weighted_recall']:.4f}")
    print(f"✓ Baseline F1-Score:  {metrics['weighted_f1']:.4f}")
    
    return y_pred, metrics

def simulate_attack(X_test, y_test, model, scaler):
    """Simulate adversarial attack chain"""
    print_banner("STEP 3: Simulating Adversarial Attack")
    
    logger.info("Initializing attack controller...")
    controller = AttackController(
        config={
            'drift_strength': 0.4,
            'drop_fraction': 0.2,
            'epsilon': 0.05,
            'random_state': 42
        },
        model=model,
        scaler=scaler,
        track_history=True
    )
    
    # Set baseline
    controller.set_baseline(X_test, y_test)
    print("✓ Baseline set for attack controller")
    
    # Apply attack chain
    logger.info("Applying attack chain: drift → corruption → evasion")
    X_attacked, y_attacked, attack_metadata = controller.apply_attack_chain(
        X_test, y_test,
        attack_sequence=['drift', 'corruption', 'evasion']
    )
    
    # Get attack effectiveness
    chain_eval = attack_metadata[-1].get('chain_evaluation', {})
    
    if not chain_eval:
        logger.warning("No chain evaluation found in metadata")
        # Create fallback metrics
        chain_eval = {
            'attack_effectiveness': 0.0,
            'model_degradation': {
                'accuracy_drop': 0.0,
                'f1_drop': 0.0
            },
            'prediction_analysis': {
                'prediction_flip_rate': 0.0
            }
        }
    
    # Extract metrics safely
    degradation = chain_eval.get('model_degradation', {})
    pred_analysis = chain_eval.get('prediction_analysis', {})
    
    print(f"\n✓ Attack Chain Applied:")
    print(f"  - Attack Effectiveness: {chain_eval.get('attack_effectiveness', 0):.3f}")
    print(f"  - Accuracy Drop: {degradation.get('accuracy_drop', 0):.3f}")
    print(f"  - F1 Drop: {degradation.get('f1_drop', 0):.3f}")
    print(f"  - Prediction Flip Rate: {pred_analysis.get('prediction_flip_rate', 0):.3f}")
    
    return X_attacked, y_attacked, attack_metadata

def trigger_self_healing(model, X_test, X_attacked, y_attacked, X_train, y_train):
    """Trigger self-healing workflow"""
    print_banner("STEP 4: Triggering Self-Healing Workflow")
    
    logger.info("Initializing self-healing orchestrator...")
    config = SelfHealingConfig()
    orchestrator = SelfHealingOrchestrator(config)
    
    print("✓ Orchestrator initialized")
    print(f"  - Policy: {config.decision_engine.policy}")
    print(f"  - Drift methods: {config.drift_detector.methods}")
    print(f"  - Health thresholds: recall={config.health_monitor.recall_threshold}")
    
    # Use smaller sample for demo
    X_ref = X_test.iloc[:1000]
    X_cur = X_attacked.iloc[:1000]
    y_cur = y_attacked.iloc[:1000]
    X_train_sample = X_train.iloc[:2000]
    y_train_sample = y_train.iloc[:2000]
    
    logger.info("Running healing workflow...")
    start_time = time.time()
    
    workflow_report = orchestrator.run_healing_workflow(
        X_reference=X_ref,
        X_current=X_cur,
        y_current=y_cur,
        current_model=model,
        X_train=X_train_sample,
        y_train=y_train_sample,
        dry_run=True  # Set to False to actually retrain
    )
    
    duration = time.time() - start_time
    
    print(f"\n✓ Workflow completed in {duration:.2f}s")
    print(f"  - Success: {workflow_report['success']}")
    print(f"  - Final State: {workflow_report['final_state']}")
    print(f"  - Recommended Action: {workflow_report['final_action']}")
    
    # Extract stage results
    stages = workflow_report.get('stages', {})
    if 'health_monitoring' in stages:
        health = stages['health_monitoring'].get('result', {})
        print(f"\n✓ Health Check:")
        print(f"  - Is Healthy: {health.get('is_healthy', 'N/A')}")
        print(f"  - Accuracy: {health.get('metrics', {}).get('accuracy', 'N/A')}")
    
    if 'drift_detection' in stages:
        drift = stages['drift_detection'].get('result', {})
        print(f"\n✓ Drift Detection:")
        print(f"  - Drift Detected: {drift.get('drift_detected', 'N/A')}")
        print(f"  - Drift Ratio: {drift.get('drift_ratio', 'N/A'):.3f}" if isinstance(drift.get('drift_ratio'), float) else "  - Drift Ratio: N/A")
    
    return workflow_report

def compare_results(baseline_metrics, attack_metadata, workflow_report):
    """Compare before/after metrics"""
    print_banner("STEP 5: Comparative Analysis")
    
    chain_eval = attack_metadata[-1].get('chain_evaluation', {})
    degradation = chain_eval.get('model_degradation', {})
    
    comparison = {
        'baseline': {
            'accuracy': baseline_metrics['accuracy'],
            'f1': baseline_metrics['weighted_f1'],
        },
        'after_attack': {
            'accuracy': baseline_metrics['accuracy'] - degradation.get('accuracy_drop', 0),
            'f1': baseline_metrics['weighted_f1'] - degradation.get('f1_drop', 0),
        },
        'attack_impact': {
            'accuracy_drop': degradation.get('accuracy_drop', 0),
            'f1_drop': degradation.get('f1_drop', 0),
            'effectiveness': chain_eval.get('attack_effectiveness', 0)
        },
        'healing_response': {
            'action_taken': workflow_report.get('final_action', 'N/A'),
            'workflow_success': workflow_report.get('success', False),
            'duration': workflow_report.get('duration_seconds', 0)
        }
    }
    
    print("Metric Comparison:")
    print(f"\n{'Metric':<20} {'Baseline':<12} {'After Attack':<12} {'Drop':<12}")
    print("-" * 60)
    print(f"{'Accuracy':<20} {comparison['baseline']['accuracy']:<12.4f} "
          f"{comparison['after_attack']['accuracy']:<12.4f} "
          f"{comparison['attack_impact']['accuracy_drop']:<12.4f}")
    print(f"{'F1-Score':<20} {comparison['baseline']['f1']:<12.4f} "
          f"{comparison['after_attack']['f1']:<12.4f} "
          f"{comparison['attack_impact']['f1_drop']:<12.4f}")
    
    print(f"\nAttack Effectiveness: {comparison['attack_impact']['effectiveness']:.3f}")
    print(f"Healing Action: {comparison['healing_response']['action_taken']}")
    print(f"Workflow Duration: {comparison['healing_response']['duration']:.2f}s")
    
    return comparison

def generate_report(comparison, attack_metadata, workflow_report):
    """Generate comprehensive JSON report"""
    print_banner("STEP 6: Generating Report")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'demo_type': 'complete_workflow',
        'summary': comparison,
        'attack_details': [
            {
                'attack_type': meta.get('attack_type', 'unknown'),
                'metrics': meta.get('evaluation', {}).get('model_degradation', {})
            }
            for meta in attack_metadata
        ],
        'workflow_details': {
            'workflow_id': workflow_report.get('workflow_id'),
            'stages': list(workflow_report.get('stages', {}).keys()),
            'final_state': workflow_report.get('final_state'),
            'success': workflow_report.get('success')
        }
    }
    
    # Save report
    output_path = Path('data/logs/complete_workflow_report.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✓ Report saved to: {output_path}")
    
    return report

def main():
    """Main workflow execution"""
    print_banner("🛡️  HYDRA-IDS COMPLETE WORKFLOW DEMONSTRATION")
    
    start_time = time.time()
    
    try:
        # Step 1: Load assets
        X_train, y_train, X_test, y_test, model, scaler = load_assets(sample_size=5000)
        
        # Step 2: Baseline evaluation
        y_pred_baseline, baseline_metrics = evaluate_baseline(model, X_test, y_test)
        
        # Step 3: Simulate attack
        X_attacked, y_attacked, attack_metadata = simulate_attack(
            X_test, y_test, model, scaler
        )
        
        # Step 4: Self-healing
        workflow_report = trigger_self_healing(
            model, X_test, X_attacked, y_attacked, X_train, y_train
        )
        
        # Step 5: Compare results
        comparison = compare_results(baseline_metrics, attack_metadata, workflow_report)
        
        # Step 6: Generate report
        report = generate_report(comparison, attack_metadata, workflow_report)
        
        # Final summary
        total_duration = time.time() - start_time
        
        print_banner("✅ WORKFLOW COMPLETED SUCCESSFULLY")
        
        print(f"Total Execution Time: {total_duration:.2f}s\n")
        print("Summary:")
        print(f"  ✓ Baseline model evaluated")
        print(f"  ✓ 3-stage attack chain simulated")
        print(f"  ✓ Self-healing workflow executed")
        print(f"  ✓ Comprehensive report generated")
        print(f"\nRecommended Action: {workflow_report.get('final_action', 'N/A')}")
        print(f"System Status: {'🟢 Healthy' if workflow_report.get('success') else '🔴 Issues Detected'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
