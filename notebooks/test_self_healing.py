# -*- coding: utf-8 -*-
"""
Simple test of the enhanced self-healing system components.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path so we can import self_healing
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("="*80)
print("SELF-HEALING SYSTEM COMPONENT TEST")
print("="*80)
print()

# Test 1: Import all components
print("Test 1: Importing components...")
try:
    from self_healing import (
        SelfHealingOrchestrator,
        HealthMonitor,
        DriftDetector,
        DecisionEngine,
        Retrainer,
        RollbackManager,
        AlertSystem,
        SelfHealingConfig
    )
    print("   [OK] All components imported successfully")
except Exception as e:
    print(f"   [FAIL] Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 2: Initialize components
print("Test 2: Initializing components...")
try:
    config = SelfHealingConfig()
    print(f"   [OK] Configuration created")
    
    health_monitor = HealthMonitor(config.health_monitor)
    print(f"   [OK] HealthMonitor initialized")
    
    drift_detector = DriftDetector(config.drift_detector)
    print(f"   [OK] DriftDetector initialized")
    
    decision_engine = DecisionEngine(config.decision_engine)
    print(f"   [OK] DecisionEngine initialized")
    
    retrainer = Retrainer(config.retrainer)
    print(f"   [OK] Retrainer initialized")
    
    rollback_manager = RollbackManager(config.rollback, "models/baseline")
    print(f"   [OK] RollbackManager initialized")
    
    alert_system = AlertSystem(config.alert_system)
    print(f"   [OK] AlertSystem initialized")
    
    orchestrator = SelfHealingOrchestrator(config)
    print(f"   [OK] Orchestrator initialized")
    
except Exception as e:
    print(f"   [FAIL] Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 3: Configuration
print("Test 3: Configuration settings...")
print(f"   Policy: {config.decision_engine.policy}")
print(f"   Drift methods: {config.drift_detector.methods}")
print(f"   Max retrains/day: {config.decision_engine.max_retrains_per_day}")
print(f"   Model type: {config.retrainer.model_type}")
print(f"   Alert channels: Console (default)")

print()

# Test 4: Test with synthetic data
print("Test 4: Testing with synthetic data...")
try:
    import numpy as np
    import pandas as pd
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X_ref = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    X_cur = pd.DataFrame(
        np.random.randn(n_samples, n_features) + 0.1,  # Slight drift
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.randint(0, 2, n_samples)
    
    # Test health monitoring
    health_result = health_monitor.evaluate(y_true, y_pred)
    print(f"   [OK] Health Monitor: is_healthy={health_result['is_healthy']}, "
          f"recall={health_result['metrics']['recall']:.3f}")
    
    # Test drift detection
    drift_result = drift_detector.detect(X_ref, X_cur)
    print(f"   [OK] Drift Detector: drift_detected={drift_result['drift_detected']}, "
          f"ratio={drift_result['drift_ratio']:.3f}")
    
    # Test decision engine
    decision_result = decision_engine.decide(health_result, drift_result)
    print(f"   [OK] Decision Engine: action={decision_result['action']}, "
          f"confidence={decision_result['confidence']:.2f}")
    
    # Test alert system
    alert_result = alert_system.send_alert(
        "Test Alert",
        "This is a test alert",
        "INFO"
    )
    print(f"   [OK] Alert System: sent={alert_result['sent']}")
    
except Exception as e:
    print(f"   [FAIL] Synthetic test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 5: Component summaries
print("Test 5: Component summaries...")
try:
    health_summary = health_monitor.get_health_summary()
    print(f"   [OK] Health Monitor: {health_summary['evaluation_count']} evaluations")
    
    drift_summary = drift_detector.get_drift_summary()
    print(f"   [OK] Drift Detector: {drift_summary['detections_performed']} detections")
    
    decision_summary = decision_engine.get_decision_summary()
    print(f"   [OK] Decision Engine: {decision_summary['decisions_made']} decisions")
    
    alert_summary = alert_system.get_alert_summary()
    print(f"   [OK] Alert System: {alert_summary['alerts_sent']} alerts sent")
    
except Exception as e:
    print(f"   [FAIL] Summary failed: {e}")
    sys.exit(1)

print()
print("="*80)
print("ALL TESTS PASSED!")
print("="*80)
print()
print("The enhanced self-healing system is working correctly!")
print("All components initialized and tested successfully.")
print()
